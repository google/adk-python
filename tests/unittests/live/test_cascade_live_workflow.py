# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""`CascadeLive` as a workflow node: what context reaches the inner text model.

The workflow is ``START -> prep (function) -> summarizer (text agent) -> voice
(live agent)``. The voice agent's model is either a `CascadeLive` around a
recording text reasoner, or a fake native live model that records the request
it was connected with. Both are driven through `Runner.run_live`.

The voice agent runs in task mode: a chat-mode agent cannot follow another
node, so task mode is how a live node receives earlier context.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
from typing import Any
from typing import AsyncGenerator
from typing import AsyncIterator

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.events.event import Event
from google.adk.live import AgentSpokenOutput
from google.adk.live import AudioChunk
from google.adk.live import CascadeLive
from google.adk.live import LiveEgress
from google.adk.live import LiveIngress
from google.adk.live import LiveRequestQueue
from google.adk.live import UserTurnFinished
from google.adk.models.base_llm import BaseLlm
from google.adk.models.base_llm_connection import BaseLlmConnection
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow._base_node import START
from google.adk.workflow._workflow import Workflow
from google.genai import types
import pytest
from typing_extensions import override

_TIMEOUT_S = 60.0
_PREP_MARKER = 'PREP_PAPAYA_17'
_SUMMARY_MARKER = 'SUMMARY_MANGO_42'


def prep(node_input: Any) -> str:
  del node_input
  return f'fact from prep: {_PREP_MARKER}'


class _FakeStt(LiveIngress):
  """Treats each audio blob as one finished utterance of its bytes."""

  async def __call__(
      self, audio: AsyncIterator[types.Blob]
  ) -> AsyncGenerator[UserTurnFinished, None]:
    async for blob in audio:
      yield UserTurnFinished(text=blob.data.decode())


class _FakeTts(LiveEgress):

  async def __call__(
      self, text: AsyncIterator[str], *, cancel: asyncio.Event
  ) -> AsyncGenerator[AudioChunk | AgentSpokenOutput, None]:
    async for chunk in text:
      if cancel.is_set():
        return
      yield AudioChunk(
          blob=types.Blob(data=chunk.encode(), mime_type='audio/pcm')
      )
      yield AgentSpokenOutput(text=chunk)


class _TextModel(BaseLlm):
  """A text model replaying one response per call and recording requests."""

  model: str = 'recording-text'
  responses: list[LlmResponse] = []
  requests: list[LlmRequest] = []

  @override
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    self.requests.append(llm_request.model_copy(deep=True))
    index = len(self.requests) - 1
    if index < len(self.responses):
      yield self.responses[index]
    else:
      yield _text('OK.')


class _NativeLiveConnection(BaseLlmConnection):
  """Answers each realtime blob with the next scripted response."""

  def __init__(self, model: _NativeLiveModel):
    self._model = model
    self._out: asyncio.Queue[LlmResponse] = asyncio.Queue()

  async def send_history(self, history: list[types.Content]) -> None:
    self._model.histories.append(copy.deepcopy(history))
    # Like Gemini Live, answer a history that ends with a user turn.
    if history and history[-1].role == 'user':
      self._respond()

  async def send_content(self, content: types.Content) -> None:
    self._model.sent_contents.append(content)

  async def send_realtime(self, blob: Any) -> None:
    if isinstance(blob, types.Blob):
      self._respond()

  def _respond(self) -> None:
    self._out.put_nowait(self._model.responses.pop(0))
    self._out.put_nowait(LlmResponse(turn_complete=True))

  async def receive(self) -> AsyncGenerator[LlmResponse, None]:
    while True:
      yield await self._out.get()

  async def close(self) -> None:
    pass


class _NativeLiveModel(BaseLlm):
  """A native live model fake recording the connect request and history."""

  model: str = 'fake-native-live'
  responses: list[LlmResponse] = []
  connect_requests: list[LlmRequest] = []
  histories: list[list[types.Content]] = []
  sent_contents: list[types.Content] = []

  @override
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    raise NotImplementedError
    yield  # pylint: disable=unreachable

  @contextlib.asynccontextmanager
  async def connect(
      self, llm_request: LlmRequest
  ) -> AsyncGenerator[BaseLlmConnection, None]:
    self.connect_requests.append(llm_request.model_copy(deep=True))
    yield _NativeLiveConnection(self)


def _text(text: str) -> LlmResponse:
  return LlmResponse(
      content=types.Content(role='model', parts=[types.Part(text=text)])
  )


def _finish_task() -> LlmResponse:
  return LlmResponse(
      content=types.Content(
          role='model',
          parts=[
              types.Part.from_function_call(
                  name='finish_task', args={'result': 'voice_done'}
              )
          ],
      )
  )


def _summarize(contents: list[types.Content]) -> list[str]:
  """One readable line per content, for assertions."""
  lines = []
  for content in contents:
    for part in content.parts or []:
      if part.text is not None:
        lines.append(f'{content.role}: {part.text}')
      elif part.function_call:
        lines.append(f'{content.role}: call {part.function_call.name}')
      elif part.function_response:
        lines.append(f'{content.role}: response {part.function_response.name}')
      else:
        lines.append(f'{content.role}: <other part>')
  return lines


def _build_workflow(voice_model: BaseLlm) -> Workflow:
  """Returns `START -> prep -> summarizer -> voice`, with a task-mode voice."""
  summarizer = LlmAgent(
      name='summarizer',
      model=_TextModel(responses=[_text(f'summary: {_SUMMARY_MARKER}')]),
      instruction='Summarize the input.',
  )
  voice = LlmAgent(
      name='voice',
      model=voice_model,
      mode='task',
      instruction='Talk to the user about the summary.',
  )
  return Workflow(
      name='cascade_wf',
      edges=[
          (START, prep),
          (prep, summarizer),
          (summarizer, voice),
      ],
  )


async def _run(
    wf: Workflow,
    utterances: list[bytes],
    *,
    close_after_turns: int | None = None,
) -> list[Event]:
  """Runs the workflow live; sends each utterance once the previous turn ends.

  The voice agent speaks first, answering its node input, so the first
  utterance also waits for a completed turn.

  Args:
    wf: The workflow to run.
    utterances: Audio payloads, one per user turn.
    close_after_turns: Close the live queue after this many completed turns
      (for chat mode, which never finishes on its own).

  Returns:
    All events yielded by `run_live`.
  """
  session_service = InMemorySessionService()
  runner = Runner(app_name=wf.name, node=wf, session_service=session_service)
  session = await session_service.create_session(app_name=wf.name, user_id='u')
  queue = LiveRequestQueue()
  pending = list(utterances)

  events: list[Event] = []
  turns = 0

  async def consume() -> None:
    nonlocal turns
    async for event in runner.run_live(
        user_id='u',
        session_id=session.id,
        live_request_queue=queue,
        run_config=RunConfig(
            get_session_config={'state_delta': {'__START__': 'go'}}
        ),
    ):
      events.append(event)
      if not event.turn_complete:
        continue
      turns += 1
      if close_after_turns is not None and turns >= close_after_turns:
        queue.close()
      elif pending:
        queue.send_realtime(
            types.Blob(data=pending.pop(0), mime_type='audio/pcm')
        )

  await asyncio.wait_for(consume(), timeout=_TIMEOUT_S)
  return events


def _first_voice_reply(events: list[Event]) -> str | None:
  for event in events:
    if event.author != 'voice':
      continue
    if event.output_transcription and event.output_transcription.finished:
      return event.output_transcription.text
    if event.content and event.content.parts and event.content.parts[0].text:
      return event.content.parts[0].text
  return None


def _build_cascade(reasoner: _TextModel) -> CascadeLive:
  return CascadeLive(model=reasoner, stt=_FakeStt(), tts=_FakeTts())


# A task-mode node sees only its own node input (the summarizer's output) as
# the leading user turn, not the output of earlier nodes such as `prep`.
_NODE_INPUT_TURN = f'user: summary: {_SUMMARY_MARKER}'


@pytest.mark.asyncio
async def test_cascade_task_node_receives_node_input_and_turns():
  """The reasoner sees the node input, then each turn, but not earlier nodes."""
  reasoner = _TextModel(
      responses=[_text('Opening.'), _text('Reply one.'), _finish_task()]
  )
  wf = _build_workflow(_build_cascade(reasoner))

  events = await _run(wf, [b'turn one', b'turn two'])

  outputs = [e.output for e in events if e.output is not None]
  assert outputs[-1] == {'result': 'voice_done'}
  assert [_summarize(r.contents) for r in reasoner.requests] == [
      # Answered before the user said anything.
      [_NODE_INPUT_TURN],
      [_NODE_INPUT_TURN, 'model: Opening.', 'user: turn one'],
      [
          _NODE_INPUT_TURN,
          'model: Opening.',
          'user: turn one',
          'model: Reply one.',
          'user: turn two',
      ],
  ]
  first = reasoner.requests[0]
  assert _PREP_MARKER not in str(first.contents)
  assert 'Talk to the user' in str(first.config.system_instruction)
  assert 'finish_task' in first.tools_dict


@pytest.mark.asyncio
async def test_cascade_task_node_matches_native_live():
  """Cascade gets the history native live gets, and replies."""
  native = _NativeLiveModel(responses=[_text('Opening.'), _finish_task()])
  reasoner = _TextModel(responses=[_text('Opening.'), _finish_task()])

  native_events = await _run(_build_workflow(native), [b'hello voice'])
  cascade_events = await _run(
      _build_workflow(_build_cascade(reasoner)), [b'hello voice']
  )

  native_connect = _summarize(native.connect_requests[0].contents)
  native_history = _summarize(native.histories[0])
  cascade_first = _summarize(reasoner.requests[0].contents)
  assert native_connect == native_history == cascade_first == [_NODE_INPUT_TURN]
  # Both answer the node input before the user speaks.
  # The native fake replies in text, cascade in audio with a transcription.
  assert _first_voice_reply(native_events) == 'Opening.'
  assert _first_voice_reply(cascade_events) == 'Opening.'

  # The live request reaches the cascade reasoner unchanged otherwise.
  assert (
      reasoner.requests[0].config.system_instruction
      == native.connect_requests[0].config.system_instruction
  )
