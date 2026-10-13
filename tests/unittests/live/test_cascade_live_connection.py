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

"""Tests for `CascadeLiveConnection`: text reasoning between STT and TTS.

Cascade replaces the monolithic speech-to-speech connection with a text
reasoning agent wrapped in two audio transforms. This exercises the whole
loop with all three stages faked, so what it actually proves is the wiring:
that audio the client sent reaches the ingress transform, that the transcript
the ingress produced reaches the agent's own (text) model as an ordinary user
turn, and that the text the model generated comes back to the client as audio
the egress transform synthesized.
"""

from __future__ import annotations

import asyncio
from contextlib import aclosing
import logging
from typing import Any
from typing import AsyncGenerator
from typing import AsyncIterator
from typing import NamedTuple
from typing import Optional

from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.llm_agent import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.events.event import Event
from google.adk.live import AgentSpokenOutput
from google.adk.live import AudioChunk
from google.adk.live import CascadeLive
from google.adk.live import LiveEgress
from google.adk.live import LiveIngress
from google.adk.live import UserTurnFinished
from google.adk.live._cascade_live_connection import CascadeLiveConnection
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.genai import types
import pytest
from typing_extensions import override
from websockets.exceptions import ConnectionClosedOK

from .. import testing_utils

_MAX_EVENTS = 20
# Only catches hangs. The first `run_live` in a process can take many seconds
# to lazily import optional model SDKs, so a tight limit fails on slow machines.
_TIMEOUT_S = 60.0
_MIME_TYPE = 'audio/pcm'

# What the user "says", and what the model is primed to answer.
_UTTERANCE = 'what is the weather'
_ANSWER = 'Sunny and 70.'


class FakeStt(LiveIngress):
  """``Blob -> IngressEvent``. Treats each blob as one complete utterance.

  Real endpointing is a provider concern and the whole point of the ingress
  contract is that the framework does not care how the decision is made -- so
  this decides immediately.
  """

  def __init__(self) -> None:
    self.heard: list[bytes] = []
    self.sessions = 0

  async def __call__(
      self, audio: AsyncIterator[types.Blob]
  ) -> AsyncGenerator[UserTurnFinished, None]:
    self.sessions += 1
    async for blob in audio:
      self.heard.append(blob.data)
      yield UserTurnFinished(text=blob.data.decode())


class FakeTts(LiveEgress):
  """``str -> EgressEvent``. Synthesizes by prefixing, so output is traceable.

  Emits `AgentSpokenOutput` only for text that was actually turned into audio,
  because `AgentSpokenOutput` -- not the model's generated text -- is what
  history records.
  """

  PREFIX = b'<tts>'

  def __init__(self, *, fail_on: Optional[str] = None) -> None:
    self.synthesized: list[str] = []
    # Raises instead of synthesizing this chunk.
    self.fail_on = fail_on

  async def __call__(
      self, text: AsyncIterator[str], *, cancel: asyncio.Event
  ) -> AsyncGenerator[AudioChunk | AgentSpokenOutput, None]:
    async for chunk in text:
      if cancel.is_set():
        return
      if chunk == self.fail_on:
        raise RuntimeError(f'TTS failed on {chunk!r}')
      self.synthesized.append(chunk)
      yield AudioChunk(
          blob=types.Blob(
              data=self.PREFIX + chunk.encode(), mime_type=_MIME_TYPE
          )
      )
      yield AgentSpokenOutput(text=chunk)


class ScriptedReasoner(BaseLlm):
  """A text model that replays one scripted response list per turn, in order.

  `MockModel` yields a single response per call, which cannot express the
  streaming contract cascade has to honour: N fragments marked
  ``partial=True`` followed by an aggregate marked ``partial=False`` that
  repeats all of them (base_llm.py:163-193).
  """

  model: str = 'scripted'
  turns: list[list[LlmResponse]] = []
  requests: list[LlmRequest] = []

  @override
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    self.requests.append(llm_request)
    turn = len(self.requests) - 1
    for response in self.turns[turn] if turn < len(self.turns) else []:
      yield response


class _Run(NamedTuple):
  """What one cascaded live run produced."""

  events: list[Event]
  session: Session


def _audio_parts(events: list[Event]) -> list[bytes]:
  """The audio payloads a client would have played."""
  return [
      part.inline_data.data
      for event in events
      if event.content and event.content.parts
      for part in event.content.parts
      if part.inline_data
  ]


def _spoken(events: list[Event]) -> str:
  """Everything the egress voiced, concatenated in the order it went out.

  Reads the partial per-sentence transcriptions; the final one per turn
  repeats them.
  """
  return ''.join(
      event.output_transcription.text
      for event in events
      if event.output_transcription and event.partial
  )


def _replayed(request: LlmRequest) -> list[str]:
  """History as the reasoner sees it: one entry per part, in order.

  Rendering the whole request rather than filtering it is deliberate -- the
  history defects are about duplication and ordering, and both are invisible
  to an assertion that only looks at one kind of part.
  """
  rendered: list[str] = []
  for content in request.contents or []:
    for part in content.parts or []:
      if part.text:
        rendered.append(f'{content.role}:{part.text}')
      elif part.function_call:
        rendered.append(f'call:{part.function_call.name}')
      elif part.function_response:
        rendered.append(f'response:{part.function_response.name}')
  return rendered


def _model_text(text: str, *, partial: Optional[bool] = None) -> LlmResponse:
  """A text response, optionally marked as a streaming fragment."""
  return LlmResponse(
      content=types.Content(role='model', parts=[types.Part(text=text)]),
      partial=partial,
  )


def _call(
    name: str, args: dict[str, Any], *, partial: bool = False
) -> LlmResponse:
  """A response carrying one function call."""
  return LlmResponse(
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id='fc-1', name=name, args=args
                  )
              )
          ],
      ),
      partial=partial or None,
  )


def _text_and_call(
    text: str, name: str, args: dict[str, Any], *, thought: str = ''
) -> LlmResponse:
  """The aggregate that closes a turn mixing speech with a tool call.

  Shape taken from the streaming contract (base_llm.py:195-200): the final
  non-partial response repeats the turn's text *and* carries the call.
  """
  parts = [
      types.Part(text=text),
      types.Part(
          function_call=types.FunctionCall(id='fc-1', name=name, args=args)
      ),
  ]
  if thought:
    parts.insert(0, types.Part(text=thought, thought=True))
  return LlmResponse(content=types.Content(role='model', parts=parts))


def _user(text: str) -> types.Content:
  """Typed client text, the way the flow hands it to `_send_content`."""
  return types.Content(role='user', parts=[types.Part(text=text)])


def _blob(text: str) -> types.Blob:
  """One frame of client audio, whose payload `FakeStt` reads back as text."""
  return types.Blob(data=text.encode(), mime_type=_MIME_TYPE)


def _weather_tool() -> Any:
  """A tool that exists only so the live tool loop has something to close on."""

  def get_weather(city: str) -> str:
    """Reports the weather.

    Args:
      city: The city to report on.
    """
    del city
    return _ANSWER

  return get_weather


def _has_call(response: LlmResponse) -> bool:
  """Whether the response carries a function call."""
  return bool(response.content) and any(
      part.function_call for part in response.content.parts or []
  )


async def _drain_turn(connection: CascadeLiveConnection) -> list[LlmResponse]:
  """Reads one turn off a directly driven connection, as the flow would.

  `_run_live` drives whole runs through audio, which cannot express a partial
  client update -- only `_send_content` can, so tests for it hold the
  connection themselves and consume `receive()` in the flow's place.

  Returns the responses read, up to the turn's completion or its function
  call, which pauses the turn until the tool response arrives.
  """
  responses: list[LlmResponse] = []

  async def _read() -> None:
    async with aclosing(connection.receive()) as agen:
      async for response in agen:
        responses.append(response)
        if response.turn_complete or _has_call(response):
          return

  await asyncio.wait_for(_read(), timeout=_TIMEOUT_S)
  return responses


async def _run_live(
    reasoner: BaseLlm,
    stt: FakeStt,
    tts: FakeTts,
    *,
    utterances: tuple[str, ...] = (_UTTERANCE,),
    tools: Optional[list[Any]] = None,
) -> _Run:
  """Speaks each utterance in turn and collects what the client would see.

  Returns once the last turn completes. The session is re-read afterwards so
  tests can assert on what was *persisted* rather than only on what was
  streamed - the two differ for partial events (runners.py:1743).
  """
  agent = Agent(
      name='root_agent',
      # Live composition is expressed entirely through `model`: `CascadeLive`
      # is itself a BaseLlm wrapping a plain text reasoner plus two transforms.
      model=CascadeLive(model=reasoner, stt=stt, tts=tts),
      tools=tools or [],
  )

  session_service = InMemorySessionService()
  session = await session_service.create_session(app_name='app', user_id='u')
  runner = Runner(app_name='app', agent=agent, session_service=session_service)

  live_request_queue = LiveRequestQueue()
  pending = list(utterances)
  live_request_queue.send_realtime(_blob(pending.pop(0)))

  events: list[Event] = []
  overran = False

  async def _consume() -> None:
    nonlocal overran
    async with aclosing(
        runner.run_live(
            user_id='u',
            session_id=session.id,
            live_request_queue=live_request_queue,
            run_config=RunConfig(response_modalities=['AUDIO']),
        )
    ) as agen:
      async for event in agen:
        events.append(event)
        if len(events) >= _MAX_EVENTS:
          overran = True
          return
        if event.turn_complete:
          if not pending:
            return
          live_request_queue.send_realtime(_blob(pending.pop(0)))

  try:
    await asyncio.wait_for(_consume(), timeout=_TIMEOUT_S)
  except asyncio.TimeoutError:
    pytest.fail(f'Cascaded run never completed. Got events: {events}')

  # Say so outright: a truncated event list otherwise surfaces as a baffling
  # assertion diff rather than "the run produced more events than expected".
  if overran:
    pytest.fail(
        f'Cascaded run exceeded {_MAX_EVENTS} events without completing.'
        f' Got events: {events}'
    )

  return _Run(
      events=events,
      session=await session_service.get_session(
          app_name='app', user_id='u', session_id=session.id
      ),
  )


@pytest.mark.asyncio
async def test_cascade_reasons_in_text_between_stt_and_tts():
  """One utterance in as audio, one answer out as audio, text in between."""
  stt, tts = FakeStt(), FakeTts()
  model = testing_utils.MockModel.create([_ANSWER])

  run = await _run_live(model, stt, tts)

  # The ingress saw the client's audio, and only the client's audio.
  assert stt.heard == [_UTTERANCE.encode()]

  # The reasoner was the agent's own text model, and it received the
  # transcript as an ordinary user turn -- indistinguishable from typed text.
  # One request for the one turn `_run_live` drove: `MockModel.connect` also
  # records a request, so this count is what proves the S2S connect path was
  # bypassed rather than merely wrapped.
  assert len(model.requests) == 1
  assert model.requests[0].contents[-1] == types.Content(
      role='user', parts=[types.Part(text=_UTTERANCE)]
  )

  # The egress synthesized the model's text, and the client got that audio.
  assert tts.synthesized == [_ANSWER]
  assert _audio_parts(run.events) == [FakeTts.PREFIX + _ANSWER.encode()]

  # The transcripts on both edges come for free in cascade: input transcription
  # *is* the STT output, output transcription *is* the pre-synthesis text.
  assert [
      e.input_transcription.text for e in run.events if e.input_transcription
  ] == [_UTTERANCE]
  assert [
      (e.output_transcription.text, e.output_transcription.finished, e.partial)
      for e in run.events
      if e.output_transcription
  ] == [(_ANSWER, False, True), (_ANSWER, True, False)]

  assert run.events[-1].turn_complete


@pytest.mark.asyncio
async def test_streamed_fragments_are_spoken_once():
  """The aggregate that closes a streamed turn repeats it - do not respeak it.

  Regression: with no guard the egress synthesized every fragment *and* the
  aggregate that repeats them, so the user heard the whole answer twice.
  """
  stt, tts = FakeStt(), FakeTts()
  reasoner = ScriptedReasoner(
      turns=[
          [
              _model_text('Hello ', partial=True),
              _model_text('world.', partial=True),
              _model_text('Hello world.', partial=False),
          ],
          [_model_text(_ANSWER)],
      ]
  )

  # The follow-up utterance is what makes history observable: the reasoner
  # sees the record of the first turn in the request for the second.
  run = await _run_live(
      reasoner, stt, tts, utterances=(_UTTERANCE, 'and tomorrow')
  )

  # Incremental synthesis survives - fragments were spoken as they arrived,
  # which is the whole point of streaming into TTS.
  assert tts.synthesized == ['Hello ', 'world.', _ANSWER]
  assert _audio_parts(run.events)[:2] == [
      FakeTts.PREFIX + b'Hello ',
      FakeTts.PREFIX + b'world.',
  ]
  assert _spoken(run.events) == 'Hello world.' + _ANSWER

  # And history records the turn once, not once per fragment plus a repeat.
  assert [
      part.text
      for content in reasoner.requests[-1].contents
      if content.role == 'model'
      for part in content.parts or []
      if part.text
  ] == ['Hello world.']


@pytest.mark.asyncio
async def test_output_transcription_is_partial_per_sentence_then_final():
  """Each sentence streams as a partial; the turn ends with one final.

  Mirrors GeminiLlmConnection. The runner persists only non-partial
  transcriptions, so the session holds the turn's speech once.
  """
  stt, tts = FakeStt(), FakeTts()
  reasoner = ScriptedReasoner(
      turns=[[
          _model_text('Hello ', partial=True),
          _model_text('world.', partial=True),
          _model_text('Hello world.', partial=False),
      ]]
  )

  run = await _run_live(reasoner, stt, tts)

  assert [
      (e.output_transcription.text, e.output_transcription.finished, e.partial)
      for e in run.events
      if e.output_transcription
  ] == [
      ('Hello ', False, True),
      ('world.', False, True),
      ('Hello world.', True, False),
  ]
  assert [
      e.output_transcription.text
      for e in run.session.events
      if e.output_transcription
  ] == ['Hello world.']


@pytest.mark.asyncio
async def test_single_non_partial_response_is_still_spoken():
  """A producer that never streams must not be filtered out wholesale."""
  stt, tts = FakeStt(), FakeTts()
  reasoner = ScriptedReasoner(turns=[[_model_text(_ANSWER, partial=False)]])

  run = await _run_live(reasoner, stt, tts)

  assert tts.synthesized == [_ANSWER]
  assert _audio_parts(run.events) == [FakeTts.PREFIX + _ANSWER.encode()]


@pytest.mark.asyncio
async def test_function_call_is_taken_from_the_aggregate_not_a_fragment():
  """A fragment's call may carry incomplete args and is never persisted.

  Regression: acting on the fragment invoked the tool with empty ``args`` and
  emitted a ``partial=True`` event, which `runners.py:1743` drops - leaving
  the session with a ``functionResponse`` and no matching ``functionCall``.
  """
  called_with: list[dict[str, Any]] = []

  def get_weather(city: str) -> str:
    """Reports the weather.

    Args:
      city: The city to report on.
    """
    called_with.append({'city': city})
    return _ANSWER

  stt, tts = FakeStt(), FakeTts()
  reasoner = ScriptedReasoner(
      turns=[
          [
              _call('get_weather', {}, partial=True),
              _call('get_weather', {'city': 'Tokyo'}),
          ],
          [_model_text(_ANSWER)],
      ]
  )

  run = await _run_live(reasoner, stt, tts, tools=[get_weather])

  assert called_with == [{'city': 'Tokyo'}]

  call_events = [e for e in run.events if e.get_function_calls()]
  assert len(call_events) == 1
  assert not call_events[0].partial
  assert call_events[0].get_function_calls()[0].args == {'city': 'Tokyo'}

  # The call has to reach the session too, or the recorded conversation holds
  # a response answering nothing and cannot be replayed.
  assert [
      call.name for e in run.session.events for call in e.get_function_calls()
  ] == ['get_weather']
  assert [
      response.name
      for e in run.session.events
      for response in e.get_function_responses()
  ] == ['get_weather']


@pytest.mark.asyncio
async def test_thoughts_are_not_spoken():
  """Chain of thought is reasoning, not speech: TTS must never see it."""
  stt, tts = FakeStt(), FakeTts()
  reasoner = ScriptedReasoner(
      turns=[[
          LlmResponse(
              content=types.Content(
                  role='model',
                  parts=[
                      types.Part(
                          text='The user wants the weather.', thought=True
                      ),
                      types.Part(text=_ANSWER),
                  ],
              )
          )
      ]]
  )

  run = await _run_live(reasoner, stt, tts)

  assert tts.synthesized == [_ANSWER]
  assert _audio_parts(run.events) == [FakeTts.PREFIX + _ANSWER.encode()]
  assert _spoken(run.events) == _ANSWER


@pytest.mark.asyncio
async def test_mixed_text_and_call_is_recorded_once_and_in_order():
  """Text spoken before a tool call is history once, and before the call.

  Regression: the aggregate that carries both was appended by `_stream_text`
  *and* the egress text by `_run_turn`, so the answer appeared twice - and the
  copy from the aggregate landed first, putting the call before the speech it
  followed.
  """
  stt, tts = FakeStt(), FakeTts()
  reasoner = ScriptedReasoner(
      turns=[
          [
              _model_text('Let me check. ', partial=True),
              _text_and_call(
                  'Let me check. ', 'get_weather', {'city': 'Tokyo'}
              ),
          ],
          [_model_text(_ANSWER)],
          [_model_text('Also sunny.')],
      ]
  )

  await _run_live(
      reasoner,
      stt,
      tts,
      utterances=(_UTTERANCE, 'and tomorrow'),
      tools=[_weather_tool()],
  )

  # The last request is what the reasoner is fed on the *next* turn, i.e. the
  # replayed conversation history -- which is where a duplicated or misordered
  # record shows up.
  assert _replayed(reasoner.requests[-1]) == [
      f'user:{_UTTERANCE}',
      'model:Let me check. ',
      'call:get_weather',
      'response:get_weather',
      f'model:{_ANSWER}',
      'user:and tomorrow',
  ]


@pytest.mark.asyncio
async def test_unstreamed_text_beside_a_call_is_spoken_then_recorded():
  """A model that emits its text and its call at once still says the text.

  `AgentSpokenOutput` is the only writer of assistant text to history, so text
  that never reaches the egress is text the turn forgets. Handing it to TTS is
  what keeps the preamble to a tool call both audible and recorded.
  """
  stt, tts = FakeStt(), FakeTts()
  reasoner = ScriptedReasoner(
      turns=[
          [_text_and_call('Let me check. ', 'get_weather', {'city': 'Tokyo'})],
          [_model_text(_ANSWER)],
          [_model_text('Also sunny.')],
      ]
  )

  run = await _run_live(
      reasoner,
      stt,
      tts,
      utterances=(_UTTERANCE, 'and tomorrow'),
      tools=[_weather_tool()],
  )

  assert tts.synthesized == ['Let me check. ', _ANSWER, 'Also sunny.']
  assert _audio_parts(run.events)[0] == FakeTts.PREFIX + b'Let me check. '
  assert _replayed(reasoner.requests[-1]) == [
      f'user:{_UTTERANCE}',
      'model:Let me check. ',
      'call:get_weather',
      'response:get_weather',
      f'model:{_ANSWER}',
      'user:and tomorrow',
  ]


@pytest.mark.asyncio
async def test_function_call_response_carries_only_the_call():
  """The emitted call drops the aggregate's text and thoughts.

  Text already reached the client as spoken output; repeating it in the call
  event would persist it a second time, as generated rather than spoken text.
  """
  stt, tts = FakeStt(), FakeTts()
  call = _text_and_call('Let me check. ', 'get_weather', {'city': 'Tokyo'})
  call.content.parts.insert(
      0, types.Part(text='The user wants the weather.', thought=True)
  )
  reasoner = ScriptedReasoner(turns=[[call], [_model_text(_ANSWER)]])

  run = await _run_live(reasoner, stt, tts, tools=[_weather_tool()])

  call_events = [e for e in run.events if e.get_function_calls()]
  assert len(call_events) == 1
  assert all(part.function_call for part in call_events[0].content.parts)
  # The spoken preamble is persisted once, as the turn's final transcription.
  assert not [
      part.text
      for e in run.session.events
      if e.author == 'root_agent' and e.content
      for part in e.content.parts or []
      if part.text
  ]
  assert [
      e.output_transcription.text
      for e in run.session.events
      if e.output_transcription
  ] == ['Let me check. ', _ANSWER]


@pytest.mark.asyncio
async def test_call_only_turn_still_records_the_call():
  """A turn that says nothing must still leave its call in history.

  The model goes straight to a function call without speaking any text first,
  so the egress emits no `AgentSpokenOutput` for the turn - and
  `AgentSpokenOutput` is what normally writes assistant text to history. This
  covers the path where there is no such event to write from: without the
  call, the function response replayed after it answers nothing and the
  reasoner cannot make sense of its own conversation.
  """
  stt, tts = FakeStt(), FakeTts()
  reasoner = ScriptedReasoner(
      turns=[
          [_call('get_weather', {'city': 'Tokyo'})],
          [_model_text(_ANSWER)],
          [_model_text('Also sunny.')],
      ]
  )

  await _run_live(
      reasoner,
      stt,
      tts,
      utterances=(_UTTERANCE, 'and tomorrow'),
      tools=[_weather_tool()],
  )

  assert tts.synthesized == [_ANSWER, 'Also sunny.']
  assert _replayed(reasoner.requests[-1]) == [
      f'user:{_UTTERANCE}',
      'call:get_weather',
      'response:get_weather',
      f'model:{_ANSWER}',
      'user:and tomorrow',
  ]


@pytest.mark.asyncio
async def test_thoughts_do_not_reach_history_via_the_aggregate():
  """Thought filtering has to hold for the aggregate, not just the egress.

  Regression: the aggregate behind a function call was appended verbatim, so
  a turn that thought before calling a tool read its own scratchpad back on
  the next turn as something it had said out loud.
  """
  stt, tts = FakeStt(), FakeTts()
  reasoner = ScriptedReasoner(
      turns=[
          [
              _model_text('Let me check. ', partial=True),
              _text_and_call(
                  'Let me check. ',
                  'get_weather',
                  {'city': 'Tokyo'},
                  thought='The user wants the weather.',
              ),
          ],
          [_model_text(_ANSWER)],
          [_model_text('Also sunny.')],
      ]
  )

  run = await _run_live(
      reasoner,
      stt,
      tts,
      utterances=(_UTTERANCE, 'and tomorrow'),
      tools=[_weather_tool()],
  )

  assert 'The user wants the weather.' not in _spoken(run.events)
  assert not [
      entry
      for entry in _replayed(reasoner.requests[-1])
      if 'wants the weather' in entry
  ]


@pytest.mark.asyncio
async def test_partial_client_content_is_not_recorded():
  """Fragments of a client's text are not history; the final content is.

  The framework refuses to persist a partial anywhere else
  (base_llm_flow.py:1037, runners.py:1743). Recording them here would replay
  every prefix of the user's message back to the reasoner alongside the
  message itself.
  """
  stt, tts = FakeStt(), FakeTts()
  reasoner = ScriptedReasoner(
      turns=[[_model_text(_ANSWER)], [_model_text('Also sunny.')]]
  )
  connection = CascadeLiveConnection(
      LlmRequest(),
      reasoner,
      stt,
      tts,
  )

  # A client streaming typed text: prefixes, then the whole message.
  await connection._send_content(_user('what is'), partial=True)
  await connection._send_content(_user('what is the'), partial=True)
  await connection._send_content(_user(_UTTERANCE))
  await _drain_turn(connection)

  # The follow-up is what makes history observable, as elsewhere here.
  await connection._send_content(_user('and tomorrow'))
  await _drain_turn(connection)
  await connection.close()

  assert _replayed(reasoner.requests[-1]) == [
      f'user:{_UTTERANCE}',
      f'model:{_ANSWER}',
      'user:and tomorrow',
  ]


class _ClosingReasoner(ScriptedReasoner):
  """Records when its response stream is closed."""

  closed: bool = False

  @override
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    try:
      async for response in super().generate_content_async(llm_request, stream):
        yield response
    finally:
      self.closed = True


class _FirstSentenceTts(LiveEgress):
  """Speaks the first text chunk, then stops reading and returns."""

  async def __call__(
      self, text: AsyncIterator[str], *, cancel: asyncio.Event
  ) -> AsyncGenerator[AudioChunk | AgentSpokenOutput, None]:
    async for chunk in text:
      yield AgentSpokenOutput(text=chunk)
      return


@pytest.mark.asyncio
async def test_tts_returning_early_closes_the_llm_stream(monkeypatch):
  """A TTS that stops early ends the turn and closes the LLM stream."""
  reasoner = _ClosingReasoner(
      turns=[[
          _model_text('One.', partial=True),
          _model_text(' Two.', partial=True),
          _model_text('One. Two.'),
      ]]
  )
  connection = CascadeLiveConnection(
      LlmRequest(), reasoner, FakeStt(), _FirstSentenceTts()
  )
  # Sampled when `turn_complete` is emitted: checking after the turn would
  # also pass when garbage collection closes the abandoned stream.
  closed_at_turn_complete = []
  emit = connection._emit

  def _emit(response: LlmResponse) -> None:
    if response.turn_complete:
      closed_at_turn_complete.append(reasoner.closed)
    emit(response)

  monkeypatch.setattr(connection, '_emit', _emit)

  await connection._send_content(_user(_UTTERANCE))
  await _drain_turn(connection)
  await connection.close()

  assert closed_at_turn_complete == [True]


@pytest.mark.asyncio
async def test_tts_failure_before_a_call_drops_the_call():
  """A call whose preamble the TTS failed on does not run on a later turn.

  Regression: the call was stashed before its preamble was spoken and cleared
  only on success, so the next turn emitted it even though it made no call.
  """
  reasoner = ScriptedReasoner(
      turns=[
          [_text_and_call('Let me check. ', 'get_weather', {'city': 'Tokyo'})],
          [_model_text(_ANSWER)],
      ]
  )
  connection = CascadeLiveConnection(
      LlmRequest(), reasoner, FakeStt(), FakeTts(fail_on='Let me check. ')
  )

  await connection._send_content(_user(_UTTERANCE))
  failed = await _drain_turn(connection)
  await connection._send_content(_user('and tomorrow'))
  answered = await _drain_turn(connection)
  await connection.close()

  assert failed[-1].error_code == 'CASCADE_TURN_FAILED'
  assert not [r for r in failed + answered if _has_call(r)]
  assert answered[-1].turn_complete
  assert _replayed(reasoner.requests[-1]) == [
      f'user:{_UTTERANCE}',
      'user:and tomorrow',
  ]


@pytest.mark.asyncio
async def test_tts_failure_keeps_sentences_already_spoken():
  """Sentences voiced before a TTS failure reach history and transcription.

  Regression: spoken text was recorded only when the turn succeeded, so the
  next turn did not know what the user had already heard.
  """
  reasoner = ScriptedReasoner(
      turns=[
          [
              _model_text('First. ', partial=True),
              _model_text('Second.', partial=True),
              _model_text('First. Second.'),
          ],
          [_model_text(_ANSWER)],
      ]
  )
  connection = CascadeLiveConnection(
      LlmRequest(), reasoner, FakeStt(), FakeTts(fail_on='Second.')
  )

  await connection._send_content(_user(_UTTERANCE))
  failed = await _drain_turn(connection)
  await connection._send_content(_user('and tomorrow'))
  await _drain_turn(connection)
  await connection.close()

  assert [
      (r.output_transcription.text, r.output_transcription.finished)
      for r in failed
      if r.output_transcription
  ] == [('First. ', False), ('First. ', True)]
  assert failed[-1].error_code == 'CASCADE_TURN_FAILED'
  assert _replayed(reasoner.requests[-1]) == [
      f'user:{_UTTERANCE}',
      'model:First. ',
      'user:and tomorrow',
  ]


@pytest.mark.parametrize(
    'segment_end',
    [
        types.ActivityEnd(),
        types.LiveClientRealtimeInput(audio_stream_end=True),
    ],
    ids=['activity_end', 'audio_stream_end'],
)
@pytest.mark.asyncio
async def test_audio_after_segment_end_opens_a_new_stt_session(segment_end):
  """Ending an audio segment finalizes it without ending ingress for good.

  Both signals end the STT stream so the transform can flush; the next audio
  must still be transcribed, in a fresh session.
  """
  stt, tts = FakeStt(), FakeTts()
  reasoner = ScriptedReasoner(
      turns=[[_model_text(_ANSWER)], [_model_text('Also sunny.')]]
  )
  connection = CascadeLiveConnection(
      LlmRequest(),
      reasoner,
      stt,
      tts,
  )

  await connection.send_realtime(_blob(_UTTERANCE))
  await _drain_turn(connection)
  await connection.send_realtime(segment_end)
  await connection.send_realtime(_blob('and tomorrow'))
  await _drain_turn(connection)
  await connection.close()

  assert stt.heard == [_UTTERANCE.encode(), b'and tomorrow']
  assert stt.sessions == 2


@pytest.mark.asyncio
async def test_segment_end_without_audio_opens_no_stt_session():
  """A segment end with no audio in it does not start an empty STT session."""
  stt, tts = FakeStt(), FakeTts()
  reasoner = ScriptedReasoner(turns=[[_model_text(_ANSWER)]])
  connection = CascadeLiveConnection(
      LlmRequest(),
      reasoner,
      stt,
      tts,
  )

  await connection.send_realtime(types.ActivityEnd())
  await connection.send_realtime(_blob(_UTTERANCE))
  await _drain_turn(connection)
  await connection.close()

  assert stt.heard == [_UTTERANCE.encode()]
  assert stt.sessions == 1


@pytest.mark.asyncio
async def test_video_frames_are_dropped_not_sent_to_stt(caplog):
  """A camera frame mid-session is dropped and does not end the session."""
  stt, tts = FakeStt(), FakeTts()
  reasoner = ScriptedReasoner(
      turns=[[_model_text(_ANSWER)], [_model_text('Also sunny.')]]
  )
  connection = CascadeLiveConnection(
      LlmRequest(),
      reasoner,
      stt,
      tts,
  )
  frame = types.Blob(data=b'\xff\xd8', mime_type='image/jpeg')

  await connection.send_realtime(_blob(_UTTERANCE))
  await _drain_turn(connection)
  with caplog.at_level(logging.WARNING):
    await connection.send_realtime(frame)
    await connection.send_realtime(frame)
  await connection.send_realtime(_blob('and tomorrow'))
  await _drain_turn(connection)
  await connection.close()

  assert stt.heard == [_UTTERANCE.encode(), b'and tomorrow']
  assert stt.sessions == 1
  assert len(reasoner.requests) == 2
  dropped = [r for r in caplog.records if 'image/jpeg' in r.getMessage()]
  assert len(dropped) == 1


@pytest.mark.asyncio
async def test_receive_reports_close_instead_of_spinning():
  """A closed connection must end the flow's receive loop, not re-arm it.

  `_receive_from_model` re-enters `receive()` in a ``while True:``
  (base_llm_flow.py:1085), so a sentinel left on the queue is read again
  immediately and the flow spins forever. This mirrors that loop, under a
  timeout so a regression fails instead of hanging the suite.
  """
  stt, tts = FakeStt(), FakeTts()
  reasoner = ScriptedReasoner(turns=[[_model_text(_ANSWER)]])
  connection = CascadeLiveConnection(
      LlmRequest(),
      reasoner,
      stt,
      tts,
  )

  await connection.close()

  async def _drain_like_the_flow() -> None:
    while True:
      async with aclosing(connection.receive()) as agen:
        async for _ in agen:
          pass
      await asyncio.sleep(0)

  with pytest.raises(ConnectionClosedOK):
    await asyncio.wait_for(_drain_like_the_flow(), timeout=2.0)


@pytest.mark.asyncio
async def test_closing_the_queue_ends_the_live_run():
  """Closing the client queue ends `run_live` rather than stranding it."""
  stt, tts = FakeStt(), FakeTts()
  reasoner = ScriptedReasoner(turns=[[_model_text(_ANSWER)]])
  agent = Agent(
      name='root_agent',
      model=CascadeLive(model=reasoner, stt=stt, tts=tts),
  )

  session_service = InMemorySessionService()
  session = await session_service.create_session(app_name='app', user_id='u')
  runner = Runner(app_name='app', agent=agent, session_service=session_service)

  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(_blob(_UTTERANCE))

  events: list[Event] = []

  async def _consume_to_completion() -> None:
    # Deliberately no early return: this ends only when `run_live` does.
    async with aclosing(
        runner.run_live(
            user_id='u',
            session_id=session.id,
            live_request_queue=live_request_queue,
            run_config=RunConfig(response_modalities=['AUDIO']),
        )
    ) as agen:
      async for event in agen:
        events.append(event)
        if event.turn_complete:
          live_request_queue.close()

  try:
    await asyncio.wait_for(_consume_to_completion(), timeout=_TIMEOUT_S)
  except asyncio.TimeoutError:
    pytest.fail(f'run_live never returned after close. Got events: {events}')

  assert events[-1].turn_complete


class _FailingStt(LiveIngress):
  """An STT that fails on the first audio, e.g. on a bad API key."""

  async def __call__(
      self, audio: AsyncIterator[types.Blob]
  ) -> AsyncGenerator[UserTurnFinished, None]:
    async for _ in audio:
      raise RuntimeError('invalid API key')
    yield UserTurnFinished(text='')  # pragma: no cover


@pytest.mark.asyncio
async def test_stt_failure_is_raised_from_run_live():
  """An STT failure ends `run_live` with the error, not as a normal close."""
  reasoner = ScriptedReasoner(turns=[[_model_text(_ANSWER)]])
  agent = Agent(
      name='root_agent',
      model=CascadeLive(model=reasoner, stt=_FailingStt(), tts=FakeTts()),
  )

  session_service = InMemorySessionService()
  session = await session_service.create_session(app_name='app', user_id='u')
  runner = Runner(app_name='app', agent=agent, session_service=session_service)

  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(_blob(_UTTERANCE))

  async def _consume() -> None:
    async with aclosing(
        runner.run_live(
            user_id='u',
            session_id=session.id,
            live_request_queue=live_request_queue,
            run_config=RunConfig(response_modalities=['AUDIO']),
        )
    ) as agen:
      async for _ in agen:
        pass

  with pytest.raises(RuntimeError, match='invalid API key'):
    await asyncio.wait_for(_consume(), timeout=_TIMEOUT_S)
  assert not reasoner.requests
