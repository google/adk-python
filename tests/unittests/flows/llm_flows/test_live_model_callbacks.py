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

"""Tests for before/after model callbacks on the Live API flow."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any
from typing import Iterable
from typing import Optional
from unittest import mock

from google.adk.agents.llm_agent import Agent
from google.adk.events.event import Event
from google.adk.flows.llm_flows import contents as contents_processor
from google.adk.flows.llm_flows.base_llm_flow import _ReconnectMode
from google.adk.flows.llm_flows.base_llm_flow import _ReconnectSentinel
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.live import LiveRequest
from google.adk.live import LiveRequestQueue
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.function_tool import FunctionTool
from google.genai import types
import pytest

from ... import testing_utils


class _Flow(BaseLlmFlow):
  pass


class _StopReceive(Exception):
  """Breaks the ``_receive_from_model`` outer loop once turns are exhausted."""


def _fake_connection(*turns: list[LlmResponse]):
  """Builds a fake connection serving one turn per ``receive()`` call.

  When the batches run out, ``_StopReceive`` ends the loop.
  """
  remaining = [list(turn) for turn in turns]

  async def _receive():
    if not remaining:
      raise _StopReceive()
    for response in remaining.pop(0):
      yield response
    if not remaining:
      raise _StopReceive()

  connection = mock.AsyncMock()
  connection.receive = mock.Mock(side_effect=_receive)
  return connection


class _LiveMockModel(testing_utils.MockModel):
  """Yields a fake connection on ``connect()`` and records incoming requests.

  When ``run_live`` restarts after a blocked turn, it calls ``connect()``
  again. Reusing one connection across calls lets tests script both sessions
  in order and check each request in ``self.requests``.
  """

  connection: Any = None

  @contextlib.asynccontextmanager
  async def connect(self, llm_request: LlmRequest):
    self.requests.append(llm_request)
    yield self.connection


def _fake_send_connection():
  connection = mock.AsyncMock()
  connection._send_content = mock.AsyncMock()
  connection.send_realtime = mock.AsyncMock()
  connection.close = mock.AsyncMock()
  return connection


async def _make_context(*, plugins=None, before=None, after=None):
  model = testing_utils.MockModel.create(responses=[])
  agent = Agent(
      name='test_agent',
      model=model,
      before_model_callback=before,
      after_model_callback=after,
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, plugins=plugins or []
  )
  invocation_context.live_request_queue = LiveRequestQueue()
  return invocation_context


async def _append_user_event(invocation_context, text: str) -> None:
  """Records a user turn on the session, as the live flow would."""
  await invocation_context.session_service.append_event(
      session=invocation_context.session,
      event=Event(
          invocation_id=invocation_context.invocation_id,
          author='user',
          content=types.Content(role='user', parts=[types.Part(text=text)]),
      ),
  )


def _content_texts(
    contents: Optional[Iterable[Optional[types.Content]]],
) -> list[str]:
  """Flattens the text parts of a sequence of contents."""
  return [
      part.text
      for content in contents or []
      if content
      for part in content.parts or []
      if part.text
  ]


def _event_texts(events: Iterable[Event]) -> list[str]:
  """Flattens the text parts of every event carrying content."""
  return _content_texts(event.content for event in events)


async def _collect_receive(
    flow, invocation_context, connection, llm_request=None
):
  """Drains ``_receive_from_model``, returning its events and the request."""
  llm_request = llm_request if llm_request is not None else LlmRequest()
  events = []
  try:
    async with testing_utils.Aclosing(
        flow._receive_from_model(
            connection, 'e1', invocation_context, llm_request
        )
    ) as agen:
      async for event in agen:
        events.append(event)
  except _StopReceive:
    pass
  return events, llm_request


async def _drive_send(flow, invocation_context, requests, llm_request=None):
  """Enqueues requests and runs ``_send_to_model`` to the end.

  Returns a ``(connection, delivered)`` pair, where ``delivered`` holds the
  events ``_send_to_model`` put on the invocation's event queue.

  ``_send_to_model`` runs in the send task, which cannot yield events to the
  caller of ``run_live()``, so it enqueues them instead. This stands in for the
  Runner as the queue's consumer: it appends non-partial events to the session
  and acknowledges them, which ``_enqueue_event`` blocks on.
  """
  invocation_context._event_queue = asyncio.Queue()
  delivered: list[Event] = []

  async def _consume() -> None:
    while True:
      event, processed = await invocation_context._event_queue.get()
      delivered.append(event)
      if not event.partial:
        await invocation_context.session_service.append_event(
            session=invocation_context.session, event=event
        )
      if processed is not None:
        processed.set()

  consumer = asyncio.create_task(_consume())
  for request in requests:
    invocation_context.live_request_queue.send(request)
  invocation_context.live_request_queue.close()
  connection = _fake_send_connection()
  try:
    await flow._send_to_model(
        connection,
        invocation_context,
        llm_request if llm_request is not None else LlmRequest(),
    )
  finally:
    consumer.cancel()
  return connection, delivered


# --- Builders ---------------------------------------------------------------


def _content_request(text: str, *, partial: bool = False) -> LiveRequest:
  return LiveRequest(
      content=types.Content(role='user', parts=[types.Part(text=text)]),
      partial=partial,
  )


def noop_tool() -> dict:
  """A tool that does nothing."""
  return {}


def _connection_llm_request() -> LlmRequest:
  """A populated ``LlmRequest``, as the live connection was opened with."""
  llm_request = LlmRequest(
      model='gemini-live-model',
      contents=[
          types.Content(role='user', parts=[types.Part(text='earlier turn')])
      ],
      config=types.GenerateContentConfig(temperature=0.25),
      live_connect_config=types.LiveConnectConfig(
          response_modalities=[types.Modality.AUDIO]
      ),
  )
  llm_request.tools_dict = {'noop_tool': FunctionTool(noop_tool)}
  return llm_request


def _output_transcription(text: str, *, finished: bool = False) -> LlmResponse:
  """Model output transcription."""
  return LlmResponse(
      output_transcription=types.Transcription(text=text, finished=finished),
      partial=not finished,
  )


def _input_transcription(text: str, *, finished: bool) -> LlmResponse:
  """Model transcription of the user's spoken input."""
  return LlmResponse(
      input_transcription=types.Transcription(text=text, finished=finished),
      partial=not finished,
  )


def _recorder():
  """Returns ``(callback, seen)``: a pass-through callback and its record."""
  seen = []

  def callback(**kwargs):
    seen.append(kwargs)
    return None

  return callback, seen


def _blocker(text: str):
  """Returns a callback that blocks the turn with a replacement response."""

  def callback(**kwargs):
    del kwargs
    return LlmResponse(
        content=types.Content(role='model', parts=[types.Part(text=text)])
    )

  return callback


# --- Before model callback: live text input ---------------------------------


@pytest.mark.asyncio
async def test_before_model_callback_fires_once_for_text_input():
  """A typed live message runs before_model_callback exactly once."""
  before, seen = _recorder()
  invocation_context = await _make_context(before=before)

  await _drive_send(_Flow(), invocation_context, [_content_request('hello')])

  assert len(seen) == 1
  assert seen[0]['llm_request'].contents[0].parts[0].text == 'hello'


@pytest.mark.asyncio
async def test_before_model_callback_receives_connection_fields_for_text():
  """before_model_callback sees the live model, config and tools."""
  before, seen = _recorder()
  invocation_context = await _make_context(before=before)

  await _drive_send(
      _Flow(),
      invocation_context,
      [_content_request('hello')],
      _connection_llm_request(),
  )

  request = seen[0]['llm_request']
  assert request.model == 'gemini-live-model'
  assert request.config.temperature == 0.25
  assert list(request.tools_dict) == ['noop_tool']
  assert request.live_connect_config.response_modalities == [
      types.Modality.AUDIO
  ]


@pytest.mark.asyncio
async def test_before_model_callback_skips_partial_text_input():
  """Partial typed messages do not run before_model_callback."""
  before, seen = _recorder()
  invocation_context = await _make_context(before=before)

  await _drive_send(
      _Flow(),
      invocation_context,
      [
          _content_request('hel', partial=True),
          _content_request('hello'),
      ],
  )

  assert len(seen) == 1
  assert seen[0]['llm_request'].contents[0].parts[0].text == 'hello'


@pytest.mark.asyncio
async def test_blocked_text_input_keeps_user_event_in_session():
  """Blocked typed input is still recorded in the session."""
  invocation_context = await _make_context(before=_blocker('blocked'))

  await _drive_send(_Flow(), invocation_context, [_content_request('hello')])

  user_events = [
      event
      for event in invocation_context.session.events
      if event.author == 'user'
  ]
  assert _event_texts(user_events) == ['hello']


@pytest.mark.asyncio
async def test_blocked_text_input_appends_blocked_event_to_session():
  """A blocked typed turn records the callback's reply in the session."""
  invocation_context = await _make_context(before=_blocker('blocked'))

  await _drive_send(_Flow(), invocation_context, [_content_request('hello')])

  assert 'blocked' in _event_texts(invocation_context.session.events)


@pytest.mark.asyncio
async def test_blocked_text_input_is_not_sent_to_the_model():
  """Blocked typed input is never forwarded to the model."""
  invocation_context = await _make_context(before=_blocker('blocked'))

  connection, _ = await _drive_send(
      _Flow(), invocation_context, [_content_request('hello')]
  )

  connection._send_content.assert_not_called()


@pytest.mark.asyncio
async def test_blocked_text_input_is_delivered_to_the_runner():
  """A blocked typed turn reaches the caller of ``run_live()``.

  ``_send_to_model`` runs in the send task, so appending to the session alone
  would persist the event without ever surfacing it. Enqueuing it puts it on
  the stream the Runner merges into what ``run_live()`` yields.
  """
  invocation_context = await _make_context(before=_blocker('blocked'))

  _, delivered = await _drive_send(
      _Flow(), invocation_context, [_content_request('hello')]
  )

  assert _event_texts(delivered) == ['blocked']


# --- Before model callback: live audio input --------------------------------


@pytest.mark.asyncio
async def test_before_model_callback_fires_once_for_spoken_input():
  """A completed spoken utterance runs before_model_callback once."""
  before, seen = _recorder()
  invocation_context = await _make_context(before=before)
  connection = _fake_connection([_input_transcription('hello', finished=True)])

  await _collect_receive(_Flow(), invocation_context, connection)

  assert len(seen) == 1
  assert seen[0]['llm_request'].contents[0].parts[0].text == 'hello'


@pytest.mark.asyncio
async def test_before_model_callback_receives_connection_fields_for_speech():
  """Spoken input callback also sees the live model, config and tools."""
  before, seen = _recorder()
  invocation_context = await _make_context(before=before)
  connection = _fake_connection([_input_transcription('hello', finished=True)])

  await _collect_receive(
      _Flow(), invocation_context, connection, _connection_llm_request()
  )

  request = seen[0]['llm_request']
  assert request.model == 'gemini-live-model'
  assert request.config.temperature == 0.25
  assert list(request.tools_dict) == ['noop_tool']
  # Only the transcribed utterance, not the request's prior history.
  assert _content_texts(request.contents) == ['hello']


@pytest.mark.asyncio
async def test_before_model_callback_skips_partial_input_transcription():
  """Partial speech transcriptions do not run before_model_callback."""
  before, seen = _recorder()
  invocation_context = await _make_context(before=before)
  connection = _fake_connection([
      _input_transcription('hello', finished=False),
      _input_transcription(' there', finished=False),
      _input_transcription('hello there', finished=True),
  ])

  await _collect_receive(_Flow(), invocation_context, connection)

  assert len(seen) == 1
  assert seen[0]['llm_request'].contents[0].parts[0].text == 'hello there'


@pytest.mark.asyncio
async def test_blocked_spoken_input_still_yields_transcription_event():
  """Blocking spoken input still emits what the user said."""
  invocation_context = await _make_context(before=_blocker('blocked'))
  connection = _fake_connection(
      [_input_transcription('hello there', finished=True)]
  )

  events, _ = await _collect_receive(_Flow(), invocation_context, connection)

  transcribed = [
      event.input_transcription.text
      for event in events
      if event.input_transcription
  ]
  assert transcribed == ['hello there']


@pytest.mark.asyncio
async def test_blocked_spoken_input_yields_blocked_event():
  """Blocking spoken input emits the callback's replacement reply."""
  invocation_context = await _make_context(before=_blocker('blocked'))
  connection = _fake_connection(
      [_input_transcription('hello there', finished=True)]
  )

  events, _ = await _collect_receive(_Flow(), invocation_context, connection)

  blocked = [event for event in events if _event_texts([event]) == ['blocked']]
  assert len(blocked) == 1
  assert blocked[0].turn_complete is True


@pytest.mark.asyncio
async def test_blocked_spoken_input_restarts_the_session():
  """Blocking spoken input ends the live session and restarts it."""
  invocation_context = await _make_context(before=_blocker('blocked'))
  connection = _fake_connection(
      [_input_transcription('hello there', finished=True)]
  )

  events, _ = await _collect_receive(_Flow(), invocation_context, connection)

  assert isinstance(events[-1], _ReconnectSentinel)
  assert events[-1].mode is _ReconnectMode.RESTART


# --- After model callback: model output -------------------------------------


@pytest.mark.asyncio
async def test_after_model_callback_fires_on_partial_output():
  """Each streamed output fragment runs after_model_callback."""
  after, seen = _recorder()
  invocation_context = await _make_context(after=after)
  connection = _fake_connection([
      _output_transcription('Hello'),
      _output_transcription(' there'),
  ])

  await _collect_receive(_Flow(), invocation_context, connection)

  assert len(seen) == 2
  assert all(call['llm_response'].partial for call in seen)


@pytest.mark.asyncio
async def test_after_model_callback_fires_for_output_transcription():
  """Transcribed audio output runs after_model_callback."""
  after, seen = _recorder()
  invocation_context = await _make_context(after=after)
  connection = _fake_connection([_output_transcription('Hello')])

  await _collect_receive(_Flow(), invocation_context, connection)

  assert len(seen) == 1
  assert seen[0]['llm_response'].output_transcription.text == 'Hello'


@pytest.mark.parametrize(
    'response',
    [
        pytest.param(
            LlmResponse(
                input_transcription=types.Transcription(
                    text='hello', finished=True
                )
            ),
            id='input_transcription',
        ),
        pytest.param(
            LlmResponse(
                content=types.Content(
                    role='model',
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                name='noop_tool', args={}
                            )
                        )
                    ],
                )
            ),
            id='tool_call',
        ),
        pytest.param(LlmResponse(turn_complete=True), id='turn_complete'),
        pytest.param(LlmResponse(interrupted=True), id='interrupted'),
        pytest.param(
            LlmResponse(
                content=types.Content(
                    role='model', parts=[types.Part(text='Hello')]
                ),
                partial=True,
            ),
            id='text_content',
        ),
    ],
)
@pytest.mark.asyncio
async def test_after_model_callback_does_not_fire_for(response):
  """Responses with no output transcription do not run after_model_callback."""
  after, seen = _recorder()
  invocation_context = await _make_context(after=after)
  connection = _fake_connection([response])
  # Function calls are resolved against ``tools_dict``, so the tool has to be
  # registered for a response carrying one to reach postprocessing.
  llm_request = LlmRequest()
  llm_request.tools_dict = {'noop_tool': FunctionTool(noop_tool)}

  await _collect_receive(_Flow(), invocation_context, connection, llm_request)

  assert seen == []


@pytest.mark.asyncio
async def test_after_model_callback_receives_accumulated_output_transcription():
  """after_model_callback sees the accumulated spoken answer so far."""
  after, seen = _recorder()
  invocation_context = await _make_context(after=after)
  connection = _fake_connection([
      _output_transcription('Hello'),
      _output_transcription(' there'),
  ])

  await _collect_receive(_Flow(), invocation_context, connection)

  assert [call['llm_response'].output_transcription.text for call in seen] == [
      'Hello',
      'Hello there',
  ]


@pytest.mark.asyncio
async def test_after_model_callback_does_not_fire_when_nothing_accumulated():
  """The buffer is cleared on ``finished``, so there is nothing left to screen."""
  after, seen = _recorder()
  invocation_context = await _make_context(after=after)
  connection = _fake_connection([
      _output_transcription('How can I help?', finished=True),
  ])

  await _collect_receive(_Flow(), invocation_context, connection)

  assert seen == []


@pytest.mark.asyncio
async def test_blocked_model_output_yields_blocked_event():
  """Blocking model output emits the callback's replacement reply."""
  invocation_context = await _make_context(after=_blocker('blocked'))
  connection = _fake_connection([_output_transcription('a secret')])

  events, _ = await _collect_receive(_Flow(), invocation_context, connection)

  blocked = [event for event in events if _event_texts([event]) == ['blocked']]
  assert len(blocked) == 1
  assert blocked[0].turn_complete is True


@pytest.mark.asyncio
async def test_blocked_model_output_restarts_the_session():
  """Blocking model output ends the live session and restarts it."""
  invocation_context = await _make_context(after=_blocker('blocked'))
  connection = _fake_connection([_output_transcription('a secret')])

  events, _ = await _collect_receive(_Flow(), invocation_context, connection)

  assert isinstance(events[-1], _ReconnectSentinel)
  assert events[-1].mode is _ReconnectMode.RESTART


@pytest.mark.asyncio
async def test_blocked_model_output_suppresses_remaining_turn():
  """Nothing else from a blocked answer reaches the user."""
  invocation_context = await _make_context(after=_blocker('blocked'))
  connection = _fake_connection([
      _output_transcription('a secret'),
      _output_transcription(' and more'),
      LlmResponse(turn_complete=True),
  ])

  events, _ = await _collect_receive(_Flow(), invocation_context, connection)

  assert isinstance(events[-1], _ReconnectSentinel)
  assert events[-1].mode is _ReconnectMode.RESTART
  assert not [event for event in events if event.output_transcription]
  assert _event_texts([event for event in events if event.turn_complete]) == [
      'blocked'
  ]


# --- Restarting after a blocked turn ----------------------------------------


@pytest.mark.asyncio
async def test_restart_opens_a_new_session_with_rebuilt_history():
  """A blocked turn reconnects with the session history."""
  connection = _fake_connection([_output_transcription('a secret')])
  model = _LiveMockModel(model='mock', responses=[], connection=connection)
  invocation_context = await _make_context(after=_blocker('blocked'))
  invocation_context.agent.model = model
  invocation_context.live_session_resumption_handle = 'handle-123'
  await _append_user_event(invocation_context, 'said during the session')

  # `_Flow` is a bare BaseLlmFlow with no request processors. The rebuild under
  # test is the contents processor's work, so register it.
  flow = _Flow()
  flow.request_processors.append(contents_processor.request_processor)

  async def _drain():
    async with testing_utils.Aclosing(
        flow.run_live(invocation_context)
    ) as agen:
      async for _ in agen:
        pass

  # The fake connection ends the run by raising once its turns run out. The
  # timeout keeps a restart that never settles from hanging the suite.
  with contextlib.suppress(_StopReceive):
    await asyncio.wait_for(_drain(), timeout=5)

  assert len(model.requests) == 2
  restarted_request = model.requests[1]
  assert restarted_request.live_connect_config.session_resumption is None
  assert 'said during the session' in _content_texts(restarted_request.contents)


# --- Clean turns ------------------------------------------------------------


@pytest.mark.asyncio
async def test_clean_turn_passes_through_unchanged():
  """Callbacks returning None leave the turn's events untouched."""
  before, before_seen = _recorder()
  after, after_seen = _recorder()
  invocation_context = await _make_context(before=before, after=after)
  connection = _fake_connection([
      _output_transcription('Hello'),
      _output_transcription(' there'),
  ])

  events, _ = await _collect_receive(_Flow(), invocation_context, connection)

  assert len(after_seen) == 2
  assert before_seen == []
  assert not any(isinstance(event, _ReconnectSentinel) for event in events)
  assert [
      event.output_transcription.text
      for event in events
      if event.output_transcription
  ] == ['Hello', ' there']


@pytest.mark.asyncio
async def test_turn_without_callbacks_passes_through_unchanged():
  """An agent with no callbacks streams its turn unchanged."""
  invocation_context = await _make_context()
  connection = _fake_connection([
      _output_transcription('Hello'),
      _output_transcription(' there'),
  ])

  events, _ = await _collect_receive(_Flow(), invocation_context, connection)

  assert not any(isinstance(event, _ReconnectSentinel) for event in events)
  assert [
      event.output_transcription.text
      for event in events
      if event.output_transcription
  ] == ['Hello', ' there']
