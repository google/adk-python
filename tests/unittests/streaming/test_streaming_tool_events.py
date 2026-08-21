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

"""Tests for a live streaming tool that yields user-facing events.

A plain value is a result for the model and goes back over the live connection
as a FunctionResponse. An ``Event`` is a message for the user: streamed to the
client, never sent over the live connection, and recorded in the session like
any other event.
"""

from __future__ import annotations

import asyncio
from contextlib import aclosing
import itertools
from typing import Any
from typing import AsyncGenerator

from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.llm_agent import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.flows.llm_flows import contents
from google.adk.flows.llm_flows.functions import _message_content_for_user
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.platform import time as platform_time
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.function_tool import FunctionTool
from google.genai import types
import pytest

from .. import testing_utils

_TOOL_NAME = 'analyze'


def _texts(events: list[Event]) -> list[str]:
  """Returns the text of every event carrying a single text part."""
  out = []
  for event in events:
    if not event.content or not event.content.parts:
      continue
    for part in event.content.parts:
      if part.text:
        out.append(part.text)
  return out


def _function_responses(contents: list[types.Content]) -> list[Any]:
  """Returns the payload of every FunctionResponse sent to the model."""
  out = []
  for content in contents:
    for part in content.parts or []:
      if part.function_response:
        out.append(part.function_response.response)
  return out


async def _run_live_until(
    tool: Any,
    *,
    stop_when: Any,
    monkeypatch: pytest.MonkeyPatch,
    max_events: int = 25,
) -> tuple[list[Event], list[types.Content]]:
  """Runs a live turn calling ``tool`` once; captures both directions.

  The mock connection replays its canned responses forever, so consumption
  stops as soon as ``stop_when`` has seen what it needs.

  Args:
    tool: The streaming tool to register on the agent.
    stop_when: Called with the events and the contents sent to the model;
      truthy to stop consuming.
    monkeypatch: Used to intercept what is sent over the live connection.
    max_events: Backstop, so a ``stop_when`` that is never satisfied cannot
      hang the test.

  Returns:
    The events yielded to the client, and the contents pushed to the model.
  """
  to_model: list[types.Content] = []

  async def _record_send_content(self: Any, content: types.Content) -> None:
    del self  # Unused.
    to_model.append(content)

  monkeypatch.setattr(
      testing_utils.MockLlmConnection, 'send_content', _record_send_content
  )

  function_call = types.Part.from_function_call(
      name=_TOOL_NAME, args={'query': 'sales'}
  )
  mock_model = testing_utils.MockModel.create([
      LlmResponse(
          content=types.Content(role='model', parts=[function_call]),
          turn_complete=False,
      ),
      LlmResponse(turn_complete=True),
  ])
  root_agent = Agent(name='root_agent', model=mock_model, tools=[tool])

  session_service = InMemorySessionService()
  session = await session_service.create_session(app_name='app', user_id='u')
  runner = Runner(
      app_name='app', agent=root_agent, session_service=session_service
  )

  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      types.Blob(data=b'question', mime_type='audio/pcm')
  )

  events: list[Event] = []

  async def _consume() -> None:
    async with aclosing(
        runner.run_live(
            user_id='u',
            session_id=session.id,
            live_request_queue=live_request_queue,
            run_config=RunConfig(response_modalities=['TEXT']),
        )
    ) as agen:
      async for event in agen:
        events.append(event)
        if stop_when(events, to_model) or len(events) >= max_events:
          # Let the tool's in-flight sends reach the connection.
          for _ in range(30):
            await asyncio.sleep(0)
          return

  try:
    await asyncio.wait_for(_consume(), timeout=10.0)
  except (asyncio.TimeoutError, asyncio.CancelledError):
    pass

  return events, to_model


@pytest.mark.asyncio
async def test_event_goes_to_the_user_and_not_back_as_a_tool_result(
    monkeypatch: pytest.MonkeyPatch,
):
  """It reaches the client, and never the model as a FunctionResponse."""

  async def analyze(query: str) -> AsyncGenerator[Any, None]:
    yield Event(message=f'Connecting for {query}...')
    yield {'done': True}

  events, to_model = await _run_live_until(
      analyze,
      stop_when=lambda evs, _: 'Connecting for sales...' in _texts(evs),
      monkeypatch=monkeypatch,
  )

  assert 'Connecting for sales...' in _texts(events)
  # The message is never restated to the model as a tool result.
  assert not [
      r
      for r in _function_responses(to_model)
      if 'Connecting for sales...' in str(r)
  ]


@pytest.mark.asyncio
async def test_plain_value_goes_to_model_as_function_response(
    monkeypatch: pytest.MonkeyPatch,
):
  """A plain value yielded by a streaming tool is sent back to the model."""

  async def analyze(query: str) -> AsyncGenerator[Any, None]:
    del query  # Unused.
    yield {'stage': 'partial', 'rows': 10}

  _, to_model = await _run_live_until(
      analyze,
      stop_when=lambda _, sent: {'stage': 'partial', 'rows': 10}
      in _function_responses(sent),
      monkeypatch=monkeypatch,
  )

  assert {'stage': 'partial', 'rows': 10} in _function_responses(to_model)


@pytest.mark.asyncio
async def test_tool_mixes_any_number_of_messages_and_results(
    monkeypatch: pytest.MonkeyPatch,
):
  """Messages and results interleave freely, each going only where it belongs."""

  async def analyze(query: str) -> AsyncGenerator[Any, None]:
    del query  # Unused.
    yield Event(message='step one')
    yield {'rows': 1}
    yield Event(message='step two')
    yield {'rows': 2}
    yield Event(message='step three')

  events, to_model = await _run_live_until(
      analyze,
      stop_when=lambda evs, _: 'step three' in _texts(evs),
      monkeypatch=monkeypatch,
  )

  texts = _texts(events)
  assert (
      texts.index('step one')
      < texts.index('step two')
      < texts.index('step three')
  )

  responses = _function_responses(to_model)
  assert {'rows': 1} in responses
  assert {'rows': 2} in responses
  # Three messages, and not one of them was routed to the model.
  assert not [r for r in responses if 'step' in str(r)]


@pytest.mark.asyncio
async def test_message_and_result_are_sent_as_separate_yields(
    monkeypatch: pytest.MonkeyPatch,
):
  """The supported way to both narrate and report: yield twice."""

  async def analyze(query: str) -> AsyncGenerator[Any, None]:
    del query  # Unused.
    yield Event(message='finishing up')
    yield {'rows': 25000}

  events, to_model = await _run_live_until(
      analyze,
      stop_when=lambda _, sent: {'rows': 25000} in _function_responses(sent),
      monkeypatch=monkeypatch,
  )

  assert 'finishing up' in _texts(events)
  assert {'rows': 25000} in _function_responses(to_model)


@pytest.mark.asyncio
async def test_event_is_authored_by_the_agent_and_branched_under_its_call(
    monkeypatch: pytest.MonkeyPatch,
):
  """Authored by the agent, as a Workflow authors its nodes' events.

  Attribution to the individual tool rides on the branch instead; authoring
  by the tool would have the contents processor quote these back at the model.
  """

  async def analyze(query: str) -> AsyncGenerator[Any, None]:
    del query  # Unused.
    yield Event(message='working')

  events, _ = await _run_live_until(
      analyze,
      stop_when=lambda evs, _: 'working' in _texts(evs),
      monkeypatch=monkeypatch,
  )

  message_events = [e for e in events if _texts([e]) == ['working']]
  assert message_events
  event = message_events[0]
  assert event.author == 'root_agent'
  # The same role a tool's FunctionResponse carries.
  assert event.content.role == 'user'

  function_call_ids = [
      part.function_call.id
      for e in events
      if e.content and e.content.parts
      for part in e.content.parts
      if part.function_call
  ]
  assert function_call_ids
  assert event.branch == f'{_TOOL_NAME}@{function_call_ids[0]}'


@pytest.mark.asyncio
async def test_events_are_persisted_to_the_session(
    monkeypatch: pytest.MonkeyPatch,
):
  """A tool's user-facing events are appended to the session like any other."""

  async def analyze(query: str) -> AsyncGenerator[Any, None]:
    del query  # Unused.
    yield Event(message='persist me')

  events, _ = await _run_live_until(
      analyze,
      stop_when=lambda evs, _: 'persist me' in _texts(evs),
      monkeypatch=monkeypatch,
  )

  # The runner appends before yielding, so a yielded non-partial event is
  # already in the session.
  persisted = [e for e in events if _texts([e]) == ['persist me']]
  assert persisted
  assert persisted[0].id


@pytest.mark.asyncio
async def test_message_is_not_rewritten_as_another_agents_reply():
  """A tool's message must not come back to the model as a quoted aside.

  send_history replays the session on every connect. An event authored by the
  tool would be rewritten into "For context: [tool] said: ..."; authoring it
  by the agent keeps the text exactly as the tool wrote it.
  """
  agent = Agent(name='root_agent', model=testing_utils.MockModel.create([]))
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content=''
  )
  invocation_context.session.events = [
      Event(
          invocation_id='i',
          author='user',
          content=types.Content(
              role='user', parts=[types.Part(text='monitor XYZ')]
          ),
      ),
      # As _emit_streaming_tool_event stamps it.
      Event(
          invocation_id='i',
          author='root_agent',
          content=types.Content(
              role='user', parts=[types.Part(text='Connected to the feed.')]
          ),
          branch='analyze@fc-1',
      ),
  ]

  llm_request = LlmRequest()
  async for _ in contents.request_processor.run_async(
      invocation_context, llm_request
  ):
    pass

  replayed = [
      (content.role, part.text)
      for content in llm_request.contents
      for part in content.parts or []
      if part.text
  ]
  assert ('user', 'Connected to the feed.') in replayed
  assert not [text for _, text in replayed if 'For context:' in text]


@pytest.mark.asyncio
async def test_yielded_event_is_left_untouched_and_can_be_reused(
    monkeypatch: pytest.MonkeyPatch,
):
  """A tool may hold one Event and yield it twice; it keeps its object.

  Each delivery needs its own id and timestamp, or the two land in the session
  as one event written twice, and its own Content, or a later edit by the tool
  rewrites what was already delivered.
  """
  # A clock that ticks once per read, so re-stamping is decided by the
  # assertion rather than by the resolution of the wall clock.
  ticks = itertools.count(1_700_000_000.0, 1.0)
  platform_time.set_time_provider(lambda: next(ticks))
  try:
    reused = Event(message='tick')
    built_at = reused.timestamp

    async def analyze(query: str) -> AsyncGenerator[Any, None]:
      del query  # Unused.
      yield reused
      yield reused
      yield {'rows': 10}

    events, _ = await _run_live_until(
        analyze,
        stop_when=(
            lambda evs, _: len([t for t in _texts(evs) if t == 'tick']) > 1
        ),
        monkeypatch=monkeypatch,
    )
  finally:
    platform_time.reset_time_provider()

  # The tool's own object came back exactly as it went in.
  assert reused.author == ''
  assert reused.branch is None
  assert reused.invocation_id == ''
  assert reused.timestamp == built_at

  # Both deliveries landed, as two distinct events.
  delivered = [e for e in events if _texts([e]) == ['tick']]
  assert len(delivered) == 2
  assert delivered[0].id != delivered[1].id
  # Each is dated from when it was sent, not from when the object was built.
  assert delivered[0].timestamp != delivered[1].timestamp
  assert min(e.timestamp for e in delivered) > built_at

  # Nothing delivered shares the Content the tool kept, so editing it now
  # cannot reach back into an event already in the session.
  reused.content.parts[0].text = 'edited after the fact'
  assert _texts(delivered) == ['tick', 'tick']


@pytest.mark.asyncio
async def test_content_in_another_role_is_delivered_in_the_user_role(
    monkeypatch: pytest.MonkeyPatch,
):
  """A role the tool set to something else is stamped over, not refused.

  ``types.ModelContent`` is the form to test it with: it is the one a tool
  reaches for by name, and ``t_content`` wraps a bare ``types.Part`` in one.
  """

  async def analyze(query: str) -> AsyncGenerator[Any, None]:
    del query  # Unused.
    yield Event(content=types.ModelContent(parts=[types.Part(text='built')]))

  events, _ = await _run_live_until(
      analyze,
      stop_when=lambda evs, _: 'built' in _texts(evs),
      monkeypatch=monkeypatch,
  )

  delivered = [e for e in events if _texts([e]) == ['built']]
  assert delivered
  assert delivered[0].content.role == 'user'


@pytest.mark.asyncio
async def test_content_without_a_role_is_delivered_in_the_user_role(
    monkeypatch: pytest.MonkeyPatch,
):
  """A tool that assembles its own Content commonly leaves the role out."""

  async def analyze(query: str) -> AsyncGenerator[Any, None]:
    del query  # Unused.
    yield Event(content=types.Content(parts=[types.Part(text='working')]))

  events, _ = await _run_live_until(
      analyze,
      stop_when=lambda evs, _: 'working' in _texts(evs),
      monkeypatch=monkeypatch,
  )

  delivered = [e for e in events if _texts([e]) == ['working']]
  assert delivered
  assert delivered[0].content.role == 'user'


# --- An Event is a message, and only the message is carried over ---------
#
# A field set beyond the content is ignored rather than honored, and warned
# about; the message still goes out. Only an Event with nothing left to
# deliver is dropped -- dropped and not raised, since raising reaches the
# streaming tool's error handler and fails the whole call.

_IGNORED_EXTRA_FIELDS = 'fields beyond the message, which are ignored'
_DROPPED_NO_CONTENT = 'no content, so there is nothing to deliver'

# The message (`content`, plus the `id` and `timestamp` stamped at
# construction) and the three fields the framework owns. Asserted as an exact
# set so that a field Event gains later is covered without being named here.
_DELIVERED_EVENT_FIELDS = frozenset({
    'content',
    'id',
    'timestamp',
    'author',
    'invocation_id',
    'branch',
})


def _tool() -> FunctionTool:
  """Returns a tool to be named in the warnings."""

  async def analyze(query: str) -> AsyncGenerator[Any, None]:
    del query  # Unused.
    yield {}

  return FunctionTool(analyze)


def _carried_fields(event: Event) -> set[str]:
  """Returns the names of the fields ``event`` sets away from their default."""
  return set(
      event.model_dump(exclude_defaults=True, exclude_none=True, warnings=False)
  )


@pytest.mark.asyncio
async def test_delivered_event_carries_the_message_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
  """The whole contract, asserted on the event that reaches the user.

  The tool sets every kind of extra field there is -- overwritten by the
  framework, side-effecting, client-read, node-resolved -- and none survives.
  """
  caplog.set_level('WARNING')

  async def analyze(query: str) -> AsyncGenerator[Any, None]:
    del query  # Unused.
    yield Event(
        message='working',
        actions=EventActions(state_delta={'stage': 'done'}),
        output={'rows': 10},
        custom_metadata={'k': 'v'},
        author='analyze',
        branch='somewhere.else',
        invocation_id='not-this-one',
        turn_complete=True,
    )

  events, _ = await _run_live_until(
      analyze,
      stop_when=lambda evs, _: 'working' in _texts(evs),
      monkeypatch=monkeypatch,
  )

  delivered = [e for e in events if _texts([e]) == ['working']]
  assert delivered
  assert _carried_fields(delivered[0]) == _DELIVERED_EVENT_FIELDS
  # The three the framework owns hold its values, not the tool's.
  assert delivered[0].author == 'root_agent'
  assert delivered[0].invocation_id != 'not-this-one'
  assert delivered[0].branch.startswith(f'{_TOOL_NAME}@')
  # And the developer is told, rather than left to notice the omission.
  assert _IGNORED_EXTRA_FIELDS in caplog.text
  assert 'analyze' in caplog.text


@pytest.mark.parametrize(
    'event',
    [
        pytest.param(
            Event(
                message='saved',
                actions=EventActions(state_delta={'stage': 'done'}),
            ),
            id='actions',
        ),
        pytest.param(Event(message='typing', partial=True), id='partial'),
        pytest.param(
            Event(message='done', turn_complete=True), id='turn_complete'
        ),
        pytest.param(Event(message='hi', author='analyze'), id='author'),
        pytest.param(Event(message='hi', branch='analyze@1'), id='branch'),
        pytest.param(Event(message='hi', invocation_id='e-1'), id='invocation'),
        pytest.param(Event(message='hi', output={'rows': 10}), id='output'),
        pytest.param(
            Event(message='hi', custom_metadata={'k': 'v'}),
            id='custom_metadata',
        ),
    ],
)
def test_event_carrying_more_than_a_message_keeps_the_message(
    event: Event, caplog: pytest.LogCaptureFixture
):
  """A field beyond the content costs that field, never the message.

  Withholding the message would trade the developer's visible mistake for an
  invisible one: the field does nothing either way, and now the user hears
  nothing at all.
  """
  with caplog.at_level('WARNING'):
    content = _message_content_for_user(event, tool=_tool())

  assert content is not None
  # The warning says which tool, and what a message should look like.
  assert _IGNORED_EXTRA_FIELDS in caplog.text
  assert 'analyze' in caplog.text
  assert 'Event(message=...)' in caplog.text


@pytest.mark.parametrize(
    'event',
    [
        pytest.param(Event(), id='no_content'),
        pytest.param(Event(message=None), id='none_message'),
    ],
)
def test_event_without_content_is_dropped(
    event: Event, caplog: pytest.LogCaptureFixture
):
  """The one shape that is dropped rather than trimmed."""
  with caplog.at_level('WARNING'):
    assert _message_content_for_user(event, tool=_tool()) is None

  assert _DROPPED_NO_CONTENT in caplog.text


@pytest.mark.parametrize(
    'content',
    [
        pytest.param(
            types.ModelContent(parts=[types.Part(text='working')]),
            id='model_content',
        ),
        pytest.param(
            types.Content(role='model', parts=[types.Part(text='working')]),
            id='hand_assembled_in_the_model_role',
        ),
        pytest.param(
            types.Content(parts=[types.Part(text='working')]),
            id='hand_assembled_without_a_role',
        ),
    ],
)
def test_content_is_delivered_in_the_user_role(
    content: types.Content, caplog: pytest.LogCaptureFixture
):
  """Whatever role the tool left behind, the message goes out as ``user``.

  Not warned about: the role is not the tool's to pick, and ``user`` is the
  role every other tool output carries.
  """
  with caplog.at_level('WARNING'):
    delivered = _message_content_for_user(Event(content=content), tool=_tool())

  assert delivered is not None
  assert delivered.role == 'user'
  assert not caplog.text


@pytest.mark.parametrize(
    'event',
    [
        pytest.param(Event(message='working'), id='text'),
        pytest.param(
            Event(
                message=types.Part(
                    inline_data=types.Blob(mime_type='image/png', data=b'x')
                )
            ),
            id='inline_data',
        ),
        pytest.param(
            Event(
                content=types.Content(
                    role='user', parts=[types.Part(text='working')]
                )
            ),
            id='hand_assembled_in_the_user_role',
        ),
        pytest.param(Event(message=''), id='blank_text'),
        pytest.param(
            Event(
                content=types.Content(
                    role='user',
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=_TOOL_NAME, response={'rows': 10}
                            )
                        )
                    ],
                )
            ),
            id='function_response_part',
        ),
    ],
)
def test_event_that_is_only_a_message_warns_about_nothing(
    event: Event, caplog: pytest.LogCaptureFixture
):
  """An Event whose only field is content goes out quietly.

  Whatever is in it: the message is the tool's to compose, so a blank string
  and even a function_response part pass. Both are mistakes, just not ones
  this path is looking for.
  """
  with caplog.at_level('WARNING'):
    assert _message_content_for_user(event, tool=_tool()) is not None

  assert not caplog.text


@pytest.mark.asyncio
async def test_extra_fields_cost_the_tool_neither_the_message_nor_the_call(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
  """End to end: the message lands, the extra field does not, the tool runs on.

  Each assertion rules out one alternative: dropping the event, passing
  ``output`` through, and raising -- which would reach the handler in
  ``run_tool_and_update_queue`` and cost the tool the results it had left.
  """
  caplog.set_level('WARNING')

  async def analyze(query: str) -> AsyncGenerator[Any, None]:
    del query  # Unused.
    yield Event(message='finishing up', output={'rows': 25000})
    yield {'rows': 25000}

  events, to_model = await _run_live_until(
      analyze,
      stop_when=lambda _, sent: {'rows': 25000} in _function_responses(sent),
      monkeypatch=monkeypatch,
  )

  delivered = [e for e in events if _texts([e]) == ['finishing up']]
  assert delivered
  assert delivered[0].output is None
  assert _IGNORED_EXTRA_FIELDS in caplog.text
  assert 'analyze' in caplog.text
  # The tool was not abandoned: the yield after the warned-about one landed.
  assert {'rows': 25000} in _function_responses(to_model)
  # And the model was never told the call failed.
  assert not [
      r for r in _function_responses(to_model) if 'internal error' in str(r)
  ]
  # Nor was the message restated to it as a tool result.
  assert not [
      r for r in _function_responses(to_model) if 'finishing up' in str(r)
  ]
