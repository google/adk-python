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

"""Tests for SnowflakeCortexAgent.

Verifies the configuration surface (composition guards, credential exclusion
from ``repr`` and every serialization path, per-agent state keys) and the run
loop against a scripted Snowflake behind ``httpx.MockTransport``: thread
creation, cursor commit and continuation, SSE gating, tool trace, failure
paths that leave the cursor alone, and cancellation on disconnect.
"""

# `_state_key` is the documented shape of the session state key; the tests
# check it directly once and otherwise observe it through `state_delta`.
# pylint: disable=protected-access

from __future__ import annotations

import functools
import json
from typing import Any
from typing import AsyncGenerator
from typing import AsyncIterator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.adk.cli.utils.graph_serialization import serialize_agent
from google.adk.events.event import Event
from google.adk.labs.snowflake import SnowflakeCortexAgent
from google.adk.labs.snowflake._client import CortexApiError
from google.adk.labs.snowflake._client import CortexTransportError
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types as genai_types
import httpx
from pydantic import ValidationError
import pytest

_TOKEN = 'pat-secret-token-value'
_ACCOUNT_URL = 'https://example.snowflakecomputing.com'


def _bearer_headers(ctx: ReadonlyContext, *, token: str) -> dict[str, str]:
  del ctx
  return {'Authorization': f'Bearer {token}'}


# A `functools.partial` rather than a closure, because its `repr` prints the
# bound token. That makes "the token is absent" a real check: it would appear
# if the field were not excluded from `repr` and serialization.
_HEADER_PROVIDER = functools.partial(_bearer_headers, token=_TOKEN)


def _sse(*events: tuple[str, Any]) -> bytes:
  """Encodes `(event name, JSON payload)` pairs as one SSE byte stream."""
  return b''.join(
      f'event: {name}\ndata: {json.dumps(payload)}\n\n'.encode('utf-8')
      for name, payload in events
  )


_DONE = b'event: done\ndata: [DONE]\n\n'


def _run_stream(
    *,
    assistant_message_id: int | None = 456,
    answer: str = 'Hello',
    status: str = 'completed',
    done: bool = True,
) -> bytes:
  """A complete run: status, two text deltas, one SQL tool call, the answer."""
  events: list[tuple[str, Any]] = [
      ('metadata', {'metadata': {'role': 'user', 'message_id': 455}}),
      ('response.status', {'status': 'planning', 'sequence_number': 1}),
      (
          'response.text.delta',
          {'content_index': 0, 'sequence_number': 2, 'text': answer[:3]},
      ),
      (
          'response.text.delta',
          {'content_index': 0, 'sequence_number': 3, 'text': answer[3:]},
      ),
      (
          'response.tool_use',
          {
              'client_side_execute': False,
              'input': {'sql': 'SELECT 1'},
              'name': 'system_execute_sql',
              'tool_use_id': 't1',
              'type': 'system_execute_sql',
          },
      ),
      (
          'response.tool_result',
          {
              'content': [{'json': {'query_id': 'q1'}, 'type': 'json'}],
              'name': 'system_execute_sql',
              'status': 'success',
              'tool_use_id': 't1',
              'type': 'system_execute_sql',
          },
      ),
  ]
  metadata: dict[str, Any] = {'run_id': 'run-1', 'user_message_id': 455}
  if assistant_message_id is not None:
    events.append((
        'metadata',
        {'metadata': {'role': 'assistant', 'message_id': assistant_message_id}},
    ))
    metadata['assistant_message_id'] = assistant_message_id
  events.append((
      'response',
      {
          'content': [{'text': answer, 'type': 'text'}],
          'metadata': metadata,
          'status': status,
      },
  ))
  return _sse(*events) + (_DONE if done else b'')


class _Chunks(httpx.AsyncByteStream):
  """A scripted response body that records whether it was closed."""

  def __init__(self, body: bytes):
    self._body = body
    self.closed = False

  async def __aiter__(self) -> AsyncIterator[bytes]:
    yield self._body

  async def aclose(self) -> None:
    self.closed = True


class _FakeSnowflake:
  """A scripted Snowflake behind `httpx.MockTransport`, recording requests."""

  def __init__(
      self,
      *,
      run_body: bytes | None = None,
      thread_id: int = 123,
      run_status: int = 200,
      cancel_status: int = 200,
  ):
    self.thread_id = thread_id
    self.run_status = run_status
    self.cancel_status = cancel_status
    self.run_bodies = [run_body if run_body is not None else _run_stream()]
    self.streams: list[_Chunks] = []
    self.requests: list[httpx.Request] = []

  def http_client(self) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(self._handle))

  def _handle(self, request: httpx.Request) -> httpx.Response:
    self.requests.append(request)
    path = request.url.path
    if path.endswith('/cortex/threads'):
      return httpx.Response(200, json={'thread_id': self.thread_id})
    if path.endswith(':run'):
      if self.run_status != 200:
        return httpx.Response(self.run_status, json={'message': 'nope'})
      body = (
          self.run_bodies.pop(0)
          if len(self.run_bodies) > 1
          else (self.run_bodies[0])
      )
      stream = _Chunks(body)
      self.streams.append(stream)
      return httpx.Response(
          200, stream=stream, headers={'content-type': 'text/event-stream'}
      )
    if path.endswith('/cancel'):
      return httpx.Response(self.cancel_status)
    return httpx.Response(404)

  def paths(self, suffix: str) -> list[httpx.Request]:
    return [r for r in self.requests if r.url.path.endswith(suffix)]


def _make_agent(
    name: str = 'cortex', **overrides: object
) -> SnowflakeCortexAgent:
  """A minimal SnowflakeCortexAgent pointing at a fake account."""
  fields: dict[str, object] = {
      'name': name,
      'account_url': _ACCOUNT_URL,
      'database': 'SALES_DB',
      'schema_name': 'ANALYTICS',
      'cortex_agent_name': 'SALES_AGENT',
      'header_provider': _HEADER_PROVIDER,
  }
  fields.update(overrides)
  return SnowflakeCortexAgent(**fields)


class _StubChild(BaseAgent):
  """A runnable ADK child agent."""

  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield Event(invocation_id=ctx.invocation_id, author=self.name)


async def _invocation_context(
    agent: BaseAgent,
    *,
    text: str | None = 'hello',
    state: dict[str, Any] | None = None,
    streaming_mode: StreamingMode = StreamingMode.SSE,
) -> InvocationContext:
  """A real InvocationContext rooted at `agent`."""
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user', state=state
  )
  return InvocationContext(
      session_service=session_service,
      invocation_id='inv_1',
      agent=agent,
      session=session,
      user_content=(
          genai_types.Content(
              role='user', parts=[genai_types.Part.from_text(text=text)]
          )
          if text is not None
          else None
      ),
      run_config=RunConfig(streaming_mode=streaming_mode),
  )


async def _run(agent: BaseAgent, ctx: InvocationContext) -> list[Event]:
  return [event async for event in agent.run_async(ctx)]


def _final(events: list[Event]) -> Event:
  (final,) = [
      e
      for e in events
      if not e.partial
      and e.content
      and e.content.role == 'model'
      and not e.get_function_calls()
  ]
  return final


# --- configuration ------------------------------------------------------------


def test_standalone_agent_is_allowed():
  """An agent with neither parent nor children constructs cleanly."""
  agent = _make_agent()

  assert agent.parent_agent is None
  assert agent.sub_agents == []


def test_defaults_are_the_documented_values():
  """Options not passed take the documented defaults."""
  agent = _make_agent()

  assert agent.timeout == 900.0
  assert agent.cancel_on_disconnect is True
  assert agent.max_tool_result_bytes == 32 * 1024
  assert agent.include_thinking_in_final_event is False
  assert agent.http_client is None


@pytest.mark.parametrize('field', ['timeout', 'max_tool_result_bytes'])
def test_non_positive_bounds_are_rejected(field: str):
  """A zero timeout or result size limit fails validation."""
  with pytest.raises(ValidationError, match='greater than 0'):
    _make_agent(**{field: 0})


def test_sub_agents_are_rejected():
  """Declaring `sub_agents` fails at construction."""
  child = _StubChild(name='reviewer')

  with pytest.raises(ValueError, match='sub_agents'):
    _make_agent(sub_agents=[child])


def test_using_as_sub_agent_is_rejected():
  """A parent listing this agent in `sub_agents` fails to construct."""
  agent = _make_agent()

  with pytest.raises(ValueError, match='root agent'):
    BaseAgent(name='parent', sub_agents=[agent])

  assert agent.parent_agent is None


async def test_sub_agents_added_after_construction_are_rejected_at_run():
  """Mutating `sub_agents` past validation still fails, at the first turn."""
  agent = _make_agent()
  agent.sub_agents.append(_StubChild(name='late'))
  ctx = await _invocation_context(agent)

  with pytest.raises(ValueError, match='sub_agents'):
    await _run(agent, ctx)


async def test_sub_agents_added_by_clone_are_rejected_at_run():
  """A clone given `sub_agents` skips construction checks but cannot run."""
  agent = _make_agent()
  cloned = agent.clone(update={'sub_agents': [_StubChild(name='late')]})
  ctx = await _invocation_context(cloned)

  with pytest.raises(ValueError, match='sub_agents'):
    await _run(cloned, ctx)


def test_header_provider_is_hidden_from_repr():
  """`repr` shows neither the provider nor the token it carries."""
  agent = _make_agent()

  text = repr(agent)

  assert 'header_provider' not in text
  assert _TOKEN not in text


def test_header_provider_is_excluded_from_model_dump():
  """`model_dump` omits the provider and the token it carries."""
  agent = _make_agent()

  dumped = agent.model_dump()

  assert 'header_provider' not in dumped
  assert 'http_client' not in dumped
  assert _TOKEN not in str(dumped)
  assert dumped['cortex_agent_name'] == 'SALES_AGENT'


def test_header_provider_is_hidden_from_the_adk_web_agent_graph():
  """The `adk web` agent graph omits the provider and the token it carries."""
  agent = _make_agent()

  serialized = json.dumps(serialize_agent(agent), default=str)

  assert 'header_provider' not in serialized
  assert _TOKEN not in serialized


async def test_header_provider_stays_callable_on_the_instance():
  """Exclusion from output leaves the provider itself in place."""
  agent = _make_agent()
  ctx = await _invocation_context(agent)

  headers = agent.header_provider(ReadonlyContext(ctx))

  assert headers == {'Authorization': f'Bearer {_TOKEN}'}


def test_clone_keeps_the_header_provider():
  """A clone can still authenticate: exclusion is from output, not copies."""
  agent = _make_agent()

  cloned = agent.clone(update={'name': 'copy'})

  assert cloned.header_provider is agent.header_provider


def test_state_key_is_scoped_by_agent_name():
  """Two agents with different names keep separate Snowflake threads."""
  first = _make_agent(name='first')
  second = _make_agent(name='second')

  assert first._state_key() != second._state_key()
  assert first._state_key() == _make_agent(name='first')._state_key()


# --- first and second turn ----------------------------------------------------


async def test_first_turn_creates_a_thread_and_commits_the_cursor():
  """Turn one creates a thread, runs from message 0 and stores the cursor."""
  snowflake = _FakeSnowflake()
  agent = _make_agent(http_client=snowflake.http_client())
  ctx = await _invocation_context(agent)

  events = await _run(agent, ctx)

  (thread_request,) = snowflake.paths('/cortex/threads')
  (run_request,) = snowflake.paths(':run')
  assert thread_request.headers['authorization'] == f'Bearer {_TOKEN}'
  run_body = json.loads(run_request.content)
  assert (run_body['thread_id'], run_body['parent_message_id']) == (123, 0)
  assert run_body['messages'][0]['content'][0]['text'] == 'hello'
  final = _final(events)
  assert final.content.parts[0].text == 'Hello'
  assert final.custom_metadata['snowflake_cortex']['run_id'] == 'run-1'
  cursor = final.actions.state_delta['_snowflake_cortex_cortex']
  assert cursor['schema_version'] == 1
  assert cursor['resource_fingerprint'].startswith('sha256:')
  assert (cursor['thread_id'], cursor['parent_message_id']) == ('123', '456')


async def test_user_message_id_is_never_stored():
  """Only the assistant message may be the next parent."""
  snowflake = _FakeSnowflake()
  agent = _make_agent(http_client=snowflake.http_client())

  events = await _run(agent, await _invocation_context(agent))

  assert '455' not in json.dumps(_final(events).actions.state_delta)


async def test_second_turn_continues_the_stored_thread():
  """With a cursor in state, no thread is created and the parent advances."""
  snowflake = _FakeSnowflake(run_body=_run_stream(assistant_message_id=789))
  agent = _make_agent(http_client=snowflake.http_client())
  first_turn = _FakeSnowflake()
  seed = _make_agent(http_client=first_turn.http_client())
  stored = _final(
      await _run(seed, await _invocation_context(seed))
  ).actions.state_delta
  ctx = await _invocation_context(agent, text='and then?', state=stored)

  events = await _run(agent, ctx)

  assert snowflake.paths('/cortex/threads') == []
  run_body = json.loads(snowflake.paths(':run')[0].content)
  assert (run_body['thread_id'], run_body['parent_message_id']) == (123, 456)
  cursor = _final(events).actions.state_delta['_snowflake_cortex_cortex']
  assert (cursor['thread_id'], cursor['parent_message_id']) == ('123', '789')


async def test_runner_persists_the_cursor_between_turns():
  """Through the Runner the state delta lands in the session for turn two."""
  snowflake = _FakeSnowflake()
  snowflake.run_bodies = [
      _run_stream(assistant_message_id=456),
      _run_stream(assistant_message_id=789),
  ]
  agent = _make_agent(http_client=snowflake.http_client())
  session_service = InMemorySessionService()
  session = await session_service.create_session(app_name='app', user_id='u')
  runner = Runner(app_name='app', agent=agent, session_service=session_service)

  for text in ('first', 'second'):
    async for _ in runner.run_async(
        user_id='u',
        session_id=session.id,
        new_message=genai_types.Content(
            role='user', parts=[genai_types.Part.from_text(text=text)]
        ),
    ):
      pass

  session = await session_service.get_session(
      app_name='app', user_id='u', session_id=session.id
  )
  assert len(snowflake.paths('/cortex/threads')) == 1
  parents = [
      json.loads(r.content)['parent_message_id']
      for r in snowflake.paths(':run')
  ]
  assert parents == [0, 456]
  cursor = session.state['_snowflake_cortex_cortex']
  assert (cursor['thread_id'], cursor['parent_message_id']) == ('123', '789')


async def test_two_agents_in_one_session_keep_separate_cursors():
  """Each agent's cursor lives under its own key."""
  first = _make_agent(name='first', http_client=_FakeSnowflake().http_client())
  second = _make_agent(
      name='second',
      http_client=_FakeSnowflake(thread_id=999).http_client(),
  )
  state = _final(
      await _run(first, await _invocation_context(first))
  ).actions.state_delta
  state.update(
      _final(
          await _run(second, await _invocation_context(second, state=state))
      ).actions.state_delta
  )

  assert state['_snowflake_cortex_first']['thread_id'] == '123'
  assert state['_snowflake_cortex_second']['thread_id'] == '999'


# --- streaming and tool trace -------------------------------------------------


async def test_sse_mode_streams_partial_events():
  """With SSE streaming the deltas and progress arrive as partial events."""
  agent = _make_agent(http_client=_FakeSnowflake().http_client())
  ctx = await _invocation_context(agent, streaming_mode=StreamingMode.SSE)

  events = await _run(agent, ctx)

  partial_text = [
      e.content.parts[0].text
      for e in events
      if e.partial and e.content and e.content.parts
  ]
  assert partial_text == ['Hel', 'lo']
  assert any(e.partial and e.custom_metadata for e in events)


async def test_none_mode_yields_only_persisted_events():
  """Without SSE streaming nothing partial is yielded, the answer still is."""
  agent = _make_agent(http_client=_FakeSnowflake().http_client())
  ctx = await _invocation_context(agent, streaming_mode=StreamingMode.NONE)

  events = await _run(agent, ctx)

  assert not any(e.partial for e in events)
  assert _final(events).content.parts[0].text == 'Hello'


async def test_tool_trace_is_recorded_as_function_call_and_response():
  """Server-side tool use shows up as a call by the agent and a response."""
  agent = _make_agent(http_client=_FakeSnowflake().http_client())

  events = await _run(agent, await _invocation_context(agent))

  (call_event,) = [e for e in events if e.get_function_calls()]
  (response_event,) = [e for e in events if e.get_function_responses()]
  assert call_event.author == 'cortex'
  assert call_event.get_function_calls()[0].args == {'sql': 'SELECT 1'}
  assert response_event.author == 'system_execute_sql'
  assert response_event.get_function_responses()[0].id == 't1'
  assert not call_event.partial and not response_event.partial


async def test_reading_stops_at_done():
  """Bytes after `[DONE]` are never read, so a chatty server cannot stall."""
  body = _run_stream() + _sse(('response.status', {'status': 'late'})) * 3
  snowflake = _FakeSnowflake(run_body=body)
  agent = _make_agent(http_client=snowflake.http_client())

  events = await _run(agent, await _invocation_context(agent))

  assert not any(
      e.custom_metadata
      and e.custom_metadata['snowflake_cortex'].get('data', {}).get('status')
      == 'late'
      for e in events
  )
  assert snowflake.streams[0].closed


# --- failures leave the cursor alone ------------------------------------------


async def test_error_event_ends_the_turn_without_a_cursor_update():
  """A terminal `error` is surfaced as an error event and nothing is stored."""
  body = (
      _sse(
          ('metadata', {'metadata': {'role': 'user', 'message_id': 455}}),
          ('error', {'code': 'STREAM_TIMEOUT', 'message': 'took too long'}),
      )
      + _DONE
  )
  agent = _make_agent(http_client=_FakeSnowflake(run_body=body).http_client())

  events = await _run(agent, await _invocation_context(agent))

  (error_event,) = [e for e in events if e.error_code]
  assert error_event.error_code == 'STREAM_TIMEOUT'
  assert all(not e.actions.state_delta for e in events)


async def test_stream_cut_before_the_final_response_fails_and_keeps_the_cursor():
  """A stream that ends early is a failure, not a truncated answer."""
  body = _run_stream()[: _run_stream().index(b'event: response\n')]
  agent = _make_agent(http_client=_FakeSnowflake(run_body=body).http_client())

  with pytest.raises(CortexTransportError, match='before the final response'):
    await _run(agent, await _invocation_context(agent))


async def test_http_error_on_run_is_raised():
  """Snowflake refusing the run surfaces as an API error with its status."""
  agent = _make_agent(http_client=_FakeSnowflake(run_status=401).http_client())

  with pytest.raises(CortexApiError) as info:
    await _run(agent, await _invocation_context(agent))

  assert info.value.status_code == 401


async def test_final_response_without_done_still_completes_the_turn():
  """`[DONE]` is a compatibility sentinel; the final `response` closes the turn."""
  body = _run_stream(done=False)
  agent = _make_agent(http_client=_FakeSnowflake(run_body=body).http_client())

  events = await _run(agent, await _invocation_context(agent))

  final = _final(events)
  assert final.content.parts[0].text == 'Hello'
  cursor = final.actions.state_delta['_snowflake_cortex_cortex']
  assert cursor['parent_message_id'] == '456'


@pytest.mark.parametrize('status', ['cancelled', 'timed_out'])
async def test_non_completed_final_status_does_not_commit_the_cursor(status):
  """Only a `completed` run may become the parent of the next turn."""
  body = _run_stream(status=status)
  agent = _make_agent(http_client=_FakeSnowflake(run_body=body).http_client())

  events = await _run(agent, await _invocation_context(agent))

  final = _final(events)
  assert final.custom_metadata['snowflake_cortex']['status'] == status
  assert final.actions.state_delta == {}


async def test_missing_assistant_id_skips_the_cursor_update():
  """Without an assistant message id there is no safe parent to store."""
  body = _run_stream(assistant_message_id=None)
  agent = _make_agent(http_client=_FakeSnowflake(run_body=body).http_client())

  events = await _run(agent, await _invocation_context(agent))

  assert _final(events).actions.state_delta == {}


async def test_cursor_from_another_cortex_agent_is_refused():
  """A cursor whose fingerprint differs is never continued."""
  snowflake = _FakeSnowflake()
  agent = _make_agent(http_client=snowflake.http_client())
  other = _make_agent(cortex_agent_name='OTHER_AGENT')
  stored = {
      '_snowflake_cortex_cortex': {
          'schema_version': 1,
          'resource_fingerprint': other._resource_fingerprint(),
          'thread_id': '123',
          'parent_message_id': '456',
      }
  }
  ctx = await _invocation_context(agent, state=stored)

  with pytest.raises(ValueError, match='different account') as info:
    await _run(agent, ctx)

  assert snowflake.requests == []
  assert '123' not in str(info.value)


@pytest.mark.parametrize(
    'stored',
    ['garbage', {'schema_version': 2}, {'schema_version': 1, 'thread_id': 1}],
)
async def test_malformed_cursor_is_refused(stored: Any):
  """State that is not a cursor this version understands is an error."""
  snowflake = _FakeSnowflake()
  agent = _make_agent(http_client=snowflake.http_client())
  if isinstance(stored, dict) and 'thread_id' in stored:
    stored = {
        **stored,
        'resource_fingerprint': agent._resource_fingerprint(),
        'parent_message_id': '0',
    }
  ctx = await _invocation_context(
      agent, state={'_snowflake_cortex_cortex': stored}
  )

  with pytest.raises(ValueError, match='_snowflake_cortex_cortex'):
    await _run(agent, ctx)

  assert snowflake.requests == []


async def test_missing_user_text_is_rejected_before_any_request():
  """A turn without text cannot be sent to Snowflake."""
  snowflake = _FakeSnowflake()
  agent = _make_agent(http_client=snowflake.http_client())
  ctx = await _invocation_context(agent, text=None)

  with pytest.raises(ValueError, match='text message'):
    await _run(agent, ctx)

  assert snowflake.requests == []


# --- disconnect ---------------------------------------------------------------


async def test_disconnect_closes_upstream_and_cancels_the_run():
  """A consumer that stops reading releases Snowflake and cancels the run."""
  snowflake = _FakeSnowflake()
  agent = _make_agent(http_client=snowflake.http_client())
  generator = agent.run_async(await _invocation_context(agent))

  first = await generator.__anext__()
  await generator.aclose()

  assert first.partial is True
  assert snowflake.streams[0].closed
  (cancel,) = snowflake.paths('/cancel')
  assert cancel.url.path.endswith('/runs/123-455/cancel')


async def test_disconnect_without_the_option_does_not_cancel():
  """`cancel_on_disconnect=False` only closes the connection."""
  snowflake = _FakeSnowflake()
  agent = _make_agent(
      cancel_on_disconnect=False, http_client=snowflake.http_client()
  )
  generator = agent.run_async(await _invocation_context(agent))

  await generator.__anext__()
  await generator.aclose()

  assert snowflake.streams[0].closed
  assert snowflake.paths('/cancel') == []


async def test_disconnect_before_the_user_message_is_acknowledged_does_not_cancel():
  """Without a user message id there is no run id, so nothing is cancelled."""
  body = _sse(('response.status', {'status': 'planning', 'sequence_number': 1}))
  body += _run_stream()
  snowflake = _FakeSnowflake(run_body=body)
  agent = _make_agent(http_client=snowflake.http_client())
  generator = agent.run_async(await _invocation_context(agent))

  first = await generator.__anext__()
  await generator.aclose()

  assert first.custom_metadata['snowflake_cortex']['event'] == 'response.status'
  assert snowflake.streams[0].closed
  assert snowflake.paths('/cancel') == []


async def test_disconnect_cancel_rejected_by_snowflake_is_swallowed():
  """A 409 from the cancel endpoint (run already over) does not surface."""
  snowflake = _FakeSnowflake(cancel_status=409)
  agent = _make_agent(http_client=snowflake.http_client())
  generator = agent.run_async(await _invocation_context(agent))

  await generator.__anext__()
  await generator.aclose()

  assert len(snowflake.paths('/cancel')) == 1
  assert snowflake.streams[0].closed


async def test_disconnect_after_done_does_not_cancel():
  """Nothing is cancelled once Snowflake has finished the run."""
  snowflake = _FakeSnowflake()
  agent = _make_agent(http_client=snowflake.http_client())

  await _run(agent, await _invocation_context(agent))

  assert snowflake.paths('/cancel') == []


# --- lifecycle ----------------------------------------------------------------


async def test_cleanup_leaves_a_shared_http_client_open():
  """The application's client is the application's to close."""
  snowflake = _FakeSnowflake()
  shared = snowflake.http_client()
  agent = _make_agent(http_client=shared)
  await _run(agent, await _invocation_context(agent))

  await agent.cleanup()

  assert shared.is_closed is False
  await _run(agent, await _invocation_context(agent))
  await shared.aclose()
