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

"""Tests for tools that yield intermediate results in non-live runs."""

import asyncio
import json
from typing import Any
from typing import AsyncGenerator
from typing import Generator

from google.adk.agents.llm_agent import Agent
from google.adk.events.event import Event
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest

from ... import testing_utils


def _function_responses(event: Event) -> list[types.FunctionResponse]:
  """Returns the function responses an event carries."""
  if not event.content or not event.content.parts:
    return []
  return [
      part.function_response
      for part in event.content.parts
      if part.function_response is not None
  ]


def _tool_progress(events: list[Event]) -> list[tuple[bool, Any]]:
  """Summarizes every function response as (is intermediate, payload)."""
  return [
      (bool(event.partial), function_response.response)
      for event in events
      for function_response in _function_responses(event)
  ]


@pytest.mark.asyncio
async def test_async_generator_tool_streams_every_yield_but_the_last():
  function_call = types.Part.from_function_call(
      name='search', args={'q': 'adk'}
  )
  mock_model = testing_utils.MockModel.create(responses=[function_call, 'done'])

  async def search(q: str) -> AsyncGenerator[dict[str, Any], None]:
    yield {'status': 'inProgress', 'message': f'searching {q}'}
    yield {'status': 'inProgress', 'message': 'reading pages'}
    yield {'status': 'ok', 'result': ['page1', 'page2']}

  agent = Agent(name='root_agent', model=mock_model, tools=[search])
  runner = testing_utils.InMemoryRunner(agent)
  events = await runner.run_async('test')

  assert _tool_progress(events) == [
      (True, {'status': 'inProgress', 'message': 'searching adk'}),
      (True, {'status': 'inProgress', 'message': 'reading pages'}),
      (False, {'status': 'ok', 'result': ['page1', 'page2']}),
  ]


@pytest.mark.asyncio
async def test_only_intermediate_results_are_marked_will_continue():
  function_call = types.Part.from_function_call(name='search', args={})
  mock_model = testing_utils.MockModel.create(responses=[function_call, 'done'])

  async def search() -> AsyncGenerator[dict[str, Any], None]:
    yield {'status': 'inProgress'}
    yield {'status': 'ok'}

  agent = Agent(name='root_agent', model=mock_model, tools=[search])
  runner = testing_utils.InMemoryRunner(agent)
  events = await runner.run_async('test')

  will_continue = [
      function_response.will_continue
      for event in events
      for function_response in _function_responses(event)
  ]
  assert will_continue == [True, None]


@pytest.mark.asyncio
async def test_intermediate_results_reach_an_sse_client_as_will_continue():
  """Pins the shape a client reads off the wire, not just the Event in memory."""
  function_call = types.Part.from_function_call(name='search', args={})
  function_call.function_call.id = 'call-1'
  mock_model = testing_utils.MockModel.create(responses=[function_call, 'done'])

  async def search() -> AsyncGenerator[dict[str, Any], None]:
    yield {'status': 'inProgress'}
    yield {'status': 'ok'}

  agent = Agent(name='root_agent', model=mock_model, tools=[search])
  runner = testing_utils.InMemoryRunner(agent)
  events = await runner.run_async('test')

  intermediate = next(event for event in events if event.partial)
  # The same serialization the SSE endpoint applies.
  payload = json.loads(
      intermediate.model_dump_json(exclude_none=True, by_alias=True)
  )
  assert payload['partial'] is True
  assert payload['content']['parts'][0]['functionResponse'] == {
      'willContinue': True,
      'id': 'call-1',
      'name': 'search',
      'response': {'status': 'inProgress'},
  }


@pytest.mark.asyncio
async def test_intermediate_results_carry_the_call_id_and_tool_name():
  function_call = types.Part.from_function_call(name='search', args={})
  function_call.function_call.id = 'call-1'
  mock_model = testing_utils.MockModel.create(responses=[function_call, 'done'])

  async def search() -> AsyncGenerator[dict[str, Any], None]:
    yield {'status': 'inProgress'}
    yield {'status': 'ok'}

  agent = Agent(name='root_agent', model=mock_model, tools=[search])
  runner = testing_utils.InMemoryRunner(agent)
  events = await runner.run_async('test')

  addressing = [
      (function_response.id, function_response.name)
      for event in events
      for function_response in _function_responses(event)
  ]
  assert addressing == [('call-1', 'search'), ('call-1', 'search')]


@pytest.mark.asyncio
async def test_intermediate_results_are_not_appended_to_the_session():
  function_call = types.Part.from_function_call(name='search', args={})
  mock_model = testing_utils.MockModel.create(responses=[function_call, 'done'])

  async def search() -> AsyncGenerator[dict[str, Any], None]:
    yield {'status': 'inProgress'}
    yield {'status': 'ok'}

  agent = Agent(name='root_agent', model=mock_model, tools=[search])
  runner = testing_utils.InMemoryRunner(agent)
  await runner.run_async('test')

  assert _tool_progress(runner.session.events) == [(False, {'status': 'ok'})]


@pytest.mark.asyncio
async def test_intermediate_results_are_not_sent_to_the_model():
  function_call = types.Part.from_function_call(name='search', args={})
  mock_model = testing_utils.MockModel.create(responses=[function_call, 'done'])

  async def search() -> AsyncGenerator[dict[str, Any], None]:
    yield {'status': 'inProgress'}
    yield {'status': 'ok'}

  agent = Agent(name='root_agent', model=mock_model, tools=[search])
  runner = testing_utils.InMemoryRunner(agent)
  await runner.run_async('test')

  assert testing_utils.simplify_contents(mock_model.requests[1].contents) == [
      ('user', 'test'),
      ('model', types.Part.from_function_call(name='search', args={})),
      (
          'user',
          types.Part.from_function_response(
              name='search', response={'status': 'ok'}
          ),
      ),
  ]


@pytest.mark.asyncio
async def test_sync_generator_tool_streams_every_yield_but_the_last():
  function_call = types.Part.from_function_call(name='count', args={})
  mock_model = testing_utils.MockModel.create(responses=[function_call, 'done'])

  def count() -> Generator[dict[str, Any], None, None]:
    yield {'status': 'inProgress', 'done': 1}
    yield {'status': 'inProgress', 'done': 2}
    yield {'status': 'ok', 'done': 3}

  agent = Agent(name='root_agent', model=mock_model, tools=[count])
  runner = testing_utils.InMemoryRunner(agent)
  events = await runner.run_async('test')

  assert _tool_progress(events) == [
      (True, {'status': 'inProgress', 'done': 1}),
      (True, {'status': 'inProgress', 'done': 2}),
      (False, {'status': 'ok', 'done': 3}),
  ]


@pytest.mark.asyncio
async def test_non_dict_yields_are_wrapped_like_a_returned_value():
  function_call = types.Part.from_function_call(name='count', args={})
  mock_model = testing_utils.MockModel.create(responses=[function_call, 'done'])

  async def count() -> AsyncGenerator[int, None]:
    yield 1
    yield 2

  agent = Agent(name='root_agent', model=mock_model, tools=[count])
  runner = testing_utils.InMemoryRunner(agent)
  events = await runner.run_async('test')

  assert _tool_progress(events) == [
      (True, {'result': 1}),
      (False, {'result': 2}),
  ]


@pytest.mark.asyncio
async def test_tool_yielding_once_reports_no_intermediate_result():
  function_call = types.Part.from_function_call(name='search', args={})
  mock_model = testing_utils.MockModel.create(responses=[function_call, 'done'])

  async def search() -> AsyncGenerator[dict[str, Any], None]:
    yield {'status': 'ok'}

  agent = Agent(name='root_agent', model=mock_model, tools=[search])
  runner = testing_utils.InMemoryRunner(agent)
  events = await runner.run_async('test')

  assert _tool_progress(events) == [(False, {'status': 'ok'})]


@pytest.mark.asyncio
async def test_tool_yielding_nothing_reports_a_null_result():
  function_call = types.Part.from_function_call(name='search', args={})
  mock_model = testing_utils.MockModel.create(responses=[function_call, 'done'])

  async def search() -> AsyncGenerator[dict[str, Any], None]:
    if False:
      yield {'status': 'ok'}

  agent = Agent(name='root_agent', model=mock_model, tools=[search])
  runner = testing_utils.InMemoryRunner(agent)
  events = await runner.run_async('test')

  assert _tool_progress(events) == [(False, {'result': None})]


@pytest.mark.asyncio
async def test_yielded_event_is_delivered_as_a_user_message():
  function_call = types.Part.from_function_call(name='search', args={})
  mock_model = testing_utils.MockModel.create(responses=[function_call, 'done'])

  async def search() -> AsyncGenerator[Any, None]:
    yield Event(message='looking it up')
    yield {'status': 'ok'}

  agent = Agent(name='root_agent', model=mock_model, tools=[search])
  runner = testing_utils.InMemoryRunner(agent)
  events = await runner.run_async('test')

  # The Event is a message for the user, so it is not a function response and
  # does not displace the result the model sees.
  assert _tool_progress(events) == [(False, {'status': 'ok'})]
  messages = [
      event
      for event in events
      if event.content and event.content.parts and event.content.parts[0].text
  ]
  assert [
      (event.content.role, event.content.parts[0].text) for event in messages
  ] == [('user', 'looking it up'), ('model', 'done')]
  assert messages[0].branch.startswith('search@')


@pytest.mark.asyncio
async def test_failure_after_the_first_yield_reaches_on_tool_error():
  function_call = types.Part.from_function_call(name='search', args={})
  mock_model = testing_utils.MockModel.create(responses=[function_call, 'done'])
  errors = []

  async def search() -> AsyncGenerator[dict[str, Any], None]:
    yield {'status': 'inProgress'}
    raise ValueError('no results')

  def on_tool_error(*, tool, args, tool_context, error):
    errors.append(error)
    return {'status': 'error', 'message': str(error)}

  agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[search],
      on_tool_error_callback=on_tool_error,
  )
  runner = testing_utils.InMemoryRunner(agent)
  events = await runner.run_async('test')

  assert [str(error) for error in errors] == ['no results']
  assert _tool_progress(events) == [
      (True, {'status': 'inProgress'}),
      (False, {'status': 'error', 'message': 'no results'}),
  ]


@pytest.mark.asyncio
async def test_unhandled_failure_after_the_first_yield_propagates():
  function_call = types.Part.from_function_call(name='search', args={})
  mock_model = testing_utils.MockModel.create(responses=[function_call, 'done'])

  async def search() -> AsyncGenerator[dict[str, Any], None]:
    yield {'status': 'inProgress'}
    raise ValueError('no results')

  agent = Agent(name='root_agent', model=mock_model, tools=[search])
  runner = testing_utils.InMemoryRunner(agent)
  with pytest.raises(ValueError, match='no results'):
    await runner.run_async('test')


@pytest.mark.asyncio
async def test_parallel_generator_tools_stream_independently():
  function_calls = [
      types.Part.from_function_call(name='fast', args={}),
      types.Part.from_function_call(name='slow', args={}),
  ]
  mock_model = testing_utils.MockModel.create(
      responses=[function_calls, 'done']
  )

  async def fast() -> AsyncGenerator[dict[str, Any], None]:
    yield {'tool': 'fast', 'status': 'inProgress'}
    yield {'tool': 'fast', 'status': 'ok'}

  async def slow() -> AsyncGenerator[dict[str, Any], None]:
    yield {'tool': 'slow', 'status': 'inProgress'}
    # Yields the event loop so that `fast` finishes first, proving the two
    # calls are drained concurrently rather than one after the other.
    await asyncio.sleep(0.05)
    yield {'tool': 'slow', 'status': 'ok'}

  agent = Agent(name='root_agent', model=mock_model, tools=[fast, slow])
  runner = testing_utils.InMemoryRunner(agent)
  events = await runner.run_async('test')

  intermediate = [
      payload for partial, payload in _tool_progress(events) if partial
  ]
  final = [
      payload for partial, payload in _tool_progress(events) if not partial
  ]
  assert sorted(payload['tool'] for payload in intermediate) == ['fast', 'slow']
  # Both calls answer in one merged non-partial event, as parallel calls always
  # have.
  assert len(final) == 2
  assert all(payload['status'] == 'ok' for payload in final)


@pytest.mark.asyncio
async def test_tool_context_is_still_injected_into_a_generator_tool():
  function_call = types.Part.from_function_call(name='search', args={})
  mock_model = testing_utils.MockModel.create(responses=[function_call, 'done'])
  seen = []

  async def search(
      tool_context: ToolContext | None = None,
  ) -> AsyncGenerator[dict[str, Any], None]:
    seen.append(tool_context.function_call_id)
    yield {'status': 'inProgress'}
    yield {'status': 'ok'}

  agent = Agent(name='root_agent', model=mock_model, tools=[search])
  runner = testing_utils.InMemoryRunner(agent)
  await runner.run_async('test')

  assert len(seen) == 1 and seen[0]


@pytest.mark.asyncio
async def test_state_a_generator_tool_sets_is_applied_once_at_the_end():
  function_call = types.Part.from_function_call(name='search', args={})
  mock_model = testing_utils.MockModel.create(responses=[function_call, 'done'])

  async def search(
      tool_context: ToolContext | None = None,
  ) -> AsyncGenerator[dict[str, Any], None]:
    yield {'status': 'inProgress'}
    tool_context.state['hits'] = 2
    yield {'status': 'ok'}

  agent = Agent(name='root_agent', model=mock_model, tools=[search])
  runner = testing_utils.InMemoryRunner(agent)
  events = await runner.run_async('test')

  # An intermediate event is never applied, so it must not carry the delta.
  assert [event.actions.state_delta for event in events if event.partial] == [
      {}
  ]
  assert runner.session.state['hits'] == 2


@pytest.mark.asyncio
async def test_returning_a_plain_value_is_unchanged():
  function_call = types.Part.from_function_call(name='search', args={})
  mock_model = testing_utils.MockModel.create(responses=[function_call, 'done'])

  async def search() -> dict[str, Any]:
    return {'status': 'ok'}

  agent = Agent(name='root_agent', model=mock_model, tools=[search])
  runner = testing_utils.InMemoryRunner(agent)
  events = await runner.run_async('test')

  assert _tool_progress(events) == [(False, {'status': 'ok'})]
