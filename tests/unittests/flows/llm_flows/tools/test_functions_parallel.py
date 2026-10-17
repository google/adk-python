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

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from collections.abc import Callable
import copy
from typing import Any

from google.adk.agents.llm_agent import Agent
from google.adk.events.event_actions import EventActions
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest

from .... import testing_utils


async def _run_parallel_calls(
    tool: Callable[..., Awaitable[None]],
    args_list: list[dict[str, str]],
    initial_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Runs one model turn of parallel calls to `tool`, returns the stored state."""
  function_calls = [
      types.Part.from_function_call(name=tool.__name__, args=args)
      for args in args_list
  ]
  agent = Agent(
      name='root_agent',
      model=testing_utils.MockModel.create(responses=[function_calls, 'done']),
      tools=[tool],
  )
  runner = testing_utils.TestInMemoryRunner(agent)
  session = await runner.session_service.create_session(
      app_name=runner.app_name, user_id='test_user', state=initial_state
  )
  async for _ in runner.run_async(
      user_id='test_user',
      session_id=session.id,
      new_message=testing_utils.get_user_content('test'),
  ):
    pass
  stored = await runner.session_service.get_session(
      app_name=runner.app_name, user_id='test_user', session_id=session.id
  )
  assert stored is not None
  return stored.state


@pytest.mark.asyncio
async def test_parallel_function_calls_with_state_change():
  function_calls = [
      types.Part.from_function_call(
          name='update_session_state',
          args={'key': 'test_key1', 'value': 'test_value1'},
      ),
      types.Part.from_function_call(
          name='update_session_state',
          args={'key': 'test_key2', 'value': 'test_value2'},
      ),
      types.Part.from_function_call(
          name='transfer_to_agent', args={'agent_name': 'test_sub_agent'}
      ),
  ]
  function_responses = [
      types.Part.from_function_response(
          name='update_session_state', response={'result': None}
      ),
      types.Part.from_function_response(
          name='update_session_state', response={'result': None}
      ),
      types.Part.from_function_response(
          name='transfer_to_agent', response={'result': None}
      ),
  ]

  responses: list[types.Content] = [
      function_calls,
      'response1',
  ]
  function_called = 0
  mock_model = testing_utils.MockModel.create(responses=responses)

  async def update_session_state(
      key: str, value: str, tool_context: ToolContext
  ) -> None:
    nonlocal function_called
    function_called += 1
    tool_context.state.update({key: value})
    return

  async def transfer_to_agent(
      agent_name: str, tool_context: ToolContext
  ) -> None:
    nonlocal function_called
    function_called += 1
    tool_context.actions.transfer_to_agent = agent_name
    return

  test_sub_agent = Agent(
      name='test_sub_agent',
  )

  agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[update_session_state, transfer_to_agent],
      sub_agents=[test_sub_agent],
  )
  runner = testing_utils.TestInMemoryRunner(agent)
  events = await runner.run_async_with_new_session('test')

  # Notice that the following assertion only checks the "contents" part of the events.
  # The "actions" part will be checked later.
  assert testing_utils.simplify_events(events) == [
      ('root_agent', function_calls),
      ('root_agent', function_responses),
      ('test_sub_agent', 'response1'),
  ]

  # Asserts the function calls.
  assert function_called == 3

  # Asserts the actions in response event.
  response_event = events[1]

  assert response_event.actions == EventActions(
      state_delta={
          'test_key1': 'test_value1',
          'test_key2': 'test_value2',
      },
      transfer_to_agent='test_sub_agent',
  )


@pytest.mark.asyncio
async def test_parallel_appends_finishing_out_of_order_keep_every_item():
  b_written = asyncio.Event()

  async def append_item(item: str, tool_context: ToolContext) -> None:
    if item == 'a':
      await asyncio.wait_for(b_written.wait(), timeout=5)
    tool_context.state['items'] = tool_context.state.get('items', []) + [item]
    b_written.set()

  state = await _run_parallel_calls(append_item, [{'item': 'a'}, {'item': 'b'}])

  assert state['items'] == ['b', 'a']


@pytest.mark.asyncio
async def test_parallel_appends_finishing_in_order_store_no_duplicates():
  async def append_item(item: str, tool_context: ToolContext) -> None:
    tool_context.state['items'] = tool_context.state.get('items', []) + [item]

  state = await _run_parallel_calls(append_item, [{'item': 'a'}, {'item': 'b'}])

  assert state['items'] == ['a', 'b']


@pytest.mark.asyncio
async def test_parallel_snapshot_writes_keep_the_latest_nested_list():
  b_written = asyncio.Event()

  async def add_label(label: str, tool_context: ToolContext) -> None:
    if label == 'a':
      await asyncio.wait_for(b_written.wait(), timeout=5)
    doc = copy.deepcopy(tool_context.state['doc'])
    doc['entities'][0]['labels'].append(label)
    tool_context.state['doc'] = doc
    b_written.set()

  state = await _run_parallel_calls(
      add_label,
      [{'label': 'a'}, {'label': 'b'}],
      initial_state={'doc': {'entities': [{'name': 'e1', 'labels': []}]}},
  )

  assert state['doc'] == {'entities': [{'name': 'e1', 'labels': ['b', 'a']}]}


@pytest.mark.asyncio
async def test_parallel_writes_of_separate_dict_keys_are_all_kept():
  async def save_draft(draft_id: str, tool_context: ToolContext) -> None:
    tool_context.state['drafts'] = {draft_id: 'body'}

  state = await _run_parallel_calls(
      save_draft, [{'draft_id': 'a'}, {'draft_id': 'b'}]
  )

  assert state['drafts'] == {'a': 'body', 'b': 'body'}


@pytest.mark.asyncio
async def test_parallel_list_writes_bypassing_state_keep_call_order():
  async def set_items(item: str, tool_context: ToolContext) -> None:
    tool_context.actions.state_delta['items'] = [item]

  state = await _run_parallel_calls(
      set_items,
      [{'item': 'a'}, {'item': 'b'}],
      initial_state={'items': ['old']},
  )

  assert state['items'] == ['b']


@pytest.mark.asyncio
async def test_parallel_scalar_writes_bypassing_state_keep_call_order():
  async def set_done(done: str, tool_context: ToolContext) -> None:
    tool_context.actions.state_delta['done'] = done == 'yes'

  state = await _run_parallel_calls(
      set_done,
      [{'done': 'no'}, {'done': 'yes'}],
      initial_state={'done': False},
  )

  assert state['done'] is True
