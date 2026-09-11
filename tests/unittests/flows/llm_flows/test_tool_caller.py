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

"""Unit tests for _tool_caller."""

from __future__ import annotations

import asyncio
import contextvars
from typing import Any
from unittest import mock

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event_actions import EventActions
from google.adk.flows.llm_flows import _tool_caller
from google.adk.flows.llm_flows import functions
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest

from ... import testing_utils


def test_normalize_tool_result() -> None:
  assert _tool_caller._normalize_tool_result({'foo': 'bar'}) == {'foo': 'bar'}
  assert _tool_caller._normalize_tool_result('hello') == {'result': 'hello'}
  assert _tool_caller._normalize_tool_result(123) == {'result': 123}
  assert _tool_caller._normalize_tool_result([1, 2]) == {'result': [1, 2]}


def test_as_callback_result() -> None:
  assert _tool_caller._as_callback_result({'a': 1}) == {'a': 1}


def test_build_function_response_content() -> None:
  tool = BaseTool(name='my_tool', description='desc')
  content = _tool_caller._build_function_response_content(
      tool=tool,
      function_result={'status': 'ok'},
      function_call_id='call-123',
  )
  assert content.role == 'user'
  assert content.parts is not None
  assert len(content.parts) == 1
  fr = content.parts[0].function_response
  assert fr is not None
  assert fr.name == 'my_tool'
  assert fr.id == 'call-123'
  assert fr.response == {'status': 'ok'}


@pytest.mark.asyncio
async def test_execute_single_prepared_call_runs_tool_runner() -> None:
  tool = BaseTool(name='echo_tool', description='echo')
  tool_context = mock.create_autospec(ToolContext, instance=True)
  tool_context.actions = EventActions()
  tool_context.function_call_id = 'call-1'

  fc = types.FunctionCall(name='echo_tool', id='call-1', args={'val': 42})
  prepared = _tool_caller._PreparedFunctionCall(
      function_call=fc,
      tool=tool,
      tool_context=tool_context,
      function_args={'val': 42},
      contextvars_snapshot=contextvars.copy_context(),
  )

  invocation_context = mock.create_autospec(InvocationContext, instance=True)
  invocation_context.invocation_id = 'inv-1'
  invocation_context.branch = 'main'
  invocation_context.agent = mock.Mock()
  invocation_context.agent.name = 'test_agent'
  invocation_context.plugin_manager = mock.AsyncMock()
  invocation_context.plugin_manager.run_before_tool_callback.return_value = None
  invocation_context.plugin_manager.run_after_tool_callback.return_value = None

  agent = mock.create_autospec(LlmAgent, instance=True)
  agent.name = 'test_agent'
  agent.canonical_before_tool_callbacks = []
  agent.canonical_after_tool_callbacks = []

  runner_called = False

  async def mock_runner() -> dict[str, Any]:
    nonlocal runner_called
    runner_called = True
    return {'val': 84}

  event = await _tool_caller._execute_single_prepared_call(
      invocation_context,
      prepared,
      agent,
      tool_runner=mock_runner,
  )

  assert runner_called
  assert event is not None
  assert event.content is not None
  assert event.content.parts is not None
  fr = event.content.parts[0].function_response
  assert fr is not None
  assert fr.response == {'val': 84}


@pytest.mark.asyncio
async def test_execute_single_prepared_call_lookup_failure() -> None:
  tool = BaseTool(name='missing_tool', description='desc')
  tool_context = mock.create_autospec(ToolContext, instance=True)
  tool_context.actions = EventActions()
  tool_context.function_call_id = 'call-missing'

  fc = types.FunctionCall(name='missing_tool', id='call-missing')
  prepared = _tool_caller._PreparedFunctionCall(
      function_call=fc,
      tool=tool,
      tool_context=tool_context,
      function_args={},
      contextvars_snapshot=contextvars.copy_context(),
      tools_dict={},
      tool_lookup_error=ValueError('Tool missing_tool not found'),
  )

  invocation_context = mock.create_autospec(InvocationContext, instance=True)
  invocation_context.invocation_id = 'inv-1'
  invocation_context.branch = 'main'
  invocation_context.agent = mock.Mock()
  invocation_context.agent.name = 'test_agent'
  invocation_context.plugin_manager = mock.AsyncMock()
  invocation_context.plugin_manager.run_before_tool_callback.return_value = None
  invocation_context.plugin_manager.run_on_tool_error_callback.return_value = (
      None
  )

  after_tool_calls: list[str] = []

  def after_tool(
      tool: BaseTool,
      args: dict[str, Any],
      tool_context: ToolContext,
      tool_response: dict[str, Any],
  ) -> None:
    after_tool_calls.append(tool.name)

  agent = mock.create_autospec(LlmAgent, instance=True)
  agent.name = 'test_agent'
  agent.canonical_before_tool_callbacks = []
  agent.canonical_on_tool_error_callbacks = []
  agent.canonical_after_tool_callbacks = [after_tool]

  runner_called = False

  async def mock_runner() -> dict[str, Any]:
    nonlocal runner_called
    runner_called = True
    return {}

  event = await _tool_caller._execute_single_prepared_call(
      invocation_context,
      prepared,
      agent,
      tool_runner=mock_runner,
  )

  # Tool runner must NOT be called on lookup failure
  assert not runner_called
  # Nor the after-tool callbacks: they describe a run that never happened.
  assert not after_tool_calls
  invocation_context.plugin_manager.run_after_tool_callback.assert_not_awaited()
  assert event is not None
  assert event.content is not None
  assert event.content.parts is not None
  fr = event.content.parts[0].function_response
  assert fr is not None
  assert fr.response is not None
  assert 'missing_tool' in fr.response['error']


@pytest.mark.asyncio
async def test_lookup_failure_answerable_by_before_callback() -> None:
  tool = BaseTool(name='missing_tool', description='desc')
  tool_context = mock.create_autospec(ToolContext, instance=True)
  tool_context.actions = EventActions()
  tool_context.function_call_id = 'call-missing'

  fc = types.FunctionCall(name='missing_tool', id='call-missing')
  prepared = _tool_caller._PreparedFunctionCall(
      function_call=fc,
      tool=tool,
      tool_context=tool_context,
      function_args={},
      contextvars_snapshot=contextvars.copy_context(),
      tools_dict={},
      tool_lookup_error=ValueError('Tool missing_tool not found'),
  )

  invocation_context = mock.create_autospec(InvocationContext, instance=True)
  invocation_context.invocation_id = 'inv-1'
  invocation_context.branch = 'main'
  invocation_context.agent = mock.Mock()
  invocation_context.agent.name = 'test_agent'
  invocation_context.plugin_manager = mock.AsyncMock()
  invocation_context.plugin_manager.run_before_tool_callback.return_value = None
  invocation_context.plugin_manager.run_after_tool_callback.return_value = None

  def before_tool(
      tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
  ) -> dict[str, Any]:
    return {'answered': True}

  agent = mock.create_autospec(LlmAgent, instance=True)
  agent.name = 'test_agent'
  agent.canonical_before_tool_callbacks = [before_tool]
  agent.canonical_after_tool_callbacks = []

  runner_called = False

  async def mock_runner() -> dict[str, Any]:
    nonlocal runner_called
    runner_called = True
    return {}

  event = await _tool_caller._execute_single_prepared_call(
      invocation_context,
      prepared,
      agent,
      tool_runner=mock_runner,
  )

  assert not runner_called
  assert event is not None
  assert event.content is not None
  assert event.content.parts is not None
  fr = event.content.parts[0].function_response
  assert fr is not None
  assert fr.response == {'answered': True}


@pytest.mark.asyncio
async def test_tool_callbacks_pair_up_when_nothing_in_the_call_awaits() -> None:
  order: list[str] = []
  bookkeeping: dict[str, Any] = {}
  pairings: list[bool] = []

  def record(value: int) -> dict[str, int]:
    return {'value': value}

  def before_tool(
      tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
  ) -> None:
    bookkeeping['value'] = args['value']
    order.append(f'before:{args["value"]}')

  def after_tool(
      tool: BaseTool,
      args: dict[str, Any],
      tool_context: ToolContext,
      tool_response: dict[str, Any],
  ) -> None:
    pairings.append(bookkeeping['value'] == args['value'])
    order.append(f'after:{args["value"]}')

  agent = LlmAgent(
      name='test_agent',
      before_tool_callback=before_tool,
      after_tool_callback=after_tool,
  )
  invocation_context = await testing_utils.create_invocation_context(agent)

  await functions.handle_function_call_list_async(
      invocation_context,
      [
          types.FunctionCall(name='record', id='call-1', args={'value': 1}),
          types.FunctionCall(name='record', id='call-2', args={'value': 2}),
      ],
      {'record': FunctionTool(record)},
  )

  assert order == ['before:1', 'after:1', 'before:2', 'after:2']
  assert pairings == [True, True]


@pytest.mark.asyncio
async def test_awaiting_tool_callbacks_keep_their_state_per_call() -> None:
  order: list[str] = []
  pairings: list[bool] = []

  async def record(value: int) -> dict[str, int]:
    order.append(f'tool-start:{value}')
    await asyncio.sleep(0)
    order.append(f'tool-end:{value}')
    return {'value': value}

  async def before_tool(
      tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
  ) -> None:
    await asyncio.sleep(0)
    tool_context.state['seen'] = args['value']
    order.append(f'before:{args["value"]}')

  async def after_tool(
      tool: BaseTool,
      args: dict[str, Any],
      tool_context: ToolContext,
      tool_response: dict[str, Any],
  ) -> None:
    await asyncio.sleep(0)
    pairings.append(tool_context.state['seen'] == args['value'])
    order.append(f'after:{args["value"]}')

  agent = LlmAgent(
      name='test_agent',
      before_tool_callback=before_tool,
      after_tool_callback=after_tool,
  )
  invocation_context = await testing_utils.create_invocation_context(agent)

  await functions.handle_function_call_list_async(
      invocation_context,
      [
          types.FunctionCall(name='record', id='call-1', args={'value': 1}),
          types.FunctionCall(name='record', id='call-2', args={'value': 2}),
      ],
      {'record': FunctionTool(record)},
  )

  assert pairings == [True, True]
  for value in (1, 2):
    assert (
        order.index(f'before:{value}')
        < order.index(f'tool-start:{value}')
        < order.index(f'after:{value}')
    )
  # The tools still overlap; awaiting callbacks must not serialize the batch.
  assert order.index('tool-start:2') < order.index('tool-end:1')
