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

import contextvars
from typing import Any
from unittest import mock

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event_actions import EventActions
from google.adk.flows.llm_flows import _tool_caller
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest


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
  invocation_context.plugin_manager.run_after_tool_callback.return_value = None

  agent = mock.create_autospec(LlmAgent, instance=True)
  agent.name = 'test_agent'
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
      override_response={'error': 'Tool not found'},
      tool_lookup_error=ValueError('Tool missing_tool not found'),
      is_tool_lookup_failure=True,
  )

  invocation_context = mock.create_autospec(InvocationContext, instance=True)
  invocation_context.invocation_id = 'inv-1'
  invocation_context.branch = 'main'
  invocation_context.agent = mock.Mock()
  invocation_context.agent.name = 'test_agent'

  agent = mock.create_autospec(LlmAgent, instance=True)
  agent.name = 'test_agent'

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
  assert event is not None
  assert event.content is not None
  assert event.content.parts is not None
  fr = event.content.parts[0].function_response
  assert fr is not None
  assert fr.response == {
      'error': 'Tool not found',
  }
