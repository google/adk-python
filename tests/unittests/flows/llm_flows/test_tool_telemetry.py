# Copyright 2025 Google LLC
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

from typing import Any
from typing import Dict
from typing import Optional
from unittest import mock

from google.adk import telemetry
from google.adk.agents.llm_agent import Agent
from google.adk.events.event import Event
from google.adk.flows.llm_flows.functions import handle_function_calls_async
from google.adk.tools.function_tool import FunctionTool
from google.genai import types

from ... import testing_utils


async def invoke_tool() -> Optional[Event]:
  def simple_fn(**kwargs) -> Dict[str, Any]:
    return {'result': 'test'}

  tool = FunctionTool(simple_fn)
  model = testing_utils.MockModel.create(responses=[])
  agent = Agent(
      name='agent',
      model=model,
      tools=[tool],
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content=''
  )
  function_call = types.FunctionCall(name=tool.name, args={'a': 1, 'b': 2})
  content = types.Content(parts=[types.Part(function_call=function_call)])
  event = Event(
      invocation_id=invocation_context.invocation_id,
      author=agent.name,
      content=content,
  )
  tools_dict = {tool.name: tool}
  return await handle_function_calls_async(
      invocation_context,
      event,
      tools_dict,
  )


async def test_simple_function_with_mocked_tracer(monkeypatch):
  mock_start_as_current_span_func = mock.Mock()
  returned_context_manager_mock = mock.MagicMock()
  returned_context_manager_mock.__enter__.return_value = mock.Mock(
      name='span_mock'
  )
  mock_start_as_current_span_func.return_value = returned_context_manager_mock

  monkeypatch.setattr(
      telemetry.tracer, 'start_as_current_span', mock_start_as_current_span_func
  )

  mock_adk_trace_tool_call = mock.Mock()
  monkeypatch.setattr(
      'google.adk.flows.llm_flows.functions.trace_tool_call',
      mock_adk_trace_tool_call,
  )

  event = await invoke_tool()
  assert event is not None

  event = await invoke_tool()
  assert event is not None

  expected_span_name = 'execute_tool simple_fn'

  assert mock_start_as_current_span_func.call_count == 2
  mock_start_as_current_span_func.assert_any_call(expected_span_name)

  assert returned_context_manager_mock.__enter__.call_count == 2
  assert returned_context_manager_mock.__exit__.call_count == 2

  assert mock_adk_trace_tool_call.call_count == 2
  for call_args_item in mock_adk_trace_tool_call.call_args_list:
    kwargs = call_args_item.kwargs
    assert kwargs['tool'].name == 'simple_fn'
    assert kwargs['args'] == {'a': 1, 'b': 2}
    assert 'function_response_event' in kwargs
    assert kwargs['function_response_event'].content.parts[
        0
    ].function_response.response == {'result': 'test'}
