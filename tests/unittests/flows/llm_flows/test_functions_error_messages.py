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

"""Tests for enhanced error messages in function tool handling."""

from google.adk.agents.llm_agent import Agent
from google.adk.flows.llm_flows.functions import _get_tool
from google.adk.tools import BaseTool
from google.genai import types
import pytest

from ... import testing_utils


# Mock tool for testing error messages
class MockTool(BaseTool):
  """Mock tool for testing error messages."""

  def __init__(self, name: str = 'mock_tool'):
    super().__init__(name=name, description=f'Mock tool: {name}')

  def call(self, *args, **kwargs):
    return 'mock_response'


def test_tool_not_found_enhanced_error():
  """Verify enhanced error message for tool not found."""
  function_call = types.FunctionCall(name='nonexistent_tool', args={})
  tools_dict = {
      'get_weather': MockTool(name='get_weather'),
      'calculate_sum': MockTool(name='calculate_sum'),
      'search_database': MockTool(name='search_database'),
  }

  with pytest.raises(ValueError) as exc_info:
    _get_tool(function_call, tools_dict)

  error_msg = str(exc_info.value)

  # Verify error message components
  assert 'nonexistent_tool' in error_msg
  assert 'Available tools:' in error_msg
  assert 'get_weather' in error_msg
  assert 'Possible causes:' in error_msg
  assert 'Suggested fixes:' in error_msg


def test_tool_not_found_with_different_name():
  """Verify error message contains basic information."""
  function_call = types.FunctionCall(name='completely_different', args={})
  tools_dict = {
      'get_weather': MockTool(name='get_weather'),
      'calculate_sum': MockTool(name='calculate_sum'),
  }

  with pytest.raises(ValueError) as exc_info:
    _get_tool(function_call, tools_dict)

  error_msg = str(exc_info.value)

  # Verify error message contains basic information
  assert 'completely_different' in error_msg
  assert 'Available tools:' in error_msg


def test_tool_call_without_name_is_rejected():
  """Verify a malformed unnamed function call has a useful error."""
  function_call = types.FunctionCall(args={})

  with pytest.raises(ValueError, match="Tool 'None' not found"):
    _get_tool(function_call, {'get_weather': MockTool(name='get_weather')})


def test_tool_not_found_shows_all_tools():
  """Verify error message shows all tools (no truncation)."""
  function_call = types.FunctionCall(name='nonexistent', args={})

  # Create 100 tools
  tools_dict = {f'tool_{i}': MockTool(name=f'tool_{i}') for i in range(100)}

  with pytest.raises(ValueError) as exc_info:
    _get_tool(function_call, tools_dict)

  error_msg = str(exc_info.value)

  # Verify all tools are shown (no truncation)
  assert 'tool_0' in error_msg  # First tool shown
  assert 'tool_99' in error_msg  # Last tool also shown
  assert 'showing first 20 of' not in error_msg  # No truncation message


def test_unknown_tool_name_is_reported_to_the_model():
  """A call to a tool that is not registered continues the invocation."""

  def get_weather(city: str) -> str:
    return f'sunny in {city}'

  mock_model = testing_utils.MockModel.create(
      responses=[
          types.Part.from_function_call(name='no_such_tool', args={}),
          'sorry, let me try again',
      ]
  )
  agent = Agent(name='root_agent', model=mock_model, tools=[get_weather])
  runner = testing_utils.InMemoryRunner(agent)

  events = runner.run('what is the weather')

  assert testing_utils.simplify_events(events)[-1] == (
      'root_agent',
      'sorry, let me try again',
  )
  function_response = events[1].content.parts[0].function_response
  assert function_response.name == 'no_such_tool'
  error = function_response.response['error']
  assert 'no_such_tool' in error
  assert 'get_weather' in error
