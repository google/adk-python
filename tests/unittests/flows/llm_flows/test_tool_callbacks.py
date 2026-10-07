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

import contextvars
from typing import Any
from unittest import mock

from google.adk.agents.llm_agent import Agent
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from google.genai.types import Part
from pydantic import BaseModel
import pytest

from ... import testing_utils


def simple_function(input_str: str) -> str:
  return {'result': input_str}


def simple_function_with_error() -> str:
  raise SystemError('simple_function_with_error')


class MockBeforeToolCallback(BaseModel):
  """Mock before tool callback."""

  mock_response: dict[str, object]
  modify_tool_request: bool = False

  def __call__(
      self,
      tool: BaseTool,
      args: dict[str, Any],
      tool_context: ToolContext,
  ) -> dict[str, object]:
    if self.modify_tool_request:
      args['input_str'] = 'modified_input'
      return None
    return self.mock_response


class MockAfterToolCallback(BaseModel):
  """Mock after tool callback."""

  mock_response: dict[str, object]
  modify_tool_request: bool = False
  modify_tool_response: bool = False

  def __call__(
      self,
      tool: BaseTool,
      args: dict[str, Any],
      tool_context: ToolContext,
      tool_response: dict[str, Any] = None,
  ) -> dict[str, object]:
    if self.modify_tool_request:
      args['input_str'] = 'modified_input'
      return None
    if self.modify_tool_response:
      tool_response['result'] = 'modified_output'
      return tool_response
    return self.mock_response


class MockOnToolErrorCallback(BaseModel):
  """Mock on tool error callback."""

  mock_response: dict[str, object]
  modify_tool_response: bool = False

  def __call__(
      self,
      tool: BaseTool,
      args: dict[str, Any],
      tool_context: ToolContext,
      error: Exception,
  ) -> dict[str, object]:
    if self.modify_tool_response:
      return self.mock_response
    return None


def noop_callback(
    **kwargs,
) -> dict[str, object]:
  pass


def test_before_tool_callback():
  """Test that the before_tool_callback is called before the tool is called."""
  responses = [
      types.Part.from_function_call(name='simple_function', args={}),
      'response1',
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)
  agent = Agent(
      name='root_agent',
      model=mock_model,
      before_tool_callback=MockBeforeToolCallback(
          mock_response={'test': 'before_tool_callback'}
      ),
      tools=[simple_function],
  )

  runner = testing_utils.InMemoryRunner(agent)
  assert testing_utils.simplify_events(runner.run('test')) == [
      ('root_agent', Part.from_function_call(name='simple_function', args={})),
      (
          'root_agent',
          Part.from_function_response(
              name='simple_function', response={'test': 'before_tool_callback'}
          ),
      ),
      ('root_agent', 'response1'),
  ]


def test_before_tool_callback_noop():
  """Test that the before_tool_callback is a no-op when not overridden."""
  responses = [
      types.Part.from_function_call(
          name='simple_function', args={'input_str': 'simple_function_call'}
      ),
      'response1',
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)
  agent = Agent(
      name='root_agent',
      model=mock_model,
      before_tool_callback=noop_callback,
      tools=[simple_function],
  )

  runner = testing_utils.InMemoryRunner(agent)
  assert testing_utils.simplify_events(runner.run('test')) == [
      (
          'root_agent',
          Part.from_function_call(
              name='simple_function', args={'input_str': 'simple_function_call'}
          ),
      ),
      (
          'root_agent',
          Part.from_function_response(
              name='simple_function',
              response={'result': 'simple_function_call'},
          ),
      ),
      ('root_agent', 'response1'),
  ]


def test_before_tool_callback_modify_tool_request():
  """Test that the before_tool_callback modifies the tool request."""
  responses = [
      types.Part.from_function_call(name='simple_function', args={}),
      'response1',
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)
  agent = Agent(
      name='root_agent',
      model=mock_model,
      before_tool_callback=MockBeforeToolCallback(
          mock_response={'test': 'before_tool_callback'},
          modify_tool_request=True,
      ),
      tools=[simple_function],
  )

  runner = testing_utils.InMemoryRunner(agent)
  assert testing_utils.simplify_events(runner.run('test')) == [
      ('root_agent', Part.from_function_call(name='simple_function', args={})),
      (
          'root_agent',
          Part.from_function_response(
              name='simple_function',
              response={'result': 'modified_input'},
          ),
      ),
      ('root_agent', 'response1'),
  ]


def test_after_tool_callback():
  """Test that the after_tool_callback is called after the tool is called."""
  responses = [
      types.Part.from_function_call(
          name='simple_function', args={'input_str': 'simple_function_call'}
      ),
      'response1',
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)
  agent = Agent(
      name='root_agent',
      model=mock_model,
      after_tool_callback=MockAfterToolCallback(
          mock_response={'test': 'after_tool_callback'}
      ),
      tools=[simple_function],
  )

  runner = testing_utils.InMemoryRunner(agent)
  assert testing_utils.simplify_events(runner.run('test')) == [
      (
          'root_agent',
          Part.from_function_call(
              name='simple_function', args={'input_str': 'simple_function_call'}
          ),
      ),
      (
          'root_agent',
          Part.from_function_response(
              name='simple_function', response={'test': 'after_tool_callback'}
          ),
      ),
      ('root_agent', 'response1'),
  ]


def test_after_tool_callback_noop():
  """Test that the after_tool_callback is a no-op when not overridden."""
  responses = [
      types.Part.from_function_call(
          name='simple_function', args={'input_str': 'simple_function_call'}
      ),
      'response1',
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)
  agent = Agent(
      name='root_agent',
      model=mock_model,
      after_tool_callback=noop_callback,
      tools=[simple_function],
  )

  runner = testing_utils.InMemoryRunner(agent)
  assert testing_utils.simplify_events(runner.run('test')) == [
      (
          'root_agent',
          Part.from_function_call(
              name='simple_function', args={'input_str': 'simple_function_call'}
          ),
      ),
      (
          'root_agent',
          Part.from_function_response(
              name='simple_function',
              response={'result': 'simple_function_call'},
          ),
      ),
      ('root_agent', 'response1'),
  ]


def test_after_tool_callback_modify_tool_response():
  """Test that the after_tool_callback modifies the tool response."""
  responses = [
      types.Part.from_function_call(
          name='simple_function', args={'input_str': 'simple_function_call'}
      ),
      'response1',
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)
  agent = Agent(
      name='root_agent',
      model=mock_model,
      after_tool_callback=MockAfterToolCallback(
          mock_response={'result': 'after_tool_callback'},
          modify_tool_response=True,
      ),
      tools=[simple_function],
  )

  runner = testing_utils.InMemoryRunner(agent)
  assert testing_utils.simplify_events(runner.run('test')) == [
      (
          'root_agent',
          Part.from_function_call(
              name='simple_function', args={'input_str': 'simple_function_call'}
          ),
      ),
      (
          'root_agent',
          Part.from_function_response(
              name='simple_function',
              response={'result': 'modified_output'},
          ),
      ),
      ('root_agent', 'response1'),
  ]


def test_on_tool_error_callback_tool_not_found_noop():
  """Test that a no-op on_tool_error_callback keeps the default error response."""
  responses = [
      types.Part.from_function_call(
          name='nonexistent_function',
          args={'input_str': 'simple_function_call'},
      ),
      'response1',
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)
  agent = Agent(
      name='root_agent',
      model=mock_model,
      on_tool_error_callback=noop_callback,
      tools=[simple_function],
  )

  runner = testing_utils.InMemoryRunner(agent)
  events = runner.run('test')

  assert testing_utils.simplify_events(events)[-1] == (
      'root_agent',
      'response1',
  )
  function_response = events[1].content.parts[0].function_response
  assert function_response.name == 'nonexistent_function'
  assert 'nonexistent_function' in function_response.response['error']


def test_on_tool_error_callback_tool_not_found_modify_tool_response():
  """Test that the on_tool_error_callback modifies the tool response when the tool is not found."""
  responses = [
      types.Part.from_function_call(
          name='nonexistent_function',
          args={'input_str': 'simple_function_call'},
      ),
      'response1',
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)
  agent = Agent(
      name='root_agent',
      model=mock_model,
      on_tool_error_callback=MockOnToolErrorCallback(
          mock_response={'result': 'on_tool_error_callback_response'},
          modify_tool_response=True,
      ),
      tools=[simple_function],
  )

  runner = testing_utils.InMemoryRunner(agent)
  assert testing_utils.simplify_events(runner.run('test')) == [
      (
          'root_agent',
          Part.from_function_call(
              name='nonexistent_function',
              args={'input_str': 'simple_function_call'},
          ),
      ),
      (
          'root_agent',
          Part.from_function_response(
              name='nonexistent_function',
              response={'result': 'on_tool_error_callback_response'},
          ),
      ),
      ('root_agent', 'response1'),
  ]


def test_on_tool_error_callback_stops_on_empty_dict():
  """Test that an empty error recovery response stops the callback chain."""

  def empty_response_callback(tool, args, tool_context, error):
    return {}

  unexpected_callback = mock.Mock(
      side_effect=AssertionError('callback chain should have stopped')
  )
  responses = [
      types.Part.from_function_call(name='missing_tool', args={}),
      'response1',
  ]
  agent = Agent(
      name='root_agent',
      model=testing_utils.MockModel.create(responses=responses),
      on_tool_error_callback=[empty_response_callback, unexpected_callback],
      tools=[simple_function],
  )

  events = testing_utils.InMemoryRunner(agent).run('test')

  assert testing_utils.simplify_events(events) == [
      (
          'root_agent',
          Part.from_function_call(name='missing_tool', args={}),
      ),
      (
          'root_agent',
          Part.from_function_response(name='missing_tool', response={}),
      ),
      ('root_agent', 'response1'),
  ]
  unexpected_callback.assert_not_called()


async def test_on_tool_error_callback_tool_error_noop():
  """Test that the on_tool_error_callback is a no-op when the tool returns an error."""
  responses = [
      types.Part.from_function_call(
          name='simple_function_with_error',
          args={},
      ),
      'response1',
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)
  agent = Agent(
      name='root_agent',
      model=mock_model,
      on_tool_error_callback=noop_callback,
      tools=[simple_function_with_error],
  )

  runner = testing_utils.InMemoryRunner(agent)
  with pytest.raises(SystemError):
    await runner.run_async('test')


def test_on_tool_error_callback_tool_error_modify_tool_response():
  """Test that the on_tool_error_callback modifies the tool response when the tool returns an error."""

  async def async_on_tool_error_callback(
      tool: BaseTool,
      args: dict[str, Any],
      tool_context: ToolContext,
      error: Exception,
  ) -> dict[str, object]:
    if tool.name == 'simple_function_with_error':
      return {'result': 'async_on_tool_error_callback_response'}
    return None

  responses = [
      types.Part.from_function_call(
          name='simple_function_with_error',
          args={},
      ),
      'response1',
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)
  agent = Agent(
      name='root_agent',
      model=mock_model,
      on_tool_error_callback=async_on_tool_error_callback,
      tools=[simple_function_with_error],
  )

  runner = testing_utils.InMemoryRunner(agent)
  assert testing_utils.simplify_events(runner.run('test')) == [
      (
          'root_agent',
          Part.from_function_call(
              name='simple_function_with_error',
              args={},
          ),
      ),
      (
          'root_agent',
          Part.from_function_response(
              name='simple_function_with_error',
              response={'result': 'async_on_tool_error_callback_response'},
          ),
      ),
      ('root_agent', 'response1'),
  ]


def test_before_tool_callback_lambda_with_arbitrary_param_names():
  """Test that before_tool_callback works with lambda having non-matching param names."""
  captured = []
  responses = [
      types.Part.from_function_call(name='simple_function', args={}),
      'response1',
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)
  agent = Agent(
      name='root_agent',
      model=mock_model,
      before_tool_callback=lambda t, a, tc: captured.append((t.name, a)),
      tools=[simple_function],
  )

  runner = testing_utils.InMemoryRunner(agent)
  runner.run('test')
  assert len(captured) == 1
  assert captured[0] == ('simple_function', {})


_ambient_value: contextvars.ContextVar[str] = contextvars.ContextVar(
    'ambient_value', default='unset'
)


def read_ambient_value() -> dict[str, object]:
  return {'result': _ambient_value.get()}


def test_before_tool_callback_contextvar_reaches_the_tool():
  """A contextvar set in before_tool_callback is still set when the tool runs."""

  def before_tool_callback(
      tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
  ) -> None:
    _ambient_value.set('set-by-callback')
    return None

  responses = [
      types.Part.from_function_call(name='read_ambient_value', args={}),
      'response1',
  ]
  agent = Agent(
      name='root_agent',
      model=testing_utils.MockModel.create(responses=responses),
      before_tool_callback=before_tool_callback,
      tools=[read_ambient_value],
  )

  events = testing_utils.InMemoryRunner(agent).run('test')

  assert testing_utils.simplify_events(events)[1] == (
      'root_agent',
      Part.from_function_response(
          name='read_ambient_value', response={'result': 'set-by-callback'}
      ),
  )
  # The callback ran in a task of its own, so it must not have leaked its value
  # into the caller.
  assert _ambient_value.get() == 'unset'


def test_before_tool_callback_contextvars_isolated_between_parallel_calls():
  """Parallel calls each see their own callback's contextvar, not a sibling's."""
  call_index = 0

  def before_tool_callback(
      tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
  ) -> None:
    nonlocal call_index
    call_index += 1
    _ambient_value.set(f'call-{call_index}')
    return None

  responses = [
      [
          types.Part.from_function_call(name='read_ambient_value', args={}),
          types.Part.from_function_call(name='read_ambient_value', args={}),
      ],
      'response1',
  ]
  agent = Agent(
      name='root_agent',
      model=testing_utils.MockModel.create(responses=responses),
      before_tool_callback=before_tool_callback,
      tools=[read_ambient_value],
  )

  events = testing_utils.InMemoryRunner(agent).run('test')

  assert testing_utils.simplify_events(events)[1] == (
      'root_agent',
      [
          Part.from_function_response(
              name='read_ambient_value', response={'result': 'call-1'}
          ),
          Part.from_function_response(
              name='read_ambient_value', response={'result': 'call-2'}
          ),
      ],
  )
