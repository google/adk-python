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

from google.adk.agents.llm_agent import Agent
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from google.genai.types import Part
from pydantic import BaseModel

from ... import testing_utils


def simple_function(input_str: str) -> str:
  return {'result': input_str}


class MockBeforeToolCallback(BaseModel):
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


def noop_callback(
    **kwargs,
) -> dict[str, object]:
  pass


def test_before_tool_callback():
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
