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

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.agent_tool import AgentTool
from google.genai.types import Part
from pydantic import BaseModel
import pytest
from pytest import mark

from .. import testing_utils

pytestmark = pytest.mark.skip(
    reason='Skipping until tool.func evaluations are fixed (async)'
)


function_call_custom = Part.from_function_call(
    name='tool_agent', args={'custom_input': 'test1'}
)

function_call_no_schema = Part.from_function_call(
    name='tool_agent', args={'request': 'test1'}
)

function_response_custom = Part.from_function_response(
    name='tool_agent', response={'custom_output': 'response1'}
)

function_response_no_schema = Part.from_function_response(
    name='tool_agent', response={'result': 'response1'}
)


def change_state_callback(callback_context: CallbackContext):
  callback_context.state['state_1'] = 'changed_value'
  print('change_state_callback: ', callback_context.state)


def test_no_schema():
  mock_model = testing_utils.MockModel.create(
      responses=[
          function_call_no_schema,
          'response1',
          'response2',
      ]
  )

  tool_agent = Agent(
      name='tool_agent',
      model=mock_model,
  )

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[AgentTool(agent=tool_agent)],
  )

  runner = testing_utils.InMemoryRunner(root_agent)

  assert testing_utils.simplify_events(runner.run('test1')) == [
      ('root_agent', function_call_no_schema),
      ('root_agent', function_response_no_schema),
      ('root_agent', 'response2'),
  ]


def test_update_state():
  """The agent tool can read and change parent state."""

  mock_model = testing_utils.MockModel.create(
      responses=[
          function_call_no_schema,
          '{"custom_output": "response1"}',
          'response2',
      ]
  )

  tool_agent = Agent(
      name='tool_agent',
      model=mock_model,
      instruction='input: {state_1}',
      before_agent_callback=change_state_callback,
  )

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[AgentTool(agent=tool_agent)],
  )

  runner = testing_utils.InMemoryRunner(root_agent)
  runner.session.state['state_1'] = 'state1_value'

  runner.run('test1')
  assert (
      'input: changed_value' in mock_model.requests[1].config.system_instruction
  )
  assert runner.session.state['state_1'] == 'changed_value'


@mark.parametrize(
    'env_variables',
    [
        'GOOGLE_AI',
        # TODO(wanyif): re-enable after fix.
        # 'VERTEX',
    ],
    indirect=True,
)
def test_custom_schema():
  class CustomInput(BaseModel):
    custom_input: str

  class CustomOutput(BaseModel):
    custom_output: str

  mock_model = testing_utils.MockModel.create(
      responses=[
          function_call_custom,
          '{"custom_output": "response1"}',
          'response2',
      ]
  )

  tool_agent = Agent(
      name='tool_agent',
      model=mock_model,
      input_schema=CustomInput,
      output_schema=CustomOutput,
      output_key='tool_output',
  )

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[AgentTool(agent=tool_agent)],
  )

  runner = testing_utils.InMemoryRunner(root_agent)
  runner.session.state['state_1'] = 'state1_value'

  assert testing_utils.simplify_events(runner.run('test1')) == [
      ('root_agent', function_call_custom),
      ('root_agent', function_response_custom),
      ('root_agent', 'response2'),
  ]

  assert runner.session.state['tool_output'] == {'custom_output': 'response1'}

  assert len(mock_model.requests) == 3
  # The second request is the tool agent request.
  assert mock_model.requests[1].config.response_schema == CustomOutput
  assert mock_model.requests[1].config.response_mime_type == 'application/json'
