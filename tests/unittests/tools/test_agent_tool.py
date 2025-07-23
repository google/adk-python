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

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import Agent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.tools.agent_tool import AgentTool
from google.adk.utils.variant_utils import GoogleLLMVariant
from google.genai import types
from google.genai.types import Part
from pydantic import BaseModel
from pytest import mark

from .. import testing_utils

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


def test_update_artifacts():
  """The agent tool can read and write artifacts."""

  async def before_tool_agent(callback_context: CallbackContext):
    # Artifact 1 should be available in the tool agent.
    artifact = await callback_context.load_artifact('artifact_1')
    await callback_context.save_artifact(
        'artifact_2', Part.from_text(text=artifact.text + ' 2')
    )

  tool_agent = SequentialAgent(
      name='tool_agent',
      before_agent_callback=before_tool_agent,
  )

  async def before_main_agent(callback_context: CallbackContext):
    await callback_context.save_artifact(
        'artifact_1', Part.from_text(text='test')
    )

  async def after_main_agent(callback_context: CallbackContext):
    # Artifact 2 should be available after the tool agent.
    artifact_2 = await callback_context.load_artifact('artifact_2')
    await callback_context.save_artifact(
        'artifact_3', Part.from_text(text=artifact_2.text + ' 3')
    )

  mock_model = testing_utils.MockModel.create(
      responses=[function_call_no_schema, 'response2']
  )
  root_agent = Agent(
      name='root_agent',
      before_agent_callback=before_main_agent,
      after_agent_callback=after_main_agent,
      tools=[AgentTool(agent=tool_agent)],
      model=mock_model,
  )

  runner = testing_utils.InMemoryRunner(root_agent)
  runner.run('test1')

  artifacts_path = f'test_app/test_user/{runner.session_id}'
  assert runner.runner.artifact_service.artifacts == {
      f'{artifacts_path}/artifact_1': [Part.from_text(text='test')],
      f'{artifacts_path}/artifact_2': [Part.from_text(text='test 2')],
      f'{artifacts_path}/artifact_3': [Part.from_text(text='test 2 3')],
  }


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


@mark.parametrize(
    'env_variables',
    [
        'VERTEX',  # Test VERTEX_AI variant
    ],
    indirect=True,
)
def test_agent_tool_response_schema_no_output_schema_vertex_ai():
  """Test AgentTool with no output schema has string response schema for VERTEX_AI."""
  tool_agent = Agent(
      name='tool_agent',
      model=testing_utils.MockModel.create(responses=['test response']),
  )

  agent_tool = AgentTool(agent=tool_agent)
  declaration = agent_tool._get_declaration()

  assert declaration.name == 'tool_agent'
  assert declaration.parameters.type == 'OBJECT'
  assert declaration.parameters.properties['request'].type == 'STRING'
  # Should have string response schema for VERTEX_AI
  assert declaration.response is not None
  assert declaration.response.type == types.Type.STRING


@mark.parametrize(
    'env_variables',
    [
        'VERTEX',  # Test VERTEX_AI variant
    ],
    indirect=True,
)
def test_agent_tool_response_schema_with_output_schema_vertex_ai():
  """Test AgentTool with output schema has object response schema for VERTEX_AI."""

  class CustomOutput(BaseModel):
    custom_output: str

  tool_agent = Agent(
      name='tool_agent',
      model=testing_utils.MockModel.create(responses=['test response']),
      output_schema=CustomOutput,
  )

  agent_tool = AgentTool(agent=tool_agent)
  declaration = agent_tool._get_declaration()

  assert declaration.name == 'tool_agent'
  # Should have object response schema for VERTEX_AI when output_schema exists
  assert declaration.response is not None
  assert declaration.response.type == types.Type.OBJECT


@mark.parametrize(
    'env_variables',
    [
        'GOOGLE_AI',  # Test GEMINI_API variant
    ],
    indirect=True,
)
def test_agent_tool_response_schema_gemini_api():
  """Test AgentTool with GEMINI_API variant has no response schema."""

  class CustomOutput(BaseModel):
    custom_output: str

  tool_agent = Agent(
      name='tool_agent',
      model=testing_utils.MockModel.create(responses=['test response']),
      output_schema=CustomOutput,
  )

  agent_tool = AgentTool(agent=tool_agent)
  declaration = agent_tool._get_declaration()

  assert declaration.name == 'tool_agent'
  # GEMINI_API should not have response schema
  assert declaration.response is None


@mark.parametrize(
    'env_variables',
    [
        'VERTEX',  # Test VERTEX_AI variant
    ],
    indirect=True,
)
def test_agent_tool_response_schema_with_input_schema_vertex_ai():
  """Test AgentTool with input and output schemas for VERTEX_AI."""

  class CustomInput(BaseModel):
    custom_input: str

  class CustomOutput(BaseModel):
    custom_output: str

  tool_agent = Agent(
      name='tool_agent',
      model=testing_utils.MockModel.create(responses=['test response']),
      input_schema=CustomInput,
      output_schema=CustomOutput,
  )

  agent_tool = AgentTool(agent=tool_agent)
  declaration = agent_tool._get_declaration()

  assert declaration.name == 'tool_agent'
  assert declaration.parameters.type == 'OBJECT'
  assert declaration.parameters.properties['custom_input'].type == 'STRING'
  # Should have object response schema for VERTEX_AI when output_schema exists
  assert declaration.response is not None
  assert declaration.response.type == types.Type.OBJECT


@mark.parametrize(
    'env_variables',
    [
        'VERTEX',  # Test VERTEX_AI variant
    ],
    indirect=True,
)
def test_agent_tool_response_schema_with_input_schema_no_output_vertex_ai():
  """Test AgentTool with input schema but no output schema for VERTEX_AI."""

  class CustomInput(BaseModel):
    custom_input: str

  tool_agent = Agent(
      name='tool_agent',
      model=testing_utils.MockModel.create(responses=['test response']),
      input_schema=CustomInput,
  )

  agent_tool = AgentTool(agent=tool_agent)
  declaration = agent_tool._get_declaration()

  assert declaration.name == 'tool_agent'
  assert declaration.parameters.type == 'OBJECT'
  assert declaration.parameters.properties['custom_input'].type == 'STRING'
  # Should have string response schema for VERTEX_AI when no output_schema
  assert declaration.response is not None
  assert declaration.response.type == types.Type.STRING
