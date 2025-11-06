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
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.plugins.plugin_manager import PluginManager
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.tool_context import ToolContext
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


@mark.asyncio
async def test_agent_tool_inherits_parent_app_name(monkeypatch):
  parent_app_name = 'parent_app'
  captured: dict[str, str] = {}

  class RecordingSessionService(InMemorySessionService):

    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ):
      captured['session_app_name'] = app_name
      return await super().create_session(
          app_name=app_name,
          user_id=user_id,
          state=state,
          session_id=session_id,
      )

  monkeypatch.setattr(
      'google.adk.sessions.in_memory_session_service.InMemorySessionService',
      RecordingSessionService,
  )

  async def _empty_async_generator():
    if False:
      yield None

  class StubRunner:

    def __init__(
        self,
        *,
        app_name: str,
        agent: Agent,
        artifact_service,
        session_service,
        memory_service,
        credential_service,
        plugins,
    ):
      del artifact_service, memory_service, credential_service
      captured['runner_app_name'] = app_name
      self.agent = agent
      self.session_service = session_service
      self.plugin_manager = PluginManager(plugins=plugins)
      self.app_name = app_name

    def run_async(
        self,
        *,
        user_id: str,
        session_id: str,
        invocation_id: Optional[str] = None,
        new_message: Optional[types.Content] = None,
        state_delta: Optional[dict[str, Any]] = None,
        run_config: Optional[RunConfig] = None,
    ):
      del (
          user_id,
          session_id,
          invocation_id,
          new_message,
          state_delta,
          run_config,
      )
      return _empty_async_generator()

    async def close(self):
      """Mock close method."""
      pass

  monkeypatch.setattr('google.adk.runners.Runner', StubRunner)

  tool_agent = Agent(
      name='tool_agent',
      model='test-model',
  )
  agent_tool = AgentTool(agent=tool_agent)
  root_agent = Agent(
      name='root_agent',
      model='test-model',
      tools=[agent_tool],
  )

  artifact_service = InMemoryArtifactService()
  parent_session_service = InMemorySessionService()
  parent_session = await parent_session_service.create_session(
      app_name=parent_app_name,
      user_id='user',
  )
  invocation_context = InvocationContext(
      artifact_service=artifact_service,
      session_service=parent_session_service,
      memory_service=InMemoryMemoryService(),
      plugin_manager=PluginManager(),
      invocation_id='invocation-id',
      agent=root_agent,
      session=parent_session,
      run_config=RunConfig(),
  )
  tool_context = ToolContext(invocation_context)

  assert tool_context._invocation_context.app_name == parent_app_name

  await agent_tool.run_async(
      args={'request': 'hello'},
      tool_context=tool_context,
  )

  assert captured['runner_app_name'] == parent_app_name
  assert captured['session_app_name'] == parent_app_name


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


def test_use_plugins():
  """The agent tool can use plugins from parent runner."""

  class ModelResponseCapturePlugin(BasePlugin):

    def __init__(self):
      super().__init__('plugin')
      self.model_responses = {}

    async def after_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> Optional[LlmResponse]:
      response_text = []
      for part in llm_response.content.parts:
        if not part.text:
          continue
        response_text.append(part.text)
      if response_text:
        if callback_context.agent_name not in self.model_responses:
          self.model_responses[callback_context.agent_name] = []
        self.model_responses[callback_context.agent_name].append(
            ''.join(response_text)
        )

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

  model_response_capture = ModelResponseCapturePlugin()
  runner = testing_utils.InMemoryRunner(
      root_agent, plugins=[model_response_capture]
  )

  assert testing_utils.simplify_events(runner.run('test1')) == [
      ('root_agent', function_call_no_schema),
      ('root_agent', function_response_no_schema),
      ('root_agent', 'response2'),
  ]

  # should be able to capture response from both root and tool agent.
  assert model_response_capture.model_responses == {
      'tool_agent': ['response1'],
      'root_agent': ['response2'],
  }


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


@mark.asyncio
async def test_update_artifacts():
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

  async def load_artifact(filename: str):
    return await runner.runner.artifact_service.load_artifact(
        app_name='test_app',
        user_id='test_user',
        session_id=runner.session_id,
        filename=filename,
    )

  assert await runner.runner.artifact_service.list_artifact_keys(
      app_name='test_app', user_id='test_user', session_id=runner.session_id
  ) == ['artifact_1', 'artifact_2', 'artifact_3']

  assert await load_artifact('artifact_1') == Part.from_text(text='test')
  assert await load_artifact('artifact_2') == Part.from_text(text='test 2')
  assert await load_artifact('artifact_3') == Part.from_text(text='test 2 3')


@mark.parametrize(
    'env_variables',
    [
        'GOOGLE_AI',
        # TODO(wanyif): re-enable after fix.
        # 'VERTEX',
    ],
    indirect=True,
)
def test_custom_schema(env_variables):
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
def test_agent_tool_response_schema_no_output_schema_vertex_ai(
    env_variables,
):
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
def test_agent_tool_response_schema_with_output_schema_vertex_ai(
    env_variables,
):
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
def test_agent_tool_response_schema_gemini_api(
    env_variables,
):
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
def test_agent_tool_response_schema_with_input_schema_vertex_ai(
    env_variables,
):
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
def test_agent_tool_response_schema_with_input_schema_no_output_vertex_ai(
    env_variables,
):
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
