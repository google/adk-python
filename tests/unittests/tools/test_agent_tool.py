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

from typing import Any
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.features import FeatureName
from google.adk.features._feature_registry import temporary_feature_override
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
import pytest
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


def test_include_plugins_default_true():
  """Test that plugins are propagated by default (include_plugins=True)."""

  # Create a test plugin that tracks callbacks
  class TrackingPlugin(BasePlugin):

    def __init__(self, name: str):
      super().__init__(name)
      self.before_agent_calls = 0

    async def before_agent_callback(self, **kwargs):
      self.before_agent_calls += 1

  tracking_plugin = TrackingPlugin(name='tracking')

  mock_model = testing_utils.MockModel.create(
      responses=[function_call_no_schema, 'response1', 'response2']
  )

  tool_agent = Agent(
      name='tool_agent',
      model=mock_model,
  )

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[AgentTool(agent=tool_agent)],  # Default include_plugins=True
  )

  runner = testing_utils.InMemoryRunner(root_agent, plugins=[tracking_plugin])
  runner.run('test1')

  # Plugin should be called for both root_agent and tool_agent
  assert tracking_plugin.before_agent_calls == 2


def test_include_plugins_explicit_true():
  """Test that plugins are propagated when include_plugins=True."""

  class TrackingPlugin(BasePlugin):

    def __init__(self, name: str):
      super().__init__(name)
      self.before_agent_calls = 0

    async def before_agent_callback(self, **kwargs):
      self.before_agent_calls += 1

  tracking_plugin = TrackingPlugin(name='tracking')

  mock_model = testing_utils.MockModel.create(
      responses=[function_call_no_schema, 'response1', 'response2']
  )

  tool_agent = Agent(
      name='tool_agent',
      model=mock_model,
  )

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[AgentTool(agent=tool_agent, include_plugins=True)],
  )

  runner = testing_utils.InMemoryRunner(root_agent, plugins=[tracking_plugin])
  runner.run('test1')

  # Plugin should be called for both root_agent and tool_agent
  assert tracking_plugin.before_agent_calls == 2


def test_include_plugins_false():
  """Test that plugins are NOT propagated when include_plugins=False."""

  class TrackingPlugin(BasePlugin):

    def __init__(self, name: str):
      super().__init__(name)
      self.before_agent_calls = 0

    async def before_agent_callback(self, **kwargs):
      self.before_agent_calls += 1

  tracking_plugin = TrackingPlugin(name='tracking')

  mock_model = testing_utils.MockModel.create(
      responses=[function_call_no_schema, 'response1', 'response2']
  )

  tool_agent = Agent(
      name='tool_agent',
      model=mock_model,
  )

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[AgentTool(agent=tool_agent, include_plugins=False)],
  )

  runner = testing_utils.InMemoryRunner(root_agent, plugins=[tracking_plugin])
  runner.run('test1')

  # Plugin should only be called for root_agent, not tool_agent
  assert tracking_plugin.before_agent_calls == 1


def test_agent_tool_description_with_input_schema():
  """Test that agent description is propagated when using input_schema."""

  class CustomInput(BaseModel):
    """This is the Pydantic model docstring."""

    custom_input: str

  agent_description = 'This is the agent description that should be used'
  tool_agent = Agent(
      name='tool_agent',
      model=testing_utils.MockModel.create(responses=['test response']),
      description=agent_description,
      input_schema=CustomInput,
  )

  agent_tool = AgentTool(agent=tool_agent)
  declaration = agent_tool._get_declaration()

  # The description should come from the agent, not the Pydantic model
  assert declaration.description == agent_description


@pytest.fixture
def enable_json_schema_feature():
  """Fixture to enable JSON_SCHEMA_FOR_FUNC_DECL feature for a test."""
  with temporary_feature_override(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, True):
    yield


def test_agent_tool_no_schema_with_json_schema_feature(
    enable_json_schema_feature,
):
  """Test AgentTool without input_schema uses parameters_json_schema when feature enabled."""
  tool_agent = Agent(
      name='tool_agent',
      description='A tool agent for testing.',
      model=testing_utils.MockModel.create(responses=['test response']),
  )

  agent_tool = AgentTool(agent=tool_agent)
  declaration = agent_tool._get_declaration()

  assert declaration.model_dump(exclude_none=True) == {
      'name': 'tool_agent',
      'description': 'A tool agent for testing.',
      'parameters_json_schema': {
          'type': 'object',
          'properties': {
              'request': {'type': 'string'},
          },
          'required': ['request'],
      },
  }


@mark.parametrize(
    'env_variables',
    [
        'VERTEX',  # Test VERTEX_AI variant
    ],
    indirect=True,
)
def test_agent_tool_response_json_schema_no_output_schema_vertex_ai(
    env_variables,
    enable_json_schema_feature,
):
  """Test AgentTool with no output schema uses response_json_schema for VERTEX_AI when feature enabled."""
  tool_agent = Agent(
      name='tool_agent',
      description='A tool agent for testing.',
      model=testing_utils.MockModel.create(responses=['test response']),
  )

  agent_tool = AgentTool(agent=tool_agent)
  declaration = agent_tool._get_declaration()

  assert declaration.model_dump(exclude_none=True) == {
      'name': 'tool_agent',
      'description': 'A tool agent for testing.',
      'parameters_json_schema': {
          'type': 'object',
          'properties': {
              'request': {'type': 'string'},
          },
          'required': ['request'],
      },
      'response_json_schema': {'type': 'string'},
  }


@mark.parametrize(
    'env_variables',
    [
        'VERTEX',  # Test VERTEX_AI variant
    ],
    indirect=True,
)
def test_agent_tool_response_json_schema_with_output_schema_vertex_ai(
    env_variables,
    enable_json_schema_feature,
):
  """Test AgentTool with output schema uses response_json_schema for VERTEX_AI when feature enabled."""

  class CustomOutput(BaseModel):
    custom_output: str

  tool_agent = Agent(
      name='tool_agent',
      description='A tool agent for testing.',
      model=testing_utils.MockModel.create(responses=['test response']),
      output_schema=CustomOutput,
  )

  agent_tool = AgentTool(agent=tool_agent)
  declaration = agent_tool._get_declaration()

  assert declaration.model_dump(exclude_none=True) == {
      'name': 'tool_agent',
      'description': 'A tool agent for testing.',
      'parameters_json_schema': {
          'type': 'object',
          'properties': {
              'request': {'type': 'string'},
          },
          'required': ['request'],
      },
      'response_json_schema': {'type': 'object'},
  }


@mark.parametrize(
    'env_variables',
    [
        'GOOGLE_AI',  # Test GEMINI_API variant
    ],
    indirect=True,
)
def test_agent_tool_no_response_json_schema_gemini_api(
    env_variables,
    enable_json_schema_feature,
):
  """Test AgentTool with GEMINI_API variant has no response_json_schema when feature enabled."""

  class CustomOutput(BaseModel):
    custom_output: str

  tool_agent = Agent(
      name='tool_agent',
      description='A tool agent for testing.',
      model=testing_utils.MockModel.create(responses=['test response']),
      output_schema=CustomOutput,
  )

  agent_tool = AgentTool(agent=tool_agent)
  declaration = agent_tool._get_declaration()

  # GEMINI_API should not have response_json_schema
  assert declaration.model_dump(exclude_none=True) == {
      'name': 'tool_agent',
      'description': 'A tool agent for testing.',
      'parameters_json_schema': {
          'type': 'object',
          'properties': {
              'request': {'type': 'string'},
          },
          'required': ['request'],
      },
  }


@mark.parametrize(
    'env_variables',
    [
        'VERTEX',  # Test VERTEX_AI variant
    ],
    indirect=True,
)
def test_agent_tool_with_input_schema_uses_json_schema_feature(
    env_variables,
    enable_json_schema_feature,
):
  """Test AgentTool with input_schema uses parameters_json_schema when feature enabled."""

  class CustomInput(BaseModel):
    custom_input: str

  class CustomOutput(BaseModel):
    custom_output: str

  tool_agent = Agent(
      name='tool_agent',
      description='A tool agent for testing.',
      model=testing_utils.MockModel.create(responses=['test response']),
      input_schema=CustomInput,
      output_schema=CustomOutput,
  )

  agent_tool = AgentTool(agent=tool_agent)
  declaration = agent_tool._get_declaration()

  # When input_schema is provided, build_function_declaration uses Pydantic's
  # model_json_schema() which includes additional fields like 'title'
  assert declaration.model_dump(exclude_none=True) == {
      'name': 'tool_agent',
      'description': 'A tool agent for testing.',
      'parameters_json_schema': {
          'properties': {
              'custom_input': {'title': 'Custom Input', 'type': 'string'},
          },
          'required': ['custom_input'],
          'title': 'CustomInput',
          'type': 'object',
      },
      'response_json_schema': {'type': 'object'},
  }


@mark.asyncio
async def test_run_async_handles_none_parts_in_response():
  """Verify run_async handles None parts in response without raising TypeError."""

  # Mock model for the tool_agent that returns content with parts=None
  # This simulates the condition causing the TypeError
  tool_agent_model = testing_utils.MockModel.create(
      responses=[
          LlmResponse(
              content=types.Content(parts=None),
          )
      ]
  )

  tool_agent = Agent(
      name='tool_agent',
      model=tool_agent_model,
  )

  agent_tool = AgentTool(agent=tool_agent)

  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )

  invocation_context = InvocationContext(
      invocation_id='invocation_id',
      agent=tool_agent,
      session=session,
      session_service=session_service,
  )
  tool_context = ToolContext(invocation_context=invocation_context)

  # This should not raise `TypeError: 'NoneType' object is not iterable`.
  tool_result = await agent_tool.run_async(
      args={'request': 'test request'}, tool_context=tool_context
  )

  assert tool_result == ''


class TestAgentToolWithCompositeAgents:
  """Tests for AgentTool wrapping composite agents (SequentialAgent, etc.)."""

  def test_sequential_agent_with_first_sub_agent_input_schema(self):
    """Test that AgentTool exposes input_schema from first sub-agent of SequentialAgent."""

    class CustomInput(BaseModel):
      query: str
      language: str

    first_agent = Agent(
        name='first_agent',
        model=testing_utils.MockModel.create(responses=['response1']),
        input_schema=CustomInput,
    )

    second_agent = Agent(
        name='second_agent',
        model=testing_utils.MockModel.create(responses=['response2']),
    )

    sequence = SequentialAgent(
        name='sequence',
        description='Process the query through multiple steps',
        sub_agents=[first_agent, second_agent],
    )

    agent_tool = AgentTool(agent=sequence)
    declaration = agent_tool._get_declaration()

    # Should expose CustomInput schema, not fallback to 'request'
    assert declaration.name == 'sequence'
    assert declaration.description == 'Process the query through multiple steps'
    assert declaration.parameters.properties['query'].type == 'STRING'
    assert declaration.parameters.properties['language'].type == 'STRING'
    assert 'request' not in declaration.parameters.properties

  def test_sequential_agent_without_input_schema_falls_back_to_request(self):
    """Test that AgentTool falls back to 'request' when no sub-agent has input_schema."""

    first_agent = Agent(
        name='first_agent',
        model=testing_utils.MockModel.create(responses=['response1']),
    )

    second_agent = Agent(
        name='second_agent',
        model=testing_utils.MockModel.create(responses=['response2']),
    )

    sequence = SequentialAgent(
        name='sequence',
        description='Process the query through multiple steps',
        sub_agents=[first_agent, second_agent],
    )

    agent_tool = AgentTool(agent=sequence)
    declaration = agent_tool._get_declaration()

    # Should fall back to 'request' parameter
    assert declaration.name == 'sequence'
    assert declaration.parameters.properties['request'].type == 'STRING'
    assert 'query' not in declaration.parameters.properties

  @mark.parametrize(
      'env_variables',
      [
          'VERTEX',
      ],
      indirect=True,
  )
  def test_sequential_agent_with_last_sub_agent_output_schema(
      self, env_variables
  ):
    """Test that AgentTool uses output_schema from last sub-agent of SequentialAgent."""

    class CustomOutput(BaseModel):
      result: str

    first_agent = Agent(
        name='first_agent',
        model=testing_utils.MockModel.create(responses=['response1']),
    )

    second_agent = Agent(
        name='second_agent',
        model=testing_utils.MockModel.create(responses=['response2']),
        output_schema=CustomOutput,
    )

    sequence = SequentialAgent(
        name='sequence',
        description='Process the query',
        sub_agents=[first_agent, second_agent],
    )

    agent_tool = AgentTool(agent=sequence)
    declaration = agent_tool._get_declaration()

    # Should have object response schema from last sub-agent
    assert declaration.response is not None
    assert declaration.response.type == types.Type.OBJECT

  def test_nested_sequential_agent_input_schema(self):
    """Test that AgentTool recursively finds input_schema in nested composite agents."""

    class CustomInput(BaseModel):
      deep_query: str

    inner_agent = Agent(
        name='inner_agent',
        model=testing_utils.MockModel.create(responses=['response1']),
        input_schema=CustomInput,
    )

    inner_sequence = SequentialAgent(
        name='inner_sequence',
        sub_agents=[inner_agent],
    )

    outer_sequence = SequentialAgent(
        name='outer_sequence',
        description='Nested sequence',
        sub_agents=[inner_sequence],
    )

    agent_tool = AgentTool(agent=outer_sequence)
    declaration = agent_tool._get_declaration()

    # Should recursively find CustomInput from inner_agent
    assert declaration.name == 'outer_sequence'
    assert 'deep_query' in declaration.parameters.properties
    assert declaration.parameters.properties['deep_query'].type == 'STRING'
    assert 'request' not in declaration.parameters.properties

  @mark.parametrize(
      'env_variables',
      [
          'GOOGLE_AI',
          'VERTEX',
      ],
      indirect=True,
  )
  def test_sequential_agent_custom_schema_end_to_end(self, env_variables):
    """Test end-to-end flow with SequentialAgent using custom input/output schema."""

    class CustomInput(BaseModel):
      custom_input: str

    class CustomOutput(BaseModel):
      custom_output: str

    function_call_seq = Part.from_function_call(
        name='sequence', args={'custom_input': 'test_input'}
    )

    mock_model = testing_utils.MockModel.create(
        responses=[
            function_call_seq,
            '{"custom_output": "step1_response"}',
            '{"custom_output": "final_response"}',
            'root_response',
        ]
    )

    first_agent = Agent(
        name='first_agent',
        model=mock_model,
        input_schema=CustomInput,
    )

    second_agent = Agent(
        name='second_agent',
        model=mock_model,
        output_schema=CustomOutput,
        output_key='seq_output',
    )

    sequence = SequentialAgent(
        name='sequence',
        description='A sequential pipeline',
        sub_agents=[first_agent, second_agent],
    )

    root_agent = Agent(
        name='root_agent',
        model=mock_model,
        tools=[AgentTool(agent=sequence)],
    )

    runner = testing_utils.InMemoryRunner(root_agent)
    runner.run('test1')

    # Verify the tool declaration sent to LLM has the correct schema
    # The first request is from root_agent, which should have the tool declaration
    first_request = mock_model.requests[0]
    tool_declarations = first_request.config.tools
    assert len(tool_declarations) == 1

    sequence_tool = tool_declarations[0].function_declarations[0]
    assert sequence_tool.name == 'sequence'
    # Should have 'custom_input' parameter from first sub-agent's input_schema
    assert 'custom_input' in sequence_tool.parameters.properties
    # Should NOT have the fallback 'request' parameter
    assert 'request' not in sequence_tool.parameters.properties

  def test_empty_sequential_agent_falls_back_to_request(self):
    """Test that AgentTool with empty SequentialAgent falls back to 'request'."""

    sequence = SequentialAgent(
        name='empty_sequence',
        description='An empty sequence',
        sub_agents=[],
    )

    agent_tool = AgentTool(agent=sequence)
    declaration = agent_tool._get_declaration()

    # Should fall back to 'request' parameter
    assert declaration.parameters.properties['request'].type == 'STRING'
