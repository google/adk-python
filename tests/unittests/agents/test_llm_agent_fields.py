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

"""Unit tests for canonical_xxx fields in LlmAgent."""

import logging
from typing import Any
from typing import Optional
from unittest import mock

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.anthropic_llm import Claude
from google.adk.models.google_llm import Gemini
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.registry import LLMRegistry
from google.adk.planners.built_in_planner import BuiltInPlanner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.google_search_tool import google_search
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools.vertex_ai_search_tool import VertexAiSearchTool
from google.genai import types
from pydantic import BaseModel
import pytest


async def _create_readonly_context(
    agent: LlmAgent, state: Optional[dict[str, Any]] = None
) -> ReadonlyContext:
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user', state=state
  )
  invocation_context = InvocationContext(
      invocation_id='test_id',
      agent=agent,
      session=session,
      session_service=session_service,
  )
  return ReadonlyContext(invocation_context)


@pytest.mark.parametrize(
    ('default_model', 'expected_model_name', 'expected_model_type'),
    [
        (LlmAgent.DEFAULT_MODEL, LlmAgent.DEFAULT_MODEL, Gemini),
        ('gemini-2.0-flash', 'gemini-2.0-flash', Gemini),
    ],
)
def test_canonical_model_default_fallback(
    default_model, expected_model_name, expected_model_type
):
  original_default = LlmAgent._default_model
  LlmAgent.set_default_model(default_model)
  try:
    agent = LlmAgent(name='test_agent')
    assert isinstance(agent.canonical_model, expected_model_type)
    assert agent.canonical_model.model == expected_model_name
  finally:
    LlmAgent.set_default_model(original_default)


def test_canonical_model_str():
  agent = LlmAgent(name='test_agent', model='gemini-pro')

  assert agent.canonical_model.model == 'gemini-pro'


def test_canonical_model_llm():
  llm = LLMRegistry.new_llm('gemini-pro')
  agent = LlmAgent(name='test_agent', model=llm)

  assert agent.canonical_model == llm


def test_canonical_model_inherit():
  sub_agent = LlmAgent(name='sub_agent')
  parent_agent = LlmAgent(
      name='parent_agent', model='gemini-pro', sub_agents=[sub_agent]
  )

  assert sub_agent.canonical_model == parent_agent.canonical_model


async def test_canonical_instruction_str():
  agent = LlmAgent(name='test_agent', instruction='instruction')
  ctx = await _create_readonly_context(agent)

  canonical_instruction, bypass_state_injection = (
      await agent.canonical_instruction(ctx)
  )
  assert canonical_instruction == 'instruction'
  assert not bypass_state_injection


async def test_canonical_instruction():
  def _instruction_provider(ctx: ReadonlyContext) -> str:
    return f'instruction: {ctx.state["state_var"]}'

  agent = LlmAgent(name='test_agent', instruction=_instruction_provider)
  ctx = await _create_readonly_context(
      agent, state={'state_var': 'state_value'}
  )

  canonical_instruction, bypass_state_injection = (
      await agent.canonical_instruction(ctx)
  )
  assert canonical_instruction == 'instruction: state_value'
  assert bypass_state_injection


async def test_async_canonical_instruction():
  async def _instruction_provider(ctx: ReadonlyContext) -> str:
    return f'instruction: {ctx.state["state_var"]}'

  agent = LlmAgent(name='test_agent', instruction=_instruction_provider)
  ctx = await _create_readonly_context(
      agent, state={'state_var': 'state_value'}
  )

  canonical_instruction, bypass_state_injection = (
      await agent.canonical_instruction(ctx)
  )
  assert canonical_instruction == 'instruction: state_value'
  assert bypass_state_injection


async def test_canonical_global_instruction_str():
  agent = LlmAgent(name='test_agent', global_instruction='global instruction')
  ctx = await _create_readonly_context(agent)

  canonical_instruction, bypass_state_injection = (
      await agent.canonical_global_instruction(ctx)
  )
  assert canonical_instruction == 'global instruction'
  assert not bypass_state_injection


async def test_canonical_global_instruction():
  def _global_instruction_provider(ctx: ReadonlyContext) -> str:
    return f'global instruction: {ctx.state["state_var"]}'

  agent = LlmAgent(
      name='test_agent', global_instruction=_global_instruction_provider
  )
  ctx = await _create_readonly_context(
      agent, state={'state_var': 'state_value'}
  )

  canonical_global_instruction, bypass_state_injection = (
      await agent.canonical_global_instruction(ctx)
  )
  assert canonical_global_instruction == 'global instruction: state_value'
  assert bypass_state_injection


async def test_async_canonical_global_instruction():
  async def _global_instruction_provider(ctx: ReadonlyContext) -> str:
    return f'global instruction: {ctx.state["state_var"]}'

  agent = LlmAgent(
      name='test_agent', global_instruction=_global_instruction_provider
  )
  ctx = await _create_readonly_context(
      agent, state={'state_var': 'state_value'}
  )
  canonical_global_instruction, bypass_state_injection = (
      await agent.canonical_global_instruction(ctx)
  )
  assert canonical_global_instruction == 'global instruction: state_value'
  assert bypass_state_injection


def test_output_schema_with_sub_agents_will_not_throw():
  class Schema(BaseModel):
    pass

  sub_agent = LlmAgent(
      name='sub_agent',
  )

  agent = LlmAgent(
      name='test_agent',
      output_schema=Schema,
      sub_agents=[sub_agent],
  )

  # Transfer is not disabled
  assert not agent.disallow_transfer_to_parent
  assert not agent.disallow_transfer_to_peers

  assert agent.output_schema == Schema
  assert agent.sub_agents == [sub_agent]


def test_output_schema_with_tools_will_not_throw():
  class Schema(BaseModel):
    pass

  def _a_tool():
    pass

  LlmAgent(
      name='test_agent',
      output_schema=Schema,
      tools=[_a_tool],
  )


def test_before_model_callback():
  def _before_model_callback(
      callback_context: CallbackContext,
      llm_request: LlmRequest,
  ) -> None:
    return None

  agent = LlmAgent(
      name='test_agent', before_model_callback=_before_model_callback
  )

  # TODO: add more logic assertions later.
  assert agent.before_model_callback is not None


def test_validate_generate_content_config_thinking_config_allow():
  """Tests that thinking_config is now allowed directly in the agent init."""
  agent = LlmAgent(
      name='test_agent',
      generate_content_config=types.GenerateContentConfig(
          thinking_config=types.ThinkingConfig(include_thoughts=True)
      ),
  )
  assert agent.generate_content_config.thinking_config.include_thoughts is True


def test_thinking_config_precedence_warning():
  """Tests that a UserWarning is issued when both manual config and planner exist."""

  config = types.GenerateContentConfig(
      thinking_config=types.ThinkingConfig(include_thoughts=True)
  )
  planner = BuiltInPlanner(
      thinking_config=types.ThinkingConfig(include_thoughts=True)
  )

  with pytest.warns(
      UserWarning, match="planner's configuration will take precedence"
  ):
    LlmAgent(name='test_agent', generate_content_config=config, planner=planner)


def test_validate_generate_content_config_tools_throw():
  """Tests that tools cannot be set directly in config."""
  with pytest.raises(ValueError):
    _ = LlmAgent(
        name='test_agent',
        generate_content_config=types.GenerateContentConfig(
            tools=[types.Tool(function_declarations=[])]
        ),
    )


def test_validate_generate_content_config_system_instruction_throw():
  """Tests that system instructions cannot be set directly in config."""
  with pytest.raises(ValueError):
    _ = LlmAgent(
        name='test_agent',
        generate_content_config=types.GenerateContentConfig(
            system_instruction='system instruction'
        ),
    )


def test_validate_generate_content_config_response_schema_throw():
  """Tests that response schema cannot be set directly in config."""

  class Schema(BaseModel):
    pass

  with pytest.raises(ValueError):
    _ = LlmAgent(
        name='test_agent',
        generate_content_config=types.GenerateContentConfig(
            response_schema=Schema
        ),
    )


def test_allow_transfer_by_default():
  sub_agent = LlmAgent(name='sub_agent')
  agent = LlmAgent(name='test_agent', sub_agents=[sub_agent])

  assert not agent.disallow_transfer_to_parent
  assert not agent.disallow_transfer_to_peers


# TODO(b/448114567): Remove TestCanonicalTools once the workaround
# is no longer needed.
class TestCanonicalTools:
  """Unit tests for canonical_tools in LlmAgent."""

  @staticmethod
  def _my_tool(sides: int) -> int:
    return sides

  async def test_handle_google_search_with_other_tools(self):
    """Test that google_search is wrapped into an agent."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-pro',
        tools=[
            self._my_tool,
            GoogleSearchTool(bypass_multi_tools_limit=True),
        ],
    )
    ctx = await _create_readonly_context(agent)
    tools = await agent.canonical_tools(ctx)

    assert len(tools) == 2
    assert tools[0].name == '_my_tool'
    assert tools[0].__class__.__name__ == 'FunctionTool'
    assert tools[1].name == 'google_search_agent'
    assert tools[1].__class__.__name__ == 'GoogleSearchAgentTool'

  async def test_handle_google_search_with_other_tools_no_bypass(self):
    """Test that google_search is not wrapped into an agent."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-pro',
        tools=[
            self._my_tool,
            GoogleSearchTool(bypass_multi_tools_limit=False),
        ],
    )
    ctx = await _create_readonly_context(agent)
    tools = await agent.canonical_tools(ctx)

    assert len(tools) == 2
    assert tools[0].name == '_my_tool'
    assert tools[0].__class__.__name__ == 'FunctionTool'
    assert tools[1].name == 'google_search'
    assert tools[1].__class__.__name__ == 'GoogleSearchTool'

  async def test_handle_google_search_only(self):
    """Test that google_search is not wrapped into an agent."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-pro',
        tools=[
            google_search,
        ],
    )
    ctx = await _create_readonly_context(agent)
    tools = await agent.canonical_tools(ctx)

    assert len(tools) == 1
    assert tools[0].name == 'google_search'
    assert tools[0].__class__.__name__ == 'GoogleSearchTool'

  async def test_function_tool_only(self):
    """Test that function tool is not affected."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-pro',
        tools=[
            self._my_tool,
        ],
    )
    ctx = await _create_readonly_context(agent)
    tools = await agent.canonical_tools(ctx)

    assert len(tools) == 1
    assert tools[0].name == '_my_tool'
    assert tools[0].__class__.__name__ == 'FunctionTool'

  @mock.patch(
      'google.auth.default',
      mock.MagicMock(return_value=('credentials', 'project')),
  )
  async def test_handle_vais_with_other_tools(self):
    """Test that VertexAiSearchTool is replaced with Discovery Engine Search."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-pro',
        tools=[
            self._my_tool,
            VertexAiSearchTool(
                data_store_id='test_data_store_id',
                bypass_multi_tools_limit=True,
            ),
        ],
    )
    ctx = await _create_readonly_context(agent)
    tools = await agent.canonical_tools(ctx)

    assert len(tools) == 2
    assert tools[0].name == '_my_tool'
    assert tools[0].__class__.__name__ == 'FunctionTool'
    assert tools[1].name == 'discovery_engine_search'
    assert tools[1].__class__.__name__ == 'DiscoveryEngineSearchTool'

  async def test_handle_vais_with_other_tools_no_bypass(self):
    """Test that VertexAiSearchTool is not replaced."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-pro',
        tools=[
            self._my_tool,
            VertexAiSearchTool(
                data_store_id='test_data_store_id',
                bypass_multi_tools_limit=False,
            ),
        ],
    )
    ctx = await _create_readonly_context(agent)
    tools = await agent.canonical_tools(ctx)

    assert len(tools) == 2
    assert tools[0].name == '_my_tool'
    assert tools[0].__class__.__name__ == 'FunctionTool'
    assert tools[1].name == 'vertex_ai_search'
    assert tools[1].__class__.__name__ == 'VertexAiSearchTool'

  async def test_handle_vais_only(self):
    """Test that VertexAiSearchTool is not wrapped into an agent."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-pro',
        tools=[
            VertexAiSearchTool(data_store_id='test_data_store_id'),
        ],
    )
    ctx = await _create_readonly_context(agent)
    tools = await agent.canonical_tools(ctx)

    assert len(tools) == 1
    assert tools[0].name == 'vertex_ai_search'
    assert tools[0].__class__.__name__ == 'VertexAiSearchTool'


# Tests for multi-provider model support via string model names
@pytest.mark.parametrize(
    'model_name',
    [
        'gemini-1.5-flash',
        'gemini-2.0-flash-exp',
    ],
)
def test_agent_with_gemini_string_model(model_name):
  """Test that Agent accepts Gemini model strings and resolves to Gemini."""
  agent = LlmAgent(name='test_agent', model=model_name)
  assert isinstance(agent.canonical_model, Gemini)
  assert agent.canonical_model.model == model_name


@pytest.mark.parametrize(
    'model_name',
    [
        'claude-3-5-sonnet-v2@20241022',
        'claude-sonnet-4@20250514',
    ],
)
def test_agent_with_claude_string_model(model_name):
  """Test that Agent accepts Claude model strings and resolves to Claude."""
  agent = LlmAgent(name='test_agent', model=model_name)
  assert isinstance(agent.canonical_model, Claude)
  assert agent.canonical_model.model == model_name


@pytest.mark.parametrize(
    'model_name',
    [
        'openai/gpt-4o',
        'groq/llama3-70b-8192',
        'anthropic/claude-3-opus-20240229',
    ],
)
def test_agent_with_litellm_string_model(model_name):
  """Test that Agent accepts LiteLLM provider strings."""
  agent = LlmAgent(name='test_agent', model=model_name)
  assert isinstance(agent.canonical_model, LiteLlm)
  assert agent.canonical_model.model == model_name


def test_builtin_planner_overwrite_logging(caplog):
  """Tests that the planner logs an DEBUG message when overwriting a config."""

  planner = BuiltInPlanner(
      thinking_config=types.ThinkingConfig(include_thoughts=True)
  )

  # Create a request that already has a thinking_config
  req = LlmRequest(
      contents=[],
      config=types.GenerateContentConfig(
          thinking_config=types.ThinkingConfig(include_thoughts=True)
      ),
  )

  with caplog.at_level(
      logging.DEBUG, logger='google_adk.google.adk.planners.built_in_planner'
  ):
    planner.apply_thinking_config(req)
  assert (
      'Overwriting `thinking_config` from `generate_content_config`'
      in caplog.text
  )
