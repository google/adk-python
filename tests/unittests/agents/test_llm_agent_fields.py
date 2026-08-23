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
import warnings

from google.adk.agents.base_agent import BaseAgent
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
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.enterprise_search_tool import EnterpriseWebSearchTool
from google.adk.tools.function_tool import FunctionTool
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
        ('gemini-2.5-flash', 'gemini-2.5-flash', Gemini),
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


def test_canonical_model_str_resolved_once():
  agent = LlmAgent(name='test_agent', model='gemini-pro')

  with mock.patch.object(
      LLMRegistry, 'new_llm', wraps=LLMRegistry.new_llm
  ) as new_llm:
    first = agent.canonical_model
    second = agent.canonical_model
    third = agent.canonical_model

  assert new_llm.call_count == 1
  assert first is second is third


def test_canonical_model_str_resolved_again_after_reassignment():
  agent = LlmAgent(name='test_agent', model='gemini-pro')
  first = agent.canonical_model

  agent.model = 'gemini-2.5-flash'
  second = agent.canonical_model

  assert second is not first
  assert second.model == 'gemini-2.5-flash'


def test_canonical_model_str_not_stale_after_model_copy():
  agent = LlmAgent(name='test_agent', model='gemini-pro')
  assert agent.canonical_model.model == 'gemini-pro'

  copied = agent.model_copy(update={'model': 'gemini-2.5-flash'})

  assert copied.canonical_model.model == 'gemini-2.5-flash'
  assert agent.canonical_model.model == 'gemini-pro'


def test_canonical_live_model_str_resolved_once():
  agent = LlmAgent(name='test_agent', model='gemini-pro')

  with mock.patch.object(
      LLMRegistry, 'new_llm', wraps=LLMRegistry.new_llm
  ) as new_llm:
    first = agent.canonical_live_model
    second = agent.canonical_live_model

  assert new_llm.call_count == 1
  assert first is second


def test_canonical_live_model_default_fallback():
  original_default = LlmAgent._default_live_model
  LlmAgent.set_default_live_model('gemini-2.0-flash')
  try:
    agent = LlmAgent(name='test_agent')
    assert agent.canonical_live_model.model == 'gemini-2.0-flash'
  finally:
    LlmAgent.set_default_live_model(original_default)


def test_canonical_live_model_str():
  agent = LlmAgent(name='test_agent', model='gemini-pro')
  assert agent.canonical_live_model.model == 'gemini-pro'


def test_canonical_live_model_llm():
  llm = LLMRegistry.new_llm('gemini-pro')
  agent = LlmAgent(name='test_agent', model=llm)
  assert agent.canonical_live_model == llm


def test_canonical_live_model_inherit():
  sub_agent = LlmAgent(name='sub_agent')
  parent_agent = LlmAgent(
      name='parent_agent', model='gemini-pro', sub_agents=[sub_agent]
  )
  assert sub_agent.canonical_live_model == parent_agent.canonical_live_model


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


def test_validate_generate_content_config_http_options_base_url_throw():
  """Tests that a transport base URL cannot be set directly in config."""
  with pytest.raises(ValueError):
    _ = LlmAgent(
        name='test_agent',
        generate_content_config=types.GenerateContentConfig(
            http_options=types.HttpOptions(base_url='http://example.invalid')
        ),
    )


def test_validate_generate_content_config_http_options_allowed():
  """Tests that request-time http options remain settable in config."""
  extra_body = {'tool_config': {'function_calling_config': {'mode': 'AUTO'}}}
  agent = LlmAgent(
      name='test_agent',
      generate_content_config=types.GenerateContentConfig(
          http_options=types.HttpOptions(timeout=1000, extra_body=extra_body)
      ),
  )

  assert agent.generate_content_config.http_options.timeout == 1000
  assert agent.generate_content_config.http_options.extra_body == extra_body


def test_allow_transfer_by_default():
  sub_agent = LlmAgent(name='sub_agent')
  agent = LlmAgent(name='test_agent', sub_agents=[sub_agent])

  assert not agent.disallow_transfer_to_parent
  assert not agent.disallow_transfer_to_peers


# Pending cleanup: remove TestCanonicalTools once the workaround
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

  async def test_handle_google_search_in_hierarchy_with_bypass(self):
    """Test that google_search with bypass is wrapped when in an agent hierarchy."""
    search_agent = LlmAgent(
        name='search_agent',
        model='gemini-pro',
        tools=[GoogleSearchTool(bypass_multi_tools_limit=True)],
    )
    _ = LlmAgent(
        name='root_agent',
        model='gemini-pro',
        sub_agents=[search_agent],
    )
    ctx = await _create_readonly_context(search_agent)
    tools = await search_agent.canonical_tools(ctx)

    assert len(tools) == 1
    assert tools[0].name == 'google_search_agent'
    assert tools[0].__class__.__name__ == 'GoogleSearchAgentTool'

  async def test_handle_google_search_in_hierarchy_no_bypass(self):
    """Test that google_search without bypass is not wrapped even in a hierarchy."""
    search_agent = LlmAgent(
        name='search_agent',
        model='gemini-pro',
        tools=[google_search],
    )
    _ = LlmAgent(
        name='root_agent',
        model='gemini-pro',
        sub_agents=[search_agent],
    )
    ctx = await _create_readonly_context(search_agent)
    tools = await search_agent.canonical_tools(ctx)

    assert len(tools) == 1
    assert tools[0].name == 'google_search'
    assert tools[0].__class__.__name__ == 'GoogleSearchTool'

  @mock.patch(
      'google.auth.default',
      mock.MagicMock(return_value=('credentials', 'project')),
  )
  async def test_handle_vais_in_hierarchy_with_bypass(self):
    """Test that VertexAiSearchTool with bypass is replaced when in an agent hierarchy."""
    search_agent = LlmAgent(
        name='search_agent',
        model='gemini-pro',
        tools=[
            VertexAiSearchTool(
                data_store_id='test_data_store_id',
                bypass_multi_tools_limit=True,
            ),
        ],
    )
    _ = LlmAgent(
        name='root_agent',
        model='gemini-pro',
        sub_agents=[search_agent],
    )
    ctx = await _create_readonly_context(search_agent)
    tools = await search_agent.canonical_tools(ctx)

    assert len(tools) == 1
    assert tools[0].name == 'discovery_engine_search'
    assert tools[0].__class__.__name__ == 'DiscoveryEngineSearchTool'

  async def test_handle_vais_in_hierarchy_no_bypass(self):
    """Test that VertexAiSearchTool without bypass is not replaced even in a hierarchy."""
    search_agent = LlmAgent(
        name='search_agent',
        model='gemini-pro',
        tools=[
            VertexAiSearchTool(
                data_store_id='test_data_store_id',
                bypass_multi_tools_limit=False,
            ),
        ],
    )
    _ = LlmAgent(
        name='root_agent',
        model='gemini-pro',
        sub_agents=[search_agent],
    )
    ctx = await _create_readonly_context(search_agent)
    tools = await search_agent.canonical_tools(ctx)

    assert len(tools) == 1
    assert tools[0].name == 'vertex_ai_search'
    assert tools[0].__class__.__name__ == 'VertexAiSearchTool'

  async def test_handle_enterprise_web_search_in_hierarchy(self):
    """Enterprise web search without bypass remains a built-in search tool in a hierarchy."""
    search_agent = LlmAgent(
        name='search_agent',
        model='gemini-pro',
        tools=[EnterpriseWebSearchTool()],
    )
    _ = LlmAgent(
        name='root_agent',
        model='gemini-pro',
        sub_agents=[search_agent],
    )
    ctx = await _create_readonly_context(search_agent)
    tools = await search_agent.canonical_tools(ctx)

    assert len(tools) == 1
    assert tools[0].name == 'enterprise_web_search'
    assert tools[0].__class__.__name__ == 'EnterpriseWebSearchTool'

  async def test_multiple_tools_resolution(self):
    """Test that multiple tools are resolved correctly."""

    def _tool_1():
      pass

    def _tool_2():
      pass

    agent = LlmAgent(
        name='test_agent',
        model='gemini-pro',
        tools=[_tool_1, _tool_2],
    )
    ctx = await _create_readonly_context(agent)
    tools = await agent.canonical_tools(ctx)

    assert len(tools) == 2
    assert tools[0].name == '_tool_1'
    assert tools[1].name == '_tool_2'

  async def test_canonical_tools_graceful_degradation_on_toolset_error(self):
    """Test that canonical_tools returns tools from working toolsets when one fails."""
    from google.adk.tools.base_tool import BaseTool
    from google.adk.tools.base_toolset import BaseToolset

    class FailingToolset(BaseToolset):

      async def get_tools(self, readonly_context=None):
        raise ConnectionError('MCP server unavailable')

    class WorkingToolset(BaseToolset):

      async def get_tools(self, readonly_context=None):
        tool = mock.MagicMock(spec=BaseTool)
        tool.name = 'working_tool'
        tool._get_declaration = mock.MagicMock(return_value=None)
        return [tool]

    def _regular_tool():
      pass

    agent = LlmAgent(
        name='test_agent',
        model='gemini-pro',
        tools=[_regular_tool, FailingToolset(), WorkingToolset()],
    )
    ctx = await _create_readonly_context(agent)
    tools = await agent.canonical_tools(ctx)

    # Should have the regular tool + working toolset tool, but not crash
    assert len(tools) == 2
    assert tools[0].name == '_regular_tool'
    assert tools[1].name == 'working_tool'

  async def test_canonical_tools_reports_the_toolset_it_dropped(self, caplog):
    """A toolset that fails to load is reported at error level, with context."""
    from google.adk.tools.base_toolset import BaseToolset

    class FailingToolset(BaseToolset):

      async def get_tools(self, readonly_context=None):
        raise ConnectionError('MCP server unavailable')

    agent = LlmAgent(
        name='test_agent',
        model='gemini-pro',
        tools=[FailingToolset(tool_name_prefix='books')],
    )
    ctx = await _create_readonly_context(agent)

    with caplog.at_level(logging.ERROR, logger='google_adk'):
      tools = await agent.canonical_tools(ctx)

    assert tools == []
    record = next(
        r for r in caplog.records if 'failed to load' in r.getMessage()
    )
    message = record.getMessage()
    assert 'test_agent' in message
    assert 'FailingToolset' in message
    assert 'books' in message
    assert 'MCP server unavailable' in message
    # The traceback is what identifies where inside the toolset it broke.
    assert record.exc_info is not None


# Tests for multi-provider model support via string model names
@pytest.mark.parametrize(
    'model_name',
    [
        'gemini-2.5-flash',
        'gemini-2.5-pro',
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


def _callback_a(**kwargs) -> None:
  return None


def _callback_b(**kwargs) -> None:
  return None


_OMITTED = object()

# (field name, name of the canonical property that resolves it)
_CANONICAL_CALLBACK_PROPERTIES = [
    ('before_model_callback', 'canonical_before_model_callbacks'),
    ('after_model_callback', 'canonical_after_model_callbacks'),
    ('on_model_error_callback', 'canonical_on_model_error_callbacks'),
    ('before_tool_callback', 'canonical_before_tool_callbacks'),
    ('after_tool_callback', 'canonical_after_tool_callbacks'),
    ('on_tool_error_callback', 'canonical_on_tool_error_callbacks'),
]


@pytest.mark.parametrize(
    'field_name, property_name', _CANONICAL_CALLBACK_PROPERTIES
)
@pytest.mark.parametrize('value', [_OMITTED, None], ids=['omitted', 'none'])
def test_canonical_callbacks_unset_resolves_to_empty_list(
    field_name, property_name, value
):
  """Callers iterate the canonical list directly, so it is never None."""
  kwargs = {} if value is _OMITTED else {field_name: value}
  agent = LlmAgent(name='test_agent', **kwargs)

  assert getattr(agent, property_name) == []


@pytest.mark.parametrize(
    'field_name, property_name', _CANONICAL_CALLBACK_PROPERTIES
)
def test_canonical_callbacks_single_callable_resolves_to_one_element_list(
    field_name, property_name
):
  """A bare callable is wrapped so callers only ever handle the list form."""
  agent = LlmAgent(name='test_agent', **{field_name: _callback_a})

  assert getattr(agent, property_name) == [_callback_a]


@pytest.mark.parametrize(
    'field_name, property_name', _CANONICAL_CALLBACK_PROPERTIES
)
def test_canonical_callbacks_list_keeps_declaration_order(
    field_name, property_name
):
  """Order matters: the chain stops at the first callback that answers."""
  agent = LlmAgent(
      name='test_agent', **{field_name: [_callback_a, _callback_b]}
  )

  assert getattr(agent, property_name) == [_callback_a, _callback_b]


def test_canonical_model_skips_non_llm_agent_ancestor():
  """A non-LLM ancestor in the tree does not stop model inheritance."""
  leaf = LlmAgent(name='leaf_agent')
  non_llm_agent = BaseAgent(name='non_llm_agent', sub_agents=[leaf])
  _ = LlmAgent(
      name='root_agent', model='gemini-2.5-flash', sub_agents=[non_llm_agent]
  )

  assert leaf.canonical_model.model == 'gemini-2.5-flash'


def test_canonical_model_uses_nearest_ancestor_with_a_model():
  leaf = LlmAgent(name='leaf_agent')
  middle = LlmAgent(
      name='middle_agent', model='gemini-2.0-flash', sub_agents=[leaf]
  )
  _ = LlmAgent(name='root_agent', model='gemini-2.5-flash', sub_agents=[middle])

  assert leaf.canonical_model.model == 'gemini-2.0-flash'


def test_canonical_live_model_falls_back_to_live_default_through_ancestors():
  """Walking up model-less ancestors in live mode ends at the live default."""
  original_model = LlmAgent._default_model
  original_live_model = LlmAgent._default_live_model
  LlmAgent.set_default_model('gemini-2.5-flash')
  LlmAgent.set_default_live_model('gemini-2.0-flash-live-001')
  try:
    leaf = LlmAgent(name='leaf_agent')
    _ = LlmAgent(name='root_agent', sub_agents=[leaf])

    assert leaf.canonical_live_model.model == 'gemini-2.0-flash-live-001'
    assert leaf.canonical_model.model == 'gemini-2.5-flash'
  finally:
    LlmAgent.set_default_model(original_model)
    LlmAgent.set_default_live_model(original_live_model)


async def test_canonical_global_instruction_str_warns_deprecated():
  agent = LlmAgent(name='test_agent', global_instruction='global instruction')
  ctx = await _create_readonly_context(agent)

  with pytest.warns(
      DeprecationWarning, match='global_instruction field is deprecated'
  ):
    instruction, bypass_state_injection = (
        await agent.canonical_global_instruction(ctx)
    )

  assert instruction == 'global instruction'
  assert not bypass_state_injection


async def test_canonical_global_instruction_unset_does_not_warn():
  """Agents that never opted into the deprecated field must stay quiet."""
  agent = LlmAgent(name='test_agent')
  ctx = await _create_readonly_context(agent)

  with warnings.catch_warnings():
    warnings.simplefilter('error', DeprecationWarning)
    instruction, bypass_state_injection = (
        await agent.canonical_global_instruction(ctx)
    )

  assert instruction == ''
  assert not bypass_state_injection


def test_validate_generate_content_config_none_becomes_empty_config():
  agent = LlmAgent(name='test_agent', generate_content_config=None)
  other_agent = LlmAgent(name='other_agent', generate_content_config=None)

  assert agent.generate_content_config == types.GenerateContentConfig()
  # Each agent must own its config, otherwise one agent's later edits would
  # silently apply to every other agent.
  assert (
      agent.generate_content_config is not other_agent.generate_content_config
  )


def _plain_tool_1():
  pass


def _plain_tool_2():
  pass


def _toolset_tool_1():
  pass


def _toolset_tool_2():
  pass


class _TwoToolToolset(BaseToolset):
  """A toolset that expands into two tools and records the context it saw."""

  def __init__(self):
    super().__init__()
    self.received_context = 'get_tools was never called'

  async def get_tools(self, readonly_context=None):
    self.received_context = readonly_context
    return [
        FunctionTool(func=_toolset_tool_1),
        FunctionTool(func=_toolset_tool_2),
    ]


async def test_canonical_tools_flattens_toolsets_in_declared_order():
  """Toolsets resolve concurrently but must land in the declared position."""
  agent = LlmAgent(
      name='test_agent',
      model='gemini-pro',
      tools=[_plain_tool_1, _TwoToolToolset(), _plain_tool_2],
  )
  ctx = await _create_readonly_context(agent)

  tools = await agent.canonical_tools(ctx)

  assert [tool.name for tool in tools] == [
      '_plain_tool_1',
      '_toolset_tool_1',
      '_toolset_tool_2',
      '_plain_tool_2',
  ]


async def test_canonical_tools_without_context_passes_none_to_toolset():
  """Callers outside an invocation (e.g. agent cards) pass no context."""
  toolset = _TwoToolToolset()
  agent = LlmAgent(
      name='test_agent', model='gemini-pro', tools=[_plain_tool_1, toolset]
  )

  tools = await agent.canonical_tools()

  assert [tool.name for tool in tools] == [
      '_plain_tool_1',
      '_toolset_tool_1',
      '_toolset_tool_2',
  ]
  assert toolset.received_context is None
