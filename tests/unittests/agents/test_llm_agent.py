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

"""Unit tests for the new workflow LlmAgent."""

from __future__ import annotations

from typing import Any
from typing import AsyncGenerator
from typing import Optional

from google.adk.agents.context import Context
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm._base_llm_agent import BaseLlmAgent
from google.adk.agents.llm._single_llm_agent import _SingleLlmAgent
from google.adk.agents.llm_agent import Agent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.events.event import Event
from google.adk.models.registry import LLMRegistry
from google.adk.planners.built_in_planner import BuiltInPlanner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow import BaseNode
from google.genai import types
from pydantic import BaseModel
import pytest
from typing_extensions import override

from .. import testing_utils

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class StubNode(BaseNode):
  """Minimal BaseNode stub for testing LlmAgent node construction."""

  name: str = ''
  description: str = ''
  events_to_yield: list[Event] = []

  @override
  def get_name(self) -> str:
    return self.name

  @override
  async def run(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    for event in self.events_to_yield:
      yield event


def _make_text_event(author: str, text: str) -> Event:
  """Create a simple text Event."""
  return Event(
      author=author,
      content=types.Content(
          role='model',
          parts=[types.Part(text=text)],
      ),
  )


def _make_mock_model(responses: list[str]) -> testing_utils.MockModel:
  """Create a MockModel with text responses."""
  return testing_utils.MockModel.create(responses)


# ===================================================================
# Tests: _SingleLlmAgent.from_base_llm_agent
# ===================================================================


class TestFromBaseLlmAgent:

  def test_creates_with_same_name(self):
    agent = BaseLlmAgent(name='test_agent', description='desc')
    result = _SingleLlmAgent.from_base_llm_agent(agent)
    assert result.name == 'test_agent'

  def test_copies_description(self):
    agent = BaseLlmAgent(name='test', description='my description')
    result = _SingleLlmAgent.from_base_llm_agent(agent)
    assert result.description == 'my description'

  def test_copies_model(self):
    agent = BaseLlmAgent(name='test', model='gemini-2.0-flash')
    result = _SingleLlmAgent.from_base_llm_agent(agent)
    assert result.model == 'gemini-2.0-flash'

  def test_copies_model_instance(self):
    mock_model = _make_mock_model(['response'])
    agent = BaseLlmAgent(name='test', model=mock_model)
    result = _SingleLlmAgent.from_base_llm_agent(agent)
    assert result.model is mock_model

  def test_copies_instruction(self):
    agent = BaseLlmAgent(name='test', instruction='Be helpful')
    result = _SingleLlmAgent.from_base_llm_agent(agent)
    assert result.instruction == 'Be helpful'

  def test_copies_instruction_provider(self):
    def my_instruction(ctx):
      return 'dynamic'

    agent = BaseLlmAgent(name='test', instruction=my_instruction)
    result = _SingleLlmAgent.from_base_llm_agent(agent)
    assert result.instruction is my_instruction

  def test_copies_tools(self):
    def my_tool():
      pass

    agent = BaseLlmAgent(name='test', tools=[my_tool])
    result = _SingleLlmAgent.from_base_llm_agent(agent)
    assert len(result.tools) == 1
    assert result.tools[0] is my_tool

  def test_tools_is_new_list(self):
    """Modifying the result's tools doesn't affect the original."""

    def tool_a():
      pass

    def tool_b():
      pass

    agent = BaseLlmAgent(name='test', tools=[tool_a])
    result = _SingleLlmAgent.from_base_llm_agent(agent)
    result.tools.append(tool_b)
    assert len(agent.tools) == 1

  def test_copies_disallow_transfer_to_parent(self):
    agent = BaseLlmAgent(name='test', disallow_transfer_to_parent=True)
    result = _SingleLlmAgent.from_base_llm_agent(agent)
    assert result.disallow_transfer_to_parent is True

  def test_copies_disallow_transfer_to_peers(self):
    agent = BaseLlmAgent(name='test', disallow_transfer_to_peers=True)
    result = _SingleLlmAgent.from_base_llm_agent(agent)
    assert result.disallow_transfer_to_peers is True

  def test_copies_output_key(self):
    agent = BaseLlmAgent(name='test', output_key='result')
    result = _SingleLlmAgent.from_base_llm_agent(agent)
    assert result.output_key == 'result'

  def test_copies_include_contents(self):
    agent = BaseLlmAgent(name='test', include_contents='none')
    result = _SingleLlmAgent.from_base_llm_agent(agent)
    assert result.include_contents == 'none'

  def test_copies_before_model_callback(self):
    def my_callback(ctx, req):
      pass

    agent = BaseLlmAgent(name='test', before_model_callback=my_callback)
    result = _SingleLlmAgent.from_base_llm_agent(agent)
    assert result.before_model_callback is my_callback

  def test_copies_after_model_callback(self):
    def my_callback(ctx, resp):
      pass

    agent = BaseLlmAgent(name='test', after_model_callback=my_callback)
    result = _SingleLlmAgent.from_base_llm_agent(agent)
    assert result.after_model_callback is my_callback

  def test_copies_before_tool_callback(self):
    def my_callback(tool, args, ctx):
      pass

    agent = BaseLlmAgent(name='test', before_tool_callback=my_callback)
    result = _SingleLlmAgent.from_base_llm_agent(agent)
    assert result.before_tool_callback is my_callback

  def test_copies_after_tool_callback(self):
    def my_callback(tool, args, ctx, result):
      pass

    agent = BaseLlmAgent(name='test', after_tool_callback=my_callback)
    result = _SingleLlmAgent.from_base_llm_agent(agent)
    assert result.after_tool_callback is my_callback

  def test_result_is_single_llm_agent(self):
    agent = BaseLlmAgent(name='test')
    result = _SingleLlmAgent.from_base_llm_agent(agent)
    assert isinstance(result, _SingleLlmAgent)

  def test_has_workflow_edges(self):
    """Created _SingleLlmAgent has call_llm/execute_tools edges."""
    agent = BaseLlmAgent(name='test')
    result = _SingleLlmAgent.from_base_llm_agent(agent)
    assert result.graph is not None

  def test_default_fields_preserved(self):
    """Fields not set on source agent have default values."""
    agent = BaseLlmAgent(name='test')
    result = _SingleLlmAgent.from_base_llm_agent(agent)
    assert result.model == ''
    assert result.instruction == ''
    assert result.tools == []
    assert result.disallow_transfer_to_parent is False
    assert result.disallow_transfer_to_peers is False
    assert result.before_model_callback is None
    assert result.after_model_callback is None

  def test_all_llm_fields_copied(self):
    """All BaseLlmAgent fields (except sub_agents/parent_agent) are copied."""
    agent = BaseLlmAgent(name='src')
    result = _SingleLlmAgent.from_base_llm_agent(agent)
    for field_name, info in BaseLlmAgent.model_fields.items():
      if info.init is not False and field_name != 'sub_agents':
        assert hasattr(result, field_name), f'{field_name} not copied'


# ===================================================================
# Tests: LlmAgent._build_nodes (via construction)
# ===================================================================


class TestBuildNodes:

  def test_coordinator_is_single_llm_agent(self):
    agent = LlmAgent(name='parent')
    assert isinstance(agent.nodes[0], _SingleLlmAgent)

  def test_coordinator_has_same_name(self):
    agent = LlmAgent(name='parent')
    assert agent.nodes[0].name == 'parent'

  def test_coordinator_is_first_node(self):
    sub = _SingleLlmAgent(name='child')
    agent = LlmAgent(name='parent', sub_agents=[sub])
    assert agent.nodes[0].name == 'parent'

  def test_coordinator_has_model_from_parent(self):
    mock_model = _make_mock_model(['resp'])
    agent = LlmAgent(name='parent', model=mock_model)
    coordinator = agent.nodes[0]
    assert coordinator.model is mock_model

  def test_coordinator_has_instruction_from_parent(self):
    agent = LlmAgent(name='parent', instruction='Be a coordinator')
    coordinator = agent.nodes[0]
    assert coordinator.instruction == 'Be a coordinator'

  def test_coordinator_has_tools_from_parent(self):
    def my_tool():
      pass

    agent = LlmAgent(name='parent', tools=[my_tool])
    coordinator = agent.nodes[0]
    assert len(coordinator.tools) == 1
    assert coordinator.tools[0] is my_tool

  def test_no_sub_agents_only_coordinator(self):
    agent = LlmAgent(name='parent')
    assert len(agent.nodes) == 1

  def test_single_llm_agent_sub_agent_reused(self):
    """Pre-existing _SingleLlmAgent is kept as-is."""
    sub = _SingleLlmAgent(name='child')
    agent = LlmAgent(name='parent', sub_agents=[sub])
    assert agent.nodes[1] is sub

  def test_new_llm_agent_sub_agent_reused(self):
    """Pre-existing (new) LlmAgent sub-agent is kept as-is."""
    sub = LlmAgent(name='child')
    agent = LlmAgent(name='parent', sub_agents=[sub])
    assert agent.nodes[1] is sub

  def test_base_llm_agent_leaf_becomes_single_llm_agent(self):
    """BaseLlmAgent with no sub_agents is converted to _SingleLlmAgent."""
    sub = BaseLlmAgent(name='leaf')
    agent = LlmAgent(name='parent', sub_agents=[sub])
    assert isinstance(agent.nodes[1], _SingleLlmAgent)
    assert agent.nodes[1].name == 'leaf'

  def test_base_llm_agent_leaf_copies_fields(self):
    """Converted BaseLlmAgent carries over its LLM fields."""
    mock_model = _make_mock_model(['resp'])
    sub = BaseLlmAgent(
        name='leaf',
        model=mock_model,
        instruction='leaf instruction',
    )
    agent = LlmAgent(name='parent', sub_agents=[sub])
    leaf_node = agent.nodes[1]
    assert leaf_node.model is mock_model
    assert leaf_node.instruction == 'leaf instruction'

  def test_base_llm_agent_with_sub_agents_kept(self):
    """BaseLlmAgent with sub_agents is kept as-is (not converted)."""
    grandchild = BaseLlmAgent(name='grandchild')
    sub = BaseLlmAgent(name='branch', sub_agents=[grandchild])
    agent = LlmAgent(name='parent', sub_agents=[sub])
    assert agent.nodes[1] is sub

  def test_multiple_sub_agents_ordering(self):
    """Nodes list maintains sub-agent order after coordinator."""
    a = _SingleLlmAgent(name='a')
    b = _SingleLlmAgent(name='b')
    c = BaseLlmAgent(name='c')
    agent = LlmAgent(name='parent', sub_agents=[a, b, c])
    names = [n.name for n in agent.nodes]
    assert names == ['parent', 'a', 'b', 'c']


# ===================================================================
# Tests: LlmAgent.model_post_init
# ===================================================================


class TestModelPostInit:

  def test_parent_agent_set_on_sub_agents(self):
    """After construction, sub_agents have parent_agent = LlmAgent."""
    sub = _SingleLlmAgent(name='child')
    agent = LlmAgent(name='parent', sub_agents=[sub])
    # BaseAgent.model_post_init sets parent_agent after the reset.
    assert sub.parent_agent is agent

  def test_shared_sub_agent_two_parents(self):
    """Same sub-agent instance can be used in two LlmAgent parents."""
    shared = _SingleLlmAgent(name='shared')
    parent_a = LlmAgent(name='a', sub_agents=[shared])
    # Without the parent_agent reset, this would raise ValueError.
    parent_b = LlmAgent(name='b', sub_agents=[shared])

    # Both parents have the shared agent in their nodes.
    assert shared in parent_a.nodes
    assert shared in parent_b.nodes
    # Last parent wins parent_agent ownership.
    assert shared.parent_agent is parent_b

  def test_shared_sub_agent_multiple_parents(self):
    """Shared instance works across three parents."""
    shared = _SingleLlmAgent(name='shared')
    p1 = LlmAgent(name='p1', sub_agents=[shared])
    p2 = LlmAgent(name='p2', sub_agents=[shared])
    p3 = LlmAgent(name='p3', sub_agents=[shared])
    assert shared.parent_agent is p3

  def test_nodes_populated(self):
    """nodes list is populated after construction."""
    agent = LlmAgent(name='test')
    assert len(agent.nodes) >= 1
    assert agent.nodes[0].name == 'test'


# ===================================================================
# Tests: LlmAgent._run_async_impl
# ===================================================================


class TestRunAsyncImpl:

  @pytest.mark.asyncio
  async def test_creates_root_context_and_runs(self):
    """_run_async_impl creates a Context and delegates to Mesh.run."""
    mock_model = _make_mock_model(['Hello world'])
    agent = LlmAgent(name='test_agent', model=mock_model)
    ic = await testing_utils.create_invocation_context(agent, user_content='hi')

    events = []
    async for event in agent._run_async_impl(ic):
      events.append(event)

    # Should have produced events from the LLM response.
    assert len(events) > 0

  @pytest.mark.asyncio
  async def test_model_response_text(self):
    """LLM response text is yielded as event content."""
    mock_model = _make_mock_model(['Hello from mock'])
    agent = LlmAgent(name='test_agent', model=mock_model)
    ic = await testing_utils.create_invocation_context(agent, user_content='hi')

    events = []
    async for event in agent._run_async_impl(ic):
      events.append(event)

    # Find the model response event.
    text_events = [
        e
        for e in events
        if hasattr(e, 'content') and e.content and e.content.parts
    ]
    texts = [p.text for e in text_events for p in e.content.parts if p.text]
    assert 'Hello from mock' in texts

  @pytest.mark.asyncio
  async def test_events_have_correct_author(self):
    """Events from the coordinator have the agent's name as author."""
    mock_model = _make_mock_model(['response'])
    agent = LlmAgent(name='my_agent', model=mock_model)
    ic = await testing_utils.create_invocation_context(agent, user_content='hi')

    events = []
    async for event in agent._run_async_impl(ic):
      events.append(event)

    content_events = [
        e
        for e in events
        if hasattr(e, 'content') and e.content and e.content.parts
    ]
    # Content events should be authored by the agent.
    for e in content_events:
      assert e.author == 'my_agent'


# ===================================================================
# Tests: end-to-end with InMemoryRunner
# ===================================================================


class TestEndToEnd:

  @pytest.mark.asyncio
  async def test_simple_agent_with_runner(self):
    """Run a simple LlmAgent through InMemoryRunner."""
    mock_model = _make_mock_model(['Hello!'])
    agent = LlmAgent(name='test_agent', model=mock_model)
    runner = testing_utils.TestInMemoryRunner(agent=agent)

    events = await runner.run_async_with_new_session('hi')

    # Should have at least one event with the model response.
    assert len(events) > 0
    text_events = [e for e in events if e.content and e.content.parts]
    texts = [p.text for e in text_events for p in e.content.parts if p.text]
    assert 'Hello!' in texts

  @pytest.mark.asyncio
  async def test_agent_with_sub_agents(self):
    """LlmAgent with sub-agents constructs correctly for runner."""
    mock_model = _make_mock_model(['parent response'])
    sub_model = _make_mock_model(['sub response'])
    sub = _SingleLlmAgent(name='helper', model=sub_model)
    agent = LlmAgent(
        name='coordinator',
        model=mock_model,
        sub_agents=[sub],
    )
    runner = testing_utils.TestInMemoryRunner(agent=agent)

    events = await runner.run_async_with_new_session('help me')

    assert len(events) > 0


# ===================================================================
# Helpers for canonical_* tests
# ===================================================================


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


# ===================================================================
# Tests: canonical_model
# ===================================================================


class TestCanonicalModel:

  def test_returns_base_llm_for_string_model(self):
    agent = LlmAgent(name='test', model='gemini-pro')
    assert agent.canonical_model.model == 'gemini-pro'

  def test_returns_base_llm_instance(self):
    llm = LLMRegistry.new_llm('gemini-pro')
    agent = LlmAgent(name='test', model=llm)
    assert agent.canonical_model == llm

  def test_inherits_from_parent(self):
    sub = LlmAgent(name='sub')
    parent = LlmAgent(name='parent', model='gemini-pro', sub_agents=[sub])
    assert sub.canonical_model == parent.canonical_model

  def test_default_model_fallback(self):
    agent = LlmAgent(name='test')
    assert agent.canonical_model.model == LlmAgent.DEFAULT_MODEL


# ===================================================================
# Tests: canonical_instruction
# ===================================================================


class TestCanonicalInstruction:

  @pytest.mark.asyncio
  async def test_string_instruction(self):
    agent = LlmAgent(name='test', instruction='Be helpful')
    ctx = await _create_readonly_context(agent)
    instruction, bypass = await agent.canonical_instruction(ctx)
    assert instruction == 'Be helpful'
    assert not bypass

  @pytest.mark.asyncio
  async def test_provider_instruction(self):
    def provider(ctx: ReadonlyContext) -> str:
      return f'state: {ctx.state["key"]}'

    agent = LlmAgent(name='test', instruction=provider)
    ctx = await _create_readonly_context(agent, state={'key': 'value'})
    instruction, bypass = await agent.canonical_instruction(ctx)
    assert instruction == 'state: value'
    assert bypass

  @pytest.mark.asyncio
  async def test_async_provider_instruction(self):
    async def provider(ctx: ReadonlyContext) -> str:
      return 'async instruction'

    agent = LlmAgent(name='test', instruction=provider)
    ctx = await _create_readonly_context(agent)
    instruction, bypass = await agent.canonical_instruction(ctx)
    assert instruction == 'async instruction'
    assert bypass


# ===================================================================
# Tests: canonical_global_instruction
# ===================================================================


class TestCanonicalGlobalInstruction:

  @pytest.mark.asyncio
  async def test_string_global_instruction(self):
    agent = LlmAgent(name='test', global_instruction='global')
    ctx = await _create_readonly_context(agent)
    instruction, bypass = await agent.canonical_global_instruction(ctx)
    assert instruction == 'global'
    assert not bypass


# ===================================================================
# Tests: canonical_tools
# ===================================================================


class TestCanonicalTools:

  @pytest.mark.asyncio
  async def test_function_tools(self):
    def my_tool():
      pass

    agent = LlmAgent(name='test', model='gemini-pro', tools=[my_tool])
    ctx = await _create_readonly_context(agent)
    tools = await agent.canonical_tools(ctx)
    assert len(tools) == 1
    assert tools[0].name == 'my_tool'

  @pytest.mark.asyncio
  async def test_multiple_tools(self):
    def tool_a():
      pass

    def tool_b():
      pass

    agent = LlmAgent(name='test', model='gemini-pro', tools=[tool_a, tool_b])
    ctx = await _create_readonly_context(agent)
    tools = await agent.canonical_tools(ctx)
    assert len(tools) == 2


# ===================================================================
# Tests: canonical_*_callbacks
# ===================================================================


class TestCanonicalCallbacks:

  def test_before_model_callback_single(self):
    def cb(ctx, req):
      pass

    agent = LlmAgent(name='test', before_model_callback=cb)
    assert agent.canonical_before_model_callbacks == [cb]

  def test_before_model_callback_list(self):
    def cb1(ctx, req):
      pass

    def cb2(ctx, req):
      pass

    agent = LlmAgent(name='test', before_model_callback=[cb1, cb2])
    assert agent.canonical_before_model_callbacks == [cb1, cb2]

  def test_before_model_callback_none(self):
    agent = LlmAgent(name='test')
    assert agent.canonical_before_model_callbacks == []

  def test_after_model_callback(self):
    def cb(ctx, resp):
      pass

    agent = LlmAgent(name='test', after_model_callback=cb)
    assert agent.canonical_after_model_callbacks == [cb]

  def test_before_tool_callback(self):
    def cb(tool, args, ctx):
      pass

    agent = LlmAgent(name='test', before_tool_callback=cb)
    assert agent.canonical_before_tool_callbacks == [cb]

  def test_after_tool_callback(self):
    def cb(tool, args, ctx, result):
      pass

    agent = LlmAgent(name='test', after_tool_callback=cb)
    assert agent.canonical_after_tool_callbacks == [cb]

  def test_on_model_error_callback(self):
    def cb(ctx, req, err):
      pass

    agent = LlmAgent(name='test', on_model_error_callback=cb)
    assert agent.canonical_on_model_error_callbacks == [cb]

  def test_on_tool_error_callback(self):
    def cb(tool, args, ctx, err):
      pass

    agent = LlmAgent(name='test', on_tool_error_callback=cb)
    assert agent.canonical_on_tool_error_callbacks == [cb]


# ===================================================================
# Tests: set_default_model
# ===================================================================


class TestSetDefaultModel:

  def test_set_string_model(self):
    original = LlmAgent._default_model
    try:
      LlmAgent.set_default_model('gemini-2.0-flash')
      agent = LlmAgent(name='test')
      assert agent.canonical_model.model == 'gemini-2.0-flash'
    finally:
      LlmAgent._default_model = original

  def test_set_llm_instance(self):
    original = LlmAgent._default_model
    llm = LLMRegistry.new_llm('gemini-pro')
    try:
      LlmAgent.set_default_model(llm)
      agent = LlmAgent(name='test')
      assert agent.canonical_model == llm
    finally:
      LlmAgent._default_model = original

  def test_rejects_empty_string(self):
    with pytest.raises(ValueError):
      LlmAgent.set_default_model('')

  def test_rejects_invalid_type(self):
    with pytest.raises(TypeError):
      LlmAgent.set_default_model(123)


# ===================================================================
# Tests: validate_generate_content_config
# ===================================================================


class TestValidateGenerateContentConfig:

  def test_rejects_tools_in_config(self):
    with pytest.raises(ValueError):
      LlmAgent(
          name='test',
          generate_content_config=types.GenerateContentConfig(
              tools=[types.Tool(function_declarations=[])]
          ),
      )

  def test_rejects_system_instruction_in_config(self):
    with pytest.raises(ValueError):
      LlmAgent(
          name='test',
          generate_content_config=types.GenerateContentConfig(
              system_instruction='instruction'
          ),
      )

  def test_rejects_response_schema_in_config(self):
    class Schema(BaseModel):
      pass

    with pytest.raises(ValueError):
      LlmAgent(
          name='test',
          generate_content_config=types.GenerateContentConfig(
              response_schema=Schema
          ),
      )

  def test_allows_thinking_config(self):
    agent = LlmAgent(
        name='test',
        generate_content_config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(include_thoughts=True)
        ),
    )
    assert (
        agent.generate_content_config.thinking_config.include_thoughts is True
    )

  def test_none_config_accepted(self):
    agent = LlmAgent(name='test')
    # When not set, generate_content_config stays None.
    assert agent.generate_content_config is None


# ===================================================================
# Tests: thinking config warning
# ===================================================================


class TestThinkingConfigWarning:

  def test_warns_when_both_thinking_configs(self):
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(include_thoughts=True)
    )
    planner = BuiltInPlanner(
        thinking_config=types.ThinkingConfig(include_thoughts=True)
    )
    with pytest.warns(
        UserWarning, match="planner's configuration will take precedence"
    ):
      LlmAgent(name='test', generate_content_config=config, planner=planner)


# ===================================================================
# Tests: output_schema wider type
# ===================================================================


class TestOutputSchema:

  def test_accepts_base_model(self):
    class MySchema(BaseModel):
      name: str

    agent = LlmAgent(name='test', output_schema=MySchema)
    assert agent.output_schema == MySchema

  def test_accepts_none(self):
    agent = LlmAgent(name='test')
    assert agent.output_schema is None


# ===================================================================
# Tests: Agent alias and re-exports
# ===================================================================


class TestAliasAndReExports:

  def test_agent_is_llm_agent(self):
    assert Agent is LlmAgent

  def test_callback_type_re_exports(self):
    from google.adk.agents.llm_agent import AfterModelCallback
    from google.adk.agents.llm_agent import AfterToolCallback
    from google.adk.agents.llm_agent import BeforeModelCallback
    from google.adk.agents.llm_agent import BeforeToolCallback
    from google.adk.agents.llm_agent import InstructionProvider
    from google.adk.agents.llm_agent import OnModelErrorCallback
    from google.adk.agents.llm_agent import OnToolErrorCallback
    from google.adk.agents.llm_agent import ToolUnion

    # Just verify they are importable and not None.
    assert BeforeModelCallback is not None
    assert AfterModelCallback is not None
    assert BeforeToolCallback is not None
    assert AfterToolCallback is not None
    assert InstructionProvider is not None
    assert OnModelErrorCallback is not None
    assert OnToolErrorCallback is not None
    assert ToolUnion is not None

  def test_convert_tool_union_re_export(self):
    from google.adk.agents.llm_agent import _convert_tool_union_to_tools

    assert callable(_convert_tool_union_to_tools)


# ===================================================================
# Tests: output_key (end-to-end via runner)
# ===================================================================


def _final_events(events: list[Event]) -> list[Event]:
  """Return only final-response events authored by the agent."""
  return [
      e
      for e in events
      if (e.content or e.output is not None)
      and e.is_final_response()
      and e.author
      and e.author != 'user'
  ]


class TestOutputKey:

  @pytest.mark.asyncio
  async def test_saves_to_state(self):
    """output_key stores text in state_delta."""
    mock_model = _make_mock_model(['Test response'])
    agent = LlmAgent(name='test_agent', model=mock_model, output_key='result')
    runner = testing_utils.TestInMemoryRunner(agent=agent)

    events = await runner.run_async_with_new_session('hi')
    final = _final_events(events)

    assert len(final) == 1
    assert final[0].actions.state_delta['result'] == 'Test response'

  @pytest.mark.asyncio
  async def test_no_output_key(self):
    """Without output_key, state_delta is empty."""
    mock_model = _make_mock_model(['Test response'])
    agent = LlmAgent(name='test_agent', model=mock_model)
    runner = testing_utils.TestInMemoryRunner(agent=agent)

    events = await runner.run_async_with_new_session('hi')
    final = _final_events(events)

    assert len(final) == 1
    assert len(final[0].actions.state_delta) == 0

  @pytest.mark.asyncio
  async def test_with_output_schema(self):
    """output_key + output_schema validates and stores parsed dict."""

    class MockSchema(BaseModel):
      message: str
      confidence: float

    json_content = '{"message": "Hello", "confidence": 0.95}'
    mock_model = _make_mock_model([json_content])
    agent = LlmAgent(
        name='test_agent',
        model=mock_model,
        output_key='result',
        output_schema=MockSchema,
    )
    runner = testing_utils.TestInMemoryRunner(agent=agent)

    events = await runner.run_async_with_new_session('hi')
    final = _final_events(events)

    assert len(final) == 1
    assert final[0].actions.state_delta['result'] == {
        'message': 'Hello',
        'confidence': 0.95,
    }


# ===================================================================
# Tests: event.output for single_turn/task + output_schema (e2e)
# ===================================================================


class TestEventData:

  @pytest.mark.asyncio
  async def test_sets_event_data_single_turn_with_output_schema(self):
    """single_turn agent with output_schema sets event.output."""

    class MySchema(BaseModel):
      answer: str
      score: float

    json_content = '{"answer": "hello", "score": 0.9}'
    mock_model = _make_mock_model([json_content])
    agent = LlmAgent(
        name='test_agent',
        model=mock_model,
        mode='single_turn',
        output_schema=MySchema,
    )
    runner = testing_utils.TestInMemoryRunner(agent=agent)

    events = await runner.run_async_with_new_session('hi')
    final = _final_events(events)

    assert len(final) == 1
    assert final[0].output == {'answer': 'hello', 'score': 0.9}
    assert len(final[0].actions.state_delta) == 0

  @pytest.mark.asyncio
  async def test_event_data_and_state_when_output_key_set(self):
    """single_turn with output_schema + output_key: sets both."""

    class MySchema(BaseModel):
      answer: str

    json_content = '{"answer": "hello"}'
    mock_model = _make_mock_model([json_content])
    agent = LlmAgent(
        name='test_agent',
        model=mock_model,
        mode='single_turn',
        output_schema=MySchema,
        output_key='result',
    )
    runner = testing_utils.TestInMemoryRunner(agent=agent)

    events = await runner.run_async_with_new_session('hi')
    final = _final_events(events)

    assert len(final) == 1
    assert final[0].output == {'answer': 'hello'}
    assert final[0].actions.state_delta['result'] == {'answer': 'hello'}

  @pytest.mark.skip(
      reason='event.output not set for task mode with output_schema yet'
  )
  @pytest.mark.asyncio
  async def test_sets_event_data_task_mode_with_output_schema(self):
    """task mode agent with output_schema sets event.output."""

    class MySchema(BaseModel):
      answer: str
      score: float

    json_content = '{"answer": "hello", "score": 0.9}'
    mock_model = _make_mock_model([json_content])
    agent = LlmAgent(
        name='test_agent',
        model=mock_model,
        mode='task',
        output_schema=MySchema,
    )
    runner = testing_utils.TestInMemoryRunner(agent=agent)

    events = await runner.run_async_with_new_session('hi')
    final = _final_events(events)

    assert len(final) == 1
    assert final[0].output == {'answer': 'hello', 'score': 0.9}
    assert len(final[0].actions.state_delta) == 0

  @pytest.mark.skip(
      reason='event.output not set for task mode with output_schema yet'
  )
  @pytest.mark.asyncio
  async def test_event_data_and_state_task_mode_with_output_key(self):
    """task mode with output_schema + output_key: sets both."""

    class MySchema(BaseModel):
      answer: str

    json_content = '{"answer": "hello"}'
    mock_model = _make_mock_model([json_content])
    agent = LlmAgent(
        name='test_agent',
        model=mock_model,
        mode='task',
        output_schema=MySchema,
        output_key='result',
    )
    runner = testing_utils.TestInMemoryRunner(agent=agent)

    events = await runner.run_async_with_new_session('hi')
    final = _final_events(events)

    assert len(final) == 1
    assert final[0].output == {'answer': 'hello'}
    assert final[0].actions.state_delta['result'] == {'answer': 'hello'}

  @pytest.mark.asyncio
  async def test_no_event_data_for_chat_mode(self):
    """Chat mode agent with output_schema does not set event.output."""

    class MySchema(BaseModel):
      answer: str

    json_content = '{"answer": "hello"}'
    mock_model = _make_mock_model([json_content])
    agent = LlmAgent(
        name='test_agent',
        model=mock_model,
        mode='chat',
        output_schema=MySchema,
    )
    runner = testing_utils.TestInMemoryRunner(agent=agent)

    events = await runner.run_async_with_new_session('hi')
    final = _final_events(events)

    assert len(final) == 1
    assert final[0].output is None

  @pytest.mark.asyncio
  async def test_event_data_as_plain_text_without_output_schema(self):
    """single_turn without output_schema sets event.output as plain text."""
    mock_model = _make_mock_model(['plain text'])
    agent = LlmAgent(
        name='test_agent',
        model=mock_model,
        mode='single_turn',
    )
    runner = testing_utils.TestInMemoryRunner(agent=agent)

    events = await runner.run_async_with_new_session('hi')
    final = _final_events(events)

    assert len(final) == 1
    assert final[0].output == 'plain text'


# ===================================================================
# Helpers for output_key unit tests
# ===================================================================


def _create_test_event(
    author: str = 'test_agent',
    content_text: str = 'Hello world',
    is_final: bool = True,
    invocation_id: str = 'test_invocation',
    node_path: str = '',
) -> Event:
  """Helper to create test events."""
  from google.adk.events.event_actions import EventActions

  parts = [types.Part.from_text(text=content_text)] if content_text else []
  content = types.Content(role='model', parts=parts) if parts else None

  event = Event(
      invocation_id=invocation_id,
      author=author,
      node_path=node_path,
      content=content,
      actions=EventActions(),
  )

  if not is_final:
    event.partial = True

  return event


# ===================================================================
# Tests: output_key node_path scoping (_maybe_save_output_to_state)
# ===================================================================


class TestOutputKeyNodePath:

  def test_captures_coordinator_child_node_path(self):
    """Events from coordinator's internal nodes (call_llm) are captured."""
    agent = LlmAgent(name='helper', output_key='result')
    # Event from the internal coordinator's call_llm node.
    event = _create_test_event(
        author='helper',
        content_text='Response from coordinator',
        node_path='helper/call_llm',
    )

    agent._maybe_save_output_to_state(event, 'helper')

    assert event.actions.state_delta['result'] == 'Response from coordinator'

  def test_skips_same_name_sub_agent(self):
    """Events from a sub-agent with the same name are NOT captured."""
    agent = LlmAgent(name='helper', output_key='result')
    # Event from a sub-agent that shares the parent's name.
    event = _create_test_event(
        author='helper',
        content_text='Response from sub-agent',
        node_path='helper/helper',
    )

    agent._maybe_save_output_to_state(event, 'helper')

    assert len(event.actions.state_delta) == 0

  def test_skips_same_name_different_node_path(self):
    """Events from a different agent with the same name are skipped."""
    agent = LlmAgent(name='helper', output_key='result')
    # Event from a different agent with same name but under a different
    # parent — not a descendant of 'helper'.
    event = _create_test_event(
        author='helper',
        content_text='Response from other helper',
        node_path='other_parent/helper',
    )

    agent._maybe_save_output_to_state(event, 'helper')

    assert len(event.actions.state_delta) == 0
