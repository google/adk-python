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

"""Tests for _agent_tree_cloner."""

from __future__ import annotations

from collections.abc import AsyncGenerator
import ssl
from typing import Any
from typing import Literal

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.loop_agent import LoopAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.optimization._agent_tree_cloner import clone_agent_with_params
from google.adk.planners.built_in_planner import BuiltInPlanner
from google.adk.tools.agent_tool import AgentTool
from google.genai import types
from pydantic import model_validator
from pydantic import PrivateAttr
from pydantic import ValidationError
import pytest


@pytest.fixture(autouse=True)
def _suppress_experimental_warning(monkeypatch: pytest.MonkeyPatch) -> None:
  """Suppresses experimental feature warnings during unit tests."""
  monkeypatch.setenv("ADK_SUPPRESS_EXPERIMENTAL_FEATURE_WARNINGS", "true")


class _StubBaseLlm(BaseLlm):
  """Minimal BaseLlm subclass carrying mutable _additional_args like LiteLlm."""

  model: str = "openai/gpt-4o"
  _additional_args: dict[str, Any] = PrivateAttr(default_factory=dict)

  def __init__(
      self, model: str = "openai/gpt-4o", **additional_args: Any
  ) -> None:
    super().__init__(model=model)
    self._additional_args = dict(additional_args)

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    del llm_request, stream
    if False:  # pragma: no cover
      yield LlmResponse()


def _noop_tool(query: str) -> str:
  """Plain function tool used to verify non-AgentTool preservation."""
  return query


def test_empty_params_clones_tree_and_isolates_mutable_state() -> None:
  """Cloning with empty params copies agents, configs, sub-agents, and tools."""
  sub = LlmAgent(
      name="Sub",
      model="gemini-2.5-flash",
      generate_content_config=types.GenerateContentConfig(temperature=0.2),
  )
  helper = LlmAgent(name="Helper", model="gemini-2.5-flash-lite")
  root = LlmAgent(
      name="Root",
      model="gemini-2.5-pro",
      instruction="Original instruction",
      generate_content_config=types.GenerateContentConfig(temperature=0.5),
      sub_agents=[sub],
      tools=[_noop_tool, AgentTool(agent=helper)],
  )

  cloned = clone_agent_with_params(root, {})

  assert isinstance(cloned, LlmAgent)
  assert cloned is not root
  assert cloned.name == "Root"
  assert cloned.model == "gemini-2.5-pro"
  assert cloned.instruction == "Original instruction"
  assert cloned.parent_agent is None
  assert cloned.generate_content_config is not root.generate_content_config
  assert cloned.generate_content_config is not None
  assert cloned.generate_content_config.temperature == pytest.approx(0.5)

  # Mutating the cloned config must not affect the original agent.
  cloned.generate_content_config.temperature = 0.9
  assert root.generate_content_config is not None
  assert root.generate_content_config.temperature == pytest.approx(0.5)

  # Sub-agent is cloned and re-parented to cloned, leaving original intact.
  assert len(cloned.sub_agents) == 1
  cloned_sub = cloned.sub_agents[0]
  assert isinstance(cloned_sub, LlmAgent)
  assert cloned_sub is not sub
  assert cloned_sub.parent_agent is cloned
  assert sub.parent_agent is root
  assert cloned_sub.generate_content_config is not sub.generate_content_config

  # Plain function tool is preserved; AgentTool and its child are cloned.
  assert cloned.tools[0] is _noop_tool
  cloned_tool = cloned.tools[1]
  assert isinstance(cloned_tool, AgentTool)
  assert cloned_tool is not root.tools[1]
  assert cloned_tool.agent is not helper
  assert cloned_tool.agent.parent_agent is None


def test_updates_direct_llm_agent_fields_without_mutating_original() -> None:
  """Direct LlmAgent fields are updated on the clone only."""
  root = LlmAgent(
      name="Root",
      model="gemini-2.5-flash",
      instruction="Initial",
      description="Initial description",
      output_key="initial_out",
  )

  cloned = clone_agent_with_params(
      root,
      {
          "model": "gemini-2.5-pro",
          "instruction": "Updated",
          "description": "Updated description",
          "output_key": "updated_out",
          "disallow_transfer_to_parent": True,
          "disallow_transfer_to_peers": True,
      },
  )

  assert isinstance(cloned, LlmAgent)
  assert cloned.model == "gemini-2.5-pro"
  assert cloned.instruction == "Updated"
  assert cloned.description == "Updated description"
  assert cloned.output_key == "updated_out"
  assert cloned.disallow_transfer_to_parent is True
  assert cloned.disallow_transfer_to_peers is True

  assert root.model == "gemini-2.5-flash"
  assert root.instruction == "Initial"
  assert root.description == "Initial description"
  assert root.output_key == "initial_out"
  assert root.disallow_transfer_to_parent is False
  assert root.disallow_transfer_to_peers is False


def test_clones_base_llm_model_when_swept_or_inherited() -> None:
  """BaseLlm instances and their _additional_args are isolated across clones."""
  baseline_llm = _StubBaseLlm("openai/gpt-4o-mini", api_base="https://a.test")
  swept_llm = _StubBaseLlm("openai/gpt-4o", api_base="https://b.test")
  root = LlmAgent(name="Root", model=baseline_llm)

  # Case 1: BaseLlm inherited from baseline agent when sweeping another field.
  clone_inherited = clone_agent_with_params(root, {"instruction": "Changed"})
  assert isinstance(clone_inherited, LlmAgent)
  assert isinstance(clone_inherited.model, _StubBaseLlm)
  assert clone_inherited.model is not baseline_llm
  assert (
      clone_inherited.model._additional_args
      is not baseline_llm._additional_args
  )
  clone_inherited.model._additional_args["api_base"] = "https://mutated.test"
  assert baseline_llm._additional_args["api_base"] == "https://a.test"

  # Case 2: BaseLlm passed as a swept parameter value across two clones.
  clone_a = clone_agent_with_params(root, {"model": swept_llm})
  clone_b = clone_agent_with_params(root, {"model": swept_llm})
  assert isinstance(clone_a, LlmAgent) and isinstance(clone_b, LlmAgent)
  assert isinstance(clone_a.model, _StubBaseLlm)
  assert isinstance(clone_b.model, _StubBaseLlm)
  assert clone_a.model is not swept_llm
  assert clone_a.model is not clone_b.model
  clone_a.model._additional_args["timeout"] = 30
  assert "timeout" not in swept_llm._additional_args
  assert "timeout" not in clone_b.model._additional_args


def test_dotted_generate_content_config_scalars_and_nested_thinking() -> None:
  """Dotted generate_content_config paths merge cleanly with baseline fields."""
  root = LlmAgent(
      name="Root",
      model="gemini-2.5-flash",
      generate_content_config=types.GenerateContentConfig(
          temperature=0.1,
          top_p=0.95,
          thinking_config=types.ThinkingConfig(
              include_thoughts=True,
              thinking_budget=128,
          ),
      ),
  )

  cloned = clone_agent_with_params(
      root,
      {
          "generate_content_config.temperature": 0.7,
          "generate_content_config.top_k": 40,
          "generate_content_config.max_output_tokens": 1024,
          "generate_content_config.seed": 42,
          "generate_content_config.thinking_config.thinking_budget": 2048,
      },
  )

  assert isinstance(cloned, LlmAgent)
  cfg = cloned.generate_content_config
  assert cfg is not None
  assert cfg.temperature == pytest.approx(0.7)
  assert cfg.top_p == pytest.approx(0.95)
  assert cfg.top_k == pytest.approx(40.0)
  assert cfg.max_output_tokens == 1024
  assert cfg.seed == 42
  assert cfg.thinking_config is not None
  assert cfg.thinking_config.include_thoughts is True
  assert cfg.thinking_config.thinking_budget == 2048

  # Baseline remains untouched.
  assert root.generate_content_config is not None
  assert root.generate_content_config.temperature == pytest.approx(0.1)
  assert root.generate_content_config.top_k is None
  assert root.generate_content_config.thinking_config is not None
  assert root.generate_content_config.thinking_config.thinking_budget == 128


def test_dotted_generate_content_config_when_baseline_config_is_none() -> None:
  """Dotted generate_content_config paths create a config when base is None."""
  root = LlmAgent(name="Root", model="gemini-2.5-flash")
  root.generate_content_config = None

  cloned = clone_agent_with_params(
      root,
      {
          "generate_content_config.temperature": 0.3,
          "generate_content_config.thinking_config.thinking_budget": 512,
      },
  )

  assert isinstance(cloned, LlmAgent)
  assert root.generate_content_config is None
  assert cloned.generate_content_config is not None
  assert cloned.generate_content_config.temperature == pytest.approx(0.3)
  assert cloned.generate_content_config.thinking_config is not None
  assert cloned.generate_content_config.thinking_config.thinking_budget == 512


def test_whole_generate_content_config_object_mapping_and_none() -> None:
  """Whole generate_content_config accepts a model, a dict, or None."""
  root = LlmAgent(
      name="Root",
      model="gemini-2.5-flash",
      generate_content_config=types.GenerateContentConfig(temperature=0.2),
  )

  # 1. Whole GenerateContentConfig instance is deep-copied.
  provided_cfg = types.GenerateContentConfig(temperature=0.8, top_p=0.9)
  clone_obj = clone_agent_with_params(
      root, {"generate_content_config": provided_cfg}
  )
  assert isinstance(clone_obj, LlmAgent)
  assert clone_obj.generate_content_config is not None
  assert clone_obj.generate_content_config is not provided_cfg
  assert clone_obj.generate_content_config.temperature == pytest.approx(0.8)
  provided_cfg.temperature = 0.0
  assert clone_obj.generate_content_config.temperature == pytest.approx(0.8)

  # 2. Whole dict is validated into a GenerateContentConfig.
  raw_dict: dict[str, Any] = {
      "temperature": 0.4,
      "thinking_config": {"thinking_budget": 256},
  }
  clone_dict = clone_agent_with_params(
      root, {"generate_content_config": raw_dict}
  )
  assert isinstance(clone_dict, LlmAgent)
  assert clone_dict.generate_content_config is not None
  assert clone_dict.generate_content_config.temperature == pytest.approx(0.4)
  assert clone_dict.generate_content_config.thinking_config is not None
  assert (
      clone_dict.generate_content_config.thinking_config.thinking_budget == 256
  )

  # 3. None clears generate_content_config on the clone only.
  clone_none = clone_agent_with_params(root, {"generate_content_config": None})
  assert isinstance(clone_none, LlmAgent)
  assert clone_none.generate_content_config is None
  assert root.generate_content_config is not None
  assert root.generate_content_config.temperature == pytest.approx(0.2)


def test_builtin_planner_stays_synced_with_thinking_config() -> None:
  """BuiltInPlanner.thinking_config tracks swept thinking_config updates."""
  root = LlmAgent(
      name="Root",
      model="gemini-2.5-flash",
      planner=BuiltInPlanner(
          thinking_config=types.ThinkingConfig(
              include_thoughts=True,
              thinking_budget=128,
          )
      ),
  )

  # Baseline and non-thinking updates copy BuiltInPlanner so it is never shared.
  untouched = clone_agent_with_params(
      root, {"generate_content_config.temperature": 0.6}
  )
  assert isinstance(untouched, LlmAgent)
  assert isinstance(untouched.planner, BuiltInPlanner)
  assert isinstance(root.planner, BuiltInPlanner)
  assert untouched.planner is not root.planner
  assert untouched.planner.thinking_config is not root.planner.thinking_config
  assert untouched.planner.thinking_config.thinking_budget == 128

  # Dotted update merges with planner's thinking_config and updates both.
  cloned = clone_agent_with_params(
      root,
      {"generate_content_config.thinking_config.thinking_budget": 1024},
  )
  assert isinstance(cloned, LlmAgent)
  assert isinstance(cloned.planner, BuiltInPlanner)
  assert cloned.planner is not root.planner
  assert cloned.planner.thinking_config.thinking_budget == 1024
  assert cloned.planner.thinking_config.include_thoughts is True
  assert cloned.generate_content_config is not None
  assert cloned.generate_content_config.thinking_config is not None
  assert cloned.generate_content_config.thinking_config.thinking_budget == 1024
  assert cloned.generate_content_config.thinking_config.include_thoughts is True
  assert isinstance(root.planner, BuiltInPlanner)
  assert root.planner.thinking_config.thinking_budget == 128

  # Clearing generate_content_config to None also clears BuiltInPlanner.
  cleared = clone_agent_with_params(root, {"generate_content_config": None})
  assert isinstance(cleared, LlmAgent)
  assert cleared.planner is None

  # Whole config without thinking_config also clears BuiltInPlanner.
  no_thinking = clone_agent_with_params(
      root,
      {
          "generate_content_config": types.GenerateContentConfig(
              temperature=0.5, thinking_config=None
          )
      },
  )
  assert isinstance(no_thinking, LlmAgent)
  assert no_thinking.planner is None

  # Explicit planner override in flat_params is respected and not overwritten.
  explicit_planner = BuiltInPlanner(
      thinking_config=types.ThinkingConfig(thinking_budget=4096)
  )
  with_explicit = clone_agent_with_params(
      root,
      {
          "planner": explicit_planner,
          "generate_content_config.thinking_config.thinking_budget": 512,
      },
  )
  assert isinstance(with_explicit, LlmAgent)
  assert isinstance(with_explicit.planner, BuiltInPlanner)
  assert with_explicit.planner.thinking_config.thinking_budget == 4096


def test_http_options_preserves_live_ssl_context_and_isolates_dicts() -> None:
  """Live non-deepcopyable objects on http_options survive cloning cleanly."""
  ssl_ctx = ssl.create_default_context()
  http_options = types.HttpOptions(
      timeout=1000,
      headers={"x-env": "prod"},
      extra_body={"custom": "val"},
      client_args={"verify": ssl_ctx},
      async_client_args={"verify": ssl_ctx},
      retry_options=types.HttpRetryOptions(attempts=3),
  )
  root = LlmAgent(
      name="Root",
      model="gemini-2.5-flash",
      generate_content_config=types.GenerateContentConfig(
          temperature=0.2,
          http_options=http_options,
      ),
  )

  # 1. Cloning while updating another config field keeps ssl_ctx reference
  # and copies the surrounding dicts.
  cloned = clone_agent_with_params(
      root, {"generate_content_config.temperature": 0.6}
  )
  assert isinstance(cloned, LlmAgent)
  assert cloned.generate_content_config is not None
  cloned_http = cloned.generate_content_config.http_options
  assert cloned_http is not None
  assert cloned_http is not http_options
  assert cloned_http.client_args is not None
  assert cloned_http.client_args["verify"] is ssl_ctx
  assert cloned_http.async_client_args is not None
  assert cloned_http.async_client_args["verify"] is ssl_ctx
  assert cloned_http.headers is not http_options.headers
  assert cloned_http.retry_options is not http_options.retry_options
  cloned_http.headers["x-env"] = "staging"
  assert http_options.headers["x-env"] == "prod"

  # 2. Merging a dotted http_options mapping updates specified fields and keeps
  # existing headers + ssl_ctx.
  merged = clone_agent_with_params(
      root, {"generate_content_config.http_options.timeout": 5000}
  )
  assert isinstance(merged, LlmAgent)
  assert merged.generate_content_config is not None
  merged_http = merged.generate_content_config.http_options
  assert merged_http is not None
  assert merged_http.timeout == 5000
  assert merged_http.headers == {"x-env": "prod"}
  assert merged_http.client_args is not None
  assert merged_http.client_args["verify"] is ssl_ctx

  # 3. Setting http_options when base had None works with a dict.
  bare = LlmAgent(name="Bare", model="gemini-2.5-flash")
  with_http = clone_agent_with_params(
      bare, {"generate_content_config.http_options": {"timeout": 2500}}
  )
  assert isinstance(with_http, LlmAgent)
  assert with_http.generate_content_config is not None
  assert with_http.generate_content_config.http_options is not None
  assert with_http.generate_content_config.http_options.timeout == 2500


def test_clones_four_level_composite_workflow_tree_and_reparents() -> None:
  """Clones Root -> Sequential -> Parallel -> Loop -> Leaf LlmAgent cleanly."""
  leaf_a = LlmAgent(
      name="DraftWriter",
      model="gemini-2.5-flash",
      instruction="Draft v1",
  )
  leaf_b = LlmAgent(
      name="FactChecker",
      model="gemini-2.5-flash",
      instruction="Check facts",
  )
  loop = LoopAgent(
      name="RefinementLoop",
      max_iterations=2,
      sub_agents=[leaf_a],
  )
  parallel = ParallelAgent(
      name="FanOut",
      sub_agents=[loop, leaf_b],
  )
  pipeline = SequentialAgent(
      name="Pipeline",
      description="Original pipeline",
      sub_agents=[parallel],
  )
  root = LlmAgent(
      name="Orchestrator",
      model="gemini-2.5-flash",
      sub_agents=[pipeline],
  )

  cloned = clone_agent_with_params(
      root,
      {
          "model": "gemini-2.5-pro",
          "sub_agents.Pipeline.description": "Optimized pipeline",
          (
              "sub_agents.Pipeline.sub_agents.FanOut.sub_agents."
              "RefinementLoop.max_iterations"
          ): 5,
          (
              "sub_agents.Pipeline.sub_agents.FanOut.sub_agents."
              "RefinementLoop.sub_agents.DraftWriter.model"
          ): "gemini-2.5-pro",
          (
              "sub_agents.Pipeline.sub_agents.FanOut.sub_agents."
              "RefinementLoop.sub_agents.DraftWriter."
              "generate_content_config.temperature"
          ): 0.3,
          (
              "sub_agents.Pipeline.sub_agents.FanOut.sub_agents."
              "FactChecker.model"
          ): "gemini-2.5-flash-lite",
      },
  )

  assert isinstance(cloned, LlmAgent)
  assert cloned.model == "gemini-2.5-pro"
  c_pipeline = cloned.sub_agents[0]
  assert isinstance(c_pipeline, SequentialAgent)
  assert c_pipeline.description == "Optimized pipeline"
  assert c_pipeline.parent_agent is cloned

  c_parallel = c_pipeline.sub_agents[0]
  assert isinstance(c_parallel, ParallelAgent)
  assert c_parallel.parent_agent is c_pipeline

  c_loop = c_parallel.sub_agents[0]
  c_checker = c_parallel.sub_agents[1]
  assert isinstance(c_loop, LoopAgent)
  assert isinstance(c_checker, LlmAgent)
  assert c_loop.max_iterations == 5
  assert c_loop.parent_agent is c_parallel
  assert c_checker.model == "gemini-2.5-flash-lite"
  assert c_checker.parent_agent is c_parallel

  c_writer = c_loop.sub_agents[0]
  assert isinstance(c_writer, LlmAgent)
  assert c_writer.model == "gemini-2.5-pro"
  assert c_writer.generate_content_config is not None
  assert c_writer.generate_content_config.temperature == pytest.approx(0.3)
  assert c_writer.parent_agent is c_loop

  # Verify original hierarchy and parent pointers were never modified.
  assert root.model == "gemini-2.5-flash"
  assert pipeline.description == "Original pipeline"
  assert pipeline.parent_agent is root
  assert parallel.parent_agent is pipeline
  assert loop.max_iterations == 2
  assert loop.parent_agent is parallel
  assert leaf_a.model == "gemini-2.5-flash"
  assert leaf_a.generate_content_config is None
  assert leaf_a.parent_agent is loop
  assert leaf_b.model == "gemini-2.5-flash"
  assert leaf_b.parent_agent is parallel


def test_sweeping_sub_agents_list_clones_supplied_children() -> None:
  """Sweeping sub_agents directly clones the supplied agents to avoid sharing."""
  original_child = LlmAgent(name="OriginalChild", model="gemini-2.5-flash")
  root = LlmAgent(
      name="Root", model="gemini-2.5-flash", sub_agents=[original_child]
  )
  replacement_child = LlmAgent(name="Replacement", model="gemini-2.5-flash")

  clone_a = clone_agent_with_params(
      root,
      {
          "sub_agents": [replacement_child],
          "sub_agents.Replacement.model": "gemini-2.5-pro",
      },
  )
  clone_b = clone_agent_with_params(
      root,
      {
          "sub_agents": [replacement_child],
          "sub_agents.Replacement.model": "gemini-2.5-flash-lite",
      },
  )

  assert replacement_child.parent_agent is None
  assert replacement_child.model == "gemini-2.5-flash"
  assert clone_a.sub_agents[0] is not replacement_child
  assert clone_b.sub_agents[0] is not replacement_child
  assert clone_a.sub_agents[0] is not clone_b.sub_agents[0]
  assert clone_a.sub_agents[0].parent_agent is clone_a
  assert clone_b.sub_agents[0].parent_agent is clone_b
  assert isinstance(clone_a.sub_agents[0], LlmAgent)
  assert isinstance(clone_b.sub_agents[0], LlmAgent)
  assert clone_a.sub_agents[0].model == "gemini-2.5-pro"
  assert clone_b.sub_agents[0].model == "gemini-2.5-flash-lite"


def test_clones_nested_agent_tools_and_sub_agents_inside_agent_tools() -> None:
  """Supports sub_agents.<A>.tools.<B> and tools.<A>.sub_agents.<B> paths."""
  inner_tool_agent = LlmAgent(name="SqlTool", model="gemini-2.5-flash")
  sub_with_tool = LlmAgent(
      name="Analyst",
      model="gemini-2.5-flash",
      tools=[_noop_tool, AgentTool(agent=inner_tool_agent)],
  )

  leaf_in_tool = LlmAgent(name="Fetcher", model="gemini-2.5-flash")
  tool_with_sub = LlmAgent(
      name="RouterTool",
      model="gemini-2.5-flash",
      sub_agents=[leaf_in_tool],
  )
  root = LlmAgent(
      name="Coordinator",
      model="gemini-2.5-flash",
      sub_agents=[sub_with_tool],
      tools=[AgentTool(agent=tool_with_sub)],
  )

  cloned = clone_agent_with_params(
      root,
      {
          "sub_agents.Analyst.tools.SqlTool.model": "gemini-2.5-pro",
          "tools.RouterTool.model": "gemini-2.5-flash-lite",
          "tools.RouterTool.sub_agents.Fetcher.model": "gemini-2.5-pro",
      },
  )

  assert isinstance(cloned, LlmAgent)
  c_analyst = cloned.sub_agents[0]
  assert isinstance(c_analyst, LlmAgent)
  assert c_analyst.tools[0] is _noop_tool
  c_sql_tool = c_analyst.tools[1]
  assert isinstance(c_sql_tool, AgentTool)
  assert isinstance(c_sql_tool.agent, LlmAgent)
  assert c_sql_tool.agent.model == "gemini-2.5-pro"
  assert c_sql_tool.agent.parent_agent is None
  assert inner_tool_agent.model == "gemini-2.5-flash"

  c_router_tool = cloned.tools[0]
  assert isinstance(c_router_tool, AgentTool)
  assert isinstance(c_router_tool.agent, LlmAgent)
  assert c_router_tool.agent.model == "gemini-2.5-flash-lite"
  assert c_router_tool.agent.parent_agent is None
  c_fetcher = c_router_tool.agent.sub_agents[0]
  assert isinstance(c_fetcher, LlmAgent)
  assert c_fetcher.model == "gemini-2.5-pro"
  assert c_fetcher.parent_agent is c_router_tool.agent
  assert leaf_in_tool.model == "gemini-2.5-flash"


def test_single_turn_and_dual_bound_sub_agents_rebind_and_reject_tools_path() -> (
    None
):
  """Sub-agents also wrapped in AgentTools use sub_agents.* and rebind cleanly."""
  specialist = LlmAgent(
      name="Specialist",
      model="gemini-2.5-flash",
      mode="single_turn",
  )
  root = LlmAgent(
      name="Coordinator",
      model="gemini-2.5-flash",
      sub_agents=[specialist],
  )

  # 1. Addressing via sub_agents.Specialist.model updates the sub-agent and
  # keeps the single-turn AgentTool wrapper synchronized across chained clones.
  first = clone_agent_with_params(
      root, {"sub_agents.Specialist.model": "gemini-2.5-pro"}
  )
  second = clone_agent_with_params(
      first, {"sub_agents.Specialist.model": "gemini-2.5-flash-lite"}
  )
  assert isinstance(second, LlmAgent)
  agent_tools = [t for t in second.tools if isinstance(t, AgentTool)]
  assert len(agent_tools) == 1
  assert agent_tools[0].agent is second.sub_agents[0]
  assert isinstance(second.sub_agents[0], LlmAgent)
  assert second.sub_agents[0].model == "gemini-2.5-flash-lite"

  # 2. Addressing a sub-agent via tools.<Name>.* fails fast with a clear hint
  # instead of silently dropping the parameter.
  with pytest.raises(
      ValueError,
      match=(
          r"has no AgentTool agent named 'Specialist'.*"
          r"use 'sub_agents\.Specialist\.model'"
      ),
  ):
    clone_agent_with_params(root, {"tools.Specialist.model": "gemini-2.5-pro"})

  # 3. A manually dual-bound sub-agent (present in both sub_agents and a user
  # AgentTool without mode="single_turn") also rebinds via sub_agents.* and
  # rejects tools.*.
  manual_sub = LlmAgent(name="Helper", model="gemini-2.5-flash")
  manual_root = LlmAgent(name="Coordinator", model="gemini-2.5-flash")
  manual_root.sub_agents = [manual_sub]
  manual_root.tools = [AgentTool(agent=manual_sub)]

  rebound = clone_agent_with_params(
      manual_root, {"sub_agents.Helper.model": "gemini-2.5-pro"}
  )
  assert isinstance(rebound, LlmAgent)
  assert isinstance(rebound.tools[0], AgentTool)
  assert rebound.tools[0].agent is rebound.sub_agents[0]
  assert isinstance(rebound.sub_agents[0], LlmAgent)
  assert rebound.sub_agents[0].model == "gemini-2.5-pro"

  with pytest.raises(
      ValueError,
      match=(
          r"has no AgentTool agent named 'Helper'.*"
          r"use 'sub_agents\.Helper\.model'"
      ),
  ):
    clone_agent_with_params(
        manual_root, {"tools.Helper.model": "gemini-2.5-pro"}
    )


def test_canonical_prefixes_disambiguate_collisions_and_preserve_caller_dicts() -> (
    None
):
  """Explicit prefixes separate colliding names and do not mutate input dicts."""
  planner_sub = LlmAgent(name="planner", model="gemini-2.5-flash")
  sub_helper = LlmAgent(name="Helper", model="gemini-2.5-flash")
  tool_helper = LlmAgent(name="Helper", model="gemini-2.5-flash")
  root = LlmAgent(
      name="Coordinator",
      model="gemini-2.5-flash",
      generate_content_config=types.GenerateContentConfig(
          temperature=0.1,
          thinking_config=types.ThinkingConfig(thinking_budget=256),
      ),
      sub_agents=[planner_sub, sub_helper],
      tools=[AgentTool(agent=tool_helper)],
  )
  shared_thinking = {"include_thoughts": True}

  clone_a = clone_agent_with_params(
      root,
      {
          "generate_content_config.thinking_config": shared_thinking,
          "generate_content_config.top_p": 0.5,
          "sub_agents.planner.model": "gemini-2.5-pro",
          "sub_agents.Helper.model": "gemini-2.5-flash-lite",
          "tools.Helper.model": "gemini-2.5-pro",
      },
  )
  clone_b = clone_agent_with_params(
      root,
      {
          "generate_content_config.thinking_config": shared_thinking,
          "generate_content_config.top_p": 0.9,
      },
  )

  # Caller's mapping was never mutated in place during merge.
  assert shared_thinking == {"include_thoughts": True}
  assert isinstance(clone_a, LlmAgent) and isinstance(clone_b, LlmAgent)
  assert isinstance(clone_a.sub_agents[0], LlmAgent)
  assert isinstance(clone_a.sub_agents[1], LlmAgent)
  assert clone_a.sub_agents[0].model == "gemini-2.5-pro"
  assert clone_a.sub_agents[1].model == "gemini-2.5-flash-lite"
  assert isinstance(clone_a.tools[0], AgentTool)
  assert isinstance(clone_a.tools[0].agent, LlmAgent)
  assert clone_a.tools[0].agent.model == "gemini-2.5-pro"
  for cloned, expected_top_p in ((clone_a, 0.5), (clone_b, 0.9)):
    assert cloned.generate_content_config is not None
    assert cloned.generate_content_config.top_p == pytest.approx(expected_top_p)
    assert cloned.generate_content_config.thinking_config == (
        types.ThinkingConfig(include_thoughts=True, thinking_budget=256)
    )


@pytest.mark.parametrize(
    ("bad_params", "expected_pattern"),
    [
        (
            {"sub_agents.Ghost.model": "gemini-2.5-pro"},
            r"has no sub-agent named 'Ghost'\. Sub-agents: \['Pipeline',"
            r" 'Worker'\]",
        ),
        (
            {"sub_agents.Worker": "gemini-2.5-pro"},
            r"names sub-agent 'Worker' but no field",
        ),
        (
            {"sub_agents.Worker.": "gemini-2.5-pro"},
            r"names sub-agent 'Worker' but no field",
        ),
        (
            {"tools.Ghost.model": "gemini-2.5-pro"},
            r"has no AgentTool agent named 'Ghost'\. AgentTools: \['Helper'\]",
        ),
        (
            {"tools.Helper": "gemini-2.5-pro"},
            r"names AgentTool 'Helper' but no field",
        ),
        (
            {"tools.Helper.": "gemini-2.5-pro"},
            r"names AgentTool 'Helper' but no field",
        ),
        (
            {"generate_content_config.": 0.5},
            r"names no config field",
        ),
        (
            {"sub_agents.Pipeline.generate_content_config.temperature": 0.5},
            r"SequentialAgent \('Pipeline'\) has no generate_content_config",
        ),
        (
            {
                "generate_content_config": types.GenerateContentConfig(
                    temperature=0.2
                ),
                "generate_content_config.temperature": 0.7,
            },
            r"Cannot set both 'generate_content_config' as a whole and",
        ),
        (
            {
                "generate_content_config.thinking_config": {
                    "thinking_budget": 128
                },
                "generate_content_config.thinking_config.thinking_budget": 512,
            },
            r"sets a field inside 'generate_content_config\.thinking_config'",
        ),
        (
            {
                "generate_content_config.thinking_config.thinking_budget": 512,
                "generate_content_config.thinking_config": {
                    "thinking_budget": 128
                },
            },
            r"sets a field inside 'generate_content_config\.thinking_config'",
        ),
        (
            {"planner.thinking_config": None},
            r"sets a field inside 'planner'; only 'generate_content_config\.",
        ),
        (
            {"Worker.model": "gemini-2.5-pro"},
            r"is not valid on 'Root'\. Prefix child agents with 'sub_agents\.'",
        ),
        (
            {"temperature": 0.5},
            r"is not a field of LlmAgent \('Root'\); use"
            r" 'generate_content_config\.temperature'",
        ),
    ],
)
def test_invalid_parameter_paths_fail_fast(
    bad_params: dict[str, Any],
    expected_pattern: str,
) -> None:
  """Invalid parameter paths raise descriptive ValueErrors before execution."""
  worker = LlmAgent(name="Worker", model="gemini-2.5-flash")
  pipeline = SequentialAgent(name="Pipeline", sub_agents=[])
  helper = LlmAgent(name="Helper", model="gemini-2.5-flash")
  root = LlmAgent(
      name="Root",
      model="gemini-2.5-flash",
      sub_agents=[worker, pipeline],
      tools=[AgentTool(agent=helper)],
  )

  with pytest.raises(ValueError, match=expected_pattern):
    clone_agent_with_params(root, bad_params)


def test_unknown_generate_content_config_subfield_raises_validation_error() -> (
    None
):
  """Unknown GenerateContentConfig subfields are rejected by Pydantic."""
  root = LlmAgent(name="Root", model="gemini-2.5-flash")

  with pytest.raises(ValidationError):
    clone_agent_with_params(
        root, {"generate_content_config.not_a_real_field": 123}
    )


class _EffortGenerateContentConfig(types.GenerateContentConfig):
  """Provider-style GenerateContentConfig subclass like Anthropic/OpenAI."""

  effort: Literal["low", "medium", "high"] | None = None

  @model_validator(mode="after")
  def _reject_thinking_level(self) -> _EffortGenerateContentConfig:
    if self.thinking_config and self.thinking_config.thinking_level is not None:
      raise ValueError("thinking_level is not supported alongside effort.")
    return self


def test_config_updates_are_deep_copied_across_clones() -> None:
  """Mutable sub-models supplied via dotted overrides never alias."""
  shared_thinking = types.ThinkingConfig(thinking_budget=512)
  root = LlmAgent(
      name="Root",
      model="gemini-2.5-flash",
      generate_content_config=types.GenerateContentConfig(temperature=0.2),
  )

  clone_a = clone_agent_with_params(
      root, {"generate_content_config.thinking_config": shared_thinking}
  )
  clone_b = clone_agent_with_params(
      root, {"generate_content_config.thinking_config": shared_thinking}
  )

  assert isinstance(clone_a, LlmAgent)
  assert isinstance(clone_b, LlmAgent)
  assert clone_a.generate_content_config is not None
  assert clone_b.generate_content_config is not None
  tc_a = clone_a.generate_content_config.thinking_config
  tc_b = clone_b.generate_content_config.thinking_config
  assert tc_a is not None and tc_b is not None
  assert tc_a is not shared_thinking
  assert tc_a is not tc_b
  tc_a.thinking_budget = 2048
  assert shared_thinking.thinking_budget == 512
  assert tc_b.thinking_budget == 512


def test_recursive_merge_preserves_nested_model_dict_and_http_siblings() -> (
    None
):
  """Depth-3 models, dict fields, and http_options keep sibling fields."""
  ssl_ctx = ssl.create_default_context()
  base_cfg = types.GenerateContentConfig(
      labels={"env": "prod", "team": "adk"},
      tool_config=types.ToolConfig(
          function_calling_config=types.FunctionCallingConfig(
              mode=types.FunctionCallingConfigMode.AUTO,
              allowed_function_names=["search"],
          )
      ),
      http_options=types.HttpOptions(
          headers={"x-env": "prod", "x-team": "adk"},
          retry_options=types.HttpRetryOptions(attempts=3, initial_delay=1.5),
          client_args={"verify": ssl_ctx, "timeout": 10},
      ),
  )
  root = LlmAgent(
      name="Root",
      model="gemini-2.5-flash",
      generate_content_config=base_cfg,
  )

  cloned = clone_agent_with_params(
      root,
      {
          "generate_content_config.labels.env": "staging",
          "generate_content_config.tool_config.function_calling_config.mode": (
              types.FunctionCallingConfigMode.ANY
          ),
          "generate_content_config.http_options.headers.x-env": "staging",
          "generate_content_config.http_options.retry_options.attempts": 5,
          "generate_content_config.http_options.client_args.timeout": 30,
      },
  )

  assert isinstance(cloned, LlmAgent)
  cfg = cloned.generate_content_config
  assert cfg is not None
  assert cfg.labels == {"env": "staging", "team": "adk"}
  assert base_cfg.labels == {"env": "prod", "team": "adk"}
  assert cfg.tool_config is not None
  fc = cfg.tool_config.function_calling_config
  assert fc is not None
  assert fc.mode == types.FunctionCallingConfigMode.ANY
  assert fc.allowed_function_names == ["search"]
  assert base_cfg.tool_config is not None
  assert base_cfg.tool_config.function_calling_config is not None
  assert (
      base_cfg.tool_config.function_calling_config.mode
      == types.FunctionCallingConfigMode.AUTO
  )
  assert cfg.http_options is not None
  assert cfg.http_options.headers == {"x-env": "staging", "x-team": "adk"}
  assert cfg.http_options.retry_options == types.HttpRetryOptions(
      attempts=5, initial_delay=1.5
  )
  assert cfg.http_options.client_args == {"verify": ssl_ctx, "timeout": 30}
  assert cfg.http_options.client_args["verify"] is ssl_ctx
  assert base_cfg.http_options is not None
  assert base_cfg.http_options.headers == {"x-env": "prod", "x-team": "adk"}
  assert base_cfg.http_options.retry_options == types.HttpRetryOptions(
      attempts=3, initial_delay=1.5
  )
  assert base_cfg.http_options.client_args == {"verify": ssl_ctx, "timeout": 10}


def test_generate_content_config_subclass_is_preserved_and_validated() -> None:
  """Provider GenerateContentConfig subclasses keep extra fields and rules."""
  root = LlmAgent(
      name="Root",
      model="gemini-2.5-flash",
      generate_content_config=_EffortGenerateContentConfig(
          effort="low", temperature=0.2
      ),
  )

  dotted_clone = clone_agent_with_params(
      root, {"generate_content_config.effort": "high"}
  )
  whole_clone = clone_agent_with_params(
      root, {"generate_content_config": {"effort": "medium"}}
  )

  assert isinstance(dotted_clone, LlmAgent)
  assert isinstance(
      dotted_clone.generate_content_config, _EffortGenerateContentConfig
  )
  assert dotted_clone.generate_content_config.effort == "high"
  assert dotted_clone.generate_content_config.temperature == pytest.approx(0.2)
  assert isinstance(whole_clone, LlmAgent)
  assert isinstance(
      whole_clone.generate_content_config, _EffortGenerateContentConfig
  )
  assert whole_clone.generate_content_config.effort == "medium"

  with pytest.raises(ValidationError, match="thinking_level is not supported"):
    clone_agent_with_params(
        root,
        {
            "generate_content_config.thinking_config.thinking_level": (
                types.ThinkingLevel.HIGH
            )
        },
    )


def test_camel_case_config_aliases_apply_and_sync_planner() -> None:
  """camelCase aliases work for config, http_options, and BuiltInPlanner."""
  root = LlmAgent(
      name="Root",
      model="gemini-2.5-flash",
      planner=BuiltInPlanner(
          thinking_config=types.ThinkingConfig(
              include_thoughts=True, thinking_budget=1024
          )
      ),
      generate_content_config=types.GenerateContentConfig(
          temperature=0.3,
          http_options=types.HttpOptions(headers={"x-a": "1"}),
      ),
  )

  cloned = clone_agent_with_params(
      root,
      {
          "generate_content_config.maxOutputTokens": 512,
          "generate_content_config.httpOptions.apiVersion": "v1",
          "generate_content_config.thinkingConfig.thinkingBudget": 256,
      },
  )

  assert isinstance(cloned, LlmAgent)
  cfg = cloned.generate_content_config
  assert cfg is not None
  assert cfg.max_output_tokens == 512
  assert cfg.temperature == pytest.approx(0.3)
  assert cfg.http_options is not None
  assert cfg.http_options.api_version == "v1"
  assert cfg.http_options.headers == {"x-a": "1"}
  expected_tc = types.ThinkingConfig(include_thoughts=True, thinking_budget=256)
  assert cfg.thinking_config == expected_tc
  assert isinstance(cloned.planner, BuiltInPlanner)
  assert cloned.planner.thinking_config == expected_tc

  with pytest.raises(ValueError, match="Duplicate override"):
    clone_agent_with_params(
        root,
        {
            "generate_content_config.max_output_tokens": 128,
            "generate_content_config.maxOutputTokens": 256,
        },
    )


def test_swept_builtin_planner_is_isolated_across_clones() -> None:
  """A swept BuiltInPlanner instance is copied per clone, not shared."""
  shared_planner = BuiltInPlanner(
      thinking_config=types.ThinkingConfig(thinking_budget=1024)
  )
  root = LlmAgent(name="Root", model="gemini-2.5-flash")

  clone_a = clone_agent_with_params(root, {"planner": shared_planner})
  clone_b = clone_agent_with_params(root, {"planner": shared_planner})

  assert isinstance(clone_a, LlmAgent)
  assert isinstance(clone_b, LlmAgent)
  assert isinstance(clone_a.planner, BuiltInPlanner)
  assert isinstance(clone_b.planner, BuiltInPlanner)
  assert clone_a.planner is not shared_planner
  assert clone_a.planner is not clone_b.planner
  assert clone_a.planner.thinking_config is not shared_planner.thinking_config
  assert clone_a.planner.thinking_config is not clone_b.planner.thinking_config
  clone_a.planner.thinking_config.thinking_budget = 256
  assert shared_planner.thinking_config.thinking_budget == 1024
  assert clone_b.planner.thinking_config.thinking_budget == 1024


@pytest.mark.parametrize(
    ("bad_params", "expected_pattern"),
    [
        (
            {"generate_content_config.system_instruction": "Be brief."},
            r"LlmAgent\.instruction",
        ),
        (
            {"generate_content_config": {"system_instruction": "Be brief."}},
            r"LlmAgent\.instruction",
        ),
        (
            {
                "generate_content_config.tools": [
                    types.Tool(
                        function_declarations=[
                            types.FunctionDeclaration(name="ping")
                        ]
                    )
                ]
            },
            r"LlmAgent\.tools",
        ),
        (
            {
                "generate_content_config.response_schema": types.Schema(
                    type=types.Type.STRING
                )
            },
            r"LlmAgent\.output_schema",
        ),
        (
            {
                "generate_content_config.http_options.base_url": (
                    "https://example.test"
                )
            },
            r"Base URL",
        ),
    ],
)
def test_forbidden_llm_agent_config_fields_fail_fast(
    bad_params: dict[str, Any],
    expected_pattern: str,
) -> None:
  """LlmAgent.validate_generate_content_config runs on swept configs."""
  root = LlmAgent(name="Root", model="gemini-2.5-flash")

  with pytest.raises(ValueError, match=expected_pattern):
    clone_agent_with_params(root, bad_params)


def test_agent_tool_metadata_syncs_with_cloned_agent() -> None:
  """AgentTool name/description track overridden child fields unless custom."""
  helper = LlmAgent(
      name="Helper", model="gemini-2.5-flash", description="Old helper"
  )
  aliased = LlmAgent(
      name="Aliased", model="gemini-2.5-flash", description="Old aliased"
  )
  aliased_tool = AgentTool(agent=aliased)
  aliased_tool.name = "custom_tool_alias"
  root = LlmAgent(
      name="Root",
      model="gemini-2.5-flash",
      tools=[AgentTool(agent=helper), aliased_tool],
  )

  cloned = clone_agent_with_params(
      root,
      {
          "tools.Helper.name": "HelperV2",
          "tools.Helper.description": "New helper",
          "tools.Aliased.name": "AliasedV2",
          "tools.Aliased.description": "New aliased",
      },
  )

  assert isinstance(cloned, LlmAgent)
  tool_0, tool_1 = cloned.tools
  assert isinstance(tool_0, AgentTool)
  assert isinstance(tool_1, AgentTool)
  assert tool_0.name == "HelperV2"
  assert tool_0.description == "New helper"
  assert tool_1.name == "custom_tool_alias"
  assert tool_1.description == "New aliased"
  assert root.tools[0].name == "Helper"
  assert root.tools[0].description == "Old helper"


def test_shared_agent_across_levels_preserves_identity_and_applies_override() -> (
    None
):
  """An agent in sub_agents and an AgentTool on another sibling shares one clone."""
  a = LlmAgent(name="A", model="old")
  b = LlmAgent(name="B", model="m", tools=[AgentTool(agent=a)])
  root = LlmAgent(name="root", model="m", sub_agents=[a, b])

  cloned = clone_agent_with_params(root, {"sub_agents.A.model": "new"})
  assert isinstance(cloned, LlmAgent)
  cloned_a, cloned_b = cloned.sub_agents
  assert isinstance(cloned_a, LlmAgent)
  assert isinstance(cloned_b, LlmAgent)
  assert cloned_a.model == "new"
  assert cloned_b.tools[0].agent.model == "new"
  assert cloned_a is cloned_b.tools[0].agent

  # Reverse sibling order (AgentTool sibling before sub_agent A) also shares one clone.
  a2 = LlmAgent(name="A", model="old")
  b2 = LlmAgent(name="B", model="m", tools=[AgentTool(agent=a2)])
  root_rev = LlmAgent(name="root_rev", model="m", sub_agents=[b2, a2])
  cloned_rev = clone_agent_with_params(root_rev, {"sub_agents.A.model": "new"})
  assert isinstance(cloned_rev, LlmAgent)
  cloned_b_rev, cloned_a_rev = cloned_rev.sub_agents
  assert isinstance(cloned_a_rev, LlmAgent)
  assert isinstance(cloned_b_rev, LlmAgent)
  assert cloned_a_rev.model == "new"
  assert cloned_b_rev.tools[0].agent.model == "new"
  assert cloned_a_rev is cloned_b_rev.tools[0].agent


def test_dotted_thinking_override_preserves_planner_only_fields() -> None:
  """Dotted thinking_config override starts from BuiltInPlanner when both are set."""
  root = LlmAgent(
      name="root",
      model="gemini-2.5-flash",
      generate_content_config=types.GenerateContentConfig(
          thinking_config=types.ThinkingConfig(thinking_budget=64)
      ),
      planner=BuiltInPlanner(
          thinking_config=types.ThinkingConfig(
              include_thoughts=True, thinking_budget=128
          )
      ),
  )

  cloned = clone_agent_with_params(
      root, {"generate_content_config.thinking_config.thinking_budget": 1024}
  )
  assert isinstance(cloned, LlmAgent)
  assert isinstance(cloned.planner, BuiltInPlanner)
  assert cloned.planner.thinking_config.include_thoughts is True
  assert cloned.planner.thinking_config.thinking_budget == 1024
  assert cloned.generate_content_config is not None
  assert cloned.generate_content_config.thinking_config is not None
  assert cloned.generate_content_config.thinking_config.include_thoughts is True
  assert cloned.generate_content_config.thinking_config.thinking_budget == 1024


def test_ambiguous_duplicate_child_or_tool_name_raises() -> None:
  """Targeting a duplicate sub-agent or AgentTool name fails fast with ValueError."""
  dup_sub = SequentialAgent(
      name="root",
      sub_agents=[
          LlmAgent(name="helper", model="m1"),
          LlmAgent(name="helper", model="m2"),
      ],
  )
  with pytest.raises(ValueError, match=r"multiple sub-agents named 'helper'"):
    clone_agent_with_params(dup_sub, {"sub_agents.helper.model": "X"})

  dup_tool = LlmAgent(
      name="root",
      model="m",
      tools=[
          AgentTool(agent=LlmAgent(name="helper", model="m1")),
          AgentTool(agent=LlmAgent(name="helper", model="m2")),
      ],
  )
  with pytest.raises(
      ValueError, match=r"multiple AgentTool agents named 'helper'"
  ):
    clone_agent_with_params(dup_tool, {"tools.helper.model": "X"})
