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

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any
from typing import Dict
from typing import Optional

environment_simulation_logger = logging.getLogger(
    "environment_simulation_logger"
)

from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.environment_simulation.environment_simulation_config import EnvironmentSimulationConfig
from google.adk.tools.environment_simulation.environment_simulation_config import InjectionConfig
from google.adk.tools.environment_simulation.environment_simulation_config import MockStrategy as MockStrategyEnum
from google.adk.tools.environment_simulation.strategies import base as base_mock_strategies
from google.adk.tools.environment_simulation.strategies import tool_spec_mock_strategy
from google.adk.tools.environment_simulation.tool_connection_analyzer import ToolConnectionAnalyzer
from google.adk.tools.environment_simulation.tool_connection_map import ToolConnectionMap
from google.genai import types as genai_types

from ...utils._event_loop_cache import per_loop_value
from ..tool_args_matcher._semantic_arg_matcher import SemanticArgMatcher
from ..tool_args_matcher._tool_args_matcher import BaseArgMatcher
from ..tool_args_matcher._tool_args_matcher import ToolArgsMatcher

# Upper bound on args_matcher comparisons an engine runs at once. A tool
# recorded many times produces many injection configs, and a semantic
# comparison makes a model call for each of them.
_MAX_CONCURRENT_ARG_MATCHES = 8


def _create_mock_strategy(
    mock_strategy_type: MockStrategyEnum,
    llm_name: str,
    llm_config: genai_types.GenerateContentConfig,
) -> base_mock_strategies.MockStrategy:
  """Creates a mock strategy based on the given type."""
  if mock_strategy_type == MockStrategyEnum.MOCK_STRATEGY_TOOL_SPEC:
    return tool_spec_mock_strategy.ToolSpecMockStrategy(llm_name, llm_config)
  if mock_strategy_type == MockStrategyEnum.MOCK_STRATEGY_TRACING:
    return base_mock_strategies.TracingMockStrategy(llm_name, llm_config)
  raise ValueError(f"Unknown mock strategy type: {mock_strategy_type}")


def _with_default_semantic_model(
    args_matcher: ToolArgsMatcher, model: str
) -> ToolArgsMatcher:
  """Returns a copy whose semantic matchers without a model use `model`.

  The copy leaves the caller's config untouched, so the same config can be
  given to engines with different simulation models.
  """

  def _resolve(matcher: BaseArgMatcher) -> BaseArgMatcher:
    if isinstance(matcher, SemanticArgMatcher):
      if matcher.model is None:
        return matcher.model_copy(update={"model": model}, deep=True)
      return matcher.model_copy(deep=True)
    return matcher

  return args_matcher.model_copy(
      update={
          "default_matcher": _resolve(args_matcher.default_matcher),
          "per_arg_matchers": {
              arg_name: _resolve(matcher)
              for arg_name, matcher in args_matcher.per_arg_matchers.items()
          },
      }
  )


class EnvironmentSimulationEngine:
  """Core engine to handle the simulation logic."""

  def __init__(self, config: EnvironmentSimulationConfig):
    self._config = config
    self._tool_sim_configs = {
        c.tool_name: c for c in config.tool_simulation_configs
    }
    # Keyed by tool name and the index of the injection config.
    self._args_matchers: dict[tuple[str, int], ToolArgsMatcher] = {
        (c.tool_name, index): _with_default_semantic_model(
            injection_config.args_matcher, config.simulation_model
        )
        for c in config.tool_simulation_configs
        for index, injection_config in enumerate(c.injection_configs)
        if injection_config.args_matcher is not None
    }
    self._is_analyzed = False
    self._tool_connection_map: Optional[ToolConnectionMap] = None
    self._analyzer = ToolConnectionAnalyzer(
        llm_name=config.simulation_model,
        llm_config=config.simulation_model_configuration,
    )
    self._state_store: Dict[str, Any] = {}
    self._random_generator = random.Random()
    self._seeded_generators: dict[tuple[str, int], random.Random] = {}
    for c in config.tool_simulation_configs:
      for i, injection_config in enumerate(c.injection_configs):
        if injection_config.random_seed is not None:
          self._seeded_generators[(c.tool_name, i)] = random.Random(
              injection_config.random_seed
          )
    self._environment_data = config.environment_data
    self._tracing = config.tracing

  async def _match_with_args_matcher(
      self,
      *,
      tool_name: str,
      args: dict[str, Any],
      match_args: dict[str, Any],
      args_matcher: ToolArgsMatcher,
  ) -> bool:
    """Compares a call with an injection config through its args_matcher."""
    semaphore = per_loop_value(
        self,
        "_match_semaphores",
        lambda: asyncio.Semaphore(_MAX_CONCURRENT_ARG_MATCHES),
    )
    async with semaphore:
      result = await args_matcher.match(
          tool_name=tool_name,
          expected=match_args,
          actual=args,
          short_circuit=True,
      )
    environment_simulation_logger.debug(
        "args_matcher for tool '%s' matched=%s per_arg=%s",
        tool_name,
        result.matched,
        {name: r.matched for name, r in result.per_arg.items()},
    )
    return result.matched

  @staticmethod
  def _matches_exactly(
      args: dict[str, Any], injection_config: InjectionConfig
  ) -> bool:
    """Checks whether a call equals an injection config's match_args.

    Every entry of match_args must appear in the call with an equal value, the
    check used for configs without an args_matcher. A config whose
    args_matcher disallows extra arguments also requires the call to have no
    argument that match_args does not name.
    """
    match_args = injection_config.match_args or {}
    if not all(item in args.items() for item in match_args.items()):
      return False
    args_matcher = injection_config.args_matcher
    if args_matcher is not None and not args_matcher.allow_extra_args:
      return all(arg_name in match_args for arg_name in args)
    return True

  def _draw(
      self, tool_name: str, index: int, injection_config: InjectionConfig
  ) -> bool:
    """Draws whether a matching injection config applies to this call."""
    generator = self._seeded_generators.get(
        (tool_name, index), self._random_generator
    )
    return generator.random() < injection_config.injection_probability

  async def _match_remaining_configs(
      self,
      *,
      tool_name: str,
      args: dict[str, Any],
      injection_configs: list[InjectionConfig],
      drawn: set[int],
  ) -> InjectionConfig | None:
    """Finds the first applicable config among those not yet drawn.

    Comparisons through an args_matcher may call a model, so they start
    together and run concurrently. The engine semaphore admits them in
    priority order, and those still waiting are cancelled once an injection
    applies.
    """
    pending_matches = {
        index: asyncio.create_task(
            self._match_with_args_matcher(
                tool_name=tool_name,
                args=args,
                match_args=injection_config.match_args,
                args_matcher=self._args_matchers[(tool_name, index)],
            )
        )
        for index, injection_config in enumerate(injection_configs)
        if index not in drawn
        and injection_config.match_args
        and (tool_name, index) in self._args_matchers
    }
    try:
      for index, injection_config in enumerate(injection_configs):
        if index in drawn:
          continue
        if index in pending_matches:
          if not await pending_matches[index]:
            continue
        elif injection_config.match_args:
          if not self._matches_exactly(args, injection_config):
            continue
        if self._draw(tool_name, index, injection_config):
          return injection_config
      return None
    finally:
      for task in pending_matches.values():
        task.cancel()
      await asyncio.gather(*pending_matches.values(), return_exceptions=True)

  async def simulate(
      self, tool: BaseTool, args: Dict[str, Any], tool_context: Any
  ) -> Optional[Dict[str, Any]]:
    """Simulates a tool call."""
    if tool.name not in self._tool_sim_configs:
      return None

    tool_sim_config = self._tool_sim_configs[tool.name]

    if not self._is_analyzed and any(
        c.mock_strategy_type != MockStrategyEnum.MOCK_STRATEGY_UNSPECIFIED
        for c in self._config.tool_simulation_configs
    ):
      agent = tool_context._invocation_context.agent
      if isinstance(agent, LlmAgent):
        tools = await agent.canonical_tools(tool_context)
        self._tool_connection_map = await self._analyzer.analyze(tools)
      self._is_analyzed = True

    injection_configs = tool_sim_config.injection_configs
    # An exact hit must not depend on a model's verdict, so configs whose
    # match_args equal the call are tried first, in order, before any
    # args_matcher comparison starts. The check stops at the first config
    # without match_args: such a config applies to every call, so it keeps its
    # priority over the configs after it. Each config is drawn at most once per
    # call: one that matched exactly but lost its draw is not compared again.
    drawn: set[int] = set()
    matched_injection = None
    for index, injection_config in enumerate(injection_configs):
      if not injection_config.match_args:
        break
      if not self._matches_exactly(args, injection_config):
        continue
      drawn.add(index)
      if self._draw(tool.name, index, injection_config):
        matched_injection = injection_config
        break

    if matched_injection is None:
      matched_injection = await self._match_remaining_configs(
          tool_name=tool.name,
          args=args,
          injection_configs=injection_configs,
          drawn=drawn,
      )

    if matched_injection is not None:
      await asyncio.sleep(matched_injection.injected_latency_seconds)
      if matched_injection.injected_error is not None:
        return {
            "error_code": (
                matched_injection.injected_error.injected_http_error_code
            ),
            "error_message": matched_injection.injected_error.error_message,
        }
      if matched_injection.injected_response is not None:
        return matched_injection.injected_response

    # If no injection was applied, fall back to the mock strategy.
    if (
        tool_sim_config.mock_strategy_type
        == MockStrategyEnum.MOCK_STRATEGY_UNSPECIFIED
    ):
      environment_simulation_logger.warning(
          "Tool '%s' did not hit any injection config and has no mock strategy"
          " configured. Returning no-op.",
          tool.name,
      )
      return None

    mock_strategy = _create_mock_strategy(
        tool_sim_config.mock_strategy_type,
        self._config.simulation_model,
        self._config.simulation_model_configuration,
    )
    return await mock_strategy.mock(
        tool,
        args,
        tool_context,
        self._tool_connection_map,
        self._state_store,
        self._environment_data,
        self._tracing,
    )
