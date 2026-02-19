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
import concurrent.futures
import logging
import random
import time
from typing import Any
from typing import Dict
from typing import Optional

agent_simulator_logger = logging.getLogger("agent_simulator_logger")

from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.agent_simulator.agent_simulator_config import AgentSimulatorConfig
from google.adk.tools.agent_simulator.agent_simulator_config import MockStrategy as MockStrategyEnum
from google.adk.tools.agent_simulator.agent_simulator_config import ToolSimulationConfig
from google.adk.tools.agent_simulator.strategies import base as base_mock_strategies
from google.adk.tools.agent_simulator.strategies import tool_spec_mock_strategy
from google.adk.tools.agent_simulator.tool_connection_analyzer import ToolConnectionAnalyzer
from google.adk.tools.agent_simulator.tool_connection_map import ToolConnectionMap
from google.adk.tools.base_tool import BaseTool


def _create_mock_strategy(
    mock_strategy_type: MockStrategyEnum,
    llm_name: str,
    llm_config: genai_types.GenerateContentConfig,
) -> base_mock_strategies.MockStrategy:
  """Creates a mock strategy based on the given type."""
  if mock_strategy_type == MockStrategyEnum.MOCK_STRATEGY_TOOL_SPEC:
    return tool_spec_mock_strategy.ToolSpecMockStrategy(llm_name, llm_config)
  if mock_strategy_type == MockStrategyEnum.MOCK_STRATEGY_TRACING:
    return base_mock_strategies.TracingMockStrategy()
  raise ValueError(f"Unknown mock strategy type: {mock_strategy_type}")


class AgentSimulatorEngine:
  """Core engine to handle the simulation logic."""

  def __init__(self, config: AgentSimulatorConfig):
    self._config = config
    self._tool_sim_configs = {
        c.tool_name: c for c in config.tool_simulation_configs
    }
    self._is_analyzed = False
    self._tool_connection_map: Optional[ToolConnectionMap] = None
    self._analyzer = ToolConnectionAnalyzer(
        llm_name=config.simulation_model,
        llm_config=config.simulation_model_configuration,
    )
    self._state_store = {}
    self._random_generator = random.Random()
    self._environment_data = config.environment_data

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

    for injection_config in tool_sim_config.injection_configs:
      if injection_config.match_args:
        if not all(
            item in args.items() for item in injection_config.match_args.items()
        ):
          continue

      if injection_config.random_seed is not None:
        self._random_generator.seed(injection_config.random_seed)

      if (
          self._random_generator.random()
          < injection_config.injection_probability
      ):
        time.sleep(injection_config.injected_latency_seconds)
        if injection_config.injected_error:
          return {
              "error_code": (
                  injection_config.injected_error.injected_http_error_code
              ),
              "error_message": injection_config.injected_error.error_message,
          }
        if injection_config.injected_response:
          return injection_config.injected_response

    # If no injection was applied, fall back to the mock strategy.
    if (
        tool_sim_config.mock_strategy_type
        == MockStrategyEnum.MOCK_STRATEGY_UNSPECIFIED
    ):
      agent_simulator_logger.warning(
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
    )
