# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law of a-specific language governing permissions and
# limitations under the License.

import logging
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.tools.agent_simulator.agent_simulator_config import AgentSimulatorConfig
from google.adk.tools.agent_simulator.agent_simulator_config import InjectedError
from google.adk.tools.agent_simulator.agent_simulator_config import InjectionConfig
from google.adk.tools.agent_simulator.agent_simulator_config import MockStrategy
from google.adk.tools.agent_simulator.agent_simulator_config import ToolSimulationConfig
from google.adk.tools.agent_simulator.agent_simulator_engine import AgentSimulatorEngine
from google.genai import types as genai_types
import pytest


@patch(
    "google.adk.tools.agent_simulator.agent_simulator_engine.ToolConnectionAnalyzer"
)
@patch(
    "google.adk.tools.agent_simulator.agent_simulator_engine._create_mock_strategy"
)
@pytest.mark.asyncio
class TestAgentSimulatorEngineSimulate:
  """Test cases for the simulate method of AgentSimulatorEngine."""

  async def test_simulate_no_op_for_unconfigured_tool(
      self, mock_create_strategy, mock_analyzer
  ):
    """Test that simulate returns None for a tool not in the config."""
    config = AgentSimulatorConfig(
        tool_simulation_configs=[
            ToolSimulationConfig(
                tool_name="configured_tool",
                mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
            )
        ],
        simulation_model="test-model",
        simulation_model_configuration=genai_types.GenerateContentConfig(),
    )
    engine = AgentSimulatorEngine(config)
    mock_tool = MagicMock()
    mock_tool.name = "unconfigured_tool"
    result = await engine.simulate(mock_tool, {}, MagicMock())
    assert result is None

  async def test_injection_with_matching_args(
      self, mock_create_strategy, mock_analyzer
  ):
    """Test that an injection is applied when match_args match."""
    config = AgentSimulatorConfig(
        tool_simulation_configs=[
            ToolSimulationConfig(
                tool_name="test_tool",
                injection_configs=[
                    InjectionConfig(
                        match_args={"param": "value"},
                        injected_response={"injected": True},
                    )
                ],
            )
        ],
        simulation_model="test-model",
        simulation_model_configuration=genai_types.GenerateContentConfig(),
    )
    engine = AgentSimulatorEngine(config)
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    result = await engine.simulate(mock_tool, {"param": "value"}, MagicMock())
    assert result == {"injected": True}

  async def test_injection_not_applied_with_mismatched_args(
      self, mock_create_strategy, mock_analyzer
  ):
    """Test that an injection is not applied when match_args do not match."""
    mock_strategy_instance = MagicMock()
    mock_strategy_instance.mock = AsyncMock(return_value={"mocked": True})
    mock_create_strategy.return_value = mock_strategy_instance
    config = AgentSimulatorConfig(
        tool_simulation_configs=[
            ToolSimulationConfig(
                tool_name="test_tool",
                injection_configs=[
                    InjectionConfig(
                        match_args={"param": "value"},
                        injected_response={"injected": True},
                    )
                ],
                mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
            )
        ],
        simulation_model="test-model",
        simulation_model_configuration=genai_types.GenerateContentConfig(),
    )
    engine = AgentSimulatorEngine(config)
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    result = await engine.simulate(
        mock_tool, {"param": "different_value"}, MagicMock()
    )
    assert result == {"mocked": True}
    mock_create_strategy.assert_called_once_with(
        config.tool_simulation_configs[0].mock_strategy_type,
        config.simulation_model,
        config.simulation_model_configuration,
    )
    mock_strategy_instance.mock.assert_awaited_once()

  async def test_no_op_when_no_injection_hit_and_unspecified_strategy(
      self, mock_create_strategy, mock_analyzer, caplog
  ):
    """Test for no-op and warning when no injection hits and mock strategy is unspecified."""
    config = AgentSimulatorConfig(
        tool_simulation_configs=[
            ToolSimulationConfig(
                tool_name="test_tool",
                injection_configs=[
                    InjectionConfig(
                        match_args={"param": "value"},
                        injected_response={"injected": True},
                    )
                ],
                mock_strategy_type=MockStrategy.MOCK_STRATEGY_UNSPECIFIED,
            )
        ],
        simulation_model="test-model",
        simulation_model_configuration=genai_types.GenerateContentConfig(),
    )
    engine = AgentSimulatorEngine(config)
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"

    caplog.set_level(logging.WARNING, logger="agent_simulator_logger")
    with caplog.at_level(logging.WARNING, logger="agent_simulator_logger"):
      result = await engine.simulate(
          mock_tool, {"param": "different_value"}, MagicMock()
      )
      assert result is None
      assert (
          "did not hit any injection config and has no mock strategy"
          in caplog.text
      )
    mock_create_strategy.assert_not_called()

  async def test_injection_with_random_seed_is_deterministic(
      self, mock_create_strategy, mock_analyzer
  ):
    """Test that an injection with a random_seed is deterministic."""
    # With seed=42, random.random() is > 0.5, so this will NOT be injected
    # and should fall back to the mock strategy.
    mock_strategy_instance = MagicMock()
    mock_strategy_instance.mock = AsyncMock(return_value={"mocked": True})
    mock_create_strategy.return_value = mock_strategy_instance
    config_mocked = AgentSimulatorConfig(
        tool_simulation_configs=[
            ToolSimulationConfig(
                tool_name="test_tool",
                injection_configs=[
                    InjectionConfig(
                        injection_probability=0.5,
                        random_seed=42,  # A fixed seed
                        injected_response={"injected": True},
                    )
                ],
                mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
            )
        ],
        simulation_model="test-model",
        simulation_model_configuration=genai_types.GenerateContentConfig(),
    )
    engine_mocked = AgentSimulatorEngine(config_mocked)
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"

    result1 = await engine_mocked.simulate(mock_tool, {}, MagicMock())
    assert result1 == {"mocked": True}
    mock_create_strategy.assert_called_once_with(
        config_mocked.tool_simulation_configs[0].mock_strategy_type,
        config_mocked.simulation_model,
        config_mocked.simulation_model_configuration,
    )
    mock_strategy_instance.mock.assert_awaited_once()

    mock_create_strategy.reset_mock()
    mock_strategy_instance.mock.reset_mock()

    # With seed=100, random.random() is < 0.5, so this WILL be injected.
    config_injected = AgentSimulatorConfig(
        tool_simulation_configs=[
            ToolSimulationConfig(
                tool_name="test_tool",
                injection_configs=[
                    InjectionConfig(
                        injection_probability=0.5,
                        random_seed=100,  # A different fixed seed
                        injected_response={"injected": True},
                    )
                ],
                mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
            )
        ],
        simulation_model="test-model",
        simulation_model_configuration=genai_types.GenerateContentConfig(),
    )
    engine_injected = AgentSimulatorEngine(config_injected)
    result2 = await engine_injected.simulate(mock_tool, {}, MagicMock())
    assert result2 == {"injected": True}
    mock_create_strategy.assert_not_called()
