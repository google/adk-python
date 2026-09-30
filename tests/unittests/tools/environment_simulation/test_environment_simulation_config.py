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

"""Tests for the environment simulation config validators."""

from google.adk.tools.environment_simulation.environment_simulation_config import EnvironmentSimulationConfig
from google.adk.tools.environment_simulation.environment_simulation_config import InjectedError
from google.adk.tools.environment_simulation.environment_simulation_config import InjectionConfig
from google.adk.tools.environment_simulation.environment_simulation_config import MockStrategy
from google.adk.tools.environment_simulation.environment_simulation_config import ToolSimulationConfig
from pydantic import ValidationError
import pytest


def _injected_error() -> InjectedError:
  return InjectedError(injected_http_error_code=404, error_message="not found")


class TestInjectionConfig:
  """Tests for InjectionConfig.check_injected_error_or_response."""

  def test_neither_error_nor_response_raises(self):
    """An injection that injects nothing has no effect and is rejected."""
    with pytest.raises(ValidationError, match="but not both, and not neither"):
      InjectionConfig()

  def test_both_error_and_response_raises(self):
    """The two are mutually exclusive: a call cannot both fail and succeed."""
    with pytest.raises(ValidationError, match="but not both, and not neither"):
      InjectionConfig(
          injected_error=_injected_error(),
          injected_response={"status": "ok"},
      )

  def test_only_error_is_accepted(self):
    config = InjectionConfig(injected_error=_injected_error())

    assert config.injected_error.injected_http_error_code == 404
    assert config.injected_response is None

  def test_only_response_is_accepted(self):
    config = InjectionConfig(injected_response={"status": "ok"})

    assert config.injected_response == {"status": "ok"}
    assert config.injected_error is None


class TestToolSimulationConfig:
  """Tests for ToolSimulationConfig.check_mock_strategy_type."""

  def test_no_injections_and_unspecified_strategy_raises(self):
    """With neither injections nor a strategy the tool cannot be simulated."""
    with pytest.raises(
        ValidationError,
        match="mock_strategy_type cannot be MOCK_STRATEGY_UNSPECIFIED",
    ):
      ToolSimulationConfig(tool_name="my_tool")

  def test_injections_alone_are_enough(self):
    """Injections handle the call, so no mock strategy is required."""
    config = ToolSimulationConfig(
        tool_name="my_tool",
        injection_configs=[InjectionConfig(injected_error=_injected_error())],
    )

    assert config.mock_strategy_type is MockStrategy.MOCK_STRATEGY_UNSPECIFIED
    assert len(config.injection_configs) == 1
    assert config.injection_configs[0].injected_error.error_message == (
        "not found"
    )

  def test_strategy_alone_is_enough(self):
    """A strategy handles every call, so no injections are required."""
    config = ToolSimulationConfig(
        tool_name="my_tool",
        mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
    )

    assert config.injection_configs == []


class TestEnvironmentSimulationConfig:
  """Tests for EnvironmentSimulationConfig.check_tool_simulation_configs."""

  def test_explicitly_empty_tool_simulation_configs_raises(self):
    with pytest.raises(
        ValidationError, match="tool_simulation_configs must be provided"
    ):
      EnvironmentSimulationConfig(tool_simulation_configs=[])

  def test_duplicate_tool_names_raise_and_name_the_duplicate(self):
    """Two configs for one tool are ambiguous, so the second is an error."""
    tool_config = ToolSimulationConfig(
        tool_name="dup_tool",
        mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
    )

    with pytest.raises(
        ValidationError, match="Duplicate tool_name found: dup_tool"
    ):
      EnvironmentSimulationConfig(
          tool_simulation_configs=[tool_config, tool_config.model_copy()]
      )

  def test_distinct_tool_names_are_kept_in_order(self):
    config = EnvironmentSimulationConfig(
        tool_simulation_configs=[
            ToolSimulationConfig(
                tool_name="first",
                mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
            ),
            ToolSimulationConfig(
                tool_name="second",
                mock_strategy_type=MockStrategy.MOCK_STRATEGY_TRACING,
            ),
        ]
    )

    assert [c.tool_name for c in config.tool_simulation_configs] == [
        "first",
        "second",
    ]
    assert config.tool_simulation_configs[1].mock_strategy_type is (
        MockStrategy.MOCK_STRATEGY_TRACING
    )
