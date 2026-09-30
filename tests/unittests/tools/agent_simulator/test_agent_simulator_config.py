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

"""Tests for the deprecated AgentSimulatorConfig alias."""

import warnings

from google.adk.tools.agent_simulator.agent_simulator_config import AgentSimulatorConfig
from google.adk.tools.environment_simulation.environment_simulation_config import MockStrategy
from google.adk.tools.environment_simulation.environment_simulation_config import ToolSimulationConfig
import pytest


def _tool_configs() -> list[ToolSimulationConfig]:
  return [
      ToolSimulationConfig(
          tool_name="my_tool",
          mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
      )
  ]


def test_tracing_path_is_forwarded_to_tracing():
  """The renamed field must still reach the new `tracing` field."""
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    config = AgentSimulatorConfig(
        tool_simulation_configs=_tool_configs(),
        tracing_path="prior_run_trace",
    )

  assert config.tracing == "prior_run_trace"


def test_tracing_path_emits_deprecation_warning():
  with pytest.warns(DeprecationWarning, match="`tracing_path` is deprecated"):
    AgentSimulatorConfig(
        tool_simulation_configs=_tool_configs(),
        tracing_path="prior_run_trace",
    )


def test_explicit_tracing_wins_over_tracing_path():
  """When both are given the new field is authoritative, not the alias."""
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    config = AgentSimulatorConfig(
        tool_simulation_configs=_tool_configs(),
        tracing="explicit_trace",
        tracing_path="legacy_trace",
    )

  assert config.tracing == "explicit_trace"


def test_tracing_alone_does_not_warn():
  """Callers already on the new field must not see a deprecation warning."""
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    config = AgentSimulatorConfig(
        tool_simulation_configs=_tool_configs(),
        tracing="explicit_trace",
    )

  assert config.tracing == "explicit_trace"
  assert not [w for w in caught if "tracing_path" in str(w.message)]
