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

from typing import Any
import warnings

from google.adk.tools.environment_simulation.environment_simulation_config import EnvironmentSimulationConfig
from google.adk.tools.environment_simulation.environment_simulation_config import InjectedError
from google.adk.tools.environment_simulation.environment_simulation_config import InjectionConfig
from google.adk.tools.environment_simulation.environment_simulation_config import MockStrategy
from google.adk.tools.environment_simulation.environment_simulation_config import ToolSimulationConfig
from pydantic import model_validator

warnings.warn(
    "google.adk.tools.agent_simulator.agent_simulator_config is moved to"
    " google.adk.tools.environment_simulation.environment_simulation_config",
    DeprecationWarning,
    stacklevel=2,
)


class AgentSimulatorConfig(EnvironmentSimulationConfig):
  """Deprecated AgentSimulatorConfig alias.

  Forwards tracing_path to tracing.
  """

  @model_validator(mode="before")
  @classmethod
  def convert_tracing_path(cls, data: Any) -> Any:
    """Convert tracing_path to tracing."""
    if isinstance(data, dict) and "tracing_path" in data:
      warnings.warn(
          "`tracing_path` is deprecated. Use `tracing` instead.",
          DeprecationWarning,
          stacklevel=2,
      )
      if "tracing" not in data:
        data["tracing"] = data.pop("tracing_path")
      else:
        data.pop("tracing_path")
    return data


__all__ = [
    "AgentSimulatorConfig",
    "InjectedError",
    "InjectionConfig",
    "MockStrategy",
    "ToolSimulationConfig",
]
