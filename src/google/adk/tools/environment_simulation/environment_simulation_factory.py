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
from typing import Awaitable
from typing import Callable
from typing import Dict
from typing import Optional

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.environment_simulation.environment_simulation_config import EnvironmentSimulationConfig
from google.adk.tools.environment_simulation.environment_simulation_engine import EnvironmentSimulationEngine
from google.adk.tools.environment_simulation.environment_simulation_plugin import EnvironmentSimulationPlugin

from ...features import experimental
from ...features import FeatureName


@experimental(FeatureName.ENVIRONMENT_SIMULATION)
class EnvironmentSimulationFactory:
  """Factory for creating EnvironmentSimulation instances."""

  @staticmethod
  def create_callback(
      config: EnvironmentSimulationConfig,
  ) -> Callable[
      [BaseTool, Dict[str, Any], Any], Awaitable[Optional[Dict[str, Any]]]
  ]:
    """Creates a callback function for EnvironmentSimulation.

    Args:
      config: The configuration for the EnvironmentSimulation.

    Returns:
      A callable that can be used as a before_tool_callback or
      after_tool_callback.
    """
    simulator_engine = EnvironmentSimulationEngine(config)

    async def _environment_simulation_callback(
        tool: BaseTool, args: Dict[str, Any], tool_context: Any
    ) -> Optional[Dict[str, Any]]:
      return await simulator_engine.simulate(tool, args, tool_context)

    return _environment_simulation_callback

  @staticmethod
  def create_plugin(
      config: EnvironmentSimulationConfig,
  ) -> EnvironmentSimulationPlugin:
    """Creates an ADK Plugin for EnvironmentSimulation.

    Args:
      config: The configuration for the EnvironmentSimulation.

    Returns:
      An instance of EnvironmentSimulationPlugin that can be used as an ADK
      plugin.
    """
    simulator_engine = EnvironmentSimulationEngine(config)
    return EnvironmentSimulationPlugin(simulator_engine)
