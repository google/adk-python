# Copyright 2024 Google LLC
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
from typing import Dict
from typing import Optional

from google.adk.plugins import BasePlugin
from google.adk.tools.agent_simulator.agent_simulator_config import AgentSimulatorConfig
from google.adk.tools.agent_simulator.agent_simulator_engine import AgentSimulatorEngine
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext


class AgentSimulatorPlugin(BasePlugin):
  """ADK Plugin for AgentSimulator."""

  name: str = "AgentSimulator"

  def __init__(self, simulator_engine: AgentSimulatorEngine):
    self._simulator_engine = simulator_engine

  async def before_tool_callback(
      self, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
  ) -> Optional[Dict[str, Any]]:
    """Invokes the AgentSimulatorEngine before a tool call."""
    return await self._simulator_engine.simulate(tool, tool_args, tool_context)
