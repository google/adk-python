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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.tools.agent_simulator.agent_simulator_plugin import AgentSimulatorPlugin
import pytest


@pytest.mark.asyncio
class TestAgentSimulatorPlugin:
  """Test cases for the AgentSimulatorPlugin."""

  @pytest.fixture
  def mock_simulator_engine(self):
    """Fixture for a mock AgentSimulatorEngine."""
    engine = MagicMock()
    engine.simulate = AsyncMock()
    return engine

  async def test_before_tool_callback(self, mock_simulator_engine):
    """Test that the before_tool_callback calls the engine's simulate method."""
    plugin = AgentSimulatorPlugin(mock_simulator_engine)

    mock_tool = MagicMock()
    mock_args = {}
    mock_context = MagicMock()

    await plugin.before_tool_callback(mock_tool, mock_args, mock_context)

    mock_simulator_engine.simulate.assert_awaited_once_with(
        mock_tool, mock_args, mock_context
    )
