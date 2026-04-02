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

from google.adk.tools.environment_simulation import EnvironmentSimulationFactory
from google.adk.tools.environment_simulation.environment_simulation_config import EnvironmentSimulationConfig
from google.adk.tools.environment_simulation.environment_simulation_config import MockStrategy
from google.adk.tools.environment_simulation.environment_simulation_config import ToolSimulationConfig
from google.adk.tools.environment_simulation.environment_simulation_plugin import EnvironmentSimulationPlugin
import pytest


@pytest.mark.asyncio
@patch(
    "google.adk.tools.environment_simulation.environment_simulation_factory.EnvironmentSimulationEngine"
)
class TestEnvironmentSimulationFactory:
  """Test cases for the EnvironmentSimulation factory class."""

  @pytest.fixture
  def mock_config(self):
    """Fixture for a basic EnvironmentSimulationConfig."""
    return EnvironmentSimulationConfig(
        tool_simulation_configs=[
            ToolSimulationConfig(
                tool_name="test_tool",
                mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
            )
        ]
    )

  async def test_create_callback(self, mock_engine_class, mock_config):
    """Test that create_callback returns a valid callable."""
    mock_engine_instance = MagicMock()
    mock_engine_instance.simulate = AsyncMock(return_value=None)
    mock_engine_class.return_value = mock_engine_instance

    callback = EnvironmentSimulationFactory.create_callback(mock_config)
    assert callable(callback)
    await callback(MagicMock(), {}, MagicMock())

    mock_engine_class.assert_called_once_with(mock_config)
    mock_engine_instance.simulate.assert_awaited_once()

  @patch(
      "google.adk.tools.environment_simulation.environment_simulation_factory.EnvironmentSimulationPlugin"
  )
  def test_create_plugin(
      self, mock_plugin_class, mock_engine_class, mock_config
  ):
    """Test that create_plugin returns a valid EnvironmentSimulationPlugin instance."""
    plugin = EnvironmentSimulationFactory.create_plugin(mock_config)
    mock_engine_class.assert_called_once_with(mock_config)
    mock_plugin_class.assert_called_once_with(mock_engine_class.return_value)
    assert plugin == mock_plugin_class.return_value
