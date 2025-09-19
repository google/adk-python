# Copyright 2025 Google LLC
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

from unittest.mock import Mock

from google.adk.agents.base_agent import BaseAgent
from google.adk.apps.app import App
from google.adk.plugins.base_plugin import BasePlugin


class TestApp:
  """Tests for App class."""

  def test_app_initialization(self):
    """Test that the app is initialized correctly without plugins."""
    mock_agent = Mock(spec=BaseAgent)
    app = App(name="test_app", root_agent=mock_agent)
    assert app.name == "test_app"
    assert app.root_agent == mock_agent
    assert app.plugins == []

  def test_app_initialization_with_plugins(self):
    """Test that the app is initialized correctly with plugins."""
    mock_agent = Mock(spec=BaseAgent)
    mock_plugin = Mock(spec=BasePlugin)
    app = App(name="test_app", root_agent=mock_agent, plugins=[mock_plugin])
    assert app.name == "test_app"
    assert app.root_agent == mock_agent
    assert app.plugins == [mock_plugin]
