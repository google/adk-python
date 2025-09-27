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
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
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

  def test_app_initialization_without_cache_config(self):
    """Test that the app is initialized correctly without context cache config."""
    mock_agent = Mock(spec=BaseAgent)
    app = App(name="test_app", root_agent=mock_agent)
    assert app.name == "test_app"
    assert app.root_agent == mock_agent
    assert app.context_cache_config is None

  def test_app_initialization_with_cache_config(self):
    """Test that the app is initialized correctly with context cache config."""
    mock_agent = Mock(spec=BaseAgent)
    cache_config = ContextCacheConfig(
        cache_intervals=15, ttl_seconds=3600, min_tokens=1024
    )

    app = App(
        name="test_app",
        root_agent=mock_agent,
        context_cache_config=cache_config,
    )

    assert app.name == "test_app"
    assert app.root_agent == mock_agent
    assert app.context_cache_config == cache_config
    assert app.context_cache_config.cache_intervals == 15
    assert app.context_cache_config.ttl_seconds == 3600
    assert app.context_cache_config.min_tokens == 1024

  def test_app_initialization_with_resumability_config(self):
    """Test that the app is initialized correctly with app config."""
    mock_agent = Mock(spec=BaseAgent)
    resumability_config = ResumabilityConfig(
        is_resumable=True,
    )
    app = App(
        name="test_app",
        root_agent=mock_agent,
        resumability_config=resumability_config,
    )

    assert app.name == "test_app"
    assert app.root_agent == mock_agent
    assert app.resumability_config == resumability_config
    assert app.resumability_config.is_resumable

  def test_app_with_all_components(self):
    """Test app with all components: agent, plugins, and cache config."""
    mock_agent = Mock(spec=BaseAgent)
    mock_plugin = Mock(spec=BasePlugin)
    cache_config = ContextCacheConfig(
        cache_intervals=20, ttl_seconds=7200, min_tokens=2048
    )
    resumability_config = ResumabilityConfig(
        is_resumable=True,
    )

    app = App(
        name="full_test_app",
        root_agent=mock_agent,
        plugins=[mock_plugin],
        context_cache_config=cache_config,
        resumability_config=resumability_config,
    )

    assert app.name == "full_test_app"
    assert app.root_agent == mock_agent
    assert app.plugins == [mock_plugin]
    assert app.context_cache_config == cache_config
    assert app.resumability_config == resumability_config
    assert app.resumability_config.is_resumable

  def test_app_cache_config_defaults(self):
    """Test that cache config has proper defaults when created."""
    mock_agent = Mock(spec=BaseAgent)
    cache_config = ContextCacheConfig()  # Use defaults

    app = App(
        name="default_cache_app",
        root_agent=mock_agent,
        context_cache_config=cache_config,
    )

    assert app.context_cache_config.cache_intervals == 10  # Default
    assert app.context_cache_config.ttl_seconds == 1800  # Default 30 minutes
    assert app.context_cache_config.min_tokens == 0  # Default

  def test_app_context_cache_config_is_optional(self):
    """Test that context_cache_config is truly optional."""
    mock_agent = Mock(spec=BaseAgent)

    # Should work without context_cache_config
    app = App(name="no_cache_app", root_agent=mock_agent)
    assert app.context_cache_config is None

    # Should work with explicit None
    app = App(
        name="explicit_none_app",
        root_agent=mock_agent,
        context_cache_config=None,
    )
    assert app.context_cache_config is None

  def test_app_resumability_config_defaults(self):
    """Test that app config has proper defaults when created."""
    mock_agent = Mock(spec=BaseAgent)

    app = App(
        name="default_resumability_config_app",
        root_agent=mock_agent,
        resumability_config=ResumabilityConfig(),
    )
    assert app.resumability_config is not None
    assert not app.resumability_config.is_resumable  # Default

  def test_app_resumability_config_is_optional(self):
    """Test that resumability_config is truly optional."""
    mock_agent = Mock(spec=BaseAgent)

    app = App(name="no_resumability_config_app", root_agent=mock_agent)
    assert app.resumability_config is None

    app = App(
        name="explicit_none_resumability_config_app",
        root_agent=mock_agent,
        resumability_config=None,
    )
    assert app.resumability_config is None
