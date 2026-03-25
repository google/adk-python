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

"""Unit tests for the new on_run_error_callback and on_agent_error_callback
lifecycle hooks added to BasePlugin and PluginManager (issue #4774).

These tests focus on:
  1. PluginManager._run_error_callbacks — fan-out semantics (all plugins
     notified even when one raises).
  2. PluginManager.run_on_run_error_callback and
     run_on_agent_error_callback — correct argument forwarding.
  3. BasePlugin default no-op implementations return None.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import Mock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.plugins.plugin_manager import PluginManager
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ObservabilityPlugin(BasePlugin):
  """A test plugin that records every on_*_error_callback invocation."""

  __test__ = False

  def __init__(self, name: str, *, raise_on_error: bool = False):
    super().__init__(name)
    self.agent_error_calls: list[dict] = []
    self.run_error_calls: list[dict] = []
    self._raise_on_error = raise_on_error

  async def on_agent_error_callback(
      self, *, agent, callback_context, error
  ) -> None:
    self.agent_error_calls.append(
        {"agent": agent, "callback_context": callback_context, "error": error}
    )
    if self._raise_on_error:
      raise RuntimeError(
          f"{self.name} intentionally raised in on_agent_error_callback"
      )

  async def on_run_error_callback(self, *, invocation_context, error) -> None:
    self.run_error_calls.append(
        {"invocation_context": invocation_context, "error": error}
    )
    if self._raise_on_error:
      raise RuntimeError(
          f"{self.name} intentionally raised in on_run_error_callback"
      )


# ---------------------------------------------------------------------------
# BasePlugin default no-op tests
# ---------------------------------------------------------------------------


class TestBasePluginDefaults:
  """Verifies that the default BasePlugin implementations are safe no-ops."""

  @pytest.mark.asyncio
  async def test_on_agent_error_callback_returns_none_by_default(self):
    """Default on_agent_error_callback must return None without raising."""

    class MinimalPlugin(BasePlugin):
      pass

    plugin = MinimalPlugin(name="minimal")
    result = await plugin.on_agent_error_callback(
        agent=Mock(),
        callback_context=Mock(),
        error=ValueError("boom"),
    )
    assert result is None

  @pytest.mark.asyncio
  async def test_on_run_error_callback_returns_none_by_default(self):
    """Default on_run_error_callback must return None without raising."""

    class MinimalPlugin(BasePlugin):
      pass

    plugin = MinimalPlugin(name="minimal")
    result = await plugin.on_run_error_callback(
        invocation_context=Mock(),
        error=RuntimeError("boom"),
    )
    assert result is None


# ---------------------------------------------------------------------------
# PluginManager._run_error_callbacks fan-out semantics
# ---------------------------------------------------------------------------


class TestRunErrorCallbacksFanOut:
  """Verifies that _run_error_callbacks notifies ALL plugins regardless of
  earlier plugin failures (unlike the early-exit _run_callbacks)."""

  @pytest.mark.asyncio
  async def test_all_plugins_notified_when_no_failures(self):
    """All plugins must be called when none raise."""
    plugin_a = ObservabilityPlugin("a")
    plugin_b = ObservabilityPlugin("b")
    manager = PluginManager(plugins=[plugin_a, plugin_b])

    inv_ctx = Mock(spec=InvocationContext)
    error = ValueError("run exploded")

    await manager.run_on_run_error_callback(
        invocation_context=inv_ctx, error=error
    )

    assert len(plugin_a.run_error_calls) == 1
    assert len(plugin_b.run_error_calls) == 1
    assert plugin_a.run_error_calls[0]["error"] is error
    assert plugin_b.run_error_calls[0]["error"] is error

  @pytest.mark.asyncio
  async def test_subsequent_plugins_called_even_when_first_raises(self):
    """If plugin_a raises during on_run_error_callback, plugin_b must still
    be called — a broken observability plugin must not silence others."""
    plugin_a = ObservabilityPlugin("a", raise_on_error=True)
    plugin_b = ObservabilityPlugin("b")
    manager = PluginManager(plugins=[plugin_a, plugin_b])

    inv_ctx = Mock(spec=InvocationContext)
    error = ValueError("original error")

    # Should NOT raise even though plugin_a raises internally
    await manager.run_on_run_error_callback(
        invocation_context=inv_ctx, error=error
    )

    assert len(plugin_b.run_error_calls) == 1
    assert plugin_b.run_error_calls[0]["error"] is error

  @pytest.mark.asyncio
  async def test_agent_error_all_plugins_notified_when_no_failures(self):
    """All plugins must be called for on_agent_error_callback."""
    plugin_a = ObservabilityPlugin("a")
    plugin_b = ObservabilityPlugin("b")
    manager = PluginManager(plugins=[plugin_a, plugin_b])

    agent = Mock(spec=BaseAgent)
    cb_ctx = Mock(spec=CallbackContext)
    error = RuntimeError("agent died")

    await manager.run_on_agent_error_callback(
        agent=agent, callback_context=cb_ctx, error=error
    )

    assert len(plugin_a.agent_error_calls) == 1
    assert len(plugin_b.agent_error_calls) == 1
    assert plugin_a.agent_error_calls[0]["agent"] is agent
    assert plugin_b.agent_error_calls[0]["error"] is error

  @pytest.mark.asyncio
  async def test_agent_error_subsequent_plugins_called_even_when_first_raises(
      self,
  ):
    """Broken plugin in on_agent_error_callback must not block others."""
    plugin_a = ObservabilityPlugin("a", raise_on_error=True)
    plugin_b = ObservabilityPlugin("b")
    manager = PluginManager(plugins=[plugin_a, plugin_b])

    agent = Mock(spec=BaseAgent)
    cb_ctx = Mock(spec=CallbackContext)
    error = RuntimeError("agent died")

    await manager.run_on_agent_error_callback(
        agent=agent, callback_context=cb_ctx, error=error
    )

    assert len(plugin_b.agent_error_calls) == 1

  @pytest.mark.asyncio
  async def test_no_plugins_registered_does_not_raise(self):
    """run_on_run_error_callback with zero plugins must be a safe no-op."""
    manager = PluginManager()
    await manager.run_on_run_error_callback(
        invocation_context=Mock(), error=ValueError("x")
    )

  @pytest.mark.asyncio
  async def test_no_plugins_registered_agent_does_not_raise(self):
    """run_on_agent_error_callback with zero plugins must be a safe no-op."""
    manager = PluginManager()
    await manager.run_on_agent_error_callback(
        agent=Mock(), callback_context=Mock(), error=ValueError("x")
    )


# ---------------------------------------------------------------------------
# PluginManager — argument forwarding
# ---------------------------------------------------------------------------


class TestArgumentForwarding:
  """Verifies exact argument passing to each plugin callback."""

  @pytest.mark.asyncio
  async def test_run_on_run_error_callback_passes_correct_kwargs(self):
    plugin = ObservabilityPlugin("p")
    manager = PluginManager(plugins=[plugin])

    inv_ctx = Mock(spec=InvocationContext)
    error = KeyError("missing_key")

    await manager.run_on_run_error_callback(
        invocation_context=inv_ctx, error=error
    )

    call = plugin.run_error_calls[0]
    assert call["invocation_context"] is inv_ctx
    assert call["error"] is error

  @pytest.mark.asyncio
  async def test_run_on_agent_error_callback_passes_correct_kwargs(self):
    plugin = ObservabilityPlugin("p")
    manager = PluginManager(plugins=[plugin])

    agent = Mock(spec=BaseAgent)
    cb_ctx = Mock(spec=CallbackContext)
    error = TypeError("bad_type")

    await manager.run_on_agent_error_callback(
        agent=agent, callback_context=cb_ctx, error=error
    )

    call = plugin.agent_error_calls[0]
    assert call["agent"] is agent
    assert call["callback_context"] is cb_ctx
    assert call["error"] is error


# ---------------------------------------------------------------------------
# Contrast with _run_callbacks early-exit semantics
# ---------------------------------------------------------------------------


class TestErrorCallbacksDoNotEarlyExit:
  """Confirms error callbacks ignore non-None returns (no early exit)."""

  @pytest.mark.asyncio
  async def test_on_run_error_return_value_is_ignored(self):
    """on_run_error_callback returning a value must not stop other plugins."""

    class ReturningPlugin(BasePlugin):

      def __init__(self, name):
        super().__init__(name)
        self.called = False

      async def on_run_error_callback(self, **kwargs):
        self.called = True
        return "non-none-value-that-should-be-ignored"

    p1 = ReturningPlugin("p1")
    p2 = ReturningPlugin("p2")
    manager = PluginManager(plugins=[p1, p2])

    await manager.run_on_run_error_callback(
        invocation_context=Mock(), error=ValueError("x")
    )

    assert p1.called
    assert p2.called  # must NOT be short-circuited

  @pytest.mark.asyncio
  async def test_on_agent_error_return_value_is_ignored(self):
    """on_agent_error_callback returning a value must not stop other plugins."""

    class ReturningPlugin(BasePlugin):

      def __init__(self, name):
        super().__init__(name)
        self.called = False

      async def on_agent_error_callback(self, **kwargs):
        self.called = True
        return "non-none-value"

    p1 = ReturningPlugin("p1")
    p2 = ReturningPlugin("p2")
    manager = PluginManager(plugins=[p1, p2])

    await manager.run_on_agent_error_callback(
        agent=Mock(), callback_context=Mock(), error=ValueError("x")
    )

    assert p1.called
    assert p2.called
