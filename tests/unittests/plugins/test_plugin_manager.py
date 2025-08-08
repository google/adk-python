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

"""Unit tests for the PluginManager."""

from __future__ import annotations

from unittest.mock import Mock

from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
# Assume the following path to your modules
# You might need to adjust this based on your project structure.
from google.adk.plugins.plugin_manager import PluginCallbackName
from google.adk.plugins.plugin_manager import PluginManager
import pytest


# A helper class to use in tests instead of mocks.
# This makes tests more explicit and easier to debug.
class TestPlugin(BasePlugin):
  __test__ = False
  """
  A test plugin that can be configured to return specific values or raise
  exceptions for any callback, and it logs which callbacks were invoked.
  """

  def __init__(self, name: str):
    super().__init__(name)
    # A log to track the names of callbacks that have been called.
    self.call_log: list[PluginCallbackName] = []
    # A map to configure return values for specific callbacks.
    self.return_values: dict[PluginCallbackName, any] = {}
    # A map to configure exceptions to be raised by specific callbacks.
    self.exceptions_to_raise: dict[PluginCallbackName, Exception] = {}

  async def _handle_callback(self, name: PluginCallbackName):
    """Generic handler for all callback methods."""
    self.call_log.append(name)
    if name in self.exceptions_to_raise:
      raise self.exceptions_to_raise[name]
    return self.return_values.get(name)

  # Implement all callback methods from the BasePlugin interface.
  async def on_user_message_callback(self, **kwargs):
    return await self._handle_callback("on_user_message_callback")

  async def before_run_callback(self, **kwargs):
    return await self._handle_callback("before_run_callback")

  async def after_run_callback(self, **kwargs):
    return await self._handle_callback("after_run_callback")

  async def on_event_callback(self, **kwargs):
    return await self._handle_callback("on_event_callback")

  async def before_agent_callback(self, **kwargs):
    return await self._handle_callback("before_agent_callback")

  async def after_agent_callback(self, **kwargs):
    return await self._handle_callback("after_agent_callback")

  async def before_tool_callback(self, **kwargs):
    return await self._handle_callback("before_tool_callback")

  async def after_tool_callback(self, **kwargs):
    return await self._handle_callback("after_tool_callback")

  async def on_tool_error_callback(self, **kwargs):
    return await self._handle_callback("on_tool_error_callback")

  async def before_model_callback(self, **kwargs):
    return await self._handle_callback("before_model_callback")

  async def after_model_callback(self, **kwargs):
    return await self._handle_callback("after_model_callback")

  async def on_model_error_callback(self, **kwargs):
    return await self._handle_callback("on_model_error_callback")


@pytest.fixture
def service() -> PluginManager:
  """Provides a clean PluginManager instance for each test."""
  return PluginManager()


@pytest.fixture
def plugin1() -> TestPlugin:
  """Provides a clean instance of our test plugin named 'plugin1'."""
  return TestPlugin(name="plugin1")


@pytest.fixture
def plugin2() -> TestPlugin:
  """Provides a clean instance of our test plugin named 'plugin2'."""
  return TestPlugin(name="plugin2")


def test_register_and_get_plugin(service: PluginManager, plugin1: TestPlugin):
  """Tests successful registration and retrieval of a plugin."""
  service.register_plugin(plugin1)

  assert len(service.plugins) == 1
  assert service.plugins[0] is plugin1
  assert service.get_plugin("plugin1") is plugin1


def test_register_duplicate_plugin_name_raises_value_error(
    service: PluginManager, plugin1: TestPlugin
):
  """Tests that registering a plugin with a duplicate name raises an error."""
  plugin1_duplicate = TestPlugin(name="plugin1")
  service.register_plugin(plugin1)

  with pytest.raises(
      ValueError, match="Plugin with name 'plugin1' already registered."
  ):
    service.register_plugin(plugin1_duplicate)


@pytest.mark.asyncio
async def test_early_exit_stops_subsequent_plugins(
    service: PluginManager, plugin1: TestPlugin, plugin2: TestPlugin
):
  """Tests the core "early exit" logic: if a plugin returns a value,

  subsequent plugins for that callback should not be executed.
  """
  # Configure plugin1 to return a value, simulating a cache hit.
  mock_response = Mock(spec=LlmResponse)
  plugin1.return_values["before_run_callback"] = mock_response

  service.register_plugin(plugin1)
  service.register_plugin(plugin2)

  # Execute the callback chain.
  result = await service.run_before_run_callback(invocation_context=Mock())

  # Assert that the final result is the one returned by the first plugin.
  assert result is mock_response
  # Assert that the first plugin was called.
  assert "before_run_callback" in plugin1.call_log
  # CRITICAL: Assert that the second plugin was never called.
  assert "before_run_callback" not in plugin2.call_log


@pytest.mark.asyncio
async def test_normal_flow_all_plugins_are_called(
    service: PluginManager, plugin1: TestPlugin, plugin2: TestPlugin
):
  """Tests that if no plugin returns a value, all plugins in the chain

  are executed in order.
  """
  # By default, plugins are configured to return None.
  service.register_plugin(plugin1)
  service.register_plugin(plugin2)

  result = await service.run_before_run_callback(invocation_context=Mock())

  # The final result should be None as no plugin interrupted the flow.
  assert result is None
  # Both plugins must have been called.
  assert "before_run_callback" in plugin1.call_log
  assert "before_run_callback" in plugin2.call_log


@pytest.mark.asyncio
async def test_plugin_exception_is_wrapped_in_runtime_error(
    service: PluginManager, plugin1: TestPlugin
):
  """Tests that if a plugin callback raises an exception, the PluginManager

  catches it and raises a descriptive RuntimeError.
  """
  # Configure the plugin to raise an error during a specific callback.
  original_exception = ValueError("Something went wrong inside the plugin!")
  plugin1.exceptions_to_raise["before_run_callback"] = original_exception
  service.register_plugin(plugin1)

  with pytest.raises(RuntimeError) as excinfo:
    await service.run_before_run_callback(invocation_context=Mock())

  # Check that the error message is informative.
  assert "Error in plugin 'plugin1'" in str(excinfo.value)
  assert "before_run_callback" in str(excinfo.value)
  # Check that the original exception is chained for better tracebacks.
  assert excinfo.value.__cause__ is original_exception


@pytest.mark.asyncio
async def test_all_callbacks_are_supported(
    service: PluginManager, plugin1: TestPlugin
):
  """Tests that all callbacks defined in the BasePlugin interface are supported

  by the PluginManager.
  """
  service.register_plugin(plugin1)
  mock_context = Mock()
  mock_user_message = Mock()

  # Test all callbacks
  await service.run_on_user_message_callback(
      user_message=mock_user_message, invocation_context=mock_context
  )
  await service.run_before_run_callback(invocation_context=mock_context)
  await service.run_after_run_callback(invocation_context=mock_context)
  await service.run_on_event_callback(
      invocation_context=mock_context, event=mock_context
  )
  await service.run_before_agent_callback(
      agent=mock_context, callback_context=mock_context
  )
  await service.run_after_agent_callback(
      agent=mock_context, callback_context=mock_context
  )
  await service.run_before_tool_callback(
      tool=mock_context, tool_args={}, tool_context=mock_context
  )
  await service.run_after_tool_callback(
      tool=mock_context, tool_args={}, tool_context=mock_context, result={}
  )
  await service.run_on_tool_error_callback(
      tool=mock_context,
      tool_args={},
      tool_context=mock_context,
      error=mock_context,
  )
  await service.run_before_model_callback(
      callback_context=mock_context, llm_request=mock_context
  )
  await service.run_after_model_callback(
      callback_context=mock_context, llm_response=mock_context
  )
  await service.run_on_model_error_callback(
      callback_context=mock_context,
      llm_request=mock_context,
      error=mock_context,
  )

  # Verify all callbacks were logged
  expected_callbacks = [
      "on_user_message_callback",
      "before_run_callback",
      "after_run_callback",
      "on_event_callback",
      "before_agent_callback",
      "after_agent_callback",
      "before_tool_callback",
      "after_tool_callback",
      "on_tool_error_callback",
      "before_model_callback",
      "after_model_callback",
      "on_model_error_callback",
  ]
  assert set(plugin1.call_log) == set(expected_callbacks)
