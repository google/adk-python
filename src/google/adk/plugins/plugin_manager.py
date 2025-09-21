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

from __future__ import annotations

import logging
from typing import Any
from typing import List
from typing import Literal
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types

from .base_plugin import BasePlugin

if TYPE_CHECKING:
  from ..agents.base_agent import BaseAgent
  from ..agents.callback_context import CallbackContext
  from ..agents.invocation_context import InvocationContext
  from ..events.event import Event
  from ..models.llm_request import LlmRequest
  from ..models.llm_response import LlmResponse
  from ..tools.base_tool import BaseTool
  from ..tools.tool_context import ToolContext

# A type alias for the names of the available plugin callbacks.
# This helps with static analysis and prevents typos when calling run_callbacks.
PluginCallbackName = Literal[
    "on_user_message_callback",
    "before_run_callback",
    "after_run_callback",
    "on_event_callback",
    "before_agent_callback",
    "after_agent_callback",
    "before_tool_callback",
    "after_tool_callback",
    "before_model_callback",
    "after_model_callback",
    "on_tool_error_callback",
    "on_model_error_callback",
]

logger = logging.getLogger("google_adk." + __name__)


class PluginManager:
  """Manages the registration and execution of plugins.

  The PluginManager is an internal class that orchestrates the invocation of
  plugin callbacks at key points in the SDK's execution lifecycle. It maintains
  a list of registered plugins and ensures they are called in the order they
  were registered.

  The core execution logic implements an "early exit" strategy: if any plugin
  callback returns a non-`None` value, the execution of subsequent plugins for
  that specific event is halted, and the returned value is propagated up the
  call stack. This allows plugins to short-circuit operations like agent runs,
  tool calls, or model requests.
  """

  def __init__(self, plugins: Optional[List[BasePlugin]] = None):
    """Initializes the plugin service.

    Args:
      plugins: An optional list of plugins to register upon initialization.
    """
    self.plugins: List[BasePlugin] = []
    if plugins:
      for plugin in plugins:
        self.register_plugin(plugin)

  def register_plugin(self, plugin: BasePlugin) -> None:
    """Registers a new plugin.

    Args:
      plugin: The plugin instance to register.

    Raises:
      ValueError: If a plugin with the same name is already registered.
    """
    if any(p.name == plugin.name for p in self.plugins):
      raise ValueError(f"Plugin with name '{plugin.name}' already registered.")
    self.plugins.append(plugin)
    logger.info("Plugin '%s' registered.", plugin.name)

  def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
    """Retrieves a registered plugin by its name.

    Args:
      plugin_name: The name of the plugin to retrieve.

    Returns:
      The plugin instance if found, otherwise `None`.
    """
    return next((p for p in self.plugins if p.name == plugin_name), None)

  async def run_on_user_message_callback(
      self,
      *,
      user_message: types.Content,
      invocation_context: InvocationContext,
  ) -> Optional[types.Content]:
    """Runs the `on_user_message_callback` for all plugins."""
    return await self._run_callbacks(
        "on_user_message_callback",
        user_message=user_message,
        invocation_context=invocation_context,
    )

  async def run_before_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> Optional[types.Content]:
    """Runs the `before_run_callback` for all plugins."""
    return await self._run_callbacks(
        "before_run_callback", invocation_context=invocation_context
    )

  async def run_after_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> Optional[None]:
    """Runs the `after_run_callback` for all plugins."""
    return await self._run_callbacks(
        "after_run_callback", invocation_context=invocation_context
    )

  async def run_on_event_callback(
      self, *, invocation_context: InvocationContext, event: Event
  ) -> Optional[Event]:
    """Runs the `on_event_callback` for all plugins."""
    return await self._run_callbacks(
        "on_event_callback",
        invocation_context=invocation_context,
        event=event,
    )

  async def run_before_agent_callback(
      self, *, agent: BaseAgent, callback_context: CallbackContext
  ) -> Optional[types.Content]:
    """Runs the `before_agent_callback` for all plugins."""
    return await self._run_callbacks(
        "before_agent_callback",
        agent=agent,
        callback_context=callback_context,
    )

  async def run_after_agent_callback(
      self, *, agent: BaseAgent, callback_context: CallbackContext
  ) -> Optional[types.Content]:
    """Runs the `after_agent_callback` for all plugins."""
    return await self._run_callbacks(
        "after_agent_callback",
        agent=agent,
        callback_context=callback_context,
    )

  async def run_before_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Optional[dict]:
    """Runs the `before_tool_callback` for all plugins."""
    return await self._run_callbacks(
        "before_tool_callback",
        tool=tool,
        tool_args=tool_args,
        tool_context=tool_context,
    )

  async def run_after_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      result: dict,
  ) -> Optional[dict]:
    """Runs the `after_tool_callback` for all plugins."""
    return await self._run_callbacks(
        "after_tool_callback",
        tool=tool,
        tool_args=tool_args,
        tool_context=tool_context,
        result=result,
    )

  async def run_on_model_error_callback(
      self,
      *,
      callback_context: CallbackContext,
      llm_request: LlmRequest,
      error: Exception,
  ) -> Optional[LlmResponse]:
    """Runs the `on_model_error_callback` for all plugins."""
    return await self._run_callbacks(
        "on_model_error_callback",
        callback_context=callback_context,
        llm_request=llm_request,
        error=error,
    )

  async def run_before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> Optional[LlmResponse]:
    """Runs the `before_model_callback` for all plugins."""
    return await self._run_callbacks(
        "before_model_callback",
        callback_context=callback_context,
        llm_request=llm_request,
    )

  async def run_after_model_callback(
      self, *, callback_context: CallbackContext, llm_response: LlmResponse
  ) -> Optional[LlmResponse]:
    """Runs the `after_model_callback` for all plugins."""
    return await self._run_callbacks(
        "after_model_callback",
        callback_context=callback_context,
        llm_response=llm_response,
    )

  async def run_on_tool_error_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      error: Exception,
  ) -> Optional[dict]:
    """Runs the `on_tool_error_callback` for all plugins."""
    return await self._run_callbacks(
        "on_tool_error_callback",
        tool=tool,
        tool_args=tool_args,
        tool_context=tool_context,
        error=error,
    )

  async def _run_callbacks(
      self, callback_name: PluginCallbackName, **kwargs: Any
  ) -> Optional[Any]:
    """Executes a specific callback for all registered plugins.

    This private method iterates through the plugins and calls the specified
    callback method on each one, passing the provided keyword arguments.

    The execution stops as soon as a plugin's callback returns a non-`None`
    value. This "early exit" value is then returned by this method. If all
    plugins are executed and all return `None`, this method also returns `None`.

    Args:
      callback_name: The name of the callback method to execute.
      **kwargs: Keyword arguments to be passed to the callback method.

    Returns:
      The first non-`None` value returned by a plugin callback, or `None` if
      all callbacks return `None`.

    Raises:
      RuntimeError: If a plugin encounters an unhandled exception during
        execution. The original exception is chained.
    """
    for plugin in self.plugins:
      # Each plugin might not implement all callbacks. The base class provides
      # default `pass` implementations, so `getattr` will always succeed.
      callback_method = getattr(plugin, callback_name)
      try:
        result = await callback_method(**kwargs)
        if result is not None:
          # Early exit: A plugin has returned a value. We stop
          # processing further plugins and return this value immediately.
          logger.debug(
              "Plugin '%s' returned a value for callback '%s', exiting early.",
              plugin.name,
              callback_name,
          )
          return result
      except Exception as e:
        error_message = (
            f"Error in plugin '{plugin.name}' during '{callback_name}'"
            f" callback: {e}"
        )
        logger.error(error_message, exc_info=True)
        raise RuntimeError(error_message) from e

    return None
