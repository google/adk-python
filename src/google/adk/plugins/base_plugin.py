# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may in obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from abc import ABC
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING
from typing import TypeVar

from google.genai import types

from ..agents.base_agent import BaseAgent
from ..agents.callback_context import CallbackContext
from ..events.event import Event
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..tools.base_tool import BaseTool
from ..utils.feature_decorator import working_in_progress

if TYPE_CHECKING:
  from ..agents.invocation_context import InvocationContext
  from ..tools.tool_context import ToolContext


# Type alias: The value may or may not be awaitable, and value is optional.
T = TypeVar("T")


class BasePlugin(ABC):
  """Base class for creating plugins.

  Plugins provide a structured way to intercept and modify agent, tool, and
  LLM behaviors at critical execution points in a callback manner. While agent
  callbacks apply to a particular agent, plugins applies globally to all
  agents added in the runner. Plugins are best used for adding custom behaviors
  like logging, monitoring, caching, or modifying requests and responses at key
  stages.

  A plugin can implement one or more methods of callbacks, but should not
  implement the same method of callback for multiple times.

  Relation with [Agent callbacks](https://google.github.io/adk-docs/callbacks/):

  **Execution Order**
  Similar to Agent callbacks, Plugins are executed in the order they are
  registered. However, Plugin and Agent Callbacks are executed sequentially,
  with Plugins takes precedence over agent callbacks. When the callback in a
  plugin returns a value, it will short circuit all remaining plugins and
  agent callbacks, causing all remaining plugins and agent callbacks
  to be skipped.

  **Change Propagation**
  Plugins and agent callbacks can both modify the value of the input parameters,
  including agent input, tool input, and LLM request/response, etc. They work in
  the exactly same way. The modifications will be visible and passed to the next
  callback in the chain. For example, if a plugin modifies the tool input with
  before_tool_callback, the modified tool input will be passed to the
  before_tool_callback of the next plugin, and further passed to the agent
  callbacks if not short circuited.

  To use a plugin, implement the desired callback methods and pass an instance
  of your custom plugin class to the ADK Runner.

  Examples:
      A simple plugin that logs every tool call.

      >>> class ToolLoggerPlugin(BasePlugin):
      ..   def __init__(self):
      ..     super().__init__(name="tool_logger")
      ..
      ..   async def before_tool_callback(
      ..       self, *, tool: BaseTool, tool_args: dict[str, Any],
      tool_context:
      ToolContext
      ..   ):
      ..     print(f"[{self.name}] Calling tool '{tool.name}' with args:
      {tool_args}")
      ..
      ..   async def after_tool_callback(
      ..       self, *, tool: BaseTool, tool_args: dict, tool_context:
      ToolContext, result: dict
      ..   ):
      ..     print(f"[{self.name}] Tool '{tool.name}' finished with result:
      {result}")
      ..
      >>> # Add the plugin to ADK Runner
      >>> # runner = Runner(
      >>> #     ...
      >>> #     plugins=[ToolLoggerPlugin(), AgentPolicyPlugin()],
      >>> # )
  """

  def __init__(self, name: str):
    """Initializes the plugin.

    Args:
      name: A unique identifier for this plugin instance.
    """
    super().__init__()
    self.name = name

  async def on_user_message_callback(
      self,
      *,
      invocation_context: InvocationContext,
      user_message: types.Content,
  ) -> Optional[types.Content]:
    """Callback executed when a user message is received before an invocation starts.

    This callback helps logging and modifying the user message before the
    runner starts the invocation.

    Args:
      invocation_context: The context for the entire invocation.
      user_message: The message content input by user.

    Returns:
      An optional `types.Content` to be returned to the ADK. Returning a
      value to replace the user message. Returning `None` to proceed
      normally.
    """
    pass

  async def before_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> Optional[types.Content]:
    """Callback executed before the ADK runner runs.

    This is the first callback to be called in the lifecycle, ideal for global
    setup or initialization tasks.

    Args:
      invocation_context: The context for the entire invocation, containing
        session information, the root agent, etc.

    Returns:
      An optional `Event` to be returned to the ADK. Returning a value to
      halt execution of the runner and ends the runner with that event. Return
      `None` to proceed normally.
    """
    pass

  async def on_event_callback(
      self, *, invocation_context: InvocationContext, event: Event
  ) -> Optional[Event]:
    """Callback executed after an event is yielded from runner.

    This is the ideal place to make modification to the event before the event
    is handled by the underlying agent app.

    Args:
      invocation_context: The context for the entire invocation.
      event: The event raised by the runner.

    Returns:
      An optional value. A non-`None` return may be used by the framework to
      modify or replace the response. Returning `None` allows the original
      response to be used.
    """
    pass

  async def after_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> Optional[None]:
    """Callback executed after an ADK runner run has completed.

    This is the final callback in the ADK lifecycle, suitable for cleanup, final
    logging, or reporting tasks.

    Args:
      invocation_context: The context for the entire invocation.

    Returns:
      None
    """
    pass

  async def before_agent_callback(
      self, *, agent: BaseAgent, callback_context: CallbackContext
  ) -> Optional[types.Content]:
    """Callback executed before an agent's primary logic is invoked.

    This callback can be used for logging, setup, or to short-circuit the
    agent's execution by returning a value.

    Args:
      agent: The agent that is about to run.
      callback_context: The context for the agent invocation.

    Returns:
      An optional `types.Content` object. If a value is returned, it will bypass
      the agent's callbacks and its execution, and return this value directly.
      Returning `None` allows the agent to proceed normally.
    """
    pass

  async def after_agent_callback(
      self, *, agent: BaseAgent, callback_context: CallbackContext
  ) -> Optional[types.Content]:
    """Callback executed after an agent's primary logic has completed.

    This callback can be used to inspect, log, or modify the agent's final
    result before it is returned.

    Args:
      agent: The agent that has just run.
      callback_context: The context for the agent invocation.

    Returns:
      An optional `types.Content` object. If a value is returned, it will
      replace the agent's original result. Returning `None` uses the original,
      unmodified result.
    """
    pass

  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> Optional[LlmResponse]:
    """Callback executed before a request is sent to the model.

    This provides an opportunity to inspect, log, or modify the `LlmRequest`
    object. It can also be used to implement caching by returning a cached
    `LlmResponse`, which would skip the actual model call.

    Args:
      callback_context: The context for the current agent call.
      llm_request: The prepared request object to be sent to the model.

    Returns:
      An optional value. The interpretation of a non-`None` trigger an early
      exit and returns the response immediately. Returning `None` allows the LLM
      request to proceed normally.
    """
    pass

  async def after_model_callback(
      self, *, callback_context: CallbackContext, llm_response: LlmResponse
  ) -> Optional[LlmResponse]:
    """Callback executed after a response is received from the model.

    This is the ideal place to log model responses, collect metrics on token
    usage, or perform post-processing on the raw `LlmResponse`.

    Args:
      callback_context: The context for the current agent call.
      llm_response: The response object received from the model.

    Returns:
      An optional value. A non-`None` return may be used by the framework to
      modify or replace the response. Returning `None` allows the original
      response to be used.
    """
    pass

  async def on_model_error_callback(
      self,
      *,
      callback_context: CallbackContext,
      llm_request: LlmRequest,
      error: Exception,
  ) -> Optional[LlmResponse]:
    """Callback executed when a model call encounters an error.

    This callback provides an opportunity to handle model errors gracefully,
    potentially providing alternative responses or recovery mechanisms.

    Args:
      callback_context: The context for the current agent call.
      llm_request: The request that was sent to the model when the error
        occurred.
      error: The exception that was raised during model execution.

    Returns:
      An optional LlmResponse. If an LlmResponse is returned, it will be used
      instead of propagating the error. Returning `None` allows the original
      error to be raised.
    """
    pass

  async def before_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Optional[dict]:
    """Callback executed before a tool is called.

    This callback is useful for logging tool usage, input validation, or
    modifying the arguments before they are passed to the tool.

    Args:
      tool: The tool instance that is about to be executed.
      tool_args: The dictionary of arguments to be used for invoking the tool.
      tool_context: The context specific to the tool execution.

    Returns:
      An optional dictionary. If a dictionary is returned, it will stop the tool
      execution and return this response immediately. Returning `None` uses the
      original, unmodified arguments.
    """
    pass

  async def after_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      result: dict,
  ) -> Optional[dict]:
    """Callback executed after a tool has been called.

    This callback allows for inspecting, logging, or modifying the result
    returned by a tool.

    Args:
      tool: The tool instance that has just been executed.
      tool_args: The original arguments that were passed to the tool.
      tool_context: The context specific to the tool execution.
      result: The dictionary returned by the tool invocation.

    Returns:
      An optional dictionary. If a dictionary is returned, it will **replace**
      the original result from the tool. This allows for post-processing or
      altering tool outputs. Returning `None` uses the original, unmodified
      result.
    """
    pass

  async def on_tool_error_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      error: Exception,
  ) -> Optional[dict]:
    """Callback executed when a tool call encounters an error.

    This callback provides an opportunity to handle tool errors gracefully,
    potentially providing alternative responses or recovery mechanisms.

    Args:
      tool: The tool instance that encountered an error.
      tool_args: The arguments that were passed to the tool.
      tool_context: The context specific to the tool execution.
      error: The exception that was raised during tool execution.

    Returns:
      An optional dictionary. If a dictionary is returned, it will be used as
      the tool response instead of propagating the error. Returning `None`
      allows the original error to be raised.
    """
    pass
