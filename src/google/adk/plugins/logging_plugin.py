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

import logging
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from ..agents.base_agent import BaseAgent
from ..agents.callback_context import CallbackContext
from ..events.event import Event
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..tools.base_tool import BaseTool
from ..tools.tool_context import ToolContext
from ._callback_helpers import _classify_tool_result_error
from ._callback_helpers import _is_durable_node_error
from ._callback_helpers import _mark_tool_error_recorded
from ._callback_helpers import _pop_tool_error_recorded
from ._callback_helpers import _validate_tool_result_classifier
from ._callback_helpers import ToolResultClassifier
from .base_plugin import BasePlugin

if TYPE_CHECKING:
  from ..agents.invocation_context import InvocationContext

logger = logging.getLogger("google_adk." + __name__)


class LoggingPlugin(BasePlugin):
  """A plugin that logs important information at each callback point.

  This plugin helps print all critical events in the console. It is not a
  replacement of existing logging in ADK. It rather helps terminal based
  debugging by showing all logs in the console, and serves as a simple demo for
  everyone to leverage when developing new plugins.

  This plugin helps users track the invocation status by logging:
  - User messages and invocation context
  - Agent execution flow
  - LLM requests and responses, with the finish reason of a final response
  - Tool calls with arguments and results
  - Events and final responses, with the path, output and failure of the
    workflow node that emitted them
  - Errors during model and tool execution

  A tool call is logged as a ``TOOL ERROR`` when the tool raises, and also
  when its result reports a failure without raising: a
  ``ReflectAndRetryToolPlugin`` answer, an MCP ``CallToolResult`` with
  ``isError`` set, or a result that ``tool_result_classifier`` classifies as
  an error. A raised error is logged once, even when a later plugin's answer
  to it reaches ``after_tool_callback`` as the call's result.

  Example:
      >>> logging_plugin = LoggingPlugin()
      >>> runner = Runner(
      ...     agents=[my_agent],
      ...     # ...
      ...     plugins=[logging_plugin],
      ... )
  """

  def __init__(
      self,
      name: str = "logging_plugin",
      *,
      tool_result_classifier: Optional[ToolResultClassifier] = None,
  ):
    """Initialize the logging plugin.

    Args:
      name: The name of the plugin instance.
      tool_result_classifier: Classifies the results of the application's own
        tools as a tool error or a completed call; see
        ``ToolResultClassifier``. One that raises or returns an unusable
        verdict at run time is skipped with a warning that names only the
        tool, and the built-in rule applies.

    Raises:
      ValueError: If ``tool_result_classifier`` is not callable, is
        asynchronous, has no signature that ``inspect`` can read, or cannot
        take the keyword arguments ``tool``, ``tool_args``, ``tool_context``
        and ``result``.
    """
    super().__init__(name)
    _validate_tool_result_classifier(tool_result_classifier)
    self._tool_result_classifier = tool_result_classifier

  @override
  async def on_user_message_callback(
      self,
      *,
      invocation_context: InvocationContext,
      user_message: types.Content,
  ) -> Optional[types.Content]:
    """Log user message and invocation start."""
    self._log(f"🚀 USER MESSAGE RECEIVED")
    self._log(f"   Invocation ID: {invocation_context.invocation_id}")
    self._log(f"   Session ID: {invocation_context.session.id}")
    self._log(f"   User ID: {invocation_context.user_id}")
    self._log(f"   App Name: {invocation_context.app_name}")
    agent = invocation_context.agent
    self._log(
        "   Root Agent:"
        f" {agent.name if agent is not None and hasattr(agent, 'name') else 'Unknown'}"
    )
    self._log(f"   User Content: {self._format_content(user_message)}")
    if invocation_context.branch:
      self._log(f"   Branch: {invocation_context.branch}")
    return None

  @override
  async def before_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> Optional[types.Content]:
    """Log invocation start."""
    self._log(f"🏃 INVOCATION STARTING")
    self._log(f"   Invocation ID: {invocation_context.invocation_id}")
    agent = invocation_context.agent
    self._log(
        "   Starting Agent:"
        f" {agent.name if agent is not None and hasattr(agent, 'name') else 'Unknown'}"
    )
    return None

  @override
  async def on_event_callback(
      self, *, invocation_context: InvocationContext, event: Event
  ) -> Optional[Event]:
    """Log events yielded from the runner."""
    self._log(f"📢 EVENT YIELDED")
    self._log(f"   Event ID: {event.id}")
    self._log(f"   Author: {event.author}")
    self._log(f"   Content: {self._format_content(event.content)}")
    self._log(f"   Final Response: {event.is_final_response()}")

    if event.get_function_calls():
      func_calls = [fc.name for fc in event.get_function_calls()]
      self._log(f"   Function Calls: {func_calls}")

    if event.get_function_responses():
      func_responses = [fr.name for fr in event.get_function_responses()]
      self._log(f"   Function Responses: {func_responses}")

    if event.long_running_tool_ids:
      self._log(f"   Long Running Tools: {list(event.long_running_tool_ids)}")

    # The same events BigQueryAgentAnalyticsPlugin records as NODE_OUTPUT and
    # NODE_ERROR: a final event of a workflow node. An output its content
    # already carries is not repeated, and an error code that is a model
    # finish or block reason is a model termination, not a node failure.
    node_info = event.node_info
    if node_info.path:
      self._log(f"   Node Path: {node_info.path}")
      if event.partial is not True:
        if event.output is not None and node_info.message_as_output is not True:
          self._log(f"   Node Output: {self._format_value(event.output)}")
        if _is_durable_node_error(event):
          self._log(
              f"   Node Error: [{event.error_code}]"
              f" {event.error_message or ''}".rstrip()
          )

    return None

  @override
  async def after_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> Optional[None]:
    """Log invocation completion."""
    self._log(f"✅ INVOCATION COMPLETED")
    self._log(f"   Invocation ID: {invocation_context.invocation_id}")
    agent = invocation_context.agent
    self._log(
        "   Final Agent:"
        f" {agent.name if agent is not None and hasattr(agent, 'name') else 'Unknown'}"
    )
    return None

  @override
  async def before_agent_callback(
      self, *, agent: BaseAgent, callback_context: CallbackContext
  ) -> Optional[types.Content]:
    """Log agent execution start."""
    self._log(f"🤖 AGENT STARTING")
    self._log(f"   Agent Name: {callback_context.agent_name}")
    self._log(f"   Invocation ID: {callback_context.invocation_id}")
    if callback_context._invocation_context.branch:
      self._log(f"   Branch: {callback_context._invocation_context.branch}")
    return None

  @override
  async def after_agent_callback(
      self, *, agent: BaseAgent, callback_context: CallbackContext
  ) -> Optional[types.Content]:
    """Log agent execution completion."""
    self._log(f"🤖 AGENT COMPLETED")
    self._log(f"   Agent Name: {callback_context.agent_name}")
    self._log(f"   Invocation ID: {callback_context.invocation_id}")
    return None

  @override
  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> Optional[LlmResponse]:
    """Log LLM request before sending to model."""
    self._log(f"🧠 LLM REQUEST")
    self._log(f"   Model: {llm_request.model or 'default'}")
    self._log(f"   Agent: {callback_context.agent_name}")

    # Log system instruction if present
    system_instruction = llm_request.config.system_instruction
    if system_instruction:
      rendered_instruction = self._render_system_instruction(system_instruction)
      sys_instruction = rendered_instruction[:200]
      if len(rendered_instruction) > 200:
        sys_instruction += "..."
      self._log(f"   System Instruction: '{sys_instruction}'")

    # Note: Content logging removed due to type compatibility issues
    # Users can still see content in the LLM response

    # Log available tools
    if llm_request.tools_dict:
      tool_names = list(llm_request.tools_dict.keys())
      self._log(f"   Available Tools: {tool_names}")

    return None

  @override
  async def after_model_callback(
      self, *, callback_context: CallbackContext, llm_response: LlmResponse
  ) -> Optional[LlmResponse]:
    """Log LLM response after receiving from model.

    Only the final response of a model call logs why the model stopped, its
    error and finish reason: a streamed chunk can carry the same values, and
    logging them there too would log one termination twice.
    """
    self._log(f"🧠 LLM RESPONSE")
    self._log(f"   Agent: {callback_context.agent_name}")
    is_final = llm_response.partial is not True

    if is_final and llm_response.error_code:
      self._log(f"   ❌ ERROR - Code: {llm_response.error_code}")
      self._log(f"   Error Message: {llm_response.error_message}")
    else:
      self._log(f"   Content: {self._format_content(llm_response.content)}")
      if llm_response.partial:
        self._log(f"   Partial: {llm_response.partial}")
      if llm_response.turn_complete is not None:
        self._log(f"   Turn Complete: {llm_response.turn_complete}")

    if is_final and llm_response.finish_reason:
      finish_reason = llm_response.finish_reason
      self._log(
          f"   Finish Reason: {getattr(finish_reason, 'value', finish_reason)}"
      )

    # Log usage metadata if available
    if llm_response.usage_metadata:
      self._log(
          "   Token Usage - Input:"
          f" {llm_response.usage_metadata.prompt_token_count}, Output:"
          f" {llm_response.usage_metadata.candidates_token_count}"
      )

    return None

  @override
  async def before_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Optional[dict[str, Any]]:
    """Log tool execution start."""
    self._log(f"🔧 TOOL STARTING")
    self._log(f"   Tool Name: {tool.name}")
    self._log(f"   Agent: {tool_context.agent_name}")
    self._log(f"   Function Call ID: {tool_context.function_call_id}")
    self._log(f"   Arguments: {self._format_args(tool_args)}")
    return None

  @override
  async def after_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      result: dict[str, Any],
  ) -> Optional[dict[str, Any]]:
    """Log tool execution completion, or the failure its result reports.

    The answer a later plugin gave to an error this plugin's
    ``on_tool_error_callback`` already logged is not logged again. A result
    that reports a failure without raising is logged as a ``TOOL ERROR``.
    """
    if _pop_tool_error_recorded(self, tool_context):
      return None
    error_message = _classify_tool_result_error(
        tool=tool,
        tool_args=tool_args,
        tool_context=tool_context,
        result=result,
        tool_result_classifier=self._tool_result_classifier,
        log=logger,
    )
    if error_message is not None:
      self._log(f"🔧 TOOL ERROR")
      self._log(f"   Tool Name: {tool.name}")
      self._log(f"   Agent: {tool_context.agent_name}")
      self._log(f"   Function Call ID: {tool_context.function_call_id}")
      self._log(f"   Arguments: {self._format_args(tool_args)}")
      self._log(f"   Result: {self._format_args(result)}")
      self._log(f"   Error: {error_message}")
      return None
    self._log(f"🔧 TOOL COMPLETED")
    self._log(f"   Tool Name: {tool.name}")
    self._log(f"   Agent: {tool_context.agent_name}")
    self._log(f"   Function Call ID: {tool_context.function_call_id}")
    self._log(f"   Result: {self._format_args(result)}")
    return None

  @override
  async def on_model_error_callback(
      self,
      *,
      callback_context: CallbackContext,
      llm_request: LlmRequest,
      error: Exception,
  ) -> Optional[LlmResponse]:
    """Log LLM error."""
    self._log(f"🧠 LLM ERROR")
    self._log(f"   Agent: {callback_context.agent_name}")
    self._log(f"   Error: {error}")

    return None

  @override
  async def on_tool_error_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      error: Exception,
  ) -> Optional[dict[str, Any]]:
    """Log tool error."""
    lines = [
        "🔧 TOOL ERROR",
        f"   Tool Name: {tool.name}",
        f"   Agent: {tool_context.agent_name}",
        f"   Function Call ID: {tool_context.function_call_id}",
        f"   Arguments: {self._format_args(tool_args)}",
        f"   Error: {error}",
    ]
    # Marked once every line is rendered: after_tool_callback skips a marked
    # call, so failing to render a line must not leave the call marked without
    # its log lines.
    _mark_tool_error_recorded(self, tool_context)
    for line in lines:
      self._log(line)
    return None

  def _log(self, message: str) -> None:
    """Internal method to format and print log messages."""
    # ANSI color codes: \033[90m for grey, \033[0m to reset
    formatted_message: str = f"\033[90m[{self.name}] {message}\033[0m"
    print(formatted_message)

  def _render_system_instruction(self, system_instruction: Any) -> str:
    """Renders a system instruction for logging.

    The field accepts a string, a Content, a Part, a File, an image, or a list
    of those. Only the text-bearing shapes are worth rendering in a log line,
    so anything else falls back to its string form.
    """
    if isinstance(system_instruction, str):
      return system_instruction
    if isinstance(system_instruction, types.Content):
      return self._format_content(system_instruction)
    if isinstance(system_instruction, types.Part):
      return self._format_content(types.Content(parts=[system_instruction]))
    if isinstance(system_instruction, list):
      return " | ".join(
          self._render_system_instruction(entry) for entry in system_instruction
      )
    return str(system_instruction)

  def _format_content(
      self, content: Optional[types.Content], max_length: int = 200
  ) -> str:
    """Format content for logging, truncating if too long."""
    if not content or not content.parts:
      return "None"

    parts = []
    for part in content.parts:
      if part.text:
        text = part.text.strip()
        if len(text) > max_length:
          text = text[:max_length] + "..."
        parts.append(f"text: '{text}'")
      elif part.function_call:
        parts.append(f"function_call: {part.function_call.name}")
      elif part.function_response:
        parts.append(f"function_response: {part.function_response.name}")
      elif part.code_execution_result:
        parts.append("code_execution_result")
      else:
        parts.append("other_part")

    return " | ".join(parts)

  def _format_args(self, args: dict[str, Any], max_length: int = 300) -> str:
    """Format arguments dictionary for logging."""
    if not args:
      return "{}"

    formatted = str(args)
    if len(formatted) > max_length:
      formatted = formatted[:max_length] + "...}"
    return formatted

  def _format_value(self, value: Any, max_length: int = 300) -> str:
    """Format any value, such as a node output, for logging."""
    formatted = repr(value) if isinstance(value, str) else str(value)
    if len(formatted) > max_length:
      formatted = formatted[:max_length] + "..."
    return formatted
