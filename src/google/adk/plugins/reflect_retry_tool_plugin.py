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

import asyncio
from enum import Enum
import json
from typing import Any
from typing import Optional

from pydantic import BaseModel

from ..tools.base_tool import BaseTool
from ..tools.tool_context import ToolContext
from ..utils.feature_decorator import experimental
from .base_plugin import BasePlugin

REFLECT_AND_RETRY_RESPONSE_TYPE = "ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN"
GLOBAL_SCOPE_KEY = "__global_reflect_and_retry_scope__"

# A mapping from a tool's name to its consecutive failure count.
PerToolFailuresCounter = dict[str, int]


class TrackingScope(Enum):
  """Defines the lifecycle scope for tracking tool failure counts."""

  INVOCATION = "invocation"
  GLOBAL = "global"


class ToolFailureResponse(BaseModel):
  """Response containing tool failure details and retry guidance."""

  response_type: str = REFLECT_AND_RETRY_RESPONSE_TYPE
  error_type: str = ""
  error_details: str = ""
  retry_count: int = 0
  reflection_guidance: str = ""


@experimental
class ReflectAndRetryToolPlugin(BasePlugin):
  """Provides self-healing, concurrent-safe error recovery for tool failures.

  This plugin intercepts tool failures, provides structured guidance to the LLM
  for reflection and correction, and retries the operation up to a configurable
  limit.

  **Key Features:**

  - **Concurrency Safe:** Uses locking to safely handle parallel tool
  executions
  - **Configurable Scope:** Tracks failures per-invocation (default) or globally
    using the `TrackingScope` enum.
  - **Extensible Scoping:** The `_get_scope_key` method can be overridden to
    implement custom tracking logic (e.g., per-user or per-session).
  - **Granular Tracking:** Failure counts are tracked per-tool within the
    defined scope. A success with one tool resets its counter without affecting
    others.
  - **Custom Error Extraction:** Supports detecting errors in normal tool
  responses
  that
    don't throw exceptions, by overriding the `extract_error_from_result`
    method.

  **Example:**
  ```python
  from my_project.plugins import ReflectAndRetryToolPlugin, TrackingScope

  # Example 1: (MOST COMMON USAGE):
  # Track failures only within the current agent invocation (default).
  error_handling_plugin = ReflectAndRetryToolPlugin(max_retries=3)

  # Example 2:
  # Track failures globally across all turns and users.
  global_error_handling_plugin = ReflectAndRetryToolPlugin(max_retries=5,
  scope=TrackingScope.GLOBAL)

  # Example 3:
  # Retry on failures but do not throw exceptions.
  error_handling_plugin =
    ReflectAndRetryToolPlugin(max_retries=3,
    throw_exception_if_retry_exceeded=False)

  # Example 4:
  # Track failures in successful tool responses that contain errors.
  class CustomRetryPlugin(ReflectAndRetryToolPlugin):
    async def extract_error_from_result(self, *, tool, tool_args,tool_context,
    result):
      # Detect error based on response content
      if result.get('status') == 'error':
          return result
      return None  # No error detected
  error_handling_plugin = CustomRetryPlugin(max_retries=5)
  ```
  """

  def __init__(
      self,
      name: str = "reflect_retry_tool_plugin",
      max_retries: int = 3,
      throw_exception_if_retry_exceeded: bool = True,
      tracking_scope: TrackingScope = TrackingScope.INVOCATION,
  ):
    """Initializes the ReflectAndRetryToolPlugin.

    Args:
        name: Plugin instance identifier.
        max_retries: Maximum consecutive failures before giving up (0 = no
          retries).
        throw_exception_if_retry_exceeded: If True, raises the final exception
          when the retry limit is reached. If False, returns guidance instead.
        tracking_scope: Determines the lifecycle of the error tracking state.
          Defaults to `TrackingScope.INVOCATION` tracking per-invocation.
    """
    super().__init__(name)
    if max_retries < 0:
      raise ValueError("max_retries must be a non-negative integer.")
    self.max_retries = max_retries
    self.throw_exception_if_retry_exceeded = throw_exception_if_retry_exceeded
    self.scope = tracking_scope
    self._scoped_failure_counters: dict[str, PerToolFailuresCounter] = {}
    self._lock = asyncio.Lock()

  async def after_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      result: Any,
  ) -> Optional[dict[str, Any]]:
    """Handles successful tool calls or extracts and processes errors.

    Args:
      tool: The tool that was called.
      tool_args: The arguments passed to the tool.
      tool_context: The context of the tool call.
      result: The result of the tool call.

    Returns:
      An optional dictionary containing reflection guidance if an error is
      detected, or None if the tool call was successful or the
      response is already a reflection message.
    """
    if (
        isinstance(result, dict)
        and result.get("response_type") == REFLECT_AND_RETRY_RESPONSE_TYPE
    ):
      return None

    error = await self.extract_error_from_result(
        tool=tool, tool_args=tool_args, tool_context=tool_context, result=result
    )

    if error:
      return await self._handle_tool_error(tool, tool_args, tool_context, error)

    # On success, reset the failure count for this specific tool within
    # its scope.
    await self._reset_failures_for_tool(tool_context, tool.name)
    return None

  async def extract_error_from_result(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      result: Any,
  ) -> Optional[dict[str, Any]]:
    """Extracts an error from a successful tool result and triggers retry logic.

    This is useful when tool call finishes successfully but the result contains
    an error object like {"error": ...} that should be handled by the plugin.

    By overriding this method, you can trigger retry logic on these successful
    results that contain errors.

    Args:
      tool: The tool that was called.
      tool_args: The arguments passed to the tool.
      tool_context: The context of the tool call.
      result: The result of the tool call.

    Returns:
      The extracted error if any, or None if no error was detected.
    """
    return None

  async def on_tool_error_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      error: Exception,
  ) -> Optional[dict[str, Any]]:
    """Handles tool exceptions by providing reflection guidance.

    Args:
      tool: The tool that was called.
      tool_args: The arguments passed to the tool.
      tool_context: The context of the tool call.
      error: The exception raised by the tool.

    Returns:
      An optional dictionary containing reflection guidance for the error.
    """
    return await self._handle_tool_error(tool, tool_args, tool_context, error)

  async def _handle_tool_error(
      self,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      error: Any,
  ) -> Optional[dict[str, Any]]:
    """Central, thread-safe logic for processing tool errors.

    Args:
      tool: The tool that was called.
      tool_args: The arguments passed to the tool.
      tool_context: The context of the tool call.
      error: The error to be handled.

    Returns:
      An optional dictionary containing reflection guidance for the error.
    """
    if self.max_retries == 0:
      if self.throw_exception_if_retry_exceeded:
        raise error
      return self._get_tool_retry_exceed_msg(tool, error, tool_args)

    scope_key = self._get_scope_key(tool_context)
    async with self._lock:
      tool_failure_counter = self._scoped_failure_counters.setdefault(
          scope_key, {}
      )
      current_retries = tool_failure_counter.get(tool.name, 0) + 1
      tool_failure_counter[tool.name] = current_retries

      if current_retries <= self.max_retries:
        return self._create_tool_reflection_response(
            tool, tool_args, error, current_retries
        )

      # Max Retry exceeded
      if self.throw_exception_if_retry_exceeded:
        raise error
      else:
        return self._get_tool_retry_exceed_msg(tool, tool_args, error)

  def _get_scope_key(self, tool_context: ToolContext) -> str:
    """Returns a unique key for the state dictionary based on the scope.

    This method can be overridden in a subclass to implement custom scoping
    logic, for example, tracking failures on a per-user or per-session basis.
    """
    if self.scope is TrackingScope.INVOCATION:
      return tool_context.invocation_id
    elif self.scope is TrackingScope.GLOBAL:
      return GLOBAL_SCOPE_KEY
    raise ValueError(f"Unknown scope: {self.scope}")

  async def _reset_failures_for_tool(
      self, tool_context: ToolContext, tool_name: str
  ) -> None:
    """Atomically resets the failure count for a tool and cleans up state."""
    scope = self._get_scope_key(tool_context)
    async with self._lock:
      if scope in self._scoped_failure_counters:
        state = self._scoped_failure_counters[scope]
        state.pop(tool_name, None)

  def _ensure_exception(self, error: Any) -> Exception:
    """Ensures the given error is an Exception instance, wrapping if not."""
    return error if isinstance(error, Exception) else Exception(str(error))

  def _format_error_details(self, error: Any) -> str:
    """Formats error details for inclusion in the reflection message."""
    if isinstance(error, Exception):
      return f"{type(error).__name__}: {str(error)}"
    return str(error)

  def _create_tool_reflection_response(
      self,
      tool: BaseTool,
      tool_args: dict[str, Any],
      error: Any,
      retry_count: int,
  ) -> dict[str, Any]:
    """Generates structured reflection guidance for tool failures."""
    args_summary = json.dumps(tool_args, indent=2, default=str)
    error_details = self._format_error_details(error)

    reflection_message = f"""
The call to tool `{tool.name}` failed.

**Error Details:**
```
{error_details}
```

**Tool Arguments Used:**
```json
{args_summary}
```

**Reflection Guidance:**
This is retry attempt **{retry_count} of {self.max_retries}**. Analyze the error and the arguments you provided. Do not repeat the exact same call. Consider the following before your next attempt:

1.  **Invalid Parameters**: Does the error suggest that one or more arguments are incorrect, badly formatted, or missing? Review the tool's schema and your arguments.
2.  **State or Preconditions**: Did a previous step fail or not produce the necessary state/resource for this tool to succeed?
3.  **Alternative Approach**: Is this the right tool for the job? Could another tool or a different sequence of steps achieve the goal?
4.  **Simplify the Task**: Can you break the problem down into smaller, simpler steps?
5.  **Wrong Function Name**: Does the error indicates the tool is not found? Please check again and only use available tools.

Formulate a new plan based on your analysis and try a corrected or different approach.
"""

    return ToolFailureResponse(
        error_type=(
            type(error).__name__
            if isinstance(error, Exception)
            else "ToolError"
        ),
        error_details=str(error),
        retry_count=retry_count,
        reflection_guidance=reflection_message.strip(),
    ).model_dump(mode="json")

  def _get_tool_retry_exceed_msg(
      self,
      tool: BaseTool,
      tool_args: dict[str, Any],
      error: Exception,
  ) -> dict[str, Any]:
    """Generates guidance when the maximum retry limit is exceeded."""
    error_details = self._format_error_details(error)
    args_summary = json.dumps(tool_args, indent=2, default=str)

    reflection_message = f"""
The tool `{tool.name}` has failed consecutively {self.max_retries} times and the retry limit has been exceeded.

**Last Error:**
```
{error_details}
```

**Last Arguments Used:**
```json
{args_summary}
```

**Final Instruction:**
**Do not attempt to use the `{tool.name}` tool again for this task.** You must now try a different approach. Acknowledge the failure and devise a new strategy, potentially using other available tools or informing the user that the task cannot be completed.
"""

    return ToolFailureResponse(
        error_type=(
            type(error).__name__
            if isinstance(error, Exception)
            else "ToolError"
        ),
        error_details=str(error),
        retry_count=self.max_retries,
        reflection_guidance=reflection_message.strip(),
    ).model_dump(mode="json")
