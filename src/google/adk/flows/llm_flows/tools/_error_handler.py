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

"""Error handling, failure detection, and error callback dispatch for tools."""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any
from typing import cast
from typing import Optional
from typing import TYPE_CHECKING

from ....tools.base_tool import BaseTool
from ....tools.tool_context import ToolContext
from ....utils._callback_pipeline import _invoke_callback
from ....utils._callback_pipeline import _run_callbacks
from ....utils._callback_pipeline import _stop_on_non_none

if TYPE_CHECKING:
  from ....agents.invocation_context import InvocationContext
  from ....agents.llm_agent import LlmAgent

logger = logging.getLogger('google_adk.' + __name__)


def detect_error_type_for_telemetry(
    tool: BaseTool,
    tool_context: ToolContext,
    function_response: Any,
) -> Optional[str]:
  """Detects an error type from a tool response for telemetry purposes.

  This does not modify the response. `_detect_error_in_response` is an optional
  per-tool hook accessed via `getattr` to avoid adding a public API on
  `BaseTool`. Any exception raised by the detector is logged and swallowed so
  that telemetry logic never breaks tool execution.

  Args:
    tool: The tool whose response is being inspected.
    tool_context: The tool context for the current invocation. Detection is
      skipped when the tool is requesting auth or confirmation.
    function_response: The raw response returned by the tool.

  Returns:
    The error type string reported by the tool's `_detect_error_in_response`
    hook, or `None` if no error was detected, no hook is defined, or the hook
    raised an exception.
  """
  try:
    if (
        tool_context.actions.requested_auth_configs
        or tool_context.actions.requested_tool_confirmations
    ):
      return None
    detector = getattr(tool, '_detect_error_in_response', None)
    if detector is None:
      return None
    return cast('str | None', detector(function_response))
  except Exception:  # pylint: disable=broad-exception-caught
    # Never let telemetry logic break tool execution.
    logger.exception(
        'Error while detecting error type for telemetry from tool %r.',
        getattr(tool, 'name', tool),
    )
    return None


def build_tool_not_found_response(
    tool_name: str, tools_dict: dict[str, BaseTool]
) -> dict[str, str]:
  """Returns the error payload for a tool name the model made up.

  A name the model invented is the model's own mistake to correct, so it is
  reported back to the model the way a malformed argument list already is,
  rather than raised out of the invocation.
  """
  available = ', '.join(tools_dict) or 'none'
  return {
      'error': (
          f'Invoking `{tool_name}()` failed as no tool with that name is'
          f' available. The tools you can call are: {available}. You could'
          ' retry, but it is IMPORTANT that you only call a tool from that'
          ' list.'
      )
  }


async def run_on_tool_error_callbacks(
    *,
    invocation_context: InvocationContext,
    agent: LlmAgent,
    tool: BaseTool,
    tool_args: dict[str, Any],
    tool_context: ToolContext,
    error: Exception,
) -> Optional[dict[str, Any]]:
  """Runs the on_tool_error_callbacks for the given tool."""
  error_response = (
      await invocation_context.plugin_manager.run_on_tool_error_callback(
          tool=tool,
          tool_args=tool_args,
          tool_context=tool_context,
          error=error,
      )
  )
  if error_response is not None:
    return error_response

  return await _run_callbacks(
      agent.canonical_on_tool_error_callbacks,  # type: ignore[arg-type]
      _stop_on_non_none,
      tool=tool,
      args=tool_args,
      tool_context=tool_context,
      error=error,
  )


async def notify_tool_cancelled(
    *,
    invocation_context: InvocationContext,
    agent: Optional[LlmAgent] = None,
    tool: BaseTool,
    tool_args: dict[str, Any],
    tool_context: ToolContext,
    cancelled: asyncio.CancelledError,
) -> None:
  """Runs the `on_tool_error` callbacks on tool call cancellation.

  `asyncio.CancelledError` is not an `Exception`, so a cancelled call skips the
  handler that reports a failed call to the error callbacks, and whatever a
  before-tool callback set up for the call is never released. This gives all
  callbacks the chance to perform teardown and cleanup.

  They run for cleanup only: a cancelled call is never recovered, so return
  values are ignored and all callbacks are executed regardless of responses.
  Exceptions or further cancellations raised by any callback are caught and
  logged so that remaining callbacks still run before the cancellation
  propagates.

  Args:
    invocation_context: The invocation the call belongs to.
    agent: The agent that owns the tool, or `None` when the tool runs inside a
      workflow tool node.
    tool: The tool whose call was cancelled.
    tool_args: The arguments the tool was called with.
    tool_context: The tool context of the call.
    cancelled: The cancellation that interrupted the call.
  """
  for plugin in invocation_context.plugin_manager.plugins:
    try:
      callback_method = getattr(plugin, 'on_tool_error_callback', None)
      if callback_method is not None:
        result = callback_method(
            tool=tool,
            tool_args=tool_args,
            tool_context=tool_context,
            error=cancelled,
        )
        if inspect.isawaitable(result):
          await result
    except asyncio.CancelledError:
      logger.warning(
          'Plugin %r on_tool_error_callback was cancelled while tool %r was'
          ' being cancelled; continuing remaining cleanup callbacks.',
          plugin.name,
          tool.name,
      )
    except Exception:  # pylint: disable=broad-exception-caught
      logger.exception(
          'Plugin %r on_tool_error_callback raised while tool %r was being'
          ' cancelled; the cancellation propagates regardless.',
          plugin.name,
          tool.name,
      )

  if agent is None:
    return

  for callback in agent.canonical_on_tool_error_callbacks:
    try:
      result = _invoke_callback(
          callback,
          tool=tool,
          args=tool_args,
          tool_context=tool_context,
          error=cancelled,
      )
      if inspect.isawaitable(result):
        await result
    except asyncio.CancelledError:
      logger.warning(
          'Agent %r on_tool_error callback was cancelled while tool %r was'
          ' being cancelled; continuing remaining cleanup callbacks.',
          agent.name,
          tool.name,
      )
    except Exception:  # pylint: disable=broad-exception-caught
      logger.exception(
          'Agent %r on_tool_error callback raised while tool %r was being'
          ' cancelled; the cancellation propagates regardless.',
          agent.name,
          tool.name,
      )
