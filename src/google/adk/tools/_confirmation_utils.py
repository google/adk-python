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

"""Helpers for asking the user to confirm a tool call."""

from __future__ import annotations

from typing import Any

from .base_tool import BaseTool
from .tool_context import ToolContext


async def apply_confirmation_gate(
    tool: BaseTool,
    function_args: dict[str, Any],
    tool_context: ToolContext,
) -> dict[str, str] | None:
  """Answers a call whose tool is waiting on a human, instead of running it.

  A call that still needs confirmation records the request on
  `tool_context.actions.requested_tool_confirmations`, the same way a tool
  that calls `tool_context.request_confirmation()` itself does.

  Args:
    tool: The tool the call names.
    function_args: The arguments the call carries.
    tool_context: The context the call will run in.

  Returns:
    The response to answer the call with, or None if the call may proceed.

  Raises:
    ValueError: If `tool_context.function_call_id` is missing or empty when
      requesting confirmation.
  """
  requires_confirmation = await tool.check_require_confirmation(
      function_args, tool_context
  )
  # Holding a call back from the model is restrictive, and the hook is declared
  # to answer with a bool, so anything other than True lets the call through.
  if requires_confirmation is not True:
    return None

  confirmation = tool_context.tool_confirmation
  if confirmation is None:
    tool_context.request_confirmation(
        hint=(
            f'Please approve or reject the tool call {tool.name}() by'
            ' responding with a FunctionResponse with an expected'
            ' ToolConfirmation payload.'
        ),
    )
    return {
        'error': (
            'This tool call requires confirmation, please approve or reject.'
        )
    }
  if not confirmation.confirmed:
    return {'error': 'This tool call is rejected.'}
  return None
