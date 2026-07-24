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
from typing import AsyncGenerator
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from . import functions
from ...agents.invocation_context import InvocationContext
from ...agents.readonly_context import ReadonlyContext
from ...events.event import Event
from ...models.llm_request import LlmRequest
from ...tools.base_tool import BaseTool
from ...tools.tool_confirmation import ToolConfirmation
from ...tools.tool_context import ToolContext
from ._base_llm_processor import BaseLlmRequestProcessor
from .functions import REQUEST_CONFIRMATION_FUNCTION_CALL_NAME

if TYPE_CHECKING:
  pass


logger = logging.getLogger("google_adk." + __name__)


def _parse_tool_confirmation(response: dict[str, Any]) -> ToolConfirmation:
  """Parses ToolConfirmation from a function response dict."""
  return ToolConfirmation.from_response_dict(response)


async def _resolve_confirmation_targets(
    invocation_context: InvocationContext,
    events: list[Event],
    confirmation_fc_ids: set[str],
    confirmations_by_fc_id: dict[str, ToolConfirmation],
    tools_dict: dict[str, BaseTool],
) -> tuple[dict[str, ToolConfirmation], dict[str, types.FunctionCall]]:
  """Find original function calls for confirmed tools and validate them.

  Scans events for ``adk_request_confirmation`` function calls whose IDs
  are in *confirmation_fc_ids*, extracts the ``originalFunctionCall`` from
  their args, validates that they are registered, actually require confirmation,
  and match the original function calls in history, and maps each confirmation
  to the original FC ID.

  Args:
    invocation_context: Current invocation context.
    events: Session events to scan.
    confirmation_fc_ids: IDs of ``adk_request_confirmation`` function calls.
    confirmations_by_fc_id: Mapping of confirmation FC ID ->
      ``ToolConfirmation``.
    tools_dict: Dictionary of registered tools.

  Returns:
    Tuple of ``(tool_confirmation_dict, original_fcs_dict)`` where both
    are keyed by the ORIGINAL function call IDs.

  Raises:
    ValueError: If validation of any confirmation target fails.
  """
  tool_confirmation_dict: dict[str, ToolConfirmation] = {}
  original_fcs_dict: dict[str, types.FunctionCall] = {}

  history_fcs = {
      fc.id: (fc, ev)
      for ev in events
      for fc in ev.get_function_calls()
      if fc.id and fc.name != REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
  }
  history_fr_events = {
      fr.id: ev for ev in events for fr in ev.get_function_responses() if fr.id
  }

  for event in events:
    event_function_calls = event.get_function_calls()
    if not event_function_calls:
      continue

    for function_call in event_function_calls:
      if not function_call.id or function_call.id not in confirmation_fc_ids:
        continue

      args = function_call.args
      if not args or "originalFunctionCall" not in args:
        continue
      original_function_call = types.FunctionCall(
          **args["originalFunctionCall"]
      )
      if not original_function_call.id:
        raise ValueError("Original function call ID is missing.")
      tool_name = original_function_call.name
      if not tool_name:
        raise ValueError("Original function call name is missing.")

      # Check 1: Is the tool registered?
      original_fc_info = history_fcs.get(original_function_call.id)
      if not original_fc_info:
        raise ValueError(
            f"Original function call for ID '{original_function_call.id}' not"
            " found in session history."
        )
      original_fc_in_history, original_fc_event = original_fc_info

      # If this tool call was authored by another agent, skip it to let that
      # agent's processor handle it.
      agent = invocation_context.agent
      if agent and original_fc_event.author != agent.name:
        continue

      tool = tools_dict.get(tool_name)
      if not tool:
        raise ValueError(
            f"Tool '{original_function_call.name}' is not registered."
        )

      # Check 2: Does the tool require confirmation for these arguments?
      # We check if it is either statically required, or if it was dynamically
      # requested in the session history.
      temp_tool_context = ToolContext(
          invocation_context=invocation_context,
          function_call_id=original_function_call.id,
      )
      requires_confirmation = await tool.check_require_confirmation(
          original_function_call.args or {}, temp_tool_context
      )

      requested_in_history = False
      if not requires_confirmation:
        # Search the history for the response event of the original tool call
        original_response_event = history_fr_events.get(
            original_function_call.id
        )
        if (
            original_response_event
            and original_response_event.actions.requested_tool_confirmations
        ):
          requested_in_history = (
              original_function_call.id
              in original_response_event.actions.requested_tool_confirmations
          )

      if not requires_confirmation and not requested_in_history:
        raise ValueError(
            f"Tool '{original_function_call.name}' does not require"
            " confirmation."
        )

      # Check 3: Does the original function call match name and arguments?
      if original_fc_in_history.name != original_function_call.name:
        raise ValueError(
            f"Function call name mismatch for ID '{original_function_call.id}':"
            f" history has '{original_fc_in_history.name}', confirmation has"
            f" '{original_function_call.name}'."
        )

      hist_args = original_fc_in_history.args or {}
      conf_args = original_function_call.args or {}
      if hist_args != conf_args:
        raise ValueError(
            "Function call arguments mismatch for ID"
            f" '{original_function_call.id}'."
        )

      tool_confirmation_dict[original_function_call.id] = (
          confirmations_by_fc_id[function_call.id]
      )
      original_fcs_dict[original_function_call.id] = original_function_call

  return tool_confirmation_dict, original_fcs_dict


class _RequestConfirmationLlmRequestProcessor(BaseLlmRequestProcessor):
  """Handles tool confirmation information to build the LLM request."""

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:

    agent = invocation_context.agent

    # Only look at events in the current branch.
    events = invocation_context._get_events(current_branch=True)
    if not events:
      return

    # Step 1: Find the last user-authored event and parse confirmation
    # responses from it.
    confirmations_by_fc_id: dict[str, ToolConfirmation] = {}
    for k in range(len(events) - 1, -1, -1):
      event = events[k]
      if not event.author or event.author != "user":
        continue
      responses = event.get_function_responses()
      if not responses:
        return

      for function_response in responses:
        if function_response.name != REQUEST_CONFIRMATION_FUNCTION_CALL_NAME:
          continue
        if not function_response.id or function_response.response is None:
          continue
        confirmations_by_fc_id[function_response.id] = _parse_tool_confirmation(
            function_response.response
        )
      break

    if not confirmations_by_fc_id:
      return

    # Resolve all canonical tools and build tools_dict
    tools_dict = {}
    if agent is not None and hasattr(agent, "canonical_tools"):
      tools_dict = {
          tool.name: tool
          for tool in await agent.canonical_tools(
              ReadonlyContext(invocation_context)
          )
      }

    # Step 2: Resolve confirmation targets using extracted helper.
    confirmation_fc_ids = set(confirmations_by_fc_id.keys())
    tools_to_resume_with_confirmation, tools_to_resume_with_args = (
        await _resolve_confirmation_targets(
            invocation_context,
            events,
            confirmation_fc_ids,
            confirmations_by_fc_id,
            tools_dict,
        )
    )

    if not tools_to_resume_with_confirmation:
      return

    # Step 3: Remove tools that have already been confirmed (dedup).
    for event in reversed(events):
      if event.author == "user":
        break
      fr_list = event.get_function_responses()
      if not fr_list:
        continue

      for function_response in fr_list:
        if function_response.id in tools_to_resume_with_confirmation:
          tools_to_resume_with_confirmation.pop(function_response.id)
          tools_to_resume_with_args.pop(function_response.id)
      if not tools_to_resume_with_confirmation:
        break

    if not tools_to_resume_with_confirmation:
      return

    # Step 4: Re-execute the confirmed tools.
    if function_response_event := await functions.handle_function_call_list_async(
        invocation_context,
        list(tools_to_resume_with_args.values()),
        tools_dict,
        set(tools_to_resume_with_confirmation.keys()),
        tools_to_resume_with_confirmation,
    ):
      yield function_response_event
    return


request_processor = _RequestConfirmationLlmRequestProcessor()
