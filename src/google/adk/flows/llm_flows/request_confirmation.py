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
from typing import cast
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
from .agent_transfer import _build_transfer_tool
from .agent_transfer import _get_transfer_targets
from .functions import REQUEST_CONFIRMATION_FUNCTION_CALL_NAME

if TYPE_CHECKING:
  from ...agents.llm_agent import LlmAgent


logger = logging.getLogger("google_adk." + __name__)


def _parse_tool_confirmation(response: dict[str, Any]) -> ToolConfirmation:
  """Parses ToolConfirmation from a function response dict."""
  return ToolConfirmation.from_response_dict(response)


def _get_original_function_call_args(
    function_call: types.FunctionCall,
) -> dict[str, Any] | None:
  """Returns the raw ``originalFunctionCall`` payload of a confirmation call.

  Both the dedup pre-pass and ``_resolve_confirmation_targets`` read the
  original function call out of an ``adk_request_confirmation`` call's args.
  They must agree on what counts as a well-formed payload, otherwise a
  confirmation could be skipped by one and processed by the other.

  Args:
    function_call: An ``adk_request_confirmation`` function call.

  Returns:
    The ``originalFunctionCall`` dict, or ``None`` if it is absent or malformed.
  """
  args = function_call.args
  if not args:
    return None
  original_function_call = args.get("originalFunctionCall")
  if not isinstance(original_function_call, dict):
    return None
  return original_function_call


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
  # IDs of function calls for which a tool dynamically requested confirmation.
  # This accumulates over ALL events rather than keeping one event per ID: once
  # the confirmed tool is re-executed it emits a second function response with
  # the same ID and no `requested_tool_confirmations`, which would otherwise
  # shadow the original request.
  dynamically_requested_fc_ids: set[str] = set()
  for ev in events:
    requested_tool_confirmations = ev.actions.requested_tool_confirmations or {}
    if not requested_tool_confirmations:
      continue
    for fr in ev.get_function_responses():
      if fr.id and fr.id in requested_tool_confirmations:
        dynamically_requested_fc_ids.add(fr.id)

  for event in events:
    event_function_calls = event.get_function_calls()
    if not event_function_calls:
      continue

    for function_call in event_function_calls:
      if not function_call.id or function_call.id not in confirmation_fc_ids:
        continue

      original_function_call_args = _get_original_function_call_args(
          function_call
      )
      if original_function_call_args is None:
        continue
      original_function_call = types.FunctionCall(**original_function_call_args)
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

      requested_in_history = (
          original_function_call.id in dynamically_requested_fc_ids
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


def _map_confirmation_to_original_fc_ids(
    events: list[Event],
    confirmation_fc_ids: set[str],
) -> dict[str, str]:
  """Maps each confirmation function call ID to its original function call ID.

  This is a cheap, validation-free pre-pass so that already-consumed
  confirmations can be dropped *before* the expensive and strict
  ``_resolve_confirmation_targets``.

  Args:
    events: Session events to scan.
    confirmation_fc_ids: IDs of ``adk_request_confirmation`` function calls.

  Returns:
    Mapping of confirmation FC ID -> original FC ID. Confirmations whose
    original function call cannot be determined are omitted.
  """
  mapping: dict[str, str] = {}
  for event in events:
    for function_call in event.get_function_calls():
      if not function_call.id or function_call.id not in confirmation_fc_ids:
        continue
      original_function_call_args = _get_original_function_call_args(
          function_call
      )
      # Mirror the `is None` check in `_resolve_confirmation_targets`: an empty
      # payload must reach the strict validation there and be rejected, not be
      # quietly dropped here (dropping it would skip the dedup and produce a
      # confusing downstream error instead).
      if original_function_call_args is None:
        continue
      original_fc_id = original_function_call_args.get("id")
      if original_fc_id:
        mapping[function_call.id] = original_fc_id
  return mapping


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

    # Step 2: Drop confirmations that have already been consumed.
    #
    # This must happen BEFORE resolving targets. The processor re-runs on every
    # LLM step of the invocation, and the approval stays the last user event for
    # the rest of the turn, so a confirmation the previous step already acted on
    # is seen again here. Re-validating consumed state is not just wasted work:
    # the session and the toolset have moved on since the approval, so the
    # strict checks in `_resolve_confirmation_targets` can now legitimately fail
    # and abort the invocation.
    confirmation_to_original_fc_id = _map_confirmation_to_original_fc_ids(
        events, set(confirmations_by_fc_id.keys())
    )
    responded_fc_ids: set[str] = set()
    for event in reversed(events):
      if event.author == "user":
        break
      for function_response in event.get_function_responses():
        if function_response.id:
          responded_fc_ids.add(function_response.id)

    confirmations_by_fc_id = {
        confirmation_fc_id: confirmation
        for confirmation_fc_id, confirmation in confirmations_by_fc_id.items()
        if confirmation_to_original_fc_id.get(confirmation_fc_id)
        not in responded_fc_ids
    }

    if not confirmations_by_fc_id:
      return

    # Resolve all canonical tools and build tools_dict. Deliberately after the
    # dedup above so a consumed confirmation does not force a toolset
    # resolution, which can be a remote call for e.g. MCP toolsets.
    tools_dict = {}
    if agent is not None and hasattr(agent, "canonical_tools"):
      tools_dict = {
          tool.name: tool
          for tool in await cast("LlmAgent", agent).canonical_tools(
              ReadonlyContext(invocation_context)
          )
      }

    from ...agents.llm_agent import LlmAgent

    if isinstance(agent, LlmAgent):
      transfer_targets = _get_transfer_targets(agent)
      if transfer_targets:
        transfer_tool = _build_transfer_tool(transfer_targets)
        tools_dict[transfer_tool.name] = transfer_tool

    # Step 3: Resolve confirmation targets using extracted helper.
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
