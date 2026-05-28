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

import json
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
from ...tools.tool_confirmation import ToolConfirmation
from ._base_llm_processor import BaseLlmRequestProcessor
from .functions import REQUEST_CONFIRMATION_FUNCTION_CALL_NAME

if TYPE_CHECKING:
  from ...agents.llm_agent import LlmAgent


logger = logging.getLogger('google_adk.' + __name__)


def _parse_tool_confirmation(response: dict[str, Any]) -> ToolConfirmation:
  """Parse ToolConfirmation from a function response dict.

  Handles both the direct dict format and the ADK client's
  ``{'response': json_string}`` wrapper format.

  """
  if response and len(response.values()) == 1 and 'response' in response.keys():
    return ToolConfirmation.model_validate(json.loads(response['response']))
  return ToolConfirmation.model_validate(response)


def _resolve_confirmation_targets(
    events: list[Event],
    confirmation_fc_ids: set[str],
    confirmations_by_fc_id: dict[str, ToolConfirmation],
) -> tuple[dict[str, ToolConfirmation], dict[str, types.FunctionCall]]:
  """Find original function calls for confirmed tools.

  Scans events for ``adk_request_confirmation`` function calls whose IDs
  are in *confirmation_fc_ids*, extracts the ``originalFunctionCall`` from
  their args, and maps each confirmation to the original FC ID.

  Args:
    events: Session events to scan.
    confirmation_fc_ids: IDs of ``adk_request_confirmation`` function calls.
    confirmations_by_fc_id: Mapping of confirmation FC ID ->
      ``ToolConfirmation``.

  Returns:
    Tuple of ``(tool_confirmation_dict, original_fcs_dict)`` where both
    are keyed by the ORIGINAL function call IDs.
  """
  tool_confirmation_dict: dict[str, ToolConfirmation] = {}
  original_fcs_dict: dict[str, types.FunctionCall] = {}

  for event in events:
    event_function_calls = event.get_function_calls()
    if not event_function_calls:
      continue

    for function_call in event_function_calls:
      if function_call.id not in confirmation_fc_ids:
        continue

      args = function_call.args
      if 'originalFunctionCall' not in args:
        continue
      original_function_call = types.FunctionCall(
          **args['originalFunctionCall']
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
    from ...agents.llm_agent import LlmAgent

    agent = invocation_context.agent

    # Only look at events in the current branch.
    events = invocation_context._get_events(current_branch=True)
    if not events:
      return

    # Step 1: Find the last user-authored event and parse confirmation
    # responses from it.
    confirmations_by_fc_id: dict[str, ToolConfirmation] = {}
    confirmation_event_index = -1
    for k in range(len(events) - 1, -1, -1):
      event = events[k]
      if not event.author or event.author != 'user':
        continue
      responses = event.get_function_responses()
      if not responses:
        return

      for function_response in responses:
        if function_response.name != REQUEST_CONFIRMATION_FUNCTION_CALL_NAME:
          continue
        confirmations_by_fc_id[function_response.id] = _parse_tool_confirmation(
            function_response.response
        )
      confirmation_event_index = k
      break

    if not confirmations_by_fc_id:
      return

    # Step 2: Resolve confirmation targets using extracted helper.
    confirmation_fc_ids = set(confirmations_by_fc_id.keys())
    tools_to_resume_with_confirmation, tools_to_resume_with_args = (
        _resolve_confirmation_targets(
            events, confirmation_fc_ids, confirmations_by_fc_id
        )
    )

    if not tools_to_resume_with_confirmation:
      return

    # Step 3: Remove tools that have already been confirmed (dedup).
    for i in range(len(events) - 1, confirmation_event_index, -1):
      event = events[i]
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
        tools_to_resume_with_args.values(),
        {
            tool.name: tool
            for tool in await agent.canonical_tools(
                ReadonlyContext(invocation_context)
            )
        },
        tools_to_resume_with_confirmation.keys(),
        tools_to_resume_with_confirmation,
    ):
      yield function_response_event
    return


request_processor = _RequestConfirmationLlmRequestProcessor()
