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

"""Provides an LLM request preprocessor for handling the approval workflow.

This module defines `_ApprovalLlmRequestProcessor`, which integrates into the LLM
request lifecycle to:
1. Check for incoming approval responses from the user/client.
2. Update the session state with new grants based on these responses.
3. Identify and prepare previously suspended function calls (due to pending approvals)
   that can now be resumed with the new grants.
4. Generate necessary events to reflect grant updates and to trigger the execution
   of resumed function calls.

It also includes helper functions for extracting approval-related information from events.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, AsyncGenerator, Optional, TYPE_CHECKING, Tuple

from google.genai import types
from typing_extensions import override

from .approval_grant import ApprovalGrant
from .approval_handler import ApprovalHandler
from .approval_request import ApprovalRequest, ApprovalResponse, FunctionCallStatus
from ..agents.invocation_context import InvocationContext
from ..events import EventActions
from ..events.event import Event
from ..flows.llm_flows import functions
from ..flows.llm_flows._base_llm_processor import BaseLlmRequestProcessor
from ..models.llm_request import LlmRequest
from ..sessions import State

if TYPE_CHECKING:
  from ..agents.llm_agent import LlmAgent

# The name of the function call for requesting approval
REQUEST_APPROVAL_FUNCTION_CALL_NAME = "adk_request_approval"


class _ApprovalLlmRequestProcessor(BaseLlmRequestProcessor):
  """Processes incoming approval responses and manages suspended function calls.

  This preprocessor runs before the main LLM content generation. It inspects the latest
  user event for any `ApprovalResponse` function responses. If found, it updates the
  session's grants. Then, it checks if any function calls were previously suspended
  pending these approvals. If such calls exist and can now proceed (or are still
  partially pending), it prepares them for re-evaluation or execution.
  """

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    """Executes the approval processing logic.

    Checks for approval responses in the last user event, updates grants, and
    resumes or further processes suspended function calls.

    Args:
        invocation_context: The current invocation context, providing access to the
                            session, agent, and other invocation details.
        llm_request: The current LLM request object (not directly modified by this
                     processor but part of the interface).

    Yields:
        `Event` objects representing state changes (grant updates) or new function
        call processing events if calls are resumed.
    """
    from ..agents.llm_agent import LlmAgent

    agent = invocation_context.agent
    if not isinstance(agent, LlmAgent):
      return

    if not (events := invocation_context.session.events):
      return

    state = invocation_context.session.state

    if not (approval_responses := self._get_approval_responses(events)):
      return

    if grant_update_actions := self._process_approvals(
        approval_responses, state
    ):
      yield Event(
          invocation_id=invocation_context.invocation_id,
          author=invocation_context.agent.name,
          actions=grant_update_actions,
          branch=invocation_context.branch,
      )
      state.update(grant_update_actions.state_delta)
    else:
      invocation_context.end_invocation = True
      return

    if function_calls_content := self._get_suspended_function_calls_content(
        state
    ):
      # Reset suspended_function_calls in state (need to ensure a state_delta event is sent later to track this)
      state["approvals__suspended_function_calls"] = []

      function_calls_event = Event(
          invocation_id=invocation_context.invocation_id,
          author=invocation_context.agent.name,
          branch=invocation_context.branch,
          content=function_calls_content,
      )

      if function_response_event := await functions.handle_function_calls_async(
          invocation_context,
          function_calls_event,
          {tool.name: tool for tool in await agent.canonical_tools()},
      ):
        if "approvals" not in function_response_event.actions.state_delta:
          function_response_event.actions.state_delta["approvals"] = {}

        # reset suspended_function_calls if no state_deltas created - all were passed through grant testing logic again
        # function_response_event.actions.state_delta["approvals__suspended_function_calls"] = function_response_event.actions.state_delta.get("approvals__suspended_function_calls")

        yield function_response_event
        state.update(function_response_event.actions.state_delta)

  def _get_approval_responses(
      self, events: list[Event]
  ) -> list[ApprovalResponse]:
    """Extracts `ApprovalResponse` objects from the last user event.

    Searches the most recent event authored by the "user" for function responses
    named `REQUEST_APPROVAL_FUNCTION_CALL_NAME` and parses them into
    `ApprovalResponse` model instances.

    Args:
        events: A list of all events in the current session, ordered chronologically.

    Returns:
        A list of `ApprovalResponse` objects found, or an empty list if none are present.
    """
    if not (user_event := get_last_user_response(events)):
      return []

    return [
        ApprovalResponse.model_validate(function_call_response.response)
        for function_call_response in user_event.get_function_responses()
        if function_call_response.name == REQUEST_APPROVAL_FUNCTION_CALL_NAME
    ]

  def _process_approvals(
      self, approval_responses: list[ApprovalResponse], state: dict[str, Any]
  ) -> Optional[EventActions]:
    """Processes approval responses to update grants in the state.

    Compares grants from `approval_responses` with existing grants in `state`
    and generates `EventActions` to update `state["approvals__grants"]` if
    new grants are found.

    Args:
        approval_responses: A list of `ApprovalResponse` objects from the user.
        state: The current session state dictionary.

    Returns:
        An `EventActions` object with a `state_delta` for grant updates if new
        grants were processed, otherwise `None`.
    """
    initial_grants = self._get_grants(state)
    if extra_grants := ApprovalHandler.parse_and_store_approval_responses(
        initial_grants=initial_grants,
        approval_responses=approval_responses,
    ):
      return EventActions(
          state_delta={
              "approvals__grants": [
                  grant.model_dump(mode="json")
                  for grant in initial_grants + extra_grants
              ]
          },
      )

  def _get_suspended_function_calls_content(
      self, state: dict[str, Any]
  ) -> Optional[types.Content]:
    """Creates a `types.Content` object containing currently suspended function calls.

    Retrieves function calls from `state["approvals__suspended_function_calls"]` that
    have a status of "suspended" and packages them into a `types.Content` object,
    which can be used to re-trigger their processing.

    Args:
        state: The current session state dictionary.

    Returns:
        A `types.Content` object with parts for each suspended function call, or
        `None` if no calls are currently suspended.
    """
    suspended_function_calls = self._get_suspended_function_calls(state)
    # align with existing function calls? Or no point?
    # Create new event with function calls?
    return types.Content(
        role="model",
        parts=[
            types.Part(function_call=function_call)
            for function_call in suspended_function_calls
        ],
    )

  def _get_grants(self, state: dict[str, Any]) -> list[ApprovalGrant]:
    """Retrieves and validates `ApprovalGrant` objects from the state.

    Args:
        state: The current session state dictionary.

    Returns:
        A list of `ApprovalGrant` model instances.
    """
    return [
        ApprovalGrant.model_validate(grant_dict)
        for grant_dict in state.get("approvals__grants", [])
    ]

  def _get_suspended_function_calls(self, state: dict[str, Any]) -> list[types.FunctionCall]:
    """Extracts unique, currently suspended function calls from the state.

    Filters `state["approvals__suspended_function_calls"]` for entries with status
    "suspended" and returns a list of unique `types.FunctionCall` objects.
    Uniqueness is based on the function call ID.

    Args:
        state: The current session state dictionary.

    Returns:
        A list of unique `types.FunctionCall` objects that are suspended.
    """
    function_calls = [
        FunctionCallStatus.model_validate(suspended_function_call).function_call
        for suspended_function_call in state.get(
            "approvals__suspended_function_calls", []
        )
        if FunctionCallStatus.model_validate(suspended_function_call).status == "suspended"
    ]
    function_calls_dict = {fc.id: fc for fc in function_calls}
    return list(function_calls_dict.values())


# Create the preprocessor instance
request_processor = _ApprovalLlmRequestProcessor()


def get_last_user_response(events: list[Event]) -> Optional[Event]:
  """Retrieves the most recent user event containing function responses.

  Iterates backwards through the event list to find the latest event authored
  by "user" that also contains at least one function response.

  Args:
      events: The list of all session events.

  Returns:
      The last relevant user `Event`, or `None` if not found.
  """
  user_events = list(get_user_responses(events, max_events=1))
  if len(user_events) == 1:
    return user_events[0]


def get_user_responses(events: list[Event], max_events: Optional[int]=None) -> Generator[Event, None, None]:
  """Yields user events containing function responses, from most recent.

  Iterates backwards through the event list, yielding events that are authored
  by "user" and contain function responses.

  Args:
      events: The list of all session events.
      max_events: Optional limit on the number of events to yield.

  Yields:
      `Event` objects that meet the criteria.
  """
  events_emitted = 0
  for k in range(len(events) - 1, -1, -1):
    event = events[k]
    # Look for first event authored by user
    if not event.author or event.author != "user":
      continue

    responses = event.get_function_responses()
    if not responses:
      return

    yield event
    events_emitted += 1
    if max_events is not None and events_emitted >= max_events:
      break
