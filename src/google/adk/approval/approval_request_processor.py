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

"""Preprocessor for handling approval requests in the LLM flow."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, AsyncGenerator, Optional, TYPE_CHECKING, Tuple

from google.genai import types
from typing_extensions import override

from .approval_grant import ApprovalGrant
from .approval_handler import ApprovalHandler
from .approval_request import ApprovalRequest, ApprovalResponse
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
  """Handles approval information to build the LLM request."""

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    from ..agents.llm_agent import LlmAgent

    agent = invocation_context.agent
    if not isinstance(agent, LlmAgent):
      return

    if not (events := invocation_context.session.events):
      return

    state = invocation_context.session.state

    if not (approval_responses := self._get_approval_responses(events)):
      return

    if grant_update_actions := self._process_approvals(approval_responses, state):
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

    if function_calls_content := self._get_suspended_function_calls_content(state):
      # Reset suspended_function_calls in state (need to ensure a state_delta event is sent later to track this)
      state.update(
        {
          "approvals": {
            **state.get("approvals", {}),
            "suspended_function_calls": []
          }
        }
      )

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
          function_response_event.actions.state_delta['approvals'] = {}

        # reset suspended_function_calls if no state_deltas created - all were passed through grant testing logic again
        function_response_event.actions.state_delta["approvals"] = {
          **state.get("approvals", {}),
          **function_response_event.actions.state_delta.get("approvals")
        }

        yield function_response_event
        state.update(function_response_event.actions.state_delta)

  def _get_approval_responses(self, events: list[Event]) -> list[ApprovalResponse]:
    if not (user_event := get_last_user_response(events)):
      return []

    return [
        ApprovalResponse.model_validate(function_call_response.response)
        for function_call_response in user_event.get_function_responses()
        if function_call_response.name == REQUEST_APPROVAL_FUNCTION_CALL_NAME
    ]

  def _process_approvals(self, approval_responses: list[ApprovalResponse], state: dict[str, Any]) -> Optional[EventActions]:
    initial_grants = self._get_grants(state)
    if extra_grants := ApprovalHandler.parse_and_store_approval_responses(
        initial_grants=initial_grants,
        approval_responses=approval_responses,
    ):
      return EventActions(
        state_delta={
            "approvals": {
                **state.get("approvals", {}),
                "grants": [
                    grant.model_dump(mode="json") for grant in initial_grants + extra_grants
                ]
            }
        },
      )

  def _get_suspended_function_calls_content(self, state) -> Optional[types.Content]:
    suspended_function_calls = self._get_suspended_function_calls(state)
    # align with existing function calls? Or no point?
    # Create new event with function calls?
    return types.Content(
      role="model",
      parts=[
          types.Part(function_call=function_call)
          for function_call in suspended_function_calls
      ]
    )

  def _get_grants(self, state) -> list[ApprovalGrant]:
    return [
      ApprovalGrant.model_validate(grant_dict)
      for grant_dict in state.get("approvals", {}).get("grants", [])
    ]

  def _get_suspended_function_calls(self, state) -> list[types.FunctionCall]:
    function_calls = [
      types.FunctionCall.model_validate(suspended_function_call)
      for suspended_function_call in
      state.get("approvals", {}).get("suspended_function_calls", [])
    ]
    function_calls_dict = {fc.id: fc for fc in function_calls}
    return list(function_calls_dict.values())


# Create the preprocessor instance
request_processor = _ApprovalLlmRequestProcessor()


def get_last_user_response(events) -> Event:
    user_events = list(get_user_responses(events, max_events=1))
    if len(user_events) == 1:
        return user_events[0]


def get_user_responses(events, max_events=None):
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
