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

from typing_extensions import override

from ..agents.invocation_context import InvocationContext
from ..agents.readonly_context import ReadonlyContext
from ..events.event import Event
from ..flows.llm_flows._base_llm_processor import BaseLlmRequestProcessor
from ..flows.llm_flows.tools._functions import handle_function_calls_async
from ..models.llm_request import LlmRequest
from ..sessions.state import State
from ..utils._function_call_names import REQUEST_EUC_FUNCTION_CALL_NAME
from ._auth_resume import find_requested_auth_configs
from ._auth_resume import store_auth_response

# Prefix used by toolset auth credential IDs.
# Auth requests with this prefix are for toolset authentication (before tool
# listing) and don't require resuming a function call.
TOOLSET_AUTH_CREDENTIAL_ID_PREFIX = "_adk_toolset_auth_"

logger = logging.getLogger("google_adk." + __name__)


async def _store_auth_and_collect_resume_targets(
    events: list[Event],
    auth_fc_ids: set[str],
    auth_responses: dict[str, Any],
    state: State,
) -> set[str]:
  """Store auth credentials and return original function call IDs to resume.

  Scans session events for ``adk_request_credential`` function calls whose
  IDs are in *auth_fc_ids*, pins each client response (`AuthConfig` or
  `AuthCredential` dict) against the server-issued request via
  ``store_auth_response``, and returns the set of original function call IDs
  that should be re-executed (excluding toolset auth).

  Args:
    events: Session events to scan.
    auth_fc_ids: IDs of ``adk_request_credential`` function calls to match.
    auth_responses: Mapping of FC ID -> auth response payload from the client.
    state: Session state for temporary credential storage.

  Returns:
    Set of original function call IDs to resume.
  """
  requested_by_id = find_requested_auth_configs(events, auth_fc_ids)

  authorized_keys: set[str] = set()
  for fc_id in auth_fc_ids:
    if fc_id not in auth_responses:
      continue
    requested_args = requested_by_id.get(fc_id)
    if requested_args is None:
      logger.warning(
          "Ignoring auth response for function call ID %r, which this session"
          " never requested.",
          fc_id,
      )
      continue

    stored_config = await store_auth_response(
        requested=requested_args.auth_config,
        response=auth_responses[fc_id],
        state=state,
        interrupt_id=fc_id,
    )
    if stored_config is not None and stored_config.credential_key:
      authorized_keys.add(stored_config.credential_key)

  tools_to_resume: set[str] = set()
  for requested_args in requested_by_id.values():
    if not requested_args.function_call_id.startswith(
        TOOLSET_AUTH_CREDENTIAL_ID_PREFIX
    ):
      tools_to_resume.add(requested_args.function_call_id)

  matching_events: list[Event] = []
  for event in events:
    actions = getattr(event, "actions", None)
    if actions and actions.requested_auth_configs:
      if any(
          fc_id in actions.requested_auth_configs for fc_id in tools_to_resume
      ):
        matching_events.append(event)

  for event in matching_events:
    actions = getattr(event, "actions", None)
    if actions and actions.requested_auth_configs:
      for (
          original_fc_id,
          config,
      ) in actions.requested_auth_configs.items():
        if config.credential_key in authorized_keys:
          tools_to_resume.add(original_fc_id)

  return tools_to_resume


class _AuthLlmRequestProcessor(BaseLlmRequestProcessor):
  """Handles auth information to build the LLM request."""

  name = "auth"

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    agent = invocation_context.agent
    if agent is None or not hasattr(agent, "canonical_tools"):
      return
    events = invocation_context._get_events(current_branch=True)
    if not events:
      return

    # Find the last user-authored event with function responses to
    # identify adk_request_credential responses.
    last_event_with_content = None
    for i in range(len(events) - 1, -1, -1):
      event = events[i]
      if event.content is not None:
        last_event_with_content = event
        break

    if not last_event_with_content or last_event_with_content.author != "user":
      return

    responses = last_event_with_content.get_function_responses()
    if not responses:
      return

    # Collect adk_request_credential function response IDs and their
    # response dicts.
    auth_fc_ids: set[str] = set()
    auth_responses: dict[str, Any] = {}
    for function_call_response in responses:
      if function_call_response.name != REQUEST_EUC_FUNCTION_CALL_NAME:
        continue
      auth_fc_ids.add(function_call_response.id)
      auth_responses[function_call_response.id] = (
          function_call_response.response
      )

    if not auth_fc_ids:
      return

    # Store credentials and collect tools to resume.
    tools_to_resume = await _store_auth_and_collect_resume_targets(
        events, auth_fc_ids, auth_responses, invocation_context.session.state
    )

    if not tools_to_resume:
      return

    # Find the original function call event and re-execute the tools
    # that needed auth.
    for i in range(len(events) - 2, -1, -1):
      event = events[i]
      function_calls = event.get_function_calls()
      if not function_calls:
        continue

      if any([
          function_call.id in tools_to_resume
          for function_call in function_calls
      ]):
        # If this tool call was authored by another agent, skip it to let
        # that agent's own auth processor handle it. Without this check, a
        # shared session's events could cause one agent to resume and
        # execute a different agent's auth-gated tool call using its own
        # (potentially differently-scoped) canonical_tools.
        if event.author != agent.name:
          continue
        if function_response_event := await handle_function_calls_async(
            invocation_context,
            event,
            {
                tool.name: tool
                for tool in await agent.canonical_tools(
                    ReadonlyContext(invocation_context)
                )
            },
            tools_to_resume,
        ):
          yield function_response_event
        return
    return


request_processor = _AuthLlmRequestProcessor()
