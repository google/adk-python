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

from typing import Any
from typing import AsyncGenerator

from typing_extensions import override

from ..agents.invocation_context import InvocationContext
from ..agents.readonly_context import ReadonlyContext
from ..events.event import Event
from ..flows.llm_flows import functions
from ..flows.llm_flows._base_llm_processor import BaseLlmRequestProcessor
from ..flows.llm_flows.functions import REQUEST_EUC_FUNCTION_CALL_NAME
from ..models.llm_request import LlmRequest
from ..sessions.state import State
from .auth_handler import AuthHandler
from .auth_tool import AuthConfig
from .auth_tool import AuthToolArguments

# Prefix used by toolset auth credential IDs.
# Auth requests with this prefix are for toolset authentication (before tool
# listing) and don't require resuming a function call.
TOOLSET_AUTH_CREDENTIAL_ID_PREFIX = '_adk_toolset_auth_'


def _find_function_call(
    events: list[Event], function_call_id: str
) -> Any | None:
  for event in events:
    for function_call in event.get_function_calls():
      if function_call.id == function_call_id:
        return function_call
  return None


def _has_requested_auth_config(
    events: list[Event], function_call_id: str
) -> bool:
  return any(
      function_call_id in event.actions.requested_auth_configs
      for event in events
  )


def _is_valid_auth_resume_target(
    events: list[Event], args: AuthToolArguments
) -> bool:
  if args.function_call_id.startswith(TOOLSET_AUTH_CREDENTIAL_ID_PREFIX):
    return False
  if not _has_requested_auth_config(events, args.function_call_id):
    return False
  if not args.function_call_digest:
    return False

  function_call = _find_function_call(events, args.function_call_id)
  if not function_call:
    return False

  return (
      functions.function_call_digest(function_call) == args.function_call_digest
  )


async def _store_auth_and_collect_resume_targets(
    events: list[Event],
    auth_fc_ids: set[str],
    auth_responses: dict[str, Any],
    state: State,
) -> set[str]:
  """Store auth credentials and return original function call IDs to resume.

  Scans session events for ``adk_request_credential`` function calls whose
  IDs are in *auth_fc_ids*, extracts ``credential_key`` from their
  ``AuthToolArguments`` args, merges ``credential_key`` into the
  corresponding auth response, stores credentials via ``AuthHandler``,
  and returns the set of original function call IDs that should be
  re-executed (excluding toolset auth).

  Args:
    events: Session events to scan.
    auth_fc_ids: IDs of ``adk_request_credential`` function calls to match.
    auth_responses: Mapping of FC ID -> auth config response dict from the
      client.
    state: Session state for temporary credential storage.

  Returns:
    Set of original function call IDs to resume.
  """
  # Step 1: Scan events for matching adk_request_credential function calls
  # to extract AuthToolArguments (contains credential_key).
  requested_auth_args_by_id: dict[str, AuthToolArguments] = {}
  for event in events:
    event_function_calls = event.get_function_calls()
    if not event_function_calls:
      continue
    for function_call in event_function_calls:
      if (
          function_call.id not in auth_fc_ids
          or function_call.name != REQUEST_EUC_FUNCTION_CALL_NAME
      ):
        continue
      try:
        requested_auth_args_by_id[function_call.id] = (
            AuthToolArguments.model_validate(function_call.args)
        )
      except TypeError:
        continue

  # Step 2: Store credentials. Merge credential_key from the original
  # request into the client's auth response before storing.
  for fc_id in auth_fc_ids:
    if fc_id not in auth_responses:
      continue
    auth_config = AuthConfig.model_validate(auth_responses[fc_id])
    requested_auth_args = requested_auth_args_by_id.get(fc_id)
    if requested_auth_args and (
        requested_auth_args.auth_config.credential_key is not None
    ):
      auth_config.credential_key = (
          requested_auth_args.auth_config.credential_key
      )
    await AuthHandler(auth_config=auth_config).parse_and_store_auth_response(
        state=state
    )

  # Step 3: Collect original function call IDs to resume, skipping
  # toolset auth entries which don't map to a resumable function call.
  tools_to_resume: set[str] = set()
  for args in requested_auth_args_by_id.values():
    if _is_valid_auth_resume_target(events, args):
      tools_to_resume.add(args.function_call_id)

  return tools_to_resume


class _AuthLlmRequestProcessor(BaseLlmRequestProcessor):
  """Handles auth information to build the LLM request."""

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    agent = invocation_context.agent
    if not hasattr(agent, 'canonical_tools'):
      return
    events = invocation_context.session.events
    if not events:
      return

    # Find the last user-authored event with function responses to
    # identify adk_request_credential responses.
    last_event_with_content = None
    last_event_with_content_index = -1
    for i in range(len(events) - 1, -1, -1):
      event = events[i]
      if event.content is not None:
        last_event_with_content = event
        last_event_with_content_index = i
        break

    if not last_event_with_content or last_event_with_content.author != 'user':
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
    prior_events = events[:last_event_with_content_index]
    tools_to_resume = await _store_auth_and_collect_resume_targets(
        prior_events,
        auth_fc_ids,
        auth_responses,
        invocation_context.session.state,
    )

    if not tools_to_resume:
      return

    # Find the original function call event and re-execute the tools
    # that needed auth.
    for i in range(last_event_with_content_index - 1, -1, -1):
      event = events[i]
      function_calls = event.get_function_calls()
      if not function_calls:
        continue

      if any([
          function_call.id in tools_to_resume
          for function_call in function_calls
      ]):
        if function_response_event := await functions.handle_function_calls_async(
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
