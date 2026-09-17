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

from pydantic import ValidationError
from typing_extensions import override

from ..agents.invocation_context import InvocationContext
from ..agents.readonly_context import ReadonlyContext
from ..events.event import Event
from ..flows.llm_flows._base_llm_processor import BaseLlmRequestProcessor
from ..flows.llm_flows.functions import handle_function_calls_async
from ..flows.llm_flows.functions import REQUEST_EUC_FUNCTION_CALL_NAME
from ..models.llm_request import LlmRequest
from ..sessions.state import State
from .auth_credential import AuthCredential
from .auth_handler import AuthHandler
from .auth_tool import AuthConfig
from .auth_tool import AuthToolArguments


def _merge_credential_oauth2_fields(
    target_cred: AuthCredential | None,
    source_cred: AuthCredential | None,
) -> AuthCredential | None:
  """Merges OAuth2 fields from source_cred into target_cred if target_cred fields are None.

  If target_cred is None, returns source_cred.
  Otherwise, merges fields and returns target_cred.
  """
  if not source_cred:
    return target_cred
  if not target_cred:
    return source_cred

  if target_cred.oauth2 is None and source_cred.oauth2 is not None:
    target_cred.oauth2 = source_cred.oauth2.model_copy(deep=True)
  elif target_cred.oauth2 and source_cred.oauth2:
    target = target_cred.oauth2
    source = source_cred.oauth2
    for field in [
        "client_id",
        "client_secret",
        "redirect_uri",
        "code_verifier",
        "code_challenge_method",
    ]:
      if getattr(target, field) is None:
        setattr(target, field, getattr(source, field))

    # token_endpoint_auth_method has a default value "client_secret_basic" in OAuth2Auth model.
    # We only merge it if it wasn't explicitly set in target.
    target_fields_set = getattr(target, "model_fields_set", None)
    if (
        target_fields_set is None
        or "token_endpoint_auth_method" not in target_fields_set
    ):
      target.token_endpoint_auth_method = source.token_endpoint_auth_method

  return target_cred


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
  IDs are in *auth_fc_ids*, rebuilds each auth config from the request this
  server issued and takes only the exchanged credential from the client's
  response, stores credentials via ``AuthHandler``, and returns the set of
  original function call IDs that should be re-executed (excluding toolset
  auth).

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
  requested_auth_config_by_id: dict[str, AuthConfig] = {}
  for event in events:
    event_function_calls = event.get_function_calls()
    if not event_function_calls:
      continue
    try:
      for function_call in event_function_calls:
        if (
            function_call.id in auth_fc_ids
            and function_call.name == REQUEST_EUC_FUNCTION_CALL_NAME
        ):
          args = AuthToolArguments.model_validate(function_call.args)
          requested_auth_config_by_id[function_call.id] = args.auth_config
    except TypeError:
      continue

  # Step 2: Store credentials. The client's response supplies the result of the
  # user's browser round trip; the auth scheme, the raw credential and the
  # credential key come from the request this server issued.
  authorized_keys: set[str] = set()
  for fc_id in auth_fc_ids:
    if fc_id not in auth_responses:
      continue
    requested_auth_config = requested_auth_config_by_id.get(fc_id)
    if requested_auth_config is None:
      # Nothing to pin against, so the response would get to choose both the
      # credential it is exchanged with and the endpoint that goes to.
      logger.warning(
          "Ignoring auth response for function call ID %r, which this session"
          " never requested.",
          fc_id,
      )
      continue

    try:
      client_auth_config = AuthConfig.model_validate(auth_responses[fc_id])
    except (ValidationError, TypeError):
      # The response is attacker-reachable, so a malformed one skips this
      # function call instead of ending the whole invocation.
      logger.warning(
          "Ignoring malformed auth response for function call ID %r.", fc_id
      )
      continue

    # The scheme names the token endpoint the developer's secret is posted to
    # and the raw credential names the OAuth2 client it is posted as, so both
    # stay exactly as this server issued them. Only the outcome of the browser
    # round trip comes from the client, backfilled from the request for the
    # fields the client did not echo back.
    auth_config = requested_auth_config.model_copy(deep=True)
    auth_config.exchanged_auth_credential = _merge_credential_oauth2_fields(
        client_auth_config.exchanged_auth_credential,
        auth_config.exchanged_auth_credential,
    )
    if auth_config.credential_key:
      authorized_keys.add(auth_config.credential_key)

    await AuthHandler(auth_config=auth_config).parse_and_store_auth_response(
        state=state
    )

  # Step 3: Collect original function call IDs to resume, skipping
  # toolset auth entries which don't map to a resumable function call.
  tools_to_resume: set[str] = set()
  for fc_id in auth_fc_ids:
    requested_auth_config = requested_auth_config_by_id.get(fc_id)
    if not requested_auth_config:
      continue
    # Re-parse to get function_call_id (AuthConfig doesn't carry it;
    # AuthToolArguments does).
    for event in events:
      event_function_calls = event.get_function_calls()
      if not event_function_calls:
        continue
      for function_call in event_function_calls:
        if (
            function_call.id == fc_id
            and function_call.name == REQUEST_EUC_FUNCTION_CALL_NAME
        ):
          args = AuthToolArguments.model_validate(function_call.args)
          if args.function_call_id.startswith(
              TOOLSET_AUTH_CREDENTIAL_ID_PREFIX
          ):
            continue
          tools_to_resume.add(args.function_call_id)

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
