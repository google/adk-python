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

"""Function call ID utilities, auth/confirmation event generation, and dispatch entry points."""

from __future__ import annotations

import logging
from typing import Dict
from typing import Optional
from typing import TYPE_CHECKING

from google.adk.platform import uuid as platform_uuid
from google.genai import types

from . import _batch_executor as _batch_tool_executor
from ....auth.auth_tool import AuthConfig
from ....auth.auth_tool import AuthToolArguments
from ....events.event import Event
from ....tools.base_tool import BaseTool
from ....tools.tool_confirmation import ToolConfirmation
from ..core._utils import require_agent_name as _require_agent_name

if TYPE_CHECKING:
  from ....agents.invocation_context import InvocationContext

AF_FUNCTION_CALL_ID_PREFIX = 'adk-'
REQUEST_EUC_FUNCTION_CALL_NAME = 'adk_request_credential'
REQUEST_CONFIRMATION_FUNCTION_CALL_NAME = 'adk_request_confirmation'
REQUEST_INPUT_FUNCTION_CALL_NAME = 'adk_request_input'

logger = logging.getLogger('google_adk.' + __name__)


def generate_client_function_call_id() -> str:
  return f'{AF_FUNCTION_CALL_ID_PREFIX}{platform_uuid.new_uuid()}'


def _new_client_function_call_id() -> str:
  """Generates a client function call id, resolved through the compat shim.

  Callers monkeypatch
  `google.adk.flows.llm_flows.functions.generate_client_function_call_id` to
  make ids deterministic; `cli/agent_test_runner.py` does this when rebuilding
  golden files, and so do tests outside ADK. Before this module was split out
  of `functions.py`, every id in this module came from that one module global,
  so a single patch covered all of them.

  Going through the shim keeps that true. Calling the module-local
  `generate_client_function_call_id()` directly would bypass those patches,
  because the shim's re-export is a separate binding.
  """
  from .. import functions

  return functions.generate_client_function_call_id()


def populate_client_function_call_id(model_response_event: Event) -> None:
  if not model_response_event.get_function_calls():
    return
  for function_call in model_response_event.get_function_calls():
    if not function_call.id:
      function_call.id = _new_client_function_call_id()


def remove_client_function_call_id(content: Optional[types.Content]) -> None:
  """Removes ADK-generated function call IDs from content before sending to LLM.

  Strips client-side function call/response IDs that start with 'adk-' prefix
  to avoid sending internal tracking IDs to the model.

  Args:
    content: Content containing function calls/responses to clean.
  """
  if content and content.parts:
    for part in content.parts:
      if (
          part.function_call
          and part.function_call.id
          and part.function_call.id.startswith(AF_FUNCTION_CALL_ID_PREFIX)
      ):
        part.function_call.id = None
      if (
          part.function_response
          and part.function_response.id
          and part.function_response.id.startswith(AF_FUNCTION_CALL_ID_PREFIX)
      ):
        part.function_response.id = None


def get_long_running_function_calls(
    function_calls: list[types.FunctionCall],
    tools_dict: dict[str, BaseTool],
) -> set[str]:
  long_running_tool_ids: set[str] = set()
  for function_call in function_calls:
    if (
        function_call.name in tools_dict
        and tools_dict[function_call.name].is_long_running
        and function_call.id is not None
    ):
      long_running_tool_ids.add(function_call.id)

  return long_running_tool_ids


def build_auth_request_event(
    invocation_context: InvocationContext,
    auth_requests: Dict[str, AuthConfig],
    *,
    author: Optional[str] = None,
    role: Optional[str] = None,
) -> Event:
  """Builds an auth request event with function calls for each auth request.

  This is a shared helper used by both tool-level auth (when a tool requests
  auth during execution) and toolset-level auth (before tool listing).

  Args:
    invocation_context: The invocation context.
    auth_requests: Dict mapping function_call_id to AuthConfig.
    author: The event author. Defaults to agent name.
    role: The content role. Defaults to None.

  Returns:
    Event with auth request function calls.
  """
  parts: list[types.Part] = []
  long_running_tool_ids: set[str] = set()

  deduplicated_requests: dict[str, AuthConfig] = {}
  seen_keys = set()
  for function_call_id, auth_config in auth_requests.items():
    key = auth_config.credential_key
    if not key:
      deduplicated_requests[function_call_id] = auth_config
    elif key not in seen_keys:
      seen_keys.add(key)
      deduplicated_requests[function_call_id] = auth_config

  for function_call_id, auth_config in deduplicated_requests.items():
    request_id = _new_client_function_call_id()
    request_euc_function_call = types.FunctionCall(
        name=REQUEST_EUC_FUNCTION_CALL_NAME,
        id=request_id,
        args=AuthToolArguments(
            function_call_id=function_call_id,
            auth_config=auth_config,
        ).model_dump(mode='json', exclude_none=True, by_alias=True),
    )
    long_running_tool_ids.add(request_id)
    parts.append(types.Part(function_call=request_euc_function_call))

  return Event(
      invocation_id=invocation_context.invocation_id,
      author=author or _require_agent_name(invocation_context),
      branch=invocation_context.branch,
      content=types.Content(parts=parts, role=role),
      long_running_tool_ids=long_running_tool_ids,
  )


def generate_auth_event(
    invocation_context: InvocationContext,
    function_response_event: Event,
) -> Optional[Event]:
  """Generates an auth request event from a function response event.

  This is used for tool-level auth where a tool requests credentials during
  execution.

  Args:
    invocation_context: The invocation context.
    function_response_event: The function response event with auth requests.

  Returns:
    Event with auth request function calls, or None if no auth requested.
  """
  if not function_response_event.actions.requested_auth_configs:
    return None

  return build_auth_request_event(
      invocation_context,
      function_response_event.actions.requested_auth_configs,
      role=(
          function_response_event.content.role
          if function_response_event.content is not None
          else None
      ),
  )


def generate_request_confirmation_event(
    invocation_context: InvocationContext,
    function_call_event: Event,
    function_response_event: Event,
) -> Optional[Event]:
  """Generates a request confirmation event from a function response event."""
  if not function_response_event.actions.requested_tool_confirmations:
    return None
  parts: list[types.Part] = []
  long_running_tool_ids: set[str] = set()
  function_calls = function_call_event.get_function_calls()
  for (
      function_call_id,
      tool_confirmation,
  ) in function_response_event.actions.requested_tool_confirmations.items():
    original_function_call = next(
        (fc for fc in function_calls if fc.id == function_call_id), None
    )
    if not original_function_call:
      continue
    request_id = _new_client_function_call_id()
    request_confirmation_function_call = types.FunctionCall(
        name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
        id=request_id,
        args={
            'originalFunctionCall': original_function_call.model_dump(
                exclude_none=True, by_alias=True
            ),
            'toolConfirmation': tool_confirmation.model_dump(
                by_alias=True, exclude_none=True
            ),
        },
    )
    long_running_tool_ids.add(request_id)
    parts.append(types.Part(function_call=request_confirmation_function_call))

  return Event(
      invocation_id=invocation_context.invocation_id,
      author=_require_agent_name(invocation_context),
      branch=invocation_context.branch,
      content=types.Content(parts=parts, role='model'),
      long_running_tool_ids=long_running_tool_ids,
  )


async def handle_function_call_list_async(
    invocation_context: InvocationContext,
    function_calls: list[types.FunctionCall],
    tools_dict: dict[str, BaseTool],
    filters: Optional[set[str]] = None,
    tool_confirmation_dict: Optional[dict[str, ToolConfirmation]] = None,
) -> Optional[Event]:
  """Calls the functions and returns the function response event."""
  return await _batch_tool_executor.handle_function_call_list_async(
      invocation_context=invocation_context,
      function_calls=function_calls,
      tools_dict=tools_dict,
      filters=filters,
      tool_confirmation_dict=tool_confirmation_dict,
  )


async def handle_function_calls_async(
    invocation_context: InvocationContext,
    function_call_event: Event,
    tools_dict: dict[str, BaseTool],
    filters: Optional[set[str]] = None,
    tool_confirmation_dict: Optional[dict[str, ToolConfirmation]] = None,
) -> Optional[Event]:
  """Calls the functions and returns the function response event."""
  from .. import functions

  function_calls = function_call_event.get_function_calls()
  return await functions.handle_function_call_list_async(
      invocation_context,
      function_calls,
      tools_dict,
      filters,
      tool_confirmation_dict,
  )


async def handle_function_calls_live(
    invocation_context: InvocationContext,
    function_call_event: Event,
    tools_dict: dict[str, BaseTool],
) -> Event | None:
  """Calls the functions and returns the function response event."""
  return await _batch_tool_executor.handle_function_calls_live(
      invocation_context=invocation_context,
      function_call_event=function_call_event,
      tools_dict=tools_dict,
  )


def find_event_by_function_call_id(
    events: list[Event],
    function_call_id: str,
) -> Optional[Event]:
  """Finds the function call event that matches the function call id."""
  for event in reversed(events):
    for function_call in event.get_function_calls():
      if function_call.id == function_call_id:
        return event
  return None


def _collect_function_call_ids(events: list[Event]) -> set[str]:
  """Returns the ids of every function call recorded in ``events``."""
  call_ids: set[str] = set()
  for event in events:
    for function_call in event.get_function_calls():
      if function_call.id:
        call_ids.add(function_call.id)
  return call_ids


def find_matching_function_call(
    events: list[Event],
    function_response_event: Optional[Event] = None,
) -> Optional[Event]:
  """Finds the function call event that matches the function response id."""
  if not events:
    return None

  target_response_event = (
      function_response_event
      if function_response_event is not None
      else events[-1]
  )
  function_responses = target_response_event.get_function_responses()
  if not function_responses:
    return None

  function_call_id = function_responses[0].id
  if not function_call_id:
    return None

  if events and events[-1].id == target_response_event.id:
    search_space = events[:-1]
  else:
    search_space = events

  return find_event_by_function_call_id(search_space, function_call_id)
