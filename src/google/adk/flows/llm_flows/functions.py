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

"""Handles function calling for LLM flow."""

from __future__ import annotations

import logging
from typing import Dict
from typing import Optional
from typing import TYPE_CHECKING

from google.adk.platform import uuid as platform_uuid
from google.genai import types

from . import _batch_tool_executor
from ...auth.auth_tool import AuthConfig
from ...auth.auth_tool import AuthToolArguments
from ...events.event import Event
from ...tools.base_tool import BaseTool
from ...tools.tool_confirmation import ToolConfirmation
# Re-export definitions from submodules for full backward compatibility
from ._batch_tool_executor import _execute_prepared_function_calls_async as _execute_prepared_function_calls_async
from ._batch_tool_executor import _execute_prepared_function_calls_live as _execute_prepared_function_calls_live
from ._batch_tool_executor import _gather_or_cancel as _gather_or_cancel
from ._batch_tool_executor import _is_non_blocking_tool as _is_non_blocking_tool
from ._batch_tool_executor import _launch_non_blocking_call_live as _launch_non_blocking_call_live
from ._batch_tool_executor import _merge_and_trace_function_response_events as _merge_and_trace_function_response_events
from ._batch_tool_executor import _prepare_function_calls as _prepare_function_calls
from ._batch_tool_executor import deep_merge_dicts as deep_merge_dicts
from ._batch_tool_executor import merge_parallel_function_response_events as merge_parallel_function_response_events
from ._invocation_utils import require_agent_name as _require_agent_name
from ._tool_caller import _as_callback_result as _as_callback_result
from ._tool_caller import _as_function_response_part as _as_function_response_part
from ._tool_caller import _build_function_response_content as _build_function_response_content
from ._tool_caller import _build_response_event as _build_response_event
from ._tool_caller import _call_tool_async as _call_tool_async
from ._tool_caller import _call_tool_in_thread_pool as _call_tool_in_thread_pool
from ._tool_caller import _create_tool_context as _create_tool_context
from ._tool_caller import _emit_streaming_tool_event as _emit_streaming_tool_event
from ._tool_caller import _execute_single_prepared_call as _execute_single_prepared_call
from ._tool_caller import _execute_single_prepared_call_async as _execute_single_prepared_call_async
from ._tool_caller import _execute_single_prepared_call_live as _execute_single_prepared_call_live
from ._tool_caller import _extract_media_from_entry as _extract_media_from_entry
from ._tool_caller import _extract_multimodal_parts as _extract_multimodal_parts
from ._tool_caller import _get_tool as _get_tool
from ._tool_caller import _get_tool_and_context as _get_tool_and_context
from ._tool_caller import _get_tool_thread_pool as _get_tool_thread_pool
from ._tool_caller import _is_live_request_queue_annotation as _is_live_request_queue_annotation
from ._tool_caller import _is_sync_tool as _is_sync_tool
from ._tool_caller import _MAX_MEDIA_CONTAINER_DEPTH as _MAX_MEDIA_CONTAINER_DEPTH
from ._tool_caller import _message_content_for_user as _message_content_for_user
from ._tool_caller import _MESSAGE_EVENT_FIELDS as _MESSAGE_EVENT_FIELDS
from ._tool_caller import _normalize_tool_result as _normalize_tool_result
from ._tool_caller import _prepare_single as _prepare_single
from ._tool_caller import _PreparedFunctionCall as _PreparedFunctionCall
from ._tool_caller import _process_function_live_helper as _process_function_live_helper
from ._tool_caller import _TOOL_THREAD_POOL_LOCK as _TOOL_THREAD_POOL_LOCK
from ._tool_caller import _TOOL_THREAD_POOLS as _TOOL_THREAD_POOLS
from ._tool_caller import _try_decode_computer_use_image as _try_decode_computer_use_image

if TYPE_CHECKING:
  from ...agents.invocation_context import InvocationContext

AF_FUNCTION_CALL_ID_PREFIX = 'adk-'
REQUEST_EUC_FUNCTION_CALL_NAME = 'adk_request_credential'
REQUEST_CONFIRMATION_FUNCTION_CALL_NAME = 'adk_request_confirmation'
REQUEST_INPUT_FUNCTION_CALL_NAME = 'adk_request_input'

logger = logging.getLogger('google_adk.' + __name__)


def generate_client_function_call_id() -> str:
  return f'{AF_FUNCTION_CALL_ID_PREFIX}{platform_uuid.new_uuid()}'


def populate_client_function_call_id(model_response_event: Event) -> None:
  if not model_response_event.get_function_calls():
    return
  for function_call in model_response_event.get_function_calls():
    if not function_call.id:
      function_call.id = generate_client_function_call_id()


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
    request_id = generate_client_function_call_id()
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
    request_id = generate_client_function_call_id()
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
  function_calls = function_call_event.get_function_calls()
  return await handle_function_call_list_async(
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
) -> Optional[Event]:
  """Finds the function call event that matches the function response id of the last event."""
  if not events:
    return None

  last_event = events[-1]
  function_responses = last_event.get_function_responses()
  if not function_responses:
    return None

  function_call_id = function_responses[0].id
  if function_call_id is None:
    return None
  return find_event_by_function_call_id(events[:-1], function_call_id)
