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

"""Backward-compatibility shim for flows/llm_flows/tools/."""

from __future__ import annotations

from .core._utils import require_agent_name as _require_agent_name
from .tools import _batch_executor as _batch_tool_executor
from .tools._batch_executor import _execute_prepared_function_calls_async as _execute_prepared_function_calls_async
from .tools._batch_executor import _execute_prepared_function_calls_live as _execute_prepared_function_calls_live
from .tools._batch_executor import _gather_or_cancel as _gather_or_cancel
from .tools._batch_executor import _is_non_blocking_tool as _is_non_blocking_tool
from .tools._batch_executor import _launch_non_blocking_call_live as _launch_non_blocking_call_live
from .tools._batch_executor import _merge_and_trace_function_response_events as _merge_and_trace_function_response_events
from .tools._batch_executor import _prepare_function_calls as _prepare_function_calls
from .tools._batch_executor import deep_merge_dicts as deep_merge_dicts
from .tools._batch_executor import merge_parallel_function_response_events as merge_parallel_function_response_events
from .tools._caller import _as_callback_result as _as_callback_result
from .tools._caller import _as_function_response_part as _as_function_response_part
from .tools._caller import _build_function_response_content as _build_function_response_content
from .tools._caller import _build_response_event as _build_response_event
from .tools._caller import _call_tool_async as _call_tool_async
from .tools._caller import _call_tool_in_thread_pool as _call_tool_in_thread_pool
from .tools._caller import _create_tool_context as _create_tool_context
from .tools._caller import _emit_streaming_tool_event as _emit_streaming_tool_event
from .tools._caller import _execute_single_prepared_call as _execute_single_prepared_call
from .tools._caller import _execute_single_prepared_call_async as _execute_single_prepared_call_async
from .tools._caller import _execute_single_prepared_call_live as _execute_single_prepared_call_live
from .tools._caller import _extract_media_from_entry as _extract_media_from_entry
from .tools._caller import _extract_multimodal_parts as _extract_multimodal_parts
from .tools._caller import _get_tool as _get_tool
from .tools._caller import _get_tool_and_context as _get_tool_and_context
from .tools._caller import _get_tool_thread_pool as _get_tool_thread_pool
from .tools._caller import _is_live_request_queue_annotation as _is_live_request_queue_annotation
from .tools._caller import _is_sync_tool as _is_sync_tool
from .tools._caller import _MAX_MEDIA_CONTAINER_DEPTH as _MAX_MEDIA_CONTAINER_DEPTH
from .tools._caller import _message_content_for_user as _message_content_for_user
from .tools._caller import _MESSAGE_EVENT_FIELDS as _MESSAGE_EVENT_FIELDS
from .tools._caller import _normalize_tool_result as _normalize_tool_result
from .tools._caller import _prepare_single as _prepare_single
from .tools._caller import _PreparedFunctionCall as _PreparedFunctionCall
from .tools._caller import _process_function_live_helper as _process_function_live_helper
from .tools._caller import _TOOL_THREAD_POOL_LOCK as _TOOL_THREAD_POOL_LOCK
from .tools._caller import _TOOL_THREAD_POOLS as _TOOL_THREAD_POOLS
from .tools._caller import _try_decode_computer_use_image as _try_decode_computer_use_image
from .tools._functions import _collect_function_call_ids as _collect_function_call_ids
from .tools._functions import AF_FUNCTION_CALL_ID_PREFIX as AF_FUNCTION_CALL_ID_PREFIX
from .tools._functions import build_auth_request_event as build_auth_request_event
from .tools._functions import find_event_by_function_call_id as find_event_by_function_call_id
from .tools._functions import find_matching_function_call as find_matching_function_call
from .tools._functions import generate_auth_event as generate_auth_event
from .tools._functions import generate_client_function_call_id as generate_client_function_call_id
from .tools._functions import generate_request_confirmation_event as generate_request_confirmation_event
from .tools._functions import get_long_running_function_calls as get_long_running_function_calls
from .tools._functions import handle_function_call_list_async as handle_function_call_list_async
from .tools._functions import handle_function_calls_async as handle_function_calls_async
from .tools._functions import handle_function_calls_live as handle_function_calls_live
from .tools._functions import logger as logger
from .tools._functions import populate_client_function_call_id as populate_client_function_call_id
from .tools._functions import remove_client_function_call_id as remove_client_function_call_id
from .tools._functions import REQUEST_CONFIRMATION_FUNCTION_CALL_NAME as REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
from .tools._functions import REQUEST_EUC_FUNCTION_CALL_NAME as REQUEST_EUC_FUNCTION_CALL_NAME
from .tools._functions import REQUEST_INPUT_FUNCTION_CALL_NAME as REQUEST_INPUT_FUNCTION_CALL_NAME

__all__ = [
    'AF_FUNCTION_CALL_ID_PREFIX',
    'REQUEST_CONFIRMATION_FUNCTION_CALL_NAME',
    'REQUEST_EUC_FUNCTION_CALL_NAME',
    'REQUEST_INPUT_FUNCTION_CALL_NAME',
    '_MAX_MEDIA_CONTAINER_DEPTH',
    '_MESSAGE_EVENT_FIELDS',
    '_PreparedFunctionCall',
    '_TOOL_THREAD_POOLS',
    '_TOOL_THREAD_POOL_LOCK',
    '_as_callback_result',
    '_as_function_response_part',
    '_batch_tool_executor',
    '_build_function_response_content',
    '_build_response_event',
    '_call_tool_async',
    '_call_tool_in_thread_pool',
    '_collect_function_call_ids',
    '_create_tool_context',
    '_emit_streaming_tool_event',
    '_execute_prepared_function_calls_async',
    '_execute_prepared_function_calls_live',
    '_execute_single_prepared_call',
    '_execute_single_prepared_call_async',
    '_execute_single_prepared_call_live',
    '_extract_media_from_entry',
    '_extract_multimodal_parts',
    '_gather_or_cancel',
    '_get_tool',
    '_get_tool_and_context',
    '_get_tool_thread_pool',
    '_is_live_request_queue_annotation',
    '_is_non_blocking_tool',
    '_is_sync_tool',
    '_launch_non_blocking_call_live',
    '_merge_and_trace_function_response_events',
    '_message_content_for_user',
    '_normalize_tool_result',
    '_prepare_function_calls',
    '_prepare_single',
    '_process_function_live_helper',
    '_require_agent_name',
    '_try_decode_computer_use_image',
    'build_auth_request_event',
    'deep_merge_dicts',
    'find_event_by_function_call_id',
    'find_matching_function_call',
    'generate_auth_event',
    'generate_client_function_call_id',
    'generate_request_confirmation_event',
    'get_long_running_function_calls',
    'handle_function_call_list_async',
    'handle_function_calls_async',
    'handle_function_calls_live',
    'logger',
    'merge_parallel_function_response_events',
    'populate_client_function_call_id',
    'remove_client_function_call_id',
]
