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

"""Backward compatibility module for contents processing.

Contents processing has moved to ``google.adk.flows.llm_flows.context._contents``.
This module re-exports all symbols for backward compatibility.
"""

from __future__ import annotations

from ._fencing import _is_other_agent_reply as _is_other_agent_reply
from ._fencing import _present_other_agent_message as _present_other_agent_message
from .context._compaction import _process_compaction_events as _process_compaction_events
from .context._compaction import _recover_compacted_function_calls as _recover_compacted_function_calls
from .context._contents import _add_instructions_to_user_content as _add_instructions_to_user_content
from .context._contents import _add_model_input_context_to_user_content as _add_model_input_context_to_user_content
from .context._contents import _build_task_input_user_content as _build_task_input_user_content
from .context._contents import _contains_empty_content as _contains_empty_content
from .context._contents import _ContentLlmRequestProcessor as _ContentLlmRequestProcessor
from .context._contents import _copy_content_for_request as _copy_content_for_request
from .context._contents import _get_contents as _get_contents
from .context._contents import _get_current_turn_contents as _get_current_turn_contents
from .context._contents import _id_pairing_model_types as _id_pairing_model_types
from .context._contents import _is_adk_framework_event as _is_adk_framework_event
from .context._contents import _is_auth_event as _is_auth_event
from .context._contents import _is_direct_transfer as _is_direct_transfer
from .context._contents import _is_event_belongs_to_branch as _is_event_belongs_to_branch
from .context._contents import _is_function_call_event as _is_function_call_event
from .context._contents import _is_live_model_media_event_with_inline_data as _is_live_model_media_event_with_inline_data
from .context._contents import _is_part_invisible as _is_part_invisible
from .context._contents import _is_request_confirmation_event as _is_request_confirmation_event
from .context._contents import _is_submitted_tool_result as _is_submitted_tool_result
from .context._contents import _should_include_event_in_context as _should_include_event_in_context
from .context._contents import _SINGLE_TURN_NUDGE as _SINGLE_TURN_NUDGE
from .context._contents import logger as logger
from .context._contents import request_processor as request_processor
from .tools._rearranger import _drop_orphaned_function_responses as _drop_orphaned_function_responses
from .tools._rearranger import _rearrange_events_for_async_function_responses_in_history as _rearrange_events_for_async_function_responses_in_history
from .tools._rearranger import _rearrange_events_for_latest_function_response as _rearrange_events_for_latest_function_response
from .tools._rearranger import drop_orphaned_function_calls as drop_orphaned_function_calls

__all__ = [
    '_ContentLlmRequestProcessor',
    '_SINGLE_TURN_NUDGE',
    '_add_instructions_to_user_content',
    '_add_model_input_context_to_user_content',
    '_build_task_input_user_content',
    '_contains_empty_content',
    '_copy_content_for_request',
    '_drop_orphaned_function_responses',
    '_get_contents',
    '_get_current_turn_contents',
    '_id_pairing_model_types',
    '_is_adk_framework_event',
    '_is_auth_event',
    '_is_direct_transfer',
    '_is_event_belongs_to_branch',
    '_is_function_call_event',
    '_is_live_model_media_event_with_inline_data',
    '_is_other_agent_reply',
    '_is_part_invisible',
    '_is_request_confirmation_event',
    '_is_submitted_tool_result',
    '_present_other_agent_message',
    '_process_compaction_events',
    '_rearrange_events_for_async_function_responses_in_history',
    '_rearrange_events_for_latest_function_response',
    '_recover_compacted_function_calls',
    '_should_include_event_in_context',
    'drop_orphaned_function_calls',
    'logger',
    'request_processor',
]
