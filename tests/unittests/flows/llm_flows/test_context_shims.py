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

"""Tests ensuring backward-compatibility shims for context modules export expected symbols."""

from google.adk.flows import llm_flows
from google.adk.flows.llm_flows import _content_compaction as shim_content_compaction
from google.adk.flows.llm_flows import compaction as shim_compaction
from google.adk.flows.llm_flows import contents as shim_contents
from google.adk.flows.llm_flows import context_cache_processor as shim_cache
from google.adk.flows.llm_flows import interactions_processor as shim_interactions
from google.adk.flows.llm_flows.context import _cache as ctx_cache
from google.adk.flows.llm_flows.context import _compaction as ctx_compaction
from google.adk.flows.llm_flows.context import _contents as ctx_contents
from google.adk.flows.llm_flows.context import _interactions as ctx_interactions


def test_contents_shim_exports():
  """All symbols in contents shim match context._contents and related modules."""
  assert shim_contents.request_processor is ctx_contents.request_processor
  assert (
      shim_contents._ContentLlmRequestProcessor
      is ctx_contents._ContentLlmRequestProcessor
  )
  assert shim_contents._SINGLE_TURN_NUDGE is ctx_contents._SINGLE_TURN_NUDGE
  assert shim_contents._get_contents is ctx_contents._get_contents
  assert (
      shim_contents._get_current_turn_contents
      is ctx_contents._get_current_turn_contents
  )
  assert (
      shim_contents._contains_empty_content
      is ctx_contents._contains_empty_content
  )
  assert (
      shim_contents._copy_content_for_request
      is ctx_contents._copy_content_for_request
  )
  assert (
      shim_contents._id_pairing_model_types
      is ctx_contents._id_pairing_model_types
  )
  assert (
      shim_contents._is_event_belongs_to_branch
      is ctx_contents._is_event_belongs_to_branch
  )
  assert (
      shim_contents._is_adk_framework_event
      is ctx_contents._is_adk_framework_event
  )
  assert shim_contents._is_auth_event is ctx_contents._is_auth_event
  assert shim_contents._is_direct_transfer is ctx_contents._is_direct_transfer
  assert (
      shim_contents._is_function_call_event
      is ctx_contents._is_function_call_event
  )
  assert (
      shim_contents._is_live_model_media_event_with_inline_data
      is ctx_contents._is_live_model_media_event_with_inline_data
  )
  assert shim_contents._is_part_invisible is ctx_contents._is_part_invisible
  assert (
      shim_contents._is_request_confirmation_event
      is ctx_contents._is_request_confirmation_event
  )
  assert (
      shim_contents._is_submitted_tool_result
      is ctx_contents._is_submitted_tool_result
  )
  assert (
      shim_contents._should_include_event_in_context
      is ctx_contents._should_include_event_in_context
  )
  assert (
      shim_contents._add_instructions_to_user_content
      is ctx_contents._add_instructions_to_user_content
  )
  assert (
      shim_contents._add_model_input_context_to_user_content
      is ctx_contents._add_model_input_context_to_user_content
  )
  assert (
      shim_contents._build_task_input_user_content
      is ctx_contents._build_task_input_user_content
  )
  assert shim_contents.logger is ctx_contents.logger
  # Re-exported from compaction
  assert (
      shim_contents._process_compaction_events
      is ctx_compaction._process_compaction_events
  )
  assert (
      shim_contents._recover_compacted_function_calls
      is ctx_compaction._recover_compacted_function_calls
  )
  # Re-exported from fencing & rearranger
  assert hasattr(shim_contents, '_is_other_agent_reply')
  assert hasattr(shim_contents, '_present_other_agent_message')
  assert hasattr(
      shim_contents, '_rearrange_events_for_latest_function_response'
  )
  assert hasattr(shim_contents, '_drop_orphaned_function_responses')
  assert hasattr(shim_contents, 'drop_orphaned_function_calls')


def test_compaction_shim_exports():
  """All symbols in compaction shim match context._compaction."""
  assert shim_compaction.request_processor is ctx_compaction.request_processor
  assert (
      shim_compaction.CompactionRequestProcessor
      is ctx_compaction.CompactionRequestProcessor
  )


def test_content_compaction_shim_exports():
  """All symbols in _content_compaction shim match context._compaction."""
  assert (
      shim_content_compaction._process_compaction_events
      is ctx_compaction._process_compaction_events
  )
  assert (
      shim_content_compaction._recover_compacted_function_calls
      is ctx_compaction._recover_compacted_function_calls
  )


def test_context_cache_processor_shim_exports():
  """All symbols in context_cache_processor shim match context._cache."""
  assert shim_cache.request_processor is ctx_cache.request_processor
  assert (
      shim_cache.ContextCacheRequestProcessor
      is ctx_cache.ContextCacheRequestProcessor
  )
  assert shim_cache.logger is ctx_cache.logger


def test_interactions_processor_shim_exports():
  """All symbols in interactions_processor shim match context._interactions."""
  assert (
      shim_interactions.request_processor is ctx_interactions.request_processor
  )
  assert (
      shim_interactions.InteractionsRequestProcessor
      is ctx_interactions.InteractionsRequestProcessor
  )
  assert (
      shim_interactions._is_event_in_branch
      is ctx_interactions._is_event_in_branch
  )
  assert (
      shim_interactions._find_previous_interaction_state
      is ctx_interactions._find_previous_interaction_state
  )
  assert shim_interactions.logger is ctx_interactions.logger


def test_package_level_imports():
  """Modules are accessible as attributes on google.adk.flows.llm_flows."""
  assert hasattr(llm_flows, 'contents')
  assert hasattr(llm_flows, 'context')
  assert hasattr(llm_flows.context, '_cache')
  assert hasattr(llm_flows.context, '_compaction')
  assert hasattr(llm_flows.context, '_contents')
  assert hasattr(llm_flows.context, '_interactions')
