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

"""Tests ensuring backward-compatibility shims for extensions export expected symbols."""

from google.adk.flows import llm_flows
from google.adk.flows.llm_flows import _code_execution as shim_code_execution
from google.adk.flows.llm_flows import _nl_planning as shim_planning
from google.adk.flows.llm_flows import agent_transfer as shim_agent_transfer
from google.adk.flows.llm_flows.extensions import _agent_transfer as ext_agent_transfer
from google.adk.flows.llm_flows.extensions import _code_execution as ext_code_execution
from google.adk.flows.llm_flows.extensions import _planning as ext_planning


def test_agent_transfer_shim_exports():
  """All symbols in agent_transfer shim match extensions._agent_transfer."""
  assert (
      shim_agent_transfer.request_processor
      is ext_agent_transfer.request_processor
  )
  assert (
      shim_agent_transfer._get_transfer_targets
      is ext_agent_transfer._get_transfer_targets
  )
  assert (
      shim_agent_transfer._build_transfer_instruction_body
      is ext_agent_transfer._build_transfer_instruction_body
  )
  assert (
      shim_agent_transfer._build_transfer_instructions
      is ext_agent_transfer._build_transfer_instructions
  )
  assert (
      shim_agent_transfer._build_transfer_tool
      is ext_agent_transfer._build_transfer_tool
  )
  assert (
      shim_agent_transfer._get_incompatible_builtin_tool_error
      is ext_agent_transfer._get_incompatible_builtin_tool_error
  )
  assert (
      shim_agent_transfer._AgentTransferLlmRequestProcessor
      is ext_agent_transfer._AgentTransferLlmRequestProcessor
  )
  assert shim_agent_transfer._AgentLike is ext_agent_transfer._AgentLike
  assert (
      shim_agent_transfer._build_target_agents_info
      is ext_agent_transfer._build_target_agents_info
  )
  assert shim_agent_transfer.line_break == '\n'


def test_code_execution_shim_exports():
  """All symbols in _code_execution shim match extensions._code_execution."""
  assert (
      shim_code_execution.request_processor
      is ext_code_execution.request_processor
  )
  assert (
      shim_code_execution.response_processor
      is ext_code_execution.response_processor
  )
  assert (
      shim_code_execution.get_content_as_bytes
      is ext_code_execution.get_content_as_bytes
  )
  assert (
      shim_code_execution._DATA_FILE_HELPER_LIB
      is ext_code_execution._DATA_FILE_HELPER_LIB
  )
  assert (
      shim_code_execution._DATA_FILE_UTIL_MAP
      is ext_code_execution._DATA_FILE_UTIL_MAP
  )
  assert (
      shim_code_execution._NON_BUILTIN_EXECUTOR_INSTRUCTION
      is ext_code_execution._NON_BUILTIN_EXECUTOR_INSTRUCTION
  )
  assert shim_code_execution.DataFileUtil is ext_code_execution.DataFileUtil
  assert (
      shim_code_execution._extract_and_replace_inline_files
      is ext_code_execution._extract_and_replace_inline_files
  )
  assert (
      shim_code_execution._get_data_file_preprocessing_code
      is ext_code_execution._get_data_file_preprocessing_code
  )
  assert (
      shim_code_execution._get_or_set_execution_id
      is ext_code_execution._get_or_set_execution_id
  )
  assert (
      shim_code_execution._post_process_code_execution_result
      is ext_code_execution._post_process_code_execution_result
  )
  assert (
      shim_code_execution._run_post_processor
      is ext_code_execution._run_post_processor
  )
  assert (
      shim_code_execution._run_pre_processor
      is ext_code_execution._run_pre_processor
  )
  assert shim_code_execution.logger is ext_code_execution.logger


def test_planning_shim_exports():
  """All symbols in _nl_planning shim match extensions._planning."""
  assert shim_planning.request_processor is ext_planning.request_processor
  assert shim_planning.response_processor is ext_planning.response_processor
  assert shim_planning._get_planner is ext_planning._get_planner
  assert (
      shim_planning._remove_thought_from_request
      is ext_planning._remove_thought_from_request
  )
  assert (
      shim_planning._NlPlanningRequestProcessor
      is ext_planning._NlPlanningRequestProcessor
  )
  assert shim_planning._NlPlanningResponse is ext_planning._NlPlanningResponse


def test_package_level_imports():
  """Modules are accessible as attributes on google.adk.flows.llm_flows."""
  assert hasattr(llm_flows, '_code_execution')
  assert hasattr(llm_flows, '_nl_planning')
  assert hasattr(llm_flows, 'extensions')
  assert hasattr(llm_flows.extensions, '_agent_transfer')
  assert hasattr(llm_flows.extensions, '_code_execution')
  assert hasattr(llm_flows.extensions, '_planning')
