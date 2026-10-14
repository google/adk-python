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

"""Tests for google.adk.utils._function_call_names."""

from __future__ import annotations

from google.adk.flows.llm_flows import functions as llm_functions
from google.adk.flows.llm_flows.tools import _functions as tools_functions
from google.adk.utils._function_call_names import AF_FUNCTION_CALL_ID_PREFIX
from google.adk.utils._function_call_names import CLIENT_FUNCTION_CALL_NAMES
from google.adk.utils._function_call_names import REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
from google.adk.utils._function_call_names import REQUEST_EUC_FUNCTION_CALL_NAME
from google.adk.utils._function_call_names import REQUEST_INPUT_FUNCTION_CALL_NAME
from google.adk.workflow.utils import _workflow_hitl_utils


def test_function_call_name_constants():
  assert AF_FUNCTION_CALL_ID_PREFIX == 'adk-'
  assert REQUEST_EUC_FUNCTION_CALL_NAME == 'adk_request_credential'
  assert REQUEST_CONFIRMATION_FUNCTION_CALL_NAME == 'adk_request_confirmation'
  assert REQUEST_INPUT_FUNCTION_CALL_NAME == 'adk_request_input'
  assert CLIENT_FUNCTION_CALL_NAMES == frozenset({
      'adk_request_credential',
      'adk_request_confirmation',
      'adk_request_input',
  })


def test_backward_compatible_reexports():
  assert (
      tools_functions.AF_FUNCTION_CALL_ID_PREFIX == AF_FUNCTION_CALL_ID_PREFIX
  )
  assert (
      tools_functions.REQUEST_EUC_FUNCTION_CALL_NAME
      == REQUEST_EUC_FUNCTION_CALL_NAME
  )
  assert (
      tools_functions.REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
      == REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
  )
  assert (
      tools_functions.REQUEST_INPUT_FUNCTION_CALL_NAME
      == REQUEST_INPUT_FUNCTION_CALL_NAME
  )

  assert llm_functions.AF_FUNCTION_CALL_ID_PREFIX == AF_FUNCTION_CALL_ID_PREFIX
  assert (
      llm_functions.REQUEST_EUC_FUNCTION_CALL_NAME
      == REQUEST_EUC_FUNCTION_CALL_NAME
  )
  assert (
      llm_functions.REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
      == REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
  )
  assert (
      llm_functions.REQUEST_INPUT_FUNCTION_CALL_NAME
      == REQUEST_INPUT_FUNCTION_CALL_NAME
  )

  assert (
      _workflow_hitl_utils.REQUEST_EUC_FUNCTION_CALL_NAME
      == REQUEST_EUC_FUNCTION_CALL_NAME
  )
  assert (
      _workflow_hitl_utils.REQUEST_INPUT_FUNCTION_CALL_NAME
      == REQUEST_INPUT_FUNCTION_CALL_NAME
  )
