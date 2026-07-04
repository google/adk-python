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

from a2a.types import TaskState
from google.adk.a2a.converters.long_running_functions import LongRunningFunctions
from google.adk.a2a.converters.part_converter import convert_genai_part_to_a2a_part
from google.adk.flows.llm_flows.functions import REQUEST_EUC_FUNCTION_CALL_NAME
from google.genai import types as genai_types


def _long_running_call_part(function_name: str):
  """Builds the A2A DataPart produced for a long-running function call."""
  genai_part = genai_types.Part(
      function_call=genai_types.FunctionCall(
          id="call-1", name=function_name, args={}
      )
  )
  return convert_genai_part_to_a2a_part(genai_part)


class TestMarkLongRunningFunctionCall:
  """Test cases for LongRunningFunctions._mark_long_running_function_call."""

  def test_euc_request_maps_to_auth_required(self):
    """An EUC (credential) request must produce the auth_required state.

    The function name lives in the DataPart's ``data`` (the FunctionCall
    dump), not in ``metadata`` -- reading it from ``metadata`` always
    yielded ``None`` and mislabeled the task as input_required.
    """
    lrf = LongRunningFunctions()

    lrf._mark_long_running_function_call(
        _long_running_call_part(REQUEST_EUC_FUNCTION_CALL_NAME)
    )

    assert lrf._task_state == TaskState.auth_required

  def test_regular_function_maps_to_input_required(self):
    """A non-EUC long-running function stays in the input_required state."""
    lrf = LongRunningFunctions()

    lrf._mark_long_running_function_call(_long_running_call_part("do_thing"))

    assert lrf._task_state == TaskState.input_required
