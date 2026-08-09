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

from google.adk.a2a import _compat
from google.adk.a2a.converters.long_running_functions import LongRunningFunctions
from google.adk.a2a.converters.part_converter import A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY
from google.adk.a2a.converters.utils import _get_adk_metadata_key
from google.adk.events.event import Event
from google.genai import types


def test_default_converter_returns_a2a_long_running_function_call():
  """The default converter must translate GenAI parts into A2A parts."""
  function_call = types.Part(
      function_call=types.FunctionCall(
          id="call-1", name="request_approval", args={}
      )
  )
  event = Event(
      invocation_id="invocation-1",
      author="agent",
      content=types.Content(role="model", parts=[function_call]),
      long_running_tool_ids={"call-1"},
  )
  long_running_functions = LongRunningFunctions()

  processed_event = long_running_functions.process_event(event)
  result = long_running_functions.create_long_running_function_call_event(
      "task-1", "context-1"
  )

  assert processed_event.content is not None
  assert processed_event.content.parts == []
  assert result is not None
  assert result.status.state == _compat.TS_INPUT_REQUIRED
  assert result.status.message is not None
  result_part = result.status.message.parts[0]
  assert _compat.is_data_part(result_part)
  assert (
      _compat.part_metadata(result_part)[
          _get_adk_metadata_key(A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY)
      ]
      is True
  )
