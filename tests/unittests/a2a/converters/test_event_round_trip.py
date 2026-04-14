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

"""Round trip tests for ADK and A2A event converters."""

from __future__ import annotations

from typing import Dict
from unittest.mock import Mock

from a2a.types import TaskArtifactUpdateEvent
from a2a.types import TaskStatusUpdateEvent
from google.adk.a2a.converters.from_adk_event import convert_event_to_a2a_events
from google.adk.a2a.converters.from_adk_event import create_error_status_event
from google.adk.a2a.converters.to_adk_event import _parse_adk_metadata_value
from google.adk.a2a.converters.to_adk_event import convert_a2a_artifact_update_to_event
from google.adk.a2a.converters.to_adk_event import convert_a2a_status_update_to_event
from google.adk.a2a.converters.utils import _get_adk_metadata_key
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.genai import types as genai_types


def test_round_trip_text_event():
  original_event = Event(
      invocation_id="test_invocation",
      author="test_agent",
      branch="main",
      content=genai_types.Content(
          role="model",
          parts=[genai_types.Part.from_text(text="Hello world!")],
      ),
      partial=False,
  )
  agents_artifacts: Dict[str, str] = {}

  a2a_events = convert_event_to_a2a_events(
      event=original_event,
      agents_artifacts=agents_artifacts,
      task_id="task1",
      context_id="context1",
  )

  assert len(a2a_events) == 1
  a2a_event = a2a_events[0]
  assert isinstance(a2a_event, TaskArtifactUpdateEvent)

  mock_context = Mock(
      spec=InvocationContext, invocation_id="test_invocation", branch="main"
  )

  restored_event = convert_a2a_artifact_update_to_event(
      a2a_artifact_update=a2a_event,
      author="test_agent",
      invocation_context=mock_context,
  )

  assert restored_event is not None
  assert restored_event.author == original_event.author
  assert restored_event.invocation_id == original_event.invocation_id
  assert restored_event.branch == original_event.branch
  assert restored_event.partial == original_event.partial
  assert len(restored_event.content.parts) == len(original_event.content.parts)
  assert (
      restored_event.content.parts[0].text
      == original_event.content.parts[0].text
  )


def test_round_trip_error_status_event():
  original_event = Event(
      invocation_id="error_inv",
      author="error_agent",
      branch="main",
      error_message="Test Error",
  )

  a2a_event = create_error_status_event(
      event=original_event,
      task_id="task2",
      context_id="ctx2",
  )

  assert isinstance(a2a_event, TaskStatusUpdateEvent)

  mock_context = Mock(
      spec=InvocationContext, invocation_id="error_inv", branch="main"
  )

  restored_event = convert_a2a_status_update_to_event(
      a2a_status_update=a2a_event,
      author="error_agent",
      invocation_context=mock_context,
  )

  assert restored_event is not None
  assert restored_event.author == original_event.author
  assert restored_event.invocation_id == original_event.invocation_id
  assert restored_event.branch == original_event.branch
  assert len(restored_event.content.parts) == 1
  assert restored_event.content.parts[0].text == "Test Error"


def test_round_trip_function_call_event():
  original_event = Event(
      invocation_id="test_invocation",
      author="test_agent",
      branch="main",
      content=genai_types.Content(
          role="model",
          parts=[
              genai_types.Part.from_function_call(
                  name="my_function",
                  args={"arg1": "value1"},
              )
          ],
      ),
      partial=False,
  )
  agents_artifacts: Dict[str, str] = {}

  a2a_events = convert_event_to_a2a_events(
      event=original_event,
      agents_artifacts=agents_artifacts,
      task_id="task1",
      context_id="context1",
  )

  assert len(a2a_events) == 1
  a2a_event = a2a_events[0]

  mock_context = Mock(
      spec=InvocationContext, invocation_id="test_invocation", branch="main"
  )

  restored_event = convert_a2a_artifact_update_to_event(
      a2a_artifact_update=a2a_event,
      author="test_agent",
      invocation_context=mock_context,
  )

  assert restored_event is not None
  assert restored_event.author == original_event.author
  assert restored_event.invocation_id == original_event.invocation_id
  assert restored_event.branch == original_event.branch
  assert len(restored_event.content.parts) == 1
  assert restored_event.content.parts[0].function_call.name == "my_function"
  assert restored_event.content.parts[0].function_call.args == {
      "arg1": "value1"
  }


def test_round_trip_function_response_event():
  original_event = Event(
      invocation_id="test_invocation",
      author="test_agent",
      branch="main",
      content=genai_types.Content(
          role="user",
          parts=[
              genai_types.Part.from_function_response(
                  name="my_function",
                  response={"result": "success"},
              )
          ],
      ),
      partial=False,
  )
  agents_artifacts: Dict[str, str] = {}

  a2a_events = convert_event_to_a2a_events(
      event=original_event,
      agents_artifacts=agents_artifacts,
      task_id="task1",
      context_id="context1",
  )

  assert len(a2a_events) == 1
  a2a_event = a2a_events[0]

  mock_context = Mock(
      spec=InvocationContext, invocation_id="test_invocation", branch="main"
  )

  restored_event = convert_a2a_artifact_update_to_event(
      a2a_artifact_update=a2a_event,
      author="test_agent",
      invocation_context=mock_context,
  )

  assert restored_event is not None
  assert restored_event.author == original_event.author
  assert restored_event.invocation_id == original_event.invocation_id
  assert restored_event.branch == original_event.branch
  assert len(restored_event.content.parts) == 1
  assert restored_event.content.parts[0].function_response.name == "my_function"
  assert restored_event.content.parts[0].function_response.response == {
      "result": "success"
  }


def test_round_trip_custom_metadata_preserves_structured_values():
  original_custom_metadata = {
      "flag": True,
      "count": 42,
      "nested": {"key": "val"},
      "tags": ["a", "b"],
  }
  original_event = Event(
      invocation_id="test_invocation",
      author="test_agent",
      branch="main",
      content=genai_types.Content(
          role="model",
          parts=[genai_types.Part.from_text(text="Hello world!")],
      ),
      custom_metadata=original_custom_metadata,
  )
  agents_artifacts: Dict[str, str] = {}

  a2a_events = convert_event_to_a2a_events(
      event=original_event,
      agents_artifacts=agents_artifacts,
      task_id="task1",
      context_id="context1",
  )

  assert len(a2a_events) == 1
  a2a_event = a2a_events[0]
  assert isinstance(a2a_event, TaskArtifactUpdateEvent)

  serialized_metadata = a2a_event.artifact.metadata[
      _get_adk_metadata_key("custom_metadata")
  ]

  assert not isinstance(serialized_metadata, str)
  assert (
      _parse_adk_metadata_value(serialized_metadata) == original_custom_metadata
  )


def test_serialize_value_handles_non_serializable_nested_types():
  """Regression: non-JSON-native types inside dicts/lists must not crash."""
  from datetime import datetime
  from datetime import timezone

  from google.adk.a2a.converters.from_adk_event import _serialize_value

  ts = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
  value = {
      "created_at": ts,
      "tags": {"alpha", "beta"},
      "normal": 42,
      "nested_list": [True, ts],
  }

  result = _serialize_value(value)

  # Result must be fully JSON-serializable (no crash)
  import json

  json_str = json.dumps(result)
  parsed = json.loads(json_str)

  # Leaf types preserved
  assert parsed["normal"] == 42

  # datetime falls back to str representation
  assert isinstance(parsed["created_at"], str)
  assert "2026" in parsed["created_at"]

  # set becomes a sorted list of strings
  assert isinstance(parsed["tags"], list)
  assert set(parsed["tags"]) == {"alpha", "beta"}

  # nested list with mixed types
  assert parsed["nested_list"][0] is True
  assert isinstance(parsed["nested_list"][1], str)
