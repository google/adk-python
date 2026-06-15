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
from google.adk.a2a.converters.to_adk_event import convert_a2a_artifact_update_to_event
from google.adk.a2a.converters.to_adk_event import convert_a2a_status_update_to_event
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
