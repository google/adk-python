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

from __future__ import annotations

from unittest.mock import Mock

from a2a.types import Part as A2APart
from a2a.types import TaskArtifactUpdateEvent
from a2a.types import TaskStatusUpdateEvent
from google.adk.a2a.converters.from_adk_event import convert_event_to_a2a_events
from google.adk.events import event_actions
from google.adk.events.event import Event
from google.genai import types as genai_types
import pytest


class TestFromAdk:
  """Test suite for from_adk functions."""

  def setup_method(self):
    """Set up test fixtures."""
    self.mock_event = Mock(spec=Event)
    self.mock_event.id = "test-event-id"
    self.mock_event.invocation_id = "test-invocation-id"
    self.mock_event.author = "test-author"
    self.mock_event.branch = None
    self.mock_event.content = None
    self.mock_event.error_code = None
    self.mock_event.error_message = None
    self.mock_event.grounding_metadata = None
    self.mock_event.citation_metadata = None
    self.mock_event.custom_metadata = None
    self.mock_event.usage_metadata = None
    self.mock_event.actions = None
    self.mock_event.partial = True
    self.mock_event.long_running_tool_ids = None

  def test_convert_event_to_a2a_events_artifact_update(self):
    """Event with content produces a TaskArtifactUpdateEvent."""
    self.mock_event.content = genai_types.Content(
        parts=[genai_types.Part(text="hello")], role="model"
    )
    self.mock_event.author = "agent-1"

    agents_artifacts = {}
    mock_a2a_part = A2APart(text="hello")
    mock_convert_part = Mock(return_value=[mock_a2a_part])

    result = convert_event_to_a2a_events(
        self.mock_event,
        agents_artifacts,
        task_id="task-123",
        context_id="context-456",
        part_converter=mock_convert_part,
    )

    assert len(result) == 1
    assert isinstance(result[0], TaskArtifactUpdateEvent)
    assert result[0].task_id == "task-123"
    assert result[0].context_id == "context-456"
    assert "agent-1" in agents_artifacts

  def test_convert_event_to_a2a_events_error(self):
    """Event with error_code produces no events (error is handled separately)."""
    self.mock_event.error_code = "ERR001"
    self.mock_event.error_message = "Something went wrong"

    result = convert_event_to_a2a_events(
        self.mock_event,
        {},
        task_id="task-123",
        context_id="context-456",
    )

    assert len(result) == 0

  def test_convert_event_to_a2a_events_none_event(self):
    """None event raises ValueError."""
    with pytest.raises(ValueError, match="Event cannot be None"):
      convert_event_to_a2a_events(None, {})

  def test_convert_event_to_a2a_events_none_artifacts(self):
    """None agents_artifacts raises ValueError."""
    with pytest.raises(ValueError, match="Agents artifacts cannot be None"):
      convert_event_to_a2a_events(self.mock_event, None)

  def test_convert_event_to_a2a_events_with_actions(self):
    """Event with actions but no content produces a TaskStatusUpdateEvent."""
    self.mock_event.actions = event_actions.EventActions()
    self.mock_event.actions.artifact_delta["image"] = 0

    result = convert_event_to_a2a_events(
        self.mock_event,
        {},
        task_id="task-123",
        context_id="context-456",
    )

    assert len(result) == 1
    assert isinstance(result[0], TaskStatusUpdateEvent)
    assert result[0].task_id == "task-123"
    assert result[0].context_id == "context-456"
