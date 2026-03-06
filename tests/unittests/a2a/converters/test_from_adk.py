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
from unittest.mock import patch
import uuid

from a2a.types import Part as A2APart
from a2a.types import TaskArtifactUpdateEvent
from a2a.types import TaskState
from a2a.types import TaskStatusUpdateEvent
from a2a.types import TextPart
from google.adk.a2a.converters.from_adk_event import convert_event_to_a2a_events
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
    """Test conversion of event to TaskArtifactUpdateEvent."""
    # Setup event with content
    self.mock_event.content = genai_types.Content(
        parts=[genai_types.Part(text="hello")], role="model"
    )
    self.mock_event.author = "agent-1"

    agents_artifacts = {}

    # Mock part converter to return a standard text part
    mock_a2a_part = A2APart(root=TextPart(text="hello"))
    mock_a2a_part.root.metadata = {}
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
    assert result[0].artifact.parts == [mock_a2a_part]
    assert "agent-1" in agents_artifacts  # Artifact ID should be stored

  def test_convert_event_to_a2a_events_error(self):
    """Test conversion of event with error to TaskStatusUpdateEvent."""
    self.mock_event.error_code = "ERR001"
    self.mock_event.error_message = "Something went wrong"

    agents_artifacts = {}

    result = convert_event_to_a2a_events(
        self.mock_event,
        agents_artifacts,
        task_id="task-123",
        context_id="context-456",
    )

    # Should not return any artifact events
    assert len(result) == 0

  def test_convert_event_to_a2a_events_none_event(self):
    """Test convert_event_to_a2a_events with None event."""
    with pytest.raises(ValueError, match="Event cannot be None"):
      convert_event_to_a2a_events(None, {})

  def test_convert_event_to_a2a_events_none_artifacts(self):
    """Test convert_event_to_a2a_events with None agents_artifacts."""
    with pytest.raises(ValueError, match="Agents artifacts cannot be None"):
      convert_event_to_a2a_events(self.mock_event, None)
