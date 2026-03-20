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

from a2a.types import Artifact
from a2a.types import Message
from a2a.types import Part as A2APart
from a2a.types import Task
from a2a.types import TaskArtifactUpdateEvent
from a2a.types import TaskState
from a2a.types import TaskStatus
from a2a.types import TaskStatusUpdateEvent
from a2a.types import TextPart
from google.adk.a2a.converters.part_converter import A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY
from google.adk.a2a.converters.to_adk_event import convert_a2a_artifact_update_to_event
from google.adk.a2a.converters.to_adk_event import convert_a2a_message_to_event
from google.adk.a2a.converters.to_adk_event import convert_a2a_status_update_to_event
from google.adk.a2a.converters.to_adk_event import convert_a2a_task_to_event
from google.adk.a2a.converters.utils import _get_adk_metadata_key
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types as genai_types
import pytest


class TestToAdk:
  """Test suite for to_adk functions."""

  def setup_method(self):
    """Set up test fixtures."""
    self.mock_context = Mock(spec=InvocationContext)
    self.mock_context.invocation_id = "test-invocation"
    self.mock_context.branch = "test-branch"

  def test_convert_a2a_message_to_event_success(self):
    """Test successful conversion of A2A message to Event."""
    a2a_part = Mock(spec=A2APart)
    a2a_part.root = Mock()
    a2a_part.root.metadata = {}
    message = Message(message_id="msg-1", role="user", parts=[a2a_part])

    mock_genai_part = genai_types.Part.from_text(text="hello")
    mock_part_converter = Mock(return_value=[mock_genai_part])

    event = convert_a2a_message_to_event(
        message,
        author="test-author",
        invocation_context=self.mock_context,
        part_converter=mock_part_converter,
    )

    assert event.author == "test-author"
    assert event.invocation_id == "test-invocation"
    assert event.branch == "test-branch"
    assert len(event.content.parts) == 1
    assert event.content.parts[0] == mock_genai_part

  def test_convert_a2a_message_to_event_none(self):
    """Test convert_a2a_message_to_event with None."""
    with pytest.raises(ValueError, match="A2A message cannot be None"):
      convert_a2a_message_to_event(None)

  def test_convert_a2a_task_to_event_success(self):
    """Test successful conversion of A2A task to Event."""
    a2a_part = Mock(spec=A2APart)
    a2a_part.root = Mock()
    a2a_part.root.metadata = {}
    task = Task(
        id="task-1",
        status=TaskStatus(
            state=TaskState.submitted, timestamp="2024-01-01T00:00:00Z"
        ),
        context_id="context-1",
        history=[Message(message_id="msg-1", role="agent", parts=[a2a_part])],
        artifacts=[
            Artifact(
                artifact_id="art-1", artifact_type="message", parts=[a2a_part]
            )
        ],
    )

    mock_genai_part = genai_types.Part.from_text(text="task artifact text")
    mock_part_converter = Mock(return_value=[mock_genai_part])

    event = convert_a2a_task_to_event(
        task,
        author="test-author",
        invocation_context=self.mock_context,
        part_converter=mock_part_converter,
    )

    assert event.author == "test-author"
    assert event.invocation_id == "test-invocation"
    assert len(event.content.parts) == 1
    assert event.content.parts[0] == mock_genai_part

  def test_convert_a2a_task_to_event_none(self):
    """Test convert_a2a_task_to_event with None."""
    with pytest.raises(ValueError, match="A2A task cannot be None"):
      convert_a2a_task_to_event(None)

  def test_convert_a2a_status_update_to_event_success(self):
    """Test successful conversion of A2A status update to Event."""
    a2a_part = Mock(spec=A2APart)
    a2a_part.root = Mock()
    a2a_part.root.metadata = {
        _get_adk_metadata_key(A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY): True
    }
    update = TaskStatusUpdateEvent(
        task_id="task-1",
        status=TaskStatus(
            state=TaskState.input_required,
            timestamp="now",
            message=Message(
                message_id="m1",
                role="agent",
                parts=[a2a_part],
            ),
        ),
        context_id="context-1",
        final=False,
    )

    mock_genai_part = genai_types.Part(
        function_call=genai_types.FunctionCall(
            name="status update text", args={"arg": "value"}, id="call-1"
        )
    )
    mock_part_converter = Mock(return_value=[mock_genai_part])

    event = convert_a2a_status_update_to_event(
        update,
        author="test-author",
        invocation_context=self.mock_context,
        part_converter=mock_part_converter,
    )

    assert event.author == "test-author"
    assert event.invocation_id == "test-invocation"
    assert len(event.content.parts) == 1
    assert event.content.parts[0] == mock_genai_part

  def test_convert_a2a_status_update_to_event_none(self):
    """Test convert_a2a_status_update_to_event with None."""
    with pytest.raises(ValueError, match="A2A status update cannot be None"):
      convert_a2a_status_update_to_event(None)

  def test_convert_a2a_artifact_update_to_event_success(self):
    """Test successful conversion of A2A artifact update to Event."""
    a2a_part = Mock(spec=A2APart)
    a2a_part.root = Mock()
    a2a_part.root.metadata = {}
    update = TaskArtifactUpdateEvent(
        task_id="task-1",
        artifact=Artifact(
            artifact_id="art-1", artifact_type="message", parts=[a2a_part]
        ),
        append=True,
        context_id="context-1",
        last_chunk=False,
    )

    mock_genai_part = genai_types.Part.from_text(text="artifact chunk text")
    mock_part_converter = Mock(return_value=[mock_genai_part])

    event = convert_a2a_artifact_update_to_event(
        update,
        author="test-author",
        invocation_context=self.mock_context,
        part_converter=mock_part_converter,
    )

    assert event.author == "test-author"
    assert event.invocation_id == "test-invocation"
    assert event.partial is True
    assert len(event.content.parts) == 1
    assert event.content.parts[0] == mock_genai_part

  def test_convert_a2a_artifact_update_to_event_none(self):
    """Test convert_a2a_artifact_update_to_event with None."""
    with pytest.raises(ValueError, match="A2A artifact update cannot be None"):
      convert_a2a_artifact_update_to_event(None)
