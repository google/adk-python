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

from datetime import datetime
from datetime import timezone
from unittest.mock import Mock

from a2a.types import Artifact
from a2a.types import Message
from a2a.types import Part as A2APart
from a2a.types import Role
from a2a.types import Task
from a2a.types import TaskArtifactUpdateEvent
from a2a.types import TaskState
from a2a.types import TaskStatus
from a2a.types import TaskStatusUpdateEvent
from google.adk.a2a.converters.part_converter import A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY
from google.adk.a2a.converters.to_adk_event import convert_a2a_artifact_update_to_event
from google.adk.a2a.converters.to_adk_event import convert_a2a_message_to_event
from google.adk.a2a.converters.to_adk_event import convert_a2a_status_update_to_event
from google.adk.a2a.converters.to_adk_event import convert_a2a_task_to_event
from google.adk.a2a.converters.to_adk_event import MOCK_FUNCTION_CALL_FOR_REQUIRED_USER_AUTH
from google.adk.a2a.converters.to_adk_event import MOCK_FUNCTION_CALL_FOR_REQUIRED_USER_INPUT
from google.adk.a2a.converters.utils import _get_adk_metadata_key
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types as genai_types
import pytest


def _make_task_status(state: TaskState) -> TaskStatus:
  """Helper to create a TaskStatus with the given state."""
  status = TaskStatus(state=state)
  status.timestamp.FromDatetime(datetime.now(timezone.utc))
  return status


class TestToAdk:
  """Test suite for to_adk functions."""

  def setup_method(self):
    """Set up test fixtures."""
    self.mock_context = Mock(spec=InvocationContext)
    self.mock_context.invocation_id = "test-invocation"
    self.mock_context.branch = "test-branch"

  def test_convert_a2a_message_to_event_success(self):
    """A2A message with parts converts to event with those parts."""
    a2a_part = A2APart(text="hello source")
    message = Message(message_id="msg-1", role=Role.ROLE_USER, parts=[a2a_part])

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
    """None message raises ValueError."""
    with pytest.raises(ValueError, match="A2A message cannot be None"):
      convert_a2a_message_to_event(None)

  def test_convert_a2a_message_to_event_restores_actions_from_metadata(self):
    """Actions in message metadata are restored into the event."""
    message = Message(
        message_id="msg-1",
        role=Role.ROLE_USER,
        parts=[A2APart(text="hello")],
    )
    message.metadata[_get_adk_metadata_key("actions")] = {
        "stateDelta": {"saved_key": "saved-value"}
    }

    mock_genai_part = genai_types.Part.from_text(text="hello")
    mock_part_converter = Mock(return_value=[mock_genai_part])

    event = convert_a2a_message_to_event(
        message,
        author="test-author",
        invocation_context=self.mock_context,
        part_converter=mock_part_converter,
    )

    assert event.actions.state_delta == {"saved_key": "saved-value"}
    assert event.content is not None
    assert event.content.parts[0] == mock_genai_part

  def test_convert_a2a_message_to_event_returns_action_only_event(self):
    """Message with no parts but actions metadata produces an action-only event."""
    message = Message(message_id="msg-1", role=Role.ROLE_USER, parts=[])
    message.metadata[_get_adk_metadata_key("actions")] = {
        "stateDelta": {"saved_key": "saved-value"}
    }

    event = convert_a2a_message_to_event(
        message,
        author="test-author",
        invocation_context=self.mock_context,
        part_converter=Mock(),
    )

    assert event is not None
    assert event.actions.state_delta == {"saved_key": "saved-value"}
    assert event.content is None

  def test_convert_a2a_task_to_event_success(self):
    """Task with artifact parts converts to event with those parts."""
    a2a_part = A2APart(text="task text")
    artifact = Artifact(artifact_id="art-1", parts=[a2a_part])
    task = Task(
        id="task-1",
        context_id="context-1",
        artifacts=[artifact],
    )
    task.status.CopyFrom(_make_task_status(TaskState.TASK_STATE_SUBMITTED))
    task.history.append(Message(message_id="msg-1", role=Role.ROLE_AGENT))

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

  def test_convert_a2a_task_to_event_returns_action_only_event(self):
    """Task artifact with actions metadata produces an action-only event."""
    artifact = Artifact(artifact_id="art-1", parts=[])
    artifact.metadata[_get_adk_metadata_key("actions")] = {
        "stateDelta": {"saved_key": "saved-value"}
    }
    task = Task(id="task-1", context_id="context-1", artifacts=[artifact])
    task.status.CopyFrom(_make_task_status(TaskState.TASK_STATE_SUBMITTED))

    event = convert_a2a_task_to_event(
        task,
        author="test-author",
        invocation_context=self.mock_context,
        part_converter=Mock(),
    )

    assert event is not None
    assert event.actions.state_delta == {"saved_key": "saved-value"}
    assert event.content is None

  def test_convert_a2a_task_to_event_merges_actions_across_artifacts(self):
    """Actions are merged across multiple artifact metadata entries."""
    art1 = Artifact(artifact_id="art-1", parts=[])
    art1.metadata[_get_adk_metadata_key("actions")] = {
        "stateDelta": {"first_key": "first-value"}
    }
    art2 = Artifact(artifact_id="art-2", parts=[])

    task = Task(id="task-1", context_id="context-1", artifacts=[art1, art2])
    task.status.CopyFrom(_make_task_status(TaskState.TASK_STATE_SUBMITTED))

    event = convert_a2a_task_to_event(
        task,
        author="test-author",
        invocation_context=self.mock_context,
        part_converter=Mock(),
    )

    assert event is not None
    assert event.actions.state_delta == {"first_key": "first-value"}
    assert event.content is None

  def test_convert_a2a_task_to_event_overwrites_nested_state_delta_values(self):
    """Later artifact metadata overwrites earlier ones at the top level."""
    art1 = Artifact(artifact_id="art-1", parts=[])
    art1.metadata[_get_adk_metadata_key("actions")] = {
        "stateDelta": {"settings": {"theme": "light", "language": "en"}}
    }
    art2 = Artifact(artifact_id="art-2", parts=[])
    art2.metadata[_get_adk_metadata_key("actions")] = {
        "stateDelta": {"settings": {"theme": "dark"}}
    }

    task = Task(id="task-1", context_id="context-1", artifacts=[art1, art2])
    task.status.CopyFrom(_make_task_status(TaskState.TASK_STATE_SUBMITTED))

    event = convert_a2a_task_to_event(
        task,
        author="test-author",
        invocation_context=self.mock_context,
        part_converter=Mock(),
    )

    assert event is not None
    assert event.actions.state_delta == {"settings": {"theme": "dark"}}
    assert event.content is None

  def test_convert_a2a_task_to_event_merges_status_and_artifact_actions(self):
    """Actions from artifact metadata and status message metadata are merged."""
    art = Artifact(artifact_id="art-1", parts=[])
    art.metadata[_get_adk_metadata_key("actions")] = {
        "stateDelta": {"saved_key": "saved-value"}
    }

    status_msg = Message(message_id="msg-1", role=Role.ROLE_AGENT, parts=[A2APart(text="need input")])
    status_msg.metadata[_get_adk_metadata_key("actions")] = {
        "transferToAgent": "agent-2"
    }
    status = TaskStatus(state=TaskState.TASK_STATE_INPUT_REQUIRED, message=status_msg)
    status.timestamp.FromDatetime(datetime.now(timezone.utc))

    task = Task(id="task-1", context_id="context-1", artifacts=[art])
    task.status.CopyFrom(status)

    mock_genai_part = genai_types.Part.from_text(text="need input")

    event = convert_a2a_task_to_event(
        task,
        author="test-author",
        invocation_context=self.mock_context,
        part_converter=Mock(return_value=[mock_genai_part]),
    )

    assert event is not None
    assert event.actions.state_delta == {"saved_key": "saved-value"}
    assert event.actions.transfer_to_agent == "agent-2"
    assert event.content is not None
    assert (
        event.content.parts[0].function_call.name
        == MOCK_FUNCTION_CALL_FOR_REQUIRED_USER_INPUT
    )

  def test_convert_a2a_task_to_event_auth_required_uses_auth_args_key(self):
    """Test auth-required state populates the function call with auth args."""
    a2a_part = A2APart(text="need auth")
    task = Task(
        id="task-1",
        context_id="context-1",
        status=TaskStatus(
            state=TaskState.TASK_STATE_AUTH_REQUIRED,
            message=Message(
                message_id="m1",
                role=Role.ROLE_AGENT,
                parts=[a2a_part],
            ),
        ),
    )

    mock_genai_part = genai_types.Part.from_text(text="need auth")

    event = convert_a2a_task_to_event(
        task,
        author="test-author",
        invocation_context=self.mock_context,
        part_converter=Mock(return_value=[mock_genai_part]),
    )

    assert event is not None
    assert event.content is not None
    assert (
        event.content.parts[0].function_call.name
        == MOCK_FUNCTION_CALL_FOR_REQUIRED_USER_AUTH
    )
    # auth_required state should populate the auth_required arg key, not
    # input_required.
    assert (
        event.content.parts[0].function_call.args["auth_required"]
        == "need auth"
    )
    assert "input_required" not in event.content.parts[0].function_call.args

  def test_convert_a2a_task_to_event_multiple_parts_replaces_last_text(self):
    """input_required with multiple parts injects mock function call for the last text part."""
    status_msg = Message(
        message_id="m1",
        role=Role.ROLE_AGENT,
        parts=[A2APart(text="part1"), A2APart(text="part2")],
    )
    status = TaskStatus(
        state=TaskState.TASK_STATE_INPUT_REQUIRED, message=status_msg
    )
    status.timestamp.FromDatetime(datetime.now(timezone.utc))

    task = Task(id="task-1", context_id="context-1")
    task.status.CopyFrom(status)

    mock_genai_part_1 = genai_types.Part.from_text(text="Part 1")
    mock_genai_part_2 = genai_types.Part.from_text(text="Part 2")

    part_converter_mock = Mock()
    part_converter_mock.side_effect = [[mock_genai_part_1], [mock_genai_part_2]]

    event = convert_a2a_task_to_event(
        task,
        author="test-author",
        invocation_context=self.mock_context,
        part_converter=part_converter_mock,
    )

    assert event is not None
    assert event.content is not None
    assert len(event.content.parts) == 2
    assert event.content.parts[0].text == "Part 1"
    assert (
        event.content.parts[1].function_call.name
        == MOCK_FUNCTION_CALL_FOR_REQUIRED_USER_INPUT
    )

  def test_convert_a2a_task_to_event_no_text_parts(self):
    """input_required with no text parts does not inject mock function call."""
    # Use a non-text part (inline_data)
    a2a_part = A2APart(raw=b"fake", media_type="image/jpeg")
    status_msg = Message(message_id="m1", role=Role.ROLE_AGENT, parts=[a2a_part])
    status = TaskStatus(
        state=TaskState.TASK_STATE_INPUT_REQUIRED, message=status_msg
    )
    status.timestamp.FromDatetime(datetime.now(timezone.utc))

    task = Task(id="task-1", context_id="context-1")
    task.status.CopyFrom(status)

    mock_image_part = genai_types.Part(
        inline_data=genai_types.Blob(mime_type="image/jpeg", data=b"fake")
    )

    event = convert_a2a_task_to_event(
        task,
        author="test-author",
        invocation_context=self.mock_context,
        part_converter=Mock(return_value=[mock_image_part]),
    )

    assert event is not None
    assert event.content is not None
    assert event.content.parts == [mock_image_part]

  def test_convert_a2a_status_update_to_event_success(self):
    """Status update with a message converts to event with those parts."""
    a2a_part = A2APart(text="status text")
    a2a_part.metadata[_get_adk_metadata_key(A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY)] = True

    status_msg = Message(message_id="m1", role=Role.ROLE_AGENT, parts=[a2a_part])
    status = TaskStatus(state=TaskState.TASK_STATE_INPUT_REQUIRED, message=status_msg)
    status.timestamp.FromDatetime(datetime.now(timezone.utc))

    update = TaskStatusUpdateEvent(task_id="task-1", context_id="context-1")
    update.status.CopyFrom(status)

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
    """None status update raises ValueError."""
    with pytest.raises(ValueError, match="A2A status update cannot be None"):
      convert_a2a_status_update_to_event(None)

  def test_convert_a2a_artifact_update_to_event_success(self):
    """Artifact update with parts converts to a partial event."""
    a2a_part = A2APart(text="chunk text")
    artifact = Artifact(artifact_id="art-1", parts=[a2a_part])

    update = TaskArtifactUpdateEvent(
        task_id="task-1",
        context_id="context-1",
        artifact=artifact,
        append=True,
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
    """None artifact update raises ValueError."""
    with pytest.raises(ValueError, match="A2A artifact update cannot be None"):
      convert_a2a_artifact_update_to_event(None)
