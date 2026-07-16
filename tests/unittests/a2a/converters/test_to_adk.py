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

import json
from unittest.mock import Mock

from a2a.types import Message
from a2a.types import Part as A2APart
from a2a.types import Task
from a2a.types import TaskArtifactUpdateEvent
from google.adk.a2a import _compat
from google.adk.a2a.converters.part_converter import A2A_DATA_PART_END_TAG
from google.adk.a2a.converters.part_converter import A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY
from google.adk.a2a.converters.part_converter import A2A_DATA_PART_START_TAG
from google.adk.a2a.converters.part_converter import A2A_DATA_PART_TEXT_MIME_TYPE
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


def _make_a2a_part_for_test(metadata=None):
  """Returns a real proto Part on 1.x, or Mock(spec=A2APart) on 0.3."""
  if _compat.IS_A2A_V1:
    p = _compat.make_text_part("test")
    if metadata:
      _compat.set_part_metadata(p, metadata)
    return p
  else:
    from unittest.mock import Mock

    from a2a.types import TextPart

    m = Mock(spec=A2APart)
    m.root = Mock(spec=TextPart)
    m.root.metadata = metadata or {}
    return m


class TestToAdk:
  """Test suite for to_adk functions."""

  def setup_method(self):
    """Set up test fixtures."""
    self.mock_context = Mock(spec=InvocationContext)
    self.mock_context.invocation_id = "test-invocation"
    self.mock_context.branch = "test-branch"

  def test_convert_a2a_message_to_event_success(self):
    """Test successful conversion of A2A message to Event."""
    a2a_part = _make_a2a_part_for_test({})
    message = Message(
        message_id="msg-1", role=_compat.ROLE_USER, parts=[a2a_part]
    )

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

  def test_convert_a2a_message_to_event_restores_actions_from_metadata(self):
    """Test A2A message conversion restores ADK actions metadata."""
    a2a_part = _make_a2a_part_for_test({})
    message = Message(
        message_id="msg-1",
        role=_compat.ROLE_USER,
        parts=[a2a_part],
        metadata={
            _get_adk_metadata_key("actions"): {
                "stateDelta": {"saved_key": "saved-value"}
            }
        },
    )

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
    """Test A2A message conversion returns action-only events."""
    message = Message(
        message_id="msg-1",
        role=_compat.ROLE_USER,
        parts=[],
        metadata={
            _get_adk_metadata_key("actions"): {
                "stateDelta": {"saved_key": "saved-value"}
            }
        },
    )

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
    """Test successful conversion of A2A task to Event."""
    a2a_part = _make_a2a_part_for_test({})
    task = Task(
        id="task-1",
        status=_compat.make_task_status(
            _compat.TS_SUBMITTED, timestamp="2024-01-01T00:00:00Z"
        ),
        context_id="context-1",
        history=[
            Message(
                message_id="msg-1", role=_compat.ROLE_AGENT, parts=[a2a_part]
            )
        ],
        artifacts=[
            _compat.make_artifact(
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

  def test_convert_a2a_task_to_event_returns_action_only_event(self):
    """Test A2A task conversion returns action-only events."""
    task = Task(
        id="task-1",
        status=_compat.make_task_status(
            _compat.TS_SUBMITTED, timestamp="2024-01-01T00:00:00Z"
        ),
        context_id="context-1",
        artifacts=[
            _compat.make_artifact(
                artifact_id="art-1",
                artifact_type="message",
                parts=[],
                metadata={
                    _get_adk_metadata_key("actions"): {
                        "stateDelta": {"saved_key": "saved-value"}
                    }
                },
            )
        ],
    )

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
    """Test task conversion merges actions across artifact metadata."""
    task = Task(
        id="task-1",
        status=_compat.make_task_status(
            _compat.TS_SUBMITTED, timestamp="2024-01-01T00:00:00Z"
        ),
        context_id="context-1",
        artifacts=[
            _compat.make_artifact(
                artifact_id="art-1",
                artifact_type="message",
                parts=[],
                metadata={
                    _get_adk_metadata_key("actions"): {
                        "stateDelta": {"first_key": "first-value"}
                    }
                },
            ),
            _compat.make_artifact(
                artifact_id="art-2",
                artifact_type="message",
                parts=[],
                metadata={},
            ),
        ],
    )

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
    """Test task conversion preserves top-level state overwrite semantics."""
    task = Task(
        id="task-1",
        status=_compat.make_task_status(
            _compat.TS_SUBMITTED, timestamp="2024-01-01T00:00:00Z"
        ),
        context_id="context-1",
        artifacts=[
            _compat.make_artifact(
                artifact_id="art-1",
                artifact_type="message",
                parts=[],
                metadata={
                    _get_adk_metadata_key("actions"): {
                        "stateDelta": {
                            "settings": {
                                "theme": "light",
                                "language": "en",
                            }
                        }
                    }
                },
            ),
            _compat.make_artifact(
                artifact_id="art-2",
                artifact_type="message",
                parts=[],
                metadata={
                    _get_adk_metadata_key("actions"): {
                        "stateDelta": {"settings": {"theme": "dark"}}
                    }
                },
            ),
        ],
    )

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
    """Test task conversion merges status and artifact actions."""
    a2a_part = _make_a2a_part_for_test({})
    task = Task(
        id="task-1",
        status=_compat.make_task_status(
            _compat.TS_INPUT_REQUIRED,
            timestamp="2024-01-01T00:00:00Z",
            message=Message(
                message_id="msg-1",
                role=_compat.ROLE_AGENT,
                parts=[a2a_part],
                metadata={
                    _get_adk_metadata_key("actions"): {
                        "transferToAgent": "agent-2"
                    }
                },
            ),
        ),
        context_id="context-1",
        artifacts=[
            _compat.make_artifact(
                artifact_id="art-1",
                artifact_type="message",
                parts=[],
                metadata={
                    _get_adk_metadata_key("actions"): {
                        "stateDelta": {"saved_key": "saved-value"}
                    }
                },
            )
        ],
    )

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
    assert (
        event.content.parts[0].function_call.args["input_required"]
        == "need input"
    )

  def test_convert_a2a_task_to_event_auth_required_uses_auth_args_key(self):
    """Test auth-required state populates the function call with auth args."""
    a2a_part = _make_a2a_part_for_test({})
    task = _compat.make_task(
        id="task-1",
        context_id="context-1",
        kind="task",
        status=_compat.make_task_status(
            _compat.TS_AUTH_REQUIRED,
            timestamp="now",
            message=Message(
                message_id="m1",
                role=_compat.ROLE_AGENT,
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
    """Test converting A2A task with multiple text parts, only replacing the last text."""
    part1 = _make_a2a_part_for_test({})
    part2 = _make_a2a_part_for_test({})

    task = _compat.make_task(
        id="task-1",
        context_id="context-1",
        kind="task",
        status=_compat.make_task_status(
            _compat.TS_INPUT_REQUIRED,
            timestamp="now",
            message=Message(
                message_id="m1",
                role=_compat.ROLE_AGENT,
                parts=[part1, part2],
            ),
        ),
    )

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
    """Test converting A2A task with no text parts should not inject function call."""
    # A real non-text (data) part; converter output is mocked below.
    part1 = _compat.make_data_part(data={"placeholder": True})

    task = _compat.make_task(
        id="task-1",
        context_id="context-1",
        kind="task",
        status=_compat.make_task_status(
            _compat.TS_INPUT_REQUIRED,
            timestamp="now",
            message=Message(
                message_id="m1",
                role=_compat.ROLE_AGENT,
                parts=[part1],
            ),
        ),
    )
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

  def test_convert_a2a_task_to_event_data_part_input_required(self):
    """Input-required prompt carried in a data part becomes a function call."""
    # A real non-text (data) part; converter output is mocked below.
    part1 = _compat.make_data_part(data={"placeholder": True})

    task = _compat.make_task(
        id="task-1",
        context_id="context-1",
        kind="task",
        status=_compat.make_task_status(
            _compat.TS_INPUT_REQUIRED,
            timestamp="now",
            message=Message(
                message_id="m1",
                role=_compat.ROLE_AGENT,
                parts=[part1],
            ),
        ),
    )

    prompt = {
        "id": "abc123",
        "text": "Please confirm this action. Do you want to continue?",
    }
    data_part_json = json.dumps({"data": prompt, "kind": "data"}).encode(
        "utf-8"
    )
    mock_data_blob_part = genai_types.Part(
        inline_data=genai_types.Blob(
            mime_type=A2A_DATA_PART_TEXT_MIME_TYPE,
            data=A2A_DATA_PART_START_TAG
            + data_part_json
            + A2A_DATA_PART_END_TAG,
        )
    )

    event = convert_a2a_task_to_event(
        task,
        author="test-author",
        invocation_context=self.mock_context,
        part_converter=Mock(return_value=[mock_data_blob_part]),
    )

    assert event is not None
    assert event.content is not None
    assert (
        event.content.parts[0].function_call.name
        == MOCK_FUNCTION_CALL_FOR_REQUIRED_USER_INPUT
    )
    assert event.content.parts[0].function_call.args["input_required"] == prompt
    assert event.long_running_tool_ids

  def test_convert_a2a_task_to_event_data_part_malformed_json(self):
    """A malformed data-part blob is left untouched (no crash, no fc)."""
    # A real non-text (data) part; converter output is mocked below.
    part1 = _compat.make_data_part(data={"placeholder": True})

    task = _compat.make_task(
        id="task-1",
        context_id="context-1",
        kind="task",
        status=_compat.make_task_status(
            _compat.TS_INPUT_REQUIRED,
            timestamp="now",
            message=Message(
                message_id="m1",
                role=_compat.ROLE_AGENT,
                parts=[part1],
            ),
        ),
    )

    mock_bad_blob_part = genai_types.Part(
        inline_data=genai_types.Blob(
            mime_type=A2A_DATA_PART_TEXT_MIME_TYPE,
            data=A2A_DATA_PART_START_TAG + b"not-json" + A2A_DATA_PART_END_TAG,
        )
    )

    event = convert_a2a_task_to_event(
        task,
        author="test-author",
        invocation_context=self.mock_context,
        part_converter=Mock(return_value=[mock_bad_blob_part]),
    )

    assert event is not None
    assert event.content is not None
    assert event.content.parts == [mock_bad_blob_part]
    assert not event.long_running_tool_ids

  def test_convert_a2a_status_update_to_event_success(self):
    """Test successful conversion of A2A status update to Event."""
    a2a_part = _make_a2a_part_for_test({
        _get_adk_metadata_key(A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY): True
    })
    update = _compat.make_task_status_update_event(
        task_id="task-1",
        status=_compat.make_task_status(
            _compat.TS_INPUT_REQUIRED,
            timestamp="now",
            message=Message(
                message_id="m1",
                role=_compat.ROLE_AGENT,
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
    a2a_part = _make_a2a_part_for_test({})
    update = TaskArtifactUpdateEvent(
        task_id="task-1",
        artifact=_compat.make_artifact(
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

  def test_convert_a2a_message_to_event_user_role(self) -> None:
    """Test that A2A user role maps to GenAI content role 'user'."""
    a2a_part = _make_a2a_part_for_test({})
    message = Message(
        message_id="msg-1", role=_compat.ROLE_USER, parts=[a2a_part]
    )

    mock_genai_part = genai_types.Part.from_text(text="hello from user")
    mock_part_converter = Mock(return_value=[mock_genai_part])

    event = convert_a2a_message_to_event(
        message,
        author="user",
        invocation_context=self.mock_context,
        part_converter=mock_part_converter,
    )

    assert event.content.role == "user"

  def test_convert_a2a_message_to_event_agent_role(self) -> None:
    """Test that A2A agent role maps to GenAI content role 'model'."""
    a2a_part = _make_a2a_part_for_test({})
    message = Message(
        message_id="msg-1", role=_compat.ROLE_AGENT, parts=[a2a_part]
    )

    mock_genai_part = genai_types.Part.from_text(text="hello from agent")
    mock_part_converter = Mock(return_value=[mock_genai_part])

    event = convert_a2a_message_to_event(
        message,
        author="test-agent",
        invocation_context=self.mock_context,
        part_converter=mock_part_converter,
    )

    assert event.content.role == "model"
