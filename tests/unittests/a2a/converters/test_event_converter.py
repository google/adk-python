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

from unittest.mock import Mock
from unittest.mock import patch

from a2a.types import Message
from a2a.types import Part
from a2a.types import Role
from a2a.types import Task
from a2a.types import TaskState
from a2a.types import TaskStatusUpdateEvent
from google.adk.a2a.converters.event_converter import _create_artifact_id
from google.adk.a2a.converters.event_converter import _create_error_status_event
from google.adk.a2a.converters.event_converter import _create_status_update_event
from google.adk.a2a.converters.event_converter import _get_context_metadata
from google.adk.a2a.converters.event_converter import _serialize_metadata_value
from google.adk.a2a.converters.event_converter import ARTIFACT_ID_SEPARATOR
from google.adk.a2a.converters.event_converter import convert_a2a_message_to_event
from google.adk.a2a.converters.event_converter import convert_a2a_task_to_event
from google.adk.a2a.converters.event_converter import convert_event_to_a2a_events
from google.adk.a2a.converters.event_converter import convert_event_to_a2a_message
from google.adk.a2a.converters.event_converter import DEFAULT_ERROR_MESSAGE
from google.adk.a2a.converters.part_converter import convert_genai_part_to_a2a_part
from google.adk.a2a.converters.utils import _get_adk_metadata_key
from google.adk.a2a.converters.utils import ADK_METADATA_KEY_PREFIX
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.genai import types as genai_types
import pytest


class TestEventConverter:
  """Test suite for event_converter module."""

  def setup_method(self):
    """Set up test fixtures."""
    self.mock_session = Mock()
    self.mock_session.id = "test-session-id"

    self.mock_invocation_context = Mock(spec=InvocationContext)
    self.mock_invocation_context.app_name = "test-app"
    self.mock_invocation_context.user_id = "test-user"
    self.mock_invocation_context.session = self.mock_session

    self.mock_event = Mock(spec=Event)
    self.mock_event.id = None
    self.mock_event.invocation_id = "test-invocation-id"
    self.mock_event.author = "test-author"
    self.mock_event.branch = None
    self.mock_event.grounding_metadata = None
    self.mock_event.custom_metadata = None
    self.mock_event.usage_metadata = None
    self.mock_event.error_code = None
    self.mock_event.error_message = None
    self.mock_event.content = None
    self.mock_event.long_running_tool_ids = None
    self.mock_event.actions = None

  def test_get_adk_event_metadata_key_success(self):
    """Metadata key is formed by prefixing the given key."""
    key = "test_key"
    result = _get_adk_metadata_key(key)
    assert result == f"{ADK_METADATA_KEY_PREFIX}{key}"

  def test_create_error_status_event_is_final(self):
    """Error status events must be marked final."""
    result = _create_error_status_event(
        self.mock_event,
        self.mock_invocation_context,
        task_id="test-task-id",
        context_id="test-context-id",
    )

    assert result.final is True

  def test_get_adk_event_metadata_key_empty_string(self):
    """Empty string key raises ValueError."""
    with pytest.raises(ValueError, match="cannot be empty or None"):
      _get_adk_metadata_key("")

  def test_get_adk_event_metadata_key_none(self):
    """None key raises ValueError."""
    with pytest.raises(ValueError, match="cannot be empty or None"):
      _get_adk_metadata_key(None)

  def test_serialize_metadata_value_with_model_dump(self):
    """Values with model_dump are serialized via that method."""
    mock_value = Mock()
    mock_value.model_dump.return_value = {"key": "value"}

    result = _serialize_metadata_value(mock_value)

    assert result == {"key": "value"}
    mock_value.model_dump.assert_called_once_with(
        exclude_none=True, by_alias=True
    )

  def test_serialize_metadata_value_with_model_dump_exception(self):
    """When model_dump raises, falls back to str() with a warning."""
    mock_value = Mock()
    mock_value.model_dump.side_effect = Exception("Serialization failed")

    with patch(
        "google.adk.a2a.converters.event_converter.logger"
    ) as mock_logger:
      result = _serialize_metadata_value(mock_value)

    assert result == str(mock_value)
    mock_logger.warning.assert_called_once()

  def test_serialize_metadata_value_without_model_dump(self):
    """Plain string values are returned as-is."""
    assert _serialize_metadata_value("simple_string") == "simple_string"

  def test_get_context_metadata_success(self):
    """Context metadata contains all required ADK keys."""
    result = _get_context_metadata(
        self.mock_event, self.mock_invocation_context
    )

    for key in [
        f"{ADK_METADATA_KEY_PREFIX}app_name",
        f"{ADK_METADATA_KEY_PREFIX}user_id",
        f"{ADK_METADATA_KEY_PREFIX}session_id",
        f"{ADK_METADATA_KEY_PREFIX}invocation_id",
        f"{ADK_METADATA_KEY_PREFIX}author",
        f"{ADK_METADATA_KEY_PREFIX}event_id",
    ]:
      assert key in result

  def test_get_context_metadata_with_optional_fields(self):
    """Optional fields are included when present."""
    self.mock_event.branch = "test-branch"
    self.mock_event.error_code = "ERROR_001"
    mock_metadata = Mock()
    mock_metadata.model_dump.return_value = {"test": "value"}
    self.mock_event.grounding_metadata = mock_metadata
    self.mock_event.actions = Mock()
    self.mock_event.actions.model_dump.return_value = {"test_actions": "value"}

    result = _get_context_metadata(
        self.mock_event, self.mock_invocation_context
    )

    assert f"{ADK_METADATA_KEY_PREFIX}branch" in result
    assert f"{ADK_METADATA_KEY_PREFIX}grounding_metadata" in result
    assert f"{ADK_METADATA_KEY_PREFIX}actions" in result
    assert result[f"{ADK_METADATA_KEY_PREFIX}branch"] == "test-branch"
    assert result[f"{ADK_METADATA_KEY_PREFIX}actions"] == {
        "test_actions": "value"
    }

  def test_get_context_metadata_none_event(self):
    """None event raises ValueError."""
    with pytest.raises(ValueError, match="Event cannot be None"):
      _get_context_metadata(None, self.mock_invocation_context)

  def test_get_context_metadata_none_context(self):
    """None context raises ValueError."""
    with pytest.raises(ValueError, match="Invocation context cannot be None"):
      _get_context_metadata(self.mock_event, None)

  def test_create_artifact_id(self):
    """Artifact ID is formed by joining components with the separator."""
    result = _create_artifact_id(
        "test-app", "user123", "session456", "test.txt", 1
    )
    expected = f"test-app{ARTIFACT_ID_SEPARATOR}user123{ARTIFACT_ID_SEPARATOR}session456{ARTIFACT_ID_SEPARATOR}test.txt{ARTIFACT_ID_SEPARATOR}1"
    assert result == expected

  @patch(
      "google.adk.a2a.converters.event_converter.convert_event_to_a2a_message"
  )
  @patch("google.adk.a2a.converters.event_converter._create_error_status_event")
  @patch(
      "google.adk.a2a.converters.event_converter._create_status_update_event"
  )
  def test_convert_event_to_a2a_events_full_scenario(
      self, mock_create_running, mock_create_error, mock_convert_message
  ):
    """Event with error and message produces both error and running events."""
    self.mock_event.error_code = "ERROR_001"

    mock_message = Mock(spec=Message)
    mock_convert_message.return_value = mock_message
    mock_error_event = Mock()
    mock_create_error.return_value = mock_error_event
    mock_running_event = Mock()
    mock_create_running.return_value = mock_running_event

    result = convert_event_to_a2a_events(
        self.mock_event, self.mock_invocation_context
    )

    mock_create_error.assert_called_once_with(
        self.mock_event, self.mock_invocation_context, None, None
    )
    mock_create_running.assert_called_once_with(
        mock_message, self.mock_invocation_context, self.mock_event, None, None
    )
    assert len(result) == 2
    assert mock_error_event in result
    assert mock_running_event in result

  def test_convert_event_to_a2a_events_empty_scenario(self):
    """Event with no content or error produces no events."""
    result = convert_event_to_a2a_events(
        self.mock_event, self.mock_invocation_context
    )
    assert result == []

  def test_convert_event_to_a2a_events_none_event(self):
    """None event raises ValueError."""
    with pytest.raises(ValueError, match="Event cannot be None"):
      convert_event_to_a2a_events(None, self.mock_invocation_context)

  def test_convert_event_to_a2a_events_none_context(self):
    """None context raises ValueError."""
    with pytest.raises(ValueError, match="Invocation context cannot be None"):
      convert_event_to_a2a_events(self.mock_event, None)

  @patch(
      "google.adk.a2a.converters.event_converter.convert_event_to_a2a_message"
  )
  def test_convert_event_to_a2a_events_message_only(self, mock_convert_message):
    """Event with message only produces one running event."""
    mock_message = Mock(spec=Message)
    mock_convert_message.return_value = mock_message

    with patch(
        "google.adk.a2a.converters.event_converter._create_status_update_event"
    ) as mock_create_running:
      mock_running_event = Mock()
      mock_create_running.return_value = mock_running_event

      result = convert_event_to_a2a_events(
          self.mock_event, self.mock_invocation_context
      )

      assert len(result) == 1
      assert result[0] == mock_running_event
      mock_create_running.assert_called_once_with(
          mock_message,
          self.mock_invocation_context,
          self.mock_event,
          None,
          None,
      )

  @patch(
      "google.adk.a2a.converters.event_converter.convert_event_to_a2a_message"
  )
  def test_convert_event_to_a2a_events_with_task_id_and_context_id(
      self, mock_convert_message
  ):
    """Custom task_id and context_id are forwarded to status_update_event."""
    mock_message = Mock(spec=Message)
    mock_message.parts = []
    mock_convert_message.return_value = mock_message

    with patch(
        "google.adk.a2a.converters.event_converter._create_status_update_event"
    ) as mock_create_running:
      mock_create_running.return_value = Mock()

      convert_event_to_a2a_events(
          self.mock_event, self.mock_invocation_context, "task-1", "ctx-1"
      )

      mock_create_running.assert_called_once_with(
          mock_message,
          self.mock_invocation_context,
          self.mock_event,
          "task-1",
          "ctx-1",
      )

  def test_convert_event_to_a2a_events_user_role(self):
    """User-authored event uses ROLE_USER."""
    mock_message = Mock(spec=Message)
    mock_message.parts = []

    with patch(
        "google.adk.a2a.converters.event_converter.convert_event_to_a2a_message"
    ) as mock_convert_message:
      mock_convert_message.return_value = mock_message
      self.mock_event.author = "user"

      with patch(
          "google.adk.a2a.converters.event_converter._create_status_update_event"
      ) as mock_create_running:
        mock_create_running.return_value = Mock()

        convert_event_to_a2a_events(
            self.mock_event, self.mock_invocation_context, "t", "c"
        )

        mock_convert_message.assert_called_once_with(
            self.mock_event,
            role=Role.ROLE_USER,
            part_converter=convert_genai_part_to_a2a_part,
        )

  def test_create_status_update_event_yields_auth_required_state(self):
    """Message with auth-required pattern sets TASK_STATE_AUTH_REQUIRED."""
    from google.adk.a2a.converters.part_converter import A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY
    from google.adk.a2a.converters.part_converter import A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL
    from google.adk.a2a.converters.part_converter import A2A_DATA_PART_METADATA_TYPE_KEY
    from google.adk.flows.llm_flows.functions import REQUEST_EUC_FUNCTION_CALL_NAME
    from google.protobuf import json_format

    # Build a proto Part that is a long-running function call to request_euc
    part = Part()
    json_format.ParseDict(
        {
            "data": {
                "name": REQUEST_EUC_FUNCTION_CALL_NAME,
                "id": "fc-1",
                "args": {},
            }
        },
        part,
    )
    part.metadata[_get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)] = (
        A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL
    )
    part.metadata[
        _get_adk_metadata_key(A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY)
    ] = True

    msg = Message(message_id="m1", role=Role.ROLE_AGENT, parts=[part])

    result = _create_status_update_event(
        msg, self.mock_invocation_context, self.mock_event, "t", "c"
    )

    assert isinstance(result, TaskStatusUpdateEvent)
    assert result.status.state == TaskState.TASK_STATE_AUTH_REQUIRED

  def test_create_status_update_event_yields_input_required_state(self):
    """Message with non-auth long-running call sets TASK_STATE_INPUT_REQUIRED."""
    from google.adk.a2a.converters.part_converter import A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY
    from google.adk.a2a.converters.part_converter import A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL
    from google.adk.a2a.converters.part_converter import A2A_DATA_PART_METADATA_TYPE_KEY
    from google.protobuf import json_format

    part = Part()
    json_format.ParseDict(
        {"data": {"name": "some_other_tool", "id": "fc-2", "args": {}}}, part
    )
    part.metadata[_get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)] = (
        A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL
    )
    part.metadata[
        _get_adk_metadata_key(A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY)
    ] = True

    msg = Message(message_id="m1", role=Role.ROLE_AGENT, parts=[part])

    result = _create_status_update_event(
        msg, self.mock_invocation_context, self.mock_event, "t", "c"
    )

    assert result.status.state == TaskState.TASK_STATE_INPUT_REQUIRED


class TestA2AToEventConverters:
  """Test suite for A2A to Event conversion functions."""

  def setup_method(self):
    """Set up test fixtures."""
    self.mock_invocation_context = Mock(spec=InvocationContext)
    self.mock_invocation_context.invocation_id = "test-invocation-id"
    self.mock_invocation_context.branch = "test-branch"

  def test_convert_a2a_task_to_event_no_message(self):
    """Task with no message, artifacts, or history produces a minimal event."""
    mock_task = Mock(spec=Task)
    mock_task.artifacts = None
    mock_task.status = None
    mock_task.history = []

    result = convert_a2a_task_to_event(
        mock_task, "test-author", self.mock_invocation_context
    )

    assert result.author == "test-author"
    assert result.branch == "test-branch"
    assert result.invocation_id == "test-invocation-id"

  @patch("google.adk.a2a.converters.event_converter.platform_uuid.new_uuid")
  def test_convert_a2a_task_to_event_default_author(self, mock_uuid):
    """Task with no context uses default author and generates a UUID."""
    mock_task = Mock(spec=Task)
    mock_task.artifacts = None
    mock_task.status = None
    mock_task.history = []
    mock_uuid.return_value = "generated-uuid"

    result = convert_a2a_task_to_event(mock_task)

    assert result.author == "a2a agent"
    assert result.branch is None
    assert result.invocation_id == "generated-uuid"

  def test_convert_a2a_task_to_event_none_task(self):
    """None task raises ValueError."""
    with pytest.raises(ValueError, match="A2A task cannot be None"):
      convert_a2a_task_to_event(None)

  def test_convert_a2a_task_to_event_message_conversion_error(self):
    """Conversion error in message is wrapped as RuntimeError."""
    mock_message = Mock(spec=Message, parts=[Mock()])
    mock_status = Mock(message=mock_message)
    mock_task = Mock(spec=Task, artifacts=None, status=mock_status, history=[])

    with patch(
        "google.adk.a2a.converters.event_converter.convert_a2a_message_to_event"
    ) as mock_convert_message:
      mock_convert_message.side_effect = Exception("Conversion failed")

      with pytest.raises(RuntimeError, match="Failed to convert task message"):
        convert_a2a_task_to_event(mock_task, "test-author")

  def test_convert_a2a_message_to_event_success(self):
    """Message parts are converted and placed in the event content."""
    a2a_part = Part(text="source part")
    mock_genai_part = genai_types.Part(text="test content")
    mock_convert_part = Mock(return_value=mock_genai_part)
    mock_message = Mock(spec=Message, parts=[a2a_part])

    result = convert_a2a_message_to_event(
        mock_message,
        "test-author",
        self.mock_invocation_context,
        mock_convert_part,
    )

    assert result.author == "test-author"
    assert result.branch == "test-branch"
    assert result.invocation_id == "test-invocation-id"
    assert result.content.role == "model"
    assert len(result.content.parts) == 1
    assert result.content.parts[0].text == "test content"
    mock_convert_part.assert_called_once_with(a2a_part)

  def test_convert_a2a_message_to_event_empty_parts(self):
    """Message with empty parts produces an event with empty content."""
    mock_message = Mock(spec=Message, parts=[])

    result = convert_a2a_message_to_event(
        mock_message, "test-author", self.mock_invocation_context
    )

    assert result.author == "test-author"
    assert result.content.role == "model"
    assert len(result.content.parts) == 0

  def test_convert_a2a_message_to_event_none_message(self):
    """None message raises ValueError."""
    with pytest.raises(ValueError, match="A2A message cannot be None"):
      convert_a2a_message_to_event(None)

  def test_convert_a2a_message_to_event_part_conversion_fails(self):
    """Failed part conversion produces an event with no parts."""
    a2a_part = Part(text="some text")
    mock_convert_part = Mock(return_value=None)
    mock_message = Mock(spec=Message, parts=[a2a_part])

    result = convert_a2a_message_to_event(
        mock_message,
        "test-author",
        self.mock_invocation_context,
        mock_convert_part,
    )

    assert result.author == "test-author"
    assert len(result.content.parts) == 0

  def test_convert_a2a_message_to_event_part_conversion_exception(self):
    """Part conversion exception is skipped; remaining parts are included."""
    a2a_part1 = Part(text="text1")
    a2a_part2 = Part(text="text2")
    mock_genai_part = genai_types.Part(text="successful conversion")
    mock_convert_part = Mock(
        side_effect=[Exception("Conversion failed"), mock_genai_part]
    )
    mock_message = Mock(spec=Message, parts=[a2a_part1, a2a_part2])

    result = convert_a2a_message_to_event(
        mock_message,
        "test-author",
        self.mock_invocation_context,
        mock_convert_part,
    )

    assert len(result.content.parts) == 1
    assert result.content.parts[0].text == "successful conversion"

  @patch("google.adk.a2a.converters.event_converter.platform_uuid.new_uuid")
  def test_convert_a2a_message_to_event_default_author(self, mock_uuid):
    """No invocation context uses default author and generated UUID."""
    mock_message = Mock(spec=Message, parts=[])
    mock_uuid.return_value = "generated-uuid"

    result = convert_a2a_message_to_event(mock_message)

    assert result.author == "a2a agent"
    assert result.branch is None
    assert result.invocation_id == "generated-uuid"

  def test_convert_event_to_a2a_message_returns_none_for_empty_content(self):
    """Event with no content produces None."""
    mock_event = Mock(spec=Event)
    mock_event.content = None

    result = convert_event_to_a2a_message(mock_event)

    assert result is None

  def test_convert_event_to_a2a_message_with_text_part(self):
    """Event with text part produces A2A message with matching text part."""
    mock_event = Mock(spec=Event)
    mock_event.long_running_tool_ids = None
    mock_event.content = genai_types.Content(
        parts=[genai_types.Part(text="hello")], role="model"
    )

    result = convert_event_to_a2a_message(
        mock_event, part_converter=convert_genai_part_to_a2a_part
    )

    assert result is not None
    assert len(result.parts) == 1
    assert result.parts[0].WhichOneof("content") == "text"
    assert result.parts[0].text == "hello"
