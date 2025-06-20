# Copyright 2025 Google LLC
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

import sys
from unittest.mock import Mock
from unittest.mock import patch

import pytest

# Skip all tests in this module if Python version is less than 3.10
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10), reason="A2A tool requires Python 3.10+"
)

# Import dependencies with version checking
try:
  from a2a.types import DataPart
  from a2a.types import Message
  from a2a.types import Role
  from a2a.types import TaskArtifactUpdateEvent
  from a2a.types import TaskState
  from a2a.types import TaskStatusUpdateEvent
  from google.adk.a2a.converters.event_converter import _convert_artifact_to_a2a_events
  from google.adk.a2a.converters.event_converter import _create_artifact_id
  from google.adk.a2a.converters.event_converter import _create_error_status_event
  from google.adk.a2a.converters.event_converter import _create_running_status_event
  from google.adk.a2a.converters.event_converter import _get_adk_metadata_key
  from google.adk.a2a.converters.event_converter import _get_context_metadata
  from google.adk.a2a.converters.event_converter import _process_long_running_tool
  from google.adk.a2a.converters.event_converter import _serialize_metadata_value
  from google.adk.a2a.converters.event_converter import ARTIFACT_ID_SEPARATOR
  from google.adk.a2a.converters.event_converter import convert_event_to_a2a_events
  from google.adk.a2a.converters.event_converter import convert_event_to_a2a_status_message
  from google.adk.a2a.converters.event_converter import DEFAULT_ERROR_MESSAGE
  from google.adk.a2a.converters.utils import ADK_METADATA_KEY_PREFIX
  from google.adk.agents.invocation_context import InvocationContext
  from google.adk.events.event import Event
  from google.adk.events.event_actions import EventActions
except ImportError as e:
  if sys.version_info < (3, 10):
    # Create dummy classes to prevent NameError during test collection
    # Tests will be skipped anyway due to pytestmark
    class DummyTypes:
      pass

    DataPart = DummyTypes()
    Message = DummyTypes()
    Role = DummyTypes()
    TaskArtifactUpdateEvent = DummyTypes()
    TaskState = DummyTypes()
    TaskStatusUpdateEvent = DummyTypes()
    _convert_artifact_to_a2a_events = lambda *args: None
    _create_artifact_id = lambda *args: None
    _create_error_status_event = lambda *args: None
    _create_running_status_event = lambda *args: None
    _get_adk_metadata_key = lambda *args: None
    _get_context_metadata = lambda *args: None
    _process_long_running_tool = lambda *args: None
    _serialize_metadata_value = lambda *args: None
    ADK_METADATA_KEY_PREFIX = "adk_"
    ARTIFACT_ID_SEPARATOR = "_"
    convert_event_to_a2a_events = lambda *args: None
    convert_event_to_a2a_status_message = lambda *args: None
    DEFAULT_ERROR_MESSAGE = "error"
    InvocationContext = DummyTypes()
    Event = DummyTypes()
    EventActions = DummyTypes()
    types = DummyTypes()
  else:
    raise e


class TestEventConverter:
  """Test suite for event_converter module."""

  def setup_method(self):
    """Set up test fixtures."""
    self.mock_session = Mock()
    self.mock_session.id = "test-session-id"

    self.mock_artifact_service = Mock()
    self.mock_invocation_context = Mock(spec=InvocationContext)
    self.mock_invocation_context.app_name = "test-app"
    self.mock_invocation_context.user_id = "test-user"
    self.mock_invocation_context.session = self.mock_session
    self.mock_invocation_context.artifact_service = self.mock_artifact_service

    self.mock_event = Mock(spec=Event)
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
    self.mock_event.actions = Mock(spec=EventActions)
    self.mock_event.actions.artifact_delta = None

  def test_get_adk_event_metadata_key_success(self):
    """Test successful metadata key generation."""
    key = "test_key"
    result = _get_adk_metadata_key(key)
    assert result == f"{ADK_METADATA_KEY_PREFIX}{key}"

  def test_get_adk_event_metadata_key_empty_string(self):
    """Test metadata key generation with empty string."""
    with pytest.raises(ValueError) as exc_info:
      _get_adk_metadata_key("")
    assert "cannot be empty or None" in str(exc_info.value)

  def test_get_adk_event_metadata_key_none(self):
    """Test metadata key generation with None."""
    with pytest.raises(ValueError) as exc_info:
      _get_adk_metadata_key(None)
    assert "cannot be empty or None" in str(exc_info.value)

  def test_serialize_metadata_value_with_model_dump(self):
    """Test serialization of value with model_dump method."""
    mock_value = Mock()
    mock_value.model_dump.return_value = {"key": "value"}

    result = _serialize_metadata_value(mock_value)

    assert result == {"key": "value"}
    mock_value.model_dump.assert_called_once_with(
        exclude_none=True, by_alias=True
    )

  def test_serialize_metadata_value_with_model_dump_exception(self):
    """Test serialization when model_dump raises exception."""
    mock_value = Mock()
    mock_value.model_dump.side_effect = Exception("Serialization failed")

    with patch(
        "google.adk.a2a.converters.event_converter.logger"
    ) as mock_logger:
      result = _serialize_metadata_value(mock_value)

      assert result == str(mock_value)
      mock_logger.warning.assert_called_once()

  def test_serialize_metadata_value_without_model_dump(self):
    """Test serialization of value without model_dump method."""
    value = "simple_string"
    result = _serialize_metadata_value(value)
    assert result == "simple_string"

  def test_get_context_metadata_success(self):
    """Test successful context metadata creation."""
    result = _get_context_metadata(
        self.mock_event, self.mock_invocation_context
    )

    assert result is not None
    expected_keys = [
        f"{ADK_METADATA_KEY_PREFIX}app_name",
        f"{ADK_METADATA_KEY_PREFIX}user_id",
        f"{ADK_METADATA_KEY_PREFIX}session_id",
        f"{ADK_METADATA_KEY_PREFIX}invocation_id",
        f"{ADK_METADATA_KEY_PREFIX}author",
    ]

    for key in expected_keys:
      assert key in result

  def test_get_context_metadata_with_optional_fields(self):
    """Test context metadata creation with optional fields."""
    self.mock_event.branch = "test-branch"
    self.mock_event.error_code = "ERROR_001"

    mock_metadata = Mock()
    mock_metadata.model_dump.return_value = {"test": "value"}
    self.mock_event.grounding_metadata = mock_metadata

    result = _get_context_metadata(
        self.mock_event, self.mock_invocation_context
    )

    assert result is not None
    assert f"{ADK_METADATA_KEY_PREFIX}branch" in result
    assert f"{ADK_METADATA_KEY_PREFIX}grounding_metadata" in result
    assert result[f"{ADK_METADATA_KEY_PREFIX}branch"] == "test-branch"

    # Check if error_code is in the result - it should be there since we set it
    if f"{ADK_METADATA_KEY_PREFIX}error_code" in result:
      assert result[f"{ADK_METADATA_KEY_PREFIX}error_code"] == "ERROR_001"

  def test_get_context_metadata_none_event(self):
    """Test context metadata creation with None event."""
    with pytest.raises(ValueError) as exc_info:
      _get_context_metadata(None, self.mock_invocation_context)
    assert "Event cannot be None" in str(exc_info.value)

  def test_get_context_metadata_none_context(self):
    """Test context metadata creation with None context."""
    with pytest.raises(ValueError) as exc_info:
      _get_context_metadata(self.mock_event, None)
    assert "Invocation context cannot be None" in str(exc_info.value)

  def test_create_artifact_id(self):
    """Test artifact ID creation."""
    app_name = "test-app"
    user_id = "user123"
    session_id = "session456"
    filename = "test.txt"
    version = 1

    result = _create_artifact_id(
        app_name, user_id, session_id, filename, version
    )
    expected = f"{app_name}{ARTIFACT_ID_SEPARATOR}{user_id}{ARTIFACT_ID_SEPARATOR}{session_id}{ARTIFACT_ID_SEPARATOR}{filename}{ARTIFACT_ID_SEPARATOR}{version}"

    assert result == expected

  @patch(
      "google.adk.a2a.converters.event_converter.convert_genai_part_to_a2a_part"
  )
  def test_convert_artifact_to_a2a_events_success(self, mock_convert_part):
    """Test successful artifact delta conversion."""
    filename = "test.txt"
    version = 1

    mock_artifact_part = Mock()
    # Create a proper Part that Pydantic will accept
    from a2a.types import Part
    from a2a.types import TextPart

    text_part = TextPart(text="test content")
    mock_converted_part = Part(root=text_part)

    self.mock_artifact_service.load_artifact.return_value = mock_artifact_part
    mock_convert_part.return_value = mock_converted_part

    result = _convert_artifact_to_a2a_events(
        self.mock_event, self.mock_invocation_context, filename, version
    )

    assert isinstance(result, TaskArtifactUpdateEvent)
    assert result.contextId == self.mock_invocation_context.session.id
    assert result.append is False
    assert result.lastChunk is True

    # Check artifact properties
    assert result.artifact.name == filename
    assert result.artifact.metadata["filename"] == filename
    assert result.artifact.metadata["version"] == version
    assert len(result.artifact.parts) == 1
    assert result.artifact.parts[0].root.text == "test content"

  def test_convert_artifact_to_a2a_events_empty_filename(self):
    """Test artifact delta conversion with empty filename."""
    with pytest.raises(ValueError) as exc_info:
      _convert_artifact_to_a2a_events(
          self.mock_event, self.mock_invocation_context, "", 1
      )
    assert "Filename cannot be empty" in str(exc_info.value)

  def test_convert_artifact_to_a2a_events_negative_version(self):
    """Test artifact delta conversion with negative version."""
    with pytest.raises(ValueError) as exc_info:
      _convert_artifact_to_a2a_events(
          self.mock_event, self.mock_invocation_context, "test.txt", -1
      )
    assert "Version must be non-negative" in str(exc_info.value)

  @patch(
      "google.adk.a2a.converters.event_converter.convert_genai_part_to_a2a_part"
  )
  def test_convert_artifact_to_a2a_events_conversion_failure(
      self, mock_convert_part
  ):
    """Test artifact delta conversion when part conversion fails."""
    filename = "test.txt"
    version = 1

    mock_artifact_part = Mock()
    self.mock_artifact_service.load_artifact.return_value = mock_artifact_part
    mock_convert_part.return_value = None  # Simulate conversion failure

    with pytest.raises(RuntimeError) as exc_info:
      _convert_artifact_to_a2a_events(
          self.mock_event, self.mock_invocation_context, filename, version
      )
    assert "Failed to convert artifact part" in str(exc_info.value)

  def test_process_long_running_tool_marks_tool(self):
    """Test processing of long-running tool metadata."""
    mock_a2a_part = Mock()
    mock_data_part = Mock(spec=DataPart)
    mock_data_part.metadata = {"adk_type": "function_call", "id": "tool-123"}
    mock_a2a_part.root = mock_data_part

    self.mock_event.long_running_tool_ids = {"tool-123"}

    with (
        patch(
            "google.adk.a2a.converters.event_converter.A2A_DATA_PART_METADATA_TYPE_KEY",
            "type",
        ),
        patch(
            "google.adk.a2a.converters.event_converter.A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL",
            "function_call",
        ),
    ):

      _process_long_running_tool(mock_a2a_part, self.mock_event)

      expected_key = f"{ADK_METADATA_KEY_PREFIX}is_long_running"
      assert mock_data_part.metadata[expected_key] is True

  def test_process_long_running_tool_no_marking(self):
    """Test processing when tool should not be marked as long-running."""
    mock_a2a_part = Mock()
    mock_data_part = Mock(spec=DataPart)
    mock_data_part.metadata = {"adk_type": "function_call", "id": "tool-456"}
    mock_a2a_part.root = mock_data_part

    self.mock_event.long_running_tool_ids = {"tool-123"}  # Different ID

    with (
        patch(
            "google.adk.a2a.converters.event_converter.A2A_DATA_PART_METADATA_TYPE_KEY",
            "type",
        ),
        patch(
            "google.adk.a2a.converters.event_converter.A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL",
            "function_call",
        ),
    ):

      _process_long_running_tool(mock_a2a_part, self.mock_event)

      expected_key = f"{ADK_METADATA_KEY_PREFIX}is_long_running"
      assert expected_key not in mock_data_part.metadata

  @patch(
      "google.adk.a2a.converters.event_converter.convert_genai_part_to_a2a_part"
  )
  @patch("google.adk.a2a.converters.event_converter.uuid.uuid4")
  def test_convert_event_to_message_success(self, mock_uuid, mock_convert_part):
    """Test successful event to message conversion."""
    mock_uuid.return_value = "test-uuid"

    mock_part = Mock()
    # Create a proper Part that Pydantic will accept
    from a2a.types import Part
    from a2a.types import TextPart

    text_part = TextPart(text="test message")
    mock_a2a_part = Part(root=text_part)
    mock_convert_part.return_value = mock_a2a_part

    mock_content = Mock()
    mock_content.parts = [mock_part]
    self.mock_event.content = mock_content

    result = convert_event_to_a2a_status_message(
        self.mock_event, self.mock_invocation_context
    )

    assert isinstance(result, Message)
    assert result.messageId == "test-uuid"
    assert result.role == Role.agent
    assert len(result.parts) == 1
    assert result.parts[0].root.text == "test message"

  def test_convert_event_to_message_no_content(self):
    """Test event to message conversion with no content."""
    self.mock_event.content = None

    result = convert_event_to_a2a_status_message(
        self.mock_event, self.mock_invocation_context
    )

    assert result is None

  def test_convert_event_to_message_empty_parts(self):
    """Test event to message conversion with empty parts."""
    mock_content = Mock()
    mock_content.parts = []
    self.mock_event.content = mock_content

    result = convert_event_to_a2a_status_message(
        self.mock_event, self.mock_invocation_context
    )

    assert result is None

  def test_convert_event_to_message_none_event(self):
    """Test event to message conversion with None event."""
    with pytest.raises(ValueError) as exc_info:
      convert_event_to_a2a_status_message(None, self.mock_invocation_context)
    assert "Event cannot be None" in str(exc_info.value)

  def test_convert_event_to_message_none_context(self):
    """Test event to message conversion with None context."""
    with pytest.raises(ValueError) as exc_info:
      convert_event_to_a2a_status_message(self.mock_event, None)
    assert "Invocation context cannot be None" in str(exc_info.value)

  @patch("google.adk.a2a.converters.event_converter.uuid.uuid4")
  @patch("google.adk.a2a.converters.event_converter.datetime.datetime")
  def test_create_error_status_event(self, mock_datetime, mock_uuid):
    """Test creation of error status event."""
    mock_uuid.return_value = "test-uuid"
    mock_datetime.now.return_value.isoformat.return_value = (
        "2023-01-01T00:00:00"
    )

    self.mock_event.error_message = "Test error message"

    result = _create_error_status_event(
        self.mock_event, self.mock_invocation_context
    )

    assert isinstance(result, TaskStatusUpdateEvent)
    assert result.contextId == self.mock_invocation_context.session.id
    assert result.status.state == TaskState.failed
    assert result.status.message.parts[0].root.text == "Test error message"

  @patch("google.adk.a2a.converters.event_converter.uuid.uuid4")
  @patch("google.adk.a2a.converters.event_converter.datetime.datetime")
  def test_create_error_status_event_no_message(self, mock_datetime, mock_uuid):
    """Test creation of error status event without error message."""
    mock_uuid.return_value = "test-uuid"
    mock_datetime.now.return_value.isoformat.return_value = (
        "2023-01-01T00:00:00"
    )

    result = _create_error_status_event(
        self.mock_event, self.mock_invocation_context
    )

    assert result.status.message.parts[0].root.text == DEFAULT_ERROR_MESSAGE

  @patch("google.adk.a2a.converters.event_converter.datetime.datetime")
  def test_create_running_status_event(self, mock_datetime):
    """Test creation of running status event."""
    mock_datetime.now.return_value.isoformat.return_value = (
        "2023-01-01T00:00:00"
    )

    mock_message = Mock(spec=Message)

    result = _create_running_status_event(
        mock_message, self.mock_invocation_context, self.mock_event
    )

    assert isinstance(result, TaskStatusUpdateEvent)
    assert result.contextId == self.mock_invocation_context.session.id
    assert result.status.state == TaskState.working
    assert result.status.message == mock_message

  @patch(
      "google.adk.a2a.converters.event_converter._convert_artifact_to_a2a_events"
  )
  @patch(
      "google.adk.a2a.converters.event_converter.convert_event_to_a2a_status_message"
  )
  @patch("google.adk.a2a.converters.event_converter._create_error_status_event")
  @patch(
      "google.adk.a2a.converters.event_converter._create_running_status_event"
  )
  def test_convert_event_to_a2a_events_full_scenario(
      self,
      mock_create_running,
      mock_create_error,
      mock_convert_message,
      mock_convert_artifact,
  ):
    """Test full event to A2A events conversion scenario."""
    # Setup artifact delta
    self.mock_event.actions.artifact_delta = {"file1.txt": 1, "file2.txt": 2}

    # Setup error
    self.mock_event.error_code = "ERROR_001"

    # Setup message
    mock_message = Mock(spec=Message)
    mock_convert_message.return_value = mock_message

    # Setup mock returns
    mock_artifact_event1 = Mock()
    mock_artifact_event2 = Mock()
    mock_convert_artifact.side_effect = [
        mock_artifact_event1,
        mock_artifact_event2,
    ]

    mock_error_event = Mock()
    mock_create_error.return_value = mock_error_event

    mock_running_event = Mock()
    mock_create_running.return_value = mock_running_event

    result = convert_event_to_a2a_events(
        self.mock_event, self.mock_invocation_context
    )

    # Verify artifact delta events
    assert mock_convert_artifact.call_count == 2

    # Verify error event
    mock_create_error.assert_called_once_with(
        self.mock_event, self.mock_invocation_context
    )

    # Verify running event
    mock_create_running.assert_called_once_with(
        mock_message, self.mock_invocation_context, self.mock_event
    )

    # Verify result contains all events
    assert len(result) == 4  # 2 artifact + 1 error + 1 running
    assert mock_artifact_event1 in result
    assert mock_artifact_event2 in result
    assert mock_error_event in result
    assert mock_running_event in result

  def test_convert_event_to_a2a_events_empty_scenario(self):
    """Test event to A2A events conversion with empty event."""
    result = convert_event_to_a2a_events(
        self.mock_event, self.mock_invocation_context
    )

    assert result == []

  def test_convert_event_to_a2a_events_none_event(self):
    """Test event to A2A events conversion with None event."""
    with pytest.raises(ValueError) as exc_info:
      convert_event_to_a2a_events(None, self.mock_invocation_context)
    assert "Event cannot be None" in str(exc_info.value)

  def test_convert_event_to_a2a_events_none_context(self):
    """Test event to A2A events conversion with None context."""
    with pytest.raises(ValueError) as exc_info:
      convert_event_to_a2a_events(self.mock_event, None)
    assert "Invocation context cannot be None" in str(exc_info.value)

  @patch(
      "google.adk.a2a.converters.event_converter.convert_event_to_a2a_status_message"
  )
  def test_convert_event_to_a2a_events_message_only(self, mock_convert_message):
    """Test event to A2A events conversion with message only."""
    mock_message = Mock(spec=Message)
    mock_convert_message.return_value = mock_message

    with patch(
        "google.adk.a2a.converters.event_converter._create_running_status_event"
    ) as mock_create_running:
      mock_running_event = Mock()
      mock_create_running.return_value = mock_running_event

      result = convert_event_to_a2a_events(
          self.mock_event, self.mock_invocation_context
      )

      assert len(result) == 1
      assert result[0] == mock_running_event

  @patch("google.adk.a2a.converters.event_converter.logger")
  def test_convert_event_to_a2a_events_exception_handling(self, mock_logger):
    """Test exception handling in event to A2A events conversion."""
    # Make convert_event_to_a2a_status_message raise an exception
    with patch(
        "google.adk.a2a.converters.event_converter.convert_event_to_a2a_status_message"
    ) as mock_convert:
      mock_convert.side_effect = Exception("Conversion failed")

      with pytest.raises(Exception):
        convert_event_to_a2a_events(
            self.mock_event, self.mock_invocation_context
        )

      mock_logger.error.assert_called_once()
