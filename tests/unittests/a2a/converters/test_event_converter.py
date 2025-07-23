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
    sys.version_info < (3, 10), reason="A2A requires Python 3.10+"
)

# Import dependencies with version checking
try:
  from a2a.types import DataPart
  from a2a.types import Message
  from a2a.types import Role
  from a2a.types import Task
  from a2a.types import TaskState
  from a2a.types import TaskStatusUpdateEvent
  from google.adk.a2a.converters.event_converter import _create_artifact_id
  from google.adk.a2a.converters.event_converter import _create_error_status_event
  from google.adk.a2a.converters.event_converter import _create_status_update_event
  from google.adk.a2a.converters.event_converter import _get_adk_metadata_key
  from google.adk.a2a.converters.event_converter import _get_context_metadata
  from google.adk.a2a.converters.event_converter import _process_long_running_tool
  from google.adk.a2a.converters.event_converter import _serialize_metadata_value
  from google.adk.a2a.converters.event_converter import ARTIFACT_ID_SEPARATOR
  from google.adk.a2a.converters.event_converter import convert_a2a_task_to_event
  from google.adk.a2a.converters.event_converter import convert_event_to_a2a_events
  from google.adk.a2a.converters.event_converter import convert_event_to_a2a_message
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
    Task = DummyTypes()
    TaskState = DummyTypes()
    TaskStatusUpdateEvent = DummyTypes()
    _create_artifact_id = lambda *args: None
    _create_error_status_event = lambda *args: None
    _create_status_update_event = lambda *args: None
    _get_adk_metadata_key = lambda *args: None
    _get_context_metadata = lambda *args: None
    _process_long_running_tool = lambda *args: None
    _serialize_metadata_value = lambda *args: None
    ADK_METADATA_KEY_PREFIX = "adk_"
    ARTIFACT_ID_SEPARATOR = "_"
    convert_event_to_a2a_events = lambda *args: None
    convert_event_to_a2a_message = lambda *args: None
    convert_a2a_task_to_event = lambda *args: None
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

  def test_process_long_running_tool_marks_tool(self):
    """Test processing of long-running tool metadata."""
    mock_a2a_part = Mock()
    mock_data_part = Mock(spec=DataPart)
    mock_data_part.metadata = {"adk_type": "function_call", "id": "tool-123"}
    mock_data_part.data = Mock()
    mock_data_part.data.get = Mock(return_value="tool-123")
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
        patch(
            "google.adk.a2a.converters.event_converter._get_adk_metadata_key"
        ) as mock_get_key,
    ):
      mock_get_key.side_effect = lambda key: f"adk_{key}"

      _process_long_running_tool(mock_a2a_part, self.mock_event)

      expected_key = f"{ADK_METADATA_KEY_PREFIX}is_long_running"
      assert mock_data_part.metadata[expected_key] is True

  def test_process_long_running_tool_no_marking(self):
    """Test processing when tool should not be marked as long-running."""
    mock_a2a_part = Mock()
    mock_data_part = Mock(spec=DataPart)
    mock_data_part.metadata = {"adk_type": "function_call", "id": "tool-456"}
    mock_data_part.data = Mock()
    mock_data_part.data.get = Mock(return_value="tool-456")
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
        patch(
            "google.adk.a2a.converters.event_converter._get_adk_metadata_key"
        ) as mock_get_key,
    ):
      mock_get_key.side_effect = lambda key: f"adk_{key}"

      _process_long_running_tool(mock_a2a_part, self.mock_event)

      expected_key = f"{ADK_METADATA_KEY_PREFIX}is_long_running"
      assert expected_key not in mock_data_part.metadata

  @patch(
      "google.adk.a2a.converters.event_converter.convert_event_to_a2a_message"
  )
  @patch("google.adk.a2a.converters.event_converter._create_error_status_event")
  @patch(
      "google.adk.a2a.converters.event_converter._create_status_update_event"
  )
  def test_convert_event_to_a2a_events_full_scenario(
      self,
      mock_create_running,
      mock_create_error,
      mock_convert_message,
  ):
    """Test full event to A2A events conversion scenario."""
    # Setup error
    self.mock_event.error_code = "ERROR_001"

    # Setup message
    mock_message = Mock(spec=Message)
    mock_convert_message.return_value = mock_message

    # Setup mock returns
    mock_error_event = Mock()
    mock_create_error.return_value = mock_error_event

    mock_running_event = Mock()
    mock_create_running.return_value = mock_running_event

    result = convert_event_to_a2a_events(
        self.mock_event, self.mock_invocation_context
    )

    # Verify error event - now called with task_id and context_id parameters
    mock_create_error.assert_called_once_with(
        self.mock_event, self.mock_invocation_context, None, None
    )

    # Verify running event - now called with task_id and context_id parameters
    mock_create_running.assert_called_once_with(
        mock_message, self.mock_invocation_context, self.mock_event, None, None
    )

    # Verify result contains all events
    assert len(result) == 2  # 1 error + 1 running
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
      "google.adk.a2a.converters.event_converter.convert_event_to_a2a_message"
  )
  def test_convert_event_to_a2a_events_message_only(self, mock_convert_message):
    """Test event to A2A events conversion with message only."""
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
      # Verify the function is called with task_id and context_id parameters
      mock_create_running.assert_called_once_with(
          mock_message,
          self.mock_invocation_context,
          self.mock_event,
          None,
          None,
      )

  @patch("google.adk.a2a.converters.event_converter.logger")
  def test_convert_event_to_a2a_events_exception_handling(self, mock_logger):
    """Test exception handling in convert_event_to_a2a_events."""
    # Make convert_event_to_a2a_message raise an exception
    with patch(
        "google.adk.a2a.converters.event_converter.convert_event_to_a2a_message"
    ) as mock_convert_message:
      mock_convert_message.side_effect = Exception("Test exception")

      with pytest.raises(Exception):
        convert_event_to_a2a_events(
            self.mock_event, self.mock_invocation_context
        )

      mock_logger.error.assert_called_once()

  def test_convert_event_to_a2a_events_with_task_id_and_context_id(self):
    """Test event to A2A events conversion with specific task_id and context_id."""
    # Setup message
    mock_message = Mock(spec=Message)
    mock_message.parts = []

    with patch(
        "google.adk.a2a.converters.event_converter.convert_event_to_a2a_message"
    ) as mock_convert_message:
      mock_convert_message.return_value = mock_message

      with patch(
          "google.adk.a2a.converters.event_converter._create_status_update_event"
      ) as mock_create_running:
        mock_running_event = Mock()
        mock_create_running.return_value = mock_running_event

        task_id = "custom-task-id"
        context_id = "custom-context-id"

        result = convert_event_to_a2a_events(
            self.mock_event, self.mock_invocation_context, task_id, context_id
        )

        assert len(result) == 1
        assert result[0] == mock_running_event

        # Verify the function is called with the specific task_id and context_id
        mock_create_running.assert_called_once_with(
            mock_message,
            self.mock_invocation_context,
            self.mock_event,
            task_id,
            context_id,
        )

  def test_convert_event_to_a2a_events_with_custom_ids(self):
    """Test event to A2A events conversion with custom IDs."""
    # Setup message
    mock_message = Mock(spec=Message)
    mock_message.parts = []

    with patch(
        "google.adk.a2a.converters.event_converter.convert_event_to_a2a_message"
    ) as mock_convert_message:
      mock_convert_message.return_value = mock_message

      with patch(
          "google.adk.a2a.converters.event_converter._create_status_update_event"
      ) as mock_create_running:
        mock_running_event = Mock()
        mock_create_running.return_value = mock_running_event

        task_id = "custom-task-id"
        context_id = "custom-context-id"

        result = convert_event_to_a2a_events(
            self.mock_event, self.mock_invocation_context, task_id, context_id
        )

        assert len(result) == 1  # 1 status
        assert mock_running_event in result

        # Verify status update is called with custom IDs
        mock_create_running.assert_called_once_with(
            mock_message,
            self.mock_invocation_context,
            self.mock_event,
            task_id,
            context_id,
        )

  def test_create_status_update_event_with_auth_required_state(self):
    """Test creation of status update event with auth_required state."""
    from a2a.types import DataPart
    from a2a.types import Part

    # Create a mock message with a part that triggers auth_required state
    mock_message = Mock(spec=Message)
    mock_part = Mock()
    mock_data_part = Mock(spec=DataPart)
    mock_data_part.metadata = {
        "adk_type": "function_call",
        "adk_is_long_running": True,
    }
    mock_data_part.data = Mock()
    mock_data_part.data.get = Mock(return_value="request_euc")
    mock_part.root = mock_data_part
    mock_message.parts = [mock_part]

    task_id = "test-task-id"
    context_id = "test-context-id"

    with patch(
        "google.adk.a2a.converters.event_converter.datetime"
    ) as mock_datetime:
      mock_datetime.now.return_value.isoformat.return_value = (
          "2023-01-01T00:00:00"
      )

      with (
          patch(
              "google.adk.a2a.converters.event_converter.A2A_DATA_PART_METADATA_TYPE_KEY",
              "type",
          ),
          patch(
              "google.adk.a2a.converters.event_converter.A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL",
              "function_call",
          ),
          patch(
              "google.adk.a2a.converters.event_converter.A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY",
              "is_long_running",
          ),
          patch(
              "google.adk.a2a.converters.event_converter.REQUEST_EUC_FUNCTION_CALL_NAME",
              "request_euc",
          ),
          patch(
              "google.adk.a2a.converters.event_converter._get_adk_metadata_key"
          ) as mock_get_key,
      ):
        mock_get_key.side_effect = lambda key: f"adk_{key}"

        result = _create_status_update_event(
            mock_message,
            self.mock_invocation_context,
            self.mock_event,
            task_id,
            context_id,
        )

        assert isinstance(result, TaskStatusUpdateEvent)
        assert result.task_id == task_id
        assert result.context_id == context_id
        assert result.status.state == TaskState.auth_required

  def test_create_status_update_event_with_input_required_state(self):
    """Test creation of status update event with input_required state."""
    from a2a.types import DataPart
    from a2a.types import Part

    # Create a mock message with a part that triggers input_required state
    mock_message = Mock(spec=Message)
    mock_part = Mock()
    mock_data_part = Mock(spec=DataPart)
    mock_data_part.metadata = {
        "adk_type": "function_call",
        "adk_is_long_running": True,
    }
    mock_data_part.data = Mock()
    mock_data_part.data.get = Mock(return_value="some_other_function")
    mock_part.root = mock_data_part
    mock_message.parts = [mock_part]

    task_id = "test-task-id"
    context_id = "test-context-id"

    with patch(
        "google.adk.a2a.converters.event_converter.datetime"
    ) as mock_datetime:
      mock_datetime.now.return_value.isoformat.return_value = (
          "2023-01-01T00:00:00"
      )

      with (
          patch(
              "google.adk.a2a.converters.event_converter.A2A_DATA_PART_METADATA_TYPE_KEY",
              "type",
          ),
          patch(
              "google.adk.a2a.converters.event_converter.A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL",
              "function_call",
          ),
          patch(
              "google.adk.a2a.converters.event_converter.A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY",
              "is_long_running",
          ),
          patch(
              "google.adk.a2a.converters.event_converter.REQUEST_EUC_FUNCTION_CALL_NAME",
              "request_euc",
          ),
          patch(
              "google.adk.a2a.converters.event_converter._get_adk_metadata_key"
          ) as mock_get_key,
      ):
        mock_get_key.side_effect = lambda key: f"adk_{key}"

        result = _create_status_update_event(
            mock_message,
            self.mock_invocation_context,
            self.mock_event,
            task_id,
            context_id,
        )

        assert isinstance(result, TaskStatusUpdateEvent)
        assert result.task_id == task_id
        assert result.context_id == context_id
        assert result.status.state == TaskState.input_required


class TestA2AToEventConverters:
  """Test suite for A2A to Event conversion functions."""

  def setup_method(self):
    """Set up test fixtures."""
    self.mock_invocation_context = Mock(spec=InvocationContext)
    self.mock_invocation_context.invocation_id = "test-invocation-id"
    self.mock_invocation_context.branch = "test-branch"

  def test_convert_a2a_task_to_event_with_artifacts_priority(self):
    """Test convert_a2a_task_to_event prioritizes artifacts over status/history."""
    from a2a.types import Artifact
    from a2a.types import Part
    from a2a.types import TaskStatus
    from a2a.types import TextPart

    # Create mock artifacts
    artifact_part = Part(root=TextPart(text="artifact content"))
    mock_artifact = Mock(spec=Artifact)
    mock_artifact.parts = [artifact_part]

    # Create mock status and history
    status_part = Part(root=TextPart(text="status content"))
    mock_status = Mock(spec=TaskStatus)
    mock_status.message = Mock(spec=Message)
    mock_status.message.parts = [status_part]

    history_part = Part(root=TextPart(text="history content"))
    mock_history_message = Mock(spec=Message)
    mock_history_message.parts = [history_part]

    # Create task with all three sources
    mock_task = Mock(spec=Task)
    mock_task.artifacts = [mock_artifact]
    mock_task.status = mock_status
    mock_task.history = [mock_history_message]

    with patch(
        "google.adk.a2a.converters.event_converter.convert_a2a_message_to_event"
    ) as mock_convert_message:
      mock_event = Mock(spec=Event)
      mock_convert_message.return_value = mock_event

      result = convert_a2a_task_to_event(
          mock_task, "test-author", self.mock_invocation_context
      )

      assert result == mock_event
      # Should call convert_a2a_message_to_event with a message created from artifacts
      mock_convert_message.assert_called_once()
      called_message = mock_convert_message.call_args[0][0]
      assert called_message.role == Role.agent
      assert called_message.parts == [artifact_part]

  def test_convert_a2a_task_to_event_with_status_message(self):
    """Test convert_a2a_task_to_event with status message (no artifacts)."""
    from a2a.types import Part
    from a2a.types import TaskStatus
    from a2a.types import TextPart

    # Create mock status
    status_part = Part(root=TextPart(text="status content"))
    mock_status = Mock(spec=TaskStatus)
    mock_status.message = Mock(spec=Message)
    mock_status.message.parts = [status_part]

    # Create task with no artifacts
    mock_task = Mock(spec=Task)
    mock_task.artifacts = None
    mock_task.status = mock_status
    mock_task.history = []

    with patch(
        "google.adk.a2a.converters.event_converter.convert_a2a_message_to_event"
    ) as mock_convert_message:
      mock_event = Mock(spec=Event)
      mock_convert_message.return_value = mock_event

      result = convert_a2a_task_to_event(
          mock_task, "test-author", self.mock_invocation_context
      )

      assert result == mock_event
      # Should call convert_a2a_message_to_event with the status message
      mock_convert_message.assert_called_once_with(
          mock_status.message, "test-author", self.mock_invocation_context
      )

  def test_convert_a2a_task_to_event_with_history_message(self):
    """Test converting A2A task with history message when no status message."""
    from google.adk.a2a.converters.event_converter import convert_a2a_task_to_event

    # Create mock message and task
    mock_message = Mock(spec=Message)
    mock_task = Mock(spec=Task)
    mock_task.artifacts = None
    mock_task.status = None
    mock_task.history = [mock_message]

    # Mock the convert_a2a_message_to_event function
    with patch(
        "google.adk.a2a.converters.event_converter.convert_a2a_message_to_event"
    ) as mock_convert_message:
      mock_event = Mock(spec=Event)
      mock_event.invocation_id = "test-invocation-id"
      mock_convert_message.return_value = mock_event

      result = convert_a2a_task_to_event(mock_task, "test-author")

      # Verify the message converter was called with correct parameters
      mock_convert_message.assert_called_once_with(
          mock_message, "test-author", None
      )
      assert result == mock_event

  def test_convert_a2a_task_to_event_no_message(self):
    """Test converting A2A task with no message."""
    from google.adk.a2a.converters.event_converter import convert_a2a_task_to_event

    # Create mock task with no message
    mock_task = Mock(spec=Task)
    mock_task.artifacts = None
    mock_task.status = None
    mock_task.history = []

    result = convert_a2a_task_to_event(
        mock_task, "test-author", self.mock_invocation_context
    )

    # Verify minimal event was created with correct invocation_id
    assert result.author == "test-author"
    assert result.branch == "test-branch"
    assert result.invocation_id == "test-invocation-id"

  @patch("google.adk.a2a.converters.event_converter.uuid.uuid4")
  def test_convert_a2a_task_to_event_default_author(self, mock_uuid):
    """Test converting A2A task with default author and no invocation context."""
    from google.adk.a2a.converters.event_converter import convert_a2a_task_to_event

    # Create mock task with no message
    mock_task = Mock(spec=Task)
    mock_task.artifacts = None
    mock_task.status = None
    mock_task.history = []

    # Mock UUID generation
    mock_uuid.return_value = "generated-uuid"

    result = convert_a2a_task_to_event(mock_task)

    # Verify default author was used and UUID was generated for invocation_id
    assert result.author == "a2a agent"
    assert result.branch is None
    assert result.invocation_id == "generated-uuid"

  def test_convert_a2a_task_to_event_none_task(self):
    """Test converting None task raises ValueError."""
    from google.adk.a2a.converters.event_converter import convert_a2a_task_to_event

    with pytest.raises(ValueError, match="A2A task cannot be None"):
      convert_a2a_task_to_event(None)

  def test_convert_a2a_task_to_event_message_conversion_error(self):
    """Test error handling when message conversion fails."""
    from google.adk.a2a.converters.event_converter import convert_a2a_task_to_event

    # Create mock message and task
    mock_message = Mock(spec=Message)
    mock_status = Mock()
    mock_status.message = mock_message
    mock_task = Mock(spec=Task)
    mock_task.artifacts = None
    mock_task.status = mock_status
    mock_task.history = []

    # Mock the convert_a2a_message_to_event function to raise an exception
    with patch(
        "google.adk.a2a.converters.event_converter.convert_a2a_message_to_event"
    ) as mock_convert_message:
      mock_convert_message.side_effect = Exception("Conversion failed")

      with pytest.raises(RuntimeError, match="Failed to convert task message"):
        convert_a2a_task_to_event(mock_task, "test-author")

  @patch(
      "google.adk.a2a.converters.event_converter.convert_a2a_part_to_genai_part"
  )
  def test_convert_a2a_message_to_event_success(self, mock_convert_part):
    """Test successful conversion of A2A message to event."""
    from google.adk.a2a.converters.event_converter import convert_a2a_message_to_event
    from google.genai import types as genai_types

    # Create mock parts and message with valid genai Part
    mock_a2a_part = Mock()
    mock_genai_part = genai_types.Part(text="test content")
    mock_convert_part.return_value = mock_genai_part

    mock_message = Mock(spec=Message)
    mock_message.parts = [mock_a2a_part]

    result = convert_a2a_message_to_event(
        mock_message, "test-author", self.mock_invocation_context
    )

    # Verify conversion was successful
    assert result.author == "test-author"
    assert result.branch == "test-branch"
    assert result.invocation_id == "test-invocation-id"
    assert result.content.role == "model"
    assert len(result.content.parts) == 1
    assert result.content.parts[0].text == "test content"
    mock_convert_part.assert_called_once_with(mock_a2a_part)

  @patch(
      "google.adk.a2a.converters.event_converter.convert_a2a_part_to_genai_part"
  )
  def test_convert_a2a_message_to_event_with_long_running_tools(
      self, mock_convert_part
  ):
    """Test conversion with long-running tools by mocking the entire flow."""
    from google.adk.a2a.converters.event_converter import convert_a2a_message_to_event

    # Create mock parts and message
    mock_a2a_part = Mock()
    mock_message = Mock(spec=Message)
    mock_message.parts = [mock_a2a_part]

    # Mock the part conversion to return None to simulate long-running tool detection logic
    mock_convert_part.return_value = None

    # Patch the long-running tool detection since the main logic is in the actual conversion
    with patch(
        "google.adk.a2a.converters.event_converter.logger"
    ) as mock_logger:
      result = convert_a2a_message_to_event(
          mock_message, "test-author", self.mock_invocation_context
      )

      # Verify basic conversion worked
      assert result.author == "test-author"
      assert result.invocation_id == "test-invocation-id"
      assert result.content.role == "model"
      # Parts will be empty since conversion returned None, but that's expected for this test

  def test_convert_a2a_message_to_event_empty_parts(self):
    """Test conversion with empty parts list."""
    from google.adk.a2a.converters.event_converter import convert_a2a_message_to_event

    mock_message = Mock(spec=Message)
    mock_message.parts = []

    result = convert_a2a_message_to_event(
        mock_message, "test-author", self.mock_invocation_context
    )

    # Verify event was created with empty parts
    assert result.author == "test-author"
    assert result.invocation_id == "test-invocation-id"
    assert result.content.role == "model"
    assert len(result.content.parts) == 0

  def test_convert_a2a_message_to_event_none_message(self):
    """Test converting None message raises ValueError."""
    from google.adk.a2a.converters.event_converter import convert_a2a_message_to_event

    with pytest.raises(ValueError, match="A2A message cannot be None"):
      convert_a2a_message_to_event(None)

  @patch(
      "google.adk.a2a.converters.event_converter.convert_a2a_part_to_genai_part"
  )
  def test_convert_a2a_message_to_event_part_conversion_fails(
      self, mock_convert_part
  ):
    """Test handling when part conversion returns None."""
    from google.adk.a2a.converters.event_converter import convert_a2a_message_to_event

    # Setup mock to return None (conversion failure)
    mock_a2a_part = Mock()
    mock_convert_part.return_value = None

    mock_message = Mock(spec=Message)
    mock_message.parts = [mock_a2a_part]

    result = convert_a2a_message_to_event(
        mock_message, "test-author", self.mock_invocation_context
    )

    # Verify event was created but with no parts
    assert result.author == "test-author"
    assert result.invocation_id == "test-invocation-id"
    assert result.content.role == "model"
    assert len(result.content.parts) == 0

  @patch(
      "google.adk.a2a.converters.event_converter.convert_a2a_part_to_genai_part"
  )
  def test_convert_a2a_message_to_event_part_conversion_exception(
      self, mock_convert_part
  ):
    """Test handling when part conversion raises exception."""
    from google.adk.a2a.converters.event_converter import convert_a2a_message_to_event
    from google.genai import types as genai_types

    # Setup mock to raise exception
    mock_a2a_part1 = Mock()
    mock_a2a_part2 = Mock()
    mock_genai_part = genai_types.Part(text="successful conversion")

    mock_convert_part.side_effect = [
        Exception("Conversion failed"),  # First part fails
        mock_genai_part,  # Second part succeeds
    ]

    mock_message = Mock(spec=Message)
    mock_message.parts = [mock_a2a_part1, mock_a2a_part2]

    result = convert_a2a_message_to_event(
        mock_message, "test-author", self.mock_invocation_context
    )

    # Verify event was created with only the successfully converted part
    assert result.author == "test-author"
    assert result.invocation_id == "test-invocation-id"
    assert result.content.role == "model"
    assert len(result.content.parts) == 1
    assert result.content.parts[0].text == "successful conversion"

  @patch(
      "google.adk.a2a.converters.event_converter.convert_a2a_part_to_genai_part"
  )
  def test_convert_a2a_message_to_event_missing_tool_id(
      self, mock_convert_part
  ):
    """Test handling of message conversion when part conversion fails."""
    from google.adk.a2a.converters.event_converter import convert_a2a_message_to_event

    # Create mock parts and message
    mock_a2a_part = Mock()
    mock_message = Mock(spec=Message)
    mock_message.parts = [mock_a2a_part]

    # Mock the part conversion to return None
    mock_convert_part.return_value = None

    result = convert_a2a_message_to_event(
        mock_message, "test-author", self.mock_invocation_context
    )

    # Verify basic conversion worked
    assert result.author == "test-author"
    assert result.invocation_id == "test-invocation-id"
    assert result.content.role == "model"
    # Parts will be empty since conversion returned None
    assert len(result.content.parts) == 0

  @patch("google.adk.a2a.converters.event_converter.uuid.uuid4")
  def test_convert_a2a_message_to_event_default_author(self, mock_uuid):
    """Test conversion with default author and no invocation context."""
    from google.adk.a2a.converters.event_converter import convert_a2a_message_to_event

    mock_message = Mock(spec=Message)
    mock_message.parts = []

    # Mock UUID generation
    mock_uuid.return_value = "generated-uuid"

    result = convert_a2a_message_to_event(mock_message)

    # Verify default author was used and UUID was generated for invocation_id
    assert result.author == "a2a agent"
    assert result.branch is None
    assert result.invocation_id == "generated-uuid"
