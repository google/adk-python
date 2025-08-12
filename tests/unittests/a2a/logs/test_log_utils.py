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

"""Tests for log_utils module."""

import json
from unittest.mock import Mock
from unittest.mock import patch

import pytest

# Import the actual A2A types that we need to mock
try:
  from a2a.types import DataPart as A2ADataPart
  from a2a.types import Message as A2AMessage
  from a2a.types import MessageSendConfiguration
  from a2a.types import MessageSendParams
  from a2a.types import Part as A2APart
  from a2a.types import Role
  from a2a.types import SendMessageRequest
  from a2a.types import Task as A2ATask
  from a2a.types import TaskState
  from a2a.types import TaskStatus
  from a2a.types import TextPart as A2ATextPart

  A2A_AVAILABLE = True
except ImportError:
  A2A_AVAILABLE = False


class TestBuildMessagePartLog:
  """Test suite for build_message_part_log function."""

  @pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A types not available")
  def test_text_part_short_text(self):
    """Test TextPart with short text."""
    # Import here to avoid import issues at module level
    from google.adk.a2a.logs.log_utils import build_message_part_log

    # Create real A2A objects
    text_part = A2ATextPart(text="Hello, world!")
    part = A2APart(root=text_part)

    result = build_message_part_log(part)

    assert result == "TextPart: Hello, world!"

  @pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A types not available")
  def test_text_part_long_text(self):
    """Test TextPart with long text that gets truncated."""
    from google.adk.a2a.logs.log_utils import build_message_part_log

    long_text = "x" * 150  # Long text that should be truncated
    text_part = A2ATextPart(text=long_text)
    part = A2APart(root=text_part)

    result = build_message_part_log(part)

    expected = f"TextPart: {'x' * 100}..."
    assert result == expected

  @pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A types not available")
  def test_data_part_simple_data(self):
    """Test DataPart with simple data."""
    from google.adk.a2a.logs.log_utils import build_message_part_log

    data_part = A2ADataPart(data={"key1": "value1", "key2": 42})
    part = A2APart(root=data_part)

    result = build_message_part_log(part)

    expected_data = {"key1": "value1", "key2": 42}
    expected = f"DataPart: {json.dumps(expected_data, indent=2)}"
    assert result == expected

  @pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A types not available")
  def test_data_part_large_values(self):
    """Test DataPart with large values that get summarized."""
    from google.adk.a2a.logs.log_utils import build_message_part_log

    large_dict = {f"key{i}": f"value{i}" for i in range(50)}
    large_list = list(range(100))

    data_part = A2ADataPart(
        data={
            "small_value": "hello",
            "large_dict": large_dict,
            "large_list": large_list,
            "normal_int": 42,
        }
    )
    part = A2APart(root=data_part)

    result = build_message_part_log(part)

    # Large values should be replaced with type names
    assert "small_value" in result
    assert "hello" in result
    assert "<dict>" in result
    assert "<list>" in result
    assert "normal_int" in result
    assert "42" in result

  def test_other_part_type(self):
    """Test handling of other part types (not Text or Data)."""
    from google.adk.a2a.logs.log_utils import build_message_part_log

    # Create a mock part that will fall through to the else case
    mock_root = Mock()
    mock_root.__class__.__name__ = "MockOtherPart"
    # Ensure metadata attribute doesn't exist or returns None to avoid JSON serialization issues
    mock_root.metadata = None

    mock_part = Mock()
    mock_part.root = mock_root
    mock_part.model_dump_json.return_value = '{"some": "data"}'

    result = build_message_part_log(mock_part)

    expected = 'MockOtherPart: {"some": "data"}'
    assert result == expected


class TestBuildA2ARequestLog:
  """Test suite for build_a2a_request_log function."""

  def test_request_with_parts_and_config(self):
    """Test request logging with message parts and configuration."""
    from google.adk.a2a.logs.log_utils import build_a2a_request_log

    # Create mock request with all components
    req = SendMessageRequest(
        id="req-123",
        method="message/send",
        jsonrpc="2.0",
        params=MessageSendParams(
            message=A2AMessage(
                message_id="msg-456",
                role="user",
                task_id="task-789",
                context_id="ctx-101",
                parts=[
                    A2APart(root=A2ATextPart(text="Part 1")),
                    A2APart(root=A2ATextPart(text="Part 2")),
                ],
                metadata={"msg_key": "msg_value"},
            ),
            configuration=MessageSendConfiguration(
                accepted_output_modes=["text", "image"],
                blocking=True,
                history_length=10,
                push_notification_config=None,
            ),
            metadata={"key1": "value1"},
        ),
    )

    with patch(
        "google.adk.a2a.logs.log_utils.build_message_part_log"
    ) as mock_build_part:
      mock_build_part.side_effect = lambda part: f"Mock part: {id(part)}"

      result = build_a2a_request_log(req)

    # Verify all components are present
    assert "req-123" in result
    assert "message/send" in result
    assert "2.0" in result
    assert "msg-456" in result
    assert "user" in result
    assert "task-789" in result
    assert "ctx-101" in result
    assert "Part 0:" in result
    assert "Part 1:" in result
    assert '"blocking": true' in result
    assert '"history_length": 10' in result
    assert '"key1": "value1"' in result

  def test_request_without_parts(self):
    """Test request logging without message parts."""
    from google.adk.a2a.logs.log_utils import build_a2a_request_log

    req = Mock()
    req.id = "req-123"
    req.method = "message/send"
    req.jsonrpc = "2.0"

    req.params.message.message_id = "msg-456"
    req.params.message.role = "user"
    req.params.message.task_id = "task-789"
    req.params.message.context_id = "ctx-101"
    req.params.message.parts = None  # No parts
    req.params.message.metadata = None  # No message metadata

    req.params.configuration = None  # No configuration
    req.params.metadata = None  # No metadata

    result = build_a2a_request_log(req)

    assert "No parts" in result
    assert "Configuration:\nNone" in result
    # When metadata is None, it's not included in the output
    assert "Metadata:" not in result

  def test_request_with_empty_parts_list(self):
    """Test request logging with empty parts list."""
    from google.adk.a2a.logs.log_utils import build_a2a_request_log

    req = Mock()
    req.id = "req-123"
    req.method = "sendMessage"
    req.jsonrpc = "2.0"

    req.params.message.message_id = "msg-456"
    req.params.message.role = "user"
    req.params.message.task_id = "task-789"
    req.params.message.context_id = "ctx-101"
    req.params.message.parts = []  # Empty parts list
    req.params.message.metadata = None  # No message metadata

    req.params.configuration = None
    req.params.metadata = None

    result = build_a2a_request_log(req)

    assert "No parts" in result


class TestBuildA2AResponseLog:
  """Test suite for build_a2a_response_log function."""

  def test_error_response(self):
    """Test error response logging."""
    from google.adk.a2a.logs.log_utils import build_a2a_response_log

    resp = Mock()
    resp.root.error.code = 500
    resp.root.error.message = "Internal Server Error"
    resp.root.error.data = {"details": "Something went wrong"}
    resp.root.id = "resp-error"
    resp.root.jsonrpc = "2.0"

    result = build_a2a_response_log(resp)

    assert "Type: ERROR" in result
    assert "Error Code: 500" in result
    assert "Internal Server Error" in result
    assert '"details": "Something went wrong"' in result
    assert "resp-error" in result
    assert "2.0" in result

  def test_error_response_no_data(self):
    """Test error response logging without error data."""
    from google.adk.a2a.logs.log_utils import build_a2a_response_log

    resp = Mock()
    resp.root.error.code = 404
    resp.root.error.message = "Not Found"
    resp.root.error.data = None
    resp.root.id = "resp-404"
    resp.root.jsonrpc = "2.0"

    result = build_a2a_response_log(resp)

    assert "Type: ERROR" in result
    assert "Error Code: 404" in result
    assert "Not Found" in result
    assert "Error Data: None" in result

  @pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A types not available")
  def test_success_response_with_task(self):
    """Test success response logging with Task result."""
    # Use module-level imported types consistently
    from google.adk.a2a.logs.log_utils import build_a2a_response_log

    task_status = TaskStatus(state=TaskState.working)
    task = A2ATask(id="task-123", context_id="ctx-456", status=task_status)

    resp = Mock()
    resp.root.result = task
    resp.root.id = "resp-789"
    resp.root.jsonrpc = "2.0"

    # Remove error attribute to ensure success path
    delattr(resp.root, "error")

    result = build_a2a_response_log(resp)

    assert "Type: SUCCESS" in result
    assert "Result Type: Task" in result
    assert "Task ID: task-123" in result
    assert "Context ID: ctx-456" in result
    # Handle both structured format and JSON fallback due to potential isinstance failures
    assert (
        "Status State: TaskState.working" in result
        or "Status State: working" in result
        or '"state":"working"' in result
        or '"state": "working"' in result
    )

  @pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A types not available")
  def test_success_response_with_task_and_status_message(self):
    """Test success response with Task that has status message."""
    from google.adk.a2a.logs.log_utils import build_a2a_response_log

    # Create status message using module-level imported types
    status_message = A2AMessage(
        message_id="status-msg-123",
        role=Role.agent,
        parts=[
            A2APart(root=A2ATextPart(text="Status part 1")),
            A2APart(root=A2ATextPart(text="Status part 2")),
        ],
    )

    task_status = TaskStatus(state=TaskState.working, message=status_message)
    task = A2ATask(
        id="task-123",
        context_id="ctx-456",
        status=task_status,
        history=[],
        artifacts=None,
    )

    resp = Mock()
    resp.root.result = task
    resp.root.id = "resp-789"
    resp.root.jsonrpc = "2.0"

    # Remove error attribute to ensure success path
    delattr(resp.root, "error")

    result = build_a2a_response_log(resp)

    assert "ID: status-msg-123" in result
    # Handle both structured format and JSON fallback
    assert (
        "Role: Role.agent" in result
        or "Role: agent" in result
        or '"role":"agent"' in result
        or '"role": "agent"' in result
    )
    assert "Message Parts:" in result

  @pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A types not available")
  def test_success_response_with_message(self):
    """Test success response logging with Message result."""
    from google.adk.a2a.logs.log_utils import build_a2a_response_log

    # Use module-level imported types consistently
    message = A2AMessage(
        message_id="msg-123",
        role=Role.agent,
        task_id="task-456",
        context_id="ctx-789",
        parts=[A2APart(root=A2ATextPart(text="Message part 1"))],
    )

    resp = Mock()
    resp.root.result = message
    resp.root.id = "resp-101"
    resp.root.jsonrpc = "2.0"

    # Remove error attribute to ensure success path
    delattr(resp.root, "error")

    result = build_a2a_response_log(resp)

    assert "Type: SUCCESS" in result
    assert "Result Type: Message" in result
    assert "Message ID: msg-123" in result
    # Handle both structured format and JSON fallback
    assert (
        "Role: Role.agent" in result
        or "Role: agent" in result
        or '"role":"agent"' in result
        or '"role": "agent"' in result
    )
    assert "Task ID: task-456" in result
    assert "Context ID: ctx-789" in result

  def test_success_response_with_message_no_parts(self):
    """Test success response with Message that has no parts."""
    from google.adk.a2a.logs.log_utils import build_a2a_response_log

    # Use mock for this case since we want to test empty parts handling
    message = Mock()
    message.__class__.__name__ = "Message"
    message.message_id = "msg-empty"
    message.role = "agent"
    message.task_id = "task-empty"
    message.context_id = "ctx-empty"
    message.parts = None  # No parts
    message.model_dump_json.return_value = '{"message": "empty"}'

    resp = Mock()
    resp.root.result = message
    resp.root.id = "resp-empty"
    resp.root.jsonrpc = "2.0"

    # Remove error attribute to ensure success path
    delattr(resp.root, "error")

    result = build_a2a_response_log(resp)

    assert "Type: SUCCESS" in result
    assert "Result Type: Message" in result

  def test_success_response_with_other_result_type(self):
    """Test success response with result type that's not Task or Message."""
    from google.adk.a2a.logs.log_utils import build_a2a_response_log

    other_result = Mock()
    other_result.__class__.__name__ = "OtherResult"
    other_result.model_dump_json.return_value = '{"other": "data"}'

    resp = Mock()
    resp.root.result = other_result
    resp.root.id = "resp-other"
    resp.root.jsonrpc = "2.0"

    # Remove error attribute to ensure success path
    delattr(resp.root, "error")

    result = build_a2a_response_log(resp)

    assert "Type: SUCCESS" in result
    assert "Result Type: OtherResult" in result
    assert "JSON Data:" in result
    assert '"other": "data"' in result

  def test_success_response_without_model_dump_json(self):
    """Test success response with result that doesn't have model_dump_json."""
    from google.adk.a2a.logs.log_utils import build_a2a_response_log

    other_result = Mock()
    other_result.__class__.__name__ = "SimpleResult"
    # Don't add model_dump_json method
    del other_result.model_dump_json

    resp = Mock()
    resp.root.result = other_result
    resp.root.id = "resp-simple"
    resp.root.jsonrpc = "2.0"

    # Remove error attribute to ensure success path
    delattr(resp.root, "error")

    result = build_a2a_response_log(resp)

    assert "Type: SUCCESS" in result
    assert "Result Type: SimpleResult" in result

  def test_build_message_part_log_with_metadata(self):
    """Test build_message_part_log with metadata in the part."""
    from google.adk.a2a.logs.log_utils import build_message_part_log

    mock_root = Mock()
    mock_root.__class__.__name__ = "MockPartWithMetadata"
    mock_root.metadata = {"key": "value", "nested": {"data": "test"}}

    mock_part = Mock()
    mock_part.root = mock_root
    mock_part.model_dump_json.return_value = '{"content": "test"}'

    result = build_message_part_log(mock_part)

    assert "MockPartWithMetadata:" in result
    assert "Part Metadata:" in result
    assert '"key": "value"' in result
    assert '"nested"' in result

  def test_build_a2a_request_log_with_message_metadata(self):
    """Test request logging with message metadata."""
    from google.adk.a2a.logs.log_utils import build_a2a_request_log

    req = Mock()
    req.id = "req-with-metadata"
    req.method = "sendMessage"
    req.jsonrpc = "2.0"

    req.params.message.message_id = "msg-with-metadata"
    req.params.message.role = "user"
    req.params.message.task_id = "task-metadata"
    req.params.message.context_id = "ctx-metadata"
    req.params.message.parts = []
    req.params.message.metadata = {"msg_type": "test", "priority": "high"}

    req.params.configuration = None
    req.params.metadata = None

    result = build_a2a_request_log(req)

    assert "Metadata:" in result
    assert '"msg_type": "test"' in result
    assert '"priority": "high"' in result
