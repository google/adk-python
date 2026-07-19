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

"""Tests for log_utils module."""

import sys
from unittest.mock import Mock
from unittest.mock import patch

import pytest

# Skip all tests in this module if Python version is less than 3.10.
pytestmark = [
    pytest.mark.skipif(
        sys.version_info < (3, 10), reason="A2A requires Python 3.10+"
    ),
]

# Import dependencies with version checking. Tests use version-agnostic builders
# from ``_compat`` so they run on both a2a-sdk 0.3.x and 1.x.
try:
  from google.adk.a2a import _compat
  from google.adk.a2a.logs.log_utils import build_a2a_request_log
  from google.adk.a2a.logs.log_utils import build_a2a_response_log
  from google.adk.a2a.logs.log_utils import build_message_part_log
except ImportError as e:
  if sys.version_info < (3, 10):
    # Imports are not needed since tests will be skipped due to pytestmark.
    # The imported names are only used within test methods, not at module level,
    # so no NameError occurs during module compilation.
    pass
  else:
    raise e


class TestBuildMessagePartLog:
  """Test suite for build_message_part_log function."""

  def test_text_part_short_text(self):
    """Test TextPart with short text."""

    # Create real A2A objects (version-agnostic).
    part = _compat.make_text_part("Hello, world!")

    result = build_message_part_log(part)

    assert result == "TextPart: Hello, world!"

  def test_text_part_long_text(self):
    """Test TextPart with long text that gets truncated."""

    long_text = "x" * 150  # Long text that should be truncated
    part = _compat.make_text_part(long_text)

    result = build_message_part_log(part)

    expected = f"TextPart: {'x' * 100}..."
    assert result == expected

  def test_data_part_simple_data(self):
    """Test DataPart with simple data."""

    part = _compat.make_data_part(data={"key1": "value1", "key2": 42})

    result = build_message_part_log(part)

    # The integer value renders as a float on 1.x (proto Struct numbers).
    assert result.startswith("DataPart: ")
    assert '"key1": "value1"' in result
    if _compat.IS_A2A_V1:
      assert '"key2": 42.0' in result
    else:
      assert '"key2": 42' in result

  def test_data_part_large_values(self):
    """Test DataPart with large values that get summarized."""

    large_dict = {f"key{i}": f"value{i}" for i in range(50)}
    large_list = list(range(100))

    part = _compat.make_data_part(
        data={
            "small_value": "hello",
            "large_dict": large_dict,
            "large_list": large_list,
            "normal_int": 42,
        }
    )

    result = build_message_part_log(part)

    # Large values should be replaced with type names.
    assert "small_value" in result
    assert "hello" in result
    assert "<dict>" in result
    assert "<list>" in result
    assert "normal_int" in result
    # The integer value renders as a float on 1.x (proto Struct numbers).
    if _compat.IS_A2A_V1:
      assert "42.0" in result
    else:
      assert "42" in result

  @pytest.mark.skipif(
      _compat.IS_A2A_V1, reason="0.3-only .root/model_dump fallback structure"
  )
  def test_other_part_type(self):
    """Test handling of other part types (not Text or Data)."""

    # Create a mock part that will fall through to the else (file/other) case.
    mock_root = Mock()
    mock_root.__class__.__name__ = "MockOtherPart"
    # Ensure metadata attribute doesn't exist or returns None to avoid JSON
    # serialization issues.
    mock_root.metadata = None

    mock_part = Mock()
    mock_part.root = mock_root
    # The version-agnostic serializer uses model_dump (not model_dump_json) on
    # 0.3.x; a bare Mock isn't serializable so the fallback marker is emitted.
    mock_part.model_dump.return_value = {"some": "data"}

    result = build_message_part_log(mock_part)

    # On 0.3.x the else branch labels non-text/data parts as "FilePart".
    assert result.startswith("FilePart: ")


class TestBuildA2ARequestLog:
  """Test suite for build_a2a_request_log function."""

  def test_request_with_parts(self):
    """Test request logging of message parts."""

    # Create a real A2A request message (version-agnostic). Message metadata is
    # exercised separately in ``test_build_a2a_request_log_with_message_metadata``;
    # it is omitted here because on 1.x it is stored as a proto Struct that the
    # logger serializes via json.dumps.
    req = _compat.make_message(
        message_id="msg-456",
        role="user",
        task_id="task-789",
        context_id="ctx-101",
        parts=[
            _compat.make_text_part("Part 1"),
            _compat.make_text_part("Part 2"),
        ],
    )

    with patch(
        "google.adk.a2a.logs.log_utils.build_message_part_log"
    ) as mock_build_part:
      mock_build_part.side_effect = lambda part: f"Mock part: {id(part)}"

      result = build_a2a_request_log(req)

    # Verify all components are present.
    assert "msg-456" in result
    # Role prints as a string on 0.3.x and as an int enum on 1.x.
    if _compat.IS_A2A_V1:
      assert str(_compat.ROLE_USER) in result
    else:
      assert "user" in result
    assert "task-789" in result
    assert "ctx-101" in result
    assert "Part 0:" in result
    assert "Part 1:" in result

  def test_request_without_parts(self):
    """Test request logging without message parts."""

    req = Mock()

    req.message_id = "msg-456"
    req.role = "user"
    req.task_id = "task-789"
    req.context_id = "ctx-101"
    req.parts = None  # No parts
    req.metadata = None  # No message metadata

    result = build_a2a_request_log(req)

    assert "No parts" in result

  def test_request_with_empty_parts_list(self):
    """Test request logging with empty parts list."""

    req = Mock()

    req.message_id = "msg-456"
    req.role = "user"
    req.task_id = "task-789"
    req.context_id = "ctx-101"
    req.parts = []  # Empty parts list
    req.metadata = None  # No message metadata

    result = build_a2a_request_log(req)

    assert "No parts" in result


class TestBuildA2AResponseLog:
  """Test suite for build_a2a_response_log function."""

  def test_success_response_with_client_event(self):
    """Test success response logging with Task result."""

    task_status = _compat.make_task_status(_compat.TS_WORKING)
    task = _compat.make_task(
        id="task-123", context_id="ctx-456", status=task_status
    )

    resp = (task, None)

    result = build_a2a_response_log(resp)

    assert "Type: SUCCESS" in result
    assert "Result Type: ClientEvent" in result
    assert "Task ID: task-123" in result
    assert "Context ID: ctx-456" in result
    # The state renders as an enum/string on 0.3.x and as an int on 1.x.
    if _compat.IS_A2A_V1:
      assert f"Status State: {_compat.TS_WORKING}" in result
    else:
      assert (
          "Status State: TaskState.working" in result
          or "Status State: working" in result
          or '"state":"working"' in result
          or '"state": "working"' in result
      )

  def test_success_response_with_task_and_status_message(self):
    """Test success response with Task that has status message."""

    # Create status message (version-agnostic).
    status_message = _compat.make_message(
        message_id="status-msg-123",
        role=_compat.ROLE_AGENT,
        parts=[
            _compat.make_text_part("Status part 1"),
            _compat.make_text_part("Status part 2"),
        ],
    )

    task_status = _compat.make_task_status(
        _compat.TS_WORKING, message=status_message
    )
    task = _compat.make_task(
        id="task-123",
        context_id="ctx-456",
        status=task_status,
        history=[],
        artifacts=None,
    )

    resp = (task, None)

    result = build_a2a_response_log(resp)

    assert "ID: status-msg-123" in result
    # Role prints as a string on 0.3.x and as an int enum on 1.x.
    if _compat.IS_A2A_V1:
      assert f"Role: {_compat.ROLE_AGENT}" in result
    else:
      assert (
          "Role: Role.agent" in result
          or "Role: agent" in result
          or '"role":"agent"' in result
          or '"role": "agent"' in result
      )
    assert "Message Parts:" in result

  def test_success_response_with_message(self):
    """Test success response logging with Message result."""

    # Create a real A2A message (version-agnostic).
    message = _compat.make_message(
        message_id="msg-123",
        role=_compat.ROLE_AGENT,
        task_id="task-456",
        context_id="ctx-789",
        parts=[_compat.make_text_part("Message part 1")],
    )

    resp = message

    result = build_a2a_response_log(resp)

    assert "Type: SUCCESS" in result
    assert "Result Type: Message" in result
    assert "Message ID: msg-123" in result
    # Role prints as a string on 0.3.x and as an int enum on 1.x.
    if _compat.IS_A2A_V1:
      assert f"Role: {_compat.ROLE_AGENT}" in result
    else:
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

    # Use mock for this case since we want to test empty parts handling.
    message = Mock()
    message.__class__.__name__ = "Message"
    message.message_id = "msg-empty"
    message.role = "agent"
    message.task_id = "task-empty"
    message.context_id = "ctx-empty"
    message.parts = None  # No parts
    message.metadata = None  # No metadata (keep json.dumps happy)
    message.model_dump_json.return_value = '{"message": "empty"}'

    resp = message

    result = build_a2a_response_log(resp)

    assert "Type: SUCCESS" in result
    assert "Result Type: Message" in result

  def test_success_response_with_other_result_type(self):
    """Test success response with result type that's not Task or Message."""

    other_result = Mock()
    other_result.__class__.__name__ = "OtherResult"
    other_result.model_dump_json.return_value = '{"other": "data"}'

    resp = other_result

    result = build_a2a_response_log(resp)

    assert "Type: SUCCESS" in result
    assert "Result Type: OtherResult" in result
    assert "JSON Data:" in result
    assert '"other": "data"' in result

  def test_success_response_without_model_dump_json(self):
    """Test success response with result that doesn't have model_dump_json."""

    other_result = Mock()
    other_result.__class__.__name__ = "SimpleResult"
    # Don't add model_dump_json method.
    del other_result.model_dump_json

    resp = other_result

    result = build_a2a_response_log(resp)

    assert "Type: SUCCESS" in result
    assert "Result Type: SimpleResult" in result

  @pytest.mark.skipif(
      _compat.IS_A2A_V1, reason="0.3-only .root/model_dump fallback structure"
  )
  def test_build_message_part_log_with_metadata(self):
    """Test build_message_part_log with metadata in the part."""

    mock_root = Mock()
    mock_root.__class__.__name__ = "MockPartWithMetadata"
    mock_root.metadata = {"key": "value", "nested": {"data": "test"}}

    mock_part = Mock()
    mock_part.root = mock_root
    mock_part.model_dump.return_value = {"content": "test"}

    result = build_message_part_log(mock_part)

    # On 0.3.x the part metadata is read from ``part.root.metadata``.
    assert "Part Metadata:" in result
    assert '"key": "value"' in result
    assert '"nested"' in result

  def test_build_a2a_request_log_with_message_metadata(self):
    """Test request logging with message metadata."""

    req = Mock()

    req.message_id = "msg-with-metadata"
    req.role = "user"
    req.task_id = "task-metadata"
    req.context_id = "ctx-metadata"
    req.parts = []
    req.metadata = {"msg_type": "test", "priority": "high"}

    result = build_a2a_request_log(req)

    assert "Metadata:" in result
    assert '"msg_type": "test"' in result
    assert '"priority": "high"' in result
