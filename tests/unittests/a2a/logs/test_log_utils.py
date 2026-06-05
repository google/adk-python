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

import pytest

# Skip all tests in this module if Python version is less than 3.10
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10), reason="A2A requires Python 3.10+"
)

try:
  from a2a.types import Message as A2AMessage
  from a2a.types import Part as A2APart
  from a2a.types import Role
  from a2a.types import StreamResponse
  from a2a.types import Task as A2ATask
  from a2a.types import TaskState
  from a2a.types import TaskStatus
  from google.adk.a2a.logs.log_utils import build_a2a_request_log
  from google.adk.a2a.logs.log_utils import build_a2a_response_log
  from google.adk.a2a.logs.log_utils import build_message_part_log
  from google.protobuf import json_format
except ImportError as e:
  if sys.version_info < (3, 10):
    pass
  else:
    raise e


class TestBuildMessagePartLog:
  """Test suite for build_message_part_log function."""

  def test_text_part_short_text(self):
    """Text Part with short text produces 'TextPart: <text>'."""
    part = A2APart(text="Hello, world!")

    result = build_message_part_log(part)

    assert result.startswith("TextPart: Hello, world!")

  def test_text_part_long_text(self):
    """Text Part with long text gets truncated at 100 chars."""
    long_text = "x" * 150
    part = A2APart(text=long_text)

    result = build_message_part_log(part)

    assert result.startswith("TextPart: " + "x" * 100 + "...")

  def test_data_part_simple_data(self):
    """Data Part with simple data shows its keys and values."""
    part = A2APart()
    json_format.ParseDict({"data": {"key1": "value1", "key2": 42}}, part)

    result = build_message_part_log(part)

    assert "DataPart:" in result
    assert "key1" in result

  def test_url_part(self):
    """URL Part shows the URL."""
    part = A2APart(url="gs://bucket/file.txt", media_type="text/plain")

    result = build_message_part_log(part)

    assert "UrlPart:" in result
    assert "gs://bucket/file.txt" in result

  def test_empty_part_returns_string(self):
    """Empty Part (no content set) returns a string without crashing."""
    part = A2APart()

    result = build_message_part_log(part)

    assert isinstance(result, str)


class TestBuildA2ARequestLog:
  """Test suite for build_a2a_request_log function."""

  def test_request_with_parts(self):
    """Request with parts logs all part indices."""
    msg = A2AMessage(
        message_id="msg-456",
        role=Role.ROLE_USER,
        parts=[A2APart(text="Part 1"), A2APart(text="Part 2")],
    )

    result = build_a2a_request_log(msg)

    assert "msg-456" in result
    assert "Part 0:" in result
    assert "Part 1:" in result

  def test_request_without_parts(self):
    """Request with no parts shows 'No parts'."""
    msg = A2AMessage(message_id="msg-456", role=Role.ROLE_USER)

    result = build_a2a_request_log(msg)

    assert "No parts" in result

  def test_request_with_metadata(self):
    """Request with metadata includes metadata in the log."""
    msg = A2AMessage(message_id="msg-1", role=Role.ROLE_USER)
    msg.metadata["msg_type"] = "test"
    msg.metadata["priority"] = "high"

    result = build_a2a_request_log(msg)

    assert "Metadata:" in result
    assert "msg_type" in result


class TestBuildA2AResponseLog:
  """Test suite for build_a2a_response_log function."""

  def test_response_with_stream_response_task(self):
    """StreamResponse with task payload logs task details."""
    status = TaskStatus(state=TaskState.TASK_STATE_WORKING)
    status.timestamp.GetCurrentTime()
    task = A2ATask(id="task-123", context_id="ctx-456")
    task.status.CopyFrom(status)

    stream_resp = StreamResponse(task=task)

    result = build_a2a_response_log(stream_resp)

    assert "Type: SUCCESS" in result
    assert "task-123" in result

  def test_response_with_message(self):
    """A2AMessage response logs message details."""
    message = A2AMessage(
        message_id="msg-123",
        role=Role.ROLE_AGENT,
        parts=[A2APart(text="Hello")],
    )

    result = build_a2a_response_log(message)

    assert "Type: SUCCESS" in result
    assert "msg-123" in result

  def test_response_with_tuple_legacy(self):
    """Legacy tuple (task, update) response is handled."""
    status = TaskStatus(state=TaskState.TASK_STATE_WORKING)
    status.timestamp.GetCurrentTime()
    task = A2ATask(id="task-123", context_id="ctx-456")
    task.status.CopyFrom(status)

    resp = (task, None)

    result = build_a2a_response_log(resp)

    assert "Type: SUCCESS" in result
    assert "ClientEvent" in result
    assert "task-123" in result
