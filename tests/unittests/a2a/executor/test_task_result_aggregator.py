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

import pytest

# Skip all tests in this module if Python version is less than 3.10
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10), reason="A2A requires Python 3.10+"
)

# Import dependencies with version checking
try:
  from a2a.types import Message
  from a2a.types import Part
  from a2a.types import Role
  from a2a.types import TaskState
  from a2a.types import TaskStatus
  from a2a.types import TaskStatusUpdateEvent
  from a2a.types import TextPart
  from google.adk.a2a.executor.task_result_aggregator import TaskResultAggregator
except ImportError as e:
  if sys.version_info < (3, 10):
    # Create dummy classes to prevent NameError during test collection
    # Tests will be skipped anyway due to pytestmark
    class DummyTypes:
      pass

    TaskState = DummyTypes()
    TaskStatus = DummyTypes()
    TaskStatusUpdateEvent = DummyTypes()
    TaskResultAggregator = DummyTypes()
  else:
    raise e


def create_test_message(text: str) -> Message:
  """Helper function to create a test Message object."""
  return Message(
      message_id="test-msg",
      role=Role.agent,
      parts=[Part(root=TextPart(text=text))],
  )


class TestTaskResultAggregator:
  """Test suite for TaskResultAggregator class."""

  def setup_method(self):
    """Set up test fixtures."""
    self.aggregator = TaskResultAggregator()

  def test_initial_state(self):
    """Test the initial state of the aggregator."""
    assert self.aggregator.task_state == TaskState.working
    assert self.aggregator.task_status_message is None

  def test_process_failed_event(self):
    """Test processing a failed event."""
    status_message = create_test_message("Failed to process")
    event = TaskStatusUpdateEvent(
        task_id="test-task",
        context_id="test-context",
        status=TaskStatus(state=TaskState.failed, message=status_message),
        final=True,
    )

    self.aggregator.process_event(event)
    assert self.aggregator.task_state == TaskState.failed
    assert self.aggregator.task_status_message == status_message
    # Verify the event state was modified to working
    assert event.status.state == TaskState.working

  def test_process_auth_required_event(self):
    """Test processing an auth_required event."""
    status_message = create_test_message("Authentication needed")
    event = TaskStatusUpdateEvent(
        task_id="test-task",
        context_id="test-context",
        status=TaskStatus(
            state=TaskState.auth_required, message=status_message
        ),
        final=False,
    )

    self.aggregator.process_event(event)
    assert self.aggregator.task_state == TaskState.auth_required
    assert self.aggregator.task_status_message == status_message
    # Verify the event state was modified to working
    assert event.status.state == TaskState.working

  def test_process_input_required_event(self):
    """Test processing an input_required event."""
    status_message = create_test_message("Input required")
    event = TaskStatusUpdateEvent(
        task_id="test-task",
        context_id="test-context",
        status=TaskStatus(
            state=TaskState.input_required, message=status_message
        ),
        final=False,
    )

    self.aggregator.process_event(event)
    assert self.aggregator.task_state == TaskState.input_required
    assert self.aggregator.task_status_message == status_message
    # Verify the event state was modified to working
    assert event.status.state == TaskState.working

  def test_status_message_with_none_message(self):
    """Test that status message handles None message properly."""
    event = TaskStatusUpdateEvent(
        task_id="test-task",
        context_id="test-context",
        status=TaskStatus(state=TaskState.failed, message=None),
        final=True,
    )

    self.aggregator.process_event(event)
    assert self.aggregator.task_state == TaskState.failed
    assert self.aggregator.task_status_message is None

  def test_priority_order_failed_over_auth(self):
    """Test that failed state takes priority over auth_required."""
    # First set auth_required
    auth_message = create_test_message("Auth required")
    auth_event = TaskStatusUpdateEvent(
        task_id="test-task",
        context_id="test-context",
        status=TaskStatus(state=TaskState.auth_required, message=auth_message),
        final=False,
    )
    self.aggregator.process_event(auth_event)
    assert self.aggregator.task_state == TaskState.auth_required
    assert self.aggregator.task_status_message == auth_message

    # Then process failed - should override
    failed_message = create_test_message("Failed")
    failed_event = TaskStatusUpdateEvent(
        task_id="test-task",
        context_id="test-context",
        status=TaskStatus(state=TaskState.failed, message=failed_message),
        final=True,
    )
    self.aggregator.process_event(failed_event)
    assert self.aggregator.task_state == TaskState.failed
    assert self.aggregator.task_status_message == failed_message

  def test_priority_order_auth_over_input(self):
    """Test that auth_required state takes priority over input_required."""
    # First set input_required
    input_message = create_test_message("Input needed")
    input_event = TaskStatusUpdateEvent(
        task_id="test-task",
        context_id="test-context",
        status=TaskStatus(
            state=TaskState.input_required, message=input_message
        ),
        final=False,
    )
    self.aggregator.process_event(input_event)
    assert self.aggregator.task_state == TaskState.input_required
    assert self.aggregator.task_status_message == input_message

    # Then process auth_required - should override
    auth_message = create_test_message("Auth needed")
    auth_event = TaskStatusUpdateEvent(
        task_id="test-task",
        context_id="test-context",
        status=TaskStatus(state=TaskState.auth_required, message=auth_message),
        final=False,
    )
    self.aggregator.process_event(auth_event)
    assert self.aggregator.task_state == TaskState.auth_required
    assert self.aggregator.task_status_message == auth_message

  def test_ignore_non_status_update_events(self):
    """Test that non-TaskStatusUpdateEvent events are ignored."""
    mock_event = Mock()

    initial_state = self.aggregator.task_state
    initial_message = self.aggregator.task_status_message
    self.aggregator.process_event(mock_event)

    # State should remain unchanged
    assert self.aggregator.task_state == initial_state
    assert self.aggregator.task_status_message == initial_message

  def test_working_state_does_not_override_higher_priority(self):
    """Test that working state doesn't override higher priority states."""
    # First set failed state
    failed_message = create_test_message("Failure message")
    failed_event = TaskStatusUpdateEvent(
        task_id="test-task",
        context_id="test-context",
        status=TaskStatus(state=TaskState.failed, message=failed_message),
        final=True,
    )
    self.aggregator.process_event(failed_event)
    assert self.aggregator.task_state == TaskState.failed
    assert self.aggregator.task_status_message == failed_message

    # Then process working - should not override state and should not update message
    # because the current task state is not working
    working_event = TaskStatusUpdateEvent(
        task_id="test-task",
        context_id="test-context",
        status=TaskStatus(state=TaskState.working),
        final=False,
    )
    self.aggregator.process_event(working_event)
    assert self.aggregator.task_state == TaskState.failed
    # Working events don't update the status message when task state is not working
    assert self.aggregator.task_status_message == failed_message

  def test_status_message_priority_ordering(self):
    """Test that status messages follow the same priority ordering as states."""
    # Start with input_required
    input_message = create_test_message("Input message")
    input_event = TaskStatusUpdateEvent(
        task_id="test-task",
        context_id="test-context",
        status=TaskStatus(
            state=TaskState.input_required, message=input_message
        ),
        final=False,
    )
    self.aggregator.process_event(input_event)
    assert self.aggregator.task_status_message == input_message

    # Override with auth_required
    auth_message = create_test_message("Auth message")
    auth_event = TaskStatusUpdateEvent(
        task_id="test-task",
        context_id="test-context",
        status=TaskStatus(state=TaskState.auth_required, message=auth_message),
        final=False,
    )
    self.aggregator.process_event(auth_event)
    assert self.aggregator.task_status_message == auth_message

    # Override with failed
    failed_message = create_test_message("Failed message")
    failed_event = TaskStatusUpdateEvent(
        task_id="test-task",
        context_id="test-context",
        status=TaskStatus(state=TaskState.failed, message=failed_message),
        final=True,
    )
    self.aggregator.process_event(failed_event)
    assert self.aggregator.task_status_message == failed_message

    # Working should not override failed message because current task state is failed
    working_message = create_test_message("Working message")
    working_event = TaskStatusUpdateEvent(
        task_id="test-task",
        context_id="test-context",
        status=TaskStatus(state=TaskState.working, message=working_message),
        final=False,
    )
    self.aggregator.process_event(working_event)
    # State should still be failed, and message should remain the failed message
    # because working events only update message when task state is working
    assert self.aggregator.task_state == TaskState.failed
    assert self.aggregator.task_status_message == failed_message

  def test_process_working_event_updates_message(self):
    """Test that working state events update the status message."""
    working_message = create_test_message("Working on task")
    event = TaskStatusUpdateEvent(
        task_id="test-task",
        context_id="test-context",
        status=TaskStatus(state=TaskState.working, message=working_message),
        final=False,
    )

    self.aggregator.process_event(event)
    assert self.aggregator.task_state == TaskState.working
    assert self.aggregator.task_status_message == working_message
    # Verify the event state was modified to working (should remain working)
    assert event.status.state == TaskState.working

  def test_working_event_with_none_message(self):
    """Test that working state events handle None message properly."""
    event = TaskStatusUpdateEvent(
        task_id="test-task",
        context_id="test-context",
        status=TaskStatus(state=TaskState.working, message=None),
        final=False,
    )

    self.aggregator.process_event(event)
    assert self.aggregator.task_state == TaskState.working
    assert self.aggregator.task_status_message is None

  def test_working_event_updates_message_regardless_of_state(self):
    """Test that working events update message only when current task state is working."""
    # First set auth_required state
    auth_message = create_test_message("Auth required")
    auth_event = TaskStatusUpdateEvent(
        task_id="test-task",
        context_id="test-context",
        status=TaskStatus(state=TaskState.auth_required, message=auth_message),
        final=False,
    )
    self.aggregator.process_event(auth_event)
    assert self.aggregator.task_state == TaskState.auth_required
    assert self.aggregator.task_status_message == auth_message

    # Then process working - should not update message because task state is not working
    working_message = create_test_message("Working on auth")
    working_event = TaskStatusUpdateEvent(
        task_id="test-task",
        context_id="test-context",
        status=TaskStatus(state=TaskState.working, message=working_message),
        final=False,
    )
    self.aggregator.process_event(working_event)
    assert (
        self.aggregator.task_state == TaskState.auth_required
    )  # State unchanged
    assert (
        self.aggregator.task_status_message == auth_message
    )  # Message unchanged because task state is not working
