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
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

# Skip all tests in this module if Python version is less than 3.10
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10), reason="A2A tool requires Python 3.10+"
)

# Import dependencies with version checking
try:
  from a2a.server.agent_execution.context import RequestContext
  from a2a.server.events.event_queue import EventQueue
  from a2a.types import Message
  from a2a.types import TaskState
  from a2a.types import TextPart
  from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
  from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutorConfig
  from google.adk.events.event import Event
  from google.adk.runners import Runner
except ImportError as e:
  if sys.version_info < (3, 10):
    # Create dummy classes to prevent NameError during test collection
    # Tests will be skipped anyway due to pytestmark
    class DummyTypes:
      pass

    RequestContext = DummyTypes()
    EventQueue = DummyTypes()
    Message = DummyTypes()
    Role = DummyTypes()
    TaskState = DummyTypes()
    TaskStatus = DummyTypes()
    TaskStatusUpdateEvent = DummyTypes()
    TextPart = DummyTypes()
    A2aAgentExecutor = DummyTypes()
    A2aAgentExecutorConfig = DummyTypes()
    Event = DummyTypes()
    Runner = DummyTypes()
  else:
    raise e


class TestA2aAgentExecutor:
  """Test suite for A2aAgentExecutor class."""

  def setup_method(self):
    """Set up test fixtures."""
    self.mock_runner = Mock(spec=Runner)
    self.mock_runner.app_name = "test-app"
    self.mock_runner.session_service = Mock()
    self.mock_runner._new_invocation_context = Mock()
    self.mock_runner.run_async = AsyncMock()

    self.mock_config = Mock(spec=A2aAgentExecutorConfig)
    self.executor = A2aAgentExecutor(
        runner=self.mock_runner, config=self.mock_config
    )

    self.mock_context = Mock(spec=RequestContext)
    self.mock_context.message = Mock(spec=Message)
    self.mock_context.message.parts = [Mock(spec=TextPart)]
    self.mock_context.current_task = None
    self.mock_context.task_id = "test-task-id"
    self.mock_context.context_id = "test-context-id"

    self.mock_event_queue = Mock(spec=EventQueue)

  async def _create_async_generator(self, items):
    """Helper to create async generator from items."""
    for item in items:
      yield item

  @pytest.mark.asyncio
  async def test_execute_success_new_task(self):
    """Test successful execution of a new task."""
    # Setup
    with patch(
        "google.adk.a2a.executor.a2a_agent_executor.convert_a2a_request_to_adk_run_args"
    ) as mock_convert:
      mock_convert.return_value = {
          "user_id": "test-user",
          "session_id": "test-session",
          "new_message": Mock(),
          "run_config": Mock(),
      }

      # Mock session service
      mock_session = Mock()
      mock_session.id = "test-session"
      self.mock_runner.session_service.get_session = AsyncMock(
          return_value=mock_session
      )

      # Mock invocation context
      mock_invocation_context = Mock()
      self.mock_runner._new_invocation_context.return_value = (
          mock_invocation_context
      )

      # Mock agent run with proper async generator
      mock_event = Mock(spec=Event)

      # Configure run_async to return the async generator when awaited
      async def mock_run_async(**kwargs):
        async for item in self._create_async_generator([mock_event]):
          yield item

      self.mock_runner.run_async = mock_run_async

      with patch(
          "google.adk.a2a.executor.a2a_agent_executor.convert_event_to_a2a_events"
      ) as mock_convert_events:
        mock_convert_events.return_value = []

        # Execute
        await self.executor.execute(self.mock_context, self.mock_event_queue)

        # Verify task submitted event was enqueued
        assert self.mock_event_queue.enqueue_event.call_count >= 3
        submitted_event = self.mock_event_queue.enqueue_event.call_args_list[0][
            0
        ][0]
        assert submitted_event.status.state == TaskState.submitted
        assert submitted_event.final == False

        # Verify working event was enqueued
        working_event = self.mock_event_queue.enqueue_event.call_args_list[1][
            0
        ][0]
        assert working_event.status.state == TaskState.working
        assert working_event.final == False

        # Verify final event was enqueued with proper message field
        final_event = self.mock_event_queue.enqueue_event.call_args_list[-1][0][
            0
        ]
        assert final_event.final == True
        # The TaskResultAggregator is created with default state (working), and since no messages
        # are processed, it will publish a status event with the current state
        assert hasattr(final_event.status, "message")
        assert final_event.status.state == TaskState.working

  @pytest.mark.asyncio
  async def test_execute_no_message_error(self):
    """Test execution fails when no message is provided."""
    self.mock_context.message = None

    with pytest.raises(ValueError, match="A2A request must have a message"):
      await self.executor.execute(self.mock_context, self.mock_event_queue)

  @pytest.mark.asyncio
  async def test_execute_existing_task(self):
    """Test execution with existing task (no submitted event)."""
    self.mock_context.current_task = Mock()
    self.mock_context.task_id = "existing-task-id"

    with patch(
        "google.adk.a2a.executor.a2a_agent_executor.convert_a2a_request_to_adk_run_args"
    ) as mock_convert:
      mock_convert.return_value = {
          "user_id": "test-user",
          "session_id": "test-session",
          "new_message": Mock(),
          "run_config": Mock(),
      }

      # Mock session service
      mock_session = Mock()
      mock_session.id = "test-session"
      self.mock_runner.session_service.get_session = AsyncMock(
          return_value=mock_session
      )

      # Mock invocation context
      mock_invocation_context = Mock()
      self.mock_runner._new_invocation_context.return_value = (
          mock_invocation_context
      )

      # Mock agent run with proper async generator
      mock_event = Mock(spec=Event)

      # Configure run_async to return the async generator when awaited
      async def mock_run_async(**kwargs):
        async for item in self._create_async_generator([mock_event]):
          yield item

      self.mock_runner.run_async = mock_run_async

      with patch(
          "google.adk.a2a.executor.a2a_agent_executor.convert_event_to_a2a_events"
      ) as mock_convert_events:
        mock_convert_events.return_value = []

        # Execute
        await self.executor.execute(self.mock_context, self.mock_event_queue)

        # Verify no submitted event (first call should be working event)
        working_event = self.mock_event_queue.enqueue_event.call_args_list[0][
            0
        ][0]
        assert working_event.status.state == TaskState.working
        assert working_event.final == False

        # Verify final event was enqueued with proper message field
        final_event = self.mock_event_queue.enqueue_event.call_args_list[-1][0][
            0
        ]
        assert final_event.final == True
        # The TaskResultAggregator is created with default state (working), and since no messages
        # are processed, it will publish a status event with the current state
        assert hasattr(final_event.status, "message")
        assert final_event.status.state == TaskState.working

  @pytest.mark.asyncio
  async def test_prepare_session_new_session(self):
    """Test session preparation when session doesn't exist."""
    run_args = {
        "user_id": "test-user",
        "session_id": None,
        "new_message": Mock(),
        "run_config": Mock(),
    }

    # Mock session service
    self.mock_runner.session_service.get_session = AsyncMock(return_value=None)
    mock_session = Mock()
    mock_session.id = "new-session-id"
    self.mock_runner.session_service.create_session = AsyncMock(
        return_value=mock_session
    )

    # Execute
    result = await self.executor._prepare_session(
        self.mock_context, run_args, self.mock_runner
    )

    # Verify session was created
    assert result == mock_session
    assert run_args["session_id"] is not None
    self.mock_runner.session_service.create_session.assert_called_once()

  @pytest.mark.asyncio
  async def test_prepare_session_existing_session(self):
    """Test session preparation when session exists."""
    run_args = {
        "user_id": "test-user",
        "session_id": "existing-session",
        "new_message": Mock(),
        "run_config": Mock(),
    }

    # Mock session service
    mock_session = Mock()
    mock_session.id = "existing-session"
    self.mock_runner.session_service.get_session = AsyncMock(
        return_value=mock_session
    )

    # Execute
    result = await self.executor._prepare_session(
        self.mock_context, run_args, self.mock_runner
    )

    # Verify existing session was returned
    assert result == mock_session
    self.mock_runner.session_service.create_session.assert_not_called()

  def test_constructor_with_callable_runner(self):
    """Test constructor with callable runner."""
    callable_runner = Mock()
    executor = A2aAgentExecutor(runner=callable_runner, config=self.mock_config)

    assert executor._runner == callable_runner
    assert executor._config == self.mock_config

  @pytest.mark.asyncio
  async def test_resolve_runner_direct_instance(self):
    """Test _resolve_runner with direct Runner instance."""
    # Setup - already using direct runner instance in setup_method
    runner = await self.executor._resolve_runner()
    assert runner == self.mock_runner

  @pytest.mark.asyncio
  async def test_resolve_runner_sync_callable(self):
    """Test _resolve_runner with sync callable that returns Runner."""

    def create_runner():
      return self.mock_runner

    executor = A2aAgentExecutor(runner=create_runner, config=self.mock_config)
    runner = await executor._resolve_runner()
    assert runner == self.mock_runner

  @pytest.mark.asyncio
  async def test_resolve_runner_async_callable(self):
    """Test _resolve_runner with async callable that returns Runner."""

    async def create_runner():
      return self.mock_runner

    executor = A2aAgentExecutor(runner=create_runner, config=self.mock_config)
    runner = await executor._resolve_runner()
    assert runner == self.mock_runner

  @pytest.mark.asyncio
  async def test_resolve_runner_invalid_type(self):
    """Test _resolve_runner with invalid runner type."""
    executor = A2aAgentExecutor(runner="invalid", config=self.mock_config)

    with pytest.raises(
        TypeError, match="Runner must be a Runner instance or a callable"
    ):
      await executor._resolve_runner()

  @pytest.mark.asyncio
  async def test_resolve_runner_callable_with_parameters(self):
    """Test _resolve_runner with callable that normally takes parameters."""

    def create_runner(*args, **kwargs):
      # In real usage, this might use the args/kwargs to configure the runner
      # For testing, we'll just return the mock runner
      return self.mock_runner

    executor = A2aAgentExecutor(runner=create_runner, config=self.mock_config)
    runner = await executor._resolve_runner()
    assert runner == self.mock_runner

  @pytest.mark.asyncio
  async def test_resolve_runner_caching(self):
    """Test that _resolve_runner caches the result and doesn't call the callable multiple times."""
    call_count = 0

    def create_runner():
      nonlocal call_count
      call_count += 1
      return self.mock_runner

    executor = A2aAgentExecutor(runner=create_runner, config=self.mock_config)

    # First call should invoke the callable
    runner1 = await executor._resolve_runner()
    assert runner1 == self.mock_runner
    assert call_count == 1

    # Second call should return cached result, not invoke callable again
    runner2 = await executor._resolve_runner()
    assert runner2 == self.mock_runner
    assert runner1 is runner2  # Same instance
    assert call_count == 1  # Callable was not called again

    # Verify that self._runner is now the resolved Runner instance
    assert executor._runner is self.mock_runner

  @pytest.mark.asyncio
  async def test_resolve_runner_async_caching(self):
    """Test that _resolve_runner caches async callable results correctly."""
    call_count = 0

    async def create_runner():
      nonlocal call_count
      call_count += 1
      return self.mock_runner

    executor = A2aAgentExecutor(runner=create_runner, config=self.mock_config)

    # First call should invoke the async callable
    runner1 = await executor._resolve_runner()
    assert runner1 == self.mock_runner
    assert call_count == 1

    # Second call should return cached result, not invoke callable again
    runner2 = await executor._resolve_runner()
    assert runner2 == self.mock_runner
    assert runner1 is runner2  # Same instance
    assert call_count == 1  # Async callable was not called again

    # Verify that self._runner is now the resolved Runner instance
    assert executor._runner is self.mock_runner

  @pytest.mark.asyncio
  async def test_execute_with_sync_callable_runner(self):
    """Test execution with sync callable runner."""

    def create_runner():
      return self.mock_runner

    executor = A2aAgentExecutor(runner=create_runner, config=self.mock_config)

    with patch(
        "google.adk.a2a.executor.a2a_agent_executor.convert_a2a_request_to_adk_run_args"
    ) as mock_convert:
      mock_convert.return_value = {
          "user_id": "test-user",
          "session_id": "test-session",
          "new_message": Mock(),
          "run_config": Mock(),
      }

      # Mock session service
      mock_session = Mock()
      mock_session.id = "test-session"
      self.mock_runner.session_service.get_session = AsyncMock(
          return_value=mock_session
      )

      # Mock invocation context
      mock_invocation_context = Mock()
      self.mock_runner._new_invocation_context.return_value = (
          mock_invocation_context
      )

      # Mock agent run with proper async generator
      mock_event = Mock(spec=Event)

      async def mock_run_async(**kwargs):
        async for item in self._create_async_generator([mock_event]):
          yield item

      self.mock_runner.run_async = mock_run_async

      with patch(
          "google.adk.a2a.executor.a2a_agent_executor.convert_event_to_a2a_events"
      ) as mock_convert_events:
        mock_convert_events.return_value = []

        # Execute
        await executor.execute(self.mock_context, self.mock_event_queue)

        # Verify task submitted event was enqueued
        assert self.mock_event_queue.enqueue_event.call_count >= 3
        submitted_event = self.mock_event_queue.enqueue_event.call_args_list[0][
            0
        ][0]
        assert submitted_event.status.state == TaskState.submitted
        assert submitted_event.final == False

        # Verify final event was enqueued with proper message field
        final_event = self.mock_event_queue.enqueue_event.call_args_list[-1][0][
            0
        ]
        assert final_event.final == True
        # The TaskResultAggregator is created with default state (working), and since no messages
        # are processed, it will publish a status event with the current state
        assert hasattr(final_event.status, "message")
        assert final_event.status.state == TaskState.working

  @pytest.mark.asyncio
  async def test_execute_with_async_callable_runner(self):
    """Test execution with async callable runner."""

    async def create_runner():
      return self.mock_runner

    executor = A2aAgentExecutor(runner=create_runner, config=self.mock_config)

    with patch(
        "google.adk.a2a.executor.a2a_agent_executor.convert_a2a_request_to_adk_run_args"
    ) as mock_convert:
      mock_convert.return_value = {
          "user_id": "test-user",
          "session_id": "test-session",
          "new_message": Mock(),
          "run_config": Mock(),
      }

      # Mock session service
      mock_session = Mock()
      mock_session.id = "test-session"
      self.mock_runner.session_service.get_session = AsyncMock(
          return_value=mock_session
      )

      # Mock invocation context
      mock_invocation_context = Mock()
      self.mock_runner._new_invocation_context.return_value = (
          mock_invocation_context
      )

      # Mock agent run with proper async generator
      mock_event = Mock(spec=Event)

      async def mock_run_async(**kwargs):
        async for item in self._create_async_generator([mock_event]):
          yield item

      self.mock_runner.run_async = mock_run_async

      with patch(
          "google.adk.a2a.executor.a2a_agent_executor.convert_event_to_a2a_events"
      ) as mock_convert_events:
        mock_convert_events.return_value = []

        # Execute
        await executor.execute(self.mock_context, self.mock_event_queue)

        # Verify task submitted event was enqueued
        assert self.mock_event_queue.enqueue_event.call_count >= 3
        submitted_event = self.mock_event_queue.enqueue_event.call_args_list[0][
            0
        ][0]
        assert submitted_event.status.state == TaskState.submitted
        assert submitted_event.final == False

        # Verify final event was enqueued with proper message field
        final_event = self.mock_event_queue.enqueue_event.call_args_list[-1][0][
            0
        ]
        assert final_event.final == True
        # The TaskResultAggregator is created with default state (working), and since no messages
        # are processed, it will publish a status event with the current state
        assert hasattr(final_event.status, "message")
        assert final_event.status.state == TaskState.working

  @pytest.mark.asyncio
  async def test_handle_request_integration(self):
    """Test the complete request handling flow."""
    # Setup context with task_id
    self.mock_context.task_id = "test-task-id"

    # Setup detailed mocks
    with patch(
        "google.adk.a2a.executor.a2a_agent_executor.convert_a2a_request_to_adk_run_args"
    ) as mock_convert:
      mock_convert.return_value = {
          "user_id": "test-user",
          "session_id": "test-session",
          "new_message": Mock(),
          "run_config": Mock(),
      }

      # Mock session service
      mock_session = Mock()
      mock_session.id = "test-session"
      self.mock_runner.session_service.get_session = AsyncMock(
          return_value=mock_session
      )

      # Mock invocation context
      mock_invocation_context = Mock()
      self.mock_runner._new_invocation_context.return_value = (
          mock_invocation_context
      )

      # Mock agent run with multiple events using proper async generator
      mock_events = [Mock(spec=Event), Mock(spec=Event)]

      # Configure run_async to return the async generator when awaited
      async def mock_run_async(**kwargs):
        async for item in self._create_async_generator(mock_events):
          yield item

      self.mock_runner.run_async = mock_run_async

      with patch(
          "google.adk.a2a.executor.a2a_agent_executor.convert_event_to_a2a_events"
      ) as mock_convert_events:
        mock_convert_events.return_value = [Mock()]

        with patch(
            "google.adk.a2a.executor.a2a_agent_executor.TaskResultAggregator"
        ) as mock_aggregator_class:
          mock_aggregator = Mock()
          mock_aggregator.task_state = TaskState.working
          # Mock the task_status_message property to return None by default
          mock_aggregator.task_status_message = None
          mock_aggregator_class.return_value = mock_aggregator

          # Execute
          await self.executor._handle_request(
              self.mock_context, self.mock_event_queue
          )

          # Verify working event was enqueued
          working_events = [
              call[0][0]
              for call in self.mock_event_queue.enqueue_event.call_args_list
              if hasattr(call[0][0], "status")
              and call[0][0].status.state == TaskState.working
          ]
          assert len(working_events) >= 1

          # Verify aggregator processed events
          assert mock_aggregator.process_event.call_count == len(mock_events)

          # Verify final event has message field from aggregator and state is completed when aggregator state is working
          final_events = [
              call[0][0]
              for call in self.mock_event_queue.enqueue_event.call_args_list
              if hasattr(call[0][0], "final") and call[0][0].final == True
          ]
          assert len(final_events) >= 1
          final_event = final_events[-1]  # Get the last final event
          assert (
              final_event.status.message == mock_aggregator.task_status_message
          )
          # When aggregator state is working but no message, final event should be working
          assert final_event.status.state == TaskState.working

  @pytest.mark.asyncio
  async def test_cancel_with_task_id(self):
    """Test cancellation with a task ID."""
    self.mock_context.task_id = "test-task-id"

    # The current implementation raises NotImplementedError
    with pytest.raises(
        NotImplementedError, match="Cancellation is not supported"
    ):
      await self.executor.cancel(self.mock_context, self.mock_event_queue)

  @pytest.mark.asyncio
  async def test_cancel_without_task_id(self):
    """Test cancellation without a task ID."""
    self.mock_context.task_id = None

    # The current implementation raises NotImplementedError regardless of task_id
    with pytest.raises(
        NotImplementedError, match="Cancellation is not supported"
    ):
      await self.executor.cancel(self.mock_context, self.mock_event_queue)

  @pytest.mark.asyncio
  async def test_execute_with_exception_handling(self):
    """Test execution with exception handling."""
    self.mock_context.task_id = "test-task-id"
    self.mock_context.current_task = (
        None  # Make sure it goes through submitted event creation
    )

    with patch(
        "google.adk.a2a.executor.a2a_agent_executor.convert_a2a_request_to_adk_run_args"
    ) as mock_convert:
      mock_convert.side_effect = Exception("Test error")

      # Execute (should not raise since we catch the exception)
      await self.executor.execute(self.mock_context, self.mock_event_queue)

      # Verify both submitted and failure events were enqueued
      # First call should be submitted event, last should be failure event
      assert self.mock_event_queue.enqueue_event.call_count >= 2

      # Check submitted event (first)
      submitted_event = self.mock_event_queue.enqueue_event.call_args_list[0][
          0
      ][0]
      assert submitted_event.status.state == TaskState.submitted
      assert submitted_event.final == False

      # Check failure event (last)
      failure_event = self.mock_event_queue.enqueue_event.call_args_list[-1][0][
          0
      ]
      assert failure_event.status.state == TaskState.failed
      assert failure_event.final == True

  @pytest.mark.asyncio
  async def test_handle_request_with_aggregator_message(self):
    """Test that the final task status event includes message from aggregator."""
    # Setup context with task_id
    self.mock_context.task_id = "test-task-id"

    # Create a test message to be returned by the aggregator
    from a2a.types import Message
    from a2a.types import Role
    from a2a.types import TextPart

    test_message = Mock(spec=Message)
    test_message.message_id = "test-message-id"
    test_message.role = Role.agent
    test_message.parts = [Mock(spec=TextPart)]

    # Setup detailed mocks
    with patch(
        "google.adk.a2a.executor.a2a_agent_executor.convert_a2a_request_to_adk_run_args"
    ) as mock_convert:
      mock_convert.return_value = {
          "user_id": "test-user",
          "session_id": "test-session",
          "new_message": Mock(),
          "run_config": Mock(),
      }

      # Mock session service
      mock_session = Mock()
      mock_session.id = "test-session"
      self.mock_runner.session_service.get_session = AsyncMock(
          return_value=mock_session
      )

      # Mock invocation context
      mock_invocation_context = Mock()
      self.mock_runner._new_invocation_context.return_value = (
          mock_invocation_context
      )

      # Mock agent run with multiple events using proper async generator
      mock_events = [Mock(spec=Event), Mock(spec=Event)]

      # Configure run_async to return the async generator when awaited
      async def mock_run_async(**kwargs):
        async for item in self._create_async_generator(mock_events):
          yield item

      self.mock_runner.run_async = mock_run_async

      with patch(
          "google.adk.a2a.executor.a2a_agent_executor.convert_event_to_a2a_events"
      ) as mock_convert_events:
        mock_convert_events.return_value = [Mock()]

        with patch(
            "google.adk.a2a.executor.a2a_agent_executor.TaskResultAggregator"
        ) as mock_aggregator_class:
          mock_aggregator = Mock()
          mock_aggregator.task_state = TaskState.completed
          # Mock the task_status_message property to return a test message
          mock_aggregator.task_status_message = test_message
          mock_aggregator_class.return_value = mock_aggregator

          # Execute
          await self.executor._handle_request(
              self.mock_context, self.mock_event_queue
          )

          # Verify final event has message field from aggregator
          final_events = [
              call[0][0]
              for call in self.mock_event_queue.enqueue_event.call_args_list
              if hasattr(call[0][0], "final") and call[0][0].final == True
          ]
          assert len(final_events) >= 1
          final_event = final_events[-1]  # Get the last final event
          assert final_event.status.message == test_message
          # When aggregator state is completed (not working), final event should be completed
          assert final_event.status.state == TaskState.completed

  @pytest.mark.asyncio
  async def test_handle_request_with_non_working_aggregator_state(self):
    """Test that when aggregator state is not working, it preserves the original state."""
    # Setup context with task_id
    self.mock_context.task_id = "test-task-id"

    # Create a test message to be returned by the aggregator
    from a2a.types import Message
    from a2a.types import Role
    from a2a.types import TextPart

    test_message = Mock(spec=Message)
    test_message.message_id = "test-message-id"
    test_message.role = Role.agent
    test_message.parts = [Mock(spec=TextPart)]

    # Setup detailed mocks
    with patch(
        "google.adk.a2a.executor.a2a_agent_executor.convert_a2a_request_to_adk_run_args"
    ) as mock_convert:
      mock_convert.return_value = {
          "user_id": "test-user",
          "session_id": "test-session",
          "new_message": Mock(),
          "run_config": Mock(),
      }

      # Mock session service
      mock_session = Mock()
      mock_session.id = "test-session"
      self.mock_runner.session_service.get_session = AsyncMock(
          return_value=mock_session
      )

      # Mock invocation context
      mock_invocation_context = Mock()
      self.mock_runner._new_invocation_context.return_value = (
          mock_invocation_context
      )

      # Mock agent run with multiple events using proper async generator
      mock_events = [Mock(spec=Event), Mock(spec=Event)]

      # Configure run_async to return the async generator when awaited
      async def mock_run_async(**kwargs):
        async for item in self._create_async_generator(mock_events):
          yield item

      self.mock_runner.run_async = mock_run_async

      with patch(
          "google.adk.a2a.executor.a2a_agent_executor.convert_event_to_a2a_events"
      ) as mock_convert_events:
        mock_convert_events.return_value = [Mock()]

        with patch(
            "google.adk.a2a.executor.a2a_agent_executor.TaskResultAggregator"
        ) as mock_aggregator_class:
          mock_aggregator = Mock()
          # Test with failed state - should preserve failed state
          mock_aggregator.task_state = TaskState.failed
          mock_aggregator.task_status_message = test_message
          mock_aggregator_class.return_value = mock_aggregator

          # Execute
          await self.executor._handle_request(
              self.mock_context, self.mock_event_queue
          )

          # Verify final event preserves the non-working state
          final_events = [
              call[0][0]
              for call in self.mock_event_queue.enqueue_event.call_args_list
              if hasattr(call[0][0], "final") and call[0][0].final == True
          ]
          assert len(final_events) >= 1
          final_event = final_events[-1]  # Get the last final event
          assert final_event.status.message == test_message
          # When aggregator state is failed (not working), final event should keep failed state
          assert final_event.status.state == TaskState.failed

  @pytest.mark.asyncio
  async def test_handle_request_with_working_state_publishes_artifact_and_completed(
      self,
  ):
    """Test that when aggregator state is working, it publishes artifact update and completed status."""
    # Setup context with task_id
    self.mock_context.task_id = "test-task-id"
    self.mock_context.context_id = "test-context-id"

    # Create a test message to be returned by the aggregator
    from a2a.types import Message
    from a2a.types import Part
    from a2a.types import Role
    from a2a.types import TextPart

    test_message = Mock(spec=Message)
    test_message.message_id = "test-message-id"
    test_message.role = Role.agent
    test_message.parts = [Part(root=TextPart(text="test content"))]

    # Setup detailed mocks
    with patch(
        "google.adk.a2a.executor.a2a_agent_executor.convert_a2a_request_to_adk_run_args"
    ) as mock_convert:
      mock_convert.return_value = {
          "user_id": "test-user",
          "session_id": "test-session",
          "new_message": Mock(),
          "run_config": Mock(),
      }

      # Mock session service
      mock_session = Mock()
      mock_session.id = "test-session"
      self.mock_runner.session_service.get_session = AsyncMock(
          return_value=mock_session
      )

      # Mock invocation context
      mock_invocation_context = Mock()
      self.mock_runner._new_invocation_context.return_value = (
          mock_invocation_context
      )

      # Mock agent run with multiple events using proper async generator
      mock_events = [Mock(spec=Event), Mock(spec=Event)]

      # Configure run_async to return the async generator when awaited
      async def mock_run_async(**kwargs):
        async for item in self._create_async_generator(mock_events):
          yield item

      self.mock_runner.run_async = mock_run_async

      with patch(
          "google.adk.a2a.executor.a2a_agent_executor.convert_event_to_a2a_events"
      ) as mock_convert_events:
        mock_convert_events.return_value = [Mock()]

        with patch(
            "google.adk.a2a.executor.a2a_agent_executor.TaskResultAggregator"
        ) as mock_aggregator_class:
          mock_aggregator = Mock()
          # Test with working state - should publish artifact update and completed status
          mock_aggregator.task_state = TaskState.working
          mock_aggregator.task_status_message = test_message
          mock_aggregator_class.return_value = mock_aggregator

          # Execute
          await self.executor._handle_request(
              self.mock_context, self.mock_event_queue
          )

          # Verify artifact update event was published
          artifact_events = [
              call[0][0]
              for call in self.mock_event_queue.enqueue_event.call_args_list
              if hasattr(call[0][0], "artifact")
              and call[0][0].last_chunk == True
          ]
          assert len(artifact_events) == 1
          artifact_event = artifact_events[0]
          assert artifact_event.task_id == "test-task-id"
          assert artifact_event.context_id == "test-context-id"
          # Check that artifact parts correspond to message parts
          assert len(artifact_event.artifact.parts) == len(test_message.parts)
          assert artifact_event.artifact.parts == test_message.parts

          # Verify final status event was published with completed state
          final_events = [
              call[0][0]
              for call in self.mock_event_queue.enqueue_event.call_args_list
              if hasattr(call[0][0], "final") and call[0][0].final == True
          ]
          assert len(final_events) >= 1
          final_event = final_events[-1]  # Get the last final event
          assert final_event.status.state == TaskState.completed
          assert final_event.task_id == "test-task-id"
          assert final_event.context_id == "test-context-id"

  @pytest.mark.asyncio
  async def test_handle_request_with_non_working_state_publishes_status_only(
      self,
  ):
    """Test that when aggregator state is not working, it publishes only the status event."""
    # Setup context with task_id
    self.mock_context.task_id = "test-task-id"
    self.mock_context.context_id = "test-context-id"

    # Create a test message to be returned by the aggregator
    from a2a.types import Message
    from a2a.types import Part
    from a2a.types import Role
    from a2a.types import TextPart

    test_message = Mock(spec=Message)
    test_message.message_id = "test-message-id"
    test_message.role = Role.agent
    test_message.parts = [Part(root=TextPart(text="test content"))]

    # Setup detailed mocks
    with patch(
        "google.adk.a2a.executor.a2a_agent_executor.convert_a2a_request_to_adk_run_args"
    ) as mock_convert:
      mock_convert.return_value = {
          "user_id": "test-user",
          "session_id": "test-session",
          "new_message": Mock(),
          "run_config": Mock(),
      }

      # Mock session service
      mock_session = Mock()
      mock_session.id = "test-session"
      self.mock_runner.session_service.get_session = AsyncMock(
          return_value=mock_session
      )

      # Mock invocation context
      mock_invocation_context = Mock()
      self.mock_runner._new_invocation_context.return_value = (
          mock_invocation_context
      )

      # Mock agent run with multiple events using proper async generator
      mock_events = [Mock(spec=Event), Mock(spec=Event)]

      # Configure run_async to return the async generator when awaited
      async def mock_run_async(**kwargs):
        async for item in self._create_async_generator(mock_events):
          yield item

      self.mock_runner.run_async = mock_run_async

      with patch(
          "google.adk.a2a.executor.a2a_agent_executor.convert_event_to_a2a_events"
      ) as mock_convert_events:
        mock_convert_events.return_value = [Mock()]

        with patch(
            "google.adk.a2a.executor.a2a_agent_executor.TaskResultAggregator"
        ) as mock_aggregator_class:
          mock_aggregator = Mock()
          # Test with auth_required state - should publish only status event
          mock_aggregator.task_state = TaskState.auth_required
          mock_aggregator.task_status_message = test_message
          mock_aggregator_class.return_value = mock_aggregator

          # Execute
          await self.executor._handle_request(
              self.mock_context, self.mock_event_queue
          )

          # Verify no artifact update event was published
          artifact_events = [
              call[0][0]
              for call in self.mock_event_queue.enqueue_event.call_args_list
              if hasattr(call[0][0], "artifact")
              and call[0][0].last_chunk == True
          ]
          assert len(artifact_events) == 0

          # Verify final status event was published with the actual state and message
          final_events = [
              call[0][0]
              for call in self.mock_event_queue.enqueue_event.call_args_list
              if hasattr(call[0][0], "final") and call[0][0].final == True
          ]
          assert len(final_events) >= 1
          final_event = final_events[-1]  # Get the last final event
          assert final_event.status.state == TaskState.auth_required
          assert final_event.status.message == test_message
          assert final_event.task_id == "test-task-id"
          assert final_event.context_id == "test-context-id"
