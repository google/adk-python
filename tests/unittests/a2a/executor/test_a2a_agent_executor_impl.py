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

from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import Message
from a2a.types import Task
from a2a.types import TaskState
from a2a.types import TaskStatus
from a2a.types import TaskStatusUpdateEvent
from a2a.types import TextPart
from google.adk.a2a.converters.request_converter import AgentRunRequest
from google.adk.a2a.converters.utils import _get_adk_metadata_key
from google.adk.a2a.executor.a2a_agent_executor_impl import _A2aAgentExecutor as A2aAgentExecutor
from google.adk.a2a.executor.a2a_agent_executor_impl import _NEW_A2A_ADK_INTEGRATION_EXTENSION
from google.adk.a2a.executor.a2a_agent_executor_impl import A2aAgentExecutorConfig
from google.adk.a2a.executor.config import ExecuteInterceptor
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.runners import RunConfig
from google.adk.runners import Runner
from google.genai.types import Content
import pytest


class TestA2aAgentExecutor:
  """Test suite for A2aAgentExecutor class."""

  def setup_method(self):
    """Set up test fixtures."""
    self.mock_runner = Mock(spec=Runner)
    self.mock_runner.app_name = "test-app"
    self.mock_runner.session_service = Mock()
    self.mock_runner._new_invocation_context = Mock()
    self.mock_runner.run_async = AsyncMock()

    self.mock_a2a_part_converter = Mock()
    self.mock_gen_ai_part_converter = Mock()
    self.mock_request_converter = Mock()
    self.mock_event_converter = Mock()
    self.mock_config = A2aAgentExecutorConfig(
        a2a_part_converter=self.mock_a2a_part_converter,
        gen_ai_part_converter=self.mock_gen_ai_part_converter,
        request_converter=self.mock_request_converter,
        adk_event_converter=self.mock_event_converter,
    )
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

    self.expected_metadata = {
        _get_adk_metadata_key("app_name"): "test-app",
        _get_adk_metadata_key("user_id"): "test-user",
        _get_adk_metadata_key("session_id"): "test-session",
        _NEW_A2A_ADK_INTEGRATION_EXTENSION: {"adk_agent_executor_v2": True},
    }

  async def _create_async_generator(self, items):
    """Helper to create async generator from items."""
    for item in items:
      yield item

  @pytest.mark.asyncio
  async def test_execute_success_new_task(self):
    """Test successful execution of a new task."""
    # Setup
    self.mock_request_converter.return_value = AgentRunRequest(
        user_id="test-user",
        session_id="test-session",
        new_message=Mock(spec=Content),
        run_config=Mock(spec=RunConfig),
    )
    # Mock session service
    mock_session = Mock()
    mock_session.id = "test-session"
    self.mock_runner.session_service.get_session = AsyncMock(
        return_value=mock_session
    )

    # Mock agent run with proper async generator
    mock_event = Event(
        invocation_id="invocation-id",
        author="test-agent",
        branch="main",
        partial=False,
    )

    # Configure run_async to return the async generator when awaited
    async def mock_run_async(**kwargs):
      async for item in self._create_async_generator([mock_event]):
        yield item

    self.mock_runner.run_async = mock_run_async

    # Mock event converter to return a working status update
    working_event = TaskStatusUpdateEvent(
        task_id="test-task-id",
        status=TaskStatus(state=TaskState.working, timestamp="now"),
        context_id="test-context-id",
        final=False,
    )
    self.mock_event_converter.return_value = [working_event]

    # Execute
    await self.executor.execute(self.mock_context, self.mock_event_queue)

    # Verify request converter was called with proper arguments
    self.mock_request_converter.assert_called_once_with(
        self.mock_context, self.mock_a2a_part_converter
    )

    # Verify event converter was called with proper arguments
    self.mock_event_converter.assert_called_once_with(
        mock_event,
        {},  # agents_artifact (initially empty)
        self.mock_context.task_id,
        self.mock_context.context_id,
        self.mock_gen_ai_part_converter,
    )

    # Verify task submitted event was enqueued
    # call 0: submitted
    # call 1: working (from converter)
    # call 2: completed (final)
    assert self.mock_event_queue.enqueue_event.call_count >= 3

    submitted_event = self.mock_event_queue.enqueue_event.call_args_list[0][0][
        0
    ]
    assert isinstance(submitted_event, Task)
    assert submitted_event.status.state == TaskState.submitted
    assert submitted_event.metadata == self.expected_metadata

    # Verify working event was enqueued
    enqueued_working_event = self.mock_event_queue.enqueue_event.call_args_list[
        1
    ][0][0]
    assert isinstance(enqueued_working_event, TaskStatusUpdateEvent)
    assert enqueued_working_event.status.state == TaskState.working
    assert enqueued_working_event.metadata == self.expected_metadata

    # Verify converted event was enqueued
    converted_event = self.mock_event_queue.enqueue_event.call_args_list[2][0][
        0
    ]
    assert converted_event == working_event
    assert converted_event.metadata == self.expected_metadata

    # Verify final event was enqueued
    final_event = self.mock_event_queue.enqueue_event.call_args_list[-1][0][0]
    assert final_event.final == True
    assert final_event.status.state == TaskState.completed
    assert final_event.metadata == self.expected_metadata

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

    self.mock_request_converter.return_value = AgentRunRequest(
        user_id="test-user",
        session_id="test-session",
        new_message=Mock(spec=Content),
        run_config=Mock(spec=RunConfig),
    )

    # Mock session service
    mock_session = Mock()
    mock_session.id = "test-session"
    self.mock_runner.session_service.get_session = AsyncMock(
        return_value=mock_session
    )

    # Mock agent run with proper async generator
    mock_event = Event(
        invocation_id="invocation-id",
        author="test-agent",
        branch="main",
        partial=False,
    )

    # Configure run_async to return the async generator when awaited
    async def mock_run_async(**kwargs):
      async for item in self._create_async_generator([mock_event]):
        yield item

    self.mock_runner.run_async = mock_run_async

    # Mock event converter
    working_event = TaskStatusUpdateEvent(
        task_id="existing-task-id",
        status=TaskStatus(state=TaskState.working, timestamp="now"),
        context_id="test-context-id",
        final=False,
    )
    self.mock_event_converter.return_value = [working_event]

    # Execute
    await self.executor.execute(self.mock_context, self.mock_event_queue)

    # Verify submitted event was NOT enqueued for existing task
    # So we check first event is working state
    first_event = self.mock_event_queue.enqueue_event.call_args_list[0][0][0]
    assert isinstance(first_event, TaskStatusUpdateEvent)
    assert first_event.status.state == TaskState.working
    assert first_event.metadata == self.expected_metadata

    # Verify manual working event is FIRST
    assert isinstance(first_event, TaskStatusUpdateEvent)
    assert first_event.status.state == TaskState.working

    # Verify converted event was enqueued
    converted_event = self.mock_event_queue.enqueue_event.call_args_list[1][0][
        0
    ]
    assert converted_event == working_event
    assert converted_event.metadata == self.expected_metadata

    # Verify final event
    final_event = self.mock_event_queue.enqueue_event.call_args_list[-1][0][0]
    assert final_event.final == True
    assert final_event.status.state == TaskState.completed
    assert final_event.metadata == self.expected_metadata

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
  async def test_handle_request_integration(self):
    """Test the complete request handling flow."""
    # Setup context with task_id
    self.mock_context.task_id = "test-task-id"

    # Setup detailed mocks
    self.mock_request_converter.return_value = AgentRunRequest(
        user_id="test-user",
        session_id="test-session",
        new_message=Mock(spec=Content),
        run_config=Mock(spec=RunConfig),
    )

    # Mock session service
    mock_session = Mock()
    mock_session.id = "test-session"
    self.mock_runner.session_service.get_session = AsyncMock(
        return_value=mock_session
    )

    # Mock agent run with multiple events using proper async generator
    mock_events = [
        Event(
            invocation_id="invocation-id",
            author="test-agent",
            branch="main",
            partial=False,
        ),
        Event(
            invocation_id="invocation-id",
            author="test-agent",
            branch="main",
            partial=False,
        ),
    ]

    # Configure run_async to return the async generator when awaited
    async def mock_run_async(**kwargs):
      async for item in self._create_async_generator(mock_events):
        yield item

    self.mock_runner.run_async = mock_run_async

    # Mock event converter to return events
    working_event = TaskStatusUpdateEvent(
        task_id="test-task-id",
        status=TaskStatus(state=TaskState.working, timestamp="now"),
        context_id="test-context-id",
        final=False,
    )
    self.mock_event_converter.return_value = [working_event]

    # Initialize executor context attributes as they would be in execute()
    self.executor._invocation_metadata = {}
    self.executor._executor_context = Mock()

    # Execute
    await self.executor._handle_request(
        self.mock_context,
        self.executor._executor_context,
        self.mock_event_queue,
        self.mock_runner,
        self.mock_request_converter.return_value,
    )

    # Verify events enqueued
    # Should check for working events
    working_events = [
        call[0][0]
        for call in self.mock_event_queue.enqueue_event.call_args_list
        if hasattr(call[0][0], "status")
        and call[0][0].status.state == TaskState.working
    ]
    # Each ADK event generates 1 working event in this mock setup
    assert len(working_events) >= len(mock_events)

    # Verify final event is completed
    final_events = [
        call[0][0]
        for call in self.mock_event_queue.enqueue_event.call_args_list
        if hasattr(call[0][0], "final") and call[0][0].final == True
    ]
    assert len(final_events) >= 1
    final_event = final_events[-1]
    assert final_event.status.state == TaskState.completed

  @pytest.mark.asyncio
  async def test_cancel_with_task_id(self):
    """Test cancellation with a task ID."""
    self.mock_context.task_id = "test-task-id"

    with pytest.raises(
        NotImplementedError, match="Cancellation is not supported"
    ):
      await self.executor.cancel(self.mock_context, self.mock_event_queue)

  @pytest.mark.asyncio
  async def test_execute_with_exception_handling(self):
    """Test execution with exception handling."""
    self.mock_context.task_id = "test-task-id"
    self.mock_context.current_task = None

    self.mock_request_converter.side_effect = Exception("Test error")

    # Execute (should not raise since we catch the exception)
    await self.executor.execute(self.mock_context, self.mock_event_queue)

    # Check failure event (last)
    failure_event = self.mock_event_queue.enqueue_event.call_args_list[-1][0][0]
    assert failure_event.status.state == TaskState.failed
    assert failure_event.final == True
    assert "Test error" in failure_event.status.message.parts[0].root.text

  @pytest.mark.asyncio
  async def test_handle_request_with_non_working_state(self):
    """Test handle request when a non-working state is encountered."""
    # Setup context with task_id
    self.mock_context.task_id = "test-task-id"
    self.mock_context.context_id = "test-context-id"

    # Mock agent run event
    mock_event = Event(
        invocation_id="invocation-id",
        author="test-agent",
        branch="main",
        partial=False,
    )
    mock_event.error_code = "ERROR"

    async def mock_run_async(**kwargs):
      async for item in self._create_async_generator([mock_event]):
        yield item

    self.mock_runner.run_async = mock_run_async

    # Mock event converter to return a FAILED event
    failed_event = TaskStatusUpdateEvent(
        task_id="test-task-id",
        status=TaskStatus(state=TaskState.failed, timestamp="now"),
        context_id="test-context-id",
        final=False,
    )
    self.mock_event_converter.return_value = [failed_event]

    run_request = AgentRunRequest(
        user_id="test-user",
        session_id="test-session",
        new_message=Mock(spec=Content),
        run_config=Mock(spec=RunConfig),
    )

    # Initialize executor context attributes
    self.executor._invocation_metadata = {}
    self.executor._executor_context = Mock()

    # Execute
    await self.executor._handle_request(
        self.mock_context,
        self.executor._executor_context,
        self.mock_event_queue,
        self.mock_runner,
        run_request,
    )

    # Verify final event is FAILED, not COMPLETED
    final_events = [
        call[0][0]
        for call in self.mock_event_queue.enqueue_event.call_args_list
        if hasattr(call[0][0], "final") and call[0][0].final == True
    ]
    assert len(final_events) >= 1
    # The last event should be the synthesized final event
    final_event = final_events[-1]
    assert final_event.status.state == TaskState.failed

  @pytest.mark.asyncio
  async def test_handle_request_with_error_message(self):
    """Test handle request when an error message is present without an error code."""
    self.mock_context.task_id = "test-task-id"
    self.mock_context.context_id = "test-context-id"

    # Mock agent run event with only error_message
    mock_event = Event(
        invocation_id="invocation-id",
        author="test-agent",
        branch="main",
        partial=False,
    )
    mock_event.error_code = None
    mock_event.error_message = "Test Error Message"

    async def mock_run_async(**kwargs):
      async for item in self._create_async_generator([mock_event]):
        yield item

    self.mock_runner.run_async = mock_run_async
    self.mock_event_converter.return_value = []

    run_request = AgentRunRequest(
        user_id="test-user",
        session_id="test-session",
        new_message=Mock(spec=Content),
        run_config=Mock(spec=RunConfig),
    )

    executor_context = Mock()
    executor_context.app_name = "test-app"
    executor_context.user_id = "test-user"
    executor_context.session_id = "test-session"

    await self.executor._handle_request(
        self.mock_context,
        executor_context,
        self.mock_event_queue,
        self.mock_runner,
        run_request,
    )

    final_events = [
        call[0][0]
        for call in self.mock_event_queue.enqueue_event.call_args_list
        if hasattr(call[0][0], "final") and call[0][0].final == True
    ]
    assert len(final_events) >= 1
    final_event = final_events[-1]
    assert final_event.status.state == TaskState.failed
    assert final_event.metadata == self.expected_metadata

  @pytest.mark.asyncio
  async def test_interceptors(self):
    """Test interceptors execution."""
    # Setup interceptors
    before_interceptor = AsyncMock(return_value=self.mock_context)
    after_event_interceptor = AsyncMock()
    after_event_interceptor.side_effect = lambda ctx, a2a, adk: a2a
    after_agent_interceptor = AsyncMock()
    after_agent_interceptor.side_effect = lambda ctx, event: event

    interceptor = ExecuteInterceptor(
        before_agent=before_interceptor,
        after_event=after_event_interceptor,
        after_agent=after_agent_interceptor,
    )

    self.mock_config.execute_interceptors = [interceptor]

    # Mock run
    mock_event = Event(
        invocation_id="invocation-id",
        author="test-agent",
        branch="main",
        partial=False,
    )

    async def mock_run_async(**kwargs):
      async for item in self._create_async_generator([mock_event]):
        yield item

    self.mock_runner.run_async = mock_run_async

    # Mock event converter
    working_event = TaskStatusUpdateEvent(
        task_id="test-task-id",
        status=TaskStatus(state=TaskState.working, timestamp="now"),
        context_id="test-context-id",
        final=False,
    )
    self.mock_event_converter.return_value = [working_event]

    # Pre-setup request converter
    self.mock_request_converter.return_value = AgentRunRequest(
        user_id="test-user",
        session_id="test-session",
        new_message=Mock(spec=Content),
        run_config=Mock(spec=RunConfig),
    )

    # Mock session
    mock_session = Mock()
    mock_session.id = "test-session"
    self.mock_runner.session_service.get_session = AsyncMock(
        return_value=mock_session
    )

    # Execute
    await self.executor.execute(self.mock_context, self.mock_event_queue)

    # Verify interceptors called
    before_interceptor.assert_called_once_with(self.mock_context)
    # after_event called for each event
    assert after_event_interceptor.call_count >= 1
    after_agent_interceptor.assert_called_once()

  @pytest.mark.asyncio
  @patch("google.adk.a2a.executor.a2a_agent_executor_impl.handle_user_input")
  async def test_execute_missing_user_input(self, mock_handle_user_input):
    """Test when handle_user_input returns a missing user input event."""
    self.mock_context.current_task = Mock()
    self.mock_context.task_id = "test-task-id"
    self.mock_context.context_id = "test-context-id"

    # Set up handle_user_input to return an event
    missing_event = TaskStatusUpdateEvent(
        task_id="test-task-id",
        status=TaskStatus(state=TaskState.input_required, timestamp="now"),
        context_id="test-context-id",
        final=False,
    )
    mock_handle_user_input.return_value = missing_event

    self.mock_runner.session_service.get_session = AsyncMock(
        return_value=Mock(id="test-session")
    )
    self.mock_request_converter.return_value = AgentRunRequest(
        user_id="test-user",
        session_id="test-session",
        new_message=Mock(spec=Content),
        run_config=Mock(spec=RunConfig),
    )

    # Execute
    await self.executor.execute(self.mock_context, self.mock_event_queue)

    # Verify that the missing_event was enqueued
    self.mock_event_queue.enqueue_event.assert_called_once_with(missing_event)

    # Verify that metadata was injected
    enqueued_event = self.mock_event_queue.enqueue_event.call_args[0][0]
    assert enqueued_event.metadata == self.expected_metadata

  @pytest.mark.asyncio
  async def test_resolve_session_creates_new_session(self):
    """Test that _resolve_session creates a new session if it doesn't exist."""
    self.mock_runner.session_service.get_session = AsyncMock(return_value=None)

    new_session = Mock()
    new_session.id = "new-session-id"
    self.mock_runner.session_service.create_session = AsyncMock(
        return_value=new_session
    )

    run_request = AgentRunRequest(
        user_id="test-user",
        session_id="old-session-id",
        new_message=Mock(spec=Content),
        run_config=Mock(spec=RunConfig),
    )

    await self.executor._resolve_session(run_request, self.mock_runner)

    self.mock_runner.session_service.get_session.assert_called_once_with(
        app_name=self.mock_runner.app_name,
        user_id="test-user",
        session_id="old-session-id",
    )
    self.mock_runner.session_service.create_session.assert_called_once_with(
        app_name=self.mock_runner.app_name,
        user_id="test-user",
        state={},
        session_id="old-session-id",
    )
    assert run_request.session_id == "new-session-id"

  @pytest.mark.asyncio
  async def test_execute_enqueue_error_in_exception_handler(self):
    """Test failure event publishing handles exception during enqueue."""
    self.mock_context.task_id = "test-task-id"
    self.mock_request_converter.side_effect = Exception("Test error")

    # Make enqueue_event raise an exception
    self.mock_event_queue.enqueue_event.side_effect = Exception("Enqueue error")

    # This should not raise an exception itself
    await self.executor.execute(self.mock_context, self.mock_event_queue)

    # Verify enqueue_event was called to publish the error event
    assert self.mock_event_queue.enqueue_event.call_count == 1

  @pytest.mark.asyncio
  @patch("google.adk.a2a.executor.a2a_agent_executor_impl.LongRunningFunctions")
  async def test_long_running_functions_final_event(self, mock_lrf_class):
    """Test _handle_request when there are long running function calls."""
    self.mock_context.task_id = "test-task-id"
    self.mock_context.context_id = "test-context-id"

    # Set up mock LongRunningFunctions
    mock_lrf = mock_lrf_class.return_value
    mock_lrf.process_event.side_effect = lambda e: e
    mock_lrf.has_long_running_function_calls.return_value = True

    lrf_event = TaskStatusUpdateEvent(
        task_id="test-task-id",
        status=TaskStatus(state=TaskState.input_required, timestamp="now"),
        context_id="test-context-id",
        final=False,
    )
    mock_lrf.create_long_running_function_call_event.return_value = lrf_event

    self.mock_request_converter.return_value = AgentRunRequest(
        user_id="test-user",
        session_id="test-session",
        new_message=Mock(spec=Content),
        run_config=Mock(spec=RunConfig),
    )

    mock_session = Mock()
    mock_session.id = "test-session"
    self.mock_runner.session_service.get_session = AsyncMock(
        return_value=mock_session
    )

    mock_event = Event(
        invocation_id="invocation-id",
        author="test-agent",
        branch="main",
        partial=False,
    )

    async def mock_run_async(**kwargs):
      async for item in self._create_async_generator([mock_event]):
        yield item

    self.mock_runner.run_async = mock_run_async
    self.mock_event_converter.return_value = []

    self.executor._invocation_metadata = {}
    self.executor._executor_context = Mock()

    await self.executor._handle_request(
        self.mock_context,
        self.executor._executor_context,
        self.mock_event_queue,
        self.mock_runner,
        self.mock_request_converter.return_value,
    )

    # Verify final event is the long running function call event
    final_events = [
        call[0][0]
        for call in self.mock_event_queue.enqueue_event.call_args_list
        if call[0][0] == lrf_event
    ]
    assert len(final_events) >= 1

  @pytest.mark.asyncio
  async def test_after_event_interceptor_returns_none(self):
    """Test after_event_interceptor returning None drops the event."""
    # Setup interceptor returning None
    after_event_interceptor = AsyncMock()
    after_event_interceptor.side_effect = lambda ctx, a2a, adk: None

    interceptor = ExecuteInterceptor(
        after_event=after_event_interceptor,
    )
    self.mock_config.execute_interceptors = [interceptor]

    self.mock_context.task_id = "test-task-id"
    self.mock_context.context_id = "test-context-id"

    self.mock_request_converter.return_value = AgentRunRequest(
        user_id="test-user",
        session_id="test-session",
        new_message=Mock(spec=Content),
        run_config=Mock(spec=RunConfig),
    )

    mock_event = Event(
        invocation_id="invocation-id",
        author="test-agent",
        branch="main",
        partial=False,
    )

    async def mock_run_async(**kwargs):
      async for item in self._create_async_generator([mock_event]):
        yield item

    self.mock_runner.run_async = mock_run_async

    # Event converter returns one event
    working_event = TaskStatusUpdateEvent(
        task_id="test-task-id",
        status=TaskStatus(state=TaskState.working, timestamp="now"),
        context_id="test-context-id",
        final=False,
    )
    self.mock_event_converter.return_value = [working_event]

    self.executor._executor_context = Mock()
    await self.executor._handle_request(
        self.mock_context,
        self.executor._executor_context,
        self.mock_event_queue,
        self.mock_runner,
        self.mock_request_converter.return_value,
    )

    # Since the interceptor returns None, working_event should NOT be enqueued
    # The only event enqueued by _handle_request should be the final event
    assert self.mock_event_queue.enqueue_event.call_count == 1
    final_event = self.mock_event_queue.enqueue_event.call_args_list[0][0][0]
    assert final_event.status.state == TaskState.completed
