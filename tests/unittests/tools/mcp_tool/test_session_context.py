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

from __future__ import annotations

import asyncio
from datetime import timedelta
from unittest.mock import patch

from google.adk.tools.mcp_tool.session_context import SessionContext
import pytest


class MockClientSession:
  """Mock ClientSession for testing."""

  def __init__(self, *args, **kwargs):
    self._initialized = False
    self._args = args
    self._kwargs = kwargs

  async def initialize(self):
    """Mock initialize method."""
    self._initialized = True
    return self

  async def __aenter__(self):
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    return False


class MockClient:
  """Mock MCP client."""

  def __init__(
      self,
      transports=None,
      raise_on_enter=None,
      delay_on_enter=0,
  ):
    self._transports = transports or ("read_stream", "write_stream")
    self._raise_on_enter = raise_on_enter
    self._delay_on_enter = delay_on_enter
    self._entered = False
    self._exited = False

  async def __aenter__(self):
    if self._delay_on_enter > 0:
      await asyncio.sleep(self._delay_on_enter)
    if self._raise_on_enter:
      raise self._raise_on_enter
    self._entered = True
    return self._transports

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    self._exited = True
    return False


class TestSessionContext:
  """Test suite for SessionContext class."""

  @pytest.mark.asyncio
  async def test_start_success_ready_event_set_and_session_returned(self):
    """Test that start() sets _ready_event and returns session."""
    mock_client = MockClient()
    session_context = SessionContext(
        mock_client, timeout=5.0, sse_read_timeout=None
    )

    # Mock ClientSession
    mock_session = MockClientSession()

    with patch(
        "google.adk.tools.mcp_tool.session_context.ClientSession"
    ) as mock_session_class:
      mock_session_class.return_value = mock_session

      session = await session_context.start()

      # Verify ready_event was set
      assert session_context._ready_event.is_set()

      # Verify session was returned
      assert session == mock_session
      assert session_context.session == mock_session

      # Verify initialize was called
      assert mock_session._initialized

      # Verify task was created and is still running (waiting for close)
      assert session_context._task is not None
      assert not session_context._task.done()

      # Clean up
      await session_context.close()

  @pytest.mark.asyncio
  async def test_start_raises_connection_error_on_exception(self):
    """Test that start() raises ConnectionError when exception occurs."""
    test_exception = ValueError("Connection failed")
    mock_client = MockClient(raise_on_enter=test_exception)
    session_context = SessionContext(
        mock_client, timeout=5.0, sse_read_timeout=None
    )

    with pytest.raises(ConnectionError) as exc_info:
      await session_context.start()

    # Verify ConnectionError message contains original exception
    assert "Failed to create MCP session" in str(exc_info.value)
    assert "Connection failed" in str(exc_info.value)

    # Verify ready_event was set (in finally block)
    assert session_context._ready_event.is_set()

  @pytest.mark.asyncio
  async def test_start_raises_connection_error_on_cancelled_error(self):
    """Test that start() raises ConnectionError when CancelledError occurs."""
    mock_client = MockClient()
    session_context = SessionContext(
        mock_client, timeout=5.0, sse_read_timeout=None
    )

    # Mock session that will cause cancellation
    mock_session = MockClientSession()

    # Make initialize raise CancelledError
    async def cancelled_initialize():
      raise asyncio.CancelledError("Task cancelled")

    mock_session.initialize = cancelled_initialize

    with patch(
        "google.adk.tools.mcp_tool.session_context.ClientSession"
    ) as mock_session_class:
      mock_session_class.return_value = mock_session

      # Should raise ConnectionError (not CancelledError directly)
      with pytest.raises(ConnectionError) as exc_info:
        await session_context.start()

      # Verify it's a ConnectionError about cancellation
      assert "Failed to create MCP session" in str(exc_info.value)
      assert "task cancelled" in str(exc_info.value)

      # Verify ready_event was set
      assert session_context._ready_event.is_set()

  @pytest.mark.asyncio
  async def test_close_cleans_up_task(self):
    """Test that close() properly cleans up the task."""
    mock_client = MockClient()
    session_context = SessionContext(
        mock_client, timeout=5.0, sse_read_timeout=None
    )

    # Mock ClientSession
    mock_session = MockClientSession()

    with patch(
        "google.adk.tools.mcp_tool.session_context.ClientSession"
    ) as mock_session_class:
      mock_session_class.return_value = mock_session

      # Start the session context
      await session_context.start()

      # Verify task is running
      assert session_context._task is not None
      assert not session_context._task.done()

      # Close the session context
      await session_context.close()

      # Wait a bit for cleanup
      await asyncio.sleep(0.1)

      # Verify close_event was set
      assert session_context._close_event.is_set()

      # Verify task completed (may take a moment)
      # The task should finish after close_event is set
      assert session_context._task.done()

  @pytest.mark.asyncio
  async def test_session_exception_does_not_break_event_loop(self):
    """Test that session exceptions don't break the event loop."""
    mock_client = MockClient()
    session_context = SessionContext(
        mock_client, timeout=5.0, sse_read_timeout=None
    )

    # Mock ClientSession that raises exception during use
    mock_session = MockClientSession()

    async def failing_operation():
      raise RuntimeError("Session operation failed")

    mock_session.failing_operation = failing_operation

    with patch(
        "google.adk.tools.mcp_tool.session_context.ClientSession"
    ) as mock_session_class:
      mock_session_class.return_value = mock_session

      # Start the session context
      session = await session_context.start()

      # Use session and trigger exception
      with pytest.raises(RuntimeError, match="Session operation failed"):
        await session.failing_operation()

      # Close the session context - should not break event loop
      await session_context.close()

      # Verify event loop is still healthy by running another task
      result = await asyncio.sleep(0.01)
      assert result is None

  @pytest.mark.asyncio
  async def test_async_context_manager(self):
    """Test using SessionContext as async context manager."""
    mock_client = MockClient()
    mock_session = MockClientSession()

    with patch(
        "google.adk.tools.mcp_tool.session_context.ClientSession"
    ) as mock_session_class:
      mock_session_class.return_value = mock_session

      async with SessionContext(
          mock_client, timeout=5.0, sse_read_timeout=None
      ) as session:
        assert session == mock_session
        # Verify initialize was called by checking _initialized flag
        assert session._initialized

  @pytest.mark.asyncio
  async def test_timeout_during_connection(self):
    """Test timeout during client connection."""
    # Client that takes longer than timeout
    mock_client = MockClient(delay_on_enter=10.0)
    session_context = SessionContext(
        mock_client, timeout=0.1, sse_read_timeout=None
    )

    with pytest.raises(ConnectionError) as exc_info:
      await session_context.start()

    assert "Failed to create MCP session" in str(exc_info.value)

  @pytest.mark.asyncio
  async def test_timeout_during_initialization(self):
    """Test timeout during session initialization."""
    mock_client = MockClient()
    session_context = SessionContext(
        mock_client, timeout=0.1, sse_read_timeout=None
    )

    # Mock ClientSession with slow initialize
    mock_session = MockClientSession()

    async def slow_initialize():
      await asyncio.sleep(1.0)
      return mock_session

    mock_session.initialize = slow_initialize

    with patch(
        "google.adk.tools.mcp_tool.session_context.ClientSession"
    ) as mock_session_class:
      mock_session_class.return_value = mock_session

      with pytest.raises(ConnectionError) as exc_info:
        await session_context.start()

      assert "Failed to create MCP session" in str(exc_info.value)

  @pytest.mark.asyncio
  async def test_stdio_client_with_read_timeout(self):
    """Test stdio client includes read_timeout_seconds parameter."""
    mock_client = MockClient()
    session_context = SessionContext(
        mock_client, timeout=5.0, sse_read_timeout=None, is_stdio=True
    )

    mock_session = MockClientSession()

    with patch(
        "google.adk.tools.mcp_tool.session_context.ClientSession"
    ) as mock_session_class:
      mock_session_class.return_value = mock_session

      await session_context.start()

      # Verify ClientSession was called with read_timeout_seconds for stdio
      call_args = mock_session_class.call_args
      assert "read_timeout_seconds" in call_args.kwargs
      assert call_args.kwargs["read_timeout_seconds"] == timedelta(seconds=5.0)

      await session_context.close()

  @pytest.mark.asyncio
  async def test_non_stdio_client_without_read_timeout(self):
    """Test non-stdio client does not include read_timeout_seconds."""
    mock_client = MockClient()
    session_context = SessionContext(
        mock_client, timeout=5.0, sse_read_timeout=None, is_stdio=False
    )

    mock_session = MockClientSession()

    with patch(
        "google.adk.tools.mcp_tool.session_context.ClientSession"
    ) as mock_session_class:
      mock_session_class.return_value = mock_session

      await session_context.start()

      # Verify ClientSession was called with read_timeout_seconds=None for non-stdio
      # when sse_read_timeout is None
      call_args = mock_session_class.call_args
      assert "read_timeout_seconds" in call_args.kwargs
      assert call_args.kwargs["read_timeout_seconds"] is None

      await session_context.close()

  @pytest.mark.asyncio
  async def test_sse_read_timeout_passed_to_client_session(self):
    """Test that sse_read_timeout is passed to ClientSession for non-stdio."""
    mock_client = MockClient()
    session_context = SessionContext(
        mock_client, timeout=5.0, sse_read_timeout=300.0, is_stdio=False
    )

    mock_session = MockClientSession()

    with patch(
        "google.adk.tools.mcp_tool.session_context.ClientSession"
    ) as mock_session_class:
      mock_session_class.return_value = mock_session

      await session_context.start()

      # Verify ClientSession was called with sse_read_timeout
      call_args = mock_session_class.call_args
      assert "read_timeout_seconds" in call_args.kwargs
      assert call_args.kwargs["read_timeout_seconds"] == timedelta(
          seconds=300.0
      )

      await session_context.close()

  @pytest.mark.asyncio
  async def test_close_multiple_times(self):
    """Test that close() can be called multiple times safely."""
    mock_client = MockClient()
    session_context = SessionContext(
        mock_client, timeout=5.0, sse_read_timeout=None
    )

    mock_session = MockClientSession()

    with patch(
        "google.adk.tools.mcp_tool.session_context.ClientSession"
    ) as mock_session_class:
      mock_session_class.return_value = mock_session

      await session_context.start()

      # Close multiple times
      await session_context.close()
      await session_context.close()
      await session_context.close()

      # Should not raise exception
      assert session_context._close_event.is_set()

  @pytest.mark.asyncio
  async def test_close_before_start(self):
    """Test that close() works even if start() was never called."""
    mock_client = MockClient()
    session_context = SessionContext(
        mock_client, timeout=5.0, sse_read_timeout=None
    )

    # Close before starting should not raise
    await session_context.close()

    assert session_context._close_event.is_set()

  @pytest.mark.asyncio
  async def test_close_before_start_ends(self):
    """Test that close() before start() ends the task."""
    # Client has enough time to delay the start task
    mock_client = MockClient(delay_on_enter=10.0)
    session_context = SessionContext(
        mock_client, timeout=5.0, sse_read_timeout=None
    )

    start_task = asyncio.create_task(session_context.start())
    await asyncio.sleep(0.1)
    assert not start_task.done()

    # Call close before start() ends the task
    await session_context.close()
    await asyncio.sleep(0.1)

    assert start_task.done()
    assert isinstance(
        start_task.exception(), ConnectionError
    ) and "task cancelled" in str(start_task.exception())

  @pytest.mark.asyncio
  async def test_close_before_start_called(self):
    """Test that close() before start() called sets the close event."""
    mock_client = MockClient()
    session_context = SessionContext(
        mock_client, timeout=5.0, sse_read_timeout=None
    )

    # Call close() before start() called
    await session_context.close()
    await asyncio.sleep(0.1)

    assert session_context._task is None
    assert session_context._close_event.is_set()

    with pytest.raises(ConnectionError) as exc_info:
      await session_context.start()

    assert "session already closed" in str(exc_info.value)
    assert session_context._task is None

  @pytest.mark.asyncio
  async def test_session_property(self):
    """Test that session property returns the managed session."""
    mock_client = MockClient()
    session_context = SessionContext(
        mock_client, timeout=5.0, sse_read_timeout=None
    )

    # Initially None
    assert session_context.session is None

    mock_session = MockClientSession()

    with patch(
        "google.adk.tools.mcp_tool.session_context.ClientSession"
    ) as mock_session_class:
      mock_session_class.return_value = mock_session

      await session_context.start()

      # Should return the session
      assert session_context.session == mock_session

      await session_context.close()

  @pytest.mark.asyncio
  async def test_client_cleanup_on_exception(self):
    """Test that client is properly cleaned up even when exception occurs."""
    test_exception = RuntimeError("Test error")
    mock_client = MockClient(raise_on_enter=test_exception)
    session_context = SessionContext(
        mock_client, timeout=5.0, sse_read_timeout=None
    )

    with pytest.raises(ConnectionError):
      await session_context.start()

    # Wait a bit for cleanup
    await asyncio.sleep(0.1)

    # Verify task completed
    assert session_context._task.done()

  @pytest.mark.asyncio
  async def test_close_handles_cancelled_error(self):
    """Test that close() handles CancelledError gracefully."""
    mock_client = MockClient()
    session_context = SessionContext(
        mock_client, timeout=5.0, sse_read_timeout=None
    )

    mock_session = MockClientSession()

    with patch(
        "google.adk.tools.mcp_tool.session_context.ClientSession"
    ) as mock_session_class:
      mock_session_class.return_value = mock_session

      await session_context.start()

      # Cancel the task
      if session_context._task:
        session_context._task.cancel()

      # Close should handle CancelledError gracefully
      await session_context.close()

      # Should not raise exception
      assert session_context._close_event.is_set()

  @pytest.mark.asyncio
  async def test_close_handles_exception_during_cleanup(self):
    """Test that close() handles exceptions during cleanup gracefully."""
    mock_client = MockClient()
    session_context = SessionContext(
        mock_client, timeout=5.0, sse_read_timeout=None
    )

    # Create a mock session that raises during exit
    class FailingMockSession(MockClientSession):

      async def __aexit__(self, exc_type, exc_val, exc_tb):
        raise RuntimeError("Cleanup failed")

    failing_session = FailingMockSession()

    with patch(
        "google.adk.tools.mcp_tool.session_context.ClientSession"
    ) as mock_session_class:
      mock_session_class.return_value = failing_session

      await session_context.start()

      # Close should handle the exception gracefully
      await session_context.close()

      # Should not raise exception
      assert session_context._close_event.is_set()


class TestRunGuarded:
  """Tests for SessionContext._run_guarded()."""

  @pytest.mark.asyncio
  async def test_run_guarded_normal_completion(self):
    """Test _run_guarded returns result when coroutine completes normally."""
    mock_client = MockClient()
    mock_session = MockClientSession()

    with patch(
        "google.adk.tools.mcp_tool.session_context.ClientSession"
    ) as mock_session_class:
      mock_session_class.return_value = mock_session

      session_context = SessionContext(
          mock_client, timeout=5.0, sse_read_timeout=None
      )
      await session_context.start()

      async def my_coro():
        return "hello"

      result = await session_context._run_guarded(my_coro())
      assert result == "hello"

      await session_context.close()

  @pytest.mark.asyncio
  async def test_run_guarded_background_task_crash(self):
    """Test _run_guarded raises ConnectionError when background task dies."""
    mock_client = MockClient()
    mock_session = MockClientSession()

    with patch(
        "google.adk.tools.mcp_tool.session_context.ClientSession"
    ) as mock_session_class:
      mock_session_class.return_value = mock_session

      session_context = SessionContext(
          mock_client, timeout=5.0, sse_read_timeout=None
      )
      await session_context.start()

      crash_error = RuntimeError("403 Forbidden")

      async def hanging_coro():
        await asyncio.sleep(100)

      # Kill the background task and replace with one that's already failed
      session_context._task.cancel()
      try:
        await session_context._task
      except (asyncio.CancelledError, Exception):
        pass

      async def failing_task():
        raise crash_error

      session_context._task = asyncio.create_task(failing_task())
      await asyncio.sleep(0.01)

      coro = hanging_coro()
      with pytest.raises(ConnectionError) as exc_info:
        await session_context._run_guarded(coro)

      assert "403 Forbidden" in str(exc_info.value)
      assert exc_info.value.__cause__ is crash_error

  @pytest.mark.asyncio
  async def test_run_guarded_task_already_dead(self):
    """Test _run_guarded raises immediately when task is already done."""
    mock_client = MockClient()
    mock_session = MockClientSession()

    with patch(
        "google.adk.tools.mcp_tool.session_context.ClientSession"
    ) as mock_session_class:
      mock_session_class.return_value = mock_session

      session_context = SessionContext(
          mock_client, timeout=5.0, sse_read_timeout=None
      )
      await session_context.start()

      original_error = ValueError("transport died")

      async def failing_task():
        raise original_error

      session_context._task.cancel()
      try:
        await session_context._task
      except (asyncio.CancelledError, Exception):
        pass

      session_context._task = asyncio.create_task(failing_task())
      await asyncio.sleep(0.01)

      async def my_coro():
        return "should not reach"

      coro = my_coro()
      with pytest.raises(ConnectionError) as exc_info:
        await session_context._run_guarded(coro)

      assert "already terminated" in str(exc_info.value)
      assert exc_info.value.__cause__ is original_error

  @pytest.mark.asyncio
  async def test_run_guarded_cancels_coroutine_on_crash(self):
    """Test that _run_guarded cancels the coroutine when the task crashes."""
    mock_client = MockClient()
    mock_session = MockClientSession()

    coro_was_cancelled = False

    with patch(
        "google.adk.tools.mcp_tool.session_context.ClientSession"
    ) as mock_session_class:
      mock_session_class.return_value = mock_session

      session_context = SessionContext(
          mock_client, timeout=5.0, sse_read_timeout=None
      )
      await session_context.start()

      async def slow_coro():
        nonlocal coro_was_cancelled
        try:
          await asyncio.sleep(100)
        except asyncio.CancelledError:
          coro_was_cancelled = True
          raise

      # Replace the background task with one that will fail soon
      session_context._close_event.set()
      await asyncio.sleep(0.05)

      crash_error = RuntimeError("connection lost")

      async def crashing_task():
        await asyncio.sleep(0.05)
        raise crash_error

      session_context._task = asyncio.create_task(crashing_task())

      with pytest.raises(ConnectionError):
        await session_context._run_guarded(slow_coro())

      assert coro_was_cancelled

  @pytest.mark.asyncio
  async def test_run_guarded_coroutine_raises(self):
    """Test _run_guarded propagates coroutine exceptions unwrapped."""
    mock_client = MockClient()
    mock_session = MockClientSession()

    with patch(
        "google.adk.tools.mcp_tool.session_context.ClientSession"
    ) as mock_session_class:
      mock_session_class.return_value = mock_session

      session_context = SessionContext(
          mock_client, timeout=5.0, sse_read_timeout=None
      )
      await session_context.start()

      original_error = ValueError("invalid arguments")

      async def failing_coro():
        raise original_error

      with pytest.raises(ValueError) as exc_info:
        await session_context._run_guarded(failing_coro())

      assert exc_info.value is original_error

      await session_context.close()

  @pytest.mark.asyncio
  async def test_run_guarded_no_task(self):
    """Test _run_guarded raises when task has not been started."""
    mock_client = MockClient()
    session_context = SessionContext(
        mock_client, timeout=5.0, sse_read_timeout=None
    )

    async def my_coro():
      return "hello"

    coro = my_coro()
    with pytest.raises(ConnectionError, match="not been started"):
      await session_context._run_guarded(coro)


# Mock MCP Tool from mcp.types
class MockMCPTool:
  """Mock MCP Tool for testing."""

  def __init__(
      self,
      name="test_tool",
      description="Test tool description",
      outputSchema=None,
      meta=None,
  ):
    self.name = name
    self.description = description
    self.meta = meta
    self.inputSchema = {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "First parameter"},
            "param2": {"type": "integer", "description": "Second parameter"},
        },
        "required": ["param1"],
    }
    self.outputSchema = outputSchema
