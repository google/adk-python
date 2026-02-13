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

import asyncio
from datetime import timedelta
import hashlib
from io import StringIO
import json
import sys
from unittest.mock import ANY
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

from google.adk.platform import thread as platform_thread
from google.adk.tools.mcp_tool.mcp_session_manager import MCPSessionManager
from google.adk.tools.mcp_tool.mcp_session_manager import retry_on_errors
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
from mcp import StdioServerParameters
import pytest


class MockClientSession:
  """Mock ClientSession for testing."""

  def __init__(self):
    self._read_stream = Mock()
    self._write_stream = Mock()
    self._read_stream._closed = False
    self._write_stream._closed = False
    self.initialize = AsyncMock()


class MockAsyncExitStack:
  """Mock AsyncExitStack for testing."""

  def __init__(self):
    self.aclose = AsyncMock()
    self.enter_async_context = AsyncMock()

  async def __aenter__(self):
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    pass


class MockSessionContext:
  """Mock SessionContext for testing."""

  def __init__(self, session=None):
    """Initialize MockSessionContext.

    Args:
        session: The mock session to return from __aenter__ and session property.
    """
    self._session = session
    self._aenter_mock = AsyncMock(return_value=session)
    self._aexit_mock = AsyncMock(return_value=False)

  @property
  def session(self):
    """Get the mock session."""
    return self._session

  async def __aenter__(self):
    """Enter the async context manager."""
    return await self._aenter_mock()

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    """Exit the async context manager."""
    return await self._aexit_mock(exc_type, exc_val, exc_tb)


class TestMCPSessionManager:
  """Test suite for MCPSessionManager class."""

  def setup_method(self):
    """Set up test fixtures."""
    self.mock_stdio_params = StdioServerParameters(
        command="test_command", args=[]
    )
    self.mock_stdio_connection_params = StdioConnectionParams(
        server_params=self.mock_stdio_params, timeout=5.0
    )

  def test_init_with_stdio_server_parameters(self):
    """Test initialization with StdioServerParameters (deprecated)."""
    with patch(
        "google.adk.tools.mcp_tool.mcp_session_manager.logger"
    ) as mock_logger:
      manager = MCPSessionManager(self.mock_stdio_params)

      # Should log deprecation warning
      mock_logger.warning.assert_called_once()
      assert "StdioServerParameters is not recommended" in str(
          mock_logger.warning.call_args
      )

      # Should convert to StdioConnectionParams
      assert isinstance(manager._connection_params, StdioConnectionParams)
      assert manager._connection_params.server_params == self.mock_stdio_params
      assert manager._connection_params.timeout == 5

  def test_init_with_stdio_connection_params(self):
    """Test initialization with StdioConnectionParams."""
    manager = MCPSessionManager(self.mock_stdio_connection_params)

    assert manager._connection_params == self.mock_stdio_connection_params
    assert manager._errlog == sys.stderr
    assert manager._sessions == {}

  def test_init_with_sse_connection_params(self):
    """Test initialization with SseConnectionParams."""
    sse_params = SseConnectionParams(
        url="https://example.com/mcp",
        headers={"Authorization": "Bearer token"},
        timeout=10.0,
    )
    manager = MCPSessionManager(sse_params)

    assert manager._connection_params == sse_params

  def test_init_with_streamable_http_params(self):
    """Test initialization with StreamableHTTPConnectionParams."""
    http_params = StreamableHTTPConnectionParams(
        url="https://example.com/mcp", timeout=15.0
    )
    manager = MCPSessionManager(http_params)

    assert manager._connection_params == http_params

  @patch("google.adk.tools.mcp_tool.mcp_session_manager.streamablehttp_client")
  def test_init_with_streamable_http_custom_httpx_factory(
      self, mock_streamablehttp_client
  ):
    """Test that streamablehttp_client is called with custom httpx_client_factory."""
    custom_httpx_factory = Mock()

    http_params = StreamableHTTPConnectionParams(
        url="https://example.com/mcp",
        timeout=15.0,
        httpx_client_factory=custom_httpx_factory,
    )
    manager = MCPSessionManager(http_params)

    manager._create_client()

    mock_streamablehttp_client.assert_called_once_with(
        url="https://example.com/mcp",
        headers=None,
        timeout=timedelta(seconds=15.0),
        sse_read_timeout=timedelta(seconds=300.0),
        terminate_on_close=True,
        httpx_client_factory=custom_httpx_factory,
    )

  @patch("google.adk.tools.mcp_tool.mcp_session_manager.streamablehttp_client")
  def test_init_with_streamable_http_default_httpx_factory(
      self, mock_streamablehttp_client
  ):
    """Test that streamablehttp_client is called with default httpx_client_factory."""
    http_params = StreamableHTTPConnectionParams(
        url="https://example.com/mcp", timeout=15.0
    )
    manager = MCPSessionManager(http_params)

    manager._create_client()

    mock_streamablehttp_client.assert_called_once_with(
        url="https://example.com/mcp",
        headers=None,
        timeout=timedelta(seconds=15.0),
        sse_read_timeout=timedelta(seconds=300.0),
        terminate_on_close=True,
        httpx_client_factory=StreamableHTTPConnectionParams.model_fields[
            "httpx_client_factory"
        ].get_default(),
    )

  def test_generate_session_key_stdio(self):
    """Test session key generation for stdio connections."""
    manager = MCPSessionManager(self.mock_stdio_connection_params)

    # For stdio, headers should be ignored and return constant key
    key1 = manager._generate_session_key({"Authorization": "Bearer token"})
    key2 = manager._generate_session_key(None)

    assert key1 == "stdio_session"
    assert key2 == "stdio_session"
    assert key1 == key2

  def test_generate_session_key_sse(self):
    """Test session key generation for SSE connections."""
    sse_params = SseConnectionParams(url="https://example.com/mcp")
    manager = MCPSessionManager(sse_params)

    headers1 = {"Authorization": "Bearer token1"}
    headers2 = {"Authorization": "Bearer token2"}

    key1 = manager._generate_session_key(headers1)
    key2 = manager._generate_session_key(headers2)
    key3 = manager._generate_session_key(headers1)

    # Different headers should generate different keys
    assert key1 != key2
    # Same headers should generate same key
    assert key1 == key3

    # Should be deterministic hash
    headers_json = json.dumps(headers1, sort_keys=True)
    expected_hash = hashlib.md5(headers_json.encode()).hexdigest()
    assert key1 == f"session_{expected_hash}"

  def test_merge_headers_stdio(self):
    """Test header merging for stdio connections."""
    manager = MCPSessionManager(self.mock_stdio_connection_params)

    # Stdio connections don't support headers
    headers = manager._merge_headers({"Authorization": "Bearer token"})
    assert headers is None

  def test_merge_headers_sse(self):
    """Test header merging for SSE connections."""
    base_headers = {"Content-Type": "application/json"}
    sse_params = SseConnectionParams(
        url="https://example.com/mcp", headers=base_headers
    )
    manager = MCPSessionManager(sse_params)

    # With additional headers
    additional = {"Authorization": "Bearer token"}
    merged = manager._merge_headers(additional)

    expected = {
        "Content-Type": "application/json",
        "Authorization": "Bearer token",
    }
    assert merged == expected

  def test_is_session_disconnected(self):
    """Test session disconnection detection."""
    manager = MCPSessionManager(self.mock_stdio_connection_params)

    # Create mock session
    session = MockClientSession()

    # Not disconnected
    assert not manager._is_session_disconnected(session)

    # Disconnected - read stream closed
    session._read_stream._closed = True
    assert manager._is_session_disconnected(session)

  @pytest.mark.asyncio
  async def test_create_session_stdio_new(self):
    """Test creating a new stdio session."""
    manager = MCPSessionManager(self.mock_stdio_connection_params)

    mock_exit_stack = MockAsyncExitStack()

    with patch(
        "google.adk.tools.mcp_tool.mcp_session_manager.stdio_client"
    ) as mock_stdio:
      with patch(
          "google.adk.tools.mcp_tool.mcp_session_manager.AsyncExitStack"
      ) as mock_exit_stack_class:
        with patch(
            "google.adk.tools.mcp_tool.mcp_session_manager.SessionContext"
        ) as mock_session_context_class:

          # Setup mocks
          mock_exit_stack_class.return_value = mock_exit_stack
          mock_stdio.return_value = AsyncMock()

          # Mock SessionContext using MockSessionContext
          # Create a mock session that will be returned by SessionContext
          mock_session = AsyncMock()
          mock_session_context = MockSessionContext(session=mock_session)
          mock_session_context_class.return_value = mock_session_context
          mock_exit_stack.enter_async_context.return_value = mock_session

          # Create session
          session = await manager.create_session()

          # Verify session creation
          assert session == mock_session
          assert len(manager._sessions) == 1
          assert "stdio_session" in manager._sessions
          session_data = manager._sessions["stdio_session"]
          assert len(session_data) == 3
          assert session_data[0] == mock_session
          assert session_data[2] == asyncio.get_running_loop()

          # Verify SessionContext was created
          mock_session_context_class.assert_called_once()
          # Verify enter_async_context was called (which internally calls __aenter__)
          mock_exit_stack.enter_async_context.assert_called_once()

  @pytest.mark.asyncio
  async def test_create_session_reuse_existing(self):
    """Test reusing an existing connected session."""
    manager = MCPSessionManager(self.mock_stdio_connection_params)

    # Create mock existing session
    existing_session = MockClientSession()
    existing_exit_stack = MockAsyncExitStack()
    manager._sessions["stdio_session"] = (
        existing_session,
        existing_exit_stack,
        asyncio.get_running_loop(),
    )

    # Session is connected
    existing_session._read_stream._closed = False
    existing_session._write_stream._closed = False

    session = await manager.create_session()

    # Should reuse existing session
    assert session == existing_session
    assert len(manager._sessions) == 1

    # Should not create new session
    existing_session.initialize.assert_not_called()

  @pytest.mark.asyncio
  @patch("google.adk.tools.mcp_tool.mcp_session_manager.stdio_client")
  @patch("google.adk.tools.mcp_tool.mcp_session_manager.AsyncExitStack")
  @patch("google.adk.tools.mcp_tool.mcp_session_manager.SessionContext")
  async def test_create_session_timeout(
      self, mock_session_context_class, mock_exit_stack_class, mock_stdio
  ):
    """Test session creation timeout."""
    manager = MCPSessionManager(self.mock_stdio_connection_params)

    mock_exit_stack = MockAsyncExitStack()

    mock_exit_stack_class.return_value = mock_exit_stack
    mock_stdio.return_value = AsyncMock()

    # Mock SessionContext
    mock_session_context = AsyncMock()
    mock_session_context.__aenter__ = AsyncMock(
        return_value=MockClientSession()
    )
    mock_session_context.__aexit__ = AsyncMock(return_value=False)
    mock_session_context_class.return_value = mock_session_context

    # Mock enter_async_context to raise TimeoutError (simulating asyncio.wait_for timeout)
    mock_exit_stack.enter_async_context = AsyncMock(
        side_effect=asyncio.TimeoutError("Test timeout")
    )

    # Expect ConnectionError due to timeout
    with pytest.raises(ConnectionError, match="Failed to create MCP session"):
      await manager.create_session()

    # Verify SessionContext was created
    mock_session_context_class.assert_called_once()
    # Verify session was not added to pool
    assert not manager._sessions
    # Verify cleanup was called
    mock_exit_stack.aclose.assert_called_once()

  @pytest.mark.asyncio
  async def test_close_success(self):
    """Test successful cleanup of all sessions."""
    manager = MCPSessionManager(self.mock_stdio_connection_params)

    # Add mock sessions
    session1 = MockClientSession()
    exit_stack1 = MockAsyncExitStack()
    session2 = MockClientSession()
    exit_stack2 = MockAsyncExitStack()

    manager._sessions["session1"] = (
        session1,
        exit_stack1,
        asyncio.get_running_loop(),
    )
    manager._sessions["session2"] = (
        session2,
        exit_stack2,
        asyncio.get_running_loop(),
    )

    await manager.close()

    # All sessions should be closed
    exit_stack1.aclose.assert_called_once()
    exit_stack2.aclose.assert_called_once()
    assert len(manager._sessions) == 0

  @pytest.mark.asyncio
  @patch("google.adk.tools.mcp_tool.mcp_session_manager.logger")
  async def test_close_with_errors(self, mock_logger):
    """Test cleanup when some sessions fail to close."""
    manager = MCPSessionManager(self.mock_stdio_connection_params)

    # Add mock sessions
    session1 = MockClientSession()
    exit_stack1 = MockAsyncExitStack()
    exit_stack1.aclose.side_effect = Exception("Close error 1")

    session2 = MockClientSession()
    exit_stack2 = MockAsyncExitStack()

    manager._sessions["session1"] = (
        session1,
        exit_stack1,
        asyncio.get_running_loop(),
    )
    manager._sessions["session2"] = (
        session2,
        exit_stack2,
        asyncio.get_running_loop(),
    )

    # Should not raise exception
    await manager.close()

    # Good session should still be closed
    exit_stack2.aclose.assert_called_once()
    assert len(manager._sessions) == 0

    # Error should be logged via logger.warning
    mock_logger.warning.assert_called_once()
    args, kwargs = mock_logger.warning.call_args
    assert "Error during session cleanup for session1: Close error 1" in args[0]
    assert kwargs.get("exc_info")

  @pytest.mark.asyncio
  @patch("google.adk.tools.mcp_tool.mcp_session_manager.stdio_client")
  @patch("google.adk.tools.mcp_tool.mcp_session_manager.AsyncExitStack")
  @patch("google.adk.tools.mcp_tool.mcp_session_manager.SessionContext")
  async def test_create_and_close_session_in_different_tasks(
      self, mock_session_context_class, mock_exit_stack_class, mock_stdio
  ):
    """Test creating and closing a session in different tasks."""
    manager = MCPSessionManager(self.mock_stdio_connection_params)

    mock_exit_stack_class.return_value = MockAsyncExitStack()
    mock_stdio.return_value = AsyncMock()

    # Mock SessionContext
    mock_session_context = AsyncMock()
    mock_session_context.__aenter__ = AsyncMock(
        return_value=MockClientSession()
    )
    mock_session_context.__aexit__ = AsyncMock(return_value=False)
    mock_session_context_class.return_value = mock_session_context

    # Create session in a new task
    await asyncio.create_task(manager.create_session())

    # Close session in another task
    await asyncio.create_task(manager.close())

    # Verify session was closed
    assert not manager._sessions

  @pytest.mark.asyncio
  async def test_session_lock_different_loops(self):
    """Verify that _session_lock returns different locks for different loops."""

    manager = MCPSessionManager(self.mock_stdio_connection_params)

    # Access in current loop
    lock1 = manager._session_lock
    assert isinstance(lock1, asyncio.Lock)

    # Access in a different loop (in a separate thread)
    lock_container = []

    def run_in_thread():
      loop2 = asyncio.new_event_loop()
      asyncio.set_event_loop(loop2)
      try:

        async def get_lock():
          return manager._session_lock

        lock_container.append(loop2.run_until_complete(get_lock()))
      finally:
        loop2.close()

    thread = platform_thread.create_thread(target=run_in_thread)
    thread.start()
    thread.join()

    assert lock_container
    lock2 = lock_container[0]
    assert isinstance(lock2, asyncio.Lock)
    assert lock1 is not lock2

  @pytest.mark.asyncio
  async def test_cleanup_session_cross_loop(self):
    """Verify that _cleanup_session uses run_coroutine_threadsafe for different loops."""
    manager = MCPSessionManager(self.mock_stdio_connection_params)
    mock_exit_stack = MockAsyncExitStack()

    # Create a dummy loop that is "running" in another thread
    loop2 = asyncio.new_event_loop()
    try:
      with patch(
          "google.adk.tools.mcp_tool.mcp_session_manager.asyncio.run_coroutine_threadsafe"
      ) as mock_run_threadsafe:
        with patch(
            "google.adk.tools.mcp_tool.mcp_session_manager.logger"
        ) as mock_logger:
          # We need to mock the return value of run_coroutine_threadsafe to be a future
          mock_future = Mock()
          mock_run_threadsafe.return_value = mock_future

          await manager._cleanup_session("test_session", mock_exit_stack, loop2)

          # Verify run_coroutine_threadsafe was called
          # ANY is used because a new coroutine object is created each time
          mock_run_threadsafe.assert_called_once_with(ANY, loop2)

          mock_logger.info.assert_any_call(
              "Scheduling cleanup of session test_session on its original"
              " event loop."
          )
          mock_future.add_done_callback.assert_called_once()
    finally:
      loop2.close()

  @pytest.mark.asyncio
  async def test_create_session_cleans_up_without_aclose_if_loop_is_different(
      self,
  ):
    """Verify that sessions from different loops are cleaned up without calling aclose()."""
    manager = MCPSessionManager(self.mock_stdio_connection_params)

    # 1. Simulate a session created in a "different" loop
    mock_session = MockClientSession()
    mock_exit_stack = MockAsyncExitStack()
    # Use a dummy object as a different loop
    different_loop = Mock(spec=asyncio.AbstractEventLoop)

    manager._sessions["stdio_session"] = (
        mock_session,
        mock_exit_stack,
        different_loop,
    )

    # 2. Mock creation of a new session
    # We need to mock create_client, wait_for, and SessionContext
    with patch.object(manager, "_create_client") as mock_create_client:
      with patch(
          "google.adk.tools.mcp_tool.mcp_session_manager.asyncio.wait_for"
      ) as mock_wait_for:
        with patch(
            "google.adk.tools.mcp_tool.mcp_session_manager.SessionContext"
        ) as mock_session_context_class:
          # Setup mocks for new session creation
          mock_create_client.return_value = AsyncMock()
          new_session = MockClientSession()
          mock_wait_for.return_value = new_session
          mock_session_context_class.return_value = AsyncMock()

          # 3. Call create_session
          session = await manager.create_session()

          # 4. Verify results
          assert session == new_session
          assert len(manager._sessions) == 1
          # Verify that old exit_stack.aclose was NOT called since loop was different
          mock_exit_stack.aclose.assert_not_called()

  @pytest.mark.asyncio
  async def test_close_skips_aclose_for_different_loop_sessions(self):
    """Verify that close() skips aclose() for sessions from different loops."""
    manager = MCPSessionManager(self.mock_stdio_connection_params)

    # Add one session from same loop and one from different loop
    current_loop = asyncio.get_running_loop()
    different_loop = Mock(spec=asyncio.AbstractEventLoop)

    session1 = MockClientSession()
    exit_stack1 = MockAsyncExitStack()
    manager._sessions["session1"] = (session1, exit_stack1, current_loop)

    session2 = MockClientSession()
    exit_stack2 = MockAsyncExitStack()
    manager._sessions["session2"] = (session2, exit_stack2, different_loop)

    await manager.close()

    # exit_stack1 should be closed, exit_stack2 should be skipped
    exit_stack1.aclose.assert_called_once()
    exit_stack2.aclose.assert_not_called()
    assert len(manager._sessions) == 0


@pytest.mark.asyncio
async def test_retry_on_errors_decorator():
  """Test the retry_on_errors decorator."""

  call_count = 0

  @retry_on_errors
  async def mock_function(self):
    nonlocal call_count
    call_count += 1
    if call_count == 1:
      raise ConnectionError("Resource closed")
    return "success"

  mock_self = Mock()
  result = await mock_function(mock_self)

  assert result == "success"
  assert call_count == 2  # First call fails, second succeeds


@pytest.mark.asyncio
async def test_retry_on_errors_decorator_does_not_retry_cancelled_error():
  """Test the retry_on_errors decorator does not retry cancellation."""

  call_count = 0

  @retry_on_errors
  async def mock_function(self):
    nonlocal call_count
    call_count += 1
    raise asyncio.CancelledError()

  mock_self = Mock()
  with pytest.raises(asyncio.CancelledError):
    await mock_function(mock_self)

  assert call_count == 1


@pytest.mark.asyncio
async def test_retry_on_errors_decorator_does_not_retry_when_task_is_cancelling():
  """Test the retry_on_errors decorator does not retry when cancelling."""

  call_count = 0

  @retry_on_errors
  async def mock_function(self):
    nonlocal call_count
    call_count += 1
    raise ConnectionError("Resource closed")

  class _MockTask:

    def cancelling(self):
      return 1

  mock_self = Mock()
  with patch.object(asyncio, "current_task", return_value=_MockTask()):
    with pytest.raises(ConnectionError):
      await mock_function(mock_self)

  assert call_count == 1


@pytest.mark.asyncio
async def test_retry_on_errors_decorator_does_not_retry_exception_from_cancel():
  """Test the retry_on_errors decorator does not retry exceptions on cancel."""

  call_count = 0

  @retry_on_errors
  async def mock_function(self):
    nonlocal call_count
    call_count += 1
    try:
      raise asyncio.CancelledError()
    except asyncio.CancelledError:
      raise ConnectionError("Resource closed")

  mock_self = Mock()
  with pytest.raises(ConnectionError):
    await mock_function(mock_self)

  assert call_count == 1
