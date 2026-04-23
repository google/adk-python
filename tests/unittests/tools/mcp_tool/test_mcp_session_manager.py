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

  @patch("google.adk.tools.mcp_tool.mcp_session_manager.sse_client")
  def test_init_with_sse_custom_httpx_factory(self, mock_sse_client):
    """Test that sse_client is called with custom httpx_client_factory."""
    custom_httpx_factory = Mock()

    sse_params = SseConnectionParams(
        url="https://example.com/mcp",
        timeout=10.0,
        httpx_client_factory=custom_httpx_factory,
    )
    manager = MCPSessionManager(sse_params)

    manager._create_client()

    mock_sse_client.assert_called_once_with(
        url="https://example.com/mcp",
        headers=None,
        timeout=10.0,
        sse_read_timeout=300.0,
        httpx_client_factory=custom_httpx_factory,
    )

  @patch("google.adk.tools.mcp_tool.mcp_session_manager.sse_client")
  def test_init_with_sse_default_httpx_factory(self, mock_sse_client):
    """Test that sse_client is called with default httpx_client_factory."""
    sse_params = SseConnectionParams(
        url="https://example.com/mcp",
        timeout=10.0,
    )
    manager = MCPSessionManager(sse_params)

    manager._create_client()

    mock_sse_client.assert_called_once_with(
        url="https://example.com/mcp",
        headers=None,
        timeout=10.0,
        sse_read_timeout=300.0,
        httpx_client_factory=SseConnectionParams.model_fields[
            "httpx_client_factory"
        ].get_default(),
    )

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
  async def test_pickle_mcp_session_manager(self):
    """Verify that MCPSessionManager can be pickled and unpickled."""
    import pickle

    manager = MCPSessionManager(self.mock_stdio_connection_params)

    # Access the lock to ensure it's initialized
    lock = manager._session_lock
    assert isinstance(lock, asyncio.Lock)

    # Add a mock session to verify it's cleared on pickling
    manager._sessions["test"] = (Mock(), Mock(), asyncio.get_running_loop())

    # Pickle and unpickle
    pickled = pickle.dumps(manager)
    unpickled = pickle.loads(pickled)

    # Verify basics are restored
    assert unpickled._connection_params == manager._connection_params

    # Verify transient/unpicklable members are re-initialized or cleared
    assert unpickled._sessions == {}
    assert unpickled._session_lock_map == {}
    assert isinstance(unpickled._lock_map_lock, type(manager._lock_map_lock))
    assert unpickled._lock_map_lock is not manager._lock_map_lock
    assert unpickled._errlog == sys.stderr

    # Verify we can still get a lock in the new instance
    new_lock = unpickled._session_lock
    assert isinstance(new_lock, asyncio.Lock)
    assert new_lock is not lock


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


class TestMCPSessionManagerGetSessionContext:
  """Tests for MCPSessionManager._get_session_context.

  This is the lookup that allows McpTool to obtain the SessionContext
  for the current session and call `_run_guarded` on it.
  """

  def setup_method(self):
    """Set up a manager with stdio params."""
    self.params = StdioServerParameters(command="echo", args=[])
    self.manager = MCPSessionManager(self.params)

  def test_returns_none_when_no_session_exists(self):
    """With an empty pool, _get_session_context returns None."""
    assert self.manager._get_session_context() is None
    assert self.manager._get_session_context(headers={"x": "y"}) is None

  def test_returns_stored_session_context_for_stdio(self):
    """A stored SessionContext is returned for the stdio session key."""
    fake_ctx = MockSessionContext()
    # Stdio uses a constant session key, so headers are ignored.
    self.manager._session_contexts["stdio_session"] = fake_ctx

    assert self.manager._get_session_context() is fake_ctx
    assert self.manager._get_session_context(headers={"x": "y"}) is fake_ctx

  def test_returns_correct_session_context_per_header_set(self):
    """Different header sets produce different keys, so different contexts."""
    sse_params = SseConnectionParams(url="https://example.com/mcp")
    manager = MCPSessionManager(sse_params)

    ctx_a = MockSessionContext()
    ctx_b = MockSessionContext()
    key_a = manager._generate_session_key(
        manager._merge_headers({"x-token": "a"})
    )
    key_b = manager._generate_session_key(
        manager._merge_headers({"x-token": "b"})
    )
    manager._session_contexts[key_a] = ctx_a
    manager._session_contexts[key_b] = ctx_b

    assert manager._get_session_context(headers={"x-token": "a"}) is ctx_a
    assert manager._get_session_context(headers={"x-token": "b"}) is ctx_b
    # Unknown header set returns None.
    assert manager._get_session_context(headers={"x-token": "c"}) is None

  def test_session_contexts_dict_is_independent_of_sessions_tuple(self):
    """Backward-compat guard: _sessions remains the tuple shape.

    Downstream tests poke at `_sessions` directly using tuple unpacking
    (`session, exit_stack, loop = manager._sessions[key]`). This test
    ensures we did not switch to a dataclass that would break that.
    """
    mock_session = MockClientSession()
    mock_exit_stack = MockAsyncExitStack()
    mock_loop = Mock()
    mock_ctx = MockSessionContext()

    self.manager._sessions["stdio_session"] = (
        mock_session,
        mock_exit_stack,
        mock_loop,
    )
    self.manager._session_contexts["stdio_session"] = mock_ctx

    # Should be unpackable as a 3-tuple.
    session, exit_stack, loop = self.manager._sessions["stdio_session"]
    assert session is mock_session
    assert exit_stack is mock_exit_stack
    assert loop is mock_loop

    # And the SessionContext is reachable independently.
    assert self.manager._get_session_context() is mock_ctx

  def test_pickling_round_trip_clears_runtime_state(self):
    """__getstate__/__setstate__ should drop runtime SessionContext refs."""
    import pickle

    self.manager._session_contexts["stdio_session"] = MockSessionContext()

    restored = pickle.loads(pickle.dumps(self.manager))

    # Runtime state must not survive pickling.
    assert restored._session_contexts == {}
    assert restored._sessions == {}


class TestMCPSessionManagerCreateSessionFlagOff:
  """Pin down that create_session does NOT consult task aliveness when off.

  Existing callers broke under an earlier unconditional version of this
  fix because the new `_is_task_alive` check caused session re-creation
  paths to fire when the test mocks did not have a live `_task`. The
  check must be gated behind the feature flag.
  """

  def setup_method(self):
    from google.adk.features import FeatureName  # noqa: F401

    self.params = StdioServerParameters(command="echo", args=[])
    self.manager = MCPSessionManager(self.params)

  @pytest.mark.asyncio
  async def test_existing_session_reused_when_flag_off_even_with_dead_ctx(
      self,
  ):
    """A 'dead' SessionContext does not invalidate the session when off."""
    from google.adk.features import FeatureName
    from google.adk.features._feature_registry import temporary_feature_override

    # Pre-populate a healthy-looking session and a SessionContext whose
    # _task looks dead.
    healthy_session = MockClientSession()
    dead_ctx = MockSessionContext()
    dead_ctx._is_task_alive = False  # pretend the task died
    self.manager._sessions["stdio_session"] = (
        healthy_session,
        MockAsyncExitStack(),
        asyncio.get_running_loop(),
    )
    self.manager._session_contexts["stdio_session"] = dead_ctx

    # With flag OFF, the create_session must reuse the existing session
    # rather than tearing it down because of the dead _task.
    with temporary_feature_override(
        FeatureName._MCP_GRACEFUL_ERROR_HANDLING, False
    ):
      returned = await self.manager.create_session()

    assert returned is healthy_session

  @pytest.mark.asyncio
  async def test_existing_session_recreated_when_flag_on_with_dead_ctx(
      self,
  ):
    """And confirm: with flag ON, the dead _task DOES trigger re-creation."""
    from google.adk.features import FeatureName
    from google.adk.features._feature_registry import temporary_feature_override

    healthy_session = MockClientSession()
    dead_ctx = MockSessionContext()
    dead_ctx._is_task_alive = False
    # Mark the existing exit_stack so we can confirm the new one is different.
    old_exit_stack = MockAsyncExitStack()
    self.manager._sessions["stdio_session"] = (
        healthy_session,
        old_exit_stack,
        asyncio.get_running_loop(),
    )
    self.manager._session_contexts["stdio_session"] = dead_ctx

    # Patch the SessionContext used inside create_session so we don't
    # actually try to launch a real subprocess. Mirrors the patching
    # pattern used by `test_create_session_stdio_new`.
    new_session = MockClientSession()
    mock_exit_stack = MockAsyncExitStack()
    mock_session_ctx = MockSessionContext(session=new_session)

    with temporary_feature_override(
        FeatureName._MCP_GRACEFUL_ERROR_HANDLING, True
    ):
      with patch("google.adk.tools.mcp_tool.mcp_session_manager.stdio_client"):
        with patch(
            "google.adk.tools.mcp_tool.mcp_session_manager.AsyncExitStack"
        ) as mock_exit_stack_class:
          with patch(
              "google.adk.tools.mcp_tool.mcp_session_manager.SessionContext"
          ) as mock_session_context_class:
            mock_exit_stack_class.return_value = mock_exit_stack
            mock_session_context_class.return_value = mock_session_ctx
            mock_exit_stack.enter_async_context.return_value = new_session

            returned = await self.manager.create_session()

    assert returned is new_session
    # The original 'healthy_session' was torn down because dead_ctx
    # told us the task was gone.
    assert returned is not healthy_session


class TestMCPGracefulErrorHandlingFlagContract:
  """Pin down the public contract that GE will rely on to enable the fix.

  GE will flip this fix on by setting an environment variable in their
  deployment config (per Sasha's confirmation: "environment variable, GE
  team is responsible for setting it"). The deployment expects:

    * `ADK_ENABLE_MCP_GRACEFUL_ERROR_HANDLING=1`  enables the fix
    * absence of the variable                      keeps it disabled
    * `ADK_DISABLE_MCP_GRACEFUL_ERROR_HANDLING=1` is the kill switch

  These tests are guards: if anyone refactors the feature-flag framework
  in a way that changes how the env var is read (renames it, caches the
  value at import time, requires a binary push, etc.), these tests fail
  loudly so we don't silently break GE's rollout.
  """

  def test_default_state_is_off_so_cl_is_a_noop(self):
    """The CL must be a no-op until GE explicitly enables it."""
    import os

    from google.adk.features import FeatureName
    from google.adk.features import is_feature_enabled

    enable = "ADK_ENABLE_MCP_GRACEFUL_ERROR_HANDLING"
    disable = "ADK_DISABLE_MCP_GRACEFUL_ERROR_HANDLING"
    saved = {k: os.environ.pop(k) for k in (enable, disable) if k in os.environ}
    try:
      assert (
          is_feature_enabled(FeatureName._MCP_GRACEFUL_ERROR_HANDLING) is False
      )
    finally:
      os.environ.update(saved)

  def test_env_var_enable_flips_flag_on_at_runtime(self):
    """The env var GE will set must turn the fix on without a rebuild."""
    import os

    from google.adk.features import FeatureName
    from google.adk.features import is_feature_enabled

    enable = "ADK_ENABLE_MCP_GRACEFUL_ERROR_HANDLING"
    saved = os.environ.pop(enable, None)
    try:
      os.environ[enable] = "1"
      assert (
          is_feature_enabled(FeatureName._MCP_GRACEFUL_ERROR_HANDLING) is True
      )
      # And once it's removed, we revert. Confirms the value is read
      # live from os.environ on every call (no caching, no binary push).
      del os.environ[enable]
      assert (
          is_feature_enabled(FeatureName._MCP_GRACEFUL_ERROR_HANDLING) is False
      )
    finally:
      if saved is not None:
        os.environ[enable] = saved

  def test_env_var_disable_acts_as_kill_switch(self):
    """The disable env var lets consumers turn off without a rebuild."""
    import os

    from google.adk.features import FeatureName
    from google.adk.features import is_feature_enabled
    from google.adk.features._feature_registry import temporary_feature_override

    disable = "ADK_DISABLE_MCP_GRACEFUL_ERROR_HANDLING"
    enable = "ADK_ENABLE_MCP_GRACEFUL_ERROR_HANDLING"
    saved_disable = os.environ.pop(disable, None)
    saved_enable = os.environ.pop(enable, None)
    try:
      # If a future default flip ever turns this on by default, the
      # disable env var should still let consumers turn it back off
      # without a rebuild.
      os.environ[disable] = "1"
      assert (
          is_feature_enabled(FeatureName._MCP_GRACEFUL_ERROR_HANDLING) is False
      )
      # And confirm: a programmatic override takes precedence over the
      # disable env var (priority order documented in _feature_registry).
      with temporary_feature_override(
          FeatureName._MCP_GRACEFUL_ERROR_HANDLING, True
      ):
        assert (
            is_feature_enabled(FeatureName._MCP_GRACEFUL_ERROR_HANDLING) is True
        )
    finally:
      if saved_disable is not None:
        os.environ[disable] = saved_disable
      if saved_enable is not None:
        os.environ[enable] = saved_enable
