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

import asyncio
from datetime import timedelta
import hashlib
from io import StringIO
import json
import sys
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

# Skip all tests in this module if Python version is less than 3.10
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10), reason="MCP tool requires Python 3.10+"
)

# Import dependencies with version checking
try:
  from google.adk.tools.mcp_tool.mcp_session_manager import MCPSessionManager
  from google.adk.tools.mcp_tool.mcp_session_manager import retry_on_closed_resource
  from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams
  from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
  from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
except ImportError as e:
  if sys.version_info < (3, 10):
    # Create dummy classes to prevent NameError during test collection
    # Tests will be skipped anyway due to pytestmark
    class DummyClass:
      pass

    MCPSessionManager = DummyClass
    retry_on_closed_resource = lambda x: x
    SseConnectionParams = DummyClass
    StdioConnectionParams = DummyClass
    StreamableHTTPConnectionParams = DummyClass
  else:
    raise e

# Import real MCP classes
try:
  from mcp import StdioServerParameters
except ImportError:
  # Create a mock if MCP is not available
  class StdioServerParameters:

    def __init__(self, command="test_command", args=None):
      self.command = command
      self.args = args or []


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

    mock_session = MockClientSession()
    mock_exit_stack = MockAsyncExitStack()

    with patch(
        "google.adk.tools.mcp_tool.mcp_session_manager.stdio_client"
    ) as mock_stdio:
      with patch(
          "google.adk.tools.mcp_tool.mcp_session_manager.AsyncExitStack"
      ) as mock_exit_stack_class:
        with patch(
            "google.adk.tools.mcp_tool.mcp_session_manager.ClientSession"
        ) as mock_session_class:

          # Setup mocks
          mock_exit_stack_class.return_value = mock_exit_stack
          mock_stdio.return_value = AsyncMock()
          mock_exit_stack.enter_async_context.side_effect = [
              ("read", "write"),  # First call returns transports
              mock_session,  # Second call returns session
          ]
          mock_session_class.return_value = mock_session

          # Create session
          session = await manager.create_session()

          # Verify session creation
          assert session == mock_session
          assert len(manager._sessions) == 1
          assert "stdio_session" in manager._sessions

          # Verify session was initialized
          mock_session.initialize.assert_called_once()

  @pytest.mark.asyncio
  async def test_create_session_reuse_existing(self):
    """Test reusing an existing connected session."""
    manager = MCPSessionManager(self.mock_stdio_connection_params)

    # Create mock existing session
    existing_session = MockClientSession()
    existing_exit_stack = MockAsyncExitStack()
    manager._sessions["stdio_session"] = (existing_session, existing_exit_stack)

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
  @patch("google.adk.tools.mcp_tool.mcp_session_manager.ClientSession")
  async def test_create_session_timeout(
      self, mock_session_class, mock_exit_stack_class, mock_stdio
  ):
    """Test session creation timeout."""
    manager = MCPSessionManager(self.mock_stdio_connection_params)

    mock_session = MockClientSession()
    mock_exit_stack = MockAsyncExitStack()

    mock_exit_stack_class.return_value = mock_exit_stack
    mock_stdio.return_value = AsyncMock()
    mock_exit_stack.enter_async_context.side_effect = [
        ("read", "write"),  # First call returns transports
        mock_session,  # Second call returns session
    ]
    mock_session_class.return_value = mock_session

    # Simulate timeout during session initialization
    mock_session.initialize.side_effect = asyncio.TimeoutError("Test timeout")

    # Expect ConnectionError due to timeout
    with pytest.raises(ConnectionError, match="Failed to create MCP session"):
      await manager.create_session()

    # Verify ClientSession called with timeout
    mock_session_class.assert_called_with(
        "read",
        "write",
        read_timeout_seconds=timedelta(
            seconds=manager._connection_params.timeout
        ),
    )
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

    manager._sessions["session1"] = (session1, exit_stack1)
    manager._sessions["session2"] = (session2, exit_stack2)

    await manager.close()

    # All sessions should be closed
    exit_stack1.aclose.assert_called_once()
    exit_stack2.aclose.assert_called_once()
    assert len(manager._sessions) == 0

  @pytest.mark.asyncio
  async def test_close_with_errors(self):
    """Test cleanup when some sessions fail to close."""
    manager = MCPSessionManager(self.mock_stdio_connection_params)

    # Add mock sessions
    session1 = MockClientSession()
    exit_stack1 = MockAsyncExitStack()
    exit_stack1.aclose.side_effect = Exception("Close error 1")

    session2 = MockClientSession()
    exit_stack2 = MockAsyncExitStack()

    manager._sessions["session1"] = (session1, exit_stack1)
    manager._sessions["session2"] = (session2, exit_stack2)

    custom_errlog = StringIO()
    manager._errlog = custom_errlog

    # Should not raise exception
    await manager.close()

    # Good session should still be closed
    exit_stack2.aclose.assert_called_once()
    assert len(manager._sessions) == 0

    # Error should be logged
    error_output = custom_errlog.getvalue()
    assert "Warning: Error during MCP session cleanup" in error_output
    assert "Close error 1" in error_output


def test_retry_on_closed_resource_decorator():
  """Test the retry_on_closed_resource decorator."""

  call_count = 0

  @retry_on_closed_resource
  async def mock_function(self):
    nonlocal call_count
    call_count += 1
    if call_count == 1:
      import anyio

      raise anyio.ClosedResourceError("Resource closed")
    return "success"

  @pytest.mark.asyncio
  async def test_retry():
    nonlocal call_count
    call_count = 0

    mock_self = Mock()
    result = await mock_function(mock_self)

    assert result == "success"
    assert call_count == 2  # First call fails, second succeeds

  # Run the test
  import asyncio

  asyncio.run(test_retry())
