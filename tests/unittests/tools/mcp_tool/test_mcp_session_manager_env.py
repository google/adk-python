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

"""Unit tests for MCPSessionManager environment variable functionality."""

from types import MappingProxyType
from typing import Any
from typing import Dict
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.readonly_context import ReadonlyContext
import pytest


# Mock MCP imports to avoid dependency issues in tests
@pytest.fixture(autouse=True)
def mock_mcp_imports():
  """Mock MCP imports to avoid import errors in testing."""
  from unittest.mock import MagicMock

  with patch.dict(
      'sys.modules',
      {
          'mcp': MagicMock(),
          'mcp.types': MagicMock(),
          'mcp.client.sse': MagicMock(),
          'mcp.client.stdio': MagicMock(),
          'mcp.client.streamable_http': MagicMock(),
      },
  ):
    # Mock the specific classes we need
    mock_stdio_params = MagicMock()
    mock_client_session = MagicMock()

    with (
        patch(
            'google.adk.tools.mcp_tool.mcp_session_manager.StdioServerParameters',
            mock_stdio_params,
        ),
        patch(
            'google.adk.tools.mcp_tool.mcp_session_manager.ClientSession',
            mock_client_session,
        ),
    ):
      yield


# Import after mocking to avoid MCP dependency issues
from google.adk.tools.mcp_tool.mcp_session_manager import MCPSessionManager, ContextToEnvMapperCallback


@pytest.fixture
def mock_stdio_params():
  """Create a mock StdioServerParameters instance."""
  from unittest.mock import MagicMock

  mock_params = MagicMock()
  mock_params.command = 'npx'
  mock_params.args = ['-y', '@modelcontextprotocol/server-filesystem']
  mock_params.env = {'EXISTING_VAR': 'existing_value'}
  mock_params.cwd = None
  mock_params.encoding = None
  mock_params.encoding_error_handler = None
  return mock_params


@pytest.fixture
def mock_sse_params():
  """Create a mock SseServerParams instance."""
  from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams

  return SseServerParams(
      url='http://localhost:3000/sse',
      headers={'Authorization': 'Bearer token'},
      timeout=5,
      sse_read_timeout=300,
  )


@pytest.fixture
def sample_context_to_env_mapper_callback():
  """Create a sample context to env mapper callback."""

  def env_callback(state: Dict[str, Any]) -> Dict[str, str]:
    env_vars = {}
    if 'api_key' in state:
      env_vars['API_KEY'] = state['api_key']
    if 'workspace_path' in state:
      env_vars['WORKSPACE_PATH'] = state['workspace_path']
    return env_vars

  return env_callback


@pytest.fixture
def mock_readonly_context():
  """Create a mock ReadonlyContext with sample state."""
  context = MagicMock(spec=ReadonlyContext)
  context.state = MappingProxyType({
      'api_key': 'test_api_key_123',
      'workspace_path': '/home/user/workspace',
      'other_data': 'some_value',
  })
  return context


class TestMCPSessionManagerEnv:
  """Test environment variable functionality in MCPSessionManager."""

  def test_init_with_env_callback(
      self, mock_stdio_params, sample_context_to_env_mapper_callback
  ):
    """Test MCPSessionManager initialization with context to env mapper callback."""
    session_manager = MCPSessionManager(
        connection_params=mock_stdio_params,
        context_to_env_mapper_callback=sample_context_to_env_mapper_callback,
    )

    assert (
        session_manager._context_to_env_mapper_callback
        == sample_context_to_env_mapper_callback
    )
    assert session_manager._connection_params == mock_stdio_params

  def test_init_without_env_callback(self, mock_stdio_params):
    """Test MCPSessionManager initialization without environment callback."""
    session_manager = MCPSessionManager(connection_params=mock_stdio_params)

    assert session_manager._context_to_env_mapper_callback is None
    assert session_manager._connection_params == mock_stdio_params

  def test_extract_env_from_context_with_callback(
      self,
      mock_stdio_params,
      sample_context_to_env_mapper_callback,
      mock_readonly_context,
  ):
    """Test environment extraction from context with callback."""
    session_manager = MCPSessionManager(
        connection_params=mock_stdio_params,
        context_to_env_mapper_callback=sample_context_to_env_mapper_callback,
    )

    env_vars = session_manager._extract_env_from_context(mock_readonly_context)

    expected_env = {
        'API_KEY': 'test_api_key_123',
        'WORKSPACE_PATH': '/home/user/workspace',
    }
    assert env_vars == expected_env

  def test_extract_env_from_context_without_callback(
      self, mock_stdio_params, mock_readonly_context
  ):
    """Test environment extraction without callback returns empty dict."""
    session_manager = MCPSessionManager(connection_params=mock_stdio_params)

    env_vars = session_manager._extract_env_from_context(mock_readonly_context)

    assert env_vars == {}

  def test_extract_env_from_context_no_context(
      self, mock_stdio_params, sample_context_to_env_mapper_callback
  ):
    """Test environment extraction with no context returns empty dict."""
    session_manager = MCPSessionManager(
        connection_params=mock_stdio_params,
        context_to_env_mapper_callback=sample_context_to_env_mapper_callback,
    )

    env_vars = session_manager._extract_env_from_context(None)

    assert env_vars == {}

  def test_extract_env_from_context_callback_exception(
      self, mock_stdio_params, mock_readonly_context
  ):
    """Test environment extraction handles callback exceptions gracefully."""

    def failing_callback(state: Dict[str, Any]) -> Dict[str, str]:
      raise ValueError('Callback failed')

    session_manager = MCPSessionManager(
        connection_params=mock_stdio_params,
        context_to_env_mapper_callback=failing_callback,
    )

    env_vars = session_manager._extract_env_from_context(mock_readonly_context)

    assert env_vars == {}

  def test_inject_env_vars_stdio_params(self, mock_stdio_params):
    """Test environment variable injection for StdioServerParameters."""
    # Mock the isinstance check to return True for our mock
    mock_stdio_params.__class__.__name__ = 'StdioServerParameters'

    session_manager = MCPSessionManager(connection_params=mock_stdio_params)

    new_env_vars = {'API_KEY': 'test_key', 'NEW_VAR': 'new_value'}

    with patch(
        'google.adk.tools.mcp_tool.mcp_session_manager.StdioServerParameters'
    ) as mock_stdio_class:
      updated_params = session_manager._inject_env_vars(new_env_vars)

      # Verify StdioServerParameters was called with merged env vars
      mock_stdio_class.assert_called_once_with(
          command='npx',
          args=['-y', '@modelcontextprotocol/server-filesystem'],
          env={
              'EXISTING_VAR': 'existing_value',
              'API_KEY': 'test_key',
              'NEW_VAR': 'new_value',
          },
          cwd=None,
          encoding=None,
          encoding_error_handler=None,
      )

  def test_inject_env_vars_no_existing_env(self, mock_stdio_params):
    """Test environment variable injection when no existing env vars."""
    # Set up mock with no existing env and set class name
    mock_stdio_params.env = None
    mock_stdio_params.__class__.__name__ = 'StdioServerParameters'

    session_manager = MCPSessionManager(connection_params=mock_stdio_params)

    new_env_vars = {'API_KEY': 'test_key', 'NEW_VAR': 'new_value'}

    with patch(
        'google.adk.tools.mcp_tool.mcp_session_manager.StdioServerParameters'
    ) as mock_stdio_class:
      updated_params = session_manager._inject_env_vars(new_env_vars)

      # Verify StdioServerParameters was called with only new env vars
      mock_stdio_class.assert_called_once_with(
          command='npx',
          args=['-y', '@modelcontextprotocol/server-filesystem'],
          env={'API_KEY': 'test_key', 'NEW_VAR': 'new_value'},
          cwd=None,
          encoding=None,
          encoding_error_handler=None,
      )

  def test_inject_env_vars_empty_env(self, mock_stdio_params):
    """Test environment variable injection with empty env vars."""
    mock_stdio_params.__class__.__name__ = 'StdioServerParameters'

    session_manager = MCPSessionManager(connection_params=mock_stdio_params)

    # No new env vars to inject
    empty_env_vars = {}

    updated_params = session_manager._inject_env_vars(empty_env_vars)

    # Should return original params when no env vars to inject
    assert updated_params == mock_stdio_params

  def test_inject_env_vars_non_stdio_params(self, mock_sse_params):
    """Test that _inject_env_vars is not called for non-StdioServerParameters.

    Since the refactoring moved the logic inside the isinstance check,
    this method will only be called for StdioServerParameters.
    """
    session_manager = MCPSessionManager(connection_params=mock_sse_params)

    # Since _inject_env_vars is only called for StdioServerParameters,
    # we test that it would fail if called with non-stdio params and non-empty env
    non_empty_env_vars = {'API_KEY': 'test_key'}

    # This test verifies that if _inject_env_vars were called with non-stdio params,
    # it would fail because SseServerParams doesn't have 'command' attribute
    with pytest.raises(AttributeError):
      # This should fail because SseServerParams doesn't have 'command' attribute
      session_manager._inject_env_vars(non_empty_env_vars)

  @pytest.mark.asyncio
  async def test_create_session_with_env_injection(
      self,
      mock_stdio_params,
      sample_context_to_env_mapper_callback,
      mock_readonly_context,
  ):
    """Test create_session with environment variable injection."""
    from unittest.mock import AsyncMock

    session_manager = MCPSessionManager(
        connection_params=mock_stdio_params,
        context_to_env_mapper_callback=sample_context_to_env_mapper_callback,
    )

    # Mock the necessary components
    mock_exit_stack = AsyncMock()
    mock_client = AsyncMock()
    mock_session = AsyncMock()
    mock_transports = [AsyncMock(), AsyncMock()]

    with (
        patch(
            'google.adk.tools.mcp_tool.mcp_session_manager.AsyncExitStack',
            return_value=mock_exit_stack,
        ),
        patch(
            'google.adk.tools.mcp_tool.mcp_session_manager.stdio_client',
            return_value=mock_client,
        ),
        patch(
            'google.adk.tools.mcp_tool.mcp_session_manager.ClientSession',
            return_value=mock_session,
        ),
        patch(
            'google.adk.tools.mcp_tool.mcp_session_manager.isinstance'
        ) as mock_isinstance,
    ):

      # Make isinstance return True only for StdioServerParameters check
      def isinstance_side_effect(obj, cls):
        # Check if this is our mock stdio params
        if obj == mock_stdio_params:
          # For our mock object, check the class name/type
          if (
              hasattr(cls, '__name__')
              and cls.__name__ == 'StdioServerParameters'
          ):
            return True
          elif 'StdioServerParameters' in str(cls):
            return True
          elif hasattr(cls, '__name__') and cls.__name__ in [
              'SseServerParams',
              'StreamableHTTPServerParams',
          ]:
            return False
          elif any(
              name in str(cls)
              for name in ['SseServerParams', 'StreamableHTTPServerParams']
          ):
            return False
          # If it's the first call with the mocked class, return True (this is the StdioServerParameters check)
          elif str(cls).startswith('<MagicMock'):
            return True
        # For other objects, use the original isinstance
        try:
          original_result = (
              isinstance.__wrapped__(obj, cls)
              if hasattr(isinstance, '__wrapped__')
              else type(obj) is cls
          )
          return original_result
        except:
          return False

      mock_isinstance.side_effect = isinstance_side_effect
      mock_exit_stack.enter_async_context.side_effect = [
          mock_transports,
          mock_session,
      ]

      result = await session_manager.create_session(mock_readonly_context)

      # Verify the result
      assert result == mock_session
      assert session_manager._session == mock_session

  @pytest.mark.asyncio
  async def test_create_session_without_context(
      self, mock_stdio_params, sample_context_to_env_mapper_callback
  ):
    """Test create_session without readonly_context."""
    from unittest.mock import AsyncMock

    session_manager = MCPSessionManager(
        connection_params=mock_stdio_params,
        context_to_env_mapper_callback=sample_context_to_env_mapper_callback,
    )

    # Mock the necessary components
    mock_exit_stack = AsyncMock()
    mock_client = AsyncMock()
    mock_session = AsyncMock()
    mock_transports = [AsyncMock(), AsyncMock()]

    with (
        patch(
            'google.adk.tools.mcp_tool.mcp_session_manager.AsyncExitStack',
            return_value=mock_exit_stack,
        ),
        patch(
            'google.adk.tools.mcp_tool.mcp_session_manager.stdio_client',
            return_value=mock_client,
        ),
        patch(
            'google.adk.tools.mcp_tool.mcp_session_manager.ClientSession',
            return_value=mock_session,
        ),
        patch(
            'google.adk.tools.mcp_tool.mcp_session_manager.isinstance'
        ) as mock_isinstance,
    ):

      # Make isinstance return True only for StdioServerParameters check
      def isinstance_side_effect(obj, cls):
        # Check if this is our mock stdio params
        if obj == mock_stdio_params:
          # For our mock object, check the class name/type
          if (
              hasattr(cls, '__name__')
              and cls.__name__ == 'StdioServerParameters'
          ):
            return True
          elif 'StdioServerParameters' in str(cls):
            return True
          elif hasattr(cls, '__name__') and cls.__name__ in [
              'SseServerParams',
              'StreamableHTTPServerParams',
          ]:
            return False
          elif any(
              name in str(cls)
              for name in ['SseServerParams', 'StreamableHTTPServerParams']
          ):
            return False
          # If it's the first call with the mocked class, return True (this is the StdioServerParameters check)
          elif str(cls).startswith('<MagicMock'):
            return True
        # For other objects, use the original isinstance
        try:
          original_result = (
              isinstance.__wrapped__(obj, cls)
              if hasattr(isinstance, '__wrapped__')
              else type(obj) is cls
          )
          return original_result
        except:
          return False

      mock_isinstance.side_effect = isinstance_side_effect
      mock_exit_stack.enter_async_context.side_effect = [
          mock_transports,
          mock_session,
      ]

      result = await session_manager.create_session(None)

      # Verify the result
      assert result == mock_session
      assert session_manager._session == mock_session
