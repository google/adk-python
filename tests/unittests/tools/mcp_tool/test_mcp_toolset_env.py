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

"""Unit tests for MCPToolset environment variable functionality."""

from types import MappingProxyType
from typing import Any
from typing import Dict
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.mcp_tool.mcp_session_manager import ContextToEnvMapperCallback
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
      },
  ):
    # Mock the specific classes we need
    mock_stdio_params = MagicMock()
    mock_list_tools_result = MagicMock()

    with (
        patch(
            'google.adk.tools.mcp_tool.mcp_toolset.StdioServerParameters',
            mock_stdio_params,
        ),
        patch(
            'google.adk.tools.mcp_tool.mcp_toolset.ListToolsResult',
            mock_list_tools_result,
        ),
    ):
      yield


# Import after mocking to avoid MCP dependency issues
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset


@pytest.fixture
def mock_stdio_params():
  """Create a mock StdioServerParameters instance."""
  from unittest.mock import MagicMock

  mock_params = MagicMock()
  mock_params.command = 'npx'
  mock_params.args = ['-y', '@modelcontextprotocol/server-filesystem']
  mock_params.env = {'EXISTING_VAR': 'existing_value'}
  return mock_params


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


class TestMCPToolsetEnv:
  """Test environment variable functionality in MCPToolset."""

  def test_init_with_env_callback(
      self, mock_stdio_params, sample_context_to_env_mapper_callback
  ):
    """Test MCPToolset initialization with context to env mapper callback."""
    with patch(
        'google.adk.tools.mcp_tool.mcp_toolset.MCPSessionManager'
    ) as mock_session_manager:
      toolset = MCPToolset(
          connection_params=mock_stdio_params,
          context_to_env_mapper_callback=sample_context_to_env_mapper_callback,
      )

      # Verify the session manager was created with the context to env mapper callback
      mock_session_manager.assert_called_once_with(
          connection_params=mock_stdio_params,
          errlog=toolset._errlog,
          context_to_env_mapper_callback=sample_context_to_env_mapper_callback,
      )

      assert (
          toolset._context_to_env_mapper_callback
          == sample_context_to_env_mapper_callback
      )

  def test_init_without_env_callback(self, mock_stdio_params):
    """Test MCPToolset initialization without environment callback."""
    with patch(
        'google.adk.tools.mcp_tool.mcp_toolset.MCPSessionManager'
    ) as mock_session_manager:
      toolset = MCPToolset(connection_params=mock_stdio_params)

      # Verify the session manager was created without env callback
      mock_session_manager.assert_called_once_with(
          connection_params=mock_stdio_params,
          errlog=toolset._errlog,
          context_to_env_mapper_callback=None,
      )

      assert toolset._context_to_env_mapper_callback is None

  @pytest.mark.asyncio
  async def test_get_tools_passes_context_to_session_manager(
      self,
      mock_stdio_params,
      sample_context_to_env_mapper_callback,
      mock_readonly_context,
  ):
    """Test that get_tools passes readonly_context to session manager."""
    with patch(
        'google.adk.tools.mcp_tool.mcp_toolset.MCPSessionManager'
    ) as mock_session_manager_class:
      # Set up mock session manager instance
      mock_session_manager = AsyncMock()
      mock_session = AsyncMock()
      mock_session.list_tools.return_value = MagicMock(tools=[])
      mock_session_manager.create_session.return_value = mock_session
      mock_session_manager_class.return_value = mock_session_manager

      toolset = MCPToolset(
          connection_params=mock_stdio_params,
          context_to_env_mapper_callback=sample_context_to_env_mapper_callback,
      )

      # Call get_tools with readonly_context
      await toolset.get_tools(mock_readonly_context)

      # Verify create_session was called with the readonly_context
      mock_session_manager.create_session.assert_called_once_with(
          mock_readonly_context
      )

  @pytest.mark.asyncio
  async def test_get_tools_without_context(
      self, mock_stdio_params, sample_context_to_env_mapper_callback
  ):
    """Test that get_tools works without readonly_context."""
    with patch(
        'google.adk.tools.mcp_tool.mcp_toolset.MCPSessionManager'
    ) as mock_session_manager_class:
      # Set up mock session manager instance
      mock_session_manager = AsyncMock()
      mock_session = AsyncMock()
      mock_session.list_tools.return_value = MagicMock(tools=[])
      mock_session_manager.create_session.return_value = mock_session
      mock_session_manager_class.return_value = mock_session_manager

      toolset = MCPToolset(
          connection_params=mock_stdio_params,
          context_to_env_mapper_callback=sample_context_to_env_mapper_callback,
      )

      # Call get_tools without readonly_context
      await toolset.get_tools(None)

      # Verify create_session was called with None
      mock_session_manager.create_session.assert_called_once_with(None)



  def test_both_auth_and_env_callbacks(self, mock_stdio_params):
    """Test MCPToolset initialization with both auth and env callbacks."""

    def auth_callback(state):
      return None, None

    def env_callback(state):
      return {'TEST_VAR': 'test_value'}

    with patch(
        'google.adk.tools.mcp_tool.mcp_toolset.MCPSessionManager'
    ) as mock_session_manager:
      toolset = MCPToolset(
          connection_params=mock_stdio_params,
          auth_transform_callback=auth_callback,
          context_to_env_mapper_callback=env_callback,
      )

      # Verify both callbacks are stored
      assert toolset._auth_transform_callback == auth_callback
      assert toolset._context_to_env_mapper_callback == env_callback

      # Verify the session manager was created with the env callback
      mock_session_manager.assert_called_once_with(
          connection_params=mock_stdio_params,
          errlog=toolset._errlog,
          context_to_env_mapper_callback=env_callback,
      )

  @pytest.mark.asyncio
  async def test_integration_env_extraction_and_injection(
      self, mock_stdio_params, mock_readonly_context
  ):
    """Test end-to-end environment variable extraction and injection."""

    def env_callback(state: Dict[str, Any]) -> Dict[str, str]:
      return {
          'API_KEY': state.get('api_key', ''),
          'WORKSPACE_PATH': state.get('workspace_path', ''),
      }

    with (
        patch(
            'google.adk.tools.mcp_tool.mcp_toolset.MCPSessionManager'
        ) as mock_session_manager_class,
        patch(
            'google.adk.tools.mcp_tool.mcp_toolset.MCPTool'
        ) as mock_mcp_tool_class,
    ):

      # Set up mock session manager instance
      mock_session_manager = AsyncMock()
      mock_session = AsyncMock()
      mock_tool_response = MagicMock()
      mock_tool_response.tools = []
      mock_session.list_tools.return_value = mock_tool_response
      mock_session_manager.create_session.return_value = mock_session
      mock_session_manager_class.return_value = mock_session_manager

      toolset = MCPToolset(
          connection_params=mock_stdio_params,
          context_to_env_mapper_callback=env_callback,
      )

      # Call get_tools with context containing state
      tools = await toolset.get_tools(mock_readonly_context)

      # Verify the session manager's create_session was called with the context
      mock_session_manager.create_session.assert_called_once_with(
          mock_readonly_context
      )

      # Verify list_tools was called on the session
      mock_session.list_tools.assert_called_once()

      assert isinstance(tools, list)
