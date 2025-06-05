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

"""Unit tests for MCPToolset auth functionality."""

import pytest
from typing import Any, Dict, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch
from types import MappingProxyType

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes
from google.adk.auth.auth_schemes import AuthScheme
from google.adk.tools.mcp_tool.mcp_toolset import AuthTransformCallback

# Mock MCP imports to avoid dependency issues in tests
@pytest.fixture(autouse=True)
def mock_mcp_imports():
    """Mock MCP imports to avoid import errors in testing."""
    from unittest.mock import MagicMock
    
    with patch.dict('sys.modules', {
        'mcp': MagicMock(),
        'mcp.types': MagicMock(),
    }):
        # Mock the specific classes we need
        mock_stdio_params = MagicMock()
        mock_list_tools_result = MagicMock()
        mock_tool = MagicMock()
        
        with patch('google.adk.tools.mcp_tool.mcp_toolset.StdioServerParameters', mock_stdio_params), \
             patch('google.adk.tools.mcp_tool.mcp_toolset.ListToolsResult', mock_list_tools_result):
            yield


# Import after mocking to avoid MCP dependency issues
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset


@pytest.fixture
def mock_auth_scheme():
    """Create a mock AuthScheme for testing."""
    from fastapi.openapi.models import HTTPBearer
    return HTTPBearer(bearerFormat="JWT")


@pytest.fixture
def mock_auth_credential():
    """Create a mock AuthCredential for testing."""
    return AuthCredential(
        auth_type=AuthCredentialTypes.API_KEY,
        api_key="test_api_key"
    )


@pytest.fixture
def mock_connection_params():
    """Create mock connection parameters."""
    mock_params = MagicMock()
    mock_params.command = "test_command"
    return mock_params


@pytest.fixture
def mock_readonly_context_with_auth(mock_auth_scheme, mock_auth_credential):
    """Create a ReadonlyContext with auth information."""
    context = MagicMock(spec=ReadonlyContext)
    context.state = MappingProxyType({
        "auth_scheme": mock_auth_scheme,
        "auth_credential": mock_auth_credential,
        "other_data": "test_value"
    })
    return context


@pytest.fixture
def mock_readonly_context_no_auth():
    """Create a ReadonlyContext without auth information."""
    context = MagicMock(spec=ReadonlyContext)
    context.state = MappingProxyType({
        "other_data": "test_value"
    })
    return context


@pytest.fixture
def mock_readonly_context_custom_auth():
    """Create a ReadonlyContext with custom auth format."""
    context = MagicMock(spec=ReadonlyContext)
    context.state = MappingProxyType({
        "custom_auth": {
            "type": "bearer",
            "token": "custom_token_123"
        },
        "other_data": "test_value"
    })
    return context


class TestMCPToolsetAuth:
    """Test auth functionality in MCPToolset."""

    def test_init_with_auth_callback(self, mock_connection_params):
        """Test MCPToolset initialization with auth transform callback."""
        def custom_auth_callback(state: Dict[str, Any]) -> Tuple[Optional[AuthScheme], Optional[AuthCredential]]:
            return None, None
        
        with patch('google.adk.tools.mcp_tool.mcp_toolset.MCPSessionManager'):
            toolset = MCPToolset(
                connection_params=mock_connection_params,
                auth_transform_callback=custom_auth_callback
            )
            
            assert toolset._auth_transform_callback == custom_auth_callback

    def test_init_without_auth_callback(self, mock_connection_params):
        """Test MCPToolset initialization without auth transform callback."""
        with patch('google.adk.tools.mcp_tool.mcp_toolset.MCPSessionManager'):
            toolset = MCPToolset(connection_params=mock_connection_params)
            
            assert toolset._auth_transform_callback is None

    def test_extract_auth_direct_extraction(self, mock_connection_params, mock_readonly_context_with_auth,
                                           mock_auth_scheme, mock_auth_credential):
        """Test direct auth extraction from context state."""
        with patch('google.adk.tools.mcp_tool.mcp_toolset.MCPSessionManager'):
            toolset = MCPToolset(connection_params=mock_connection_params)
            
            auth_scheme, auth_credential = toolset._extract_auth_from_context(mock_readonly_context_with_auth)
            
            assert auth_scheme == mock_auth_scheme
            assert auth_credential == mock_auth_credential

    def test_extract_auth_no_context(self, mock_connection_params):
        """Test auth extraction with no context."""
        with patch('google.adk.tools.mcp_tool.mcp_toolset.MCPSessionManager'):
            toolset = MCPToolset(connection_params=mock_connection_params)
            
            auth_scheme, auth_credential = toolset._extract_auth_from_context(None)
            
            assert auth_scheme is None
            assert auth_credential is None

    def test_extract_auth_no_auth_in_context(self, mock_connection_params, mock_readonly_context_no_auth):
        """Test auth extraction when context has no auth information."""
        with patch('google.adk.tools.mcp_tool.mcp_toolset.MCPSessionManager'):
            toolset = MCPToolset(connection_params=mock_connection_params)
            
            auth_scheme, auth_credential = toolset._extract_auth_from_context(mock_readonly_context_no_auth)
            
            assert auth_scheme is None
            assert auth_credential is None

    def test_extract_auth_with_callback(self, mock_connection_params, mock_readonly_context_custom_auth,
                                       mock_auth_scheme, mock_auth_credential):
        """Test auth extraction using custom callback."""
        def custom_auth_callback(state: Dict[str, Any]) -> Tuple[Optional[AuthScheme], Optional[AuthCredential]]:
            custom_auth = state.get("custom_auth")
            if custom_auth and custom_auth.get("type") == "bearer":
                # Return mock auth objects for testing
                return mock_auth_scheme, mock_auth_credential
            return None, None
        
        with patch('google.adk.tools.mcp_tool.mcp_toolset.MCPSessionManager'):
            toolset = MCPToolset(
                connection_params=mock_connection_params,
                auth_transform_callback=custom_auth_callback
            )
            
            auth_scheme, auth_credential = toolset._extract_auth_from_context(mock_readonly_context_custom_auth)
            
            assert auth_scheme == mock_auth_scheme
            assert auth_credential == mock_auth_credential

    def test_extract_auth_callback_fallback(self, mock_connection_params, mock_readonly_context_with_auth,
                                           mock_auth_scheme, mock_auth_credential):
        """Test fallback to direct extraction when callback fails."""
        def failing_callback(state: Dict[str, Any]) -> Tuple[Optional[AuthScheme], Optional[AuthCredential]]:
            raise ValueError("Callback failed")
        
        with patch('google.adk.tools.mcp_tool.mcp_toolset.MCPSessionManager'):
            toolset = MCPToolset(
                connection_params=mock_connection_params,
                auth_transform_callback=failing_callback
            )
            
            # Should fall back to direct extraction
            auth_scheme, auth_credential = toolset._extract_auth_from_context(mock_readonly_context_with_auth)
            
            assert auth_scheme == mock_auth_scheme
            assert auth_credential == mock_auth_credential

    @pytest.mark.asyncio
    async def test_get_tools_passes_auth_to_mcp_tool(self, mock_connection_params, mock_readonly_context_with_auth,
                                                     mock_auth_scheme, mock_auth_credential):
        """Test that get_tools passes auth parameters to MCPTool constructor."""
        # Mock the MCP session and tools
        mock_session = AsyncMock()
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool description"

        mock_tools_response = MagicMock()
        mock_tools_response.tools = [mock_tool]
        mock_session.list_tools.return_value = mock_tools_response

        with patch('google.adk.tools.mcp_tool.mcp_toolset.MCPSessionManager') as mock_session_manager_class, \
             patch('google.adk.tools.mcp_tool.mcp_toolset.MCPTool') as mock_mcp_tool_class:

            # Setup session manager mock to return async mock properly
            mock_session_manager = AsyncMock()
            mock_session_manager.create_session.return_value = mock_session
            mock_session_manager_class.return_value = mock_session_manager

            # Setup MCPTool mock
            mock_mcp_tool_instance = MagicMock()
            mock_mcp_tool_class.return_value = mock_mcp_tool_instance

            toolset = MCPToolset(connection_params=mock_connection_params)

            # Mock the _is_tool_selected method to return True
            toolset._is_tool_selected = MagicMock(return_value=True)

            tools = await toolset.get_tools(mock_readonly_context_with_auth)
            
            # Verify MCPTool was called with auth parameters
            mock_mcp_tool_class.assert_called_once_with(
                mcp_tool=mock_tool,
                mcp_session_manager=mock_session_manager,
                auth_scheme=mock_auth_scheme,
                auth_credential=mock_auth_credential
            )
            
            assert len(tools) == 1
            # The returned tool should be an MCPTool instance (mock)
            assert isinstance(tools[0], type(mock_mcp_tool_instance))

    @pytest.mark.asyncio
    async def test_get_tools_no_auth_in_context(self, mock_connection_params, mock_readonly_context_no_auth):
        """Test that get_tools handles no auth in context gracefully."""
        # Mock the MCP session and tools
        mock_session = AsyncMock()
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool description"

        mock_tools_response = MagicMock()
        mock_tools_response.tools = [mock_tool]
        mock_session.list_tools.return_value = mock_tools_response

        with patch('google.adk.tools.mcp_tool.mcp_toolset.MCPSessionManager') as mock_session_manager_class, \
             patch('google.adk.tools.mcp_tool.mcp_toolset.MCPTool') as mock_mcp_tool_class:

            # Setup session manager mock to return async mock properly
            mock_session_manager = AsyncMock()
            mock_session_manager.create_session.return_value = mock_session
            mock_session_manager_class.return_value = mock_session_manager

            # Setup MCPTool mock
            mock_mcp_tool_instance = MagicMock()
            mock_mcp_tool_class.return_value = mock_mcp_tool_instance

            toolset = MCPToolset(connection_params=mock_connection_params)

            # Mock the _is_tool_selected method to return True
            toolset._is_tool_selected = MagicMock(return_value=True)

            tools = await toolset.get_tools(mock_readonly_context_no_auth)
            
            # Verify MCPTool was called with None auth parameters
            mock_mcp_tool_class.assert_called_once_with(
                mcp_tool=mock_tool,
                mcp_session_manager=mock_session_manager,
                auth_scheme=None,
                auth_credential=None
            )
            
            assert len(tools) == 1
            assert tools[0] == mock_mcp_tool_instance

    def test_extract_auth_invalid_types(self, mock_connection_params):
        """Test auth extraction with invalid types in context."""
        # Create context with invalid auth types
        context = MagicMock(spec=ReadonlyContext)
        context.state = MappingProxyType({
            "auth_scheme": "invalid_scheme",  # Should be AuthScheme instance
            "auth_credential": {"invalid": "credential"}  # Should be AuthCredential instance
        })
        
        with patch('google.adk.tools.mcp_tool.mcp_toolset.MCPSessionManager'):
            toolset = MCPToolset(connection_params=mock_connection_params)
            
            auth_scheme, auth_credential = toolset._extract_auth_from_context(context)
            
            # Both should be None due to invalid types
            assert auth_scheme is None
            assert auth_credential is None
