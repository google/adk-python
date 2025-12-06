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

from types import MappingProxyType
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_schemes import AuthScheme
from google.adk.tools.mcp_tool.mcp_toolset import AuthExtractionError
from google.adk.tools.mcp_tool.mcp_toolset import GetAuthFromContextCallback
import pytest


# Mock MCP imports to avoid dependency issues in tests
@pytest.fixture(autouse=True)
def mock_mcp_imports():
  """Mock MCP imports to avoid import errors in testing."""
  from unittest.mock import MagicMock

  with patch.dict(
      "sys.modules",
      {
          "mcp": MagicMock(),
          "mcp.types": MagicMock(),
      },
  ):
    # Mock the specific classes we need
    mock_stdio_params = MagicMock()
    mock_list_tools_result = MagicMock()
    mock_tool = MagicMock()

    with (
        patch(
            "google.adk.tools.mcp_tool.mcp_toolset.StdioServerParameters",
            mock_stdio_params,
        ),
        patch(
            "google.adk.tools.mcp_tool.mcp_toolset.ListToolsResult",
            mock_list_tools_result,
        ),
    ):
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
      auth_type=AuthCredentialTypes.API_KEY, api_key="test_api_key"
  )


@pytest.fixture
def mock_connection_params():
  """Create mock connection parameters."""
  mock_params = MagicMock()
  mock_params.command = "test_command"
  mock_params.timeout = 10.0  # Add timeout to avoid asyncio.wait_for issues
  return mock_params


@pytest.fixture
def mock_readonly_context_with_auth(mock_auth_scheme, mock_auth_credential):
  """Create a ReadonlyContext with auth information."""
  context = MagicMock(spec=ReadonlyContext)
  context.state = MappingProxyType({
      "auth_scheme": mock_auth_scheme,
      "auth_credential": mock_auth_credential,
      "other_data": "test_value",
  })
  return context


class TestMCPToolsetAuth:
  """Test auth functionality in MCPToolset."""

  def test_init_with_auth_callback(self, mock_connection_params):
    """Test MCPToolset initialization with auth extraction callback."""

    def custom_auth_callback(
        state: Dict[str, Any],
    ) -> Tuple[Optional[AuthScheme], Optional[AuthCredential]]:
      return None, None

    with patch("google.adk.tools.mcp_tool.mcp_toolset.MCPSessionManager"):
      toolset = MCPToolset(
          connection_params=mock_connection_params,
          get_auth_from_context_fn=custom_auth_callback,
      )

      assert toolset._get_auth_from_context_fn == custom_auth_callback

  def test_init_without_auth_callback(self, mock_connection_params):
    """Test MCPToolset initialization without auth extraction callback."""
    with patch("google.adk.tools.mcp_tool.mcp_toolset.MCPSessionManager"):
      toolset = MCPToolset(connection_params=mock_connection_params)

      assert toolset._get_auth_from_context_fn is None

  def test_extract_auth_no_context_returns_init_values(
      self,
      mock_connection_params,
      mock_auth_scheme,
      mock_auth_credential,
  ):
    """Test fallback to __init__ auth_scheme and auth_credential when context is None."""
    with patch("google.adk.tools.mcp_tool.mcp_toolset.MCPSessionManager"):
      toolset = MCPToolset(
          connection_params=mock_connection_params,
          auth_scheme=mock_auth_scheme,
          auth_credential=mock_auth_credential,
      )
      auth_scheme, auth_credential = toolset._extract_auth_from_context(None)
      assert auth_scheme == mock_auth_scheme
      assert auth_credential == mock_auth_credential

  def test_extract_auth_no_context_no_init_values(self, mock_connection_params):
    """Test auth extraction with no context and no init values returns (None, None)."""
    with patch("google.adk.tools.mcp_tool.mcp_toolset.MCPSessionManager"):
      toolset = MCPToolset(connection_params=mock_connection_params)
      auth_scheme, auth_credential = toolset._extract_auth_from_context(None)
      assert auth_scheme is None
      assert auth_credential is None

  def test_extract_auth_no_callback_returns_init_values(
      self, mock_connection_params, mock_auth_scheme, mock_auth_credential
  ):
    """Test that context without callback returns __init__ values."""
    context = MagicMock(spec=ReadonlyContext)
    context.state = MagicMock(return_value={"some_data": "value"})
    with patch("google.adk.tools.mcp_tool.mcp_toolset.MCPSessionManager"):
      toolset = MCPToolset(
          connection_params=mock_connection_params,
          auth_scheme=mock_auth_scheme,
          auth_credential=mock_auth_credential,
      )
      auth_scheme, auth_credential = toolset._extract_auth_from_context(context)
      assert auth_scheme == mock_auth_scheme
      assert auth_credential == mock_auth_credential

  def test_extract_auth_with_callback_success(
      self,
      mock_connection_params,
      mock_auth_scheme,
      mock_auth_credential,
  ):
    """Test successful auth extraction using custom callback."""

    def custom_auth_callback(state: Dict[str, Any]):
      if state.get("custom_auth"):
        return mock_auth_scheme, mock_auth_credential
      return None, None

    context = MagicMock(spec=ReadonlyContext)
    context.state = MagicMock(return_value={"custom_auth": {"type": "bearer"}})

    with patch("google.adk.tools.mcp_tool.mcp_toolset.MCPSessionManager"):
      toolset = MCPToolset(
          connection_params=mock_connection_params,
          get_auth_from_context_fn=custom_auth_callback,
      )
      auth_scheme, auth_credential = toolset._extract_auth_from_context(context)
      assert auth_scheme == mock_auth_scheme
      assert auth_credential == mock_auth_credential

  def test_extract_auth_fallback_to_direct_state_lookup(
      self,
      mock_connection_params,
      mock_auth_scheme,
      mock_auth_credential,
  ):
    """Test fallback to direct state lookup when no callback is provided."""
    context = MagicMock(spec=ReadonlyContext)
    context.state = MagicMock(
        return_value={
            "auth_scheme": mock_auth_scheme,
            "auth_credential": mock_auth_credential,
        }
    )

    with patch("google.adk.tools.mcp_tool.mcp_toolset.MCPSessionManager"):
      toolset = MCPToolset(
          connection_params=mock_connection_params,
      )
      auth_scheme, auth_credential = toolset._extract_auth_from_context(context)
      assert auth_scheme == mock_auth_scheme
      assert auth_credential == mock_auth_credential

  def test_extract_auth_callback_execution_error(
      self,
      mock_connection_params,
  ):
    """Test that callback execution errors are wrapped in AuthExtractionError."""

    def failing_callback(state: Dict[str, Any]):
      raise ValueError("Callback failed")

    context = MagicMock(spec=ReadonlyContext)
    context.state = MagicMock(return_value={})

    with patch("google.adk.tools.mcp_tool.mcp_toolset.MCPSessionManager"):
      toolset = MCPToolset(
          connection_params=mock_connection_params,
          get_auth_from_context_fn=failing_callback,
      )

      with pytest.raises(
          AuthExtractionError, match="Auth extraction callback failed"
      ):
        toolset._extract_auth_from_context(context)

  def test_extract_auth_callback_invalid_return_type(
      self,
      mock_connection_params,
  ):
    """Test that invalid callback return types raise AuthExtractionError."""

    def invalid_callback(
        state: Dict[str, Any],
    ) -> Any:  # Use Any to avoid type checker
      return "invalid_return"  # Should return tuple

    context = MagicMock(spec=ReadonlyContext)
    context.state = MagicMock(return_value={})

    with patch("google.adk.tools.mcp_tool.mcp_toolset.MCPSessionManager"):
      toolset = MCPToolset(
          connection_params=mock_connection_params,
          get_auth_from_context_fn=invalid_callback,  # type: ignore
      )

      with pytest.raises(AuthExtractionError, match="must return a tuple"):
        toolset._extract_auth_from_context(context)

  def test_extract_auth_callback_invalid_tuple_length(
      self,
      mock_connection_params,
  ):
    """Test that callback returning wrong tuple length raises AuthExtractionError."""

    def invalid_callback(
        state: Dict[str, Any],
    ) -> Any:  # Use Any to avoid type checker
      return (None,)  # Should return tuple of length 2

    context = MagicMock(spec=ReadonlyContext)
    context.state = MagicMock(return_value={})

    with patch("google.adk.tools.mcp_tool.mcp_toolset.MCPSessionManager"):
      toolset = MCPToolset(
          connection_params=mock_connection_params,
          get_auth_from_context_fn=invalid_callback,  # type: ignore
      )

      with pytest.raises(AuthExtractionError, match="must return a tuple"):
        toolset._extract_auth_from_context(context)

  def test_extract_auth_callback_invalid_auth_scheme_type(
      self,
      mock_connection_params,
  ):
    """Test that invalid auth_scheme type raises AuthExtractionError."""

    def invalid_callback(
        state: Dict[str, Any],
    ) -> Any:  # Use Any to avoid type checker
      return (
          "invalid_auth_scheme",
          None,
      )  # auth_scheme should be AuthScheme or None

    context = MagicMock(spec=ReadonlyContext)
    context.state = MagicMock(return_value={})

    with patch("google.adk.tools.mcp_tool.mcp_toolset.MCPSessionManager"):
      toolset = MCPToolset(
          connection_params=mock_connection_params,
          get_auth_from_context_fn=invalid_callback,  # type: ignore
      )

      with pytest.raises(AuthExtractionError, match="invalid auth_scheme type"):
        toolset._extract_auth_from_context(context)

  def test_extract_auth_callback_invalid_auth_credential_type(
      self,
      mock_connection_params,
      mock_auth_scheme,
  ):
    """Test that invalid auth_credential type raises AuthExtractionError."""

    def invalid_callback(
        state: Dict[str, Any],
    ) -> Any:  # Use Any to avoid type checker
      return (
          mock_auth_scheme,
          "invalid_credential",
      )  # credential should be AuthCredential or None

    context = MagicMock(spec=ReadonlyContext)
    context.state = MagicMock(return_value={})

    with patch("google.adk.tools.mcp_tool.mcp_toolset.MCPSessionManager"):
      toolset = MCPToolset(
          connection_params=mock_connection_params,
          get_auth_from_context_fn=invalid_callback,  # type: ignore
      )

      with pytest.raises(
          AuthExtractionError, match="invalid auth_credential type"
      ):
        toolset._extract_auth_from_context(context)

  def test_extract_auth_state_extraction_error(
      self,
      mock_connection_params,
  ):
    """Test that state extraction errors are wrapped in AuthExtractionError."""

    def valid_callback(state: Dict[str, Any]):
      return None, None

    context = MagicMock(spec=ReadonlyContext)
    context.state = MagicMock(side_effect=TypeError("State extraction failed"))

    with patch("google.adk.tools.mcp_tool.mcp_toolset.MCPSessionManager"):
      toolset = MCPToolset(
          connection_params=mock_connection_params,
          get_auth_from_context_fn=valid_callback,
      )

      with pytest.raises(AuthExtractionError, match="Failed to extract state"):
        toolset._extract_auth_from_context(context)

  def test_extract_auth_callback_allows_none_values(
      self,
      mock_connection_params,
  ):
    """Test that callback can return None values for auth_scheme and auth_credential."""

    def callback_returning_none(state: Dict[str, Any]):
      return None, None

    context = MagicMock(spec=ReadonlyContext)
    context.state = MagicMock(return_value={})

    with patch("google.adk.tools.mcp_tool.mcp_toolset.MCPSessionManager"):
      toolset = MCPToolset(
          connection_params=mock_connection_params,
          get_auth_from_context_fn=callback_returning_none,
      )

      auth_scheme, auth_credential = toolset._extract_auth_from_context(context)
      assert auth_scheme is None
      assert auth_credential is None

  @pytest.mark.asyncio
  async def test_get_tools_passes_auth_to_mcp_tool(
      self,
      mock_connection_params,
      mock_readonly_context_with_auth,
      mock_auth_scheme,
      mock_auth_credential,
  ):
    """Test that get_tools passes auth parameters to MCPTool constructor."""
    # Mock the MCP session and tools
    mock_session = AsyncMock()
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "Test tool description"

    mock_tools_response = MagicMock()
    mock_tools_response.tools = [mock_tool]
    mock_session.list_tools.return_value = mock_tools_response

    with (
        patch(
            "google.adk.tools.mcp_tool.mcp_toolset.MCPSessionManager"
        ) as mock_session_manager_class,
        patch(
            "google.adk.tools.mcp_tool.mcp_toolset.MCPTool"
        ) as mock_mcp_tool_class,
    ):

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
      assert len(tools) == 1
      mock_mcp_tool_class.assert_called_once()
      call_args = mock_mcp_tool_class.call_args
      assert "auth_scheme" in call_args.kwargs
      assert "auth_credential" in call_args.kwargs
