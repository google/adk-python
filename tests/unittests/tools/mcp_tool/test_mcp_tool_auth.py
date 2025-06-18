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

"""Unit tests for MCPTool auth functionality."""

import logging
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi.openapi.models import HTTPBearer
from fastapi.openapi.models import SecuritySchemeType
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.tools.tool_context import ToolContext
import pytest


# Mock MCP imports to avoid dependency issues in tests
@pytest.fixture(autouse=True)
def mock_mcp_imports():
  """Mock MCP imports to avoid import errors in testing."""
  with patch.dict(
      "sys.modules",
      {
          "mcp": MagicMock(),
          "mcp.types": MagicMock(),
          "mcp.client": MagicMock(),
          "mcp.client.stdio": MagicMock(),
          "mcp.client.sse": MagicMock(),
          "mcp.client.streamable_http": MagicMock(),
      },
  ):
    # Mock the Tool class from mcp.types
    mock_tool_class = MagicMock()
    mock_tool_class.name = "test_tool"
    mock_tool_class.description = "Test tool description"
    mock_tool_class.inputSchema = {"type": "object", "properties": {}}

    with patch(
        "google.adk.tools.mcp_tool.mcp_tool.McpBaseTool", mock_tool_class
    ):
      yield


# Import after mocking to avoid MCP dependency issues
from google.adk.tools.mcp_tool.mcp_tool import MCPTool


@pytest.fixture
def mock_auth_scheme():
  """Create a mock AuthScheme for testing."""
  return HTTPBearer(scheme="bearer", bearerFormat="JWT")


@pytest.fixture
def mock_auth_credential():
  """Create a mock AuthCredential for testing."""
  return AuthCredential(
      auth_type=AuthCredentialTypes.API_KEY, api_key="test_api_key"
  )


@pytest.fixture
def mock_mcp_tool():
  """Create a mock MCP tool."""
  tool = MagicMock()
  tool.name = "test_tool"
  tool.description = "Test tool description"
  tool.inputSchema = {"type": "object", "properties": {}}
  return tool


@pytest.fixture
def mock_session_manager():
  """Create a mock MCP session manager."""
  return MagicMock()


class TestMCPToolAuth:
  """Test auth functionality in MCPTool."""

  def test_init_with_auth(
      self,
      mock_mcp_tool,
      mock_session_manager,
      mock_auth_scheme,
      mock_auth_credential,
  ):
    """Test MCPTool initialization with auth parameters."""
    tool = MCPTool(
        mcp_tool=mock_mcp_tool,
        mcp_session_manager=mock_session_manager,
        auth_scheme=mock_auth_scheme,
        auth_credential=mock_auth_credential,
    )

    assert tool._auth_scheme == mock_auth_scheme
    assert tool._auth_credential == mock_auth_credential

  def test_init_without_auth(self, mock_mcp_tool, mock_session_manager):
    """Test MCPTool initialization without auth parameters."""
    tool = MCPTool(
        mcp_tool=mock_mcp_tool, mcp_session_manager=mock_session_manager
    )

    assert tool._auth_scheme is None
    assert tool._auth_credential is None

  @pytest.mark.asyncio
  async def test_run_async_with_auth_logging(
      self,
      mock_mcp_tool,
      mock_session_manager,
      mock_auth_scheme,
      mock_auth_credential,
      caplog,
  ):
    """Test that run_async logs auth information when available."""
    # Create mock session
    mock_session = AsyncMock()
    mock_session.call_tool.return_value = {"result": "success"}
    mock_session_manager.create_session = AsyncMock(return_value=mock_session)

    # Create tool with auth
    tool = MCPTool(
        mcp_tool=mock_mcp_tool,
        mcp_session_manager=mock_session_manager,
        auth_scheme=mock_auth_scheme,
        auth_credential=mock_auth_credential,
    )

    # Create mock tool context
    mock_tool_context = MagicMock(spec=ToolContext)

    # Set logging level to capture info logs
    with caplog.at_level(logging.INFO):
      result = await tool.run_async(
          args={"test": "value"}, tool_context=mock_tool_context
      )

    # Verify the tool was called
    mock_session.call_tool.assert_called_once_with(
        "test_tool", arguments={"test": "value"}
    )
    assert result == {"result": "success"}

    # Check that auth information was logged
    assert "MCPTool 'test_tool' has authentication configured" in caplog.text
    assert "scheme=HTTPBearer" in caplog.text
    assert "credential=AuthCredential" in caplog.text

  @pytest.mark.asyncio
  async def test_run_async_without_auth_no_logging(
      self, mock_mcp_tool, mock_session_manager, caplog
  ):
    """Test that run_async doesn't log auth info when no auth is configured."""
    # Create mock session
    mock_session = AsyncMock()
    mock_session.call_tool.return_value = {"result": "success"}
    mock_session_manager.create_session = AsyncMock(return_value=mock_session)

    # Create tool without auth
    tool = MCPTool(
        mcp_tool=mock_mcp_tool, mcp_session_manager=mock_session_manager
    )

    # Create mock tool context
    mock_tool_context = MagicMock(spec=ToolContext)

    # Set logging level to capture info logs
    with caplog.at_level(logging.INFO):
      result = await tool.run_async(
          args={"test": "value"}, tool_context=mock_tool_context
      )

    # Verify the tool was called
    mock_session.call_tool.assert_called_once_with(
        "test_tool", arguments={"test": "value"}
    )
    assert result == {"result": "success"}

    # Check that no auth information was logged
    assert "has authentication configured" not in caplog.text
