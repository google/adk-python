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
from io import StringIO
import sys
import unittest
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

from google.adk.auth.auth_credential import AuthCredential
import pytest

# Skip all tests in this module if Python version is less than 3.10
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10), reason="MCP tool requires Python 3.10+"
)

# Import dependencies with version checking
try:
  from google.adk.agents.readonly_context import ReadonlyContext
  from google.adk.tools.mcp_tool.mcp_session_manager import MCPSessionManager
  from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams
  from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
  from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
  from google.adk.tools.mcp_tool.mcp_tool import MCPTool
  from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
  from mcp import StdioServerParameters
except ImportError as e:
  if sys.version_info < (3, 10):
    # Create dummy classes to prevent NameError during test collection
    # Tests will be skipped anyway due to pytestmark
    class DummyClass:
      pass

    class StdioServerParameters:

      def __init__(self, command="test_command", args=None):
        self.command = command
        self.args = args or []

    MCPSessionManager = DummyClass
    SseConnectionParams = DummyClass
    StdioConnectionParams = DummyClass
    StreamableHTTPConnectionParams = DummyClass
    MCPTool = DummyClass
    MCPToolset = DummyClass
    ReadonlyContext = DummyClass
  else:
    raise e


class MockMCPTool:
  """Mock MCP Tool for testing."""

  def __init__(self, name, description="Test tool description"):
    self.name = name
    self.description = description
    self.inputSchema = {
        "type": "object",
        "properties": {"param": {"type": "string"}},
    }


class MockListToolsResult:
  """Mock ListToolsResult for testing."""

  def __init__(self, tools):
    self.tools = tools


class TestMCPToolset:
  """Test suite for MCPToolset class."""

  def setup_method(self):
    """Set up test fixtures."""
    self.mock_stdio_params = StdioServerParameters(
        command="test_command", args=[]
    )
    self.mock_session_manager = Mock(spec=MCPSessionManager)
    self.mock_session = AsyncMock()
    self.mock_session_manager.create_session = AsyncMock(
        return_value=self.mock_session
    )

  def test_init_basic(self):
    """Test basic initialization with StdioServerParameters."""
    toolset = MCPToolset(connection_params=self.mock_stdio_params)

    # Note: StdioServerParameters gets converted to StdioConnectionParams internally
    assert toolset._errlog == sys.stderr
    assert toolset._auth_scheme is None
    assert toolset._auth_credential is None

  def test_init_with_stdio_connection_params(self):
    """Test initialization with StdioConnectionParams."""
    stdio_params = StdioConnectionParams(
        server_params=self.mock_stdio_params, timeout=10.0
    )
    toolset = MCPToolset(connection_params=stdio_params)

    assert toolset._connection_params == stdio_params

  def test_init_with_sse_connection_params(self):
    """Test initialization with SseConnectionParams."""
    sse_params = SseConnectionParams(
        url="https://example.com/mcp", headers={"Authorization": "Bearer token"}
    )
    toolset = MCPToolset(connection_params=sse_params)

    assert toolset._connection_params == sse_params

  def test_init_with_streamable_http_params(self):
    """Test initialization with StreamableHTTPConnectionParams."""
    http_params = StreamableHTTPConnectionParams(
        url="https://example.com/mcp",
        headers={"Content-Type": "application/json"},
    )
    toolset = MCPToolset(connection_params=http_params)

    assert toolset._connection_params == http_params

  def test_init_with_tool_filter_list(self):
    """Test initialization with tool filter as list."""
    tool_filter = ["tool1", "tool2"]
    toolset = MCPToolset(
        connection_params=self.mock_stdio_params, tool_filter=tool_filter
    )

    # The tool filter is stored in the parent BaseToolset class
    # We can verify it by checking the filtering behavior in get_tools
    assert toolset._is_tool_selected is not None

  def test_init_with_auth(self):
    """Test initialization with authentication."""
    # Create real auth scheme instances
    from fastapi.openapi.models import OAuth2

    auth_scheme = OAuth2(flows={})
    from google.adk.auth.auth_credential import OAuth2Auth

    auth_credential = AuthCredential(
        auth_type="oauth2",
        oauth2=OAuth2Auth(client_id="test_id", client_secret="test_secret"),
    )

    toolset = MCPToolset(
        connection_params=self.mock_stdio_params,
        auth_scheme=auth_scheme,
        auth_credential=auth_credential,
    )

    assert toolset._auth_scheme == auth_scheme
    assert toolset._auth_credential == auth_credential

  def test_init_missing_connection_params(self):
    """Test initialization with missing connection params raises error."""
    with pytest.raises(ValueError, match="Missing connection params"):
      MCPToolset(connection_params=None)

  @pytest.mark.asyncio
  async def test_get_tools_basic(self):
    """Test getting tools without filtering."""
    # Mock tools from MCP server
    mock_tools = [
        MockMCPTool("tool1"),
        MockMCPTool("tool2"),
        MockMCPTool("tool3"),
    ]
    self.mock_session.list_tools = AsyncMock(
        return_value=MockListToolsResult(mock_tools)
    )

    toolset = MCPToolset(connection_params=self.mock_stdio_params)
    toolset._mcp_session_manager = self.mock_session_manager

    tools = await toolset.get_tools()

    assert len(tools) == 3
    for tool in tools:
      assert isinstance(tool, MCPTool)
    assert tools[0].name == "tool1"
    assert tools[1].name == "tool2"
    assert tools[2].name == "tool3"

  @pytest.mark.asyncio
  async def test_get_tools_with_list_filter(self):
    """Test getting tools with list-based filtering."""
    # Mock tools from MCP server
    mock_tools = [
        MockMCPTool("tool1"),
        MockMCPTool("tool2"),
        MockMCPTool("tool3"),
    ]
    self.mock_session.list_tools = AsyncMock(
        return_value=MockListToolsResult(mock_tools)
    )

    tool_filter = ["tool1", "tool3"]
    toolset = MCPToolset(
        connection_params=self.mock_stdio_params, tool_filter=tool_filter
    )
    toolset._mcp_session_manager = self.mock_session_manager

    tools = await toolset.get_tools()

    assert len(tools) == 2
    assert tools[0].name == "tool1"
    assert tools[1].name == "tool3"

  @pytest.mark.asyncio
  async def test_get_tools_with_function_filter(self):
    """Test getting tools with function-based filtering."""
    # Mock tools from MCP server
    mock_tools = [
        MockMCPTool("read_file"),
        MockMCPTool("write_file"),
        MockMCPTool("list_directory"),
    ]
    self.mock_session.list_tools = AsyncMock(
        return_value=MockListToolsResult(mock_tools)
    )

    def file_tools_filter(tool, context):
      """Filter for file-related tools only."""
      return "file" in tool.name

    toolset = MCPToolset(
        connection_params=self.mock_stdio_params, tool_filter=file_tools_filter
    )
    toolset._mcp_session_manager = self.mock_session_manager

    tools = await toolset.get_tools()

    assert len(tools) == 2
    assert tools[0].name == "read_file"
    assert tools[1].name == "write_file"

  @pytest.mark.asyncio
  async def test_get_tools_with_header_provider(self):
    """Test get_tools with a header_provider."""
    mock_tools = [MockMCPTool("tool1"), MockMCPTool("tool2")]
    self.mock_session.list_tools = AsyncMock(
        return_value=MockListToolsResult(mock_tools)
    )
    mock_readonly_context = Mock(spec=ReadonlyContext)
    expected_headers = {"X-Tenant-ID": "test-tenant"}
    header_provider = Mock(return_value=expected_headers)

    toolset = MCPToolset(
        connection_params=self.mock_stdio_params,
        header_provider=header_provider,
    )
    toolset._mcp_session_manager = self.mock_session_manager

    tools = await toolset.get_tools(readonly_context=mock_readonly_context)

    assert len(tools) == 2
    header_provider.assert_called_once_with(mock_readonly_context)
    self.mock_session_manager.create_session.assert_called_once_with(
        headers=expected_headers
    )

  @pytest.mark.asyncio
  async def test_close_success(self):
    """Test successful cleanup."""
    toolset = MCPToolset(connection_params=self.mock_stdio_params)
    toolset._mcp_session_manager = self.mock_session_manager

    await toolset.close()

    self.mock_session_manager.close.assert_called_once()

  @pytest.mark.asyncio
  async def test_close_with_exception(self):
    """Test cleanup when session manager raises exception."""
    toolset = MCPToolset(connection_params=self.mock_stdio_params)
    toolset._mcp_session_manager = self.mock_session_manager

    # Mock close to raise an exception
    self.mock_session_manager.close = AsyncMock(
        side_effect=Exception("Cleanup error")
    )

    custom_errlog = StringIO()
    toolset._errlog = custom_errlog

    # Should not raise exception
    await toolset.close()

    # Should log the error
    error_output = custom_errlog.getvalue()
    assert "Warning: Error during McpToolset cleanup" in error_output
    assert "Cleanup error" in error_output

  @pytest.mark.asyncio
  async def test_get_tools_with_timeout(self):
    """Test get_tools with timeout."""
    stdio_params = StdioConnectionParams(
        server_params=self.mock_stdio_params, timeout=0.01
    )
    toolset = MCPToolset(connection_params=stdio_params)
    toolset._mcp_session_manager = self.mock_session_manager

    async def long_running_list_tools():
      await asyncio.sleep(0.1)
      return MockListToolsResult([])

    self.mock_session.list_tools = long_running_list_tools

    with pytest.raises(
        ConnectionError, match="Failed to get tools from MCP server."
    ):
      await toolset.get_tools()

  @pytest.mark.asyncio
  async def test_get_tools_retry_decorator(self):
    """Test that get_tools has retry decorator applied."""
    toolset = MCPToolset(connection_params=self.mock_stdio_params)

    # Check that the method has the retry decorator
    assert hasattr(toolset.get_tools, "__wrapped__")
