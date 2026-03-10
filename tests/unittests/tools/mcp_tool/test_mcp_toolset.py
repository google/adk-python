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
import base64
from io import StringIO
import json
import sys
import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.auth.auth_credential import AuthCredential
from google.adk.tools.load_mcp_resource_tool import LoadMcpResourceTool
from google.adk.tools.mcp_tool.mcp_session_manager import MCPSessionManager
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
from google.adk.tools.mcp_tool.mcp_tool import MCPTool
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from mcp import StdioServerParameters
from mcp.types import BlobResourceContents
from mcp.types import ListResourcesResult
from mcp.types import ReadResourceResult
from mcp.types import Resource
from mcp.types import TextResourceContents
import pytest


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


class TestMcpToolset:
  """Test suite for McpToolset class."""

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
    toolset = McpToolset(connection_params=self.mock_stdio_params)

    # Note: StdioServerParameters gets converted to StdioConnectionParams internally
    assert toolset._errlog == sys.stderr
    assert toolset._auth_scheme is None
    assert toolset._auth_credential is None
    assert toolset._use_mcp_resources is False

  def test_init_with_use_mcp_resources(self):
    """Test initialization with use_mcp_resources."""
    toolset = McpToolset(
        connection_params=self.mock_stdio_params, use_mcp_resources=True
    )
    assert toolset._use_mcp_resources is True

  def test_init_with_stdio_connection_params(self):
    """Test initialization with StdioConnectionParams."""
    stdio_params = StdioConnectionParams(
        server_params=self.mock_stdio_params, timeout=10.0
    )
    toolset = McpToolset(connection_params=stdio_params)

    assert toolset._connection_params == stdio_params

  def test_init_with_sse_connection_params(self):
    """Test initialization with SseConnectionParams."""
    sse_params = SseConnectionParams(
        url="https://example.com/mcp", headers={"Authorization": "Bearer token"}
    )
    toolset = McpToolset(connection_params=sse_params)

    assert toolset._connection_params == sse_params

  def test_init_with_streamable_http_params(self):
    """Test initialization with StreamableHTTPConnectionParams."""
    http_params = StreamableHTTPConnectionParams(
        url="https://example.com/mcp",
        headers={"Content-Type": "application/json"},
    )
    toolset = McpToolset(connection_params=http_params)

    assert toolset._connection_params == http_params

  def test_init_with_tool_filter_list(self):
    """Test initialization with tool filter as list."""
    tool_filter = ["tool1", "tool2"]
    toolset = McpToolset(
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

    toolset = McpToolset(
        connection_params=self.mock_stdio_params,
        auth_scheme=auth_scheme,
        auth_credential=auth_credential,
    )

    assert toolset._auth_scheme == auth_scheme
    assert toolset._auth_credential == auth_credential

  def test_init_missing_connection_params(self):
    """Test initialization with missing connection params raises error."""
    with pytest.raises(ValueError, match="Missing connection params"):
      McpToolset(connection_params=None)

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

    toolset = McpToolset(
        connection_params=self.mock_stdio_params, use_mcp_resources=True
    )
    toolset._mcp_session_manager = self.mock_session_manager

    tools = await toolset.get_tools()

    assert len(tools) == 4
    for tool in tools[:3]:
      assert isinstance(tool, MCPTool)
    assert isinstance(tools[3], LoadMcpResourceTool)
    assert tools[0].name == "tool1"
    assert tools[1].name == "tool2"
    assert tools[2].name == "tool3"
    assert tools[3].name == "load_mcp_resource"

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
    toolset = McpToolset(
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

    toolset = McpToolset(
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

    toolset = McpToolset(
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
    toolset = McpToolset(connection_params=self.mock_stdio_params)
    toolset._mcp_session_manager = self.mock_session_manager

    await toolset.close()

    self.mock_session_manager.close.assert_called_once()

  @pytest.mark.asyncio
  async def test_close_with_exception(self):
    """Test cleanup when session manager raises exception."""
    toolset = McpToolset(connection_params=self.mock_stdio_params)
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
    toolset = McpToolset(connection_params=stdio_params)
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
    toolset = McpToolset(connection_params=self.mock_stdio_params)

    # Check that the method has the retry decorator
    assert hasattr(toolset.get_tools, "__wrapped__")

  @pytest.mark.asyncio
  async def test_mcp_toolset_with_prefix(self):
    """Test that McpToolset correctly applies the tool_name_prefix."""
    # Mock the connection parameters
    mock_connection_params = MagicMock()
    mock_connection_params.timeout = None

    # Mock the MCPSessionManager and its create_session method
    mock_session_manager = MagicMock()
    mock_session = MagicMock()

    # Mock the list_tools response from the MCP server
    mock_tool1 = MagicMock()
    mock_tool1.name = "tool1"
    mock_tool1.description = "tool 1 desc"
    mock_tool2 = MagicMock()
    mock_tool2.name = "tool2"
    mock_tool2.description = "tool 2 desc"
    list_tools_result = MagicMock()
    list_tools_result.tools = [mock_tool1, mock_tool2]
    mock_session.list_tools = AsyncMock(return_value=list_tools_result)
    mock_session_manager.create_session = AsyncMock(return_value=mock_session)

    # Create an instance of McpToolset with a prefix
    toolset = McpToolset(
        connection_params=mock_connection_params,
        tool_name_prefix="my_prefix",
        use_mcp_resources=True,
    )

    # Replace the internal session manager with our mock
    toolset._mcp_session_manager = mock_session_manager

    # Get the tools from the toolset
    tools = await toolset.get_tools()

    # The get_tools method in McpToolset returns MCPTool objects, which are
    # instances of BaseTool. The prefixing is handled by the BaseToolset,
    # so we need to call get_tools_with_prefix to get the prefixed tools.
    prefixed_tools = await toolset.get_tools_with_prefix()

    # Assert that the tools are prefixed correctly
    assert len(prefixed_tools) == 3
    assert prefixed_tools[0].name == "my_prefix_tool1"
    assert prefixed_tools[1].name == "my_prefix_tool2"
    assert prefixed_tools[2].name == "my_prefix_load_mcp_resource"

    # Assert that the original tools are not modified
    assert tools[0].name == "tool1"
    assert tools[1].name == "tool2"
    assert tools[2].name == "load_mcp_resource"

  def test_init_with_progress_callback(self):
    """Test initialization with progress_callback."""

    async def my_progress_callback(
        progress: float, total: float | None, message: str | None
    ) -> None:
      pass

    toolset = McpToolset(
        connection_params=self.mock_stdio_params,
        progress_callback=my_progress_callback,
    )

    assert toolset._progress_callback == my_progress_callback

  @pytest.mark.asyncio
  async def test_get_tools_passes_progress_callback_to_mcp_tools(self):
    """Test that get_tools passes progress_callback to created MCPTool instances."""
    progress_updates = []

    async def my_progress_callback(
        progress: float, total: float | None, message: str | None
    ) -> None:
      progress_updates.append((progress, total, message))

    mock_tools = [MockMCPTool("tool1"), MockMCPTool("tool2")]
    self.mock_session.list_tools = AsyncMock(
        return_value=MockListToolsResult(mock_tools)
    )

    toolset = McpToolset(
        connection_params=self.mock_stdio_params,
        progress_callback=my_progress_callback,
    )
    toolset._mcp_session_manager = self.mock_session_manager

    tools = await toolset.get_tools()

    assert len(tools) == 2
    # Verify each tool has the progress_callback set
    for tool in tools:
      assert tool._progress_callback == my_progress_callback

  def test_init_with_progress_callback_factory(self):
    """Test initialization with a ProgressCallbackFactory."""

    def my_callback_factory(tool_name: str, *, readonly_context=None, **kwargs):
      async def callback(
          progress: float, total: float | None, message: str | None
      ) -> None:
        pass

      return callback

    toolset = McpToolset(
        connection_params=self.mock_stdio_params,
        progress_callback=my_callback_factory,
    )

    assert toolset._progress_callback == my_callback_factory

  @pytest.mark.asyncio
  async def test_get_tools_passes_factory_to_mcp_tools(self):
    """Test that get_tools passes factory directly to MCPTool instances.

    The factory is resolved at runtime in McpTool._run_async_impl, not at
    tool creation time. This allows the factory to receive ReadonlyContext.
    """

    def my_callback_factory(tool_name: str, *, readonly_context=None, **kwargs):
      async def callback(
          progress: float, total: float | None, message: str | None
      ) -> None:
        pass

      return callback

    mock_tools = [MockMCPTool("tool1"), MockMCPTool("tool2")]
    self.mock_session.list_tools = AsyncMock(
        return_value=MockListToolsResult(mock_tools)
    )

    toolset = McpToolset(
        connection_params=self.mock_stdio_params,
        progress_callback=my_callback_factory,
    )
    toolset._mcp_session_manager = self.mock_session_manager

    tools = await toolset.get_tools()

    assert len(tools) == 2
    # Factory is passed directly to each tool (resolved at runtime)
    for tool in tools:
      assert tool._progress_callback == my_callback_factory

  @pytest.mark.asyncio
  async def test_list_resources(self):
    """Test listing resources."""
    resources = [
        Resource(
            name="file1.txt", mime_type="text/plain", uri="file:///file1.txt"
        ),
        Resource(
            name="data.json",
            mime_type="application/json",
            uri="file:///data.json",
        ),
    ]
    list_resources_result = ListResourcesResult(resources=resources)
    self.mock_session.list_resources = AsyncMock(
        return_value=list_resources_result
    )

    toolset = McpToolset(connection_params=self.mock_stdio_params)
    toolset._mcp_session_manager = self.mock_session_manager

    result = await toolset.list_resources()

    assert result == ["file1.txt", "data.json"]
    self.mock_session.list_resources.assert_called_once()

  @pytest.mark.asyncio
  async def test_get_resource_info_success(self):
    """Test getting resource info for an existing resource."""
    resources = [
        Resource(
            name="file1.txt", mime_type="text/plain", uri="file:///file1.txt"
        ),
        Resource(
            name="data.json",
            mime_type="application/json",
            uri="file:///data.json",
        ),
    ]
    list_resources_result = ListResourcesResult(resources=resources)
    self.mock_session.list_resources = AsyncMock(
        return_value=list_resources_result
    )

    toolset = McpToolset(connection_params=self.mock_stdio_params)
    toolset._mcp_session_manager = self.mock_session_manager

    result = await toolset.get_resource_info("data.json")

    assert result == {
        "name": "data.json",
        "mime_type": "application/json",
        "uri": "file:///data.json",
    }
    self.mock_session.list_resources.assert_called_once()

  @pytest.mark.asyncio
  async def test_get_resource_info_not_found(self):
    """Test getting resource info for a non-existent resource."""
    resources = [
        Resource(
            name="file1.txt", mime_type="text/plain", uri="file:///file1.txt"
        ),
    ]
    list_resources_result = ListResourcesResult(resources=resources)
    self.mock_session.list_resources = AsyncMock(
        return_value=list_resources_result
    )

    toolset = McpToolset(connection_params=self.mock_stdio_params)
    toolset._mcp_session_manager = self.mock_session_manager

    with pytest.raises(
        ValueError, match="Resource with name 'other.json' not found."
    ):
      await toolset.get_resource_info("other.json")

  @pytest.mark.parametrize(
      "name,mime_type,content,encoding",
      [
          ("file1.txt", "text/plain", "hello world", None),
          (
              "data.json",
              "application/json",
              '{"key": "value"}',
              None,
          ),
          (
              "file1_b64.txt",
              "text/plain",
              base64.b64encode(b"hello world").decode("ascii"),
              "base64",
          ),
          (
              "data_b64.json",
              "application/json",
              base64.b64encode(b'{"key": "value"}').decode("ascii"),
              "base64",
          ),
          (
              "data.bin",
              "application/octet-stream",
              base64.b64encode(b"\x01\x02\x03").decode("ascii"),
              "base64",
          ),
      ],
  )
  @pytest.mark.asyncio
  async def test_read_resource(self, name, mime_type, content, encoding):
    """Test reading various resource types."""
    uri = f"file:///{name}"
    # Mock list_resources for get_resource_info
    resources = [Resource(name=name, mime_type=mime_type, uri=uri)]
    list_resources_result = ListResourcesResult(resources=resources)
    self.mock_session.list_resources = AsyncMock(
        return_value=list_resources_result
    )

    # Mock read_resource
    if encoding == "base64":
      contents = [
          BlobResourceContents(uri=uri, mimeType=mime_type, blob=content)
      ]
    else:
      contents = [
          TextResourceContents(uri=uri, mimeType=mime_type, text=content)
      ]

    read_resource_result = ReadResourceResult(contents=contents)
    self.mock_session.read_resource = AsyncMock(
        return_value=read_resource_result
    )

    toolset = McpToolset(connection_params=self.mock_stdio_params)
    toolset._mcp_session_manager = self.mock_session_manager

    result = await toolset.read_resource(name)

    assert result == contents
    self.mock_session.list_resources.assert_called_once()
    self.mock_session.read_resource.assert_called_once_with(uri=uri)
