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
import itertools
import pickle
import sys
import time
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlowAuthorizationCode
from fastapi.openapi.models import OAuthFlows
from google.adk.agents.context import Context
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import HttpAuth
from google.adk.auth.auth_credential import HttpCredentials
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_tool import AuthConfig
from google.adk.tools.load_mcp_resource_tool import LoadMcpResourceTool
from google.adk.tools.mcp_tool import mcp_toolset as mcp_toolset_module
from google.adk.tools.mcp_tool.mcp_session_manager import _http_debug_var
from google.adk.tools.mcp_tool.mcp_session_manager import _SESSION_IDLE_TTL_SECONDS
from google.adk.tools.mcp_tool.mcp_session_manager import MCPSessionManager
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
from google.adk.tools.mcp_tool.mcp_tool import MCPTool
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_toolset import McpToolsetConfig
from google.adk.tools.tool_configs import ToolArgsConfig
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

  @pytest.fixture
  def allow_config_stdio_servers(self):
    """Opts this process in to stdio MCP servers declared in agent configs."""
    mcp_toolset_module._set_allow_config_stdio_servers(True)
    try:
      yield
    finally:
      mcp_toolset_module._set_allow_config_stdio_servers(None)

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

  def test_connection_params(self):
    """Test getting connection params."""
    toolset = McpToolset(connection_params=self.mock_stdio_params)
    assert toolset.connection_params == self.mock_stdio_params

  def test_auth_scheme(self):
    """Test getting auth scheme."""
    toolset = McpToolset(connection_params=self.mock_stdio_params)
    assert toolset.auth_scheme is None

  def test_auth_credential(self):
    """Test getting auth credential."""
    toolset = McpToolset(connection_params=self.mock_stdio_params)
    assert toolset.auth_credential is None

  def test_error_log(self):
    """Test getting error log."""
    toolset = McpToolset(connection_params=self.mock_stdio_params)
    assert toolset.errlog == sys.stderr

  def test_auth_scheme_with_value(self):
    """Test getting auth scheme when provided at initialization."""
    auth_scheme = OAuth2(
        flows=OAuthFlows(
            authorizationCode=OAuthFlowAuthorizationCode(
                authorizationUrl="https://example.com/auth",
                tokenUrl="https://example.com/token",
                scopes={"read": "Read access"},
            )
        )
    )
    toolset = McpToolset(
        connection_params=self.mock_stdio_params,
        auth_scheme=auth_scheme,
    )
    assert toolset.auth_scheme == auth_scheme

  def test_require_confirmation(self):
    """Test getting require_confirmation flag."""
    toolset = McpToolset(
        connection_params=self.mock_stdio_params,
        require_confirmation=True,
    )
    assert toolset.require_confirmation is True

  def test_header_provider(self):
    """Test getting header_provider."""
    mock_header_provider = Mock()
    toolset = McpToolset(
        connection_params=self.mock_stdio_params,
        header_provider=mock_header_provider,
    )
    assert toolset.header_provider == mock_header_provider

  def test_auth_credential_with_value(self):
    """Test getting auth credential when provided at initialization."""
    mock_credential = Mock(spec=AuthCredential)
    toolset = McpToolset(
        connection_params=self.mock_stdio_params,
        auth_credential=mock_credential,
    )
    assert toolset.auth_credential == mock_credential

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

    # The tool filter is stored on the parent BaseToolset class.
    assert toolset.tool_filter == tool_filter

  def test_init_with_auth(self):
    """Test initialization with authentication."""
    # Create real auth scheme instances

    auth_scheme = OAuth2(flows={})

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

  def test_init_with_auth_and_credential_key(self):
    """Test initialization with authentication and a custom credential_key."""

    auth_scheme = OAuth2(flows={})

    auth_credential = AuthCredential(
        auth_type="oauth2",
        oauth2=OAuth2Auth(client_id="test_id", client_secret="test_secret"),
    )

    toolset = McpToolset(
        connection_params=self.mock_stdio_params,
        auth_scheme=auth_scheme,
        auth_credential=auth_credential,
        credential_key="my_custom_key",
    )

    assert toolset._auth_scheme == auth_scheme
    assert toolset._auth_credential == auth_credential
    assert toolset._auth_config.credential_key == "my_custom_key"

  @pytest.mark.usefixtures("allow_config_stdio_servers")
  def test_from_config_with_credential_key(self):
    """Test that from_config correctly parses credential_key."""

    auth_scheme = OAuth2(flows={})

    config = ToolArgsConfig(
        stdio_server_params=self.mock_stdio_params,
        auth_scheme=auth_scheme,
        credential_key="my_custom_key",
    )
    toolset = McpToolset.from_config(config, "")

    assert isinstance(toolset._auth_scheme, OAuth2)
    assert toolset._auth_config.credential_key == "my_custom_key"

  def test_from_config_rejects_stdio_server_params(self):
    """Config-supplied stdio servers are rejected by default."""
    config = ToolArgsConfig(stdio_server_params=self.mock_stdio_params)

    with pytest.raises(ValueError, match="not allowed in agent configs"):
      McpToolset.from_config(config, "")

  def test_from_config_rejects_stdio_connection_params(self):
    """The stdio_connection_params spelling is rejected the same way."""
    config = ToolArgsConfig(
        stdio_connection_params=StdioConnectionParams(
            server_params=self.mock_stdio_params
        )
    )

    with pytest.raises(ValueError, match="not allowed in agent configs"):
      McpToolset.from_config(config, "")

  def test_from_config_rejection_names_the_env_var(self):
    """The error tells the operator how to opt in."""
    config = ToolArgsConfig(stdio_server_params=self.mock_stdio_params)

    with pytest.raises(
        ValueError, match=mcp_toolset_module.ALLOW_CONFIG_STDIO_SERVERS_ENV_VAR
    ):
      McpToolset.from_config(config, "")

  def test_from_config_allows_stdio_when_env_var_set(self, monkeypatch):
    """The environment variable opts a whole process in."""
    monkeypatch.setenv(
        mcp_toolset_module.ALLOW_CONFIG_STDIO_SERVERS_ENV_VAR, "1"
    )
    config = ToolArgsConfig(stdio_server_params=self.mock_stdio_params)

    toolset = McpToolset.from_config(config, "")

    assert isinstance(toolset, McpToolset)

  @pytest.mark.usefixtures("allow_config_stdio_servers")
  def test_from_config_allows_stdio_when_set_programmatically(self):
    """An embedding application can opt in without touching the environment."""
    config = ToolArgsConfig(stdio_server_params=self.mock_stdio_params)

    toolset = McpToolset.from_config(config, "")

    assert isinstance(toolset, McpToolset)

  def test_programmatic_setting_overrides_env_var(self, monkeypatch):
    """An explicit False wins over an environment variable that says yes."""
    monkeypatch.setenv(
        mcp_toolset_module.ALLOW_CONFIG_STDIO_SERVERS_ENV_VAR, "1"
    )
    monkeypatch.setattr(
        mcp_toolset_module, "_allow_config_stdio_servers", False
    )
    config = ToolArgsConfig(stdio_server_params=self.mock_stdio_params)

    with pytest.raises(ValueError, match="not allowed in agent configs"):
      McpToolset.from_config(config, "")

  def test_from_config_allows_remote_connection_params(self):
    """Remote MCP servers are unaffected: they launch no local process."""
    config = ToolArgsConfig(
        sse_connection_params=SseConnectionParams(url="https://example.com/sse")
    )

    toolset = McpToolset.from_config(config, "")

    assert isinstance(toolset, McpToolset)

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
  async def test_get_tools_returns_sorted_by_name(self):
    """Test that get_tools returns tools sorted by name for cache stability."""
    # Mock tools from MCP server in non-alphabetical order.
    mock_tools = [
        MockMCPTool("charlie"),
        MockMCPTool("alpha"),
        MockMCPTool("bravo"),
    ]
    self.mock_session.list_tools = AsyncMock(
        return_value=MockListToolsResult(mock_tools)
    )

    toolset = McpToolset(connection_params=self.mock_stdio_params)
    toolset._mcp_session_manager = self.mock_session_manager

    tools = await toolset.get_tools()

    assert [tool.name for tool in tools] == ["alpha", "bravo", "charlie"]

  @pytest.mark.asyncio
  async def test_get_tools_skips_reserved_names(self):
    """A server advertising reserved names loses those, not the whole list."""
    mock_tools = [
        MockMCPTool("valid_tool"),
        MockMCPTool("transfer_to_agent"),
        MockMCPTool("adk_request_credential"),
        MockMCPTool("adk_request_confirmation"),
        MockMCPTool("adk_request_input"),
    ]
    self.mock_session.list_tools = AsyncMock(
        return_value=MockListToolsResult(mock_tools)
    )

    toolset = McpToolset(connection_params=self.mock_stdio_params)
    toolset._mcp_session_manager = self.mock_session_manager

    tools = await toolset.get_tools()

    assert [tool.name for tool in tools] == ["valid_tool"]

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
  async def test_get_tools_with_async_header_provider(self):
    """Test get_tools with an async header_provider."""
    mock_tools = [MockMCPTool("tool1"), MockMCPTool("tool2")]
    self.mock_session.list_tools = AsyncMock(
        return_value=MockListToolsResult(mock_tools)
    )
    mock_readonly_context = Mock(spec=ReadonlyContext)
    expected_headers = {"X-Tenant-ID": "test-tenant"}

    async def header_provider(_context):
      return expected_headers

    toolset = McpToolset(
        connection_params=self.mock_stdio_params,
        header_provider=header_provider,
    )
    toolset._mcp_session_manager = self.mock_session_manager

    tools = await toolset.get_tools(readonly_context=mock_readonly_context)

    assert len(tools) == 2
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

    # Should not raise exception, should log the warning
    await toolset.close()

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
            name="file1.txt", mimeType="text/plain", uri="file:///file1.txt"
        ),
        Resource(
            name="data.json",
            mimeType="application/json",
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
            name="file1.txt", mimeType="text/plain", uri="file:///file1.txt"
        ),
        Resource(
            name="data.json",
            mimeType="application/json",
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
        "mimeType": "application/json",
        "uri": "file:///data.json",
    }
    self.mock_session.list_resources.assert_called_once()

  @pytest.mark.asyncio
  async def test_get_resource_info_keeps_the_1x_key_names(self):
    """This dict goes straight to the caller, so its keys are contractual.

    2.x renames `mimeType` the way it renamed `isError`, and nothing else
    reads it, so a rename here is silent all the way out. `meta` has to
    survive the alias dump that prevents that.
    """
    resources = [
        Resource(
            name="data.json",
            mimeType="application/json",
            uri="file:///data.json",
            _meta={"trace": "t"},
        )
    ]
    list_resources_result = ListResourcesResult(resources=resources)
    self.mock_session.list_resources = AsyncMock(
        return_value=list_resources_result
    )

    toolset = McpToolset(connection_params=self.mock_stdio_params)
    toolset._mcp_session_manager = self.mock_session_manager

    result = await toolset.get_resource_info("data.json")

    assert result["mimeType"] == "application/json"
    assert "mime_type" not in result
    assert result["meta"] == {"trace": "t"}
    assert "_meta" not in result

  @pytest.mark.asyncio
  async def test_get_resource_info_not_found(self):
    """Test getting resource info for a non-existent resource."""
    resources = [
        Resource(
            name="file1.txt", mimeType="text/plain", uri="file:///file1.txt"
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
    resources = [Resource(name=name, mimeType=mime_type, uri=uri)]
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

  @pytest.mark.asyncio
  async def test_sampling_callback_invoked(self):

    called = {"value": False}

    async def mock_sampling_handler(messages, params=None, context=None):
      called["value"] = True

      assert isinstance(messages, list)
      assert messages[0]["role"] == "user"

      return {
          "model": "test-model",
          "role": "assistant",
          "content": {"type": "text", "text": "sampling response"},
          "stopReason": "endTurn",
      }

    toolset = McpToolset(
        connection_params=StreamableHTTPConnectionParams(
            url="http://localhost:9999",
            timeout=10,
        ),
        sampling_callback=mock_sampling_handler,
    )

    messages = [{"role": "user", "content": {"type": "text", "text": "hello"}}]

    result = await toolset._sampling_callback(messages)

    assert called["value"] is True
    assert result["role"] == "assistant"
    assert result["content"]["text"] == "sampling response"

  @pytest.mark.asyncio
  async def test_elicitation_callback_plumbed_to_session_manager(self):
    """Elicitation callback reaches the session manager unchanged."""

    # pylint: disable=protected-access
    async def mock_elicitation_handler(context, params):
      del context, params
      return {"action": "decline"}

    toolset = McpToolset(
        connection_params=StreamableHTTPConnectionParams(
            url="http://localhost:9999",
            timeout=10,
        ),
        elicitation_callback=mock_elicitation_handler,
    )
    assert toolset._elicitation_callback is mock_elicitation_handler
    assert (
        toolset._mcp_session_manager._elicitation_callback
        is mock_elicitation_handler
    )
    # pylint: enable=protected-access

  @pytest.mark.asyncio
  async def test_elicitation_callback_defaults_to_none(self):
    # pylint: disable=protected-access
    toolset = McpToolset(connection_params=self.mock_stdio_params)
    assert toolset._elicitation_callback is None
    assert toolset._mcp_session_manager._elicitation_callback is None
    # pylint: enable=protected-access

  @pytest.mark.asyncio
  async def test_get_auth_headers_includes_additional_headers(self):
    credential = AuthCredential(
        auth_type=AuthCredentialTypes.HTTP,
        http=HttpAuth(
            scheme="bearer",
            credentials=HttpCredentials(token="token"),
            additional_headers={"X-API-Key": "secret"},
        ),
    )
    auth_config = AuthConfig(
        auth_scheme=OAuth2(flows={}),
        raw_auth_credential=credential,
    )
    auth_config.exchanged_auth_credential = credential
    toolset = McpToolset(connection_params=self.mock_stdio_params)
    toolset._auth_config = auth_config

    headers = toolset._get_auth_headers()

    assert headers["Authorization"] == "Bearer token"
    assert headers["X-API-Key"] == "secret"

  def test_pickle_mcp_toolset(self):
    toolset = McpToolset(connection_params=self.mock_stdio_params)
    pickled = pickle.dumps(toolset)
    unpickled = pickle.loads(pickled)
    assert unpickled._connection_params == self.mock_stdio_params
    assert unpickled._errlog == sys.stderr


class TestMcpToolsetHttpDebug:
  """Tests that McpToolset._execute_with_session captures HTTP debug info based on context mutability."""

  @pytest.mark.asyncio
  @patch(
      "google.adk.tools.mcp_tool.mcp_toolset.logger.isEnabledFor",
      return_value=True,
  )
  async def test_execute_with_session_captures_http_debug_when_context_is_mutable(
      self, mock_is_enabled
  ):
    mock_session_manager = MagicMock(spec=MCPSessionManager)
    mock_session = AsyncMock()
    mock_session_manager.create_session.return_value = mock_session

    toolset = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(command="mock"), timeout=5
        )
    )
    toolset._mcp_session_manager = mock_session_manager

    # Mock Context (mutable)
    mock_invocation_context = Mock(spec=InvocationContext)
    mock_invocation_context._custom_metadata = {}
    mock_ctx_session = Mock()
    mock_ctx_session.state = {}
    mock_invocation_context.session = mock_ctx_session
    context = Context(mock_invocation_context)

    async def dummy_coro(session):
      debug_list = _http_debug_var.get(None)
      if debug_list is not None:
        debug_list.append(
            {"url": "https://example.com/api", "status_code": 200}
        )
      return "done"

    res = await toolset._execute_with_session(
        dummy_coro, "error", readonly_context=context
    )
    assert res == "done"

    assert "http_debug_info" in context.custom_metadata
    debug_info = context.custom_metadata["http_debug_info"]
    assert len(debug_info) == 1
    assert debug_info[0]["url"] == "https://example.com/api"
    assert debug_info[0]["status_code"] == 200

  @pytest.mark.asyncio
  @patch(
      "google.adk.tools.mcp_tool.mcp_toolset.logger.isEnabledFor",
      return_value=True,
  )
  async def test_execute_with_session_captures_http_debug_when_context_is_readonly(
      self, mock_is_enabled
  ):
    mock_session_manager = MagicMock(spec=MCPSessionManager)
    mock_session = AsyncMock()
    mock_session_manager.create_session.return_value = mock_session

    toolset = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(command="mock"), timeout=5
        )
    )
    toolset._mcp_session_manager = mock_session_manager

    # Mock ReadonlyContext (read-only)
    mock_invocation_context = Mock(spec=InvocationContext)
    mock_invocation_context._custom_metadata = {}
    mock_ctx_session = Mock()
    mock_ctx_session.state = {}
    mock_invocation_context.session = mock_ctx_session
    context = ReadonlyContext(mock_invocation_context)

    async def dummy_coro(session):
      debug_list = _http_debug_var.get(None)
      if debug_list is not None:
        debug_list.append(
            {"url": "https://example.com/api", "status_code": 200}
        )
      return "done"

    res = await toolset._execute_with_session(
        dummy_coro, "error", readonly_context=context
    )
    assert res == "done"

    assert "http_debug_info" in context.custom_metadata
    debug_info = context.custom_metadata["http_debug_info"]
    assert len(debug_info) == 1
    assert debug_info[0]["url"] == "https://example.com/api"
    assert debug_info[0]["status_code"] == 200


class TestMcpToolsetConfig:
  """Test suite for the McpToolsetConfig connection-params validator."""

  def _stdio_server_params(self):
    return StdioServerParameters(command="test_command", args=[])

  def test_no_connection_params_is_rejected(self):
    """A toolset with no transport configured cannot connect to anything."""
    with pytest.raises(ValueError, match="Exactly one of"):
      McpToolsetConfig()

  def test_two_connection_params_are_rejected(self):
    """The transports are mutually exclusive; two of them is ambiguous."""
    with pytest.raises(ValueError, match="Exactly one of"):
      McpToolsetConfig(
          stdio_server_params=self._stdio_server_params(),
          sse_connection_params=SseConnectionParams(
              url="https://example.com/mcp"
          ),
      )

  def test_stdio_server_params_alone_is_accepted(self):
    config = McpToolsetConfig(stdio_server_params=self._stdio_server_params())

    assert config.stdio_server_params.command == "test_command"
    assert config.stdio_connection_params is None
    assert config.sse_connection_params is None
    assert config.streamable_http_connection_params is None

  def test_stdio_connection_params_alone_is_accepted(self):
    config = McpToolsetConfig(
        stdio_connection_params=StdioConnectionParams(
            server_params=self._stdio_server_params(), timeout=10.0
        )
    )

    assert config.stdio_connection_params.timeout == 10.0

  def test_sse_connection_params_alone_is_accepted(self):
    config = McpToolsetConfig(
        sse_connection_params=SseConnectionParams(url="https://example.com/mcp")
    )

    assert config.sse_connection_params.url == "https://example.com/mcp"

  def test_streamable_http_connection_params_alone_is_accepted(self):
    config = McpToolsetConfig(
        streamable_http_connection_params=StreamableHTTPConnectionParams(
            url="https://example.com/mcp"
        )
    )

    assert (
        config.streamable_http_connection_params.url
        == "https://example.com/mcp"
    )

  def test_non_transport_fields_do_not_satisfy_the_validator(self):
    """Auth/filter fields are not transports and cannot stand in for one."""
    with pytest.raises(ValueError, match="Exactly one of"):
      McpToolsetConfig(tool_filter=["tool1"], credential_key="key")

  def test_use_mcp_resources_defaults_to_false(self):
    config = McpToolsetConfig(stdio_server_params=self._stdio_server_params())

    assert config.use_mcp_resources is False


class TestMcpToolsetToolListCache:
  """Test suite for reusing the MCP server's tools/list response."""

  # The cache and its session manager are internal state that these tests
  # substitute and assert on directly.
  # pylint: disable=protected-access

  def setup_method(self):
    """Set up a toolset whose session manager keys sessions by headers."""
    self.mock_stdio_params = StdioServerParameters(
        command="test_command", args=[]
    )
    self.mock_session = AsyncMock()
    self.mock_session.list_tools = AsyncMock(
        return_value=MockListToolsResult(
            [MockMCPTool("tool1"), MockMCPTool("tool2")]
        )
    )
    self.mock_session_manager = Mock(spec=MCPSessionManager)
    self.mock_session_manager.create_session = AsyncMock(
        return_value=self.mock_session
    )
    self.mock_session_manager._session_key_for = Mock(
        side_effect=lambda headers=None: repr(sorted((headers or {}).items()))
    )

  def _toolset(self, **kwargs) -> McpToolset:
    toolset = McpToolset(connection_params=self.mock_stdio_params, **kwargs)
    toolset._mcp_session_manager = self.mock_session_manager
    return toolset

  @pytest.mark.asyncio
  async def test_tool_list_is_not_cached_by_default(self):
    """Without an explicit TTL the server is still listed on every call."""
    toolset = self._toolset()

    await toolset.get_tools()
    await toolset.get_tools()

    assert self.mock_session.list_tools.await_count == 2
    self.mock_session_manager._session_key_for.assert_not_called()

  @pytest.mark.asyncio
  async def test_second_call_reuses_the_cached_tool_list(self):
    """A second call within the TTL serves the same tools without listing."""
    toolset = self._toolset(tool_list_cache_ttl_seconds=60)

    first = await toolset.get_tools()
    second = await toolset.get_tools()

    assert self.mock_session.list_tools.await_count == 1
    assert [tool.name for tool in first] == ["tool1", "tool2"]
    assert [tool.name for tool in second] == ["tool1", "tool2"]

  @pytest.mark.asyncio
  async def test_expired_entry_is_refetched(self):
    """Once the TTL lapses the server is consulted again."""
    toolset = self._toolset(tool_list_cache_ttl_seconds=60)

    await toolset.get_tools()
    for entry in toolset._tool_list_cache.values():
      entry.expires_at = time.monotonic() - 1
    await toolset.get_tools()

    assert self.mock_session.list_tools.await_count == 2

  @pytest.mark.asyncio
  async def test_different_identities_do_not_share_a_cache_entry(self):
    """Tools listed for one tenant are never served to another."""
    headers = {"X-Tenant-ID": "tenant-a"}
    toolset = self._toolset(
        tool_list_cache_ttl_seconds=60,
        header_provider=lambda _context: dict(headers),
    )
    context = Mock(spec=ReadonlyContext)

    await toolset.get_tools(readonly_context=context)
    headers["X-Tenant-ID"] = "tenant-b"
    await toolset.get_tools(readonly_context=context)
    headers["X-Tenant-ID"] = "tenant-a"
    await toolset.get_tools(readonly_context=context)

    # Two listings for two tenants; the third call reuses tenant-a's entry.
    assert self.mock_session.list_tools.await_count == 2

  @pytest.mark.asyncio
  async def test_tool_filter_still_runs_on_a_cache_hit(self):
    """Caching skips the round trip, not the context-dependent filtering."""
    allowed = {"tool1"}
    toolset = self._toolset(
        tool_list_cache_ttl_seconds=60,
        tool_filter=lambda tool, _context: tool.name in allowed,
    )

    first = await toolset.get_tools()
    allowed.clear()
    allowed.add("tool2")
    second = await toolset.get_tools()

    assert self.mock_session.list_tools.await_count == 1
    assert [tool.name for tool in first] == ["tool1"]
    assert [tool.name for tool in second] == ["tool2"]

  @pytest.mark.asyncio
  async def test_close_clears_the_cache(self):
    """Closing the toolset drops tool lists along with the sessions."""
    toolset = self._toolset(tool_list_cache_ttl_seconds=60)
    await toolset.get_tools()
    assert toolset._tool_list_cache

    await toolset.close()

    assert not toolset._tool_list_cache

  @pytest.mark.asyncio
  async def test_pickled_state_drops_the_cache(self):
    """Cache keys name sessions that do not survive pickling."""
    toolset = self._toolset(tool_list_cache_ttl_seconds=60)
    await toolset.get_tools()
    assert toolset._tool_list_cache

    assert not toolset.__getstate__()["_tool_list_cache"]

  @pytest.mark.parametrize("ttl", [0, -1])
  def test_non_positive_ttl_is_rejected(self, ttl):
    """A zero or negative TTL is a mistake, not a way to disable caching."""
    with pytest.raises(ValueError, match="must be positive"):
      McpToolset(
          connection_params=self.mock_stdio_params,
          tool_list_cache_ttl_seconds=ttl,
      )

  @pytest.mark.asyncio
  async def test_expired_entries_for_other_keys_are_swept(self):
    """A key that never comes back is still reclaimed.

    A read only evicts the key it was asked for, so a `header_provider` that
    mints a fresh value per request would otherwise grow the cache forever.
    """
    counter = itertools.count()
    toolset = self._toolset(
        tool_list_cache_ttl_seconds=60,
        header_provider=lambda _context: {"X-Request-ID": str(next(counter))},
    )
    context = Mock(spec=ReadonlyContext)

    await toolset.get_tools(readonly_context=context)
    for entry in toolset._tool_list_cache.values():
      entry.expires_at = time.monotonic() - 1
    await toolset.get_tools(readonly_context=context)

    # The first key expired and was swept even though it was never read again.
    assert len(toolset._tool_list_cache) == 1

  @pytest.mark.asyncio
  async def test_unexpired_entries_are_capped(self):
    """The cap holds even when every key is still inside its TTL."""
    counter = itertools.count()
    toolset = self._toolset(
        tool_list_cache_ttl_seconds=3600,
        header_provider=lambda _context: {"X-Request-ID": str(next(counter))},
    )
    context = Mock(spec=ReadonlyContext)

    for _ in range(mcp_toolset_module._MAX_TOOL_LIST_CACHE_ENTRIES + 10):
      await toolset.get_tools(readonly_context=context)

    assert (
        len(toolset._tool_list_cache)
        == mcp_toolset_module._MAX_TOOL_LIST_CACHE_ENTRIES
    )

  @pytest.mark.asyncio
  async def test_the_cap_evicts_the_least_recently_used_entry(self):
    """A key that keeps being read survives a flood of one-shot keys."""
    headers = {"X-Tenant-ID": "keeper"}
    toolset = self._toolset(
        tool_list_cache_ttl_seconds=3600,
        header_provider=lambda _context: dict(headers),
    )
    context = Mock(spec=ReadonlyContext)
    await toolset.get_tools(readonly_context=context)
    keeper_key = next(iter(toolset._tool_list_cache))

    for i in range(mcp_toolset_module._MAX_TOOL_LIST_CACHE_ENTRIES - 1):
      headers["X-Tenant-ID"] = f"one-shot-{i}"
      await toolset.get_tools(readonly_context=context)
      # Touch the keeper so it stays the most recently used entry.
      headers["X-Tenant-ID"] = "keeper"
      await toolset.get_tools(readonly_context=context)

    headers["X-Tenant-ID"] = "overflow"
    await toolset.get_tools(readonly_context=context)

    assert keeper_key in toolset._tool_list_cache
    assert (
        len(toolset._tool_list_cache)
        == mcp_toolset_module._MAX_TOOL_LIST_CACHE_ENTRIES
    )


class TestMcpToolsetSessionInUse:
  """Tests that a session in use is held out of the pool's idle sweep."""

  @pytest.mark.asyncio
  async def test_execute_with_session_is_not_swept_mid_call(self):
    """A toolset call in flight must not have its session torn down."""
    toolset = McpToolset(
        connection_params=StreamableHTTPConnectionParams(
            url="http://example.com/mcp"
        )
    )
    manager = toolset._mcp_session_manager
    session_key = manager._generate_session_key(manager._merge_headers(None))

    pooled_session = Mock()
    pooled_session._read_stream = Mock(_closed=False)
    pooled_session._write_stream = Mock(_closed=False)
    exit_stack = AsyncMock()
    manager._sessions[session_key] = (
        pooled_session,
        exit_stack,
        asyncio.get_running_loop(),
    )

    call_started = asyncio.Event()
    finish_call = asyncio.Event()

    async def slow_coro(session):
      call_started.set()
      await finish_call.wait()
      return "done"

    call = asyncio.ensure_future(
        toolset._execute_with_session(slow_coro, "error")
    )
    await asyncio.wait_for(call_started.wait(), timeout=5.0)

    # The call has outlived the idle TTL and an unrelated caller sweeps.
    manager._session_last_used[session_key] = (
        time.monotonic() - 10 * _SESSION_IDLE_TTL_SECONDS
    )
    manager._evict_idle_sessions(keep_key="key_of_another_caller")

    assert session_key in manager._sessions
    exit_stack.aclose.assert_not_called()

    finish_call.set()
    assert await asyncio.wait_for(call, timeout=5.0) == "done"
    assert (
        time.monotonic() - manager._session_last_used[session_key]
        < _SESSION_IDLE_TTL_SECONDS
    )
