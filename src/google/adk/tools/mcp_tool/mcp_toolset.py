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

from __future__ import annotations

import logging
import sys
from typing import List
from typing import Optional
from typing import TextIO
from typing import Union

from pydantic import model_validator
from typing_extensions import override

from ...agents.readonly_context import ReadonlyContext
from ...auth.auth_credential import AuthCredential
from ...auth.auth_schemes import AuthScheme
from ..base_tool import BaseTool
from ..base_toolset import BaseToolset
from ..base_toolset import ToolPredicate
from ..tool_configs import BaseToolConfig
from ..tool_configs import ToolArgsConfig
from .mcp_session_manager import MCPSessionManager
from .mcp_session_manager import retry_on_closed_resource
from .mcp_session_manager import SseConnectionParams
from .mcp_session_manager import StdioConnectionParams
from .mcp_session_manager import StreamableHTTPConnectionParams

# Attempt to import MCP Tool from the MCP library, and hints user to upgrade
# their Python version to 3.10 if it fails.
try:
  from mcp import StdioServerParameters
  from mcp.types import ListToolsResult
except ImportError as e:
  import sys

  if sys.version_info < (3, 10):
    raise ImportError(
        "MCP Tool requires Python 3.10 or above. Please upgrade your Python"
        " version."
    ) from e
  else:
    raise e

from .mcp_tool import MCPTool

logger = logging.getLogger("google_adk." + __name__)


class MCPToolset(BaseToolset):
  """Connects to a MCP Server, and retrieves MCP Tools into ADK Tools.

  This toolset manages the connection to an MCP server and provides tools
  that can be used by an agent. It properly implements the BaseToolset
  interface for easy integration with the agent framework.

  Usage::

    toolset = MCPToolset(
        connection_params=StdioServerParameters(
            command='npx',
            args=["-y", "@modelcontextprotocol/server-filesystem"],
        ),
        tool_filter=['read_file', 'list_directory']  # Optional: filter specific tools
    )

    # Use in an agent
    agent = LlmAgent(
        model='gemini-2.0-flash',
        name='enterprise_assistant',
        instruction='Help user accessing their file systems',
        tools=[toolset],
    )

    # Cleanup is handled automatically by the agent framework
    # But you can also manually close if needed:
    # await toolset.close()
  """

  def __init__(
      self,
      *,
      connection_params: Union[
          StdioServerParameters,
          StdioConnectionParams,
          SseConnectionParams,
          StreamableHTTPConnectionParams,
      ],
      tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
      errlog: TextIO = sys.stderr,
      auth_scheme: Optional[AuthScheme] = None,
      auth_credential: Optional[AuthCredential] = None,
  ):
    """Initializes the MCPToolset.

    Args:
      connection_params: The connection parameters to the MCP server. Can be:
        ``StdioConnectionParams`` for using local mcp server (e.g. using ``npx`` or
        ``python3``); or ``SseConnectionParams`` for a local/remote SSE server; or
        ``StreamableHTTPConnectionParams`` for local/remote Streamable http
        server. Note, ``StdioServerParameters`` is also supported for using local
        mcp server (e.g. using ``npx`` or ``python3`` ), but it does not support
        timeout, and we recommend to use ``StdioConnectionParams`` instead when
        timeout is needed.
      tool_filter: Optional filter to select specific tools. Can be either: - A
        list of tool names to include - A ToolPredicate function for custom
        filtering logic
      errlog: TextIO stream for error logging.
      auth_scheme: The auth scheme of the tool for tool calling
      auth_credential: The auth credential of the tool for tool calling
    """
    super().__init__(tool_filter=tool_filter)

    if not connection_params:
      raise ValueError("Missing connection params in MCPToolset.")

    self._connection_params = connection_params
    self._errlog = errlog

    # Create the session manager that will handle the MCP connection
    self._mcp_session_manager = MCPSessionManager(
        connection_params=self._connection_params,
        errlog=self._errlog,
    )
    self._auth_scheme = auth_scheme
    self._auth_credential = auth_credential

  @retry_on_closed_resource
  async def get_tools(
      self,
      readonly_context: Optional[ReadonlyContext] = None,
  ) -> List[BaseTool]:
    """Return all tools in the toolset based on the provided context.

    Args:
        readonly_context: Context used to filter tools available to the agent.
            If None, all tools in the toolset are returned.

    Returns:
        List[BaseTool]: A list of tools available under the specified context.
    """
    # Get session from session manager
    session = await self._mcp_session_manager.create_session()

    # Fetch available tools from the MCP server
    tools_response: ListToolsResult = await session.list_tools()

    # Apply filtering based on context and tool_filter
    tools = []
    for tool in tools_response.tools:
      mcp_tool = MCPTool(
          mcp_tool=tool,
          mcp_session_manager=self._mcp_session_manager,
          auth_scheme=self._auth_scheme,
          auth_credential=self._auth_credential,
      )

      if self._is_tool_selected(mcp_tool, readonly_context):
        tools.append(mcp_tool)
    return tools

  async def close(self) -> None:
    """Performs cleanup and releases resources held by the toolset.

    This method closes the MCP session and cleans up all associated resources.
    It's designed to be safe to call multiple times and handles cleanup errors
    gracefully to avoid blocking application shutdown.
    """
    try:
      await self._mcp_session_manager.close()
    except Exception as e:
      # Log the error but don't re-raise to avoid blocking shutdown
      print(f"Warning: Error during MCPToolset cleanup: {e}", file=self._errlog)

  @override
  @classmethod
  def from_config(
      cls: type[MCPToolset], config: ToolArgsConfig, config_abs_path: str
  ) -> MCPToolset:
    """Creates an MCPToolset from a configuration object."""
    mcp_toolset_config = MCPToolsetConfig.model_validate(config.model_dump())

    if mcp_toolset_config.stdio_server_params:
      connection_params = mcp_toolset_config.stdio_server_params
    elif mcp_toolset_config.stdio_connection_params:
      connection_params = mcp_toolset_config.stdio_connection_params
    elif mcp_toolset_config.sse_connection_params:
      connection_params = mcp_toolset_config.sse_connection_params
    elif mcp_toolset_config.streamable_http_connection_params:
      connection_params = mcp_toolset_config.streamable_http_connection_params
    else:
      raise ValueError("No connection params found in MCPToolsetConfig.")

    return cls(
        connection_params=connection_params,
        tool_filter=mcp_toolset_config.tool_filter,
        auth_scheme=mcp_toolset_config.auth_scheme,
        auth_credential=mcp_toolset_config.auth_credential,
    )


class MCPToolsetConfig(BaseToolConfig):
  """The config for MCPToolset."""

  stdio_server_params: Optional[StdioServerParameters] = None

  stdio_connection_params: Optional[StdioConnectionParams] = None

  sse_connection_params: Optional[SseConnectionParams] = None

  streamable_http_connection_params: Optional[
      StreamableHTTPConnectionParams
  ] = None

  tool_filter: Optional[List[str]] = None

  auth_scheme: Optional[AuthScheme] = None

  auth_credential: Optional[AuthCredential] = None

  @model_validator(mode="after")
  def _check_only_one_params_field(self):
    param_fields = [
        self.stdio_server_params,
        self.stdio_connection_params,
        self.sse_connection_params,
        self.streamable_http_connection_params,
    ]
    populated_fields = [f for f in param_fields if f is not None]

    if len(populated_fields) != 1:
      raise ValueError(
          "Exactly one of stdio_server_params, stdio_connection_params,"
          " sse_connection_params, streamable_http_connection_params must be"
          " set."
      )
    return self
