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

from contextlib import AsyncExitStack
import sys
from types import TracebackType
from typing import List
from typing import Optional
from typing import override
from typing import TextIO
from typing import Type

from ...agents.readonly_context import ReadonlyContext
from ..base_toolset import BaseToolPredicate
from ..base_toolset import BaseToolset
from .mcp_session_manager import MCPSessionManager
from .mcp_session_manager import retry_on_closed_resource
from .mcp_session_manager import SseServerParams

# Attempt to import MCP Tool from the MCP library, and hints user to upgrade
# their Python version to 3.10 if it fails.
try:
  from mcp import ClientSession
  from mcp import StdioServerParameters
  from mcp.types import ListToolsResult
except ImportError as e:
  import sys

  if sys.version_info < (3, 10):
    raise ImportError(
        'MCP Tool requires Python 3.10 or above. Please upgrade your Python'
        ' version.'
    ) from e
  else:
    raise e

from .mcp_tool import MCPTool


class MCPToolset(BaseToolset):
  """Connects to a MCP Server, and retrieves MCP Tools into ADK Tools.

  Usage:
  ```
  root_agent = LlmAgent(
      tools=MCPToolset(
          connection_params=StdioServerParameters(
              command='npx',
              args=["-y", "@modelcontextprotocol/server-filesystem"],
          )
      )
  )
  ```
  """

  def __init__(
      self,
      *,
      connection_params: StdioServerParameters | SseServerParams,
      errlog: TextIO = sys.stderr,
      tool_predicate: Optional[BaseToolPredicate] = None,
  ):
    """Initializes the MCPToolset.

    Args:
      connection_params: The connection parameters to the MCP server. Can be:
        `StdioServerParameters` for using local mcp server (e.g. using `npx` or
        `python3`); or `SseServerParams` for a local/remote SSE server.
    """

    if not connection_params:
      raise ValueError('Missing connection params in MCPToolset.')
    self.connection_params = connection_params
    self.errlog = errlog
    self.exit_stack = AsyncExitStack()

    self.session_manager = MCPSessionManager(
        connection_params=self.connection_params,
        exit_stack=self.exit_stack,
        errlog=self.errlog,
    )
    self.session = None
    self.tool_predicate = tool_predicate

  async def _initialize(self) -> ClientSession:
    """Connects to the MCP Server and initializes the ClientSession."""
    self.session = await self.session_manager.create_session()
    return self.session

  @override
  async def close(self):
    """Closes the connection to MCP Server."""
    await self.exit_stack.aclose()

  @retry_on_closed_resource('_initialize')
  @override
  async def get_tools(
      self,
      readony_context: ReadonlyContext = None,
  ) -> List[MCPTool]:
    """Loads all tools from the MCP Server.

    Returns:
      A list of MCPTools imported from the MCP Server.
    """
    if not self.session:
      await self._initialize()
    tools_response: ListToolsResult = await self.session.list_tools()
    return [
        MCPTool(
            mcp_tool=tool,
            mcp_session=self.session,
            mcp_session_manager=self.session_manager,
        )
        for tool in tools_response.tools
        if self.tool_predicate is None
        or self.tool_predicate(tool, readony_context)
    ]
