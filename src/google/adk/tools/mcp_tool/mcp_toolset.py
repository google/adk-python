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
from typing import List, Optional, TextIO, Tuple, Type

from .mcp_session_manager import MCPSessionManager, SseServerParams, retry_on_closed_resource

# Attempt to import MCP Tool from the MCP library, and hints user to upgrade
# their Python version to 3.10 if it fails.
try:
  from mcp import ClientSession, StdioServerParameters
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


class MCPToolset:
  """Connects to a MCP Server, and retrieves MCP Tools into ADK Tools.

  Usage:
  Example 1: (using from_server helper):
  ```
  async def load_tools():
    return await MCPToolset.from_server(
      connection_params=StdioServerParameters(
          command='npx',
          args=["-y", "@modelcontextprotocol/server-filesystem"],
          )
    )

  # Use the tools in an LLM agent
  tools, exit_stack = await load_tools()
  agent = LlmAgent(
      tools=tools
  )
  ...
  await exit_stack.aclose()
  ```

  Example 2: (using `async with`):

  ```
  async def load_tools():
    async with MCPToolset(
      connection_params=SseServerParams(url="http://0.0.0.0:8090/sse")
    ) as toolset:
      tools = await toolset.load_tools()

      agent = LlmAgent(
          ...
          tools=tools
      )
  ```

  Example 3: (provide AsyncExitStack):
  ```
  async def load_tools():
    async_exit_stack = AsyncExitStack()
    toolset = MCPToolset(
      connection_params=StdioServerParameters(...),
    )
    async_exit_stack.enter_async_context(toolset)
    tools = await toolset.load_tools()
    agent = LlmAgent(
        ...
        tools=tools
    )
    ...
    await async_exit_stack.aclose()

  ```

  Attributes:
    connection_params: The connection parameters to the MCP server. Can be
      either `StdioServerParameters` or `SseServerParams`.
    exit_stack: The async exit stack to manage the connection to the MCP server.
    session: The MCP session being initialized with the connection.
  """

  def __init__(
      self,
      *,
      connection_params: StdioServerParameters | SseServerParams,
      errlog: TextIO = sys.stderr,
      exit_stack=AsyncExitStack(),
  ):
    """Initializes the MCPToolset.

    Usage:
    Example 1: (using from_server helper):
    ```
    async def load_tools():
      return await MCPToolset.from_server(
        connection_params=StdioServerParameters(
            command='npx',
            args=["-y", "@modelcontextprotocol/server-filesystem"],
            )
      )

    # Use the tools in an LLM agent
    tools, exit_stack = await load_tools()
    agent = LlmAgent(
        tools=tools
    )
    ...
    await exit_stack.aclose()
    ```

    Example 2: (using `async with`):

    ```
    async def load_tools():
      async with MCPToolset(
        connection_params=SseServerParams(url="http://0.0.0.0:8090/sse")
      ) as toolset:
        tools = await toolset.load_tools()

        agent = LlmAgent(
            ...
            tools=tools
        )
    ```

    Example 3: (provide AsyncExitStack):
    ```
    async def load_tools():
      async_exit_stack = AsyncExitStack()
      toolset = MCPToolset(
        connection_params=StdioServerParameters(...),
      )
      async_exit_stack.enter_async_context(toolset)
      tools = await toolset.load_tools()
      agent = LlmAgent(
          ...
          tools=tools
      )
      ...
      await async_exit_stack.aclose()

    ```

    Args:
      connection_params: The connection parameters to the MCP server. Can be:
        `StdioServerParameters` for using local mcp server (e.g. using `npx` or
        `python3`); or `SseServerParams` for a local/remote SSE server.
    """
    if not connection_params:
      raise ValueError('Missing connection params in MCPToolset.')
    self.connection_params = connection_params
    self.errlog = errlog
    self.exit_stack = exit_stack

    self.session_manager = MCPSessionManager(
        connection_params=self.connection_params,
        exit_stack=self.exit_stack,
        errlog=self.errlog,
    )

  @classmethod
  async def from_server(
      cls,
      *,
      connection_params: StdioServerParameters | SseServerParams,
      async_exit_stack: Optional[AsyncExitStack] = None,
      errlog: TextIO = sys.stderr,
  ) -> Tuple[List[MCPTool], AsyncExitStack]:
    """Retrieve all tools from the MCP connection.

    Usage:
    ```
    async def load_tools():
      tools, exit_stack = await MCPToolset.from_server(
        connection_params=StdioServerParameters(
            command='npx',
            args=["-y", "@modelcontextprotocol/server-filesystem"],
        )
      )
    ```

    Args:
      connection_params: The connection parameters to the MCP server.
      async_exit_stack: The async exit stack to use. If not provided, a new
        AsyncExitStack will be created.

    Returns:
      A tuple of the list of MCPTools and the AsyncExitStack.
      - tools: The list of MCPTools.
      - async_exit_stack: The AsyncExitStack used to manage the connection to
        the MCP server. Use `await async_exit_stack.aclose()` to close the
        connection when server shuts down.
    """
    async_exit_stack = async_exit_stack or AsyncExitStack()
    toolset = cls(
        connection_params=connection_params,
        exit_stack=async_exit_stack,
        errlog=errlog,
    )

    await async_exit_stack.enter_async_context(toolset)
    tools = await toolset.load_tools()
    return (tools, async_exit_stack)

  async def _initialize(self) -> ClientSession:
    """Connects to the MCP Server and initializes the ClientSession."""
    self.session = await self.session_manager.create_session()
    return self.session

  async def _exit(self):
    """Closes the connection to MCP Server."""
    await self.exit_stack.aclose()

  @retry_on_closed_resource('_initialize')
  async def load_tools(self) -> List[MCPTool]:
    """Loads all tools from the MCP Server.

    Returns:
      A list of MCPTools imported from the MCP Server.
    """
    tools_response: ListToolsResult = await self.session.list_tools()
    return [
        MCPTool(
            mcp_tool=tool,
            mcp_session=self.session,
            mcp_session_manager=self.session_manager,
        )
        for tool in tools_response.tools
    ]

  async def __aenter__(self):
    try:
      await self._initialize()
      return self
    except Exception as e:
      raise e

  async def __aexit__(
      self,
      exc_type: Optional[Type[BaseException]],
      exc: Optional[BaseException],
      tb: Optional[TracebackType],
  ) -> None:
    await self._exit()
