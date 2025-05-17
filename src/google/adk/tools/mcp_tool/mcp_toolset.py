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
from contextlib import AsyncExitStack
import logging
import os
import signal
import sys
from typing import List
from typing import Optional
from typing import TextIO
from typing import Union

from typing_extensions import override

from ...agents.readonly_context import ReadonlyContext
from ..base_tool import BaseTool
from ..base_toolset import BaseToolset
from ..base_toolset import ToolPredicate
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
        "MCP Tool requires Python 3.10 or above. Please upgrade your Python"
        " version."
    ) from e
  else:
    raise e

from .mcp_tool import MCPTool

logger = logging.getLogger("google_adk." + __name__)


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
      tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
  ):
    """Initializes the MCPToolset.

    Args:
      connection_params: The connection parameters to the MCP server. Can be:
        `StdioServerParameters` for using local mcp server (e.g. using `npx` or
        `python3`); or `SseServerParams` for a local/remote SSE server.
      errlog: (Optional) TextIO stream for error logging. Use only for
        initializing a local stdio MCP session.
    """

    if not connection_params:
      raise ValueError("Missing connection params in MCPToolset.")
    self._connection_params = connection_params
    self._errlog = errlog
    self._exit_stack = AsyncExitStack()
    self._creator_task_id = None
    self._process_pid = None  # Store the subprocess PID

    self._session_manager = MCPSessionManager(
        connection_params=self._connection_params,
        exit_stack=self._exit_stack,
        errlog=self._errlog,
    )
    self._session = None
    self.tool_filter = tool_filter
    self._initialized = False

  async def _initialize(self) -> ClientSession:
    """Connects to the MCP Server and initializes the ClientSession."""
    # Store the current task ID when initializing
    self._creator_task_id = id(asyncio.current_task())
    self._session, process = await self._session_manager.create_session()
    # Store the process PID if available
    if process and hasattr(process, "pid"):
      self._process_pid = process.pid
    self._initialized = True
    return self._session

  def _is_selected(
      self, tool: BaseTool, readonly_context: Optional[ReadonlyContext]
  ) -> bool:
    """Checks if a tool should be selected based on the tool filter."""
    if self.tool_filter is None:
      return True
    if isinstance(self.tool_filter, ToolPredicate):
      return self.tool_filter(tool, readonly_context)
    if isinstance(self.tool_filter, list):
      return tool.name in self.tool_filter
    return False

  @override
  async def close(self):
    """Safely closes the connection to MCP Server with guaranteed resource cleanup."""
    if not self._initialized:
      return  # Nothing to close

    logger.info("Closing MCP Toolset")

    # Step 1: Try graceful shutdown of the session if it exists
    if self._session:
      try:
        logger.info("Attempting graceful session shutdown")
        await self._session.shutdown()
      except Exception as e:
        logger.warning(f"Session shutdown error (continuing cleanup): {e}")

    # Step 2: Try to close the exit stack
    try:
      logger.info("Closing AsyncExitStack")
      await self._exit_stack.aclose()
      # If we get here, the exit stack closed successfully
      logger.info("AsyncExitStack closed successfully")
      return
    except RuntimeError as e:
      if "Attempted to exit cancel scope in a different task" in str(e):
        logger.warning("Task mismatch during shutdown - using fallback cleanup")
        # Continue to manual cleanup
      else:
        logger.error(f"Unexpected RuntimeError: {e}")
        # Continue to manual cleanup
    except Exception as e:
      logger.error(f"Error during exit stack closure: {e}")
      # Continue to manual cleanup

    # Step 3: Manual cleanup of the subprocess if we have its PID
    if self._process_pid:
      await self._ensure_process_terminated(self._process_pid)

    # Step 4: Ask the session manager to do any additional cleanup it can
    await self._session_manager._emergency_cleanup()

  async def _ensure_process_terminated(self, pid):
    """Ensure a process is terminated using its PID."""
    try:
      # Check if process exists
      os.kill(pid, 0)  # This just checks if the process exists

      logger.info(f"Terminating process with PID {pid}")
      # First try SIGTERM for graceful shutdown
      os.kill(pid, signal.SIGTERM)

      # Give it a moment to terminate
      for _ in range(30):  # wait up to 3 seconds
        await asyncio.sleep(0.1)
        try:
          os.kill(pid, 0)  # Process still exists
        except ProcessLookupError:
          logger.info(f"Process {pid} terminated successfully")
          return

      # If we get here, process didn't terminate gracefully
      logger.warning(
          f"Process {pid} didn't terminate gracefully, using SIGKILL"
      )
      os.kill(pid, signal.SIGKILL)

    except ProcessLookupError:
      logger.info(f"Process {pid} already terminated")
    except Exception as e:
      logger.error(f"Error terminating process {pid}: {e}")

  @retry_on_closed_resource("_initialize")
  @override
  async def get_tools(
      self,
      readonly_context: Optional[ReadonlyContext] = None,
  ) -> List[MCPTool]:
    """Loads all tools from the MCP Server.

    Returns:
      A list of MCPTools imported from the MCP Server.
    """
    if not self._session:
      await self._initialize()
    tools_response: ListToolsResult = await self._session.list_tools()
    tools = []
    for tool in tools_response.tools:
      mcp_tool = MCPTool(
          mcp_tool=tool,
          mcp_session=self._session,
          mcp_session_manager=self._session_manager,
      )

      if self._is_selected(mcp_tool, readonly_context):
        tools.append(mcp_tool)
    return tools
