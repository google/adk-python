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
from typing import Optional

from google.genai.types import FunctionDeclaration
from typing_extensions import override

from .._gemini_schema_util import _to_gemini_schema
from .mcp_session_manager import MCPSessionManager
from .mcp_session_manager import retry_on_closed_resource

# Attempt to import MCP Tool from the MCP library, and hints user to upgrade
# their Python version to 3.10 if it fails.
try:
  from mcp.types import Tool as McpBaseTool
except ImportError as e:
  import sys

  if sys.version_info < (3, 10):
    raise ImportError(
        "MCP Tool requires Python 3.10 or above. Please upgrade your Python"
        " version."
    ) from e
  else:
    raise e


from ...auth.auth_credential import AuthCredential
from ...auth.auth_schemes import AuthScheme
from ..base_tool import BaseTool
from ..tool_context import ToolContext

logger = logging.getLogger("google_adk." + __name__)


class MCPTool(BaseTool):
  """Turns an MCP Tool into an ADK Tool.

  Internally, the tool initializes from a MCP Tool, and uses the MCP Session to
  call the tool.
  """

  def __init__(
      self,
      *,
      mcp_tool: McpBaseTool,
      mcp_session_manager: MCPSessionManager,
      auth_scheme: Optional[AuthScheme] = None,
      auth_credential: Optional[AuthCredential] = None,
  ):
    """Initializes an MCPTool.

    This tool wraps an MCP Tool interface and uses a session manager to
    communicate with the MCP server.

    Args:
        mcp_tool: The MCP tool to wrap.
        mcp_session_manager: The MCP session manager to use for communication.
        auth_scheme: The authentication scheme to use.
        auth_credential: The authentication credential to use.

    Raises:
        ValueError: If mcp_tool or mcp_session_manager is None.
    """
    if mcp_tool is None:
      raise ValueError("mcp_tool cannot be None")
    if mcp_session_manager is None:
      raise ValueError("mcp_session_manager cannot be None")
    super().__init__(
        name=mcp_tool.name,
        description=mcp_tool.description if mcp_tool.description else "",
    )
    self._mcp_tool = mcp_tool
    self._mcp_session_manager = mcp_session_manager
    # TODO(cheliu): Support passing auth to MCP Server.
    self._auth_scheme = auth_scheme
    self._auth_credential = auth_credential

  @override
  def _get_declaration(self) -> FunctionDeclaration:
    """Gets the function declaration for the tool.

    Returns:
        FunctionDeclaration: The Gemini function declaration for the tool.
    """
    schema_dict = self._mcp_tool.inputSchema
    parameters = _to_gemini_schema(schema_dict)
    function_decl = FunctionDeclaration(
        name=self.name, description=self.description, parameters=parameters
    )
    return function_decl

  @retry_on_closed_resource("_mcp_session_manager")
  async def run_async(self, *, args, tool_context: ToolContext):
    """Runs the tool asynchronously.

    Args:
        args: The arguments as a dict to pass to the tool.
        tool_context: The tool context of the current invocation.

    Returns:
        Any: The response from the tool.
    """
    # Get the session from the session manager
    session = await self._mcp_session_manager.create_session()

    response = await session.call_tool(self.name, arguments=args)
    return response
