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

from typing import Optional

from google.genai.types import FunctionDeclaration
from typing_extensions import override

from .mcp_session_manager import MCPSessionManager, retry_on_closed_resource

# Attempt to import MCP Tool from the MCP library, and hints user to upgrade
# their Python version to 3.10 if it fails.
try:
  from mcp import ClientSession
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


from ..base_tool import BaseTool
from ...auth.auth_credential import AuthCredential
from ...auth.auth_schemes import AuthScheme
from ..openapi_tool.openapi_spec_parser.rest_api_tool import to_gemini_schema
from ..tool_context import ToolContext


class MCPTool(BaseTool):
  """Turns a MCP Tool into a Vertex Agent Framework Tool.

  Internally, the tool initializes from a MCP Tool, and uses the MCP Session to
  call the tool.
  """

  def __init__(
      self,
      mcp_tool: McpBaseTool,
      mcp_session: ClientSession,
      mcp_session_manager: MCPSessionManager,
      auth_scheme: Optional[AuthScheme] = None,
      auth_credential: Optional[AuthCredential] | None = None,
  ):
    """Initializes a MCPTool.

    This tool wraps a MCP Tool interface and an active MCP Session. It invokes
    the MCP Tool through executing the tool from remote MCP Session.

    Example:
        tool = MCPTool(mcp_tool=mcp_tool, mcp_session=mcp_session)

    Args:
        mcp_tool: The MCP tool to wrap.
        mcp_session: The MCP session to use to call the tool.
        auth_scheme: The authentication scheme to use.
        auth_credential: The authentication credential to use.

    Raises:
        ValueError: If mcp_tool or mcp_session is None.
    """
    if mcp_tool is None:
      raise ValueError("mcp_tool cannot be None")
    if mcp_session is None:
      raise ValueError("mcp_session cannot be None")
    self.name = mcp_tool.name
    self.description = mcp_tool.description if mcp_tool.description else ""
    self.mcp_tool = mcp_tool
    self.mcp_session = mcp_session
    self.mcp_session_manager = mcp_session_manager
    # TODO(cheliu): Support passing auth to MCP Server.
    self.auth_scheme = auth_scheme
    self.auth_credential = auth_credential

  async def _reinitialize_session(self):
    self.mcp_session = await self.mcp_session_manager.create_session()

  @override
  def _get_declaration(self) -> FunctionDeclaration:
    """Gets the function declaration for the tool.

    Returns:
        FunctionDeclaration: The Gemini function declaration for the tool.
    """
    schema_dict = self.mcp_tool.inputSchema
    parameters = to_gemini_schema(schema_dict)
    function_decl = FunctionDeclaration(
        name=self.name, description=self.description, parameters=parameters
    )
    return function_decl

  @override
  @retry_on_closed_resource("_reinitialize_session")
  async def run_async(self, *, args, tool_context: ToolContext):
    """Runs the tool asynchronously.

    Args:
        args: The arguments as a dict to pass to the tool.
        tool_context: The tool context from upper level ADK agent.

    Returns:
        Any: The response from the tool.
    """
    # TODO(cheliu): Support passing tool context to MCP Server.
    try:
      response = await self.mcp_session.call_tool(self.name, arguments=args)
      return response
    except Exception as e:
      print(e)
      raise e
