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
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import TextIO
from typing import Tuple
from typing import Union

from ...agents.readonly_context import ReadonlyContext
from ...auth.auth_credential import AuthCredential
from ...auth.auth_schemes import AuthScheme
from ..base_tool import BaseTool
from ..base_toolset import BaseToolset
from ..base_toolset import ToolPredicate
from .mcp_session_manager import ContextToEnvMapperCallback
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

# Type definition for auth transformation callback
AuthTransformCallback = Callable[
    [Dict[str, Any]], Tuple[Optional[AuthScheme], Optional[AuthCredential]]
]


class MCPToolset(BaseToolset):
  """Connects to a MCP Server, and retrieves MCP Tools into ADK Tools.

  This toolset manages the connection to an MCP server and provides tools
  that can be used by an agent. It properly implements the BaseToolset
  interface for easy integration with the agent framework.

  Usage:
  ```python
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
  ```
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
      auth_transform_callback: Optional[AuthTransformCallback] = None,
      context_to_env_mapper_callback: Optional[
          ContextToEnvMapperCallback
      ] = None,
      errlog: TextIO = sys.stderr,
  ):
    """Initializes the MCPToolset.

    Args:
      connection_params: The connection parameters to the MCP server. Can be:
        `StdioConnectionParams` for using local mcp server (e.g. using `npx` or
        `python3`); or `SseConnectionParams` for a local/remote SSE server; or
        `StreamableHTTPConnectionParams` for local/remote Streamable http
        server. Note, `StdioServerParameters` is also supported for using local
        mcp server (e.g. using `npx` or `python3` ), but it does not support
        timeout, and we recommend to use `StdioConnectionParams` instead when
        timeout is needed.
      tool_filter: Optional filter to select specific tools. Can be either: - A
        list of tool names to include - A ToolPredicate function for custom
        filtering logic
      auth_transform_callback: Optional callback function to transform auth data
        from ReadonlyContext.state into AuthScheme and AuthCredential. If None,
        the toolset will look for 'auth_scheme' and 'auth_credential' keys
        directly in the context state.
      context_to_env_mapper_callback: Optional callback function to transform session
        state into environment variables for the MCP connection. Takes a
        dictionary of session state and returns a dictionary of environment
        variables to be injected into the MCP connection.
      errlog: TextIO stream for error logging.
    """
    super().__init__(tool_filter=tool_filter)

    if not connection_params:
      raise ValueError("Missing connection params in MCPToolset.")

    self._connection_params = connection_params
    self._auth_transform_callback = auth_transform_callback
    self._context_to_env_mapper_callback = context_to_env_mapper_callback
    self._errlog = errlog

    # Create the session manager that will handle the MCP connection
    self._mcp_session_manager = MCPSessionManager(
        connection_params=self._connection_params,
        errlog=self._errlog,
        context_to_env_mapper_callback=self._context_to_env_mapper_callback,
    )

    self._session = None

  def _extract_auth_from_context(
      self, readonly_context: Optional[ReadonlyContext]
  ) -> Tuple[Optional[AuthScheme], Optional[AuthCredential]]:
    """Extracts auth scheme and credential from readonly context.

    Args:
        readonly_context: The readonly context containing state information.

    Returns:
        Tuple of (AuthScheme, AuthCredential) or (None, None) if not found.
    """
    if not readonly_context or not readonly_context.state:
      return None, None

    state_dict = dict(readonly_context.state)

    # If a custom auth transform callback is provided, use it
    if self._auth_transform_callback:
      try:
        return self._auth_transform_callback(state_dict)
      except Exception as e:
        logger.warning(
            f"Auth transform callback failed: {e}. Falling back to direct"
            " extraction."
        )

    # Direct extraction: look for 'auth_scheme' and 'auth_credential' keys
    auth_scheme = state_dict.get("auth_scheme")
    auth_credential = state_dict.get("auth_credential")

    # Validate types - auth_scheme should be an AuthScheme instance
    if auth_scheme is not None:
      try:
        # Check if it's a valid AuthScheme (Union of SecurityScheme types)
        from fastapi.openapi.models import SecurityScheme

        from ...auth.auth_schemes import OpenIdConnectWithConfig

        if not isinstance(
            auth_scheme, (SecurityScheme, OpenIdConnectWithConfig)
        ):
          logger.warning(
              f"Invalid auth_scheme type in context: {type(auth_scheme)}"
          )
          auth_scheme = None
      except Exception as e:
        logger.warning(f"Error validating auth_scheme: {e}")
        auth_scheme = None

    if auth_credential is not None and not isinstance(
        auth_credential, AuthCredential
    ):
      logger.warning(
          f"Invalid auth_credential type in context: {type(auth_credential)}"
      )
      auth_credential = None

    return auth_scheme, auth_credential

  @retry_on_closed_resource("_reinitialize_session")
  async def get_tools(
      self,
      readonly_context: Optional[ReadonlyContext] = None,
  ) -> List[BaseTool]:
    """Return all tools in the toolset based on the provided context.

    Args:
        readonly_context: Context used to filter tools available to the agent.
            If None, all tools in the toolset are returned. The context may
            also contain auth information in its state.

    Returns:
        List[BaseTool]: A list of tools available under the specified context.
    """
    # Get session from session manager
    if not self._session:
      self._session = await self._mcp_session_manager.create_session(
          readonly_context
      )

    # Extract auth information from context
    auth_scheme, auth_credential = self._extract_auth_from_context(
        readonly_context
    )

    # Fetch available tools from the MCP server
    tools_response: ListToolsResult = await self._session.list_tools()

    # Apply filtering based on context and tool_filter
    tools = []
    for tool in tools_response.tools:
      mcp_tool = MCPTool(
          mcp_tool=tool,
          mcp_session_manager=self._mcp_session_manager,
          auth_scheme=auth_scheme,
          auth_credential=auth_credential,
      )

      if self._is_tool_selected(mcp_tool, readonly_context):
        tools.append(mcp_tool)
    return tools

  async def _reinitialize_session(self):
    """Reinitializes the session when connection is lost."""
    # Close the old session and clear cache
    await self._mcp_session_manager.close()
    # Note: We'll need the readonly_context for reinitializing, but it's not
    # available here. The session will be recreated on the next get_tools call
    # with the proper context.
    self._session = None

    # Tools will be reloaded on next get_tools call

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
    finally:
      # Clear session reference
      self._session = None
