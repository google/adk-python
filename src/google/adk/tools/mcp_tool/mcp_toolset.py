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

from __future__ import annotations

import asyncio
import base64
import logging
import sys
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import TextIO
from typing import TypeVar
from typing import Union
import warnings

from mcp import StdioServerParameters
from mcp.shared.session import ProgressFnT
from mcp.types import ListResourcesResult
from mcp.types import ListToolsResult
from pydantic import model_validator
from typing_extensions import override

from ...agents.readonly_context import ReadonlyContext
from ...auth.auth_credential import AuthCredential
from ...auth.auth_schemes import AuthScheme
from ...auth.auth_tool import AuthConfig
from ..base_tool import BaseTool
from ..base_toolset import BaseToolset
from ..base_toolset import ToolPredicate
from ..load_mcp_resource_tool import LoadMcpResourceTool
from ..tool_configs import BaseToolConfig
from ..tool_configs import ToolArgsConfig
from ..tool_context import ToolContext
from .mcp_session_manager import MCPSessionManager
from .mcp_session_manager import retry_on_errors
from .mcp_session_manager import SseConnectionParams
from .mcp_session_manager import StdioConnectionParams
from .mcp_session_manager import StreamableHTTPConnectionParams
from .mcp_tool import MCPTool
from .mcp_tool import ProgressCallbackFactory

logger = logging.getLogger("google_adk." + __name__)


T = TypeVar("T")


class McpToolset(BaseToolset):
  """Connects to a MCP Server, and retrieves MCP Tools into ADK Tools.

  This toolset manages the connection to an MCP server and provides tools
  that can be used by an agent. It properly implements the BaseToolset
  interface for easy integration with the agent framework.

  Usage::

    toolset = McpToolset(
        connection_params=StdioServerParameters(
            command='npx',
            args=["-y", "@modelcontextprotocol/server-filesystem"],
        ),
        tool_filter=['read_file', 'list_directory']  # Optional: filter specific
        tools
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
      tool_name_prefix: Optional[str] = None,
      errlog: TextIO = sys.stderr,
      auth_scheme: Optional[AuthScheme] = None,
      auth_credential: Optional[AuthCredential] = None,
      require_confirmation: Union[bool, Callable[..., bool]] = False,
      header_provider: Optional[
          Callable[[ReadonlyContext], Dict[str, str]]
      ] = None,
      progress_callback: Optional[
          Union[ProgressFnT, ProgressCallbackFactory]
      ] = None,
      use_mcp_resources: Optional[bool] = False,
  ):
    """Initializes the McpToolset.

    Args:
      connection_params: The connection parameters to the MCP server. Can be:
        ``StdioConnectionParams`` for using local mcp server (e.g. using ``npx``
        or ``python3``); or ``SseConnectionParams`` for a local/remote SSE
        server; or ``StreamableHTTPConnectionParams`` for local/remote
        Streamable http server. Note, ``StdioServerParameters`` is also
        supported for using local mcp server (e.g. using ``npx`` or ``python3``
        ), but it does not support timeout, and we recommend to use
        ``StdioConnectionParams`` instead when timeout is needed.
      tool_filter: Optional filter to select specific tools. Can be either: - A
        list of tool names to include - A ToolPredicate function for custom
        filtering logic
      tool_name_prefix: A prefix to be added to the name of each tool in this
        toolset.
      errlog: TextIO stream for error logging.
      auth_scheme: The auth scheme of the tool for tool calling
      auth_credential: The auth credential of the tool for tool calling
      require_confirmation: Whether tools in this toolset require confirmation.
        Can be a single boolean or a callable to apply to all tools.
      header_provider: A callable that takes a ReadonlyContext and returns a
        dictionary of headers to be used for the MCP session.
      progress_callback: Optional callback to receive progress notifications
        from MCP server during long-running tool execution. Can be either:

        - A ``ProgressFnT`` callback that receives (progress, total, message).
          This callback will be shared by all tools in the toolset.

        - A ``ProgressCallbackFactory`` that creates per-tool callbacks. The
          factory receives (tool_name, callback_context, **kwargs) and returns
          a ProgressFnT or None. This allows different tools to have different
          progress handling logic and access/modify session state via the
          CallbackContext. The **kwargs parameter allows for future
          extensibility.
      use_mcp_resources: Whether the agent should have access to MCP resources.
        This will add a `load_mcp_resource` tool to the toolset and include
        available resources in the agent context. Defaults to False.
    """

    super().__init__(tool_filter=tool_filter, tool_name_prefix=tool_name_prefix)

    if not connection_params:
      raise ValueError("Missing connection params in McpToolset.")

    self._connection_params = connection_params
    self._errlog = errlog
    self._header_provider = header_provider
    self._progress_callback = progress_callback

    # Create the session manager that will handle the MCP connection
    self._mcp_session_manager = MCPSessionManager(
        connection_params=self._connection_params,
        errlog=self._errlog,
    )
    self._auth_scheme = auth_scheme
    self._auth_credential = auth_credential
    self._require_confirmation = require_confirmation
    # Store auth config as instance variable so ADK can populate
    # exchanged_auth_credential in-place before calling get_tools()
    self._auth_config: Optional[AuthConfig] = (
        AuthConfig(
            auth_scheme=auth_scheme,
            raw_auth_credential=auth_credential,
        )
        if auth_scheme
        else None
    )
    self._use_mcp_resources = use_mcp_resources

  def _get_auth_headers(self) -> Optional[Dict[str, str]]:
    """Build authentication headers from exchanged credential.

    Returns:
        Dictionary of auth headers, or None if no auth configured.
    """
    if not self._auth_config or not self._auth_config.exchanged_auth_credential:
      return None

    credential = self._auth_config.exchanged_auth_credential
    headers: Optional[Dict[str, str]] = None

    if credential.oauth2:
      headers = {"Authorization": f"Bearer {credential.oauth2.access_token}"}
    elif credential.http:
      # Handle HTTP authentication schemes
      if (
          credential.http.scheme.lower() == "bearer"
          and credential.http.credentials
          and credential.http.credentials.token
      ):
        headers = {
            "Authorization": f"Bearer {credential.http.credentials.token}"
        }
      elif credential.http.scheme.lower() == "basic":
        # Handle basic auth
        if (
            credential.http.credentials
            and credential.http.credentials.username
            and credential.http.credentials.password
        ):
          credentials_str = (
              f"{credential.http.credentials.username}"
              f":{credential.http.credentials.password}"
          )
          encoded_credentials = base64.b64encode(
              credentials_str.encode()
          ).decode()
          headers = {"Authorization": f"Basic {encoded_credentials}"}
      elif credential.http.credentials and credential.http.credentials.token:
        # Handle other HTTP schemes with token
        headers = {
            "Authorization": (
                f"{credential.http.scheme} {credential.http.credentials.token}"
            )
        }
    elif credential.api_key:
      # For API key, use the auth scheme to determine header name
      if self._auth_config.auth_scheme:
        from fastapi.openapi.models import APIKeyIn

        if hasattr(self._auth_config.auth_scheme, "in_"):
          if self._auth_config.auth_scheme.in_ == APIKeyIn.header:
            headers = {self._auth_config.auth_scheme.name: credential.api_key}
          else:
            logger.warning(
                "McpToolset only supports header-based API key authentication."
                " Configured location: %s",
                self._auth_config.auth_scheme.in_,
            )
        else:
          # Default to using scheme name as header
          headers = {self._auth_config.auth_scheme.name: credential.api_key}

    return headers

  async def _execute_with_session(
      self,
      coroutine_func: Callable[[Any], Awaitable[T]],
      error_message: str,
      readonly_context: Optional[ReadonlyContext] = None,
  ) -> T:
    """Creates a session and executes a coroutine with it."""
    headers: Dict[str, str] = {}

    # Add headers from header_provider if available
    if self._header_provider and readonly_context:
      provider_headers = self._header_provider(readonly_context)
      if provider_headers:
        headers.update(provider_headers)

    # Add auth headers from exchanged credential if available
    auth_headers = self._get_auth_headers()
    if auth_headers:
      headers.update(auth_headers)

    session = await self._mcp_session_manager.create_session(
        headers=headers if headers else None
    )
    timeout_in_seconds = (
        self._connection_params.timeout
        if hasattr(self._connection_params, "timeout")
        else None
    )
    try:
      return await asyncio.wait_for(
          coroutine_func(session), timeout=timeout_in_seconds
      )
    except Exception as e:
      raise ConnectionError(f"{error_message}: {e}") from e

  @retry_on_errors
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
    # Fetch available tools from the MCP server
    tools_response: ListToolsResult = await self._execute_with_session(
        lambda session: session.list_tools(),
        "Failed to get tools from MCP server",
        readonly_context,
    )

    # Apply filtering based on context and tool_filter
    tools = []
    for tool in tools_response.tools:
      mcp_tool = MCPTool(
          mcp_tool=tool,
          mcp_session_manager=self._mcp_session_manager,
          auth_scheme=self._auth_scheme,
          auth_credential=self._auth_credential,
          require_confirmation=self._require_confirmation,
          header_provider=self._header_provider,
          progress_callback=self._progress_callback
          if hasattr(self, "_progress_callback")
          else None,
      )

      if self._is_tool_selected(mcp_tool, readonly_context):
        tools.append(mcp_tool)

    if self._use_mcp_resources:
      load_resource_tool = LoadMcpResourceTool(
          mcp_toolset=self,
      )
      tools.append(load_resource_tool)

    return tools

  async def read_resource(
      self, name: str, readonly_context: Optional[ReadonlyContext] = None
  ) -> Any:
    """Fetches and returns a list of contents of the named resource.

    Args:
      name: The name of the resource to fetch.
      readonly_context: Context used to provide headers for the MCP session.

    Returns:
      List of contents of the resource.
    """
    resource_info = await self.get_resource_info(name, readonly_context)
    if "uri" not in resource_info:
      raise ValueError(f"Resource '{name}' has no URI.")

    result: Any = await self._execute_with_session(
        lambda session: session.read_resource(uri=resource_info["uri"]),
        f"Failed to get resource {name} from MCP server",
        readonly_context,
    )
    return result.contents

  async def list_resources(
      self, readonly_context: Optional[ReadonlyContext] = None
  ) -> list[str]:
    """Returns a list of resource names available on the MCP server."""
    result: ListResourcesResult = await self._execute_with_session(
        lambda session: session.list_resources(),
        "Failed to list resources from MCP server",
        readonly_context,
    )
    return [resource.name for resource in result.resources]

  async def get_resource_info(
      self, name: str, readonly_context: Optional[ReadonlyContext] = None
  ) -> dict[str, Any]:
    """Returns metadata about a specific resource (name, MIME type, etc.)."""
    result: ListResourcesResult = await self._execute_with_session(
        lambda session: session.list_resources(),
        "Failed to list resources from MCP server",
        readonly_context,
    )
    for resource in result.resources:
      if resource.name == name:
        return resource.model_dump(mode="json", exclude_none=True)
    raise ValueError(f"Resource with name '{name}' not found.")

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
      print(f"Warning: Error during McpToolset cleanup: {e}", file=self._errlog)

  @override
  def get_auth_config(self) -> Optional[AuthConfig]:
    """Returns the auth config for this toolset.

    ADK will populate exchanged_auth_credential on this config before calling
    get_tools(). The toolset can then access the ready-to-use credential via
    self._auth_config.exchanged_auth_credential.
    """
    return self._auth_config

  @override
  @classmethod
  def from_config(
      cls: type[McpToolset], config: ToolArgsConfig, config_abs_path: str
  ) -> McpToolset:
    """Creates an McpToolset from a configuration object."""
    mcp_toolset_config = McpToolsetConfig.model_validate(config.model_dump())

    if mcp_toolset_config.stdio_server_params:
      connection_params = mcp_toolset_config.stdio_server_params
    elif mcp_toolset_config.stdio_connection_params:
      connection_params = mcp_toolset_config.stdio_connection_params
    elif mcp_toolset_config.sse_connection_params:
      connection_params = mcp_toolset_config.sse_connection_params
    elif mcp_toolset_config.streamable_http_connection_params:
      connection_params = mcp_toolset_config.streamable_http_connection_params
    else:
      raise ValueError("No connection params found in McpToolsetConfig.")

    return cls(
        connection_params=connection_params,
        tool_filter=mcp_toolset_config.tool_filter,
        tool_name_prefix=mcp_toolset_config.tool_name_prefix,
        auth_scheme=mcp_toolset_config.auth_scheme,
        auth_credential=mcp_toolset_config.auth_credential,
        use_mcp_resources=mcp_toolset_config.use_mcp_resources,
    )


class MCPToolset(McpToolset):
  """Deprecated name, use `McpToolset` instead."""

  def __init__(self, *args, **kwargs):
    warnings.warn(
        "MCPToolset class is deprecated, use `McpToolset` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    super().__init__(*args, **kwargs)


class McpToolsetConfig(BaseToolConfig):
  """The config for McpToolset."""

  stdio_server_params: Optional[StdioServerParameters] = None

  stdio_connection_params: Optional[StdioConnectionParams] = None

  sse_connection_params: Optional[SseConnectionParams] = None

  streamable_http_connection_params: Optional[
      StreamableHTTPConnectionParams
  ] = None

  tool_filter: Optional[List[str]] = None

  tool_name_prefix: Optional[str] = None

  auth_scheme: Optional[AuthScheme] = None

  auth_credential: Optional[AuthCredential] = None

  use_mcp_resources: bool = False

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
