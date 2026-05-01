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
import os
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

from mcp import SamplingCapability
from mcp import StdioServerParameters
from mcp.client.session import SamplingFnT
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
from ._internal import sanitize_header_value
from ._internal import validate_header_format
from ._internal import validate_header_name
from ._internal import validate_header_value
from .mcp_session_manager import MCPSessionManager
from .mcp_session_manager import retry_on_errors
from .mcp_session_manager import SseConnectionParams
from .mcp_session_manager import StdioConnectionParams
from .mcp_session_manager import StreamableHTTPConnectionParams
from .mcp_tool import MCPTool
from .mcp_tool import ProgressCallbackFactory
from .types import HeaderProvider

logger = logging.getLogger("google_adk." + __name__)

T = TypeVar("T")


def create_session_state_header_provider(
    state_key: str,
    header_name: str = "Authorization",
    header_format: str = "Bearer {value}",
    default_value: Optional[str] = None,
    strict: bool = False,
) -> HeaderProvider:
  """Creates a header provider that extracts values from session state.

  This utility function generates a header_provider callable that can be used
  with McpToolset to automatically extract values from the session state and
  format them as HTTP headers for MCP server connections.

  .. warning::
      **Security Best Practice**: For sensitive, short-lived tokens like JWTs,
      use ``request_state`` instead of ``session.state`` to avoid persisting
      sensitive data to the database. Pass tokens via
      ``RunAgentRequest.request_state``, which will override ``session.state``
      for the duration of the request without being persisted.

  **Security Features:**

  - RFC 7230 compliant HTTP header validation and sanitization
  - Automatic protection against header injection attacks
  - Support for secure token propagation via session state
  - Configurable strict validation for header values

  **Security Best Practices:**

  1. **Token Security**: Use ``request_state`` for sensitive, short-lived tokens
    (JWTs, API keys) instead of ``session.state`` to avoid persisting
    sensitive data.

  2. **Header Validation**: Header names and values are automatically validated
    according to RFC 7230 to prevent injection attacks.

  3. **Complex Data**: For complex data structures, pre-serialize them or use
    ``state_header_format`` to ensure proper string representation.

  4. **Strict Mode**: Enable ``state_header_strict=True`` in configuration to
    catch non-primitive type errors early.

  Args:
      state_key: The key to look up in session.state (or request_state).
      header_name: The HTTP header name to set (default: 'Authorization').
      header_format: Format string for the header value. Use {value} as a
        placeholder for the state value (default: 'Bearer {value}').
      default_value: Default value if state_key is not found in session state.
        If None, the header is omitted when the key is missing.
      strict: If True, raises ValueError when non-primitive types are
        encountered. If False (default), logs a warning instead.

  Returns:
      A callable that takes a ReadonlyContext and returns a dictionary of
      headers to be used for the MCP session.

  Raises:
      ValueError: If strict=True and a non-primitive type is found in state,
        or if header_name is invalid.

  Example::

      # Example 1: Using request_state for JWT tokens (recommended)
      toolset = McpToolset(
          connection_params=StreamableHTTPConnectionParams(
              url="http://api.example.com/mcp"
          ),
          header_provider=create_session_state_header_provider(
              state_key="jwt_token",  # Will read from request_state first
              header_name="Authorization",
              header_format="Bearer {value}"
          )
      )

      # Client sends request with ephemeral JWT
      response = await agent.run(
          RunAgentRequest(
              session_id="user-123",
              request_state={  # Ephemeral, not persisted
                  "jwt_token": "eyJhbG..."
              }
          )
      )
  """
  # Validate header name and format upfront to prevent injection attacks
  validate_header_name(header_name)
  validate_header_format(header_format)

  def provider(ctx: ReadonlyContext) -> Dict[str, str]:
    value = ctx.state.get(state_key, default_value)
    # Skip header if value is None or empty string
    if value is None or value == "":
      return {}

    validate_header_value(state_key, value, strict=strict)
    formatted_value = header_format.format(value=value)
    # Strip CRLF from the interpolated value to prevent header injection.
    # The format string is validated at construction time, but the runtime
    # value comes from session state and must never contain CRLF here.
    formatted_value = formatted_value.replace("\r", "").replace("\n", "")
    sanitized_value = sanitize_header_value(formatted_value)

    return {header_name: sanitized_value}

  return provider


def create_combined_header_provider(
    providers: List[HeaderProvider],
) -> HeaderProvider:
  """Creates a header provider that combines multiple providers.

  Args:
      providers: A list of header providers to combine.

  Returns:
      A single header provider that merges the results of all input providers.
  """

  def combined_provider(ctx: ReadonlyContext) -> Dict[str, str]:
    headers = {}
    num_providers = len(providers)
    for i, provider in enumerate(providers):
      try:
        provider_headers = provider(ctx)
        if provider_headers:
          headers.update(provider_headers)
      except Exception as e:
        logger.error(f"Header provider {i+1}/{num_providers} failed: {e}")
        raise

    if headers:
      logger.debug(
          f"Combined header provider generated {len(headers)} total headers"
      )
    return headers

  return combined_provider


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
        model='gemini-2.5-flash',
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
      header_provider: Optional[HeaderProvider] = None,
      progress_callback: Optional[
          Union[ProgressFnT, ProgressCallbackFactory]
      ] = None,
      use_mcp_resources: Optional[bool] = False,
      sampling_callback: Optional[SamplingFnT] = None,
      sampling_capabilities: Optional[SamplingCapability] = None,
      credential_key: str | None = None,
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
        from MCP server during long-running tool execution. Can be either:  - A
        ``ProgressFnT`` callback that receives (progress, total, message). This
        callback will be shared by all tools in the toolset.  - A
        ``ProgressCallbackFactory`` that creates per-tool callbacks. The factory
        receives (tool_name, callback_context, **kwargs) and returns a
        ProgressFnT or None. This allows different tools to have different
        progress handling logic and access/modify session state via the
        CallbackContext. The **kwargs parameter allows for future extensibility.
      use_mcp_resources: Whether the agent should have access to MCP resources.
        This will add a `load_mcp_resource` tool to the toolset and include
        available resources in the agent context. Defaults to False.
      sampling_callback: Optional callback to handle sampling requests from the
        MCP server.
      sampling_capabilities: Optional capabilities for sampling.
      credential_key: A user specified key used to load and save this credential
        in a credential service. Used with auth_scheme.
    """

    # --- BEGIN BOUND TOKEN PATCH ---
    # Set GOOGLE_API_PREVENT_AGENT_TOKEN_SHARING_FOR_GCP_SERVICES to false
    # to disable bound token sharing. Tracking on
    # https://github.com/google/adk-python/issues/5361
    os.environ["GOOGLE_API_PREVENT_AGENT_TOKEN_SHARING_FOR_GCP_SERVICES"] = (
        "false"
    )
    # --- END BOUND TOKEN  PATCH ---

    super().__init__(tool_filter=tool_filter, tool_name_prefix=tool_name_prefix)

    self._sampling_callback = sampling_callback
    self._sampling_capabilities = sampling_capabilities

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
        sampling_callback=self._sampling_callback,
        sampling_capabilities=self._sampling_capabilities,
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
            credential_key=credential_key,
        )
        if auth_scheme
        else None
    )
    self._use_mcp_resources = use_mcp_resources

  def _get_auth_headers(
      self, readonly_context: Optional[ReadonlyContext] = None
  ) -> Optional[Dict[str, str]]:
    """Build authentication headers from exchanged credential.

    Args:
      readonly_context: Readonly context to get credentials from.

    Returns:
        Dictionary of auth headers, or None if no auth configured.
    """
    if not self._auth_config:
      return None

    credential = None
    if readonly_context:
      credential = readonly_context.get_credential(
          self._auth_config.credential_key
      )

    if not credential:
      credential = self._auth_config.exchanged_auth_credential

    if not credential:
      return None
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

      if credential.http.additional_headers:
        headers = headers or {}
        headers.update(credential.http.additional_headers)
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
    auth_headers = self._get_auth_headers(readonly_context)
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
      logger.exception(
          f"Exception during MCP session execution: {error_message}: {e}"
      )
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

    # Build header_provider from state_header_mapping.
    providers = []

    if mcp_toolset_config.state_header_mapping:
      state_mapping = mcp_toolset_config.state_header_mapping
      state_format = mcp_toolset_config.state_header_format or {}

      providers.extend([
          create_session_state_header_provider(
              state_key=state_key,
              header_name=header_name,
              header_format=state_format.get(header_name, "{value}"),
              default_value=None,
              strict=mcp_toolset_config.state_header_strict,
          )
          for state_key, header_name in state_mapping.items()
      ])

    header_provider = (
        create_combined_header_provider(providers) if providers else None
    )

    return cls(
        connection_params=connection_params,
        tool_filter=mcp_toolset_config.tool_filter,
        tool_name_prefix=mcp_toolset_config.tool_name_prefix,
        auth_scheme=mcp_toolset_config.auth_scheme,
        auth_credential=mcp_toolset_config.auth_credential,
        credential_key=mcp_toolset_config.credential_key,
        use_mcp_resources=mcp_toolset_config.use_mcp_resources,
        header_provider=header_provider,
    )

  def __getstate__(self):
    """Custom pickling to exclude non-picklable runtime objects."""
    state = self.__dict__.copy()
    # Remove unpicklable file-like objects
    state.pop("_errlog", None)
    return state

  def __setstate__(self, state):
    """Custom unpickling to restore state."""
    self.__dict__.update(state)
    # Default to sys.stderr if _errlog was removed during pickling
    if not hasattr(self, "_errlog") or self._errlog is None:
      self._errlog = sys.stderr


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

  credential_key: str | None = None

  use_mcp_resources: bool = False

  state_header_mapping: Optional[Dict[str, str]] = None
  """Maps session state keys to HTTP header names.

  When specified, values from the session state will be extracted and passed
  as HTTP headers to the MCP server. This is useful for propagating
  authentication tokens or other context from the ADK session to the MCP server.

  Example::

      state_header_mapping:
        jwt_token: Authorization
        tenant_id: X-Tenant-ID

  This will read `session.state["jwt_token"]` and set it as the
  "Authorization" header, and read `session.state["tenant_id"]` and
  set it as the "X-Tenant-ID" header.
  """

  state_header_format: Optional[Dict[str, str]] = None
  """Optional formatting for header values extracted from session state.

  Supports format strings with {value} as a placeholder for the state value.
  Only applies to headers specified in state_header_mapping.

  Example::

      state_header_format:
        Authorization: "Bearer {value}"
        X-API-Key: "key:{value}"

  If not specified for a particular header, the value from session state is
  used as-is.
  """

  state_header_strict: bool = False
  """Enable strict type validation for state header values.

  When True, raises ValueError if state values are non-primitive types
  (not str, int, float, or bool). This helps catch configuration errors
  early by preventing accidental serialization of complex objects into headers.

  When False (default), non-primitive types trigger a warning but are still
  formatted into headers.

  Example::

      state_header_strict: true  # Raises error on non-primitive types
  """

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
