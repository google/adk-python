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
import collections
import dataclasses
import inspect
import logging
import sys
import time
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
from mcp.client.session import ElicitationFnT
from mcp.client.session import SamplingFnT
from mcp.shared.session import ProgressFnT
from mcp.types import ListResourcesResult
from mcp.types import ListToolsResult
from mcp.types import Tool as McpBaseTool
from pydantic import model_validator
from typing_extensions import override

from ...agents.readonly_context import ReadonlyContext
from ...auth._auth_headers import build_auth_headers
from ...auth.auth_credential import AuthCredential
from ...auth.auth_schemes import AuthScheme
from ...auth.auth_tool import AuthConfig
from ...utils.env_utils import is_env_enabled
from ..base_tool import BaseTool
from ..base_toolset import BaseToolset
from ..base_toolset import ToolPredicate
from ..load_mcp_resource_tool import LoadMcpResourceTool
from ..tool_configs import BaseToolConfig
from ..tool_configs import ToolArgsConfig
from .mcp_session_manager import _http_debug_var
from .mcp_session_manager import MCPSessionManager
from .mcp_session_manager import retry_on_errors
from .mcp_session_manager import SseConnectionParams
from .mcp_session_manager import StdioConnectionParams
from .mcp_session_manager import StreamableHTTPConnectionParams
from .mcp_tool import _RESERVED_TOOL_NAMES
from .mcp_tool import MCPTool
from .mcp_tool import ProgressCallbackFactory

logger = logging.getLogger("google_adk." + __name__)

ALLOW_CONFIG_STDIO_SERVERS_ENV_VAR = "ADK_ALLOW_CONFIG_STDIO_MCP_SERVERS"

# In-process override for `ALLOW_CONFIG_STDIO_SERVERS_ENV_VAR`. `None` means
# "not set, defer to the environment variable".
_allow_config_stdio_servers: Optional[bool] = None


def _set_allow_config_stdio_servers(value: Optional[bool]) -> None:
  """Overrides whether agent configs may declare stdio MCP servers.

  Applications that embed ADK and load only trusted agent configs can call this
  at startup instead of setting `ADK_ALLOW_CONFIG_STDIO_MCP_SERVERS`.

  Args:
    value: True to allow, False to deny, None to defer to the environment
      variable.
  """
  global _allow_config_stdio_servers
  _allow_config_stdio_servers = value


def _allow_config_stdio_servers_enabled() -> bool:
  """Returns whether agent configs may declare stdio MCP servers."""
  if _allow_config_stdio_servers is not None:
    return _allow_config_stdio_servers
  return is_env_enabled(ALLOW_CONFIG_STDIO_SERVERS_ENV_VAR)


T = TypeVar("T")

# Hard ceiling on cached tool lists, so a `header_provider` minting a fresh
# value per request cannot grow the cache without bound while entries are
# still unexpired.
_MAX_TOOL_LIST_CACHE_ENTRIES = 64


@dataclasses.dataclass
class _CachedToolList:
  """A ``tools/list`` response and the monotonic time it stops being usable."""

  tools: List[McpBaseTool]
  expires_at: float


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
      connection_params: (
          StdioServerParameters
          | StdioConnectionParams
          | SseConnectionParams
          | StreamableHTTPConnectionParams
      ),
      tool_filter: ToolPredicate | list[str] | None = None,
      tool_name_prefix: str | None = None,
      tool_list_cache_ttl_seconds: float | None = None,
      errlog: TextIO = sys.stderr,
      auth_scheme: AuthScheme | None = None,
      auth_credential: AuthCredential | None = None,
      require_confirmation: bool | Callable[..., bool] = False,
      header_provider: (
          Callable[
              [ReadonlyContext],
              dict[str, str] | Awaitable[dict[str, str]],
          ]
          | None
      ) = None,
      progress_callback: ProgressFnT | ProgressCallbackFactory | None = None,
      use_mcp_resources: bool | None = False,
      sampling_callback: SamplingFnT | None = None,
      sampling_capabilities: SamplingCapability | None = None,
      elicitation_callback: ElicitationFnT | None = None,
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
      tool_list_cache_ttl_seconds: If set, reuse the MCP server's ``tools/list``
        response for this many seconds instead of listing on every
        ``get_tools()`` call. Entries are keyed the way MCP sessions are pooled,
        so each ``header_provider`` identity gets its own. ADK does not
        subscribe to ``notifications/tools/list_changed``, so a tool the server
        adds or removes goes unnoticed until the entry expires. The cache lives
        on this toolset instance, so sharing it means sharing the instance.
        Defaults to None, which lists on every call.
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
      elicitation_callback: Optional callback to handle elicitation requests
        from the MCP server (``elicitation/create``), including URL-mode
        elicitations used for out-of-band flows such as auth challenges.
      credential_key: A user specified key used to load and save this credential
        in a credential service. Used with auth_scheme.
    """

    super().__init__(tool_filter=tool_filter, tool_name_prefix=tool_name_prefix)

    self._sampling_callback = sampling_callback
    self._sampling_capabilities = sampling_capabilities
    self._elicitation_callback = elicitation_callback

    if not connection_params:
      raise ValueError("Missing connection params in McpToolset.")

    if (
        tool_list_cache_ttl_seconds is not None
        and tool_list_cache_ttl_seconds <= 0
    ):
      raise ValueError(
          "tool_list_cache_ttl_seconds must be positive, got"
          f" {tool_list_cache_ttl_seconds}."
      )

    self._tool_list_cache_ttl_seconds = tool_list_cache_ttl_seconds
    # Ordered least- to most-recently used, so the cap evicts from the front.
    self._tool_list_cache: collections.OrderedDict[str, _CachedToolList] = (
        collections.OrderedDict()
    )
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
        elicitation_callback=self._elicitation_callback,
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

    return build_auth_headers(credential, self._auth_config.auth_scheme)

  @property
  def connection_params(self) -> Union[
      StdioServerParameters,
      StdioConnectionParams,
      SseConnectionParams,
      StreamableHTTPConnectionParams,
  ]:
    return self._connection_params

  @property
  def auth_scheme(self) -> Optional[AuthScheme]:
    return self._auth_scheme

  @property
  def auth_credential(self) -> Optional[AuthCredential]:
    return self._auth_credential

  @property
  def require_confirmation(self) -> Union[bool, Callable[..., bool]]:
    return self._require_confirmation

  @property
  def header_provider(
      self,
  ) -> Optional[
      Callable[
          [ReadonlyContext], Union[Dict[str, str], Awaitable[Dict[str, str]]]
      ]
  ]:
    return self._header_provider

  @property
  def errlog(self) -> TextIO:
    return self._errlog

  async def _build_headers(
      self, readonly_context: Optional[ReadonlyContext] = None
  ) -> Dict[str, str]:
    """Builds the per-request headers for an MCP session.

    Args:
      readonly_context: Context passed to the header provider and used to look
        up the exchanged credential.

    Returns:
        The merged header provider and auth headers, empty if there are none.
    """
    headers: Dict[str, str] = {}

    # Add headers from header_provider if available
    if self._header_provider and readonly_context:
      provider_headers = self._header_provider(readonly_context)
      if inspect.isawaitable(provider_headers):
        provider_headers = await provider_headers
      if provider_headers:
        headers.update(provider_headers)

    # Add auth headers from exchanged credential if available
    auth_headers = self._get_auth_headers(readonly_context)
    if auth_headers:
      headers.update(auth_headers)

    return headers

  async def _execute_with_session(
      self,
      coroutine_func: Callable[[Any], Awaitable[T]],
      error_message: str,
      readonly_context: Optional[ReadonlyContext] = None,
      headers: Optional[Dict[str, str]] = None,
  ) -> T:
    """Creates a session and executes a coroutine with it.

    Args:
      coroutine_func: Receives the session and performs the MCP call.
      error_message: Prefix for the ConnectionError raised on failure.
      readonly_context: Context used to build headers, unless `headers` is
        already supplied.
      headers: Headers from a previous `_build_headers` call, for callers that
        need them before opening the session. None means build them here; pass
        an empty dict for "no headers".

    Returns:
        Whatever `coroutine_func` returned.
    """
    current_debug: list[dict[str, Any]] = []
    debug_token = (
        _http_debug_var.set(current_debug)
        if logger.isEnabledFor(logging.DEBUG)
        else None
    )
    if headers is None:
      headers = await self._build_headers(readonly_context)

    session_headers = headers if headers else None
    try:
      session = await self._mcp_session_manager.create_session(
          headers=session_headers
      )
      timeout_in_seconds = (
          self._connection_params.timeout
          if hasattr(self._connection_params, "timeout")
          else None
      )
      # Hold the session out of the pool's idle sweep while it is in use, so
      # another caller's sweep cannot close the transport mid-call.
      self._mcp_session_manager._begin_session_use(session_headers)  # pylint: disable=protected-access
      try:
        return await asyncio.wait_for(
            coroutine_func(session), timeout=timeout_in_seconds
        )
      except Exception as e:
        logger.exception(
            f"Exception during MCP session execution: {error_message}: {e}"
        )
        raise ConnectionError(f"{error_message}: {e}") from e
      finally:
        self._mcp_session_manager._end_session_use(session_headers)  # pylint: disable=protected-access
    finally:
      if debug_token is not None:
        _http_debug_var.reset(debug_token)
        if current_debug and readonly_context is not None:
          # pylint: disable=protected-access
          inv_ctx = getattr(readonly_context, "_invocation_context", None)
          if inv_ctx is not None:
            inv_ctx._custom_metadata.setdefault("http_debug_info", []).extend(
                current_debug
            )

  def _tool_list_cache_key(self, headers: Dict[str, str]) -> Optional[str]:
    """Returns the cache key for these headers, or None if caching is off."""
    if self._tool_list_cache_ttl_seconds is None:
      return None
    # Key on the session pool key, so an entry never outlives the identity it
    # was fetched with: a different tenant reaches a different session.
    # pylint: disable-next=protected-access
    return self._mcp_session_manager._session_key_for(headers or None)

  def _read_tool_list_cache(
      self, cache_key: Optional[str]
  ) -> Optional[List[McpBaseTool]]:
    """Returns the unexpired cached tool list, or None on a miss."""
    if cache_key is None:
      return None
    entry = self._tool_list_cache.get(cache_key)
    if entry is None:
      return None
    if entry.expires_at <= time.monotonic():
      del self._tool_list_cache[cache_key]
      return None
    self._tool_list_cache.move_to_end(cache_key)
    return entry.tools

  def _write_tool_list_cache(
      self, cache_key: Optional[str], mcp_tools: List[McpBaseTool]
  ) -> None:
    """Caches a tool list, unless caching is off."""
    ttl_seconds = self._tool_list_cache_ttl_seconds
    if cache_key is None or ttl_seconds is None:
      return

    # Reads only evict the key they were asked for, so a key that never comes
    # back is never reclaimed. Bound the cache here instead: sweep what has
    # expired, then cap what has not.
    now = time.monotonic()
    for key, entry in list(self._tool_list_cache.items()):
      if entry.expires_at <= now:
        del self._tool_list_cache[key]

    self._tool_list_cache[cache_key] = _CachedToolList(
        tools=list(mcp_tools),
        expires_at=now + ttl_seconds,
    )
    self._tool_list_cache.move_to_end(cache_key)
    while len(self._tool_list_cache) > _MAX_TOOL_LIST_CACHE_ENTRIES:
      self._tool_list_cache.popitem(last=False)

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
    headers = await self._build_headers(readonly_context)
    cache_key = self._tool_list_cache_key(headers)
    mcp_tools = self._read_tool_list_cache(cache_key)

    if mcp_tools is None:
      # Fetch available tools from the MCP server
      tools_response: ListToolsResult = await self._execute_with_session(
          lambda session: session.list_tools(),
          "Failed to get tools from MCP server",
          readonly_context,
          headers=headers,
      )
      mcp_tools = tools_response.tools
      self._write_tool_list_cache(cache_key, mcp_tools)

    # Apply filtering based on context and tool_filter. This runs on every call
    # even on a cache hit, so only the round trip is skipped.
    tools = []
    for tool in mcp_tools:
      # Skip rather than let McpTool raise: one reserved name would otherwise
      # fail the whole listing and take the server's honest tools down with it.
      if tool.name in _RESERVED_TOOL_NAMES:
        logger.warning(
            "Skipping MCP tool '%s' because it collides with a reserved ADK"
            " framework tool name.",
            tool.name,
        )
        continue

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

    # Sort by name for a stable order across turns. The MCP server's
    # list_tools() order is not contractual; an unstable order would
    # invalidate the context cache every turn.
    tools.sort(key=lambda tool: tool.name)

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
    self._tool_list_cache.clear()
    try:
      await self._mcp_session_manager.close()
    except Exception as e:
      # Log the error but don't re-raise to avoid blocking shutdown
      logger.warning("Error during McpToolset cleanup: %s", e)

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

    if (
        mcp_toolset_config.stdio_server_params
        or mcp_toolset_config.stdio_connection_params
    ) and not _allow_config_stdio_servers_enabled():
      raise ValueError(
          "Stdio MCP servers are not allowed in agent configs: the"
          " config-supplied 'command' is launched as a local process when the"
          " agent starts, so an untrusted config would be able to run"
          " arbitrary code. Construct the McpToolset in Python code instead,"
          " use a remote transport (sse_connection_params or"
          " streamable_http_connection_params), or set"
          f" {ALLOW_CONFIG_STDIO_SERVERS_ENV_VAR}=1 if this application only"
          " loads agent configs it trusts."
      )

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
        tool_list_cache_ttl_seconds=(
            mcp_toolset_config.tool_list_cache_ttl_seconds
        ),
        auth_scheme=mcp_toolset_config.auth_scheme,
        auth_credential=mcp_toolset_config.auth_credential,
        credential_key=mcp_toolset_config.credential_key,
        use_mcp_resources=mcp_toolset_config.use_mcp_resources,
    )

  def __getstate__(self):
    """Custom pickling to exclude non-picklable runtime objects."""
    state = self.__dict__.copy()
    # Remove unpicklable file-like objects
    state.pop("_errlog", None)
    # The session pool does not survive pickling, so neither should tool lists
    # cached against its keys.
    state["_tool_list_cache"] = collections.OrderedDict()
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

  tool_list_cache_ttl_seconds: float | None = None

  auth_scheme: Optional[AuthScheme] = None

  auth_credential: Optional[AuthCredential] = None

  credential_key: str | None = None

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
