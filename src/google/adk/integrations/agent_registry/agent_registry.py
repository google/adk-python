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

"""Client library for interacting with the Google Cloud Agent Registry within ADK."""

from __future__ import annotations

from enum import Enum
import logging
import os
import re
import threading
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Literal
from typing import Mapping
from typing import TypedDict
from urllib.parse import urlparse

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_schemes import AuthScheme
from google.adk.auth.auth_tool import AuthConfig
from google.adk.integrations.agent_identity.gcp_auth_provider_scheme import GcpAuthProviderScheme
from google.adk.telemetry.tracing import GCP_MCP_SERVER_DESTINATION_ID
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.utils import _mtls_utils
from google.adk.utils._google_client_headers import merge_tracking_headers
import google.auth
from google.auth.transport import mtls
from google.auth.transport import requests as requests_auth
import httpx
import requests
from typing_extensions import override

from ...dependencies._mcp import StdioServerParameters

# pylint: disable=g-import-not-at-top
try:
  from a2a.types import AgentSkill
  from google.adk.a2a import _compat
  from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
except ImportError as e:
  raise ImportError(
      "AgentRegistry requires the 'a2a-sdk' package. "
      "Please install it using 'pip install google-adk[a2a]'."
  ) from e
# pylint: enable=g-import-not-at-top

logger = logging.getLogger("google_adk." + __name__)

AGENT_REGISTRY_BASE_URL = "https://agentregistry.googleapis.com/v1"
AGENT_REGISTRY_MTLS_BASE_URL = "https://agentregistry.mtls.googleapis.com/v1"

_TRANSPORT_MAPPING = {
    "HTTP_JSON": _compat.TP_HTTP_JSON,
    "JSONRPC": _compat.TP_JSONRPC,
    "GRPC": _compat.TP_GRPC,
}

# The runtime binding lookup blocks the event loop, so it must not wait long.
_REBIND_TIMEOUT_SECONDS = 10.0
_RUNTIME_REFERENCE_ATTRIBUTE = (
    "agentregistry.googleapis.com/system/RuntimeReference"
)


# An MCPToolset for a single registered MCP server. Adds the special
# gcp.mcp.server.destination.id custom_metadata key on each returned tool. This special key is
# added to execute_tool spans in google.adk.telemetry.tracing
class AgentRegistrySingleMcpToolset(McpToolset):

  # Set when the auth scheme came from a binding. The class default also covers
  # toolsets pickled by versions that did not have it.
  _binding_registry: AgentRegistry | None = None

  def __init__(
      self,
      *,
      destination_resource_id: str | None,
      connection_params: (
          StdioServerParameters
          | StdioConnectionParams
          | SseConnectionParams
          | StreamableHTTPConnectionParams
      ),
      tool_name_prefix: str | None = None,
      header_provider: (
          Callable[[ReadonlyContext], Dict[str, str]] | None
      ) = None,
      auth_scheme: AuthScheme | None = None,
      auth_credential: AuthCredential | None = None,
      binding_registry: AgentRegistry | None = None,
  ):
    super().__init__(
        connection_params=connection_params,
        tool_name_prefix=tool_name_prefix,
        header_provider=header_provider,
        auth_scheme=auth_scheme,
        auth_credential=auth_credential,
    )
    self.destination_resource_id = destination_resource_id
    self._binding_registry = binding_registry
    self._rebind_lock = threading.Lock()

  def __getstate__(self) -> Dict[str, Any]:
    state: Dict[str, Any] = super().__getstate__()  # type: ignore[no-untyped-call]
    state.pop("_rebind_lock", None)
    return state

  def __setstate__(self, state: Dict[str, Any]) -> None:
    super().__setstate__(state)  # type: ignore[no-untyped-call]
    self._rebind_lock = threading.Lock()

  @override
  def get_auth_config(self) -> AuthConfig | None:
    if self._binding_registry and os.environ.get(
        "GOOGLE_CLOUD_AGENT_ENGINE_ID"
    ):
      # Agent Engine serves requests on several threads. Until the first one
      # has rebound, the others wait here rather than use the old scheme.
      with self._rebind_lock:
        if self._binding_registry:
          self._rebind_for_deployment(self._binding_registry)
          self._binding_registry = None
    return super().get_auth_config()

  def _rebind_for_deployment(self, registry: AgentRegistry) -> None:
    """Re-resolves the binding for the Agent Engine deployment.

    The toolset is usually built on the machine that deploys the agent, so its
    binding was chosen before there was a deployment to scope the lookup to.

    Args:
      registry: The registry the binding was first resolved with.
    """
    resource_id = self.destination_resource_id
    try:
      source_identifier = (
          registry._get_deployed_agent_id()  # pylint: disable=protected-access
          if resource_id
          else None
      )
    except Exception as e:  # pylint: disable=broad-except
      logger.warning(
          "Failed to look up this Agent Engine deployment in Agent Registry;"
          " keeping the auth provider chosen at deploy time for %s: %s",
          resource_id,
          e,
      )
      return
    if not resource_id or not source_identifier:
      logger.warning(
          "Could not find this Agent Engine deployment in Agent Registry;"
          " keeping the auth provider chosen at deploy time for %s.",
          resource_id,
      )
      return
    continue_uri = (
        self._auth_scheme.continue_uri
        if isinstance(self._auth_scheme, GcpAuthProviderScheme)
        else None
    )
    try:
      auth_scheme = registry._fetch_available_auth_scheme(  # pylint: disable=protected-access
          resource_id, source_identifier, continue_uri=continue_uri
      )
    except Exception as e:  # pylint: disable=broad-except
      logger.warning(
          "Failed to fetch the bindings available to %s; keeping the auth"
          " provider chosen at deploy time for %s: %s",
          source_identifier,
          resource_id,
          e,
      )
      return
    if auth_scheme is None:
      # Deleting a binding has to revoke it, not fall back to it.
      logger.warning(
          "Agent Registry has no binding for %s available to %s; calling it"
          " without an auth provider.",
          resource_id,
          source_identifier,
      )
      self._auth_scheme = None
      self._auth_config = None
      return
    logger.info(
        "Using auth provider %s for %s, from the bindings available to %s.",
        auth_scheme.name,
        resource_id,
        source_identifier,
    )
    self._auth_scheme = auth_scheme
    self._auth_config = AuthConfig(
        auth_scheme=auth_scheme, raw_auth_credential=self._auth_credential
    )

  @override
  async def get_tools(
      self, readonly_context: ReadonlyContext | None = None
  ) -> List[BaseTool]:
    tools: List[BaseTool] = await super().get_tools(readonly_context)

    # Noop if there is no destination_resource_id
    if self.destination_resource_id is None:
      return tools

    for tool in tools:
      if not tool.custom_metadata:
        tool.custom_metadata = {}

      tool.custom_metadata[GCP_MCP_SERVER_DESTINATION_ID] = (
          self.destination_resource_id
      )
    return tools


class _MtlsEndpoint(Enum):
  """The mTLS endpoint setting."""

  AUTO = "auto"
  ALWAYS = "always"
  NEVER = "never"


class _ProtocolType(str, Enum):
  """Supported agent protocol types."""

  TYPE_UNSPECIFIED = "TYPE_UNSPECIFIED"
  A2A_AGENT = "A2A_AGENT"
  CUSTOM = "CUSTOM"


class Interface(TypedDict, total=False):
  """Details for a single connection interface."""

  url: str
  protocolBinding: str


class Endpoint(TypedDict, total=False):
  """Full metadata for a registered Endpoint."""

  name: str
  endpointId: str
  displayName: str
  description: str
  interfaces: List[Interface]
  createTime: str
  updateTime: str
  attributes: Dict[str, Any]


def _is_google_api(url: str) -> bool:
  """Checks if the given URL points to a Google API endpoint over https."""
  parsed_url = urlparse(url)
  if parsed_url.scheme != "https" or not parsed_url.hostname:
    return False
  return (
      parsed_url.hostname == "googleapis.com"
      or parsed_url.hostname.endswith(".googleapis.com")
  )


def _update_time_key(binding: Dict[str, Any]) -> tuple[str, str]:
  """Returns a sort key for a binding's `updateTime`, which is always in UTC."""
  # The fraction has 0, 3, 6 or 9 digits, so pad it before comparing.
  seconds, _, fraction = (
      binding.get("updateTime", "").removesuffix("Z").partition(".")
  )
  return seconds, fraction.ljust(9, "0")


def _select_auth_provider(
    bindings: List[Dict[str, Any]], resource_name: str
) -> str | None:
  """Returns the auth provider of the most recently updated binding.

  Args:
    bindings: Bindings targeting the resource with an auth provider.
    resource_name: The resource name, only used for logging.
  """
  if not bindings:
    return None
  newest = max(bindings, key=_update_time_key)
  chosen: str = newest["authProviderBinding"]["authProvider"]
  auth_providers = sorted(
      {b["authProviderBinding"]["authProvider"] for b in bindings}
  )
  if len(auth_providers) > 1:
    logger.warning(
        "%s is bound to %d auth providers (%s); using %s, from the most"
        " recently updated binding.",
        resource_name,
        len(auth_providers),
        ", ".join(auth_providers),
        chosen,
    )
  return chosen


class AgentRegistry:
  """Client for interacting with the Google Cloud Agent Registry service.

  Unlike a standard REST client library, this class provides higher-level
  abstractions for ADK integration. It surfaces the agent registry service
  methods along with helper methods like `get_mcp_toolset` and
  `get_remote_a2a_agent` that automatically resolve connection details and
  handle authentication to produce ready-to-use ADK components.
  """

  # The class default also covers registries pickled by older versions.
  _deployed_agent_id: str | None = None

  def __init__(
      self,
      project_id: str | None = None,
      location: str | None = None,
      header_provider: (
          Callable[[ReadonlyContext], Dict[str, str]] | None
      ) = None,
  ):
    """Initializes the AgentRegistry client.

    Args:
      project_id: The Google Cloud project ID.
      location: The Google Cloud location (region).
      header_provider: Optional provider for custom headers.
    """
    self.project_id = project_id
    self.location = location

    if not self.project_id or not self.location:
      raise ValueError("project_id and location must be provided")

    self._base_path = f"projects/{self.project_id}/locations/{self.location}"
    self._header_provider = header_provider
    self._connect_lock = threading.Lock()
    self._connect()

  def _connect(self) -> None:
    """Loads default credentials and configures the session that uses them."""
    try:
      credentials, _ = google.auth.default()
    except google.auth.exceptions.DefaultCredentialsError as e:
      raise RuntimeError(
          f"Failed to get default Google Cloud credentials: {e}"
      ) from e

    session = requests_auth.AuthorizedSession(credentials=credentials)
    use_client_cert = _use_client_cert_effective()
    client_cert_source = None
    if use_client_cert:
      client_cert_source = (
          mtls.default_client_cert_source()
          if mtls.has_default_client_cert_source()
          else None
      )
      session.configure_mtls_channel(client_cert_source)
    self._use_mtls = _should_use_mtls_endpoint(client_cert_source)
    self._base_url = (
        AGENT_REGISTRY_MTLS_BASE_URL
        if self._use_mtls
        else AGENT_REGISTRY_BASE_URL
    )
    self._credentials = credentials
    # Set last: other threads treat a non-None session as fully connected.
    self._session = session

  def _ensure_connected(self) -> None:
    if self._session is not None:
      return
    with self._connect_lock:
      if self._session is None:
        self._connect()

  def __getstate__(self) -> Dict[str, Any]:
    state = self.__dict__.copy()
    # Toolsets built here are pickled when an agent is deployed. Carrying the
    # credentials along would ship the deployer's default credentials, refresh
    # token included, into the deployed agent, which would then call Google
    # APIs as the deployer. The copy loads its own on first use instead.
    state["_credentials"] = None
    state["_session"] = None
    del state["_connect_lock"]
    return state

  def __setstate__(self, state: Dict[str, Any]) -> None:
    self.__dict__.update(state)
    self._connect_lock = threading.Lock()

  def _get_auth_headers(self) -> Dict[str, str]:
    """Refreshes credentials and returns authorization headers."""
    self._ensure_connected()
    try:
      request = google.auth.transport.requests.Request()
      self._credentials.refresh(request)
      headers = {
          "Authorization": f"Bearer {self._credentials.token}",
          "Content-Type": "application/json",
      }
      return headers
    except google.auth.exceptions.RefreshError as e:
      raise RuntimeError(
          f"Failed to refresh Google Cloud credentials: {e}"
      ) from e

  def _make_request(
      self,
      path: str,
      method: str = "GET",
      params: Dict[str, Any] | None = None,
      json_data: Dict[str, Any] | None = None,
      timeout: float | None = None,
  ) -> Dict[str, Any]:
    """Helper function to make requests to the Agent Registry API."""
    self._ensure_connected()
    if path.startswith("projects/"):
      url = f"{self._base_url}/{path}"
    else:
      url = f"{self._base_url}/{self._base_path}/{path}"
    quota_project_id = (
        getattr(self._credentials, "quota_project_id", None) or self.project_id
    )
    headers = merge_tracking_headers(
        {"x-goog-user-project": quota_project_id} if quota_project_id else {}
    )
    try:
      # Using AuthorizedSession for internal API calls to handle mTLS/Auth.
      if method == "POST":
        response = self._session.post(url, headers=headers, json=json_data)
      else:
        # Without a timeout, AuthorizedSession applies its own default.
        kwargs = {"timeout": timeout} if timeout else {}
        response = self._session.get(
            url, headers=headers, params=params, **kwargs
        )
      response.raise_for_status()
      data: Dict[str, Any] = response.json()
      return data
    except requests.exceptions.HTTPError as e:
      raise RuntimeError(
          f"API request failed with status {e.response.status_code}:"
          f" {e.response.text}"
      ) from e
    except requests.exceptions.RequestException as e:
      raise RuntimeError(f"API request failed (network error): {e}") from e
    except Exception as e:
      raise RuntimeError(f"API request failed: {e}") from e

  def _search(
      self,
      resource_type: str,
      *,
      search_string: str | None = None,
      search_type: Literal["KEYWORD", "SEMANTIC"] | None = None,
      filter_str: str | None = None,
      order_by: str | None = None,
      page_size: int | None = None,
      page_token: str | None = None,
  ) -> Dict[str, Any]:
    """Helper function to execute search requests."""
    json_data: dict[str, Any] = {}
    if search_string is not None:
      json_data["searchString"] = search_string
    if search_type is not None:
      json_data["searchType"] = search_type
    if filter_str is not None:
      json_data["filter"] = filter_str
    if order_by is not None:
      json_data["orderBy"] = order_by
    if page_size is not None:
      json_data["pageSize"] = page_size
    if page_token is not None:
      json_data["pageToken"] = page_token
    return self._make_request(
        f"{resource_type}:search", method="POST", json_data=json_data
    )

  def _get_connection_uri(
      self,
      resource_details: Mapping[str, Any],
      protocol_type: _ProtocolType | None = None,
      protocol_binding: _compat.TransportProtocol | None = None,
  ) -> tuple[str | None, str | None, _compat.TransportProtocol | None]:
    """Extracts the first matching URI based on type and binding filters."""
    protocols = list(resource_details.get("protocols", []))
    if "interfaces" in resource_details:
      protocols.append({"interfaces": resource_details["interfaces"]})

    for p in protocols:
      if protocol_type and p.get("type") != protocol_type:
        continue
      protocol_version = p.get("protocolVersion")
      for i in p.get("interfaces", []):
        mapped_binding = _TRANSPORT_MAPPING.get(i.get("protocolBinding"))
        if protocol_binding and mapped_binding != protocol_binding:
          continue
        if url := i.get("url"):
          if self._use_mtls:
            url = _mtls_utils.effective_googleapis_endpoint(url)
          return url, protocol_version, mapped_binding

    return None, None, None

  def _resolve_auth_provider_scheme(
      self,
      resource_id: str | None,
      resource_name: str,
      *,
      continue_uri: str | None = None,
  ) -> GcpAuthProviderScheme | None:
    """Resolves the auth scheme a registered resource is bound to.

    Args:
      resource_id: The stable identifier of the resource (for example an
        `agentId` or an `mcpServerId`), matched against the binding targets.
      resource_name: The resource name, only used for logging.
      continue_uri: Optional continue URI to override what is in the auth
        provider.

    Returns:
      The scheme for the bound auth provider, or None if the resource is not
      bound to one or the bindings could not be read.
    """
    if not resource_id:
      return None
    try:
      auth_provider = _select_auth_provider(
          self._list_auth_provider_bindings("bindings", resource_id),
          resource_name,
      )
    except Exception as e:  # pylint: disable=broad-except
      logger.warning("Failed to fetch bindings for %s: %s", resource_name, e)
      return None
    if not auth_provider:
      return None
    return GcpAuthProviderScheme(name=auth_provider, continue_uri=continue_uri)

  def _get_deployed_agent_id(self) -> str | None:
    """Returns the Agent Registry ID of the Agent Engine deployment, if found.

    Raises:
      RuntimeError: If the agents could not be listed.
    """
    if self._deployed_agent_id:
      return self._deployed_agent_id
    engine_id = os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID")
    location = os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_LOCATION")
    if not engine_id or not location:
      return None
    runtime_path = f"/locations/{location}/reasoningEngines/{engine_id}"
    data = self._make_request(
        "agents",
        params={
            "filter": (
                f'attributes."{_RUNTIME_REFERENCE_ATTRIBUTE}".uri:'
                f'"{runtime_path}"'
            )
        },
        timeout=_REBIND_TIMEOUT_SECONDS,
    )
    for agent in data.get("agents", []):
      uri = (
          agent.get("attributes", {})
          .get(_RUNTIME_REFERENCE_ATTRIBUTE, {})
          .get("uri", "")
      )
      # The filter matches substrings, so engine 12 would also match 123.
      if uri.endswith(runtime_path) and agent.get("agentId"):
        self._deployed_agent_id = agent["agentId"]
        return self._deployed_agent_id
    return None

  def _fetch_available_auth_scheme(
      self,
      resource_id: str,
      source_identifier: str,
      *,
      continue_uri: str | None = None,
  ) -> GcpAuthProviderScheme | None:
    """Resolves the auth scheme a deployed agent may use for a resource.

    Unlike ListBindings, FetchAvailableBindings only returns bindings made for
    this agent or for no agent in particular, and drops those whose auth
    provider the caller cannot retrieve credentials from.

    Args:
      resource_id: The stable identifier of the target resource.
      source_identifier: The deployed agent's Agent Registry identifier.
      continue_uri: Optional continue URI to override what is in the auth
        provider.

    Returns:
      The scheme, or None if no binding is available to the agent.

    Raises:
      RuntimeError: If the bindings could not be read.
    """
    bindings = self._list_auth_provider_bindings(
        "bindings:fetchAvailable",
        resource_id,
        params={
            "sourceIdentifier": source_identifier,
            "targetIdentifier": resource_id,
        },
        timeout=_REBIND_TIMEOUT_SECONDS,
    )
    own_bindings = [
        b
        for b in bindings
        if b.get("source", {}).get("identifier") == source_identifier
    ]
    auth_provider = _select_auth_provider(own_bindings or bindings, resource_id)
    if not auth_provider:
      return None
    return GcpAuthProviderScheme(name=auth_provider, continue_uri=continue_uri)

  def _list_auth_provider_bindings(
      self,
      path: str,
      resource_id: str,
      params: Dict[str, str] | None = None,
      timeout: float | None = None,
  ) -> List[Dict[str, Any]]:
    """Returns every binding at `path` targeting `resource_id` with an auth provider.

    Args:
      path: The bindings method to call.
      resource_id: The stable identifier of the target resource.
      params: Query parameters for every page.
      timeout: Seconds to wait for each page.

    Raises:
      RuntimeError: If a request fails.
    """
    params = dict(params or {})
    bindings = []
    while True:
      data = self._make_request(path, params=params, timeout=timeout)
      for b in data.get("bindings", []):
        target_id = b.get("target", {}).get("identifier", "")
        auth_provider = b.get("authProviderBinding", {}).get("authProvider")
        if target_id.endswith(resource_id) and auth_provider:
          bindings.append(b)
      page_token = data.get("nextPageToken")
      if not page_token:
        return bindings
      params["pageToken"] = page_token

  def _clean_name(self, name: str) -> str:
    """Cleans a string to be a valid Python identifier for agent names."""
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    clean = re.sub(r"_+", "_", clean)
    clean = clean.strip("_")
    if clean and not clean[0].isalpha() and clean[0] != "_":
      clean = "_" + clean
    return clean

  # --- MCP Server Methods ---

  def list_mcp_servers(
      self,
      filter_str: str | None = None,
      page_size: int | None = None,
      page_token: str | None = None,
  ) -> Dict[str, Any]:
    """Fetches a list of MCP Servers."""
    params = {}
    if filter_str:
      params["filter"] = filter_str
    if page_size:
      params["pageSize"] = str(page_size)
    if page_token:
      params["pageToken"] = page_token
    return self._make_request("mcpServers", params=params)

  def search_mcp_servers(
      self,
      *,
      search_string: str | None = None,
      search_type: Literal["KEYWORD", "SEMANTIC"] | None = None,
      filter_str: str | None = None,
      order_by: str | None = None,
      page_size: int | None = None,
      page_token: str | None = None,
  ) -> Dict[str, Any]:
    """Searches registered MCP Servers."""
    return self._search(
        "mcpServers",
        search_string=search_string,
        search_type=search_type,
        filter_str=filter_str,
        order_by=order_by,
        page_size=page_size,
        page_token=page_token,
    )

  def get_mcp_server(self, name: str) -> Dict[str, Any]:
    """Retrieves details of a specific MCP Server."""
    return self._make_request(name)

  def get_mcp_toolset(
      self,
      mcp_server_name: str,
      auth_scheme: AuthScheme | None = None,
      auth_credential: AuthCredential | None = None,
      *,
      continue_uri: str | None = None,
  ) -> McpToolset:
    """Constructs an McpToolset from a registered MCP Server.

    If `auth_scheme` is omitted, it is automatically resolved from the server's
    IAM bindings via `GcpAuthProviderScheme`.

    Args:
      mcp_server_name: Resource name of the MCP Server.
      auth_scheme: Optional auth scheme. Resolved via bindings if omitted.
      auth_credential: Optional auth credential.
      continue_uri: Optional continue URI to override what is in the auth
        provider.

    Returns:
      An McpToolset for the MCP server.
    """
    server_details = self.get_mcp_server(mcp_server_name)
    name = self._clean_name(server_details.get("displayName", mcp_server_name))
    mcp_server_id = server_details.get("mcpServerId")
    if not isinstance(mcp_server_id, str):
      mcp_server_id = None

    endpoint_uri, _, _ = self._get_connection_uri(
        server_details, protocol_binding=_compat.TP_JSONRPC
    )
    if not endpoint_uri:
      endpoint_uri, _, _ = self._get_connection_uri(
          server_details, protocol_binding=_compat.TP_HTTP_JSON
      )
    if not endpoint_uri:
      raise ValueError(
          f"MCP Server endpoint URI not found for: {mcp_server_name}"
      )

    binding_registry = None
    if not auth_scheme:
      auth_scheme = self._resolve_auth_provider_scheme(
          mcp_server_id, mcp_server_name, continue_uri=continue_uri
      )
      # Without a binding, combined_header_provider sends default credentials
      # to Google API endpoints, so only a binding's scheme is re-resolved.
      if auth_scheme:
        binding_registry = self

    connection_params = StreamableHTTPConnectionParams(
        url=endpoint_uri,
    )

    def combined_header_provider(context: ReadonlyContext) -> Dict[str, str]:
      headers = {}
      if (
          not auth_scheme
          and not auth_credential
          and _is_google_api(endpoint_uri)
      ):
        headers.update(self._get_auth_headers())
      if self._header_provider:
        headers.update(self._header_provider(context))
      return headers

    return AgentRegistrySingleMcpToolset(
        destination_resource_id=mcp_server_id,
        connection_params=connection_params,
        tool_name_prefix=name,
        header_provider=combined_header_provider,
        auth_scheme=auth_scheme,
        auth_credential=auth_credential,
        binding_registry=binding_registry,
    )

  # --- Endpoint Methods ---

  def list_endpoints(
      self,
      filter_str: str | None = None,
      page_size: int | None = None,
      page_token: str | None = None,
  ) -> Dict[str, Any]:
    """Fetches a list of Endpoints."""
    params = {}
    if filter_str:
      params["filter"] = filter_str
    if page_size:
      params["pageSize"] = str(page_size)
    if page_token:
      params["pageToken"] = page_token
    return self._make_request("endpoints", params=params)

  def get_endpoint(self, name: str) -> Endpoint:
    """Retrieves details of a specific Endpoint."""
    return self._make_request(name)  # type: ignore

  def get_model_name(self, endpoint_name: str) -> str:
    """Retrieves and parses an endpoint into a model resource name.

    Args:
      endpoint_name: The full resource name of the endpoint.

    Returns:
      The resolved model resource name string (e.g.
      projects/.../locations/.../publishers/google/models/...).
    """
    endpoint_details = self.get_endpoint(endpoint_name)
    uri, _, _ = self._get_connection_uri(endpoint_details)
    if not uri:
      raise ValueError(
          f"Connection URI not found for endpoint: {endpoint_name}"
      )

    uri = re.sub(r":\w+$", "", uri)

    if uri.startswith("projects/"):
      return uri

    match = re.search(r"(projects/.+)", uri)
    if match:
      return match.group(1)

    return uri

  # --- Agent Methods ---

  def list_agents(
      self,
      filter_str: str | None = None,
      page_size: int | None = None,
      page_token: str | None = None,
  ) -> Dict[str, Any]:
    """Fetches a list of registered A2A Agents."""
    params = {}
    if filter_str:
      params["filter"] = filter_str
    if page_size:
      params["pageSize"] = str(page_size)
    if page_token:
      params["pageToken"] = page_token
    return self._make_request("agents", params=params)

  def search_agents(
      self,
      *,
      search_string: str | None = None,
      search_type: Literal["KEYWORD", "SEMANTIC"] | None = None,
      filter_str: str | None = None,
      order_by: str | None = None,
      page_size: int | None = None,
      page_token: str | None = None,
  ) -> Dict[str, Any]:
    """Searches registered A2A Agents."""
    return self._search(
        "agents",
        search_string=search_string,
        search_type=search_type,
        filter_str=filter_str,
        order_by=order_by,
        page_size=page_size,
        page_token=page_token,
    )

  def get_agent_info(self, name: str) -> Dict[str, Any]:
    """Retrieves detailed metadata of a specific A2A Agent."""
    return self._make_request(name)

  def get_remote_a2a_agent(
      self,
      agent_name: str,
      auth_scheme: AuthScheme | None = None,
      auth_credential: AuthCredential | None = None,
      *,
      httpx_client: httpx.AsyncClient | None = None,
      continue_uri: str | None = None,
  ) -> RemoteA2aAgent:
    """Creates a RemoteA2aAgent instance for a registered A2A Agent.

    If `auth_scheme` is omitted, it is automatically resolved from the agent's
    IAM bindings via `GcpAuthProviderScheme`.

    Args:
      agent_name: Resource name of the A2A Agent.
      auth_scheme: Optional auth scheme. Resolved via bindings if omitted.
      auth_credential: Optional auth credential.
      httpx_client: Optional shared HTTP client.
      continue_uri: Optional continue URI to override what is in the auth
        provider.

    Returns:
      A RemoteA2aAgent for the registered agent.
    """
    agent_info = self.get_agent_info(agent_name)

    agent_id = agent_info.get("agentId")
    if not isinstance(agent_id, str):
      agent_id = None
    if not auth_scheme:
      auth_scheme = self._resolve_auth_provider_scheme(
          agent_id, agent_name, continue_uri=continue_uri
      )

    # Try to use the full agent card if available
    card = agent_info.get("card", {})
    card_content = card.get("content")
    if card.get("type") == "A2A_AGENT_CARD" and card_content:
      agent_card = _compat.parse_agent_card(card_content)
      # Clean the name to be a valid identifier
      name = self._clean_name(agent_card.name)

      return RemoteA2aAgent(
          name=name,
          agent_card=agent_card,
          description=agent_card.description,
          httpx_client=httpx_client,
          auth_scheme=auth_scheme,
          auth_credential=auth_credential,
      )

    name = self._clean_name(agent_info.get("displayName", agent_name))
    description = agent_info.get("description", "")
    version = agent_info.get("version", "")

    url, protocol_version, protocol_binding = self._get_connection_uri(
        agent_info, protocol_type=_ProtocolType.A2A_AGENT
    )
    if not url:
      raise ValueError(f"A2A connection URI not found for Agent: {agent_name}")

    skills = []
    for s in agent_info.get("skills", []):
      skills.append(
          AgentSkill(
              id=s.get("id"),
              name=s.get("name"),
              description=s.get("description", ""),
              tags=s.get("tags", []),
              examples=s.get("examples", []),
          )
      )

    binding = protocol_binding or _compat.TP_HTTP_JSON
    agent_card = _compat.build_agent_card(
        name=name,
        description=description,
        version=version,
        url=url,
        protocol_binding=getattr(binding, "value", binding),
        protocol_version=protocol_version,
        skills=skills,
        default_input_modes=["text"],
        default_output_modes=["text"],
    )

    return RemoteA2aAgent(
        name=name,
        agent_card=agent_card,
        description=description,
        httpx_client=httpx_client,
        auth_scheme=auth_scheme,
        auth_credential=auth_credential,
    )


def _use_client_cert_effective() -> bool:
  """Returns whether client certificate should be used for mTLS."""
  try:
    # If the google.auth.transport.mtls.should_use_client_cert function is
    # available, use it to determine whether client certificate should be used.
    return bool(mtls.should_use_client_cert())
  except (ImportError, AttributeError):
    use_client_cert_str = os.getenv(
        "GOOGLE_API_USE_CLIENT_CERTIFICATE", "false"
    ).lower()
    return use_client_cert_str == "true"


def _should_use_mtls_endpoint(client_cert_source: Any | None = None) -> bool:
  """Returns whether the mTLS endpoint should be used."""
  try:
    return bool(
        mtls.should_use_mtls_endpoint(
            client_cert_available=client_cert_source is not None
        )
    )
  except (ImportError, AttributeError):
    pass
  use_mtls_endpoint_str = os.getenv(
      "GOOGLE_API_USE_MTLS_ENDPOINT", _MtlsEndpoint.AUTO.value
  ).lower()
  try:
    use_mtls_endpoint = _MtlsEndpoint(use_mtls_endpoint_str)
  except ValueError:
    use_mtls_endpoint = _MtlsEndpoint.AUTO
  return (use_mtls_endpoint is _MtlsEndpoint.ALWAYS) or (
      use_mtls_endpoint is _MtlsEndpoint.AUTO and client_cert_source is not None
  )
