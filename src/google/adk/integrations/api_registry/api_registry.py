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

from enum import Enum
import os
from typing import Any
from typing import Callable
from urllib.parse import urlparse

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.base_toolset import ToolPredicate
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
import google.auth
from google.auth.transport import mtls
from google.auth.transport import requests as requests_auth
import google.auth.transport.requests
import requests

API_REGISTRY_URL = "https://cloudapiregistry.googleapis.com"
API_REGISTRY_MTLS_URL = "https://cloudapiregistry.mtls.googleapis.com"


class _MtlsEndpoint(Enum):
  """The mTLS endpoint setting."""

  AUTO = "auto"
  ALWAYS = "always"
  NEVER = "never"


def _is_google_api(url: str) -> bool:
  """Returns True if the URL points to a Google API host."""
  hostname = urlparse(url).hostname
  return hostname is not None and (
      hostname == "googleapis.com" or hostname.endswith(".googleapis.com")
  )


def _use_client_cert_effective() -> bool:
  """Returns whether client certificate should be used for mTLS."""
  try:
    # Prefer google.auth.transport.mtls.should_use_client_cert when available.
    return bool(mtls.should_use_client_cert())
  except (ImportError, AttributeError):
    use_client_cert_str = os.getenv(
        "GOOGLE_API_USE_CLIENT_CERTIFICATE", "false"
    ).lower()
    return use_client_cert_str == "true"


def _get_api_registry_base_url(client_cert_source: Any | None = None) -> str:
  """Returns the base URL based on mTLS configuration and cert availability."""
  use_mtls_endpoint_str = os.getenv(
      "GOOGLE_API_USE_MTLS_ENDPOINT", _MtlsEndpoint.AUTO.value
  ).lower()
  try:
    use_mtls_endpoint = _MtlsEndpoint(use_mtls_endpoint_str)
  except ValueError:
    use_mtls_endpoint = _MtlsEndpoint.AUTO
  if (use_mtls_endpoint is _MtlsEndpoint.ALWAYS) or (
      use_mtls_endpoint is _MtlsEndpoint.AUTO and client_cert_source is not None
  ):
    return API_REGISTRY_MTLS_URL
  return API_REGISTRY_URL


class ApiRegistry:
  """Registry that provides McpToolsets for MCP servers registered in API Registry."""

  def __init__(
      self,
      api_registry_project_id: str,
      location: str = "global",
      header_provider: (
          Callable[[ReadonlyContext], dict[str, str]] | None
      ) = None,
  ):
    """Initialize the API Registry.

    Args:
      api_registry_project_id: The project ID for the Google Cloud API Registry.
      location: The location of the API Registry resources.
      header_provider: Optional function to provide additional headers for MCP
        server calls.
    """
    self.api_registry_project_id = api_registry_project_id
    self.location = location
    self._credentials, _ = google.auth.default()
    self._mcp_servers: dict[str, dict[str, Any]] = {}
    self._header_provider = header_provider

    # Use an AuthorizedSession so credential refresh and, when a client
    # certificate is available, mutual TLS are handled by google-auth. The base
    # URL is selected dynamically so the .mtls. endpoint is used when mTLS is
    # active. This mirrors AgentRegistry.
    self._session = requests_auth.AuthorizedSession(
        credentials=self._credentials
    )
    client_cert_source = None
    if _use_client_cert_effective() and mtls.has_default_client_cert_source():
      client_cert_source = mtls.default_client_cert_source()
      self._session.configure_mtls_channel(client_cert_source)
    base_url = _get_api_registry_base_url(client_cert_source)

    url = f"{base_url}/v1beta/projects/{self.api_registry_project_id}/locations/{self.location}/mcpServers"

    try:
      # AuthorizedSession attaches the Authorization header on each request.
      quota_project_id = getattr(self._credentials, "quota_project_id", None)
      headers = (
          {"x-goog-user-project": quota_project_id} if quota_project_id else {}
      )
      page_token = None
      while True:
        params = {}
        if page_token:
          params["pageToken"] = page_token

        response = self._session.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        mcp_servers_list = data.get("mcpServers", [])
        for server in mcp_servers_list:
          server_name = server.get("name", "")
          if server_name:
            self._mcp_servers[server_name] = server

        page_token = data.get("nextPageToken")
        if not page_token:
          break
    except (requests.exceptions.RequestException, ValueError) as e:
      # Handle error in fetching or parsing tool definitions
      raise RuntimeError(
          f"Error fetching MCP servers from API Registry: {e}"
      ) from e

  def get_toolset(
      self,
      mcp_server_name: str,
      tool_filter: ToolPredicate | list[str] | None = None,
      tool_name_prefix: str | None = None,
  ) -> McpToolset:
    """Return the MCP Toolset based on the params.

    Args:
      mcp_server_name: Filter to select the MCP server name to get tools from.
      tool_filter: Optional filter to select specific tools. Can be a list of
        tool names or a ToolPredicate function.
      tool_name_prefix: Optional prefix to prepend to the names of the tools
        returned by the toolset.

    Returns:
      McpToolset: A toolset for the MCP server specified.
    """
    server = self._mcp_servers.get(mcp_server_name)
    if not server:
      raise ValueError(
          f"MCP server {mcp_server_name} not found in API Registry."
      )
    if not server.get("urls"):
      raise ValueError(f"MCP server {mcp_server_name} has no URLs.")

    mcp_server_url = server["urls"][0]

    # Only prepend "https://" if the URL doesn't already have a scheme
    if not mcp_server_url.startswith(("http://", "https://")):
      mcp_server_url = "https://" + mcp_server_url

    # Only attach the runtime's Application Default Credentials to Google API
    # hosts. The server URL comes from the API Registry listing and may point at
    # a non-Google host, which must not receive the runtime's Google
    # credentials. This mirrors AgentRegistry, which gates the same credentials
    # with _is_google_api. Non-Google servers can authenticate via
    # header_provider.
    headers = self._get_auth_headers() if _is_google_api(mcp_server_url) else {}

    return McpToolset(
        connection_params=StreamableHTTPConnectionParams(
            url=mcp_server_url,
            headers=headers,
        ),
        tool_filter=tool_filter,
        tool_name_prefix=tool_name_prefix,
        header_provider=self._header_provider,
    )

  def _get_auth_headers(self) -> dict[str, str]:
    """Refreshes credentials and returns authorization headers."""
    request = google.auth.transport.requests.Request()
    self._credentials.refresh(request)
    headers = {
        "Authorization": f"Bearer {self._credentials.token}",
    }
    # Add quota project header if available in ADC
    quota_project_id = getattr(self._credentials, "quota_project_id", None)
    if quota_project_id:
      headers["x-goog-user-project"] = quota_project_id
    return headers
