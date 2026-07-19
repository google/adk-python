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

import os
from typing import Any
from typing import Callable

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.base_toolset import ToolPredicate
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.utils import _mtls_utils
import google.auth
from google.auth.transport import mtls
from google.auth.transport import requests as requests_auth
import requests

API_REGISTRY_URL = "https://cloudapiregistry.googleapis.com"
API_REGISTRY_MTLS_URL = "https://cloudapiregistry.mtls.googleapis.com"


def _get_api_registry_url(client_cert_source: Any | None = None) -> str:
  """Returns the base URL based on mTLS configuration and cert availability."""
  use_mtls_endpoint_str = os.getenv(
      "GOOGLE_API_USE_MTLS_ENDPOINT", _mtls_utils.MtlsEndpoint.AUTO.value
  ).lower()
  try:
    use_mtls_endpoint = _mtls_utils.MtlsEndpoint(use_mtls_endpoint_str)
  except ValueError:
    use_mtls_endpoint = _mtls_utils.MtlsEndpoint.AUTO
  if (use_mtls_endpoint is _mtls_utils.MtlsEndpoint.ALWAYS) or (
      use_mtls_endpoint is _mtls_utils.MtlsEndpoint.AUTO
      and client_cert_source is not None
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

    use_client_cert = _mtls_utils.use_client_cert_effective()
    client_cert_source = None
    if use_client_cert:
      client_cert_source = (
          mtls.default_client_cert_source()
          if mtls.has_default_client_cert_source()
          else None
      )
    base_url = _get_api_registry_url(client_cert_source)

    url = f"{base_url}/v1beta/projects/{self.api_registry_project_id}/locations/{self.location}/mcpServers"

    try:
      quota_project_id = getattr(self._credentials, "quota_project_id", None)
      headers = {
          "Content-Type": "application/json",
      }
      if quota_project_id:
        headers["x-goog-user-project"] = quota_project_id

      page_token = None
      with requests_auth.AuthorizedSession(
          credentials=self._credentials
      ) as session:
        if use_client_cert:
          session.configure_mtls_channel(client_cert_source)
        while True:
          params = {
              # Include all the apis including disabled ones. API registry no longer supports enabling APIs.
              "filter": "enabled=false"
          }
          if page_token:
            params["pageToken"] = page_token

          response = session.get(url, headers=headers, params=params)
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
    headers = self._get_auth_headers()

    # Only prepend "https://" if the URL doesn't already have a scheme
    if not mcp_server_url.startswith(("http://", "https://")):
      mcp_server_url = "https://" + mcp_server_url

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
