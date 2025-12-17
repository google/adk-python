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

import sys
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import google.auth
import google.auth.transport.requests
import httpx

from .base_toolset import ToolPredicate
from .mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
from .mcp_tool.mcp_toolset import McpToolset

API_REGISTRY_URL = "https://cloudapiregistry.googleapis.com"


class ApiRegistry:
  """Registry that provides McpToolsets for MCP servers registered in API Registry."""

  def __init__(
      self,
      api_registry_project_id: str,
      location: str = "global",
      header_provider: Optional[
          Callable[[ReadonlyContext], Dict[str, str]]
      ] = None,
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
    self._mcp_servers: Dict[str, Dict[str, Any]] = {}
    self._header_provider = header_provider

    url = f"{API_REGISTRY_URL}/v1beta/projects/{self.api_registry_project_id}/locations/{self.location}/mcpServers"
    try:
      request = google.auth.transport.requests.Request()
      self._credentials.refresh(request)
      headers = {
          "Authorization": f"Bearer {self._credentials.token}",
          "Content-Type": "application/json",
      }
      with httpx.Client() as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        mcp_servers_list = response.json().get("mcpServers", [])
        for server in mcp_servers_list:
          server_name = server.get("name", "")
          if server_name:
            self._mcp_servers[server_name] = server
    except (httpx.HTTPError, ValueError) as e:
      # Handle error in fetching or parsing tool definitions
      raise RuntimeError(
          f"Error fetching MCP servers from API Registry: {e}"
      ) from e

  def get_toolset(
      self,
      mcp_server_name: str,
      tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
      tool_name_prefix: Optional[str] = None,
  ) -> McpToolset:
    """Return the MCP Toolset based on the params.

    Args:
      mcp_server_name: Filter to select the MCP server name to get tools
        from.
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
    request = google.auth.transport.requests.Request()
    self._credentials.refresh(request)
    headers = {
        "Authorization": f"Bearer {self._credentials.token}",
    }
    return McpToolset(
        connection_params=StreamableHTTPConnectionParams(
            url="https://" + mcp_server_url,
            headers=headers,
        ),
        tool_filter=tool_filter,
        tool_name_prefix=tool_name_prefix,
        header_provider=self._header_provider,
    )
