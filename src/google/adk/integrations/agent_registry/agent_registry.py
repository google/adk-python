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
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union
from urllib.parse import parse_qs
from urllib.parse import urlparse

from a2a.client.client_factory import minimal_agent_card
from a2a.types import AgentCapabilities
from a2a.types import AgentCard
from a2a.types import AgentSkill
from a2a.types import TransportProtocol as A2ATransport
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
import google.auth
import google.auth.transport.requests
import httpx

logger = logging.getLogger("google_adk." + __name__)

AGENT_REGISTRY_BASE_URL = "https://agentregistry.googleapis.com/v1alpha"


class _ProtocolType(str, Enum):
  """Supported agent protocol types."""

  TYPE_UNSPECIFIED = "TYPE_UNSPECIFIED"
  A2A_AGENT = "A2A_AGENT"
  CUSTOM = "CUSTOM"


class AgentRegistry:
  """Client for interacting with the Google Cloud Agent Registry service.

  Unlike a standard REST client library, this class provides higher-level
  abstractions for ADK integration. It surfaces the agent registry service
  methods along with helper methods like `get_mcp_toolset` and
  `get_remote_a2a_agent` that automatically resolve connection details and
  handle authentication to produce ready-to-use ADK components.
  """

  def __init__(
      self,
      project_id: Optional[str] = None,
      location: Optional[str] = None,
      header_provider: Optional[
          Callable[[ReadonlyContext], Dict[str, str]]
      ] = None,
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
    try:
      self._credentials, _ = google.auth.default()
    except google.auth.exceptions.DefaultCredentialsError as e:
      raise RuntimeError(
          f"Failed to get default Google Cloud credentials: {e}"
      ) from e

  def _get_auth_headers(self) -> Dict[str, str]:
    """Refreshes credentials and returns authorization headers."""
    try:
      request = google.auth.transport.requests.Request()
      self._credentials.refresh(request)
      headers = {
          "Authorization": f"Bearer {self._credentials.token}",
          "Content-Type": "application/json",
      }
      quota_project_id = getattr(self._credentials, "quota_project_id", None)
      if quota_project_id:
        headers["x-goog-user-project"] = quota_project_id
      return headers
    except google.auth.exceptions.RefreshError as e:
      raise RuntimeError(
          f"Failed to refresh Google Cloud credentials: {e}"
      ) from e

  def _make_request(
      self, path: str, params: Optional[Dict[str, Any]] = None
  ) -> Dict[str, Any]:
    """Helper function to make GET requests to the Agent Registry API."""
    if path.startswith("projects/"):
      url = f"{AGENT_REGISTRY_BASE_URL}/{path}"
    else:
      url = f"{AGENT_REGISTRY_BASE_URL}/{self._base_path}/{path}"

    try:
      headers = self._get_auth_headers()
      with httpx.Client() as client:
        response = client.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
      raise RuntimeError(
          f"API request failed with status {e.response.status_code}:"
          f" {e.response.text}"
      ) from e
    except httpx.RequestError as e:
      raise RuntimeError(f"API request failed (network error): {e}") from e
    except Exception as e:
      raise RuntimeError(f"API request failed: {e}") from e

  def _get_connection_uri(
      self,
      resource_details: Dict[str, Any],
      protocol_type: Optional[_ProtocolType] = None,
      protocol_binding: Optional[A2ATransport] = None,
  ) -> Optional[str]:
    """Extracts the first matching URI based on type and binding filters."""
    protocols = list(resource_details.get("protocols", []))
    if "interfaces" in resource_details:
      protocols.append({"interfaces": resource_details["interfaces"]})

    for p in protocols:
      if protocol_type and p.get("type") != protocol_type:
        continue
      for i in p.get("interfaces", []):
        if protocol_binding and i.get("protocolBinding") != protocol_binding:
          continue
        if url := i.get("url"):
          return url

    return None

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
      filter_str: Optional[str] = None,
      page_size: Optional[int] = None,
      page_token: Optional[str] = None,
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

  def get_mcp_server(self, name: str) -> Dict[str, Any]:
    """Retrieves details of a specific MCP Server."""
    return self._make_request(name)

  def get_mcp_toolset(self, mcp_server_name: str) -> McpToolset:
    """Constructs an McpToolset instance from a registered MCP Server."""
    server_details = self.get_mcp_server(mcp_server_name)
    name = self._clean_name(server_details.get("displayName", mcp_server_name))

    endpoint_uri = self._get_connection_uri(
        server_details, protocol_binding=A2ATransport.jsonrpc
    ) or self._get_connection_uri(
        server_details, protocol_binding=A2ATransport.http_json
    )
    if not endpoint_uri:
      raise ValueError(
          f"MCP Server endpoint URI not found for: {mcp_server_name}"
      )

    connection_params = StreamableHTTPConnectionParams(
        url=endpoint_uri, headers=self._get_auth_headers()
    )
    return McpToolset(
        connection_params=connection_params,
        tool_name_prefix=name,
        header_provider=self._header_provider,
    )

  # --- Agent Methods ---

  def list_agents(
      self,
      filter_str: Optional[str] = None,
      page_size: Optional[int] = None,
      page_token: Optional[str] = None,
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

  def get_agent_info(self, name: str) -> Dict[str, Any]:
    """Retrieves detailed metadata of a specific A2A Agent."""
    return self._make_request(name)

  def get_remote_a2a_agent(self, agent_name: str) -> RemoteA2aAgent:
    """Creates a RemoteA2aAgent instance for a registered A2A Agent."""
    agent_info = self.get_agent_info(agent_name)
    name = self._clean_name(agent_info.get("displayName", agent_name))
    description = agent_info.get("description", "")
    version = agent_info.get("version", "")

    url = self._get_connection_uri(
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

    agent_card = AgentCard(
        name=name,
        description=description,
        version=version,
        url=url,
        skills=skills,
        capabilities=AgentCapabilities(streaming=False, polling=False),
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
    )

    return RemoteA2aAgent(
        name=name,
        agent_card=agent_card,
        description=description,
    )
