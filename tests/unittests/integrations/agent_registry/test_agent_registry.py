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

from unittest.mock import MagicMock
from unittest.mock import patch

from a2a.types import TransportProtocol as A2ATransport
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.integrations.agent_registry import _ProtocolType
from google.adk.integrations.agent_registry import AgentRegistry
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
import httpx
import pytest


class TestAgentRegistry:

  @pytest.fixture
  def registry(self):
    with patch("google.auth.default", return_value=(MagicMock(), "project-id")):
      return AgentRegistry(project_id="test-project", location="global")

  def test_init_raises_value_error_if_params_missing(self):
    with pytest.raises(
        ValueError, match="project_id and location must be provided"
    ):
      AgentRegistry(project_id=None, location=None)

  def test_get_connection_uri_mcp_interfaces_top_level(self, registry):
    resource_details = {
        "interfaces": [
            {"url": "https://mcp-v1main.com", "protocolBinding": "JSONRPC"}
        ]
    }
    uri = registry._get_connection_uri(
        resource_details, protocol_binding=A2ATransport.jsonrpc
    )
    assert uri == "https://mcp-v1main.com"

  def test_get_connection_uri_agent_nested_protocols(self, registry):
    resource_details = {
        "protocols": [{
            "type": _ProtocolType.A2A_AGENT,
            "interfaces": [{
                "url": "https://my-agent.com",
                "protocolBinding": A2ATransport.jsonrpc,
            }],
        }]
    }
    uri = registry._get_connection_uri(
        resource_details, protocol_type=_ProtocolType.A2A_AGENT
    )
    assert uri == "https://my-agent.com"

  def test_get_connection_uri_filtering(self, registry):
    resource_details = {
        "protocols": [
            {
                "type": "CUSTOM",
                "interfaces": [{"url": "https://custom.com"}],
            },
            {
                "type": _ProtocolType.A2A_AGENT,
                "interfaces": [{
                    "url": "https://my-agent.com",
                    "protocolBinding": A2ATransport.http_json,
                }],
            },
        ]
    }
    # Filter by type
    uri = registry._get_connection_uri(
        resource_details, protocol_type=_ProtocolType.A2A_AGENT
    )
    assert uri == "https://my-agent.com"

    # Filter by binding
    uri = registry._get_connection_uri(
        resource_details, protocol_binding=A2ATransport.http_json
    )
    assert uri == "https://my-agent.com"

    # No match
    uri = registry._get_connection_uri(
        resource_details,
        protocol_type=_ProtocolType.A2A_AGENT,
        protocol_binding=A2ATransport.jsonrpc,
    )
    assert uri is None

  def test_get_connection_uri_returns_none_if_no_interfaces(self, registry):
    resource_details = {}
    uri = registry._get_connection_uri(resource_details)
    assert uri is None

  def test_get_connection_uri_returns_none_if_no_url_in_interfaces(
      self, registry
  ):
    resource_details = {"interfaces": [{"protocolBinding": "HTTP"}]}
    uri = registry._get_connection_uri(resource_details)
    assert uri is None

  @patch("httpx.Client")
  def test_list_agents(self, mock_httpx, registry):
    mock_response = MagicMock()
    mock_response.json.return_value = {"agents": []}
    mock_response.raise_for_status = MagicMock()
    mock_httpx.return_value.__enter__.return_value.get.return_value = (
        mock_response
    )

    # Mock auth refresh
    registry._credentials.token = "token"
    registry._credentials.refresh = MagicMock()

    agents = registry.list_agents()
    assert agents == {"agents": []}

  @patch("httpx.Client")
  def test_get_mcp_server(self, mock_httpx, registry):
    mock_response = MagicMock()
    mock_response.json.return_value = {"name": "test-mcp"}
    mock_response.raise_for_status = MagicMock()
    mock_httpx.return_value.__enter__.return_value.get.return_value = (
        mock_response
    )

    registry._credentials.token = "token"
    registry._credentials.refresh = MagicMock()

    server = registry.get_mcp_server("test-mcp")
    assert server == {"name": "test-mcp"}

  @patch("httpx.Client")
  def test_get_mcp_toolset(self, mock_httpx, registry):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "displayName": "TestPrefix",
        "interfaces": [{
            "url": "https://mcp.com",
            "protocolBinding": A2ATransport.jsonrpc,
        }],
    }
    mock_response.raise_for_status = MagicMock()
    mock_httpx.return_value.__enter__.return_value.get.return_value = (
        mock_response
    )

    registry._credentials.token = "token"
    registry._credentials.refresh = MagicMock()

    toolset = registry.get_mcp_toolset("test-mcp")
    assert isinstance(toolset, McpToolset)
    assert toolset.tool_name_prefix == "TestPrefix"

  @patch("httpx.Client")
  def test_get_remote_a2a_agent(self, mock_httpx, registry):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "displayName": "TestAgent",
        "description": "Test Desc",
        "version": "1.0",
        "protocols": [{
            "type": _ProtocolType.A2A_AGENT,
            "interfaces": [{
                "url": "https://my-agent.com",
                "protocolBinding": A2ATransport.jsonrpc,
            }],
        }],
        "skills": [{"id": "s1", "name": "Skill 1", "description": "Desc 1"}],
    }
    mock_response.raise_for_status = MagicMock()
    mock_httpx.return_value.__enter__.return_value.get.return_value = (
        mock_response
    )

    registry._credentials.token = "token"
    registry._credentials.refresh = MagicMock()

    agent = registry.get_remote_a2a_agent("test-agent")
    assert isinstance(agent, RemoteA2aAgent)
    assert agent.name == "TestAgent"
    assert agent.description == "Test Desc"
    assert agent._agent_card.url == "https://my-agent.com"
    assert agent._agent_card.version == "1.0"
    assert len(agent._agent_card.skills) == 1
    assert agent._agent_card.skills[0].name == "Skill 1"

  def test_get_auth_headers(self, registry):
    registry._credentials.token = "fake-token"
    registry._credentials.refresh = MagicMock()
    registry._credentials.quota_project_id = "quota-project"

    headers = registry._get_auth_headers()
    assert headers["Authorization"] == "Bearer fake-token"
    assert headers["x-goog-user-project"] == "quota-project"

  @patch("httpx.Client")
  def test_make_request_raises_http_status_error(self, mock_httpx, registry):
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = "Not Found"
    error = httpx.HTTPStatusError(
        "Error", request=MagicMock(), response=mock_response
    )
    mock_httpx.return_value.__enter__.return_value.get.side_effect = error

    registry._credentials.token = "token"
    registry._credentials.refresh = MagicMock()

    with pytest.raises(
        RuntimeError, match="API request failed with status 404"
    ):
      registry._make_request("test-path")

  @patch("httpx.Client")
  def test_make_request_raises_request_error(self, mock_httpx, registry):
    error = httpx.RequestError("Connection failed", request=MagicMock())
    mock_httpx.return_value.__enter__.return_value.get.side_effect = error

    registry._credentials.token = "token"
    registry._credentials.refresh = MagicMock()

    with pytest.raises(
        RuntimeError, match="API request failed \(network error\)"
    ):
      registry._make_request("test-path")

  @patch("httpx.Client")
  def test_make_request_raises_generic_exception(self, mock_httpx, registry):
    mock_httpx.return_value.__enter__.return_value.get.side_effect = Exception(
        "Generic error"
    )

    registry._credentials.token = "token"
    registry._credentials.refresh = MagicMock()

    with pytest.raises(RuntimeError, match="API request failed: Generic error"):
      registry._make_request("test-path")
