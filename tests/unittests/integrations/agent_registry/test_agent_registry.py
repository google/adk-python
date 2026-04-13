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


from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from a2a.types import TransportProtocol as A2ATransport
from fastapi.openapi.models import OAuth2
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.integrations.agent_registry import _ProtocolType
from google.adk.integrations.agent_registry import AgentRegistry
from google.adk.telemetry.tracing import GCP_MCP_SERVER_DESTINATION_ID
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
import httpx
from mcp import ClientSession
from mcp.types import ListToolsResult
from mcp.types import Tool
import pytest


class TestAgentRegistry:

  @pytest.fixture
  def registry(self):
    with patch("google.auth.default", return_value=(MagicMock(), "project-id")):
      return AgentRegistry(project_id="test-project", location="global")

  @pytest.mark.asyncio
  @patch("httpx.Client")
  @patch(
      "google.adk.tools.mcp_tool.mcp_session_manager.MCPSessionManager.create_session",
      new_callable=AsyncMock,
  )
  async def test_get_mcp_toolset_adds_destination_id(
      self, mock_create_session, mock_httpx, registry
  ):
    """Test that tools from get_mcp_toolset have the destination ID."""
    # Arrange
    mcp_server_name = "test-mcp-server"
    mock_api_response = MagicMock()
    mock_api_response.json.return_value = {
        "displayName": "TestPrefix",
        "mcpServerId": (
            "urn:mcp:googleapis.com:projects:1234:locations:global:bigquery"
        ),
        "interfaces": [{
            "url": "https://mcp.com",
            "protocolBinding": "JSONRPC",
        }],
    }
    mock_httpx.return_value.__enter__.return_value.get.return_value = (
        mock_api_response
    )

    registry._credentials.token = "token"
    registry._credentials.refresh = MagicMock()

    mock_session = AsyncMock(spec=ClientSession)
    mock_create_session.return_value = mock_session

    # Mock the tools returned by list_tools
    mock_session.list_tools.return_value = ListToolsResult(
        tools=[
            Tool(
                name="tool1",
                description="d1",
                inputs={},
                outputs={},
                inputSchema={},
            ),
            Tool(
                name="tool2",
                description="d2",
                inputs={},
                outputs={},
                inputSchema={},
            ),
        ]
    )

    # Act
    toolset = registry.get_mcp_toolset(mcp_server_name)
    tools = await toolset.get_tools()

    # Assert
    assert isinstance(toolset, McpToolset)
    mock_session.list_tools.assert_called_once_with()
    assert len(tools) == 2
    for tool in tools:
      assert tool.custom_metadata is not None
      assert (
          tool.custom_metadata.get(GCP_MCP_SERVER_DESTINATION_ID)
          == "urn:mcp:googleapis.com:projects:1234:locations:global:bigquery"
      )

  @pytest.mark.asyncio
  @patch("httpx.Client")
  @patch(
      "google.adk.tools.mcp_tool.mcp_session_manager.MCPSessionManager.create_session",
      new_callable=AsyncMock,
  )
  async def test_get_mcp_toolset_handles_missing_destination_id(
      self, mock_create_session, mock_httpx, registry
  ):
    """Test get_mcp_toolset when the destination ID is missing."""
    # Arrange
    mcp_server_name = "test-mcp-server"
    mock_api_response = MagicMock()
    mock_api_response.json.return_value = {
        "displayName": "TestPrefix",
        # "mcpServerId" is intentionally omitted
        "interfaces": [{
            "url": "https://mcp.com",
            "protocolBinding": "JSONRPC",
        }],
    }
    mock_httpx.return_value.__enter__.return_value.get.return_value = (
        mock_api_response
    )

    registry._credentials.token = "token"
    registry._credentials.refresh = MagicMock()

    mock_session = AsyncMock(spec=ClientSession)
    mock_create_session.return_value = mock_session

    # Mock the tools returned by list_tools
    mock_session.list_tools.return_value = ListToolsResult(
        tools=[
            Tool(
                name="tool1",
                description="d1",
                inputs={},
                outputs={},
                inputSchema={},
            ),
        ]
    )

    # Act
    toolset = registry.get_mcp_toolset(mcp_server_name)
    tools = await toolset.get_tools()

    # Assert
    assert isinstance(toolset, McpToolset)
    mock_session.list_tools.assert_called_once_with()
    assert len(tools) == 1
    for tool in tools:
      # The custom_metadata shouldn't have been added
      assert tool.custom_metadata is None

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
    uri, version, binding = registry._get_connection_uri(
        resource_details, protocol_binding=A2ATransport.jsonrpc
    )
    assert uri == "https://mcp-v1main.com"
    assert version is None
    assert binding == "JSONRPC"

  def test_get_connection_uri_agent_nested_protocols(self, registry):
    resource_details = {
        "protocols": [{
            "type": _ProtocolType.A2A_AGENT,
            "interfaces": [{
                "url": "https://my-agent.com",
                "protocolBinding": "JSONRPC",
            }],
        }]
    }
    uri, version, binding = registry._get_connection_uri(
        resource_details, protocol_type=_ProtocolType.A2A_AGENT
    )
    assert uri == "https://my-agent.com"
    assert version is None
    assert binding == A2ATransport.jsonrpc

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
                    "protocolBinding": "HTTP_JSON",
                }],
            },
        ]
    }
    # Filter by type
    uri, version, binding = registry._get_connection_uri(
        resource_details, protocol_type=_ProtocolType.A2A_AGENT
    )
    assert uri == "https://my-agent.com"
    assert version is None
    assert binding == A2ATransport.http_json

    # Filter by binding
    uri, version, binding = registry._get_connection_uri(
        resource_details, protocol_binding=A2ATransport.http_json
    )
    assert uri == "https://my-agent.com"
    assert version is None
    assert binding == A2ATransport.http_json

    # No match
    uri, version, binding = registry._get_connection_uri(
        resource_details,
        protocol_type=_ProtocolType.A2A_AGENT,
        protocol_binding=A2ATransport.jsonrpc,
    )
    assert uri is None
    assert version is None
    assert binding is None

  def test_get_connection_uri_returns_none_if_no_interfaces(self, registry):
    resource_details = {}
    uri, version, binding = registry._get_connection_uri(resource_details)
    assert uri is None
    assert version is None
    assert binding is None

  def test_get_connection_uri_returns_none_if_no_url_in_interfaces(
      self, registry
  ):
    resource_details = {"interfaces": [{"protocolBinding": "HTTP"}]}
    uri, version, binding = registry._get_connection_uri(resource_details)
    assert uri is None
    assert version is None
    assert binding is None

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
  def test_list_endpoints(self, mock_httpx, registry):
    mock_response = MagicMock()
    mock_response.json.return_value = {"endpoints": []}
    mock_response.raise_for_status = MagicMock()
    mock_httpx.return_value.__enter__.return_value.get.return_value = (
        mock_response
    )

    # Mock auth refresh
    registry._credentials.token = "token"
    registry._credentials.refresh = MagicMock()

    endpoints = registry.list_endpoints()
    assert endpoints == {"endpoints": []}

  @patch("httpx.Client")
  def test_get_endpoint(self, mock_httpx, registry):
    mock_response = MagicMock()
    mock_response.json.return_value = {"name": "test-endpoint"}
    mock_response.raise_for_status = MagicMock()
    mock_httpx.return_value.__enter__.return_value.get.return_value = (
        mock_response
    )

    registry._credentials.token = "token"
    registry._credentials.refresh = MagicMock()

    server = registry.get_endpoint("test-endpoint")
    assert server == {"name": "test-endpoint"}

  @pytest.mark.parametrize(
      "url, expected_auth",
      [
          ("https://mcp.com", False),
          ("https://mcp.googleapis.com/v1", True),
          ("https://example.com/googleapis/v1", False),
      ],
  )
  @patch("httpx.Client")
  def test_get_mcp_toolset_auth_headers(
      self, mock_httpx, registry, url, expected_auth
  ):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "displayName": "TestPrefix",
        "interfaces": [{
            "url": url,
            "protocolBinding": "JSONRPC",
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
    if expected_auth:
      assert toolset._connection_params.headers is not None
      assert (
          toolset._connection_params.headers["Authorization"] == "Bearer token"
      )
    else:
      assert toolset._connection_params.headers is None

  @patch("httpx.Client")
  def test_get_mcp_toolset_with_auth(self, mock_httpx, registry):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "displayName": "TestPrefix",
        "interfaces": [{
            "url": "https://mcp.com",
            "protocolBinding": "JSONRPC",
        }],
    }
    mock_response.raise_for_status = MagicMock()
    mock_httpx.return_value.__enter__.return_value.get.return_value = (
        mock_response
    )

    registry._credentials.token = "token"
    registry._credentials.refresh = MagicMock()

    auth_scheme = OAuth2(flows={})
    auth_credential = AuthCredential(
        auth_type="oauth2",
        oauth2=OAuth2Auth(client_id="test_id", client_secret="test_secret"),
    )

    toolset = registry.get_mcp_toolset(
        "test-mcp", auth_scheme=auth_scheme, auth_credential=auth_credential
    )
    assert isinstance(toolset, McpToolset)
    auth_config = toolset.get_auth_config()
    assert auth_config is not None
    assert auth_config.auth_scheme == auth_scheme
    assert auth_config.raw_auth_credential == auth_credential

  @patch("httpx.Client")
  def test_get_remote_a2a_agent(self, mock_httpx, registry):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "displayName": "TestAgent",
        "description": "Test Desc",
        "version": "1.0",
        "protocols": [{
            "type": _ProtocolType.A2A_AGENT,
            "protocolVersion": "0.4.0",
            "interfaces": [{
                "url": "https://my-agent.com",
                "protocolBinding": "HTTP_JSON",
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
    assert agent._agent_card.preferred_transport == A2ATransport.http_json
    assert agent._agent_card.protocol_version == "0.4.0"

  @patch("httpx.Client")
  def test_get_remote_a2a_agent_defaults(self, mock_httpx, registry):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "displayName": "TestAgent",
        "description": "Test Desc",
        "version": "1.0",
        "protocols": [{
            "type": _ProtocolType.A2A_AGENT,
            "interfaces": [{
                "url": "https://my-agent.com",
            }],
        }],
    }
    mock_response.raise_for_status = MagicMock()
    mock_httpx.return_value.__enter__.return_value.get.return_value = (
        mock_response
    )

    registry._credentials.token = "token"
    registry._credentials.refresh = MagicMock()

    agent = registry.get_remote_a2a_agent("test-agent")
    assert isinstance(agent, RemoteA2aAgent)
    assert agent._agent_card.preferred_transport == A2ATransport.http_json
    assert agent._agent_card.protocol_version == "0.3.0"

  @patch("httpx.Client")
  def test_get_remote_a2a_agent_with_card(self, mock_httpx, registry):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "name": "projects/p/locations/l/agents/a",
        "card": {
            "type": "A2A_AGENT_CARD",
            "content": {
                "name": "CardName",
                "description": "CardDesc",
                "version": "2.0",
                "url": "https://card-url.com",
                "skills": [{
                    "id": "s1",
                    "name": "S1",
                    "description": "D1",
                    "tags": ["t1"],
                }],
                "capabilities": {"streaming": True, "polling": False},
                "defaultInputModes": ["text"],
                "defaultOutputModes": ["text"],
            },
        },
    }
    mock_response.raise_for_status = MagicMock()
    mock_httpx.return_value.__enter__.return_value.get.return_value = (
        mock_response
    )

    registry._credentials.token = "token"
    registry._credentials.refresh = MagicMock()

    agent = registry.get_remote_a2a_agent("test-agent")
    assert isinstance(agent, RemoteA2aAgent)
    assert agent.name == "CardName"
    assert agent.description == "CardDesc"
    assert agent._agent_card.version == "2.0"
    assert agent._agent_card.url == "https://card-url.com"
    assert agent._agent_card.capabilities.streaming is True
    assert len(agent._agent_card.skills) == 1
    assert agent._agent_card.skills[0].name == "S1"

  @patch("httpx.Client")
  def test_get_remote_a2a_agent_with_httpx_client(self, mock_httpx, registry):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "displayName": "TestAgent",
        "description": "Test Desc",
        "version": "1.0",
        "protocols": [{
            "type": _ProtocolType.A2A_AGENT,
            "interfaces": [{
                "url": "https://my-agent.com",
            }],
        }],
    }
    mock_response.raise_for_status = MagicMock()
    mock_httpx.return_value.__enter__.return_value.get.return_value = (
        mock_response
    )

    custom_client = httpx.AsyncClient()
    agent = registry.get_remote_a2a_agent(
        "test-agent", httpx_client=custom_client
    )
    assert agent._httpx_client is custom_client

  @patch("httpx.Client")
  def test_get_remote_a2a_agent_configures_transports(
      self, mock_httpx, registry
  ):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "displayName": "TestAgent",
        "protocols": [{
            "type": _ProtocolType.A2A_AGENT,
            "interfaces": [{
                "url": "https://my-agent.com",
                "protocolBinding": A2ATransport.jsonrpc,
            }],
        }],
    }
    mock_response.raise_for_status = MagicMock()
    mock_httpx.return_value.__enter__.return_value.get.return_value = (
        mock_response
    )

    registry._credentials.token = "token"
    registry._credentials.refresh = MagicMock()

    agent = registry.get_remote_a2a_agent("test-agent")
    assert agent._agent_card.preferred_transport == A2ATransport.jsonrpc

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
        RuntimeError, match=r"API request failed \(network error\)"
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

  @patch.object(AgentRegistry, "get_endpoint")
  def test_get_model_name_starts_with_projects(
      self, mock_get_endpoint, registry
  ):
    mock_get_endpoint.return_value = {
        "interfaces": [{"url": "projects/p1/locations/l1/models/m1"}]
    }
    model_name = registry.get_model_name("test-endpoint")
    assert model_name == "projects/p1/locations/l1/models/m1"

  @patch.object(AgentRegistry, "get_endpoint")
  def test_get_model_name_contains_projects(self, mock_get_endpoint, registry):
    mock_get_endpoint.return_value = {
        "interfaces": [{
            "url": (
                "https://vertexai.googleapis.com/v1/projects/p1/locations/l1/models/m1"
            )
        }]
    }
    model_name = registry.get_model_name("test-endpoint")
    assert model_name == "projects/p1/locations/l1/models/m1"

  @patch.object(AgentRegistry, "get_endpoint")
  def test_get_model_name_strips_suffix(self, mock_get_endpoint, registry):
    mock_get_endpoint.return_value = {
        "interfaces": [{"url": "projects/p1/locations/l1/models/m1:predict"}]
    }
    model_name = registry.get_model_name("test-endpoint")
    assert model_name == "projects/p1/locations/l1/models/m1"

  @patch.object(AgentRegistry, "get_endpoint")
  def test_get_model_name_raises_value_error_if_no_uri(
      self, mock_get_endpoint, registry
  ):
    mock_get_endpoint.return_value = {}
    with pytest.raises(ValueError, match="Connection URI not found"):
      registry.get_model_name("test-endpoint")

  @patch.object(AgentRegistry, "_make_request")
  def test_get_mcp_toolset_with_binding(self, mock_make_request, registry):
    def side_effect(*args, **kwargs):
      if args[0] == "test-mcp":
        return {
            "displayName": "TestPrefix",
            "mcpServerId": "server-456",
            "interfaces": [{
                "url": "https://mcp.com",
                "protocolBinding": "JSONRPC",
            }],
        }
      if args[0] == "bindings":
        return {
            "bindings": [{
                "target": {
                    "identifier": (
                        "urn:mcp:projects-123:projects:123:locations:l:mcpServers:server-456"
                    )
                },
                "authProviderBinding": {
                    "authProvider": (
                        "projects/123/locations/l/authProviders/ap-789"
                    )
                },
            }]
        }
      return {}

    mock_make_request.side_effect = side_effect

    registry._credentials.token = "token"
    registry._credentials.refresh = MagicMock()

    toolset = registry.get_mcp_toolset(
        "test-mcp", continue_uri="https://override.com/continue"
    )
    assert isinstance(toolset, McpToolset)
    assert toolset._auth_scheme is not None
    assert (
        toolset._auth_scheme.name
        == "projects/123/locations/l/authProviders/ap-789"
    )
    assert toolset._auth_scheme.continue_uri == "https://override.com/continue"
