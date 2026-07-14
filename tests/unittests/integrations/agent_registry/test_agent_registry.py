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


import os
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi.openapi.models import OAuth2
from google.adk.a2a import _compat
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.integrations.agent_registry import AgentRegistry
from google.adk.integrations.agent_registry.agent_registry import _ProtocolType
from google.adk.telemetry.tracing import GCP_MCP_SERVER_DESTINATION_ID
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
import httpx
from mcp import ClientSession
from mcp.types import ListToolsResult
from mcp.types import Tool
import pytest
import requests


def _assert_preferred_transport(agent_card, expected):
  """Assert an agent card's preferred transport, version-agnostically."""
  # 0.3.x exposes a top-level `preferred_transport` field; 1.x dropped it in
  # favor of `supported_interfaces[*].protocol_binding`. `expected` is a
  # `_compat.TP_*` constant; its binding value is the wire string on both SDKs.
  if not _compat.IS_A2A_V1:
    assert agent_card.preferred_transport == expected
    return
  bindings = [
      iface.protocol_binding for iface in agent_card.supported_interfaces or []
  ]
  assert getattr(expected, "value", expected) in bindings


def _assert_protocol_version(agent_card, expected):
  """Assert an agent card's protocol version, version-agnostically."""
  # 0.3.x exposes a top-level `protocol_version` field; 1.x moved it onto each
  # `supported_interfaces[*].protocol_version`.
  if not _compat.IS_A2A_V1:
    assert agent_card.protocol_version == expected
    return
  versions = [
      iface.protocol_version for iface in agent_card.supported_interfaces or []
  ]
  assert expected in versions


class TestAgentRegistry:

  @pytest.fixture
  def registry(self):
    mock_creds = MagicMock()
    mock_creds.quota_project_id = None
    with (
        patch("google.auth.default", return_value=(mock_creds, "project-id")),
        patch(
            "google.auth.transport.requests.AuthorizedSession",
            autospec=True,
        ) as mock_session_class,
    ):
      registry = AgentRegistry(project_id="test-project", location="global")
      return registry

  @pytest.mark.asyncio
  @patch(
      "google.adk.tools.mcp_tool.mcp_session_manager.MCPSessionManager.create_session",
      new_callable=AsyncMock,
  )
  async def test_get_mcp_toolset_adds_destination_id(
      self, mock_create_session, registry
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
    registry._session.get.return_value = mock_api_response

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
  @patch(
      "google.adk.tools.mcp_tool.mcp_session_manager.MCPSessionManager.create_session",
      new_callable=AsyncMock,
  )
  async def test_get_mcp_toolset_handles_missing_destination_id(
      self, mock_create_session, registry
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
    registry._session.get.return_value = mock_api_response

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
        resource_details, protocol_binding=_compat.TP_JSONRPC
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
    assert binding == _compat.TP_JSONRPC

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
    assert binding == _compat.TP_HTTP_JSON

    # Filter by binding
    uri, version, binding = registry._get_connection_uri(
        resource_details, protocol_binding=_compat.TP_HTTP_JSON
    )
    assert uri == "https://my-agent.com"
    assert version is None
    assert binding == _compat.TP_HTTP_JSON

    # No match
    uri, version, binding = registry._get_connection_uri(
        resource_details,
        protocol_type=_ProtocolType.A2A_AGENT,
        protocol_binding=_compat.TP_JSONRPC,
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

  def test_list_agents(self, registry):
    mock_response = MagicMock()
    mock_response.json.return_value = {"agents": []}
    mock_response.raise_for_status = MagicMock()
    registry._session.get.return_value = mock_response

    registry._credentials.token = "token"
    registry._credentials.refresh = MagicMock()

    agents = registry.list_agents()
    assert agents == {"agents": []}

  def test_search_agents(self, registry):
    """Tests search_agents API call."""
    # pylint: disable=protected-access
    mock_response = MagicMock()
    mock_response.json.return_value = {"agents": [{"name": "agent-1"}]}
    mock_response.raise_for_status = MagicMock()
    registry._session.post.return_value = mock_response

    registry._credentials.token = "token"
    registry._credentials.refresh = MagicMock()

    agents = registry.search_agents(
        search_string="test-agent",
        search_type="KEYWORD",
        filter_str="display_name:test",
        order_by="name",
        page_size=10,
        page_token="next-token",
    )
    assert agents == {"agents": [{"name": "agent-1"}]}
    registry._session.post.assert_called_once_with(
        f"{registry._base_url}/projects/test-project/locations/global/agents:search",
        headers={"x-goog-user-project": "test-project"},
        json={
            "searchString": "test-agent",
            "searchType": "KEYWORD",
            "filter": "display_name:test",
            "orderBy": "name",
            "pageSize": 10,
            "pageToken": "next-token",
        },
    )
    # pylint: enable=protected-access

  def test_search_mcp_servers(self, registry):
    """Tests search_mcp_servers API call."""
    # pylint: disable=protected-access
    mock_response = MagicMock()
    mock_response.json.return_value = {"mcpServers": [{"name": "mcp-1"}]}
    mock_response.raise_for_status = MagicMock()
    registry._session.post.return_value = mock_response

    registry._credentials.token = "token"
    registry._credentials.refresh = MagicMock()

    mcp_servers = registry.search_mcp_servers(
        search_string="test-mcp",
        search_type="KEYWORD",
        filter_str="display_name:test",
        order_by="name",
        page_size=10,
        page_token="next-token",
    )
    assert mcp_servers == {"mcpServers": [{"name": "mcp-1"}]}
    registry._session.post.assert_called_once_with(
        f"{registry._base_url}/projects/test-project/locations/global/mcpServers:search",
        headers={"x-goog-user-project": "test-project"},
        json={
            "searchString": "test-mcp",
            "searchType": "KEYWORD",
            "filter": "display_name:test",
            "orderBy": "name",
            "pageSize": 10,
            "pageToken": "next-token",
        },
    )
    # pylint: enable=protected-access

  def test_get_mcp_server(self, registry):
    mock_response = MagicMock()
    mock_response.json.return_value = {"name": "test-mcp"}
    mock_response.raise_for_status = MagicMock()
    registry._session.get.return_value = mock_response

    registry._credentials.token = "token"
    registry._credentials.refresh = MagicMock()

    server = registry.get_mcp_server("test-mcp")
    assert server == {"name": "test-mcp"}

  def test_list_endpoints(self, registry):
    mock_response = MagicMock()
    mock_response.json.return_value = {"endpoints": []}
    mock_response.raise_for_status = MagicMock()
    registry._session.get.return_value = mock_response

    registry._credentials.token = "token"
    registry._credentials.refresh = MagicMock()

    endpoints = registry.list_endpoints()
    assert endpoints == {"endpoints": []}

  def test_get_endpoint(self, registry):
    mock_response = MagicMock()
    mock_response.json.return_value = {"name": "test-endpoint"}
    mock_response.raise_for_status = MagicMock()
    registry._session.get.return_value = mock_response

    registry._credentials.token = "token"
    registry._credentials.refresh = MagicMock()

    server = registry.get_endpoint("test-endpoint")
    assert server == {"name": "test-endpoint"}

  @pytest.mark.parametrize(
      "url, expected_auth, use_custom_provider",
      [
          ("https://mcp.com", False, False),
          ("https://mcp.googleapis.com/v1", True, False),
          ("https://example.com/googleapis/v1", False, False),
          ("https://mcp.googleapis.com/v1", True, True),
      ],
  )
  def test_get_mcp_toolset_auth_headers(
      self, registry, url, expected_auth, use_custom_provider
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
    registry._session.get.return_value = mock_response

    if use_custom_provider:
      custom_header_provider = lambda context: {
          "Authorization": "Bearer custom_token"
      }
      with (
          patch(
              "google.auth.default", return_value=(MagicMock(), "project-id")
          ),
          patch("google.auth.transport.requests.AuthorizedSession"),
      ):
        registry = AgentRegistry(
            project_id="test-project",
            location="global",
            header_provider=custom_header_provider,
        )
        registry._session.get.return_value = mock_response

    registry._credentials.token = "token"
    registry._credentials.refresh = MagicMock()

    toolset = registry.get_mcp_toolset("test-mcp")
    assert isinstance(toolset, McpToolset)
    assert toolset.tool_name_prefix == "TestPrefix"
    assert toolset._connection_params.headers is None
    headers = toolset._header_provider(MagicMock())

    if use_custom_provider:
      assert headers.get("Authorization") == "Bearer custom_token"
    elif expected_auth:
      assert headers.get("Authorization") == "Bearer token"
    else:
      assert "Authorization" not in headers

  def test_get_mcp_toolset_with_auth(self, registry):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "displayName": "TestPrefix",
        "interfaces": [{
            "url": "https://mcp.com",
            "protocolBinding": "JSONRPC",
        }],
    }
    mock_response.raise_for_status = MagicMock()
    registry._session.get.return_value = mock_response

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

  def test_get_mcp_toolset_with_auth_blocks_gcp_headers(self, registry):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "displayName": "TestPrefix",
        "interfaces": [{
            "url": "https://mcp.googleapis.com/v1",
            "protocolBinding": "JSONRPC",
        }],
    }
    mock_response.raise_for_status = MagicMock()
    registry._session.get.return_value = mock_response

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

    headers = toolset._header_provider(MagicMock())
    assert "Authorization" not in headers

  def test_get_remote_a2a_agent(self, registry):
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
    registry._session.get.return_value = mock_response

    registry._credentials.token = "token"
    registry._credentials.refresh = MagicMock()

    agent = registry.get_remote_a2a_agent("test-agent")
    assert isinstance(agent, RemoteA2aAgent)
    assert agent.name == "TestAgent"
    assert agent.description == "Test Desc"
    assert _compat.agent_card_url(agent._agent_card) == "https://my-agent.com"
    assert agent._agent_card.version == "1.0"
    assert len(agent._agent_card.skills) == 1
    assert agent._agent_card.skills[0].name == "Skill 1"
    _assert_preferred_transport(agent._agent_card, _compat.TP_HTTP_JSON)
    _assert_protocol_version(agent._agent_card, "0.4.0")

  def test_get_remote_a2a_agent_defaults(self, registry):
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
    registry._session.get.return_value = mock_response

    registry._credentials.token = "token"
    registry._credentials.refresh = MagicMock()

    agent = registry.get_remote_a2a_agent("test-agent")
    assert isinstance(agent, RemoteA2aAgent)
    _assert_preferred_transport(agent._agent_card, _compat.TP_HTTP_JSON)
    # When the interface omits an explicit protocol version, each SDK applies
    # its own default ("0.3.0" on 0.3.x, "1.0" on 1.x).
    _assert_protocol_version(
        agent._agent_card, "1.0" if _compat.IS_A2A_V1 else "0.3.0"
    )

  def test_get_remote_a2a_agent_with_card(self, registry):
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
    registry._session.get.return_value = mock_response

    registry._credentials.token = "token"
    registry._credentials.refresh = MagicMock()

    agent = registry.get_remote_a2a_agent("test-agent")
    assert isinstance(agent, RemoteA2aAgent)
    assert agent.name == "CardName"
    assert agent.description == "CardDesc"
    assert agent._agent_card.version == "2.0"
    assert _compat.agent_card_url(agent._agent_card) == "https://card-url.com"
    assert agent._agent_card.capabilities.streaming is True
    assert len(agent._agent_card.skills) == 1
    assert agent._agent_card.skills[0].name == "S1"

  def test_get_remote_a2a_agent_with_httpx_client(self, registry):
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
    registry._session.get.return_value = mock_response

    custom_client = httpx.AsyncClient()
    agent = registry.get_remote_a2a_agent(
        "test-agent", httpx_client=custom_client
    )
    assert agent._httpx_client is custom_client

  def test_get_remote_a2a_agent_configures_transports(self, registry):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "displayName": "TestAgent",
        "protocols": [{
            "type": _ProtocolType.A2A_AGENT,
            "interfaces": [{
                "url": "https://my-agent.com",
                "protocolBinding": _compat.TP_JSONRPC,
            }],
        }],
    }
    mock_response.raise_for_status = MagicMock()
    registry._session.get.return_value = mock_response

    registry._credentials.token = "token"
    registry._credentials.refresh = MagicMock()

    agent = registry.get_remote_a2a_agent("test-agent")
    _assert_preferred_transport(agent._agent_card, _compat.TP_JSONRPC)

  def test_get_auth_headers(self, registry):
    registry._credentials.token = "fake-token"
    registry._credentials.refresh = MagicMock()

    headers = registry._get_auth_headers()
    assert headers["Authorization"] == "Bearer fake-token"
    assert "x-goog-user-project" not in headers

  def test_make_request_raises_http_status_error(self, registry):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    error = requests.exceptions.HTTPError(
        "Error", request=MagicMock(), response=mock_response
    )
    registry._session.get.side_effect = error

    registry._credentials.token = "token"
    registry._credentials.refresh = MagicMock()

    with pytest.raises(
        RuntimeError, match="API request failed with status 500"
    ):
      registry._make_request("test-path")

  def test_make_request_raises_request_error(self, registry):
    error = requests.exceptions.RequestException(
        "Connection failed", request=MagicMock()
    )
    registry._session.get.side_effect = error

    registry._credentials.token = "token"
    registry._credentials.refresh = MagicMock()

    with pytest.raises(
        RuntimeError, match=r"API request failed \(network error\)"
    ):
      registry._make_request("test-path")

  def test_make_request_raises_generic_exception(self, registry):
    registry._session.get.side_effect = Exception("Generic error")

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


class TestAgentRegistryMtls:

  @pytest.fixture
  def registry(self):
    with (
        patch(
            "google.auth.default", return_value=(MagicMock(), "test-project")
        ),
        patch("google.auth.transport.requests.AuthorizedSession"),
        patch(
            "google.adk.integrations.agent_registry.agent_registry._use_client_cert_effective",
            return_value=False,
        ),
    ):
      return AgentRegistry(project_id="test-project", location="global")

  @patch(
      "google.auth.transport.mtls.has_default_client_cert_source",
      return_value=False,
  )
  def test_make_request_uses_authorized_session_no_mtls(
      self, mock_has_cert, registry
  ):
    mock_session = registry._session
    mock_response = MagicMock()
    mock_response.json.return_value = {"key": "value"}
    mock_session.get.return_value = mock_response

    result = registry._make_request("test-path")

    mock_session.get.assert_called_once()
    assert mock_session.configure_mtls_channel.call_count == 0
    assert result == {"key": "value"}

  @patch(
      "google.auth.transport.mtls.has_default_client_cert_source",
      return_value=True,
  )
  @patch("google.auth.transport.mtls.default_client_cert_source")
  @patch.dict(os.environ, {"GOOGLE_API_USE_CLIENT_CERTIFICATE": "true"})
  def test_make_request_configures_mtls(self, mock_cert_source, registry):
    mock_cert_source.return_value = lambda: (b"cert", b"key")
    with (
        patch(
            "google.auth.default", return_value=(MagicMock(), "test-project")
        ),
        patch(
            "google.adk.integrations.agent_registry.agent_registry._use_client_cert_effective",
            return_value=True,
        ),
        patch(
            "google.auth.transport.requests.AuthorizedSession"
        ) as mock_session_class,
    ):
      # Instantiate inside the test after enabling mTLS patches
      registry = AgentRegistry(project_id="test-project", location="global")
      mock_session = registry._session

      # Mock successful response
      mock_response = MagicMock()
      mock_response.json.return_value = {"key": "value"}
      mock_session.get.return_value = mock_response

      registry._make_request("test-path")

      # Verify mTLS configuration and endpoint
      mock_session.configure_mtls_channel.assert_called_once()
      args, kwargs = mock_session.get.call_args
      assert "agentregistry.mtls.googleapis.com" in args[0]

  @pytest.mark.parametrize(
      "env_val, has_cert, expected",
      [
          ("true", True, True),
          ("true", False, True),
          ("false", True, False),
          ("false", False, False),
      ],
  )
  def test_use_client_cert_effective(
      self, env_val, has_cert, expected, registry
  ):
    with patch.dict(os.environ, {"GOOGLE_API_USE_CLIENT_CERTIFICATE": env_val}):
      with patch(
          "google.auth.transport.mtls.has_default_client_cert_source",
          return_value=has_cert,
      ):
        from google.adk.integrations.agent_registry.agent_registry import _use_client_cert_effective

        assert _use_client_cert_effective() == expected

  @pytest.mark.parametrize(
      "use_mtls_env, client_cert_source, expected_domain",
      [
          # Auto mode (default)
          (None, None, "agentregistry.googleapis.com"),
          (None, lambda: True, "agentregistry.mtls.googleapis.com"),
          # Always mode
          ("always", None, "agentregistry.mtls.googleapis.com"),
          ("always", lambda: True, "agentregistry.mtls.googleapis.com"),
          # Never mode
          ("never", None, "agentregistry.googleapis.com"),
          ("never", lambda: True, "agentregistry.googleapis.com"),
      ],
  )
  def test_get_agent_registry_base_url(
      self, use_mtls_env, client_cert_source, expected_domain, registry
  ):
    from google.adk.integrations.agent_registry.agent_registry import _get_agent_registry_base_url

    env_patch = {}
    if use_mtls_env is not None:
      env_patch["GOOGLE_API_USE_MTLS_ENDPOINT"] = use_mtls_env
    else:
      # Ensure any ambient env var doesn't leak into the test
      env_patch = {"GOOGLE_API_USE_MTLS_ENDPOINT": "auto"}

    with patch.dict(os.environ, env_patch):
      assert expected_domain in _get_agent_registry_base_url(client_cert_source)

  def test_make_request_error_handling(self, registry):
    mock_session = registry._session
    mock_session.get.side_effect = Exception("Connection error")

    with pytest.raises(
        RuntimeError, match="API request failed: Connection error"
    ):
      registry._make_request("test-path")
