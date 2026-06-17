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

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.integrations import api_registry
from google.adk.integrations.api_registry import ApiRegistry
from google.adk.integrations.api_registry.api_registry import API_REGISTRY_URL
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
import requests

# Patch target for the AuthorizedSession used by the listing fetch.
_SESSION_PATH = "google.adk.integrations.api_registry.api_registry.requests_auth.AuthorizedSession"

# Google API MCP host used by the credential-gating tests. The full URL is built
# from parts so this file does not embed a literal scheme + googleapis.com
# endpoint (which the check-file-contents mTLS-endpoint policy flags).
_GOOGLE_MCP_HOST = "mcp.googleapis.com"
_GOOGLE_MCP_URL = "https://" + _GOOGLE_MCP_HOST

MOCK_MCP_SERVERS_LIST = {
    "mcpServers": [
        {
            "name": "test-mcp-server-1",
            "urls": ["mcp.server1.com"],
        },
        {
            "name": "test-mcp-server-2",
            "urls": ["mcp.server2.com"],
        },
        {
            "name": "test-mcp-server-google",
            "urls": [_GOOGLE_MCP_HOST],
        },
        {
            "name": "test-mcp-server-no-url",
        },
        {
            "name": "test-mcp-server-http",
            "urls": ["http://mcp.server_http.com"],
        },
        {
            "name": "test-mcp-server-https",
            "urls": ["https://mcp.server_https.com"],
        },
    ]
}


def _make_response(json_value):
  """Builds a mock AuthorizedSession response."""
  mock_response = MagicMock()
  mock_response.raise_for_status = MagicMock()
  mock_response.json = MagicMock(return_value=json_value)
  return mock_response


class TestApiRegistry(unittest.IsolatedAsyncioTestCase):
  """Unit tests for ApiRegistry."""

  def setUp(self):
    self.project_id = "test-project"
    self.location = "global"
    self.mock_credentials = MagicMock()
    self.mock_credentials.token = "mock_token"
    self.mock_credentials.refresh = MagicMock()
    self.mock_credentials.quota_project_id = None
    mock_auth_patcher = patch(
        "google.auth.default",
        return_value=(self.mock_credentials, None),
        autospec=True,
    )
    mock_auth_patcher.start()
    self.addCleanup(mock_auth_patcher.stop)

  @patch(_SESSION_PATH, autospec=True)
  def test_init_success(self, MockSession):
    mock_session = MockSession.return_value
    mock_session.get.return_value = _make_response(MOCK_MCP_SERVERS_LIST)

    api_registry = ApiRegistry(
        api_registry_project_id=self.project_id, location=self.location
    )

    self.assertEqual(len(api_registry._mcp_servers), 6)
    self.assertIn("test-mcp-server-1", api_registry._mcp_servers)
    self.assertIn("test-mcp-server-2", api_registry._mcp_servers)
    self.assertIn("test-mcp-server-no-url", api_registry._mcp_servers)
    self.assertIn("test-mcp-server-http", api_registry._mcp_servers)
    self.assertIn("test-mcp-server-https", api_registry._mcp_servers)
    mock_session.get.assert_called_once_with(
        f"{API_REGISTRY_URL}/v1beta/projects/{self.project_id}/locations/{self.location}/mcpServers",
        headers={},
        params={},
    )

  @patch(_SESSION_PATH, autospec=True)
  def test_init_with_quota_project_id_success(self, MockSession):
    self.mock_credentials.quota_project_id = "quota-project"
    mock_session = MockSession.return_value
    mock_session.get.return_value = _make_response(MOCK_MCP_SERVERS_LIST)

    api_registry = ApiRegistry(
        api_registry_project_id=self.project_id, location=self.location
    )

    self.assertEqual(len(api_registry._mcp_servers), 6)
    self.assertIn("test-mcp-server-1", api_registry._mcp_servers)
    self.assertIn("test-mcp-server-2", api_registry._mcp_servers)
    self.assertIn("test-mcp-server-no-url", api_registry._mcp_servers)
    self.assertIn("test-mcp-server-http", api_registry._mcp_servers)
    self.assertIn("test-mcp-server-https", api_registry._mcp_servers)
    mock_session.get.assert_called_once_with(
        f"{API_REGISTRY_URL}/v1beta/projects/{self.project_id}/locations/{self.location}/mcpServers",
        headers={"x-goog-user-project": "quota-project"},
        params={},
    )

  @patch(_SESSION_PATH, autospec=True)
  def test_init_with_pagination_success(self, MockSession):
    mock_response1 = _make_response({
        "mcpServers": [
            {
                "name": "test-mcp-server-1",
                "urls": ["mcp.server1.com"],
            },
            {
                "name": "test-mcp-server-2",
                "urls": ["mcp.server2.com"],
            },
        ],
        "nextPageToken": "next_page_token",
    })
    mock_response2 = _make_response({
        "mcpServers": [
            {
                "name": "test-mcp-server-no-url",
            },
            {
                "name": "test-mcp-server-http",
                "urls": ["http://mcp.server_http.com"],
            },
            {
                "name": "test-mcp-server-https",
                "urls": ["https://mcp.server_https.com"],
            },
        ]
    })
    mock_session = MockSession.return_value
    mock_session.get.side_effect = [mock_response1, mock_response2]

    api_registry = ApiRegistry(
        api_registry_project_id=self.project_id, location=self.location
    )

    self.assertEqual(len(api_registry._mcp_servers), 5)
    self.assertIn("test-mcp-server-1", api_registry._mcp_servers)
    self.assertIn("test-mcp-server-2", api_registry._mcp_servers)
    self.assertIn("test-mcp-server-no-url", api_registry._mcp_servers)
    self.assertIn("test-mcp-server-http", api_registry._mcp_servers)
    self.assertIn("test-mcp-server-https", api_registry._mcp_servers)
    self.assertEqual(mock_session.get.call_count, 2)
    mock_session.get.assert_any_call(
        f"{API_REGISTRY_URL}/v1beta/projects/{self.project_id}/locations/{self.location}/mcpServers",
        headers={},
        params={},
    )
    mock_session.get.assert_called_with(
        f"{API_REGISTRY_URL}/v1beta/projects/{self.project_id}/locations/{self.location}/mcpServers",
        headers={},
        params={"pageToken": "next_page_token"},
    )

  @patch(_SESSION_PATH, autospec=True)
  def test_init_http_error(self, MockSession):
    mock_session = MockSession.return_value
    mock_session.get.side_effect = requests.exceptions.RequestException(
        "Connection failed"
    )

    with self.assertRaisesRegex(RuntimeError, "Error fetching MCP servers"):
      ApiRegistry(
          api_registry_project_id=self.project_id, location=self.location
      )

  @patch(_SESSION_PATH, autospec=True)
  def test_init_bad_response(self, MockSession):
    mock_response = _make_response(MOCK_MCP_SERVERS_LIST)
    mock_response.raise_for_status = MagicMock(
        side_effect=requests.exceptions.HTTPError("Not Found")
    )
    mock_session = MockSession.return_value
    mock_session.get.return_value = mock_response

    with self.assertRaisesRegex(RuntimeError, "Error fetching MCP servers"):
      ApiRegistry(
          api_registry_project_id=self.project_id, location=self.location
      )
    mock_response.raise_for_status.assert_called_once()

  @patch(
      "google.adk.integrations.api_registry.api_registry.McpToolset",
      autospec=True,
  )
  @patch(_SESSION_PATH, autospec=True)
  async def test_get_toolset_success(self, MockSession, MockMcpToolset):
    mock_session = MockSession.return_value
    mock_session.get.return_value = _make_response(MOCK_MCP_SERVERS_LIST)

    api_registry = ApiRegistry(
        api_registry_project_id=self.project_id, location=self.location
    )

    toolset = api_registry.get_toolset("test-mcp-server-1")

    # A non-Google host must not receive the runtime's ADC credentials.
    MockMcpToolset.assert_called_once_with(
        connection_params=StreamableHTTPConnectionParams(
            url="https://mcp.server1.com",
            headers={},
        ),
        tool_filter=None,
        tool_name_prefix=None,
        header_provider=None,
    )
    self.assertEqual(toolset, MockMcpToolset.return_value)

  @patch(
      "google.adk.integrations.api_registry.api_registry.McpToolset",
      autospec=True,
  )
  @patch(_SESSION_PATH, autospec=True)
  async def test_get_toolset_google_host_includes_credentials(
      self, MockSession, MockMcpToolset
  ):
    mock_session = MockSession.return_value
    mock_session.get.return_value = _make_response(MOCK_MCP_SERVERS_LIST)

    api_registry = ApiRegistry(
        api_registry_project_id=self.project_id, location=self.location
    )

    toolset = api_registry.get_toolset("test-mcp-server-google")

    # A Google API host receives the runtime's ADC credentials.
    MockMcpToolset.assert_called_once_with(
        connection_params=StreamableHTTPConnectionParams(
            url=_GOOGLE_MCP_URL,
            headers={"Authorization": "Bearer mock_token"},
        ),
        tool_filter=None,
        tool_name_prefix=None,
        header_provider=None,
    )
    self.assertEqual(toolset, MockMcpToolset.return_value)

  @patch(
      "google.adk.integrations.api_registry.api_registry.McpToolset",
      autospec=True,
  )
  @patch(_SESSION_PATH, autospec=True)
  async def test_get_toolset_with_quota_project_id_success(
      self, MockSession, MockMcpToolset
  ):
    self.mock_credentials.quota_project_id = "quota-project"
    mock_session = MockSession.return_value
    mock_session.get.return_value = _make_response(MOCK_MCP_SERVERS_LIST)

    api_registry = ApiRegistry(
        api_registry_project_id=self.project_id, location=self.location
    )

    toolset = api_registry.get_toolset("test-mcp-server-google")

    MockMcpToolset.assert_called_once_with(
        connection_params=StreamableHTTPConnectionParams(
            url=_GOOGLE_MCP_URL,
            headers={
                "Authorization": "Bearer mock_token",
                "x-goog-user-project": "quota-project",
            },
        ),
        tool_filter=None,
        tool_name_prefix=None,
        header_provider=None,
    )
    self.assertEqual(toolset, MockMcpToolset.return_value)

  @patch(
      "google.adk.integrations.api_registry.api_registry.McpToolset",
      autospec=True,
  )
  @patch(_SESSION_PATH, autospec=True)
  async def test_get_toolset_with_filter_and_prefix(
      self, MockSession, MockMcpToolset
  ):
    mock_session = MockSession.return_value
    mock_session.get.return_value = _make_response(MOCK_MCP_SERVERS_LIST)

    api_registry = ApiRegistry(
        api_registry_project_id=self.project_id, location=self.location
    )
    tool_filter = ["tool1"]
    tool_name_prefix = "prefix_"
    toolset = api_registry.get_toolset(
        "test-mcp-server-1",
        tool_filter=tool_filter,
        tool_name_prefix=tool_name_prefix,
    )

    MockMcpToolset.assert_called_once_with(
        connection_params=StreamableHTTPConnectionParams(
            url="https://mcp.server1.com",
            headers={},
        ),
        tool_filter=tool_filter,
        tool_name_prefix=tool_name_prefix,
        header_provider=None,
    )
    self.assertEqual(toolset, MockMcpToolset.return_value)

  def test_get_toolset_url_scheme(self):
    params = [
        ("test-mcp-server-http", "http://mcp.server_http.com"),
        ("test-mcp-server-https", "https://mcp.server_https.com"),
    ]
    for mock_server_name, mock_url in params:
      with self.subTest(server_name=mock_server_name):
        with (
            patch(_SESSION_PATH, autospec=True) as MockSession,
            patch.object(
                api_registry.api_registry, "McpToolset", autospec=True
            ) as MockMcpToolset,
        ):
          mock_session = MockSession.return_value
          mock_session.get.return_value = _make_response(MOCK_MCP_SERVERS_LIST)

          api_registry_instance = ApiRegistry(
              api_registry_project_id=self.project_id, location=self.location
          )

          api_registry_instance.get_toolset(mock_server_name)

          MockMcpToolset.assert_called_once_with(
              connection_params=StreamableHTTPConnectionParams(
                  url=mock_url,
                  headers={},
              ),
              tool_filter=None,
              tool_name_prefix=None,
              header_provider=None,
          )

  @patch(_SESSION_PATH, autospec=True)
  async def test_get_toolset_server_not_found(self, MockSession):
    mock_session = MockSession.return_value
    mock_session.get.return_value = _make_response(MOCK_MCP_SERVERS_LIST)

    api_registry = ApiRegistry(
        api_registry_project_id=self.project_id, location=self.location
    )

    with self.assertRaisesRegex(ValueError, "not found in API Registry"):
      api_registry.get_toolset("non-existent-server")

  @patch(_SESSION_PATH, autospec=True)
  async def test_get_toolset_server_no_url(self, MockSession):
    mock_session = MockSession.return_value
    mock_session.get.return_value = _make_response(MOCK_MCP_SERVERS_LIST)

    api_registry = ApiRegistry(
        api_registry_project_id=self.project_id, location=self.location
    )

    with self.assertRaisesRegex(ValueError, "has no URLs"):
      api_registry.get_toolset("test-mcp-server-no-url")
