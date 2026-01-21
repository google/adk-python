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

import sys
import unittest
from unittest.mock import create_autospec
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.tools import api_registry
from google.adk.tools.api_registry import ApiRegistry
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
import httpx

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

  @patch("httpx.Client", autospec=True)
  def test_init_success(self, MockHttpClient):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value=MOCK_MCP_SERVERS_LIST)
    mock_client_instance = MockHttpClient.return_value
    mock_client_instance.__enter__.return_value = mock_client_instance
    mock_client_instance.get.return_value = mock_response

    api_registry = ApiRegistry(
        api_registry_project_id=self.project_id, location=self.location
    )

    self.assertEqual(len(api_registry._mcp_servers), 5)
    self.assertIn("test-mcp-server-1", api_registry._mcp_servers)
    self.assertIn("test-mcp-server-2", api_registry._mcp_servers)
    self.assertIn("test-mcp-server-no-url", api_registry._mcp_servers)
    self.assertIn("test-mcp-server-http", api_registry._mcp_servers)
    self.assertIn("test-mcp-server-https", api_registry._mcp_servers)
    mock_client_instance.get.assert_called_once_with(
        f"https://cloudapiregistry.googleapis.com/v1beta/projects/{self.project_id}/locations/{self.location}/mcpServers",
        headers={
            "Authorization": "Bearer mock_token",
            "Content-Type": "application/json",
        },
        params={},
    )

  @patch("httpx.Client", autospec=True)
  def test_init_with_quota_project_id_success(self, MockHttpClient):
    self.mock_credentials.quota_project_id = "quota-project"
    mock_response = create_autospec(httpx.Response, instance=True)
    mock_response.json.return_value = MOCK_MCP_SERVERS_LIST
    mock_client_instance = MockHttpClient.return_value
    mock_client_instance.__enter__.return_value = mock_client_instance
    mock_client_instance.get.return_value = mock_response

    api_registry = ApiRegistry(
        api_registry_project_id=self.project_id, location=self.location
    )

    self.assertEqual(len(api_registry._mcp_servers), 5)
    self.assertIn("test-mcp-server-1", api_registry._mcp_servers)
    self.assertIn("test-mcp-server-2", api_registry._mcp_servers)
    self.assertIn("test-mcp-server-no-url", api_registry._mcp_servers)
    self.assertIn("test-mcp-server-http", api_registry._mcp_servers)
    self.assertIn("test-mcp-server-https", api_registry._mcp_servers)
    mock_client_instance.get.assert_called_once_with(
        f"https://cloudapiregistry.googleapis.com/v1beta/projects/{self.project_id}/locations/{self.location}/mcpServers",
        headers={
            "Authorization": "Bearer mock_token",
            "Content-Type": "application/json",
            "x-goog-user-project": "quota-project",
        },
        params={},
    )

  @patch("httpx.Client", autospec=True)
  def test_init_with_pagination_success(self, MockHttpClient):
    mock_response1 = create_autospec(httpx.Response, instance=True)
    mock_response1.json.return_value = {
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
    }
    mock_response2 = create_autospec(httpx.Response, instance=True)
    mock_response2.json.return_value = {
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
    }
    mock_client_instance = MockHttpClient.return_value
    mock_client_instance.__enter__.return_value = mock_client_instance
    mock_client_instance.get.side_effect = [mock_response1, mock_response2]

    api_registry = ApiRegistry(
        api_registry_project_id=self.project_id, location=self.location
    )

    self.assertEqual(len(api_registry._mcp_servers), 5)
    self.assertIn("test-mcp-server-1", api_registry._mcp_servers)
    self.assertIn("test-mcp-server-2", api_registry._mcp_servers)
    self.assertIn("test-mcp-server-no-url", api_registry._mcp_servers)
    self.assertIn("test-mcp-server-http", api_registry._mcp_servers)
    self.assertIn("test-mcp-server-https", api_registry._mcp_servers)
    self.assertEqual(mock_client_instance.get.call_count, 2)
    mock_client_instance.get.assert_any_call(
        f"https://cloudapiregistry.googleapis.com/v1beta/projects/{self.project_id}/locations/{self.location}/mcpServers",
        headers={
            "Authorization": "Bearer mock_token",
            "Content-Type": "application/json",
        },
        params={},
    )
    mock_client_instance.get.assert_called_with(
        f"https://cloudapiregistry.googleapis.com/v1beta/projects/{self.project_id}/locations/{self.location}/mcpServers",
        headers={
            "Authorization": "Bearer mock_token",
            "Content-Type": "application/json",
        },
        params={"pageToken": "next_page_token"},
    )

  @patch("httpx.Client", autospec=True)
  def test_init_http_error(self, MockHttpClient):
    mock_client_instance = MockHttpClient.return_value
    mock_client_instance.__enter__.return_value = mock_client_instance
    mock_client_instance.get.side_effect = httpx.RequestError(
        "Connection failed"
    )

    with self.assertRaisesRegex(RuntimeError, "Error fetching MCP servers"):
      ApiRegistry(
          api_registry_project_id=self.project_id, location=self.location
      )

  @patch("httpx.Client", autospec=True)
  def test_init_bad_response(self, MockHttpClient):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=MagicMock()
        )
    )
    mock_client_instance = MockHttpClient.return_value
    mock_client_instance.__enter__.return_value = mock_client_instance
    mock_client_instance.get.return_value = mock_response

    with self.assertRaisesRegex(RuntimeError, "Error fetching MCP servers"):
      ApiRegistry(
          api_registry_project_id=self.project_id, location=self.location
      )
    mock_response.raise_for_status.assert_called_once()

  @patch("google.adk.tools.api_registry.McpToolset", autospec=True)
  @patch("httpx.Client", autospec=True)
  async def test_get_toolset_success(self, MockHttpClient, MockMcpToolset):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value=MOCK_MCP_SERVERS_LIST)
    mock_client_instance = MockHttpClient.return_value
    mock_client_instance.__enter__.return_value = mock_client_instance
    mock_client_instance.get.return_value = mock_response

    api_registry = ApiRegistry(
        api_registry_project_id=self.project_id, location=self.location
    )

    toolset = api_registry.get_toolset("test-mcp-server-1")

    MockMcpToolset.assert_called_once_with(
        connection_params=StreamableHTTPConnectionParams(
            url="https://mcp.server1.com",
            headers={"Authorization": "Bearer mock_token"},
        ),
        tool_filter=None,
        tool_name_prefix=None,
        header_provider=None,
    )
    self.assertEqual(toolset, MockMcpToolset.return_value)

  @patch("google.adk.tools.api_registry.McpToolset", autospec=True)
  @patch("httpx.Client", autospec=True)
  async def test_get_toolset_with_quota_project_id_success(
      self, MockHttpClient, MockMcpToolset
  ):
    self.mock_credentials.quota_project_id = "quota-project"
    mock_response = create_autospec(httpx.Response, instance=True)
    mock_response.json.return_value = MOCK_MCP_SERVERS_LIST
    mock_client_instance = MockHttpClient.return_value
    mock_client_instance.__enter__.return_value = mock_client_instance
    mock_client_instance.get.return_value = mock_response

    api_registry = ApiRegistry(
        api_registry_project_id=self.project_id, location=self.location
    )

    toolset = api_registry.get_toolset("test-mcp-server-1")

    MockMcpToolset.assert_called_once_with(
        connection_params=StreamableHTTPConnectionParams(
            url="https://mcp.server1.com",
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

  @patch("google.adk.tools.api_registry.McpToolset", autospec=True)
  @patch("httpx.Client", autospec=True)
  async def test_get_toolset_with_filter_and_prefix(
      self, MockHttpClient, MockMcpToolset
  ):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value=MOCK_MCP_SERVERS_LIST)
    mock_client_instance = MockHttpClient.return_value
    mock_client_instance.__enter__.return_value = mock_client_instance
    mock_client_instance.get.return_value = mock_response

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
            headers={"Authorization": "Bearer mock_token"},
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
            patch.object(httpx, "Client", autospec=True) as MockHttpClient,
            patch.object(
                api_registry, "McpToolset", autospec=True
            ) as MockMcpToolset,
        ):
          mock_response = create_autospec(httpx.Response, instance=True)
          mock_response.json.return_value = MOCK_MCP_SERVERS_LIST
          mock_client_instance = MockHttpClient.return_value
          mock_client_instance.__enter__.return_value = mock_client_instance
          mock_client_instance.get.return_value = mock_response

          api_registry_instance = ApiRegistry(
              api_registry_project_id=self.project_id, location=self.location
          )

          api_registry_instance.get_toolset(mock_server_name)

          MockMcpToolset.assert_called_once_with(
              connection_params=StreamableHTTPConnectionParams(
                  url=mock_url,
                  headers={"Authorization": "Bearer mock_token"},
              ),
              tool_filter=None,
              tool_name_prefix=None,
              header_provider=None,
          )

  @patch("httpx.Client", autospec=True)
  async def test_get_toolset_server_not_found(self, MockHttpClient):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value=MOCK_MCP_SERVERS_LIST)
    mock_client_instance = MockHttpClient.return_value
    mock_client_instance.__enter__.return_value = mock_client_instance
    mock_client_instance.get.return_value = mock_response

    api_registry = ApiRegistry(
        api_registry_project_id=self.project_id, location=self.location
    )

    with self.assertRaisesRegex(ValueError, "not found in API Registry"):
      api_registry.get_toolset("non-existent-server")

  @patch("httpx.Client", autospec=True)
  async def test_get_toolset_server_no_url(self, MockHttpClient):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value=MOCK_MCP_SERVERS_LIST)
    mock_client_instance = MockHttpClient.return_value
    mock_client_instance.__enter__.return_value = mock_client_instance
    mock_client_instance.get.return_value = mock_response

    api_registry = ApiRegistry(
        api_registry_project_id=self.project_id, location=self.location
    )

    with self.assertRaisesRegex(ValueError, "has no URLs"):
      api_registry.get_toolset("test-mcp-server-no-url")
