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
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.integrations import api_registry
from google.adk.integrations.api_registry import ApiRegistry
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
import requests

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

    mock_session_patcher = patch(
        "google.auth.transport.requests.AuthorizedSession",
        autospec=True,
    )
    self.mock_session_class = mock_session_patcher.start()
    self.mock_session = self.mock_session_class.return_value
    self.mock_session.__enter__.return_value = self.mock_session
    self.addCleanup(mock_session_patcher.stop)

    mock_use_cert_patcher = patch(
        "google.adk.integrations.api_registry.api_registry._mtls_utils.use_client_cert_effective",
        return_value=False,
    )
    mock_use_cert_patcher.start()
    self.addCleanup(mock_use_cert_patcher.stop)

  def test_init_success(self):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value=MOCK_MCP_SERVERS_LIST)
    self.mock_session.get.return_value = mock_response

    api_registry_instance = ApiRegistry(
        api_registry_project_id=self.project_id, location=self.location
    )

    self.assertEqual(len(api_registry_instance._mcp_servers), 5)
    self.assertIn("test-mcp-server-1", api_registry_instance._mcp_servers)
    self.assertIn("test-mcp-server-2", api_registry_instance._mcp_servers)
    self.assertIn("test-mcp-server-no-url", api_registry_instance._mcp_servers)
    self.assertIn("test-mcp-server-http", api_registry_instance._mcp_servers)
    self.assertIn("test-mcp-server-https", api_registry_instance._mcp_servers)
    self.mock_session.get.assert_called_once_with(
        f"https://cloudapiregistry.googleapis.com/v1beta/projects/{self.project_id}/locations/{self.location}/mcpServers",
        headers={
            "Content-Type": "application/json",
        },
        params={"filter": "enabled=false"},
    )

  def test_init_with_quota_project_id_success(self):
    self.mock_credentials.quota_project_id = "quota-project"
    mock_response = MagicMock()
    mock_response.json.return_value = MOCK_MCP_SERVERS_LIST
    self.mock_session.get.return_value = mock_response

    api_registry_instance = ApiRegistry(
        api_registry_project_id=self.project_id, location=self.location
    )

    self.assertEqual(len(api_registry_instance._mcp_servers), 5)
    self.mock_session.get.assert_called_once_with(
        f"https://cloudapiregistry.googleapis.com/v1beta/projects/{self.project_id}/locations/{self.location}/mcpServers",
        headers={
            "Content-Type": "application/json",
            "x-goog-user-project": "quota-project",
        },
        params={"filter": "enabled=false"},
    )

  def test_init_with_pagination_success(self):
    mock_response1 = MagicMock()
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
    mock_response2 = MagicMock()
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
    self.mock_session.get.side_effect = [mock_response1, mock_response2]

    api_registry_instance = ApiRegistry(
        api_registry_project_id=self.project_id, location=self.location
    )

    self.assertEqual(len(api_registry_instance._mcp_servers), 5)
    self.assertEqual(self.mock_session.get.call_count, 2)
    self.mock_session.get.assert_any_call(
        f"https://cloudapiregistry.googleapis.com/v1beta/projects/{self.project_id}/locations/{self.location}/mcpServers",
        headers={
            "Content-Type": "application/json",
        },
        params={"filter": "enabled=false"},
    )
    self.mock_session.get.assert_called_with(
        f"https://cloudapiregistry.googleapis.com/v1beta/projects/{self.project_id}/locations/{self.location}/mcpServers",
        headers={
            "Content-Type": "application/json",
        },
        params={"filter": "enabled=false", "pageToken": "next_page_token"},
    )

  def test_init_http_error(self):
    self.mock_session.get.side_effect = requests.exceptions.RequestException(
        "Connection failed"
    )

    with self.assertRaisesRegex(RuntimeError, "Error fetching MCP servers"):
      ApiRegistry(
          api_registry_project_id=self.project_id, location=self.location
      )

  def test_init_bad_response(self):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock(
        side_effect=requests.exceptions.HTTPError(
            "Not Found", request=MagicMock(), response=MagicMock()
        )
    )
    self.mock_session.get.return_value = mock_response

    with self.assertRaisesRegex(RuntimeError, "Error fetching MCP servers"):
      ApiRegistry(
          api_registry_project_id=self.project_id, location=self.location
      )
    mock_response.raise_for_status.assert_called_once()

  @patch(
      "google.adk.integrations.api_registry.api_registry.McpToolset",
      autospec=True,
  )
  def test_get_toolset_success(self, MockMcpToolset):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value=MOCK_MCP_SERVERS_LIST)
    self.mock_session.get.return_value = mock_response

    api_registry_instance = ApiRegistry(
        api_registry_project_id=self.project_id, location=self.location
    )

    toolset = api_registry_instance.get_toolset("test-mcp-server-1")

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

  @patch(
      "google.adk.integrations.api_registry.api_registry.McpToolset",
      autospec=True,
  )
  def test_get_toolset_with_quota_project_id_success(self, MockMcpToolset):
    self.mock_credentials.quota_project_id = "quota-project"
    mock_response = MagicMock()
    mock_response.json.return_value = MOCK_MCP_SERVERS_LIST
    self.mock_session.get.return_value = mock_response

    api_registry_instance = ApiRegistry(
        api_registry_project_id=self.project_id, location=self.location
    )

    toolset = api_registry_instance.get_toolset("test-mcp-server-1")

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

  @patch(
      "google.adk.integrations.api_registry.api_registry.McpToolset",
      autospec=True,
  )
  def test_get_toolset_with_filter_and_prefix(self, MockMcpToolset):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value=MOCK_MCP_SERVERS_LIST)
    self.mock_session.get.return_value = mock_response

    api_registry_instance = ApiRegistry(
        api_registry_project_id=self.project_id, location=self.location
    )
    tool_filter = ["tool1"]
    tool_name_prefix = "prefix_"
    toolset = api_registry_instance.get_toolset(
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
            patch.object(
                api_registry.api_registry, "McpToolset", autospec=True
            ) as MockMcpToolset,
        ):
          mock_response = MagicMock()
          mock_response.json.return_value = MOCK_MCP_SERVERS_LIST
          self.mock_session.get.return_value = mock_response

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

  def test_get_toolset_server_not_found(self):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value=MOCK_MCP_SERVERS_LIST)
    self.mock_session.get.return_value = mock_response

    api_registry_instance = ApiRegistry(
        api_registry_project_id=self.project_id, location=self.location
    )

    with self.assertRaisesRegex(ValueError, "not found in API Registry"):
      api_registry_instance.get_toolset("non-existent-server")

  def test_get_toolset_server_no_url(self):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value=MOCK_MCP_SERVERS_LIST)
    self.mock_session.get.return_value = mock_response

    api_registry_instance = ApiRegistry(
        api_registry_project_id=self.project_id, location=self.location
    )

    with self.assertRaisesRegex(ValueError, "has no URLs"):
      api_registry_instance.get_toolset("test-mcp-server-no-url")


class TestApiRegistryMtls(unittest.IsolatedAsyncioTestCase):

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

  @patch(
      "google.auth.transport.mtls.has_default_client_cert_source",
      return_value=True,
  )
  @patch("google.auth.transport.mtls.default_client_cert_source")
  @patch.dict(os.environ, {"GOOGLE_API_USE_CLIENT_CERTIFICATE": "true"})
  def test_init_configures_mtls(self, mock_cert_source, _mock_has_cert):
    mock_cert_source.return_value = lambda: (b"cert", b"key")
    with (
        patch(
            "google.adk.integrations.api_registry.api_registry._mtls_utils.use_client_cert_effective",
            return_value=True,
        ),
        patch(
            "google.auth.transport.requests.AuthorizedSession",
            autospec=True,
        ) as mock_session_class,
    ):
      mock_response = MagicMock()
      mock_response.raise_for_status = MagicMock()
      mock_response.json.return_value = MOCK_MCP_SERVERS_LIST
      mock_session = mock_session_class.return_value
      mock_session.__enter__.return_value = mock_session
      mock_session.get.return_value = mock_response

      _ = ApiRegistry(
          api_registry_project_id=self.project_id, location=self.location
      )

      mock_session.configure_mtls_channel.assert_called_once()
      args, _ = mock_session.get.call_args
      self.assertIn("cloudapiregistry.mtls.googleapis.com", args[0])
