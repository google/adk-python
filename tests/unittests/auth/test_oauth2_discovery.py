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

import json
import ssl
from unittest.mock import call
from unittest.mock import Mock
from unittest.mock import patch

from google.adk.auth.oauth2_discovery import AuthorizationServerMetadata
from google.adk.auth.oauth2_discovery import OAuth2DiscoveryManager
from google.adk.auth.oauth2_discovery import ProtectedResourceMetadata
from google.adk.utils import _mtls_utils
import httpx
import pytest


class TestOAuth2Discovery:
  """Tests for the OAuth2DiscoveryManager class."""

  @pytest.fixture
  def auth_server_metadata(self):
    """Create AuthorizationServerMetadata object."""
    return AuthorizationServerMetadata(
        issuer="https://auth.example.com",
        authorization_endpoint="https://auth.example.com/authorize",
        token_endpoint="https://auth.example.com/token",
        scopes_supported=["read", "write"],
    )

  @pytest.fixture
  def resource_metadata(self):
    """Create ProtectedResourceMetadata object."""
    return ProtectedResourceMetadata(
        resource="https://resource.example.com",
        authorization_servers=["https://auth.example.com"],
    )

  @pytest.fixture
  def mock_failed_response(self):
    """Create a mock HTTP response with a failure status."""
    response = Mock()
    response.raise_for_status.side_effect = httpx.HTTPError("Failed")
    return response

  @pytest.fixture
  def mock_empty_response(self):
    """Create a mock HTTP response with an empty JSON body."""
    response = Mock()
    response.json = lambda: {}
    return response

  @pytest.fixture
  def mock_invalid_json_response(self):
    """Create a mock HTTP response with an invalid JSON body."""
    response = Mock()
    response.json.side_effect = json.decoder.JSONDecodeError(
        "Invalid JSON", "invalid_json", 0
    )
    return response

  def mock_success_response(self, json_data):
    """Create a mock HTTP successful response with auth server metadata."""
    response = Mock()
    response.json = json_data.model_dump
    return response

  @patch("httpx.AsyncClient.get")
  @pytest.mark.asyncio
  async def test_discover_auth_server_metadata_failed(
      self,
      mock_get,
      mock_failed_response,
  ):
    """Test discovering auth server metadata with failed response."""

    mock_get.side_effect = mock_failed_response
    discovery_manager = OAuth2DiscoveryManager()
    result = await discovery_manager.discover_auth_server_metadata(
        "https://auth.example.com"
    )
    assert not result
    mock_get.assert_has_calls([
        call(
            "https://auth.example.com/.well-known/oauth-authorization-server",
            timeout=5,
        ),
        call(
            "https://auth.example.com/.well-known/openid-configuration",
            timeout=5,
        ),
    ])

  @pytest.mark.asyncio
  async def test_discover_metadata_invalid_url(self):
    """Test discovering resource/auth metadata with an invalid URL."""
    discovery_manager = OAuth2DiscoveryManager()
    result = await discovery_manager.discover_auth_server_metadata("bad_url")
    assert not result
    result = await discovery_manager.discover_resource_metadata("bad_url")
    assert not result

  @patch("httpx.AsyncClient.get")
  @pytest.mark.asyncio
  async def test_discover_auth_server_metadata_without_path(
      self,
      mock_get,
      auth_server_metadata,
      mock_empty_response,
  ):
    """Test discovering auth server metadata with an issuer URL without a path."""

    mock_get.side_effect = [
        mock_empty_response,
        self.mock_success_response(auth_server_metadata),
    ]
    discovery_manager = OAuth2DiscoveryManager()
    result = await discovery_manager.discover_auth_server_metadata(
        "https://auth.example.com/"
    )
    assert result == auth_server_metadata
    mock_get.assert_has_calls([
        call(
            "https://auth.example.com/.well-known/oauth-authorization-server",
            timeout=5,
        ),
        call(
            "https://auth.example.com/.well-known/openid-configuration",
            timeout=5,
        ),
    ])

  @patch("httpx.AsyncClient.get")
  @pytest.mark.asyncio
  async def test_discover_auth_server_metadata_with_path(
      self,
      mock_get,
      auth_server_metadata,
      mock_failed_response,
      mock_invalid_json_response,
  ):
    """Test discovering auth server metadata with an issuer URL with a path."""

    auth_server_metadata.issuer = "https://auth.example.com/oauth"
    mock_get.side_effect = [
        mock_failed_response,
        mock_invalid_json_response,
        self.mock_success_response(auth_server_metadata),
    ]
    discovery_manager = OAuth2DiscoveryManager()
    result = await discovery_manager.discover_auth_server_metadata(
        "https://auth.example.com/oauth"
    )
    assert result == auth_server_metadata
    mock_get.assert_has_calls([
        call(
            "https://auth.example.com/.well-known/oauth-authorization-server/oauth",
            timeout=5,
        ),
        call(
            "https://auth.example.com/.well-known/openid-configuration/oauth",
            timeout=5,
        ),
        call(
            "https://auth.example.com/oauth/.well-known/openid-configuration",
            timeout=5,
        ),
    ])

  @patch("httpx.AsyncClient.get")
  @pytest.mark.asyncio
  async def test_discover_auth_server_metadata_discard_mismatched_issuer(
      self,
      mock_get,
      auth_server_metadata,
  ):
    """Test discover_auth_server_metadata() discards response with mismatched issuer."""

    bad_auth_server_metadata = auth_server_metadata.model_copy(
        update={"issuer": "https://bad.example.com"}
    )
    mock_get.side_effect = [
        self.mock_success_response(bad_auth_server_metadata),
        self.mock_success_response(auth_server_metadata),
    ]
    discovery_manager = OAuth2DiscoveryManager()
    result = await discovery_manager.discover_auth_server_metadata(
        "https://auth.example.com"
    )
    assert result == auth_server_metadata
    mock_get.assert_has_calls([
        call(
            "https://auth.example.com/.well-known/oauth-authorization-server",
            timeout=5,
        ),
        call(
            "https://auth.example.com/.well-known/openid-configuration",
            timeout=5,
        ),
    ])

  @patch("httpx.AsyncClient.get")
  @pytest.mark.asyncio
  async def test_discover_auth_server_metadata_issuer_trailing_slash(
      self,
      mock_get,
      auth_server_metadata,
  ):
    """Test issuer match tolerates a trailing slash on the returned issuer.

    Some servers (e.g. FastMCP with GoogleProvider) return an issuer with a
    trailing slash even though the requested issuer_url has none.
    """

    auth_server_metadata.issuer = "https://auth.example.com/"
    mock_get.side_effect = [
        self.mock_success_response(auth_server_metadata),
    ]
    discovery_manager = OAuth2DiscoveryManager()
    result = await discovery_manager.discover_auth_server_metadata(
        "https://auth.example.com"
    )
    assert result == auth_server_metadata

  @patch("httpx.AsyncClient.get")
  @pytest.mark.asyncio
  async def test_discover_resource_metadata_failed(
      self,
      mock_get,
      mock_failed_response,
  ):
    """Test discovering resource metadata fails."""

    mock_get.return_value = mock_failed_response
    discovery_manager = OAuth2DiscoveryManager()
    result = await discovery_manager.discover_resource_metadata(
        "https://resource.example.com"
    )
    assert not result
    mock_get.assert_called_once_with(
        "https://resource.example.com/.well-known/oauth-protected-resource",
        timeout=5,
    )

  @patch("httpx.AsyncClient.get")
  @pytest.mark.asyncio
  async def test_discover_resource_metadata_without_path(
      self, mock_get, resource_metadata
  ):
    """Test discovering resource metadata with a resource URL without a path."""
    mock_get.return_value = self.mock_success_response(resource_metadata)
    discovery_manager = OAuth2DiscoveryManager()
    result = await discovery_manager.discover_resource_metadata(
        "https://resource.example.com/"
    )
    assert result == resource_metadata
    mock_get.assert_called_once_with(
        "https://resource.example.com/.well-known/oauth-protected-resource",
        timeout=5,
    )

  @patch("httpx.AsyncClient.get")
  @pytest.mark.asyncio
  async def test_discover_resource_metadata_with_path(
      self, mock_get, resource_metadata
  ):
    """Test discovering resource metadata with a resource URL with a path."""
    resource_metadata.resource = "https://resource.example.com/tenant1"
    mock_get.return_value = self.mock_success_response(resource_metadata)
    discovery_manager = OAuth2DiscoveryManager()
    result = await discovery_manager.discover_resource_metadata(
        "https://resource.example.com/tenant1"
    )
    assert result == resource_metadata
    mock_get.assert_called_once_with(
        "https://resource.example.com/.well-known/oauth-protected-resource/tenant1",
        timeout=5,
    )

  @patch("httpx.AsyncClient.get")
  @pytest.mark.asyncio
  async def test_discover_resource_metadata_discard_mismatched_resource(
      self,
      mock_get,
      resource_metadata,
  ):
    """Test discover_resource_metadata() discards response with mismatched resource."""

    resource_metadata.resource = "https://bad.example.com"
    mock_get.return_value = self.mock_success_response(resource_metadata)
    discovery_manager = OAuth2DiscoveryManager()
    result = await discovery_manager.discover_resource_metadata(
        "https://resource.example.com"
    )
    assert not result
    mock_get.assert_called_once_with(
        "https://resource.example.com/.well-known/oauth-protected-resource",
        timeout=5,
    )

  @pytest.mark.parametrize(
      "client_cert_ctx, expected_url",
      [
          (
              ssl.create_default_context(),
              "https://mcp.mtls.googleapis.com/.well-known/oauth-protected-resource/v1",
          ),
          (
              None,
              "https://mcp.googleapis.com/.well-known/oauth-protected-resource/v1",
          ),
      ],
  )
  @patch.dict("os.environ", {"GOOGLE_API_USE_MTLS_ENDPOINT": "auto"})
  @pytest.mark.asyncio
  async def test_discover_resource_metadata_mtls_routing(
      self, client_cert_ctx, expected_url
  ):
    """A Google resource is discovered on its mTLS host only with a client cert."""
    resource_url = "https://mcp.googleapis.com/v1"
    sent_requests = []

    # Like the real server, report the host the request arrived on.
    async def handle_async_request(_, request):
      sent_requests.append(request)
      return httpx.Response(
          200,
          json={
              "resource": f"https://{request.url.host}/v1",
              "authorization_servers": [f"https://{request.url.host}"],
          },
          request=request,
      )

    with (
        patch.object(
            _mtls_utils,
            "_client_cert_ssl_context",
            autospec=True,
            return_value=client_cert_ctx,
        ),
        patch.object(
            httpx.AsyncHTTPTransport,
            "handle_async_request",
            handle_async_request,
        ),
    ):
      result = await OAuth2DiscoveryManager().discover_resource_metadata(
          resource_url
      )

    assert [str(r.url) for r in sent_requests] == [expected_url]
    assert result == ProtectedResourceMetadata(
        resource="https://mcp.googleapis.com/v1",
        authorization_servers=["https://mcp.googleapis.com"],
    )

  @pytest.mark.parametrize(
      "client_cert_ctx, expected_url",
      [
          (
              ssl.create_default_context(),
              "https://auth.mtls.googleapis.com/.well-known/oauth-authorization-server",
          ),
          (
              None,
              "https://auth.googleapis.com/.well-known/oauth-authorization-server",
          ),
      ],
  )
  @patch.dict("os.environ", {"GOOGLE_API_USE_MTLS_ENDPOINT": "auto"})
  @pytest.mark.asyncio
  async def test_discover_auth_server_metadata_mtls_routing(
      self, client_cert_ctx, expected_url
  ):
    """A Google auth server is discovered on its mTLS host only with a cert."""
    issuer_url = "https://auth.googleapis.com"
    sent_requests = []

    # Like the real server, report the host the request arrived on.
    async def handle_async_request(_, request):
      sent_requests.append(request)
      issuer = f"https://{request.url.host}"
      metadata = {
          "issuer": issuer,
          "authorization_endpoint": f"{issuer}/authorize",
          "token_endpoint": f"{issuer}/token",
      }
      return httpx.Response(200, json=metadata, request=request)

    with (
        patch.object(
            _mtls_utils,
            "_client_cert_ssl_context",
            autospec=True,
            return_value=client_cert_ctx,
        ),
        patch.object(
            httpx.AsyncHTTPTransport,
            "handle_async_request",
            handle_async_request,
        ),
    ):
      result = await OAuth2DiscoveryManager().discover_auth_server_metadata(
          issuer_url
      )

    assert [str(r.url) for r in sent_requests] == [expected_url]
    assert result == AuthorizationServerMetadata(
        issuer="https://auth.googleapis.com",
        authorization_endpoint="https://auth.googleapis.com/authorize",
        token_endpoint="https://auth.googleapis.com/token",
    )

  @pytest.mark.asyncio
  async def test_discovery_skips_client_cert_for_other_hosts(self):
    """Non-Google auth servers and resources never get the client cert."""
    with (
        patch.object(
            _mtls_utils,
            "_client_cert_ssl_context",
            autospec=True,
            return_value=ssl.create_default_context(),
        ) as mock_ctx,
        patch.object(
            httpx.AsyncHTTPTransport,
            "handle_async_request",
            side_effect=httpx.ConnectError("offline"),
        ),
    ):
      manager = OAuth2DiscoveryManager()
      await manager.discover_auth_server_metadata("https://auth.example.com")
      await manager.discover_resource_metadata("https://mcp.example.com/v1")

    mock_ctx.assert_not_called()

  @pytest.mark.asyncio
  async def test_discover_resource_metadata_rejects_other_mtls_host(self):
    """Only the requested Google API's own mTLS host is accepted."""
    with patch.object(
        httpx.AsyncHTTPTransport,
        "handle_async_request",
        autospec=True,
        side_effect=lambda _, request: httpx.Response(
            200,
            json={"resource": "https://other.mtls.googleapis.com/v1"},
            request=request,
        ),
    ):
      result = await OAuth2DiscoveryManager().discover_resource_metadata(
          "https://mcp.googleapis.com/v1"
      )

    assert result is None
