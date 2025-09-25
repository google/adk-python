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

import json
from unittest.mock import call
from unittest.mock import Mock
from unittest.mock import patch

from google.adk.auth.oauth2_discovery import AuthorizationServerMetadata
from google.adk.auth.oauth2_discovery import OAuth2DiscoveryManager
from google.adk.auth.oauth2_discovery import ProtectedResourceMetadata
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
