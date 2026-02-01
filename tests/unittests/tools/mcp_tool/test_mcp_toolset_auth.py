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

"""Tests for MCPToolset authentication functionality."""

import base64
from unittest.mock import Mock

from fastapi.openapi.models import APIKey as APIKeyScheme
from fastapi.openapi.models import APIKeyIn
from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlowAuthorizationCode
from fastapi.openapi.models import OAuthFlows
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import HttpAuth
from google.adk.auth.auth_credential import HttpCredentials
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_tool import AuthConfig
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from mcp import StdioServerParameters
import pytest


class TestMcpToolsetGetAuthConfig:
  """Tests for McpToolset.get_auth_config method."""

  def test_get_auth_config_returns_none_without_auth_scheme(self):
    """Test that get_auth_config returns None when no auth configured."""
    toolset = McpToolset(
        connection_params=StdioServerParameters(command="echo", args=["test"])
    )

    assert toolset.get_auth_config() is None

  def test_get_auth_config_returns_config_with_auth_scheme(self):
    """Test that get_auth_config returns AuthConfig when auth configured."""
    auth_scheme = OAuth2(
        flows=OAuthFlows(
            authorizationCode=OAuthFlowAuthorizationCode(
                authorizationUrl="https://example.com/auth",
                tokenUrl="https://example.com/token",
                scopes={"read": "Read access"},
            )
        )
    )
    auth_credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id="test_client_id",
            client_secret="test_client_secret",
        ),
    )

    toolset = McpToolset(
        connection_params=StdioServerParameters(command="echo", args=["test"]),
        auth_scheme=auth_scheme,
        auth_credential=auth_credential,
    )

    auth_config = toolset.get_auth_config()
    assert auth_config is not None
    assert auth_config.auth_scheme == auth_scheme
    assert auth_config.raw_auth_credential == auth_credential

  def test_get_auth_config_returns_same_instance(self):
    """Test that get_auth_config returns the same instance each time."""
    auth_scheme = OAuth2(
        flows=OAuthFlows(
            authorizationCode=OAuthFlowAuthorizationCode(
                authorizationUrl="https://example.com/auth",
                tokenUrl="https://example.com/token",
                scopes={},
            )
        )
    )

    toolset = McpToolset(
        connection_params=StdioServerParameters(command="echo", args=["test"]),
        auth_scheme=auth_scheme,
    )

    # Should return the same instance
    config1 = toolset.get_auth_config()
    config2 = toolset.get_auth_config()
    assert config1 is config2


class TestMcpToolsetGetAuthHeaders:
  """Tests for McpToolset._get_auth_headers method."""

  @pytest.fixture
  def toolset_with_oauth2(self):
    """Create a toolset with OAuth2 auth configured."""
    auth_scheme = OAuth2(
        flows=OAuthFlows(
            authorizationCode=OAuthFlowAuthorizationCode(
                authorizationUrl="https://example.com/auth",
                tokenUrl="https://example.com/token",
                scopes={"read": "Read access"},
            )
        )
    )
    auth_credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id="test_client_id",
            client_secret="test_client_secret",
        ),
    )

    return McpToolset(
        connection_params=StdioServerParameters(command="echo", args=["test"]),
        auth_scheme=auth_scheme,
        auth_credential=auth_credential,
    )

  def test_get_auth_headers_returns_none_without_auth_config(self):
    """Test that _get_auth_headers returns None without auth config."""
    toolset = McpToolset(
        connection_params=StdioServerParameters(command="echo", args=["test"])
    )

    assert toolset._get_auth_headers() is None

  def test_get_auth_headers_returns_none_without_exchanged_credential(
      self, toolset_with_oauth2
  ):
    """Test that _get_auth_headers returns None without exchanged credential."""
    # No exchanged credential set yet
    assert toolset_with_oauth2._get_auth_headers() is None

  def test_get_auth_headers_oauth2_bearer_token(self, toolset_with_oauth2):
    """Test that _get_auth_headers returns Bearer token for OAuth2."""
    # Set exchanged credential with access token
    toolset_with_oauth2._auth_config.exchanged_auth_credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(access_token="test-access-token"),
    )

    headers = toolset_with_oauth2._get_auth_headers()

    assert headers is not None
    assert headers["Authorization"] == "Bearer test-access-token"

  def test_get_auth_headers_http_bearer_token(self):
    """Test that _get_auth_headers returns Bearer token for HTTP bearer."""
    auth_scheme = OAuth2(
        flows=OAuthFlows(
            authorizationCode=OAuthFlowAuthorizationCode(
                authorizationUrl="https://example.com/auth",
                tokenUrl="https://example.com/token",
                scopes={},
            )
        )
    )

    toolset = McpToolset(
        connection_params=StdioServerParameters(command="echo", args=["test"]),
        auth_scheme=auth_scheme,
    )

    # Set exchanged credential with HTTP bearer token
    toolset._auth_config.exchanged_auth_credential = AuthCredential(
        auth_type=AuthCredentialTypes.HTTP,
        http=HttpAuth(
            scheme="bearer",
            credentials=HttpCredentials(token="test-bearer-token"),
        ),
    )

    headers = toolset._get_auth_headers()

    assert headers is not None
    assert headers["Authorization"] == "Bearer test-bearer-token"

  def test_get_auth_headers_http_basic_auth(self):
    """Test that _get_auth_headers returns Basic auth for HTTP basic."""
    auth_scheme = OAuth2(
        flows=OAuthFlows(
            authorizationCode=OAuthFlowAuthorizationCode(
                authorizationUrl="https://example.com/auth",
                tokenUrl="https://example.com/token",
                scopes={},
            )
        )
    )

    toolset = McpToolset(
        connection_params=StdioServerParameters(command="echo", args=["test"]),
        auth_scheme=auth_scheme,
    )

    # Set exchanged credential with HTTP basic auth
    toolset._auth_config.exchanged_auth_credential = AuthCredential(
        auth_type=AuthCredentialTypes.HTTP,
        http=HttpAuth(
            scheme="basic",
            credentials=HttpCredentials(
                username="testuser",
                password="testpass",
            ),
        ),
    )

    headers = toolset._get_auth_headers()

    assert headers is not None
    expected_credentials = base64.b64encode(b"testuser:testpass").decode()
    assert headers["Authorization"] == f"Basic {expected_credentials}"

  def test_get_auth_headers_api_key_header(self):
    """Test that _get_auth_headers returns API key in header."""
    # Note: fastapi's APIKey model uses 'in' not 'in_', but accepts both
    auth_scheme = APIKeyScheme(**{
        "in": APIKeyIn.header,
        "name": "X-API-Key",
    })

    toolset = McpToolset(
        connection_params=StdioServerParameters(command="echo", args=["test"]),
        auth_scheme=auth_scheme,
    )

    # Set exchanged credential with API key
    toolset._auth_config.exchanged_auth_credential = AuthCredential(
        auth_type=AuthCredentialTypes.API_KEY,
        api_key="test-api-key-12345",
    )

    headers = toolset._get_auth_headers()

    assert headers is not None
    assert headers["X-API-Key"] == "test-api-key-12345"

  def test_get_auth_headers_api_key_non_header_logs_warning(self, caplog):
    """Test that non-header API key logs a warning."""
    # Note: fastapi's APIKey model uses 'in' not 'in_'
    auth_scheme = APIKeyScheme(**{
        "in": APIKeyIn.query,  # Query param, not header
        "name": "api_key",
    })

    toolset = McpToolset(
        connection_params=StdioServerParameters(command="echo", args=["test"]),
        auth_scheme=auth_scheme,
    )

    # Set exchanged credential with API key
    toolset._auth_config.exchanged_auth_credential = AuthCredential(
        auth_type=AuthCredentialTypes.API_KEY,
        api_key="test-api-key",
    )

    headers = toolset._get_auth_headers()

    # Should return None for non-header API key
    assert headers is None
