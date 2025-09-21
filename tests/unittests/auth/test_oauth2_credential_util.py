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

import time
from unittest.mock import Mock

from authlib.oauth2.rfc6749 import OAuth2Token
from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlowAuthorizationCode
from fastapi.openapi.models import OAuthFlows
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_schemes import OpenIdConnectWithConfig
from google.adk.auth.oauth2_credential_util import create_oauth2_session
from google.adk.auth.oauth2_credential_util import update_credential_with_tokens


class TestOAuth2CredentialUtil:
  """Test suite for OAuth2 credential utility functions."""

  def test_create_oauth2_session_openid_connect(self):
    """Test create_oauth2_session with OpenID Connect scheme."""
    scheme = OpenIdConnectWithConfig(
        type_="openIdConnect",
        openId_connect_url=(
            "https://example.com/.well-known/openid_configuration"
        ),
        authorization_endpoint="https://example.com/auth",
        token_endpoint="https://example.com/token",
        scopes=["openid", "profile"],
    )
    credential = AuthCredential(
        auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
        oauth2=OAuth2Auth(
            client_id="test_client_id",
            client_secret="test_client_secret",
            redirect_uri="https://example.com/callback",
            state="test_state",
        ),
    )

    client, token_endpoint = create_oauth2_session(scheme, credential)

    assert client is not None
    assert token_endpoint == "https://example.com/token"
    assert client.client_id == "test_client_id"
    assert client.client_secret == "test_client_secret"

  def test_create_oauth2_session_oauth2_scheme(self):
    """Test create_oauth2_session with OAuth2 scheme."""
    flows = OAuthFlows(
        authorizationCode=OAuthFlowAuthorizationCode(
            authorizationUrl="https://example.com/auth",
            tokenUrl="https://example.com/token",
            scopes={"read": "Read access", "write": "Write access"},
        )
    )
    scheme = OAuth2(type_="oauth2", flows=flows)
    credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id="test_client_id",
            client_secret="test_client_secret",
            redirect_uri="https://example.com/callback",
        ),
    )

    client, token_endpoint = create_oauth2_session(scheme, credential)

    assert client is not None
    assert token_endpoint == "https://example.com/token"

  def test_create_oauth2_session_invalid_scheme(self):
    """Test create_oauth2_session with invalid scheme."""
    scheme = Mock()  # Invalid scheme type
    credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id="test_client_id",
            client_secret="test_client_secret",
        ),
    )

    client, token_endpoint = create_oauth2_session(scheme, credential)

    assert client is None
    assert token_endpoint is None

  def test_create_oauth2_session_missing_credentials(self):
    """Test create_oauth2_session with missing credentials."""
    scheme = OpenIdConnectWithConfig(
        type_="openIdConnect",
        openId_connect_url=(
            "https://example.com/.well-known/openid_configuration"
        ),
        authorization_endpoint="https://example.com/auth",
        token_endpoint="https://example.com/token",
        scopes=["openid"],
    )
    credential = AuthCredential(
        auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
        oauth2=OAuth2Auth(
            client_id="test_client_id",
            # Missing client_secret
        ),
    )

    client, token_endpoint = create_oauth2_session(scheme, credential)

    assert client is None
    assert token_endpoint is None

  def test_update_credential_with_tokens(self):
    """Test update_credential_with_tokens function."""
    credential = AuthCredential(
        auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
        oauth2=OAuth2Auth(
            client_id="test_client_id",
            client_secret="test_client_secret",
        ),
    )

    # Store the expected expiry time to avoid timing issues
    expected_expires_at = int(time.time()) + 3600
    tokens = OAuth2Token({
        "access_token": "new_access_token",
        "refresh_token": "new_refresh_token",
        "expires_at": expected_expires_at,
        "expires_in": 3600,
    })

    update_credential_with_tokens(credential, tokens)

    assert credential.oauth2.access_token == "new_access_token"
    assert credential.oauth2.refresh_token == "new_refresh_token"
    assert credential.oauth2.expires_at == expected_expires_at
    assert credential.oauth2.expires_in == 3600
