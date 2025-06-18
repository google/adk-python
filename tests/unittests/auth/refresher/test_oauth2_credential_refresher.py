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
from unittest.mock import patch

from authlib.oauth2.rfc6749 import OAuth2Token
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_schemes import OpenIdConnectWithConfig
from google.adk.auth.refresher.oauth2_credential_refresher import OAuth2CredentialRefresher
import pytest


class TestOAuth2CredentialRefresher:
  """Test suite for OAuth2CredentialRefresher."""

  @patch("google.adk.auth.refresher.oauth2_credential_refresher.OAuth2Token")
  @pytest.mark.asyncio
  async def test_needs_refresh_token_not_expired(self, mock_oauth2_token):
    """Test needs_refresh when token is not expired."""
    mock_token_instance = Mock()
    mock_token_instance.is_expired.return_value = False
    mock_oauth2_token.return_value = mock_token_instance

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
            client_secret="test_client_secret",
            access_token="existing_token",
            expires_at=int(time.time()) + 3600,
        ),
    )

    refresher = OAuth2CredentialRefresher()
    needs_refresh = await refresher.is_refresh_needed(credential, scheme)

    assert not needs_refresh

  @patch("google.adk.auth.refresher.oauth2_credential_refresher.OAuth2Token")
  @pytest.mark.asyncio
  async def test_needs_refresh_token_expired(self, mock_oauth2_token):
    """Test needs_refresh when token is expired."""
    mock_token_instance = Mock()
    mock_token_instance.is_expired.return_value = True
    mock_oauth2_token.return_value = mock_token_instance

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
            client_secret="test_client_secret",
            access_token="existing_token",
            expires_at=int(time.time()) - 3600,  # Expired
        ),
    )

    refresher = OAuth2CredentialRefresher()
    needs_refresh = await refresher.is_refresh_needed(credential, scheme)

    assert needs_refresh

  @patch("google.adk.auth.oauth2_credential_util.OAuth2Session")
  @patch("google.adk.auth.oauth2_credential_util.OAuth2Token")
  @pytest.mark.asyncio
  async def test_refresh_token_expired_success(
      self, mock_oauth2_token, mock_oauth2_session
  ):
    """Test successful token refresh when token is expired."""
    # Setup mock token
    mock_token_instance = Mock()
    mock_token_instance.is_expired.return_value = True
    mock_oauth2_token.return_value = mock_token_instance

    # Setup mock session
    mock_client = Mock()
    mock_oauth2_session.return_value = mock_client
    mock_tokens = OAuth2Token({
        "access_token": "refreshed_access_token",
        "refresh_token": "refreshed_refresh_token",
        "expires_at": int(time.time()) + 3600,
        "expires_in": 3600,
    })
    mock_client.refresh_token.return_value = mock_tokens

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
            client_secret="test_client_secret",
            access_token="old_token",
            refresh_token="old_refresh_token",
            expires_at=int(time.time()) - 3600,  # Expired
        ),
    )

    refresher = OAuth2CredentialRefresher()
    result = await refresher.refresh(credential, scheme)

    # Verify token refresh was successful
    assert result.oauth2.access_token == "refreshed_access_token"
    assert result.oauth2.refresh_token == "refreshed_refresh_token"
    mock_client.refresh_token.assert_called_once()

  @pytest.mark.asyncio
  async def test_refresh_no_oauth2_credential(self):
    """Test refresh with no OAuth2 credential returns original."""
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
        # No oauth2 field
    )

    refresher = OAuth2CredentialRefresher()
    result = await refresher.refresh(credential, scheme)

    assert result == credential

  @pytest.mark.asyncio
  async def test_needs_refresh_google_oauth2_json_expired(self):
    """Test needs_refresh with Google OAuth2 JSON credential that is expired."""
    import json
    from unittest.mock import patch

    # Mock Google OAuth2 JSON credential data
    google_oauth2_json = json.dumps({
        "client_id": "test_client_id",
        "client_secret": "test_client_secret",
        "refresh_token": "test_refresh_token",
        "type": "authorized_user",
    })

    credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        google_oauth2_json=google_oauth2_json,
    )

    # Mock the Google Credentials class
    with patch(
        "google.adk.auth.refresher.oauth2_credential_refresher.Credentials"
    ) as mock_credentials:
      mock_google_credential = Mock()
      mock_google_credential.expired = True
      mock_google_credential.refresh_token = "test_refresh_token"
      mock_credentials.from_authorized_user_info.return_value = (
          mock_google_credential
      )

      refresher = OAuth2CredentialRefresher()
      needs_refresh = await refresher.is_refresh_needed(credential, None)

      assert needs_refresh

  @pytest.mark.asyncio
  async def test_needs_refresh_google_oauth2_json_not_expired(self):
    """Test needs_refresh with Google OAuth2 JSON credential that is not expired."""
    import json
    from unittest.mock import patch

    # Mock Google OAuth2 JSON credential data
    google_oauth2_json = json.dumps({
        "client_id": "test_client_id",
        "client_secret": "test_client_secret",
        "refresh_token": "test_refresh_token",
        "type": "authorized_user",
    })

    credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        google_oauth2_json=google_oauth2_json,
    )

    # Mock the Google Credentials class
    with patch(
        "google.adk.auth.refresher.oauth2_credential_refresher.Credentials"
    ) as mock_credentials:
      mock_google_credential = Mock()
      mock_google_credential.expired = False
      mock_google_credential.refresh_token = "test_refresh_token"
      mock_credentials.from_authorized_user_info.return_value = (
          mock_google_credential
      )

      refresher = OAuth2CredentialRefresher()
      needs_refresh = await refresher.is_refresh_needed(credential, None)

      assert not needs_refresh

  @pytest.mark.asyncio
  async def test_refresh_google_oauth2_json_success(self):
    """Test successful refresh of Google OAuth2 JSON credential."""
    import json
    from unittest.mock import patch

    # Mock Google OAuth2 JSON credential data
    google_oauth2_json = json.dumps({
        "client_id": "test_client_id",
        "client_secret": "test_client_secret",
        "refresh_token": "test_refresh_token",
        "type": "authorized_user",
    })

    credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        google_oauth2_json=google_oauth2_json,
    )

    # Mock the Google Credentials and Request classes
    with patch(
        "google.adk.auth.refresher.oauth2_credential_refresher.Credentials"
    ) as mock_credentials:
      with patch(
          "google.adk.auth.refresher.oauth2_credential_refresher.Request"
      ) as mock_request:
        mock_google_credential = Mock()
        mock_google_credential.expired = True
        mock_google_credential.refresh_token = "test_refresh_token"
        mock_google_credential.to_json.return_value = json.dumps({
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "refresh_token": "new_refresh_token",
            "access_token": "new_access_token",
            "type": "authorized_user",
        })
        mock_credentials.from_authorized_user_info.return_value = (
            mock_google_credential
        )

        refresher = OAuth2CredentialRefresher()
        result = await refresher.refresh(credential, None)

        mock_google_credential.refresh.assert_called_once()
        assert (
            result.google_oauth2_json != google_oauth2_json
        )  # Should be updated

  @pytest.mark.asyncio
  async def test_needs_refresh_no_oauth2_credential(self):
    """Test needs_refresh with no OAuth2 credential returns False."""
    credential = AuthCredential(
        auth_type=AuthCredentialTypes.HTTP,
        # No oauth2 field
    )

    refresher = OAuth2CredentialRefresher()
    needs_refresh = await refresher.is_refresh_needed(credential, None)

    assert not needs_refresh
