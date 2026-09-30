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

"""Tests for OAuth2CredentialExchanger."""

import copy
import time
from unittest.mock import MagicMock
from unittest.mock import patch

from authlib.common.errors import AuthlibBaseError
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_schemes import AuthSchemeType
from google.adk.auth.auth_schemes import OpenIdConnectWithConfig
from google.adk.tools.openapi_tool.auth.credential_exchangers import OAuth2CredentialExchanger
from google.adk.tools.openapi_tool.auth.credential_exchangers.base_credential_exchanger import AuthCredentialMissingError
import pytest
from requests.exceptions import ConnectionError as RequestsConnectionError

_CREATE_OAUTH2_SESSION = "google.adk.tools.openapi_tool.auth.credential_exchangers.oauth2_exchanger.create_oauth2_session"


@pytest.fixture
def oauth2_exchanger():
  return OAuth2CredentialExchanger()


@pytest.fixture
def auth_scheme():
  openid_config = OpenIdConnectWithConfig(
      type_=AuthSchemeType.openIdConnect,
      authorization_endpoint="https://example.com/auth",
      token_endpoint="https://example.com/token",
      scopes=["openid", "profile"],
  )
  return openid_config


def test_check_scheme_credential_type_success(oauth2_exchanger, auth_scheme):
  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id="test_client",
          client_secret="test_secret",
          redirect_uri="http://localhost:8080",
      ),
  )
  # Check that the method does not raise an exception
  oauth2_exchanger._check_scheme_credential_type(auth_scheme, auth_credential)


def test_check_scheme_credential_type_missing_credential(
    oauth2_exchanger, auth_scheme
):
  # Test case: auth_credential is None
  with pytest.raises(ValueError) as exc_info:
    oauth2_exchanger._check_scheme_credential_type(auth_scheme, None)
  assert "auth_credential is empty" in str(exc_info.value)


def test_check_scheme_credential_type_invalid_scheme_type(
    oauth2_exchanger, auth_scheme: OpenIdConnectWithConfig
):
  """Test case: Invalid AuthSchemeType."""
  # Test case: Invalid AuthSchemeType
  invalid_scheme = copy.deepcopy(auth_scheme)
  invalid_scheme.type_ = AuthSchemeType.apiKey
  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id="test_client",
          client_secret="test_secret",
          redirect_uri="http://localhost:8080",
      ),
  )
  with pytest.raises(ValueError) as exc_info:
    oauth2_exchanger._check_scheme_credential_type(
        invalid_scheme, auth_credential
    )
  assert "Invalid security scheme" in str(exc_info.value)


def test_check_scheme_credential_type_missing_openid_connect(
    oauth2_exchanger, auth_scheme
):
  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
  )
  with pytest.raises(ValueError) as exc_info:
    oauth2_exchanger._check_scheme_credential_type(auth_scheme, auth_credential)
  assert "auth_credential is not configured with oauth2" in str(exc_info.value)


def test_generate_auth_token_success(
    oauth2_exchanger, auth_scheme, monkeypatch
):
  """Test case: Successful generation of access token."""
  # Test case: Successful generation of access token
  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id="test_client",
          client_secret="test_secret",
          redirect_uri="http://localhost:8080",
          auth_response_uri="https://example.com/callback?code=test_code",
          access_token="test_access_token",
      ),
  )
  updated_credential = oauth2_exchanger.generate_auth_token(auth_credential)

  assert updated_credential.auth_type == AuthCredentialTypes.HTTP
  assert updated_credential.http.scheme == "bearer"
  assert updated_credential.http.credentials.token == "test_access_token"


def test_exchange_credential_generate_auth_token(
    oauth2_exchanger, auth_scheme, monkeypatch
):
  """Test exchange_credential when auth_response_uri is present."""
  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id="test_client",
          client_secret="test_secret",
          redirect_uri="http://localhost:8080",
          auth_response_uri="https://example.com/callback?code=test_code",
          access_token="test_access_token",
      ),
  )

  updated_credential = oauth2_exchanger.exchange_credential(
      auth_scheme, auth_credential
  )

  assert updated_credential.auth_type == AuthCredentialTypes.HTTP
  assert updated_credential.http.scheme == "bearer"
  assert updated_credential.http.credentials.token == "test_access_token"


def test_exchange_credential_auth_missing(oauth2_exchanger, auth_scheme):
  """Test exchange_credential when auth_credential is missing."""
  with pytest.raises(ValueError) as exc_info:
    oauth2_exchanger.exchange_credential(auth_scheme, None)
  assert "auth_credential is empty. Please create AuthCredential using" in str(
      exc_info.value
  )


def _oauth2_credential(*, access_token, refresh_token=None, expires_at=None):
  return AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id="test_client",
          client_secret="test_secret",
          redirect_uri="http://localhost:8080",
          access_token=access_token,
          refresh_token=refresh_token,
          expires_at=expires_at,
      ),
  )


def test_exchange_credential_refreshes_expired_token(
    oauth2_exchanger, auth_scheme
):
  """Expired access token + refresh token -> returns the refreshed token."""
  auth_credential = _oauth2_credential(
      access_token="stale_access_token",
      refresh_token="test_refresh_token",
      expires_at=int(time.time()) - 3600,
  )

  mock_client = MagicMock()
  mock_client.refresh_token.return_value = {
      "access_token": "refreshed_access_token",
      "refresh_token": "refreshed_refresh_token",
      "expires_at": int(time.time()) + 3600,
      "expires_in": 3600,
  }

  with patch(_CREATE_OAUTH2_SESSION) as mock_create_session:
    mock_create_session.return_value = (
        mock_client,
        "https://example.com/token",
    )
    updated_credential = oauth2_exchanger.exchange_credential(
        auth_scheme, auth_credential
    )

  mock_client.refresh_token.assert_called_once_with(
      url="https://example.com/token",
      refresh_token="test_refresh_token",
  )
  assert updated_credential.auth_type == AuthCredentialTypes.HTTP
  assert updated_credential.http.scheme == "bearer"
  assert updated_credential.http.credentials.token == "refreshed_access_token"


@pytest.mark.parametrize(
    "error",
    [
        AuthlibBaseError("refresh failed"),
        RequestsConnectionError("network down"),
    ],
)
def test_exchange_credential_refresh_failure_falls_back(
    oauth2_exchanger, auth_scheme, error
):
  """A caught OAuth/transport error -> falls back to the existing token."""
  auth_credential = _oauth2_credential(
      access_token="stale_access_token",
      refresh_token="test_refresh_token",
      expires_at=int(time.time()) - 3600,
  )

  mock_client = MagicMock()
  mock_client.refresh_token.side_effect = error

  with patch(_CREATE_OAUTH2_SESSION) as mock_create_session:
    mock_create_session.return_value = (
        mock_client,
        "https://example.com/token",
    )
    updated_credential = oauth2_exchanger.exchange_credential(
        auth_scheme, auth_credential
    )

  assert updated_credential.auth_type == AuthCredentialTypes.HTTP
  assert updated_credential.http.scheme == "bearer"
  assert updated_credential.http.credentials.token == "stale_access_token"


def test_exchange_credential_unexpected_error_propagates(
    oauth2_exchanger, auth_scheme
):
  """Errors outside the caught families are not swallowed."""
  auth_credential = _oauth2_credential(
      access_token="stale_access_token",
      refresh_token="test_refresh_token",
      expires_at=int(time.time()) - 3600,
  )

  mock_client = MagicMock()
  mock_client.refresh_token.side_effect = ValueError("unexpected")

  with patch(_CREATE_OAUTH2_SESSION) as mock_create_session:
    mock_create_session.return_value = (
        mock_client,
        "https://example.com/token",
    )
    with pytest.raises(ValueError):
      oauth2_exchanger.exchange_credential(auth_scheme, auth_credential)


def test_exchange_credential_not_expired_no_refresh(
    oauth2_exchanger, auth_scheme
):
  """A valid (unexpired) access token is wrapped as-is, no refresh attempted."""
  auth_credential = _oauth2_credential(
      access_token="valid_access_token",
      refresh_token="test_refresh_token",
      expires_at=int(time.time()) + 3600,
  )

  with patch(_CREATE_OAUTH2_SESSION) as mock_create_session:
    updated_credential = oauth2_exchanger.exchange_credential(
        auth_scheme, auth_credential
    )

  mock_create_session.assert_not_called()
  assert updated_credential.auth_type == AuthCredentialTypes.HTTP
  assert updated_credential.http.credentials.token == "valid_access_token"


def test_exchange_credential_expired_no_refresh_token_no_refresh(
    oauth2_exchanger, auth_scheme
):
  """An expired token without a refresh token is wrapped as-is (unchanged)."""
  auth_credential = _oauth2_credential(
      access_token="stale_access_token",
      refresh_token=None,
      expires_at=int(time.time()) - 3600,
  )

  with patch(_CREATE_OAUTH2_SESSION) as mock_create_session:
    updated_credential = oauth2_exchanger.exchange_credential(
        auth_scheme, auth_credential
    )

  mock_create_session.assert_not_called()
  assert updated_credential.auth_type == AuthCredentialTypes.HTTP
  assert updated_credential.http.credentials.token == "stale_access_token"
