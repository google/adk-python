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

"""Tests for OAuth2CredentialExchanger."""

import copy
from unittest.mock import MagicMock

from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_schemes import AuthSchemeType
from google.adk.auth.auth_schemes import OpenIdConnectWithConfig
from google.adk.tools.openapi_tool.auth.credential_exchangers import OAuth2CredentialExchanger
from google.adk.tools.openapi_tool.auth.credential_exchangers.base_credential_exchanger import AuthCredentialMissingError
import pytest


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
