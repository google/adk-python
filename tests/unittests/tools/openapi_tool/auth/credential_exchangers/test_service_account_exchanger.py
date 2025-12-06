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

"""Unit tests for the service account credential exchanger."""

from unittest.mock import MagicMock

from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import ServiceAccount
from google.adk.auth.auth_credential import ServiceAccountCredential
from google.adk.auth.auth_schemes import AuthScheme
from google.adk.auth.auth_schemes import AuthSchemeType
from google.adk.tools.openapi_tool.auth.credential_exchangers.base_credential_exchanger import AuthCredentialMissingError
from google.adk.tools.openapi_tool.auth.credential_exchangers.service_account_exchanger import ServiceAccountCredentialExchanger
import google.auth
import pytest


@pytest.fixture
def service_account_exchanger():
  return ServiceAccountCredentialExchanger()


@pytest.fixture
def auth_scheme():
  scheme = MagicMock(spec=AuthScheme)
  scheme.type_ = AuthSchemeType.oauth2
  scheme.description = "Google Service Account"
  return scheme


def test_exchange_credential_success(
    service_account_exchanger, auth_scheme, monkeypatch
):
  """Test successful exchange of service account credentials."""
  mock_credentials = MagicMock()
  mock_credentials.token = "mock_access_token"

  # Mock the from_service_account_info method
  mock_from_service_account_info = MagicMock(return_value=mock_credentials)
  target_path = (
      "google.adk.tools.openapi_tool.auth.credential_exchangers."
      "service_account_exchanger.service_account.Credentials."
      "from_service_account_info"
  )
  monkeypatch.setattr(
      target_path,
      mock_from_service_account_info,
  )

  # Mock the refresh method
  mock_credentials.refresh = MagicMock()

  # Create a valid AuthCredential with service account info
  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
      service_account=ServiceAccount(
          service_account_credential=ServiceAccountCredential(
              type_="service_account",
              project_id="your_project_id",
              private_key_id="your_private_key_id",
              private_key="-----BEGIN PRIVATE KEY-----...",
              client_email="...@....iam.gserviceaccount.com",
              client_id="your_client_id",
              auth_uri="https://accounts.google.com/o/oauth2/auth",
              token_uri="https://oauth2.googleapis.com/token",
              auth_provider_x509_cert_url=(
                  "https://www.googleapis.com/oauth2/v1/certs"
              ),
              client_x509_cert_url=(
                  "https://www.googleapis.com/robot/v1/metadata/x509/..."
              ),
              universe_domain="googleapis.com",
          ),
          scopes=["https://www.googleapis.com/auth/cloud-platform"],
      ),
  )

  result = service_account_exchanger.exchange_credential(
      auth_scheme, auth_credential
  )

  assert result.auth_type == AuthCredentialTypes.HTTP
  assert result.http.scheme == "bearer"
  assert result.http.credentials.token == "mock_access_token"
  mock_from_service_account_info.assert_called_once()
  mock_credentials.refresh.assert_called_once()


def test_exchange_credential_use_default_credential_success(
    service_account_exchanger, auth_scheme, monkeypatch
):
  """Test successful exchange of service account credentials using default credential."""
  mock_credentials = MagicMock()
  mock_credentials.token = "mock_access_token"
  mock_google_auth_default = MagicMock(
      return_value=(mock_credentials, "test_project")
  )
  monkeypatch.setattr(google.auth, "default", mock_google_auth_default)

  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
      service_account=ServiceAccount(
          use_default_credential=True,
          scopes=["https://www.googleapis.com/auth/cloud-platform"],
      ),
  )

  result = service_account_exchanger.exchange_credential(
      auth_scheme, auth_credential
  )

  assert result.auth_type == AuthCredentialTypes.HTTP
  assert result.http.scheme == "bearer"
  assert result.http.credentials.token == "mock_access_token"
  # Verify google.auth.default is called with the correct scopes parameter
  mock_google_auth_default.assert_called_once_with(
      scopes=["https://www.googleapis.com/auth/cloud-platform"]
  )
  mock_credentials.refresh.assert_called_once()


def test_exchange_credential_missing_auth_credential(
    service_account_exchanger, auth_scheme
):
  """Test missing auth credential during exchange."""
  with pytest.raises(AuthCredentialMissingError) as exc_info:
    service_account_exchanger.exchange_credential(auth_scheme, None)
  assert "Service account credentials are missing" in str(exc_info.value)


def test_exchange_credential_missing_service_account_info(
    service_account_exchanger, auth_scheme
):
  """Test missing service account info during exchange."""
  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
  )
  with pytest.raises(AuthCredentialMissingError) as exc_info:
    service_account_exchanger.exchange_credential(auth_scheme, auth_credential)
  assert "Service account credentials are missing" in str(exc_info.value)


def test_exchange_credential_exchange_failure(
    service_account_exchanger, auth_scheme, monkeypatch
):
  """Test failure during service account token exchange."""
  mock_from_service_account_info = MagicMock(
      side_effect=Exception("Failed to load credentials")
  )
  target_path = (
      "google.adk.tools.openapi_tool.auth.credential_exchangers."
      "service_account_exchanger.service_account.Credentials."
      "from_service_account_info"
  )
  monkeypatch.setattr(
      target_path,
      mock_from_service_account_info,
  )

  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
      service_account=ServiceAccount(
          service_account_credential=ServiceAccountCredential(
              type_="service_account",
              project_id="your_project_id",
              private_key_id="your_private_key_id",
              private_key="-----BEGIN PRIVATE KEY-----...",
              client_email="...@....iam.gserviceaccount.com",
              client_id="your_client_id",
              auth_uri="https://accounts.google.com/o/oauth2/auth",
              token_uri="https://oauth2.googleapis.com/token",
              auth_provider_x509_cert_url=(
                  "https://www.googleapis.com/oauth2/v1/certs"
              ),
              client_x509_cert_url=(
                  "https://www.googleapis.com/robot/v1/metadata/x509/..."
              ),
              universe_domain="googleapis.com",
          ),
          scopes=["https://www.googleapis.com/auth/cloud-platform"],
      ),
  )
  with pytest.raises(AuthCredentialMissingError) as exc_info:
    service_account_exchanger.exchange_credential(auth_scheme, auth_credential)
  assert "Failed to exchange service account token" in str(exc_info.value)
  mock_from_service_account_info.assert_called_once()
