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

"""Unit tests for the ServiceAccountCredentialExchanger."""

from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import ServiceAccount
from google.adk.auth.auth_credential import ServiceAccountCredential
from google.adk.auth.service_account_credential_exchanger import ServiceAccountCredentialExchanger
import pytest


class TestServiceAccountCredentialExchanger:
  """Test cases for ServiceAccountCredentialExchanger."""

  def test_init_valid_credential(self):
    """Test successful initialization with valid service account credential."""
    credential = AuthCredential(
        auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
        service_account=ServiceAccount(
            service_account_credential=ServiceAccountCredential(
                type_="service_account",
                project_id="test-project",
                private_key_id="key-id",
                private_key=(
                    "-----BEGIN PRIVATE KEY-----\nMOCK_KEY\n-----END PRIVATE"
                    " KEY-----"
                ),
                client_email="test@test-project.iam.gserviceaccount.com",
                client_id="12345",
                auth_uri="https://accounts.google.com/o/oauth2/auth",
                token_uri="https://oauth2.googleapis.com/token",
                auth_provider_x509_cert_url=(
                    "https://www.googleapis.com/oauth2/v1/certs"
                ),
                client_x509_cert_url="https://www.googleapis.com/robot/v1/metadata/x509/test%40test-project.iam.gserviceaccount.com",
                universe_domain="googleapis.com",
            ),
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        ),
    )

    exchanger = ServiceAccountCredentialExchanger(credential)
    assert exchanger._credential == credential

  def test_init_invalid_credential_type(self):
    """Test initialization with invalid credential type raises ValueError."""
    credential = AuthCredential(
        auth_type=AuthCredentialTypes.API_KEY,
        api_key="test-key",
    )

    with pytest.raises(
        ValueError, match="Credential is not a service account credential"
    ):
      ServiceAccountCredentialExchanger(credential)

  @patch(
      "google.adk.auth.service_account_credential_exchanger.service_account.Credentials.from_service_account_info"
  )
  @patch("google.adk.auth.service_account_credential_exchanger.Request")
  def test_exchange_with_explicit_credentials_success(
      self, mock_request_class, mock_from_service_account_info
  ):
    """Test successful exchange with explicit service account credentials."""
    # Setup mocks
    mock_request = MagicMock()
    mock_request_class.return_value = mock_request

    mock_credentials = MagicMock()
    mock_credentials.token = "mock_access_token"
    mock_from_service_account_info.return_value = mock_credentials

    # Create test credential
    service_account_cred = ServiceAccountCredential(
        type_="service_account",
        project_id="test-project",
        private_key_id="key-id",
        private_key=(
            "-----BEGIN PRIVATE KEY-----\nMOCK_KEY\n-----END PRIVATE KEY-----"
        ),
        client_email="test@test-project.iam.gserviceaccount.com",
        client_id="12345",
        auth_uri="https://accounts.google.com/o/oauth2/auth",
        token_uri="https://oauth2.googleapis.com/token",
        auth_provider_x509_cert_url=(
            "https://www.googleapis.com/oauth2/v1/certs"
        ),
        client_x509_cert_url="https://www.googleapis.com/robot/v1/metadata/x509/test%40test-project.iam.gserviceaccount.com",
        universe_domain="googleapis.com",
    )

    credential = AuthCredential(
        auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
        service_account=ServiceAccount(
            service_account_credential=service_account_cred,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        ),
    )

    exchanger = ServiceAccountCredentialExchanger(credential)
    result = exchanger.exchange()

    # Verify the result
    assert result.auth_type == AuthCredentialTypes.HTTP
    assert result.http.scheme == "bearer"
    assert result.http.credentials.token == "mock_access_token"

    # Verify mocks were called correctly
    mock_from_service_account_info.assert_called_once_with(
        service_account_cred.model_dump(),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    mock_credentials.refresh.assert_called_once_with(mock_request)

  @patch(
      "google.adk.auth.service_account_credential_exchanger.google.auth.default"
  )
  @patch("google.adk.auth.service_account_credential_exchanger.Request")
  def test_exchange_with_default_credentials_success(
      self, mock_request_class, mock_google_auth_default
  ):
    """Test successful exchange with default application credentials."""
    # Setup mocks
    mock_request = MagicMock()
    mock_request_class.return_value = mock_request

    mock_credentials = MagicMock()
    mock_credentials.token = "default_access_token"
    mock_google_auth_default.return_value = (mock_credentials, "test-project")

    # Create test credential with use_default_credential=True
    credential = AuthCredential(
        auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
        service_account=ServiceAccount(
            use_default_credential=True,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        ),
    )

    exchanger = ServiceAccountCredentialExchanger(credential)
    result = exchanger.exchange()

    # Verify the result
    assert result.auth_type == AuthCredentialTypes.HTTP
    assert result.http.scheme == "bearer"
    assert result.http.credentials.token == "default_access_token"

    # Verify mocks were called correctly
    mock_google_auth_default.assert_called_once()
    mock_credentials.refresh.assert_called_once_with(mock_request)

  def test_exchange_missing_service_account(self):
    """Test exchange fails when service_account is None."""
    credential = AuthCredential(
        auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
        service_account=None,
    )

    exchanger = ServiceAccountCredentialExchanger(credential)

    with pytest.raises(
        ValueError, match="Service account credentials are missing"
    ):
      exchanger.exchange()

  def test_exchange_missing_credentials_and_not_default(self):
    """Test exchange fails when credentials are missing and use_default_credential is False."""
    credential = AuthCredential(
        auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
        service_account=ServiceAccount(
            service_account_credential=None,
            use_default_credential=False,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        ),
    )

    exchanger = ServiceAccountCredentialExchanger(credential)

    with pytest.raises(
        ValueError, match="Service account credentials are missing"
    ):
      exchanger.exchange()

  @patch(
      "google.adk.auth.service_account_credential_exchanger.service_account.Credentials.from_service_account_info"
  )
  def test_exchange_credential_creation_failure(
      self, mock_from_service_account_info
  ):
    """Test exchange handles credential creation failure gracefully."""
    # Setup mock to raise exception
    mock_from_service_account_info.side_effect = Exception(
        "Invalid private key"
    )

    # Create test credential
    service_account_cred = ServiceAccountCredential(
        type_="service_account",
        project_id="test-project",
        private_key_id="key-id",
        private_key="invalid-key",
        client_email="test@test-project.iam.gserviceaccount.com",
        client_id="12345",
        auth_uri="https://accounts.google.com/o/oauth2/auth",
        token_uri="https://oauth2.googleapis.com/token",
        auth_provider_x509_cert_url=(
            "https://www.googleapis.com/oauth2/v1/certs"
        ),
        client_x509_cert_url="https://www.googleapis.com/robot/v1/metadata/x509/test%40test-project.iam.gserviceaccount.com",
        universe_domain="googleapis.com",
    )

    credential = AuthCredential(
        auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
        service_account=ServiceAccount(
            service_account_credential=service_account_cred,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        ),
    )

    exchanger = ServiceAccountCredentialExchanger(credential)

    with pytest.raises(
        ValueError, match="Failed to exchange service account token"
    ):
      exchanger.exchange()

  @patch(
      "google.adk.auth.service_account_credential_exchanger.google.auth.default"
  )
  def test_exchange_default_credential_failure(self, mock_google_auth_default):
    """Test exchange handles default credential failure gracefully."""
    # Setup mock to raise exception
    mock_google_auth_default.side_effect = Exception(
        "No default credentials found"
    )

    # Create test credential with use_default_credential=True
    credential = AuthCredential(
        auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
        service_account=ServiceAccount(
            use_default_credential=True,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        ),
    )

    exchanger = ServiceAccountCredentialExchanger(credential)

    with pytest.raises(
        ValueError, match="Failed to exchange service account token"
    ):
      exchanger.exchange()

  @patch(
      "google.adk.auth.service_account_credential_exchanger.service_account.Credentials.from_service_account_info"
  )
  @patch("google.adk.auth.service_account_credential_exchanger.Request")
  def test_exchange_refresh_failure(
      self, mock_request_class, mock_from_service_account_info
  ):
    """Test exchange handles credential refresh failure gracefully."""
    # Setup mocks
    mock_request = MagicMock()
    mock_request_class.return_value = mock_request

    mock_credentials = MagicMock()
    mock_credentials.refresh.side_effect = Exception(
        "Network error during refresh"
    )
    mock_from_service_account_info.return_value = mock_credentials

    # Create test credential
    service_account_cred = ServiceAccountCredential(
        type_="service_account",
        project_id="test-project",
        private_key_id="key-id",
        private_key=(
            "-----BEGIN PRIVATE KEY-----\nMOCK_KEY\n-----END PRIVATE KEY-----"
        ),
        client_email="test@test-project.iam.gserviceaccount.com",
        client_id="12345",
        auth_uri="https://accounts.google.com/o/oauth2/auth",
        token_uri="https://oauth2.googleapis.com/token",
        auth_provider_x509_cert_url=(
            "https://www.googleapis.com/oauth2/v1/certs"
        ),
        client_x509_cert_url="https://www.googleapis.com/robot/v1/metadata/x509/test%40test-project.iam.gserviceaccount.com",
        universe_domain="googleapis.com",
    )

    credential = AuthCredential(
        auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
        service_account=ServiceAccount(
            service_account_credential=service_account_cred,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        ),
    )

    exchanger = ServiceAccountCredentialExchanger(credential)

    with pytest.raises(
        ValueError, match="Failed to exchange service account token"
    ):
      exchanger.exchange()

  def test_exchange_none_credential_in_constructor(self):
    """Test that passing None credential raises appropriate error during construction."""
    # This test verifies behavior when _credential is None, though this shouldn't
    # happen in normal usage due to constructor validation
    credential = AuthCredential(
        auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
        service_account=ServiceAccount(
            use_default_credential=True,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        ),
    )

    exchanger = ServiceAccountCredentialExchanger(credential)
    # Manually set to None to test the validation logic
    exchanger._credential = None

    with pytest.raises(
        ValueError, match="Service account credentials are missing"
    ):
      exchanger.exchange()
