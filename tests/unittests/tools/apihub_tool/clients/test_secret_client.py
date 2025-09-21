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

"""Unit tests for the SecretManagerClient."""

import json
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.tools.apihub_tool.clients.secret_client import SecretManagerClient
import pytest

import google


class TestSecretManagerClient:
  """Tests for the SecretManagerClient class."""

  @patch("google.cloud.secretmanager.SecretManagerServiceClient")
  @patch(
      "google.adk.tools.apihub_tool.clients.secret_client.default_service_credential"
  )
  def test_init_with_default_credentials(
      self, mock_default_service_credential, mock_secret_manager_client
  ):
    """Test initialization with default credentials."""
    # Setup
    mock_credentials = MagicMock()
    mock_default_service_credential.return_value = (
        mock_credentials,
        "test-project",
    )

    # Execute
    client = SecretManagerClient()

    # Verify
    mock_default_service_credential.assert_called_once_with(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    mock_secret_manager_client.assert_called_once_with(
        credentials=mock_credentials
    )
    assert client._credentials == mock_credentials
    assert client._client == mock_secret_manager_client.return_value

  @patch("google.cloud.secretmanager.SecretManagerServiceClient")
  @patch("google.oauth2.service_account.Credentials.from_service_account_info")
  def test_init_with_service_account_json(
      self, mock_from_service_account_info, mock_secret_manager_client
  ):
    """Test initialization with service account JSON."""
    # Setup
    mock_credentials = MagicMock()
    mock_from_service_account_info.return_value = mock_credentials
    service_account_json = json.dumps({
        "type": "service_account",
        "project_id": "test-project",
        "private_key_id": "key-id",
        "private_key": "private-key",
        "client_email": "test@example.com",
    })

    # Execute
    client = SecretManagerClient(service_account_json=service_account_json)

    # Verify
    mock_from_service_account_info.assert_called_once_with(
        json.loads(service_account_json)
    )
    mock_secret_manager_client.assert_called_once_with(
        credentials=mock_credentials
    )
    assert client._credentials == mock_credentials
    assert client._client == mock_secret_manager_client.return_value

  @patch("google.cloud.secretmanager.SecretManagerServiceClient")
  def test_init_with_auth_token(self, mock_secret_manager_client):
    """Test initialization with auth token."""
    # Setup
    auth_token = "test-token"
    mock_credentials = MagicMock()

    # Mock the entire credentials creation process
    with (
        patch("google.auth.credentials.Credentials") as mock_credentials_class,
        patch("google.auth.transport.requests.Request") as mock_request,
    ):
      # Configure the mock to return our mock_credentials when instantiated
      mock_credentials_class.return_value = mock_credentials

      # Execute
      client = SecretManagerClient(auth_token=auth_token)

      # Verify
      mock_credentials.refresh.assert_called_once()
      mock_secret_manager_client.assert_called_once_with(
          credentials=mock_credentials
      )
      assert client._credentials == mock_credentials
      assert client._client == mock_secret_manager_client.return_value

  @patch(
      "google.adk.tools.apihub_tool.clients.secret_client.default_service_credential"
  )
  def test_init_with_default_credentials_error(
      self, mock_default_service_credential
  ):
    """Test initialization with default credentials that fails."""
    # Setup
    mock_default_service_credential.side_effect = Exception("Auth error")

    # Execute and verify
    with pytest.raises(
        ValueError,
        match="error occurred while trying to use default credentials",
    ):
      SecretManagerClient()

  def test_init_with_invalid_service_account_json(self):
    """Test initialization with invalid service account JSON."""
    # Execute and verify
    with pytest.raises(ValueError, match="Invalid service account JSON"):
      SecretManagerClient(service_account_json="invalid-json")

  @patch("google.cloud.secretmanager.SecretManagerServiceClient")
  @patch(
      "google.adk.tools.apihub_tool.clients.secret_client.default_service_credential"
  )
  def test_get_secret(
      self, mock_default_service_credential, mock_secret_manager_client
  ):
    """Test getting a secret."""
    # Setup
    mock_credentials = MagicMock()
    mock_default_service_credential.return_value = (
        mock_credentials,
        "test-project",
    )

    mock_client = MagicMock()
    mock_secret_manager_client.return_value = mock_client
    mock_response = MagicMock()
    mock_response.payload.data.decode.return_value = "secret-value"
    mock_client.access_secret_version.return_value = mock_response

    # Execute - use default credentials instead of auth_token
    client = SecretManagerClient()
    result = client.get_secret(
        "projects/test-project/secrets/test-secret/versions/latest"
    )

    # Verify
    assert result == "secret-value"
    mock_client.access_secret_version.assert_called_once_with(
        name="projects/test-project/secrets/test-secret/versions/latest"
    )
    mock_response.payload.data.decode.assert_called_once_with("UTF-8")

  @patch("google.cloud.secretmanager.SecretManagerServiceClient")
  @patch(
      "google.adk.tools.apihub_tool.clients.secret_client.default_service_credential"
  )
  def test_get_secret_error(
      self, mock_default_service_credential, mock_secret_manager_client
  ):
    """Test getting a secret that fails."""
    # Setup
    mock_credentials = MagicMock()
    mock_default_service_credential.return_value = (
        mock_credentials,
        "test-project",
    )

    mock_client = MagicMock()
    mock_secret_manager_client.return_value = mock_client
    mock_client.access_secret_version.side_effect = Exception("Secret error")

    # Execute and verify - use default credentials instead of auth_token
    client = SecretManagerClient()
    with pytest.raises(Exception, match="Secret error"):
      client.get_secret(
          "projects/test-project/secrets/test-secret/versions/latest"
      )
