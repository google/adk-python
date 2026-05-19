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

"""Unit tests for the ParameterManagerClient."""

import json
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.integrations.parameter_manager.parameter_client import ParameterManagerClient
from google.adk.integrations.parameter_manager.parameter_client import USER_AGENT
from google.api_core.gapic_v1 import client_info
import pytest


class TestParameterManagerClient:
  """Tests for the ParameterManagerClient class."""

  @patch("google.cloud.parametermanager_v1.ParameterManagerClient")
  @patch(
      "google.adk.integrations.parameter_manager.parameter_client.default_service_credential"
  )
  def test_init_with_default_credentials(
      self, mock_default_service_credential, mock_pm_client_class
  ):
    """Test initialization with default credentials."""
    # Setup
    mock_credentials = MagicMock()
    mock_default_service_credential.return_value = (
        mock_credentials,
        "test-project",
    )

    # Execute
    client = ParameterManagerClient()

    # Verify
    mock_default_service_credential.assert_called_once_with(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    mock_pm_client_class.assert_called_once()
    call_kwargs = mock_pm_client_class.call_args.kwargs
    assert call_kwargs["credentials"] == mock_credentials
    assert call_kwargs["client_options"] is None
    assert call_kwargs["client_info"].user_agent == USER_AGENT
    assert client._credentials == mock_credentials
    assert client._client == mock_pm_client_class.return_value

  @patch("google.cloud.parametermanager_v1.ParameterManagerClient")
  @patch("google.oauth2.service_account.Credentials.from_service_account_info")
  def test_init_with_service_account_json(
      self, mock_from_service_account_info, mock_pm_client_class
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
    client = ParameterManagerClient(service_account_json=service_account_json)

    # Verify
    mock_from_service_account_info.assert_called_once_with(
        json.loads(service_account_json)
    )
    mock_pm_client_class.assert_called_once()
    call_kwargs = mock_pm_client_class.call_args.kwargs
    assert call_kwargs["credentials"] == mock_credentials
    assert call_kwargs["client_options"] is None
    assert call_kwargs["client_info"].user_agent == USER_AGENT
    assert client._credentials == mock_credentials
    assert client._client == mock_pm_client_class.return_value

  @patch("google.cloud.parametermanager_v1.ParameterManagerClient")
  def test_init_with_auth_token(self, mock_pm_client_class):
    """Test initialization with auth token."""
    # Setup
    auth_token = "test-token"
    mock_credentials = MagicMock()

    with (
        patch("google.auth.credentials.Credentials") as mock_credentials_class,
        patch("google.auth.transport.requests.Request") as mock_request,
    ):
      mock_credentials_class.return_value = mock_credentials

      # Execute
      client = ParameterManagerClient(auth_token=auth_token)

      # Verify
      mock_credentials.refresh.assert_called_once()
      mock_pm_client_class.assert_called_once()
      call_kwargs = mock_pm_client_class.call_args.kwargs
      assert call_kwargs["credentials"] == mock_credentials
      assert call_kwargs["client_options"] is None
      assert call_kwargs["client_info"].user_agent == USER_AGENT
      assert client._credentials == mock_credentials
      assert client._client == mock_pm_client_class.return_value

  @patch("google.cloud.parametermanager_v1.ParameterManagerClient")
  @patch(
      "google.adk.integrations.parameter_manager.parameter_client.default_service_credential"
  )
  def test_init_with_location(
      self, mock_default_service_credential, mock_pm_client_class
  ):
    """Test initialization with a specific location."""
    # Setup
    mock_credentials = MagicMock()
    mock_default_service_credential.return_value = (
        mock_credentials,
        "test-project",
    )
    location = "us-central1"

    # Execute
    ParameterManagerClient(location=location)

    # Verify
    mock_pm_client_class.assert_called_once()
    call_kwargs = mock_pm_client_class.call_args.kwargs
    assert call_kwargs["credentials"] == mock_credentials
    assert call_kwargs["client_options"] == {
        "api_endpoint": f"parametermanager.{location}.rep.googleapis.com"
    }
    assert call_kwargs["client_info"].user_agent == USER_AGENT

  @patch(
      "google.adk.integrations.parameter_manager.parameter_client.default_service_credential"
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
      ParameterManagerClient()

  def test_init_with_invalid_service_account_json(self):
    """Test initialization with invalid service account JSON."""
    # Execute and verify
    with pytest.raises(ValueError, match="Invalid service account JSON"):
      ParameterManagerClient(service_account_json="invalid-json")

  @patch("google.cloud.parametermanager_v1.ParameterManagerClient")
  @patch(
      "google.adk.integrations.parameter_manager.parameter_client.default_service_credential"
  )
  def test_get_parameter(
      self, mock_default_service_credential, mock_pm_client_class
  ):
    """Test getting a parameter."""
    # Setup
    mock_credentials = MagicMock()
    mock_default_service_credential.return_value = (
        mock_credentials,
        "test-project",
    )

    mock_client = MagicMock()
    mock_pm_client_class.return_value = mock_client
    mock_response = MagicMock()
    mock_response.rendered_payload.decode.return_value = "parameter-value"
    mock_client.render_parameter_version.return_value = mock_response

    # Execute
    client = ParameterManagerClient()
    result = client.get_parameter(
        "projects/test-project/locations/global/parameters/test-param/versions/latest"
    )

    # Verify
    assert result == "parameter-value"
    mock_response.rendered_payload.decode.assert_called_once_with("UTF-8")
    # Verify render_parameter_version was called with correct request object
    call_kwargs = mock_client.render_parameter_version.call_args.kwargs
    assert (
        call_kwargs["request"].name
        == "projects/test-project/locations/global/parameters/test-param/versions/latest"
    )
    mock_response.rendered_payload.decode.assert_called_once_with("UTF-8")

  @patch("google.cloud.parametermanager_v1.ParameterManagerClient")
  @patch(
      "google.adk.integrations.parameter_manager.parameter_client.default_service_credential"
  )
  def test_get_parameter_error(
      self, mock_default_service_credential, mock_pm_client_class
  ):
    """Test getting a parameter that fails."""
    # Setup
    mock_credentials = MagicMock()
    mock_default_service_credential.return_value = (
        mock_credentials,
        "test-project",
    )

    mock_client = MagicMock()
    mock_pm_client_class.return_value = mock_client
    mock_client.render_parameter_version.side_effect = Exception("API error")

    # Execute and verify
    client = ParameterManagerClient()
    with pytest.raises(Exception, match="API error"):
      client.get_parameter(
          "projects/test-project/locations/global/parameters/test-param/versions/latest"
      )
