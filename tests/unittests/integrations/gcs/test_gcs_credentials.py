# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# # Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest import mock

from google.adk.integrations.gcs.gcs_credentials import GCS_DEFAULT_SCOPE
from google.adk.integrations.gcs.gcs_credentials import GCSCredentialsConfig
from google.auth.credentials import Credentials
import google.oauth2.credentials
import pytest


class TestGCSCredentials:
  """Test suite for GCS credentials configuration validation."""

  def test_gcs_credentials_config_client_id_secret(self):
    """Test GCSCredentialsConfig with client_id and client_secret."""
    config = GCSCredentialsConfig(client_id="abc", client_secret="def")
    assert config.client_id == "abc"
    assert config.client_secret == "def"
    assert config.scopes == GCS_DEFAULT_SCOPE
    assert config.credentials is None

  def test_gcs_credentials_config_existing_creds(self):
    """Test GCSCredentialsConfig with existing generic credentials."""
    mock_creds = mock.create_autospec(Credentials, instance=True)
    config = GCSCredentialsConfig(credentials=mock_creds)
    assert config.credentials == mock_creds
    assert config.client_id is None
    assert config.client_secret is None

  def test_gcs_credentials_config_oauth2_creds(self):
    """Test GCSCredentialsConfig with existing OAuth2 credentials."""
    mock_creds = mock.create_autospec(
        google.oauth2.credentials.Credentials, instance=True
    )
    mock_creds.client_id = "oauth_client_id"
    mock_creds.client_secret = "oauth_client_secret"
    mock_creds.scopes = ["fake_scope"]
    config = GCSCredentialsConfig(credentials=mock_creds)
    assert config.client_id == "oauth_client_id"
    assert config.client_secret == "oauth_client_secret"
    assert config.scopes == ["fake_scope"]

  def test_gcs_credentials_config_validation_errors(self):
    """Test GCSCredentialsConfig validation errors."""
    with pytest.raises(ValueError):
      GCSCredentialsConfig()

    with pytest.raises(ValueError):
      GCSCredentialsConfig(client_id="abc")

    mock_creds = mock.create_autospec(Credentials, instance=True)
    with pytest.raises(ValueError):
      GCSCredentialsConfig(
          credentials=mock_creds, client_id="abc", client_secret="def"
      )
