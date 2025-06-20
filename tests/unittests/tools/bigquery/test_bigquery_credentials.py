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

from unittest import mock

from google.adk.tools.bigquery.bigquery_credentials import BigQueryCredentialsConfig
# Mock the Google OAuth and API dependencies
import google.auth.credentials
import google.oauth2.credentials
import pytest


class TestBigQueryCredentials:
  """Test suite for BigQueryCredentials configuration validation.

  This class tests the credential configuration logic that ensures
  either existing credentials or client ID/secret pairs are provided.
  """

  def test_valid_credentials_object_auth_credentials(self):
    """Test that providing valid Credentials object works correctly with
    google.auth.credentials.Credentials.

    When a user already has valid OAuth credentials, they should be able
    to pass them directly without needing to provide client ID/secret.
    """
    # Create a mock auth credentials object
    # auth_creds = google.auth.credentials.Credentials()
    auth_creds = mock.create_autospec(
        google.auth.credentials.Credentials, instance=True
    )

    config = BigQueryCredentialsConfig(credentials=auth_creds)

    # Verify that the credentials are properly stored and attributes are extracted
    assert config.credentials == auth_creds
    assert config.client_id is None
    assert config.client_secret is None
    assert config.scopes == ["https://www.googleapis.com/auth/bigquery"]

  def test_valid_credentials_object_oauth2_credentials(self):
    """Test that providing valid Credentials object works correctly with
    google.oauth2.credentials.Credentials.

    When a user already has valid OAuth credentials, they should be able
    to pass them directly without needing to provide client ID/secret.
    """
    # Create a mock oauth2 credentials object
    oauth2_creds = google.oauth2.credentials.Credentials(
        "test_token",
        client_id="test_client_id",
        client_secret="test_client_secret",
        scopes=["https://www.googleapis.com/auth/calendar"],
    )

    config = BigQueryCredentialsConfig(credentials=oauth2_creds)

    # Verify that the credentials are properly stored and attributes are extracted
    assert config.credentials == oauth2_creds
    assert config.client_id == "test_client_id"
    assert config.client_secret == "test_client_secret"
    assert config.scopes == ["https://www.googleapis.com/auth/calendar"]

  def test_valid_client_id_secret_pair_default_scope(self):
    """Test that providing client ID and secret with default scope works.

    This tests the scenario where users want to create new OAuth credentials
    from scratch using their application's client ID and secret and does not
    specify the scopes explicitly.
    """
    config = BigQueryCredentialsConfig(
        client_id="test_client_id",
        client_secret="test_client_secret",
    )

    assert config.credentials is None
    assert config.client_id == "test_client_id"
    assert config.client_secret == "test_client_secret"
    assert config.scopes == ["https://www.googleapis.com/auth/bigquery"]

  def test_valid_client_id_secret_pair_w_scope(self):
    """Test that providing client ID and secret with explicit scopes works.

    This tests the scenario where users want to create new OAuth credentials
    from scratch using their application's client ID and secret and does specify
    the scopes explicitly.
    """
    config = BigQueryCredentialsConfig(
        client_id="test_client_id",
        client_secret="test_client_secret",
        scopes=[
            "https://www.googleapis.com/auth/bigquery",
            "https://www.googleapis.com/auth/drive",
        ],
    )

    assert config.credentials is None
    assert config.client_id == "test_client_id"
    assert config.client_secret == "test_client_secret"
    assert config.scopes == [
        "https://www.googleapis.com/auth/bigquery",
        "https://www.googleapis.com/auth/drive",
    ]

  def test_valid_client_id_secret_pair_w_empty_scope(self):
    """Test that providing client ID and secret with empty scope works.

    This tests the corner case scenario where users want to create new OAuth
    credentials from scratch using their application's client ID and secret but
    specifies empty scope, in which case the default BQ scope is used.
    """
    config = BigQueryCredentialsConfig(
        client_id="test_client_id",
        client_secret="test_client_secret",
        scopes=[],
    )

    assert config.credentials is None
    assert config.client_id == "test_client_id"
    assert config.client_secret == "test_client_secret"
    assert config.scopes == ["https://www.googleapis.com/auth/bigquery"]

  def test_missing_client_secret_raises_error(self):
    """Test that missing client secret raises appropriate validation error.

    This ensures that incomplete OAuth configuration is caught early
    rather than failing during runtime.
    """
    with pytest.raises(
        ValueError,
        match=(
            "Must provide either credentials or client_id and client_secret"
            " pair"
        ),
    ):
      BigQueryCredentialsConfig(client_id="test_client_id")

  def test_missing_client_id_raises_error(self):
    """Test that missing client ID raises appropriate validation error."""
    with pytest.raises(
        ValueError,
        match=(
            "Must provide either credentials or client_id and client_secret"
            " pair"
        ),
    ):
      BigQueryCredentialsConfig(client_secret="test_client_secret")

  def test_empty_configuration_raises_error(self):
    """Test that completely empty configuration is rejected.

    Users must provide either existing credentials or the components
    needed to create new ones.
    """
    with pytest.raises(
        ValueError,
        match=(
            "Must provide either credentials or client_id and client_secret"
            " pair"
        ),
    ):
      BigQueryCredentialsConfig()
