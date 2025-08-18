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

from google.adk.tools.spanner.spanner_credentials import SpannerCredentialsConfig
# Mock the Google OAuth and API dependencies
import google.auth.credentials
import google.oauth2.credentials
import pytest


class TestSpannerCredentials:
  """Test suite for Spanner credentials configuration validation.

  This class tests the credential configuration logic that ensures
  either existing credentials or client ID/secret pairs are provided.
  """

  def test_valid_credentials_object_oauth2_credentials(self):
    """Test that providing valid Credentials object works correctly with google.oauth2.credentials.Credentials.

    When a user already has valid OAuth credentials, they should be able
    to pass them directly without needing to provide client ID/secret.
    """
    # Create a mock oauth2 credentials object
    oauth2_creds = google.oauth2.credentials.Credentials(
        "test_token",
        client_id="test_client_id",
        client_secret="test_client_secret",
        scopes=[],
    )

    config = SpannerCredentialsConfig(credentials=oauth2_creds)

    # Verify that the credentials are properly stored and attributes are
    # extracted
    assert config.credentials == oauth2_creds
    assert config.client_id == "test_client_id"
    assert config.client_secret == "test_client_secret"
    assert config.scopes == [
        "https://www.googleapis.com/auth/spanner.admin",
        "https://www.googleapis.com/auth/spanner.data",
    ]

    assert config._token_cache_key == "spanner_token_cache"  # pylint: disable=protected-access
