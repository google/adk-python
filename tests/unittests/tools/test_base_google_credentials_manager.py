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


import json
from unittest.mock import Mock
from unittest.mock import patch

from google.adk.auth.auth_tool import AuthConfig
from google.adk.tools._google_credentials import GoogleCredentialsManager
from google.adk.tools.bigquery.bigquery_credentials import BIGQUERY_TOKEN_CACHE_KEY
from google.adk.tools.bigquery.bigquery_credentials import BigQueryCredentialsConfig
from google.adk.tools.tool_context import ToolContext
from google.auth.credentials import Credentials as AuthCredentials
from google.auth.exceptions import RefreshError
# Mock the Google OAuth and API dependencies
from google.oauth2.credentials import Credentials as OAuthCredentials
import pytest


class TestGoogleCredentialsManager:
  """Test suite for GoogleCredentialsManager OAuth flow handling.

  This class tests the complex credential management logic including
  credential validation, refresh, OAuth flow orchestration, and the
  new token caching functionality through tool_context.state.
  """

  @pytest.fixture
  def mock_tool_context(self):
    """Create a mock ToolContext for testing.

    The ToolContext is the interface between tools and the broader
    agent framework, handling OAuth flows and state management.
    Now includes state dictionary for testing caching behavior.
    """
    context = Mock(spec=ToolContext)
    context.get_auth_response = Mock(return_value=None)
    context.request_credential = Mock()
    context.state = {}
    return context

  @pytest.fixture
  def credentials_config(self):
    """Create a basic credentials configuration for testing."""
    return BigQueryCredentialsConfig(
        client_id="test_client_id",
        client_secret="test_client_secret",
        scopes=["https://www.googleapis.com/auth/calendar"],
    )

  @pytest.fixture
  def manager(self, credentials_config):
    """Create a credentials manager instance for testing."""
    return GoogleCredentialsManager(credentials_config)

  @pytest.mark.parametrize(
      ("credentials_class",),
      [
          pytest.param(OAuthCredentials, id="oauth"),
          pytest.param(AuthCredentials, id="auth"),
      ],
  )
  @pytest.mark.asyncio
  async def test_get_valid_credentials_with_valid_existing_creds(
      self, manager, mock_tool_context, credentials_class
  ):
    """Test that valid existing credentials are returned immediately.

    When credentials are already valid, no refresh or OAuth flow
    should be needed. This is the optimal happy path scenario.
    """
    # Create mock credentials that are already valid
    mock_creds = Mock(spec=credentials_class)
    mock_creds.valid = True
    manager.credentials_config.credentials = mock_creds

    result = await manager.get_valid_credentials(mock_tool_context)

    assert result == mock_creds
    # Verify no OAuth flow was triggered
    mock_tool_context.get_auth_response.assert_not_called()
    mock_tool_context.request_credential.assert_not_called()

  @pytest.mark.parametrize(
      ("valid",),
      [
          pytest.param(False, id="invalid"),
          pytest.param(True, id="valid"),
      ],
  )
  @pytest.mark.asyncio
  async def test_get_valid_credentials_with_existing_non_oauth_creds(
      self, manager, mock_tool_context, valid
  ):
    """Test that existing non-oauth credentials are returned immediately.

    When credentials are of non-oauth type, no refresh or OAuth flow
    is triggered irrespective of whether or not it is valid.
    """
    # Create mock credentials that are already valid
    mock_creds = Mock(spec=AuthCredentials)
    mock_creds.valid = valid
    manager.credentials_config.credentials = mock_creds

    result = await manager.get_valid_credentials(mock_tool_context)

    assert result == mock_creds
    # Verify no OAuth flow was triggered
    mock_tool_context.get_auth_response.assert_not_called()
    mock_tool_context.request_credential.assert_not_called()

  @pytest.mark.asyncio
  async def test_get_credentials_from_cache_when_none_in_manager(
      self, manager, mock_tool_context
  ):
    """Test retrieving credentials from tool_context cache when manager has none.

    This tests the new caching functionality where credentials can be
    retrieved from the tool context state when the manager instance
    doesn't have them loaded.
    """
    # Manager starts with no credentials
    manager.credentials_config.credentials = None

    # Create mock cached credentials JSON that would be stored in cache
    mock_cached_creds_json = json.dumps({
        "token": "cached_token",
        "refresh_token": "cached_refresh_token",
        "client_id": "test_client_id",
        "client_secret": "test_client_secret",
    })

    # Set up the tool context state to contain cached credentials
    mock_tool_context.state[BIGQUERY_TOKEN_CACHE_KEY] = mock_cached_creds_json

    # Mock the Credentials.from_authorized_user_info method
    with patch(
        "google.oauth2.credentials.Credentials.from_authorized_user_info"
    ) as mock_from_json:
      mock_creds = Mock(spec=OAuthCredentials)
      mock_creds.valid = True
      mock_from_json.return_value = mock_creds

      result = await manager.get_valid_credentials(mock_tool_context)

      # Verify credentials were created from cached JSON
      mock_from_json.assert_called_once_with(
          json.loads(mock_cached_creds_json), manager.credentials_config.scopes
      )
      # Verify loaded credentials were not cached into manager
      assert manager.credentials_config.credentials is None
      # Verify valid cached credentials were returned
      assert result == mock_creds

  @pytest.mark.asyncio
  async def test_no_credentials_in_manager_or_cache(
      self, manager, mock_tool_context
  ):
    """Test OAuth flow when no credentials exist in manager or cache.

    This tests the scenario where both the manager and cache are empty,
    requiring a new OAuth flow to be initiated.
    """
    # Manager starts with no credentials
    manager.credentials_config.credentials = None
    # Cache is also empty (state dict doesn't contain the key)

    result = await manager.get_valid_credentials(mock_tool_context)

    # Should trigger OAuth flow and return None (flow in progress)
    assert result is None
    mock_tool_context.request_credential.assert_called_once()

  @pytest.mark.asyncio
  @patch("google.auth.transport.requests.Request")
  async def test_refresh_cached_credentials_success(
      self, mock_request_class, manager, mock_tool_context
  ):
    """Test successful refresh of expired credentials retrieved from cache.

    This tests the interaction between caching and refresh functionality,
    ensuring that expired cached credentials can be refreshed properly.
    """
    # Manager starts with no default credentials
    manager.credentials_config.credentials = None

    # Create mock cached credentials JSON
    mock_cached_creds_json = json.dumps({
        "token": "expired_token",
        "refresh_token": "valid_refresh_token",
        "client_id": "test_client_id",
        "client_secret": "test_client_secret",
    })

    mock_refreshed_creds_json = json.dumps({
        "token": "new_token",
        "refresh_token": "valid_refresh_token",
        "client_id": "test_client_id",
        "client_secret": "test_client_secret",
    })

    # Set up the tool context state to contain cached credentials
    mock_tool_context.state[BIGQUERY_TOKEN_CACHE_KEY] = mock_cached_creds_json

    # Create expired cached credentials with refresh token
    mock_cached_creds = Mock(spec=OAuthCredentials)
    mock_cached_creds.valid = False
    mock_cached_creds.expired = True
    mock_cached_creds.refresh_token = "valid_refresh_token"
    mock_cached_creds.to_json.return_value = mock_refreshed_creds_json

    # Mock successful refresh
    def mock_refresh(request):
      mock_cached_creds.valid = True

    mock_cached_creds.refresh = Mock(side_effect=mock_refresh)

    # Mock the Credentials.from_authorized_user_info method
    with patch(
        "google.oauth2.credentials.Credentials.from_authorized_user_info"
    ) as mock_from_json:
      mock_from_json.return_value = mock_cached_creds

      result = await manager.get_valid_credentials(mock_tool_context)

      # Verify credentials were created from cached JSON
      mock_from_json.assert_called_once_with(
          json.loads(mock_cached_creds_json), manager.credentials_config.scopes
      )
      # Verify refresh was attempted and succeeded
      mock_cached_creds.refresh.assert_called_once()
      # Verify refreshed credentials were not cached into manager
      assert manager.credentials_config.credentials is None
      # Verify refreshed credentials were cached
      assert (
          "new_token"
          == json.loads(mock_tool_context.state[BIGQUERY_TOKEN_CACHE_KEY])[
              "token"
          ]
      )
      assert result == mock_cached_creds

  @pytest.mark.asyncio
  @patch("google.auth.transport.requests.Request")
  async def test_get_valid_credentials_with_refresh_success(
      self, mock_request_class, manager, mock_tool_context
  ):
    """Test successful credential refresh when tokens are expired.

    This tests the automatic token refresh capability that prevents
    users from having to re-authenticate for every expired token.
    """
    # Create expired credentials with refresh token
    mock_creds = Mock(spec=OAuthCredentials)
    mock_creds.valid = False
    mock_creds.expired = True
    mock_creds.refresh_token = "refresh_token"

    # Mock successful refresh
    def mock_refresh(request):
      mock_creds.valid = True

    mock_creds.refresh = Mock(side_effect=mock_refresh)
    manager.credentials_config.credentials = mock_creds

    result = await manager.get_valid_credentials(mock_tool_context)

    assert result == mock_creds
    mock_creds.refresh.assert_called_once()
    # Verify credentials were cached after successful refresh
    assert manager.credentials_config.credentials == mock_creds

  @pytest.mark.asyncio
  @patch("google.auth.transport.requests.Request")
  async def test_get_valid_credentials_with_refresh_failure(
      self, mock_request_class, manager, mock_tool_context
  ):
    """Test OAuth flow trigger when credential refresh fails.

    When refresh tokens expire or become invalid, the system should
    gracefully fall back to requesting a new OAuth flow.
    """
    # Create expired credentials that fail to refresh
    mock_creds = Mock(spec=OAuthCredentials)
    mock_creds.valid = False
    mock_creds.expired = True
    mock_creds.refresh_token = "expired_refresh_token"
    mock_creds.refresh = Mock(side_effect=RefreshError("Refresh failed"))
    manager.credentials_config.credentials = mock_creds

    result = await manager.get_valid_credentials(mock_tool_context)

    # Should trigger OAuth flow and return None (flow in progress)
    assert result is None
    mock_tool_context.request_credential.assert_called_once()

  @pytest.mark.asyncio
  async def test_oauth_flow_completion_with_caching(
      self, manager, mock_tool_context
  ):
    """Test successful OAuth flow completion with proper credential caching.

    This tests the happy path where a user completes the OAuth flow
    and the system successfully creates and caches new credentials
    in both the manager and the tool context state.
    """
    # Mock OAuth response indicating completed flow
    mock_auth_response = Mock()
    mock_auth_response.oauth2.access_token = "new_access_token"
    mock_auth_response.oauth2.refresh_token = "new_refresh_token"
    mock_tool_context.get_auth_response.return_value = mock_auth_response

    # Create a mock credentials instance that will represent our created credentials
    mock_creds = Mock(spec=OAuthCredentials)
    # Make the JSON match what a real Credentials object would produce
    mock_creds_json = (
        '{"token": "new_access_token", "refresh_token": "new_refresh_token",'
        ' "token_uri": "https://oauth2.googleapis.com/token", "client_id":'
        ' "test_client_id", "client_secret": "test_client_secret", "scopes":'
        ' ["https://www.googleapis.com/auth/calendar"], "universe_domain":'
        ' "googleapis.com", "account": ""}'
    )
    mock_creds.to_json.return_value = mock_creds_json

    # Use the full module path as it appears in the project structure
    with patch(
        "google.adk.tools._google_credentials.google.oauth2.credentials.Credentials",
        return_value=mock_creds,
    ) as mock_credentials_class:
      result = await manager.get_valid_credentials(mock_tool_context)

      # Verify new credentials were created
      assert result == mock_creds
      # Verify credentials are created with correct parameters
      mock_credentials_class.assert_called_once()
      call_kwargs = mock_credentials_class.call_args[1]
      assert call_kwargs["token"] == "new_access_token"
      assert call_kwargs["refresh_token"] == "new_refresh_token"

      # Verify credentials are not cached in manager
      assert manager.credentials_config.credentials is None
      # Verify credentials are also cached in tool context state
      assert (
          mock_tool_context.state[BIGQUERY_TOKEN_CACHE_KEY] == mock_creds_json
      )

  @pytest.mark.asyncio
  async def test_oauth_flow_in_progress(self, manager, mock_tool_context):
    """Test OAuth flow initiation when no auth response is available.

    This tests the case where the OAuth flow needs to be started,
    and the user hasn't completed authorization yet.
    """
    # No existing credentials, no auth response (flow not completed)
    manager.credentials_config.credentials = None
    mock_tool_context.get_auth_response.return_value = None

    result = await manager.get_valid_credentials(mock_tool_context)

    # Should return None and request credential flow
    assert result is None
    mock_tool_context.request_credential.assert_called_once()

    # Verify the auth configuration includes correct scopes and endpoints
    call_args = mock_tool_context.request_credential.call_args[0][0]
    assert isinstance(call_args, AuthConfig)

  @pytest.mark.asyncio
  async def test_cache_persistence_across_manager_instances(
      self, credentials_config, mock_tool_context
  ):
    """Test that cached credentials persist across different manager instances.

    This tests the key benefit of the tool context caching - that
    credentials can be shared between different instances of the
    credential manager, avoiding redundant OAuth flows.
    """
    # Create first manager instance and simulate OAuth completion
    manager1 = GoogleCredentialsManager(credentials_config)

    # Mock OAuth response for first manager
    mock_auth_response = Mock()
    mock_auth_response.oauth2.access_token = "cached_access_token"
    mock_auth_response.oauth2.refresh_token = "cached_refresh_token"
    mock_tool_context.get_auth_response.return_value = mock_auth_response

    # Create the mock credentials instance that will be returned by the constructor
    mock_creds = Mock(spec=OAuthCredentials)
    # Make sure our mock JSON matches the structure that real Credentials objects produce
    mock_creds_json = (
        '{"token": "cached_access_token", "refresh_token":'
        ' "cached_refresh_token", "token_uri":'
        ' "https://oauth2.googleapis.com/token", "client_id": "test_client_id",'
        ' "client_secret": "test_client_secret", "scopes":'
        ' ["https://www.googleapis.com/auth/calendar"], "universe_domain":'
        ' "googleapis.com", "account": ""}'
    )
    mock_creds.to_json.return_value = mock_creds_json
    mock_creds.valid = True

    # Use the correct module path - without the 'src.' prefix
    with patch(
        "google.adk.tools._google_credentials.google.oauth2.credentials.Credentials",
        return_value=mock_creds,
    ) as mock_credentials_class:
      # Complete OAuth flow with first manager
      result1 = await manager1.get_valid_credentials(mock_tool_context)

      # Verify credentials were cached in tool context
      assert BIGQUERY_TOKEN_CACHE_KEY in mock_tool_context.state
      cached_creds_json = mock_tool_context.state[BIGQUERY_TOKEN_CACHE_KEY]
      assert cached_creds_json == mock_creds_json

    # Create second manager instance (simulating new request/session)
    manager2 = GoogleCredentialsManager(credentials_config)
    credentials_config.credentials = None

    # Reset auth response to None (no new OAuth flow available)
    mock_tool_context.get_auth_response.return_value = None

    # Mock the from_authorized_user_info method for the second manager
    with patch(
        "google.adk.tools._google_credentials.google.oauth2.credentials.Credentials.from_authorized_user_info"
    ) as mock_from_json:
      mock_cached_creds = Mock(spec=OAuthCredentials)
      mock_cached_creds.valid = True
      mock_from_json.return_value = mock_cached_creds

      # Get credentials with second manager
      result2 = await manager2.get_valid_credentials(mock_tool_context)

      # Verify second manager retrieved cached credentials successfully
      assert result2 == mock_cached_creds
      assert manager2.credentials_config.credentials is None
      assert (
          cached_creds_json == mock_tool_context.state[BIGQUERY_TOKEN_CACHE_KEY]
      )
      # The from_authorized_user_info should be called with the complete JSON structure
      mock_from_json.assert_called_once()
      # Extract the actual argument that was passed to verify it's the right JSON structure
      actual_json_arg = mock_from_json.call_args[0][0]
      # We need to parse and compare the structure rather than exact string match
      # since the order of keys in JSON might differ
      import json

      expected_data = json.loads(mock_creds_json)
      actual_data = (
          actual_json_arg
          if isinstance(actual_json_arg, dict)
          else json.loads(actual_json_arg)
      )
      assert actual_data == expected_data
