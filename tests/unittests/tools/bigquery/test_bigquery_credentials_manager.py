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


from unittest.mock import Mock
from unittest.mock import patch

from google.adk.auth import AuthConfig
from google.adk.tools import ToolContext
from google.adk.tools.bigquery.bigquery_credentials import BIGQUERY_TOKEN_CACHE_KEY
from google.adk.tools.bigquery.bigquery_credentials import BigQueryCredentials
from google.adk.tools.bigquery.bigquery_credentials import BigQueryCredentialsManager
from google.auth.exceptions import RefreshError
# Mock the Google OAuth and API dependencies
from google.oauth2.credentials import Credentials
import pytest


class TestBigQueryCredentialsManager:
  """Test suite for BigQueryCredentialsManager OAuth flow handling.

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
    # Mock the get method and state dictionary for caching tests
    context.get = Mock(return_value=None)
    context.state = {}
    return context

  @pytest.fixture
  def credentials_config(self):
    """Create a basic credentials configuration for testing."""
    return BigQueryCredentials(
        client_id="test_client_id",
        client_secret="test_client_secret",
        scopes=["https://www.googleapis.com/auth/calendar"],
    )

  @pytest.fixture
  def manager(self, credentials_config):
    """Create a credentials manager instance for testing."""
    return BigQueryCredentialsManager(credentials_config)

  @pytest.mark.asyncio
  async def test_get_valid_credentials_with_valid_existing_creds(
      self, manager, mock_tool_context
  ):
    """Test that valid existing credentials are returned immediately.

    When credentials are already valid, no refresh or OAuth flow
    should be needed. This is the optimal happy path scenario.
    """
    # Create mock credentials that are already valid
    mock_creds = Mock(spec=Credentials)
    mock_creds.valid = True
    manager.credentials.credentials = mock_creds

    result = await manager.get_valid_credentials(mock_tool_context)

    assert result == mock_creds
    # Verify no OAuth flow was triggered
    mock_tool_context.get_auth_response.assert_not_called()
    mock_tool_context.request_credential.assert_not_called()
    # Verify cache retrieval wasn't needed since we had valid creds
    mock_tool_context.get.assert_not_called()

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
    manager.credentials.credentials = None

    # Create mock cached credentials that are valid
    mock_cached_creds = Mock(spec=Credentials)
    mock_cached_creds.valid = True

    # Set up the tool context to return cached credentials
    mock_tool_context.get.return_value = mock_cached_creds

    result = await manager.get_valid_credentials(mock_tool_context)

    # Verify credentials were retrieved from cache
    mock_tool_context.get.assert_called_once_with(
        BIGQUERY_TOKEN_CACHE_KEY, None
    )
    # Verify cached credentials were loaded into manager
    assert manager.credentials.credentials == mock_cached_creds
    # Verify valid cached credentials were returned
    assert result == mock_cached_creds

  @pytest.mark.asyncio
  async def test_no_credentials_in_manager_or_cache(
      self, manager, mock_tool_context
  ):
    """Test OAuth flow when no credentials exist in manager or cache.

    This tests the scenario where both the manager and cache are empty,
    requiring a new OAuth flow to be initiated.
    """
    # Manager starts with no credentials
    manager.credentials.credentials = None
    # Cache also returns None
    mock_tool_context.get.return_value = None

    result = await manager.get_valid_credentials(mock_tool_context)

    # Should trigger OAuth flow and return None (flow in progress)
    assert result is None
    mock_tool_context.get.assert_called_once_with(
        BIGQUERY_TOKEN_CACHE_KEY, None
    )
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
    # Manager starts with no credentials
    manager.credentials.credentials = None

    # Create expired cached credentials with refresh token
    mock_cached_creds = Mock(spec=Credentials)
    mock_cached_creds.valid = False
    mock_cached_creds.expired = True
    mock_cached_creds.refresh_token = "refresh_token"

    # Mock successful refresh
    def mock_refresh(request):
      mock_cached_creds.valid = True

    mock_cached_creds.refresh = Mock(side_effect=mock_refresh)
    mock_tool_context.get.return_value = mock_cached_creds

    result = await manager.get_valid_credentials(mock_tool_context)

    # Verify credentials were retrieved from cache
    mock_tool_context.get.assert_called_once_with(
        BIGQUERY_TOKEN_CACHE_KEY, None
    )
    # Verify refresh was attempted and succeeded
    mock_cached_creds.refresh.assert_called_once()
    # Verify refreshed credentials were loaded into manager
    assert manager.credentials.credentials == mock_cached_creds
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
    mock_creds = Mock(spec=Credentials)
    mock_creds.valid = False
    mock_creds.expired = True
    mock_creds.refresh_token = "refresh_token"

    # Mock successful refresh
    def mock_refresh(request):
      mock_creds.valid = True

    mock_creds.refresh = Mock(side_effect=mock_refresh)
    manager.credentials.credentials = mock_creds

    result = await manager.get_valid_credentials(mock_tool_context)

    assert result == mock_creds
    mock_creds.refresh.assert_called_once()
    # Verify credentials were cached after successful refresh
    assert manager.credentials.credentials == mock_creds

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
    mock_creds = Mock(spec=Credentials)
    mock_creds.valid = False
    mock_creds.expired = True
    mock_creds.refresh_token = "expired_refresh_token"
    mock_creds.refresh = Mock(side_effect=RefreshError("Refresh failed"))
    manager.credentials.credentials = mock_creds

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

    result = await manager.get_valid_credentials(mock_tool_context)

    # Verify new credentials were created and cached
    assert isinstance(result, Credentials)
    assert result.token == "new_access_token"
    assert result.refresh_token == "new_refresh_token"
    # Verify credentials are cached in manager
    assert manager.credentials.credentials == result
    # Verify credentials are also cached in tool context state
    assert mock_tool_context.state[BIGQUERY_TOKEN_CACHE_KEY] == result

  @pytest.mark.asyncio
  async def test_oauth_flow_in_progress(self, manager, mock_tool_context):
    """Test OAuth flow initiation when no auth response is available.

    This tests the case where the OAuth flow needs to be started,
    and the user hasn't completed authorization yet.
    """
    # No existing credentials, no auth response (flow not completed)
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
    manager1 = BigQueryCredentialsManager(credentials_config)

    # Mock OAuth response for first manager
    mock_auth_response = Mock()
    mock_auth_response.oauth2.access_token = "cached_access_token"
    mock_auth_response.oauth2.refresh_token = "cached_refresh_token"
    mock_tool_context.get_auth_response.return_value = mock_auth_response

    # Complete OAuth flow with first manager
    result1 = await manager1.get_valid_credentials(mock_tool_context)

    # Verify credentials were cached in tool context
    assert BIGQUERY_TOKEN_CACHE_KEY in mock_tool_context.state
    cached_creds = mock_tool_context.state[BIGQUERY_TOKEN_CACHE_KEY]

    # Create second manager instance (simulating new request/session)
    manager2 = BigQueryCredentialsManager(credentials_config)

    # Reset auth response to None (no new OAuth flow available)
    mock_tool_context.get_auth_response.return_value = None
    # Set up get method to return cached credentials
    mock_tool_context.get.return_value = cached_creds

    # Get credentials with second manager
    result2 = await manager2.get_valid_credentials(mock_tool_context)

    # Verify second manager retrieved cached credentials successfully
    assert result2 == cached_creds
    assert manager2.credentials.credentials == cached_creds
    # Verify no new OAuth flow was requested
    assert (
        mock_tool_context.request_credential.call_count == 0
    )  # Only from first manager
