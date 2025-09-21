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

from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlowAuthorizationCode
from fastapi.openapi.models import OAuthFlows
from google.adk.agents.callback_context import CallbackContext
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.credential_service.in_memory_credential_service import InMemoryCredentialService
import pytest


class TestInMemoryCredentialService:
  """Tests for the InMemoryCredentialService class."""

  @pytest.fixture
  def credential_service(self):
    """Create an InMemoryCredentialService instance for testing."""
    return InMemoryCredentialService()

  @pytest.fixture
  def oauth2_auth_scheme(self):
    """Create an OAuth2 auth scheme for testing."""
    flows = OAuthFlows(
        authorizationCode=OAuthFlowAuthorizationCode(
            authorizationUrl="https://example.com/oauth2/authorize",
            tokenUrl="https://example.com/oauth2/token",
            scopes={"read": "Read access", "write": "Write access"},
        )
    )
    return OAuth2(flows=flows)

  @pytest.fixture
  def oauth2_credentials(self):
    """Create OAuth2 credentials for testing."""
    return AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id="mock_client_id",
            client_secret="mock_client_secret",
            redirect_uri="https://example.com/callback",
        ),
    )

  @pytest.fixture
  def auth_config(self, oauth2_auth_scheme, oauth2_credentials):
    """Create an AuthConfig for testing."""
    exchanged_credential = oauth2_credentials.model_copy(deep=True)
    return AuthConfig(
        auth_scheme=oauth2_auth_scheme,
        raw_auth_credential=oauth2_credentials,
        exchanged_auth_credential=exchanged_credential,
    )

  @pytest.fixture
  def callback_context(self):
    """Create a mock CallbackContext for testing."""
    mock_context = Mock(spec=CallbackContext)
    mock_invocation_context = Mock()
    mock_invocation_context.app_name = "test_app"
    mock_invocation_context.user_id = "test_user"
    mock_context._invocation_context = mock_invocation_context
    return mock_context

  @pytest.fixture
  def another_callback_context(self):
    """Create another mock CallbackContext with different app/user for testing isolation."""
    mock_context = Mock(spec=CallbackContext)
    mock_invocation_context = Mock()
    mock_invocation_context.app_name = "another_app"
    mock_invocation_context.user_id = "another_user"
    mock_context._invocation_context = mock_invocation_context
    return mock_context

  def test_init(self, credential_service):
    """Test that the service initializes with an empty store."""
    assert isinstance(credential_service._credentials, dict)
    assert len(credential_service._credentials) == 0

  @pytest.mark.asyncio
  async def test_load_credential_not_found(
      self, credential_service, auth_config, callback_context
  ):
    """Test loading a credential that doesn't exist returns None."""
    result = await credential_service.load_credential(
        auth_config, callback_context
    )
    assert result is None

  @pytest.mark.asyncio
  async def test_save_and_load_credential(
      self, credential_service, auth_config, callback_context
  ):
    """Test saving and then loading a credential."""
    # Save the credential
    await credential_service.save_credential(auth_config, callback_context)

    # Load the credential
    result = await credential_service.load_credential(
        auth_config, callback_context
    )

    # Verify the credential was saved and loaded correctly
    assert result is not None
    assert result == auth_config.exchanged_auth_credential
    assert result.auth_type == AuthCredentialTypes.OAUTH2
    assert result.oauth2.client_id == "mock_client_id"

  @pytest.mark.asyncio
  async def test_save_credential_updates_existing(
      self,
      credential_service,
      auth_config,
      callback_context,
      oauth2_credentials,
  ):
    """Test that saving a credential updates an existing one."""
    # Save initial credential
    await credential_service.save_credential(auth_config, callback_context)

    # Create a new credential and update the auth_config
    new_credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id="updated_client_id",
            client_secret="updated_client_secret",
            redirect_uri="https://updated.com/callback",
        ),
    )
    auth_config.exchanged_auth_credential = new_credential

    # Save the updated credential
    await credential_service.save_credential(auth_config, callback_context)

    # Load and verify the credential was updated
    result = await credential_service.load_credential(
        auth_config, callback_context
    )
    assert result is not None
    assert result.oauth2.client_id == "updated_client_id"
    assert result.oauth2.client_secret == "updated_client_secret"

  @pytest.mark.asyncio
  async def test_credentials_isolated_by_context(
      self,
      credential_service,
      auth_config,
      callback_context,
      another_callback_context,
  ):
    """Test that credentials are isolated between different app/user contexts."""
    # Save credential in first context
    await credential_service.save_credential(auth_config, callback_context)

    # Try to load from another context
    result = await credential_service.load_credential(
        auth_config, another_callback_context
    )
    assert result is None

    # Verify original context still has the credential
    result = await credential_service.load_credential(
        auth_config, callback_context
    )
    assert result is not None

  @pytest.mark.asyncio
  async def test_multiple_credentials_same_context(
      self, credential_service, callback_context, oauth2_auth_scheme
  ):
    """Test storing multiple credentials in the same context with different keys."""
    # Create two different auth configs with different credential keys
    cred1 = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id="client1",
            client_secret="secret1",
            redirect_uri="https://example1.com/callback",
        ),
    )

    cred2 = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id="client2",
            client_secret="secret2",
            redirect_uri="https://example2.com/callback",
        ),
    )

    auth_config1 = AuthConfig(
        auth_scheme=oauth2_auth_scheme,
        raw_auth_credential=cred1,
        exchanged_auth_credential=cred1,
        credential_key="key1",
    )

    auth_config2 = AuthConfig(
        auth_scheme=oauth2_auth_scheme,
        raw_auth_credential=cred2,
        exchanged_auth_credential=cred2,
        credential_key="key2",
    )

    # Save both credentials
    await credential_service.save_credential(auth_config1, callback_context)
    await credential_service.save_credential(auth_config2, callback_context)

    # Load and verify both credentials
    result1 = await credential_service.load_credential(
        auth_config1, callback_context
    )
    result2 = await credential_service.load_credential(
        auth_config2, callback_context
    )

    assert result1 is not None
    assert result2 is not None
    assert result1.oauth2.client_id == "client1"
    assert result2.oauth2.client_id == "client2"

  def test_get_bucket_for_current_context_creates_nested_structure(
      self, credential_service, callback_context
  ):
    """Test that _get_bucket_for_current_context creates the proper nested structure."""
    storage = credential_service._get_bucket_for_current_context(
        callback_context
    )

    # Verify the nested structure was created
    assert "test_app" in credential_service._credentials
    assert "test_user" in credential_service._credentials["test_app"]
    assert isinstance(storage, dict)
    assert storage is credential_service._credentials["test_app"]["test_user"]

  def test_get_bucket_for_current_context_reuses_existing(
      self, credential_service, callback_context
  ):
    """Test that _get_bucket_for_current_context reuses existing structure."""
    # Create initial structure
    storage1 = credential_service._get_bucket_for_current_context(
        callback_context
    )
    storage1["test_key"] = "test_value"

    # Get storage again
    storage2 = credential_service._get_bucket_for_current_context(
        callback_context
    )

    # Verify it's the same storage instance
    assert storage1 is storage2
    assert storage2["test_key"] == "test_value"

  def test_get_storage_different_apps(
      self, credential_service, callback_context, another_callback_context
  ):
    """Test that different apps get different storage instances."""
    storage1 = credential_service._get_bucket_for_current_context(
        callback_context
    )
    storage2 = credential_service._get_bucket_for_current_context(
        another_callback_context
    )

    # Verify they are different storage instances
    assert storage1 is not storage2

    # Verify the structure
    assert "test_app" in credential_service._credentials
    assert "another_app" in credential_service._credentials
    assert "test_user" in credential_service._credentials["test_app"]
    assert "another_user" in credential_service._credentials["another_app"]

  @pytest.mark.asyncio
  async def test_same_user_different_apps(
      self, credential_service, auth_config
  ):
    """Test that the same user in different apps get isolated storage."""
    # Create two contexts with same user but different apps
    context1 = Mock(spec=CallbackContext)
    mock_invocation_context1 = Mock()
    mock_invocation_context1.app_name = "app1"
    mock_invocation_context1.user_id = "same_user"
    context1._invocation_context = mock_invocation_context1

    context2 = Mock(spec=CallbackContext)
    mock_invocation_context2 = Mock()
    mock_invocation_context2.app_name = "app2"
    mock_invocation_context2.user_id = "same_user"
    context2._invocation_context = mock_invocation_context2

    # Save credential in first app
    await credential_service.save_credential(auth_config, context1)

    # Try to load from second app
    result = await credential_service.load_credential(auth_config, context2)
    assert result is None

    # Verify first app still has the credential
    result = await credential_service.load_credential(auth_config, context1)
    assert result is not None

  @pytest.mark.asyncio
  async def test_same_app_different_users(
      self, credential_service, auth_config
  ):
    """Test that different users in the same app get isolated storage."""
    # Create two contexts with same app but different users
    context1 = Mock(spec=CallbackContext)
    mock_invocation_context1 = Mock()
    mock_invocation_context1.app_name = "same_app"
    mock_invocation_context1.user_id = "user1"
    context1._invocation_context = mock_invocation_context1

    context2 = Mock(spec=CallbackContext)
    mock_invocation_context2 = Mock()
    mock_invocation_context2.app_name = "same_app"
    mock_invocation_context2.user_id = "user2"
    context2._invocation_context = mock_invocation_context2

    # Save credential for first user
    await credential_service.save_credential(auth_config, context1)

    # Try to load for second user
    result = await credential_service.load_credential(auth_config, context2)
    assert result is None

    # Verify first user still has the credential
    result = await credential_service.load_credential(auth_config, context1)
    assert result is not None
