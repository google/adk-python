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

from unittest.mock import Mock

from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlowAuthorizationCode
from fastapi.openapi.models import OAuthFlows
from google.adk.agents.callback_context import CallbackContext
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.credential_service.session_state_credential_service import SessionStateCredentialService
from google.adk.sessions.state import State
import pytest


class TestSessionStateCredentialService:
  """Tests for the SessionStateCredentialService class."""

  @pytest.fixture
  def credential_service(self):
    """Create a SessionStateCredentialService instance for testing."""
    return SessionStateCredentialService()

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
    # Create a state dictionary that behaves like session state
    mock_context.state = {}
    return mock_context

  @pytest.fixture
  def another_callback_context(self):
    """Create another mock CallbackContext with different state for testing isolation."""
    mock_context = Mock(spec=CallbackContext)
    # Create a separate state dictionary to simulate different session
    mock_context.state = {}
    return mock_context

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
    """Test that credentials are isolated between different callback contexts."""
    # Save credential in first context
    await credential_service.save_credential(auth_config, callback_context)

    # Try to load from another context (should not find it)
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

  @pytest.mark.asyncio
  async def test_save_credential_with_none_exchanged_credential(
      self, credential_service, auth_config, callback_context
  ):
    """Test that saving a credential with None exchanged_auth_credential stores None."""
    # Set exchanged_auth_credential to None
    auth_config.exchanged_auth_credential = None

    # Save the credential
    await credential_service.save_credential(auth_config, callback_context)

    # Load and verify None was stored
    result = await credential_service.load_credential(
        auth_config, callback_context
    )
    assert result is None

  @pytest.mark.asyncio
  async def test_load_credential_with_empty_credential_key(
      self, credential_service, auth_config, callback_context
  ):
    """Test that loading with an empty credential key returns None."""
    # Set credential_key to empty string
    auth_config.credential_key = ""

    # Try to load credential
    result = await credential_service.load_credential(
        auth_config, callback_context
    )
    assert result is None

  @pytest.mark.asyncio
  async def test_state_persistence_across_operations(
      self, credential_service, auth_config, callback_context
  ):
    """Test that state persists across multiple operations."""
    # Save credential
    await credential_service.save_credential(auth_config, callback_context)

    # Verify state contains the credential under secret: prefix
    secret_key = State.SECRET_PREFIX + auth_config.credential_key
    assert secret_key in callback_context.state
    assert (
        callback_context.state[secret_key]
        == auth_config.exchanged_auth_credential
    )

    # Load credential
    result = await credential_service.load_credential(
        auth_config, callback_context
    )
    assert result is not None

    # Verify state still contains the credential
    assert secret_key in callback_context.state
    assert (
        callback_context.state[secret_key]
        == auth_config.exchanged_auth_credential
    )

    # Update credential
    new_credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id="updated_client_id",
            client_secret="updated_client_secret",
            redirect_uri="https://updated.com/callback",
        ),
    )
    auth_config.exchanged_auth_credential = new_credential

    # Save updated credential
    await credential_service.save_credential(auth_config, callback_context)

    # Verify state was updated
    assert callback_context.state[secret_key] == new_credential

  @pytest.mark.asyncio
  async def test_credential_key_uniqueness(
      self, credential_service, oauth2_auth_scheme, callback_context
  ):
    """Test that different credential keys store different credentials."""
    # Create credentials with different keys
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
        credential_key="unique_key_1",
    )

    auth_config2 = AuthConfig(
        auth_scheme=oauth2_auth_scheme,
        raw_auth_credential=cred2,
        exchanged_auth_credential=cred2,
        credential_key="unique_key_2",
    )

    # Save both credentials
    await credential_service.save_credential(auth_config1, callback_context)
    await credential_service.save_credential(auth_config2, callback_context)

    # Verify both exist in state with secret-prefixed keys
    sk1 = State.SECRET_PREFIX + "unique_key_1"
    sk2 = State.SECRET_PREFIX + "unique_key_2"
    assert sk1 in callback_context.state
    assert sk2 in callback_context.state
    assert callback_context.state[sk1] != callback_context.state[sk2]

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

  @pytest.mark.asyncio
  async def test_direct_state_access(
      self, credential_service, auth_config, callback_context
  ):
    """Test that the service properly accesses callback context state."""
    # Directly set a value in state
    test_credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id="direct_client_id",
            client_secret="direct_client_secret",
            redirect_uri="https://direct.com/callback",
        ),
    )
    callback_context.state[State.SECRET_PREFIX + auth_config.credential_key] = (
        test_credential
    )

    # Load using the service
    result = await credential_service.load_credential(
        auth_config, callback_context
    )
    assert result == test_credential

  @pytest.mark.asyncio
  async def test_load_falls_back_to_legacy_unprefixed_key(
      self, credential_service, auth_config, callback_context
  ):
    """Credentials stored under the old unprefixed key are still found."""
    legacy_cred = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id="legacy_client",
            client_secret="legacy_secret",
        ),
    )
    # Simulate a session persisted before the secret: migration
    callback_context.state[auth_config.credential_key] = legacy_cred

    result = await credential_service.load_credential(
        auth_config, callback_context
    )
    assert result is not None
    assert result.oauth2.client_id == "legacy_client"
    # Legacy key should be migrated: secret key populated, legacy cleared
    secret_key = State.SECRET_PREFIX + auth_config.credential_key
    assert secret_key in callback_context.state
    assert callback_context.state[secret_key] == legacy_cred
    assert callback_context.state[auth_config.credential_key] is None

  @pytest.mark.asyncio
  async def test_secret_key_takes_precedence_over_legacy(
      self, credential_service, auth_config, callback_context
  ):
    """When both keys exist, the secret-prefixed key wins."""
    old_cred = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id="old_client",
            client_secret="old_secret",
        ),
    )
    new_cred = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id="new_client",
            client_secret="new_secret",
        ),
    )
    callback_context.state[auth_config.credential_key] = old_cred
    callback_context.state[State.SECRET_PREFIX + auth_config.credential_key] = (
        new_cred
    )

    result = await credential_service.load_credential(
        auth_config, callback_context
    )
    assert result.oauth2.client_id == "new_client"

  @pytest.mark.asyncio
  async def test_explicit_none_secret_key_not_revived_by_legacy(
      self, credential_service, auth_config, callback_context
  ):
    """Explicit None in secret: key must not fall back to legacy key."""
    old_cred = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id="stale_client",
            client_secret="stale_secret",
        ),
    )
    callback_context.state[auth_config.credential_key] = old_cred
    callback_context.state[State.SECRET_PREFIX + auth_config.credential_key] = (
        None
    )

    result = await credential_service.load_credential(
        auth_config, callback_context
    )
    assert result is None
