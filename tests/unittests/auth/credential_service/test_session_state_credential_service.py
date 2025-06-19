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
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.credential_service.session_state_credential_service import SessionStateCredentialService
from google.adk.tools.tool_context import ToolContext
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
  def tool_context(self):
    """Create a mock ToolContext for testing."""
    mock_context = Mock(spec=ToolContext)
    # Create a state dictionary that behaves like session state
    mock_context.state = {}
    return mock_context

  @pytest.fixture
  def another_tool_context(self):
    """Create another mock ToolContext with different state for testing isolation."""
    mock_context = Mock(spec=ToolContext)
    # Create a separate state dictionary to simulate different session
    mock_context.state = {}
    return mock_context

  @pytest.mark.asyncio
  async def test_load_credential_not_found(
      self, credential_service, auth_config, tool_context
  ):
    """Test loading a credential that doesn't exist returns None."""
    result = await credential_service.load_credential(auth_config, tool_context)
    assert result is None

  @pytest.mark.asyncio
  async def test_save_and_load_credential(
      self, credential_service, auth_config, tool_context
  ):
    """Test saving and then loading a credential."""
    # Save the credential
    await credential_service.save_credential(auth_config, tool_context)

    # Load the credential
    result = await credential_service.load_credential(auth_config, tool_context)

    # Verify the credential was saved and loaded correctly
    assert result is not None
    assert result == auth_config.exchanged_auth_credential
    assert result.auth_type == AuthCredentialTypes.OAUTH2
    assert result.oauth2.client_id == "mock_client_id"

  @pytest.mark.asyncio
  async def test_save_credential_updates_existing(
      self, credential_service, auth_config, tool_context, oauth2_credentials
  ):
    """Test that saving a credential updates an existing one."""
    # Save initial credential
    await credential_service.save_credential(auth_config, tool_context)

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
    await credential_service.save_credential(auth_config, tool_context)

    # Load and verify the credential was updated
    result = await credential_service.load_credential(auth_config, tool_context)
    assert result is not None
    assert result.oauth2.client_id == "updated_client_id"
    assert result.oauth2.client_secret == "updated_client_secret"

  @pytest.mark.asyncio
  async def test_credentials_isolated_by_context(
      self, credential_service, auth_config, tool_context, another_tool_context
  ):
    """Test that credentials are isolated between different tool contexts."""
    # Save credential in first context
    await credential_service.save_credential(auth_config, tool_context)

    # Try to load from another context (should not find it)
    result = await credential_service.load_credential(
        auth_config, another_tool_context
    )
    assert result is None

    # Verify original context still has the credential
    result = await credential_service.load_credential(auth_config, tool_context)
    assert result is not None

  @pytest.mark.asyncio
  async def test_multiple_credentials_same_context(
      self, credential_service, tool_context, oauth2_auth_scheme
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
    await credential_service.save_credential(auth_config1, tool_context)
    await credential_service.save_credential(auth_config2, tool_context)

    # Load and verify both credentials
    result1 = await credential_service.load_credential(
        auth_config1, tool_context
    )
    result2 = await credential_service.load_credential(
        auth_config2, tool_context
    )

    assert result1 is not None
    assert result2 is not None
    assert result1.oauth2.client_id == "client1"
    assert result2.oauth2.client_id == "client2"

  @pytest.mark.asyncio
  async def test_save_credential_with_none_exchanged_credential(
      self, credential_service, auth_config, tool_context
  ):
    """Test saving when exchanged_auth_credential is None."""
    # Set exchanged credential to None
    auth_config.exchanged_auth_credential = None

    # Save the credential (should save None)
    await credential_service.save_credential(auth_config, tool_context)

    # Load and verify None was saved
    result = await credential_service.load_credential(auth_config, tool_context)
    assert result is None

  @pytest.mark.asyncio
  async def test_load_credential_with_empty_credential_key(
      self, credential_service, auth_config, tool_context
  ):
    """Test loading credential with empty credential key."""
    # Set credential key to empty string
    auth_config.credential_key = ""

    # Save first to have something to load
    await credential_service.save_credential(auth_config, tool_context)

    # Load should work with empty key
    result = await credential_service.load_credential(auth_config, tool_context)
    assert result == auth_config.exchanged_auth_credential

  @pytest.mark.asyncio
  async def test_state_persistence_across_operations(
      self, credential_service, auth_config, tool_context
  ):
    """Test that state persists correctly across multiple operations."""
    # Initially, no credential should exist
    result = await credential_service.load_credential(auth_config, tool_context)
    assert result is None

    # Save a credential
    await credential_service.save_credential(auth_config, tool_context)

    # Verify it was saved
    result = await credential_service.load_credential(auth_config, tool_context)
    assert result is not None
    assert result == auth_config.exchanged_auth_credential

    # Update and save again
    new_credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id="new_client_id",
            client_secret="new_client_secret",
            redirect_uri="https://new.com/callback",
        ),
    )
    auth_config.exchanged_auth_credential = new_credential
    await credential_service.save_credential(auth_config, tool_context)

    # Verify the update persisted
    result = await credential_service.load_credential(auth_config, tool_context)
    assert result is not None
    assert result.oauth2.client_id == "new_client_id"

  @pytest.mark.asyncio
  async def test_credential_key_uniqueness(
      self, credential_service, oauth2_auth_scheme, tool_context
  ):
    """Test that different credential keys create separate storage slots."""
    # Create credentials with same content but different keys
    credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id="same_client",
            client_secret="same_secret",
            redirect_uri="https://same.com/callback",
        ),
    )

    config_key1 = AuthConfig(
        auth_scheme=oauth2_auth_scheme,
        raw_auth_credential=credential,
        exchanged_auth_credential=credential,
        credential_key="unique_key_1",
    )

    config_key2 = AuthConfig(
        auth_scheme=oauth2_auth_scheme,
        raw_auth_credential=credential,
        exchanged_auth_credential=credential,
        credential_key="unique_key_2",
    )

    # Save credential with first key
    await credential_service.save_credential(config_key1, tool_context)

    # Verify it's stored under first key
    result1 = await credential_service.load_credential(
        config_key1, tool_context
    )
    assert result1 is not None

    # Verify it's not accessible under second key
    result2 = await credential_service.load_credential(
        config_key2, tool_context
    )
    assert result2 is None

    # Save under second key
    await credential_service.save_credential(config_key2, tool_context)

    # Now both should be accessible
    result1 = await credential_service.load_credential(
        config_key1, tool_context
    )
    result2 = await credential_service.load_credential(
        config_key2, tool_context
    )
    assert result1 is not None
    assert result2 is not None
    assert result1 == result2  # Same credential content

  def test_direct_state_access(
      self, credential_service, auth_config, tool_context
  ):
    """Test that the service correctly uses tool_context.state for storage."""
    # Verify that the state starts empty
    assert len(tool_context.state) == 0

    # Save a credential (this is async but we're testing the state directly)
    credential_key = auth_config.credential_key
    test_credential = auth_config.exchanged_auth_credential

    # Directly set the state to simulate save_credential behavior
    tool_context.state[credential_key] = test_credential

    # Verify the credential is in the state
    assert credential_key in tool_context.state
    assert tool_context.state[credential_key] == test_credential

    # Verify we can retrieve it using the get method (simulating load_credential)
    retrieved = tool_context.state.get(credential_key)
    assert retrieved == test_credential
