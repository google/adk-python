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

import asyncio
from contextlib import asynccontextmanager
import time
from typing import Optional
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from authlib.common.errors import AuthlibBaseError
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import HttpAuth
from google.adk.auth.auth_credential import HttpCredentials
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_schemes import AuthScheme
from google.adk.auth.refresher.oauth2_credential_refresher import OAuth2CredentialRefresher
from google.adk.events.event import Event
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.adk.sessions.state import StateSchemaError
from google.adk.tools.openapi_tool.auth.auth_helpers import openid_dict_to_scheme_credential
from google.adk.tools.openapi_tool.auth.auth_helpers import token_to_scheme_credential
from google.adk.tools.openapi_tool.auth.credential_exchangers.auto_auth_credential_exchanger import OAuth2CredentialExchanger
from google.adk.tools.openapi_tool.openapi_spec_parser import tool_auth_handler
from google.adk.tools.openapi_tool.openapi_spec_parser.tool_auth_handler import ToolAuthHandler
from google.adk.tools.openapi_tool.openapi_spec_parser.tool_auth_handler import ToolContextCredentialStore
from google.adk.tools.tool_context import ToolContext
from pydantic import BaseModel
import pytest


# Helper function to create a mock ToolContext
def create_mock_tool_context():
  return ToolContext(
      function_call_id='test-fc-id',
      invocation_context=InvocationContext(
          agent=LlmAgent(name='test'),
          session=Session(app_name='test', user_id='123', id='123'),
          invocation_id='123',
          session_service=InMemorySessionService(),
      ),
  )


# Test cases for OpenID Connect
class MockOpenIdConnectCredentialExchanger(OAuth2CredentialExchanger):

  def __init__(
      self, expected_scheme, expected_credential, expected_access_token
  ):
    self.expected_scheme = expected_scheme
    self.expected_credential = expected_credential
    self.expected_access_token = expected_access_token

  def exchange_credential(
      self,
      auth_scheme: AuthScheme,
      auth_credential: Optional[AuthCredential] = None,
  ) -> AuthCredential:
    if auth_credential.oauth2 and (
        auth_credential.oauth2.auth_response_uri
        or auth_credential.oauth2.auth_code
    ):
      auth_code = (
          auth_credential.oauth2.auth_response_uri
          if auth_credential.oauth2.auth_response_uri
          else auth_credential.oauth2.auth_code
      )
      # Simulate the token exchange
      updated_credential = AuthCredential(
          auth_type=AuthCredentialTypes.HTTP,  # Store as a bearer token
          http=HttpAuth(
              scheme='bearer',
              credentials=HttpCredentials(
                  token=auth_code + self.expected_access_token
              ),
          ),
      )
      return updated_credential

    # simulate the case of getting auth_uri
    return None


def get_mock_openid_scheme_credential():
  config_dict = {
      'authorization_endpoint': 'test.com',
      'token_endpoint': 'test.com',
  }
  scopes = ['test_scope']
  credential_dict = {
      'client_id': '123',
      'client_secret': '456',
      'redirect_uri': 'test.com',
  }
  return openid_dict_to_scheme_credential(config_dict, scopes, credential_dict)


# Fixture for the OpenID Connect security scheme
@pytest.fixture
def openid_connect_scheme():
  scheme, _ = get_mock_openid_scheme_credential()
  return scheme


# Fixture for a base OpenID Connect credential
@pytest.fixture
def openid_connect_credential():
  _, credential = get_mock_openid_scheme_credential()
  return credential


@pytest.mark.asyncio
async def test_openid_connect_no_auth_response(
    openid_connect_scheme, openid_connect_credential
):
  # Setup Mock exchanger
  mock_exchanger = MockOpenIdConnectCredentialExchanger(
      openid_connect_scheme, openid_connect_credential, None
  )
  tool_context = create_mock_tool_context()
  credential_store = ToolContextCredentialStore(tool_context=tool_context)
  handler = ToolAuthHandler(
      tool_context,
      openid_connect_scheme,
      openid_connect_credential,
      credential_exchanger=mock_exchanger,
      credential_store=credential_store,
  )
  result = await handler.prepare_auth_credentials()
  assert result.state == 'pending'
  assert result.auth_credential == openid_connect_credential


@pytest.mark.asyncio
async def test_openid_connect_uses_explicit_credential_key(
    openid_connect_scheme, openid_connect_credential
):
  tool_context = create_mock_tool_context()
  handler = ToolAuthHandler(
      tool_context,
      openid_connect_scheme,
      openid_connect_credential,
      credential_key='my_tool_tokens',
  )
  result = await handler.prepare_auth_credentials()
  assert result.state == 'pending'
  requested = tool_context.actions.requested_auth_configs['test-fc-id']
  assert requested.credential_key == 'my_tool_tokens'


@pytest.mark.asyncio
async def test_openid_connect_with_auth_response(
    openid_connect_scheme, openid_connect_credential, monkeypatch
):
  mock_exchanger = MockOpenIdConnectCredentialExchanger(
      openid_connect_scheme,
      openid_connect_credential,
      'test_access_token',
  )
  tool_context = create_mock_tool_context()

  mock_auth_handler = MagicMock()
  returned_credential = AuthCredential(
      auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
      oauth2=OAuth2Auth(auth_response_uri='test_auth_response_uri'),
  )
  mock_auth_handler.get_auth_response.return_value = returned_credential
  mock_auth_handler_path = 'google.adk.auth.auth_handler.AuthHandler'
  monkeypatch.setattr(
      mock_auth_handler_path, lambda *args, **kwargs: mock_auth_handler
  )

  credential_store = ToolContextCredentialStore(tool_context=tool_context)
  handler = ToolAuthHandler(
      tool_context,
      openid_connect_scheme,
      openid_connect_credential,
      credential_exchanger=mock_exchanger,
      credential_store=credential_store,
  )
  result = await handler.prepare_auth_credentials()
  assert result.state == 'done'
  assert result.auth_credential.auth_type == AuthCredentialTypes.HTTP
  assert 'test_access_token' in result.auth_credential.http.credentials.token
  # Verify that the credential was stored:
  stored_credential = credential_store.get_credential(
      openid_connect_scheme, openid_connect_credential
  )
  assert stored_credential == returned_credential
  mock_auth_handler.get_auth_response.assert_called_once()


@pytest.mark.asyncio
async def test_openid_connect_existing_token(
    openid_connect_scheme, openid_connect_credential
):
  _, existing_credential = token_to_scheme_credential(
      'oauth2Token', 'header', 'bearer', '123123123'
  )
  tool_context = create_mock_tool_context()
  # Store the credential to simulate existing credential
  credential_store = ToolContextCredentialStore(tool_context=tool_context)
  key = credential_store.get_credential_key(
      openid_connect_scheme, openid_connect_credential
  )
  credential_store.store_credential(key, existing_credential)

  handler = ToolAuthHandler(
      tool_context,
      openid_connect_scheme,
      openid_connect_credential,
      credential_store=credential_store,
  )
  result = await handler.prepare_auth_credentials()
  assert result.state == 'done'
  assert result.auth_credential == existing_credential


@patch.object(tool_auth_handler, 'OAuth2CredentialRefresher')
@pytest.mark.asyncio
async def test_openid_connect_existing_oauth2_token_refresh(
    mock_oauth2_refresher, openid_connect_scheme, openid_connect_credential
):
  """Test that OAuth2 tokens are refreshed when existing credentials are found."""
  # Create existing OAuth2 credential
  existing_credential = AuthCredential(
      auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
      oauth2=OAuth2Auth(
          client_id='test_client_id',
          client_secret='test_client_secret',
          access_token='existing_token',
          refresh_token='refresh_token',
      ),
  )

  # Mock the refreshed credential
  refreshed_credential = AuthCredential(
      auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
      oauth2=OAuth2Auth(
          client_id='test_client_id',
          client_secret='test_client_secret',
          access_token='refreshed_token',
          refresh_token='new_refresh_token',
      ),
  )

  # Setup mock OAuth2CredentialRefresher
  from unittest.mock import AsyncMock

  mock_refresher_instance = MagicMock()
  mock_refresher_instance.is_refresh_needed = AsyncMock(return_value=True)
  mock_refresher_instance.refresh = AsyncMock(return_value=refreshed_credential)
  mock_oauth2_refresher.return_value = mock_refresher_instance

  tool_context = create_mock_tool_context()
  credential_store = ToolContextCredentialStore(tool_context=tool_context)

  # Store the existing credential
  key = credential_store.get_credential_key(
      openid_connect_scheme, openid_connect_credential
  )
  credential_store.store_credential(key, existing_credential)

  handler = ToolAuthHandler(
      tool_context,
      openid_connect_scheme,
      openid_connect_credential,
      credential_store=credential_store,
  )

  result = await handler.prepare_auth_credentials()

  # Verify OAuth2CredentialRefresher was called for refresh
  mock_oauth2_refresher.assert_called_once()

  mock_refresher_instance.is_refresh_needed.assert_called_once_with(
      existing_credential
  )
  mock_refresher_instance.refresh.assert_called_once_with(
      existing_credential, openid_connect_scheme
  )

  assert result.state == 'done'
  # The result should contain the refreshed credential after exchange
  assert result.auth_credential is not None


@patch.object(tool_auth_handler, 'OAuth2CredentialRefresher')
@pytest.mark.asyncio
async def test_refreshed_credential_is_persisted_to_store(
    mock_oauth2_refresher, openid_connect_scheme, openid_connect_credential
):
  """Test that refreshed OAuth2 credentials are persisted back to the store."""
  # Create existing OAuth2 credential with an "old" refresh token.
  existing_credential = AuthCredential(
      auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
      oauth2=OAuth2Auth(
          client_id='test_client_id',
          client_secret='test_client_secret',
          access_token='old_access_token',
          refresh_token='old_refresh_token',
      ),
  )

  # The refresher will return a credential with rotated tokens.
  refreshed_credential = AuthCredential(
      auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
      oauth2=OAuth2Auth(
          client_id='test_client_id',
          client_secret='test_client_secret',
          access_token='new_access_token',
          refresh_token='new_refresh_token',
      ),
  )

  mock_refresher_instance = MagicMock()
  mock_refresher_instance.is_refresh_needed = AsyncMock(return_value=True)
  mock_refresher_instance.refresh = AsyncMock(return_value=refreshed_credential)
  mock_oauth2_refresher.return_value = mock_refresher_instance

  tool_context = create_mock_tool_context()
  credential_store = ToolContextCredentialStore(tool_context=tool_context)

  # Store the existing (stale) credential.
  key = credential_store.get_credential_key(
      openid_connect_scheme, openid_connect_credential
  )
  credential_store.store_credential(key, existing_credential)

  handler = ToolAuthHandler(
      tool_context,
      openid_connect_scheme,
      openid_connect_credential,
      credential_store=credential_store,
  )

  await handler.prepare_auth_credentials()

  # The critical assertion: the *refreshed* credential must now be in the
  # store so that the next invocation reads the new tokens, not the old ones.
  persisted = credential_store.get_credential(
      openid_connect_scheme, openid_connect_credential
  )
  assert persisted is not None
  assert persisted.oauth2.access_token == 'new_access_token'
  assert persisted.oauth2.refresh_token == 'new_refresh_token'


def _expired_stored_credential(
    access_token: str = 'old_token',
    refresh_token: str = 'valid_refresh_token',
) -> AuthCredential:
  """Returns a stored credential whose recorded expiry is already past.

  The scheduled refresh in _get_existing_credential only fires for a
  credential that reads as expired.
  """
  return AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id='cid',
          client_secret='csec',
          access_token=access_token,
          refresh_token=refresh_token,
          expires_at=int(time.time()) - 3600,
          expires_in=-3600,
      ),
  )


def _mock_token_endpoint(mock_create_oauth2_session, *responses):
  """Points the refresher at a token endpoint returning the given responses."""
  mock_client = MagicMock()
  mock_client.refresh_token = MagicMock(side_effect=list(responses))
  mock_create_oauth2_session.return_value = (
      mock_client,
      'https://test.com/token',
  )
  return mock_client


@patch(
    'google.adk.auth.refresher.oauth2_credential_refresher.create_oauth2_session'
)
@pytest.mark.asyncio
async def test_scheduled_refresh_keeps_refresh_token_when_omitted(
    mock_create_oauth2_session,
    openid_connect_scheme,
    openid_connect_credential,
):
  """Test that an expiry-driven refresh keeps a refresh token it is not resent.

  update_credential_with_tokens assigns tokens.get("refresh_token")
  unconditionally, so without preservation a provider that does not rotate
  loses it on an ordinary refresh.
  """
  tool_context = create_mock_tool_context()
  handler = ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=openid_connect_scheme,
      auth_credential=openid_connect_credential,
  )
  store = handler.credential_store
  key = store.get_credential_key(
      openid_connect_scheme, openid_connect_credential
  )
  store.store_credential(key, _expired_stored_credential())

  _mock_token_endpoint(
      mock_create_oauth2_session,
      {'access_token': 'new_token', 'expires_in': 3600},
  )

  await handler.prepare_auth_credentials()

  persisted = store.get_credential(
      openid_connect_scheme, openid_connect_credential
  )
  assert persisted.oauth2.access_token == 'new_token'
  assert persisted.oauth2.refresh_token == 'valid_refresh_token'


@patch(
    'google.adk.auth.refresher.oauth2_credential_refresher.create_oauth2_session'
)
@pytest.mark.asyncio
async def test_scheduled_refresh_takes_the_rotated_refresh_token(
    mock_create_oauth2_session,
    openid_connect_scheme,
    openid_connect_credential,
):
  """Test that preservation does not shadow a refresh token that was rotated.

  The old token is invalid the moment a new one is issued, so the response
  always wins when it carries one.
  """
  tool_context = create_mock_tool_context()
  handler = ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=openid_connect_scheme,
      auth_credential=openid_connect_credential,
  )
  store = handler.credential_store
  key = store.get_credential_key(
      openid_connect_scheme, openid_connect_credential
  )
  store.store_credential(key, _expired_stored_credential())

  _mock_token_endpoint(
      mock_create_oauth2_session,
      {
          'access_token': 'new_token',
          'refresh_token': 'rotated_refresh_token',
          'expires_in': 3600,
      },
  )

  await handler.prepare_auth_credentials()

  persisted = store.get_credential(
      openid_connect_scheme, openid_connect_credential
  )
  assert persisted.oauth2.refresh_token == 'rotated_refresh_token'


@patch(
    'google.adk.auth.refresher.oauth2_credential_refresher.create_oauth2_session'
)
@pytest.mark.asyncio
async def test_scheduled_refresh_persists_a_rotation_with_no_new_access_token(
    mock_create_oauth2_session,
    openid_connect_scheme,
    openid_connect_credential,
):
  """Test that a rotation is stored even when the access token is unchanged.

  A provider may rotate the refresh token while echoing the access token back.
  The old refresh token is dead from that moment, so a stored copy that keeps
  it has nothing left to refresh from and the next turn has to ask the user.
  """
  tool_context = create_mock_tool_context()
  handler = ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=openid_connect_scheme,
      auth_credential=openid_connect_credential,
  )
  store = handler.credential_store
  key = store.get_credential_key(
      openid_connect_scheme, openid_connect_credential
  )
  store.store_credential(key, _expired_stored_credential())

  _mock_token_endpoint(
      mock_create_oauth2_session,
      {
          'access_token': 'old_token',
          'refresh_token': 'rotated_refresh_token',
          'expires_in': 3600,
      },
  )

  await handler.prepare_auth_credentials()

  persisted = store.get_credential(
      openid_connect_scheme, openid_connect_credential
  )
  assert persisted.oauth2.refresh_token == 'rotated_refresh_token'


@patch(
    'google.adk.auth.refresher.oauth2_credential_refresher.create_oauth2_session'
)
@pytest.mark.asyncio
async def test_scheduled_refresh_leaves_401_recovery_possible(
    mock_create_oauth2_session,
    openid_connect_scheme,
    openid_connect_credential,
):
  """Test that an ordinary refresh does not disarm recovery from a later 401.

  Expiry-driven refreshes are far more common than revocations, so a scheduled
  refresh that dropped the refresh token would leave handle_unauthorized_error
  with nothing to refresh from long before any 401 arrived, and asking the
  user would be the only way back.
  """
  tool_context = create_mock_tool_context()
  handler = ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=openid_connect_scheme,
      auth_credential=openid_connect_credential,
  )
  store = handler.credential_store
  key = store.get_credential_key(
      openid_connect_scheme, openid_connect_credential
  )
  store.store_credential(key, _expired_stored_credential())

  # Neither response carries a refresh token, as a provider that reuses the
  # existing one does.
  _mock_token_endpoint(
      mock_create_oauth2_session,
      {'access_token': 'second_token', 'expires_in': 3600},
      {'access_token': 'third_token', 'expires_in': 3600},
  )

  await handler.prepare_auth_credentials()
  recovery = await handler.handle_unauthorized_error()

  assert recovery == 'refreshed'
  assert not tool_context.actions.requested_auth_configs
  persisted = store.get_credential(
      openid_connect_scheme, openid_connect_credential
  )
  assert persisted.oauth2.access_token == 'third_token'
  assert persisted.oauth2.refresh_token == 'valid_refresh_token'


@patch(
    'google.adk.auth.refresher.oauth2_credential_refresher.create_oauth2_session'
)
@pytest.mark.asyncio
async def test_refresh_and_store_keeps_a_failed_refresh_out_of_state(
    mock_create_oauth2_session,
    openid_connect_scheme,
    openid_connect_credential,
):
  """Test that a failed refresh does not persist the expiry it forced.

  Storing it would leave the credential permanently marked expired, so every
  later call would attempt a refresh that cannot work.
  """
  tool_context = create_mock_tool_context()
  handler = ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=openid_connect_scheme,
      auth_credential=openid_connect_credential,
  )
  store = handler.credential_store
  key = store.get_credential_key(
      openid_connect_scheme, openid_connect_credential
  )
  expires_at = int(time.time()) + 3600
  credential = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id='cid',
          client_secret='csec',
          access_token='live_token',
          refresh_token='revoked_refresh_token',
          expires_at=expires_at,
          expires_in=3600,
      ),
  )
  store.store_credential(key, credential)

  mock_client = MagicMock()
  mock_client.refresh_token = MagicMock(
      side_effect=AuthlibBaseError('invalid_grant')
  )
  mock_create_oauth2_session.return_value = (
      mock_client,
      'https://test.com/token',
  )

  outcome = await handler._refresh_and_store(credential, force_expiry=True)
  refreshed = outcome.new_access_token

  assert refreshed is False
  persisted = store.get_credential(
      openid_connect_scheme, openid_connect_credential
  )
  assert persisted.oauth2.expires_at == expires_at
  assert persisted.oauth2.access_token == 'live_token'


@patch(
    'google.adk.auth.refresher.oauth2_credential_refresher.create_oauth2_session'
)
@pytest.mark.asyncio
async def test_refresh_and_store_skips_a_credential_that_is_not_expired(
    mock_create_oauth2_session,
    openid_connect_scheme,
    openid_connect_credential,
):
  """Test that without a forced expiry the recorded expiry decides."""
  tool_context = create_mock_tool_context()
  handler = ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=openid_connect_scheme,
      auth_credential=openid_connect_credential,
  )
  credential = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id='cid',
          client_secret='csec',
          access_token='live_token',
          refresh_token='valid_refresh_token',
          expires_at=int(time.time()) + 3600,
          expires_in=3600,
      ),
  )

  outcome = await handler._refresh_and_store(credential)
  returned, refreshed = outcome.credential, outcome.new_access_token

  assert refreshed is False
  assert returned is credential
  mock_create_oauth2_session.assert_not_called()


class _SchemaWithoutCredentialKey(BaseModel):
  """A state schema that does not declare the credential key."""


@patch(
    'google.adk.auth.refresher.oauth2_credential_refresher.create_oauth2_session'
)
@pytest.mark.asyncio
async def test_a_rejected_credential_write_is_not_reported_as_a_failed_refresh(
    mock_create_oauth2_session,
    openid_connect_scheme,
    openid_connect_credential,
):
  """Test that a store failure surfaces rather than looking like a bad refresh.

  An app that declares a state_schema rejects the credential key, which carries
  no colon and so is not exempt from validation. Catching that alongside the
  refresh would report an already spent refresh token as a refresh failure and
  ask the user to re-authorize, hiding the real error. Under refresh token
  rotation the old token is dead by then and the new credential was never
  persisted.
  """
  tool_context = create_mock_tool_context()
  handler = ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=openid_connect_scheme,
      auth_credential=openid_connect_credential,
  )
  store = handler.credential_store
  key = store.get_credential_key(
      openid_connect_scheme, openid_connect_credential
  )
  store.store_credential(key, _expired_stored_credential())
  # Declared only once the credential is seeded, so the rejection lands on the
  # write that follows the refresh rather than on the setup.
  tool_context.state._schema = _SchemaWithoutCredentialKey

  _mock_token_endpoint(
      mock_create_oauth2_session,
      {'access_token': 'new_token', 'expires_in': 3600},
  )

  with pytest.raises(StateSchemaError):
    await handler.handle_unauthorized_error()

  # The refresh worked, so the user must not have been asked to re-authorize.
  assert not tool_context.actions.requested_auth_configs


def test_credential_key_is_stable_across_redirect_uri():
  """get_credential_key should be invariant under redirect_uri changes.

  redirect_uri is deployment configuration (which callback URL the auth
  server should redirect to), not part of the credential identity. Two
  AuthCredential instances that share the same client_id, client_secret,
  and scopes but differ only in redirect_uri should produce the same key.
  """
  scheme, _ = get_mock_openid_scheme_credential()
  credential_local = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id='client',
          client_secret='secret',
          redirect_uri='http://localhost:8001/oauth2callback',
      ),
  )
  credential_deployed = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id='client',
          client_secret='secret',
          redirect_uri='https://deployed.example.com/oauth2callback',
      ),
  )
  store = ToolContextCredentialStore(tool_context=create_mock_tool_context())

  assert store.get_credential_key(
      scheme, credential_local
  ) == store.get_credential_key(scheme, credential_deployed)


def test_legacy_credential_key_is_stable_across_redirect_uri():
  """_get_legacy_credential_key should be invariant under redirect_uri changes.

  The same redirect_uri-strip behavior must apply to the legacy key path so
  that already-stored credentials remain findable after the fix.
  """
  scheme, _ = get_mock_openid_scheme_credential()
  credential_local = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id='client',
          client_secret='secret',
          redirect_uri='http://localhost:8001/oauth2callback',
      ),
  )
  credential_deployed = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id='client',
          client_secret='secret',
          redirect_uri='https://deployed.example.com/oauth2callback',
      ),
  )
  store = ToolContextCredentialStore(tool_context=create_mock_tool_context())

  assert store._get_legacy_credential_key(
      scheme, credential_local
  ) == store._get_legacy_credential_key(scheme, credential_deployed)


def test_legacy_credential_migration(
    openid_connect_scheme, openid_connect_credential
):
  """Test that credentials stored under legacy keys are migrated to new keys."""
  tool_context = create_mock_tool_context()
  store = ToolContextCredentialStore(tool_context=tool_context)

  legacy_key = store._get_legacy_credential_key(
      openid_connect_scheme, openid_connect_credential
  )
  new_key = store.get_credential_key(
      openid_connect_scheme, openid_connect_credential
  )
  assert legacy_key != new_key

  legacy_credential = AuthCredential(
      auth_type=AuthCredentialTypes.HTTP,
      http=HttpAuth(
          scheme='bearer',
          credentials=HttpCredentials(token='legacy_token'),
      ),
  )
  store.store_credential(legacy_key, legacy_credential)

  assert new_key not in tool_context.state

  retrieved = store.get_credential(
      openid_connect_scheme, openid_connect_credential
  )

  assert retrieved == legacy_credential
  assert new_key in tool_context.state
  assert tool_context.state[new_key] == legacy_credential.model_dump(
      exclude_none=True
  )


def test_remove_credential_with_state_object(
    openid_connect_scheme, openid_connect_credential
):
  """Test that remove_credential sets the key to None in state and evicts credential."""
  tool_context = create_mock_tool_context()
  store = ToolContextCredentialStore(tool_context=tool_context)
  key = store.get_credential_key(
      openid_connect_scheme, openid_connect_credential
  )

  credential = AuthCredential(
      auth_type=AuthCredentialTypes.HTTP,
      http=HttpAuth(
          scheme='bearer',
          credentials=HttpCredentials(token='test_token'),
      ),
  )
  store.store_credential(key, credential)
  assert key in tool_context.state
  assert (
      store.get_credential(openid_connect_scheme, openid_connect_credential)
      is not None
  )

  store.remove_credential(key)
  assert tool_context.state.get(key) is None
  assert (
      store.get_credential(openid_connect_scheme, openid_connect_credential)
      is None
  )


def test_remove_credential_absent_key_is_a_noop(
    openid_connect_scheme, openid_connect_credential
):
  """Test that removing a key that was never stored does not create it."""
  tool_context = create_mock_tool_context()
  store = ToolContextCredentialStore(tool_context=tool_context)
  key = store.get_credential_key(
      openid_connect_scheme, openid_connect_credential
  )
  assert key not in tool_context.state

  store.remove_credential(key)

  assert key not in tool_context.state
  assert key not in tool_context.actions.state_delta


def test_evict_credentials_clears_both_hashed_and_legacy_keys(
    openid_connect_scheme, openid_connect_credential
):
  """Test that _evict_credentials sets both hashed and legacy keys to None."""
  tool_context = create_mock_tool_context()
  handler = ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=openid_connect_scheme,
      auth_credential=openid_connect_credential,
  )
  store = handler.credential_store

  new_key = store.get_credential_key(
      openid_connect_scheme, openid_connect_credential
  )
  legacy_key = store._get_legacy_credential_key(
      openid_connect_scheme, openid_connect_credential
  )

  # Seed state with credentials under both keys
  credential_data = {'dummy': 'token'}
  tool_context.state[new_key] = credential_data
  tool_context.state[legacy_key] = credential_data

  assert tool_context.state.get(new_key) == credential_data
  assert tool_context.state.get(legacy_key) == credential_data

  handler._evict_credentials()

  assert tool_context.state.get(new_key) is None
  assert tool_context.state.get(legacy_key) is None


@pytest.mark.asyncio
async def test_forced_expiry_evaluated_as_expired_by_real_refresher():
  """Test that setting expires_at=1 is treated as expired by real OAuth2CredentialRefresher & Authlib."""
  refresher = OAuth2CredentialRefresher()
  cred = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id='cid',
          client_secret='csec',
          access_token='tok',
          refresh_token='ref',
          expires_at=1,
          expires_in=0,
      ),
  )
  assert await refresher.is_refresh_needed(cred) is True


@patch(
    'google.adk.auth.refresher.oauth2_credential_refresher.create_oauth2_session'
)
@pytest.mark.asyncio
async def test_handle_unauthorized_error_forced_expiry_reaches_token_endpoint(
    mock_create_oauth2_session,
    openid_connect_scheme,
    openid_connect_credential,
):
  """Test that the real refresher contacts the token endpoint after a 401.

  The refresher is not mocked here; only the network boundary is. The stored
  token is still valid by its own expires_at, so the forced expiry is the only
  reason authlib can report it as expired, and the token request proves it did.
  """
  tool_context = create_mock_tool_context()
  handler = ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=openid_connect_scheme,
      auth_credential=openid_connect_credential,
  )
  store = handler.credential_store
  key = store.get_credential_key(
      openid_connect_scheme, openid_connect_credential
  )

  unexpired_cred = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id='cid',
          client_secret='csec',
          access_token='old_token',
          refresh_token='valid_refresh_token',
          expires_at=int(time.time()) + 3600,
          expires_in=3600,
      ),
  )
  store.store_credential(key, unexpired_cred)

  mock_client = MagicMock()
  mock_client.refresh_token = MagicMock(
      return_value={
          'access_token': 'new_token',
          'refresh_token': 'valid_refresh_token',
          'expires_at': int(time.time()) + 3600,
          'expires_in': 3600,
      }
  )

  # Recorded when the session is built: the refresher mutates this same
  # credential once the new token arrives, so reading it afterwards would show
  # the refreshed expiry instead of the forced one.
  forced_expiry = {}

  def fake_create_oauth2_session(auth_scheme, auth_credential):
    del auth_scheme  # Unused.
    forced_expiry['expires_at'] = auth_credential.oauth2.expires_at
    forced_expiry['expires_in'] = auth_credential.oauth2.expires_in
    return mock_client, 'https://test.com/token'

  mock_create_oauth2_session.side_effect = fake_create_oauth2_session

  recovery = await handler.handle_unauthorized_error()

  assert recovery == 'refreshed'
  mock_client.refresh_token.assert_called_once_with(
      url='https://test.com/token', refresh_token='valid_refresh_token'
  )
  # The credential handed to the session carries the forced expiry. It has to
  # be truthy, not merely non-None: authlib < 1.7 returns None from
  # is_expired() for a falsy expires_at, so 0 would skip the refresh on the
  # oldest version the authlib>=1.6.6,<2 pin allows.
  assert forced_expiry == {'expires_at': 1, 'expires_in': 0}
  updated = store.get_credential(
      openid_connect_scheme, openid_connect_credential
  )
  assert updated.oauth2.access_token == 'new_token'


@patch(
    'google.adk.auth.refresher.oauth2_credential_refresher.create_oauth2_session'
)
@pytest.mark.asyncio
async def test_handle_unauthorized_error_keeps_refresh_token_when_omitted(
    mock_create_oauth2_session,
    openid_connect_scheme,
    openid_connect_credential,
):
  """Test that a refresh response without a refresh token keeps the old one.

  A provider that does not rotate refresh tokens omits the field, and RFC 6749
  section 6 keeps the submitted one valid until a new one is issued. Persisting
  the credential without it would leave the next 401 unable to refresh.
  """
  tool_context = create_mock_tool_context()
  handler = ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=openid_connect_scheme,
      auth_credential=openid_connect_credential,
  )
  store = handler.credential_store
  key = store.get_credential_key(
      openid_connect_scheme, openid_connect_credential
  )

  store.store_credential(
      key,
      AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(
              client_id='cid',
              client_secret='csec',
              access_token='old_token',
              refresh_token='valid_refresh_token',
              expires_at=int(time.time()) + 3600,
              expires_in=3600,
          ),
      ),
  )

  # The token endpoint answers without a refresh_token, as a provider that
  # reuses the existing one does.
  mock_client = MagicMock()
  mock_client.refresh_token = MagicMock(
      return_value={'access_token': 'new_token', 'expires_in': 3600}
  )
  mock_create_oauth2_session.return_value = (
      mock_client,
      'https://test.com/token',
  )

  recovery = await handler.handle_unauthorized_error()

  assert recovery == 'refreshed'
  updated = store.get_credential(
      openid_connect_scheme, openid_connect_credential
  )
  assert updated.oauth2.access_token == 'new_token'
  assert updated.oauth2.refresh_token == 'valid_refresh_token'


@patch(
    'google.adk.auth.refresher.oauth2_credential_refresher.create_oauth2_session'
)
@pytest.mark.asyncio
async def test_real_refresher_skips_token_endpoint_without_forced_expiry(
    mock_create_oauth2_session, openid_connect_scheme
):
  """Test that an unexpired credential never reaches the token endpoint.

  This is the control for the forced expiry: without it authlib reports the
  token as valid and the refresher returns the cached credential untouched.
  """
  refresher = OAuth2CredentialRefresher()
  unexpired_cred = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id='cid',
          client_secret='csec',
          access_token='old_token',
          refresh_token='valid_refresh_token',
          expires_at=int(time.time()) + 3600,
          expires_in=3600,
      ),
  )

  refreshed = await refresher.refresh(unexpired_cred, openid_connect_scheme)

  mock_create_oauth2_session.assert_not_called()
  assert refreshed.oauth2.access_token == 'old_token'


@patch(
    'google.adk.auth.refresher.oauth2_credential_refresher.create_oauth2_session'
)
@pytest.mark.asyncio
async def test_handle_unauthorized_error_fallback_to_eviction(
    mock_create_oauth2_session,
    openid_connect_scheme,
    openid_connect_credential,
):
  """Test that handle_unauthorized_error evicts the credential if refresh fails.

  The refresher is real; the token request itself fails, as it would for a
  refresh token the provider has revoked.
  """
  tool_context = create_mock_tool_context()
  handler = ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=openid_connect_scheme,
      auth_credential=openid_connect_credential,
  )
  store = handler.credential_store
  key = store.get_credential_key(
      openid_connect_scheme, openid_connect_credential
  )

  dead_cred = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id='cid',
          client_secret='csec',
          access_token='dead_token',
          refresh_token='revoked_refresh_token',
      ),
  )
  store.store_credential(key, dead_cred)
  assert key in tool_context.state

  mock_client = MagicMock()
  mock_client.refresh_token = MagicMock(
      side_effect=AuthlibBaseError('invalid_grant')
  )
  mock_create_oauth2_session.return_value = (
      mock_client,
      'https://test.com/token',
  )

  recovery = await handler.handle_unauthorized_error()
  assert recovery == 'reauth_requested'

  assert tool_context.state.get(key) is None
  assert (
      store.get_credential(openid_connect_scheme, openid_connect_credential)
      is None
  )
  # The user was asked to re-authorize as part of the same call.
  assert 'test-fc-id' in tool_context.actions.requested_auth_configs


@pytest.mark.asyncio
async def test_handle_unauthorized_error_returns_failed_when_reauth_impossible(
    openid_connect_scheme,
):
  """Test that handle_unauthorized_error reports failure if reauth cannot be requested."""
  tool_context = create_mock_tool_context()
  incomplete_credential = AuthCredential(
      auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
      oauth2=OAuth2Auth(client_id='cid'),
  )
  handler = ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=openid_connect_scheme,
      auth_credential=incomplete_credential,
  )
  store = handler.credential_store
  key = store.get_credential_key(openid_connect_scheme, incomplete_credential)
  # No refresh token, so the reactive refresh is skipped and the call falls
  # through to the re-authorization request.
  store.store_credential(
      key,
      AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(client_id='cid', access_token='stored_token'),
      ),
  )

  # client_secret is missing, so _request_credential raises.
  recovery = await handler.handle_unauthorized_error()
  assert recovery == 'failed'
  assert not tool_context.actions.requested_auth_configs
  # Nothing is pending to replace it, so the stored credential has to survive.
  # Dropping it here would send the next call down the no-credential path in
  # prepare_auth_credentials, where the same validation error is raised rather
  # than reported.
  stored = store.get_credential(openid_connect_scheme, incomplete_credential)
  assert stored is not None
  assert stored.oauth2.access_token == 'stored_token'


@pytest.mark.asyncio
async def test_handle_unauthorized_error_non_oauth_scheme_is_not_recoverable():
  """Test that a 401 on an apiKey scheme fails without touching session state.

  Asking the user to re-authorize is meaningless for a static API key, so the
  caller must be left to surface the API error exactly as it did before 401
  handling existed.
  """
  tool_context = create_mock_tool_context()
  api_key_scheme, api_key_credential = token_to_scheme_credential(
      'apikey', 'header', 'X-API-Key', 'a_static_key'
  )
  handler = ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=api_key_scheme,
      auth_credential=api_key_credential,
  )
  store = handler.credential_store
  key = store.get_credential_key(api_key_scheme, api_key_credential)
  store.store_credential(key, api_key_credential)

  recovery = await handler.handle_unauthorized_error()

  assert recovery == 'failed'
  assert not tool_context.actions.requested_auth_configs
  # The stored credential is a copy of the tool's own static configuration,
  # so evicting it would achieve nothing; it is left untouched.
  assert tool_context.state.get(key) is not None


@patch(
    'google.adk.auth.refresher.oauth2_credential_refresher.create_oauth2_session'
)
@pytest.mark.asyncio
async def test_handle_unauthorized_error_unchanged_token_is_not_a_refresh(
    mock_create_oauth2_session,
    openid_connect_scheme,
    openid_connect_credential,
):
  """Test that a refresh returning the same access token is treated as failure.

  A provider that echoes the token back has not given us anything new, so
  retrying the API call would 401 again. Recovery has to fall through to
  re-authorization instead.
  """
  tool_context = create_mock_tool_context()
  handler = ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=openid_connect_scheme,
      auth_credential=openid_connect_credential,
  )
  store = handler.credential_store
  key = store.get_credential_key(
      openid_connect_scheme, openid_connect_credential
  )
  store.store_credential(
      key,
      AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(
              client_id='cid',
              client_secret='csec',
              access_token='stale_token',
              refresh_token='valid_refresh_token',
          ),
      ),
  )

  mock_client = MagicMock()
  mock_client.refresh_token = MagicMock(
      return_value={
          'access_token': 'stale_token',
          'refresh_token': 'valid_refresh_token',
          'expires_at': int(time.time()) + 3600,
          'expires_in': 3600,
      }
  )
  mock_create_oauth2_session.return_value = (
      mock_client,
      'https://test.com/token',
  )

  recovery = await handler.handle_unauthorized_error()

  assert recovery == 'reauth_requested'
  assert tool_context.state.get(key) is None
  assert 'test-fc-id' in tool_context.actions.requested_auth_configs


@patch(
    'google.adk.auth.refresher.oauth2_credential_refresher.create_oauth2_session'
)
@pytest.mark.asyncio
async def test_handle_unauthorized_error_rotation_alone_is_not_a_refresh(
    mock_create_oauth2_session,
    openid_connect_scheme,
    openid_connect_credential,
):
  """Test that a rotated refresh token alone does not license a retry.

  The rotation is worth storing, but the API rejected this access token and
  the provider has handed back the same one, so repeating the call would earn
  the same 401. Recovery has to fall through to re-authorization.
  """
  tool_context = create_mock_tool_context()
  handler = ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=openid_connect_scheme,
      auth_credential=openid_connect_credential,
  )
  store = handler.credential_store
  key = store.get_credential_key(
      openid_connect_scheme, openid_connect_credential
  )
  store.store_credential(
      key,
      AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(
              client_id='cid',
              client_secret='csec',
              access_token='stale_token',
              refresh_token='valid_refresh_token',
          ),
      ),
  )

  _mock_token_endpoint(
      mock_create_oauth2_session,
      {
          'access_token': 'stale_token',
          'refresh_token': 'rotated_refresh_token',
          'expires_at': int(time.time()) + 3600,
          'expires_in': 3600,
      },
  )

  recovery = await handler.handle_unauthorized_error()

  assert recovery == 'reauth_requested'
  refresh_budget_key = handler._refresh_budget_key()
  assert tool_context.state[refresh_budget_key] == 1


@patch(
    'google.adk.auth.refresher.oauth2_credential_refresher.create_oauth2_session'
)
@pytest.mark.asyncio
async def test_handle_unauthorized_error_without_refresh_token_skips_refresh(
    mock_create_oauth2_session,
    openid_connect_scheme,
    openid_connect_credential,
):
  """Test that a credential with no refresh token goes straight to reauth."""
  tool_context = create_mock_tool_context()
  handler = ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=openid_connect_scheme,
      auth_credential=openid_connect_credential,
  )
  store = handler.credential_store
  key = store.get_credential_key(
      openid_connect_scheme, openid_connect_credential
  )
  store.store_credential(
      key,
      AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(
              client_id='cid',
              client_secret='csec',
              access_token='stale_token',
          ),
      ),
  )

  recovery = await handler.handle_unauthorized_error()

  assert recovery == 'reauth_requested'
  # There is nothing to refresh with, so the token endpoint is never contacted.
  mock_create_oauth2_session.assert_not_called()
  assert tool_context.state.get(key) is None
  assert 'test-fc-id' in tool_context.actions.requested_auth_configs


def _stale_credential(access_token: str = 'stale_token') -> AuthCredential:
  """Returns a stored credential with no refresh token.

  Recovery from a 401 then has to fall through to re-authorization, which is
  the path the re-authorization budget guards.
  """
  return AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id='cid', client_secret='csec', access_token=access_token
      ),
  )


def test_claim_recovery_is_exclusive_per_credential(
    openid_connect_scheme, openid_connect_credential
):
  """Test that only one holder at a time may recover a given credential."""
  tool_context = create_mock_tool_context()
  handler = ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=openid_connect_scheme,
      auth_credential=openid_connect_credential,
  )

  with handler.claim_recovery() as claimed:
    assert claimed is True
    with handler.claim_recovery() as claimed_again:
      assert claimed_again is False

  with handler.claim_recovery() as claimed_after_release:
    assert claimed_after_release is True


def test_claim_recovery_is_released_when_the_body_raises(
    openid_connect_scheme, openid_connect_credential
):
  """Test that a recovery that blows up leaves the credential recoverable.

  Releasing in a finally is what the context manager buys over the caller
  remembering to do it, so the failure path is the one worth pinning.
  """
  tool_context = create_mock_tool_context()
  handler = ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=openid_connect_scheme,
      auth_credential=openid_connect_credential,
  )

  with pytest.raises(RuntimeError):
    with handler.claim_recovery():
      raise RuntimeError('recovery failed')

  with handler.claim_recovery() as claimed:
    assert claimed is True


def test_a_refused_claim_does_not_release_the_holders(
    openid_connect_scheme, openid_connect_credential
):
  """Test that a caller turned away cannot free the claim it did not take.

  The refused caller still leaves its with block, and releasing there would
  hand the credential to a third caller while the holder is still recovering.
  """
  tool_context = create_mock_tool_context()
  holder = ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=openid_connect_scheme,
      auth_credential=openid_connect_credential,
  )
  refused = ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=openid_connect_scheme,
      auth_credential=openid_connect_credential,
  )

  with holder.claim_recovery() as claimed:
    assert claimed is True
    with refused.claim_recovery() as refused_claim:
      assert refused_claim is False

    with refused.claim_recovery() as still_refused:
      assert still_refused is False


@pytest.mark.asyncio
async def test_only_one_concurrent_caller_claims_recovery(
    openid_connect_scheme, openid_connect_credential
):
  """Test that callers racing for one credential produce exactly one winner.

  Sequential calls would pass even if an await crept in between reading the
  flag and writing it, which is the regression that would let two callers
  recover the same credential at once.
  """
  tool_context = create_mock_tool_context()
  handlers = [
      ToolAuthHandler.from_tool_context(
          tool_context=tool_context,
          auth_scheme=openid_connect_scheme,
          auth_credential=openid_connect_credential,
      )
      for _ in range(5)
  ]

  async def claim(handler):
    with handler.claim_recovery() as claimed:
      # Held across a suspension point, as the real caller holds it while it
      # awaits the recovery and the retried call. Without one each coroutine
      # would run to completion and release before the next was scheduled,
      # and every one of them would win.
      await asyncio.sleep(0)
      return claimed

  results = await asyncio.gather(*(claim(handler) for handler in handlers))

  assert results.count(True) == 1


def test_claim_recovery_is_shared_across_tools(
    openid_connect_scheme, openid_connect_credential
):
  """Test that sibling tools sharing one credential share one lock.

  The lock is keyed by credential rather than by tool so that two tools of the
  same toolset cannot recover the same credential at once.
  """
  tool_context = create_mock_tool_context()
  first_tool = ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=openid_connect_scheme,
      auth_credential=openid_connect_credential,
  )
  sibling_tool = ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=openid_connect_scheme,
      auth_credential=openid_connect_credential,
  )

  with first_tool.claim_recovery() as claimed:
    assert claimed is True
    with sibling_tool.claim_recovery() as sibling_claim:
      assert sibling_claim is False


def test_claim_recovery_is_independent_per_credential(
    openid_connect_scheme, openid_connect_credential
):
  """Test that recovering one credential does not block an unrelated one."""
  tool_context = create_mock_tool_context()
  handler = ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=openid_connect_scheme,
      auth_credential=openid_connect_credential,
  )
  other_credential = AuthCredential(
      auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
      oauth2=OAuth2Auth(client_id='other_cid', client_secret='other_csec'),
  )
  other_handler = ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=openid_connect_scheme,
      auth_credential=other_credential,
  )

  with handler.claim_recovery() as claimed:
    assert claimed is True
    with other_handler.claim_recovery() as other_claim:
      assert other_claim is True


@pytest.mark.asyncio
async def test_handle_unauthorized_error_stops_asking_once_budget_is_spent(
    openid_connect_scheme, openid_connect_credential
):
  """Test that an endpoint returning 401 forever asks the user only once.

  The user has already re-authorized and the API still rejects the credential,
  so a second request would ask for exactly what was just granted.
  """
  tool_context = create_mock_tool_context()
  handler = ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=openid_connect_scheme,
      auth_credential=openid_connect_credential,
  )
  store = handler.credential_store
  key = store.get_credential_key(
      openid_connect_scheme, openid_connect_credential
  )
  store.store_credential(key, _stale_credential())

  assert await handler.handle_unauthorized_error() == 'reauth_requested'
  assert 'test-fc-id' in tool_context.actions.requested_auth_configs

  # The user re-authorized, so a new credential was stored, but the API
  # rejects that one too.
  tool_context.actions.requested_auth_configs.clear()
  store.store_credential(key, _stale_credential('freshly_granted_token'))

  assert await handler.handle_unauthorized_error() == 'reauth_limit_reached'
  assert not tool_context.actions.requested_auth_configs
  # The credential is deliberately left in place. Evicting it would send the
  # next call down the no-credential path in prepare_auth_credentials, which
  # asks unconditionally and would reopen the loop the budget just closed.
  stored = store.get_credential(
      openid_connect_scheme, openid_connect_credential
  )
  assert stored is not None
  assert stored.oauth2.access_token == 'freshly_granted_token'


@pytest.mark.asyncio
async def test_note_successful_call_refills_the_reauth_budget(
    openid_connect_scheme, openid_connect_credential
):
  """Test that a working call lets the user be asked again later.

  Without this the credential would be barred from re-authorization for the
  rest of the session, including for an ordinary token expiry hours later.
  """
  tool_context = create_mock_tool_context()
  handler = ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=openid_connect_scheme,
      auth_credential=openid_connect_credential,
  )
  store = handler.credential_store
  key = store.get_credential_key(
      openid_connect_scheme, openid_connect_credential
  )
  store.store_credential(key, _stale_credential())
  assert await handler.handle_unauthorized_error() == 'reauth_requested'

  handler.note_successful_call()

  store.store_credential(key, _stale_credential())
  assert await handler.handle_unauthorized_error() == 'reauth_requested'


def test_note_successful_call_writes_nothing_when_budget_is_unspent(
    openid_connect_scheme, openid_connect_credential
):
  """Test that an ordinary successful call emits no state delta.

  note_successful_call runs after every request, so it must stay silent in the
  common case where the user was never asked.
  """
  tool_context = create_mock_tool_context()
  handler = ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=openid_connect_scheme,
      auth_credential=openid_connect_credential,
  )

  handler.note_successful_call()

  assert handler._reauth_budget_key() not in tool_context.state


@pytest.mark.asyncio
async def test_the_lock_is_scoped_to_the_invocation_but_the_budget_is_not(
    openid_connect_scheme, openid_connect_credential
):
  """Test which pieces of 401 bookkeeping session services are asked to keep.

  The lock guards a single call chain, so it belongs in the temp namespace the
  session services discard, where a process that dies mid-recovery cannot leave
  it set. The budget has to outlast the turn it was spent in, because the loop
  it bounds is the user being asked again on the next turn.
  """
  tool_context = create_mock_tool_context()
  handler = ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=openid_connect_scheme,
      auth_credential=openid_connect_credential,
  )
  store = handler.credential_store
  credential_key = store.get_credential_key(
      openid_connect_scheme, openid_connect_credential
  )
  store.store_credential(credential_key, _stale_credential())

  with handler.claim_recovery():
    assert await handler.handle_unauthorized_error() == 'reauth_requested'

  assert handler._recovery_lock_key().startswith('temp:')
  outlives_the_invocation = {
      key
      for key in tool_context.actions.state_delta
      if not key.startswith('temp:')
  }
  assert outlives_the_invocation == {
      credential_key,
      handler._reauth_budget_key(),
  }


@patch(
    'google.adk.auth.refresher.oauth2_credential_refresher.create_oauth2_session'
)
@pytest.mark.asyncio
async def test_a_refresh_that_did_not_help_is_not_tried_again(
    mock_create_oauth2_session,
    openid_connect_scheme,
    openid_connect_credential,
):
  """Test that a healthy token endpoint cannot keep recovery busy forever.

  An authorization server that issues a new access token on every refresh, for
  an API that rejects every one of them, would otherwise be refreshed once per
  call for the rest of the session: the refresh reports success each time, so
  recovery returns early and never reaches a path that gives up.
  """
  tool_context = create_mock_tool_context()
  handler = ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=openid_connect_scheme,
      auth_credential=openid_connect_credential,
  )
  store = handler.credential_store
  key = store.get_credential_key(
      openid_connect_scheme, openid_connect_credential
  )
  store.store_credential(key, _expired_stored_credential())
  token_endpoint = _mock_token_endpoint(
      mock_create_oauth2_session,
      {'access_token': 'fresh_1', 'expires_in': 3600},
      {'access_token': 'fresh_2', 'expires_in': 3600},
  )

  assert await handler.handle_unauthorized_error() == 'refreshed'
  assert await handler.handle_unauthorized_error() == 'failed'

  # The second 401 cost no token endpoint round trip, and under refresh token
  # rotation no refresh token either.
  assert token_endpoint.refresh_token.call_count == 1
  # Nor did it interrupt the user. A provider still willing to issue tokens has
  # not withdrawn the grant, so a new one would be granted just as readily and
  # refused just as readily.
  assert not tool_context.actions.requested_auth_configs


@patch(
    'google.adk.auth.refresher.oauth2_credential_refresher.create_oauth2_session'
)
@pytest.mark.asyncio
async def test_note_successful_call_refills_the_refresh_budget(
    mock_create_oauth2_session,
    openid_connect_scheme,
    openid_connect_credential,
):
  """Test that a working call lets a later 401 be refreshed again.

  Without the refill, one unhelpful refresh would bar the credential from
  reactive refreshing for the rest of the session, including for an ordinary
  token expiry hours later.
  """
  tool_context = create_mock_tool_context()
  handler = ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=openid_connect_scheme,
      auth_credential=openid_connect_credential,
  )
  store = handler.credential_store
  key = store.get_credential_key(
      openid_connect_scheme, openid_connect_credential
  )
  store.store_credential(key, _expired_stored_credential())
  token_endpoint = _mock_token_endpoint(
      mock_create_oauth2_session,
      {'access_token': 'fresh_1', 'expires_in': 3600},
      {'access_token': 'fresh_2', 'expires_in': 3600},
  )
  assert await handler.handle_unauthorized_error() == 'refreshed'

  handler.note_successful_call()

  assert await handler.handle_unauthorized_error() == 'refreshed'
  assert token_endpoint.refresh_token.call_count == 2


@patch(
    'google.adk.auth.refresher.oauth2_credential_refresher.create_oauth2_session'
)
@pytest.mark.asyncio
async def test_a_spent_refresh_budget_does_not_bar_reauthorization(
    mock_create_oauth2_session,
    openid_connect_scheme,
    openid_connect_credential,
):
  """Test that giving up on refreshing does not also give up on asking.

  The refresh budget governs refreshing only. A credential left with nothing to
  refresh from has no refresh to skip, and asking the user is the one recovery
  still open to it, so it must not be held back by what earlier refreshes cost.
  """
  tool_context = create_mock_tool_context()
  handler = ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=openid_connect_scheme,
      auth_credential=openid_connect_credential,
  )
  store = handler.credential_store
  key = store.get_credential_key(
      openid_connect_scheme, openid_connect_credential
  )
  store.store_credential(key, _expired_stored_credential())
  _mock_token_endpoint(
      mock_create_oauth2_session,
      {'access_token': 'fresh_1', 'expires_in': 3600},
  )
  assert await handler.handle_unauthorized_error() == 'refreshed'

  # The provider has since invalidated the refresh token, so what comes back
  # from the store has nothing left to refresh from.
  store.store_credential(key, _stale_credential())

  assert await handler.handle_unauthorized_error() == 'reauth_requested'


_MULTI_TURN_APP = 'multi_turn_app'
_MULTI_TURN_USER = 'multi_turn_user'
_MULTI_TURN_SESSION = 'multi_turn_session'


@pytest.fixture(params=['in_memory', 'sqlite'])
def multi_turn_session_service(request, tmp_path):
  """Yields one of each kind of session service a conversation may run on.

  The in-memory service keeps sessions as live objects while a database-backed
  one round-trips them through storage, and the two do not agree on what
  survives a turn. Recovery bookkeeping has to behave the same on both, so
  every multi-turn test here runs against each.
  """
  if request.param == 'sqlite':
    return SqliteSessionService(db_path=str(tmp_path / 'sessions.db'))
  return InMemorySessionService()


@asynccontextmanager
async def _a_new_turn(session_service, auth_scheme, auth_credential, turn: int):
  """Enters a fresh invocation, and commits what it wrote on the way out.

  The session is re-read from the service and a new ToolContext is built from
  it, as Runner does at the start of every turn, so state that the service
  declines to persist really is gone by the next one. Yields the handler and
  its context.
  """
  session = await session_service.get_session(
      app_name=_MULTI_TURN_APP,
      user_id=_MULTI_TURN_USER,
      session_id=_MULTI_TURN_SESSION,
  )
  tool_context = ToolContext(
      function_call_id=f'fc-{turn}',
      invocation_context=InvocationContext(
          agent=LlmAgent(name='test'),
          session=session,
          invocation_id=f'inv-{turn}',
          session_service=session_service,
      ),
  )
  yield ToolAuthHandler.from_tool_context(
      tool_context=tool_context,
      auth_scheme=auth_scheme,
      auth_credential=auth_credential,
  ), tool_context
  await session_service.append_event(
      session,
      Event(
          author='agent',
          invocation_id=f'inv-{turn}',
          actions=tool_context.actions,
      ),
  )


async def _a_turn_the_api_rejects(
    session_service, auth_scheme, auth_credential, turn: int
) -> tuple[str, bool]:
  """Runs a turn whose API call comes back 401.

  Returns the recovery state and whether the user was asked to
  re-authorize.
  """
  async with _a_new_turn(
      session_service, auth_scheme, auth_credential, turn
  ) as (handler, tool_context):
    # Whatever the user granted on the previous turn, the API rejects it too.
    handler.credential_store.store_credential(
        handler.credential_store.get_credential_key(
            auth_scheme, auth_credential
        ),
        _stale_credential(f'token_from_turn_{turn}'),
    )
    with handler.claim_recovery():
      recovery = await handler.handle_unauthorized_error()
    return recovery, bool(tool_context.actions.requested_auth_configs)


async def _a_turn_the_api_accepts(
    session_service, auth_scheme, auth_credential, turn: int
) -> None:
  """Runs a turn whose API call is not rejected."""
  async with _a_new_turn(
      session_service, auth_scheme, auth_credential, turn
  ) as (handler, _):
    handler.note_successful_call()


@pytest.mark.asyncio
async def test_a_hopeless_401_asks_the_user_once_not_once_per_turn(
    multi_turn_session_service,
    openid_connect_scheme,
    openid_connect_credential,
):
  """Test that the reauth budget still binds after the turn it was spent in.

  The loop the budget exists to close spans turns: the user re-authorizes, the
  API rejects the newly granted token as well, and the next turn asks again.
  Bookkeeping that a session service drops at the turn boundary would be read
  back as an untouched budget and bound nothing.
  """
  await multi_turn_session_service.create_session(
      app_name=_MULTI_TURN_APP,
      user_id=_MULTI_TURN_USER,
      session_id=_MULTI_TURN_SESSION,
  )

  turns = [
      await _a_turn_the_api_rejects(
          multi_turn_session_service,
          openid_connect_scheme,
          openid_connect_credential,
          turn,
      )
      for turn in (1, 2, 3)
  ]

  assert turns == [
      ('reauth_requested', True),
      ('reauth_limit_reached', False),
      ('reauth_limit_reached', False),
  ]


@pytest.mark.asyncio
async def test_a_working_call_lets_a_later_turn_ask_again(
    multi_turn_session_service,
    openid_connect_scheme,
    openid_connect_credential,
):
  """Test that a spent budget is refilled by a success in a different turn.

  The call that proves re-authorization worked lands in a later turn than the
  request it forgives. If the refill did not outlive its own turn, one exhausted
  budget would bar the credential from re-authorization for the rest of the
  session, including for an ordinary token expiry hours later.
  """
  await multi_turn_session_service.create_session(
      app_name=_MULTI_TURN_APP,
      user_id=_MULTI_TURN_USER,
      session_id=_MULTI_TURN_SESSION,
  )
  assert await _a_turn_the_api_rejects(
      multi_turn_session_service,
      openid_connect_scheme,
      openid_connect_credential,
      1,
  ) == ('reauth_requested', True)

  await _a_turn_the_api_accepts(
      multi_turn_session_service,
      openid_connect_scheme,
      openid_connect_credential,
      2,
  )

  # The token granted in turn 1 worked, then expired the way any token does.
  assert await _a_turn_the_api_rejects(
      multi_turn_session_service,
      openid_connect_scheme,
      openid_connect_credential,
      3,
  ) == ('reauth_requested', True)
