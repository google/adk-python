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

from typing import Optional
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import HttpAuth
from google.adk.auth.auth_credential import HttpCredentials
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_schemes import AuthScheme
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.tools.openapi_tool.auth.auth_helpers import openid_dict_to_scheme_credential
from google.adk.tools.openapi_tool.auth.auth_helpers import token_to_scheme_credential
from google.adk.tools.openapi_tool.auth.credential_exchangers.auto_auth_credential_exchanger import OAuth2CredentialExchanger
from google.adk.tools.openapi_tool.openapi_spec_parser.tool_auth_handler import ToolAuthHandler
from google.adk.tools.openapi_tool.openapi_spec_parser.tool_auth_handler import ToolContextCredentialStore
from google.adk.tools.tool_context import ToolContext
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
  returned_credentail = AuthCredential(
      auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
      oauth2=OAuth2Auth(auth_response_uri='test_auth_response_uri'),
  )
  mock_auth_handler.get_auth_response.return_value = returned_credentail
  mock_auth_handler_path = 'google.adk.tools.tool_context.AuthHandler'
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
  assert stored_credential == returned_credentail
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


@patch(
    'google.adk.tools.openapi_tool.openapi_spec_parser.tool_auth_handler.OAuth2CredentialRefresher'
)
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
