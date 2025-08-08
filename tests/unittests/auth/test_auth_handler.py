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

import copy
import time
from unittest.mock import Mock
from unittest.mock import patch

from authlib.oauth2.rfc6749 import OAuth2Token
from fastapi.openapi.models import APIKey
from fastapi.openapi.models import APIKeyIn
from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlowAuthorizationCode
from fastapi.openapi.models import OAuthFlows
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_handler import AuthHandler
from google.adk.auth.auth_schemes import OpenIdConnectWithConfig
from google.adk.auth.auth_tool import AuthConfig
import pytest


# Mock classes for testing
class MockState(dict):
  """Mock State class for testing."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def get(self, key, default=None):
    return super().get(key, default)


class MockOAuth2Session:
  """Mock OAuth2Session for testing."""

  def __init__(
      self,
      client_id=None,
      client_secret=None,
      scope=None,
      redirect_uri=None,
      state=None,
  ):
    self.client_id = client_id
    self.client_secret = client_secret
    self.scope = scope
    self.redirect_uri = redirect_uri
    self.state = state

  def create_authorization_url(self, url, **kwargs):
    return f"{url}?client_id={self.client_id}&scope={self.scope}", "mock_state"

  def fetch_token(
      self,
      token_endpoint,
      authorization_response=None,
      code=None,
      grant_type=None,
  ):
    return {
        "access_token": "mock_access_token",
        "token_type": "bearer",
        "expires_in": 3600,
        "refresh_token": "mock_refresh_token",
    }


# Fixtures for common test objects
@pytest.fixture
def oauth2_auth_scheme():
  """Create an OAuth2 auth scheme for testing."""
  # Create the OAuthFlows object first
  flows = OAuthFlows(
      authorizationCode=OAuthFlowAuthorizationCode(
          authorizationUrl="https://example.com/oauth2/authorize",
          tokenUrl="https://example.com/oauth2/token",
          scopes={"read": "Read access", "write": "Write access"},
      )
  )

  # Then create the OAuth2 object with the flows
  return OAuth2(flows=flows)


@pytest.fixture
def openid_auth_scheme():
  """Create an OpenID Connect auth scheme for testing."""
  return OpenIdConnectWithConfig(
      openIdConnectUrl="https://example.com/.well-known/openid-configuration",
      authorization_endpoint="https://example.com/oauth2/authorize",
      token_endpoint="https://example.com/oauth2/token",
      scopes=["openid", "profile", "email"],
  )


@pytest.fixture
def oauth2_credentials():
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
def oauth2_credentials_with_token():
  """Create OAuth2 credentials with a token for testing."""
  return AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id="mock_client_id",
          client_secret="mock_client_secret",
          redirect_uri="https://example.com/callback",
          access_token="mock_access_token",
          refresh_token="mock_refresh_token",
      ),
  )


@pytest.fixture
def oauth2_credentials_with_auth_uri():
  """Create OAuth2 credentials with an auth URI for testing."""
  return AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id="mock_client_id",
          client_secret="mock_client_secret",
          redirect_uri="https://example.com/callback",
          auth_uri="https://example.com/oauth2/authorize?client_id=mock_client_id&scope=read,write",
          state="mock_state",
      ),
  )


@pytest.fixture
def oauth2_credentials_with_auth_code():
  """Create OAuth2 credentials with an auth code for testing."""
  return AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id="mock_client_id",
          client_secret="mock_client_secret",
          redirect_uri="https://example.com/callback",
          auth_uri="https://example.com/oauth2/authorize?client_id=mock_client_id&scope=read,write",
          state="mock_state",
          auth_code="mock_auth_code",
          auth_response_uri="https://example.com/callback?code=mock_auth_code&state=mock_state",
      ),
  )


@pytest.fixture
def auth_config(oauth2_auth_scheme, oauth2_credentials):
  """Create an AuthConfig for testing."""
  # Create a copy of the credentials for the exchanged_auth_credential
  exchanged_credential = oauth2_credentials.model_copy(deep=True)

  return AuthConfig(
      auth_scheme=oauth2_auth_scheme,
      raw_auth_credential=oauth2_credentials,
      exchanged_auth_credential=exchanged_credential,
  )


@pytest.fixture
def auth_config_with_exchanged(
    oauth2_auth_scheme, oauth2_credentials, oauth2_credentials_with_auth_uri
):
  """Create an AuthConfig with exchanged credentials for testing."""
  return AuthConfig(
      auth_scheme=oauth2_auth_scheme,
      raw_auth_credential=oauth2_credentials,
      exchanged_auth_credential=oauth2_credentials_with_auth_uri,
  )


@pytest.fixture
def auth_config_with_auth_code(
    oauth2_auth_scheme, oauth2_credentials, oauth2_credentials_with_auth_code
):
  """Create an AuthConfig with auth code for testing."""
  return AuthConfig(
      auth_scheme=oauth2_auth_scheme,
      raw_auth_credential=oauth2_credentials,
      exchanged_auth_credential=oauth2_credentials_with_auth_code,
  )


class TestAuthHandlerInit:
  """Tests for the AuthHandler initialization."""

  def test_init(self, auth_config):
    """Test the initialization of AuthHandler."""
    handler = AuthHandler(auth_config)
    assert handler.auth_config == auth_config


class TestGenerateAuthUri:
  """Tests for the generate_auth_uri method."""

  @patch("google.adk.auth.auth_handler.OAuth2Session", MockOAuth2Session)
  def test_generate_auth_uri_oauth2(self, auth_config):
    """Test generating an auth URI for OAuth2."""
    handler = AuthHandler(auth_config)
    result = handler.generate_auth_uri()

    assert result.oauth2.auth_uri.startswith(
        "https://example.com/oauth2/authorize"
    )
    assert "client_id=mock_client_id" in result.oauth2.auth_uri
    assert result.oauth2.state == "mock_state"

  @patch("google.adk.auth.auth_handler.OAuth2Session", MockOAuth2Session)
  def test_generate_auth_uri_openid(
      self, openid_auth_scheme, oauth2_credentials
  ):
    """Test generating an auth URI for OpenID Connect."""
    # Create a copy for the exchanged credential
    exchanged = oauth2_credentials.model_copy(deep=True)

    config = AuthConfig(
        auth_scheme=openid_auth_scheme,
        raw_auth_credential=oauth2_credentials,
        exchanged_auth_credential=exchanged,
    )
    handler = AuthHandler(config)
    result = handler.generate_auth_uri()

    assert result.oauth2.auth_uri.startswith(
        "https://example.com/oauth2/authorize"
    )
    assert "client_id=mock_client_id" in result.oauth2.auth_uri
    assert result.oauth2.state == "mock_state"


class TestGenerateAuthRequest:
  """Tests for the generate_auth_request method."""

  def test_non_oauth_scheme(self):
    """Test with a non-OAuth auth scheme."""
    # Use a SecurityBase instance without using APIKey which has validation issues
    api_key_scheme = APIKey(**{"name": "test_api_key", "in": APIKeyIn.header})

    credential = AuthCredential(
        auth_type=AuthCredentialTypes.API_KEY, api_key="test_api_key"
    )

    # Create a copy for the exchanged credential
    exchanged = credential.model_copy(deep=True)

    config = AuthConfig(
        auth_scheme=api_key_scheme,
        raw_auth_credential=credential,
        exchanged_auth_credential=exchanged,
    )

    handler = AuthHandler(config)
    result = handler.generate_auth_request()

    assert result == config

  def test_with_existing_auth_uri(self, auth_config_with_exchanged):
    """Test when auth_uri already exists in exchanged credential."""
    handler = AuthHandler(auth_config_with_exchanged)
    result = handler.generate_auth_request()

    assert (
        result.exchanged_auth_credential.oauth2.auth_uri
        == auth_config_with_exchanged.exchanged_auth_credential.oauth2.auth_uri
    )

  def test_missing_raw_credential(self, oauth2_auth_scheme):
    """Test when raw_auth_credential is missing."""

    config = AuthConfig(
        auth_scheme=oauth2_auth_scheme,
    )
    handler = AuthHandler(config)

    with pytest.raises(ValueError, match="requires auth_credential"):
      handler.generate_auth_request()

  def test_missing_oauth2_in_raw_credential(self, oauth2_auth_scheme):
    """Test when oauth2 is missing in raw_auth_credential."""
    credential = AuthCredential(
        auth_type=AuthCredentialTypes.API_KEY, api_key="test_api_key"
    )

    # Create a copy for the exchanged credential
    exchanged = credential.model_copy(deep=True)

    config = AuthConfig(
        auth_scheme=oauth2_auth_scheme,
        raw_auth_credential=credential,
        exchanged_auth_credential=exchanged,
    )
    handler = AuthHandler(config)

    with pytest.raises(ValueError, match="requires oauth2 in auth_credential"):
      handler.generate_auth_request()

  def test_auth_uri_in_raw_credential(
      self, oauth2_auth_scheme, oauth2_credentials_with_auth_uri
  ):
    """Test when auth_uri exists in raw_credential."""
    config = AuthConfig(
        auth_scheme=oauth2_auth_scheme,
        raw_auth_credential=oauth2_credentials_with_auth_uri,
        exchanged_auth_credential=oauth2_credentials_with_auth_uri.model_copy(
            deep=True
        ),
    )
    handler = AuthHandler(config)
    result = handler.generate_auth_request()

    assert (
        result.exchanged_auth_credential.oauth2.auth_uri
        == oauth2_credentials_with_auth_uri.oauth2.auth_uri
    )

  def test_missing_client_credentials(self, oauth2_auth_scheme):
    """Test when client_id or client_secret is missing."""
    bad_credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(redirect_uri="https://example.com/callback"),
    )

    # Create a copy for the exchanged credential
    exchanged = bad_credential.model_copy(deep=True)

    config = AuthConfig(
        auth_scheme=oauth2_auth_scheme,
        raw_auth_credential=bad_credential,
        exchanged_auth_credential=exchanged,
    )
    handler = AuthHandler(config)

    with pytest.raises(
        ValueError, match="requires both client_id and client_secret"
    ):
      handler.generate_auth_request()

  @patch("google.adk.auth.auth_handler.AuthHandler.generate_auth_uri")
  def test_generate_new_auth_uri(self, mock_generate_auth_uri, auth_config):
    """Test generating a new auth URI."""
    mock_credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id="mock_client_id",
            client_secret="mock_client_secret",
            redirect_uri="https://example.com/callback",
            auth_uri="https://example.com/generated",
            state="generated_state",
        ),
    )
    mock_generate_auth_uri.return_value = mock_credential

    handler = AuthHandler(auth_config)
    result = handler.generate_auth_request()

    assert mock_generate_auth_uri.called
    assert result.exchanged_auth_credential == mock_credential


class TestGetAuthResponse:
  """Tests for the get_auth_response method."""

  def test_get_auth_response_exists(
      self, auth_config, oauth2_credentials_with_auth_uri
  ):
    """Test retrieving an existing auth response from state."""
    handler = AuthHandler(auth_config)
    state = MockState()

    # Store a credential in the state
    credential_key = auth_config.credential_key
    state["temp:" + credential_key] = oauth2_credentials_with_auth_uri

    result = handler.get_auth_response(state)
    assert result == oauth2_credentials_with_auth_uri

  def test_get_auth_response_not_exists(self, auth_config):
    """Test retrieving a non-existent auth response from state."""
    handler = AuthHandler(auth_config)
    state = MockState()

    result = handler.get_auth_response(state)
    assert result is None


class TestParseAndStoreAuthResponse:
  """Tests for the parse_and_store_auth_response method."""

  @pytest.mark.asyncio
  async def test_non_oauth_scheme(self, auth_config_with_exchanged):
    """Test with a non-OAuth auth scheme."""
    # Modify the auth scheme type to be non-OAuth
    auth_config = copy.deepcopy(auth_config_with_exchanged)
    auth_config.auth_scheme = APIKey(
        **{"name": "test_api_key", "in": APIKeyIn.header}
    )

    handler = AuthHandler(auth_config)
    state = MockState()

    await handler.parse_and_store_auth_response(state)

    credential_key = auth_config.credential_key
    assert (
        state["temp:" + credential_key] == auth_config.exchanged_auth_credential
    )

  @patch("google.adk.auth.auth_handler.AuthHandler.exchange_auth_token")
  @pytest.mark.asyncio
  async def test_oauth_scheme(
      self, mock_exchange_token, auth_config_with_exchanged
  ):
    """Test with an OAuth auth scheme."""
    mock_exchange_token.return_value = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(access_token="exchanged_token"),
    )

    handler = AuthHandler(auth_config_with_exchanged)
    state = MockState()

    await handler.parse_and_store_auth_response(state)

    credential_key = auth_config_with_exchanged.credential_key
    assert state["temp:" + credential_key] == mock_exchange_token.return_value
    assert mock_exchange_token.called


class TestExchangeAuthToken:
  """Tests for the exchange_auth_token method."""

  @pytest.mark.asyncio
  async def test_token_exchange_not_supported(
      self, auth_config_with_auth_code, monkeypatch
  ):
    """Test when token exchange is not supported."""
    monkeypatch.setattr(
        "google.adk.auth.exchanger.oauth2_credential_exchanger.AUTHLIB_AVAILABLE",
        False,
    )

    handler = AuthHandler(auth_config_with_auth_code)
    result = await handler.exchange_auth_token()

    assert result == auth_config_with_auth_code.exchanged_auth_credential

  @pytest.mark.asyncio
  async def test_openid_missing_token_endpoint(
      self, openid_auth_scheme, oauth2_credentials_with_auth_code
  ):
    """Test OpenID Connect without a token endpoint."""
    # Create a scheme without token_endpoint
    scheme_without_token = copy.deepcopy(openid_auth_scheme)
    delattr(scheme_without_token, "token_endpoint")

    config = AuthConfig(
        auth_scheme=scheme_without_token,
        raw_auth_credential=oauth2_credentials_with_auth_code,
        exchanged_auth_credential=oauth2_credentials_with_auth_code,
    )

    handler = AuthHandler(config)
    result = await handler.exchange_auth_token()

    assert result == oauth2_credentials_with_auth_code

  @pytest.mark.asyncio
  async def test_oauth2_missing_token_url(
      self, oauth2_auth_scheme, oauth2_credentials_with_auth_code
  ):
    """Test OAuth2 without a token URL."""
    # Create a scheme without tokenUrl
    scheme_without_token = copy.deepcopy(oauth2_auth_scheme)
    scheme_without_token.flows.authorizationCode.tokenUrl = None

    config = AuthConfig(
        auth_scheme=scheme_without_token,
        raw_auth_credential=oauth2_credentials_with_auth_code,
        exchanged_auth_credential=oauth2_credentials_with_auth_code,
    )

    handler = AuthHandler(config)
    result = await handler.exchange_auth_token()

    assert result == oauth2_credentials_with_auth_code

  @pytest.mark.asyncio
  async def test_non_oauth_scheme(self, auth_config_with_auth_code):
    """Test with a non-OAuth auth scheme."""
    # Modify the auth scheme type to be non-OAuth
    auth_config = copy.deepcopy(auth_config_with_auth_code)
    auth_config.auth_scheme = APIKey(
        **{"name": "test_api_key", "in": APIKeyIn.header}
    )

    handler = AuthHandler(auth_config)
    result = await handler.exchange_auth_token()

    assert result == auth_config.exchanged_auth_credential

  @pytest.mark.asyncio
  async def test_missing_credentials(self, oauth2_auth_scheme):
    """Test with missing credentials."""
    empty_credential = AuthCredential(auth_type=AuthCredentialTypes.OAUTH2)

    config = AuthConfig(
        auth_scheme=oauth2_auth_scheme,
        exchanged_auth_credential=empty_credential,
    )

    handler = AuthHandler(config)
    result = await handler.exchange_auth_token()

    assert result == empty_credential

  @pytest.mark.asyncio
  async def test_credentials_with_token(
      self, auth_config, oauth2_credentials_with_token
  ):
    """Test when credentials already have a token."""
    config = AuthConfig(
        auth_scheme=auth_config.auth_scheme,
        raw_auth_credential=auth_config.raw_auth_credential,
        exchanged_auth_credential=oauth2_credentials_with_token,
    )

    handler = AuthHandler(config)
    result = await handler.exchange_auth_token()

    assert result == oauth2_credentials_with_token

  @patch("google.adk.auth.oauth2_credential_util.OAuth2Session")
  @pytest.mark.asyncio
  async def test_successful_token_exchange(
      self, mock_oauth2_session, auth_config_with_auth_code
  ):
    """Test a successful token exchange."""
    # Setup mock OAuth2Session
    mock_client = Mock()
    mock_oauth2_session.return_value = mock_client
    mock_tokens = OAuth2Token({
        "access_token": "mock_access_token",
        "refresh_token": "mock_refresh_token",
        "expires_at": int(time.time()) + 3600,
        "expires_in": 3600,
    })
    mock_client.fetch_token.return_value = mock_tokens

    handler = AuthHandler(auth_config_with_auth_code)
    result = await handler.exchange_auth_token()

    assert result.oauth2.access_token == "mock_access_token"
    assert result.oauth2.refresh_token == "mock_refresh_token"
    assert result.auth_type == AuthCredentialTypes.OAUTH2
