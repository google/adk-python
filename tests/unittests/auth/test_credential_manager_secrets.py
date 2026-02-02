from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlowAuthorizationCode
from fastapi.openapi.models import OAuthFlows
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.credential_manager import CredentialManager
import pytest


@pytest.fixture(autouse=True)
def clear_credential_manager_secrets():
  """Clear CredentialManager secrets buffer before/after each test."""
  CredentialManager._CLIENT_SECRETS = {}
  yield
  CredentialManager._CLIENT_SECRETS = {}


@pytest.mark.asyncio
async def test_credential_manager_redacts_secrets_in_raw_credential():
  """Test that CredentialManager redacts client_secret from raw_auth_credential upon initialization."""

  # Setup
  client_id = "test_client_id"
  client_secret = "test_client_secret"

  oauth_auth = OAuth2Auth(client_id=client_id, client_secret=client_secret)

  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2, oauth2=oauth_auth
  )

  auth_scheme = OAuth2(
      flows=OAuthFlows(
          authorizationCode=OAuthFlowAuthorizationCode(
              authorizationUrl="https://example.com/auth",
              tokenUrl="https://example.com/token",
          )
      )
  )

  auth_config = AuthConfig(
      auth_scheme=auth_scheme, raw_auth_credential=auth_credential
  )

  # Act
  manager = CredentialManager(auth_config)

  # Assert
  # 1. Check if secret is in memory map
  assert client_id in manager._CLIENT_SECRETS
  assert manager._CLIENT_SECRETS[client_id] == client_secret

  # 2. Check if secret is redacted in the manager's config
  assert (
      manager._auth_config.raw_auth_credential.oauth2.client_secret
      == "<redacted>"
  )

  # 3. Check original config is NOT modified (AuthConfig copy behavior)
  # Since we used model_copy(deep=True), calling on Pydantic model copies it.
  assert auth_config.raw_auth_credential.oauth2.client_secret == client_secret


@pytest.mark.asyncio
async def test_credential_manager_redacts_secrets_in_exchanged_credential():
  """Test that CredentialManager redacts client_secret from exchanged_auth_credential if present."""

  # Setup
  client_id = "test_client_id_exchanged"
  client_secret = "test_client_secret_exchanged"

  oauth_auth = OAuth2Auth(
      client_id=client_id,
      client_secret=client_secret,
      access_token="some_token",
  )

  exchanged_credential = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2, oauth2=oauth_auth
  )

  auth_scheme = OAuth2(
      flows=OAuthFlows(
          authorizationCode=OAuthFlowAuthorizationCode(
              authorizationUrl="https://example.com/auth",
              tokenUrl="https://example.com/token",
          )
      )
  )

  auth_config = AuthConfig(
      auth_scheme=auth_scheme,
      raw_auth_credential=None,
      exchanged_auth_credential=exchanged_credential,
  )

  # Act
  manager = CredentialManager(auth_config)

  # Assert
  assert client_id in manager._CLIENT_SECRETS
  assert manager._CLIENT_SECRETS[client_id] == client_secret

  assert (
      manager._auth_config.exchanged_auth_credential.oauth2.client_secret
      == "<redacted>"
  )


@pytest.mark.asyncio
async def test_exchange_credential_restores_secret():
  """Test that _exchange_credential restores the secret before calling exchanger."""

  # Setup
  client_id = "test_client_id_exchange"
  client_secret = "test_client_secret_exchange"

  oauth_auth = OAuth2Auth(client_id=client_id, client_secret=client_secret)

  raw_credential = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2, oauth2=oauth_auth
  )

  auth_scheme = OAuth2(
      flows=OAuthFlows(
          authorizationCode=OAuthFlowAuthorizationCode(
              authorizationUrl="https://example.com/auth",
              tokenUrl="https://example.com/token",
          )
      )
  )

  auth_config = AuthConfig(
      auth_scheme=auth_scheme, raw_auth_credential=raw_credential
  )

  manager = CredentialManager(auth_config)

  # Secret should be redacted now
  assert (
      manager._auth_config.raw_auth_credential.oauth2.client_secret
      == "<redacted>"
  )

  # Prepare a credential to be exchanged (e.g. from client response, has no secret or redacted)
  credential_to_exchange = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id=client_id,
          client_secret="<redacted>",  # or None
          auth_code="some_code",
      ),
  )

  # Mock exchanger
  mock_exchanger = AsyncMock()

  # We use side_effect to verify the secret at the moment of call, because the object is mutated later
  def check_secret(cred, scheme):
      assert cred.oauth2.client_secret == client_secret
      from google.adk.auth.exchanger.base_credential_exchanger import ExchangeResult
      return ExchangeResult(credential=credential_to_exchange, was_exchanged=True)

  mock_exchanger.exchange.side_effect = check_secret

  with patch.object(
      manager._exchanger_registry, "get_exchanger", return_value=mock_exchanger
  ):
    # Act
    result_credential, exchanged = await manager._exchange_credential(
        credential_to_exchange
    )

    # Assert
    # Verification happened in side_effect
    assert mock_exchanger.exchange.called

    # Check that the result credential (modified in place or returned) has secret REDACTED again
    assert result_credential.oauth2.client_secret == "<redacted>"
