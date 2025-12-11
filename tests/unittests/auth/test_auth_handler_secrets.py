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

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_handler import AuthHandler
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.credential_manager import CredentialManager
import pytest


class TestAuthHandlerSecrets:

  @pytest.fixture(autouse=True)
  def clear_credential_manager_secrets(self):
    """Clear CredentialManager secrets buffer before/after each test."""
    CredentialManager._CLIENT_SECRETS = {}
    yield
    CredentialManager._CLIENT_SECRETS = {}

  @pytest.mark.asyncio
  async def test_exchange_auth_token_restores_and_reredacts_secret(self):
    client_id = "test_client_id"
    secret = "super_secret_value"

    # Setup secure storage
    CredentialManager._CLIENT_SECRETS[client_id] = secret

    # Create credential with redacted secret
    credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(client_id=client_id, client_secret="<redacted>"),
    )

    auth_config = Mock(spec=AuthConfig)
    auth_config.exchanged_auth_credential = credential
    auth_config.auth_scheme = Mock()

    handler = AuthHandler(auth_config)

    # Mock exchanger
    mock_exchanger = AsyncMock()

    # Check secret inside exchange
    def check_secret(cred, scheme):
      assert cred.oauth2.client_secret == secret
      return cred

    mock_exchanger.exchange.side_effect = check_secret

    with patch(
        "google.adk.auth.auth_handler.OAuth2CredentialExchanger",
        return_value=mock_exchanger,
    ):
      await handler.exchange_auth_token()

    # Verify secret is re-redacted
    assert credential.oauth2.client_secret == "<redacted>"

  def test_generate_auth_uri_uses_restored_secret(self):
    client_id = "test_client_id"
    secret = "super_secret_value"

    # Setup secure storage
    CredentialManager._CLIENT_SECRETS[client_id] = secret

    # Create credential with redacted secret
    credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id=client_id,
            client_secret="<redacted>",
            redirect_uri="http://localhost/callback",
        ),
    )

    auth_config = Mock(spec=AuthConfig)
    auth_config.raw_auth_credential = credential
    auth_config.auth_scheme = Mock()
    # Mock flows for scopes
    auth_config.auth_scheme.flows.implicit = None
    auth_config.auth_scheme.flows.clientCredentials = None
    auth_config.auth_scheme.flows.password = None
    auth_config.auth_scheme.flows.authorizationCode.scopes = {"scope": "desc"}
    auth_config.auth_scheme.flows.authorizationCode.authorizationUrl = (
        "http://auth"
    )

    handler = AuthHandler(auth_config)

    # Mock OAuth2Session
    with (
        patch("google.adk.auth.auth_handler.OAuth2Session") as mock_session_cls,
        patch("google.adk.auth.auth_handler.AUTHLIB_AVAILABLE", True),
    ):

      mock_session = Mock()
      mock_session.create_authorization_url.return_value = (
          "http://auth?param=1",
          "state",
      )
      mock_session_cls.return_value = mock_session

      handler.generate_auth_uri()

      # Verify session was created with the REAL secret, not redacted one
      mock_session_cls.assert_called_with(
          client_id,
          secret,
          scope="scope",
          redirect_uri="http://localhost/callback",
      )
