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

"""OAuth2 credential exchanger implementation."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi.openapi.models import OAuth2
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_schemes import AuthScheme
from google.adk.auth.auth_schemes import OAuthGrantType
from google.adk.auth.auth_schemes import OpenIdConnectWithConfig
from google.adk.auth.oauth2_credential_util import create_oauth2_session
from google.adk.auth.oauth2_credential_util import update_credential_with_tokens
from google.adk.utils.feature_decorator import experimental
from typing_extensions import override

from .base_credential_exchanger import BaseCredentialExchanger
from .base_credential_exchanger import CredentialExchangeError

try:
  from authlib.integrations.requests_client import OAuth2Session

  AUTHLIB_AVAILABLE = True
except ImportError:
  AUTHLIB_AVAILABLE = False

logger = logging.getLogger("google_adk." + __name__)


@experimental
class OAuth2CredentialExchanger(BaseCredentialExchanger):
  """Exchanges OAuth2 credentials from authorization responses."""

  @override
  async def exchange(
      self,
      auth_credential: AuthCredential,
      auth_scheme: Optional[AuthScheme] = None,
  ) -> AuthCredential:
    """Exchange OAuth2 credential from authorization response.
    if credential exchange failed, the original credential will be returned.

    Args:
        auth_credential: The OAuth2 credential to exchange.
        auth_scheme: The OAuth2 authentication scheme.

    Returns:
        The exchanged credential with access token.

    Raises:
        CredentialExchangeError: If auth_scheme is missing.
    """
    if not auth_scheme:
      raise CredentialExchangeError(
          "auth_scheme is required for OAuth2 credential exchange"
      )

    if not AUTHLIB_AVAILABLE:
      # If authlib is not available, we cannot exchange the credential.
      # We return the original credential without exchange.
      # The client using this tool can decide to exchange the credential
      # themselves using other lib.
      logger.warning(
          "authlib is not available, skipping OAuth2 credential exchange."
      )
      return auth_credential

    if auth_credential.oauth2 and auth_credential.oauth2.access_token:
      return auth_credential

    # Determine grant type from auth_scheme
    grant_type = self._determine_grant_type(auth_scheme)

    if grant_type == OAuthGrantType.CLIENT_CREDENTIALS:
      return await self._exchange_client_credentials(
          auth_credential, auth_scheme
      )
    elif grant_type == OAuthGrantType.AUTHORIZATION_CODE:
      return await self._exchange_authorization_code(
          auth_credential, auth_scheme
      )
    else:
      logger.warning("Unsupported OAuth2 grant type: %s", grant_type)
      return auth_credential

  def _determine_grant_type(
      self, auth_scheme: AuthScheme
  ) -> Optional[OAuthGrantType]:
    """Determine the OAuth2 grant type from the auth scheme.

    Args:
        auth_scheme: The OAuth2 authentication scheme.

    Returns:
        The OAuth2 grant type or None if cannot be determined.
    """
    if isinstance(auth_scheme, OAuth2) and auth_scheme.flows:
      return OAuthGrantType.from_flow(auth_scheme.flows)
    elif isinstance(auth_scheme, OpenIdConnectWithConfig):
      # Check supported grant types for OIDC
      if (
          auth_scheme.grant_types_supported
          and "client_credentials" in auth_scheme.grant_types_supported
      ):
        return OAuthGrantType.CLIENT_CREDENTIALS
      else:
        # Default to authorization code if client credentials not supported
        return OAuthGrantType.AUTHORIZATION_CODE

    return None

  async def _exchange_client_credentials(
      self,
      auth_credential: AuthCredential,
      auth_scheme: AuthScheme,
  ) -> AuthCredential:
    """Exchange client credentials for access token.

    Args:
        auth_credential: The OAuth2 credential to exchange.
        auth_scheme: The OAuth2 authentication scheme.

    Returns:
        The credential with access token.
    """
    client, token_endpoint = create_oauth2_session(auth_scheme, auth_credential)
    if not client:
      logger.warning(
          "Could not create OAuth2 session for client credentials exchange"
      )
      return auth_credential

    try:
      tokens = client.fetch_token(
          token_endpoint,
          grant_type=OAuthGrantType.CLIENT_CREDENTIALS,
      )
      update_credential_with_tokens(auth_credential, tokens)
      logger.debug("Successfully exchanged client credentials for access token")
    except Exception as e:
      logger.error("Failed to exchange client credentials: %s", e)
      return auth_credential

    return auth_credential

  async def _exchange_authorization_code(
      self,
      auth_credential: AuthCredential,
      auth_scheme: AuthScheme,
  ) -> AuthCredential:
    """Exchange authorization code for access token.

    Args:
        auth_credential: The OAuth2 credential to exchange.
        auth_scheme: The OAuth2 authentication scheme.

    Returns:
        The credential with access token.
    """
    client, token_endpoint = create_oauth2_session(auth_scheme, auth_credential)
    if not client:
      logger.warning(
          "Could not create OAuth2 session for authorization code exchange"
      )
      return auth_credential

    try:
      tokens = client.fetch_token(
          token_endpoint,
          authorization_response=auth_credential.oauth2.auth_response_uri,
          code=auth_credential.oauth2.auth_code,
          grant_type=OAuthGrantType.AUTHORIZATION_CODE,
      )
      update_credential_with_tokens(auth_credential, tokens)
      logger.debug("Successfully exchanged authorization code for access token")
    except Exception as e:
      logger.error("Failed to exchange authorization code: %s", e)
      return auth_credential

    return auth_credential
