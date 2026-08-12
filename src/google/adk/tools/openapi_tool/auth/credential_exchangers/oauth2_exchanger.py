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

from __future__ import annotations

"""Credential fetcher for OpenID Connect."""

import logging
from typing import Optional

from authlib.common.errors import AuthlibBaseError
from authlib.oauth2.rfc6749 import OAuth2Token
from requests.exceptions import RequestException

from .....auth.auth_credential import AuthCredential
from .....auth.auth_credential import AuthCredentialTypes
from .....auth.auth_credential import HttpAuth
from .....auth.auth_credential import HttpCredentials
from .....auth.auth_schemes import AuthScheme
from .....auth.auth_schemes import AuthSchemeType
from .....auth.oauth2_credential_util import create_oauth2_session
from .....auth.oauth2_credential_util import update_credential_with_tokens
from .base_credential_exchanger import BaseAuthCredentialExchanger

logger = logging.getLogger("google_adk." + __name__)


class OAuth2CredentialExchanger(BaseAuthCredentialExchanger):
  """Fetches credentials for OAuth2 and OpenID Connect."""

  def _check_scheme_credential_type(
      self,
      auth_scheme: AuthScheme,
      auth_credential: Optional[AuthCredential] = None,
  ):
    if not auth_credential:
      raise ValueError(
          "auth_credential is empty. Please create AuthCredential using"
          " OAuth2Auth."
      )

    if auth_scheme.type_ not in (
        AuthSchemeType.openIdConnect,
        AuthSchemeType.oauth2,
    ):
      raise ValueError(
          "Invalid security scheme, expect AuthSchemeType.openIdConnect or "
          f"AuthSchemeType.oauth2 auth scheme, but got {auth_scheme.type_}"
      )

    if not auth_credential.oauth2 and not auth_credential.http:
      raise ValueError(
          "auth_credential is not configured with oauth2. Please"
          " create AuthCredential and set OAuth2Auth."
      )

  def generate_auth_token(
      self,
      auth_credential: Optional[AuthCredential] = None,
  ) -> AuthCredential:
    """Generates an auth token from the authorization response.

    Args:
        auth_scheme: The OpenID Connect or OAuth2 auth scheme.
        auth_credential: The auth credential.

    Returns:
        An AuthCredential object containing the HTTP bearer access token. If the
        HTTP bearer token cannot be generated, return the original credential.
    """

    if not auth_credential.oauth2.access_token:
      return auth_credential

    # Return the access token as a bearer token.
    updated_credential = AuthCredential(
        auth_type=AuthCredentialTypes.HTTP,  # Store as a bearer token
        http=HttpAuth(
            scheme="bearer",
            credentials=HttpCredentials(
                token=auth_credential.oauth2.access_token
            ),
        ),
    )
    return updated_credential

  def _is_token_expired(self, auth_credential: AuthCredential) -> bool:
    return bool(
        OAuth2Token({
            "expires_at": auth_credential.oauth2.expires_at,
            "expires_in": auth_credential.oauth2.expires_in,
        }).is_expired()
    )

  def _refresh_token(
      self,
      auth_scheme: AuthScheme,
      auth_credential: AuthCredential,
  ) -> None:
    """Refreshes the OAuth2 token in place, keeping the stale token on failure."""
    client, token_endpoint = create_oauth2_session(auth_scheme, auth_credential)
    if not client:
      logger.warning("Could not create OAuth2 session for token refresh")
      return

    try:
      tokens = client.refresh_token(
          url=token_endpoint,
          refresh_token=auth_credential.oauth2.refresh_token,
      )
      update_credential_with_tokens(auth_credential, tokens)
      logger.debug("Successfully refreshed OAuth2 tokens")
    except (AuthlibBaseError, RequestException) as e:
      logger.warning(
          "Failed to refresh OAuth2 tokens, falling back to existing token: %s",
          e,
      )

  def exchange_credential(
      self,
      auth_scheme: AuthScheme,
      auth_credential: Optional[AuthCredential] = None,
  ) -> AuthCredential:
    """Exchanges the OpenID Connect auth credential for an access token or an auth URI.

    Args:
        auth_scheme: The auth scheme.
        auth_credential: The auth credential.

    Returns:
        An AuthCredential object containing the HTTP Bearer access token.

    Raises:
        ValueError: If the auth scheme or auth credential is invalid.
    """
    self._check_scheme_credential_type(auth_scheme, auth_credential)

    # If token is already HTTPBearer token, do nothing assuming that this token
    #  is valid.
    if auth_credential.http:
      return auth_credential

    # If access token is exchanged, exchange a HTTPBearer token.
    if auth_credential.oauth2.access_token:
      # Refresh an expired access token first so the stale one is not wrapped.
      if auth_credential.oauth2.refresh_token and self._is_token_expired(
          auth_credential
      ):
        self._refresh_token(auth_scheme, auth_credential)
      return self.generate_auth_token(auth_credential)

    return None
