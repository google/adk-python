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

from __future__ import annotations

import logging

from ..utils.feature_decorator import experimental
from .auth_credential import AuthCredential
from .auth_schemes import AuthScheme
from .auth_schemes import OAuthGrantType
from .oauth2_credential_util import create_oauth2_session
from .oauth2_credential_util import update_credential_with_tokens

try:
  from authlib.oauth2.rfc6749 import OAuth2Token

  AUTHLIB_AVIALABLE = True
except ImportError:
  AUTHLIB_AVIALABLE = False


logger = logging.getLogger("google_adk." + __name__)


@experimental
class OAuth2CredentialFetcher:
  """Exchanges and refreshes an OAuth2 access token. (Experimental)"""

  def __init__(
      self,
      auth_scheme: AuthScheme,
      auth_credential: AuthCredential,
  ):
    self._auth_scheme = auth_scheme
    self._auth_credential = auth_credential

  def _update_credential(self, tokens: OAuth2Token) -> None:
    self._auth_credential.oauth2.access_token = tokens.get("access_token")
    self._auth_credential.oauth2.refresh_token = tokens.get("refresh_token")
    self._auth_credential.oauth2.expires_at = (
        int(tokens.get("expires_at")) if tokens.get("expires_at") else None
    )
    self._auth_credential.oauth2.expires_in = (
        int(tokens.get("expires_in")) if tokens.get("expires_in") else None
    )

  def exchange(self) -> AuthCredential:
    """Exchange an oauth token from the authorization response.

    Returns:
        An AuthCredential object containing the access token.
    """
    if not AUTHLIB_AVIALABLE:
      return self._auth_credential

    if (
        self._auth_credential.oauth2
        and self._auth_credential.oauth2.access_token
    ):
      return self._auth_credential

    client, token_endpoint = create_oauth2_session(
        self._auth_scheme, self._auth_credential
    )
    if not client:
      logger.warning("Could not create OAuth2 session for token exchange")
      return self._auth_credential

    try:
      tokens = client.fetch_token(
          token_endpoint,
          authorization_response=self._auth_credential.oauth2.auth_response_uri,
          code=self._auth_credential.oauth2.auth_code,
          grant_type=OAuthGrantType.AUTHORIZATION_CODE,
      )
      update_credential_with_tokens(self._auth_credential, tokens)
      logger.info("Successfully exchanged OAuth2 tokens")
    except Exception as e:
      logger.error("Failed to exchange OAuth2 tokens: %s", e)
      # Return original credential on failure
      return self._auth_credential

    return self._auth_credential

  def refresh(self) -> AuthCredential:
    """Refresh an oauth token.

    Returns:
        An AuthCredential object containing the refreshed access token.
    """
    if not AUTHLIB_AVIALABLE:
      return self._auth_credential
    credential = self._auth_credential
    if not credential.oauth2:
      return credential

    if OAuth2Token({
        "expires_at": credential.oauth2.expires_at,
        "expires_in": credential.oauth2.expires_in,
    }).is_expired():
      client, token_endpoint = create_oauth2_session(
          self._auth_scheme, self._auth_credential
      )
      if not client:
        logger.warning("Could not create OAuth2 session for token refresh")
        return credential

      try:
        tokens = client.refresh_token(
            url=token_endpoint,
            refresh_token=credential.oauth2.refresh_token,
        )
        update_credential_with_tokens(self._auth_credential, tokens)
        logger.info("Successfully refreshed OAuth2 tokens")
      except Exception as e:
        logger.error("Failed to refresh OAuth2 tokens: %s", e)
        # Return original credential on failure
        return credential

    return self._auth_credential
