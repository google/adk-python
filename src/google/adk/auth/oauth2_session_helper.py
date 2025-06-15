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
from typing import Optional
from typing import Tuple

from fastapi.openapi.models import OAuth2

from ..utils.feature_decorator import experimental
from .auth_credential import AuthCredential
from .auth_schemes import AuthScheme
from .auth_schemes import OpenIdConnectWithConfig

try:
  from authlib.integrations.requests_client import OAuth2Session
  from authlib.oauth2.rfc6749 import OAuth2Token

  AUTHLIB_AVIALABLE = True
except ImportError:
  AUTHLIB_AVIALABLE = False


logger = logging.getLogger("google_adk." + __name__)


@experimental
class OAuth2SessionHelper:
  """Helper class for managing OAuth2 sessions and token operations."""

  def __init__(
      self,
      auth_scheme: AuthScheme,
      auth_credential: AuthCredential,
  ):
    self._auth_scheme = auth_scheme
    self._auth_credential = auth_credential

  def create_oauth2_session(
      self,
  ) -> Tuple[Optional[OAuth2Session], Optional[str]]:
    """Create an OAuth2 session for token operations.

    Returns:
        Tuple of (OAuth2Session, token_endpoint) or (None, None) if cannot create session.
    """
    auth_scheme = self._auth_scheme
    auth_credential = self._auth_credential

    if isinstance(auth_scheme, OpenIdConnectWithConfig):
      if not hasattr(auth_scheme, "token_endpoint"):
        return None, None
      token_endpoint = auth_scheme.token_endpoint
      scopes = auth_scheme.scopes
    elif isinstance(auth_scheme, OAuth2):
      if (
          not auth_scheme.flows.authorizationCode
          or not auth_scheme.flows.authorizationCode.tokenUrl
      ):
        return None, None
      token_endpoint = auth_scheme.flows.authorizationCode.tokenUrl
      scopes = list(auth_scheme.flows.authorizationCode.scopes.keys())
    else:
      return None, None

    if (
        not auth_credential
        or not auth_credential.oauth2
        or not auth_credential.oauth2.client_id
        or not auth_credential.oauth2.client_secret
    ):
      return None, None

    return (
        OAuth2Session(
            auth_credential.oauth2.client_id,
            auth_credential.oauth2.client_secret,
            scope=" ".join(scopes),
            redirect_uri=auth_credential.oauth2.redirect_uri,
            state=auth_credential.oauth2.state,
        ),
        token_endpoint,
    )

  def update_credential_with_tokens(self, tokens: OAuth2Token) -> None:
    """Update the credential with new tokens.

    Args:
        tokens: The OAuth2Token object containing new token information.
    """
    self._auth_credential.oauth2.access_token = tokens.get("access_token")
    self._auth_credential.oauth2.refresh_token = tokens.get("refresh_token")
    self._auth_credential.oauth2.expires_at = (
        int(tokens.get("expires_at")) if tokens.get("expires_at") else None
    )
    self._auth_credential.oauth2.expires_in = (
        int(tokens.get("expires_in")) if tokens.get("expires_in") else None
    )
