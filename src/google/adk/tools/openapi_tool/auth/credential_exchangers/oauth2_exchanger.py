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

"""Credential fetcher for OpenID Connect."""

from typing import Optional

from .....auth.auth_credential import AuthCredential
from .....auth.auth_credential import AuthCredentialTypes
from .....auth.auth_credential import HttpAuth
from .....auth.auth_credential import HttpCredentials
from .....auth.auth_schemes import AuthScheme
from .....auth.auth_schemes import AuthSchemeType
from .base_credential_exchanger import BaseAuthCredentialExchanger


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
    # TODO(cheliu): Implement token refresh flow

    self._check_scheme_credential_type(auth_scheme, auth_credential)

    # If token is already HTTPBearer token, do nothing assuming that this token
    #  is valid.
    if auth_credential.http:
      return auth_credential

    # If access token is exchanged, exchange a HTTPBearer token.
    if auth_credential.oauth2.access_token:
      return self.generate_auth_token(auth_credential)

    return None
