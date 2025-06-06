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
import requests
from copy import deepcopy
from .....auth.auth_credential import AuthCredential
from .....auth.auth_credential import AuthCredentialTypes
from .....auth.auth_credential import HttpAuth
from .....auth.auth_credential import HttpCredentials
from .....auth.auth_schemes import AuthScheme
from .....auth.auth_schemes import AuthSchemeType
from .base_credential_exchanger import (
    BaseAuthCredentialExchanger,
    AuthCredentialMissingError,
)
from datetime import datetime, timedelta, timezone


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

  def _is_token_expired(self, oauth2) -> bool:
    """Returns True if the access token is expired."""
    if not oauth2 or not oauth2.expiry:
      return False
    return datetime.now(timezone.utc) >= oauth2.expiry

  def _has_valid_access_token(self, oauth2) -> bool:
    """Returns True if access_token is present and not empty."""
    return bool(oauth2 and oauth2.access_token and oauth2.access_token.strip())

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

  def refresh_auth_token(
      self, auth_scheme: AuthScheme, auth_credential: AuthCredential
  ) -> AuthCredential:
    """Refreshes the auth token using the refresh token if available.

    Args:
        auth_scheme: The auth scheme.
        auth_credential: The auth credential.

    Returns:
        An AuthCredential object containing the HTTP Bearer access token.

    Raises:
        AuthCredentialMissingError: If the refresh token is missing.
    """
    oauth2 = auth_credential.oauth2
    if not oauth2 or not oauth2.refresh_token:
      raise AuthCredentialMissingError(
          "refresh_token is missing in OAuth2 credential."
      )
    token_endpoint = getattr(auth_scheme, "token_endpoint", None)
    if not token_endpoint and hasattr(auth_scheme, "token_endpoint"):
      token_endpoint = auth_scheme.token_endpoint
    if not token_endpoint:
      raise AuthCredentialMissingError(
          "token_endpoint is missing in AuthScheme."
      )

    data = {
        "grant_type": "refresh_token",
        "refresh_token": oauth2.refresh_token,
        "client_id": oauth2.client_id,
        "client_secret": oauth2.client_secret,
    }
    try:
      response = requests.post(token_endpoint, data=data)
      if response.status_code == 200:
        token_data = response.json()
        new_access_token = token_data.get("access_token")
        if new_access_token:
          new_credential = deepcopy(auth_credential)
          new_credential.oauth2.access_token = new_access_token
          # Update expiry if expires_in is present
          expires_in = token_data.get("expires_in")
          if expires_in:
            new_credential.oauth2.expiry = datetime.now(
                timezone.utc
            ) + timedelta(seconds=int(expires_in))
          if "refresh_token" in token_data:
            new_credential.oauth2.refresh_token = token_data["refresh_token"]
          return self.generate_auth_token(new_credential)
        else:
          raise AuthCredentialMissingError(
              f"No access_token in token response: {token_data}"
          )
      else:
        raise AuthCredentialMissingError(
            f"Token refresh failed, status: {response.status_code}, body:"
            f" {response.text}"
        )
    except Exception as e:
      raise AuthCredentialMissingError(
          f"Exception during token refresh: {e}"
      ) from e

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

    # If token is already HTTPBearer token, do nothing assuming that this token is valid.
    if auth_credential.http:
      return auth_credential

    # If access token is present, not empty, and not expired, return it.
    if self._has_valid_access_token(
        auth_credential.oauth2
    ) and not self._is_token_expired(auth_credential.oauth2):
      return self.generate_auth_token(auth_credential)

    # If access token is missing, empty, or expired, and refresh_token exists, try to refresh.
    if (
        auth_credential.oauth2
        and auth_credential.oauth2.refresh_token
        and (
            not self._has_valid_access_token(auth_credential.oauth2)
            or self._is_token_expired(auth_credential.oauth2)
        )
    ):
      return self.refresh_auth_token(auth_scheme, auth_credential)

    raise AuthCredentialMissingError(
        "Cannot exchange credential: no valid access_token or refresh_token."
    )
