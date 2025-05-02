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

from typing import TYPE_CHECKING

from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import SecurityBase

from .auth_credential import AuthCredential
from .auth_credential import AuthCredentialTypes
from .auth_credential import OAuth2Auth
from .auth_schemes import AuthSchemeType
from .auth_schemes import OAuthGrantType
from .auth_schemes import OpenIdConnectWithConfig
from .auth_tool import AuthConfig

if TYPE_CHECKING:
  from ..sessions.state import State

try:
  from authlib.integrations.requests_client import OAuth2Session

  SUPPORT_TOKEN_EXCHANGE = True
except ImportError:
  SUPPORT_TOKEN_EXCHANGE = False


class AuthHandler:

  def __init__(self, auth_config: AuthConfig):
    self.auth_config = auth_config

  def exchange_auth_token(
      self,
  ) -> AuthCredential:
    """Generates an auth token from the authorization response.

    Returns:
        An AuthCredential object containing the access token.

    Raises:
        ValueError: If the token endpoint is not configured in the auth
            scheme.
        AuthCredentialMissingError: If the access token cannot be retrieved
            from the token endpoint.
    """
    auth_scheme = self.auth_config.auth_scheme
    auth_credential = self.auth_config.exchanged_auth_credential
    if not SUPPORT_TOKEN_EXCHANGE:
      return auth_credential
    if isinstance(auth_scheme, OpenIdConnectWithConfig):
      if not hasattr(auth_scheme, "token_endpoint"):
        return self.auth_config.exchanged_auth_credential
      token_endpoint = auth_scheme.token_endpoint
      scopes = auth_scheme.scopes
    elif isinstance(auth_scheme, OAuth2):
      if (
          not auth_scheme.flows.authorizationCode
          or not auth_scheme.flows.authorizationCode.tokenUrl
      ):
        return self.auth_config.exchanged_auth_credential
      token_endpoint = auth_scheme.flows.authorizationCode.tokenUrl
      scopes = list(auth_scheme.flows.authorizationCode.scopes.keys())
    else:
      return self.auth_config.exchanged_auth_credential

    if (
        not auth_credential
        or not auth_credential.oauth2
        or not auth_credential.oauth2.client_id
        or not auth_credential.oauth2.client_secret
        or auth_credential.oauth2.access_token
        or auth_credential.oauth2.refresh_token
    ):
      return self.auth_config.exchanged_auth_credential

    client = OAuth2Session(
        auth_credential.oauth2.client_id,
        auth_credential.oauth2.client_secret,
        scope=" ".join(scopes),
        redirect_uri=auth_credential.oauth2.redirect_uri,
        state=auth_credential.oauth2.state,
    )
    tokens = client.fetch_token(
        token_endpoint,
        authorization_response=auth_credential.oauth2.auth_response_uri,
        code=auth_credential.oauth2.auth_code,
        grant_type=OAuthGrantType.AUTHORIZATION_CODE,
    )

    updated_credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            access_token=tokens.get("access_token"),
            refresh_token=tokens.get("refresh_token"),
        ),
    )
    return updated_credential

  def parse_and_store_auth_response(self, state: State) -> None:

    credential_key = self.get_credential_key()

    state[credential_key] = self.auth_config.exchanged_auth_credential
    if not isinstance(
        self.auth_config.auth_scheme, SecurityBase
    ) or self.auth_config.auth_scheme.type_ not in (
        AuthSchemeType.oauth2,
        AuthSchemeType.openIdConnect,
    ):
      return

    state[credential_key] = self.exchange_auth_token()

  def _validate(self) -> None:
    if not self.auth_scheme:
      raise ValueError("auth_scheme is empty.")

  def get_auth_response(self, state: State) -> AuthCredential:
    credential_key = self.get_credential_key()
    return state.get(credential_key, None)

  def generate_auth_request(self) -> AuthConfig:
    if not isinstance(
        self.auth_config.auth_scheme, SecurityBase
    ) or self.auth_config.auth_scheme.type_ not in (
        AuthSchemeType.oauth2,
        AuthSchemeType.openIdConnect,
    ):
      return self.auth_config.model_copy(deep=True)

    # auth_uri already in exchanged credential
    if (
        self.auth_config.exchanged_auth_credential
        and self.auth_config.exchanged_auth_credential.oauth2
        and self.auth_config.exchanged_auth_credential.oauth2.auth_uri
    ):
      return self.auth_config.model_copy(deep=True)

    # Check if raw_auth_credential exists
    if not self.auth_config.raw_auth_credential:
      raise ValueError(
          f"Auth Scheme {self.auth_config.auth_scheme.type_} requires"
          " auth_credential."
      )

    # Check if oauth2 exists in raw_auth_credential
    if not self.auth_config.raw_auth_credential.oauth2:
      raise ValueError(
          f"Auth Scheme {self.auth_config.auth_scheme.type_} requires oauth2 in"
          " auth_credential."
      )

    # auth_uri in raw credential
    if self.auth_config.raw_auth_credential.oauth2.auth_uri:
      return AuthConfig(
          auth_scheme=self.auth_config.auth_scheme,
          raw_auth_credential=self.auth_config.raw_auth_credential,
          exchanged_auth_credential=self.auth_config.raw_auth_credential.model_copy(
              deep=True
          ),
      )

    # Check for client_id and client_secret
    if (
        not self.auth_config.raw_auth_credential.oauth2.client_id
        or not self.auth_config.raw_auth_credential.oauth2.client_secret
    ):
      raise ValueError(
          f"Auth Scheme {self.auth_config.auth_scheme.type_} requires both"
          " client_id and client_secret in auth_credential.oauth2."
      )

    # Generate new auth URI
    exchanged_credential = self.generate_auth_uri()
    return AuthConfig(
        auth_scheme=self.auth_config.auth_scheme,
        raw_auth_credential=self.auth_config.raw_auth_credential,
        exchanged_auth_credential=exchanged_credential,
    )

  def get_credential_key(self) -> str:
    """Generates a unique key for the given auth scheme and credential."""
    auth_scheme = self.auth_config.auth_scheme
    auth_credential = self.auth_config.raw_auth_credential
    if auth_scheme.model_extra:
      auth_scheme = auth_scheme.model_copy(deep=True)
      auth_scheme.model_extra.clear()
    scheme_name = (
        f"{auth_scheme.type_.name}_{hash(auth_scheme.model_dump_json())}"
        if auth_scheme
        else ""
    )
    if auth_credential.model_extra:
      auth_credential = auth_credential.model_copy(deep=True)
      auth_credential.model_extra.clear()
    credential_name = (
        f"{auth_credential.auth_type.value}_{hash(auth_credential.model_dump_json())}"
        if auth_credential
        else ""
    )

    return f"temp:adk_{scheme_name}_{credential_name}"

  def generate_auth_uri(
      self,
  ) -> AuthCredential:
    """Generates an response containing the auth uri for user to sign in.

    Returns:
        An AuthCredential object containing the auth URI and state.

    Raises:
        ValueError: If the authorization endpoint is not configured in the auth
            scheme.
    """
    auth_scheme = self.auth_config.auth_scheme
    auth_credential = self.auth_config.raw_auth_credential

    if isinstance(auth_scheme, OpenIdConnectWithConfig):
      authorization_endpoint = auth_scheme.authorization_endpoint
      scopes = auth_scheme.scopes
    else:
      authorization_endpoint = (
          auth_scheme.flows.implicit
          and auth_scheme.flows.implicit.authorizationUrl
          or auth_scheme.flows.authorizationCode
          and auth_scheme.flows.authorizationCode.authorizationUrl
          or auth_scheme.flows.clientCredentials
          and auth_scheme.flows.clientCredentials.tokenUrl
          or auth_scheme.flows.password
          and auth_scheme.flows.password.tokenUrl
      )
      scopes = (
          auth_scheme.flows.implicit
          and auth_scheme.flows.implicit.scopes
          or auth_scheme.flows.authorizationCode
          and auth_scheme.flows.authorizationCode.scopes
          or auth_scheme.flows.clientCredentials
          and auth_scheme.flows.clientCredentials.scopes
          or auth_scheme.flows.password
          and auth_scheme.flows.password.scopes
      )
      scopes = list(scopes.keys())

    client = OAuth2Session(
        auth_credential.oauth2.client_id,
        auth_credential.oauth2.client_secret,
        scope=" ".join(scopes),
        redirect_uri=auth_credential.oauth2.redirect_uri,
    )
    uri, state = client.create_authorization_url(
        url=authorization_endpoint, access_type="offline", prompt="consent"
    )
    exchanged_auth_credential = auth_credential.model_copy(deep=True)
    exchanged_auth_credential.oauth2.auth_uri = uri
    exchanged_auth_credential.oauth2.state = state

    return exchanged_auth_credential
