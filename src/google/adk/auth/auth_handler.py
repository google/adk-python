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

from typing import TYPE_CHECKING

from fastapi.openapi.models import SecurityBase

from .auth_credential import AuthCredential
from .auth_schemes import AuthSchemeType
from .auth_schemes import OpenIdConnectWithConfig
from .auth_tool import AuthConfig
from .exchanger.oauth2_credential_exchanger import OAuth2CredentialExchanger

if TYPE_CHECKING:
  from ..sessions.state import State

try:
  from authlib.common.security import generate_token
  from authlib.integrations.requests_client import OAuth2Session

  AUTHLIB_AVAILABLE = True
except ImportError:
  AUTHLIB_AVAILABLE = False


def _normalize_oauth_scopes(
    scopes: dict[str, str] | list[str] | None,
) -> list[str]:
  """Normalize OAuth scopes into the list shape expected by authlib."""
  if not scopes:
    return []
  if isinstance(scopes, dict):
    return list(scopes.keys())
  return list(scopes)


def _credential_without_client_secret(
    credential: AuthCredential | None,
) -> AuthCredential | None:
  """Returns a copy of credential with the OAuth2 client secret removed."""
  if credential is None:
    return None
  redacted = credential.model_copy(deep=True)
  if redacted.oauth2 is not None:
    redacted.oauth2.client_secret = None
  return redacted


def _without_client_secret(auth_config: AuthConfig) -> AuthConfig:
  """Returns a copy of auth_config with OAuth2 client secrets removed.

  The auth request travels to, and is echoed back by, the client, and is
  persisted in the session. The client secret belongs to the agent, never to
  the end user, so it is stripped here and re-attached from the tool's own
  configuration when the token exchange happens.
  """
  redacted = auth_config.model_copy(deep=True)
  redacted.raw_auth_credential = _credential_without_client_secret(
      redacted.raw_auth_credential
  )
  redacted.exchanged_auth_credential = _credential_without_client_secret(
      redacted.exchanged_auth_credential
  )
  return redacted


class AuthHandler:
  """A handler that handles the auth flow in Agent Development Kit to help
  orchestrate the credential request and response flow (e.g. OAuth flow)
  This class should only be used by Agent Development Kit.
  """

  def __init__(self, auth_config: AuthConfig):
    self.auth_config = auth_config

  async def exchange_auth_token(
      self,
  ) -> AuthCredential:
    exchanger = OAuth2CredentialExchanger()
    exchange_result = await exchanger.exchange(
        self.auth_config.exchanged_auth_credential, self.auth_config.auth_scheme
    )
    return exchange_result.credential

  async def parse_and_store_auth_response(self, state: State) -> None:
    credential_key = self.auth_config.credential_key
    if not credential_key:
      raise ValueError("credential_key is empty.")

    temp_credential_key = "temp:" + credential_key

    self.auth_config.exchanged_auth_credential = self._with_configured_client(
        self.auth_config.exchanged_auth_credential
    )
    credential = self.auth_config.exchanged_auth_credential
    if self._is_exchangeable(credential):
      credential = await self.exchange_auth_token()

    # Session state is readable by the client, so the secret does not go in it.
    state[temp_credential_key] = _credential_without_client_secret(credential)

  def _validate(self) -> None:
    if not self.auth_config.auth_scheme:
      raise ValueError("auth_scheme is empty.")

  def _with_configured_client(
      self, credential: AuthCredential | None
  ) -> AuthCredential | None:
    """Returns credential with the configured OAuth2 client identity restored.

    The credential comes back from the client, which must not be able to
    choose which OAuth2 client the token is exchanged for. The original is
    left untouched, so the copy held in session state keeps no secret.
    """
    raw_credential = self.auth_config.raw_auth_credential
    if (
        credential is None
        or credential.oauth2 is None
        or raw_credential is None
        or raw_credential.oauth2 is None
    ):
      return credential
    restored = credential.model_copy(deep=True)
    restored.oauth2.client_id = raw_credential.oauth2.client_id
    restored.oauth2.client_secret = raw_credential.oauth2.client_secret
    return restored

  def _is_exchangeable(self, credential: AuthCredential | None) -> bool:
    """Returns whether credential still needs, and can do, a token exchange."""
    if not isinstance(
        self.auth_config.auth_scheme, SecurityBase
    ) or self.auth_config.auth_scheme.type_ not in (
        AuthSchemeType.oauth2,
        AuthSchemeType.openIdConnect,
    ):
      return False
    oauth2 = credential.oauth2 if credential else None
    return bool(
        oauth2
        and not oauth2.access_token
        and oauth2.client_id
        and oauth2.client_secret
    )

  def _read_stored_credential(
      self, state: State
  ) -> tuple[str, AuthCredential] | None:
    """Returns the state key and credential stored for this auth config."""
    credential_key = self.auth_config.credential_key
    if not credential_key:
      return None

    # The temp credential key is the standard ADK flow; the key without the
    # 'temp:' prefix is the fallback.
    for key in ("temp:" + credential_key, credential_key):
      val = state.get(key, None)
      if isinstance(val, AuthCredential):
        return key, val
      if isinstance(val, dict):
        return key, AuthCredential.model_validate(val)
      if isinstance(val, str) and val:
        return key, self._build_credential_from_string(val)

    return None

  def has_auth_response(self, state: State) -> bool:
    """Returns whether an auth response is stored, without exchanging it."""
    return self._read_stored_credential(state) is not None

  def get_auth_response(self, state: State) -> AuthCredential | None:
    """Returns the stored auth response, exchanging it for a token if needed.

    The stored response carries no client secret, since the auth request went
    through the client. The secret configured on this handler's auth config is
    re-attached here so the exchange can happen without ever trusting the
    client's copy, and is dropped again from what goes back into the session.

    The token request blocks the calling thread. Callers that can await should
    let `CredentialManager` do the exchange instead.
    """
    stored = self._read_stored_credential(state)
    if stored is None:
      return None

    key, credential = stored
    credential = self._with_configured_client(credential)
    if not self._is_exchangeable(credential):
      return credential

    exchange_result = OAuth2CredentialExchanger()._exchange_sync(
        credential, self.auth_config.auth_scheme
    )
    state[key] = _credential_without_client_secret(exchange_result.credential)
    return exchange_result.credential

  def _build_credential_from_string(self, val: str) -> AuthCredential:
    from .auth_credential import AuthCredentialTypes
    from .auth_credential import HttpAuth
    from .auth_credential import HttpCredentials
    from .auth_credential import OAuth2Auth

    auth_scheme = self.auth_config.auth_scheme
    if not auth_scheme:
      return AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(access_token=val),
      )

    scheme_type = auth_scheme.type_
    if scheme_type == AuthSchemeType.apiKey:
      return AuthCredential(
          auth_type=AuthCredentialTypes.API_KEY,
          api_key=val,
      )
    elif scheme_type == AuthSchemeType.http:
      scheme = getattr(auth_scheme, "scheme", "bearer")
      return AuthCredential(
          auth_type=AuthCredentialTypes.HTTP,
          http=HttpAuth(
              scheme=scheme,
              credentials=HttpCredentials(token=val),
          ),
      )
    elif scheme_type in (AuthSchemeType.oauth2, AuthSchemeType.openIdConnect):
      return AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(access_token=val),
      )
    else:
      return AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(access_token=val),
      )

  def generate_auth_request(self) -> AuthConfig:
    return _without_client_secret(self._generate_auth_request())

  def _generate_auth_request(self) -> AuthConfig:
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
          credential_key=self.auth_config.credential_key,
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
        credential_key=self.auth_config.credential_key,
    )

  def generate_auth_uri(
      self,
  ) -> AuthCredential:
    """Generates a response containing the auth uri for user to sign in.

    Returns:
        An AuthCredential object containing the auth URI and state.

    Raises:
        ValueError: If the authorization endpoint is not configured in the auth
            scheme.
    """
    if not AUTHLIB_AVAILABLE:
      return (
          self.auth_config.raw_auth_credential.model_copy(deep=True)
          if self.auth_config.raw_auth_credential
          else None
      )

    auth_scheme = self.auth_config.auth_scheme
    auth_credential = self.auth_config.raw_auth_credential
    if not auth_credential or not auth_credential.oauth2:
      raise ValueError("raw_auth_credential or oauth2 is empty")

    if isinstance(auth_scheme, OpenIdConnectWithConfig):
      authorization_endpoint = auth_scheme.authorization_endpoint
      scopes = _normalize_oauth_scopes(auth_scheme.scopes)
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
      if auth_scheme.flows.implicit:
        scopes = _normalize_oauth_scopes(auth_scheme.flows.implicit.scopes)
      elif auth_scheme.flows.authorizationCode:
        scopes = _normalize_oauth_scopes(
            auth_scheme.flows.authorizationCode.scopes
        )
      elif auth_scheme.flows.clientCredentials:
        scopes = _normalize_oauth_scopes(
            auth_scheme.flows.clientCredentials.scopes
        )
      elif auth_scheme.flows.password:
        scopes = _normalize_oauth_scopes(auth_scheme.flows.password.scopes)
      else:
        scopes = []

    client = OAuth2Session(
        auth_credential.oauth2.client_id,
        auth_credential.oauth2.client_secret,
        scope=" ".join(scopes),
        redirect_uri=auth_credential.oauth2.redirect_uri,
        code_challenge_method=auth_credential.oauth2.code_challenge_method,
    )
    params = {
        "access_type": "offline",
        "prompt": auth_credential.oauth2.prompt or "consent",
    }
    if auth_credential.oauth2.audience:
      params["audience"] = auth_credential.oauth2.audience
    if auth_credential.oauth2.nonce:
      params["nonce"] = auth_credential.oauth2.nonce

    # If using PKCE with S256, ensure a code_verifier exists.
    # If not provided in the credential, generate a cryptographically secure
    # random token of 48 characters (OAuth2 recommends 43-128 characters).
    code_verifier = auth_credential.oauth2.code_verifier
    method = auth_credential.oauth2.code_challenge_method

    if method:
      if method != "S256":
        raise ValueError(
            f"Unsupported code_challenge_method: {method}. Only 'S256' is"
            " supported."
        )
      if not code_verifier:
        code_verifier = generate_token(48)

    uri, state = client.create_authorization_url(
        url=authorization_endpoint, code_verifier=code_verifier, **params
    )

    exchanged_auth_credential = auth_credential.model_copy(deep=True)
    if exchanged_auth_credential.oauth2 is not None:
      exchanged_auth_credential.oauth2.auth_uri = uri
      exchanged_auth_credential.oauth2.state = state
      if code_verifier:
        exchanged_auth_credential.oauth2.code_verifier = code_verifier

    return exchanged_auth_credential
