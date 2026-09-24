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

import logging
from typing import Optional
from typing import Tuple

from authlib.integrations.requests_client import OAuth2Session
from authlib.oauth2.rfc6749 import OAuth2Token
from fastapi.openapi.models import OAuth2

from ..utils import _mtls_utils
from ..utils.feature_decorator import experimental
from .auth_credential import AuthCredential
from .auth_schemes import AuthScheme
from .auth_schemes import OpenIdConnectWithConfig

logger = logging.getLogger("google_adk." + __name__)

# Token exchange and refresh run on worker threads, so a token endpoint that
# accepts the connection but never answers would hold a thread forever without
# this bound.
_TOKEN_REQUEST_TIMEOUT_SECONDS = 10


def _credential_without_client_secret(
    credential: Optional[AuthCredential],
) -> Optional[AuthCredential]:
  """Returns a copy of credential with the OAuth2 client secret removed.

  The client secret identifies the agent's OAuth2 client, not the end user, so
  it must not travel to the client or reach any store the client can read. Call
  sites that still need it for a token request re-attach it from the tool's own
  configuration, so dropping it here costs nothing.
  """
  if credential is None:
    return None
  redacted = credential.model_copy(deep=True)
  if redacted.oauth2 is not None:
    redacted.oauth2.client_secret = None
  return redacted


def _with_configured_client_secret(
    *,
    credential: Optional[AuthCredential],
    raw_credential: Optional[AuthCredential],
) -> Optional[AuthCredential]:
  """Returns credential with the configured OAuth2 client secret restored.

  The inverse of `_credential_without_client_secret`: a credential read back
  from a store that holds no secret needs one again before a token exchange or
  refresh. Only the secret is put back, so the rest of the stored credential
  round trips untouched.
  """
  if (
      raw_credential is None
      or raw_credential.oauth2 is None
      or credential is None
      or credential.oauth2 is None
  ):
    return credential
  restored = credential.model_copy(deep=True)
  if restored.oauth2 is not None:
    restored.oauth2.client_secret = raw_credential.oauth2.client_secret
  return restored


def _with_configured_client(
    *,
    credential: Optional[AuthCredential],
    raw_credential: Optional[AuthCredential],
) -> Optional[AuthCredential]:
  """Returns credential with the whole configured OAuth2 client restored.

  For credentials that came back through the client, which must not be able to
  pick which OAuth2 client its token is exchanged for, so the client id is
  pinned to the tool's own configuration along with the secret.
  """
  restored = _with_configured_client_secret(
      credential=credential, raw_credential=raw_credential
  )
  if (
      raw_credential is None
      or raw_credential.oauth2 is None
      or restored is None
      or restored.oauth2 is None
  ):
    return restored
  restored.oauth2.client_id = raw_credential.oauth2.client_id
  return restored


@experimental
def create_oauth2_session(
    auth_scheme: AuthScheme,
    auth_credential: AuthCredential,
) -> Tuple[Optional[OAuth2Session], Optional[str]]:
  """Create an OAuth2 session for token operations.

  Args:
      auth_scheme: The authentication scheme configuration.
      auth_credential: The authentication credential.

  Returns:
      Tuple of (OAuth2Session, token_endpoint) or (None, None) if cannot create session.
  """
  if isinstance(auth_scheme, OpenIdConnectWithConfig):
    if not hasattr(auth_scheme, "token_endpoint"):
      logger.warning("OpenIdConnect scheme missing token_endpoint")
      return None, None
    token_endpoint = auth_scheme.token_endpoint
  elif isinstance(auth_scheme, OAuth2):
    # Support both authorization code and client credentials flows
    if (
        auth_scheme.flows.authorizationCode
        and auth_scheme.flows.authorizationCode.tokenUrl
    ):
      token_endpoint = auth_scheme.flows.authorizationCode.tokenUrl
    elif (
        auth_scheme.flows.clientCredentials
        and auth_scheme.flows.clientCredentials.tokenUrl
    ):
      token_endpoint = auth_scheme.flows.clientCredentials.tokenUrl
    else:
      logger.warning(
          "OAuth2 scheme missing required flow configuration. Expected either"
          " authorizationCode.tokenUrl or clientCredentials.tokenUrl. Auth"
          " scheme: %s",
          auth_scheme,
      )
      return None, None
  else:
    logger.warning(f"Unsupported auth_scheme type: {type(auth_scheme)}")
    return None, None

  if (
      not auth_credential
      or not auth_credential.oauth2
      or not auth_credential.oauth2.client_id
      or not auth_credential.oauth2.client_secret
  ):
    return None, None

  # Scope is intentionally omitted: token exchange and refresh don't require
  # it per RFC 6749, and some providers reject it on these requests.
  session = OAuth2Session(
      auth_credential.oauth2.client_id,
      auth_credential.oauth2.client_secret,
      redirect_uri=auth_credential.oauth2.redirect_uri,
      state=auth_credential.oauth2.state,
      token_endpoint_auth_method=auth_credential.oauth2.token_endpoint_auth_method,
      code_challenge_method=auth_credential.oauth2.code_challenge_method,
      default_timeout=_TOKEN_REQUEST_TIMEOUT_SECONDS,
  )

  # When a client certificate is configured, route Google token requests through
  # the mTLS endpoint and present the cert so Context-Aware Access / token
  # binding is honored. Non-Google providers and non-cert environments keep the
  # existing behavior.
  if (
      _mtls_utils.is_non_mtls_googleapis_endpoint(token_endpoint)
      and _mtls_utils.use_client_cert_effective()
  ):
    if _mtls_utils.configure_session_for_mtls(session):
      token_endpoint = _mtls_utils.effective_googleapis_endpoint(token_endpoint)

  return session, token_endpoint


@experimental
def update_credential_with_tokens(
    auth_credential: AuthCredential, tokens: OAuth2Token
) -> None:
  """Update the credential with new tokens.

  Args:
      auth_credential: The authentication credential to update.
      tokens: The OAuth2Token object containing new token information.
  """
  if auth_credential.oauth2 and tokens:
    auth_credential.oauth2.access_token = tokens.get("access_token")
    auth_credential.oauth2.refresh_token = tokens.get("refresh_token")
    auth_credential.oauth2.id_token = tokens.get("id_token")
    auth_credential.oauth2.expires_at = (
        int(tokens.get("expires_at")) if tokens.get("expires_at") else None
    )
    auth_credential.oauth2.expires_in = (
        int(tokens.get("expires_in")) if tokens.get("expires_in") else None
    )
