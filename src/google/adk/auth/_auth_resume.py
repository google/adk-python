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

"""Shared helpers for finding auth requests and storing auth responses."""

from __future__ import annotations

from collections.abc import Collection
from collections.abc import Mapping
from collections.abc import Sequence
import logging
from typing import Any
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ..flows.llm_flows.tools._functions import REQUEST_EUC_FUNCTION_CALL_NAME
from .auth_credential import AuthCredential
from .auth_credential import AuthCredentialTypes
from .auth_handler import AuthHandler
from .auth_tool import AuthConfig
from .auth_tool import AuthToolArguments

if TYPE_CHECKING:
  from ..events.event import Event
  from ..sessions.state import State

logger = logging.getLogger("google_adk." + __name__)

_OAUTH_STATE_KEY_PREFIX = "adk_oauth_state:"
_OAUTH_CREDENTIAL_KEY_PREFIX = "adk_oauth_credential:"


def _oauth_state_key(interrupt_id: str) -> str:
  """Returns the session state key holding the generated OAuth state."""
  return f"{_OAUTH_STATE_KEY_PREFIX}{interrupt_id}"


def _oauth_credential_key(interrupt_id: str) -> str:
  """Returns the session state key holding the generated OAuth credential."""
  return f"{_OAUTH_CREDENTIAL_KEY_PREFIX}{interrupt_id}"


def _merge_credential_oauth2_fields(
    target_cred: AuthCredential | None,
    source_cred: AuthCredential | None,
) -> AuthCredential | None:
  """Backfills unset OAuth2 fields on `target_cred` from `source_cred`.

  If target_cred is None, returns source_cred.
  Otherwise, merges fields and returns target_cred.
  """
  if not source_cred:
    return target_cred
  if not target_cred:
    return source_cred

  if target_cred.oauth2 is None and source_cred.oauth2 is not None:
    target_cred.oauth2 = source_cred.oauth2.model_copy(deep=True)
  elif target_cred.oauth2 and source_cred.oauth2:
    target = target_cred.oauth2
    source = source_cred.oauth2
    for field in [
        "client_id",
        "client_secret",
        "redirect_uri",
        "code_verifier",
        "code_challenge_method",
    ]:
      if getattr(target, field) is None:
        setattr(target, field, getattr(source, field))

    # token_endpoint_auth_method has a default value "client_secret_basic" in
    # OAuth2Auth model. Only merge it if it wasn't explicitly set in target.
    target_fields_set = getattr(target, "model_fields_set", None)
    if (
        target_fields_set is None
        or "token_endpoint_auth_method" not in target_fields_set
    ):
      target.token_endpoint_auth_method = source.token_endpoint_auth_method

  return target_cred


def _build_credential_from_value(
    auth_config: AuthConfig,
    value: Any,
) -> AuthCredential:
  """Builds an AuthCredential from a raw user-provided value.

  For API_KEY, a bare string or numeric value is used as the key string
  directly.
  For all other cases, the value is parsed as an AuthCredential.
  """
  raw_cred = auth_config.raw_auth_credential
  if (
      raw_cred is not None
      and raw_cred.auth_type == AuthCredentialTypes.API_KEY
      and isinstance(value, (str, int, float))
      and not isinstance(value, bool)
  ):
    return AuthCredential(
        auth_type=AuthCredentialTypes.API_KEY,
        api_key=str(value),
    )
  return AuthCredential.model_validate(value)


def find_requested_auth_configs(
    events: Sequence[Event],
    interrupt_ids: Collection[str],
) -> dict[str, AuthToolArguments]:
  """Scans events and returns server-issued AuthToolArguments by call ID."""
  if not interrupt_ids:
    return {}
  target_ids = set(interrupt_ids)
  requested_by_id: dict[str, AuthToolArguments] = {}
  for event in events:
    event_function_calls = event.get_function_calls()
    if not event_function_calls:
      continue
    for function_call in event_function_calls:
      if (
          function_call.id in target_ids
          and function_call.name == REQUEST_EUC_FUNCTION_CALL_NAME
      ):
        try:
          requested_by_id[function_call.id] = AuthToolArguments.model_validate(
              function_call.args
          )
        except (ValidationError, TypeError):
          continue
  return requested_by_id


async def store_auth_response(
    *,
    requested: AuthConfig,
    response: Any,
    state: State,
    interrupt_id: str,
) -> AuthConfig | None:
  """Pins the client's auth response to `requested` and stores the credential.

  Accepts an `AuthConfig` dict/model, an `AuthCredential` dict/model, or a bare
  API-key string/number. When an OAuth state or PKCE verifier was recorded in
  `state` for `interrupt_id`, verifies the state round-trip and restores the
  PKCE verifier onto the credential before backfilling OAuth2 fields from
  `requested` and storing the credential via `AuthHandler`.

  Args:
    requested: The server-issued `AuthConfig` from the credential request or
      the node/tool's configured `AuthConfig` to pin against.
    response: The unwrapped auth response payload from the client.
    state: Session state where the credential is stored.
    interrupt_id: The interrupt / function call ID of the auth request.

  Returns:
    The pinned `AuthConfig` if valid and stored, or `None` if the response is
    malformed, carries no credential, or fails OAuth state verification.
  """
  try:
    exchanged_credential = AuthConfig.model_validate(
        response
    ).exchanged_auth_credential
  except (ValidationError, TypeError):
    try:
      exchanged_credential = _build_credential_from_value(requested, response)
    except (ValidationError, TypeError):
      logger.warning(
          "Ignoring malformed auth response for function call ID %r.",
          interrupt_id,
      )
      return None

  if exchanged_credential is None:
    logger.warning(
        "Ignoring auth response with no exchanged credential for function"
        " call ID %r.",
        interrupt_id,
    )
    return None

  generated_state = state.get(_oauth_state_key(interrupt_id))
  if generated_state is not None:
    oauth2 = exchanged_credential.oauth2
    if not oauth2 or oauth2.state != generated_state:
      logger.warning(
          "Ignoring auth response with mismatched OAuth state for function"
          " call ID %r.",
          interrupt_id,
      )
      return None

  credential_key = _oauth_credential_key(interrupt_id)
  stored_credential = state.get(credential_key)
  if isinstance(stored_credential, Mapping):
    try:
      stored_credential = AuthCredential.model_validate(stored_credential)
    except (ValidationError, TypeError):
      stored_credential = None
  if (
      isinstance(stored_credential, AuthCredential)
      and stored_credential.oauth2
      and exchanged_credential.oauth2
  ):
    if exchanged_credential.oauth2.code_verifier is None:
      exchanged_credential.oauth2.code_verifier = (
          stored_credential.oauth2.code_verifier
      )
    if exchanged_credential.oauth2.code_challenge_method is None:
      exchanged_credential.oauth2.code_challenge_method = (
          stored_credential.oauth2.code_challenge_method
      )
    state[credential_key] = None

  auth_config = requested.model_copy(deep=True)
  auth_config.exchanged_auth_credential = _merge_credential_oauth2_fields(
      exchanged_credential,
      auth_config.exchanged_auth_credential or auth_config.raw_auth_credential,
  )

  await AuthHandler(auth_config=auth_config).parse_and_store_auth_response(
      state=state
  )
  return auth_config
