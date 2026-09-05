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

"""Utility functions for Human-in-the-Loop (HITL) workflows."""

from __future__ import annotations

"""Utilities for ADK workflows."""

from collections.abc import Mapping
from typing import Any
from typing import TYPE_CHECKING

from google.genai import types
from pydantic import ValidationError

from ...auth.auth_credential import AuthCredentialTypes as _AuthCredentialTypes
from ...auth.auth_handler import AuthHandler
from ...auth.auth_tool import AuthConfig
from ...auth.auth_tool import AuthToolArguments
from ...events.event import Event
from ...events.request_input import RequestInput
from ...utils._schema_utils import schema_to_json_schema

if TYPE_CHECKING:
  from ...auth.auth_credential import AuthCredential
  from ...sessions.state import State

REQUEST_INPUT_FUNCTION_CALL_NAME = 'adk_request_input'
REQUEST_CREDENTIAL_FUNCTION_CALL_NAME = 'adk_request_credential'

_RESULT_KEY = 'result'
"""Key used to wrap non-dict values in a FunctionResponse dict."""


def create_request_input_event(request_input: RequestInput) -> Event:
  """Creates a RequestInput event from a RequestInput object."""
  args = request_input.model_dump(exclude={'response_schema'}, by_alias=True)
  args['response_schema'] = (
      schema_to_json_schema(request_input.response_schema)
      if request_input.response_schema is not None
      else None
  )
  return Event(
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      name=REQUEST_INPUT_FUNCTION_CALL_NAME,
                      args=args,
                      id=request_input.interrupt_id,
                  )
              )
          ],
      ),
      long_running_tool_ids=[request_input.interrupt_id],
  )


def has_request_input_function_call(event: Event) -> bool:
  """Checks if an event contains a `request_input` function call."""
  if not (event.content and event.content.parts):
    return False
  return any(
      p.function_call
      and p.function_call.name == REQUEST_INPUT_FUNCTION_CALL_NAME
      for p in event.content.parts
  )


def has_auth_request_function_call(event: Event) -> bool:
  """Checks if an event contains an `adk_request_credential` function call."""
  if not (event.content and event.content.parts):
    return False
  return any(
      p.function_call
      and p.function_call.name == REQUEST_CREDENTIAL_FUNCTION_CALL_NAME
      for p in event.content.parts
  )


def create_request_input_response(
    interrupt_id: str,
    response: Mapping[str, Any],
) -> types.Part:
  """Creates a FunctionResponse part in response to a `request_input` function call.

  Args:
    interrupt_id: The interrupt_id from an event containing a `request_input`
      function call.
    response: The response data to send back.

  Returns:
    A types.Part containing the FunctionResponse.
  """
  return types.Part(
      function_response=types.FunctionResponse(
          id=interrupt_id,
          name=REQUEST_INPUT_FUNCTION_CALL_NAME,
          response=response,
      )
  )


def get_request_input_interrupt_ids(event: Event) -> list[str]:
  """Extracts interrupt_ids from an event containing `request_input` function
  calls.
  """
  interrupt_ids: list[str] = []
  if not event.content or not event.content.parts:
    return interrupt_ids
  for part in event.content.parts:
    if (
        part.function_call
        and part.function_call.name == REQUEST_INPUT_FUNCTION_CALL_NAME
        and part.function_call.id is not None
    ):
      interrupt_ids.append(part.function_call.id)
  return interrupt_ids


# ---------------------------------------------------------------------------
# Auth credential utilities
# ---------------------------------------------------------------------------

_OAUTH_STATE_KEY_PREFIX = 'adk_oauth_state:'
"""Session state prefix under which a generated OAuth state is kept."""


def _oauth_state_key(interrupt_id: str) -> str:
  """Returns the session state key holding the generated OAuth state."""
  return f'{_OAUTH_STATE_KEY_PREFIX}{interrupt_id}'


def _build_auth_message(auth_config: AuthConfig) -> str:
  """Builds a human-readable message describing what credential is needed."""
  raw_cred = auth_config.raw_auth_credential
  if not raw_cred:
    return 'Please provide your authentication credentials.'

  auth_type = raw_cred.auth_type
  if auth_type == _AuthCredentialTypes.API_KEY:
    name = getattr(auth_config.auth_scheme, 'name', 'API key')
    return f'Please provide your API key for {name}.'
  elif auth_type in (
      _AuthCredentialTypes.OAUTH2,
      _AuthCredentialTypes.OPEN_ID_CONNECT,
  ):
    return 'Please complete the authentication flow.'

  return 'Please provide your authentication credentials.'


def _without_client_secret(auth_config: AuthConfig) -> AuthConfig:
  """Returns a copy of the auth config with the OAuth client secret removed.

  The auth request is handed to the caller, which needs the authorization uri
  and the state to complete the flow but never the developer's client secret.
  The secret stays on this side and is supplied again when the response comes
  back.

  Args:
    auth_config: The auth configuration for the node.
  """
  without_secret = auth_config.model_copy(deep=True)
  for credential in (
      without_secret.raw_auth_credential,
      without_secret.exchanged_auth_credential,
  ):
    if credential and credential.oauth2:
      credential.oauth2.client_secret = None
  return without_secret


def create_auth_request_event(
    auth_config: AuthConfig,
    interrupt_id: str,
    state: State,
) -> Event:
  """Creates an event requesting user authentication credentials.

  Args:
    auth_config: The auth configuration for the node.
    interrupt_id: The interrupt ID for this auth request.
    state: The session state. The OAuth state generated for this request is
      kept there so the resume response can be checked against it.

  Returns:
    An Event containing an ``adk_request_credential`` function call.
  """
  auth_handler = AuthHandler(auth_config)
  auth_request = auth_handler.generate_auth_request()
  generated_credential = auth_request.exchanged_auth_credential
  if (
      generated_credential
      and generated_credential.oauth2
      and generated_credential.oauth2.state
  ):
    state[_oauth_state_key(interrupt_id)] = generated_credential.oauth2.state
  args = AuthToolArguments(
      function_call_id=interrupt_id,
      auth_config=_without_client_secret(auth_request),
  ).model_dump(mode='json', exclude_none=True, by_alias=True)

  # Add message so the UI / CLI knows what to display.
  args['message'] = _build_auth_message(auth_config)

  return Event(
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      name=REQUEST_CREDENTIAL_FUNCTION_CALL_NAME,
                      id=interrupt_id,
                      args=args,
                  )
              )
          ],
      ),
      long_running_tool_ids=[interrupt_id],
  )


def _build_credential_from_value(
    auth_config: AuthConfig,
    value: Any,
) -> 'AuthCredential':
  """Builds an AuthCredential from a raw user-provided value.

  For API_KEY, the value is used as the key string directly.
  For all other types, the value is parsed as an AuthCredential dict.
  """
  from ...auth.auth_credential import AuthCredential

  raw_cred = auth_config.raw_auth_credential
  if raw_cred is None:
    return AuthCredential.model_validate(value)

  if raw_cred.auth_type == _AuthCredentialTypes.API_KEY:
    return AuthCredential(
        auth_type=_AuthCredentialTypes.API_KEY,
        api_key=str(value),
    )

  return AuthCredential.model_validate(value)


async def process_auth_resume(
    response_data: Any,
    auth_config: AuthConfig,
    state: State,
    interrupt_id: str,
) -> None:
  """Stores credentials from an auth resume response into session state.

  Only the credential is read from the response; the node's own auth config
  decides which scheme the credential is exchanged and stored under. When an
  OAuth state was generated for this request, the response must carry that
  same state back.

  Accepts multiple response formats (tried in order):
    1. A full AuthConfig dict (from web UI OAuth flow).
    2. An AuthCredential dict.
    3. A plain value (string for API key). The node's
       auth_config.raw_auth_credential.auth_type determines how the
       value is interpreted.

  The caller is responsible for unwrapping {"result": ...} wrappers
  before calling this function.

  Args:
    response_data: The unwrapped response from the client.
    auth_config: The original auth configuration for the node.
    state: The session state to store credentials in.
    interrupt_id: The interrupt ID of the auth request being resumed.

  Raises:
    ValueError: If the response does not carry back the OAuth state that was
      generated for this auth request.
  """
  try:
    exchanged_credential = AuthConfig.model_validate(
        response_data
    ).exchanged_auth_credential
  except (ValidationError, TypeError):
    exchanged_credential = _build_credential_from_value(
        auth_config, response_data
    )

  generated_state = state.get(_oauth_state_key(interrupt_id))
  if generated_state is not None:
    oauth2 = exchanged_credential.oauth2 if exchanged_credential else None
    if not oauth2 or oauth2.state != generated_state:
      raise ValueError(
          'The auth response does not carry back the state generated for this'
          ' auth request. Return the auth config from the credential request'
          ' with the authorization result filled in.'
      )

  resumed_config = auth_config.model_copy(deep=True)
  resumed_config.exchanged_auth_credential = exchanged_credential

  # The client secret is never handed out with the request, so the response
  # cannot carry it back; the node's own credential supplies it for the token
  # exchange.
  raw_oauth2 = (
      resumed_config.raw_auth_credential.oauth2
      if resumed_config.raw_auth_credential
      else None
  )
  exchanged_oauth2 = (
      resumed_config.exchanged_auth_credential.oauth2
      if resumed_config.exchanged_auth_credential
      else None
  )
  if raw_oauth2 and raw_oauth2.client_secret and exchanged_oauth2:
    exchanged_oauth2.client_secret = raw_oauth2.client_secret

  await AuthHandler(auth_config=resumed_config).parse_and_store_auth_response(
      state=state
  )


def has_auth_credential(
    auth_config: AuthConfig,
    state: State,
) -> bool:
  """Returns True if a credential for the given auth config exists in state."""
  return AuthHandler(auth_config).has_auth_response(state)
