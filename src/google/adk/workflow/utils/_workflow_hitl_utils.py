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

from collections.abc import Mapping
from typing import Any
from typing import TYPE_CHECKING

from google.genai import types

from ...auth.auth_credential import AuthCredential
from ...auth.auth_credential import AuthCredentialTypes as _AuthCredentialTypes
from ...auth.auth_credential import OAuth2Auth
from ...events.event import Event
from ...events.request_input import RequestInput
from ...utils._function_call_names import REQUEST_EUC_FUNCTION_CALL_NAME
from ...utils._function_call_names import REQUEST_INPUT_FUNCTION_CALL_NAME
from ...utils._schema_utils import schema_to_json_schema
from .._errors import WorkflowDataError

if TYPE_CHECKING:
  from ...auth.auth_tool import AuthConfig
  from ...sessions.state import State


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
      p.function_call and p.function_call.name == REQUEST_EUC_FUNCTION_CALL_NAME
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


def create_auth_request_event(
    auth_config: AuthConfig,
    interrupt_id: str,
    state: State,
) -> Event:
  """Creates an event requesting user authentication credentials.

  Args:
    auth_config: The auth configuration for the node.
    interrupt_id: The interrupt ID for this auth request.
    state: The session state. The OAuth state and any PKCE verifier
      credential generated for this request are kept there for
      ``process_auth_resume``.

  Returns:
    An Event containing an ``adk_request_credential`` function call.
  """
  from ...auth._auth_resume import _oauth_credential_key
  from ...auth._auth_resume import _oauth_state_key
  from ...auth.auth_handler import AuthHandler
  from ...auth.auth_tool import AuthToolArguments

  auth_handler = AuthHandler(auth_config)
  auth_request = auth_handler.generate_auth_request()
  generated_credential = auth_request.exchanged_auth_credential
  if generated_credential and generated_credential.oauth2:
    if generated_credential.oauth2.state:
      state[_oauth_state_key(interrupt_id)] = generated_credential.oauth2.state
    if generated_credential.oauth2.code_verifier:
      state[_oauth_credential_key(interrupt_id)] = AuthCredential(
          auth_type=generated_credential.auth_type,
          oauth2=OAuth2Auth(
              code_verifier=generated_credential.oauth2.code_verifier,
              code_challenge_method=(
                  generated_credential.oauth2.code_challenge_method
              ),
          ),
      )
      generated_credential.oauth2.code_verifier = None
  args = AuthToolArguments(
      function_call_id=interrupt_id,
      auth_config=auth_request,
  ).model_dump(mode='json', exclude_none=True, by_alias=True)

  # Add message so the UI / CLI knows what to display.
  args['message'] = _build_auth_message(auth_config)

  return Event(
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      name=REQUEST_EUC_FUNCTION_CALL_NAME,
                      id=interrupt_id,
                      args=args,
                  )
              )
          ],
      ),
      long_running_tool_ids=[interrupt_id],
  )


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
  same state back. Any PKCE verifier cached in ``state`` for this request is
  restored onto the credential and cleared from ``state``.

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
    state: The session state to read the cached OAuth state and PKCE verifier
      from, and to store exchanged credentials in.
    interrupt_id: The interrupt ID of the auth request being resumed.

  Raises:
    WorkflowDataError: If the response is malformed or does not carry back the
      OAuth state that was generated for this auth request.
  """
  from ...auth._auth_resume import store_auth_response

  stored = await store_auth_response(
      requested=auth_config,
      response=response_data,
      state=state,
      interrupt_id=interrupt_id,
  )
  if stored is None:
    raise WorkflowDataError(
        'The auth response is malformed or does not carry back the state'
        ' generated for this auth request. Return the auth config from the'
        ' credential request with the authorization result filled in.'
    )


def has_auth_credential(
    auth_config: AuthConfig,
    state: State,
) -> bool:
  """Returns True if a credential for the given auth config exists in state."""
  from ...auth.auth_handler import AuthHandler

  return AuthHandler(auth_config).has_auth_response(state)
