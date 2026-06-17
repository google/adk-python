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
) -> Event:
  """Creates an event requesting user authentication credentials.

  Args:
    auth_config: The auth configuration for the node.
    interrupt_id: The interrupt ID for this auth request.

  Returns:
    An Event containing an ``adk_request_credential`` function call.
  """
  auth_handler = AuthHandler(auth_config)
  auth_request = auth_handler.generate_auth_request()
  args = AuthToolArguments(
      function_call_id=interrupt_id,
      auth_config=auth_request,
  ).model_dump(exclude_none=True, by_alias=True)

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
) -> None:
  """Stores credentials from an auth resume response into session state.

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
  """
  try:
    response_config = AuthConfig.model_validate(response_data)
  except (ValidationError, TypeError):
    response_config = auth_config.model_copy(deep=True)
    response_config.exchanged_auth_credential = _build_credential_from_value(
        auth_config, response_data
    )

  response_config.credential_key = auth_config.credential_key
  await AuthHandler(auth_config=response_config).parse_and_store_auth_response(
      state=state
  )


def has_auth_credential(
    auth_config: AuthConfig,
    state: State,
) -> bool:
  """Returns True if a credential for the given auth config exists in state."""
  return AuthHandler(auth_config).get_auth_response(state) is not None
