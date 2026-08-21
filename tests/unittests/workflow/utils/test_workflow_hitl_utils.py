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

import json

from google.adk.events.event import Event
from google.adk.events.event import NodeInfo
from google.adk.events.request_input import RequestInput
from google.adk.workflow.utils._rehydration_utils import _ChildScanState
from google.adk.workflow.utils._workflow_hitl_utils import create_auth_request_event
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_event
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
from google.adk.workflow.utils._workflow_hitl_utils import get_request_input_interrupt_ids
from google.adk.workflow.utils._workflow_hitl_utils import has_auth_credential
from google.adk.workflow.utils._workflow_hitl_utils import has_request_input_function_call
from google.adk.workflow.utils._workflow_hitl_utils import process_auth_resume
from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_CREDENTIAL_FUNCTION_CALL_NAME
from google.genai import types
import pytest

# --- create_request_input_event ---


class TestCreateRequestInputEvent:

  def test_basic_event(self):
    ri = RequestInput(
        interrupt_id="test-id",
        message="Please approve",
    )
    event = create_request_input_event(ri)

    assert event.long_running_tool_ids == {"test-id"}
    assert event.content is not None
    assert event.content.role == "model"
    fc = event.content.parts[0].function_call
    assert fc.name == "adk_request_input"
    assert fc.id == "test-id"
    assert fc.args["message"] == "Please approve"

  def test_with_payload(self):
    ri = RequestInput(
        interrupt_id="id-1",
        payload={"key": "value"},
    )
    event = create_request_input_event(ri)
    fc = event.content.parts[0].function_call
    assert fc.args["payload"] == {"key": "value"}

  def test_with_response_schema(self):
    from pydantic import BaseModel

    class MySchema(BaseModel):
      approved: bool

    ri = RequestInput(
        interrupt_id="id-2",
        response_schema=MySchema,
    )
    event = create_request_input_event(ri)
    fc = event.content.parts[0].function_call
    schema = fc.args["response_schema"]
    assert "approved" in schema["properties"]
    assert schema["properties"]["approved"]["type"] == "boolean"


# --- has_request_input_function_call ---


class TestHasRequestInputFunctionCall:

  def test_true_for_request_input_event(self):
    event = create_request_input_event(
        RequestInput(interrupt_id="id-1", message="test")
    )
    assert has_request_input_function_call(event) is True

  def test_false_for_empty_event(self):
    assert has_request_input_function_call(Event()) is False

  def test_false_for_non_request_input(self):
    from google.genai import types

    event = Event(
        content=types.Content(
            parts=[
                types.Part(
                    function_call=types.FunctionCall(name="other_tool", args={})
                )
            ]
        )
    )
    assert has_request_input_function_call(event) is False


# --- create_request_input_response ---


class TestCreateRequestInputResponse:

  def test_creates_function_response_part(self):
    part = create_request_input_response("id-1", {"approved": True})
    assert part.function_response.id == "id-1"
    assert part.function_response.name == "adk_request_input"
    assert part.function_response.response == {"approved": True}


# --- get_request_input_interrupt_ids ---


class TestGetRequestInputInterruptIds:

  def test_extracts_ids(self):
    event = create_request_input_event(
        RequestInput(interrupt_id="id-1", message="test")
    )
    assert get_request_input_interrupt_ids(event) == ["id-1"]

  def test_empty_for_no_function_calls(self):
    assert get_request_input_interrupt_ids(Event()) == []

  def test_empty_for_non_request_input(self):
    from google.genai import types

    event = Event(
        content=types.Content(
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="other_tool", args={}, id="id-1"
                    )
                )
            ]
        )
    )
    assert get_request_input_interrupt_ids(event) == []


# --- create_auth_request_event ---


class TestCreateAuthRequestEvent:

  def test_creates_credential_request(self):
    from fastapi.openapi.models import APIKey
    from fastapi.openapi.models import APIKeyIn
    from google.adk.auth.auth_credential import AuthCredential
    from google.adk.auth.auth_credential import AuthCredentialTypes
    from google.adk.auth.auth_tool import AuthConfig

    auth_config = AuthConfig(
        auth_scheme=APIKey(**{"in": APIKeyIn.header, "name": "X-Api-Key"}),
        raw_auth_credential=AuthCredential(
            auth_type=AuthCredentialTypes.API_KEY,
            api_key="test_key",
        ),
        credential_key="test_cred",
    )
    event = create_auth_request_event(auth_config, "auth-id-1", _empty_state())

    assert event.long_running_tool_ids is not None
    fc = event.content.parts[0].function_call
    assert fc.name == REQUEST_CREDENTIAL_FUNCTION_CALL_NAME
    assert fc.id == "auth-id-1"
    assert "authConfig" in fc.args

  def test_args_are_json_serializable(self):
    from fastapi.openapi.models import OAuth2
    from fastapi.openapi.models import OAuthFlowAuthorizationCode
    from fastapi.openapi.models import OAuthFlows
    from google.adk.auth.auth_credential import AuthCredential
    from google.adk.auth.auth_credential import AuthCredentialTypes
    from google.adk.auth.auth_credential import OAuth2Auth
    from google.adk.auth.auth_tool import AuthConfig

    auth_config = AuthConfig(
        auth_scheme=OAuth2(
            flows=OAuthFlows(
                authorizationCode=OAuthFlowAuthorizationCode(
                    authorizationUrl=(
                        "https://accounts.google.com/o/oauth2/auth"
                    ),
                    tokenUrl="https://oauth2.googleapis.com/token",
                    scopes={
                        "https://www.googleapis.com/auth/calendar": (
                            "See calendars"
                        )
                    },
                )
            )
        ),
        raw_auth_credential=AuthCredential(
            auth_type=AuthCredentialTypes.OAUTH2,
            oauth2=OAuth2Auth(
                client_id="oauth_client_id",
                client_secret="oauth_client_secret",
            ),
        ),
    )
    event = create_auth_request_event(auth_config, "auth-id-1", _empty_state())

    fc = event.content.parts[0].function_call

    # python-mode dump leaves auth_scheme.type a live enum, breaking json.dumps
    json.dumps(fc.args)
    assert fc.args["authConfig"]["authScheme"]["type"] == "oauth2"


# --- process_auth_resume / has_auth_credential ---


def _api_key_auth_config(credential_key: str = "node-cred"):
  """An API-key AuthConfig, the simplest resume shape (no token exchange)."""
  from fastapi.openapi.models import APIKey
  from fastapi.openapi.models import APIKeyIn
  from google.adk.auth.auth_credential import AuthCredential
  from google.adk.auth.auth_credential import AuthCredentialTypes
  from google.adk.auth.auth_tool import AuthConfig

  return AuthConfig(
      auth_scheme=APIKey(**{"in": APIKeyIn.header, "name": "X-Api-Key"}),
      raw_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.API_KEY,
          api_key="placeholder",
      ),
      credential_key=credential_key,
  )


def _empty_state():
  from google.adk.sessions.state import State

  return State(value={}, delta={})


class TestProcessAuthResume:

  @pytest.mark.asyncio
  async def test_plain_value_becomes_api_key_credential(self):
    """A bare string resume response is interpreted per the raw credential type."""
    from google.adk.auth.auth_credential import AuthCredentialTypes

    auth_config = _api_key_auth_config()
    state = _empty_state()
    assert has_auth_credential(auth_config, state) is False

    await process_auth_resume(
        "user-supplied-key", auth_config, state, "auth-id-1"
    )

    stored = state["temp:node-cred"]
    assert stored.auth_type == AuthCredentialTypes.API_KEY
    assert stored.api_key == "user-supplied-key"
    assert has_auth_credential(auth_config, state) is True

  @pytest.mark.asyncio
  async def test_auth_config_response_stores_exchanged_credential(self):
    """A full AuthConfig response is accepted and its exchanged credential kept."""
    from google.adk.auth.auth_credential import AuthCredential
    from google.adk.auth.auth_credential import AuthCredentialTypes

    auth_config = _api_key_auth_config()
    state = _empty_state()
    response = auth_config.model_copy(deep=True)
    response.exchanged_auth_credential = AuthCredential(
        auth_type=AuthCredentialTypes.API_KEY,
        api_key="from-web-flow",
    )

    await process_auth_resume(
        response.model_dump(mode="json", exclude_none=True, by_alias=True),
        auth_config,
        state,
        "auth-id-1",
    )

    assert state["temp:node-cred"].api_key == "from-web-flow"

  @pytest.mark.asyncio
  async def test_response_cannot_redirect_storage_to_another_credential_key(
      self,
  ):
    """The node's own credential_key wins over one supplied in the response.

    Otherwise a resume payload could park the credential under a key the node
    never reads, leaving the node permanently unauthenticated.
    """
    from google.adk.auth.auth_credential import AuthCredential
    from google.adk.auth.auth_credential import AuthCredentialTypes

    auth_config = _api_key_auth_config(credential_key="node-cred")
    state = _empty_state()
    response = _api_key_auth_config(credential_key="unrelated-cred")
    response.exchanged_auth_credential = AuthCredential(
        auth_type=AuthCredentialTypes.API_KEY,
        api_key="k",
    )

    await process_auth_resume(
        response.model_dump(mode="json", exclude_none=True, by_alias=True),
        auth_config,
        state,
        "auth-id-1",
    )

    assert "temp:node-cred" in state
    assert "temp:unrelated-cred" not in state
    assert has_auth_credential(auth_config, state) is True


def _oauth_auth_config(token_url: str = "https://provider.example.com/token"):
  """An OAuth2 AuthConfig, the resume shape that runs a token exchange."""
  from fastapi.openapi.models import OAuth2
  from fastapi.openapi.models import OAuthFlowAuthorizationCode
  from fastapi.openapi.models import OAuthFlows
  from google.adk.auth.auth_credential import AuthCredential
  from google.adk.auth.auth_credential import AuthCredentialTypes
  from google.adk.auth.auth_credential import OAuth2Auth
  from google.adk.auth.auth_tool import AuthConfig

  return AuthConfig(
      auth_scheme=OAuth2(
          flows=OAuthFlows(
              authorizationCode=OAuthFlowAuthorizationCode(
                  authorizationUrl="https://provider.example.com/auth",
                  tokenUrl=token_url,
                  scopes={"read": "Read access"},
              )
          )
      ),
      raw_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(
              client_id="client-id",
              client_secret="client-secret",
          ),
      ),
      credential_key="node-cred",
  )


def _oauth_resume_response(auth_config, state_value: str):
  """The AuthConfig dict a client sends back after the authorization step."""
  from google.adk.auth.auth_credential import AuthCredential
  from google.adk.auth.auth_credential import AuthCredentialTypes
  from google.adk.auth.auth_credential import OAuth2Auth

  response = auth_config.model_copy(deep=True)
  response.exchanged_auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id="client-id",
          client_secret="client-secret",
          state=state_value,
          auth_code="authorization-code",
      ),
  )
  return response.model_dump(mode="json", exclude_none=True, by_alias=True)


def _requested_state(event) -> str:
  """Reads the OAuth state ADK generated, as the client receives it."""
  args = event.content.parts[0].function_call.args
  return args["authConfig"]["exchangedAuthCredential"]["oauth2"]["state"]


class TestProcessAuthResumeOAuth:

  @pytest.fixture(autouse=True)
  def _no_network_exchange(self, monkeypatch):
    """Records the auth scheme each exchange runs against, without network."""
    from google.adk.auth import auth_handler as auth_handler_module
    from google.adk.auth.exchanger.base_credential_exchanger import ExchangeResult

    self.exchanged_schemes = []
    recorded = self.exchanged_schemes

    class _RecordingExchanger:

      async def exchange(self, auth_credential, auth_scheme=None):
        recorded.append(auth_scheme)
        return ExchangeResult(auth_credential, True)

    monkeypatch.setattr(
        auth_handler_module,
        "OAuth2CredentialExchanger",
        _RecordingExchanger,
    )

  @pytest.mark.asyncio
  async def test_echoed_state_is_accepted(self):
    auth_config = _oauth_auth_config()
    state = _empty_state()
    event = create_auth_request_event(auth_config, "auth-id-1", state)

    await process_auth_resume(
        _oauth_resume_response(auth_config, _requested_state(event)),
        auth_config,
        state,
        "auth-id-1",
    )

    assert has_auth_credential(auth_config, state) is True

  @pytest.mark.asyncio
  async def test_response_with_another_state_is_rejected(self):
    """A response that does not echo the generated state is not exchanged."""
    auth_config = _oauth_auth_config()
    state = _empty_state()
    create_auth_request_event(auth_config, "auth-id-1", state)

    with pytest.raises(ValueError):
      await process_auth_resume(
          _oauth_resume_response(auth_config, "some-other-state"),
          auth_config,
          state,
          "auth-id-1",
      )

    assert self.exchanged_schemes == []
    assert has_auth_credential(auth_config, state) is False

  @pytest.mark.asyncio
  async def test_response_cannot_choose_the_token_endpoint(self):
    """The node's own scheme decides where the credential is exchanged."""
    auth_config = _oauth_auth_config(
        token_url="https://provider.example.com/token"
    )
    state = _empty_state()
    event = create_auth_request_event(auth_config, "auth-id-1", state)
    response = _oauth_resume_response(
        _oauth_auth_config(token_url="https://elsewhere.example.com/token"),
        _requested_state(event),
    )

    await process_auth_resume(response, auth_config, state, "auth-id-1")

    assert len(self.exchanged_schemes) == 1
    assert (
        self.exchanged_schemes[0].flows.authorizationCode.tokenUrl
        == "https://provider.example.com/token"
    )


class TestHasAuthCredential:

  @pytest.mark.asyncio
  async def test_false_for_a_different_credential_key(self):
    """Credentials are looked up per credential_key, not shared across configs."""

    auth_config = _api_key_auth_config(credential_key="node-cred")
    other_config = _api_key_auth_config(credential_key="other-cred")
    state = _empty_state()

    await process_auth_resume("key", auth_config, state, "auth-id-1")

    assert has_auth_credential(auth_config, state) is True
    assert has_auth_credential(other_config, state) is False


#
