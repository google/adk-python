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

"""Unit tests for shared auth resume helpers in google.adk.auth._auth_resume."""

from __future__ import annotations

from fastapi.openapi.models import APIKey
from fastapi.openapi.models import APIKeyIn
from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlowAuthorizationCode
from fastapi.openapi.models import OAuthFlows
from google.adk.auth import auth_handler as auth_handler_module
from google.adk.auth._auth_resume import _oauth_credential_key
from google.adk.auth._auth_resume import _oauth_state_key
from google.adk.auth._auth_resume import find_requested_auth_configs
from google.adk.auth._auth_resume import store_auth_response
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_tool import AuthToolArguments
from google.adk.auth.exchanger.base_credential_exchanger import ExchangeResult
from google.adk.events.event import Event
from google.adk.sessions.state import State
from google.genai import types
import pytest


def _empty_state() -> State:
  return State(value={}, delta={})


def _api_key_auth_config(credential_key: str = "server_api_key") -> AuthConfig:
  return AuthConfig(
      auth_scheme=APIKey(**{"in": APIKeyIn.header, "name": "X-Api-Key"}),
      raw_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.API_KEY,
          api_key="placeholder",
      ),
      credential_key=credential_key,
  )


def _oauth_auth_config(
    *,
    credential_key: str = "server_oauth_cred",
    token_url: str = "https://provider.example.com/token",
    state_token: str | None = None,
) -> AuthConfig:
  exchanged = None
  if state_token is not None:
    exchanged = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id="server-client-id",
            state=state_token,
            redirect_uri="https://app.example.com/callback",
            code_verifier="server-verifier",
            code_challenge_method="S256",
        ),
    )
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
              client_id="server-client-id",
              client_secret="server-client-secret",
              redirect_uri="https://app.example.com/callback",
              code_verifier="server-verifier",
              code_challenge_method="S256",
          ),
      ),
      exchanged_auth_credential=exchanged,
      credential_key=credential_key,
  )


def _auth_request_event(
    fc_id: str,
    original_fc_id: str,
    auth_config: AuthConfig,
) -> Event:
  args = AuthToolArguments(
      function_call_id=original_fc_id,
      auth_config=auth_config,
  ).model_dump(mode="json", exclude_none=True, by_alias=True)
  return Event(
      author="model",
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id=fc_id,
                      name="adk_request_credential",
                      args=args,
                  )
              )
          ],
      ),
  )


class TestFindRequestedAuthConfigs:
  """Tests for find_requested_auth_configs."""

  def test_returns_matching_requests_and_ignores_unknown_ids(self):
    """Returns matched AuthToolArguments and ignores unrequested call IDs."""
    cfg1 = _api_key_auth_config("cred_1")
    cfg2 = _api_key_auth_config("cred_2")
    other_event = Event(
        author="model",
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id="fc-other",
                        name="some_other_tool",
                        args={"x": 1},
                    )
                )
            ],
        ),
    )
    events = [
        other_event,
        _auth_request_event("auth-1", "tool-1", cfg1),
        _auth_request_event("auth-2", "tool-2", cfg2),
    ]

    found = find_requested_auth_configs(events, {"auth-1", "unknown-id"})

    assert set(found.keys()) == {"auth-1"}
    assert found["auth-1"].function_call_id == "tool-1"
    assert found["auth-1"].auth_config.credential_key == "cred_1"

  def test_skips_malformed_auth_tool_arguments(self):
    """Skips adk_request_credential calls whose args fail validation."""
    bad_event = Event(
        author="model",
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id="auth-bad",
                        name="adk_request_credential",
                        args={"not_valid": 123},
                    )
                )
            ],
        ),
    )

    found = find_requested_auth_configs([bad_event], {"auth-bad"})

    assert found == {}


class TestStoreAuthResponse:
  """Tests for store_auth_response."""

  @pytest.fixture(autouse=True)
  def _no_network_exchange(self, monkeypatch):
    """Replaces OAuth2CredentialExchanger with an in-memory recorder."""
    self.exchanged_schemes = []
    self.exchanged_credentials = []
    recorded_schemes = self.exchanged_schemes
    recorded_credentials = self.exchanged_credentials

    class _RecordingExchanger:

      async def exchange(self, auth_credential, auth_scheme=None):
        recorded_schemes.append(auth_scheme)
        recorded_credentials.append(auth_credential)
        return ExchangeResult(auth_credential, True)

    monkeypatch.setattr(
        auth_handler_module,
        "OAuth2CredentialExchanger",
        _RecordingExchanger,
    )

  @pytest.mark.asyncio
  async def test_accepts_auth_config_payload(self):
    """Stores the exchanged credential from a full AuthConfig dict."""
    requested = _api_key_auth_config("cred_key")
    state = _empty_state()
    response_cfg = requested.model_copy(deep=True)
    response_cfg.exchanged_auth_credential = AuthCredential(
        auth_type=AuthCredentialTypes.API_KEY,
        api_key="from-auth-config",
    )

    stored = await store_auth_response(
        requested=requested,
        response=response_cfg.model_dump(
            mode="json", exclude_none=True, by_alias=True
        ),
        state=state,
        interrupt_id="int-1",
    )

    assert stored is not None
    assert state["temp:cred_key"].api_key == "from-auth-config"

  @pytest.mark.asyncio
  async def test_accepts_auth_credential_payload(self):
    """Stores the credential from an AuthCredential dict."""
    requested = _api_key_auth_config("cred_key")
    state = _empty_state()
    cred = AuthCredential(
        auth_type=AuthCredentialTypes.API_KEY,
        api_key="from-auth-credential",
    )

    stored = await store_auth_response(
        requested=requested,
        response=cred.model_dump(mode="json", exclude_none=True, by_alias=True),
        state=state,
        interrupt_id="int-1",
    )

    assert stored is not None
    assert state["temp:cred_key"].api_key == "from-auth-credential"

  @pytest.mark.asyncio
  @pytest.mark.parametrize(
      "payload,expected",
      [
          ("bare-api-key-value", "bare-api-key-value"),
          (12345678, "12345678"),
          (12.5, "12.5"),
      ],
  )
  async def test_accepts_bare_api_key_payloads(self, payload, expected):
    """Converts bare string and numeric payloads into API_KEY credentials."""
    requested = _api_key_auth_config("cred_key")
    state = _empty_state()

    stored = await store_auth_response(
        requested=requested,
        response=payload,
        state=state,
        interrupt_id="int-1",
    )

    assert stored is not None
    assert state["temp:cred_key"].api_key == expected

  @pytest.mark.asyncio
  async def test_rejects_boolean_as_bare_api_key(self):
    """Rejects boolean payloads even though bool subclasses int."""
    requested = _api_key_auth_config("cred_key")
    state = _empty_state()

    stored = await store_auth_response(
        requested=requested,
        response=True,
        state=state,
        interrupt_id="int-1",
    )

    assert stored is None
    assert "temp:cred_key" not in state

  @pytest.mark.asyncio
  async def test_rejects_auth_config_with_missing_exchanged_credential(self):
    """Rejects an AuthConfig payload when exchanged_auth_credential is None."""
    requested = _api_key_auth_config("cred_key")
    state = _empty_state()
    echoed_without_credential = requested.model_copy(deep=True)
    echoed_without_credential.exchanged_auth_credential = None

    stored = await store_auth_response(
        requested=requested,
        response=echoed_without_credential.model_dump(
            mode="json", exclude_none=True, by_alias=True
        ),
        state=state,
        interrupt_id="int-1",
    )

    assert stored is None
    assert "temp:cred_key" not in state

  @pytest.mark.asyncio
  @pytest.mark.parametrize(
      "malformed",
      ["not-a-credential", {"auth_scheme": 7}, 12345],
  )
  async def test_malformed_response_returns_none_and_does_not_store(
      self, malformed
  ):
    """Returns None without modifying state when the response is malformed."""
    requested = _oauth_auth_config(credential_key="oauth_cred")
    state = _empty_state()

    stored = await store_auth_response(
        requested=requested,
        response=malformed,
        state=state,
        interrupt_id="int-1",
    )

    assert stored is None
    assert "temp:oauth_cred" not in state
    assert self.exchanged_schemes == []

  @pytest.mark.asyncio
  async def test_pins_credential_key_and_token_endpoint_to_requested(self):
    """Pins credential_key, tokenUrl, and client_secret to requested config."""
    requested = _oauth_auth_config(
        credential_key="pinned_key",
        token_url="https://provider.example.com/token",
        state_token="expected-state",
    )
    state = _empty_state()

    forged = _oauth_auth_config(
        credential_key="attacker_key",
        token_url="https://attacker.example.com/token",
    )
    forged.exchanged_auth_credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            auth_code="code-123",
            state="expected-state",
        ),
    )

    stored = await store_auth_response(
        requested=requested,
        response=forged.model_dump(
            mode="json", exclude_none=True, by_alias=True
        ),
        state=state,
        interrupt_id="int-1",
    )

    assert stored is not None
    assert stored.credential_key == "pinned_key"
    assert "temp:pinned_key" in state
    assert "temp:attacker_key" not in state
    assert len(self.exchanged_schemes) == 1
    assert (
        self.exchanged_schemes[0].flows.authorizationCode.tokenUrl
        == "https://provider.example.com/token"
    )
    assert (
        self.exchanged_credentials[0].oauth2.client_secret
        == "server-client-secret"
    )
    assert (
        self.exchanged_credentials[0].oauth2.code_verifier == "server-verifier"
    )

  @pytest.mark.asyncio
  async def test_backfills_oauth2_fields_from_raw_auth_credential(self):
    """Backfills OAuth2 fields from raw_auth_credential when needed."""
    requested = _oauth_auth_config(
        credential_key="workflow_oauth_cred",
        state_token=None,
    )
    assert requested.exchanged_auth_credential is None
    state = _empty_state()

    response = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(auth_code="code-456"),
    )

    stored = await store_auth_response(
        requested=requested,
        response=response.model_dump(
            mode="json", exclude_none=True, by_alias=True
        ),
        state=state,
        interrupt_id="int-1",
    )

    assert stored is not None
    assert len(self.exchanged_credentials) == 1
    exchanged_oauth2 = self.exchanged_credentials[0].oauth2
    assert exchanged_oauth2.client_id == "server-client-id"
    assert exchanged_oauth2.client_secret == "server-client-secret"
    assert exchanged_oauth2.redirect_uri == "https://app.example.com/callback"
    assert exchanged_oauth2.code_verifier == "server-verifier"
    assert exchanged_oauth2.code_challenge_method == "S256"

  @pytest.mark.asyncio
  async def test_restores_and_clears_pkce_code_verifier_from_session_state(
      self,
  ):
    """Restores cached PKCE code_verifier from session state and clears it."""
    requested = _oauth_auth_config(
        credential_key="pkce_oauth_cred",
        state_token=None,
    )
    requested.raw_auth_credential.oauth2.code_verifier = None
    requested.raw_auth_credential.oauth2.code_challenge_method = None
    state = _empty_state()
    state[_oauth_credential_key("int-1")] = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            code_verifier="cached-pkce-verifier",
            code_challenge_method="S256",
        ),
    )

    response = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(auth_code="code-789"),
    )

    stored = await store_auth_response(
        requested=requested,
        response=response.model_dump(
            mode="json", exclude_none=True, by_alias=True
        ),
        state=state,
        interrupt_id="int-1",
    )

    assert stored is not None
    assert state[_oauth_credential_key("int-1")] is None
    assert len(self.exchanged_credentials) == 1
    exchanged_oauth2 = self.exchanged_credentials[0].oauth2
    assert exchanged_oauth2.code_verifier == "cached-pkce-verifier"
    assert exchanged_oauth2.code_challenge_method == "S256"

  @pytest.mark.asyncio
  async def test_rejects_mismatched_oauth_state_from_session_state(self):
    """Rejects OAuth2 responses whose state does not match session state."""
    requested = _oauth_auth_config(credential_key="oauth_cred")
    state = _empty_state()
    state[_oauth_state_key("int-1")] = "expected-session-state"

    response = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            auth_code="code-123",
            state="wrong-state",
        ),
    )

    stored = await store_auth_response(
        requested=requested,
        response=response.model_dump(
            mode="json", exclude_none=True, by_alias=True
        ),
        state=state,
        interrupt_id="int-1",
    )

    assert stored is None
    assert "temp:oauth_cred" not in state
    assert self.exchanged_schemes == []

  @pytest.mark.asyncio
  async def test_rejects_missing_oauth_state_when_expected(self):
    """Rejects OAuth2 responses missing state when session state expects one."""
    requested = _oauth_auth_config(credential_key="oauth_cred")
    state = _empty_state()
    state[_oauth_state_key("int-1")] = "expected-session-state"

    response = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            auth_code="code-123",
            state=None,
        ),
    )

    stored = await store_auth_response(
        requested=requested,
        response=response.model_dump(
            mode="json", exclude_none=True, by_alias=True
        ),
        state=state,
        interrupt_id="int-1",
    )

    assert stored is None
    assert "temp:oauth_cred" not in state
    assert self.exchanged_schemes == []
