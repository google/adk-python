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

"""E2E Integration Test for 3LO flow using GCP Agent Identity service."""

import dataclasses
from typing import Any
from unittest import mock

import pytest

pytest.importorskip(
    "google.cloud.iamconnectorcredentials_v1alpha",
    reason="Requires google-cloud-iamconnectorcredentials",
)

from google.adk import Agent
from google.adk import Runner
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.credential_manager import CredentialManager
from google.adk.integrations.agent_identity import gcp_auth_provider
from google.adk.integrations.agent_identity import GcpAuthProvider
from google.adk.integrations.agent_identity import GcpAuthProviderScheme
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.base_authenticated_tool import BaseAuthenticatedTool
from google.cloud.iamconnectorcredentials_v1alpha import RetrieveCredentialsMetadata
from google.cloud.iamconnectorcredentials_v1alpha import RetrieveCredentialsRequest
from google.cloud.iamconnectorcredentials_v1alpha import RetrieveCredentialsResponse
from google.genai import types

from tests.unittests import testing_utils

DUMMY_TOKEN = "mock-token-3legged"
TEST_CONNECTOR_3LO = (
    "projects/my-project/locations/some-location/connectors/test-connector-3lo"
)


class DummyTool(BaseAuthenticatedTool):

  def __init__(self, auth_config: AuthConfig) -> None:
    super().__init__(
        name="dummy_tool",
        description="Dummy tool for testing 3LO.",
        auth_config=auth_config,
    )

  def _get_declaration(self) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters=types.Schema(
            type="OBJECT",
            properties={},
        ),
    )

  async def _run_async_impl(
      self, *, args: dict[str, Any] | None, tool_context: Any, credential: Any
  ) -> Any:
    # Extract and return the token to prove the provider gave us the expected credential
    if credential.http and credential.http.credentials:
      return credential.http.credentials.token
    if credential.oauth2 and credential.oauth2.access_token:
      return credential.oauth2.access_token

    return None


@dataclasses.dataclass
class _MockOperation:
  done: bool
  response_obj: Any = None
  metadata_obj: Any = None
  error: Any = None
  metadata: Any = dataclasses.field(init=False, default=None)
  response: Any = dataclasses.field(init=False, default=None)
  operation: Any = dataclasses.field(init=False)

  def __post_init__(self) -> None:
    if self.metadata_obj:
      self.metadata = mock.Mock()
      self.metadata.value = RetrieveCredentialsMetadata.serialize(
          self.metadata_obj
      )
    if self.response_obj:
      self.response = mock.Mock()
      self.response.value = RetrieveCredentialsResponse.serialize(
          self.response_obj
      )
    self.operation = self

  def HasField(self, field_name: str) -> bool:
    return getattr(self, field_name, None) is not None


class MockGcpClient:
  """Lightweight in-memory mock for Agent Identity Credentials service 3LO Consent Flow."""

  def __init__(self) -> None:
    self.finalized_connectors = set()

  def retrieve_credentials(
      self,
      request: RetrieveCredentialsRequest | dict[str, Any] | None = None,
      **kwargs: Any,
  ) -> _MockOperation:
    connector = (
        request.get("connector")
        if isinstance(request, dict)
        else getattr(request, "connector", None)
    )

    if connector in self.finalized_connectors:
      mock_credential = RetrieveCredentialsResponse(
          token=DUMMY_TOKEN, header="Authorization: Bearer"
      )
      return _MockOperation(done=True, response_obj=mock_credential)

    # Otherwise, return Consent Required
    # Auto-finalize for the next call to simulate user approval flow
    self.finalized_connectors.add(connector)

    mock_metadata = RetrieveCredentialsMetadata(
        uri_consent_required=RetrieveCredentialsMetadata.UriConsentRequired(
            authorization_uri="http://mock-auth-uri",
            consent_nonce="mock-consent-nonce",
        )
    )
    return _MockOperation(done=False, metadata_obj=mock_metadata)


# Mocked execution; pin to a single LLM backend to avoid duplicate runs.
@pytest.mark.parametrize("llm_backend", ["GOOGLE_AI"], indirect=True)
@pytest.mark.asyncio
async def test_gcp_agent_identity_3lo_user_consent_flow() -> None:
  # Clear registry to isolate tests
  CredentialManager._auth_provider_registry._providers.clear()

  # 1. Setup mocked GCP Client to simulate stateful 3LO process
  mock_gcp_client = MockGcpClient()

  with mock.patch.object(
      gcp_auth_provider,
      "Client",
      autospec=True,
  ) as mock_client_cls:
    mock_client_cls.return_value.retrieve_credentials.side_effect = (
        mock_gcp_client.retrieve_credentials
    )

    # 2. Configure Auth and DummyTool
    auth_scheme = GcpAuthProviderScheme(
        name=TEST_CONNECTOR_3LO,
        scopes=["test-scope"],
        continue_uri="https://example.com/continue",
    )
    auth_config = AuthConfig(auth_scheme=auth_scheme)
    dummy_tool = DummyTool(auth_config=auth_config)

    # 3. Setup LLM, Agent, and Runner
    # We mock the LLM to just issue the tool call to 'dummy_tool'
    mock_model = testing_utils.MockModel.create(
        responses=[
            types.Part.from_function_call(name="dummy_tool", args={}),
            "I am waiting for your authorization.",
            "Tool executed successfully.",
        ]
    )

    agent = Agent(
        name="test_agent",
        model=mock_model,
        instruction="You are an agent. Use the dummy_tool when needed.",
        tools=[dummy_tool],
    )

    runner = Runner(
        app_name="test_mcp_3lo_app",
        agent=agent,
        session_service=InMemorySessionService(),
        auto_create_session=True,
    )

    # 4. Register Auth Provider
    CredentialManager.register_auth_provider(GcpAuthProvider())

    # 5. Execute Flow
    session = await runner.session_service.create_session(
        app_name="test_mcp_3lo_app", user_id="test_user"
    )

    event_list = []

    # Step 5a: User sends message, Agent requests credential
    async for event in runner.run_async(
        user_id="test_user",
        session_id=session.id,
        new_message=types.UserContent(
            parts=[types.Part(text="Get me the token.")]
        ),
    ):
      event_list.append(event)

    def _find_auth_request_event(events):
      for event in events:
        for part in event.content.parts:
          if (
              part.function_call
              and part.function_call.name == "adk_request_credential"
          ):
            return event
      return None

    auth_request_event = _find_auth_request_event(event_list)

    assert (
        auth_request_event
    ), "Expected adk_request_credential tool call not found."

    # Step 5b: Simulate User Consent
    call_part = next(
        p for p in auth_request_event.content.parts if p.function_call
    )
    request_auth_config = call_part.function_call.args.get("authConfig", {})

    assert (
        request_auth_config.get("exchangedAuthCredential", {})
        .get("oauth2", {})
        .get("nonce")
        == "mock-consent-nonce"
    )

    # Step 5c: User acknowledges credential request
    response_part = types.Part.from_function_response(
        name="adk_request_credential", response=request_auth_config
    )
    response_part.function_response.id = call_part.function_call.id

    final_response_parts = []
    async for event in runner.run_async(
        user_id="test_user",
        session_id=session.id,
        new_message=types.UserContent(parts=[response_part]),
    ):
      event_list.append(event)
      if event.content:
        for part in event.content.parts:
          if part.text:
            final_response_parts.append(part.text)

    final_response_text = "".join(final_response_parts)

    # 6. Assertions

    # Assert GCP Agent Identity client was invoked for credentials twice
    # (Initial Request + Post-Consent call)
    assert mock_client_cls.return_value.retrieve_credentials.call_count == 2
    expected_request = RetrieveCredentialsRequest(
        connector=TEST_CONNECTOR_3LO,
        user_id="test_user",
        scopes=["test-scope"],
        continue_uri="https://example.com/continue",
        force_refresh=False,
    )
    mock_client_cls.return_value.retrieve_credentials.assert_called_with(
        expected_request
    )

    assert "Tool executed successfully." in final_response_text

    # Validate requests received by the mock model
    requests = mock_model.requests
    # Events:
    # 1. User Input (Get me the token.)
    # 2. LLM (I am waiting for your authorization.)
    # 3. LLM (Tool executed successfully.)
    assert len(requests) == 3

    # Extract the function response from the prompt payload sent to the LLM
    last_request = requests[-1]
    function_response = next(
        (
            p.function_response
            for p in last_request.contents[-1].parts
            if p.function_response
        ),
        None,
    )

    assert function_response is not None
    assert function_response.name == "dummy_tool"
    assert DUMMY_TOKEN in str(function_response.response)
