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

"""E2E Integration Test for GCP Agent Identity Auth Provider two-legged OAuth Flow."""

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
from google.cloud.iamconnectorcredentials_v1alpha import RetrieveCredentialsRequest
from google.cloud.iamconnectorcredentials_v1alpha import RetrieveCredentialsResponse
from google.genai import types

from tests.unittests import testing_utils

DUMMY_TOKEN = "fake-gcp-2lo-token-123"
TEST_CONNECTOR_2LO = (
    "projects/test-project/locations/global/connectors/test-connector"
)


class DummyTool(BaseAuthenticatedTool):

  def __init__(self, auth_config: AuthConfig) -> None:
    super().__init__(
        name="dummy_tool",
        description="Dummy tool for testing 2LO.",
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
    # Return the token to prove the provider gave the expected credential
    if credential.http and credential.http.credentials:
      return credential.http.credentials.token
    if credential.oauth2 and credential.oauth2.access_token:
      return credential.oauth2.access_token
    return None


# Mocked execution; pin to a single LLM backend to avoid duplicate runs.
@pytest.mark.parametrize("llm_backend", ["GOOGLE_AI"], indirect=True)
@dataclasses.dataclass
class _DummyOperation:
  done: bool = True
  error: Any = None
  metadata: Any = None
  response: Any = dataclasses.field(init=False)
  operation: Any = dataclasses.field(init=False)

  def __post_init__(self) -> None:
    self.response = mock.Mock()
    mock_credential = RetrieveCredentialsResponse(
        header="Authorization: Bearer", token=DUMMY_TOKEN
    )
    self.response.value = RetrieveCredentialsResponse.serialize(mock_credential)
    self.operation = self

  def HasField(self, field_name: str) -> bool:
    return getattr(self, field_name, None) is not None


@pytest.mark.asyncio
async def test_gcp_agent_identity_2lo_gets_token() -> None:
  """Test the end-to-end flow fetching 2LO OAuth token from GCP Agent Identity credentials service."""

  # Clear registry to isolate tests
  CredentialManager._auth_provider_registry._providers.clear()

  # 1. Setup mocked GCP Client to return the fake Bearer token
  with mock.patch.object(
      gcp_auth_provider,
      "Client",
      autospec=True,
  ) as mock_client_cls:

    mock_operation = _DummyOperation()

    mock_client_cls.return_value.retrieve_credentials.return_value = (
        mock_operation
    )

    # 2. Configure Auth and DummyTool
    auth_scheme = GcpAuthProviderScheme(
        name=TEST_CONNECTOR_2LO,
        scopes=["test-scope"],
    )
    auth_config = AuthConfig(auth_scheme=auth_scheme)
    dummy_tool = DummyTool(auth_config=auth_config)

    # 3. Setup LLM, Agent, and Runner
    # We mock the LLM to just issue the tool call to 'dummy_tool'
    mock_model = testing_utils.MockModel.create(
        responses=[
            types.Part.from_function_call(name="dummy_tool", args={}),
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
        app_name="test_mcp_2lo_app",
        agent=agent,
        session_service=InMemorySessionService(),
        auto_create_session=True,
    )

    # 4. Register Auth Provider
    CredentialManager.register_auth_provider(GcpAuthProvider())

    # 5. Execute Flow
    event_list = []
    async for event in runner.run_async(
        user_id="test_user",
        session_id="test_session1",
        new_message=types.UserContent(
            parts=[types.Part(text="Get me the token.")]
        ),
    ):
      event_list.append(event)

    # 6. Assertions

    # Assert GCP Agent Identity client was invoked for credentials
    expected_request = RetrieveCredentialsRequest(
        connector=TEST_CONNECTOR_2LO,
        user_id="test_user",
        scopes=["test-scope"],
        continue_uri="",
        force_refresh=False,
    )
    mock_client_cls.return_value.retrieve_credentials.assert_called_once_with(
        expected_request
    )

    # 3 Events: Model FunctionCall -> Tool FunctionResponse -> Final LLM Text
    assert len(event_list) == 3
    last_event = event_list[-1]
    assert last_event.content.parts[0].text == "Tool executed successfully."

    # Validate that the mock model received the query and the tool callback
    requests = mock_model.requests
    # 2 Events: User Input -> Tool FunctionResponse
    assert len(requests) == 2

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

    assert function_response.name == "dummy_tool"
    assert DUMMY_TOKEN in str(function_response.response)
