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

"""Tests for MCPToolset authentication functionality."""

import base64
from contextlib import asynccontextmanager
from typing import Any
from typing import AsyncGenerator
from typing import Callable
from unittest.mock import AsyncMock
from unittest.mock import Mock

from fastapi.openapi.models import APIKey as APIKeyScheme
from fastapi.openapi.models import APIKeyIn
from fastapi.openapi.models import HTTPBearer
from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlowAuthorizationCode
from fastapi.openapi.models import OAuthFlows
from google.adk.agents.llm_agent import LlmAgent
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import HttpAuth
from google.adk.auth.auth_credential import HttpCredentials
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.credential_service.in_memory_credential_service import InMemoryCredentialService
from google.adk.dependencies._mcp import FastMCP
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.utils._function_call_names import REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
from google.adk.utils._function_call_names import REQUEST_EUC_FUNCTION_CALL_NAME
from google.genai import types
from mcp import StdioServerParameters
import pytest

from ... import testing_utils
from ._in_memory_session import in_memory_transport


class TestMcpToolsetGetAuthConfig:
  """Tests for McpToolset.get_auth_config method."""

  def test_get_auth_config_returns_none_without_auth_scheme(self):
    """Test that get_auth_config returns None when no auth configured."""
    toolset = McpToolset(
        connection_params=StdioServerParameters(command="echo", args=["test"])
    )

    assert toolset.get_auth_config() is None

  def test_get_auth_config_returns_config_with_auth_scheme(self):
    """Test that get_auth_config returns AuthConfig when auth configured."""
    auth_scheme = OAuth2(
        flows=OAuthFlows(
            authorizationCode=OAuthFlowAuthorizationCode(
                authorizationUrl="https://example.com/auth",
                tokenUrl="https://example.com/token",
                scopes={"read": "Read access"},
            )
        )
    )
    auth_credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id="test_client_id",
            client_secret="test_client_secret",
        ),
    )

    toolset = McpToolset(
        connection_params=StdioServerParameters(command="echo", args=["test"]),
        auth_scheme=auth_scheme,
        auth_credential=auth_credential,
    )

    auth_config = toolset.get_auth_config()
    assert auth_config is not None
    assert auth_config.auth_scheme == auth_scheme
    assert auth_config.raw_auth_credential == auth_credential

  def test_get_auth_config_returns_same_instance(self):
    """Test that get_auth_config returns the same instance each time."""
    auth_scheme = OAuth2(
        flows=OAuthFlows(
            authorizationCode=OAuthFlowAuthorizationCode(
                authorizationUrl="https://example.com/auth",
                tokenUrl="https://example.com/token",
                scopes={},
            )
        )
    )

    toolset = McpToolset(
        connection_params=StdioServerParameters(command="echo", args=["test"]),
        auth_scheme=auth_scheme,
    )

    # Should return the same instance
    config1 = toolset.get_auth_config()
    config2 = toolset.get_auth_config()
    assert config1 is config2


class TestMcpToolsetGetAuthHeaders:
  """Tests for McpToolset._get_auth_headers method."""

  @pytest.fixture
  def toolset_with_oauth2(self):
    """Create a toolset with OAuth2 auth configured."""
    auth_scheme = OAuth2(
        flows=OAuthFlows(
            authorizationCode=OAuthFlowAuthorizationCode(
                authorizationUrl="https://example.com/auth",
                tokenUrl="https://example.com/token",
                scopes={"read": "Read access"},
            )
        )
    )
    auth_credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id="test_client_id",
            client_secret="test_client_secret",
        ),
    )

    return McpToolset(
        connection_params=StdioServerParameters(command="echo", args=["test"]),
        auth_scheme=auth_scheme,
        auth_credential=auth_credential,
    )

  def test_get_auth_headers_returns_none_without_auth_config(self):
    """Test that _get_auth_headers returns None without auth config."""
    toolset = McpToolset(
        connection_params=StdioServerParameters(command="echo", args=["test"])
    )

    assert toolset._get_auth_headers() is None

  def test_get_auth_headers_returns_none_without_exchanged_credential(
      self, toolset_with_oauth2
  ):
    """Test that _get_auth_headers returns None without exchanged credential."""
    # No exchanged credential set yet
    assert toolset_with_oauth2._get_auth_headers() is None

  def test_get_auth_headers_oauth2_bearer_token(self, toolset_with_oauth2):
    """Test that _get_auth_headers returns Bearer token for OAuth2."""
    # Set exchanged credential with access token
    toolset_with_oauth2._auth_config.exchanged_auth_credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(access_token="test-access-token"),
    )

    headers = toolset_with_oauth2._get_auth_headers()

    assert headers is not None
    assert headers["Authorization"] == "Bearer test-access-token"

  def test_get_auth_headers_http_bearer_token(self):
    """Test that _get_auth_headers returns Bearer token for HTTP bearer."""
    auth_scheme = OAuth2(
        flows=OAuthFlows(
            authorizationCode=OAuthFlowAuthorizationCode(
                authorizationUrl="https://example.com/auth",
                tokenUrl="https://example.com/token",
                scopes={},
            )
        )
    )

    toolset = McpToolset(
        connection_params=StdioServerParameters(command="echo", args=["test"]),
        auth_scheme=auth_scheme,
    )

    # Set exchanged credential with HTTP bearer token
    toolset._auth_config.exchanged_auth_credential = AuthCredential(
        auth_type=AuthCredentialTypes.HTTP,
        http=HttpAuth(
            scheme="bearer",
            credentials=HttpCredentials(token="test-bearer-token"),
        ),
    )

    headers = toolset._get_auth_headers()

    assert headers is not None
    assert headers["Authorization"] == "Bearer test-bearer-token"

  def test_get_auth_headers_http_basic_auth(self):
    """Test that _get_auth_headers returns Basic auth for HTTP basic."""
    auth_scheme = OAuth2(
        flows=OAuthFlows(
            authorizationCode=OAuthFlowAuthorizationCode(
                authorizationUrl="https://example.com/auth",
                tokenUrl="https://example.com/token",
                scopes={},
            )
        )
    )

    toolset = McpToolset(
        connection_params=StdioServerParameters(command="echo", args=["test"]),
        auth_scheme=auth_scheme,
    )

    # Set exchanged credential with HTTP basic auth
    toolset._auth_config.exchanged_auth_credential = AuthCredential(
        auth_type=AuthCredentialTypes.HTTP,
        http=HttpAuth(
            scheme="basic",
            credentials=HttpCredentials(
                username="testuser",
                password="testpass",
            ),
        ),
    )

    headers = toolset._get_auth_headers()

    assert headers is not None
    expected_credentials = base64.b64encode(b"testuser:testpass").decode()
    assert headers["Authorization"] == f"Basic {expected_credentials}"

  def test_get_auth_headers_api_key_header(self):
    """Test that _get_auth_headers returns API key in header."""
    # Note: fastapi's APIKey model uses 'in' not 'in_', but accepts both
    auth_scheme = APIKeyScheme(**{
        "in": APIKeyIn.header,
        "name": "X-API-Key",
    })

    toolset = McpToolset(
        connection_params=StdioServerParameters(command="echo", args=["test"]),
        auth_scheme=auth_scheme,
    )

    # Set exchanged credential with API key
    toolset._auth_config.exchanged_auth_credential = AuthCredential(
        auth_type=AuthCredentialTypes.API_KEY,
        api_key="test-api-key-12345",
    )

    headers = toolset._get_auth_headers()

    assert headers is not None
    assert headers["X-API-Key"] == "test-api-key-12345"

  def test_get_auth_headers_api_key_non_header_logs_warning(self, caplog):
    """Test that non-header API key logs a warning."""
    # Note: fastapi's APIKey model uses 'in' not 'in_'
    auth_scheme = APIKeyScheme(**{
        "in": APIKeyIn.query,  # Query param, not header
        "name": "api_key",
    })

    toolset = McpToolset(
        connection_params=StdioServerParameters(command="echo", args=["test"]),
        auth_scheme=auth_scheme,
    )

    # Set exchanged credential with API key
    toolset._auth_config.exchanged_auth_credential = AuthCredential(
        auth_type=AuthCredentialTypes.API_KEY,
        api_key="test-api-key",
    )

    headers = toolset._get_auth_headers()

    # Should return None for non-header API key
    assert headers is None

  def test_get_auth_headers_reads_from_readonly_context(
      self, toolset_with_oauth2
  ):
    """Test that _get_auth_headers reads from ReadonlyContext first."""
    from google.adk.agents.readonly_context import ReadonlyContext

    mock_readonly_context = Mock(spec=ReadonlyContext)
    mock_credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(access_token="token-from-context"),
    )
    mock_readonly_context.get_credential.return_value = mock_credential

    # Even if exchanged_auth_credential has a different value
    toolset_with_oauth2._auth_config.exchanged_auth_credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(access_token="token-from-config"),
    )

    headers = toolset_with_oauth2._get_auth_headers(
        readonly_context=mock_readonly_context
    )

    assert headers is not None
    assert headers["Authorization"] == "Bearer token-from-context"
    mock_readonly_context.get_credential.assert_called_once_with(
        toolset_with_oauth2._auth_config.credential_key
    )


_APP_NAME = "mail_app"
_USER_ID = "user"
_TOKEN = "user-token"
_SEND_EMAIL_TO_BOB = types.Part(
    function_call=types.FunctionCall(name="send_email", args={"to": "bob"})
)


@asynccontextmanager
async def _refused_connection() -> AsyncGenerator[Any, None]:
  raise PermissionError("HTTP 401: this MCP server requires a bearer token")
  yield  # pylint: disable=unreachable


def _function_call(events: list[Event], name: str) -> types.FunctionCall:
  return next(
      function_call
      for event in events
      for function_call in event.get_function_calls()
      if function_call.name == name
  )


async def _run(
    runner: Runner, session: Session, part: types.Part
) -> list[Event]:
  return [
      event
      async for event in runner.run_async(
          user_id=session.user_id,
          session_id=session.id,
          new_message=types.Content(role="user", parts=[part]),
      )
  ]


class _BearerTokenMcpServer:
  """An in-process MCP server that, like one behind auth, refuses anonymity.

  `connect` stands in for `MCPSessionManager._create_client`, which receives
  the headers every request of the new session will carry, so refusing a
  connection there is what a server answering HTTP 401 to an unauthenticated
  `initialize` amounts to.
  """

  def __init__(self) -> None:
    self.emails_sent: list[str] = []
    self.refused_connections = 0
    self._server = FastMCP("mail")

    @self._server.tool()
    def send_email(to: str) -> str:
      """Sends an email."""
      self.emails_sent.append(to)
      return f"sent to {to}"

  def connect(
      self,
      merged_headers: dict[str, str] | None = None,
      mtls_transport: Any = None,
      *,
      session_key: str | None = None,
  ) -> Any:
    del mtls_transport, session_key
    if (merged_headers or {}).get("Authorization") != f"Bearer {_TOKEN}":
      self.refused_connections += 1
      return _refused_connection()
    return in_memory_transport(self._server)


class TestMcpToolsetCredentialOnResume:
  """An approved call resumes with its toolset's credential.

  `toolset_auth` resolves an authenticated toolset's credential once per
  invocation, right before the tools are listed for the model. Resuming an
  approved call lists the agent's tools earlier, from `request_confirmation`,
  so the credential is there only if the toolset still caches the paused
  invocation's listing.
  """

  @pytest.fixture
  def server(self) -> _BearerTokenMcpServer:
    return _BearerTokenMcpServer()

  @pytest.fixture
  def session_service(self) -> InMemorySessionService:
    return InMemorySessionService()

  @pytest.fixture
  def make_runner(
      self,
      monkeypatch: pytest.MonkeyPatch,
      server: _BearerTokenMcpServer,
      session_service: InMemorySessionService,
  ) -> Callable[..., Runner]:
    """Builds runners that send email through the authenticated server.

    Each one gets its own agent and toolset, the way a new process serving the
    same sessions and credentials would build them.
    """
    credential_service = InMemoryCredentialService()

    def make_runner(*responses: Any) -> Runner:
      toolset = McpToolset(
          connection_params=StreamableHTTPConnectionParams(
              url="https://mail.example.com/mcp"
          ),
          auth_scheme=HTTPBearer(),
          require_confirmation=True,
      )
      manager = toolset._mcp_session_manager
      monkeypatch.setattr(manager, "_create_client", server.connect)
      monkeypatch.setattr(
          manager, "_get_mtls_transport", AsyncMock(return_value=None)
      )
      agent = LlmAgent(
          name="mail_agent",
          model=testing_utils.MockModel.create(responses=list(responses)),
          tools=[toolset],
      )
      return Runner(
          app_name=_APP_NAME,
          agent=agent,
          session_service=session_service,
          credential_service=credential_service,
      )

    return make_runner

  async def _pause_for_approval(
      self, runner: Runner, session: Session
  ) -> types.FunctionCall:
    """Signs the user in, then lets the model's email wait on their approval."""
    events = await _run(runner, session, types.Part(text="email bob"))
    auth_call = _function_call(events, REQUEST_EUC_FUNCTION_CALL_NAME)
    auth_response = AuthConfig.model_validate(auth_call.args["authConfig"])
    auth_response.exchanged_auth_credential = AuthCredential(
        auth_type=AuthCredentialTypes.HTTP,
        http=HttpAuth(
            scheme="bearer", credentials=HttpCredentials(token=_TOKEN)
        ),
    )
    events = await _run(
        runner,
        session,
        types.Part(
            function_response=types.FunctionResponse(
                id=auth_call.id,
                name=REQUEST_EUC_FUNCTION_CALL_NAME,
                response=auth_response.model_dump(
                    exclude_none=True, by_alias=True
                ),
            )
        ),
    )
    return _function_call(events, REQUEST_CONFIRMATION_FUNCTION_CALL_NAME)

  def _approval(self, confirmation_call: types.FunctionCall) -> types.Part:
    return types.Part(
        function_response=types.FunctionResponse(
            id=confirmation_call.id,
            name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
            response={"confirmed": True},
        )
    )

  @pytest.mark.asyncio
  async def test_approval_served_by_another_process(
      self, server, session_service, make_runner
  ):
    """An approval reaching a restarted or another instance runs the call."""
    session = await session_service.create_session(
        app_name=_APP_NAME, user_id=_USER_ID
    )
    paused_runner = make_runner(_SEND_EMAIL_TO_BOB)
    try:
      confirmation_call = await self._pause_for_approval(paused_runner, session)
    finally:
      await paused_runner.close()

    resuming_runner = make_runner("email sent")
    try:
      events = await _run(
          resuming_runner, session, self._approval(confirmation_call)
      )
    finally:
      await resuming_runner.close()

    assert server.emails_sent == ["bob"]
    assert server.refused_connections == 0
    assert testing_utils.simplify_events(events)[-1] == (
        "mail_agent",
        "email sent",
    )

  @pytest.mark.asyncio
  async def test_approval_after_another_session_listed_the_tools(
      self, server, session_service, make_runner
  ):
    """Another session's turn in between does not break a pending approval."""
    session = await session_service.create_session(
        app_name=_APP_NAME, user_id=_USER_ID
    )
    other_session = await session_service.create_session(
        app_name=_APP_NAME, user_id=_USER_ID
    )
    runner = make_runner(
        _SEND_EMAIL_TO_BOB, "hello from the other session", "email sent"
    )
    try:
      confirmation_call = await self._pause_for_approval(runner, session)
      await _run(runner, other_session, types.Part(text="hi"))

      events = await _run(runner, session, self._approval(confirmation_call))
    finally:
      await runner.close()

    assert server.emails_sent == ["bob"]
    assert server.refused_connections == 0
    assert testing_utils.simplify_events(events)[-1] == (
        "mail_agent",
        "email sent",
    )
