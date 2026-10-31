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

"""Authenticated tools drop a stored credential the server rejected."""

from unittest.mock import AsyncMock
from unittest.mock import Mock

from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlowAuthorizationCode
from fastapi.openapi.models import OAuthFlows
from google.adk.agents.invocation_context import InvocationContext
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.credential_service.in_memory_credential_service import InMemoryCredentialService
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.authenticated_function_tool import AuthenticatedFunctionTool
from google.adk.tools.base_authenticated_tool import BaseAuthenticatedTool
from google.adk.tools.mcp_tool.mcp_session_manager import MCPSessionManager
from google.adk.tools.mcp_tool.mcp_tool import McpTool
from google.adk.tools.tool_context import ToolContext
import httpx
import pytest


def _auth_config() -> AuthConfig:
  return AuthConfig(
      auth_scheme=OAuth2(
          flows=OAuthFlows(
              authorizationCode=OAuthFlowAuthorizationCode(
                  authorizationUrl="https://example.com/authorize",
                  tokenUrl="https://example.com/token",
                  scopes={"read": "read"},
              )
          )
      ),
      raw_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(client_id="client-id", client_secret="secret"),
      ),
      credential_key="reauth-test-key",
  )


def _stored_credential() -> AuthCredential:
  return AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id="client-id",
          client_secret="secret",
          access_token="rejected-token",
      ),
  )


async def _tool_context_with_stored_credential(
    config: AuthConfig,
) -> ToolContext:
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name="test_app", user_id="user", session_id="session"
  )
  credential_service = InMemoryCredentialService()
  invocation_context = InvocationContext(
      session_service=session_service,
      invocation_id="invocation",
      session=session,
      credential_service=credential_service,
  )
  tool_context = ToolContext(invocation_context, function_call_id="call-1")
  saved_config = config.model_copy(deep=True)
  saved_config.exchanged_auth_credential = _stored_credential()
  await credential_service.save_credential(saved_config, tool_context)
  return tool_context


def _status_error(status_code: int) -> httpx.HTTPStatusError:
  request = httpx.Request("GET", "https://example.com/api")
  return httpx.HTTPStatusError(
      f"HTTP {status_code}",
      request=request,
      response=httpx.Response(status_code, request=request),
  )


class _FailingTool(BaseAuthenticatedTool):
  """Raises whatever `error` is set to, or returns `result`."""

  def __init__(self, auth_config: AuthConfig):
    super().__init__(
        name="failing_tool", description="d", auth_config=auth_config
    )
    self.error: BaseException | None = None
    self.result = "ok"
    self.seen_tokens: list[str | None] = []

  async def _run_async_impl(self, *, args, tool_context, credential):
    self.seen_tokens.append(
        credential.oauth2.access_token if credential.oauth2 else None
    )
    if self.error:
      raise self.error
    return self.result


def _stored(tool_context: ToolContext, config: AuthConfig):
  return tool_context._invocation_context.credential_service.load_credential(
      config, tool_context
  )


class TestBaseAuthenticatedToolRejectedCredential:

  @pytest.mark.asyncio
  @pytest.mark.parametrize("status_code", [401, 403])
  async def test_rejected_credential_is_cleared_and_reauth_requested(
      self, status_code
  ):
    config = _auth_config()
    tool = _FailingTool(config)
    tool_context = await _tool_context_with_stored_credential(config)
    tool.error = _status_error(status_code)

    with pytest.raises(httpx.HTTPStatusError):
      await tool.run_async(args={}, tool_context=tool_context)

    assert tool.seen_tokens == ["rejected-token"]
    assert await _stored(tool_context, config) is None

    # The next call asks for a new credential instead of reusing the old one.
    tool.error = None
    result = await tool.run_async(args={}, tool_context=tool_context)

    assert result == "Pending User Authorization."
    assert tool.seen_tokens == ["rejected-token"]
    assert "call-1" in tool_context.actions.requested_auth_configs

  @pytest.mark.asyncio
  async def test_wrapped_status_error_is_recognised(self):
    config = _auth_config()
    tool = _FailingTool(config)
    tool_context = await _tool_context_with_stored_credential(config)
    try:
      try:
        raise _status_error(401)
      except httpx.HTTPStatusError as e:
        raise ConnectionError("Failed to create session") from e
    except ConnectionError as wrapped:
      tool.error = wrapped

    with pytest.raises(ConnectionError):
      await tool.run_async(args={}, tool_context=tool_context)

    assert await _stored(tool_context, config) is None

  @pytest.mark.asyncio
  @pytest.mark.parametrize(
      "error",
      [_status_error(404), _status_error(500), ValueError("bad input")],
  )
  async def test_other_errors_keep_the_stored_credential(self, error):
    config = _auth_config()
    tool = _FailingTool(config)
    tool_context = await _tool_context_with_stored_credential(config)
    tool.error = error

    with pytest.raises(type(error)):
      await tool.run_async(args={}, tool_context=tool_context)

    stored = await _stored(tool_context, config)
    assert stored is not None
    assert stored.oauth2.access_token == "rejected-token"

  @pytest.mark.asyncio
  async def test_successful_call_keeps_the_stored_credential(self):
    config = _auth_config()
    tool = _FailingTool(config)
    tool_context = await _tool_context_with_stored_credential(config)

    assert await tool.run_async(args={}, tool_context=tool_context) == "ok"
    assert await tool.run_async(args={}, tool_context=tool_context) == "ok"

    assert tool.seen_tokens == ["rejected-token"] * 2
    assert await _stored(tool_context, config) is not None

  @pytest.mark.asyncio
  async def test_static_credential_is_not_touched(self):
    config = AuthConfig(
        auth_scheme=OAuth2(flows=OAuthFlows()),
        raw_auth_credential=AuthCredential(
            auth_type=AuthCredentialTypes.API_KEY, api_key="static-key"
        ),
    )
    tool = _FailingTool(config)
    tool_context = await _tool_context_with_stored_credential(config)
    tool.error = _status_error(401)
    manager = tool._credentials_manager
    manager._save_credential = AsyncMock()

    with pytest.raises(httpx.HTTPStatusError):
      await tool.run_async(args={}, tool_context=tool_context)

    manager._save_credential.assert_not_called()


class TestAuthenticatedFunctionToolRejectedCredential:

  @pytest.mark.asyncio
  async def test_rejected_credential_is_cleared_and_reauth_requested(self):
    config = _auth_config()
    tokens = []
    fail = True

    async def call_api(credential: AuthCredential) -> str:
      tokens.append(credential.oauth2.access_token)
      if fail:
        raise _status_error(401)
      return "ok"

    tool = AuthenticatedFunctionTool(func=call_api, auth_config=config)
    tool_context = await _tool_context_with_stored_credential(config)

    with pytest.raises(httpx.HTTPStatusError):
      await tool.run_async(args={}, tool_context=tool_context)
    assert await _stored(tool_context, config) is None

    fail = False
    result = await tool.run_async(args={}, tool_context=tool_context)

    assert result == "Pending User Authorization."
    assert tokens == ["rejected-token"]
    assert "call-1" in tool_context.actions.requested_auth_configs

  @pytest.mark.asyncio
  async def test_non_auth_error_keeps_the_stored_credential(self):
    config = _auth_config()

    async def call_api(credential: AuthCredential) -> str:
      raise _status_error(500)

    tool = AuthenticatedFunctionTool(func=call_api, auth_config=config)
    tool_context = await _tool_context_with_stored_credential(config)

    with pytest.raises(httpx.HTTPStatusError):
      await tool.run_async(args={}, tool_context=tool_context)

    assert await _stored(tool_context, config) is not None


class _MockMcpTool:
  name = "mcp_tool"
  description = "d"
  inputSchema = {"type": "object", "properties": {}}
  outputSchema = None
  meta = None


class TestMcpToolRejectedCredential:

  @pytest.mark.asyncio
  async def test_unauthorized_transport_error_clears_credential(self):
    config = _auth_config()
    session = AsyncMock()
    session.call_tool.side_effect = _status_error(401)
    session_manager = Mock(spec=MCPSessionManager)
    session_manager.create_session = AsyncMock(return_value=session)
    tool = McpTool(
        mcp_tool=_MockMcpTool(),
        mcp_session_manager=session_manager,
        auth_scheme=config.auth_scheme,
        auth_credential=config.raw_auth_credential,
    )
    # McpTool derives its own credential key; build the config it will use.
    manager_config = tool._credentials_manager._auth_config
    tool_context = await _tool_context_with_stored_credential(manager_config)

    first = await tool.run_async(args={}, tool_context=tool_context)

    assert "error" in first
    assert await _stored(tool_context, manager_config) is None
    result = await tool.run_async(args={}, tool_context=tool_context)
    assert result == "Pending User Authorization."
    assert session.call_tool.call_count == 1
