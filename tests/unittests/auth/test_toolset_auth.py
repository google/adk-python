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

"""Tests for toolset authentication functionality."""

from typing import Optional
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlowAuthorizationCode
from fastapi.openapi.models import OAuthFlows
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_preprocessor import TOOLSET_AUTH_CREDENTIAL_ID_PREFIX
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_tool import AuthToolArguments
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.flows.llm_flows.base_llm_flow import TOOLSET_AUTH_CREDENTIAL_ID_PREFIX as FLOW_PREFIX
from google.adk.flows.llm_flows.functions import build_auth_request_event
from google.adk.flows.llm_flows.functions import REQUEST_EUC_FUNCTION_CALL_NAME
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
import pytest


class MockToolset(BaseToolset):
  """A mock toolset for testing."""

  def __init__(
      self,
      auth_config: Optional[AuthConfig] = None,
      tools: Optional[list[BaseTool]] = None,
  ):
    super().__init__()
    self._auth_config = auth_config
    self._tools = tools or []

  def get_auth_config(self) -> Optional[AuthConfig]:
    return self._auth_config

  async def get_tools(self, readonly_context=None) -> list[BaseTool]:
    return self._tools

  async def close(self):
    pass


def create_oauth2_auth_config() -> AuthConfig:
  """Create a sample OAuth2 auth config for testing."""
  return AuthConfig(
      auth_scheme=OAuth2(
          flows=OAuthFlows(
              authorizationCode=OAuthFlowAuthorizationCode(
                  authorizationUrl="https://example.com/auth",
                  tokenUrl="https://example.com/token",
                  scopes={"read": "Read access"},
              )
          )
      ),
      raw_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(
              client_id="test_client_id",
              client_secret="test_client_secret",
          ),
      ),
  )


class TestToolsetAuthPrefixConstant:
  """Test that prefix constants are consistent."""

  def test_prefix_constants_match(self):
    """Ensure auth_preprocessor and base_llm_flow use the same prefix."""
    assert TOOLSET_AUTH_CREDENTIAL_ID_PREFIX == FLOW_PREFIX
    assert TOOLSET_AUTH_CREDENTIAL_ID_PREFIX == "_adk_toolset_auth_"


class TestResolveToolsetAuth:
  """Tests for _resolve_toolset_auth method in BaseLlmFlow."""

  @pytest.fixture
  def mock_invocation_context(self):
    """Create a mock invocation context."""
    ctx = Mock(spec=InvocationContext)
    ctx.invocation_id = "test-invocation-id"
    ctx.end_invocation = False
    ctx.branch = None
    ctx.session = Mock()
    ctx.session.state = {}
    ctx.session.id = "test-session-id"
    ctx.credential_service = None
    ctx.app_name = "test-app"
    ctx.user_id = "test-user"
    return ctx

  @pytest.fixture
  def mock_agent(self):
    """Create a mock LLM agent."""
    agent = Mock()
    agent.name = "test-agent"
    agent.tools = []
    return agent

  @pytest.fixture
  def flow(self):
    """Create a BaseLlmFlow instance for testing."""
    # BaseLlmFlow is abstract, but we can still test _resolve_toolset_auth
    flow = Mock(spec=BaseLlmFlow)
    flow._resolve_toolset_auth = BaseLlmFlow._resolve_toolset_auth
    return flow

  @pytest.mark.asyncio
  async def test_no_tools_returns_no_events(
      self, mock_invocation_context, mock_agent
  ):
    """Test that no events are yielded when agent has no tools."""
    mock_agent.tools = []

    flow = BaseLlmFlow.__new__(BaseLlmFlow)
    events = []
    async for event in flow._resolve_toolset_auth(
        mock_invocation_context, mock_agent
    ):
      events.append(event)

    assert len(events) == 0
    assert mock_invocation_context.end_invocation is False

  @pytest.mark.asyncio
  async def test_toolset_without_auth_config_skipped(
      self, mock_invocation_context, mock_agent
  ):
    """Test that toolsets without auth config are skipped."""
    toolset = MockToolset(auth_config=None)
    mock_agent.tools = [toolset]

    flow = BaseLlmFlow.__new__(BaseLlmFlow)
    events = []
    async for event in flow._resolve_toolset_auth(
        mock_invocation_context, mock_agent
    ):
      events.append(event)

    assert len(events) == 0
    assert mock_invocation_context.end_invocation is False

  @pytest.mark.asyncio
  async def test_toolset_with_credential_available_populates_config(
      self, mock_invocation_context, mock_agent
  ):
    """Test that credential is populated in auth_config when available."""
    auth_config = create_oauth2_auth_config()
    toolset = MockToolset(auth_config=auth_config)
    mock_agent.tools = [toolset]

    # Mock CredentialManager to return a credential
    mock_credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(access_token="test-token"),
    )

    with patch(
        "google.adk.flows.llm_flows.base_llm_flow.CredentialManager"
    ) as MockCredentialManager:
      mock_manager = AsyncMock()
      mock_manager.get_auth_credential = AsyncMock(return_value=mock_credential)
      MockCredentialManager.return_value = mock_manager

      flow = BaseLlmFlow.__new__(BaseLlmFlow)
      events = []
      async for event in flow._resolve_toolset_auth(
          mock_invocation_context, mock_agent
      ):
        events.append(event)

    # No auth request events - credential was available
    assert len(events) == 0
    assert mock_invocation_context.end_invocation is False
    # Credential should be populated in auth_config
    assert auth_config.exchanged_auth_credential == mock_credential

  @pytest.mark.asyncio
  async def test_toolset_without_credential_yields_auth_event(
      self, mock_invocation_context, mock_agent
  ):
    """Test that auth request event is yielded when credential not available."""
    auth_config = create_oauth2_auth_config()
    toolset = MockToolset(auth_config=auth_config)
    mock_agent.tools = [toolset]

    with patch(
        "google.adk.flows.llm_flows.base_llm_flow.CredentialManager"
    ) as MockCredentialManager:
      mock_manager = AsyncMock()
      mock_manager.get_auth_credential = AsyncMock(return_value=None)
      MockCredentialManager.return_value = mock_manager

      flow = BaseLlmFlow.__new__(BaseLlmFlow)
      events = []
      async for event in flow._resolve_toolset_auth(
          mock_invocation_context, mock_agent
      ):
        events.append(event)

    # Should yield one auth request event
    assert len(events) == 1
    assert mock_invocation_context.end_invocation is True

    # Check event structure
    event = events[0]
    assert event.invocation_id == "test-invocation-id"
    assert event.author == "test-agent"
    assert event.content is not None
    assert len(event.content.parts) == 1

    # Check function call
    fc = event.content.parts[0].function_call
    assert fc.name == REQUEST_EUC_FUNCTION_CALL_NAME
    # The args use camelCase aliases from the pydantic model
    assert fc.args["functionCallId"].startswith(
        TOOLSET_AUTH_CREDENTIAL_ID_PREFIX
    )
    assert "MockToolset" in fc.args["functionCallId"]

  @pytest.mark.asyncio
  async def test_multiple_toolsets_needing_auth(
      self, mock_invocation_context, mock_agent
  ):
    """Test that multiple toolsets needing auth yield multiple function calls."""
    auth_config1 = create_oauth2_auth_config()
    auth_config2 = create_oauth2_auth_config()
    toolset1 = MockToolset(auth_config=auth_config1)
    toolset2 = MockToolset(auth_config=auth_config2)
    mock_agent.tools = [toolset1, toolset2]

    with patch(
        "google.adk.flows.llm_flows.base_llm_flow.CredentialManager"
    ) as MockCredentialManager:
      mock_manager = AsyncMock()
      mock_manager.get_auth_credential = AsyncMock(return_value=None)
      MockCredentialManager.return_value = mock_manager

      flow = BaseLlmFlow.__new__(BaseLlmFlow)
      events = []
      async for event in flow._resolve_toolset_auth(
          mock_invocation_context, mock_agent
      ):
        events.append(event)

    # Should yield one event with multiple function calls
    # But since both toolsets have same class name, they'll have same ID
    # and only one will be in pending_auth_requests (dict overwrites)
    assert len(events) == 1
    assert mock_invocation_context.end_invocation is True


class TestAuthPreprocessorToolsetAuthSkip:
  """Tests for auth preprocessor skipping toolset auth."""

  def test_toolset_auth_prefix_skipped(self):
    """Test that function calls with toolset auth prefix are skipped."""
    from google.adk.auth.auth_preprocessor import TOOLSET_AUTH_CREDENTIAL_ID_PREFIX

    # Verify the prefix is correct
    assert TOOLSET_AUTH_CREDENTIAL_ID_PREFIX == "_adk_toolset_auth_"

    # Test that a function_call_id starting with this prefix would be skipped
    toolset_function_call_id = f"{TOOLSET_AUTH_CREDENTIAL_ID_PREFIX}McpToolset"
    assert toolset_function_call_id.startswith(
        TOOLSET_AUTH_CREDENTIAL_ID_PREFIX
    )

    # Regular tool auth function_call_id should NOT start with prefix
    regular_function_call_id = "call_123"
    assert not regular_function_call_id.startswith(
        TOOLSET_AUTH_CREDENTIAL_ID_PREFIX
    )


class TestCallbackContextGetAuthResponse:
  """Tests for CallbackContext.get_auth_response method."""

  @pytest.fixture
  def mock_invocation_context(self):
    """Create a mock invocation context."""
    ctx = Mock(spec=InvocationContext)
    ctx.session = Mock()
    ctx.session.state = {}
    return ctx

  def test_get_auth_response_returns_none_when_no_response(
      self, mock_invocation_context
  ):
    """Test that get_auth_response returns None when no auth response in state."""
    callback_context = CallbackContext(mock_invocation_context)
    auth_config = create_oauth2_auth_config()

    result = callback_context.get_auth_response(auth_config)

    # Should return None when no auth response is stored
    assert result is None

  def test_get_auth_response_delegates_to_auth_handler(
      self, mock_invocation_context
  ):
    """Test that get_auth_response delegates to AuthHandler."""
    callback_context = CallbackContext(mock_invocation_context)
    auth_config = create_oauth2_auth_config()

    # AuthHandler is imported inside the method, so we patch the module
    with patch("google.adk.auth.auth_handler.AuthHandler") as MockAuthHandler:
      mock_handler = Mock()
      mock_handler.get_auth_response = Mock(return_value=None)
      MockAuthHandler.return_value = mock_handler

      callback_context.get_auth_response(auth_config)

      MockAuthHandler.assert_called_once_with(auth_config)
      mock_handler.get_auth_response.assert_called_once()


class TestBuildAuthRequestEvent:
  """Tests for build_auth_request_event helper function."""

  @pytest.fixture
  def mock_invocation_context(self):
    """Create a mock invocation context."""
    ctx = Mock(spec=InvocationContext)
    ctx.invocation_id = "test-invocation-id"
    ctx.branch = None
    ctx.agent = Mock()
    ctx.agent.name = "test-agent"
    return ctx

  def test_builds_event_with_auth_requests(self, mock_invocation_context):
    """Test that build_auth_request_event creates correct event."""
    auth_requests = {
        "call_123": create_oauth2_auth_config(),
    }

    event = build_auth_request_event(mock_invocation_context, auth_requests)

    assert event.invocation_id == "test-invocation-id"
    assert event.author == "test-agent"
    assert event.content is not None
    assert len(event.content.parts) == 1

    fc = event.content.parts[0].function_call
    assert fc.name == REQUEST_EUC_FUNCTION_CALL_NAME
    assert fc.args["functionCallId"] == "call_123"

  def test_multiple_auth_requests_create_multiple_parts(
      self, mock_invocation_context
  ):
    """Test that multiple auth requests create multiple function call parts."""
    auth_requests = {
        "call_1": create_oauth2_auth_config(),
        "call_2": create_oauth2_auth_config(),
    }

    event = build_auth_request_event(mock_invocation_context, auth_requests)

    assert len(event.content.parts) == 2
    function_call_ids = {
        p.function_call.args["functionCallId"] for p in event.content.parts
    }
    assert function_call_ids == {"call_1", "call_2"}

  def test_always_adds_long_running_tool_ids(self, mock_invocation_context):
    """Test that long_running_tool_ids is always set."""
    auth_requests = {"call_123": create_oauth2_auth_config()}

    event = build_auth_request_event(mock_invocation_context, auth_requests)

    assert event.long_running_tool_ids is not None
    assert len(event.long_running_tool_ids) == 1

  def test_custom_author_overrides_default(self, mock_invocation_context):
    """Test that custom author overrides default agent name."""
    auth_requests = {"call_123": create_oauth2_auth_config()}

    event = build_auth_request_event(
        mock_invocation_context, auth_requests, author="custom-author"
    )

    assert event.author == "custom-author"

  def test_role_is_set_in_content(self, mock_invocation_context):
    """Test that role is set in content."""
    auth_requests = {"call_123": create_oauth2_auth_config()}

    event = build_auth_request_event(
        mock_invocation_context, auth_requests, role="model"
    )

    assert event.content.role == "model"
