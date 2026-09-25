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

"""Unit tests for auth_preprocessor module."""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlowAuthorizationCode
from fastapi.openapi.models import OAuthFlows
from google.adk.agents.invocation_context import InvocationContext
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_handler import AuthHandler
from google.adk.auth.auth_preprocessor import _AuthLlmRequestProcessor
from google.adk.auth.auth_preprocessor import _store_auth_and_collect_resume_targets
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_tool import AuthToolArguments
from google.adk.events.event import Event
from google.adk.flows.llm_flows.functions import REQUEST_EUC_FUNCTION_CALL_NAME
from google.adk.models.llm_request import LlmRequest
from google.genai import types
import pytest


class TestAuthLlmRequestProcessor:
  """Tests for _AuthLlmRequestProcessor class."""

  @pytest.fixture
  def processor(self):
    """Create an _AuthLlmRequestProcessor instance."""
    return _AuthLlmRequestProcessor()

  @pytest.fixture
  def mock_llm_agent(self):
    """Create a mock LlmAgent."""
    from google.adk.agents.llm_agent import LlmAgent

    agent = Mock(spec=LlmAgent)
    agent.name = 'test_agent'
    agent.canonical_tools = AsyncMock(return_value=[])
    return agent

  @pytest.fixture
  def mock_non_llm_agent(self):
    """Create a mock non-LLM agent."""
    agent = Mock()
    agent.__class__.__name__ = 'BaseAgent'
    return agent

  @pytest.fixture
  def mock_session(self):
    """Create a mock session."""
    session = Mock()
    session.state = {}
    session.events = []
    return session

  @pytest.fixture
  def mock_invocation_context(self, mock_llm_agent, mock_session):
    """Create a mock invocation context."""
    context = Mock(spec=InvocationContext)
    context.agent = mock_llm_agent
    context.session = mock_session
    context._get_events.side_effect = lambda **_: context.session.events
    return context

  @pytest.fixture
  def mock_llm_request(self):
    """Create a mock LlmRequest."""
    return Mock(spec=LlmRequest)

  @pytest.fixture
  def mock_auth_config(self):
    """Create a mock AuthConfig."""
    config = Mock(spec=AuthConfig)
    config.credential_key = None
    config.auth_scheme = None
    config.raw_auth_credential = None
    config.exchanged_auth_credential = None
    config.model_copy.return_value = config
    return config

  @pytest.fixture
  def mock_function_response_with_auth(self, mock_auth_config):
    """Create a mock function response with auth data."""
    function_response = Mock()
    function_response.name = REQUEST_EUC_FUNCTION_CALL_NAME
    function_response.id = 'auth_response_id'
    function_response.response = mock_auth_config
    return function_response

  @pytest.fixture
  def mock_function_response_without_auth(self):
    """Create a mock function response without auth data."""
    function_response = Mock()
    function_response.name = 'some_other_function'
    function_response.id = 'other_response_id'
    return function_response

  @pytest.fixture
  def mock_user_event_with_auth_response(
      self, mock_function_response_with_auth
  ):
    """Create a mock user event with auth response."""
    event = Mock(spec=Event)
    event.author = 'user'
    event.content = Mock()  # Non-None content
    event.get_function_calls.return_value = []
    event.get_function_responses.return_value = [
        mock_function_response_with_auth
    ]
    return event

  @pytest.fixture
  def mock_user_event_without_auth_response(
      self, mock_function_response_without_auth
  ):
    """Create a mock user event without auth response."""
    event = Mock(spec=Event)
    event.author = 'user'
    event.content = Mock()  # Non-None content
    event.get_function_responses.return_value = [
        mock_function_response_without_auth
    ]
    return event

  @pytest.fixture
  def mock_user_event_no_responses(self):
    """Create a mock user event with no responses."""
    event = Mock(spec=Event)
    event.author = 'user'
    event.content = Mock()  # Non-None content
    event.get_function_responses.return_value = []
    return event

  @pytest.fixture
  def mock_agent_event(self):
    """Create a mock agent-authored event."""
    event = Mock(spec=Event)
    event.author = 'test_agent'
    event.content = Mock()  # Non-None content
    return event

  @pytest.fixture
  def mock_event_no_content(self):
    """Create a mock event with no content."""
    event = Mock(spec=Event)
    event.author = 'user'
    event.content = None
    return event

  @pytest.fixture
  def mock_agent_event_with_content(self):
    """Create a mock agent event with content."""
    event = Mock(spec=Event)
    event.author = 'test_agent'
    event.content = Mock()  # Non-None content
    return event

  @pytest.mark.asyncio
  async def test_non_llm_agent_returns_early(
      self, processor, mock_llm_request, mock_session
  ):
    """Test that non-LLM agents return early."""
    mock_context = Mock(spec=InvocationContext)
    # Using spec=[] ensures hasattr(agent, 'canonical_tools') returns False.
    mock_context.agent = Mock(spec=[])
    mock_context.agent.__class__.__name__ = 'BaseAgent'
    mock_context.session = mock_session

    result = []
    async for event in processor.run_async(mock_context, mock_llm_request):
      result.append(event)

    assert result == []

  @pytest.mark.asyncio
  async def test_empty_events_returns_early(
      self, processor, mock_invocation_context, mock_llm_request
  ):
    """Test that empty events list returns early."""
    mock_invocation_context.session.events = []

    result = []
    async for event in processor.run_async(
        mock_invocation_context, mock_llm_request
    ):
      result.append(event)

    assert result == []

  @pytest.mark.asyncio
  async def test_no_events_with_content_returns_early(
      self,
      processor,
      mock_invocation_context,
      mock_llm_request,
      mock_event_no_content,
  ):
    """Test that no events with content returns early."""
    mock_invocation_context.session.events = [mock_event_no_content]

    result = []
    async for event in processor.run_async(
        mock_invocation_context, mock_llm_request
    ):
      result.append(event)

    assert result == []

  @pytest.mark.asyncio
  async def test_last_event_with_content_not_user_authored_returns_early(
      self,
      processor,
      mock_invocation_context,
      mock_llm_request,
      mock_event_no_content,
      mock_agent_event_with_content,
  ):
    """Test that last event with content not user-authored returns early."""
    # Mix of events: user event with no content, then agent event with content
    mock_invocation_context.session.events = [
        mock_event_no_content,
        mock_agent_event_with_content,
    ]

    result = []
    async for event in processor.run_async(
        mock_invocation_context, mock_llm_request
    ):
      result.append(event)

    assert result == []

  @pytest.mark.asyncio
  async def test_last_event_no_responses_returns_early(
      self,
      processor,
      mock_invocation_context,
      mock_llm_request,
      mock_user_event_no_responses,
  ):
    """Test that user event with no responses returns early."""
    mock_invocation_context.session.events = [mock_user_event_no_responses]

    result = []
    async for event in processor.run_async(
        mock_invocation_context, mock_llm_request
    ):
      result.append(event)

    assert result == []

  @pytest.mark.asyncio
  async def test_last_event_no_auth_responses_returns_early(
      self,
      processor,
      mock_invocation_context,
      mock_llm_request,
      mock_user_event_without_auth_response,
  ):
    """Test that user event with non-auth responses returns early."""
    mock_invocation_context.session.events = [
        mock_user_event_without_auth_response
    ]

    result = []
    async for event in processor.run_async(
        mock_invocation_context, mock_llm_request
    ):
      result.append(event)

    assert result == []

  @pytest.mark.asyncio
  @patch('google.adk.auth.auth_preprocessor.AuthHandler')
  @patch('google.adk.auth.auth_tool.AuthConfig.model_validate')
  async def test_ignores_auth_responses_outside_current_branch(
      self,
      mock_auth_config_validate,
      mock_auth_handler_class,
      processor,
      mock_invocation_context,
      mock_llm_request,
      mock_user_event_with_auth_response,
  ):
    """Test auth responses hidden by branch filtering are ignored."""
    mock_invocation_context.session.events = [
        mock_user_event_with_auth_response
    ]
    mock_invocation_context._get_events.side_effect = None
    mock_invocation_context._get_events.return_value = []

    result = []
    async for event in processor.run_async(
        mock_invocation_context, mock_llm_request
    ):
      result.append(event)

    mock_invocation_context._get_events.assert_called_once_with(
        current_branch=True
    )
    mock_auth_config_validate.assert_not_called()
    mock_auth_handler_class.assert_not_called()
    assert result == []

  @pytest.mark.asyncio
  @patch('google.adk.auth.auth_preprocessor.AuthHandler')
  @patch('google.adk.auth.auth_tool.AuthConfig.model_validate')
  @patch('google.adk.auth.auth_tool.AuthToolArguments.model_validate')
  async def test_processes_auth_response_successfully(
      self,
      mock_auth_tool_args_validate,
      mock_auth_config_validate,
      mock_auth_handler_class,
      processor,
      mock_invocation_context,
      mock_llm_request,
      mock_user_event_with_auth_response,
      mock_auth_config,
  ):
    """Test successful processing of auth response in last event."""
    # Setup mocks
    mock_auth_config_validate.return_value = mock_auth_config
    mock_auth_handler = Mock(spec=AuthHandler)
    mock_auth_handler.parse_and_store_auth_response = AsyncMock()
    mock_auth_handler_class.return_value = mock_auth_handler

    # The request this response answers; only a matching one is honoured.
    auth_tool_args = Mock(spec=AuthToolArguments)
    auth_tool_args.function_call_id = 'tool_id_1'
    auth_tool_args.auth_config = mock_auth_config
    mock_auth_tool_args_validate.return_value = auth_tool_args

    system_function_call = Mock()
    system_function_call.id = 'auth_response_id'
    system_function_call.name = REQUEST_EUC_FUNCTION_CALL_NAME
    system_function_call.args = {
        'function_call_id': 'tool_id_1',
        'auth_config': mock_auth_config,
    }

    system_event = Mock(spec=Event)
    system_event.content = Mock()  # Non-None content
    system_event.get_function_calls.return_value = [system_function_call]

    mock_invocation_context.session.events = [
        system_event,
        mock_user_event_with_auth_response,
    ]

    result = []
    async for event in processor.run_async(
        mock_invocation_context, mock_llm_request
    ):
      result.append(event)

    # Verify auth config validation was called
    mock_auth_config_validate.assert_called_once()

    # Verify auth handler was created with the config
    mock_auth_handler_class.assert_called_once_with(
        auth_config=mock_auth_config
    )

    # Verify parse_and_store_auth_response was called
    mock_auth_handler.parse_and_store_auth_response.assert_called_once_with(
        state=mock_invocation_context.session.state
    )

  @pytest.mark.asyncio
  @patch('google.adk.auth.auth_preprocessor.AuthHandler')
  @patch('google.adk.auth.auth_tool.AuthConfig.model_validate')
  @patch('google.adk.auth.auth_preprocessor.handle_function_calls_async')
  async def test_processes_multiple_auth_responses_and_resumes_tools(
      self,
      mock_handle_function_calls,
      mock_auth_config_validate,
      mock_auth_handler_class,
      processor,
      mock_invocation_context,
      mock_llm_request,
      mock_auth_config,
  ):
    """Test processing multiple auth responses and resuming tools."""
    # Create multiple auth responses
    auth_response_1 = Mock()
    auth_response_1.name = REQUEST_EUC_FUNCTION_CALL_NAME
    auth_response_1.id = 'auth_id_1'
    auth_response_1.response = mock_auth_config

    auth_response_2 = Mock()
    auth_response_2.name = REQUEST_EUC_FUNCTION_CALL_NAME
    auth_response_2.id = 'auth_id_2'
    auth_response_2.response = mock_auth_config

    user_event_with_multiple_responses = Mock(spec=Event)
    user_event_with_multiple_responses.author = 'user'
    user_event_with_multiple_responses.content = Mock()  # Non-None content
    user_event_with_multiple_responses.get_function_responses.return_value = [
        auth_response_1,
        auth_response_2,
    ]
    user_event_with_multiple_responses.get_function_calls.return_value = []

    # Create system function call events
    system_function_call_1 = Mock()
    system_function_call_1.id = 'auth_id_1'
    system_function_call_1.name = REQUEST_EUC_FUNCTION_CALL_NAME
    system_function_call_1.args = {
        'function_call_id': 'tool_id_1',
        'auth_config': mock_auth_config,
    }

    system_function_call_2 = Mock()
    system_function_call_2.id = 'auth_id_2'
    system_function_call_2.name = REQUEST_EUC_FUNCTION_CALL_NAME
    system_function_call_2.args = {
        'function_call_id': 'tool_id_2',
        'auth_config': mock_auth_config,
    }

    system_event = Mock(spec=Event)
    system_event.content = Mock()  # Non-None content
    system_event.get_function_calls.return_value = [
        system_function_call_1,
        system_function_call_2,
    ]

    # Create original function call event
    original_function_call_1 = Mock()
    original_function_call_1.id = 'tool_id_1'

    original_function_call_2 = Mock()
    original_function_call_2.id = 'tool_id_2'

    original_event = Mock(spec=Event)
    original_event.content = Mock()  # Non-None content
    original_event.author = 'test_agent'
    original_event.get_function_calls.return_value = [
        original_function_call_1,
        original_function_call_2,
    ]

    # Setup events in order: original -> system -> user_with_responses
    mock_invocation_context.session.events = [
        original_event,
        system_event,
        user_event_with_multiple_responses,
    ]

    # Setup mocks
    mock_auth_config_validate.return_value = mock_auth_config
    mock_auth_handler = Mock(spec=AuthHandler)
    mock_auth_handler.parse_and_store_auth_response = AsyncMock()
    mock_auth_handler_class.return_value = mock_auth_handler

    mock_function_response_event = Mock(spec=Event)
    mock_handle_function_calls.return_value = mock_function_response_event

    result = []
    async for event in processor.run_async(
        mock_invocation_context, mock_llm_request
    ):
      result.append(event)

    # Verify auth responses were processed
    assert mock_auth_handler.parse_and_store_auth_response.call_count == 2

    # Verify function calls were resumed
    mock_handle_function_calls.assert_called_once()
    call_args = mock_handle_function_calls.call_args
    assert call_args[0][1] == original_event  # The original event
    assert call_args[0][3] == {'tool_id_1', 'tool_id_2'}  # Tools to resume

    # Verify the function response event was yielded
    assert result == [mock_function_response_event]

  @pytest.mark.asyncio
  @patch('google.adk.auth.auth_preprocessor.AuthHandler')
  @patch('google.adk.auth.auth_tool.AuthConfig.model_validate')
  @patch('google.adk.auth.auth_preprocessor.handle_function_calls_async')
  async def test_does_not_resume_tool_call_authored_by_another_agent(
      self,
      mock_handle_function_calls,
      mock_auth_config_validate,
      mock_auth_handler_class,
      processor,
      mock_invocation_context,
      mock_llm_request,
      mock_auth_config,
  ):
    """Refuses to resume auth-gated tool calls authored by another agent."""
    # Given a session where the original tool call was authored by another agent
    auth_response_1 = Mock()
    auth_response_1.name = REQUEST_EUC_FUNCTION_CALL_NAME
    auth_response_1.id = 'auth_id_1'
    auth_response_1.response = mock_auth_config

    user_event_with_response = Mock(spec=Event)
    user_event_with_response.author = 'user'
    user_event_with_response.content = Mock()
    user_event_with_response.get_function_responses.return_value = [
        auth_response_1
    ]
    user_event_with_response.get_function_calls.return_value = []

    system_function_call_1 = Mock()
    system_function_call_1.id = 'auth_id_1'
    system_function_call_1.name = REQUEST_EUC_FUNCTION_CALL_NAME
    system_function_call_1.args = {
        'function_call_id': 'tool_id_1',
        'auth_config': mock_auth_config,
    }

    system_event = Mock(spec=Event)
    system_event.content = Mock()
    system_event.get_function_calls.return_value = [system_function_call_1]

    original_function_call_1 = Mock()
    original_function_call_1.id = 'tool_id_1'

    # This event belongs to a DIFFERENT agent than the one running the
    # current processor - the fix must refuse to resume it.
    original_event = Mock(spec=Event)
    original_event.content = Mock()
    original_event.author = 'a_different_agent'
    original_event.get_function_calls.return_value = [original_function_call_1]

    mock_invocation_context.session.events = [
        original_event,
        system_event,
        user_event_with_response,
    ]

    mock_auth_config_validate.return_value = mock_auth_config
    mock_auth_handler = Mock(spec=AuthHandler)
    mock_auth_handler.parse_and_store_auth_response = AsyncMock()
    mock_auth_handler_class.return_value = mock_auth_handler

    # When the processor is executed with the auth response
    result = []
    async for event in processor.run_async(
        mock_invocation_context, mock_llm_request
    ):
      result.append(event)

    # Then the auth response is stored (to record the user credential)
    assert mock_auth_handler.parse_and_store_auth_response.call_count == 1

    # But the tool call is not resumed because it belongs to a different agent
    mock_handle_function_calls.assert_not_called()
    assert result == []

  @pytest.mark.asyncio
  @patch('google.adk.auth.auth_preprocessor.AuthHandler')
  @patch('google.adk.auth.auth_tool.AuthConfig.model_validate')
  async def test_no_matching_system_function_calls_returns_early(
      self,
      mock_auth_config_validate,
      mock_auth_handler_class,
      processor,
      mock_invocation_context,
      mock_llm_request,
      mock_user_event_with_auth_response,
      mock_auth_config,
  ):
    """A response with no matching request in the session is dropped."""
    # Setup mocks
    mock_auth_config_validate.return_value = mock_auth_config
    mock_auth_handler = Mock(spec=AuthHandler)
    mock_auth_handler.parse_and_store_auth_response = AsyncMock()
    mock_auth_handler_class.return_value = mock_auth_handler

    # Create a non-matching system event
    non_matching_function_call = Mock()
    non_matching_function_call.id = (  # Different from 'auth_response_id'
        'different_id'
    )
    non_matching_function_call.name = REQUEST_EUC_FUNCTION_CALL_NAME

    system_event = Mock(spec=Event)
    system_event.content = Mock()  # Non-None content
    system_event.get_function_calls.return_value = [non_matching_function_call]

    mock_invocation_context.session.events = [
        system_event,
        mock_user_event_with_auth_response,
    ]

    result = []
    async for event in processor.run_async(
        mock_invocation_context, mock_llm_request
    ):
      result.append(event)

    # The server never issued a credential request under this ID, so the
    # response must not be trusted.
    mock_auth_handler.parse_and_store_auth_response.assert_not_called()
    assert result == []

  @pytest.mark.asyncio
  @patch('google.adk.auth.auth_preprocessor.AuthHandler')
  @patch('google.adk.auth.auth_tool.AuthConfig.model_validate')
  @patch('google.adk.auth.auth_tool.AuthToolArguments.model_validate')
  async def test_handles_missing_original_function_calls(
      self,
      mock_auth_tool_args_validate,
      mock_auth_config_validate,
      mock_auth_handler_class,
      processor,
      mock_invocation_context,
      mock_llm_request,
      mock_user_event_with_auth_response,
      mock_auth_config,
  ):
    """Test handling when original function calls are not found."""
    # Setup mocks
    mock_auth_config_validate.return_value = mock_auth_config
    mock_auth_handler = Mock(spec=AuthHandler)
    mock_auth_handler.parse_and_store_auth_response = AsyncMock()
    mock_auth_handler_class.return_value = mock_auth_handler

    # Create matching system function call
    auth_tool_args = Mock(spec=AuthToolArguments)
    auth_tool_args.function_call_id = 'tool_id_1'
    auth_tool_args.auth_config = mock_auth_config
    mock_auth_tool_args_validate.return_value = auth_tool_args

    system_function_call = Mock()
    system_function_call.id = 'auth_response_id'  # Matches the response ID
    system_function_call.name = REQUEST_EUC_FUNCTION_CALL_NAME
    system_function_call.args = {
        'function_call_id': 'tool_id_1',
        'auth_config': mock_auth_config,
    }

    system_event = Mock(spec=Event)
    system_event.content = Mock()  # Non-None content
    system_event.get_function_calls.return_value = [system_function_call]

    # Create event with no function calls (original function calls missing)
    empty_event = Mock(spec=Event)
    empty_event.content = Mock()  # Non-None content
    empty_event.get_function_calls.return_value = []

    mock_user_event_with_auth_response.get_function_calls.return_value = []

    mock_invocation_context.session.events = [
        empty_event,
        system_event,
        mock_user_event_with_auth_response,
    ]

    result = []
    async for event in processor.run_async(
        mock_invocation_context, mock_llm_request
    ):
      result.append(event)

    # Should process auth response but not find original function calls
    mock_auth_handler.parse_and_store_auth_response.assert_called_once()
    assert result == []

  @pytest.mark.asyncio
  async def test_isinstance_check_for_llm_agent(
      self, processor, mock_llm_request, mock_session
  ):
    """Test that isinstance check works correctly for LlmAgent."""
    # This test ensures the isinstance check work as expected

    # Create a mock that fails isinstance check
    mock_context = Mock(spec=InvocationContext)
    # This will fail isinstance(agent, LlmAgent)
    mock_context.agent = Mock(spec=[])
    mock_context.session = mock_session

    result = []
    async for event in processor.run_async(mock_context, mock_llm_request):
      result.append(event)

    assert result == []

  @pytest.mark.asyncio
  @patch('google.adk.auth.auth_preprocessor.AuthHandler')
  @patch('google.adk.auth.auth_tool.AuthConfig.model_validate')
  @patch('google.adk.auth.auth_preprocessor.handle_function_calls_async')
  async def test_resumes_tools_by_credential_key(
      self,
      mock_handle_function_calls,
      mock_auth_config_validate,
      mock_auth_handler_class,
      processor,
      mock_invocation_context,
      mock_llm_request,
  ):
    """Test that tools are resumed by credential key matching."""
    # Setup auth response
    auth_config = Mock(spec=AuthConfig)
    auth_config.credential_key = 'test_cred_key'
    auth_config.raw_auth_credential = None
    auth_config.exchanged_auth_credential = None
    mock_auth_config_validate.return_value = auth_config

    auth_response = Mock()
    auth_response.name = REQUEST_EUC_FUNCTION_CALL_NAME
    auth_response.id = 'auth_fc_id'
    auth_response.response = auth_config

    user_event = Mock(spec=Event)
    user_event.author = 'user'
    user_event.content = Mock()
    user_event.get_function_responses.return_value = [auth_response]
    user_event.get_function_calls.return_value = []

    # Setup system event (the one that requested auth)
    system_function_call = Mock()
    system_function_call.id = 'auth_fc_id'
    system_function_call.name = REQUEST_EUC_FUNCTION_CALL_NAME
    requested_auth_config = Mock(spec=AuthConfig)
    requested_auth_config.credential_key = 'test_cred_key'
    requested_auth_config.auth_scheme = None
    requested_auth_config.raw_auth_credential = None
    requested_auth_config.exchanged_auth_credential = None
    requested_auth_config.model_copy.return_value = requested_auth_config

    system_function_call.args = {
        'function_call_id': 'original_fc_id_1',
        'auth_config': requested_auth_config,
    }

    system_event = Mock(spec=Event)
    system_event.content = Mock()
    system_event.get_function_calls.return_value = [system_function_call]

    # Setup an event with actions.requested_auth_configs
    event_with_actions = Mock(spec=Event)
    event_with_actions.content = Mock()
    event_with_actions.get_function_calls.return_value = []

    actions = Mock()
    action_config = Mock()
    action_config.credential_key = 'test_cred_key'
    actions.requested_auth_configs = {
        'original_fc_id_1': action_config,
        'original_fc_id_2': action_config,
    }
    event_with_actions.actions = actions

    # Setup original function call events
    original_fc_1 = Mock()
    original_fc_1.id = 'original_fc_id_1'
    original_fc_2 = Mock()
    original_fc_2.id = 'original_fc_id_2'

    original_event = Mock(spec=Event)
    original_event.content = Mock()
    original_event.author = 'test_agent'
    original_event.get_function_calls.return_value = [
        original_fc_1,
        original_fc_2,
    ]

    # Events in order: original -> event_with_actions -> system_event -> user_event
    mock_invocation_context.session.events = [
        original_event,
        event_with_actions,
        system_event,
        user_event,
    ]

    mock_auth_handler = Mock(spec=AuthHandler)
    mock_auth_handler.parse_and_store_auth_response = AsyncMock()
    mock_auth_handler_class.return_value = mock_auth_handler

    mock_function_response_event = Mock(spec=Event)
    mock_handle_function_calls.return_value = mock_function_response_event

    with patch(
        'google.adk.auth.auth_tool.AuthToolArguments.model_validate'
    ) as mock_auth_tool_args_validate:
      mock_args = Mock(spec=AuthToolArguments)
      mock_args.auth_config = requested_auth_config
      mock_args.function_call_id = 'original_fc_id_1'
      mock_auth_tool_args_validate.return_value = mock_args

      result = []
      async for event in processor.run_async(
          mock_invocation_context, mock_llm_request
      ):
        result.append(event)

    mock_handle_function_calls.assert_called_once()
    call_args = mock_handle_function_calls.call_args
    assert call_args[0][1] == original_event
    assert call_args[0][3] == {'original_fc_id_1', 'original_fc_id_2'}
    assert result == [mock_function_response_event]

  @pytest.mark.asyncio
  @patch('google.adk.auth.auth_preprocessor.AuthHandler')
  @patch('google.adk.auth.auth_tool.AuthConfig.model_validate')
  @patch('google.adk.auth.auth_preprocessor.handle_function_calls_async')
  async def test_does_not_resume_stale_tools_from_older_events(
      self,
      mock_handle_function_calls,
      mock_auth_config_validate,
      mock_auth_handler_class,
      processor,
      mock_invocation_context,
      mock_llm_request,
  ):
    """Test that tools from older events with matching cred key are NOT resumed."""
    # Setup auth response
    auth_config = Mock(spec=AuthConfig)
    auth_config.credential_key = 'test_cred_key'
    auth_config.raw_auth_credential = None
    auth_config.exchanged_auth_credential = None
    mock_auth_config_validate.return_value = auth_config

    auth_response = Mock()
    auth_response.name = REQUEST_EUC_FUNCTION_CALL_NAME
    auth_response.id = 'auth_fc_id'
    auth_response.response = auth_config

    user_event = Mock(spec=Event)
    user_event.author = 'user'
    user_event.content = Mock()
    user_event.get_function_responses.return_value = [auth_response]
    user_event.get_function_calls.return_value = []

    # Setup system event (the one that requested auth)
    system_function_call = Mock()
    system_function_call.id = 'auth_fc_id'
    system_function_call.name = REQUEST_EUC_FUNCTION_CALL_NAME
    requested_auth_config = Mock(spec=AuthConfig)
    requested_auth_config.credential_key = 'test_cred_key'
    requested_auth_config.auth_scheme = None
    requested_auth_config.raw_auth_credential = None
    requested_auth_config.exchanged_auth_credential = None

    system_function_call.args = {
        'function_call_id': 'original_fc_id_1',
        'auth_config': requested_auth_config,
    }

    system_event = Mock(spec=Event)
    system_event.content = Mock()
    system_event.get_function_calls.return_value = [system_function_call]

    # Setup a fresh event with actions.requested_auth_configs
    fresh_event_with_actions = Mock(spec=Event)
    fresh_event_with_actions.content = Mock()
    fresh_event_with_actions.get_function_calls.return_value = []
    actions_fresh = Mock()
    action_config_fresh = Mock()
    action_config_fresh.credential_key = 'test_cred_key'
    actions_fresh.requested_auth_configs = {
        'original_fc_id_1': action_config_fresh,
    }
    fresh_event_with_actions.actions = actions_fresh

    # Setup an OLD event with actions.requested_auth_configs that also used test_cred_key
    old_event_with_actions = Mock(spec=Event)
    old_event_with_actions.content = Mock()
    old_event_with_actions.get_function_calls.return_value = []
    actions_old = Mock()
    action_config_old = Mock()
    action_config_old.credential_key = 'test_cred_key'
    actions_old.requested_auth_configs = {'stale_fc_id': action_config_old}
    old_event_with_actions.actions = actions_old

    # Setup original function call events
    original_fc_1 = Mock()
    original_fc_1.id = 'original_fc_id_1'
    original_fc_stale = Mock()
    original_fc_stale.id = 'stale_fc_id'

    original_event = Mock(spec=Event)
    original_event.content = Mock()
    original_event.author = 'test_agent'
    original_event.get_function_calls.return_value = [
        original_fc_1,
        original_fc_stale,
    ]

    # Events in order: old_event -> original -> fresh_event -> system -> user
    mock_invocation_context.session.events = [
        old_event_with_actions,
        original_event,
        fresh_event_with_actions,
        system_event,
        user_event,
    ]

    mock_auth_handler = Mock(spec=AuthHandler)
    mock_auth_handler.parse_and_store_auth_response = AsyncMock()
    mock_auth_handler_class.return_value = mock_auth_handler

    mock_function_response_event = Mock(spec=Event)
    mock_handle_function_calls.return_value = mock_function_response_event

    with patch(
        'google.adk.auth.auth_tool.AuthToolArguments.model_validate'
    ) as mock_auth_tool_args_validate:
      mock_args = Mock(spec=AuthToolArguments)
      mock_args.auth_config = requested_auth_config
      mock_args.function_call_id = 'original_fc_id_1'
      mock_auth_tool_args_validate.return_value = mock_args

      result = []
      async for event in processor.run_async(
          mock_invocation_context, mock_llm_request
      ):
        result.append(event)

    mock_handle_function_calls.assert_called_once()
    call_args = mock_handle_function_calls.call_args
    assert call_args[0][1] == original_event
    # Should only resume original_fc_id_1, NOT stale_fc_id
    assert call_args[0][3] == {'original_fc_id_1'}
    assert result == [mock_function_response_event]

  @pytest.mark.asyncio
  @patch('google.adk.auth.auth_preprocessor.AuthHandler')
  async def test_store_auth_merges_oauth2_fields(
      self,
      mock_auth_handler_class,
  ):
    """Test that the raw credential is pinned and OAuth2 fields are merged.

    The raw credential is taken from the request wholesale. The exchanged
    credential is the client's, backfilled from the request wherever the
    client left a field empty.
    """
    # Setup AuthHandler mock
    mock_auth_handler = Mock(spec=AuthHandler)
    mock_auth_handler.parse_and_store_auth_response = AsyncMock()
    mock_auth_handler_class.return_value = mock_auth_handler

    # Create requested auth config (the one in the event history)
    # It has all OAuth2 fields populated.
    requested_oauth2 = OAuth2Auth(
        client_id='expected_client_id',
        client_secret='expected_client_secret',
        redirect_uri='expected_redirect_uri',
        code_verifier='expected_code_verifier',
        code_challenge_method='S256',
        token_endpoint_auth_method='client_secret_post',
    )
    requested_auth_config = AuthConfig(
        auth_scheme=OAuth2(
            flows=OAuthFlows(
                authorizationCode=OAuthFlowAuthorizationCode(
                    authorizationUrl='https://example.com/auth',
                    tokenUrl='https://example.com/token',
                )
            )
        ),
        raw_auth_credential=AuthCredential(
            auth_type=AuthCredentialTypes.OAUTH2,
            oauth2=requested_oauth2,
        ),
        exchanged_auth_credential=AuthCredential(
            auth_type=AuthCredentialTypes.OAUTH2,
            oauth2=requested_oauth2,
        ),
        credential_key='test_cred_key',
    )

    # Create the auth response (the one returned by the client)
    # It has some missing OAuth2 fields that should be merged.
    stored_oauth2_raw = OAuth2Auth(
        client_id=None,
        client_secret=None,
        redirect_uri=None,
        code_verifier=None,
        code_challenge_method=None,
        access_token='some_access_token',
    )
    stored_oauth2_exchanged = OAuth2Auth(
        client_id=None,
        client_secret=None,
        redirect_uri=None,
        code_verifier=None,
        code_challenge_method=None,
        access_token='some_exchanged_token',
    )
    stored_auth_config = AuthConfig(
        auth_scheme=OAuth2(
            flows=OAuthFlows(
                authorizationCode=OAuthFlowAuthorizationCode(
                    authorizationUrl='https://example.com/auth',
                    tokenUrl='https://example.com/token',
                )
            )
        ),
        raw_auth_credential=AuthCredential(
            auth_type=AuthCredentialTypes.OAUTH2,
            oauth2=stored_oauth2_raw,
        ),
        exchanged_auth_credential=AuthCredential(
            auth_type=AuthCredentialTypes.OAUTH2,
            oauth2=stored_oauth2_exchanged,
        ),
        credential_key='test_cred_key',
    )

    # Setup function call in history that requested auth
    system_function_call = Mock()
    system_function_call.id = 'auth_fc_id'
    system_function_call.name = REQUEST_EUC_FUNCTION_CALL_NAME
    system_function_call.args = {
        'function_call_id': 'original_fc_id',
        'auth_config': requested_auth_config,
    }

    system_event = Mock(spec=Event)
    system_event.content = Mock()
    system_event.get_function_calls.return_value = [system_function_call]

    # Setup state
    mock_state = Mock()

    # Call _store_auth_and_collect_resume_targets
    await _store_auth_and_collect_resume_targets(
        events=[system_event],
        auth_fc_ids={'auth_fc_id'},
        auth_responses={
            'auth_fc_id': stored_auth_config.model_dump(
                mode='json', exclude_defaults=True
            )
        },
        state=mock_state,
    )

    # Verify AuthHandler was called with merged config
    mock_auth_handler_class.assert_called_once()
    called_config = mock_auth_handler_class.call_args.kwargs['auth_config']

    # Check raw_auth_credential fields
    assert (
        called_config.raw_auth_credential.oauth2.client_id
        == 'expected_client_id'
    )
    assert (
        called_config.raw_auth_credential.oauth2.client_secret
        == 'expected_client_secret'
    )
    assert (
        called_config.raw_auth_credential.oauth2.redirect_uri
        == 'expected_redirect_uri'
    )
    assert (
        called_config.raw_auth_credential.oauth2.code_verifier
        == 'expected_code_verifier'
    )
    assert (
        called_config.raw_auth_credential.oauth2.code_challenge_method == 'S256'
    )
    assert (
        called_config.raw_auth_credential.oauth2.token_endpoint_auth_method
        == 'client_secret_post'
    )
    # The raw credential names the OAuth2 client the token is exchanged for,
    # so it is the server's copy and nothing the client put in its own copy —
    # this access token — carries over.
    assert called_config.raw_auth_credential.oauth2.access_token is None

    # Check exchanged_auth_credential fields
    assert (
        called_config.exchanged_auth_credential.oauth2.client_id
        == 'expected_client_id'
    )
    assert (
        called_config.exchanged_auth_credential.oauth2.client_secret
        == 'expected_client_secret'
    )
    assert (
        called_config.exchanged_auth_credential.oauth2.redirect_uri
        == 'expected_redirect_uri'
    )
    assert (
        called_config.exchanged_auth_credential.oauth2.code_verifier
        == 'expected_code_verifier'
    )
    assert (
        called_config.exchanged_auth_credential.oauth2.code_challenge_method
        == 'S256'
    )
    assert (
        called_config.exchanged_auth_credential.oauth2.token_endpoint_auth_method
        == 'client_secret_post'
    )
    assert (
        called_config.exchanged_auth_credential.oauth2.access_token
        == 'some_exchanged_token'
    )

  def test_merge_credential_oauth2_fields_when_target_oauth2_is_none(self):
    """Test merging fields into a target credential where target.oauth2 is None."""
    from google.adk.auth.auth_preprocessor import _merge_credential_oauth2_fields

    target = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=None,
    )
    source = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id='expected_client_id',
            client_secret='expected_client_secret',
        ),
    )

    merged = _merge_credential_oauth2_fields(target, source)
    assert merged is not None
    assert merged.oauth2 is not None
    assert merged.oauth2.client_id == 'expected_client_id'
    assert merged.oauth2.client_secret == 'expected_client_secret'


class TestRequestPinning:
  """The exchange runs against the request this server issued."""

  @staticmethod
  def _auth_scheme():
    from google.adk.auth.auth_schemes import OpenIdConnectWithConfig

    return OpenIdConnectWithConfig(
        type_='openIdConnect',
        openIdConnectUrl='https://example.com/.well-known/openid-configuration',
        authorization_endpoint='https://example.com/auth',
        token_endpoint='https://example.com/token',
        scopes=['profile'],
    )

  @staticmethod
  def _oauth2_credential():
    from google.adk.auth.auth_credential import AuthCredential
    from google.adk.auth.auth_credential import AuthCredentialTypes
    from google.adk.auth.auth_credential import OAuth2Auth

    return AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id='real-client-id',
            client_secret='server-secret',
            redirect_uri='https://example.com/callback',
        ),
    )

  def _issued_config(self):
    return AuthConfig(
        auth_scheme=self._auth_scheme(),
        raw_auth_credential=self._oauth2_credential(),
        exchanged_auth_credential=self._oauth2_credential(),
    )

  @staticmethod
  def _request_event(issued: AuthConfig) -> Event:
    """The `adk_request_credential` call this server issued."""
    return Event(
        author='model',
        content=types.Content(
            role='model',
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id='fc-1',
                        name=REQUEST_EUC_FUNCTION_CALL_NAME,
                        args=AuthToolArguments(
                            function_call_id='original-fc',
                            auth_config=issued,
                        ).model_dump(
                            mode='json', exclude_none=True, by_alias=True
                        ),
                    )
                )
            ],
        ),
    )

  @pytest.mark.asyncio
  @patch('google.adk.auth.auth_preprocessor.AuthHandler')
  async def test_scheme_comes_from_the_request_not_the_response(
      self, mock_auth_handler_class
  ):
    """Taking the scheme from the response would let a client redirect the

    token exchange, and the developer's secret with it, to itself.
    """
    from google.adk.auth.auth_preprocessor import _store_auth_and_collect_resume_targets

    issued = self._issued_config()

    forged = issued.model_copy(deep=True)
    forged.auth_scheme.token_endpoint = 'https://attacker.example/token'
    forged.auth_scheme.authorization_endpoint = 'https://attacker.example/auth'

    mock_handler = Mock()
    mock_handler.parse_and_store_auth_response = AsyncMock()
    mock_auth_handler_class.return_value = mock_handler

    await _store_auth_and_collect_resume_targets(
        events=[self._request_event(issued)],
        auth_fc_ids={'fc-1'},
        auth_responses={
            'fc-1': forged.model_dump(
                mode='json', exclude_none=True, by_alias=True
            )
        },
        state={},
    )

    used_config = mock_auth_handler_class.call_args.kwargs['auth_config']
    assert used_config.auth_scheme.token_endpoint == 'https://example.com/token'

  @pytest.mark.asyncio
  @patch('google.adk.auth.auth_preprocessor.AuthHandler')
  async def test_response_to_an_unrequested_call_id_is_ignored(
      self, mock_auth_handler_class
  ):
    """With no matching request there is nothing to pin against, so the

    response would choose both the credential key and the endpoint.
    """
    from google.adk.auth.auth_preprocessor import _store_auth_and_collect_resume_targets

    forged = self._issued_config().model_copy(deep=True)
    forged.auth_scheme.token_endpoint = 'https://attacker.example/token'

    mock_handler = Mock()
    mock_handler.parse_and_store_auth_response = AsyncMock()
    mock_auth_handler_class.return_value = mock_handler

    resumed = await _store_auth_and_collect_resume_targets(
        events=[],
        auth_fc_ids={'fc-never-issued'},
        auth_responses={
            'fc-never-issued': forged.model_dump(
                mode='json', exclude_none=True, by_alias=True
            )
        },
        state={},
    )

    mock_auth_handler_class.assert_not_called()
    mock_handler.parse_and_store_auth_response.assert_not_called()
    assert resumed == set()

  @pytest.mark.asyncio
  @pytest.mark.parametrize('malformed', ['not-a-config', {'auth_scheme': 7}])
  @patch('google.adk.auth.auth_preprocessor.AuthHandler')
  async def test_malformed_auth_response_is_skipped(
      self, mock_auth_handler_class, malformed
  ):
    """A client can send anything here, so a bad payload skips that call

    instead of raising out of the preprocessor and ending the invocation.
    """
    from google.adk.auth.auth_preprocessor import _store_auth_and_collect_resume_targets

    issued = self._issued_config()

    mock_handler = Mock()
    mock_handler.parse_and_store_auth_response = AsyncMock()
    mock_auth_handler_class.return_value = mock_handler

    await _store_auth_and_collect_resume_targets(
        events=[self._request_event(issued)],
        auth_fc_ids={'fc-1'},
        auth_responses={'fc-1': malformed},
        state={},
    )

    # Nothing is stored; the caller is left to re-request auth.
    mock_auth_handler_class.assert_not_called()
    mock_handler.parse_and_store_auth_response.assert_not_called()
