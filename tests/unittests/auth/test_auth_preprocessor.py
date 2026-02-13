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

from google.adk.agents.invocation_context import InvocationContext
from google.adk.auth.auth_handler import AuthHandler
from google.adk.auth.auth_preprocessor import _AuthLlmRequestProcessor
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_tool import AuthToolArguments
from google.adk.events.event import Event
from google.adk.flows.llm_flows.functions import REQUEST_EUC_FUNCTION_CALL_NAME
from google.adk.models.llm_request import LlmRequest
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
    return context

  @pytest.fixture
  def mock_llm_request(self):
    """Create a mock LlmRequest."""
    return Mock(spec=LlmRequest)

  @pytest.fixture
  def mock_auth_config(self):
    """Create a mock AuthConfig."""
    return Mock(spec=AuthConfig)

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
    mock_context.agent = Mock()
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
  async def test_processes_auth_response_successfully(
      self,
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

    mock_invocation_context.session.events = [
        mock_user_event_with_auth_response
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
  @patch('google.adk.flows.llm_flows.functions.handle_function_calls_async')
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

    # Create system function call events
    system_function_call_1 = Mock()
    system_function_call_1.id = 'auth_id_1'
    system_function_call_1.args = {
        'function_call_id': 'tool_id_1',
        'auth_config': mock_auth_config,
    }

    system_function_call_2 = Mock()
    system_function_call_2.id = 'auth_id_2'
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
    """Test that missing matching system function calls returns early."""
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

    # Should process auth response but not resume any tools
    mock_auth_handler.parse_and_store_auth_response.assert_called_once()
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
    mock_auth_tool_args_validate.return_value = auth_tool_args

    system_function_call = Mock()
    system_function_call.id = 'auth_response_id'  # Matches the response ID
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
    mock_context.agent = Mock()  # This will fail isinstance(agent, LlmAgent)
    mock_context.session = mock_session

    result = []
    async for event in processor.run_async(mock_context, mock_llm_request):
      result.append(event)

    assert result == []
