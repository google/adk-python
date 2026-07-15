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

import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import AsyncMock
from unittest.mock import create_autospec
from unittest.mock import Mock
from unittest.mock import patch

from a2a.client import Client as A2AClient
from a2a.client.client import ClientConfig
from a2a.client.client_factory import ClientFactory
from a2a.types import AgentCapabilities
from a2a.types import AgentCard
from a2a.types import AgentSkill
from a2a.types import Artifact
from a2a.types import Message as A2AMessage
from a2a.types import Task as A2ATask
from a2a.types import TaskArtifactUpdateEvent
from a2a.types import TaskStatus as A2ATaskStatus
from a2a.types import TaskStatusUpdateEvent
from google.adk.a2a import _compat
from google.adk.a2a.agent import ParametersConfig
from google.adk.a2a.agent import RequestInterceptor
from google.adk.a2a.agent.config import A2aRemoteAgentConfig
from google.adk.a2a.agent.utils import execute_after_request_interceptors
from google.adk.a2a.agent.utils import execute_before_request_interceptors
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.remote_a2a_agent import A2A_METADATA_PREFIX
from google.adk.agents.remote_a2a_agent import AgentCardResolutionError
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
import google.adk.agents.remote_a2a_agent as remote_a2a_agent
from google.adk.events.event import Event
from google.adk.sessions.session import Session
from google.genai import types as genai_types
import httpx
import pytest


def _make_agent_card(
    name="test-agent",
    url="https://example.com/rpc",
    description="Test agent",
    *,
    version="1.0",
    skills=None,
    **kwargs,
):
  """Build an AgentCard version-agnostically for tests."""

  if skills is None:
    skills = []
  if _compat.IS_A2A_V1:
    card = _compat.parse_agent_card({
        "name": name,
        "description": description,
        "version": version,
        "supported_interfaces": [{"url": url, "protocol_binding": "JSONRPC"}],
        "default_input_modes": ["text/plain"],
        "default_output_modes": ["text/plain"],
    })
    for skill in skills:
      card.skills.append(skill)
    return card
  else:
    return AgentCard(
        name=name,
        url=url,
        description=description,
        version=version,
        capabilities=AgentCapabilities(),
        skills=skills,
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        **kwargs,
    )


def _make_stream_message(message: A2AMessage):
  """Wrap a Message in the shape ``send_message`` yields for the active SDK.

  On 1.x ``send_message`` yields ``StreamResponse`` proto objects; on 0.3.x it
  yields the bare ``Message``. ``_compat.make_stream_normalizer`` collapses both
  back to the legacy shape, so tests build the version-correct raw item here.
  """
  if _compat.IS_A2A_V1:
    from a2a.types import StreamResponse

    resp = StreamResponse()
    resp.message.CopyFrom(message)
    return resp
  return message


# Helper function to create a proper AgentCard for testing
def create_test_agent_card(
    name: str = "test-agent",
    url: str = "https://example.com/rpc",
    description: str = "Test agent",
) -> AgentCard:
  """Create a test AgentCard with all required fields."""
  return _make_agent_card(
      name=name,
      url=url,
      description=description,
      version="1.0",
      skills=[
          AgentSkill(
              id="test-skill",
              name="Test Skill",
              description="A test skill",
              tags=["test"],
          )
      ],
  )


class TestRemoteA2aAgentInit:
  """Test RemoteA2aAgent initialization and validation."""

  def test_init_with_agent_card_object(self):
    """Test initialization with AgentCard object."""
    agent_card = create_test_agent_card()

    agent = RemoteA2aAgent(
        name="test_agent", agent_card=agent_card, description="Test description"
    )

    assert agent.name == "test_agent"
    assert agent.description == "Test description"
    assert agent._agent_card == agent_card
    assert agent._agent_card_source is None
    assert agent._httpx_client_needs_cleanup is True
    assert agent._is_resolved is False

  def test_init_with_url_string(self):
    """Test initialization with URL string."""
    agent = RemoteA2aAgent(
        name="test_agent", agent_card="https://example.com/agent.json"
    )

    assert agent.name == "test_agent"
    assert agent._agent_card is None
    assert agent._agent_card_source == "https://example.com/agent.json"

  def test_init_with_file_path(self):
    """Test initialization with file path."""
    agent = RemoteA2aAgent(name="test_agent", agent_card="/path/to/agent.json")

    assert agent.name == "test_agent"
    assert agent._agent_card is None
    assert agent._agent_card_source == "/path/to/agent.json"

  def test_init_with_shared_httpx_client(self):
    """Test initialization with shared httpx client."""
    httpx_client = httpx.AsyncClient()
    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card="https://example.com/agent.json",
        httpx_client=httpx_client,
    )

    assert agent._httpx_client is not None
    assert agent._httpx_client_needs_cleanup is False

  def test_init_with_factory(self):
    """Test initialization with shared httpx client."""
    httpx_client = httpx.AsyncClient()
    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card="https://example.com/agent.json",
        httpx_client=httpx_client,
    )

    assert agent._httpx_client == httpx_client
    assert agent._httpx_client_needs_cleanup is False

  def test_init_with_none_agent_card(self):
    """Test initialization with None agent card raises ValueError."""
    with pytest.raises(ValueError, match="agent_card cannot be None"):
      RemoteA2aAgent(name="test_agent", agent_card=None)

  def test_init_with_empty_string_agent_card(self):
    """Test initialization with empty string agent card raises ValueError."""
    with pytest.raises(ValueError, match="agent_card string cannot be empty"):
      RemoteA2aAgent(name="test_agent", agent_card="   ")

  def test_init_with_invalid_type_agent_card(self):
    """Test initialization with invalid type agent card raises TypeError."""
    with pytest.raises(TypeError, match="agent_card must be AgentCard"):
      RemoteA2aAgent(name="test_agent", agent_card=123)

  def test_init_with_custom_timeout(self):
    """Test initialization with custom timeout."""
    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card="https://example.com/agent.json",
        timeout=300.0,
    )

    assert agent._timeout == 300.0


class TestRemoteA2aAgentResolution:
  """Test agent card resolution functionality."""

  def setup_method(self):
    """Setup test fixtures."""
    self.agent_card_data = {
        "name": "test-agent",
        "url": "https://example.com/rpc",
        "description": "Test agent",
        "version": "1.0",
        "capabilities": {},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["application/json"],
        "skills": [{
            "id": "test-skill",
            "name": "Test Skill",
            "description": "A test skill",
            "tags": ["test"],
        }],
    }
    self.agent_card = create_test_agent_card()

  @pytest.mark.asyncio
  async def test_ensure_httpx_client_creates_new_client(self):
    """Test that _ensure_httpx_client creates new client when none exists."""
    agent = RemoteA2aAgent(
        name="test_agent", agent_card=create_test_agent_card()
    )

    client = await agent._ensure_httpx_client()

    assert client is not None
    assert agent._httpx_client == client
    assert agent._httpx_client_needs_cleanup is True

    if not _compat.IS_A2A_V1:
      assert agent._a2a_client_factory._config.supported_transports == [
          _compat.TransportProtocol.jsonrpc,
          _compat.TransportProtocol.http_json,
      ]
    # 1.x uses supported_protocol_bindings instead.

  @pytest.mark.asyncio
  async def test_ensure_httpx_client_reuses_existing_client(self):
    """Test that _ensure_httpx_client reuses existing client."""
    existing_client = httpx.AsyncClient()
    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card=create_test_agent_card(),
        httpx_client=existing_client,
    )

    client = await agent._ensure_httpx_client()

    assert client == existing_client
    assert agent._httpx_client_needs_cleanup is False

  @pytest.mark.asyncio
  async def test_ensure_factory_reuses_existing_client(self):
    """Test that _ensure_httpx_client reuses existing client."""
    existing_client = httpx.AsyncClient()
    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card=create_test_agent_card(),
        a2a_client_factory=ClientFactory(
            ClientConfig(httpx_client=existing_client),
        ),
    )

    client = await agent._ensure_httpx_client()

    assert client == existing_client
    assert agent._httpx_client_needs_cleanup is False

  @pytest.mark.asyncio
  async def test_ensure_httpx_client_updates_factory_with_new_client(self):
    """Test that _ensure_httpx_client updates factory with new client."""
    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card=create_test_agent_card(),
        a2a_client_factory=ClientFactory(
            ClientConfig(httpx_client=None),
        ),
    )
    assert agent._a2a_client_factory._config.httpx_client is None

    client = await agent._ensure_httpx_client()

    assert client is not None
    assert agent._httpx_client == client
    assert agent._httpx_client_needs_cleanup is True
    assert agent._a2a_client_factory._config.httpx_client == client

  @pytest.mark.asyncio
  async def test_ensure_httpx_client_reregisters_transports_with_new_client(
      self,
  ):
    """Test that _ensure_httpx_client registers transports with new client."""
    factory = ClientFactory(
        ClientConfig(httpx_client=None),
    )
    factory.register("transport_label", lambda: "test")
    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card=create_test_agent_card(),
        a2a_client_factory=factory,
    )
    assert agent._a2a_client_factory._config.httpx_client is None
    assert "transport_label" in agent._a2a_client_factory._registry

    client = await agent._ensure_httpx_client()

    assert client is not None
    assert agent._httpx_client == client
    assert agent._httpx_client_needs_cleanup is True
    assert agent._a2a_client_factory._config.httpx_client == client
    if not _compat.IS_A2A_V1:
      # On 0.3.x the factory is reconstructed preserving custom
      # transports. On 1.x the factory is recreated fresh with only the
      # standard protocol bindings, so custom transports are not
      # preserved (intended production behavior).
      assert "transport_label" in agent._a2a_client_factory._registry

  @pytest.mark.asyncio
  async def test_resolve_agent_card_from_url_success(self):
    """Test successful agent card resolution from URL."""
    agent = RemoteA2aAgent(
        name="test_agent", agent_card="https://example.com/agent.json"
    )

    with patch.object(agent, "_ensure_httpx_client") as mock_ensure_client:
      mock_client = AsyncMock()
      mock_ensure_client.return_value = mock_client

      with patch(
          "google.adk.agents.remote_a2a_agent.A2ACardResolver"
      ) as mock_resolver_class:
        mock_resolver = AsyncMock()
        mock_resolver.get_agent_card.return_value = self.agent_card
        mock_resolver_class.return_value = mock_resolver

        result = await agent._resolve_agent_card_from_url(
            "https://example.com/agent.json"
        )

        assert result == self.agent_card
        mock_resolver_class.assert_called_once_with(
            httpx_client=mock_client, base_url="https://example.com"
        )
        mock_resolver.get_agent_card.assert_called_once_with(
            relative_card_path="/agent.json"
        )

  @pytest.mark.asyncio
  async def test_resolve_agent_card_from_url_invalid_url(self):
    """Test agent card resolution from invalid URL raises error."""
    agent = RemoteA2aAgent(name="test_agent", agent_card="invalid-url")

    with pytest.raises(AgentCardResolutionError, match="Invalid URL format"):
      await agent._resolve_agent_card_from_url("invalid-url")

  @pytest.mark.asyncio
  async def test_resolve_agent_card_from_file_success(self):
    """Test successful agent card resolution from file."""
    agent = RemoteA2aAgent(name="test_agent", agent_card="/path/to/agent.json")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
      json.dump(self.agent_card_data, f)
      temp_path = f.name

    try:
      result = await agent._resolve_agent_card_from_file(temp_path)
      assert result.name == self.agent_card.name
      assert _compat.agent_card_url(result) == _compat.agent_card_url(
          self.agent_card
      )
    finally:
      Path(temp_path).unlink()

  @pytest.mark.asyncio
  async def test_resolve_agent_card_from_file_not_found(self):
    """Test agent card resolution from nonexistent file raises error."""
    agent = RemoteA2aAgent(
        name="test_agent", agent_card="/path/to/nonexistent.json"
    )

    with pytest.raises(
        AgentCardResolutionError, match="Agent card file not found"
    ):
      await agent._resolve_agent_card_from_file("/path/to/nonexistent.json")

  @pytest.mark.asyncio
  async def test_resolve_agent_card_from_file_invalid_json(self):
    """Test agent card resolution from file with invalid JSON raises error."""
    agent = RemoteA2aAgent(name="test_agent", agent_card="/path/to/agent.json")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
      f.write("invalid json")
      temp_path = f.name

    try:
      with pytest.raises(AgentCardResolutionError, match="Invalid JSON"):
        await agent._resolve_agent_card_from_file(temp_path)
    finally:
      Path(temp_path).unlink()

  @pytest.mark.asyncio
  async def test_validate_agent_card_success(self):
    """Test successful agent card validation."""
    agent_card = create_test_agent_card()
    agent = RemoteA2aAgent(name="test_agent", agent_card=agent_card)

    # Should not raise any exception
    await agent._validate_agent_card(agent_card)

  @pytest.mark.asyncio
  async def test_validate_agent_card_no_url(self):
    """Test agent card validation fails when no URL."""
    agent = RemoteA2aAgent(
        name="test_agent", agent_card=create_test_agent_card()
    )

    invalid_card = _make_agent_card(
        name="test",
        url="",  # Empty URL to trigger validation error
    )

    with pytest.raises(
        AgentCardResolutionError, match="Agent card must have a valid URL"
    ):
      await agent._validate_agent_card(invalid_card)

  @pytest.mark.asyncio
  async def test_validate_agent_card_invalid_url(self):
    """Test agent card validation fails with invalid URL."""
    agent = RemoteA2aAgent(
        name="test_agent", agent_card=create_test_agent_card()
    )

    invalid_card = _make_agent_card(
        name="test",
        url="invalid-url",  # Invalid URL to trigger validation error
    )

    with pytest.raises(AgentCardResolutionError, match="Invalid RPC URL"):
      await agent._validate_agent_card(invalid_card)

  @pytest.mark.asyncio
  async def test_ensure_resolved_with_direct_agent_card(self):
    """Test _ensure_resolved with direct agent card."""
    agent_card = create_test_agent_card()
    agent = RemoteA2aAgent(name="test_agent", agent_card=agent_card)

    with patch("httpx.AsyncClient") as mock_client_class:
      mock_client = AsyncMock()
      mock_client_class.return_value = mock_client

      with patch(
          "google.adk.agents.remote_a2a_agent.A2AClientFactory"
      ) as mock_factory_class:
        mock_factory = Mock()
        mock_a2a_client = Mock()
        mock_factory.create.return_value = mock_a2a_client
        mock_factory_class.return_value = mock_factory

        await agent._ensure_resolved()

        assert agent._is_resolved is True
        assert agent._a2a_client == mock_a2a_client

  @pytest.mark.asyncio
  async def test_ensure_resolved_with_direct_agent_card_with_factory(self):
    """Test _ensure_resolved with direct agent card."""
    agent_card = create_test_agent_card()
    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card=agent_card,
        a2a_client_factory=ClientFactory(
            ClientConfig(),
        ),
    )

    with patch("httpx.AsyncClient") as mock_client_class:
      mock_client = AsyncMock()
      mock_client_class.return_value = mock_client

      # Rebinding reconstructs the factory through
      # ``_compat.A2AClientFactory``, so the patch must target it there.
      with patch(
          "google.adk.a2a._compat.A2AClientFactory"
      ) as mock_factory_class:
        mock_a2a_client = Mock()
        mock_factory = Mock()
        mock_factory.create.return_value = mock_a2a_client
        mock_factory_class.return_value = mock_factory

        await agent._ensure_resolved()

        assert agent._is_resolved is True
        assert agent._a2a_client == mock_a2a_client

  @pytest.mark.asyncio
  async def test_ensure_resolved_with_url_source(self):
    """Test _ensure_resolved with URL source."""
    agent = RemoteA2aAgent(
        name="test_agent", agent_card="https://example.com/agent.json"
    )

    agent_card = create_test_agent_card()
    with patch.object(agent, "_resolve_agent_card") as mock_resolve:
      mock_resolve.return_value = agent_card

      with patch.object(agent, "_ensure_httpx_client") as mock_ensure_client:
        mock_client = AsyncMock()
        mock_ensure_client.return_value = mock_client

        with patch(
            "google.adk.agents.remote_a2a_agent.A2AClient"
        ) as mock_client_class:
          mock_a2a_client = AsyncMock()
          mock_client_class.return_value = mock_a2a_client

          await agent._ensure_resolved()

          assert agent._is_resolved is True
          assert agent._agent_card == agent_card
          assert agent.description == agent_card.description

  @pytest.mark.asyncio
  async def test_ensure_resolved_already_resolved(self):
    """Test _ensure_resolved when already resolved."""
    agent_card = create_test_agent_card()
    agent = RemoteA2aAgent(name="test_agent", agent_card=agent_card)

    # Set up as already resolved
    agent._is_resolved = True
    agent._a2a_client = AsyncMock()

    with patch.object(agent, "_resolve_agent_card") as mock_resolve:
      await agent._ensure_resolved()

      # Should not call resolution again
      mock_resolve.assert_not_called()


class TestRemoteA2aAgentMessageHandling:
  """Test message handling functionality."""

  def setup_method(self):
    """Setup test fixtures."""
    self.agent_card = create_test_agent_card()
    self.mock_genai_part_converter = Mock()
    self.mock_a2a_part_converter = Mock()
    self.agent = RemoteA2aAgent(
        name="test_agent",
        agent_card=self.agent_card,
        genai_part_converter=self.mock_genai_part_converter,
        a2a_part_converter=self.mock_a2a_part_converter,
    )

    # Mock session and context
    self.mock_session = Mock(spec=Session)
    self.mock_session.id = "session-123"
    self.mock_session.events = []

    self.mock_context = Mock(spec=InvocationContext)
    self.mock_context.session = self.mock_session
    self.mock_context.invocation_id = "invocation-123"
    self.mock_context.branch = "main"

  def test_create_a2a_request_for_user_function_response_no_function_call(self):
    """Test function response request creation when no function call exists."""
    with patch(
        "google.adk.agents.remote_a2a_agent.find_matching_function_call"
    ) as mock_find:
      mock_find.return_value = None

      result = self.agent._create_a2a_request_for_user_function_response(
          self.mock_context
      )

      assert result is None

  def test_create_a2a_request_for_user_function_response_success(self):
    """Test successful function response request creation."""
    # Mock function call event
    mock_function_event = Mock()
    mock_function_event.custom_metadata = {
        A2A_METADATA_PREFIX + "task_id": "task-123"
    }
    mock_function_event.content = Mock()
    mock_function_event.content.parts = [Mock()]
    mock_function_event.get_function_calls.return_value = []

    # Mock latest event with function response - set proper author
    mock_latest_event = Mock()
    mock_latest_event.author = "user"
    self.mock_session.events = [mock_latest_event]

    with patch(
        "google.adk.agents.remote_a2a_agent.find_matching_function_call"
    ) as mock_find:
      mock_find.return_value = mock_function_event

      with patch(
          "google.adk.agents.remote_a2a_agent.convert_event_to_a2a_message"
      ) as mock_convert:
        # Create a proper mock A2A message
        mock_a2a_message = create_autospec(A2AMessage, instance=True)
        mock_a2a_message.task_id = None  # Will be set by the method
        mock_convert.return_value = mock_a2a_message

        result = self.agent._create_a2a_request_for_user_function_response(
            self.mock_context
        )

        assert result is not None
        assert result == mock_a2a_message
        assert mock_a2a_message.task_id == "task-123"

  def test_construct_message_parts_from_session_success(self):
    """Test successful message parts construction from session."""
    # Mock event with text content
    mock_part = Mock()
    mock_part.text = "Hello world"

    mock_content = Mock()
    mock_content.parts = [mock_part]

    mock_event = Mock()
    mock_event.content = mock_content

    self.mock_session.events = [mock_event]

    with patch(
        "google.adk.agents.remote_a2a_agent._present_other_agent_message"
    ) as mock_convert:
      mock_convert.return_value = mock_event

      mock_a2a_part = Mock()
      self.mock_genai_part_converter.return_value = mock_a2a_part

      parts, context_id = self.agent._construct_message_parts_from_session(
          self.mock_context
      )

      assert len(parts) == 1
      assert parts[0] == mock_a2a_part
      assert context_id is None

  def test_construct_message_parts_from_session_user_input_metadata(self):
    """Test that user input metadata is added for user messages."""

    mock_part = Mock()
    mock_content = Mock()
    mock_content.parts = [mock_part]

    mock_event = Mock()
    mock_event.content = mock_content
    mock_event.author = "user"
    mock_event.custom_metadata = None

    self.mock_session.events = [mock_event]

    with patch(
        "google.adk.agents.remote_a2a_agent._present_other_agent_message"
    ) as mock_convert:
      mock_convert.return_value = mock_event

      # Converter returns a real A2A part; production stamps is_user_input.
      a2a_part = _compat.make_text_part("hi")
      self.mock_genai_part_converter.return_value = a2a_part

      parts, _ = self.agent._construct_message_parts_from_session(
          self.mock_context
      )

      assert len(parts) == 1
      assert _compat.part_metadata(parts[0]).get("is_user_input") is True

  def test_construct_message_parts_from_session_success_multiple_parts(self):
    """Test successful message parts construction from session."""
    # Mock event with text content
    mock_part = Mock()
    mock_part.text = "Hello world"

    mock_content = Mock()
    mock_content.parts = [mock_part]

    mock_event = Mock()
    mock_event.content = mock_content

    self.mock_session.events = [mock_event]

    with patch(
        "google.adk.agents.remote_a2a_agent._present_other_agent_message"
    ) as mock_convert:
      mock_convert.return_value = mock_event

      mock_a2a_part1 = Mock()
      mock_a2a_part2 = Mock()
      self.mock_genai_part_converter.return_value = [
          mock_a2a_part1,
          mock_a2a_part2,
      ]

      parts, context_id = self.agent._construct_message_parts_from_session(
          self.mock_context
      )

      assert parts == [mock_a2a_part1, mock_a2a_part2]
      assert context_id is None

  def test_construct_message_parts_from_session_empty_events(self):
    """Test message parts construction with empty events."""
    self.mock_session.events = []

    parts, context_id = self.agent._construct_message_parts_from_session(
        self.mock_context
    )

    assert parts == []
    assert context_id is None

  def test_construct_message_parts_from_session_stops_on_agent_reply(self):
    """Test message parts construction stops on agent reply by default."""
    part1 = Mock()
    part1.text = "User 1"
    content1 = Mock()
    content1.parts = [part1]
    user1 = Mock()
    user1.content = content1
    user1.author = "user"
    user1.custom_metadata = None

    part2 = Mock()
    part2.text = "Agent 1"
    content2 = Mock()
    content2.parts = [part2]
    agent1 = Mock()
    agent1.content = content2
    agent1.author = self.agent.name
    agent1.custom_metadata = {
        A2A_METADATA_PREFIX + "response": True,
    }

    agent2 = Mock()
    agent2.content = None
    agent2.author = self.agent.name
    # Just actions, no content. Not marked as a response.
    agent2.actions = Mock()
    agent2.custom_metadata = None

    part3 = Mock()
    part3.text = "User 2"
    content3 = Mock()
    content3.parts = [part3]
    user2 = Mock()
    user2.content = content3
    user2.author = "user"
    user2.custom_metadata = None

    self.mock_session.events = [user1, agent1, user2, agent2]

    def mock_converter(part):
      return _compat.make_text_part(part.text)

    self.mock_genai_part_converter.side_effect = mock_converter

    with patch(
        "google.adk.agents.remote_a2a_agent._present_other_agent_message"
    ) as mock_present:
      mock_present.side_effect = lambda event: event
      parts, context_id = self.agent._construct_message_parts_from_session(
          self.mock_context
      )
      assert len(parts) == 1
      assert _compat.part_text(parts[0]) == "User 2"
      assert context_id is None

  def test_construct_message_parts_from_session_stateless_full_history(self):
    """Test full history for stateless agent when enabled."""
    self.agent._full_history_when_stateless = True
    part1 = Mock()
    part1.text = "User 1"
    content1 = Mock()
    content1.parts = [part1]
    user1 = Mock()
    user1.content = content1
    user1.author = "user"
    user1.custom_metadata = None

    part2 = Mock()
    part2.text = "Agent 1"
    content2 = Mock()
    content2.parts = [part2]
    agent1 = Mock()
    agent1.content = content2
    agent1.author = self.agent.name
    agent1.custom_metadata = None

    part3 = Mock()
    part3.text = "User 2"
    content3 = Mock()
    content3.parts = [part3]
    user2 = Mock()
    user2.content = content3
    user2.author = "user"
    user2.custom_metadata = None

    self.mock_session.events = [user1, agent1, user2]

    def mock_converter(part):
      return _compat.make_text_part(part.text)

    self.mock_genai_part_converter.side_effect = mock_converter

    with patch(
        "google.adk.agents.remote_a2a_agent._present_other_agent_message"
    ) as mock_present:
      mock_present.side_effect = lambda event: event
      parts, context_id = self.agent._construct_message_parts_from_session(
          self.mock_context
      )
      assert len(parts) == 3
      assert _compat.part_text(parts[0]) == "User 1"
      assert _compat.part_text(parts[1]) == "Agent 1"
      assert _compat.part_text(parts[2]) == "User 2"
      assert context_id is None

  def test_construct_message_parts_from_session_stateful_partial_history(self):
    """Test partial history for stateful agent when full history is enabled."""
    self.agent._full_history_when_stateless = True
    part1 = Mock()
    part1.text = "User 1"
    content1 = Mock()
    content1.parts = [part1]
    user1 = Mock()
    user1.content = content1
    user1.author = "user"
    user1.custom_metadata = None

    part2 = Mock()
    part2.text = "Agent 1"
    content2 = Mock()
    content2.parts = [part2]
    agent1 = Mock()
    agent1.content = content2
    agent1.author = self.agent.name
    agent1.custom_metadata = {
        A2A_METADATA_PREFIX + "response": True,
        A2A_METADATA_PREFIX + "context_id": "ctx-1",
    }

    part3 = Mock()
    part3.text = "User 2"
    content3 = Mock()
    content3.parts = [part3]
    user2 = Mock()
    user2.content = content3
    user2.author = "user"
    user2.custom_metadata = None

    self.mock_session.events = [user1, agent1, user2]

    def mock_converter(part):
      return _compat.make_text_part(part.text)

    self.mock_genai_part_converter.side_effect = mock_converter

    with patch(
        "google.adk.agents.remote_a2a_agent._present_other_agent_message"
    ) as mock_present:
      mock_present.side_effect = lambda event: event
      parts, context_id = self.agent._construct_message_parts_from_session(
          self.mock_context
      )
      assert len(parts) == 1
      assert _compat.part_text(parts[0]) == "User 2"
      assert context_id == "ctx-1"

  @pytest.mark.asyncio
  async def test_handle_a2a_response_success_with_message(self):
    """Test successful A2A response handling with message."""
    mock_a2a_message = Mock(spec=A2AMessage)
    mock_a2a_message.context_id = "context-123"

    # Create a proper Event mock that can handle custom_metadata
    mock_event = Event(
        author=self.agent.name,
        invocation_id=self.mock_context.invocation_id,
        branch=self.mock_context.branch,
    )

    with patch(
        "google.adk.agents.remote_a2a_agent.convert_a2a_message_to_event"
    ) as mock_convert:
      mock_convert.return_value = mock_event

      result = await self.agent._handle_a2a_response(
          mock_a2a_message, self.mock_context
      )

      assert result == mock_event
      mock_convert.assert_called_once_with(
          mock_a2a_message,
          self.agent.name,
          self.mock_context,
          self.mock_a2a_part_converter,
      )
      # Check that metadata was added
      assert result.custom_metadata is not None
      assert A2A_METADATA_PREFIX + "context_id" in result.custom_metadata

  @pytest.mark.asyncio
  async def test_handle_a2a_response_message_converter_returns_none(self):
    """Test _handle_a2a_response returns None when message converter returns None."""
    mock_a2a_message = Mock(spec=A2AMessage)
    mock_a2a_message.context_id = "context-123"

    with patch(
        "google.adk.agents.remote_a2a_agent.convert_a2a_message_to_event"
    ) as mock_convert:
      mock_convert.return_value = None

      result = await self.agent._handle_a2a_response(
          mock_a2a_message, self.mock_context
      )

      assert result is None
      mock_convert.assert_called_once_with(
          mock_a2a_message,
          self.agent.name,
          self.mock_context,
          self.mock_a2a_part_converter,
      )

  @pytest.mark.asyncio
  async def test_handle_a2a_response_with_task_completed_and_no_update(self):
    """Test successful A2A response handling with non-streaming task and no update."""
    mock_a2a_task = Mock(spec=A2ATask)
    mock_a2a_task.id = "task-123"
    mock_a2a_task.context_id = "context-123"
    mock_a2a_task.status = Mock(spec=A2ATaskStatus)
    mock_a2a_task.status.state = _compat.TS_COMPLETED

    # Create a proper Event mock that can handle custom_metadata
    mock_a2a_part = genai_types.Part.from_text(
        text="test"
    )  # real genai part for Content
    mock_event = Event(
        author=self.agent.name,
        invocation_id=self.mock_context.invocation_id,
        branch=self.mock_context.branch,
        content=genai_types.Content(role="model", parts=[mock_a2a_part]),
    )

    with patch.object(
        remote_a2a_agent,
        "convert_a2a_task_to_event",
        autospec=True,
    ) as mock_convert:
      mock_convert.return_value = mock_event

      result = await self.agent._handle_a2a_response(
          (mock_a2a_task, None), self.mock_context
      )

      assert result == mock_event
      mock_convert.assert_called_once_with(
          mock_a2a_task,
          self.agent.name,
          self.mock_context,
          self.mock_a2a_part_converter,
      )
      # Check the parts are not updated as Thought
      assert result.content.parts[0].thought is None
      # Check that metadata was added
      assert result.custom_metadata is not None
      assert A2A_METADATA_PREFIX + "task_id" in result.custom_metadata
      assert A2A_METADATA_PREFIX + "context_id" in result.custom_metadata

  def test_construct_message_parts_from_session_preserves_order(self):
    """Test that message parts are in correct order with multi-part messages.

    This test verifies the fix for the bug where _present_other_agent_message
    creates multi-part messages with "For context:" prefix, and ensures the
    parts are in the correct chronological order (not reversed).
    """
    # Create mock events with multiple parts
    # Event 1: User message
    user_part = Mock()
    user_part.text = "User question"
    user_content = Mock()
    user_content.parts = [user_part]
    user_event = Mock()
    user_event.content = user_content
    user_event.author = "user"

    # Event 2: Other agent message (will be transformed by
    # _present_other_agent_message)
    other_agent_part1 = Mock()
    other_agent_part1.text = "For context:"
    other_agent_part2 = Mock()
    other_agent_part2.text = "[other_agent] said: Response text"
    other_agent_content = Mock()
    other_agent_content.parts = [other_agent_part1, other_agent_part2]
    other_agent_event = Mock()
    other_agent_event.content = other_agent_content
    other_agent_event.author = "other_agent"

    self.mock_session.events = [user_event, other_agent_event]

    with patch(
        "google.adk.agents.remote_a2a_agent._present_other_agent_message"
    ) as mock_present:
      # Mock _present_other_agent_message to return the transformed event
      mock_present.return_value = other_agent_event

      # Converter returns real A2A parts (production reads their metadata);
      # track the conversion order for the ordering assertions below.
      converted_order = []

      def mock_converter(part):
        converted_order.append(part.text)
        return _compat.make_text_part(part.text)

      self.mock_genai_part_converter.side_effect = mock_converter

      parts, context_id = self.agent._construct_message_parts_from_session(
          self.mock_context
      )

      # Verify the parts are in correct order
      assert len(parts) == 3  # 1 user part + 2 other agent parts
      assert context_id is None

      # Verify order: user part, then "For context:", then agent message
      assert converted_order[0] == "User question"
      assert converted_order[1] == "For context:"
      assert converted_order[2] == "[other_agent] said: Response text"
      assert _compat.part_text(parts[0]) == "User question"
      assert _compat.part_text(parts[1]) == "For context:"

  @pytest.mark.asyncio
  async def test_handle_a2a_response_with_task_submitted_and_no_update(self):
    """Test successful A2A response handling with streaming task and no update."""
    mock_a2a_task = Mock(spec=A2ATask)
    mock_a2a_task.id = "task-123"
    mock_a2a_task.context_id = "context-123"
    mock_a2a_task.status = Mock(spec=A2ATaskStatus)
    mock_a2a_task.status.state = _compat.TS_SUBMITTED

    # Create a proper Event mock that can handle custom_metadata
    mock_a2a_part = genai_types.Part.from_text(
        text="test"
    )  # real genai part for Content
    mock_event = Event(
        author=self.agent.name,
        invocation_id=self.mock_context.invocation_id,
        branch=self.mock_context.branch,
        content=genai_types.Content(role="model", parts=[mock_a2a_part]),
    )

    with patch.object(
        remote_a2a_agent,
        "convert_a2a_task_to_event",
        autospec=True,
    ) as mock_convert:
      mock_convert.return_value = mock_event

      result = await self.agent._handle_a2a_response(
          (mock_a2a_task, None), self.mock_context
      )

      assert result == mock_event
      mock_convert.assert_called_once_with(
          mock_a2a_task,
          self.agent.name,
          self.mock_context,
          self.mock_a2a_part_converter,
      )
      # Check the parts are updated as Thought
      assert result.content.parts[0].thought is True
      assert result.content.parts[0].thought_signature is None
      # Check that metadata was added
      assert result.custom_metadata is not None
      assert A2A_METADATA_PREFIX + "task_id" in result.custom_metadata
      assert A2A_METADATA_PREFIX + "context_id" in result.custom_metadata

  @pytest.mark.asyncio
  @pytest.mark.parametrize(
      "task_state,event_content",
      [
          pytest.param(
              _compat.TS_SUBMITTED,
              genai_types.Content(role="model", parts=[]),
              id="submitted_empty_parts",
          ),
          pytest.param(
              _compat.TS_WORKING,
              None,
              id="working_no_content",
          ),
      ],
  )
  async def test_handle_a2a_response_with_task_missing_content(
      self, task_state, event_content
  ):
    """Test streaming A2A response handling when content/parts are missing.

    This verifies the fix for issue #3769 where the code could raise when it
    tried to read parts[0] without checking for empty/missing content.
    """
    mock_a2a_task = create_autospec(A2ATask, instance=True)
    mock_a2a_task.id = "task-123"
    mock_a2a_task.context_id = "context-123"
    mock_a2a_task.status = create_autospec(A2ATaskStatus, instance=True)
    mock_a2a_task.status.state = task_state

    mock_event = Event(
        author=self.agent.name,
        invocation_id=self.mock_context.invocation_id,
        branch=self.mock_context.branch,
        content=event_content,
    )

    with patch.object(
        remote_a2a_agent,
        "convert_a2a_task_to_event",
        autospec=True,
    ) as mock_convert:
      mock_convert.return_value = mock_event

      result = await self.agent._handle_a2a_response(
          (mock_a2a_task, None), self.mock_context
      )

      assert result == mock_event
      assert result.custom_metadata is not None
      assert A2A_METADATA_PREFIX + "task_id" in result.custom_metadata
      assert A2A_METADATA_PREFIX + "context_id" in result.custom_metadata

  @pytest.mark.asyncio
  async def test_handle_a2a_response_with_task_working_and_no_update(self):
    """Test successful A2A response handling with streaming task and no update."""
    mock_a2a_task = Mock(spec=A2ATask)
    mock_a2a_task.id = "task-123"
    mock_a2a_task.context_id = "context-123"
    mock_a2a_task.status = Mock(spec=A2ATaskStatus)
    mock_a2a_task.status.state = _compat.TS_WORKING

    # Create a proper Event mock that can handle custom_metadata
    mock_a2a_part = genai_types.Part.from_text(
        text="test"
    )  # real genai part for Content
    mock_event = Event(
        author=self.agent.name,
        invocation_id=self.mock_context.invocation_id,
        branch=self.mock_context.branch,
        content=genai_types.Content(role="model", parts=[mock_a2a_part]),
    )

    with patch.object(
        remote_a2a_agent,
        "convert_a2a_task_to_event",
        autospec=True,
    ) as mock_convert:
      mock_convert.return_value = mock_event

      result = await self.agent._handle_a2a_response(
          (mock_a2a_task, None), self.mock_context
      )

      assert result == mock_event
      mock_convert.assert_called_once_with(
          mock_a2a_task,
          self.agent.name,
          self.mock_context,
          self.mock_a2a_part_converter,
      )
      # Check the parts are updated as Thought
      assert result.content.parts[0].thought is True
      assert result.content.parts[0].thought_signature is None
      # Check that metadata was added
      assert result.custom_metadata is not None
      assert A2A_METADATA_PREFIX + "task_id" in result.custom_metadata
      assert A2A_METADATA_PREFIX + "context_id" in result.custom_metadata

  @pytest.mark.asyncio
  async def test_handle_a2a_response_with_task_status_update_with_message(self):
    """Test handling of a task status update with a message."""
    mock_a2a_task = Mock(spec=A2ATask)
    mock_a2a_task.id = "task-123"
    mock_a2a_task.context_id = "context-123"

    mock_a2a_message = Mock(spec=A2AMessage)
    mock_update = Mock(spec=TaskStatusUpdateEvent)
    mock_update.status = Mock(A2ATaskStatus)
    mock_update.status.state = _compat.TS_COMPLETED
    mock_update.status.message = mock_a2a_message

    # Create a proper Event mock that can handle custom_metadata
    mock_a2a_part = genai_types.Part.from_text(
        text="test"
    )  # real genai part for Content
    mock_event = Event(
        author=self.agent.name,
        invocation_id=self.mock_context.invocation_id,
        branch=self.mock_context.branch,
        content=genai_types.Content(role="model", parts=[mock_a2a_part]),
    )

    with patch(
        "google.adk.agents.remote_a2a_agent.convert_a2a_message_to_event"
    ) as mock_convert:
      mock_convert.return_value = mock_event

      result = await self.agent._handle_a2a_response(
          (mock_a2a_task, mock_update), self.mock_context
      )

      assert result == mock_event
      mock_convert.assert_called_once_with(
          mock_a2a_message,
          self.agent.name,
          self.mock_context,
          self.mock_a2a_part_converter,
      )
      # Check that metadata was added
      assert result.custom_metadata is not None
      assert result.content.parts[0].thought is None
      assert A2A_METADATA_PREFIX + "task_id" in result.custom_metadata
      assert A2A_METADATA_PREFIX + "context_id" in result.custom_metadata

  @pytest.mark.asyncio
  async def test_handle_a2a_response_with_task_status_working_update_with_message(
      self,
  ):
    """Test handling of a task status update with a message."""
    mock_a2a_task = Mock(spec=A2ATask)
    mock_a2a_task.id = "task-123"
    mock_a2a_task.context_id = "context-123"

    mock_a2a_message = Mock(spec=A2AMessage)
    mock_update = Mock(spec=TaskStatusUpdateEvent)
    mock_update.status = Mock(A2ATaskStatus)
    mock_update.status.state = _compat.TS_WORKING
    mock_update.status.message = mock_a2a_message

    # Create a proper Event mock that can handle custom_metadata
    mock_a2a_part = genai_types.Part.from_text(
        text="test"
    )  # real genai part for Content
    mock_event = Event(
        author=self.agent.name,
        invocation_id=self.mock_context.invocation_id,
        branch=self.mock_context.branch,
        content=genai_types.Content(role="model", parts=[mock_a2a_part]),
    )

    with patch(
        "google.adk.agents.remote_a2a_agent.convert_a2a_message_to_event"
    ) as mock_convert:
      mock_convert.return_value = mock_event

      result = await self.agent._handle_a2a_response(
          (mock_a2a_task, mock_update), self.mock_context
      )

      assert result == mock_event
      mock_convert.assert_called_once_with(
          mock_a2a_message,
          self.agent.name,
          self.mock_context,
          self.mock_a2a_part_converter,
      )
      # Check that metadata was added
      assert result.custom_metadata is not None
      assert result.content.parts[0].thought is True
      assert A2A_METADATA_PREFIX + "task_id" in result.custom_metadata
      assert A2A_METADATA_PREFIX + "context_id" in result.custom_metadata

  @pytest.mark.asyncio
  async def test_handle_a2a_response_with_task_status_update_no_message(self):
    """Test handling of a task status update with no message."""
    mock_a2a_task = Mock(spec=A2ATask)
    mock_a2a_task.id = "task-123"

    mock_update = Mock(spec=TaskStatusUpdateEvent)
    mock_update.status = Mock(A2ATaskStatus)
    mock_update.status.state = _compat.TS_COMPLETED
    mock_update.status.message = None

    result = await self.agent._handle_a2a_response(
        (mock_a2a_task, mock_update), self.mock_context
    )

    assert result is None

  @pytest.mark.asyncio
  async def test_handle_a2a_response_with_artifact_update(self):
    """Test successful A2A response handling with artifact update."""
    mock_a2a_task = Mock(spec=A2ATask)
    mock_a2a_task.id = "task-123"
    mock_a2a_task.context_id = "context-123"

    mock_artifact = Mock(spec=Artifact)
    mock_update = Mock(spec=TaskArtifactUpdateEvent)
    mock_update.artifact = mock_artifact
    mock_update.append = False
    mock_update.last_chunk = True

    # Create a proper Event mock that can handle custom_metadata
    mock_event = Event(
        author=self.agent.name,
        invocation_id=self.mock_context.invocation_id,
        branch=self.mock_context.branch,
    )

    with patch.object(
        remote_a2a_agent,
        "convert_a2a_task_to_event",
        autospec=True,
    ) as mock_convert:
      mock_convert.return_value = mock_event

      result = await self.agent._handle_a2a_response(
          (mock_a2a_task, mock_update), self.mock_context
      )

      assert result == mock_event
      mock_convert.assert_called_once_with(
          mock_a2a_task,
          self.agent.name,
          self.mock_context,
          self.agent._a2a_part_converter,
      )
      # Check that metadata was added
      assert result.custom_metadata is not None
      assert A2A_METADATA_PREFIX + "task_id" in result.custom_metadata
      assert A2A_METADATA_PREFIX + "context_id" in result.custom_metadata

  @pytest.mark.asyncio
  async def test_handle_a2a_response_with_partial_artifact_update(self):
    """Test that partial artifact updates are ignored."""
    mock_a2a_task = Mock(spec=A2ATask)
    mock_a2a_task.id = "task-123"

    mock_update = Mock(spec=TaskArtifactUpdateEvent)
    mock_update.artifact = Mock(spec=Artifact)
    mock_update.append = True
    mock_update.last_chunk = False

    result = await self.agent._handle_a2a_response(
        (mock_a2a_task, mock_update), self.mock_context
    )

    assert result is None

  @pytest.mark.asyncio
  async def test_handle_a2a_response_with_real_empty_status_message(self):
    """A real status update without a message must not yield a spurious event."""
    mock_a2a_task = Mock(spec=A2ATask)
    mock_a2a_task.id = "task-123"
    mock_a2a_task.context_id = "context-123"

    # Real status update with NO message attached (empty proto Message on 1.x).
    update = _compat.make_task_status_update_event(
        task_id="task-123",
        context_id="context-123",
        status=_compat.make_task_status(_compat.TS_WORKING),
        final=False,
    )

    with patch(
        "google.adk.agents.remote_a2a_agent.convert_a2a_message_to_event"
    ) as mock_convert:
      result = await self.agent._handle_a2a_response(
          (mock_a2a_task, update), self.mock_context
      )

    # The empty status message must be treated as absent: the converter is not
    # called and the handler produces no spurious event.
    mock_convert.assert_not_called()
    assert result is None


class TestRemoteA2aAgentMessageHandlingFromFactory:
  """Test message handling functionality."""

  def setup_method(self):
    """Setup test fixtures."""
    self.mock_a2a_part_converter = Mock()

    self.agent_card = create_test_agent_card()
    self.agent = RemoteA2aAgent(
        name="test_agent",
        agent_card=self.agent_card,
        a2a_client_factory=ClientFactory(
            config=ClientConfig(httpx_client=httpx.AsyncClient()),
        ),
        a2a_part_converter=self.mock_a2a_part_converter,
    )

    # Mock session and context
    self.mock_session = Mock(spec=Session)
    self.mock_session.id = "session-123"
    self.mock_session.events = []

    self.mock_context = Mock(spec=InvocationContext)
    self.mock_context.session = self.mock_session
    self.mock_context.invocation_id = "invocation-123"
    self.mock_context.branch = "main"

  def test_create_a2a_request_for_user_function_response_no_function_call(self):
    """Test function response request creation when no function call exists."""
    with patch(
        "google.adk.agents.remote_a2a_agent.find_matching_function_call"
    ) as mock_find:
      mock_find.return_value = None

      result = self.agent._create_a2a_request_for_user_function_response(
          self.mock_context
      )

      assert result is None

  def test_create_a2a_request_for_user_function_response_success(self):
    """Test successful function response request creation."""
    # Mock function call event
    mock_function_event = Mock()
    mock_function_event.custom_metadata = {
        A2A_METADATA_PREFIX + "task_id": "task-123"
    }
    mock_function_event.content = Mock()
    mock_function_event.content.parts = [Mock()]
    mock_function_event.get_function_calls.return_value = []

    # Mock latest event with function response - set proper author
    mock_latest_event = Mock()
    mock_latest_event.author = "user"
    self.mock_session.events = [mock_latest_event]

    with patch(
        "google.adk.agents.remote_a2a_agent.find_matching_function_call"
    ) as mock_find:
      mock_find.return_value = mock_function_event

      with patch(
          "google.adk.agents.remote_a2a_agent.convert_event_to_a2a_message"
      ) as mock_convert:
        # Create a proper mock A2A message
        mock_a2a_message = Mock(spec=A2AMessage)
        mock_a2a_message.task_id = None  # Will be set by the method
        mock_convert.return_value = mock_a2a_message

        result = self.agent._create_a2a_request_for_user_function_response(
            self.mock_context
        )

        assert result is not None
        assert result == mock_a2a_message
        assert mock_a2a_message.task_id == "task-123"

  def test_construct_message_parts_from_session_success(self):
    """Test successful message parts construction from session."""
    # Mock event with text content
    mock_part = Mock()
    mock_part.text = "Hello world"

    mock_content = Mock()
    mock_content.parts = [mock_part]

    mock_event = Mock()
    mock_event.content = mock_content

    self.mock_session.events = [mock_event]

    with patch(
        "google.adk.agents.remote_a2a_agent._present_other_agent_message"
    ) as mock_convert:
      mock_convert.return_value = mock_event

      with patch.object(
          self.agent, "_genai_part_converter"
      ) as mock_convert_part:
        mock_a2a_part = Mock()
        mock_convert_part.return_value = mock_a2a_part

        parts, context_id = self.agent._construct_message_parts_from_session(
            self.mock_context
        )

        assert len(parts) == 1
        assert parts[0] == mock_a2a_part
        assert context_id is None

  def test_construct_message_parts_from_session_empty_events(self):
    """Test message parts construction with empty events."""
    self.mock_session.events = []

    parts, context_id = self.agent._construct_message_parts_from_session(
        self.mock_context
    )

    assert parts == []
    assert context_id is None

  @pytest.mark.asyncio
  async def test_handle_a2a_response_success_with_message(self):
    """Test successful A2A response handling with message."""
    mock_a2a_message = Mock(spec=A2AMessage)
    mock_a2a_message.context_id = "context-123"

    # Create a proper Event mock that can handle custom_metadata
    mock_event = Event(
        author=self.agent.name,
        invocation_id=self.mock_context.invocation_id,
        branch=self.mock_context.branch,
    )

    with patch(
        "google.adk.agents.remote_a2a_agent.convert_a2a_message_to_event"
    ) as mock_convert:
      mock_convert.return_value = mock_event

      result = await self.agent._handle_a2a_response(
          mock_a2a_message, self.mock_context
      )

      assert result == mock_event
      mock_convert.assert_called_once_with(
          mock_a2a_message,
          self.agent.name,
          self.mock_context,
          self.mock_a2a_part_converter,
      )
      # Check that metadata was added
      assert result.custom_metadata is not None
      assert A2A_METADATA_PREFIX + "context_id" in result.custom_metadata

  @pytest.mark.asyncio
  async def test_handle_a2a_response_with_task_completed_and_no_update(self):
    """Test successful A2A response handling with non-streaming task and no update."""
    mock_a2a_task = Mock(spec=A2ATask)
    mock_a2a_task.id = "task-123"
    mock_a2a_task.context_id = "context-123"
    mock_a2a_task.status = Mock(spec=A2ATaskStatus)
    mock_a2a_task.status.state = _compat.TS_COMPLETED

    # Create a proper Event mock that can handle custom_metadata
    mock_a2a_part = genai_types.Part.from_text(
        text="test"
    )  # real genai part for Content
    mock_event = Event(
        author=self.agent.name,
        invocation_id=self.mock_context.invocation_id,
        branch=self.mock_context.branch,
        content=genai_types.Content(role="model", parts=[mock_a2a_part]),
    )

    with patch.object(
        remote_a2a_agent,
        "convert_a2a_task_to_event",
        autospec=True,
    ) as mock_convert:
      mock_convert.return_value = mock_event

      result = await self.agent._handle_a2a_response(
          (mock_a2a_task, None), self.mock_context
      )

      assert result == mock_event
      mock_convert.assert_called_once_with(
          mock_a2a_task,
          self.agent.name,
          self.mock_context,
          self.mock_a2a_part_converter,
      )
      # Check the parts are not updated as Thought
      assert result.content.parts[0].thought is None
      # Check that metadata was added
      assert result.custom_metadata is not None
      assert A2A_METADATA_PREFIX + "task_id" in result.custom_metadata
      assert A2A_METADATA_PREFIX + "context_id" in result.custom_metadata

  @pytest.mark.asyncio
  async def test_handle_a2a_response_with_task_submitted_and_no_update(self):
    """Test successful A2A response handling with streaming task and no update."""
    mock_a2a_task = Mock(spec=A2ATask)
    mock_a2a_task.id = "task-123"
    mock_a2a_task.context_id = "context-123"
    mock_a2a_task.status = Mock(spec=A2ATaskStatus)
    mock_a2a_task.status.state = _compat.TS_SUBMITTED

    # Create a proper Event mock that can handle custom_metadata
    mock_a2a_part = genai_types.Part.from_text(
        text="test"
    )  # real genai part for Content
    mock_event = Event(
        author=self.agent.name,
        invocation_id=self.mock_context.invocation_id,
        branch=self.mock_context.branch,
        content=genai_types.Content(role="model", parts=[mock_a2a_part]),
    )

    with patch.object(
        remote_a2a_agent,
        "convert_a2a_task_to_event",
        autospec=True,
    ) as mock_convert:
      mock_convert.return_value = mock_event

      result = await self.agent._handle_a2a_response(
          (mock_a2a_task, None), self.mock_context
      )

      assert result == mock_event
      mock_convert.assert_called_once_with(
          mock_a2a_task,
          self.agent.name,
          self.mock_context,
          self.agent._a2a_part_converter,
      )
      # Check the parts are updated as Thought
      assert result.content.parts[0].thought is True
      assert result.content.parts[0].thought_signature is None
      # Check that metadata was added
      assert result.custom_metadata is not None
      assert A2A_METADATA_PREFIX + "task_id" in result.custom_metadata
      assert A2A_METADATA_PREFIX + "context_id" in result.custom_metadata

  @pytest.mark.asyncio
  async def test_handle_a2a_response_with_task_status_update_with_message(self):
    """Test handling of a task status update with a message."""
    mock_a2a_task = Mock(spec=A2ATask)
    mock_a2a_task.id = "task-123"
    mock_a2a_task.context_id = "context-123"

    mock_a2a_message = Mock(spec=A2AMessage)
    mock_update = Mock(spec=TaskStatusUpdateEvent)
    mock_update.status = Mock(A2ATaskStatus)
    mock_update.status.state = _compat.TS_COMPLETED
    mock_update.status.message = mock_a2a_message

    # Create a proper Event mock that can handle custom_metadata
    mock_a2a_part = genai_types.Part.from_text(
        text="test"
    )  # real genai part for Content
    mock_event = Event(
        author=self.agent.name,
        invocation_id=self.mock_context.invocation_id,
        branch=self.mock_context.branch,
        content=genai_types.Content(role="model", parts=[mock_a2a_part]),
    )

    with patch(
        "google.adk.agents.remote_a2a_agent.convert_a2a_message_to_event"
    ) as mock_convert:
      mock_convert.return_value = mock_event

      result = await self.agent._handle_a2a_response(
          (mock_a2a_task, mock_update), self.mock_context
      )

      assert result == mock_event
      mock_convert.assert_called_once_with(
          mock_a2a_message,
          self.agent.name,
          self.mock_context,
          self.agent._a2a_part_converter,
      )
      # Check that metadata was added
      assert result.custom_metadata is not None
      assert result.content.parts[0].thought is None
      assert A2A_METADATA_PREFIX + "task_id" in result.custom_metadata
      assert A2A_METADATA_PREFIX + "context_id" in result.custom_metadata

  @pytest.mark.asyncio
  async def test_handle_a2a_response_with_task_status_working_update_with_message(
      self,
  ):
    """Test handling of a task status update with a message."""
    mock_a2a_task = Mock(spec=A2ATask)
    mock_a2a_task.id = "task-123"
    mock_a2a_task.context_id = "context-123"

    mock_a2a_message = Mock(spec=A2AMessage)
    mock_update = Mock(spec=TaskStatusUpdateEvent)
    mock_update.status = Mock(A2ATaskStatus)
    mock_update.status.state = _compat.TS_WORKING
    mock_update.status.message = mock_a2a_message

    # Create a proper Event mock that can handle custom_metadata
    mock_a2a_part = genai_types.Part.from_text(
        text="test"
    )  # real genai part for Content
    mock_event = Event(
        author=self.agent.name,
        invocation_id=self.mock_context.invocation_id,
        branch=self.mock_context.branch,
        content=genai_types.Content(role="model", parts=[mock_a2a_part]),
    )

    with patch(
        "google.adk.agents.remote_a2a_agent.convert_a2a_message_to_event"
    ) as mock_convert:
      mock_convert.return_value = mock_event

      result = await self.agent._handle_a2a_response(
          (mock_a2a_task, mock_update), self.mock_context
      )

      assert result == mock_event
      mock_convert.assert_called_once_with(
          mock_a2a_message,
          self.agent.name,
          self.mock_context,
          self.agent._a2a_part_converter,
      )
      # Check that metadata was added
      assert result.custom_metadata is not None
      assert result.content.parts[0].thought is True
      assert A2A_METADATA_PREFIX + "task_id" in result.custom_metadata
      assert A2A_METADATA_PREFIX + "context_id" in result.custom_metadata

  @pytest.mark.asyncio
  async def test_handle_a2a_response_with_task_status_update_no_message(self):
    """Test handling of a task status update with no message."""
    mock_a2a_task = Mock(spec=A2ATask)
    mock_a2a_task.id = "task-123"

    mock_update = Mock(spec=TaskStatusUpdateEvent)
    mock_update.status = Mock(A2ATaskStatus)
    mock_update.status.state = _compat.TS_COMPLETED
    mock_update.status.message = None

    result = await self.agent._handle_a2a_response(
        (mock_a2a_task, mock_update), self.mock_context
    )

    assert result is None

  @pytest.mark.asyncio
  async def test_handle_a2a_response_with_artifact_update(self):
    """Test successful A2A response handling with artifact update."""
    mock_a2a_task = Mock(spec=A2ATask)
    mock_a2a_task.id = "task-123"
    mock_a2a_task.context_id = "context-123"

    mock_artifact = Mock(spec=Artifact)
    mock_update = Mock(spec=TaskArtifactUpdateEvent)
    mock_update.artifact = mock_artifact
    mock_update.append = False
    mock_update.last_chunk = True

    # Create a proper Event mock that can handle custom_metadata
    mock_event = Event(
        author=self.agent.name,
        invocation_id=self.mock_context.invocation_id,
        branch=self.mock_context.branch,
    )

    with patch.object(
        remote_a2a_agent,
        "convert_a2a_task_to_event",
        autospec=True,
    ) as mock_convert:
      mock_convert.return_value = mock_event

      result = await self.agent._handle_a2a_response(
          (mock_a2a_task, mock_update), self.mock_context
      )

      assert result == mock_event
      mock_convert.assert_called_once_with(
          mock_a2a_task,
          self.agent.name,
          self.mock_context,
          self.agent._a2a_part_converter,
      )
      # Check that metadata was added
      assert result.custom_metadata is not None
      assert A2A_METADATA_PREFIX + "task_id" in result.custom_metadata
      assert A2A_METADATA_PREFIX + "context_id" in result.custom_metadata

  @pytest.mark.asyncio
  async def test_handle_a2a_response_with_partial_artifact_update(self):
    """Test that partial artifact updates are ignored."""
    mock_a2a_task = Mock(spec=A2ATask)
    mock_a2a_task.id = "task-123"

    mock_update = Mock(spec=TaskArtifactUpdateEvent)
    mock_update.artifact = Mock(spec=Artifact)
    mock_update.append = True
    mock_update.last_chunk = False

    result = await self.agent._handle_a2a_response(
        (mock_a2a_task, mock_update), self.mock_context
    )

    assert result is None


class TestRemoteA2aAgentMessageHandlingV2:
  """Test _handle_a2a_response_impl functionality."""

  def setup_method(self):
    """Setup test fixtures."""
    from google.adk.a2a.agent.config import A2aRemoteAgentConfig

    self.agent_card = create_test_agent_card()
    self.mock_config = Mock(spec=A2aRemoteAgentConfig)
    self.mock_config.a2a_part_converter = Mock()
    self.mock_config.a2a_task_converter = Mock()
    self.mock_config.a2a_status_update_converter = Mock()
    self.mock_config.a2a_artifact_update_converter = Mock()
    self.mock_config.a2a_message_converter = Mock()

    self.agent = RemoteA2aAgent(
        name="test_agent",
        agent_card=self.agent_card,
        config=self.mock_config,
    )

    # Mock session and context
    self.mock_session = Mock(spec=Session)
    self.mock_session.id = "session-123"
    self.mock_session.events = []

    self.mock_context = Mock(spec=InvocationContext)
    self.mock_context.session = self.mock_session
    self.mock_context.invocation_id = "invocation-123"
    self.mock_context.branch = "main"

  @pytest.mark.asyncio
  async def test_handle_a2a_response_impl_with_message(self):
    """Test _handle_a2a_response_impl with A2AMessage."""
    mock_a2a_message = Mock(spec=A2AMessage)
    mock_a2a_message.metadata = {}
    mock_a2a_message.metadata = {}
    mock_a2a_message.context_id = "context-123"

    mock_event = Event(
        author=self.agent.name,
        invocation_id=self.mock_context.invocation_id,
        branch=self.mock_context.branch,
    )
    self.mock_config.a2a_message_converter.return_value = mock_event

    result = await self.agent._handle_a2a_response_v2(
        mock_a2a_message, self.mock_context
    )

    assert result == mock_event
    self.mock_config.a2a_message_converter.assert_called_once_with(
        mock_a2a_message,
        self.agent.name,
        self.mock_context,
        self.mock_config.a2a_part_converter,
    )
    assert result.custom_metadata is not None
    assert A2A_METADATA_PREFIX + "context_id" in result.custom_metadata
    assert (
        result.custom_metadata[A2A_METADATA_PREFIX + "context_id"]
        == "context-123"
    )

  @pytest.mark.asyncio
  async def test_handle_a2a_response_impl_with_task_and_no_update(self):
    """Test _handle_a2a_response_impl with Task and no update."""
    mock_a2a_task = Mock(spec=A2ATask)
    mock_a2a_task.id = "task-123"
    mock_a2a_task.context_id = "context-123"

    mock_event = Event(
        author=self.agent.name,
        invocation_id=self.mock_context.invocation_id,
        branch=self.mock_context.branch,
    )
    self.mock_config.a2a_task_converter.return_value = mock_event

    result = await self.agent._handle_a2a_response_v2(
        (mock_a2a_task, None), self.mock_context
    )

    assert result == mock_event
    self.mock_config.a2a_task_converter.assert_called_once_with(
        mock_a2a_task,
        self.agent.name,
        self.mock_context,
        self.mock_config.a2a_part_converter,
    )
    assert result.custom_metadata is not None
    assert A2A_METADATA_PREFIX + "task_id" in result.custom_metadata
    assert result.custom_metadata[A2A_METADATA_PREFIX + "task_id"] == "task-123"
    assert A2A_METADATA_PREFIX + "context_id" in result.custom_metadata
    assert (
        result.custom_metadata[A2A_METADATA_PREFIX + "context_id"]
        == "context-123"
    )

  @pytest.mark.asyncio
  async def test_handle_a2a_response_impl_with_task_status_update(self):
    """Test _handle_a2a_response_impl with TaskStatusUpdateEvent."""
    mock_a2a_task = Mock(spec=A2ATask)
    mock_a2a_task.id = "task-123"
    mock_a2a_task.context_id = None

    mock_update = Mock(spec=TaskStatusUpdateEvent)

    mock_event = Event(
        author=self.agent.name,
        invocation_id=self.mock_context.invocation_id,
        branch=self.mock_context.branch,
    )
    self.mock_config.a2a_status_update_converter.return_value = mock_event

    result = await self.agent._handle_a2a_response_v2(
        (mock_a2a_task, mock_update), self.mock_context
    )

    assert result == mock_event
    self.mock_config.a2a_status_update_converter.assert_called_once_with(
        mock_update,
        self.agent.name,
        self.mock_context,
        self.mock_config.a2a_part_converter,
    )
    assert result.custom_metadata is not None
    assert A2A_METADATA_PREFIX + "task_id" in result.custom_metadata
    assert result.custom_metadata[A2A_METADATA_PREFIX + "task_id"] == "task-123"
    assert A2A_METADATA_PREFIX + "context_id" not in result.custom_metadata

  @pytest.mark.asyncio
  async def test_handle_a2a_response_impl_with_task_artifact_update(self):
    """Test _handle_a2a_response_impl with TaskArtifactUpdateEvent."""
    mock_a2a_task = Mock(spec=A2ATask)
    mock_a2a_task.id = "task-123"
    mock_a2a_task.context_id = "context-123"

    mock_update = Mock(spec=TaskArtifactUpdateEvent)

    mock_event = Event(
        author=self.agent.name,
        invocation_id=self.mock_context.invocation_id,
        branch=self.mock_context.branch,
    )
    self.mock_config.a2a_artifact_update_converter.return_value = mock_event

    result = await self.agent._handle_a2a_response_v2(
        (mock_a2a_task, mock_update), self.mock_context
    )

    assert result == mock_event
    self.mock_config.a2a_artifact_update_converter.assert_called_once_with(
        mock_update,
        self.agent.name,
        self.mock_context,
        self.mock_config.a2a_part_converter,
    )
    assert result.custom_metadata is not None
    assert A2A_METADATA_PREFIX + "task_id" in result.custom_metadata
    assert result.custom_metadata[A2A_METADATA_PREFIX + "task_id"] == "task-123"
    assert A2A_METADATA_PREFIX + "context_id" in result.custom_metadata
    assert (
        result.custom_metadata[A2A_METADATA_PREFIX + "context_id"]
        == "context-123"
    )

  @pytest.mark.asyncio
  async def test_handle_a2a_response_impl_update_converter_returns_none(self):
    """Test _handle_a2a_response_impl when converter returns None."""
    mock_a2a_task = Mock(spec=A2ATask)
    mock_a2a_task.id = "task-123"

    mock_update = Mock(spec=TaskArtifactUpdateEvent)

    self.mock_config.a2a_artifact_update_converter.return_value = None

    result = await self.agent._handle_a2a_response_v2(
        (mock_a2a_task, mock_update), self.mock_context
    )

    assert result is None
    self.mock_config.a2a_artifact_update_converter.assert_called_once_with(
        mock_update,
        self.agent.name,
        self.mock_context,
        self.mock_config.a2a_part_converter,
    )

  @pytest.mark.asyncio
  async def test_handle_a2a_response_impl_message_converter_returns_none(self):
    """Test _handle_a2a_response_v2 returns None when message converter returns None."""
    mock_a2a_message = Mock(spec=A2AMessage)
    mock_a2a_message.metadata = {}
    mock_a2a_message.context_id = "context-123"

    self.mock_config.a2a_message_converter.return_value = None

    result = await self.agent._handle_a2a_response_v2(
        mock_a2a_message, self.mock_context
    )

    assert result is None
    self.mock_config.a2a_message_converter.assert_called_once_with(
        mock_a2a_message,
        self.agent.name,
        self.mock_context,
        self.mock_config.a2a_part_converter,
    )

  @pytest.mark.asyncio
  async def test_handle_a2a_response_impl_unknown_response_type(self):
    """Test _handle_a2a_response_impl with unknown response type."""
    unknown_response = object()

    result = await self.agent._handle_a2a_response_v2(
        unknown_response, self.mock_context
    )

    assert result is not None
    assert result.author == self.agent.name
    assert result.error_message == "Unknown A2A response type"
    assert result.invocation_id == self.mock_context.invocation_id
    assert result.branch == self.mock_context.branch

  @pytest.mark.asyncio
  async def test_handle_a2a_response_impl_handles_client_error(self):
    """Test _handle_a2a_response_impl catches A2AClientError."""
    mock_a2a_message = Mock(spec=A2AMessage)
    mock_a2a_message.metadata = {}
    mock_a2a_message.metadata = {}

    from google.adk.agents.remote_a2a_agent import A2AClientError

    self.mock_config.a2a_message_converter.side_effect = A2AClientError(
        "Test client error"
    )

    result = await self.agent._handle_a2a_response_v2(
        mock_a2a_message, self.mock_context
    )

    assert result is not None
    assert result.author == self.agent.name
    assert (
        "Failed to process A2A response: Test client error"
        in result.error_message
    )
    assert result.invocation_id == self.mock_context.invocation_id
    assert result.branch == self.mock_context.branch


class TestRemoteA2aAgentNoneConverterResults:
  """Regression tests for None converter results in both legacy and v2 handlers.

  Converters can legitimately return None for messages/tasks with no convertible
  parts, metadata-only events, or empty status updates. The handlers must not
  crash with AttributeError when this happens.
  """

  def setup_method(self):
    """Setup test fixtures."""
    from google.adk.a2a.agent.config import A2aRemoteAgentConfig

    self.agent_card = create_test_agent_card()

    # Legacy handler agent
    self.mock_a2a_part_converter = Mock()
    self.legacy_agent = RemoteA2aAgent(
        name="test_agent",
        agent_card=self.agent_card,
        a2a_part_converter=self.mock_a2a_part_converter,
    )

    # V2 handler agent
    self.mock_config = Mock(spec=A2aRemoteAgentConfig)
    self.mock_config.a2a_part_converter = Mock()
    self.mock_config.a2a_task_converter = Mock()
    self.mock_config.a2a_status_update_converter = Mock()
    self.mock_config.a2a_artifact_update_converter = Mock()
    self.mock_config.a2a_message_converter = Mock()
    self.mock_config.request_interceptors = None
    self.v2_agent = RemoteA2aAgent(
        name="test_agent",
        agent_card=self.agent_card,
        config=self.mock_config,
    )

    # Shared mock context
    self.mock_session = Mock(spec=Session)
    self.mock_session.id = "session-123"
    self.mock_session.events = []

    self.mock_context = Mock(spec=InvocationContext)
    self.mock_context.session = self.mock_session
    self.mock_context.invocation_id = "invocation-123"
    self.mock_context.branch = "main"

  # --- V2 handler regression tests ---

  @pytest.mark.asyncio
  async def test_v2_message_converter_returns_none(self):
    """V2 handler must not crash when message converter returns None."""
    mock_msg = Mock(spec=A2AMessage)
    mock_msg.metadata = {}
    mock_msg.context_id = None

    self.mock_config.a2a_message_converter.return_value = None

    result = await self.v2_agent._handle_a2a_response_v2(
        mock_msg, self.mock_context
    )

    assert result is None
    self.mock_config.a2a_message_converter.assert_called_once()

  @pytest.mark.asyncio
  async def test_v2_message_converter_returns_none_with_context_id(self):
    """V2 handler returns None even when message has a context_id."""
    mock_msg = Mock(spec=A2AMessage)
    mock_msg.metadata = {}
    mock_msg.context_id = "ctx-should-not-be-accessed"

    self.mock_config.a2a_message_converter.return_value = None

    result = await self.v2_agent._handle_a2a_response_v2(
        mock_msg, self.mock_context
    )

    assert result is None

  @pytest.mark.asyncio
  async def test_v2_task_converter_returns_none(self):
    """V2 handler must not crash when task converter returns None."""
    mock_task = Mock(spec=A2ATask)
    mock_task.id = "task-123"
    mock_task.context_id = "ctx-123"

    self.mock_config.a2a_task_converter.return_value = None

    result = await self.v2_agent._handle_a2a_response_v2(
        (mock_task, None), self.mock_context
    )

    assert result is None

  @pytest.mark.asyncio
  async def test_v2_status_update_converter_returns_none(self):
    """V2 handler must not crash when status update converter returns None."""
    mock_task = Mock(spec=A2ATask)
    mock_task.id = "task-123"
    mock_task.context_id = None

    mock_update = Mock(spec=TaskStatusUpdateEvent)

    self.mock_config.a2a_status_update_converter.return_value = None

    result = await self.v2_agent._handle_a2a_response_v2(
        (mock_task, mock_update), self.mock_context
    )

    assert result is None

  # --- Legacy handler regression tests ---

  @pytest.mark.asyncio
  async def test_legacy_message_converter_returns_none(self):
    """Legacy handler must not crash when message converter returns None."""
    mock_msg = Mock(spec=A2AMessage)
    mock_msg.context_id = "context-123"

    with patch(
        "google.adk.agents.remote_a2a_agent.convert_a2a_message_to_event"
    ) as mock_convert:
      mock_convert.return_value = None

      result = await self.legacy_agent._handle_a2a_response(
          mock_msg, self.mock_context
      )

      assert result is None
      mock_convert.assert_called_once()

  @pytest.mark.asyncio
  async def test_legacy_task_converter_returns_none_no_update(self):
    """Legacy handler must not crash when task converter returns None (no update)."""
    mock_task = Mock(spec=A2ATask)
    mock_task.id = "task-123"
    mock_task.context_id = None
    mock_task.status = Mock()
    mock_task.status.state = _compat.TS_COMPLETED

    with patch(
        "google.adk.agents.remote_a2a_agent.convert_a2a_task_to_event"
    ) as mock_convert:
      mock_convert.return_value = None

      result = await self.legacy_agent._handle_a2a_response(
          (mock_task, None), self.mock_context
      )

      assert result is None

  @pytest.mark.asyncio
  async def test_legacy_message_converter_returns_none_status_update(self):
    """Legacy handler must not crash when message converter returns None for status update."""
    mock_task = Mock(spec=A2ATask)
    mock_task.id = "task-123"
    mock_task.context_id = "ctx-123"

    mock_update = Mock(spec=TaskStatusUpdateEvent)
    mock_update.status = Mock()
    mock_update.status.message = Mock()
    mock_update.status.state = _compat.TS_WORKING

    with patch(
        "google.adk.agents.remote_a2a_agent.convert_a2a_message_to_event"
    ) as mock_convert:
      mock_convert.return_value = None

      result = await self.legacy_agent._handle_a2a_response(
          (mock_task, mock_update), self.mock_context
      )

      assert result is None

  @pytest.mark.asyncio
  async def test_legacy_task_converter_returns_none_artifact_update(self):
    """Legacy handler must not crash when task converter returns None for artifact update."""
    mock_task = Mock(spec=A2ATask)
    mock_task.id = "task-123"
    mock_task.context_id = None

    mock_update = Mock(spec=TaskArtifactUpdateEvent)
    mock_update.append = False
    mock_update.last_chunk = True

    with patch(
        "google.adk.agents.remote_a2a_agent.convert_a2a_task_to_event"
    ) as mock_convert:
      mock_convert.return_value = None

      result = await self.legacy_agent._handle_a2a_response(
          (mock_task, mock_update), self.mock_context
      )

      assert result is None


class TestRemoteA2aAgentExecution:
  """Test agent execution functionality."""

  def setup_method(self):
    """Setup test fixtures."""
    self.agent_card = create_test_agent_card()
    self.mock_genai_part_converter = Mock()
    self.mock_a2a_part_converter = Mock()
    self.agent = RemoteA2aAgent(
        name="test_agent",
        agent_card=self.agent_card,
        genai_part_converter=self.mock_genai_part_converter,
        a2a_part_converter=self.mock_a2a_part_converter,
    )

    # Mock session and context
    self.mock_session = Mock(spec=Session)
    self.mock_session.id = "session-123"
    self.mock_session.events = []
    self.mock_session.state = {}

    self.mock_context = Mock(spec=InvocationContext)
    self.mock_context.session = self.mock_session
    self.mock_context.invocation_id = "invocation-123"
    self.mock_context.branch = "main"

  @pytest.mark.asyncio
  async def test_run_async_impl_initialization_failure(self):
    """Test _run_async_impl when initialization fails."""
    with patch.object(self.agent, "_ensure_resolved") as mock_ensure:
      mock_ensure.side_effect = Exception("Initialization failed")

      events = []
      async for event in self.agent._run_async_impl(self.mock_context):
        events.append(event)

      assert len(events) == 1
      assert "Failed to initialize remote A2A agent" in events[0].error_message

  @pytest.mark.asyncio
  async def test_run_async_impl_no_message_parts(self):
    """Test _run_async_impl when no message parts are found."""
    with patch.object(self.agent, "_ensure_resolved"):
      with patch.object(
          self.agent, "_create_a2a_request_for_user_function_response"
      ) as mock_create_func:
        mock_create_func.return_value = None

        with patch.object(
            self.agent, "_construct_message_parts_from_session"
        ) as mock_construct:
          mock_construct.return_value = (
              [],
              None,
          )  # Tuple with empty parts and no context_id

          events = []
          async for event in self.agent._run_async_impl(self.mock_context):
            events.append(event)

          assert len(events) == 1
          assert events[0].content is not None
          assert events[0].author == self.agent.name

  @pytest.mark.asyncio
  async def test_run_async_impl_successful_request(self):
    """Test successful _run_async_impl execution."""
    with patch.object(self.agent, "_ensure_resolved"):
      with patch.object(
          self.agent, "_create_a2a_request_for_user_function_response"
      ) as mock_create_func:
        mock_create_func.return_value = None

        with patch.object(
            self.agent, "_construct_message_parts_from_session"
        ) as mock_construct:
          # Use a real A2A text part so production builds a real
          # A2A message that the 1.x send_message adapter can
          # serialize.
          mock_a2a_part = _compat.make_text_part("test")
          mock_construct.return_value = (
              [mock_a2a_part],
              "context-123",
          )  # Tuple with parts and context_id

          # Mock A2A client. Build the raw stream item version-
          # correctly (real ``StreamResponse`` on 1.x, bare
          # ``Message`` on 0.3.x) so the stream normalizer handles
          # it; the dispatch itself is mocked below.
          mock_a2a_client = create_autospec(spec=A2AClient, instance=True)
          mock_response = _make_stream_message(
              A2AMessage(
                  message_id="m1",
                  role=_compat.ROLE_USER,
                  parts=[mock_a2a_part],
              )
          )
          mock_send_message = AsyncMock()
          mock_send_message.__aiter__.return_value = [mock_response]
          mock_a2a_client.send_message.return_value = mock_send_message
          self.agent._a2a_client = mock_a2a_client

          mock_event = Event(
              author=self.agent.name,
              invocation_id=self.mock_context.invocation_id,
              branch=self.mock_context.branch,
          )

          with patch.object(self.agent, "_handle_a2a_response") as mock_handle:
            mock_handle.return_value = mock_event

            # Mock the logging functions to avoid iteration issues
            with patch(
                "google.adk.agents.remote_a2a_agent.build_a2a_request_log"
            ) as mock_req_log:
              with patch(
                  "google.adk.agents.remote_a2a_agent.build_a2a_response_log"
              ) as mock_resp_log:
                mock_req_log.return_value = "Mock request log"
                mock_resp_log.return_value = "Mock response log"

                # Patch the production serializer so metadata
                # stamping does not run MessageToDict on the
                # mock response (which crashes on 1.x).
                with patch(
                    "google.adk.a2a._compat.a2a_to_dict",
                    return_value={"k": "v"},
                ):
                  # Execute
                  events = []
                  async for event in self.agent._run_async_impl(
                      self.mock_context
                  ):
                    events.append(event)

                assert len(events) == 1
                assert events[0] == mock_event
                assert (
                    A2A_METADATA_PREFIX + "request"
                    in mock_event.custom_metadata
                )

  @pytest.mark.asyncio
  async def test_run_async_impl_a2a_client_error(self):
    """Test _run_async_impl when A2A send_message fails."""
    with patch.object(self.agent, "_ensure_resolved"):
      with patch.object(
          self.agent, "_create_a2a_request_for_user_function_response"
      ) as mock_create_func:
        mock_create_func.return_value = None

        with patch.object(
            self.agent, "_construct_message_parts_from_session"
        ) as mock_construct:
          # Return a real A2A part so the request Message builds and can
          # be serialized in the error path on both SDK versions.
          mock_a2a_part = _compat.make_text_part("test")
          mock_construct.return_value = (
              [mock_a2a_part],
              "context-123",
          )  # Tuple with parts and context_id

          # Mock A2A client that throws an exception
          mock_a2a_client = AsyncMock()
          mock_a2a_client.send_message.side_effect = Exception("Send failed")
          self.agent._a2a_client = mock_a2a_client

          # Mock the logging functions to avoid iteration issues
          with patch(
              "google.adk.agents.remote_a2a_agent.build_a2a_request_log"
          ) as mock_req_log:
            mock_req_log.return_value = "Mock request log"

            events = []
            async for event in self.agent._run_async_impl(self.mock_context):
              events.append(event)

            assert len(events) == 1
            assert "A2A request failed" in events[0].error_message

  @pytest.mark.asyncio
  async def test_run_live_impl_not_implemented(self):
    """Test that _run_live_impl raises NotImplementedError."""
    with pytest.raises(
        NotImplementedError, match="_run_live_impl.*not implemented"
    ):
      async for _ in self.agent._run_live_impl(self.mock_context):
        pass

  @pytest.mark.asyncio
  async def test_run_async_impl_with_meta_provider(self):
    """Test _run_async_impl with a2a_request_meta_provider."""
    mock_meta_provider = Mock()
    request_metadata = {"custom_meta": "value"}
    mock_meta_provider.return_value = request_metadata
    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card=self.agent_card,
        genai_part_converter=self.mock_genai_part_converter,
        a2a_part_converter=self.mock_a2a_part_converter,
        a2a_request_meta_provider=mock_meta_provider,
    )

    with patch.object(agent, "_ensure_resolved"):
      with patch.object(
          agent, "_create_a2a_request_for_user_function_response"
      ) as mock_create_func:
        mock_create_func.return_value = None

        with patch.object(
            agent, "_construct_message_parts_from_session"
        ) as mock_construct:
          # Use a real A2A text part so production builds a real
          # A2A message that the 1.x send_message adapter can
          # serialize (it CopyFrom()s the message into a proto).
          mock_a2a_part = _compat.make_text_part("test")
          mock_construct.return_value = (
              [mock_a2a_part],
              "context-123",
          )  # Tuple with parts and context_id

          # Mock A2A client. The raw stream item is built
          # version-correctly (a real ``StreamResponse`` on 1.x, a
          # bare ``Message`` on 0.3.x) so production's
          # the stream normalizer handles it; the dispatch itself
          # is mocked via ``_handle_a2a_response`` below.
          mock_a2a_client = create_autospec(spec=A2AClient, instance=True)
          mock_response = _make_stream_message(
              A2AMessage(
                  message_id="m1",
                  role=_compat.ROLE_USER,
                  parts=[mock_a2a_part],
              )
          )
          mock_send_message = AsyncMock()
          mock_send_message.__aiter__.return_value = [mock_response]
          mock_a2a_client.send_message.return_value = mock_send_message
          # Use the locally-created ``agent`` (the one with the
          # meta_provider and the patched _create/_construct), not
          # ``self.agent``.
          agent._a2a_client = mock_a2a_client

          mock_event = Event(
              author=agent.name,
              invocation_id=self.mock_context.invocation_id,
              branch=self.mock_context.branch,
          )

          with patch.object(agent, "_handle_a2a_response") as mock_handle:
            mock_handle.return_value = mock_event

            with patch(
                "google.adk.agents.remote_a2a_agent.build_a2a_request_log"
            ) as mock_req_log:
              with patch(
                  "google.adk.agents.remote_a2a_agent.build_a2a_response_log"
              ) as mock_resp_log:
                mock_req_log.return_value = "Mock request log"
                mock_resp_log.return_value = "Mock response log"

                with patch(
                    "google.adk.a2a._compat.a2a_to_dict",
                    return_value={"k": "v"},
                ):
                  events = []
                  async for event in agent._run_async_impl(self.mock_context):
                    events.append(event)

                assert len(events) == 1
                assert events[0] == mock_event
                assert (
                    A2A_METADATA_PREFIX + "request"
                    in mock_event.custom_metadata
                )


class TestRemoteA2aAgentExecutionFromFactory:
  """Test agent execution functionality (factory-constructed client)."""

  def setup_method(self):
    """Setup test fixtures."""
    self.agent_card = create_test_agent_card()
    self.agent = RemoteA2aAgent(
        name="test_agent",
        agent_card=self.agent_card,
        a2a_client_factory=ClientFactory(
            config=ClientConfig(httpx_client=httpx.AsyncClient()),
        ),
    )

    # Mock session and context
    self.mock_session = Mock(spec=Session)
    self.mock_session.id = "session-123"
    self.mock_session.events = []
    self.mock_session.state = {}

    self.mock_context = Mock(spec=InvocationContext)
    self.mock_context.session = self.mock_session
    self.mock_context.invocation_id = "invocation-123"
    self.mock_context.branch = "main"

  @pytest.mark.asyncio
  async def test_run_async_impl_initialization_failure(self):
    """Test _run_async_impl when initialization fails."""
    with patch.object(self.agent, "_ensure_resolved") as mock_ensure:
      mock_ensure.side_effect = Exception("Initialization failed")

      events = []
      async for event in self.agent._run_async_impl(self.mock_context):
        events.append(event)

      assert len(events) == 1
      assert "Failed to initialize remote A2A agent" in events[0].error_message

  @pytest.mark.asyncio
  async def test_run_async_impl_no_message_parts(self):
    """Test _run_async_impl when no message parts are found."""
    with patch.object(self.agent, "_ensure_resolved"):
      with patch.object(
          self.agent, "_create_a2a_request_for_user_function_response"
      ) as mock_create_func:
        mock_create_func.return_value = None

        with patch.object(
            self.agent, "_construct_message_parts_from_session"
        ) as mock_construct:
          mock_construct.return_value = (
              [],
              None,
          )  # Tuple with empty parts and no context_id

          events = []
          async for event in self.agent._run_async_impl(self.mock_context):
            events.append(event)

          assert len(events) == 1
          assert events[0].content is not None
          assert events[0].author == self.agent.name

  @pytest.mark.asyncio
  async def test_run_async_impl_successful_request(self):
    """Test successful _run_async_impl execution."""
    with patch.object(self.agent, "_ensure_resolved"):
      with patch.object(
          self.agent, "_create_a2a_request_for_user_function_response"
      ) as mock_create_func:
        mock_create_func.return_value = None

        with patch.object(
            self.agent, "_construct_message_parts_from_session"
        ) as mock_construct:
          # Use a real A2A text part so production builds a real
          # A2A message that the 1.x send_message adapter can
          # serialize.
          mock_a2a_part = _compat.make_text_part("test")
          mock_construct.return_value = (
              [mock_a2a_part],
              "context-123",
          )  # Tuple with parts and context_id

          # Build the raw stream item version-correctly (real
          # StreamResponse on 1.x, bare Message on 0.3.x) so
          # the stream normalizer handles it; the dispatch is mocked.
          mock_a2a_client = create_autospec(spec=A2AClient, instance=True)
          mock_response = _make_stream_message(
              A2AMessage(
                  message_id="m1",
                  role=_compat.ROLE_USER,
                  parts=[mock_a2a_part],
              )
          )
          mock_send_message = AsyncMock()
          mock_send_message.__aiter__.return_value = [mock_response]
          mock_a2a_client.send_message.return_value = mock_send_message
          self.agent._a2a_client = mock_a2a_client

          mock_event = Event(
              author=self.agent.name,
              invocation_id=self.mock_context.invocation_id,
              branch=self.mock_context.branch,
          )

          with patch.object(self.agent, "_handle_a2a_response") as mock_handle:
            mock_handle.return_value = mock_event

            with patch(
                "google.adk.agents.remote_a2a_agent.build_a2a_request_log"
            ) as mock_req_log:
              with patch(
                  "google.adk.agents.remote_a2a_agent.build_a2a_response_log"
              ) as mock_resp_log:
                mock_req_log.return_value = "Mock request log"
                mock_resp_log.return_value = "Mock response log"

                with patch(
                    "google.adk.a2a._compat.a2a_to_dict",
                    return_value={"k": "v"},
                ):
                  events = []
                  async for event in self.agent._run_async_impl(
                      self.mock_context
                  ):
                    events.append(event)

                assert len(events) == 1
                assert events[0] == mock_event
                assert (
                    A2A_METADATA_PREFIX + "request"
                    in mock_event.custom_metadata
                )

  @pytest.mark.asyncio
  async def test_run_async_impl_a2a_client_error(self):
    """Test _run_async_impl when A2A send_message fails."""
    with patch.object(self.agent, "_ensure_resolved"):
      with patch.object(
          self.agent, "_create_a2a_request_for_user_function_response"
      ) as mock_create_func:
        mock_create_func.return_value = None

        with patch.object(
            self.agent, "_construct_message_parts_from_session"
        ) as mock_construct:
          # Return a real A2A part so the request Message builds and can
          # be serialized in the error path on both SDK versions.
          mock_a2a_part = _compat.make_text_part("test")
          mock_construct.return_value = (
              [mock_a2a_part],
              "context-123",
          )  # Tuple with parts and context_id

          # Mock A2A client that throws an exception
          mock_a2a_client = AsyncMock()
          mock_a2a_client.send_message.side_effect = Exception("Send failed")
          self.agent._a2a_client = mock_a2a_client

          # Mock the logging functions to avoid iteration issues
          with patch(
              "google.adk.agents.remote_a2a_agent.build_a2a_request_log"
          ) as mock_req_log:
            mock_req_log.return_value = "Mock request log"

            events = []
            async for event in self.agent._run_async_impl(self.mock_context):
              events.append(event)

            assert len(events) == 1
            assert "A2A request failed" in events[0].error_message

  @pytest.mark.asyncio
  async def test_run_live_impl_not_implemented(self):
    """Test that _run_live_impl raises NotImplementedError."""
    with pytest.raises(
        NotImplementedError, match="_run_live_impl.*not implemented"
    ):
      async for _ in self.agent._run_live_impl(self.mock_context):
        pass


class TestRemoteA2aAgentCleanup:
  """Test cleanup functionality."""

  def setup_method(self):
    """Setup test fixtures."""
    self.agent_card = create_test_agent_card()

  @pytest.mark.asyncio
  async def test_cleanup_owns_httpx_client(self):
    """Test cleanup when agent owns httpx client."""
    agent = RemoteA2aAgent(name="test_agent", agent_card=self.agent_card)

    # Set up owned client
    mock_client = AsyncMock()
    agent._httpx_client = mock_client
    agent._httpx_client_needs_cleanup = True

    await agent.cleanup()

    mock_client.aclose.assert_called_once()
    assert agent._httpx_client is None

  @pytest.mark.asyncio
  async def test_cleanup_owns_httpx_client_factory(self):
    """Test cleanup when agent owns httpx client."""
    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card=self.agent_card,
        a2a_client_factory=ClientFactory(config=ClientConfig()),
    )

    # Set up owned client
    mock_client = AsyncMock()
    agent._httpx_client = mock_client
    agent._httpx_client_needs_cleanup = True

    await agent.cleanup()

    mock_client.aclose.assert_called_once()
    assert agent._httpx_client is None

  @pytest.mark.asyncio
  async def test_cleanup_does_not_own_httpx_client(self):
    """Test cleanup when agent does not own httpx client."""
    shared_client = AsyncMock()
    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card=self.agent_card,
        httpx_client=shared_client,
    )

    await agent.cleanup()

    # Should not close shared client
    shared_client.aclose.assert_not_called()

  @pytest.mark.asyncio
  async def test_cleanup_does_not_own_httpx_client_factory(self):
    """Test cleanup when agent does not own httpx client."""
    shared_client = AsyncMock()
    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card=self.agent_card,
        a2a_client_factory=ClientFactory(
            config=ClientConfig(httpx_client=shared_client)
        ),
    )

    await agent.cleanup()

    # Should not close shared client
    shared_client.aclose.assert_not_called()

  @pytest.mark.asyncio
  async def test_cleanup_client_close_error(self):
    """Test cleanup when client close raises error."""
    agent = RemoteA2aAgent(name="test_agent", agent_card=self.agent_card)

    mock_client = AsyncMock()
    mock_client.aclose.side_effect = Exception("Close failed")
    agent._httpx_client = mock_client
    agent._httpx_client_needs_cleanup = True

    # Should not raise exception
    await agent.cleanup()
    assert agent._httpx_client is None


class TestRemoteA2aAgentIntegration:
  """Integration tests for RemoteA2aAgent."""

  @pytest.mark.asyncio
  async def test_full_workflow_with_direct_agent_card(self):
    """Test full workflow with direct agent card."""
    agent_card = create_test_agent_card()

    agent = RemoteA2aAgent(name="test_agent", agent_card=agent_card)

    # Use a real genai Part for the session content. The instance's
    # part converter is bound at construction to the real
    # convert_genai_part_to_a2a_part (the module-level patch below does
    # not rebind it), so a real Part is needed to produce a serializable
    # A2A part on both SDK versions.
    mock_part = genai_types.Part.from_text(text="Hello world")

    mock_content = Mock()
    mock_content.parts = [mock_part]

    mock_event = Mock()
    mock_event.content = mock_content

    mock_session = Mock(spec=Session)
    mock_session.id = "session-123"
    mock_session.events = [mock_event]
    mock_session.state = {}

    mock_context = Mock(spec=InvocationContext)
    mock_context.session = mock_session
    mock_context.invocation_id = "invocation-123"
    mock_context.branch = "main"

    # Mock dependencies
    with patch(
        "google.adk.agents.remote_a2a_agent._present_other_agent_message"
    ) as mock_convert:
      mock_convert.return_value = mock_event

      with patch(
          "google.adk.agents.remote_a2a_agent.convert_genai_part_to_a2a_part"
      ) as mock_convert_part:
        # Return a real A2A text part so production builds a real
        # A2A message that the 1.x send_message adapter can serialize.
        mock_a2a_part = _compat.make_text_part("test")
        mock_convert_part.return_value = mock_a2a_part

        with patch("httpx.AsyncClient") as mock_httpx_client_class:
          mock_httpx_client = AsyncMock()
          mock_httpx_client_class.return_value = mock_httpx_client

          with patch.object(agent, "_a2a_client") as mock_a2a_client:
            # Build a real message (wrapped in a StreamResponse on
            # 1.x) so production's stream normalizer /
            # dispatch treat it as a message; the conversion itself
            # is mocked below.
            mock_a2a_message = A2AMessage(
                message_id="m1",
                role=_compat.ROLE_USER,
                parts=[mock_a2a_part],
                context_id="context-123",
            )
            mock_response = _make_stream_message(mock_a2a_message)

            mock_send_message = AsyncMock()
            mock_send_message.__aiter__.return_value = [mock_response]
            mock_a2a_client.send_message.return_value = mock_send_message

            with patch(
                "google.adk.agents.remote_a2a_agent.convert_a2a_message_to_event"
            ) as mock_convert_event:
              mock_result_event = Event(
                  author=agent.name,
                  invocation_id=mock_context.invocation_id,
                  branch=mock_context.branch,
              )
              mock_convert_event.return_value = mock_result_event

              # Mock the logging functions to avoid iteration issues
              with patch(
                  "google.adk.agents.remote_a2a_agent.build_a2a_request_log"
              ) as mock_req_log:
                with patch(
                    "google.adk.agents.remote_a2a_agent.build_a2a_response_log"
                ) as mock_resp_log:
                  mock_req_log.return_value = "Mock request log"
                  mock_resp_log.return_value = "Mock response log"

                  # Patch the production serializer so metadata
                  # stamping does not run MessageToDict on the
                  # mock response (which crashes on 1.x).
                  with patch(
                      "google.adk.a2a._compat.a2a_to_dict",
                      return_value={"k": "v"},
                  ):
                    # Execute
                    events = []
                    async for event in agent._run_async_impl(mock_context):
                      events.append(event)

                  assert len(events) == 1
                  assert events[0] == mock_result_event
                  assert (
                      A2A_METADATA_PREFIX + "request"
                      in mock_result_event.custom_metadata
                  )

                  # Verify A2A client was called
                  mock_a2a_client.send_message.assert_called_once()

  @pytest.mark.asyncio
  async def test_full_workflow_with_direct_agent_card_and_factory(self):
    """Test full workflow with direct agent card."""
    agent_card = create_test_agent_card()

    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card=agent_card,
        a2a_client_factory=ClientFactory(config=ClientConfig()),
    )

    # Use a real genai Part for the session content. The instance's
    # part converter is bound at construction to the real
    # convert_genai_part_to_a2a_part (the module-level patch below does
    # not rebind it), so a real Part is needed to produce a serializable
    # A2A part on both SDK versions.
    mock_part = genai_types.Part.from_text(text="Hello world")

    mock_content = Mock()
    mock_content.parts = [mock_part]

    mock_event = Mock()
    mock_event.content = mock_content

    mock_session = Mock(spec=Session)
    mock_session.id = "session-123"
    mock_session.events = [mock_event]
    mock_session.state = {}

    mock_context = Mock(spec=InvocationContext)
    mock_context.session = mock_session
    mock_context.invocation_id = "invocation-123"
    mock_context.branch = "main"

    # Mock dependencies
    with patch(
        "google.adk.agents.remote_a2a_agent._present_other_agent_message"
    ) as mock_convert:
      mock_convert.return_value = mock_event

      with patch(
          "google.adk.agents.remote_a2a_agent.convert_genai_part_to_a2a_part"
      ) as mock_convert_part:
        # Return a real A2A text part so production builds a real
        # A2A message that the 1.x send_message adapter can serialize.
        mock_a2a_part = _compat.make_text_part("test")
        mock_convert_part.return_value = mock_a2a_part

        with patch("httpx.AsyncClient") as mock_httpx_client_class:
          mock_httpx_client = AsyncMock()
          mock_httpx_client_class.return_value = mock_httpx_client

          with patch.object(agent, "_a2a_client") as mock_a2a_client:
            # Build a real message (wrapped in a StreamResponse on
            # 1.x) so production's stream normalizer /
            # dispatch treat it as a message; the conversion itself
            # is mocked below.
            mock_a2a_message = A2AMessage(
                message_id="m1",
                role=_compat.ROLE_USER,
                parts=[mock_a2a_part],
                context_id="context-123",
            )
            mock_response = _make_stream_message(mock_a2a_message)

            mock_send_message = AsyncMock()
            mock_send_message.__aiter__.return_value = [mock_response]
            mock_a2a_client.send_message.return_value = mock_send_message

            with patch(
                "google.adk.agents.remote_a2a_agent.convert_a2a_message_to_event"
            ) as mock_convert_event:
              mock_result_event = Event(
                  author=agent.name,
                  invocation_id=mock_context.invocation_id,
                  branch=mock_context.branch,
              )
              mock_convert_event.return_value = mock_result_event

              # Mock the logging functions to avoid iteration issues
              with patch(
                  "google.adk.agents.remote_a2a_agent.build_a2a_request_log"
              ) as mock_req_log:
                with patch(
                    "google.adk.agents.remote_a2a_agent.build_a2a_response_log"
                ) as mock_resp_log:
                  mock_req_log.return_value = "Mock request log"
                  mock_resp_log.return_value = "Mock response log"

                  # Patch the production serializer so metadata
                  # stamping does not run MessageToDict on the
                  # mock response (which crashes on 1.x).
                  with patch(
                      "google.adk.a2a._compat.a2a_to_dict",
                      return_value={"k": "v"},
                  ):
                    # Execute
                    events = []
                    async for event in agent._run_async_impl(mock_context):
                      events.append(event)

                  assert len(events) == 1
                  assert events[0] == mock_result_event
                  assert (
                      A2A_METADATA_PREFIX + "request"
                      in mock_result_event.custom_metadata
                  )

                  # Verify A2A client was called
                  mock_a2a_client.send_message.assert_called_once()


class TestRemoteA2aAgentInterceptors:

  @pytest.fixture
  def mock_context(self):
    ctx = Mock(spec=InvocationContext)
    ctx.session = Mock()
    ctx.session.state = {"key": "value"}
    return ctx

  @pytest.mark.asyncio
  async def test_execute_before_request_interceptors_none(self, mock_context):
    request = Mock(spec=A2AMessage)
    result_req, params = await execute_before_request_interceptors(
        None, mock_context, request
    )
    assert result_req is request
    assert params.client_call_context.state == {"key": "value"}

  @pytest.mark.asyncio
  async def test_execute_before_request_interceptors_empty(self, mock_context):
    request = Mock(spec=A2AMessage)
    result_req, params = await execute_before_request_interceptors(
        [], mock_context, request
    )
    assert result_req is request
    assert params.client_call_context.state == {"key": "value"}

  @pytest.mark.asyncio
  async def test_execute_before_request_interceptors_success(
      self, mock_context
  ):
    request = Mock(spec=A2AMessage)
    new_request = Mock(spec=A2AMessage)

    interceptor1 = Mock(spec=RequestInterceptor)
    interceptor1.before_request = AsyncMock(
        return_value=(
            new_request,
            ParametersConfig(
                client_call_context=_compat.ClientCallContext(
                    state={"updated": "true"}
                )
            ),
        )
    )

    result_req, params = await execute_before_request_interceptors(
        [interceptor1], mock_context, request
    )

    assert result_req is new_request
    assert params.client_call_context.state == {"updated": "true"}
    interceptor1.before_request.assert_called_once()

  @pytest.mark.asyncio
  async def test_execute_before_request_interceptors_returns_event(
      self, mock_context
  ):
    request = Mock(spec=A2AMessage)
    event = Mock(spec=Event)

    interceptor1 = Mock(spec=RequestInterceptor)
    interceptor1.before_request = AsyncMock(
        return_value=(
            event,
            ParametersConfig(
                client_call_context=_compat.ClientCallContext(
                    state={"updated": "true"}
                )
            ),
        )
    )

    interceptor2 = Mock(spec=RequestInterceptor)
    interceptor2.before_request = AsyncMock()

    result, params = await execute_before_request_interceptors(
        [interceptor1, interceptor2], mock_context, request
    )

    assert result is event
    assert params.client_call_context.state == {"updated": "true"}
    interceptor1.before_request.assert_called_once()
    interceptor2.before_request.assert_not_called()

  @pytest.mark.asyncio
  async def test_execute_before_request_interceptors_no_before_request(
      self, mock_context
  ):
    request = Mock(spec=A2AMessage)

    interceptor1 = Mock(spec=RequestInterceptor)
    interceptor1.before_request = None

    result_req, params = await execute_before_request_interceptors(
        [interceptor1], mock_context, request
    )

    assert result_req is request
    assert params.client_call_context.state == {"key": "value"}

  @pytest.mark.asyncio
  async def test_execute_after_request_interceptors_none(self, mock_context):
    response = Mock(spec=A2AMessage)
    event = Mock(spec=Event)
    result = await execute_after_request_interceptors(
        None, mock_context, response, event
    )
    assert result is event

  @pytest.mark.asyncio
  async def test_execute_after_request_interceptors_empty(self, mock_context):
    response = Mock(spec=A2AMessage)
    event = Mock(spec=Event)
    result = await execute_after_request_interceptors(
        [], mock_context, response, event
    )
    assert result is event

  @pytest.mark.asyncio
  async def test_execute_after_request_interceptors_success(self, mock_context):
    response = Mock(spec=A2AMessage)
    event = Mock(spec=Event)
    new_event = Mock(spec=Event)

    interceptor1 = Mock(spec=RequestInterceptor)
    interceptor1.after_request = AsyncMock(return_value=new_event)

    result = await execute_after_request_interceptors(
        [interceptor1], mock_context, response, event
    )

    assert result is new_event
    interceptor1.after_request.assert_called_once_with(
        mock_context, response, event
    )

  @pytest.mark.asyncio
  async def test_execute_after_request_interceptors_reverse_order(
      self, mock_context
  ):
    response = Mock(spec=A2AMessage)
    event = Mock(spec=Event)
    event1 = Mock(spec=Event)
    event2 = Mock(spec=Event)

    interceptor1 = Mock(spec=RequestInterceptor)
    interceptor1.after_request = AsyncMock(return_value=event1)

    interceptor2 = Mock(spec=RequestInterceptor)
    interceptor2.after_request = AsyncMock(return_value=event2)

    result = await execute_after_request_interceptors(
        [interceptor1, interceptor2], mock_context, response, event
    )

    assert result is event1
    interceptor2.after_request.assert_called_once_with(
        mock_context, response, event
    )
    interceptor1.after_request.assert_called_once_with(
        mock_context, response, event2
    )

  @pytest.mark.asyncio
  async def test_execute_after_request_interceptors_returns_none(
      self, mock_context
  ):
    response = Mock(spec=A2AMessage)
    event = Mock(spec=Event)

    interceptor1 = Mock(spec=RequestInterceptor)
    interceptor1.after_request = AsyncMock()

    interceptor2 = Mock(spec=RequestInterceptor)
    interceptor2.after_request = AsyncMock(return_value=None)

    result = await execute_after_request_interceptors(
        [interceptor1, interceptor2], mock_context, response, event
    )

    assert result is None
    interceptor2.after_request.assert_called_once_with(
        mock_context, response, event
    )
    interceptor1.after_request.assert_not_called()

  @pytest.mark.asyncio
  async def test_execute_after_request_interceptors_no_after_request(
      self, mock_context
  ):
    response = Mock(spec=A2AMessage)
    event = Mock(spec=Event)

    interceptor1 = Mock(spec=RequestInterceptor)
    interceptor1.after_request = None

    result = await execute_after_request_interceptors(
        [interceptor1], mock_context, response, event
    )

    assert result is event


class TestRemoteA2aAgentDeepcopy:
  """Test deepcopy functionality for RemoteA2aAgent and its config."""

  def test_deepcopy_config(self):
    """Test that A2aRemoteAgentConfig can be deepcopied with interceptors."""
    config = A2aRemoteAgentConfig()
    mock_interceptor = Mock()
    config.request_interceptors = [mock_interceptor]

    copied_config = copy.deepcopy(config)
    assert copied_config is not None

    # Verify that functions are shared (by reference)
    assert copied_config.a2a_message_converter is config.a2a_message_converter

    # Verify that request_interceptors list was copied
    assert copied_config.request_interceptors is not None
    assert len(copied_config.request_interceptors) == 1
    # Standard objects inside lists should be deepcopied (new instances)
    assert (
        copied_config.request_interceptors[0]
        is not config.request_interceptors[0]
    )
