# Copyright 2025 Google LLC
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

import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import AsyncMock
from unittest.mock import create_autospec
from unittest.mock import Mock
from unittest.mock import patch

from google.adk.events.event import Event
from google.adk.sessions.session import Session
from google.genai import types as genai_types
import httpx
import pytest

# Skip all tests in this module if Python version is less than 3.10
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10), reason="A2A requires Python 3.10+"
)

# Import dependencies with version checking
try:
  from a2a.client.client import ClientConfig
  from a2a.client.client import Consumer
  from a2a.client.client_factory import ClientFactory
  from a2a.types import AgentCapabilities
  from a2a.types import AgentCard
  from a2a.types import AgentSkill
  from a2a.types import Artifact
  from a2a.types import Message as A2AMessage
  from a2a.types import Part as A2ATaskStatus
  from a2a.types import SendMessageSuccessResponse
  from a2a.types import Task as A2ATask
  from a2a.types import TaskArtifactUpdateEvent
  from a2a.types import TaskState
  from a2a.types import TaskStatus
  from a2a.types import TaskStatusUpdateEvent
  from a2a.types import TextPart
  from google.adk.agents.invocation_context import InvocationContext
  from google.adk.agents.remote_a2a_agent import A2A_METADATA_PREFIX
  from google.adk.agents.remote_a2a_agent import AgentCardResolutionError
  from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
except ImportError as e:
  if sys.version_info < (3, 10):
    # Create dummy classes to prevent NameError during module compilation.
    # These are needed because the module has type annotations and module-level
    # helper functions that reference imported types.
    class DummyTypes:
      pass

    AgentCapabilities = DummyTypes()
    AgentCard = DummyTypes()
    AgentSkill = DummyTypes()
    A2AMessage = DummyTypes()
    SendMessageSuccessResponse = DummyTypes()
    A2ATask = DummyTypes()
    TaskStatusUpdateEvent = DummyTypes()
    Artifact = DummyTypes()
    TaskArtifactUpdateEvent = DummyTypes()
    InvocationContext = DummyTypes()
    RemoteA2aAgent = DummyTypes()
    AgentCardResolutionError = Exception
    A2A_METADATA_PREFIX = ""
  else:
    raise e


# Helper function to create a proper AgentCard for testing
def create_test_agent_card(
    name: str = "test-agent",
    url: str = "https://example.com/rpc",
    description: str = "Test agent",
) -> AgentCard:
  """Create a test AgentCard with all required fields."""
  return AgentCard(
      name=name,
      url=url,
      description=description,
      version="1.0",
      capabilities=AgentCapabilities(),
      default_input_modes=["text/plain"],
      default_output_modes=["application/json"],
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
      assert result.url == self.agent_card.url
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

    invalid_card = AgentCard(
        name="test",
        description="test",
        version="1.0",
        capabilities=AgentCapabilities(),
        default_input_modes=["text/plain"],
        default_output_modes=["application/json"],
        skills=[
            AgentSkill(
                id="test-skill",
                name="Test Skill",
                description="A test skill",
                tags=["test"],
            )
        ],
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

    invalid_card = AgentCard(
        name="test",
        url="invalid-url",
        description="test",
        version="1.0",
        capabilities=AgentCapabilities(),
        default_input_modes=["text/plain"],
        default_output_modes=["application/json"],
        skills=[
            AgentSkill(
                id="test-skill",
                name="Test Skill",
                description="A test skill",
                tags=["test"],
            )
        ],
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

      with patch(
          "google.adk.agents.remote_a2a_agent.A2AClientFactory"
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

      mock_a2a_part = Mock()
      self.mock_genai_part_converter.return_value = mock_a2a_part

      result = self.agent._construct_message_parts_from_session(
          self.mock_context
      )

      assert len(result) == 2  # Returns tuple of (parts, context_id)
      assert len(result[0]) == 1  # parts list
      assert result[0][0] == mock_a2a_part
      assert result[1] is None  # context_id

  def test_construct_message_parts_from_session_empty_events(self):
    """Test message parts construction with empty events."""
    self.mock_session.events = []

    result = self.agent._construct_message_parts_from_session(self.mock_context)

    assert len(result) == 2  # Returns tuple of (parts, context_id)
    assert result[0] == []  # empty parts list
    assert result[1] is None  # context_id

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
      )
      # Check that metadata was added
      assert result.custom_metadata is not None
      assert A2A_METADATA_PREFIX + "context_id" in result.custom_metadata

  @pytest.mark.asyncio
  async def test_handle_a2a_response_with_task_completed_and_no_update(self):
    """Test successful A2A response handling with non-streeaming task and no update."""
    mock_a2a_task = Mock(spec=A2ATask)
    mock_a2a_task.id = "task-123"
    mock_a2a_task.context_id = "context-123"
    mock_a2a_task.status = Mock(spec=A2ATaskStatus)
    mock_a2a_task.status.state = TaskState.completed

    # Create a proper Event mock that can handle custom_metadata
    mock_a2a_part = Mock(spec=TextPart)
    mock_event = Event(
        author=self.agent.name,
        invocation_id=self.mock_context.invocation_id,
        branch=self.mock_context.branch,
        content=genai_types.Content(role="model", parts=[mock_a2a_part]),
    )

    with patch(
        "google.adk.agents.remote_a2a_agent.convert_a2a_task_to_event"
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

      # Mock the converter to track the order of parts
      converted_parts = []

      def mock_converter(part):
        mock_a2a_part = Mock()
        mock_a2a_part.original_text = part.text
        converted_parts.append(mock_a2a_part)
        return mock_a2a_part

      self.mock_genai_part_converter.side_effect = mock_converter

      result = self.agent._construct_message_parts_from_session(
          self.mock_context
      )

      # Verify the parts are in correct order
      assert len(result) == 2  # Returns tuple of (parts, context_id)
      assert len(result[0]) == 3  # 1 user part + 2 other agent parts
      assert result[1] is None  # context_id

      # Verify order: user part, then "For context:", then agent message
      assert converted_parts[0].original_text == "User question"
      assert converted_parts[1].original_text == "For context:"
      assert (
          converted_parts[2].original_text
          == "[other_agent] said: Response text"
      )

  @pytest.mark.asyncio
  async def test_handle_a2a_response_with_task_submitted_and_no_update(self):
    """Test successful A2A response handling with streaming task and no update."""
    mock_a2a_task = Mock(spec=A2ATask)
    mock_a2a_task.id = "task-123"
    mock_a2a_task.context_id = "context-123"
    mock_a2a_task.status = Mock(spec=A2ATaskStatus)
    mock_a2a_task.status.state = TaskState.submitted

    # Create a proper Event mock that can handle custom_metadata
    mock_a2a_part = Mock(spec=TextPart)
    mock_event = Event(
        author=self.agent.name,
        invocation_id=self.mock_context.invocation_id,
        branch=self.mock_context.branch,
        content=genai_types.Content(role="model", parts=[mock_a2a_part]),
    )

    with patch(
        "google.adk.agents.remote_a2a_agent.convert_a2a_task_to_event"
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
    mock_update.status = Mock(TaskStatus)
    mock_update.status.state = TaskState.completed
    mock_update.status.message = mock_a2a_message

    # Create a proper Event mock that can handle custom_metadata
    mock_a2a_part = Mock(spec=TextPart)
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
    mock_update.status = Mock(TaskStatus)
    mock_update.status.state = TaskState.working
    mock_update.status.message = mock_a2a_message

    # Create a proper Event mock that can handle custom_metadata
    mock_a2a_part = Mock(spec=TextPart)
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
    mock_update.status = Mock(TaskStatus)
    mock_update.status.state = TaskState.completed
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

    with patch(
        "google.adk.agents.remote_a2a_agent.convert_a2a_task_to_event"
    ) as mock_convert:
      mock_convert.return_value = mock_event

      result = await self.agent._handle_a2a_response(
          (mock_a2a_task, mock_update), self.mock_context
      )

      assert result == mock_event
      mock_convert.assert_called_once_with(
          mock_a2a_task, self.agent.name, self.mock_context
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


class TestRemoteA2aAgentMessageHandlingFromFactory:
  """Test message handling functionality."""

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

        result = self.agent._construct_message_parts_from_session(
            self.mock_context
        )

        assert len(result) == 2  # Returns tuple of (parts, context_id)
        assert len(result[0]) == 1  # parts list
        assert result[0][0] == mock_a2a_part
        assert result[1] is None  # context_id

  def test_construct_message_parts_from_session_empty_events(self):
    """Test message parts construction with empty events."""
    self.mock_session.events = []

    result = self.agent._construct_message_parts_from_session(self.mock_context)

    assert len(result) == 2  # Returns tuple of (parts, context_id)
    assert result[0] == []  # empty parts list
    assert result[1] is None  # context_id

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
          mock_a2a_message, self.agent.name, self.mock_context
      )
      # Check that metadata was added
      assert result.custom_metadata is not None
      assert A2A_METADATA_PREFIX + "context_id" in result.custom_metadata

  @pytest.mark.asyncio
  async def test_handle_a2a_response_with_task_completed_and_no_update(self):
    """Test successful A2A response handling with non-streeaming task and no update."""
    mock_a2a_task = Mock(spec=A2ATask)
    mock_a2a_task.id = "task-123"
    mock_a2a_task.context_id = "context-123"
    mock_a2a_task.status = Mock(spec=A2ATaskStatus)
    mock_a2a_task.status.state = TaskState.completed

    # Create a proper Event mock that can handle custom_metadata
    mock_a2a_part = Mock(spec=TextPart)
    mock_event = Event(
        author=self.agent.name,
        invocation_id=self.mock_context.invocation_id,
        branch=self.mock_context.branch,
        content=genai_types.Content(role="model", parts=[mock_a2a_part]),
    )

    with patch(
        "google.adk.agents.remote_a2a_agent.convert_a2a_task_to_event"
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
    mock_a2a_task.status.state = TaskState.submitted

    # Create a proper Event mock that can handle custom_metadata
    mock_a2a_part = Mock(spec=TextPart)
    mock_event = Event(
        author=self.agent.name,
        invocation_id=self.mock_context.invocation_id,
        branch=self.mock_context.branch,
        content=genai_types.Content(role="model", parts=[mock_a2a_part]),
    )

    with patch(
        "google.adk.agents.remote_a2a_agent.convert_a2a_task_to_event"
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
    mock_update.status = Mock(TaskStatus)
    mock_update.status.state = TaskState.completed
    mock_update.status.message = mock_a2a_message

    # Create a proper Event mock that can handle custom_metadata
    mock_a2a_part = Mock(spec=TextPart)
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
    mock_update.status = Mock(TaskStatus)
    mock_update.status.state = TaskState.working
    mock_update.status.message = mock_a2a_message

    # Create a proper Event mock that can handle custom_metadata
    mock_a2a_part = Mock(spec=TextPart)
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
    mock_update.status = Mock(TaskStatus)
    mock_update.status.state = TaskState.completed
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

    with patch(
        "google.adk.agents.remote_a2a_agent.convert_a2a_task_to_event"
    ) as mock_convert:
      mock_convert.return_value = mock_event

      result = await self.agent._handle_a2a_response(
          (mock_a2a_task, mock_update), self.mock_context
      )

      assert result == mock_event
      mock_convert.assert_called_once_with(
          mock_a2a_task, self.agent.name, self.mock_context
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
          # Create proper A2A part mocks
          from a2a.client import Client as A2AClient
          from a2a.types import TextPart

          mock_a2a_part = Mock(spec=TextPart)
          mock_construct.return_value = (
              [mock_a2a_part],
              "context-123",
          )  # Tuple with parts and context_id

          # Mock A2A client
          mock_a2a_client = create_autospec(spec=A2AClient, instance=True)
          mock_response = Mock()
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

                # Mock the A2AMessage constructor
                with patch(
                    "google.adk.agents.remote_a2a_agent.A2AMessage"
                ) as mock_message_class:
                  mock_message = Mock(spec=A2AMessage)
                  mock_message_class.return_value = mock_message

                  # Add model_dump to mock_response for metadata
                  mock_response.model_dump.return_value = {"test": "response"}

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
          # Create proper A2A part mocks
          from a2a.types import TextPart

          mock_a2a_part = Mock(spec=TextPart)
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

            # Mock the A2AMessage constructor
            with patch(
                "google.adk.agents.remote_a2a_agent.A2AMessage"
            ) as mock_message_class:
              mock_message = Mock(spec=A2AMessage)
              mock_message_class.return_value = mock_message

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


class TestRemoteA2aAgentExecutionFromFactory:
  """Test agent execution functionality."""

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
          # Create proper A2A part mocks
          from a2a.client import Client as A2AClient
          from a2a.types import TextPart

          mock_a2a_part = Mock(spec=TextPart)
          mock_construct.return_value = (
              [mock_a2a_part],
              "context-123",
          )  # Tuple with parts and context_id

          # Mock A2A client
          mock_a2a_client = create_autospec(spec=A2AClient, instance=True)
          mock_response = Mock()
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

                # Mock the A2AMessage constructor
                with patch(
                    "google.adk.agents.remote_a2a_agent.A2AMessage"
                ) as mock_message_class:
                  mock_message = Mock(spec=A2AMessage)
                  mock_message_class.return_value = mock_message

                  # Add model_dump to mock_response for metadata
                  mock_response.root.model_dump.return_value = {
                      "test": "response"
                  }

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
          # Create proper A2A part mocks
          from a2a.types import TextPart

          mock_a2a_part = Mock(spec=TextPart)
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

            # Mock the A2AMessage constructor
            with patch(
                "google.adk.agents.remote_a2a_agent.A2AMessage"
            ) as mock_message_class:
              mock_message = Mock(spec=A2AMessage)
              mock_message_class.return_value = mock_message

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

    # Mock session with text event
    mock_part = Mock()
    mock_part.text = "Hello world"

    mock_content = Mock()
    mock_content.parts = [mock_part]

    mock_event = Mock()
    mock_event.content = mock_content

    mock_session = Mock(spec=Session)
    mock_session.id = "session-123"
    mock_session.events = [mock_event]

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
        from a2a.types import TextPart

        mock_a2a_part = Mock(spec=TextPart)
        mock_convert_part.return_value = mock_a2a_part

        with patch("httpx.AsyncClient") as mock_httpx_client_class:
          mock_httpx_client = AsyncMock()
          mock_httpx_client_class.return_value = mock_httpx_client

          with patch.object(agent, "_a2a_client") as mock_a2a_client:
            mock_a2a_message = create_autospec(spec=A2AMessage, instance=True)
            mock_a2a_message.context_id = "context-123"
            mock_response = mock_a2a_message

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

                  # Add model_dump to mock_response for metadata
                  mock_response.model_dump.return_value = {"test": "response"}

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

    # Mock session with text event
    mock_part = Mock()
    mock_part.text = "Hello world"

    mock_content = Mock()
    mock_content.parts = [mock_part]

    mock_event = Mock()
    mock_event.content = mock_content

    mock_session = Mock(spec=Session)
    mock_session.id = "session-123"
    mock_session.events = [mock_event]

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
        from a2a.types import TextPart

        mock_a2a_part = Mock(spec=TextPart)
        mock_convert_part.return_value = mock_a2a_part

        with patch("httpx.AsyncClient") as mock_httpx_client_class:
          mock_httpx_client = AsyncMock()
          mock_httpx_client_class.return_value = mock_httpx_client

          with patch.object(agent, "_a2a_client") as mock_a2a_client:
            mock_a2a_message = create_autospec(spec=A2AMessage, instance=True)
            mock_a2a_message.context_id = "context-123"
            mock_response = mock_a2a_message

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

                  # Add model_dump to mock_response for metadata
                  mock_response.model_dump.return_value = {"test": "response"}

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
