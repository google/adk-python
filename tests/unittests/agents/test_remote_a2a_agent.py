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
import threading
from unittest.mock import AsyncMock
from unittest.mock import create_autospec
from unittest.mock import Mock
from unittest.mock import patch

from a2a.client import Client as A2AClient
from a2a.client.client import ClientConfig
from a2a.client.client_factory import ClientFactory
from a2a.types import AgentCapabilities
from a2a.types import AgentCard
from a2a.types import AgentInterface
from a2a.types import AgentSkill
from a2a.types import Artifact
from a2a.types import Message as A2AMessage
from a2a.types import Task as A2ATask
from a2a.types import TaskArtifactUpdateEvent
from a2a.types import TaskStatus as A2ATaskStatus
from a2a.types import TaskStatusUpdateEvent
from fastapi.openapi.models import APIKey as APIKeyScheme
from fastapi.openapi.models import APIKeyIn
from google.adk.a2a import _compat
from google.adk.a2a.agent import A2aCardRequestConfig
from google.adk.a2a.agent import CardRequestInterceptor
from google.adk.a2a.agent import ParametersConfig
from google.adk.a2a.agent import RequestInterceptor
from google.adk.a2a.agent.config import A2aRemoteAgentConfig
from google.adk.a2a.agent.utils import execute_after_request_interceptors
from google.adk.a2a.agent.utils import execute_before_card_request_interceptors
from google.adk.a2a.agent.utils import execute_before_request_interceptors
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm.task._finish_task_tool import FINISH_TASK_ERROR_RESULT
from google.adk.agents.llm.task._finish_task_tool import FINISH_TASK_SUCCESS_RESULT
from google.adk.agents.llm.task._finish_task_tool import FINISH_TASK_TOOL_NAME
from google.adk.agents.remote_a2a_agent import A2A_METADATA_PREFIX
from google.adk.agents.remote_a2a_agent import AgentCardResolutionError
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
import google.adk.agents.remote_a2a_agent as remote_a2a_agent
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_preprocessor import TOOLSET_AUTH_CREDENTIAL_ID_PREFIX
from google.adk.events.event import Event
from google.adk.flows.llm_flows.functions import REQUEST_EUC_FUNCTION_CALL_NAME
from google.adk.sessions.session import Session
from google.genai import types as genai_types
import httpx
from pydantic import BaseModel
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


def _request_headers(client_call_context) -> dict[str, str]:
  """Read back the HTTP headers an interceptor set, version-agnostically."""
  if _compat.IS_A2A_V1:
    return dict(client_call_context.service_parameters or {})
  http_kwargs = client_call_context.state.get("http_kwargs", {})
  return dict(http_kwargs.get("headers", {}))


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


def _make_stream_task(task: A2ATask):
  """Wrap a Task in the shape ``send_message`` yields for the active SDK."""
  if _compat.IS_A2A_V1:
    from a2a.types import StreamResponse

    resp = StreamResponse()
    resp.task.CopyFrom(task)
    return resp
  return (task, None)


def _make_artifact_chunk(text: str, *, append: bool, last_chunk: bool):
  """Build one streamed chunk of an artifact, version-agnostically."""
  return TaskArtifactUpdateEvent(
      task_id="task-123",
      context_id="context-123",
      append=append,
      last_chunk=last_chunk,
      artifact=_compat.make_artifact(
          artifact_id="artifact-1",
          parts=[_compat.make_text_part(text)],
      ),
  )


def _make_accumulated_task(part_texts):
  """Build the running Task the stream normalizer yields alongside an update.

  The task carries the artifact parts accumulated across all chunks received
  so far, mirroring the 0.3.x ClientTaskManager / 1.x stream normalizer.
  """
  return _compat.make_task(
      id="task-123",
      status=_compat.make_task_status(_compat.TS_WORKING),
      context_id="context-123",
      artifacts=[
          _compat.make_artifact(
              artifact_id="artifact-1",
              parts=[_compat.make_text_part(text) for text in part_texts],
          )
      ],
  )


def _make_dummy_task_trigger_event(
    task_id: str = "task-1", agent_name: str = "test_agent"
) -> Event:
  """Build a dummy triggering Event containing a FunctionCall for task delegation."""
  trigger_fc = genai_types.FunctionCall(
      id=task_id, name=agent_name, args={"request": "start"}
  )
  return Event(
      author="coordinator",
      content=genai_types.Content(
          role="model", parts=[genai_types.Part(function_call=trigger_fc)]
      ),
  )


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


def _make_multi_interface_card(interfaces) -> AgentCard:
  """Build a card offering several RPC endpoints, version-agnostically.

  ``interfaces`` is a list of ``(url, transport)`` pairs; the first pair is the
  card's primary endpoint. On 1.x every pair becomes a ``supported_interfaces``
  entry; on 0.3.x the first pair is the top-level ``url``/``preferredTransport``
  and the rest land in ``additional_interfaces``.
  """
  if _compat.IS_A2A_V1:
    return _compat.parse_agent_card({
        "name": "test-agent",
        "description": "Test agent",
        "version": "1.0",
        "supported_interfaces": [
            {"url": url, "protocol_binding": transport}
            for url, transport in interfaces
        ],
        "default_input_modes": ["text/plain"],
        "default_output_modes": ["text/plain"],
    })
  (primary_url, primary_transport), *extra = interfaces
  return _make_agent_card(
      url=primary_url,
      preferred_transport=primary_transport,
      additional_interfaces=[
          AgentInterface(url=url, transport=transport)
          for url, transport in extra
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

  def test_init_with_agent_card_object_adopts_card_description(self):
    """Test description is autopopulated from a directly supplied card."""
    agent_card = create_test_agent_card(description="Converts currencies")

    agent = RemoteA2aAgent(name="test_agent", agent_card=agent_card)

    assert agent.description == "Converts currencies"

  def test_init_with_agent_card_object_keeps_explicit_description(self):
    """Test an explicit description wins over the card's."""
    agent_card = create_test_agent_card(description="Converts currencies")

    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card=agent_card,
        description="Test description",
    )

    assert agent.description == "Test description"

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
            "https://example.com/agent.json", Mock()
        )

        assert result == self.agent_card
        mock_resolver_class.assert_called_once_with(
            httpx_client=mock_client, base_url="https://example.com"
        )
        mock_resolver.get_agent_card.assert_called_once_with(
            relative_card_path="/agent.json", http_kwargs=None
        )

  @pytest.mark.asyncio
  async def test_resolve_agent_card_from_url_invalid_url(self):
    """Test agent card resolution from invalid URL raises error."""
    agent = RemoteA2aAgent(name="test_agent", agent_card="invalid-url")

    with pytest.raises(AgentCardResolutionError, match="Invalid URL format"):
      await agent._resolve_agent_card_from_url("invalid-url", Mock())

  @pytest.mark.asyncio
  async def test_card_request_interceptors_injects_headers(self):
    """Header provider headers (from session state) are sent for the card."""

    async def provider(ctx):
      return A2aCardRequestConfig(
          headers={"Authorization": f"Bearer {ctx.session.state['token']}"}
      )

    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card="https://example.com/agent.json",
        config=A2aRemoteAgentConfig(
            card_request_interceptors=[
                CardRequestInterceptor(before_request=provider)
            ]
        ),
    )
    ctx = Mock()
    ctx.session.state = {"token": "abc"}

    with patch.object(agent, "_ensure_httpx_client") as mock_ensure_client:
      mock_ensure_client.return_value = AsyncMock()
      with patch(
          "google.adk.agents.remote_a2a_agent.A2ACardResolver"
      ) as mock_resolver_class:
        mock_resolver = AsyncMock()
        mock_resolver.get_agent_card.return_value = self.agent_card
        mock_resolver_class.return_value = mock_resolver

        await agent._resolve_agent_card_from_url(
            "https://example.com/agent.json", ctx
        )

    mock_resolver.get_agent_card.assert_called_once_with(
        relative_card_path="/agent.json",
        http_kwargs={"headers": {"Authorization": "Bearer abc"}},
    )

  @pytest.mark.asyncio
  async def test_card_request_interceptors_merge_later_overrides(self):
    """Headers from multiple interceptors merge; later overrides earlier."""

    async def provider_a(ctx):
      return A2aCardRequestConfig(headers={"X-Common": "a", "X-A": "1"})

    async def provider_b(ctx):
      return A2aCardRequestConfig(headers={"X-Common": "b", "X-B": "2"})

    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card="https://example.com/agent.json",
        config=A2aRemoteAgentConfig(
            card_request_interceptors=[
                CardRequestInterceptor(before_request=provider_a),
                CardRequestInterceptor(before_request=provider_b),
            ]
        ),
    )

    with patch.object(agent, "_ensure_httpx_client") as mock_ensure_client:
      mock_ensure_client.return_value = AsyncMock()
      with patch(
          "google.adk.agents.remote_a2a_agent.A2ACardResolver"
      ) as mock_resolver_class:
        mock_resolver = AsyncMock()
        mock_resolver.get_agent_card.return_value = self.agent_card
        mock_resolver_class.return_value = mock_resolver

        await agent._resolve_agent_card_from_url(
            "https://example.com/agent.json", Mock()
        )

    mock_resolver.get_agent_card.assert_called_once_with(
        relative_card_path="/agent.json",
        http_kwargs={"headers": {"X-Common": "b", "X-A": "1", "X-B": "2"}},
    )

  @pytest.mark.asyncio
  async def test_ensure_resolved_refetches_card_when_interceptor_set(self):
    """With a card interceptor, the card is re-resolved on each invocation."""
    provider = AsyncMock(
        return_value=A2aCardRequestConfig(headers={"Authorization": "Bearer x"})
    )
    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card="https://example.com/agent.json",
        config=A2aRemoteAgentConfig(
            card_request_interceptors=[
                CardRequestInterceptor(before_request=provider)
            ]
        ),
    )

    with patch.object(
        agent, "_resolve_agent_card", new_callable=AsyncMock
    ) as mock_resolve:
      mock_resolve.return_value = self.agent_card
      with patch.object(agent, "_ensure_httpx_client") as mock_ensure:
        mock_ensure.return_value = AsyncMock()
        mock_factory = Mock()
        mock_factory.create.side_effect = [Mock(), Mock()]
        agent._a2a_client_factory = mock_factory

        client1 = await agent._ensure_resolved(Mock())
        client2 = await agent._ensure_resolved(Mock())

    assert mock_resolve.await_count == 2
    assert mock_factory.create.call_count == 2
    assert client1 is not client2
    # Shared state is NEVER mutated on the interceptor path.
    assert agent._agent_card is None
    assert agent._a2a_client is None
    assert agent._is_resolved is False

  @pytest.mark.asyncio
  async def test_card_interceptor_does_not_leak_across_sessions(self):
    """One session's card/client must not overwrite another's shared state."""

    async def provider(ctx):
      return A2aCardRequestConfig(
          headers={"Authorization": f"Bearer {ctx.session.state['token']}"}
      )

    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card="https://example.com/agent.json",
        config=A2aRemoteAgentConfig(
            card_request_interceptors=[
                CardRequestInterceptor(before_request=provider)
            ]
        ),
    )

    card_a = create_test_agent_card()
    card_b = create_test_agent_card()
    client_a = Mock()
    client_b = Mock()

    ctx_a = Mock()
    ctx_a.session.state = {"token": "AAA"}
    ctx_b = Mock()
    ctx_b.session.state = {"token": "BBB"}

    with patch.object(
        agent, "_resolve_agent_card", new_callable=AsyncMock
    ) as mock_resolve:
      mock_resolve.side_effect = [card_a, card_b]
      with patch.object(agent, "_ensure_httpx_client") as mock_ensure:
        mock_ensure.return_value = AsyncMock()
        mock_factory = Mock()
        mock_factory.create.side_effect = lambda card: (
            client_a if card is card_a else client_b
        )
        agent._a2a_client_factory = mock_factory

        result_a = await agent._ensure_resolved(ctx_a)
        result_b = await agent._ensure_resolved(ctx_b)

    assert result_a is client_a
    assert result_b is client_b
    assert agent._agent_card is None
    assert agent._a2a_client is None

  @pytest.mark.asyncio
  async def test_ensure_resolved_caches_card_without_interceptor(self):
    """Without a card interceptor, the card is resolved only once."""
    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card="https://example.com/agent.json",
    )

    with patch.object(
        agent, "_resolve_agent_card", new_callable=AsyncMock
    ) as mock_resolve:
      mock_resolve.return_value = self.agent_card
      with patch.object(agent, "_ensure_httpx_client") as mock_ensure:
        mock_ensure.return_value = AsyncMock()
        mock_factory = Mock()
        mock_factory.create.return_value = Mock()
        agent._a2a_client_factory = mock_factory

        await agent._ensure_resolved(Mock())
        await agent._ensure_resolved(Mock())

    assert mock_resolve.await_count == 1

  @pytest.mark.asyncio
  async def test_ensure_resolved_without_ctx_uses_cached_path(self):
    """_ensure_resolved() is callable with no ctx (backward compatible)."""
    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card="https://example.com/agent.json",
    )

    with patch.object(
        agent, "_resolve_agent_card", new_callable=AsyncMock
    ) as mock_resolve:
      mock_resolve.return_value = self.agent_card
      with patch.object(agent, "_ensure_httpx_client") as mock_ensure:
        mock_ensure.return_value = AsyncMock()
        mock_client = Mock()
        mock_factory = Mock()
        mock_factory.create.return_value = mock_client
        agent._a2a_client_factory = mock_factory

        # Called with no ctx argument.
        client = await agent._ensure_resolved()

    assert client is mock_client
    assert agent._a2a_client is mock_client
    assert agent._is_resolved is True
    # ctx defaults to None and is forwarded to card resolution.
    mock_resolve.assert_awaited_once_with(None)

  @pytest.mark.asyncio
  async def test_ensure_resolved_no_ctx_ignores_card_interceptors(self):
    """With interceptors but no ctx, resolution falls back to the cached path."""
    provider = AsyncMock(
        return_value=A2aCardRequestConfig(headers={"Authorization": "Bearer x"})
    )
    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card="https://example.com/agent.json",
        config=A2aRemoteAgentConfig(
            card_request_interceptors=[
                CardRequestInterceptor(before_request=provider)
            ]
        ),
    )

    with patch.object(
        agent, "_resolve_agent_card", new_callable=AsyncMock
    ) as mock_resolve:
      mock_resolve.return_value = self.agent_card
      with patch.object(agent, "_ensure_httpx_client") as mock_ensure:
        mock_ensure.return_value = AsyncMock()
        mock_factory = Mock()
        mock_factory.create.return_value = Mock()
        agent._a2a_client_factory = mock_factory

        # No ctx: must not enter the per-invocation path (would call the
        # provider with ctx=None). Falls back to cached resolution instead.
        await agent._ensure_resolved()
        await agent._ensure_resolved()

    # Cached (shared) path used: resolved once, provider never called.
    assert mock_resolve.await_count == 1
    assert agent._a2a_client is not None
    provider.assert_not_awaited()

  @pytest.mark.asyncio
  async def test_card_request_interceptors_ignored_for_direct_card(self):
    """A static AgentCard is never re-fetched even with a card interceptor."""
    provider = AsyncMock(
        return_value=A2aCardRequestConfig(headers={"Authorization": "Bearer x"})
    )
    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card=self.agent_card,
        config=A2aRemoteAgentConfig(
            card_request_interceptors=[
                CardRequestInterceptor(before_request=provider)
            ]
        ),
    )

    with patch.object(
        agent, "_resolve_agent_card", new_callable=AsyncMock
    ) as mock_resolve:
      with patch.object(agent, "_ensure_httpx_client") as mock_ensure:
        mock_ensure.return_value = AsyncMock()
        mock_factory = Mock()
        mock_factory.create.return_value = Mock()
        agent._a2a_client_factory = mock_factory

        await agent._ensure_resolved(Mock())
        await agent._ensure_resolved(Mock())

    mock_resolve.assert_not_called()
    provider.assert_not_awaited()

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
  async def test_validate_agent_card_accepts_same_origin_https_rpc_url(self):
    """A fetched card pointing back at its own origin is accepted."""
    agent = RemoteA2aAgent(
        name="test_agent", agent_card="https://example.com/agent.json"
    )

    # Should not raise any exception.
    await agent._validate_agent_card(
        create_test_agent_card(url="https://example.com/rpc")
    )

  @pytest.mark.asyncio
  async def test_validate_agent_card_rejects_cross_origin_rpc_url(self):
    """A fetched card cannot redirect RPC traffic to an unrelated host."""
    agent = RemoteA2aAgent(
        name="test_agent", agent_card="https://example.com/agent.json"
    )

    with pytest.raises(AgentCardResolutionError, match="same origin"):
      await agent._validate_agent_card(
          create_test_agent_card(url="https://attacker.example.net/rpc")
      )

  @pytest.mark.asyncio
  async def test_validate_agent_card_rejects_plain_http_rpc_url(self):
    """A fetched card cannot downgrade RPC traffic to cleartext."""
    agent = RemoteA2aAgent(
        name="test_agent", agent_card="https://example.com/agent.json"
    )

    with pytest.raises(AgentCardResolutionError, match="must use https"):
      await agent._validate_agent_card(
          create_test_agent_card(url="http://example.com/rpc")
      )

  @pytest.mark.asyncio
  @pytest.mark.parametrize(
      "rpc_url",
      [
          "http://127.0.0.1:8080/rpc",
          "http://[::1]:8080/rpc",
          "http://169.254.169.254/rpc",
          "http://metadata.internal/rpc",
      ],
  )
  async def test_validate_agent_card_rejects_internal_rpc_url(self, rpc_url):
    """A fetched card cannot aim RPC traffic at host-local or internal hosts."""
    agent = RemoteA2aAgent(
        name="test_agent", agent_card="https://example.com/agent.json"
    )

    with pytest.raises(AgentCardResolutionError):
      await agent._validate_agent_card(create_test_agent_card(url=rpc_url))

  @pytest.mark.asyncio
  async def test_validate_agent_card_allows_local_development_http(self):
    """Plain http stays allowed for a same-origin loopback card."""
    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card="http://localhost:8000/.well-known/agent.json",
    )

    # Should not raise any exception.
    await agent._validate_agent_card(
        create_test_agent_card(url="http://localhost:8000/a2a")
    )

  @pytest.mark.asyncio
  async def test_validate_agent_card_file_source_is_not_origin_checked(self):
    """A card read from a local file is configuration, not remote data."""
    agent = RemoteA2aAgent(name="test_agent", agent_card="/path/to/agent.json")

    # Should not raise any exception.
    await agent._validate_agent_card(
        create_test_agent_card(url="http://internal-host:8080/rpc")
    )

  @pytest.mark.asyncio
  @pytest.mark.parametrize(
      "interfaces",
      [
          # A second interface on the transport the client already prefers
          # displaces the benign endpoint during transport negotiation.
          [
              ("https://example.com/rpc", "JSONRPC"),
              ("http://169.254.169.254/", "JSONRPC"),
          ],
          # The primary endpoint advertises a transport the client cannot
          # speak, so negotiation falls through to the second interface.
          [
              ("https://example.com/rpc", "GRPC"),
              ("http://127.0.0.1:9000/", "HTTP+JSON"),
          ],
      ],
      ids=["displaces_primary", "primary_transport_unsupported"],
  )
  async def test_validate_agent_card_rejects_off_origin_extra_interface(
      self, interfaces
  ):
    """Every endpoint the card offers is constrained, not just the first."""
    agent = RemoteA2aAgent(
        name="test_agent", agent_card="https://example.com/agent.json"
    )

    with pytest.raises(AgentCardResolutionError):
      await agent._validate_agent_card(_make_multi_interface_card(interfaces))

  @pytest.mark.asyncio
  async def test_validate_agent_card_accepts_same_origin_extra_interface(self):
    """A card may still offer several endpoints on its own origin."""
    agent = RemoteA2aAgent(
        name="test_agent", agent_card="https://example.com/agent.json"
    )

    # Should not raise any exception.
    await agent._validate_agent_card(
        _make_multi_interface_card([
            ("https://example.com/rpc", "JSONRPC"),
            ("https://example.com/rest", "HTTP+JSON"),
        ])
    )

  def test_agent_card_rpc_urls_lists_every_endpoint(self):
    """Validation enumerates every endpoint on the card, in card order."""
    card = _make_multi_interface_card([
        ("https://example.com/rpc", "JSONRPC"),
        ("https://example.com/rest", "HTTP+JSON"),
    ])

    assert _compat.agent_card_rpc_urls(card) == [
        "https://example.com/rpc",
        "https://example.com/rest",
    ]

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

        await agent._ensure_resolved(Mock())

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

        await agent._ensure_resolved(Mock())

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

          await agent._ensure_resolved(Mock())

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
      await agent._ensure_resolved(Mock())

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
    # The response sanitizer always runs now; a bare Mock content is not
    # iterable, and there is nothing to sanitize here (no function calls), so
    # give it None to make the sanitizer a no-op.
    mock_latest_event.content = None
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

  def test_construct_message_parts_from_session_foreign_function_response_not_converted(
      self,
  ):
    """Test that foreign function responses are NOT converted to text in default mode."""
    # Mock event with a function response
    mock_fr = genai_types.FunctionResponse(
        id="fc-1", name="tool_1", response={"result": "done"}
    )
    mock_part = Mock()
    mock_part.function_response = mock_fr
    mock_part.text = None

    mock_content = Mock()
    mock_content.parts = [mock_part]

    mock_event = Mock()
    mock_event.author = "user"
    mock_event.content = mock_content
    mock_event.get_function_calls.return_value = []
    mock_event.get_function_responses.return_value = []

    self.mock_session.events = [mock_event]

    with patch(
        "google.adk.agents.remote_a2a_agent._present_other_agent_message"
    ) as mock_present:
      mock_present.return_value = mock_event

      mock_a2a_part = _compat.make_text_part("tool_response_text")
      self.mock_genai_part_converter.return_value = mock_a2a_part

      parts, _ = self.agent._construct_message_parts_from_session(
          self.mock_context
      )

      # Should call the converter, not convert to text
      self.mock_genai_part_converter.assert_called_once_with(mock_part)
      assert len(parts) == 1
      assert parts[0] == mock_a2a_part

  def test_construct_message_parts_from_session_stops_on_agent_reply_when_disabled(
      self,
  ):
    """Test message parts construction stops on agent reply when disabled."""
    self.agent._full_history_when_stateless = False
    part1 = Mock()
    part1.text = "User 1"
    content1 = Mock()
    content1.parts = [part1]
    user1 = Mock(
        live_session_id=None,
        author="user",
        custom_metadata=None,
        content=content1,
    )
    user1.get_function_calls.return_value = []
    user1.get_function_responses.return_value = []

    part2 = Mock()
    part2.text = "Agent 1"
    content2 = Mock()
    content2.parts = [part2]
    agent1 = Mock(
        live_session_id=None,
        author=self.agent.name,
        content=content2,
        custom_metadata={
            A2A_METADATA_PREFIX + "response": True,
        },
    )
    agent1.get_function_calls.return_value = []
    agent1.get_function_responses.return_value = []

    agent2 = Mock(
        live_session_id=None,
        author=self.agent.name,
        content=None,
        # Just actions, no content. Not marked as a response.
        actions=Mock(),
        custom_metadata=None,
    )
    agent2.get_function_calls.return_value = []
    agent2.get_function_responses.return_value = []

    part3 = Mock()
    part3.text = "User 2"
    content3 = Mock()
    content3.parts = [part3]
    user2 = Mock(
        live_session_id=None,
        author="user",
        content=content3,
        custom_metadata=None,
    )
    user2.get_function_calls.return_value = []
    user2.get_function_responses.return_value = []

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

  def test_construct_message_parts_from_session_stateless_full_history_when_enabled(
      self,
  ):
    """Test full history for stateless agent when enabled."""
    self.agent._full_history_when_stateless = True
    part1 = Mock()
    part1.text = "User 1"
    content1 = Mock()
    content1.parts = [part1]
    user1 = Mock(
        live_session_id=None,
        author="user",
        custom_metadata=None,
        content=content1,
    )
    user1.get_function_calls.return_value = []
    user1.get_function_responses.return_value = []

    part2 = Mock()
    part2.text = "Agent 1"
    content2 = Mock()
    content2.parts = [part2]
    agent1 = Mock(
        live_session_id=None,
        author=self.agent.name,
        content=content2,
        custom_metadata={
            A2A_METADATA_PREFIX + "response": True,
        },
    )
    agent1.get_function_calls.return_value = []
    agent1.get_function_responses.return_value = []

    part3 = Mock()
    part3.text = "User 2"
    content3 = Mock()
    content3.parts = [part3]
    user2 = Mock(
        live_session_id=None,
        author="user",
        content=content3,
        custom_metadata=None,
    )
    user2.get_function_calls.return_value = []
    user2.get_function_responses.return_value = []

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

    This verifies the fix for the case where the code could raise when it
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
  async def test_handle_a2a_response_filters_thought_parts_from_completed_task(
      self,
  ):
    """Test that thought parts are filtered from completed task response.

    When an A2A server returns a completed task with both thought and
    non-thought parts, the client should only include non-thought parts
    in the user-facing event. Fixes #4676.
    """
    mock_a2a_task = Mock(spec=A2ATask)
    mock_a2a_task.id = "task-123"
    mock_a2a_task.context_id = "context-123"
    mock_a2a_task.status = Mock(spec=A2ATaskStatus)
    mock_a2a_task.status.state = _compat.TS_COMPLETED

    # Create event with mixed thought/non-thought parts
    thought_part = genai_types.Part(text="internal reasoning", thought=True)
    answer_part = genai_types.Part(text="final answer")
    mock_event = Event(
        author=self.agent.name,
        invocation_id=self.mock_context.invocation_id,
        branch=self.mock_context.branch,
        content=genai_types.Content(
            role="model", parts=[thought_part, answer_part]
        ),
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

      # Only non-thought parts should remain
      assert len(result.content.parts) == 1
      assert result.content.parts[0].text == "final answer"
      assert result.content.parts[0].thought is None

  @pytest.mark.asyncio
  async def test_handle_a2a_response_filters_thought_parts_from_status_update(
      self,
  ):
    """Test that thought parts are filtered from completed status update.

    Fixes #4676.
    """
    mock_a2a_task = Mock(spec=A2ATask)
    mock_a2a_task.id = "task-123"
    mock_a2a_task.context_id = "context-123"

    mock_update = Mock(spec=TaskStatusUpdateEvent)
    mock_update.status = Mock(spec=A2ATaskStatus)
    mock_update.status.state = _compat.TS_COMPLETED
    mock_update.status.message = Mock(spec=A2AMessage)

    # Create event with mixed thought/non-thought parts
    thought_part = genai_types.Part(text="thinking...", thought=True)
    answer_part = genai_types.Part(text="the answer")
    mock_event = Event(
        author=self.agent.name,
        invocation_id=self.mock_context.invocation_id,
        branch=self.mock_context.branch,
        content=genai_types.Content(
            role="model", parts=[thought_part, answer_part]
        ),
    )

    with patch(
        "google.adk.agents.remote_a2a_agent.convert_a2a_message_to_event"
    ) as mock_convert:
      mock_convert.return_value = mock_event

      result = await self.agent._handle_a2a_response(
          (mock_a2a_task, mock_update), self.mock_context
      )

      # Only non-thought parts should remain
      assert len(result.content.parts) == 1
      assert result.content.parts[0].text == "the answer"

  @pytest.mark.asyncio
  async def test_handle_a2a_response_preserves_all_thought_parts_for_working(
      self,
  ):
    """Test that working state events keep all parts as thoughts.

    Intermediate events (working/submitted) should retain all parts
    marked as thought for streaming progress display.
    """
    mock_a2a_task = Mock(spec=A2ATask)
    mock_a2a_task.id = "task-123"
    mock_a2a_task.context_id = "context-123"
    mock_a2a_task.status = Mock(spec=A2ATaskStatus)
    mock_a2a_task.status.state = _compat.TS_WORKING

    part = genai_types.Part(text="still thinking", thought=True)
    mock_event = Event(
        author=self.agent.name,
        invocation_id=self.mock_context.invocation_id,
        branch=self.mock_context.branch,
        content=genai_types.Content(role="model", parts=[part]),
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

      # All parts should be marked as thought and preserved
      assert len(result.content.parts) == 1
      assert result.content.parts[0].thought is True

  @pytest.mark.asyncio
  async def test_handle_a2a_response_filters_thought_from_a2a_message(self):
    """Test thought filtering for regular A2AMessage responses.

    Fixes #4676.
    """
    mock_a2a_message = Mock(spec=A2AMessage)
    mock_a2a_message.context_id = "context-123"

    thought_part = genai_types.Part(text="reasoning", thought=True)
    answer_part = genai_types.Part(text="response")
    mock_event = Event(
        author=self.agent.name,
        invocation_id=self.mock_context.invocation_id,
        branch=self.mock_context.branch,
        content=genai_types.Content(
            role="model", parts=[thought_part, answer_part]
        ),
    )

    with patch(
        "google.adk.agents.remote_a2a_agent.convert_a2a_message_to_event"
    ) as mock_convert:
      mock_convert.return_value = mock_event

      result = await self.agent._handle_a2a_response(
          mock_a2a_message, self.mock_context
      )

      # Only non-thought parts should remain
      assert len(result.content.parts) == 1
      assert result.content.parts[0].text == "response"

  @pytest.mark.asyncio
  async def test_handle_a2a_response_with_artifact_update(self):
    """Test successful A2A response handling with artifact update."""
    mock_a2a_task = Mock(spec=A2ATask)
    mock_a2a_task.id = "task-123"
    mock_a2a_task.context_id = "context-123"

    update = _make_artifact_chunk("chunk", append=False, last_chunk=True)

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
          (mock_a2a_task, update), self.mock_context
      )

      assert result == mock_event
      mock_convert.assert_called_once()
      # Only the parts carried by this update are converted, not the
      # accumulated task.
      converted_message = mock_convert.call_args[0][0]
      assert list(converted_message.parts) == list(update.artifact.parts)
      # Check that metadata was added
      assert result.custom_metadata is not None
      assert A2A_METADATA_PREFIX + "task_id" in result.custom_metadata
      assert A2A_METADATA_PREFIX + "context_id" in result.custom_metadata

  @pytest.mark.asyncio
  async def test_handle_a2a_response_with_appended_artifact_chunk(self):
    """An appended (middle) artifact chunk emits only its own parts."""
    mock_a2a_task = Mock(spec=A2ATask)
    mock_a2a_task.id = "task-123"
    mock_a2a_task.context_id = "context-123"

    update = _make_artifact_chunk("middle", append=True, last_chunk=False)

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
          (mock_a2a_task, update), self.mock_context
      )

      assert result == mock_event
      assert result.partial is True
      converted_message = mock_convert.call_args[0][0]
      assert list(converted_message.parts) == list(update.artifact.parts)

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


class TestRemoteA2aAgentTaskModeMessageHandling:
  """Test message handling functionality under task mode."""

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
        mode="task",
    )

    # Mock session and context
    self.mock_session = Mock(spec=Session)
    self.mock_session.id = "session-123"
    self.mock_session.events = []

    self.mock_context = Mock(spec=InvocationContext)
    self.mock_context.session = self.mock_session
    self.mock_context.invocation_id = "invocation-123"
    self.mock_context.branch = "main"
    self.mock_context.isolation_scope = "task-1"

  def test_construct_message_parts_from_session_isolates_history(self):
    """Test history collection in task mode isolates to current scope."""
    # 1. Event outside task (oldest)
    event_outside_old = Mock(
        live_session_id=None,
        isolation_scope=None,
        author="user",
    )
    event_outside_old.get_function_calls.return_value = []
    event_outside_old.get_function_responses.return_value = []

    # 2. Trigger FC event (scope=None, but contains FC with id="task-1")
    trigger_fc = genai_types.FunctionCall(
        id="task-1", name="test_agent", args={"request": "start"}
    )
    trigger_part = Mock()
    trigger_part.text = "Trigger message"
    trigger_part.function_response = None

    trigger_event = Mock(
        live_session_id=None,
        isolation_scope=None,
        author="deal_agent",
    )
    trigger_event.get_function_calls.return_value = [trigger_fc]
    trigger_event.get_function_responses.return_value = []
    trigger_event.content = Mock()
    trigger_event.content.parts = [trigger_part]

    # 3. Event inside task (remote response)
    remote_part = Mock()
    remote_part.text = "remote task agent output"
    remote_part.function_response = None

    event_inside_remote = Mock(
        live_session_id=None,
        isolation_scope="task-1",
        author=self.agent.name,
        custom_metadata={
            A2A_METADATA_PREFIX + "response": True,
            A2A_METADATA_PREFIX + "context_id": "ctx-1",
        },
    )
    event_inside_remote.get_function_calls.return_value = []
    event_inside_remote.get_function_responses.return_value = []
    event_inside_remote.content = Mock()
    event_inside_remote.content.parts = [remote_part]

    # 4. Event inside task (user reply)
    user_part = Mock()
    user_part.text = "User reply"
    user_part.function_response = None

    event_inside_user = Mock(
        live_session_id=None,
        isolation_scope="task-1",
        author="user",
    )
    event_inside_user.get_function_calls.return_value = []
    event_inside_user.get_function_responses.return_value = []
    event_inside_user.content = Mock()
    event_inside_user.content.parts = [user_part]

    # 5. Event outside task (newer)
    event_outside_new = Mock(
        live_session_id=None,
        isolation_scope="task-2",
        author="other_agent",
    )
    event_outside_new.get_function_calls.return_value = []
    event_outside_new.get_function_responses.return_value = []

    # Session events (oldest to newest)
    self.mock_session.events = [
        event_outside_old,
        trigger_event,
        event_inside_remote,
        event_inside_user,
        event_outside_new,
    ]

    # Mock converter to return a real text Part
    def mock_converter(part):
      return _compat.make_text_part(getattr(part, "text", "default"))

    self.mock_genai_part_converter.side_effect = mock_converter

    with patch(
        "google.adk.agents.remote_a2a_agent._present_other_agent_message"
    ) as mock_present:
      mock_present.side_effect = lambda event: event

      parts, context_id = self.agent._construct_message_parts_from_session(
          self.mock_context
      )

      # Stateful resumption: should stop at event_inside_remote and only collect
      # event_inside_user
      assert len(parts) == 1
      assert _compat.part_text(parts[0]) == "User reply"
      assert context_id == "ctx-1"

  def test_construct_message_parts_from_session_first_turn(self):
    """Test history collection in task mode first turn (collects up to trigger FC)."""
    # 1. Event outside task (oldest)
    event_outside_old = Mock(
        live_session_id=None,
        isolation_scope=None,
        author="user",
    )
    event_outside_old.get_function_calls.return_value = []
    event_outside_old.get_function_responses.return_value = []

    # 2. Trigger FC event
    trigger_fc = genai_types.FunctionCall(
        id="task-1", name="test_agent", args={"request": "start"}
    )
    trigger_part = Mock()
    trigger_part.text = "Trigger message"
    trigger_part.function_response = None

    trigger_event = Mock(
        live_session_id=None,
        isolation_scope=None,
        author="deal_agent",
    )
    trigger_event.get_function_calls.return_value = [trigger_fc]
    trigger_event.get_function_responses.return_value = []
    trigger_event.content = Mock()
    trigger_event.content.parts = [trigger_part]

    # Session events (oldest to newest)
    self.mock_session.events = [
        event_outside_old,
        trigger_event,
    ]

    def mock_converter(part):
      return _compat.make_text_part(getattr(part, "text", "default"))

    self.mock_genai_part_converter.side_effect = mock_converter

    with patch(
        "google.adk.agents.remote_a2a_agent._present_other_agent_message"
    ) as mock_present:
      mock_present.side_effect = lambda event: event

      parts, context_id = self.agent._construct_message_parts_from_session(
          self.mock_context
      )

      # First turn: collects only trigger_event
      assert len(parts) == 1
      assert _compat.part_text(parts[0]) == "Trigger message"
      assert context_id is None

  def test_construct_message_parts_from_session_stateless_full_history(self):
    """Test full history for stateless agent in task mode when enabled."""
    self.agent._full_history_when_stateless = True

    # 1. Event outside task (oldest)
    event_outside_old = Mock(
        live_session_id=None,
        isolation_scope=None,
        author="user",
    )
    event_outside_old.get_function_calls.return_value = []
    event_outside_old.get_function_responses.return_value = []

    # 2. Trigger FC event
    trigger_fc = genai_types.FunctionCall(
        id="task-1", name="test_agent", args={"request": "start"}
    )
    trigger_part = Mock()
    trigger_part.text = "Trigger message"
    trigger_part.function_response = None

    trigger_event = Mock(
        live_session_id=None,
        isolation_scope=None,
        author="deal_agent",
    )
    trigger_event.get_function_calls.return_value = [trigger_fc]
    trigger_event.get_function_responses.return_value = []
    trigger_event.content = Mock()
    trigger_event.content.parts = [trigger_part]

    # 3. Event inside task (remote response) - STATELESS (no context_id)
    remote_part = Mock()
    remote_part.text = "remote task agent output"
    remote_part.function_response = None

    event_inside_remote = Mock(
        live_session_id=None,
        isolation_scope="task-1",
        author=self.agent.name,
        custom_metadata={
            A2A_METADATA_PREFIX + "response": True,
            # NO context_id
        },
    )
    event_inside_remote.get_function_calls.return_value = []
    event_inside_remote.get_function_responses.return_value = []
    event_inside_remote.content = Mock()
    event_inside_remote.content.parts = [remote_part]

    # 4. Event inside task (user reply)
    user_part = Mock()
    user_part.text = "User reply"
    user_part.function_response = None

    event_inside_user = Mock(
        live_session_id=None,
        isolation_scope="task-1",
        author="user",
    )
    event_inside_user.get_function_calls.return_value = []
    event_inside_user.get_function_responses.return_value = []
    event_inside_user.content = Mock()
    event_inside_user.content.parts = [user_part]

    # Session events (oldest to newest)
    self.mock_session.events = [
        event_outside_old,
        trigger_event,
        event_inside_remote,
        event_inside_user,
    ]

    def mock_converter(part):
      return _compat.make_text_part(getattr(part, "text", "default"))

    self.mock_genai_part_converter.side_effect = mock_converter

    with patch(
        "google.adk.agents.remote_a2a_agent._present_other_agent_message"
    ) as mock_present:
      mock_present.side_effect = lambda event: event

      parts, context_id = self.agent._construct_message_parts_from_session(
          self.mock_context
      )

      # Stateless resumption with full history enabled:
      # Should NOT stop at event_inside_remote.
      # Should collect: event_inside_user, event_inside_remote, trigger_event.
      assert len(parts) == 3
      assert _compat.part_text(parts[0]) == "Trigger message"
      assert _compat.part_text(parts[1]) == "remote task agent output"
      assert _compat.part_text(parts[2]) == "User reply"
      assert context_id is None

  def test_construct_message_parts_from_session_filters_sibling_fcs(self):
    """Test that sibling FunctionCalls from the coordinator are filtered out."""
    trigger_fc = genai_types.FunctionCall(
        id="task-1", name="test_agent", args={"request": "start"}
    )
    sibling_fc = genai_types.FunctionCall(
        id="sibling-2", name="other_tool", args={"other": "data"}
    )

    trigger_part = Mock()
    trigger_part.function_call = trigger_fc
    trigger_part.function_response = None
    trigger_part.text = None

    sibling_part = Mock()
    sibling_part.function_call = sibling_fc
    sibling_part.function_response = None
    sibling_part.text = None

    trigger_event = Mock(
        live_session_id=None,
        isolation_scope=None,
        author="coordinator",
    )
    trigger_event.get_function_calls.return_value = [trigger_fc, sibling_fc]
    trigger_event.get_function_responses.return_value = []
    trigger_event.content = Mock()
    trigger_event.content.parts = [trigger_part, sibling_part]

    self.mock_session.events = [trigger_event]

    def mock_converter(part):
      return _compat.make_text_part(f"FC:{part.function_call.id}")

    self.mock_genai_part_converter.side_effect = mock_converter

    with patch(
        "google.adk.agents.remote_a2a_agent._present_other_agent_message"
    ) as mock_present:
      mock_present.side_effect = lambda event: event

      parts, context_id = self.agent._construct_message_parts_from_session(
          self.mock_context
      )

      assert len(parts) == 1
      assert _compat.part_text(parts[0]) == "FC:task-1"
      assert context_id is None

  def test_construct_message_parts_from_session_foreign_function_response_converted(
      self,
  ):
    """Test that foreign function responses ARE converted to text in task mode."""
    # Mock event with a function response
    mock_fr = genai_types.FunctionResponse(
        id="fc-1", name="tool_1", response={"result": "done"}
    )
    mock_part = Mock()
    mock_part.function_response = mock_fr
    mock_part.text = None

    mock_content = Mock()
    mock_content.parts = [mock_part]

    mock_event = Mock()
    mock_event.isolation_scope = "task-1"
    mock_event.author = "user"
    mock_event.content = mock_content
    mock_event.get_function_calls.return_value = []
    mock_event.get_function_responses.return_value = []

    # Trigger event
    trigger_fc = genai_types.FunctionCall(
        id="task-1", name="test_agent", args={"request": "start"}
    )
    trigger_part = Mock()
    trigger_part.text = "Trigger message"
    trigger_part.function_response = None
    trigger_event = Mock(
        live_session_id=None,
        isolation_scope=None,
        author="deal_agent",
    )
    trigger_event.get_function_calls.return_value = [trigger_fc]
    trigger_event.get_function_responses.return_value = []
    trigger_event.content = Mock()
    trigger_event.content.parts = [trigger_part]

    self.mock_session.events = [trigger_event, mock_event]

    # Setup converter to return distinguishable parts
    def mock_converter(part):
      return _compat.make_text_part(getattr(part, "text", "default"))

    self.mock_genai_part_converter.side_effect = mock_converter

    with patch(
        "google.adk.agents.remote_a2a_agent._present_other_agent_message"
    ) as mock_present:
      mock_present.side_effect = lambda event: event

      parts, _ = self.agent._construct_message_parts_from_session(
          self.mock_context
      )

      assert len(parts) == 2
      assert _compat.part_text(parts[0]) == "Trigger message"
      # The foreign FR should be converted to text
      expected_text = 'Tool tool_1 returned: {"result": "done"}'
      assert _compat.part_text(parts[1]) == expected_text
      # Check that converter was not called for the foreign FR
      self.mock_genai_part_converter.assert_called_once_with(trigger_part)

  def test_construct_message_parts_from_session_non_foreign_fr_not_converted_when_fc_before_break(
      self,
  ):
    """Test that non-foreign FR is NOT converted to text even if its FC was before context_id break."""
    # 1. Trigger event (matching task-1)
    trigger_fc = genai_types.FunctionCall(
        id="task-1", name="test_agent", args={"request": "start"}
    )
    trigger_part = Mock()
    trigger_part.text = "Trigger message"
    trigger_part.function_response = None
    trigger_event = Mock(
        live_session_id=None, isolation_scope=None, author="deal_agent"
    )
    trigger_event.get_function_calls.return_value = [trigger_fc]
    trigger_event.get_function_responses.return_value = []
    trigger_event.content = Mock()
    trigger_event.content.parts = [trigger_part]

    # 2. Remote agent event (FC: input request) - this will be before the break
    remote_fc = genai_types.FunctionCall(
        id="fc-input-req",
        name="user_input_tool",
        args={"prompt": "enter value"},
    )
    remote_part = Mock()
    remote_part.function_call = remote_fc
    remote_part.text = "Please enter value"
    remote_event = Mock(
        live_session_id=None, isolation_scope="task-1", author="test_agent"
    )
    remote_event.get_function_calls.return_value = [remote_fc]
    remote_event.get_function_responses.return_value = []
    remote_event.content = Mock()
    remote_event.content.parts = [remote_part]
    # This event has context_id, so it will trigger the break in history walk
    remote_event.metadata = {A2A_METADATA_PREFIX + "context_id": "context-old"}

    # 3. User event (FR: user input response) - this will be after the break
    user_fr = genai_types.FunctionResponse(
        id="fc-input-req", name="user_input_tool", response={"result": "my-val"}
    )
    user_part = Mock()
    user_part.function_response = user_fr
    user_part.text = None
    user_event = Mock(
        live_session_id=None, isolation_scope="task-1", author="user"
    )
    user_event.get_function_calls.return_value = []
    user_event.get_function_responses.return_value = [user_fr]
    user_event.content = Mock()
    user_event.content.parts = [user_part]

    # Session events order (chronological): trigger, remote (break), user
    self.mock_session.events = [trigger_event, remote_event, user_event]

    # Setup converter to return distinguishable parts
    mock_a2a_part = Mock()
    self.mock_genai_part_converter.return_value = [mock_a2a_part]

    with (
        patch(
            "google.adk.agents.remote_a2a_agent._present_other_agent_message"
        ) as mock_present,
        patch(
            "google.adk.agents.remote_a2a_agent._compat.part_metadata"
        ) as mock_part_metadata,
    ):
      mock_present.side_effect = lambda event: event
      mock_part_metadata.return_value = {}

      parts, _ = self.agent._construct_message_parts_from_session(
          self.mock_context
      )

      # We expect the FR to NOT be converted to text, so the converter is called
      # and we get the mock_a2a_part back.
      assert len(parts) == 1
      assert parts[0] == mock_a2a_part
      self.mock_genai_part_converter.assert_called_once_with(user_part)


class TestRemoteA2aAgentStreamingArtifactChunks:
  """Regression tests for chunked artifact streams."""

  def setup_method(self):
    """Setup test fixtures."""
    self.agent = RemoteA2aAgent(
        name="test_agent",
        agent_card=create_test_agent_card(),
    )
    self.mock_context = Mock(spec=InvocationContext)
    self.mock_context.invocation_id = "invocation-123"
    self.mock_context.branch = "main"

  @pytest.mark.asyncio
  async def test_chunked_artifact_stream_emits_each_part_exactly_once(self):
    """A two-chunk artifact stream renders its parts without duplication."""
    chunk1 = _make_artifact_chunk("Hello, ", append=False, last_chunk=False)
    chunk2 = _make_artifact_chunk("world!", append=True, last_chunk=True)
    # (task, update) pairs as the client stream yields them: the task carries
    # the artifact parts accumulated so far.
    stream = [
        (_make_accumulated_task(["Hello, "]), chunk1),
        (_make_accumulated_task(["Hello, ", "world!"]), chunk2),
    ]

    rendered = []
    events = []
    for pair in stream:
      event = await self.agent._handle_a2a_response(pair, self.mock_context)
      events.append(event)
      if event and event.content and event.content.parts:
        rendered.extend(part.text for part in event.content.parts if part.text)

    assert "".join(rendered) == "Hello, world!"
    assert events[0].partial is True
    assert events[1].partial is False

  @pytest.mark.asyncio
  async def test_artifact_update_without_parts_is_ignored(self):
    """An artifact update carrying no parts must not emit a spurious event."""
    update = TaskArtifactUpdateEvent(
        task_id="task-123",
        context_id="context-123",
        append=False,
        last_chunk=True,
        artifact=_compat.make_artifact(artifact_id="artifact-1", parts=[]),
    )
    task = _make_accumulated_task(["already streamed"])

    result = await self.agent._handle_a2a_response(
        (task, update), self.mock_context
    )

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
    # The response sanitizer always runs now; a bare Mock content is not
    # iterable, and there is nothing to sanitize here (no function calls), so
    # give it None to make the sanitizer a no-op.
    mock_latest_event.content = None
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

    update = _make_artifact_chunk("chunk", append=False, last_chunk=True)

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
          (mock_a2a_task, update), self.mock_context
      )

      assert result == mock_event
      mock_convert.assert_called_once()
      # Only the parts carried by this update are converted, not the
      # accumulated task.
      converted_message = mock_convert.call_args[0][0]
      assert list(converted_message.parts) == list(update.artifact.parts)
      # Check that metadata was added
      assert result.custom_metadata is not None
      assert A2A_METADATA_PREFIX + "task_id" in result.custom_metadata
      assert A2A_METADATA_PREFIX + "context_id" in result.custom_metadata

  @pytest.mark.asyncio
  async def test_handle_a2a_response_with_appended_artifact_chunk(self):
    """An appended (middle) artifact chunk emits only its own parts."""
    mock_a2a_task = Mock(spec=A2ATask)
    mock_a2a_task.id = "task-123"
    mock_a2a_task.context_id = "context-123"

    update = _make_artifact_chunk("middle", append=True, last_chunk=False)

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
          (mock_a2a_task, update), self.mock_context
      )

      assert result == mock_event
      assert result.partial is True
      converted_message = mock_convert.call_args[0][0]
      assert list(converted_message.parts) == list(update.artifact.parts)


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
    self.mock_config.card_request_interceptors = None

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
    self.mock_config.card_request_interceptors = None
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
  async def test_legacy_message_converter_returns_none_artifact_update(self):
    """Legacy handler must not crash when message converter returns None for artifact update."""
    mock_task = Mock(spec=A2ATask)
    mock_task.id = "task-123"
    mock_task.context_id = None

    update = _make_artifact_chunk("chunk", append=False, last_chunk=True)

    with patch(
        "google.adk.agents.remote_a2a_agent.convert_a2a_message_to_event"
    ) as mock_convert:
      mock_convert.return_value = None

      result = await self.legacy_agent._handle_a2a_response(
          (mock_task, update), self.mock_context
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
          # _ensure_resolved now returns the client to use for the run.
          self.agent._ensure_resolved.return_value = mock_a2a_client

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
  async def test_run_async_impl_closes_stream_when_abandoned(self):
    """The A2A stream is closed when the caller stops consuming early."""
    with patch.object(self.agent, "_ensure_resolved") as mock_ensure_resolved:
      with patch.object(
          self.agent, "_create_a2a_request_for_user_function_response"
      ) as mock_create_func:
        mock_create_func.return_value = None

        with patch.object(
            self.agent, "_construct_message_parts_from_session"
        ) as mock_construct:
          mock_a2a_part = _compat.make_text_part("test")
          mock_construct.return_value = ([mock_a2a_part], "context-123")

          mock_a2a_client = create_autospec(spec=A2AClient, instance=True)
          mock_send_message = AsyncMock()
          mock_send_message.__aiter__.return_value = [
              _make_stream_message(
                  A2AMessage(
                      message_id=message_id,
                      role=_compat.ROLE_USER,
                      parts=[mock_a2a_part],
                  )
              )
              for message_id in ("m1", "m2")
          ]
          mock_a2a_client.send_message.return_value = mock_send_message
          self.agent._a2a_client = mock_a2a_client
          mock_ensure_resolved.return_value = mock_a2a_client

          mock_event = Event(
              author=self.agent.name,
              invocation_id=self.mock_context.invocation_id,
              branch=self.mock_context.branch,
          )

          with patch.object(self.agent, "_handle_a2a_response") as mock_handle:
            mock_handle.return_value = mock_event

            with patch(
                "google.adk.agents.remote_a2a_agent.build_a2a_request_log"
            ):
              with patch(
                  "google.adk.agents.remote_a2a_agent.build_a2a_response_log"
              ):
                with patch(
                    "google.adk.a2a._compat.a2a_to_dict",
                    return_value={"k": "v"},
                ):
                  agen = self.agent._run_async_impl(self.mock_context)
                  await agen.__anext__()
                  await agen.aclose()

          mock_send_message.aclose.assert_awaited_once()

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
          # _ensure_resolved now returns the client to use for the run.
          self.agent._ensure_resolved.return_value = mock_a2a_client

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
  async def test_run_async_impl_task_mode_rejects_missing_trigger_fc(self):
    """Test _run_async_impl raises ValueError when isolation_scope has no matching trigger FC."""
    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card=self.agent_card,
        mode="task",
    )
    self.mock_context.isolation_scope = "workflow_path/node@run1"
    self.mock_session.events = []

    with pytest.raises(
        ValueError, match="could not find the triggering FunctionCall"
    ):
      _ = [e async for e in agent._run_async_impl(self.mock_context)]

  @pytest.mark.asyncio
  async def test_run_async_impl_task_mode_releases_control_on_init_failure(
      self,
  ):
    """Test _run_async_impl in task mode releases control on initialization failure."""
    from google.adk.agents.llm.task._finish_task_tool import FINISH_TASK_TOOL_NAME

    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card=self.agent_card,
        mode="task",
    )
    self.mock_context.agent_states = {}
    self.mock_context.end_of_agents = {}
    self.mock_context.isolation_scope = "task-1"
    self.mock_session.events = [_make_dummy_task_trigger_event()]

    def set_agent_state_side_effect(agent_name, **kwargs):
      if kwargs.get("end_of_agent"):
        self.mock_context.end_of_agents[agent_name] = True
      else:
        self.mock_context.end_of_agents.pop(agent_name, None)

    self.mock_context.set_agent_state.side_effect = set_agent_state_side_effect

    with patch.object(agent, "_ensure_resolved") as mock_ensure:
      mock_ensure.side_effect = Exception("Init failed")
      events = []
      async for event in agent._run_async_impl(self.mock_context):
        events.append(event)

      assert len(events) == 3
      assert "Failed to initialize remote A2A agent" in events[0].error_message
      assert (
          events[1].content.parts[0].function_response.name
          == FINISH_TASK_TOOL_NAME
      )
      assert events[2].actions.end_of_agent is True

  @pytest.mark.asyncio
  async def test_run_async_impl_task_mode_releases_control_on_empty_parts(self):
    """Test _run_async_impl in task mode releases control when message parts are empty."""
    from google.adk.agents.llm.task._finish_task_tool import FINISH_TASK_TOOL_NAME

    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card=self.agent_card,
        mode="task",
    )
    self.mock_context.agent_states = {}
    self.mock_context.end_of_agents = {}
    self.mock_context.isolation_scope = "task-1"
    self.mock_session.events = [_make_dummy_task_trigger_event()]

    def set_agent_state_side_effect(agent_name, **kwargs):
      if kwargs.get("end_of_agent"):
        self.mock_context.end_of_agents[agent_name] = True
      else:
        self.mock_context.end_of_agents.pop(agent_name, None)

    self.mock_context.set_agent_state.side_effect = set_agent_state_side_effect

    with patch.object(agent, "_ensure_resolved"):
      with patch.object(
          agent, "_create_a2a_request_for_user_function_response"
      ) as mock_create_func:
        mock_create_func.return_value = None
        with patch.object(
            agent, "_construct_message_parts_from_session"
        ) as mock_construct:
          mock_construct.return_value = ([], None)

          events = []
          async for event in agent._run_async_impl(self.mock_context):
            events.append(event)

          assert len(events) == 3
          assert events[0].content is not None
          assert (
              events[1].content.parts[0].function_response.name
              == FINISH_TASK_TOOL_NAME
          )
          assert events[2].actions.end_of_agent is True

  @pytest.mark.asyncio
  async def test_run_async_impl_a2a_http_error_in_task_mode(self):
    """Test _run_async_impl task mode hand-back when A2A send_message raises HTTP error."""
    from google.adk.agents.llm.task._finish_task_tool import FINISH_TASK_ERROR_RESULT
    from google.adk.agents.llm.task._finish_task_tool import FINISH_TASK_TOOL_NAME

    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card=self.agent_card,
        genai_part_converter=self.mock_genai_part_converter,
        a2a_part_converter=self.mock_a2a_part_converter,
        mode="task",
    )

    HTTPErrorClass = _compat.A2A_HTTP_ERRORS[0]
    if _compat.IS_A2A_V1:
      error_instance = HTTPErrorClass("HTTP Error 500")
    else:
      error_instance = HTTPErrorClass(
          status_code=500, message="Internal Server Error"
      )

    self.mock_context.agent_states = {}
    self.mock_context.end_of_agents = {}
    self.mock_context.isolation_scope = "task-1"
    self.mock_session.events = [_make_dummy_task_trigger_event()]

    def set_agent_state_side_effect(agent_name, **kwargs):
      if kwargs.get("end_of_agent"):
        self.mock_context.end_of_agents[agent_name] = True
      else:
        self.mock_context.end_of_agents.pop(agent_name, None)

    self.mock_context.set_agent_state.side_effect = set_agent_state_side_effect

    # Mock _ensure_resolved to return mock client
    mock_a2a_client = Mock()
    mock_send_message = AsyncMock()
    mock_send_message.__aiter__.side_effect = error_instance
    mock_a2a_client.send_message.return_value = mock_send_message
    mock_ensure_resolved = AsyncMock(return_value=mock_a2a_client)

    with patch.object(agent, "_ensure_resolved", mock_ensure_resolved):
      with patch.object(
          agent, "_create_a2a_request_for_user_function_response"
      ) as mock_create_func:
        mock_create_func.return_value = None

        with patch.object(
            agent, "_construct_message_parts_from_session"
        ) as mock_construct:
          mock_a2a_part = _compat.make_text_part("test")
          mock_construct.return_value = (
              [mock_a2a_part],
              "context-123",
          )

          agent._a2a_client = mock_a2a_client

          with patch(
              "google.adk.agents.remote_a2a_agent.build_a2a_request_log"
          ) as mock_req_log:
            mock_req_log.return_value = "Mock request log"

            events = []
            async for event in agent._run_async_impl(self.mock_context):
              events.append(event)

            # In task mode, it should yield:
            # 1. The initial error event (Event with error_message)
            # 2. The finish_task error event (Event with user role and finish_task FR)
            # 3. The agent state event (Event with end_of_agent=True)
            assert len(events) == 3

            assert "A2A request failed" in events[0].error_message

            # The second event should be the finish_task error event
            assert events[1].content is not None
            fr = events[1].content.parts[0].function_response
            assert fr is not None
            assert fr.name == FINISH_TASK_TOOL_NAME
            assert fr.response == {"result": FINISH_TASK_ERROR_RESULT}

            # The third event should be the agent state event
            assert self.mock_context.end_of_agents[agent.name] is True

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
          # _ensure_resolved now returns the client to use for the run.
          agent._ensure_resolved.return_value = mock_a2a_client

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
          # _ensure_resolved now returns the client to use for the run.
          self.agent._ensure_resolved.return_value = mock_a2a_client

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
          # _ensure_resolved now returns the client to use for the run.
          self.agent._ensure_resolved.return_value = mock_a2a_client

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

  @pytest.mark.asyncio
  async def test_execute_before_card_request_interceptors_none(
      self, mock_context
  ):
    http_kwargs = await execute_before_card_request_interceptors(
        None, mock_context
    )
    assert http_kwargs is None

  @pytest.mark.asyncio
  async def test_execute_before_card_request_interceptors_merges(
      self, mock_context
  ):
    interceptor1 = CardRequestInterceptor(
        before_request=AsyncMock(
            return_value=A2aCardRequestConfig(
                headers={"X-Common": "a", "X-A": "1"}
            )
        )
    )
    interceptor2 = CardRequestInterceptor(
        before_request=AsyncMock(
            return_value=A2aCardRequestConfig(
                headers={"X-Common": "b", "X-B": "2"}
            )
        )
    )

    http_kwargs = await execute_before_card_request_interceptors(
        [interceptor1, interceptor2], mock_context
    )

    assert http_kwargs == {"headers": {"X-Common": "b", "X-A": "1", "X-B": "2"}}

  @pytest.mark.asyncio
  async def test_execute_before_card_request_interceptors_skips_none_provider(
      self, mock_context
  ):
    interceptor = CardRequestInterceptor(before_request=None)
    http_kwargs = await execute_before_card_request_interceptors(
        [interceptor], mock_context
    )
    assert http_kwargs is None


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


class TestFindFinishTaskArgsFromHistory:
  """Test _find_finish_task_args_from_history helper function."""

  def test_find_finish_task_args_no_filtering(self):
    # Session with multiple events
    event1 = Mock(spec=Event)
    event1.isolation_scope = "task-1"
    event1.get_function_calls.return_value = [
        genai_types.FunctionCall(
            id="fc-1", name="finish_task", args={"result": "task-1-done"}
        )
    ]

    event2 = Mock(spec=Event)
    event2.isolation_scope = "task-2"
    event2.get_function_calls.return_value = [
        genai_types.FunctionCall(
            id="fc-2", name="finish_task", args={"result": "task-2-done"}
        )
    ]

    session = Mock(spec=Session)
    session.events = [event1, event2]

    # Without isolation_scope, it should return the latest (event2)
    args = remote_a2a_agent._find_finish_task_args_from_history(session)
    assert args == {"result": "task-2-done"}

  def test_find_finish_task_args_with_filtering(self):
    # Session with multiple events
    event1 = Mock(spec=Event)
    event1.isolation_scope = "task-1"
    event1.get_function_calls.return_value = [
        genai_types.FunctionCall(
            id="fc-1", name="finish_task", args={"result": "task-1-done"}
        )
    ]

    event2 = Mock(spec=Event)
    event2.isolation_scope = "task-2"
    event2.get_function_calls.return_value = [
        genai_types.FunctionCall(
            id="fc-2", name="finish_task", args={"result": "task-2-done"}
        )
    ]

    session = Mock(spec=Session)
    session.events = [event1, event2]

    # With isolation_scope="task-1", it should return event1's args
    args = remote_a2a_agent._find_finish_task_args_from_history(
        session, "task-1"
    )
    assert args == {"result": "task-1-done"}

    # With isolation_scope="task-3", it should return None
    args = remote_a2a_agent._find_finish_task_args_from_history(
        session, "task-3"
    )
    assert args is None

  def test_find_finish_task_args_with_matching_fr_id(self):
    # Session with multiple finish_task FCs in the same scope
    event1 = Mock(spec=Event)
    event1.isolation_scope = "task-1"
    event1.get_function_calls.return_value = [
        genai_types.FunctionCall(
            id="fc-1", name="finish_task", args={"result": "first-attempt"}
        )
    ]

    event2 = Mock(spec=Event)
    event2.isolation_scope = "task-1"
    event2.get_function_calls.return_value = [
        genai_types.FunctionCall(
            id="fc-2", name="finish_task", args={"result": "second-attempt"}
        )
    ]

    session = Mock(spec=Session)
    session.events = [event1, event2]

    # Create a FR event with matching ID "fc-1" (the older one)
    fr_event = Mock(spec=Event)
    fr_event.get_function_responses.return_value = [
        genai_types.FunctionResponse(
            id="fc-1", name="finish_task", response={"result": "SUCCESS"}
        )
    ]

    # Should return event1's args because it matches fc-1, even though event2 is
    # newer
    args = remote_a2a_agent._find_finish_task_args_from_history(
        session, "task-1", completed_fr_event=fr_event
    )
    assert args == {"result": "first-attempt"}

  def test_find_finish_task_args_with_non_matching_fr_id(self):
    # Session with a finish_task FC
    event1 = Mock(spec=Event)
    event1.isolation_scope = "task-1"
    event1.get_function_calls.return_value = [
        genai_types.FunctionCall(
            id="fc-1", name="finish_task", args={"result": "done"}
        )
    ]

    session = Mock(spec=Session)
    session.events = [event1]

    # Create a FR event with a non-matching ID "fc-different"
    fr_event = Mock(spec=Event)
    fr_event.get_function_responses.return_value = [
        genai_types.FunctionResponse(
            id="fc-different",
            name="finish_task",
            response={"result": "SUCCESS"},
        )
    ]

    # Should return None because ID doesn't match
    args = remote_a2a_agent._find_finish_task_args_from_history(
        session, "task-1", completed_fr_event=fr_event
    )
    assert args is None


class _TestSingleFieldOutput(BaseModel):
  result: str


class TestRemoteA2aAgentTaskModeOutputUnwrapping:
  """Test that RemoteA2aAgent correctly unwraps task output based on schema."""

  @pytest.mark.parametrize(
      "output_schema, args, expected_output",
      [
          # Case 1: output_schema is None (default). Should NOT unwrap.
          (None, {"result": "hello"}, {"result": "hello"}),
          # Case 2: output_schema is primitive (str). Should unwrap.
          (str, {"result": "hello"}, "hello"),
          # Case 3: output_schema is BaseModel with 'result' field. Should NOT unwrap.
          (_TestSingleFieldOutput, {"result": "hello"}, {"result": "hello"}),
          # Case 4: output_schema is primitive (int). Should unwrap.
          (int, {"result": 42}, 42),
          # Case 5: Custom schema with multiple fields. Should NOT unwrap.
          (
              dict,
              {"result": "hello", "other": "world"},
              {"result": "hello", "other": "world"},
          ),
      ],
      ids=[
          "default_schema_no_unwrap",
          "primitive_str_unwrap",
          "basemodel_single_field_no_unwrap",
          "primitive_int_unwrap",
          "dict_no_unwrap",
      ],
  )
  @pytest.mark.asyncio
  async def test_output_unwrapping(self, output_schema, args, expected_output):
    agent_card = create_test_agent_card()
    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card=agent_card,
        mode="task",
        output_schema=output_schema,
    )

    mock_context = Mock(spec=InvocationContext)
    mock_context.session = Mock(spec=Session)
    mock_context.session.events = [_make_dummy_task_trigger_event()]
    mock_context.session.state = {}
    mock_context.agent_states = {}
    mock_context.end_of_agents = {}
    mock_context.isolation_scope = "task-1"
    mock_context.invocation_id = "invocation-123"
    mock_context.branch = "main"

    # Mock a2a client as regular Mock
    mock_a2a_client = Mock()
    mock_send_message = AsyncMock()
    mock_ensure_resolved = AsyncMock(return_value=mock_a2a_client)

    # Mock _ensure_resolved to return our mock client
    with patch.object(agent, "_ensure_resolved", mock_ensure_resolved):
      # Mock _construct_message_parts_from_session to avoid early exit
      with patch.object(
          agent, "_construct_message_parts_from_session"
      ) as mock_construct:
        mock_a2a_part = _compat.make_text_part("test_message")
        mock_construct.return_value = ([mock_a2a_part], "context-123")

        # Use a real A2AMessage wrapped in StreamResponse for the mock stream
        mock_a2a_message = A2AMessage(
            message_id="m1",
            role=_compat.ROLE_USER,
            parts=[mock_a2a_part],
            context_id="context-123",
        )
        mock_response = _make_stream_message(mock_a2a_message)
        mock_send_message.__aiter__.return_value = [mock_response]
        mock_a2a_client.send_message.return_value = mock_send_message
        agent._a2a_client = mock_a2a_client

        # Mock _handle_a2a_response to return a success finish_task FR event
        mock_event = Mock(spec=Event)
        mock_event.custom_metadata = {}
        mock_fr = genai_types.FunctionResponse(
            id="ft-1",
            name="finish_task",
            response={"result": "Task completed."},
        )
        mock_event.get_function_responses.return_value = [mock_fr]
        mock_event.get_function_calls.return_value = []

        with patch.object(
            agent, "_handle_a2a_response", new_callable=AsyncMock
        ) as mock_handle:
          mock_handle.return_value = mock_event

          # Mock _find_finish_task_args_from_history to return our test args
          with patch(
              "google.adk.agents.remote_a2a_agent._find_finish_task_args_from_history"
          ) as mock_find_args:
            mock_find_args.return_value = args

            events = []
            async for ev in agent._run_async_impl(mock_context):
              events.append(ev)

            # We expect at least the success event
            assert len(events) >= 1
            # The first yielded event should be the one modified with output
            success_event = events[0]
            assert success_event.output == expected_output

  @pytest.mark.asyncio
  async def test_output_unwrapping_integration_with_history(self):
    """Test that output unwrapping works when driving real events through the stream.

    This verifies that the finish_task FunctionCall event is correctly
    placed in history before the FunctionResponse event is processed,
    allowing _find_finish_task_args_from_history to find it.
    """
    output_schema = str
    expected_output = "hello"
    agent_card = create_test_agent_card()
    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card=agent_card,
        mode="task",
        output_schema=output_schema,
    )

    mock_context = Mock(spec=InvocationContext)
    mock_context.session = Mock(spec=Session)
    mock_context.session.events = [_make_dummy_task_trigger_event()]
    mock_context.session.state = {}
    mock_context.agent_states = {}
    mock_context.end_of_agents = {}
    mock_context.isolation_scope = "task-1"
    mock_context.invocation_id = "invocation-123"
    mock_context.branch = "main"

    # Mock a2a client
    mock_a2a_client = Mock()
    mock_send_message = AsyncMock()
    mock_ensure_resolved = AsyncMock(return_value=mock_a2a_client)

    # Prepare real A2A messages for FC and FR
    # 1. FunctionCall for finish_task
    fc_data = {
        "name": "finish_task",
        "args": {"result": expected_output},
        "id": "ft-1",
    }
    fc_meta = {
        "adk_type": "function_call",
    }
    fc_part = _compat.make_data_part(data=fc_data, metadata=fc_meta)
    fc_message = A2AMessage(
        message_id="m-fc",
        role=_compat.ROLE_AGENT,
        parts=[fc_part],
        context_id="context-123",
    )

    # 2. FunctionResponse for finish_task
    fr_data = {
        "name": "finish_task",
        "response": {"result": FINISH_TASK_SUCCESS_RESULT},
        "id": "ft-1",
    }
    fr_meta = {
        "adk_type": "function_response",
    }
    fr_part = _compat.make_data_part(data=fr_data, metadata=fr_meta)
    fr_message = A2AMessage(
        message_id="m-fr",
        role=_compat.ROLE_USER,
        parts=[fr_part],
        context_id="context-123",
    )

    # Mock the stream to yield FC then FR
    stream_fc = _make_stream_message(fc_message)
    stream_fr = _make_stream_message(fr_message)
    mock_send_message.__aiter__.return_value = [stream_fc, stream_fr]
    mock_a2a_client.send_message.return_value = mock_send_message
    agent._a2a_client = mock_a2a_client

    with patch.object(agent, "_ensure_resolved", mock_ensure_resolved):
      # We do NOT mock _handle_a2a_response or _find_finish_task_args_from
      # history
      with patch.object(
          agent, "_construct_message_parts_from_session"
      ) as mock_construct:
        mock_dummy_part = _compat.make_text_part("dummy_input")
        mock_construct.return_value = ([mock_dummy_part], "context-123")

        events = []
        async for ev in agent._run_async_impl(mock_context):
          events.append(ev)
          # Simulate the runner setting isolation_scope and appending to history
          if ev.isolation_scope is None:
            ev.isolation_scope = mock_context.isolation_scope
          mock_context.session.events.append(ev)

        # We expect at least the FC event and the FR event
        assert len(events) >= 2

        # Find the FR event (it should have output populated)
        fr_event = None
        for ev in events:
          if ev.get_function_responses():
            fr_event = ev
            break

        assert fr_event is not None
        assert fr_event.output == expected_output


class TestRemoteA2aAgentTaskModeFailurePropagation:
  """Test that RemoteA2aAgent propagates task failures in task mode."""

  @pytest.mark.asyncio
  async def test_fails_on_task_state_failed(self):
    agent_card = create_test_agent_card()
    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card=agent_card,
        mode="task",
    )

    mock_context = Mock(spec=InvocationContext)
    mock_context.session = Mock(spec=Session)
    mock_context.session.events = [_make_dummy_task_trigger_event()]
    mock_context.session.state = {}
    mock_context.agent_states = {}
    mock_context.end_of_agents = {}
    mock_context.isolation_scope = "task-1"
    mock_context.invocation_id = "invocation-123"
    mock_context.branch = "main"

    def set_agent_state_side_effect(agent_name, **kwargs):
      if kwargs.get("end_of_agent"):
        mock_context.end_of_agents[agent_name] = True
      else:
        mock_context.end_of_agents.pop(agent_name, None)

    mock_context.set_agent_state.side_effect = set_agent_state_side_effect

    # Mock a2a client as regular Mock
    mock_a2a_client = Mock()
    mock_send_message = AsyncMock()
    mock_ensure_resolved = AsyncMock(return_value=mock_a2a_client)

    with patch.object(agent, "_ensure_resolved", mock_ensure_resolved):
      with patch.object(
          agent, "_construct_message_parts_from_session"
      ) as mock_construct:
        mock_a2a_part = _compat.make_text_part("test_message")
        mock_construct.return_value = ([mock_a2a_part], "context-123")

        error_message = A2AMessage(
            message_id="err-msg-1",
            role=_compat.ROLE_AGENT,
            parts=[_compat.make_text_part("Simulated remote task failure")],
            context_id="context-123",
        )
        task_status = A2ATaskStatus(
            state=_compat.TS_FAILED,
            message=error_message,
        )
        failed_task = A2ATask(
            id="task-1",
            context_id="context-123",
            status=task_status,
        )

        mock_response = _make_stream_task(failed_task)
        mock_send_message.__aiter__.return_value = [mock_response]
        mock_a2a_client.send_message.return_value = mock_send_message
        agent._a2a_client = mock_a2a_client

        mock_event = Event(
            author=agent.name,
            invocation_id=mock_context.invocation_id,
            branch=mock_context.branch,
            content=genai_types.Content(
                role="model",
                parts=[
                    genai_types.Part.from_text(
                        text="Simulated remote task failure"
                    )
                ],
            ),
        )

        with patch.object(
            agent, "_handle_a2a_response", new_callable=AsyncMock
        ) as mock_handle:
          mock_handle.return_value = mock_event

          events = []
          async for ev in agent._run_async_impl(mock_context):
            events.append(ev)

          assert len(events) == 4
          assert events[0] == mock_event
          assert (
              events[1].error_message
              == "Remote A2A task failed: Simulated remote task failure"
          )
          # Verify finish_task event
          assert (
              events[2].content.parts[0].function_response.name
              == FINISH_TASK_TOOL_NAME
          )
          assert events[2].content.parts[0].function_response.response == {
              "result": FINISH_TASK_ERROR_RESULT
          }
          assert events[3].actions.end_of_agent is True

          mock_context.set_agent_state.assert_called_once_with(
              agent.name, end_of_agent=True
          )

  @pytest.mark.asyncio
  async def test_completes_on_task_state_canceled(self):
    agent_card = create_test_agent_card()
    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card=agent_card,
        mode="task",
    )

    mock_context = Mock(spec=InvocationContext)
    mock_context.session = Mock(spec=Session)
    mock_context.session.events = [_make_dummy_task_trigger_event()]
    mock_context.session.state = {}
    mock_context.agent_states = {}
    mock_context.end_of_agents = {}
    mock_context.isolation_scope = "task-1"
    mock_context.invocation_id = "invocation-123"
    mock_context.branch = "main"

    def set_agent_state_side_effect(agent_name, **kwargs):
      if kwargs.get("end_of_agent"):
        mock_context.end_of_agents[agent_name] = True
      else:
        mock_context.end_of_agents.pop(agent_name, None)

    mock_context.set_agent_state.side_effect = set_agent_state_side_effect

    mock_a2a_client = Mock()
    mock_send_message = AsyncMock()
    mock_ensure_resolved = AsyncMock(return_value=mock_a2a_client)

    with patch.object(agent, "_ensure_resolved", mock_ensure_resolved):
      with patch.object(
          agent, "_construct_message_parts_from_session"
      ) as mock_construct:
        mock_a2a_part = _compat.make_text_part("test_message")
        mock_construct.return_value = ([mock_a2a_part], "context-123")

        task_status = A2ATaskStatus(
            state=_compat.TS_CANCELED,
            message=None,
        )
        canceled_task = A2ATask(
            id="task-1",
            context_id="context-123",
            status=task_status,
        )

        mock_response = _make_stream_task(canceled_task)
        mock_send_message.__aiter__.return_value = [mock_response]
        mock_a2a_client.send_message.return_value = mock_send_message
        agent._a2a_client = mock_a2a_client

        mock_event = Event(
            author=agent.name,
            invocation_id=mock_context.invocation_id,
            branch=mock_context.branch,
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part.from_text(text="Some progress")],
            ),
        )

        with patch.object(
            agent, "_handle_a2a_response", new_callable=AsyncMock
        ) as mock_handle:
          mock_handle.return_value = mock_event

          events = []
          async for ev in agent._run_async_impl(mock_context):
            events.append(ev)

          assert len(events) == 4
          assert events[0] == mock_event
          assert (
              events[1].error_message == "Remote A2A task failed: Task canceled"
          )
          # Verify finish_task event
          assert (
              events[2].content.parts[0].function_response.name
              == FINISH_TASK_TOOL_NAME
          )
          assert events[2].content.parts[0].function_response.response == {
              "result": FINISH_TASK_ERROR_RESULT
          }
          assert events[2].output is None
          assert (
              events[2].error_message == "Remote A2A task failed: Task canceled"
          )
          assert events[3].actions.end_of_agent is True
          assert mock_context.end_of_agents[agent.name] is True
          mock_context.set_agent_state.assert_called_once_with(
              agent.name, end_of_agent=True
          )


class TestRemoteA2aAgentWorkflowOutput:
  """Tests that RemoteA2aAgent surfaces a workflow-node output value.

  Without ``_promote_response_to_output``, a ``RemoteA2aAgent`` used as
  a Workflow node leaves ``ctx.output`` as None, which causes
  downstream JoinNode aggregation to record ``None`` for that
  predecessor.
  """

  # Node path stamped on this agent's events by ``BaseAgent._run_impl``.
  _NODE_PATH = "wf/remote_agent@1"

  def _make_agent(self) -> RemoteA2aAgent:
    return RemoteA2aAgent(
        name="remote_agent",
        agent_card=create_test_agent_card(),
    )

  def test_promotes_text_content_to_output(self):
    agent = self._make_agent()
    event = Event(
        author="remote_agent",
        content=genai_types.Content(
            role="model",
            parts=[genai_types.Part(text="Findings: ok")],
        ),
    )
    event.node_info.path = self._NODE_PATH

    assert agent._promote_response_to_output(event, self._NODE_PATH) is True
    assert event.output == "Findings: ok"
    assert event.node_info.message_as_output is True

  def test_joins_multiple_text_parts(self):
    agent = self._make_agent()
    event = Event(
        author="remote_agent",
        content=genai_types.Content(
            role="model",
            parts=[
                genai_types.Part(text="line1\n"),
                genai_types.Part(text="line2"),
            ],
        ),
    )
    event.node_info.path = self._NODE_PATH

    agent._promote_response_to_output(event, self._NODE_PATH)

    assert event.output == "line1\nline2"

  def test_skips_thought_parts(self):
    agent = self._make_agent()
    event = Event(
        author="remote_agent",
        content=genai_types.Content(
            role="model",
            parts=[
                genai_types.Part(text="streaming update", thought=True),
            ],
        ),
    )
    event.node_info.path = self._NODE_PATH

    agent._promote_response_to_output(event, self._NODE_PATH)

    assert event.output is None
    assert event.node_info.message_as_output is None

  def test_skips_function_call_parts(self):
    """input-required events carry a mock function call and no text."""
    agent = self._make_agent()
    event = Event(
        author="remote_agent",
        content=genai_types.Content(
            role="model",
            parts=[
                genai_types.Part(
                    function_call=genai_types.FunctionCall(
                        id="fc1",
                        name="mock_function_call_for_required_user_input",
                        args={"input_required": "Please confirm"},
                    )
                ),
            ],
        ),
    )
    event.node_info.path = self._NODE_PATH

    agent._promote_response_to_output(event, self._NODE_PATH)

    assert event.output is None

  def test_skips_partial_events(self):
    agent = self._make_agent()
    event = Event(
        author="remote_agent",
        partial=True,
        content=genai_types.Content(
            role="model",
            parts=[genai_types.Part(text="streaming...")],
        ),
    )
    event.node_info.path = self._NODE_PATH

    agent._promote_response_to_output(event, self._NODE_PATH)

    assert event.output is None

  def test_skips_events_from_other_node_path(self):
    """Events whose node path differs are foreign, even if same-named.

    Agent names can collide across a workflow hierarchy, so promotion
    is gated on the node path rather than ``event.author``.
    """
    agent = self._make_agent()
    event = Event(
        author="remote_agent",
        content=genai_types.Content(
            role="model",
            parts=[genai_types.Part(text="Not mine")],
        ),
    )
    event.node_info.path = "wf/other_branch/remote_agent@1"

    assert agent._promote_response_to_output(event, self._NODE_PATH) is False
    assert event.output is None

  def test_preserves_existing_output(self):
    agent = self._make_agent()
    event = Event(
        author="remote_agent",
        output="preset",
        content=genai_types.Content(
            role="model",
            parts=[genai_types.Part(text="text")],
        ),
    )
    event.node_info.path = self._NODE_PATH

    agent._promote_response_to_output(event, self._NODE_PATH)

    assert event.output == "preset"

  def test_no_content_no_output(self):
    agent = self._make_agent()
    event = Event(author="remote_agent")
    event.node_info.path = self._NODE_PATH

    assert agent._promote_response_to_output(event, self._NODE_PATH) is False
    assert event.output is None

  def _make_text_event(
      self, text: str = "reply", task_state: str | None = None
  ) -> Event:
    event = Event(
        author="remote_agent",
        content=genai_types.Content(
            role="model",
            parts=[genai_types.Part(text=text)],
        ),
    )
    if task_state is not None:
      event.custom_metadata = {
          A2A_METADATA_PREFIX + "response": {"status": {"state": task_state}}
      }
    return event

  @pytest.mark.parametrize(
      "state",
      [
          "submitted",
          "working",
          "input-required",
          "auth-required",
          "unknown",
      ],
  )
  def test_skips_non_final_task_states(self, state):
    """Streaming converters may leave non-final text un-thoughted.

    The task-state check on ``custom_metadata['a2a:response']`` is the
    guard that prevents ``ctx.output`` from being overwritten by an
    intermediate event and then raising on the real final event.
    """
    agent = self._make_agent()
    event = self._make_text_event(text="in-progress chunk", task_state=state)
    event.node_info.path = self._NODE_PATH

    assert agent._promote_response_to_output(event, self._NODE_PATH) is False
    assert event.output is None

  @pytest.mark.parametrize(
      "state",
      ["completed", "failed", "canceled", "rejected"],
  )
  def test_promotes_terminal_task_states(self, state):
    agent = self._make_agent()
    event = self._make_text_event(text="final answer", task_state=state)
    event.node_info.path = self._NODE_PATH

    assert agent._promote_response_to_output(event, self._NODE_PATH) is True
    assert event.output == "final answer"

  def test_promotes_when_response_metadata_absent(self):
    """Non-Task A2A responses (plain Message) carry no task status."""
    agent = self._make_agent()
    event = self._make_text_event(text="message reply")
    event.node_info.path = self._NODE_PATH

    assert agent._promote_response_to_output(event, self._NODE_PATH) is True
    assert event.output == "message reply"

  @pytest.mark.asyncio
  async def test_run_impl_promotes_only_first_terminal_event(self):
    """Guards against ``ValueError: Output already set``.

    When the v2 converter path emits a ``working`` text event followed
    by a ``completed`` text event, the first must be passed through
    untouched and only the terminal event promoted. After that, any
    further promotable event must also be left alone.
    """

    working = self._make_text_event(
        text="thinking out loud", task_state="working"
    )
    completed = self._make_text_event(
        text="final answer", task_state="completed"
    )
    trailing = self._make_text_event(
        text="ignored trailing artifact", task_state="completed"
    )

    class _StubRemoteAgent(RemoteA2aAgent):

      async def _run_async_impl(self, ctx):
        yield working
        yield completed
        yield trailing

    agent = _StubRemoteAgent(
        name="remote_agent",
        agent_card=create_test_agent_card(),
    )

    from google.adk.apps.app import App
    from google.adk.workflow._join_node import JoinNode
    from google.adk.workflow._workflow import Workflow

    from tests.unittests import testing_utils

    workflow = Workflow(
        name="wf",
        edges=[("START", agent, JoinNode(name="join"))],
    )
    app_instance = App(name="t", root_agent=workflow)
    runner = testing_utils.InMemoryRunner(app=app_instance)

    events = await runner.run_async(testing_utils.get_user_content("start"))

    # No "Output already set" raised, and the JoinNode aggregates the
    # terminal event's text — not the working intermediate, not the
    # trailing artifact.
    join_outputs = [
        e
        for e in events
        if isinstance(e, Event)
        and e.output is not None
        and "join" in (e.node_info.path or "")
    ]
    assert join_outputs
    assert join_outputs[0].output == {"remote_agent": "final answer"}

    assert working.output is None
    assert completed.output == "final answer"
    assert trailing.output is None

  @pytest.mark.asyncio
  async def test_run_impl_promotes_output_for_each_event(self):
    """``_run_impl`` calls ``_promote_response_to_output`` per event.

    Uses a subclass that overrides ``_run_async_impl`` to yield a
    deterministic event, then drives ``_run_impl`` through the public
    workflow node entry point.
    """

    yielded_event = Event(
        author="remote_agent",
        content=genai_types.Content(
            role="model",
            parts=[genai_types.Part(text="agent reply")],
        ),
    )

    class _StubRemoteAgent(RemoteA2aAgent):

      async def _run_async_impl(self, ctx):
        yield yielded_event

    agent = _StubRemoteAgent(
        name="remote_agent",
        agent_card=create_test_agent_card(),
    )

    from google.adk.apps.app import App
    from google.adk.workflow._join_node import JoinNode
    from google.adk.workflow._workflow import Workflow

    from tests.unittests import testing_utils

    workflow = Workflow(
        name="wf",
        edges=[("START", agent, JoinNode(name="join"))],
    )
    app_instance = App(name="t", root_agent=workflow)
    runner = testing_utils.InMemoryRunner(app=app_instance)
    events = await runner.run_async(testing_utils.get_user_content("start"))

    join_outputs = [
        e
        for e in events
        if isinstance(e, Event)
        and e.output is not None
        and "join" in (e.node_info.path or "")
    ]
    assert join_outputs, "JoinNode should emit an aggregated output event"
    assert join_outputs[0].output == {"remote_agent": "agent reply"}

  @pytest.mark.asyncio
  async def test_run_impl_task_mode_failure_does_not_raise_output_already_set(
      self,
  ):
    """Guards against ``ValueError: Output already set`` during task failure.

    In task mode, the output must not be dynamically promoted from the
    original message chunk if it was already/will be set by the task
    termination finish_task event.
    """
    server_error_event = self._make_text_event(
        text="Simulated error", task_state="failed"
    )
    error_event = Event(
        author="remote_agent",
        error_message="Remote A2A task failed: Simulated error",
    )
    finish_event = Event(
        author="remote_agent",
        content=genai_types.Content(
            role="user",
            parts=[
                genai_types.Part(
                    function_response=genai_types.FunctionResponse(
                        name=FINISH_TASK_TOOL_NAME,
                        response={"result": FINISH_TASK_ERROR_RESULT},
                    )
                )
            ],
        ),
        output="Simulated error",
    )

    class _StubRemoteAgent(RemoteA2aAgent):

      async def _run_async_impl(self, ctx):
        yield server_error_event
        yield error_event
        yield finish_event

    agent = _StubRemoteAgent(
        name="remote_agent",
        agent_card=create_test_agent_card(),
        mode="task",
    )
    agent.parent_agent = Mock()

    from google.adk.apps.app import App
    from google.adk.workflow._join_node import JoinNode
    from google.adk.workflow._workflow import Workflow

    from tests.unittests import testing_utils

    workflow = Workflow(
        name="wf",
        edges=[("START", agent, JoinNode(name="join"))],
    )
    app_instance = App(name="t", root_agent=workflow)
    runner = testing_utils.InMemoryRunner(app=app_instance)

    # Should run successfully without raising "Output already set"
    events = await runner.run_async(testing_utils.get_user_content("start"))

    join_outputs = [
        e
        for e in events
        if isinstance(e, Event)
        and e.output is not None
        and "join" in (e.node_info.path or "")
    ]
    assert join_outputs
    assert join_outputs[0].output == {"remote_agent": "Simulated error"}


# ---------------------------------------------------------------------------
# Regression coverage for the A2A human-input resume rewrite (b/540026826) and
# its adversarial follow-ups: credential egress via the caller fallback, and a
# parallel real-tool + human-input resume re-creating the ValueError.
# ---------------------------------------------------------------------------

_SECRET = "ya29.super-secret-access-token"
# An adk_request_credential response payload (a serialized AuthConfig).
_AUTH_PAYLOAD = {
    "auth_scheme": {"type": "oauth2"},
    "exchanged_auth_credential": {
        "auth_type": "oauth2",
        "oauth2": {"access_token": _SECRET},
    },
}


def _resume_events(
    *,
    calls,
    responses,
    user_text=None,
    task_id="task-123",
):
  """Builds the ``[pause_event, user_response_event]`` sequence seen on resume.

  Args:
    calls: list of ``(name, id)`` function calls that paused the invocation.
    responses: list of ``(name, id, response_dict)`` user function responses.
      Unlike the harness in cl/955528102, this can place more than one
      function_response on the resume event, which is required to reproduce the
      parallel real-tool + human-input case.
    user_text: optional sibling text part appended to the response event.
    task_id: value stamped into the pausing event's a2a metadata.

  Returns:
    ``[pause_event, user_response_event]``.
  """
  call_parts = [
      genai_types.Part(
          function_call=genai_types.FunctionCall(id=cid, name=name, args={})
      )
      for name, cid in calls
  ]
  call_event = Event(
      invocation_id="inv-1",
      author="agent",
      id="e_call",
      content=genai_types.Content(role="model", parts=call_parts),
      long_running_tool_ids={cid for _, cid in calls if cid},
      custom_metadata={
          A2A_METADATA_PREFIX + "task_id": task_id,
          A2A_METADATA_PREFIX + "context_id": "context-123",
      },
  )
  response_parts = [
      genai_types.Part(
          function_response=genai_types.FunctionResponse(
              id=rid, name=name, response=response
          )
      )
      for name, rid, response in responses
  ]
  if user_text is not None:
    response_parts.append(genai_types.Part(text=user_text))
  response_event = Event(
      invocation_id="inv-1",
      author="user",
      id="e_resp",
      content=genai_types.Content(role="user", parts=response_parts),
  )
  return [call_event, response_event]


def _call_event(author, name, args, text="hello"):
  """Builds a one-turn history event: one function_call plus a text sibling."""
  parts = [
      genai_types.Part(
          function_call=genai_types.FunctionCall(
              id="fc-1", name=name, args=args
          )
      )
  ]
  if text is not None:
    parts.append(genai_types.Part(text=text))
  return Event(
      invocation_id="inv-1",
      author=author,
      id="e_auth",
      content=genai_types.Content(role="model", parts=parts),
  )


class _StatefulInterceptor:
  """A caller's interceptor that keeps state and cannot be deep-copied."""

  def __init__(self):
    self.seen = []
    self._lock = threading.Lock()

  async def before_request(self, ctx, a2a_request, parameters):
    self.seen.append(a2a_request)
    return a2a_request, parameters


def _make_agent():
  return RemoteA2aAgent(
      name="test_agent", agent_card="http://example.com/agent.json"
  )


def _make_ctx(events):
  ctx = create_autospec(InvocationContext, instance=True)
  ctx.session = create_autospec(Session, instance=True)
  ctx.session.events = events
  ctx.invocation_id = "inv-1"
  ctx.branch = None
  return ctx


def _forwarded_parts(agent, events):
  message = agent._create_a2a_request_for_user_function_response(  # pylint: disable=protected-access
      _make_ctx(events)
  )
  return list(message.parts) if message is not None else []


def _kind(part):
  # a2a 0.3.x vs 1.x differ (part.root vs flat proto); go through _compat.
  if _compat.is_data_part(part):
    return "data"
  if _compat.is_text_part(part):
    return "text"
  return "other"


def _kinds(parts):
  return [_kind(part) for part in parts]


def _data(part):
  return _compat.data_part_dict(part)


def _text(part):
  return _compat.part_text(part)


def _dump(items):
  return json.dumps([_compat.a2a_to_dict(item) for item in items], default=str)


class TestHitlResumeRewrite:
  """Regression tests for the A2A human-input resume rewrite.

  A GE workflow that pauses on a RequestInput node and then invokes an A2A
  reference node used to fail on resume with `ValueError: Message cannot contain
  both function responses and text`, because the human-input function_response
  was forwarded verbatim beside the user's text.
  """

  def test_agentflow_request_input_is_flattened(self):
    """A workflow RequestInput pause is flattened to text, not sent as data."""
    parts = _forwarded_parts(
        _make_agent(),
        _resume_events(
            calls=[("adk_request_input", "fc-1")],
            responses=[
                ("flow_request_input", "fc-1", {"company_name": "Okta"})
            ],
            user_text="Okta",
        ),
    )
    assert parts
    assert "data" not in _kinds(parts), (
        "human-input function_response survived the rewrite; ADK's Runner"
        " rejects a message mixing function responses and text"
    )

  def test_mock_input_required_is_flattened(self):
    """ADK's own mock input-required pause is still flattened, answer preserved."""
    parts = _forwarded_parts(
        _make_agent(),
        _resume_events(
            calls=[("mock_function_call_for_required_user_input", "fc-1")],
            responses=[(
                "mock_function_call_for_required_user_input",
                "fc-1",
                {"result": "Okta"},
            )],
        ),
    )
    assert "data" not in _kinds(parts)
    assert any(
        _kind(part) == "text" and "Okta" in _text(part) for part in parts
    )

  def test_request_confirmation_is_flattened(self):
    """A confirmation pause is flattened, not forwarded as a function_response."""
    parts = _forwarded_parts(
        _make_agent(),
        _resume_events(
            calls=[("adk_request_confirmation", "fc-1")],
            responses=[
                ("adk_request_confirmation", "fc-1", {"confirmed": True})
            ],
        ),
    )
    assert parts
    assert "data" not in _kinds(parts)

  def test_real_long_running_tool_response_is_preserved(self):
    """A real remote long-running tool response is preserved id-for-id."""
    parts = _forwarded_parts(
        _make_agent(),
        _resume_events(
            calls=[("ask_for_approval", "fc-1")],
            responses=[("ask_for_approval", "fc-1", {"status": "approved"})],
            user_text=None,
        ),
    )
    assert _kinds(parts) == ["data"]
    assert _data(parts[0]).get("id") == "fc-1"

  def test_real_tool_with_text_and_no_pause_never_mixes(self):
    """A real tool response plus stray text (no pause) stays an all-data resume."""
    parts = _forwarded_parts(
        _make_agent(),
        _resume_events(
            calls=[("ask_for_approval", "fc-1")],
            responses=[("ask_for_approval", "fc-1", {"status": "approved"})],
            user_text="also do X",
        ),
    )
    assert "text" not in _kinds(parts)
    assert any(_kind(part) == "data" for part in parts)

  def test_parallel_real_tool_and_human_input_never_mixes(self):
    """A real-tool + human-input resume stays all-data, never data beside text."""
    parts = _forwarded_parts(
        _make_agent(),
        _resume_events(
            calls=[
                ("ask_for_approval", "fc-real"),
                ("adk_request_input", "fc-1"),
            ],
            responses=[
                ("ask_for_approval", "fc-real", {"status": "approved"}),
                ("flow_request_input", "fc-1", {"company_name": "Okta"}),
            ],
            user_text="Okta",
        ),
    )
    kinds = _kinds(parts)
    assert not ("data" in kinds and "text" in kinds), (
        f"forwarded a function_response beside text ({kinds}); ADK's Runner"
        " rejects that combination"
    )
    assert any(
        _kind(part) == "data" and _data(part).get("id") == "fc-real"
        for part in parts
    ), "the real remote tool response must survive so the peer can resume it"

  def test_multiple_real_tools_and_human_inputs_never_mix(self):
    """N real-tool + N human-input responses in one turn stay all-data."""
    parts = _forwarded_parts(
        _make_agent(),
        _resume_events(
            calls=[
                ("ask_for_approval_1", "fc-real-1"),
                ("ask_for_approval_2", "fc-real-2"),
                ("adk_request_input", "fc-1"),
                ("adk_request_confirmation", "fc-2"),
            ],
            responses=[
                ("ask_for_approval_1", "fc-real-1", {"status": "approved"}),
                ("ask_for_approval_2", "fc-real-2", {"status": "rejected"}),
                ("flow_request_input", "fc-1", {"company_name": "Okta"}),
                ("adk_request_confirmation", "fc-2", {"confirmed": True}),
            ],
            user_text="Okta",
        ),
    )
    assert "text" not in _kinds(parts)
    ids = {_data(p).get("id") for p in parts if _kind(p) == "data"}
    assert {"fc-real-1", "fc-real-2", "fc-1", "fc-2"} <= ids

  def test_partial_auth_config_shape_is_dropped(self):
    """Fail-closed: a partial AuthConfig (auth_scheme only) is still dropped."""
    parts = _forwarded_parts(
        _make_agent(),
        _resume_events(
            calls=[("adk_request_input", "fc-1")],
            responses=[(
                "flow_request_input",
                "fc-1",
                {"auth_scheme": {"type": "oauth2"}},
            )],
            user_text="hi",
        ),
    )
    assert _kinds(parts) == ["text"]
    assert "auth_scheme" not in _text(parts[0])

  def test_credential_only_resume_returns_none_without_crashing(self):
    """A credential-only resume drops the secret and returns None, no crash."""
    message = _make_agent()._create_a2a_request_for_user_function_response(  # pylint: disable=protected-access
        _make_ctx(
            _resume_events(
                calls=[("adk_request_credential", "fc-1")],
                responses=[("adk_request_credential", "fc-1", _AUTH_PAYLOAD)],
                user_text=None,
            )
        )
    )
    assert message is None

  def test_credential_is_dropped_even_under_a_non_credential_name(self):
    """Fail-closed: a credential is dropped by AuthConfig shape, not by name."""
    parts = _forwarded_parts(
        _make_agent(),
        _resume_events(
            calls=[("adk_request_input", "fc-1")],  # NOT a credential call name
            responses=[("flow_request_input", "fc-1", _AUTH_PAYLOAD)],
            user_text="Okta",
        ),
    )
    assert "data" not in _kinds(parts)
    assert _SECRET not in _dump(parts)

  def test_credential_under_non_human_input_call_is_dropped(self):
    """Fail-closed: a credential is dropped even when its call is not a pause."""
    parts = _forwarded_parts(
        _make_agent(),
        _resume_events(
            calls=[("some_unknown_tool", "fc-1")],
            responses=[("some_unknown_tool", "fc-1", _AUTH_PAYLOAD)],
            user_text=None,
        ),
    )
    assert _SECRET not in _dump(parts)

  def test_id_less_real_tool_survives_alongside_id_less_human_input(self):
    """An id-less real tool is not flattened by an id-less human-input pause.

    An id-less human-input call (``adk_request_input``) and an id-less real tool
    call (``ask_for_approval``) share the ambiguous id-less bucket; the real
    tool's response must survive as data rather than be flattened to text.

    ``find_matching_function_call`` only engages the rewrite when the turn's
    first function_response has an id, so the turn also carries an id-bearing
    pause (``adk_request_confirmation``). That pause is a human-input answer, so
    it does not on its own force the message to stay a resume: only the id-less
    ambiguity guard keeps the real tool's response as data (without it, both
    responses would flatten to text and the peer could not resume the tool).
    """
    parts = _forwarded_parts(
        _make_agent(),
        _resume_events(
            # The two id-less calls share the ambiguous id-less bucket; the
            # id-bearing confirmation pause lets find_matching_function_call
            # engage the rewrite.
            calls=[
                ("adk_request_input", None),
                ("ask_for_approval", None),
                ("adk_request_confirmation", "fc-1"),
            ],
            responses=[
                ("adk_request_confirmation", "fc-1", {"confirmed": True}),
                ("ask_for_approval", None, {"status": "approved"}),
            ],
            user_text=None,
        ),
    )
    assert "text" not in _kinds(parts)
    assert any(
        _kind(part) == "data" and _data(part).get("name") == "ask_for_approval"
        for part in parts
    )

  def test_construct_message_parts_drops_credential_from_history(self):
    """The session-reconstruction fallback drops credential responses."""
    agent = _make_agent()
    ctx = _make_ctx([
        Event(
            invocation_id="inv-1",
            author="user",
            id="e_resp",
            content=genai_types.Content(
                role="user",
                parts=[
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            id="fc-1",
                            name="adk_request_credential",
                            response=_AUTH_PAYLOAD,
                        )
                    ),
                    genai_types.Part(text="hello"),
                ],
            ),
        )
    ])
    parts, _ = agent._construct_message_parts_from_session(ctx)  # pylint: disable=protected-access
    assert _SECRET not in _dump(parts)
    assert any(_kind(part) == "text" for part in parts)

  @pytest.mark.parametrize("author", ["test_agent", "other_agent"])
  def test_construct_message_parts_drops_credential_request_from_history(
      self, author
  ):
    """An adk_request_credential call is never forwarded.

    A flow appends this event to the session when it asks the client for a
    credential, and its arguments carry the raw client secret. It must be
    dropped whether it is replayed as the agent's own part or rendered as
    another agent's message.
    """
    agent = _make_agent()
    event = _call_event(
        author,
        "adk_request_credential",
        {"functionCallId": "toolset:test_agent", "authConfig": _AUTH_PAYLOAD},
    )
    ctx = _make_ctx([event])

    parts, _ = agent._construct_message_parts_from_session(ctx)  # pylint: disable=protected-access

    assert _SECRET not in _dump(parts)
    assert any(_kind(part) == "text" for part in parts)
    # The scrub copies; the session it read from keeps both parts.
    assert len(event.content.parts) == 2

  @pytest.mark.parametrize("author", ["test_agent", "other_agent"])
  def test_construct_message_parts_skips_credential_only_event(self, author):
    """A credential-only event empties on scrub and is then skipped.

    Without a text sibling the scrub leaves the event with no parts. The walk
    must drop such an event, so no empty or preamble-only message reaches the
    peer.
    """
    agent = _make_agent()
    ctx = _make_ctx([
        _call_event(
            author,
            "adk_request_credential",
            {
                "functionCallId": "toolset:test_agent",
                "authConfig": _AUTH_PAYLOAD,
            },
            text=None,
        )
    ])

    parts, _ = agent._construct_message_parts_from_session(ctx)  # pylint: disable=protected-access

    assert parts == []

  def test_construct_message_parts_drops_credential_call_under_other_name(self):
    """Fail-closed: a credential call is dropped by shape, not only by name.

    The AuthConfig sits under `authConfig`, one level down, because the call
    carries an AuthToolArguments envelope. Reading only the top level here would
    match nothing at all.
    """
    agent = _make_agent()
    ctx = _make_ctx([
        _call_event(
            "test_agent",
            "some_unknown_tool",  # NOT a credential call name
            {
                "functionCallId": "toolset:test_agent",
                "authConfig": _AUTH_PAYLOAD,
            },
        )
    ])

    parts, _ = agent._construct_message_parts_from_session(ctx)  # pylint: disable=protected-access

    assert _SECRET not in _dump(parts)

  def test_construct_message_parts_keeps_ordinary_call_with_auth_scheme_arg(
      self,
  ):
    """An ordinary tool taking an `auth_scheme` argument still gets forwarded.

    `auth_scheme` is a top-level key of a serialized AuthConfig, so a scrub that
    matched on the top level of the arguments would silently eat this call. In
    task mode that is invisible: the message empties and the task aborts.
    """
    agent = _make_agent()
    ctx = _make_ctx([
        _call_event(
            "test_agent",
            "register_connector",
            {"auth_scheme": "oauth2", "name": "drive"},
        )
    ])

    parts, _ = agent._construct_message_parts_from_session(ctx)  # pylint: disable=protected-access

    assert "register_connector" in _dump(parts)

  @pytest.mark.parametrize("author", ["test_agent", "other_agent"])
  def test_construct_message_parts_keeps_mock_auth_prompt(self, author):
    """The mock auth call holds the peer's prompt, not a credential.

    `to_adk_event` builds this call for an auth-required task and puts it in
    place of the peer's last text part, so dropping it throws the prompt away
    and leaves an empty message.
    """
    agent = _make_agent()
    ctx = _make_ctx([
        _call_event(
            author,
            "mock_function_call_for_required_user_auth",
            {"auth_required": "Sign in to Drive to continue"},
            text=None,
        )
    ])

    parts, _ = agent._construct_message_parts_from_session(ctx)  # pylint: disable=protected-access

    assert "Sign in to Drive to continue" in _dump(parts)

  @pytest.mark.asyncio
  async def test_run_async_impl_never_forwards_credential_to_peer(self):
    """A credential-only resume never sends the AuthConfig to the peer."""
    agent = _make_agent()
    captured = []

    async def _capture_send(request, request_metadata=None, context=None):
      del request_metadata, context  # unused; captured request is what matters
      captured.append(request)
      return
      yield  # pragma: no cover -- marks this an async generator

    fake_client = Mock()
    fake_client.send_message = _capture_send
    agent._a2a_client = fake_client  # pylint: disable=protected-access

    ctx = _make_ctx(
        _resume_events(
            calls=[("adk_request_credential", "fc-1")],
            responses=[("adk_request_credential", "fc-1", _AUTH_PAYLOAD)],
            user_text=None,
        )
    )
    with patch.object(agent, "_ensure_resolved"):
      _ = [
          event
          async for event in agent._run_async_impl(ctx)  # pylint: disable=protected-access
      ]

    assert _SECRET not in _dump(captured)


class TestRemoteA2aAgentAuth:
  """Tests for the auth_scheme / auth_credential support."""

  def _auth_scheme(self, **extra):
    return APIKeyScheme(**{"in": APIKeyIn.header, "name": "X-API-Key", **extra})

  def _agent(self, **kwargs):
    kwargs.setdefault("auth_scheme", self._auth_scheme())
    kwargs.setdefault("agent_card", create_test_agent_card())
    return RemoteA2aAgent(name="test_agent", **kwargs)

  def _credential(self, api_key="resolved-key"):
    return AuthCredential(
        auth_type=AuthCredentialTypes.API_KEY, api_key=api_key
    )

  def _message(self):
    return A2AMessage(
        message_id="message-1",
        parts=[_compat.make_text_part("hello")],
        role=_compat.ROLE_USER,
    )

  def _context(self):
    ctx = Mock(spec=InvocationContext)
    ctx.session = Mock(spec=Session)
    ctx.session.state = {}
    ctx.credential_by_key = {}
    ctx.end_invocation = False
    ctx.invocation_id = "invocation-123"
    ctx.branch = "main"
    return ctx

  async def _resolve(self, agent, ctx, **credential_manager_result):
    """Resolves the credential against a stubbed `CredentialManager`."""
    with patch(
        "google.adk.auth.credential_manager.CredentialManager"
    ) as manager:
      manager.return_value.get_auth_credential = AsyncMock(
          **credential_manager_result
      )
      return await agent._resolve_auth_credential(ctx), manager

  async def _run(self, agent, ctx, **resolve_result):
    """Runs the agent with a stubbed `_resolve_auth_credential`."""
    with (
        patch.object(
            agent, "_resolve_auth_credential", AsyncMock(**resolve_result)
        ),
        patch.object(agent, "_ensure_resolved", AsyncMock()) as ensure_resolved,
    ):
      return [e async for e in agent._run_async_impl(ctx)], ensure_resolved

  def test_init_without_auth_scheme_configures_nothing(self):
    """A credential without a scheme is ignored, matching McpToolset."""
    agent = RemoteA2aAgent(
        name="test_agent",
        agent_card=create_test_agent_card(),
        auth_credential=self._credential(),
    )

    assert agent._auth_config is None
    assert agent._config.card_request_interceptors is None
    assert agent._config.request_interceptors is None

  def test_init_with_auth_scheme_registers_interceptors(self):
    """A scheme builds the auth config and both auth interceptors."""
    credential = self._credential()

    agent = self._agent(auth_credential=credential)

    assert agent._auth_config.raw_auth_credential == credential
    assert agent._auth_config.credential_key
    assert len(agent._config.card_request_interceptors) == 1
    assert len(agent._config.request_interceptors) == 1

  def test_init_leaves_the_callers_config_alone(self):
    """One config shared by two agents must not share their credentials."""
    config = A2aRemoteAgentConfig(
        card_request_interceptors=[CardRequestInterceptor()]
    )

    agent = self._agent(config=config)

    assert len(config.card_request_interceptors) == 1
    assert not config.request_interceptors
    # Appended last, so the credential wins over the caller's own headers.
    assert len(agent._config.card_request_interceptors) == 2
    assert agent._config.card_request_interceptors[-1].before_request

  def test_init_keeps_a_stateful_caller_interceptor(self):
    """A cloned interceptor leaves the caller reading state nothing writes."""
    caller = _StatefulInterceptor()
    config = A2aRemoteAgentConfig(
        request_interceptors=[
            RequestInterceptor(before_request=caller.before_request)
        ]
    )

    agent = self._agent(config=config)

    interceptor = agent._config.request_interceptors[0]
    assert interceptor.before_request.__self__ is caller

  def test_init_rejects_an_invalid_agent_card(self):
    """The card is validated before the credential key digests it."""
    with pytest.raises(TypeError):
      self._agent(agent_card=object())

  def test_credential_key_separates_agents_on_different_remotes(self):
    """The derived key digests the scheme, so two agents would share one."""
    bank = self._agent(
        agent_card=create_test_agent_card(url="https://bank.example.com/rpc")
    )
    other = self._agent(
        agent_card=create_test_agent_card(url="https://other.example.com/rpc")
    )

    assert bank._auth_config.credential_key != other._auth_config.credential_key

  def test_credential_key_is_stable_for_the_same_remote(self):
    """The key reaches a credential service, so it cannot move between runs."""
    card = create_test_agent_card(url="https://bank.example.com/rpc")

    first = self._agent(agent_card=card)
    second = self._agent(agent_card=card)

    assert (
        first._auth_config.credential_key == second._auth_config.credential_key
    )

  def test_remote_identity_falls_back_to_an_empty_string(self):
    """The digest reads the card, so neither missing field can end the run."""
    card = Mock(spec=["name"])
    card.name = None

    with patch.object(_compat, "agent_card_url", return_value=None):
      assert remote_a2a_agent._remote_identity(card) == ""

  def test_explicit_credential_key_is_left_alone(self):
    """An explicit key is how a caller opts two agents into sharing one."""
    agent = self._agent(credential_key="shared-key")

    assert agent._auth_config.credential_key == "shared-key"

  def test_credential_key_named_by_the_scheme_is_left_alone(self):
    """`AuthConfig` reads a key off the scheme, which is just as explicit."""
    agent = self._agent(
        auth_scheme=self._auth_scheme(credential_key="shared-key")
    )

    assert agent._auth_config.credential_key == "shared-key"

  @pytest.mark.asyncio
  async def test_resolve_auth_credential_caches_credential(self):
    """A resolved credential is cached on the invocation context."""
    agent, ctx = self._agent(), self._context()
    credential = self._credential()

    event, manager = await self._resolve(agent, ctx, return_value=credential)

    assert event is None
    assert ctx.credential_by_key == {
        agent._auth_config.credential_key: credential
    }
    assert ctx.end_invocation is False
    # Resolved against a copy, so the shared config keeps no credential.
    assert manager.call_args.args[0] is not agent._auth_config

  @pytest.mark.asyncio
  async def test_resolve_auth_credential_reuses_cached_credential(self):
    """An already cached credential is not resolved again."""
    agent, ctx = self._agent(), self._context()
    ctx.credential_by_key[agent._auth_config.credential_key] = (
        self._credential()
    )

    event, manager = await self._resolve(agent, ctx)

    assert event is None
    manager.assert_not_called()

  @pytest.mark.asyncio
  async def test_resolve_auth_credential_requests_credential(self):
    """An unresolvable credential interrupts with an auth request."""
    agent, ctx = self._agent(), self._context()

    event, _ = await self._resolve(agent, ctx, return_value=None)

    function_calls = event.get_function_calls()
    assert len(function_calls) == 1
    assert function_calls[0].name == REQUEST_EUC_FUNCTION_CALL_NAME
    assert (
        function_calls[0]
        .args["functionCallId"]
        .startswith(TOOLSET_AUTH_CREDENTIAL_ID_PREFIX)
    )
    assert event.author == "test_agent"
    assert ctx.end_invocation is True
    assert not ctx.credential_by_key

  @pytest.mark.asyncio
  async def test_resolve_auth_credential_handles_invalid_config(self):
    """An invalid auth config interrupts instead of raising."""
    agent, ctx = self._agent(), self._context()

    event, _ = await self._resolve(
        agent, ctx, side_effect=ValueError("missing auth_credential")
    )

    assert event is not None
    assert ctx.end_invocation is True

  @pytest.mark.asyncio
  async def test_resolve_auth_credential_rejects_a_tokenless_credential(self):
    """A failed exchange hands back the credential with no usable token."""
    agent, ctx = self._agent(), self._context()
    tokenless = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(client_id="client-id", client_secret="secret"),
    )

    event, _ = await self._resolve(agent, ctx, return_value=tokenless)

    assert event is not None
    assert ctx.credential_by_key == {}

  @pytest.mark.asyncio
  async def test_interceptors_send_the_resolved_credential(self):
    """Both the card fetch and the message send carry the credential."""
    agent, ctx = self._agent(), self._context()
    ctx.credential_by_key[agent._auth_config.credential_key] = (
        self._credential()
    )

    http_kwargs = await execute_before_card_request_interceptors(
        agent._config.card_request_interceptors, ctx
    )
    _, parameters = await execute_before_request_interceptors(
        agent._config.request_interceptors, ctx, self._message()
    )

    assert http_kwargs == {"headers": {"X-API-Key": "resolved-key"}}
    assert (
        _request_headers(parameters.client_call_context)["X-API-Key"]
        == "resolved-key"
    )

  @pytest.mark.asyncio
  async def test_interceptors_add_no_headers_without_a_credential(self):
    """Neither call carries an empty header when nothing was resolved."""
    agent, ctx = self._agent(), self._context()

    http_kwargs = await execute_before_card_request_interceptors(
        agent._config.card_request_interceptors, ctx
    )
    _, parameters = await execute_before_request_interceptors(
        agent._config.request_interceptors, ctx, self._message()
    )

    assert http_kwargs is None
    assert not _request_headers(parameters.client_call_context)

  def test_add_request_headers_keeps_the_credential_out_of_session_state(self):
    """The legacy call context is seeded with the persisted session state."""
    session_state = {"http_kwargs": {"headers": {"X-Ext": "1"}}}
    parameters = ParametersConfig(
        client_call_context=_compat.ClientCallContext(state=session_state)
    )

    with patch.object(_compat, "IS_A2A_V1", False):
      remote_a2a_agent._add_request_headers(
          parameters, {"Authorization": "Bearer secret"}
      )

    assert session_state == {"http_kwargs": {"headers": {"X-Ext": "1"}}}
    assert parameters.client_call_context.state["http_kwargs"]["headers"] == {
        "X-Ext": "1",
        "Authorization": "Bearer secret",
    }

  def test_add_request_headers_creates_the_call_context(self):
    """A missing call context is created rather than skipped."""
    parameters = ParametersConfig()

    remote_a2a_agent._add_request_headers(parameters, {"X-API-Key": "key"})

    assert (
        _request_headers(parameters.client_call_context)["X-API-Key"] == "key"
    )

  @pytest.mark.asyncio
  async def test_run_async_impl_yields_auth_request_event(self):
    """An auth request short-circuits before the card is resolved."""
    agent = self._agent()
    auth_request_event = Event(author="test_agent", invocation_id="x")

    events, ensure_resolved = await self._run(
        agent, self._context(), return_value=auth_request_event
    )

    assert events == [auth_request_event]
    ensure_resolved.assert_not_called()

  @pytest.mark.asyncio
  async def test_run_async_impl_reports_auth_failure(self):
    """An unexpected auth failure surfaces as an error event."""
    agent = self._agent()

    events, ensure_resolved = await self._run(
        agent, self._context(), side_effect=RuntimeError("boom")
    )

    assert len(events) == 1
    assert "Failed to authenticate remote A2A agent" in events[0].error_message
    ensure_resolved.assert_not_called()

  def _task_context(self):
    ctx = self._context()
    ctx.agent_states = {}
    ctx.end_of_agents = {}
    ctx.isolation_scope = "task-1"

    def set_agent_state(agent_name, **kwargs):
      if kwargs.get("end_of_agent"):
        ctx.end_of_agents[agent_name] = True
      else:
        ctx.end_of_agents.pop(agent_name, None)

    ctx.set_agent_state.side_effect = set_agent_state
    return ctx

  @pytest.mark.asyncio
  async def test_run_async_impl_releases_task_control_on_auth_failure(self):
    """An auth failure hands the task back, as every other early exit does."""
    agent = self._agent(mode="task")

    events, _ = await self._run(
        agent, self._task_context(), side_effect=RuntimeError("boom")
    )

    assert "Failed to authenticate remote A2A agent" in events[0].error_message
    assert (
        events[1].content.parts[0].function_response.name
        == FINISH_TASK_TOOL_NAME
    )
    assert events[2].actions.end_of_agent is True

  @pytest.mark.asyncio
  async def test_run_async_impl_keeps_task_control_on_auth_request(self):
    """An auth request pauses the task rather than finishing it."""
    agent = self._agent(mode="task")
    ctx = self._task_context()
    auth_request_event = Event(author="test_agent", invocation_id="x")

    events, _ = await self._run(agent, ctx, return_value=auth_request_event)

    assert events == [auth_request_event]
    ctx.set_agent_state.assert_not_called()
