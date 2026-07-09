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

"""Tests for ``to_a2a``.

Hosting is wired through the version-agnostic
``_compat.attach_a2a_routes_to_app``
shim, so these tests run on both a2a-sdk 0.3.x and 1.x. Tests that assert the
version-specific *route attachment* internals live in
``TestAttachA2aRoutesToApp``
and are gated per SDK major. Everything else exercises the version-agnostic
``to_a2a`` surface (validation, runner/service wiring, agent-card sources,
lifespan composition) either via mocks (construction-only checks) or
behaviorally
(driving the app lifespan and asserting routes are attached).
"""

from unittest.mock import ANY
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

from a2a.types import AgentCard
from google.adk.a2a import _compat
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from google.adk.a2a.utils.agent_card_builder import AgentCardBuilder
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.auth.credential_service.in_memory_credential_service import InMemoryCredentialService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow import FunctionNode
from google.adk.workflow import START
from google.adk.workflow import Workflow
import pytest
from starlette.applications import Starlette


def _route_paths(app: Starlette) -> set:
  """Returns the set of route paths registered on an app."""
  return {getattr(r, "path", None) for r in app.routes}


def _assert_a2a_routes_attached(app: Starlette) -> None:
  """Asserts both the JSON-RPC and agent-card routes were attached."""
  paths = _route_paths(app)
  assert "/" in paths, f"missing JSON-RPC route; got {paths}"
  assert any(
      p and "agent-card" in p for p in paths
  ), f"missing agent-card route; got {paths}"


def _minimal_agent_card_dict() -> dict:
  """A minimal agent-card dict valid on both a2a-sdk majors."""
  if _compat.IS_A2A_V1:
    return {
        "name": "file_agent",
        "description": "Test agent from file",
        "version": "1.0.0",
        "supported_interfaces": [
            {"url": "http://example.com/", "protocol_binding": "JSONRPC"}
        ],
        "default_input_modes": ["text/plain"],
        "default_output_modes": ["text/plain"],
    }
  return {
      "name": "file_agent",
      "url": "http://example.com",
      "description": "Test agent from file",
      "version": "1.0.0",
      "capabilities": {},
      "skills": [],
      "defaultInputModes": ["text/plain"],
      "defaultOutputModes": ["text/plain"],
      "supportsAuthenticatedExtendedCard": False,
  }


def _make_minimal_agent_card() -> AgentCard:
  """A minimal AgentCard valid on both a2a-sdk majors."""
  return _compat.parse_agent_card(_minimal_agent_card_dict())


class TestToA2A:
  """Tests for the to_a2a function."""

  def setup_method(self):
    """Set up test fixtures."""
    self.mock_agent = Mock(spec=BaseAgent)
    self.mock_agent.name = "test_agent"
    self.mock_agent.description = "Test agent description"

  # ---------------------------------------------------------------------------
  # Construction-only checks (mock Starlette; hosting is exercised separately).
  # ---------------------------------------------------------------------------

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_default_parameters(
      self,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_agent_executor_class,
  ):
    """Test to_a2a with default parameters."""
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_agent_executor_class.return_value = Mock(spec=A2aAgentExecutor)
    mock_card_builder_class.return_value = Mock(spec=AgentCardBuilder)

    result = to_a2a(self.mock_agent)

    assert result == mock_app
    mock_starlette_class.assert_called_once_with(lifespan=ANY)
    mock_task_store_class.assert_called_once()
    mock_agent_executor_class.assert_called_once()
    mock_card_builder_class.assert_called_once_with(
        agent=self.mock_agent, rpc_url="http://localhost:8000/"
    )

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_with_custom_runner(
      self,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_agent_executor_class,
  ):
    """Test to_a2a with a custom runner."""
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_agent_executor_class.return_value = Mock(spec=A2aAgentExecutor)
    mock_card_builder_class.return_value = Mock(spec=AgentCardBuilder)
    custom_runner = Mock(spec=Runner)

    result = to_a2a(self.mock_agent, runner=custom_runner)

    assert result == mock_app
    mock_starlette_class.assert_called_once_with(lifespan=ANY)
    mock_task_store_class.assert_called_once()
    mock_agent_executor_class.assert_called_once_with(runner=custom_runner)

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_with_custom_task_store(
      self,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_agent_executor_class,
  ):
    """Test to_a2a with a custom task store does not build the default one."""
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_agent_executor_class.return_value = Mock(spec=A2aAgentExecutor)
    mock_card_builder_class.return_value = Mock(spec=AgentCardBuilder)
    custom_task_store = Mock()

    result = to_a2a(self.mock_agent, task_store=custom_task_store)

    assert result == mock_app
    mock_task_store_class.assert_not_called()

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_default_task_store_when_none(
      self,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_agent_executor_class,
  ):
    """Test to_a2a defaults to InMemoryTaskStore when task_store is None."""
    mock_starlette_class.return_value = Mock(spec=Starlette)
    mock_agent_executor_class.return_value = Mock(spec=A2aAgentExecutor)
    mock_card_builder_class.return_value = Mock(spec=AgentCardBuilder)

    to_a2a(self.mock_agent, task_store=None)

    mock_task_store_class.assert_called_once()

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_custom_host_port(
      self,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_agent_executor_class,
  ):
    """Test to_a2a with custom host and port."""
    mock_starlette_class.return_value = Mock(spec=Starlette)
    mock_agent_executor_class.return_value = Mock(spec=A2aAgentExecutor)
    mock_card_builder_class.return_value = Mock(spec=AgentCardBuilder)

    to_a2a(self.mock_agent, host="example.com", port=9000)

    mock_card_builder_class.assert_called_once_with(
        agent=self.mock_agent, rpc_url="http://example.com:9000/"
    )

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_with_custom_port_zero(
      self,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_agent_executor_class,
  ):
    """Test to_a2a with port 0 (dynamic port assignment)."""
    mock_starlette_class.return_value = Mock(spec=Starlette)
    mock_agent_executor_class.return_value = Mock(spec=A2aAgentExecutor)
    mock_card_builder_class.return_value = Mock(spec=AgentCardBuilder)

    to_a2a(self.mock_agent, port=0)

    mock_card_builder_class.assert_called_once_with(
        agent=self.mock_agent, rpc_url="http://localhost:0/"
    )

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_with_empty_string_host(
      self,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_agent_executor_class,
  ):
    """Test to_a2a with empty string host."""
    mock_starlette_class.return_value = Mock(spec=Starlette)
    mock_agent_executor_class.return_value = Mock(spec=A2aAgentExecutor)
    mock_card_builder_class.return_value = Mock(spec=AgentCardBuilder)

    to_a2a(self.mock_agent, host="")

    mock_card_builder_class.assert_called_once_with(
        agent=self.mock_agent, rpc_url="http://:8000/"
    )

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_with_negative_port(
      self,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_agent_executor_class,
  ):
    """Test to_a2a with negative port number."""
    mock_starlette_class.return_value = Mock(spec=Starlette)
    mock_agent_executor_class.return_value = Mock(spec=A2aAgentExecutor)
    mock_card_builder_class.return_value = Mock(spec=AgentCardBuilder)

    to_a2a(self.mock_agent, port=-1)

    mock_card_builder_class.assert_called_once_with(
        agent=self.mock_agent, rpc_url="http://localhost:-1/"
    )

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_with_very_large_port(
      self,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_agent_executor_class,
  ):
    """Test to_a2a with very large port number."""
    mock_starlette_class.return_value = Mock(spec=Starlette)
    mock_agent_executor_class.return_value = Mock(spec=A2aAgentExecutor)
    mock_card_builder_class.return_value = Mock(spec=AgentCardBuilder)

    to_a2a(self.mock_agent, port=65535)

    mock_card_builder_class.assert_called_once_with(
        agent=self.mock_agent, rpc_url="http://localhost:65535/"
    )

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_with_special_characters_in_host(
      self,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_agent_executor_class,
  ):
    """Test to_a2a with special characters in host name."""
    mock_starlette_class.return_value = Mock(spec=Starlette)
    mock_agent_executor_class.return_value = Mock(spec=A2aAgentExecutor)
    mock_card_builder_class.return_value = Mock(spec=AgentCardBuilder)

    to_a2a(self.mock_agent, host="test-host.example.com")

    mock_card_builder_class.assert_called_once_with(
        agent=self.mock_agent, rpc_url="http://test-host.example.com:8000/"
    )

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_with_ip_address_host(
      self,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_agent_executor_class,
  ):
    """Test to_a2a with IP address as host."""
    mock_starlette_class.return_value = Mock(spec=Starlette)
    mock_agent_executor_class.return_value = Mock(spec=A2aAgentExecutor)
    mock_card_builder_class.return_value = Mock(spec=AgentCardBuilder)

    to_a2a(self.mock_agent, host="192.168.1.1")

    mock_card_builder_class.assert_called_once_with(
        agent=self.mock_agent, rpc_url="http://192.168.1.1:8000/"
    )

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_agent_without_name(
      self,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_agent_executor_class,
  ):
    """Test to_a2a with agent that has no name."""
    self.mock_agent.name = None
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_agent_executor_class.return_value = Mock(spec=A2aAgentExecutor)
    mock_card_builder_class.return_value = Mock(spec=AgentCardBuilder)

    result = to_a2a(self.mock_agent)

    assert result == mock_app

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_creates_runner_with_correct_services(
      self,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_agent_executor_class,
  ):
    """Test that the agent executor receives a runner factory callable."""
    mock_starlette_class.return_value = Mock(spec=Starlette)
    mock_agent_executor_class.return_value = Mock(spec=A2aAgentExecutor)
    mock_card_builder_class.return_value = Mock(spec=AgentCardBuilder)

    to_a2a(self.mock_agent)

    mock_agent_executor_class.assert_called_once()
    call_args = mock_agent_executor_class.call_args
    assert "runner" in call_args[1]
    assert callable(call_args[1]["runner"])

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  @patch("google.adk.a2a.utils.agent_to_a2a.Runner")
  def test_create_runner_function_creates_runner_correctly(
      self,
      mock_runner_class,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_agent_executor_class,
  ):
    """Test that the create_runner factory builds a Runner with correct args."""
    mock_starlette_class.return_value = Mock(spec=Starlette)
    mock_agent_executor_class.return_value = Mock(spec=A2aAgentExecutor)
    mock_card_builder_class.return_value = Mock(spec=AgentCardBuilder)
    mock_runner = Mock(spec=Runner)
    mock_runner_class.return_value = mock_runner

    to_a2a(self.mock_agent)

    runner_func = mock_agent_executor_class.call_args[1]["runner"]
    runner_result = runner_func()

    mock_runner_class.assert_called_once_with(
        app_name="test_agent",
        agent=self.mock_agent,
        artifact_service=mock_runner_class.call_args[1]["artifact_service"],
        session_service=mock_runner_class.call_args[1]["session_service"],
        memory_service=mock_runner_class.call_args[1]["memory_service"],
        credential_service=mock_runner_class.call_args[1]["credential_service"],
    )
    call_args = mock_runner_class.call_args[1]
    assert isinstance(call_args["artifact_service"], InMemoryArtifactService)
    assert isinstance(call_args["session_service"], InMemorySessionService)
    assert isinstance(call_args["memory_service"], InMemoryMemoryService)
    assert isinstance(
        call_args["credential_service"], InMemoryCredentialService
    )
    assert runner_result == mock_runner

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  @patch("google.adk.a2a.utils.agent_to_a2a.Runner")
  def test_create_runner_function_with_agent_without_name(
      self,
      mock_runner_class,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_agent_executor_class,
  ):
    """Test create_runner uses a default app_name when agent has no name."""
    self.mock_agent.name = None
    mock_starlette_class.return_value = Mock(spec=Starlette)
    mock_agent_executor_class.return_value = Mock(spec=A2aAgentExecutor)
    mock_card_builder_class.return_value = Mock(spec=AgentCardBuilder)
    mock_runner_class.return_value = Mock(spec=Runner)

    to_a2a(self.mock_agent)

    runner_func = mock_agent_executor_class.call_args[1]["runner"]
    runner_func()

    assert mock_runner_class.call_args[1]["app_name"] == "adk_agent"

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_returns_starlette_app(
      self,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_agent_executor_class,
  ):
    """Test that to_a2a returns the constructed Starlette application."""
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_agent_executor_class.return_value = Mock(spec=A2aAgentExecutor)
    mock_card_builder_class.return_value = Mock(spec=AgentCardBuilder)

    result = to_a2a(self.mock_agent)

    assert result == mock_app

  # ---------------------------------------------------------------------------
  # Behavioral hosting checks (real Starlette; drive lifespan; assert routes).
  # ---------------------------------------------------------------------------

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  async def test_setup_a2a_builds_card_and_attaches_routes(
      self,
      mock_card_builder_class,
      mock_task_store_class,
      mock_agent_executor_class,
  ):
    """setup_a2a builds the agent card and attaches the A2A routes."""
    mock_agent_executor_class.return_value = Mock(spec=A2aAgentExecutor)
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder
    mock_card_builder.build = AsyncMock(return_value=_make_minimal_agent_card())

    app = to_a2a(self.mock_agent)
    async with app.router.lifespan_context(app):
      pass

    mock_card_builder.build.assert_called_once()
    _assert_a2a_routes_attached(app)

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  async def test_setup_a2a_handles_agent_card_build_failure(
      self,
      mock_card_builder_class,
      mock_task_store_class,
      mock_agent_executor_class,
  ):
    """A failure building the agent card propagates during lifespan startup."""
    mock_agent_executor_class.return_value = Mock(spec=A2aAgentExecutor)
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder
    mock_card_builder.build = AsyncMock(side_effect=Exception("Build failed"))

    app = to_a2a(self.mock_agent)
    with pytest.raises(Exception, match="Build failed"):
      async with app.router.lifespan_context(app):
        pass

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  async def test_to_a2a_with_custom_agent_card_object(
      self,
      mock_card_builder_class,
      mock_task_store_class,
      mock_agent_executor_class,
  ):
    """A provided AgentCard is used directly without building one."""
    mock_agent_executor_class.return_value = Mock(spec=A2aAgentExecutor)
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder
    mock_card_builder.build = AsyncMock()

    app = to_a2a(self.mock_agent, agent_card=_make_minimal_agent_card())
    async with app.router.lifespan_context(app):
      pass

    mock_card_builder.build.assert_not_called()
    _assert_a2a_routes_attached(app)

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("json.load")
  @patch("pathlib.Path.open")
  @patch("pathlib.Path")
  async def test_to_a2a_with_agent_card_file_path(
      self,
      mock_path_class,
      mock_open,
      mock_json_load,
      mock_card_builder_class,
      mock_task_store_class,
      mock_agent_executor_class,
  ):
    """An agent card loaded from a file path is used to attach routes."""
    mock_agent_executor_class.return_value = Mock(spec=A2aAgentExecutor)
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder
    mock_card_builder.build = AsyncMock()

    mock_path = Mock()
    mock_path_class.return_value = mock_path
    mock_file_handle = Mock()
    mock_context_manager = Mock()
    mock_context_manager.__enter__ = Mock(return_value=mock_file_handle)
    mock_context_manager.__exit__ = Mock(return_value=None)
    mock_path.open = Mock(return_value=mock_context_manager)
    mock_json_load.return_value = _minimal_agent_card_dict()

    app = to_a2a(self.mock_agent, agent_card="/path/to/agent_card.json")
    async with app.router.lifespan_context(app):
      pass

    mock_path_class.assert_called_once_with("/path/to/agent_card.json")
    mock_path.open.assert_called_once_with("r", encoding="utf-8")
    mock_json_load.assert_called_once_with(mock_file_handle)
    mock_card_builder.build.assert_not_called()
    _assert_a2a_routes_attached(app)

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("pathlib.Path.open", side_effect=FileNotFoundError("File not found"))
  @patch("pathlib.Path")
  def test_to_a2a_with_invalid_agent_card_file_path(
      self,
      mock_path_class,
      mock_open,
      mock_card_builder_class,
      mock_task_store_class,
      mock_agent_executor_class,
  ):
    """An invalid agent card file path raises at to_a2a() call time."""
    mock_agent_executor_class.return_value = Mock(spec=A2aAgentExecutor)
    mock_card_builder_class.return_value = Mock(spec=AgentCardBuilder)
    mock_path_class.return_value = Mock()

    with pytest.raises(ValueError, match="Failed to load agent card from"):
      to_a2a(self.mock_agent, agent_card="/invalid/path.json")

  async def test_to_a2a_with_lifespan(self):
    """A user lifespan runs alongside A2A setup."""
    from contextlib import asynccontextmanager

    startup_called = False
    shutdown_called = False

    @asynccontextmanager
    async def custom_lifespan(app):
      nonlocal startup_called, shutdown_called
      startup_called = True
      app.state.test_value = "hello"
      yield
      shutdown_called = True

    agent = LlmAgent(
        name="lifespan_agent", description="d", model="gemini-2.0-flash"
    )
    app = to_a2a(agent, port=8001, lifespan=custom_lifespan)

    async with app.router.lifespan_context(app):
      _assert_a2a_routes_attached(app)
      assert startup_called
      assert app.state.test_value == "hello"

    assert shutdown_called

  async def test_to_a2a_without_lifespan(self):
    """Without a user lifespan, A2A setup still attaches routes."""
    agent = LlmAgent(
        name="nolifespan_agent", description="d", model="gemini-2.0-flash"
    )
    app = to_a2a(agent, port=8001)

    async with app.router.lifespan_context(app):
      _assert_a2a_routes_attached(app)

  async def test_to_a2a_lifespan_setup_runs_before_user_lifespan(self):
    """A2A setup (routes attached) runs before the user lifespan startup."""
    from contextlib import asynccontextmanager

    call_order = []

    @asynccontextmanager
    async def custom_lifespan(app):
      # By the time the user lifespan starts, routes must already be attached.
      call_order.append("user_startup")
      assert "/" in _route_paths(app)
      yield
      call_order.append("user_shutdown")

    agent = LlmAgent(
        name="order_agent", description="d", model="gemini-2.0-flash"
    )
    app = to_a2a(agent, port=8001, lifespan=custom_lifespan)

    async with app.router.lifespan_context(app):
      pass

    assert call_order == ["user_startup", "user_shutdown"]

  # ---------------------------------------------------------------------------
  # Validation (version-agnostic).
  # ---------------------------------------------------------------------------

  def test_to_a2a_with_none_agent(self):
    """Test that to_a2a raises error when agent is None."""
    with pytest.raises(ValueError, match="Agent cannot be None or empty."):
      to_a2a(None)

  def test_to_a2a_rejects_non_agent_non_workflow(self):
    """to_a2a raises TypeError immediately for unsupported types.

    Only BaseAgent (e.g. LlmAgent) and Workflow are valid A2A roots. Other
    BaseNode subclasses (e.g. FunctionNode) and arbitrary objects must be
    rejected at call time, not silently served as a degenerate card.
    """
    with pytest.raises(
        TypeError, match="requires a BaseAgent or Workflow, got str"
    ):
      to_a2a("not an agent")

  async def test_to_a2a_succeeds_for_workflow(self):
    """to_a2a accepts a Workflow and the Starlette lifespan completes."""
    writer = LlmAgent(
        name="writer",
        model="gemini-2.5-flash",
        instruction="Write a short reply.",
    )
    workflow = Workflow(name="pipe", edges=[(START, writer)])

    app = to_a2a(workflow, port=8001)
    async with app.router.lifespan_context(app):
      _assert_a2a_routes_attached(app)

  def test_to_a2a_rejects_function_node(self):
    """to_a2a raises TypeError for a bare FunctionNode.

    FunctionNode is a BaseNode but is intended for use inside a Workflow, not
    as a standalone A2A root.
    """

    async def my_fn(node_input):
      return f"echo: {node_input}"

    fn_node = FunctionNode(func=my_fn, name="echo_fn")

    with pytest.raises(
        TypeError, match="requires a BaseAgent or Workflow, got FunctionNode"
    ):
      to_a2a(fn_node)


class TestAttachA2aRoutesToApp:
  """Tests for the version-agnostic ``_compat.attach_a2a_routes_to_app`` shim."""

  @pytest.mark.skipif(
      _compat.IS_A2A_V1,
      reason="0.3.x route attachment internals (A2AStarletteApplication)",
  )
  @patch("a2a.server.request_handlers.DefaultRequestHandler")
  @patch("a2a.server.apps.A2AStarletteApplication")
  def test_prefix_is_propagated_to_add_routes_on_0_3(
      self, mock_a2a_app_class, mock_handler_class
  ):
    """The 0.3.x branch must mount routes under the given prefix."""
    # Regression: fast_api.py hosts multiple agents on one app, each under
    # /a2a/{name}. The 0.3.x branch previously ignored prefix and called
    # add_routes_to_app(app) with defaults, so every agent collided on the
    # root RPC route and default /.well-known/... card route.
    del mock_handler_class  # Patched to avoid real handler construction.
    mock_a2a_app = Mock()
    mock_a2a_app_class.return_value = mock_a2a_app
    app = Starlette()

    _compat.attach_a2a_routes_to_app(
        app,
        agent_card=Mock(spec=AgentCard),
        agent_executor=Mock(),
        task_store=Mock(),
        prefix="/a2a/my_agent",
    )

    mock_a2a_app.add_routes_to_app.assert_called_once()
    _, kwargs = mock_a2a_app.add_routes_to_app.call_args
    assert kwargs["rpc_url"] == "/a2a/my_agent"
    assert kwargs["agent_card_url"].startswith("/a2a/my_agent/.well-known/")

  @pytest.mark.skipif(
      _compat.IS_A2A_V1,
      reason="0.3.x route attachment internals (A2AStarletteApplication)",
  )
  @patch("a2a.server.request_handlers.DefaultRequestHandler")
  @patch("a2a.server.apps.A2AStarletteApplication")
  def test_no_prefix_uses_defaults_on_0_3(
      self, mock_a2a_app_class, mock_handler_class
  ):
    """Without a prefix, routes mount at the SDK defaults (root)."""
    del mock_handler_class  # Patched to avoid real handler construction.
    mock_a2a_app = Mock()
    mock_a2a_app_class.return_value = mock_a2a_app
    app = Starlette()

    _compat.attach_a2a_routes_to_app(
        app,
        agent_card=Mock(spec=AgentCard),
        agent_executor=Mock(),
        task_store=Mock(),
    )

    mock_a2a_app.add_routes_to_app.assert_called_once_with(app)

  @pytest.mark.skipif(
      not _compat.IS_A2A_V1,
      reason="1.x route-factory attachment path",
  )
  def test_attach_routes_with_prefix_on_v1(self):
    """The 1.x branch attaches prefixed JSON-RPC and agent-card routes."""
    from a2a.server.tasks import InMemoryTaskStore

    agent_card = _compat.parse_agent_card({
        "name": "smoke",
        "description": "d",
        "version": "1.0",
        "supported_interfaces": [
            {"url": "http://localhost:8001/", "protocol_binding": "JSONRPC"}
        ],
        "default_input_modes": ["text/plain"],
        "default_output_modes": ["text/plain"],
    })

    class _NoopExecutor:

      async def execute(self, context, event_queue):
        return None

      async def cancel(self, context, event_queue):
        return None

    app = Starlette()
    before = len(app.routes)
    _compat.attach_a2a_routes_to_app(
        app,
        agent_card=agent_card,
        agent_executor=_NoopExecutor(),
        task_store=InMemoryTaskStore(),
        prefix="/a2a/smoke",
    )

    assert len(app.routes) > before
    paths = {getattr(r, "path", None) for r in app.routes}
    assert any(p and "agent-card" in p for p in paths)
    assert any(p == "/a2a/smoke" for p in paths)
