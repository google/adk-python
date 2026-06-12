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

from contextlib import asynccontextmanager
from unittest.mock import ANY
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryPushNotificationConfigStore
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard
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

# ---------------------------------------------------------------------------
# Helper: decorator order note
# @patch decorators are applied bottom-up; the innermost (closest to def)
# corresponds to the FIRST mock parameter after self.
# ---------------------------------------------------------------------------


class TestToA2A:
  """Test suite for to_a2a function."""

  def setup_method(self):
    """Set up test fixtures."""
    self.mock_agent = Mock(spec=BaseAgent)
    self.mock_agent.name = "test_agent"
    self.mock_agent.description = "Test agent description"

  # -------------------------------------------------------------------------
  # Helper: standard mock setup used by many tests
  # -------------------------------------------------------------------------
  @staticmethod
  def _setup_standard_mocks(
      mock_card_builder_class,
      mock_task_store_class,
      mock_request_handler_class,
      mock_agent_executor_class,
      mock_create_card_routes,
      mock_create_jsonrpc_routes,
      mock_create_rest_routes,
  ):
    mock_task_store = Mock(spec=InMemoryTaskStore)
    mock_task_store_class.return_value = mock_task_store
    mock_agent_executor = Mock(spec=A2aAgentExecutor)
    mock_agent_executor_class.return_value = mock_agent_executor
    mock_request_handler = Mock(spec=DefaultRequestHandler)
    mock_request_handler_class.return_value = mock_request_handler
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder
    mock_agent_card = Mock(spec=AgentCard)
    mock_card_builder.build = AsyncMock(return_value=mock_agent_card)
    mock_create_card_routes.return_value = []
    mock_create_jsonrpc_routes.return_value = []
    mock_create_rest_routes.return_value = []
    return (
        mock_task_store,
        mock_agent_executor,
        mock_request_handler,
        mock_card_builder,
        mock_agent_card,
    )

  # =========================================================================
  # Tests that verify executor / handler construction (require lifespan run)
  # =========================================================================

  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_agent_card_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_jsonrpc_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_rest_routes")
  async def test_to_a2a_default_parameters(
      self,
      mock_create_rest_routes,  # innermost → first param
      mock_create_jsonrpc_routes,
      mock_create_card_routes,
      mock_agent_executor_class,
      mock_request_handler_class,
      mock_task_store_class,
      mock_card_builder_class,  # outermost → last param
  ):
    """Test to_a2a with default parameters."""
    (
        mock_task_store,
        mock_agent_executor,
        mock_request_handler,
        mock_card_builder,
        mock_agent_card,
    ) = self._setup_standard_mocks(
        mock_card_builder_class,
        mock_task_store_class,
        mock_request_handler_class,
        mock_agent_executor_class,
        mock_create_card_routes,
        mock_create_jsonrpc_routes,
        mock_create_rest_routes,
    )

    result = to_a2a(self.mock_agent)

    assert isinstance(result, Starlette)
    mock_task_store_class.assert_called_once()
    mock_card_builder_class.assert_called_once_with(
        agent=self.mock_agent, rpc_url="http://localhost:8000/"
    )

    async with result.router.lifespan_context(result):
      pass

    mock_agent_executor_class.assert_called_once()
    mock_request_handler_class.assert_called_once_with(
        agent_executor=mock_agent_executor,
        push_config_store=ANY,
        task_store=mock_task_store,
        agent_card=mock_agent_card,
    )

  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_agent_card_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_jsonrpc_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_rest_routes")
  async def test_to_a2a_with_custom_runner(
      self,
      mock_create_rest_routes,
      mock_create_jsonrpc_routes,
      mock_create_card_routes,
      mock_agent_executor_class,
      mock_request_handler_class,
      mock_task_store_class,
      mock_card_builder_class,
  ):
    """Test to_a2a with a custom runner."""
    (
        mock_task_store,
        mock_agent_executor,
        mock_request_handler,
        mock_card_builder,
        mock_agent_card,
    ) = self._setup_standard_mocks(
        mock_card_builder_class,
        mock_task_store_class,
        mock_request_handler_class,
        mock_agent_executor_class,
        mock_create_card_routes,
        mock_create_jsonrpc_routes,
        mock_create_rest_routes,
    )
    custom_runner = Mock(spec=Runner)

    result = to_a2a(self.mock_agent, runner=custom_runner)

    assert isinstance(result, Starlette)
    mock_task_store_class.assert_called_once()
    mock_card_builder_class.assert_called_once_with(
        agent=self.mock_agent, rpc_url="http://localhost:8000/"
    )

    async with result.router.lifespan_context(result):
      pass

    mock_agent_executor_class.assert_called_once_with(runner=custom_runner)
    mock_request_handler_class.assert_called_once_with(
        agent_executor=mock_agent_executor,
        push_config_store=ANY,
        task_store=mock_task_store,
        agent_card=mock_agent_card,
    )

  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_agent_card_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_jsonrpc_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_rest_routes")
  async def test_to_a2a_passes_custom_push_config_store(
      self,
      mock_create_rest_routes,
      mock_create_jsonrpc_routes,
      mock_create_card_routes,
      mock_agent_executor_class,
      mock_request_handler_class,
      mock_task_store_class,
      mock_card_builder_class,
  ):
    """Test to_a2a forwards a custom push config store."""
    (
        mock_task_store,
        mock_agent_executor,
        mock_request_handler,
        mock_card_builder,
        mock_agent_card,
    ) = self._setup_standard_mocks(
        mock_card_builder_class,
        mock_task_store_class,
        mock_request_handler_class,
        mock_agent_executor_class,
        mock_create_card_routes,
        mock_create_jsonrpc_routes,
        mock_create_rest_routes,
    )
    custom_push_store = InMemoryPushNotificationConfigStore()

    result = to_a2a(self.mock_agent, push_config_store=custom_push_store)

    assert isinstance(result, Starlette)

    async with result.router.lifespan_context(result):
      pass

    mock_request_handler_class.assert_called_once_with(
        agent_executor=mock_agent_executor,
        push_config_store=custom_push_store,
        task_store=mock_task_store,
        agent_card=mock_agent_card,
    )

  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_agent_card_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_jsonrpc_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_rest_routes")
  async def test_to_a2a_with_custom_task_store(
      self,
      mock_create_rest_routes,
      mock_create_jsonrpc_routes,
      mock_create_card_routes,
      mock_agent_executor_class,
      mock_request_handler_class,
      mock_task_store_class,
      mock_card_builder_class,
  ):
    """Test to_a2a with a custom task store."""
    (
        _,
        mock_agent_executor,
        mock_request_handler,
        mock_card_builder,
        mock_agent_card,
    ) = self._setup_standard_mocks(
        mock_card_builder_class,
        mock_task_store_class,
        mock_request_handler_class,
        mock_agent_executor_class,
        mock_create_card_routes,
        mock_create_jsonrpc_routes,
        mock_create_rest_routes,
    )
    custom_task_store = Mock()

    result = to_a2a(self.mock_agent, task_store=custom_task_store)

    assert isinstance(result, Starlette)

    async with result.router.lifespan_context(result):
      pass

    mock_task_store_class.assert_not_called()
    mock_request_handler_class.assert_called_once_with(
        agent_executor=mock_agent_executor,
        push_config_store=ANY,
        task_store=custom_task_store,
        agent_card=mock_agent_card,
    )

  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_agent_card_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_jsonrpc_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_rest_routes")
  async def test_to_a2a_default_task_store_when_none(
      self,
      mock_create_rest_routes,
      mock_create_jsonrpc_routes,
      mock_create_card_routes,
      mock_agent_executor_class,
      mock_request_handler_class,
      mock_task_store_class,
      mock_card_builder_class,
  ):
    """Test to_a2a defaults to InMemoryTaskStore when task_store is None."""
    (
        mock_task_store,
        mock_agent_executor,
        mock_request_handler,
        mock_card_builder,
        mock_agent_card,
    ) = self._setup_standard_mocks(
        mock_card_builder_class,
        mock_task_store_class,
        mock_request_handler_class,
        mock_agent_executor_class,
        mock_create_card_routes,
        mock_create_jsonrpc_routes,
        mock_create_rest_routes,
    )

    result = to_a2a(self.mock_agent, task_store=None)

    mock_task_store_class.assert_called_once()

    async with result.router.lifespan_context(result):
      pass

    mock_request_handler_class.assert_called_once_with(
        agent_executor=mock_agent_executor,
        push_config_store=ANY,
        task_store=mock_task_store,
        agent_card=mock_agent_card,
    )

  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_custom_host_port(
      self,
      mock_starlette_class,
      mock_agent_executor_class,
      mock_request_handler_class,
      mock_task_store_class,
      mock_card_builder_class,
  ):
    """Test to_a2a with custom host and port."""
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder

    result = to_a2a(self.mock_agent, host="example.com", port=9000)

    assert result == mock_app
    mock_card_builder_class.assert_called_once_with(
        agent=self.mock_agent, rpc_url="http://example.com:9000/"
    )

  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_agent_without_name(
      self,
      mock_starlette_class,
      mock_agent_executor_class,
      mock_request_handler_class,
      mock_task_store_class,
      mock_card_builder_class,
  ):
    """Test to_a2a with agent that has no name."""
    self.mock_agent.name = None
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder

    result = to_a2a(self.mock_agent)

    assert result == mock_app
    # The create_runner function should use "adk_agent" as default name

  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_agent_card_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_jsonrpc_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_rest_routes")
  async def test_to_a2a_creates_runner_with_correct_services(
      self,
      mock_create_rest_routes,
      mock_create_jsonrpc_routes,
      mock_create_card_routes,
      mock_agent_executor_class,
      mock_request_handler_class,
      mock_task_store_class,
      mock_card_builder_class,
  ):
    """Test that the create_runner function creates Runner with correct services."""
    (
        mock_task_store,
        mock_agent_executor,
        mock_request_handler,
        mock_card_builder,
        mock_agent_card,
    ) = self._setup_standard_mocks(
        mock_card_builder_class,
        mock_task_store_class,
        mock_request_handler_class,
        mock_agent_executor_class,
        mock_create_card_routes,
        mock_create_jsonrpc_routes,
        mock_create_rest_routes,
    )

    result = to_a2a(self.mock_agent)

    assert isinstance(result, Starlette)

    async with result.router.lifespan_context(result):
      pass

    # Verify that the agent executor was created with a runner instance
    mock_agent_executor_class.assert_called_once()
    call_args = mock_agent_executor_class.call_args
    assert "runner" in call_args[1]
    runner_instance = call_args[1]["runner"]
    assert isinstance(runner_instance, Runner)

  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_agent_card_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_jsonrpc_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_rest_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.Runner")
  async def test_create_runner_function_creates_runner_correctly(
      self,
      mock_runner_class,
      mock_create_rest_routes,
      mock_create_jsonrpc_routes,
      mock_create_card_routes,
      mock_agent_executor_class,
      mock_request_handler_class,
      mock_task_store_class,
      mock_card_builder_class,
  ):
    """Test that the create_runner function creates Runner with correct parameters."""
    (
        mock_task_store,
        mock_agent_executor,
        mock_request_handler,
        mock_card_builder,
        mock_agent_card,
    ) = self._setup_standard_mocks(
        mock_card_builder_class,
        mock_task_store_class,
        mock_request_handler_class,
        mock_agent_executor_class,
        mock_create_card_routes,
        mock_create_jsonrpc_routes,
        mock_create_rest_routes,
    )
    mock_runner = Mock(spec=Runner)
    mock_runner_class.return_value = mock_runner

    result = to_a2a(self.mock_agent)

    async with result.router.lifespan_context(result):
      pass

    # Get the runner instance that was passed to A2aAgentExecutor
    call_args = mock_agent_executor_class.call_args
    runner_instance = call_args[1]["runner"]

    # Verify Runner was created with correct parameters (eagerly, not as a factory)
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
    # Runner is passed as an instance directly (not a callable factory)
    assert runner_instance == mock_runner

  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_agent_card_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_jsonrpc_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_rest_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.Runner")
  async def test_create_runner_function_with_agent_without_name(
      self,
      mock_runner_class,
      mock_create_rest_routes,
      mock_create_jsonrpc_routes,
      mock_create_card_routes,
      mock_agent_executor_class,
      mock_request_handler_class,
      mock_task_store_class,
      mock_card_builder_class,
  ):
    """Test create_runner function with agent that has no name."""
    self.mock_agent.name = None
    (
        mock_task_store,
        mock_agent_executor,
        mock_request_handler,
        mock_card_builder,
        mock_agent_card,
    ) = self._setup_standard_mocks(
        mock_card_builder_class,
        mock_task_store_class,
        mock_request_handler_class,
        mock_agent_executor_class,
        mock_create_card_routes,
        mock_create_jsonrpc_routes,
        mock_create_rest_routes,
    )
    mock_runner = Mock(spec=Runner)
    mock_runner_class.return_value = mock_runner

    result = to_a2a(self.mock_agent)

    async with result.router.lifespan_context(result):
      pass

    call_args = mock_agent_executor_class.call_args
    runner_func = call_args[1]["runner"]
    runner_func()

    # Verify Runner was created with default app_name when agent has no name
    mock_runner_class.assert_called_once_with(
        app_name="adk_agent",
        agent=self.mock_agent,
        artifact_service=mock_runner_class.call_args[1]["artifact_service"],
        session_service=mock_runner_class.call_args[1]["session_service"],
        memory_service=mock_runner_class.call_args[1]["memory_service"],
        credential_service=mock_runner_class.call_args[1]["credential_service"],
    )

  # =========================================================================
  # Async tests: setup_a2a lifespan and route wiring
  # =========================================================================

  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_agent_card_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_jsonrpc_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_rest_routes")
  async def test_setup_a2a_function_builds_agent_card_and_configures_routes(
      self,
      mock_create_rest_routes,
      mock_create_jsonrpc_routes,
      mock_create_card_routes,
      mock_agent_executor_class,
      mock_request_handler_class,
      mock_task_store_class,
      mock_card_builder_class,
  ):
    """Test that setup_a2a builds agent card and configures A2A routes."""
    (
        mock_task_store,
        mock_agent_executor,
        mock_request_handler,
        mock_card_builder,
        mock_agent_card,
    ) = self._setup_standard_mocks(
        mock_card_builder_class,
        mock_task_store_class,
        mock_request_handler_class,
        mock_agent_executor_class,
        mock_create_card_routes,
        mock_create_jsonrpc_routes,
        mock_create_rest_routes,
    )

    app = to_a2a(self.mock_agent)

    async with app.router.lifespan_context(app):
      pass

    # Verify agent card was built
    mock_card_builder.build.assert_called_once()

    # Verify executor and handler were created
    mock_agent_executor_class.assert_called_once()
    mock_request_handler_class.assert_called_once_with(
        agent_executor=mock_agent_executor,
        task_store=mock_task_store,
        agent_card=mock_agent_card,
        push_config_store=ANY,
    )

    # Verify route builders were called
    mock_create_card_routes.assert_called_once_with(mock_agent_card)
    mock_create_jsonrpc_routes.assert_called_once()
    mock_create_rest_routes.assert_called_once_with(mock_request_handler)

  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_agent_card_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_jsonrpc_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_rest_routes")
  async def test_setup_a2a_function_handles_agent_card_build_failure(
      self,
      mock_create_rest_routes,
      mock_create_jsonrpc_routes,
      mock_create_card_routes,
      mock_agent_executor_class,
      mock_request_handler_class,
      mock_task_store_class,
      mock_card_builder_class,
  ):
    """Test that setup_a2a properly handles agent card build failure."""
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder
    mock_card_builder.build = AsyncMock(side_effect=Exception("Build failed"))

    app = to_a2a(self.mock_agent)

    with pytest.raises(Exception, match="Build failed"):
      async with app.router.lifespan_context(app):
        pass

  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_returns_starlette_app(
      self,
      mock_starlette_class,
      mock_agent_executor_class,
      mock_request_handler_class,
      mock_task_store_class,
      mock_card_builder_class,
  ):
    """Test that to_a2a returns a Starlette application."""
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder

    result = to_a2a(self.mock_agent)

    assert isinstance(result, Mock)
    assert result == mock_app

  def test_to_a2a_with_none_agent(self):
    """Test that to_a2a raises error when agent is None."""
    with pytest.raises(ValueError, match="Agent cannot be None or empty."):
      to_a2a(None)

  def test_to_a2a_rejects_non_agent_non_workflow(self):
    """to_a2a raises TypeError immediately for unsupported types.

    Only BaseAgent (e.g. LlmAgent) and Workflow are valid
    A2A roots. Other BaseNode subclasses (e.g. FunctionNode) and
    arbitrary objects must be rejected at call time, not silently served
    as a degenerate "custom agent" card.
    """
    with pytest.raises(
        TypeError, match="requires a BaseAgent or Workflow, got str"
    ):
      to_a2a("not an agent")

  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_with_custom_port_zero(
      self,
      mock_starlette_class,
      mock_agent_executor_class,
      mock_request_handler_class,
      mock_task_store_class,
      mock_card_builder_class,
  ):
    """Test to_a2a with port 0."""
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder

    result = to_a2a(self.mock_agent, port=0)

    assert result == mock_app
    mock_card_builder_class.assert_called_once_with(
        agent=self.mock_agent, rpc_url="http://localhost:0/"
    )

  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_with_empty_string_host(
      self,
      mock_starlette_class,
      mock_agent_executor_class,
      mock_request_handler_class,
      mock_task_store_class,
      mock_card_builder_class,
  ):
    """Test to_a2a with empty string host."""
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder

    result = to_a2a(self.mock_agent, host="")

    assert result == mock_app
    mock_card_builder_class.assert_called_once_with(
        agent=self.mock_agent, rpc_url="http://:8000/"
    )

  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_with_negative_port(
      self,
      mock_starlette_class,
      mock_agent_executor_class,
      mock_request_handler_class,
      mock_task_store_class,
      mock_card_builder_class,
  ):
    """Test to_a2a with negative port number."""
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder

    result = to_a2a(self.mock_agent, port=-1)

    assert result == mock_app
    mock_card_builder_class.assert_called_once_with(
        agent=self.mock_agent, rpc_url="http://localhost:-1/"
    )

  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_with_very_large_port(
      self,
      mock_starlette_class,
      mock_agent_executor_class,
      mock_request_handler_class,
      mock_task_store_class,
      mock_card_builder_class,
  ):
    """Test to_a2a with very large port number."""
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder

    result = to_a2a(self.mock_agent, port=65535)

    assert result == mock_app
    mock_card_builder_class.assert_called_once_with(
        agent=self.mock_agent, rpc_url="http://localhost:65535/"
    )

  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_with_special_characters_in_host(
      self,
      mock_starlette_class,
      mock_agent_executor_class,
      mock_request_handler_class,
      mock_task_store_class,
      mock_card_builder_class,
  ):
    """Test to_a2a with special characters in host name."""
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder

    result = to_a2a(self.mock_agent, host="test-host.example.com")

    assert result == mock_app
    mock_card_builder_class.assert_called_once_with(
        agent=self.mock_agent, rpc_url="http://test-host.example.com:8000/"
    )

  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_with_ip_address_host(
      self,
      mock_starlette_class,
      mock_agent_executor_class,
      mock_request_handler_class,
      mock_task_store_class,
      mock_card_builder_class,
  ):
    """Test to_a2a with IP address as host."""
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder

    result = to_a2a(self.mock_agent, host="192.168.1.1")

    assert result == mock_app
    mock_card_builder_class.assert_called_once_with(
        agent=self.mock_agent, rpc_url="http://192.168.1.1:8000/"
    )

  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_agent_card_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_jsonrpc_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_rest_routes")
  async def test_to_a2a_with_custom_agent_card_object(
      self,
      mock_create_rest_routes,
      mock_create_jsonrpc_routes,
      mock_create_card_routes,
      mock_agent_executor_class,
      mock_request_handler_class,
      mock_task_store_class,
      mock_card_builder_class,
  ):
    """Test to_a2a with custom AgentCard object."""
    (
        mock_task_store,
        mock_agent_executor,
        mock_request_handler,
        mock_card_builder,
        mock_agent_card,
    ) = self._setup_standard_mocks(
        mock_card_builder_class,
        mock_task_store_class,
        mock_request_handler_class,
        mock_agent_executor_class,
        mock_create_card_routes,
        mock_create_jsonrpc_routes,
        mock_create_rest_routes,
    )
    custom_agent_card = Mock(spec=AgentCard)
    custom_agent_card.name = "custom_agent"

    app = to_a2a(self.mock_agent, agent_card=custom_agent_card)

    async with app.router.lifespan_context(app):
      pass

    # Verify the card builder build method was NOT called since we provided a card
    mock_card_builder.build.assert_not_called()

    # Verify handler was created with the custom card
    mock_request_handler_class.assert_called_once_with(
        agent_executor=mock_agent_executor,
        task_store=mock_task_store,
        agent_card=custom_agent_card,
        push_config_store=ANY,
    )

    # Verify route builders were called with the custom card
    mock_create_card_routes.assert_called_once_with(custom_agent_card)

  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_agent_card_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_jsonrpc_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_rest_routes")
  @patch("json.load")
  @patch("pathlib.Path.open")
  @patch("pathlib.Path")
  async def test_to_a2a_with_agent_card_file_path(
      self,
      mock_path_class,
      mock_open,
      mock_json_load,
      mock_create_rest_routes,
      mock_create_jsonrpc_routes,
      mock_create_card_routes,
      mock_agent_executor_class,
      mock_request_handler_class,
      mock_task_store_class,
      mock_card_builder_class,
  ):
    """Test to_a2a with agent card file path."""
    (
        mock_task_store,
        mock_agent_executor,
        mock_request_handler,
        mock_card_builder,
        mock_agent_card,
    ) = self._setup_standard_mocks(
        mock_card_builder_class,
        mock_task_store_class,
        mock_request_handler_class,
        mock_agent_executor_class,
        mock_create_card_routes,
        mock_create_jsonrpc_routes,
        mock_create_rest_routes,
    )

    # Mock file operations
    mock_path = Mock()
    mock_path_class.return_value = mock_path
    mock_file_handle = Mock()
    mock_context_manager = Mock()
    mock_context_manager.__enter__ = Mock(return_value=mock_file_handle)
    mock_context_manager.__exit__ = Mock(return_value=None)
    mock_path.open = Mock(return_value=mock_context_manager)

    agent_card_data = {
        "name": "file_agent",
        "description": "Test agent from file",
        "version": "1.0.0",
        "capabilities": {},
        "skills": [],
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
    }
    mock_json_load.return_value = agent_card_data

    app = to_a2a(self.mock_agent, agent_card="/path/to/agent_card.json")

    async with app.router.lifespan_context(app):
      pass

    # Verify file was opened and JSON was loaded
    mock_path_class.assert_called_once_with("/path/to/agent_card.json")
    mock_path.open.assert_called_once_with("r", encoding="utf-8")
    mock_json_load.assert_called_once_with(mock_file_handle)

    # Verify the card builder build method was NOT called since we provided a card
    mock_card_builder.build.assert_not_called()

    # Verify handler was created
    mock_request_handler_class.assert_called_once()
    args, kwargs = mock_request_handler_class.call_args
    assert kwargs.get("agent_executor") == mock_agent_executor
    assert kwargs.get("task_store") == mock_task_store

  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  @patch("pathlib.Path.open", side_effect=FileNotFoundError("File not found"))
  @patch("pathlib.Path")
  def test_to_a2a_with_invalid_agent_card_file_path(
      self,
      mock_path_class,
      mock_open,
      mock_starlette_class,
      mock_agent_executor_class,
      mock_request_handler_class,
      mock_task_store_class,
      mock_card_builder_class,
  ):
    """Test to_a2a with invalid agent card file path."""
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder
    mock_path = Mock()
    mock_path_class.return_value = mock_path

    with pytest.raises(ValueError, match="Failed to load agent card from"):
      to_a2a(self.mock_agent, agent_card="/invalid/path.json")

  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_agent_card_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_jsonrpc_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_rest_routes")
  async def test_to_a2a_with_lifespan(
      self,
      mock_create_rest_routes,
      mock_create_jsonrpc_routes,
      mock_create_card_routes,
      mock_agent_executor_class,
      mock_request_handler_class,
      mock_task_store_class,
      mock_card_builder_class,
  ):
    """Test to_a2a with a custom lifespan context manager."""
    (
        mock_task_store,
        mock_agent_executor,
        mock_request_handler,
        mock_card_builder,
        mock_agent_card,
    ) = self._setup_standard_mocks(
        mock_card_builder_class,
        mock_task_store_class,
        mock_request_handler_class,
        mock_agent_executor_class,
        mock_create_card_routes,
        mock_create_jsonrpc_routes,
        mock_create_rest_routes,
    )

    startup_called = False
    shutdown_called = False

    @asynccontextmanager
    async def custom_lifespan(app):
      nonlocal startup_called, shutdown_called
      startup_called = True
      app.state.test_value = "hello"
      yield
      shutdown_called = True

    app = to_a2a(self.mock_agent, lifespan=custom_lifespan)

    async with app.router.lifespan_context(app):
      # A2A setup should have run
      mock_agent_executor_class.assert_called_once()
      # User lifespan startup should have run
      assert startup_called
      assert app.state.test_value == "hello"

    # User lifespan shutdown should have run
    assert shutdown_called

  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_agent_card_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_jsonrpc_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_rest_routes")
  async def test_to_a2a_without_lifespan(
      self,
      mock_create_rest_routes,
      mock_create_jsonrpc_routes,
      mock_create_card_routes,
      mock_agent_executor_class,
      mock_request_handler_class,
      mock_task_store_class,
      mock_card_builder_class,
  ):
    """Test to_a2a without lifespan still runs setup_a2a."""
    (
        mock_task_store,
        mock_agent_executor,
        mock_request_handler,
        mock_card_builder,
        mock_agent_card,
    ) = self._setup_standard_mocks(
        mock_card_builder_class,
        mock_task_store_class,
        mock_request_handler_class,
        mock_agent_executor_class,
        mock_create_card_routes,
        mock_create_jsonrpc_routes,
        mock_create_rest_routes,
    )

    app = to_a2a(self.mock_agent)

    async with app.router.lifespan_context(app):
      # Verify setup_a2a ran
      mock_agent_executor_class.assert_called_once()
      mock_create_card_routes.assert_called_once_with(mock_agent_card)

  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_agent_card_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_jsonrpc_routes")
  @patch("google.adk.a2a.utils.agent_to_a2a.create_rest_routes")
  async def test_to_a2a_lifespan_setup_runs_before_user_lifespan(
      self,
      mock_create_rest_routes,
      mock_create_jsonrpc_routes,
      mock_create_card_routes,
      mock_agent_executor_class,
      mock_request_handler_class,
      mock_task_store_class,
      mock_card_builder_class,
  ):
    """Test that A2A setup runs before user lifespan startup."""
    (
        mock_task_store,
        mock_agent_executor,
        mock_request_handler,
        mock_card_builder,
        mock_agent_card,
    ) = self._setup_standard_mocks(
        mock_card_builder_class,
        mock_task_store_class,
        mock_request_handler_class,
        mock_agent_executor_class,
        mock_create_card_routes,
        mock_create_jsonrpc_routes,
        mock_create_rest_routes,
    )

    call_order = []

    original_create_card_routes = mock_create_card_routes.side_effect

    def track_card_routes(*args, **kwargs):
      call_order.append("setup_a2a")
      return []

    mock_create_card_routes.side_effect = track_card_routes

    @asynccontextmanager
    async def custom_lifespan(app):
      call_order.append("user_startup")
      yield
      call_order.append("user_shutdown")

    app = to_a2a(self.mock_agent, lifespan=custom_lifespan)

    async with app.router.lifespan_context(app):
      pass

    # A2A setup runs before user lifespan
    assert call_order == [
        "setup_a2a",
        "user_startup",
        "user_shutdown",
    ]

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
      pass

  def test_to_a2a_rejects_function_node(self):
    """to_a2a raises TypeError for a bare FunctionNode.

    FunctionNode is a BaseNode but is intended for use inside a
    Workflow, not as a standalone A2A root. Passing one directly used
    to silently produce a degenerate "custom agent" card; it now fails
    fast at to_a2a() call time.
    """

    async def my_fn(node_input):
      return f"echo: {node_input}"

    fn_node = FunctionNode(func=my_fn, name="echo_fn")

    with pytest.raises(
        TypeError, match="requires a BaseAgent or Workflow, got FunctionNode"
    ):
      to_a2a(fn_node)
