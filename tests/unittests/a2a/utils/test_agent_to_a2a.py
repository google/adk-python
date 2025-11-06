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

import sys
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

# Skip all tests in this module if Python version is less than 3.10
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10), reason="A2A requires Python 3.10+"
)

# Import dependencies with version checking
try:
  from a2a.server.apps import A2AStarletteApplication
  from a2a.server.request_handlers import DefaultRequestHandler
  from a2a.server.tasks import InMemoryTaskStore
  from a2a.types import AgentCard
  from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
  from google.adk.a2a.utils.agent_card_builder import AgentCardBuilder
  from google.adk.a2a.utils.agent_to_a2a import to_a2a
  from google.adk.agents.base_agent import BaseAgent
  from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
  from google.adk.auth.credential_service.in_memory_credential_service import InMemoryCredentialService
  from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
  from google.adk.runners import Runner
  from google.adk.sessions.in_memory_session_service import InMemorySessionService
  from starlette.applications import Starlette
except ImportError as e:
  if sys.version_info < (3, 10):
    # Imports are not needed since tests will be skipped due to pytestmark.
    # The imported names are only used within test methods, not at module level,
    # so no NameError occurs during module compilation.
    pass
  else:
    raise e


class TestToA2A:
  """Test suite for to_a2a function."""

  def setup_method(self):
    """Set up test fixtures."""
    self.mock_agent = Mock(spec=BaseAgent)
    self.mock_agent.name = "test_agent"
    self.mock_agent.description = "Test agent description"

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_default_parameters(
      self,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_request_handler_class,
      mock_agent_executor_class,
  ):
    """Test to_a2a with default parameters."""
    # Arrange
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_task_store = Mock(spec=InMemoryTaskStore)
    mock_task_store_class.return_value = mock_task_store
    mock_agent_executor = Mock(spec=A2aAgentExecutor)
    mock_agent_executor_class.return_value = mock_agent_executor
    mock_request_handler = Mock(spec=DefaultRequestHandler)
    mock_request_handler_class.return_value = mock_request_handler
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder

    # Act
    result = to_a2a(self.mock_agent)

    # Assert
    assert result == mock_app
    mock_starlette_class.assert_called_once()
    mock_task_store_class.assert_called_once()
    mock_agent_executor_class.assert_called_once()
    mock_request_handler_class.assert_called_once_with(
        agent_executor=mock_agent_executor, task_store=mock_task_store
    )
    mock_card_builder_class.assert_called_once_with(
        agent=self.mock_agent, rpc_url="http://localhost:8000/"
    )
    mock_app.add_event_handler.assert_called_once_with(
        "startup", mock_app.add_event_handler.call_args[0][1]
    )

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_with_custom_runner(
      self,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_request_handler_class,
      mock_agent_executor_class,
  ):
    """Test to_a2a with a custom runner."""
    # Arrange
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_task_store = Mock(spec=InMemoryTaskStore)
    mock_task_store_class.return_value = mock_task_store
    mock_agent_executor = Mock(spec=A2aAgentExecutor)
    mock_agent_executor_class.return_value = mock_agent_executor
    mock_request_handler = Mock(spec=DefaultRequestHandler)
    mock_request_handler_class.return_value = mock_request_handler
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder
    custom_runner = Mock(spec=Runner)

    # Act
    result = to_a2a(self.mock_agent, runner=custom_runner)

    # Assert
    assert result == mock_app
    mock_starlette_class.assert_called_once()
    mock_task_store_class.assert_called_once()
    mock_agent_executor_class.assert_called_once_with(runner=custom_runner)
    mock_request_handler_class.assert_called_once_with(
        agent_executor=mock_agent_executor, task_store=mock_task_store
    )
    mock_card_builder_class.assert_called_once_with(
        agent=self.mock_agent, rpc_url="http://localhost:8000/"
    )
    mock_app.add_event_handler.assert_called_once_with(
        "startup", mock_app.add_event_handler.call_args[0][1]
    )

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_custom_host_port(
      self,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_request_handler_class,
      mock_agent_executor_class,
  ):
    """Test to_a2a with custom host and port."""
    # Arrange
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_task_store = Mock(spec=InMemoryTaskStore)
    mock_task_store_class.return_value = mock_task_store
    mock_agent_executor = Mock(spec=A2aAgentExecutor)
    mock_agent_executor_class.return_value = mock_agent_executor
    mock_request_handler = Mock(spec=DefaultRequestHandler)
    mock_request_handler_class.return_value = mock_request_handler
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder

    # Act
    result = to_a2a(self.mock_agent, host="example.com", port=9000)

    # Assert
    assert result == mock_app
    mock_card_builder_class.assert_called_once_with(
        agent=self.mock_agent, rpc_url="http://example.com:9000/"
    )

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_agent_without_name(
      self,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_request_handler_class,
      mock_agent_executor_class,
  ):
    """Test to_a2a with agent that has no name."""
    # Arrange
    self.mock_agent.name = None
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_task_store = Mock(spec=InMemoryTaskStore)
    mock_task_store_class.return_value = mock_task_store
    mock_agent_executor = Mock(spec=A2aAgentExecutor)
    mock_agent_executor_class.return_value = mock_agent_executor
    mock_request_handler = Mock(spec=DefaultRequestHandler)
    mock_request_handler_class.return_value = mock_request_handler
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder

    # Act
    result = to_a2a(self.mock_agent)

    # Assert
    assert result == mock_app
    # The create_runner function should use "adk_agent" as default name
    # We can't directly test the create_runner function, but we can verify
    # the agent executor was created with the runner function

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_creates_runner_with_correct_services(
      self,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_request_handler_class,
      mock_agent_executor_class,
  ):
    """Test that the create_runner function creates Runner with correct services."""
    # Arrange
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_task_store = Mock(spec=InMemoryTaskStore)
    mock_task_store_class.return_value = mock_task_store
    mock_agent_executor = Mock(spec=A2aAgentExecutor)
    mock_agent_executor_class.return_value = mock_agent_executor
    mock_request_handler = Mock(spec=DefaultRequestHandler)
    mock_request_handler_class.return_value = mock_request_handler
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder

    # Act
    result = to_a2a(self.mock_agent)

    # Assert
    assert result == mock_app
    # Verify that the agent executor was created with a runner function
    mock_agent_executor_class.assert_called_once()
    call_args = mock_agent_executor_class.call_args
    assert "runner" in call_args[1]
    runner_func = call_args[1]["runner"]
    assert callable(runner_func)

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  @patch("google.adk.a2a.utils.agent_to_a2a.Runner")
  async def test_create_runner_function_creates_runner_correctly(
      self,
      mock_runner_class,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_request_handler_class,
      mock_agent_executor_class,
  ):
    """Test that the create_runner function creates Runner with correct parameters."""
    # Arrange
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_task_store = Mock(spec=InMemoryTaskStore)
    mock_task_store_class.return_value = mock_task_store
    mock_agent_executor = Mock(spec=A2aAgentExecutor)
    mock_agent_executor_class.return_value = mock_agent_executor
    mock_request_handler = Mock(spec=DefaultRequestHandler)
    mock_request_handler_class.return_value = mock_request_handler
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder
    mock_runner = Mock(spec=Runner)
    mock_runner_class.return_value = mock_runner

    # Act
    result = to_a2a(self.mock_agent)

    # Assert
    assert result == mock_app
    # Get the runner function that was passed to A2aAgentExecutor
    call_args = mock_agent_executor_class.call_args
    runner_func = call_args[1]["runner"]

    # Call the runner function to verify it creates Runner correctly
    runner_result = await runner_func()

    # Verify Runner was created with correct parameters
    mock_runner_class.assert_called_once_with(
        app_name="test_agent",
        agent=self.mock_agent,
        artifact_service=mock_runner_class.call_args[1]["artifact_service"],
        session_service=mock_runner_class.call_args[1]["session_service"],
        memory_service=mock_runner_class.call_args[1]["memory_service"],
        credential_service=mock_runner_class.call_args[1]["credential_service"],
    )

    # Verify the services are of the correct types
    call_args = mock_runner_class.call_args[1]
    assert isinstance(call_args["artifact_service"], InMemoryArtifactService)
    assert isinstance(call_args["session_service"], InMemorySessionService)
    assert isinstance(call_args["memory_service"], InMemoryMemoryService)
    assert isinstance(
        call_args["credential_service"], InMemoryCredentialService
    )

    assert runner_result == mock_runner

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  @patch("google.adk.a2a.utils.agent_to_a2a.Runner")
  async def test_create_runner_function_with_agent_without_name(
      self,
      mock_runner_class,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_request_handler_class,
      mock_agent_executor_class,
  ):
    """Test create_runner function with agent that has no name."""
    # Arrange
    self.mock_agent.name = None
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_task_store = Mock(spec=InMemoryTaskStore)
    mock_task_store_class.return_value = mock_task_store
    mock_agent_executor = Mock(spec=A2aAgentExecutor)
    mock_agent_executor_class.return_value = mock_agent_executor
    mock_request_handler = Mock(spec=DefaultRequestHandler)
    mock_request_handler_class.return_value = mock_request_handler
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder
    mock_runner = Mock(spec=Runner)
    mock_runner_class.return_value = mock_runner

    # Act
    result = to_a2a(self.mock_agent)

    # Assert
    assert result == mock_app
    # Get the runner function that was passed to A2aAgentExecutor
    call_args = mock_agent_executor_class.call_args
    runner_func = call_args[1]["runner"]

    # Call the runner function to verify it creates Runner correctly
    await runner_func()

    # Verify Runner was created with default app_name when agent has no name
    mock_runner_class.assert_called_once_with(
        app_name="adk_agent",
        agent=self.mock_agent,
        artifact_service=mock_runner_class.call_args[1]["artifact_service"],
        session_service=mock_runner_class.call_args[1]["session_service"],
        memory_service=mock_runner_class.call_args[1]["memory_service"],
        credential_service=mock_runner_class.call_args[1]["credential_service"],
    )

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  @patch("google.adk.a2a.utils.agent_to_a2a.A2AStarletteApplication")
  async def test_setup_a2a_function_builds_agent_card_and_configures_routes(
      self,
      mock_a2a_app_class,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_request_handler_class,
      mock_agent_executor_class,
  ):
    """Test that the setup_a2a function builds agent card and configures A2A routes."""
    # Arrange
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
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
    mock_a2a_app = Mock(spec=A2AStarletteApplication)
    mock_a2a_app_class.return_value = mock_a2a_app

    # Act
    result = to_a2a(self.mock_agent)

    # Assert
    assert result == mock_app
    # Get the setup_a2a function that was added as startup handler
    startup_handler = mock_app.add_event_handler.call_args[0][1]

    # Call the setup_a2a function
    await startup_handler()

    # Verify agent card was built
    mock_card_builder.build.assert_called_once()

    # Verify A2A Starlette application was created
    mock_a2a_app_class.assert_called_once_with(
        agent_card=mock_agent_card,
        http_handler=mock_request_handler,
    )

    # Verify routes were added to the main app
    mock_a2a_app.add_routes_to_app.assert_called_once_with(mock_app)

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  @patch("google.adk.a2a.utils.agent_to_a2a.A2AStarletteApplication")
  async def test_setup_a2a_function_handles_agent_card_build_failure(
      self,
      mock_a2a_app_class,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_request_handler_class,
      mock_agent_executor_class,
  ):
    """Test that setup_a2a function properly handles agent card build failure."""
    # Arrange
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_task_store = Mock(spec=InMemoryTaskStore)
    mock_task_store_class.return_value = mock_task_store
    mock_agent_executor = Mock(spec=A2aAgentExecutor)
    mock_agent_executor_class.return_value = mock_agent_executor
    mock_request_handler = Mock(spec=DefaultRequestHandler)
    mock_request_handler_class.return_value = mock_request_handler
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder
    mock_card_builder.build = AsyncMock(side_effect=Exception("Build failed"))
    mock_a2a_app = Mock(spec=A2AStarletteApplication)
    mock_a2a_app_class.return_value = mock_a2a_app

    # Act
    result = to_a2a(self.mock_agent)

    # Assert
    assert result == mock_app
    # Get the setup_a2a function that was added as startup handler
    startup_handler = mock_app.add_event_handler.call_args[0][1]

    # Call the setup_a2a function and expect it to raise the exception
    with pytest.raises(Exception, match="Build failed"):
      await startup_handler()

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_returns_starlette_app(
      self,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_request_handler_class,
      mock_agent_executor_class,
  ):
    """Test that to_a2a returns a Starlette application."""
    # Arrange
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_task_store = Mock(spec=InMemoryTaskStore)
    mock_task_store_class.return_value = mock_task_store
    mock_agent_executor = Mock(spec=A2aAgentExecutor)
    mock_agent_executor_class.return_value = mock_agent_executor
    mock_request_handler = Mock(spec=DefaultRequestHandler)
    mock_request_handler_class.return_value = mock_request_handler
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder

    # Act
    result = to_a2a(self.mock_agent)

    # Assert
    assert isinstance(result, Mock)  # Mock of Starlette
    assert result == mock_app

  def test_to_a2a_with_none_agent(self):
    """Test that to_a2a raises error when agent is None."""
    # Act & Assert
    with pytest.raises(ValueError, match="Agent cannot be None or empty."):
      to_a2a(None)

  def test_to_a2a_with_invalid_agent_type(self):
    """Test that to_a2a raises error when agent is not a BaseAgent."""
    # Arrange
    invalid_agent = "not an agent"

    # Act & Assert
    # The error occurs during startup when building the agent card
    app = to_a2a(invalid_agent)
    with pytest.raises(
        AttributeError, match="'str' object has no attribute 'name'"
    ):
      import asyncio

      asyncio.run(app.router.on_startup[0]())

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_with_custom_port_zero(
      self,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_request_handler_class,
      mock_agent_executor_class,
  ):
    """Test to_a2a with port 0 (dynamic port assignment)."""
    # Arrange
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_task_store = Mock(spec=InMemoryTaskStore)
    mock_task_store_class.return_value = mock_task_store
    mock_agent_executor = Mock(spec=A2aAgentExecutor)
    mock_agent_executor_class.return_value = mock_agent_executor
    mock_request_handler = Mock(spec=DefaultRequestHandler)
    mock_request_handler_class.return_value = mock_request_handler
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder

    # Act
    result = to_a2a(self.mock_agent, port=0)

    # Assert
    assert result == mock_app
    mock_card_builder_class.assert_called_once_with(
        agent=self.mock_agent, rpc_url="http://localhost:0/"
    )

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_with_empty_string_host(
      self,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_request_handler_class,
      mock_agent_executor_class,
  ):
    """Test to_a2a with empty string host."""
    # Arrange
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_task_store = Mock(spec=InMemoryTaskStore)
    mock_task_store_class.return_value = mock_task_store
    mock_agent_executor = Mock(spec=A2aAgentExecutor)
    mock_agent_executor_class.return_value = mock_agent_executor
    mock_request_handler = Mock(spec=DefaultRequestHandler)
    mock_request_handler_class.return_value = mock_request_handler
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder

    # Act
    result = to_a2a(self.mock_agent, host="")

    # Assert
    assert result == mock_app
    mock_card_builder_class.assert_called_once_with(
        agent=self.mock_agent, rpc_url="http://:8000/"
    )

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_with_negative_port(
      self,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_request_handler_class,
      mock_agent_executor_class,
  ):
    """Test to_a2a with negative port number."""
    # Arrange
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_task_store = Mock(spec=InMemoryTaskStore)
    mock_task_store_class.return_value = mock_task_store
    mock_agent_executor = Mock(spec=A2aAgentExecutor)
    mock_agent_executor_class.return_value = mock_agent_executor
    mock_request_handler = Mock(spec=DefaultRequestHandler)
    mock_request_handler_class.return_value = mock_request_handler
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder

    # Act
    result = to_a2a(self.mock_agent, port=-1)

    # Assert
    assert result == mock_app
    mock_card_builder_class.assert_called_once_with(
        agent=self.mock_agent, rpc_url="http://localhost:-1/"
    )

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_with_very_large_port(
      self,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_request_handler_class,
      mock_agent_executor_class,
  ):
    """Test to_a2a with very large port number."""
    # Arrange
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_task_store = Mock(spec=InMemoryTaskStore)
    mock_task_store_class.return_value = mock_task_store
    mock_agent_executor = Mock(spec=A2aAgentExecutor)
    mock_agent_executor_class.return_value = mock_agent_executor
    mock_request_handler = Mock(spec=DefaultRequestHandler)
    mock_request_handler_class.return_value = mock_request_handler
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder

    # Act
    result = to_a2a(self.mock_agent, port=65535)

    # Assert
    assert result == mock_app
    mock_card_builder_class.assert_called_once_with(
        agent=self.mock_agent, rpc_url="http://localhost:65535/"
    )

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_with_special_characters_in_host(
      self,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_request_handler_class,
      mock_agent_executor_class,
  ):
    """Test to_a2a with special characters in host name."""
    # Arrange
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_task_store = Mock(spec=InMemoryTaskStore)
    mock_task_store_class.return_value = mock_task_store
    mock_agent_executor = Mock(spec=A2aAgentExecutor)
    mock_agent_executor_class.return_value = mock_agent_executor
    mock_request_handler = Mock(spec=DefaultRequestHandler)
    mock_request_handler_class.return_value = mock_request_handler
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder

    # Act
    result = to_a2a(self.mock_agent, host="test-host.example.com")

    # Assert
    assert result == mock_app
    mock_card_builder_class.assert_called_once_with(
        agent=self.mock_agent, rpc_url="http://test-host.example.com:8000/"
    )

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  def test_to_a2a_with_ip_address_host(
      self,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_request_handler_class,
      mock_agent_executor_class,
  ):
    """Test to_a2a with IP address as host."""
    # Arrange
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_task_store = Mock(spec=InMemoryTaskStore)
    mock_task_store_class.return_value = mock_task_store
    mock_agent_executor = Mock(spec=A2aAgentExecutor)
    mock_agent_executor_class.return_value = mock_agent_executor
    mock_request_handler = Mock(spec=DefaultRequestHandler)
    mock_request_handler_class.return_value = mock_request_handler
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder

    # Act
    result = to_a2a(self.mock_agent, host="192.168.1.1")

    # Assert
    assert result == mock_app
    mock_card_builder_class.assert_called_once_with(
        agent=self.mock_agent, rpc_url="http://192.168.1.1:8000/"
    )

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  @patch("google.adk.a2a.utils.agent_to_a2a.A2AStarletteApplication")
  async def test_to_a2a_with_custom_agent_card_object(
      self,
      mock_a2a_app_class,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_request_handler_class,
      mock_agent_executor_class,
  ):
    """Test to_a2a with custom AgentCard object."""
    # Arrange
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_task_store = Mock(spec=InMemoryTaskStore)
    mock_task_store_class.return_value = mock_task_store
    mock_agent_executor = Mock(spec=A2aAgentExecutor)
    mock_agent_executor_class.return_value = mock_agent_executor
    mock_request_handler = Mock(spec=DefaultRequestHandler)
    mock_request_handler_class.return_value = mock_request_handler
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder
    mock_a2a_app = Mock(spec=A2AStarletteApplication)
    mock_a2a_app_class.return_value = mock_a2a_app

    # Create a custom agent card
    custom_agent_card = Mock(spec=AgentCard)
    custom_agent_card.name = "custom_agent"

    # Act
    result = to_a2a(self.mock_agent, agent_card=custom_agent_card)

    # Assert
    assert result == mock_app
    # Get the setup_a2a function that was added as startup handler
    startup_handler = mock_app.add_event_handler.call_args[0][1]

    # Call the setup_a2a function
    await startup_handler()

    # Verify the card builder build method was NOT called since we provided a card
    mock_card_builder.build.assert_not_called()

    # Verify A2A Starlette application was created with custom card
    mock_a2a_app_class.assert_called_once_with(
        agent_card=custom_agent_card,
        http_handler=mock_request_handler,
    )

    # Verify routes were added to the main app
    mock_a2a_app.add_routes_to_app.assert_called_once_with(mock_app)

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  @patch("google.adk.a2a.utils.agent_to_a2a.A2AStarletteApplication")
  @patch("json.load")
  @patch("pathlib.Path.open")
  @patch("pathlib.Path")
  async def test_to_a2a_with_agent_card_file_path(
      self,
      mock_path_class,
      mock_open,
      mock_json_load,
      mock_a2a_app_class,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_request_handler_class,
      mock_agent_executor_class,
  ):
    """Test to_a2a with agent card file path."""
    # Arrange
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_task_store = Mock(spec=InMemoryTaskStore)
    mock_task_store_class.return_value = mock_task_store
    mock_agent_executor = Mock(spec=A2aAgentExecutor)
    mock_agent_executor_class.return_value = mock_agent_executor
    mock_request_handler = Mock(spec=DefaultRequestHandler)
    mock_request_handler_class.return_value = mock_request_handler
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder
    mock_a2a_app = Mock(spec=A2AStarletteApplication)
    mock_a2a_app_class.return_value = mock_a2a_app

    # Mock file operations
    mock_path = Mock()
    mock_path_class.return_value = mock_path
    mock_file_handle = Mock()
    # Create a proper context manager mock
    mock_context_manager = Mock()
    mock_context_manager.__enter__ = Mock(return_value=mock_file_handle)
    mock_context_manager.__exit__ = Mock(return_value=None)
    mock_path.open = Mock(return_value=mock_context_manager)

    # Mock agent card data from file with all required fields
    agent_card_data = {
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
    mock_json_load.return_value = agent_card_data

    # Act
    result = to_a2a(self.mock_agent, agent_card="/path/to/agent_card.json")

    # Assert
    assert result == mock_app
    # Get the setup_a2a function that was added as startup handler
    startup_handler = mock_app.add_event_handler.call_args[0][1]

    # Call the setup_a2a function
    await startup_handler()

    # Verify file was opened and JSON was loaded
    mock_path_class.assert_called_once_with("/path/to/agent_card.json")
    mock_path.open.assert_called_once_with("r", encoding="utf-8")
    mock_json_load.assert_called_once_with(mock_file_handle)

    # Verify the card builder build method was NOT called since we provided a card
    mock_card_builder.build.assert_not_called()

    # Verify A2A Starlette application was created with loaded card
    mock_a2a_app_class.assert_called_once()
    args, kwargs = mock_a2a_app_class.call_args
    assert kwargs["http_handler"] == mock_request_handler
    # The agent_card should be an AgentCard object created from loaded data
    assert hasattr(kwargs["agent_card"], "name")

  @patch("google.adk.a2a.utils.agent_to_a2a.A2aAgentExecutor")
  @patch("google.adk.a2a.utils.agent_to_a2a.DefaultRequestHandler")
  @patch("google.adk.a2a.utils.agent_to_a2a.InMemoryTaskStore")
  @patch("google.adk.a2a.utils.agent_to_a2a.AgentCardBuilder")
  @patch("google.adk.a2a.utils.agent_to_a2a.Starlette")
  @patch("pathlib.Path.open", side_effect=FileNotFoundError("File not found"))
  @patch("pathlib.Path")
  def test_to_a2a_with_invalid_agent_card_file_path(
      self,
      mock_path_class,
      mock_open,
      mock_starlette_class,
      mock_card_builder_class,
      mock_task_store_class,
      mock_request_handler_class,
      mock_agent_executor_class,
  ):
    """Test to_a2a with invalid agent card file path."""
    # Arrange
    mock_app = Mock(spec=Starlette)
    mock_starlette_class.return_value = mock_app
    mock_task_store = Mock(spec=InMemoryTaskStore)
    mock_task_store_class.return_value = mock_task_store
    mock_agent_executor = Mock(spec=A2aAgentExecutor)
    mock_agent_executor_class.return_value = mock_agent_executor
    mock_request_handler = Mock(spec=DefaultRequestHandler)
    mock_request_handler_class.return_value = mock_request_handler
    mock_card_builder = Mock(spec=AgentCardBuilder)
    mock_card_builder_class.return_value = mock_card_builder

    mock_path = Mock()
    mock_path_class.return_value = mock_path

    # Act & Assert
    with pytest.raises(ValueError, match="Failed to load agent card from"):
      to_a2a(self.mock_agent, agent_card="/invalid/path.json")
