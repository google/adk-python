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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.context import Context
from google.adk.auth import auth_handler
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_tool import AuthConfig
from google.adk.events.ui_widget import UiWidget
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.genai import types
from google.genai.types import Part
import pytest


@pytest.fixture
def mock_invocation_context():
  """Create a mock invocation context for testing."""
  mock_context = MagicMock()
  mock_context.invocation_id = "test-invocation-id"
  mock_context.agent.name = "test-agent-name"
  mock_context.session.state = {"key1": "value1", "key2": "value2"}
  mock_context.session.id = "test-session-id"
  mock_context.app_name = "test-app"
  mock_context.user_id = "test-user"
  mock_context.branch = "test-branch"
  mock_context.artifact_service = None
  mock_context.credential_service = None
  mock_context.memory_service = None
  return mock_context


def test_context_branch_returns_invocation_branch(mock_invocation_context):
  """Context.branch returns the branch from the underlying invocation context."""
  mock_invocation_context.branch = "test-branch"
  context = Context(invocation_context=mock_invocation_context)

  assert context.branch == "test-branch"


@pytest.fixture
def mock_artifact_service():
  """Create a mock artifact service for testing."""
  mock_service = AsyncMock()
  mock_service.list_artifact_keys.return_value = [
      "file1.txt",
      "file2.txt",
      "file3.txt",
  ]
  return mock_service


@pytest.fixture
def mock_auth_config(mocker):
  """Create a mock auth config for testing."""
  return mocker.create_autospec(AuthConfig, instance=True)


@pytest.fixture
def mock_auth_credential(mocker):
  """Create a mock auth credential for testing."""
  mock_credential = mocker.create_autospec(AuthCredential, instance=True)
  mock_credential.auth_type = AuthCredentialTypes.OAUTH2
  return mock_credential


class TestContextInitialization:
  """Test Context initialization."""

  def test_initialization_without_function_call_id(
      self, mock_invocation_context
  ):
    """Test Context initialization without function_call_id."""
    context = Context(mock_invocation_context)

    assert context._invocation_context == mock_invocation_context
    assert context._event_actions is not None
    assert context._state is not None
    assert context.function_call_id is None
    assert context.tool_confirmation is None

  def test_initialization_with_function_call_id(self, mock_invocation_context):
    """Test Context initialization with function_call_id."""
    context = Context(
        mock_invocation_context,
        function_call_id="test-function-call-id",
    )

    assert context.function_call_id == "test-function-call-id"
    assert context.tool_confirmation is None

  def test_initialization_with_tool_confirmation(self, mock_invocation_context):
    """Test Context initialization with tool_confirmation."""
    tool_confirmation = ToolConfirmation(
        hint="test hint", payload={"key": "value"}
    )
    context = Context(
        mock_invocation_context,
        function_call_id="test-function-call-id",
        tool_confirmation=tool_confirmation,
    )

    assert context.function_call_id == "test-function-call-id"
    assert context.tool_confirmation == tool_confirmation
    assert context.tool_confirmation.hint == "test hint"
    assert context.tool_confirmation.payload == {"key": "value"}

  def test_state_property(self, mock_invocation_context):
    """Test that state property returns mutable state."""
    context = Context(mock_invocation_context)

    assert context.state["key1"] == "value1"
    assert context.state["key2"] == "value2"

  def test_actions_property(self, mock_invocation_context):
    """Test that actions property returns event_actions."""
    context = Context(mock_invocation_context)

    assert context.actions is context._event_actions


class TestContextListArtifacts:
  """Test the list_artifacts method in Context."""

  async def test_list_artifacts_returns_artifact_keys(
      self, mock_invocation_context, mock_artifact_service
  ):
    """Test that list_artifacts returns the artifact keys from the service."""
    mock_invocation_context.artifact_service = mock_artifact_service
    context = Context(mock_invocation_context)

    result = await context.list_artifacts()

    assert result == ["file1.txt", "file2.txt", "file3.txt"]
    mock_artifact_service.list_artifact_keys.assert_called_once_with(
        app_name="test-app",
        user_id="test-user",
        session_id="test-session-id",
    )

  async def test_list_artifacts_raises_value_error_when_service_is_none(
      self, mock_invocation_context
  ):
    """Test that list_artifacts raises ValueError when no artifact service."""
    mock_invocation_context.artifact_service = None
    context = Context(mock_invocation_context)

    with pytest.raises(
        ValueError, match="Artifact service is not initialized."
    ):
      await context.list_artifacts()


class TestContextSaveLoadArtifact:
  """Test save_artifact and load_artifact methods in Context."""

  async def test_save_artifact(self, mock_invocation_context):
    """Test save_artifact method."""
    artifact_service = AsyncMock()
    artifact_service.save_artifact.return_value = 1
    mock_invocation_context.artifact_service = artifact_service

    context = Context(mock_invocation_context)
    test_artifact = Part.from_text(text="test content")

    version = await context.save_artifact("test_file.txt", test_artifact)

    artifact_service.save_artifact.assert_called_once_with(
        app_name="test-app",
        user_id="test-user",
        session_id="test-session-id",
        filename="test_file.txt",
        artifact=test_artifact,
        custom_metadata=None,
    )
    assert version == 1
    assert context.actions.artifact_delta["test_file.txt"] == 1

  async def test_load_artifact(self, mock_invocation_context):
    """Test load_artifact method."""
    artifact_service = AsyncMock()
    test_artifact = Part.from_text(text="test content")
    artifact_service.load_artifact.return_value = test_artifact
    mock_invocation_context.artifact_service = artifact_service

    context = Context(mock_invocation_context)

    result = await context.load_artifact("test_file.txt")

    artifact_service.load_artifact.assert_called_once_with(
        app_name="test-app",
        user_id="test-user",
        session_id="test-session-id",
        filename="test_file.txt",
        version=None,
    )
    assert result == test_artifact

  async def test_load_artifact_with_version(self, mock_invocation_context):
    """Test load_artifact method with specific version."""
    artifact_service = AsyncMock()
    test_artifact = Part.from_text(text="test content")
    artifact_service.load_artifact.return_value = test_artifact
    mock_invocation_context.artifact_service = artifact_service

    context = Context(mock_invocation_context)

    result = await context.load_artifact("test_file.txt", version=2)

    artifact_service.load_artifact.assert_called_once_with(
        app_name="test-app",
        user_id="test-user",
        session_id="test-session-id",
        filename="test_file.txt",
        version=2,
    )
    assert result == test_artifact


class TestContextCredentialMethods:
  """Test credential methods in Context."""

  async def test_save_credential_with_service(
      self, mock_invocation_context, mock_auth_config
  ):
    """Test save_credential when credential service is available."""
    credential_service = AsyncMock()
    mock_invocation_context.credential_service = credential_service

    context = Context(mock_invocation_context)
    await context.save_credential(mock_auth_config)

    credential_service.save_credential.assert_called_once_with(
        mock_auth_config, context
    )

  async def test_save_credential_no_service(
      self, mock_invocation_context, mock_auth_config
  ):
    """Test save_credential when credential service is not available."""
    mock_invocation_context.credential_service = None

    context = Context(mock_invocation_context)

    with pytest.raises(
        ValueError, match="Credential service is not initialized"
    ):
      await context.save_credential(mock_auth_config)

  async def test_load_credential_with_service(
      self, mock_invocation_context, mock_auth_config, mock_auth_credential
  ):
    """Test load_credential when credential service is available."""
    credential_service = AsyncMock()
    credential_service.load_credential.return_value = mock_auth_credential
    mock_invocation_context.credential_service = credential_service

    context = Context(mock_invocation_context)
    result = await context.load_credential(mock_auth_config)

    credential_service.load_credential.assert_called_once_with(
        mock_auth_config, context
    )
    assert result == mock_auth_credential

  async def test_load_credential_no_service(
      self, mock_invocation_context, mock_auth_config
  ):
    """Test load_credential when credential service is not available."""
    mock_invocation_context.credential_service = None

    context = Context(mock_invocation_context)

    with pytest.raises(
        ValueError, match="Credential service is not initialized"
    ):
      await context.load_credential(mock_auth_config)


class TestContextGetAuthResponse:
  """Test get_auth_response method in Context."""

  def test_get_auth_response(self, mock_invocation_context, mock_auth_config):
    """Test get_auth_response method."""
    context = Context(mock_invocation_context)

    with patch.object(
        auth_handler, "AuthHandler", autospec=True
    ) as mock_auth_handler:
      mock_handler_instance = mock_auth_handler.return_value
      mock_handler_instance.get_auth_response.return_value = "auth-response"

      result = context.get_auth_response(mock_auth_config)

      mock_auth_handler.assert_called_once_with(mock_auth_config)
      mock_handler_instance.get_auth_response.assert_called_once_with(
          context.state
      )
      assert result == "auth-response"


class TestContextRequestCredential:
  """Test request_credential method in Context."""

  def test_request_credential_with_function_call_id(
      self, mock_invocation_context, mock_auth_config
  ):
    """Test request_credential when function_call_id is set."""
    context = Context(
        mock_invocation_context,
        function_call_id="test-function-call-id",
    )

    with patch.object(
        auth_handler, "AuthHandler", autospec=True
    ) as mock_auth_handler:
      mock_handler_instance = mock_auth_handler.return_value
      mock_handler_instance.generate_auth_request.return_value = "auth-request"

      context.request_credential(mock_auth_config)

      mock_auth_handler.assert_called_once_with(mock_auth_config)
      mock_handler_instance.generate_auth_request.assert_called_once()
      assert (
          context.actions.requested_auth_configs["test-function-call-id"]
          == "auth-request"
      )

  def test_request_credential_without_function_call_id_raises(
      self, mock_invocation_context, mock_auth_config
  ):
    """Test request_credential raises ValueError when no function_call_id."""
    context = Context(mock_invocation_context)

    with pytest.raises(
        ValueError,
        match="request_credential requires function_call_id",
    ):
      context.request_credential(mock_auth_config)


class TestContextRequestConfirmation:
  """Test request_confirmation method in Context."""

  def test_request_confirmation_with_function_call_id(
      self, mock_invocation_context
  ):
    """Test request_confirmation when function_call_id is set."""
    context = Context(
        mock_invocation_context,
        function_call_id="test-function-call-id",
    )

    context.request_confirmation(
        hint="Please confirm", payload={"action": "delete"}
    )

    confirmation = context.actions.requested_tool_confirmations[
        "test-function-call-id"
    ]
    assert confirmation.hint == "Please confirm"
    assert confirmation.payload == {"action": "delete"}

  def test_request_confirmation_with_only_hint(self, mock_invocation_context):
    """Test request_confirmation with only hint provided."""
    context = Context(
        mock_invocation_context,
        function_call_id="test-function-call-id",
    )

    context.request_confirmation(hint="Confirm this action")

    confirmation = context.actions.requested_tool_confirmations[
        "test-function-call-id"
    ]
    assert confirmation.hint == "Confirm this action"
    assert confirmation.payload is None

  def test_request_confirmation_without_function_call_id_raises(
      self, mock_invocation_context
  ):
    """Test request_confirmation raises ValueError when no function_call_id."""
    context = Context(mock_invocation_context)

    with pytest.raises(
        ValueError,
        match="request_confirmation requires function_call_id",
    ):
      context.request_confirmation()


class TestContextMemoryMethods:
  """Test memory methods in Context."""

  async def test_add_session_to_memory_success(self, mock_invocation_context):
    """Test that add_session_to_memory calls the memory service correctly."""
    memory_service = AsyncMock()
    mock_invocation_context.memory_service = memory_service

    context = Context(mock_invocation_context)
    await context.add_session_to_memory()

    memory_service.add_session_to_memory.assert_called_once_with(
        mock_invocation_context.session
    )

  async def test_add_session_to_memory_no_service_raises(
      self, mock_invocation_context
  ):
    """Test that add_session_to_memory raises ValueError when memory service is None."""
    mock_invocation_context.memory_service = None

    context = Context(mock_invocation_context)

    with pytest.raises(
        ValueError,
        match=(
            r"Cannot add session to memory: memory service is not available\."
        ),
    ):
      await context.add_session_to_memory()

  async def test_search_memory_success(self, mock_invocation_context, mocker):
    """Test that search_memory calls the memory service correctly."""
    memory_service = AsyncMock()
    mock_search_response = mocker.create_autospec(
        SearchMemoryResponse, instance=True
    )
    memory_service.search_memory.return_value = mock_search_response
    mock_invocation_context.memory_service = memory_service

    context = Context(mock_invocation_context)
    result = await context.search_memory("test query")

    memory_service.search_memory.assert_called_once_with(
        app_name="test-app",
        user_id="test-user",
        query="test query",
    )
    assert result == mock_search_response

  async def test_search_memory_no_service_raises(self, mock_invocation_context):
    """Test that search_memory raises ValueError when memory service is None."""
    mock_invocation_context.memory_service = None

    context = Context(mock_invocation_context)

    with pytest.raises(ValueError, match="Memory service is not available."):
      await context.search_memory("test query")

  async def test_add_events_to_memory_success(self, mock_invocation_context):
    """Test that add_events_to_memory calls the memory service correctly."""
    memory_service = AsyncMock()
    mock_invocation_context.memory_service = memory_service
    test_event = MagicMock()

    context = Context(mock_invocation_context)
    await context.add_events_to_memory(
        events=[test_event],
        custom_metadata={"ttl": "6000s"},
    )

    memory_service.add_events_to_memory.assert_called_once_with(
        app_name=mock_invocation_context.session.app_name,
        user_id=mock_invocation_context.session.user_id,
        session_id=mock_invocation_context.session.id,
        events=[test_event],
        custom_metadata={"ttl": "6000s"},
    )

  async def test_add_events_to_memory_no_service_raises(
      self, mock_invocation_context
  ):
    """Test that add_events_to_memory raises ValueError when no service."""
    mock_invocation_context.memory_service = None

    context = Context(mock_invocation_context)

    with pytest.raises(
        ValueError,
        match=r"Cannot add events to memory: memory service is not available\.",
    ):
      await context.add_events_to_memory(events=[MagicMock()])

  @pytest.mark.asyncio
  async def test_add_memory_forwards_metadata(self, mock_invocation_context):
    """Tests that add_memory forwards memories and metadata."""
    memory_service = AsyncMock()
    mock_invocation_context.memory_service = memory_service
    memories = [
        MemoryEntry(content=types.Content(parts=[types.Part(text="fact one")]))
    ]
    metadata = {"ttl": "6000s"}

    context = Context(mock_invocation_context)
    await context.add_memory(memories=memories, custom_metadata=metadata)

    memory_service.add_memory.assert_called_once_with(
        app_name=mock_invocation_context.session.app_name,
        user_id=mock_invocation_context.session.user_id,
        memories=memories,
        custom_metadata=metadata,
    )

  @pytest.mark.asyncio
  async def test_add_memory_accepts_memory_entries(
      self, mock_invocation_context
  ):
    """Tests that add_memory forwards MemoryEntry inputs unchanged."""
    memory_service = AsyncMock()
    mock_invocation_context.memory_service = memory_service
    memory_entry = MemoryEntry(
        content=types.Content(parts=[types.Part(text="fact one")])
    )

    context = Context(mock_invocation_context)
    await context.add_memory(memories=[memory_entry])

    memory_service.add_memory.assert_called_once_with(
        app_name=mock_invocation_context.session.app_name,
        user_id=mock_invocation_context.session.user_id,
        memories=[memory_entry],
        custom_metadata=None,
    )

  async def test_add_memory_no_service_raises(self, mock_invocation_context):
    """Test that add_memory raises ValueError when no service."""
    mock_invocation_context.memory_service = None

    context = Context(mock_invocation_context)

    with pytest.raises(
        ValueError,
        match=r"Cannot add memory: memory service is not available\.",
    ):
      await context.add_memory(
          memories=[
              MemoryEntry(
                  content=types.Content(parts=[types.Part(text="fact one")])
              )
          ]
      )


class TestContextAddUiWidget:
  """Test render_ui_widget method in Context."""

  def test_render_ui_widget(self, mock_invocation_context):
    """Test that render_ui_widget appends a widget to actions."""

    context = Context(mock_invocation_context)
    widget = UiWidget(
        id="w1",
        provider="mcp",
        payload={"resource_uri": "ui://test-app"},
    )

    context.render_ui_widget(widget)

    assert context.actions.render_ui_widgets is not None
    assert len(context.actions.render_ui_widgets) == 1
    assert context.actions.render_ui_widgets[0] is widget

  def test_render_ui_widget_multiple(self, mock_invocation_context):
    """Test that calling render_ui_widget twice yields two widgets."""

    context = Context(mock_invocation_context)
    w1 = UiWidget(
        id="w1",
        provider="mcp",
        payload={"resource_uri": "ui://app-1"},
    )
    w2 = UiWidget(
        id="w2",
        provider="mcp",
        payload={"resource_uri": "ui://app-2"},
    )

    context.render_ui_widget(w1)
    context.render_ui_widget(w2)

    assert len(context.actions.render_ui_widgets) == 2
    assert context.actions.render_ui_widgets[0] is w1
    assert context.actions.render_ui_widgets[1] is w2

  def test_render_ui_widget_duplicate(self, mock_invocation_context):
    """Test that duplicate widgets by id are not added."""

    context = Context(mock_invocation_context)
    w1 = UiWidget(
        id="w1",
        provider="mcp",
        payload={"resource_uri": "ui://app-1"},
    )
    w2 = UiWidget(
        id="w1",
        provider="mcp",
        payload={"resource_uri": "ui://app-1-mod"},
    )

    context.render_ui_widget(w1)

    with pytest.raises(
        ValueError,
        match=(
            f"UI widget with ID '{w1.id}' already exists in the current event"
            " actions."
        ),
    ):
      context.render_ui_widget(w2)

    assert len(context.actions.render_ui_widgets) == 1
    assert context.actions.render_ui_widgets[0] is w1


class TestDeriveScheduler:
  """Tests for _derive_scheduler helper."""

  def test_derive_scheduler_no_parent(self):
    from google.adk.agents.context import _derive_scheduler

    assert _derive_scheduler(None) is None

  def test_derive_scheduler_with_parent_having_scheduler(self):
    from google.adk.agents.context import _derive_scheduler

    mock_parent = MagicMock()
    mock_scheduler = MagicMock()
    mock_parent._workflow_scheduler = mock_scheduler

    assert _derive_scheduler(mock_parent) is mock_scheduler

  def test_derive_scheduler_with_parent_no_scheduler(self):
    from google.adk.agents.context import _derive_scheduler
    from google.adk.workflow._dynamic_node_scheduler import DynamicNodeScheduler

    mock_parent = MagicMock()
    mock_parent._workflow_scheduler = None

    scheduler = _derive_scheduler(mock_parent)
    assert isinstance(scheduler, DynamicNodeScheduler)


class TestContextGetInvocationContext:
  """Test get_invocation_context method in Context."""

  def test_get_invocation_context_propagates_isolation_scope(
      self, mock_invocation_context
  ):
    """Test that get_invocation_context propagates isolation_scope to the copy."""
    context = Context(mock_invocation_context)
    context.isolation_scope = "test-isolation-scope"

    # Mock model_copy to return a mock copy
    mock_copy = MagicMock()
    mock_invocation_context.model_copy.return_value = mock_copy

    result = context.get_invocation_context()

    # Verify model_copy was called with correct update dict
    mock_invocation_context.model_copy.assert_called_once_with(
        update={
            "session": context.session,
            "isolation_scope": "test-isolation-scope",
        }
    )
    assert result is mock_copy


class TestContextRunNodeInternal:
  """Tests for the internal Context._run_node_internal orchestration method."""

  @pytest.mark.asyncio
  async def test_run_node_internal_returns_ctx_and_handles_resume_inputs(
      self, mock_invocation_context, mocker
  ):
    """Test that _run_node_internal correctly handles return_ctx and resume_inputs."""
    # Arrange
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.events.event_actions import EventActions

    agent_a = LlmAgent(name="agent_a", rerun_on_resume=True)
    root = LlmAgent(name="root", sub_agents=[agent_a], rerun_on_resume=True)
    agent_a.parent_agent = root

    root_ctx = Context(mock_invocation_context, node=root, run_id="1")

    child_ctx_a = Context(
        mock_invocation_context,
        parent_ctx=root_ctx,
        node=agent_a,
        run_id="1",
        event_actions=EventActions(),
    )
    child_ctx_a.output = "a_output"

    # Mock the standalone execution boundary
    mock_run_standalone = mocker.patch.object(
        Context,
        "_run_node_standalone",
        return_value=child_ctx_a,
    )

    # Act 1: Call _run_node_internal with return_ctx=True
    result_ctx = await root_ctx._run_node_internal(
        agent_a,
        node_input="a_input",
        return_ctx=True,
        resume_inputs={"some_key": "some_val"},
    )

    # Assert 1: It returns the child context object itself, not the output!
    assert result_ctx is child_ctx_a
    assert result_ctx.output == "a_output"

    # Assert 2: resume_inputs was correctly passed to _run_node_standalone
    mock_run_standalone.assert_called_once()
    _, kwargs = mock_run_standalone.call_args
    assert kwargs.get("resume_inputs") == {"some_key": "some_val"}


class TestContextRunNodeTransferLoop:
  """Tests for Context.run_node transfer loop orchestration."""

  @pytest.mark.asyncio
  async def test_sibling_transfer_executes_target_agent(
      self, mock_invocation_context
  ):
    """Sibling transfer routes execution to the target agent under the same parent context."""
    # Arrange
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.events.event_actions import EventActions

    # Create sibling agents under a root agent
    agent_b = LlmAgent(name="agent_b", rerun_on_resume=True)
    agent_a = LlmAgent(name="agent_a", rerun_on_resume=True)
    root = LlmAgent(
        name="root", sub_agents=[agent_a, agent_b], rerun_on_resume=True
    )
    agent_a.parent_agent = root
    agent_b.parent_agent = root

    # Create root context
    root_ctx = Context(mock_invocation_context, node=root, run_id="1")
    root_ctx._child_run_counters = {}

    # Mock child contexts returned by the scheduler
    child_ctx_a = Context(
        mock_invocation_context,
        parent_ctx=root_ctx,
        node=agent_a,
        run_id="1",
        event_actions=EventActions(transfer_to_agent="agent_b"),
    )

    child_ctx_b = Context(
        mock_invocation_context,
        parent_ctx=root_ctx,
        node=agent_b,
        run_id="1",
        event_actions=EventActions(),
    )
    child_ctx_b.output = "b_output"

    # Mock the scheduler boundary
    root_ctx._workflow_scheduler = AsyncMock(
        side_effect=[child_ctx_a, child_ctx_b]
    )

    # Act
    result = await root_ctx.run_node(agent_a, node_input="a_input")

    # Assert
    assert result == "b_output"
    assert root_ctx._workflow_scheduler.call_count == 2

    # First call was agent_a
    args1, kwargs1 = root_ctx._workflow_scheduler.call_args_list[0]
    assert args1[0] is root_ctx  # parent_ctx
    assert args1[1].name == agent_a.name  # node (cloned, so compare name)
    assert args1[2] == "a_input"  # node_input
    assert kwargs1.get("run_id") == "1"

    # Second call was agent_b (transferred sibling)
    args2, kwargs2 = root_ctx._workflow_scheduler.call_args_list[1]
    assert args2[0] is root_ctx  # parent_ctx (same sibling parent!)
    assert args2[1].name == agent_b.name  # node (cloned, so compare name)
    assert args2[2] is None  # transfer input is empty
    assert kwargs2.get("run_id") == "1"  # independent counter, so still 1

  @pytest.mark.asyncio
  async def test_parent_transfer_routes_execution_to_parent_agent(
      self, mock_invocation_context
  ):
    """Parent transfer routes execution to the parent agent, climbing up the context tree."""
    # Arrange
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.events.event_actions import EventActions

    # Create hierarchical agent tree
    child = LlmAgent(name="child", rerun_on_resume=True)
    parent = LlmAgent(name="parent", sub_agents=[child], rerun_on_resume=True)
    root = LlmAgent(name="root", sub_agents=[parent], rerun_on_resume=True)
    child.parent_agent = parent
    parent.parent_agent = root

    # Create root and parent contexts
    root_ctx = Context(mock_invocation_context, node=root, run_id="1")
    root_ctx._child_run_counters = {"parent": 1}

    parent_ctx = Context(
        mock_invocation_context, parent_ctx=root_ctx, node=parent, run_id="1"
    )
    parent_ctx._child_run_counters = {"child": 1}

    # Set up scheduler on both contexts
    mock_scheduler = AsyncMock()
    root_ctx._workflow_scheduler = mock_scheduler
    parent_ctx._workflow_scheduler = mock_scheduler

    # Mock child contexts returned by the scheduler
    child_ctx = Context(
        mock_invocation_context,
        parent_ctx=parent_ctx,
        node=child,
        run_id="1",
        event_actions=EventActions(transfer_to_agent="parent"),
    )

    parent_ctx2 = Context(
        mock_invocation_context,
        parent_ctx=root_ctx,
        node=parent,
        run_id="2",
        event_actions=EventActions(),
    )
    parent_ctx2.output = "parent_output"

    mock_scheduler.side_effect = [child_ctx, parent_ctx2]

    # Act
    # We run 'child' with use_as_output=True to test flag resetting
    result = await parent_ctx.run_node(
        child, node_input="child_input", use_as_output=True
    )

    # Assert
    assert result == "parent_output"
    assert mock_scheduler.call_count == 2

    # First call: child under parent_ctx
    args1, kwargs1 = mock_scheduler.call_args_list[0]
    assert args1[0] is parent_ctx
    assert args1[1].name == child.name  # node (cloned, so compare name)
    assert kwargs1.get("use_as_output") is True

    # Second call: parent under root_ctx (climbed up!)
    args2, kwargs2 = mock_scheduler.call_args_list[1]
    assert args2[0] is root_ctx  # Climbed up to root_ctx!
    assert args2[1].name == parent.name  # node (cloned, so compare name)
    # Sibling delegation flag must be reset because we crossed parent contexts!
    assert kwargs2.get("use_as_output") is False

  @pytest.mark.asyncio
  async def test_standalone_sibling_transfer_executes_target_agent(
      self, mock_invocation_context, mocker
  ):
    """Standalone mode sibling transfer routes execution to the target agent."""
    # Arrange
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.events.event_actions import EventActions

    # Create sibling agents under a root agent
    agent_b = LlmAgent(name="agent_b", rerun_on_resume=True)
    agent_a = LlmAgent(name="agent_a", rerun_on_resume=True)
    root = LlmAgent(
        name="root", sub_agents=[agent_a, agent_b], rerun_on_resume=True
    )
    agent_a.parent_agent = root
    agent_b.parent_agent = root

    # Create root context (Mode 2: _workflow_scheduler is None)
    root_ctx = Context(mock_invocation_context, node=root, run_id="1")
    root_ctx._child_run_counters = {}
    root_ctx._workflow_scheduler = None

    # Mock child contexts returned by the standalone runner
    child_ctx_a = Context(
        mock_invocation_context,
        parent_ctx=root_ctx,
        node=agent_a,
        run_id="1",
        event_actions=EventActions(transfer_to_agent="agent_b"),
    )

    child_ctx_b = Context(
        mock_invocation_context,
        parent_ctx=root_ctx,
        node=agent_b,
        run_id="1",
        event_actions=EventActions(),
    )
    child_ctx_b.output = "standalone_b_output"

    # Mock the standalone execution boundary on the class
    mock_run_standalone = mocker.patch.object(
        Context,
        "_run_node_standalone",
        side_effect=[child_ctx_a, child_ctx_b],
    )

    # Act
    result = await root_ctx.run_node(agent_a, node_input="a_input")

    # Assert
    assert result == "standalone_b_output"
    assert mock_run_standalone.call_count == 2

    # First call was agent_a
    args1, kwargs1 = mock_run_standalone.call_args_list[0]
    assert args1[0].name == agent_a.name
    assert args1[1] == "a_input"
    assert kwargs1.get("run_id") is None

    # Second call was agent_b
    args2, kwargs2 = mock_run_standalone.call_args_list[1]
    assert args2[0].name == agent_b.name
    assert args2[1] is None
    assert kwargs2.get("run_id") is None

  @pytest.mark.asyncio
  async def test_child_transfer_routes_execution_to_child_agent(
      self, mock_invocation_context
  ):
    """Child transfer routes execution to a sub-agent (downward in hierarchy)."""
    # Arrange
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.events.event_actions import EventActions

    # Create parent-child agents
    child = LlmAgent(name="child", rerun_on_resume=True)
    parent = LlmAgent(name="parent", sub_agents=[child], rerun_on_resume=True)
    child.parent_agent = parent

    # Create parent context (the starting context of the run)
    parent_ctx = Context(mock_invocation_context, node=None, parent_ctx=None)
    mock_scheduler = AsyncMock()
    parent_ctx._workflow_scheduler = mock_scheduler

    # Mock child contexts returned by the scheduler
    # 1. Parent runs, and decides to transfer to 'child'
    parent_run_ctx = Context(
        mock_invocation_context,
        parent_ctx=parent_ctx,  # Parent is parent_ctx!
        node=parent,
        run_id="1",
        event_actions=EventActions(transfer_to_agent="child"),
    )

    # 2. Child runs successfully
    child_run_ctx = Context(
        mock_invocation_context,
        parent_ctx=parent_run_ctx,  # Parent is parent_run_ctx (inherits scheduler!)
        node=child,
        run_id="1",
        event_actions=EventActions(),
    )
    child_run_ctx.output = "child_output"

    mock_scheduler.side_effect = [parent_run_ctx, child_run_ctx]

    # Act
    result = await parent_ctx.run_node(parent, node_input="parent_input")

    # Assert
    assert result == "child_output"
    assert mock_scheduler.call_count == 2

    # First call: parent node
    args1, kwargs1 = mock_scheduler.call_args_list[0]
    assert args1[0] is parent_ctx
    assert args1[1].name == parent.name

    # Second call: child node (run under parent's execution context!)
    args2, kwargs2 = mock_scheduler.call_args_list[1]
    assert args2[0] is parent_run_ctx  # Run under parent_run_ctx!
    assert args2[1].name == child.name
    assert kwargs2.get("run_id") == "1"

  @pytest.mark.asyncio
  async def test_three_layer_transfer_round_trip(self, mock_invocation_context):
    """Verify 3-layer round trip transfer (Root -> Child -> Grandchild -> Child -> Root)."""
    # Arrange
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.events.event_actions import EventActions

    # Create 3-layer agent tree
    grandchild = LlmAgent(name="grandchild", rerun_on_resume=True)
    child = LlmAgent(
        name="child", sub_agents=[grandchild], rerun_on_resume=True
    )
    root = LlmAgent(name="root", sub_agents=[child], rerun_on_resume=True)
    grandchild.parent_agent = child
    child.parent_agent = root

    # Create root context
    root_ctx = Context(mock_invocation_context, node=None, parent_ctx=None)
    mock_scheduler = AsyncMock()
    root_ctx._workflow_scheduler = mock_scheduler

    # Step 1: Root runs (child of root_ctx)
    root_run_ctx = Context(
        mock_invocation_context,
        parent_ctx=root_ctx,  # Parent is root_ctx!
        node=root,
        run_id="1",
        event_actions=EventActions(transfer_to_agent="child"),
    )

    # Step 2: Child runs (child of root_run_ctx)
    child_run_ctx = Context(
        mock_invocation_context,
        parent_ctx=root_run_ctx,  # Parent is root_run_ctx!
        node=child,
        run_id="1",
        event_actions=EventActions(transfer_to_agent="grandchild"),
    )

    # Step 3: Grandchild runs (child of child_run_ctx)
    grandchild_run_ctx = Context(
        mock_invocation_context,
        parent_ctx=child_run_ctx,  # Parent is child_run_ctx!
        node=grandchild,
        run_id="1",
        event_actions=EventActions(transfer_to_agent="child"),
    )

    # Step 4: Child runs again (sibling of child_run_ctx, parent is root_run_ctx!)
    child_run_ctx2 = Context(
        mock_invocation_context,
        parent_ctx=root_run_ctx,  # Parent is root_run_ctx!
        node=child,
        run_id="2",
        event_actions=EventActions(transfer_to_agent="root"),
    )

    # Step 5: Root runs again (sibling of root_run_ctx, parent is root_ctx!)
    root_run_ctx2 = Context(
        mock_invocation_context,
        parent_ctx=root_ctx,  # Parent is root_ctx!
        node=root,
        run_id="2",
        event_actions=EventActions(),
    )
    root_run_ctx2.output = "final_root_output"

    mock_scheduler.side_effect = [
        root_run_ctx,
        child_run_ctx,
        grandchild_run_ctx,
        child_run_ctx2,
        root_run_ctx2,
    ]

    # Act
    result = await root_ctx.run_node(root, node_input="start")

    # Assert
    assert result == "final_root_output"
    assert mock_scheduler.call_count == 5

    # Verify call sequence
    calls = mock_scheduler.call_args_list

    # 1. root (run_id 1)
    assert calls[0][0][0] is root_ctx
    assert calls[0][0][1].name == root.name
    assert calls[0][1].get("run_id") == "1"

    # 2. child (run_id 1, under root_run_ctx)
    assert calls[1][0][0] is root_run_ctx
    assert calls[1][0][1].name == child.name
    assert calls[1][1].get("run_id") == "1"

    # 3. grandchild (run_id 1, under child_run_ctx)
    assert calls[2][0][0] is child_run_ctx
    assert calls[2][0][1].name == grandchild.name
    assert calls[2][1].get("run_id") == "1"

    # 4. child (run_id 2, under root_run_ctx - climbed up!)
    assert calls[3][0][0] is root_run_ctx
    assert calls[3][0][1].name == child.name
    assert calls[3][1].get("run_id") == "2"

    # 5. root (run_id 2, under root_ctx - climbed up!)
    assert calls[4][0][0] is root_ctx
    assert calls[4][0][1].name == root.name
    assert calls[4][1].get("run_id") == "2"

  @pytest.mark.asyncio
  async def test_transfer_preserves_use_as_output_for_original_context(
      self, mock_invocation_context
  ):
    """Verify that use_as_output is preserved when transferring back to self."""
    # Arrange
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.events.event_actions import EventActions

    # Create root and child agents
    child = LlmAgent(name="child", rerun_on_resume=True)
    root = LlmAgent(name="root", sub_agents=[child], rerun_on_resume=True)
    child.parent_agent = root

    # Create root context
    root_ctx = Context(mock_invocation_context, node=None, parent_ctx=None)
    mock_scheduler = AsyncMock()
    root_ctx._workflow_scheduler = mock_scheduler

    # Step 1: Root runs, transfers to child
    root_run_ctx1 = Context(
        mock_invocation_context,
        parent_ctx=root_ctx,
        node=root,
        run_id="1",
        event_actions=EventActions(transfer_to_agent="child"),
    )

    # Step 2: Child runs, transfers back to root
    child_run_ctx = Context(
        mock_invocation_context,
        parent_ctx=root_run_ctx1,
        node=child,
        run_id="1",
        event_actions=EventActions(transfer_to_agent="root"),
    )

    # Step 3: Root runs again, completes
    root_run_ctx2 = Context(
        mock_invocation_context,
        parent_ctx=root_ctx,
        node=root,
        run_id="2",
        event_actions=EventActions(),
    )
    root_run_ctx2.output = "final_output"

    mock_scheduler.side_effect = [
        root_run_ctx1,
        child_run_ctx,
        root_run_ctx2,
    ]

    # Act
    result = await root_ctx.run_node(
        root, node_input="start", use_as_output=True
    )

    # Assert
    assert result == "final_output"
    assert mock_scheduler.call_count == 3

    calls = mock_scheduler.call_args_list

    # 1. root (run_id 1) should have use_as_output=True
    assert calls[0][1].get("use_as_output") is True

    # 2. child (run_id 1, under root_run_ctx1) should have use_as_output=False
    assert calls[1][1].get("use_as_output") is False

    # 3. root (run_id 2, under root_ctx) should have use_as_output=True again!
    assert calls[2][1].get("use_as_output") is True
