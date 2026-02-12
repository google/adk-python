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
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.tools.tool_confirmation import ToolConfirmation
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
  mock_context.artifact_service = None
  mock_context.credential_service = None
  mock_context.memory_service = None
  return mock_context


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
