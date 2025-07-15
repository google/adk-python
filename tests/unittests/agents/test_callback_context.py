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

"""Tests for the CallbackContext class."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock

from google.adk.agents.callback_context import CallbackContext
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_tool import AuthConfig
from google.adk.tools.tool_context import ToolContext
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
def callback_context_with_artifact_service(
    mock_invocation_context, mock_artifact_service
):
  """Create a CallbackContext with a mock artifact service."""
  mock_invocation_context.artifact_service = mock_artifact_service
  return CallbackContext(mock_invocation_context)


@pytest.fixture
def callback_context_without_artifact_service(mock_invocation_context):
  """Create a CallbackContext without an artifact service."""
  mock_invocation_context.artifact_service = None
  return CallbackContext(mock_invocation_context)


@pytest.fixture
def mock_auth_config():
  """Create a mock auth config for testing."""
  mock_config = Mock(spec=AuthConfig)
  return mock_config


@pytest.fixture
def mock_auth_credential():
  """Create a mock auth credential for testing."""
  mock_credential = Mock(spec=AuthCredential)
  mock_credential.auth_type = AuthCredentialTypes.OAUTH2
  return mock_credential


class TestCallbackContextListArtifacts:
  """Test the list_artifacts method in CallbackContext."""

  @pytest.mark.asyncio
  async def test_list_artifacts_returns_artifact_keys(
      self, callback_context_with_artifact_service, mock_artifact_service
  ):
    """Test that list_artifacts returns the artifact keys from the service."""
    result = await callback_context_with_artifact_service.list_artifacts()

    assert result == ["file1.txt", "file2.txt", "file3.txt"]
    mock_artifact_service.list_artifact_keys.assert_called_once_with(
        app_name="test-app",
        user_id="test-user",
        session_id="test-session-id",
    )

  @pytest.mark.asyncio
  async def test_list_artifacts_returns_empty_list(
      self, callback_context_with_artifact_service, mock_artifact_service
  ):
    """Test that list_artifacts returns an empty list when no artifacts exist."""
    mock_artifact_service.list_artifact_keys.return_value = []

    result = await callback_context_with_artifact_service.list_artifacts()

    assert result == []
    mock_artifact_service.list_artifact_keys.assert_called_once_with(
        app_name="test-app",
        user_id="test-user",
        session_id="test-session-id",
    )

  @pytest.mark.asyncio
  async def test_list_artifacts_raises_value_error_when_service_is_none(
      self, callback_context_without_artifact_service
  ):
    """Test that list_artifacts raises ValueError when artifact service is None."""
    with pytest.raises(
        ValueError, match="Artifact service is not initialized."
    ):
      await callback_context_without_artifact_service.list_artifacts()

  @pytest.mark.asyncio
  async def test_list_artifacts_passes_through_service_exceptions(
      self, callback_context_with_artifact_service, mock_artifact_service
  ):
    """Test that list_artifacts passes through exceptions from the artifact service."""
    mock_artifact_service.list_artifact_keys.side_effect = Exception(
        "Service error"
    )

    with pytest.raises(Exception, match="Service error"):
      await callback_context_with_artifact_service.list_artifacts()


class TestCallbackContext:
  """Test suite for CallbackContext."""

  @pytest.mark.asyncio
  async def test_tool_context_inherits_list_artifacts(
      self, mock_invocation_context, mock_artifact_service
  ):
    """Test that ToolContext inherits the list_artifacts method from CallbackContext."""
    mock_invocation_context.artifact_service = mock_artifact_service
    tool_context = ToolContext(mock_invocation_context)

    result = await tool_context.list_artifacts()

    assert result == ["file1.txt", "file2.txt", "file3.txt"]
    mock_artifact_service.list_artifact_keys.assert_called_once_with(
        app_name="test-app",
        user_id="test-user",
        session_id="test-session-id",
    )

  @pytest.mark.asyncio
  async def test_tool_context_list_artifacts_raises_value_error_when_service_is_none(
      self, mock_invocation_context
  ):
    """Test that ToolContext's list_artifacts raises ValueError when artifact service is None."""
    mock_invocation_context.artifact_service = None
    tool_context = ToolContext(mock_invocation_context)

    with pytest.raises(
        ValueError, match="Artifact service is not initialized."
    ):
      await tool_context.list_artifacts()

  def test_tool_context_has_list_artifacts_method(self):
    """Test that ToolContext has the list_artifacts method available."""
    assert hasattr(ToolContext, "list_artifacts")
    assert callable(getattr(ToolContext, "list_artifacts"))

  def test_callback_context_has_list_artifacts_method(self):
    """Test that CallbackContext has the list_artifacts method available."""
    assert hasattr(CallbackContext, "list_artifacts")
    assert callable(getattr(CallbackContext, "list_artifacts"))

  def test_tool_context_shares_same_list_artifacts_method_with_callback_context(
      self,
  ):
    """Test that ToolContext and CallbackContext share the same list_artifacts method."""
    assert ToolContext.list_artifacts is CallbackContext.list_artifacts

  def test_initialization(self, mock_invocation_context):
    """Test CallbackContext initialization."""
    context = CallbackContext(mock_invocation_context)
    assert context._invocation_context == mock_invocation_context
    assert context._event_actions is not None
    assert context._state is not None

  @pytest.mark.asyncio
  async def test_save_credential_with_service(
      self, mock_invocation_context, mock_auth_config
  ):
    """Test save_credential when credential service is available."""
    # Mock credential service
    credential_service = AsyncMock()
    mock_invocation_context.credential_service = credential_service

    context = CallbackContext(mock_invocation_context)
    await context.save_credential(mock_auth_config)

    credential_service.save_credential.assert_called_once_with(
        mock_auth_config, context
    )

  @pytest.mark.asyncio
  async def test_save_credential_no_service(
      self, mock_invocation_context, mock_auth_config
  ):
    """Test save_credential when credential service is not available."""
    mock_invocation_context.credential_service = None

    context = CallbackContext(mock_invocation_context)

    with pytest.raises(
        ValueError, match="Credential service is not initialized"
    ):
      await context.save_credential(mock_auth_config)

  @pytest.mark.asyncio
  async def test_load_credential_with_service(
      self, mock_invocation_context, mock_auth_config, mock_auth_credential
  ):
    """Test load_credential when credential service is available."""
    # Mock credential service
    credential_service = AsyncMock()
    credential_service.load_credential.return_value = mock_auth_credential
    mock_invocation_context.credential_service = credential_service

    context = CallbackContext(mock_invocation_context)
    result = await context.load_credential(mock_auth_config)

    credential_service.load_credential.assert_called_once_with(
        mock_auth_config, context
    )
    assert result == mock_auth_credential

  @pytest.mark.asyncio
  async def test_load_credential_no_service(
      self, mock_invocation_context, mock_auth_config
  ):
    """Test load_credential when credential service is not available."""
    mock_invocation_context.credential_service = None

    context = CallbackContext(mock_invocation_context)

    with pytest.raises(
        ValueError, match="Credential service is not initialized"
    ):
      await context.load_credential(mock_auth_config)

  @pytest.mark.asyncio
  async def test_load_credential_returns_none(
      self, mock_invocation_context, mock_auth_config
  ):
    """Test load_credential returns None when credential not found."""
    # Mock credential service
    credential_service = AsyncMock()
    credential_service.load_credential.return_value = None
    mock_invocation_context.credential_service = credential_service

    context = CallbackContext(mock_invocation_context)
    result = await context.load_credential(mock_auth_config)

    credential_service.load_credential.assert_called_once_with(
        mock_auth_config, context
    )
    assert result is None

  @pytest.mark.asyncio
  async def test_save_artifact_integration(self, mock_invocation_context):
    """Test save_artifact to ensure credential methods follow same pattern."""
    # Mock artifact service
    artifact_service = AsyncMock()
    artifact_service.save_artifact.return_value = 1
    mock_invocation_context.artifact_service = artifact_service

    context = CallbackContext(mock_invocation_context)
    test_artifact = Part.from_text(text="test content")

    version = await context.save_artifact("test_file.txt", test_artifact)

    artifact_service.save_artifact.assert_called_once_with(
        app_name="test-app",
        user_id="test-user",
        session_id="test-session-id",
        filename="test_file.txt",
        artifact=test_artifact,
    )
    assert version == 1

  @pytest.mark.asyncio
  async def test_load_artifact_integration(self, mock_invocation_context):
    """Test load_artifact to ensure credential methods follow same pattern."""
    # Mock artifact service
    artifact_service = AsyncMock()
    test_artifact = Part.from_text(text="test content")
    artifact_service.load_artifact.return_value = test_artifact
    mock_invocation_context.artifact_service = artifact_service

    context = CallbackContext(mock_invocation_context)

    result = await context.load_artifact("test_file.txt")

    artifact_service.load_artifact.assert_called_once_with(
        app_name="test-app",
        user_id="test-user",
        session_id="test-session-id",
        filename="test_file.txt",
        version=None,
    )
    assert result == test_artifact
