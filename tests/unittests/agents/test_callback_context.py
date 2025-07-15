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

from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
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


class TestToolContextListArtifacts:
  """Test that list_artifacts is available in ToolContext through inheritance."""

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
