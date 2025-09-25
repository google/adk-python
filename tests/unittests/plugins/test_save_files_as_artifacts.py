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

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import Mock

from google.adk.agents.invocation_context import InvocationContext
from google.adk.plugins.save_files_as_artifacts_plugin import SaveFilesAsArtifactsPlugin
from google.genai import types
import pytest


class TestSaveFilesAsArtifactsPlugin:
  """Test suite for SaveFilesAsArtifactsPlugin."""

  def setup_method(self):
    """Set up test fixtures."""
    self.plugin = SaveFilesAsArtifactsPlugin()

    # Mock invocation context
    self.mock_context = Mock(spec=InvocationContext)
    self.mock_context.artifact_service = AsyncMock()
    self.mock_context.app_name = "test_app"
    self.mock_context.user_id = "test_user"
    self.mock_context.invocation_id = "test_invocation_123"
    self.mock_context.session = Mock()
    self.mock_context.session.id = "test_session"

  @pytest.mark.asyncio
  async def test_save_files_with_display_name(self):
    """Test saving files when inline_data has display_name."""
    # Create a message with inline data
    inline_data = types.Blob(
        display_name="test_document.pdf",
        data=b"test data",
        mime_type="application/pdf",
    )

    original_part = types.Part(inline_data=inline_data)
    user_message = types.Content(parts=[original_part])

    # Execute the plugin
    result = await self.plugin.on_user_message_callback(
        invocation_context=self.mock_context, user_message=user_message
    )

    # Verify artifact was saved with correct filename
    self.mock_context.artifact_service.save_artifact.assert_called_once_with(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        filename="test_document.pdf",
        artifact=original_part,
    )

    # Verify message was modified with placeholder
    assert result.parts[0].text == '[Uploaded Artifact: "test_document.pdf"]'

  @pytest.mark.asyncio
  async def test_save_files_without_display_name(self):
    """Test saving files when inline_data has no display_name."""
    # Create inline data without display_name
    inline_data = types.Blob(
        display_name=None, data=b"test data", mime_type="application/pdf"
    )

    original_part = types.Part(inline_data=inline_data)
    user_message = types.Content(parts=[original_part])

    # Execute the plugin
    result = await self.plugin.on_user_message_callback(
        invocation_context=self.mock_context, user_message=user_message
    )

    # Verify artifact was saved with generated filename
    expected_filename = "artifact_test_invocation_123_0"
    self.mock_context.artifact_service.save_artifact.assert_called_once_with(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        filename=expected_filename,
        artifact=original_part,
    )

    # Verify message was modified with generated filename
    assert result.parts[0].text == f'[Uploaded Artifact: "{expected_filename}"]'

  @pytest.mark.asyncio
  async def test_multiple_files_in_message(self):
    """Test handling multiple files in a single message."""
    # Create message with multiple inline data parts
    inline_data1 = types.Blob(
        display_name="file1.txt", data=b"file1 content", mime_type="text/plain"
    )

    inline_data2 = types.Blob(
        display_name="file2.jpg", data=b"file2 content", mime_type="image/jpeg"
    )

    user_message = types.Content(
        parts=[
            types.Part(inline_data=inline_data1),
            types.Part(text="Some text between files"),
            types.Part(inline_data=inline_data2),
        ]
    )

    # Execute the plugin
    result = await self.plugin.on_user_message_callback(
        invocation_context=self.mock_context, user_message=user_message
    )

    # Verify both artifacts were saved
    assert self.mock_context.artifact_service.save_artifact.call_count == 2

    # Check first file
    first_call = (
        self.mock_context.artifact_service.save_artifact.call_args_list[0]
    )
    assert first_call[1]["filename"] == "file1.txt"

    # Check second file
    second_call = (
        self.mock_context.artifact_service.save_artifact.call_args_list[1]
    )
    assert second_call[1]["filename"] == "file2.jpg"

    # Verify message parts were modified correctly
    assert result.parts[0].text == '[Uploaded Artifact: "file1.txt"]'
    assert result.parts[1].text == "Some text between files"  # Unchanged
    assert result.parts[2].text == '[Uploaded Artifact: "file2.jpg"]'

  @pytest.mark.asyncio
  async def test_no_artifact_service(self):
    """Test behavior when artifact service is not available."""
    # Set artifact service to None
    self.mock_context.artifact_service = None

    inline_data = types.Blob(
        display_name="test.pdf", data=b"test data", mime_type="application/pdf"
    )

    user_message = types.Content(parts=[types.Part(inline_data=inline_data)])

    # Execute the plugin
    result = await self.plugin.on_user_message_callback(
        invocation_context=self.mock_context, user_message=user_message
    )

    # Should return original message unchanged
    assert result == user_message
    assert result.parts[0].inline_data == inline_data

  @pytest.mark.asyncio
  async def test_no_parts_in_message(self):
    """Test behavior when message has no parts."""
    user_message = types.Content(parts=[])

    # Execute the plugin
    result = await self.plugin.on_user_message_callback(
        invocation_context=self.mock_context, user_message=user_message
    )

    # Should return original message unchanged
    assert result == user_message
    assert result.parts == []

    # Should not try to save any artifacts
    self.mock_context.artifact_service.save_artifact.assert_not_called()

  @pytest.mark.asyncio
  async def test_parts_without_inline_data(self):
    """Test behavior with parts that don't have inline_data."""
    user_message = types.Content(
        parts=[types.Part(text="Hello world"), types.Part(text="No files here")]
    )

    # Execute the plugin
    result = await self.plugin.on_user_message_callback(
        invocation_context=self.mock_context, user_message=user_message
    )

    # Should return original message unchanged
    assert result == user_message
    assert result.parts[0].text == "Hello world"
    assert result.parts[1].text == "No files here"

    # Should not try to save any artifacts
    self.mock_context.artifact_service.save_artifact.assert_not_called()

  @pytest.mark.asyncio
  async def test_save_artifact_failure(self):
    """Test behavior when saving artifact fails."""
    # Mock save_artifact to raise an exception
    self.mock_context.artifact_service.save_artifact.side_effect = Exception(
        "Storage error"
    )

    inline_data = types.Blob(
        display_name="test.pdf", data=b"test data", mime_type="application/pdf"
    )

    original_part = types.Part(inline_data=inline_data)
    user_message = types.Content(parts=[original_part])

    # Execute the plugin - should not raise exception
    result = await self.plugin.on_user_message_callback(
        invocation_context=self.mock_context, user_message=user_message
    )

    # Should preserve original part when saving fails
    assert result.parts[0] == original_part
    assert result.parts[0].inline_data == inline_data

  @pytest.mark.asyncio
  async def test_mixed_success_and_failure(self):
    """Test behavior when some files save successfully and others fail."""
    # Mock save_artifact to succeed on first call, fail on second
    save_calls = 0

    def mock_save_artifact(*_args, **_kwargs):
      nonlocal save_calls
      save_calls += 1
      if save_calls == 2:
        raise Exception("Storage error on second file")
      return AsyncMock()

    self.mock_context.artifact_service.save_artifact.side_effect = (
        mock_save_artifact
    )

    inline_data1 = types.Blob(
        display_name="success.pdf",
        data=b"success data",
        mime_type="application/pdf",
    )

    inline_data2 = types.Blob(
        display_name="failure.pdf",
        data=b"failure data",
        mime_type="application/pdf",
    )

    original_part2 = types.Part(inline_data=inline_data2)
    user_message = types.Content(
        parts=[types.Part(inline_data=inline_data1), original_part2]
    )

    # Execute the plugin
    result = await self.plugin.on_user_message_callback(
        invocation_context=self.mock_context, user_message=user_message
    )

    # First file should be replaced with placeholder
    assert result.parts[0].text == '[Uploaded Artifact: "success.pdf"]'

    # Second file should remain unchanged due to failure
    assert result.parts[1] == original_part2
    assert result.parts[1].inline_data == inline_data2

  @pytest.mark.asyncio
  async def test_placeholder_text_format(self):
    """Test that placeholder text is formatted correctly."""
    inline_data = types.Blob(
        display_name="test file with spaces.docx",
        data=b"document data",
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    user_message = types.Content(parts=[types.Part(inline_data=inline_data)])

    # Execute the plugin
    result = await self.plugin.on_user_message_callback(
        invocation_context=self.mock_context, user_message=user_message
    )

    # Verify exact format of placeholder text
    expected_text = '[Uploaded Artifact: "test file with spaces.docx"]'
    assert result.parts[0].text == expected_text

  def test_plugin_name_default(self):
    """Test that plugin has correct default name."""
    plugin = SaveFilesAsArtifactsPlugin()
    assert plugin.name == "save_files_as_artifacts_plugin"
