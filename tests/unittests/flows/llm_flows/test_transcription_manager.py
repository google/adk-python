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
from unittest.mock import Mock

from google.adk.flows.llm_flows.transcription_manager import TranscriptionManager
from google.genai import types
import pytest

from ... import testing_utils


class TestTranscriptionManager:
  """Test the TranscriptionManager class."""

  def setup_method(self):
    """Set up test fixtures."""
    self.manager = TranscriptionManager()

  @pytest.mark.asyncio
  async def test_handle_input_transcription(self):
    """Test handling user input transcription events."""
    invocation_context = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )

    # Set up mock session service
    mock_session_service = AsyncMock()
    invocation_context.session_service = mock_session_service

    # Create test transcription
    transcription = types.Transcription(text='Hello from user')

    # Handle transcription
    await self.manager.handle_input_transcription(
        invocation_context, transcription
    )

    # Verify session service was called
    mock_session_service.append_event.assert_not_called()

  @pytest.mark.asyncio
  async def test_handle_output_transcription(self):
    """Test handling model output transcription events."""
    agent = testing_utils.create_test_agent()
    invocation_context = await testing_utils.create_invocation_context(agent)

    # Set up mock session service
    mock_session_service = AsyncMock()
    invocation_context.session_service = mock_session_service

    # Create test transcription
    transcription = types.Transcription(text='Hello from model')

    # Handle transcription
    await self.manager.handle_output_transcription(
        invocation_context, transcription
    )

    # Verify session service was called
    mock_session_service.append_event.assert_not_called()

  @pytest.mark.asyncio
  async def test_handle_multiple_transcriptions(self):
    """Test handling multiple transcription events."""
    invocation_context = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )

    # Set up mock session service
    mock_session_service = AsyncMock()
    invocation_context.session_service = mock_session_service

    # Handle multiple input transcriptions
    for i in range(3):
      transcription = types.Transcription(text=f'User message {i}')
      await self.manager.handle_input_transcription(
          invocation_context, transcription
      )

    # Handle multiple output transcriptions
    for i in range(2):
      transcription = types.Transcription(text=f'Model response {i}')
      await self.manager.handle_output_transcription(
          invocation_context, transcription
      )

    # Verify session service was called for each transcription
    assert mock_session_service.append_event.call_count == 0

  def test_get_transcription_stats_empty_session(self):
    """Test getting transcription statistics for empty session."""
    invocation_context = Mock()
    invocation_context.session.events = []

    stats = self.manager.get_transcription_stats(invocation_context)

    expected = {
        'input_transcriptions': 0,
        'output_transcriptions': 0,
        'total_transcriptions': 0,
    }
    assert stats == expected

  def test_get_transcription_stats_with_events(self):
    """Test getting transcription statistics for session with events."""
    invocation_context = Mock()

    # Create mock events
    input_event1 = Mock()
    input_event1.input_transcription = types.Transcription(text='User 1')
    input_event1.output_transcription = None

    input_event2 = Mock()
    input_event2.input_transcription = types.Transcription(text='User 2')
    input_event2.output_transcription = None

    output_event = Mock()
    output_event.input_transcription = None
    output_event.output_transcription = types.Transcription(
        text='Model response'
    )

    regular_event = Mock()
    regular_event.input_transcription = None
    regular_event.output_transcription = None

    invocation_context.session.events = [
        input_event1,
        output_event,
        input_event2,
        regular_event,
    ]

    stats = self.manager.get_transcription_stats(invocation_context)

    expected = {
        'input_transcriptions': 2,
        'output_transcriptions': 1,
        'total_transcriptions': 3,
    }
    assert stats == expected

  def test_get_transcription_stats_missing_attributes(self):
    """Test getting transcription statistics when events don't have transcription attributes."""
    invocation_context = Mock()

    # Create mock events and explicitly set transcription attributes to None
    event1 = Mock()
    event1.input_transcription = None
    event1.output_transcription = None

    event2 = Mock()
    event2.input_transcription = None
    event2.output_transcription = None

    invocation_context.session.events = [event1, event2]

    stats = self.manager.get_transcription_stats(invocation_context)

    expected = {
        'input_transcriptions': 0,
        'output_transcriptions': 0,
        'total_transcriptions': 0,
    }
    assert stats == expected

  @pytest.mark.asyncio
  async def test_transcription_event_fields(self):
    """Test that transcription events have correct field values."""
    invocation_context = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )

    # Set up mock session service
    mock_session_service = AsyncMock()
    invocation_context.session_service = mock_session_service

    # Create test transcription with specific content
    transcription = types.Transcription(
        text='Test transcription content', finished=True
    )

    # Handle input transcription
    await self.manager.handle_input_transcription(
        invocation_context, transcription
    )

  @pytest.mark.asyncio
  async def test_transcription_with_different_data_types(self):
    """Test handling transcriptions with different data types."""
    invocation_context = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )

    # Set up mock session service
    mock_session_service = AsyncMock()
    invocation_context.session_service = mock_session_service

    # Test with transcription that has basic fields only
    transcription = types.Transcription(
        text='Advanced transcription', finished=True
    )

    # Handle transcription
    await self.manager.handle_input_transcription(
        invocation_context, transcription
    )
