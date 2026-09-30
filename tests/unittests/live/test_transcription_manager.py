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

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import Mock

from google.adk.live._transcription_manager import TranscriptionManager
from google.genai import types
import pytest

from .. import testing_utils


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
    mock_session_service.append_event.assert_not_called()

  @pytest.mark.asyncio
  async def test_transcription_event_content_input(self):
    """Test that input transcription events have correct content structure."""
    invocation_context = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )

    transcription = types.Transcription(text='Test user input')

    # Handle input transcription
    event = await self.manager.handle_input_transcription(
        invocation_context, transcription
    )

    # Verify event structure
    assert event.author == 'user'
    assert event.input_transcription == transcription
    assert event.output_transcription is None
    assert event.invocation_id == invocation_context.invocation_id
    assert isinstance(event.timestamp, float)

  @pytest.mark.asyncio
  async def test_transcription_event_content_output(self):
    """Test that output transcription events have correct content structure."""
    agent = testing_utils.create_test_agent(name='my_test_agent')
    invocation_context = await testing_utils.create_invocation_context(agent)

    transcription = types.Transcription(text='Test model output')

    # Handle output transcription
    event = await self.manager.handle_output_transcription(
        invocation_context, transcription
    )

    # Verify event structure
    assert event.author == 'my_test_agent'  # Should use agent name
    assert event.output_transcription == transcription
    assert event.input_transcription is None
    assert event.invocation_id == invocation_context.invocation_id
    assert isinstance(event.timestamp, float)

  def test_get_transcription_stats_empty_session(self):
    """Test getting statistics from a session with no events."""
    invocation_context = Mock()
    invocation_context.session = Mock()
    invocation_context.session.events = []

    stats = self.manager.get_transcription_stats(invocation_context)

    expected = {
        'input_transcriptions': 0,
        'output_transcriptions': 0,
        'total_transcriptions': 0,
    }
    assert stats == expected

  def test_get_transcription_stats_with_events(self):
    """Test getting statistics from a session with transcription events."""
    invocation_context = Mock()
    invocation_context.session = Mock()

    # Create mock events
    event1 = Mock()
    event1.input_transcription = types.Transcription(text='Input 1')
    event1.output_transcription = None

    event2 = Mock()
    event2.input_transcription = types.Transcription(text='Input 2')
    event2.output_transcription = None

    event3 = Mock()
    event3.input_transcription = None
    event3.output_transcription = types.Transcription(text='Output 1')

    event4 = Mock()
    event4.input_transcription = None
    event4.output_transcription = None

    invocation_context.session.events = [event1, event2, event3, event4]

    stats = self.manager.get_transcription_stats(invocation_context)

    expected = {
        'input_transcriptions': 2,
        'output_transcriptions': 1,
        'total_transcriptions': 3,
    }
    assert stats == expected

  def test_get_transcription_stats_missing_attributes(self):
    """Test getting statistics when events lack transcription attributes."""
    invocation_context = Mock()
    invocation_context.session = Mock()

    # Create mock events without transcription attributes
    event1 = Mock(spec=['timestamp', 'author'])
    event2 = Mock(spec=['timestamp', 'author'])

    invocation_context.session.events = [event1, event2]

    stats = self.manager.get_transcription_stats(invocation_context)

    expected = {
        'input_transcriptions': 0,
        'output_transcriptions': 0,
        'total_transcriptions': 0,
    }
    assert stats == expected
