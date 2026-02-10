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

import time
from unittest.mock import AsyncMock
from unittest.mock import Mock

from google.adk.flows.llm_flows.audio_cache_manager import AudioCacheConfig
from google.adk.flows.llm_flows.audio_cache_manager import AudioCacheManager
from google.genai import types
import pytest

from ... import testing_utils


class TestAudioCacheConfig:
  """Test the AudioCacheConfig class."""

  def test_default_values(self):
    """Test that default configuration values are set correctly."""
    config = AudioCacheConfig()
    assert config.max_cache_size_bytes == 10 * 1024 * 1024  # 10MB
    assert config.max_cache_duration_seconds == 300.0  # 5 minutes
    assert config.auto_flush_threshold == 100

  def test_custom_values(self):
    """Test that custom configuration values are set correctly."""
    config = AudioCacheConfig(
        max_cache_size_bytes=5 * 1024 * 1024,
        max_cache_duration_seconds=120.0,
        auto_flush_threshold=50,
    )
    assert config.max_cache_size_bytes == 5 * 1024 * 1024
    assert config.max_cache_duration_seconds == 120.0
    assert config.auto_flush_threshold == 50


class TestAudioCacheManager:
  """Test the AudioCacheManager class."""

  def setup_method(self):
    """Set up test fixtures."""
    self.config = AudioCacheConfig()
    self.manager = AudioCacheManager(self.config)

  @pytest.mark.asyncio
  async def test_cache_input_audio(self):
    """Test caching input audio data."""
    invocation_context = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )

    audio_blob = types.Blob(data=b'test_audio_data', mime_type='audio/pcm')

    # Initially no cache
    assert invocation_context.input_realtime_cache is None

    # Cache audio
    self.manager.cache_audio(invocation_context, audio_blob, 'input')

    # Verify cache is created and populated
    assert invocation_context.input_realtime_cache is not None
    assert len(invocation_context.input_realtime_cache) == 1

    entry = invocation_context.input_realtime_cache[0]
    assert entry.role == 'user'
    assert entry.data == audio_blob
    assert isinstance(entry.timestamp, float)

  @pytest.mark.asyncio
  async def test_cache_output_audio(self):
    """Test caching output audio data."""
    invocation_context = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )

    audio_blob = types.Blob(data=b'test_model_audio', mime_type='audio/wav')

    # Initially no cache
    assert invocation_context.output_realtime_cache is None

    # Cache audio
    self.manager.cache_audio(invocation_context, audio_blob, 'output')

    # Verify cache is created and populated
    assert invocation_context.output_realtime_cache is not None
    assert len(invocation_context.output_realtime_cache) == 1

    entry = invocation_context.output_realtime_cache[0]
    assert entry.role == 'model'
    assert entry.data == audio_blob
    assert isinstance(entry.timestamp, float)

  @pytest.mark.asyncio
  async def test_multiple_audio_caching(self):
    """Test caching multiple audio chunks."""
    invocation_context = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )

    # Cache multiple input audio chunks
    for i in range(3):
      audio_blob = types.Blob(data=f'input_{i}'.encode(), mime_type='audio/pcm')
      self.manager.cache_audio(invocation_context, audio_blob, 'input')

    # Cache multiple output audio chunks
    for i in range(2):
      audio_blob = types.Blob(
          data=f'output_{i}'.encode(), mime_type='audio/wav'
      )
      self.manager.cache_audio(invocation_context, audio_blob, 'output')

    # Verify all chunks are cached
    assert len(invocation_context.input_realtime_cache) == 3
    assert len(invocation_context.output_realtime_cache) == 2

  @pytest.mark.asyncio
  async def test_flush_caches_both(self):
    """Test flushing both input and output caches."""
    invocation_context = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )

    # Set up mock artifact service
    mock_artifact_service = AsyncMock()
    mock_artifact_service.save_artifact.return_value = 123
    invocation_context.artifact_service = mock_artifact_service

    # Cache some audio
    input_blob = types.Blob(data=b'input_data', mime_type='audio/pcm')
    output_blob = types.Blob(data=b'output_data', mime_type='audio/wav')
    self.manager.cache_audio(invocation_context, input_blob, 'input')
    self.manager.cache_audio(invocation_context, output_blob, 'output')

    # Flush caches
    await self.manager.flush_caches(invocation_context)

    # Verify caches are cleared
    assert invocation_context.input_realtime_cache == []
    assert invocation_context.output_realtime_cache == []

    # Verify artifact service was called twice (once for each cache)
    assert mock_artifact_service.save_artifact.call_count == 2

  @pytest.mark.asyncio
  async def test_flush_caches_selective(self):
    """Test selectively flushing only one cache."""
    invocation_context = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )

    # Set up mock artifact service
    mock_artifact_service = AsyncMock()
    mock_artifact_service.save_artifact.return_value = 123
    invocation_context.artifact_service = mock_artifact_service

    # Cache some audio
    input_blob = types.Blob(data=b'input_data', mime_type='audio/pcm')
    output_blob = types.Blob(data=b'output_data', mime_type='audio/wav')
    self.manager.cache_audio(invocation_context, input_blob, 'input')
    self.manager.cache_audio(invocation_context, output_blob, 'output')

    # Flush only input cache
    await self.manager.flush_caches(
        invocation_context, flush_user_audio=True, flush_model_audio=False
    )

    # Verify only input cache is cleared
    assert invocation_context.input_realtime_cache == []
    assert len(invocation_context.output_realtime_cache) == 1

    # Verify artifact service was called once
    assert mock_artifact_service.save_artifact.call_count == 1

  @pytest.mark.asyncio
  async def test_flush_empty_caches(self):
    """Test flushing when caches are empty."""
    invocation_context = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )

    # Set up mock artifact service
    mock_artifact_service = AsyncMock()
    invocation_context.artifact_service = mock_artifact_service

    # Flush empty caches (should not error)
    await self.manager.flush_caches(invocation_context)

    # Verify artifact service was not called
    mock_artifact_service.save_artifact.assert_not_called()

  @pytest.mark.asyncio
  async def test_flush_without_artifact_service(self):
    """Test flushing when no artifact service is available."""
    invocation_context = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )

    # No artifact service
    invocation_context.artifact_service = None

    # Cache some audio
    input_blob = types.Blob(data=b'input_data', mime_type='audio/pcm')
    self.manager.cache_audio(invocation_context, input_blob, 'input')

    # Flush should not error but should not clear cache either
    await self.manager.flush_caches(invocation_context)

    # Cache should remain (no actual flushing happened)
    assert len(invocation_context.input_realtime_cache) == 1

  @pytest.mark.asyncio
  async def test_flush_artifact_creation(self):
    """Test that artifacts are created correctly during flush."""
    invocation_context = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )

    # Set up mock services
    mock_artifact_service = AsyncMock()
    mock_artifact_service.save_artifact.return_value = 456
    mock_session_service = AsyncMock()

    invocation_context.artifact_service = mock_artifact_service
    invocation_context.session_service = mock_session_service

    # Cache audio with specific data
    test_data = b'specific_test_audio_data'
    audio_blob = types.Blob(data=test_data, mime_type='audio/pcm')
    self.manager.cache_audio(invocation_context, audio_blob, 'input')

    # Flush cache
    await self.manager.flush_caches(invocation_context)

    # Verify artifact was saved with correct data
    mock_artifact_service.save_artifact.assert_called_once()
    call_args = mock_artifact_service.save_artifact.call_args
    saved_artifact = call_args.kwargs['artifact']
    assert saved_artifact.inline_data.data == test_data
    assert saved_artifact.inline_data.mime_type == 'audio/pcm'

    # Verify session event was created
    mock_session_service.append_event.assert_not_called()

  def test_get_cache_stats_empty(self):
    """Test getting statistics for empty caches."""
    invocation_context = Mock()
    invocation_context.input_realtime_cache = None
    invocation_context.output_realtime_cache = None

    stats = self.manager.get_cache_stats(invocation_context)

    expected = {
        'input_chunks': 0,
        'output_chunks': 0,
        'input_bytes': 0,
        'output_bytes': 0,
        'total_chunks': 0,
        'total_bytes': 0,
    }
    assert stats == expected

  @pytest.mark.asyncio
  async def test_get_cache_stats_with_data(self):
    """Test getting statistics for caches with data."""
    invocation_context = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )

    # Cache some audio data of different sizes
    input_blob1 = types.Blob(data=b'12345', mime_type='audio/pcm')  # 5 bytes
    input_blob2 = types.Blob(
        data=b'1234567890', mime_type='audio/pcm'
    )  # 10 bytes
    output_blob = types.Blob(data=b'abc', mime_type='audio/wav')  # 3 bytes

    self.manager.cache_audio(invocation_context, input_blob1, 'input')
    self.manager.cache_audio(invocation_context, input_blob2, 'input')
    self.manager.cache_audio(invocation_context, output_blob, 'output')

    stats = self.manager.get_cache_stats(invocation_context)

    expected = {
        'input_chunks': 2,
        'output_chunks': 1,
        'input_bytes': 15,  # 5 + 10
        'output_bytes': 3,
        'total_chunks': 3,
        'total_bytes': 18,  # 15 + 3
    }
    assert stats == expected

  @pytest.mark.asyncio
  async def test_error_handling_in_flush(self):
    """Test error handling during cache flush operations."""
    invocation_context = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )

    # Set up mock artifact service that raises an error
    mock_artifact_service = AsyncMock()
    mock_artifact_service.save_artifact.side_effect = Exception(
        'Artifact service error'
    )
    invocation_context.artifact_service = mock_artifact_service

    # Cache some audio
    audio_blob = types.Blob(data=b'test_data', mime_type='audio/pcm')
    self.manager.cache_audio(invocation_context, audio_blob, 'input')

    # Flush should not raise exception but should log error and retain cache
    await self.manager.flush_caches(invocation_context)

    # Cache should remain since flush failed
    assert len(invocation_context.input_realtime_cache) == 1

  @pytest.mark.asyncio
  async def test_filename_uses_first_chunk_timestamp(self):
    """Test that the filename timestamp comes from the first audio chunk, not flush time."""
    invocation_context = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )

    # Set up mock services
    mock_artifact_service = AsyncMock()
    mock_artifact_service.save_artifact.return_value = 789
    mock_session_service = AsyncMock()

    invocation_context.artifact_service = mock_artifact_service
    invocation_context.session_service = mock_session_service

    # Cache multiple audio chunks with specific timestamps
    first_timestamp = 1234567890.123  # First chunk timestamp
    second_timestamp = 1234567891.456  # Second chunk timestamp (later)

    # Manually create audio cache entries with specific timestamps
    invocation_context.input_realtime_cache = []

    from google.adk.agents.invocation_context import RealtimeCacheEntry

    first_entry = RealtimeCacheEntry(
        role='user',
        data=types.Blob(data=b'first_chunk', mime_type='audio/pcm'),
        timestamp=first_timestamp,
    )

    second_entry = RealtimeCacheEntry(
        role='user',
        data=types.Blob(data=b'second_chunk', mime_type='audio/pcm'),
        timestamp=second_timestamp,
    )

    invocation_context.input_realtime_cache.extend([first_entry, second_entry])

    # Sleep briefly to ensure current time is different from first timestamp
    time.sleep(0.01)

    # Flush cache
    await self.manager.flush_caches(invocation_context)

    # Verify artifact was saved
    mock_artifact_service.save_artifact.assert_called_once()
    call_args = mock_artifact_service.save_artifact.call_args
    filename = call_args.kwargs['filename']

    # Extract timestamp from filename (format: input_audio_{timestamp}.pcm)
    expected_timestamp_ms = int(first_timestamp * 1000)
    assert (
        f'adk_live_audio_storage_input_audio_{expected_timestamp_ms}.pcm'
        == filename
    )

    # Verify the timestamp in filename matches first chunk, not current time
    current_timestamp_ms = int(time.time() * 1000)
    assert expected_timestamp_ms != current_timestamp_ms  # Should be different
    assert filename.startswith(
        f'adk_live_audio_storage_input_audio_{expected_timestamp_ms}'
    )

  @pytest.mark.asyncio
  async def test_flush_event_author_for_user_audio(self):
    """Test that flushed user audio events have 'user' as author."""
    invocation_context = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )

    # Set up mock artifact service
    mock_artifact_service = AsyncMock()
    mock_artifact_service.save_artifact.return_value = 123
    invocation_context.artifact_service = mock_artifact_service

    # Cache user input audio
    input_blob = types.Blob(data=b'user_audio_data', mime_type='audio/pcm')
    self.manager.cache_audio(invocation_context, input_blob, 'input')

    # Flush cache and get events
    events = await self.manager.flush_caches(
        invocation_context, flush_user_audio=True, flush_model_audio=False
    )

    # Verify event author is 'user' for user audio
    assert len(events) == 1
    assert events[0].author == 'user'
    assert events[0].content.role == 'user'

  @pytest.mark.asyncio
  async def test_flush_event_author_for_model_audio(self):
    """Test that flushed model audio events have agent name as author, not 'model'."""
    agent = testing_utils.create_test_agent(name='my_test_agent')
    invocation_context = await testing_utils.create_invocation_context(agent)

    # Set up mock artifact service
    mock_artifact_service = AsyncMock()
    mock_artifact_service.save_artifact.return_value = 123
    invocation_context.artifact_service = mock_artifact_service

    # Cache model output audio
    output_blob = types.Blob(data=b'model_audio_data', mime_type='audio/wav')
    self.manager.cache_audio(invocation_context, output_blob, 'output')

    # Flush cache and get events
    events = await self.manager.flush_caches(
        invocation_context, flush_user_audio=False, flush_model_audio=True
    )

    # Verify event author is agent name (not 'model') for model audio
    assert len(events) == 1
    assert events[0].author == 'my_test_agent'  # Agent name, not 'model'
    assert events[0].content.role == 'model'  # Role is still 'model'
