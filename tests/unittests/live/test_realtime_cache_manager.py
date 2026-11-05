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

import logging
import re
import time
from unittest.mock import AsyncMock
from unittest.mock import Mock

from google.adk.agents.invocation_context import InvocationContext
from google.adk.live._realtime_cache_manager import RealtimeCacheConfig
from google.adk.live._realtime_cache_manager import RealtimeCacheEntry
from google.adk.live._realtime_cache_manager import RealtimeCacheManager
from google.adk.live._visual_frames import unpack_visual_frames
from google.adk.platform import time as platform_time
from google.adk.utils.content_utils import _filter_media_parts
from google.genai import types
import pydantic
import pytest

from .. import testing_utils


def _image(data: bytes | None, mime_type: str = 'image/jpeg') -> types.Blob:
  return types.Blob(data=data, mime_type=mime_type)


async def _context_with_artifact_service() -> InvocationContext:
  """Returns a context whose artifact service records every save."""
  context = await testing_utils.create_invocation_context(
      testing_utils.create_test_agent()
  )
  artifact_service = AsyncMock()
  artifact_service.save_artifact.return_value = 0
  context.artifact_service = artifact_service
  return context


def _saved_blobs(context: InvocationContext) -> list[types.Blob]:
  """Returns the blobs passed to save_artifact, in call order."""
  return [
      call.kwargs['artifact'].inline_data
      for call in context.artifact_service.save_artifact.call_args_list
  ]


def _saved_frames(context: InvocationContext) -> list[list[bytes]]:
  """Returns the frame bytes in each archive passed to save_artifact."""
  return [
      [frame.blob.data for frame in unpack_visual_frames(blob.data)[0]]
      for blob in _saved_blobs(context)
  ]


class TestRealtimeCacheEntry:
  """RealtimeCacheEntry moved here verbatim; its validation must not drift."""

  def test_unknown_fields_are_rejected(self):
    # `extra="forbid"` came with the model from agents.invocation_context.
    # Losing it in the move would silently accept malformed cache entries.
    with pytest.raises(pydantic.ValidationError):
      RealtimeCacheEntry(
          role='user',
          data=types.Blob(data=b'x', mime_type='audio/pcm'),
          timestamp=0.0,
          not_a_field=1,
      )

  def test_accepts_the_declared_fields(self):
    entry = RealtimeCacheEntry(
        role='user',
        data=types.Blob(data=b'x', mime_type='audio/pcm'),
        timestamp=1.5,
    )
    assert entry.role == 'user'
    assert entry.timestamp == 1.5


class TestRealtimeCacheConfig:
  """Test the RealtimeCacheConfig class."""

  def test_default_values(self):
    """Test that default configuration values are set correctly."""
    config = RealtimeCacheConfig()
    assert config.max_cache_size_bytes == 10 * 1024 * 1024  # 10MB
    assert config.max_cache_duration_seconds == 300.0  # 5 minutes
    assert config.auto_flush_threshold == 100

  def test_custom_values(self):
    """Test that custom configuration values are set correctly."""
    config = RealtimeCacheConfig(
        max_cache_size_bytes=5 * 1024 * 1024,
        max_cache_duration_seconds=120.0,
        auto_flush_threshold=50,
    )
    assert config.max_cache_size_bytes == 5 * 1024 * 1024
    assert config.max_cache_duration_seconds == 120.0
    assert config.auto_flush_threshold == 50

  def test_visual_frame_limit_default(self):
    """The frame cache holds at most 600 frames by default."""
    config = RealtimeCacheConfig()

    assert config.max_visual_cache_frames == 600

  def test_rejects_a_visual_frame_limit_below_one(self):
    """A frame limit below 1 is rejected when the config is built."""
    with pytest.raises(ValueError, match='must be at least 1'):
      RealtimeCacheConfig(max_visual_cache_frames=0)


class TestRealtimeCacheManager:
  """Test the RealtimeCacheManager class."""

  def setup_method(self):
    """Set up test fixtures."""
    self.config = RealtimeCacheConfig()
    self.manager = RealtimeCacheManager(self.config)

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
  async def test_cache_audio_rejects_missing_byte_data(self):
    invocation_context = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )

    with pytest.raises(ValueError, match='must contain byte data'):
      self.manager.cache_audio(
          invocation_context,
          types.Blob(data=None, mime_type='audio/pcm'),
          'input',
      )

    assert invocation_context.input_realtime_cache is None

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
  async def test_flush_without_artifact_service_still_clears_cache(self):
    """Audio that cannot be saved is dropped rather than accumulated."""
    invocation_context = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )

    # No artifact service
    invocation_context.artifact_service = None

    # Cache some audio
    input_blob = types.Blob(data=b'input_data', mime_type='audio/pcm')
    output_blob = types.Blob(data=b'output_data', mime_type='audio/wav')
    self.manager.cache_audio(invocation_context, input_blob, 'input')
    self.manager.cache_audio(invocation_context, output_blob, 'output')

    # Flush should not error.
    events = await self.manager.flush_caches(invocation_context)

    # There is nowhere to write the audio, so no event is produced, but both
    # caches are emptied. Retaining them would grow for the whole session.
    assert not events
    assert not invocation_context.input_realtime_cache
    assert not invocation_context.output_realtime_cache

  @pytest.mark.asyncio
  async def test_flush_after_failure_does_not_repeat_earlier_audio(self):
    """A failed flush must not leave its audio in the next artifact."""
    invocation_context = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )

    mock_artifact_service = AsyncMock()
    mock_artifact_service.save_artifact.side_effect = [
        Exception('Artifact service error'),
        123,
    ]
    invocation_context.artifact_service = mock_artifact_service

    self.manager.cache_audio(
        invocation_context,
        types.Blob(data=b'first_turn', mime_type='audio/pcm'),
        'input',
    )
    await self.manager.flush_caches(invocation_context)

    self.manager.cache_audio(
        invocation_context,
        types.Blob(data=b'second_turn', mime_type='audio/pcm'),
        'input',
    )
    await self.manager.flush_caches(invocation_context)

    saved = mock_artifact_service.save_artifact.call_args.kwargs['artifact']
    assert saved.inline_data.data == b'second_turn'

  @pytest.mark.asyncio
  async def test_flush_keeps_audio_cached_while_saving(self):
    """Audio arriving during the save belongs to the next flush."""
    invocation_context = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )

    def cache_while_saving(**kwargs):
      self.manager.cache_audio(
          invocation_context,
          types.Blob(data=b'mid_flush', mime_type='audio/pcm'),
          'input',
      )
      return 123

    mock_artifact_service = AsyncMock()
    mock_artifact_service.save_artifact.side_effect = cache_while_saving
    invocation_context.artifact_service = mock_artifact_service

    self.manager.cache_audio(
        invocation_context,
        types.Blob(data=b'first_turn', mime_type='audio/pcm'),
        'input',
    )
    await self.manager.flush_caches(invocation_context)

    saved = mock_artifact_service.save_artifact.call_args.kwargs['artifact']
    assert saved.inline_data.data == b'first_turn'
    assert [
        entry.data.data for entry in invocation_context.input_realtime_cache
    ] == [b'mid_flush']

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

    # Flush should log the error rather than raise.
    events = await self.manager.flush_caches(invocation_context)

    # The save failed, so no event is produced, but the cache is still
    # emptied. A retry would fail the same way and the audio would pile up.
    assert not events
    assert not invocation_context.input_realtime_cache

  @pytest.mark.asyncio
  async def test_filename_drops_mime_type_parameters(self):
    """MIME parameters like `;rate=24000` stay out of the file extension."""
    invocation_context = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )
    mock_artifact_service = AsyncMock()
    mock_artifact_service.save_artifact.return_value = 0
    invocation_context.artifact_service = mock_artifact_service
    audio_blob = types.Blob(data=b'audio', mime_type='audio/pcm;rate=24000')
    self.manager.cache_audio(invocation_context, audio_blob, 'output')

    await self.manager.flush_caches(invocation_context)

    call_args = mock_artifact_service.save_artifact.call_args
    assert call_args.kwargs['filename'].endswith('.pcm')
    assert (
        call_args.kwargs['artifact'].inline_data.mime_type
        == 'audio/pcm;rate=24000'
    )

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

  @pytest.mark.asyncio
  async def test_cache_blob_keeps_non_audio_blobs_out_of_audio_caches(self):
    """Only audio/* blobs reach the audio caches."""
    ctx = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )
    in_audio = types.Blob(data=b'user_pcm', mime_type='audio/pcm')
    out_audio = types.Blob(data=b'model_wav', mime_type='audio/wav;rate=24000')
    self.manager.cache_blob(ctx, in_audio, 'input')
    self.manager.cache_blob(ctx, out_audio, 'output')
    self.manager.cache_blob(
        ctx, types.Blob(data=b'jpg', mime_type='image/jpeg'), 'input'
    )
    self.manager.cache_blob(
        ctx, types.Blob(data=b'mp4', mime_type='video/mp4'), 'output'
    )
    self.manager.cache_blob(ctx, types.Blob(data=b'raw'), 'input')

    assert [e.data for e in ctx.input_realtime_cache] == [in_audio]
    assert [e.data for e in ctx.output_realtime_cache] == [out_audio]

  @pytest.mark.asyncio
  async def test_cache_blob_caches_a_copy_of_the_blob(self):
    """Later changes to the caller's blob don't change the cached copy."""
    ctx = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )
    blob = types.Blob(data=b'chunk_1', mime_type='audio/pcm')
    self.manager.cache_blob(ctx, blob, 'input')
    blob.data = b'chunk_2'

    assert [e.data.data for e in ctx.input_realtime_cache] == [b'chunk_1']

  @pytest.mark.asyncio
  async def test_cache_blob_saves_image_frames_on_flush(self):
    """The user's image frames are cached and saved on the next flush."""
    context = await _context_with_artifact_service()
    self.manager.cache_blob(context, _image(b'user_frame'), 'input')

    events = await self.manager.flush_caches(context)

    assert [event.author for event in events] == ['user']
    assert _saved_frames(context) == [[b'user_frame']]

  @pytest.mark.asyncio
  async def test_cache_blob_does_not_save_image_frames_from_the_model(self):
    """Live models send no image frames, so any from the model are not saved."""
    context = await _context_with_artifact_service()
    self.manager.cache_blob(
        context, _image(b'model_frame', 'image/png'), 'output'
    )

    events = await self.manager.flush_caches(context)

    assert not events
    context.artifact_service.save_artifact.assert_not_called()

  @pytest.mark.parametrize(
      'mime_type',
      ['video/mp4', 'application/octet-stream', None, 'IMAGE/JPEG'],
  )
  @pytest.mark.asyncio
  async def test_cache_blob_skips_video_and_unknown_blobs(self, mime_type):
    """Blobs that are neither audio/* nor image/* are not saved."""
    context = await _context_with_artifact_service()
    self.manager.cache_blob(
        context, types.Blob(data=b'blob', mime_type=mime_type), 'input'
    )

    events = await self.manager.flush_caches(context)

    assert not events
    context.artifact_service.save_artifact.assert_not_called()

  @pytest.mark.asyncio
  async def test_cache_blob_warns_once_per_kind_of_unsaved_blob(self, caplog):
    """The first unsaved blob of each kind logs a warning; repeats don't."""
    context = await _context_with_artifact_service()
    unsaved = [
        (types.Blob(data=b'chunk', mime_type='video/mp4'), 'output'),
        (types.Blob(data=b'raw'), 'input'),
        (_image(b'model_frame'), 'output'),
    ]

    with caplog.at_level(logging.WARNING, logger='google_adk'):
      for blob, cache_type in unsaved + unsaved:
        self.manager.cache_blob(context, blob, cache_type)
      self.manager.cache_blob(context, _image(b'user_frame'), 'input')

    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    assert len(warnings) == 3
    assert "output live blobs of type 'video/mp4'" in warnings[0]
    assert 'input live blobs of type None' in warnings[1]
    assert "output live blobs of type 'image/jpeg'" in warnings[2]

  @pytest.mark.parametrize('data', [None, b''])
  @pytest.mark.asyncio
  async def test_cache_blob_skips_image_frames_without_data(self, data):
    """An image frame with no bytes is skipped instead of failing the batch."""
    context = await _context_with_artifact_service()
    self.manager.cache_blob(context, _image(data), 'input')

    events = await self.manager.flush_caches(context)

    assert not events
    context.artifact_service.save_artifact.assert_not_called()

  @pytest.mark.asyncio
  async def test_cache_blob_caches_a_copy_of_an_image_frame(self):
    """Later changes to the caller's blob don't change the cached frame."""
    context = await _context_with_artifact_service()
    blob = _image(b'frame_1')
    self.manager.cache_blob(context, blob, 'input')
    blob.data = b'frame_2'

    await self.manager.flush_caches(context)

    assert _saved_frames(context) == [[b'frame_1']]

  @pytest.mark.asyncio
  async def test_cache_blob_rejects_an_unknown_cache_type_for_images(self):
    """An image frame with a cache_type other than input or output raises."""
    context = await _context_with_artifact_service()

    with pytest.raises(ValueError, match="either 'input' or 'output'"):
      self.manager.cache_blob(context, _image(b'frame'), 'sideways')

  @pytest.mark.asyncio
  async def test_visual_cache_drops_the_oldest_frames_over_the_frame_limit(
      self,
  ):
    """Past the frame limit, the oldest frames make room for new ones."""
    manager = RealtimeCacheManager(
        RealtimeCacheConfig(max_visual_cache_frames=2)
    )
    context = await _context_with_artifact_service()
    for data in (b'frame_1', b'frame_2', b'frame_3'):
      manager.cache_blob(context, _image(data), 'input')

    await manager.flush_caches(context)

    (archive,) = _saved_blobs(context)
    frames, _ = unpack_visual_frames(archive.data)
    assert [frame.blob.data for frame in frames] == [b'frame_2', b'frame_3']

  @pytest.mark.asyncio
  async def test_flush_saves_a_single_frame_as_a_zip(self):
    """One cached frame is saved as a one-frame archive, like several are."""
    context = await _context_with_artifact_service()
    self.manager.cache_blob(context, _image(b'frame', 'image/png'), 'input')

    events = await self.manager.flush_caches(context)

    save_kwargs = context.artifact_service.save_artifact.call_args.kwargs
    archive = save_kwargs['artifact'].inline_data
    frames, manifest = unpack_visual_frames(archive.data)
    assert archive.mime_type == 'application/zip'
    assert re.fullmatch(
        r'adk_live_visual_storage_input_visual_\d+\.zip',
        save_kwargs['filename'],
    )
    assert [(frame.blob.data, frame.blob.mime_type) for frame in frames] == [
        (b'frame', 'image/png')
    ]
    assert save_kwargs['custom_metadata'] == {
        key: value for key, value in manifest.items() if key != 'frames'
    }
    file_data = events[0].content.parts[0].file_data
    assert file_data.mime_type == 'application/zip'
    assert file_data.file_uri.endswith(
        f"/_adk_live/{save_kwargs['filename']}#0"
    )

  @pytest.mark.asyncio
  async def test_flush_packs_several_frames_into_one_zip(self):
    """Several frames are saved as one ZIP archive with summary metadata."""
    context = await _context_with_artifact_service()
    self.manager.cache_blob(context, _image(b'jpeg_frame'), 'input')
    self.manager.cache_blob(context, _image(b'png_frame', 'image/png'), 'input')

    events = await self.manager.flush_caches(context)

    save_kwargs = context.artifact_service.save_artifact.call_args.kwargs
    archive = save_kwargs['artifact'].inline_data
    frames, manifest = unpack_visual_frames(archive.data)
    assert archive.mime_type == 'application/zip'
    assert save_kwargs['filename'].endswith('.zip')
    assert [(frame.blob.data, frame.blob.mime_type) for frame in frames] == [
        (b'jpeg_frame', 'image/jpeg'),
        (b'png_frame', 'image/png'),
    ]
    assert save_kwargs['custom_metadata'] == {
        key: value for key, value in manifest.items() if key != 'frames'
    }
    assert events[0].content.parts[0].file_data.mime_type == 'application/zip'

  @pytest.mark.asyncio
  async def test_flush_packs_frames_when_the_clock_steps_back(self):
    """A wall clock that steps back doesn't stop the frames being packed."""
    context = await _context_with_artifact_service()
    timestamps = iter([100.0, 99.0, 101.0])
    platform_time.set_time_provider(lambda: next(timestamps))
    try:
      for data in (b'frame_1', b'frame_2', b'frame_3'):
        self.manager.cache_blob(context, _image(data), 'input')
    finally:
      platform_time.reset_time_provider()

    events = await self.manager.flush_caches(context)

    (archive,) = _saved_blobs(context)
    _, manifest = unpack_visual_frames(archive.data)
    assert len(events) == 1
    assert [entry['offsetMs'] for entry in manifest['frames']] == [0, 0, 1000]

  @pytest.mark.asyncio
  async def test_flush_names_an_archive_after_its_start_timestamp(self):
    """An archive's name carries the same millisecond as its manifest."""
    context = await _context_with_artifact_service()
    timestamps = iter([1760000000.0006, 1760000000.2])
    platform_time.set_time_provider(lambda: next(timestamps))
    try:
      for data in (b'frame_1', b'frame_2'):
        self.manager.cache_blob(context, _image(data), 'input')
    finally:
      platform_time.reset_time_provider()

    await self.manager.flush_caches(context)

    save_kwargs = context.artifact_service.save_artifact.call_args.kwargs
    _, manifest = unpack_visual_frames(save_kwargs['artifact'].inline_data.data)
    assert manifest['startTimestampMs'] == 1760000000001
    assert save_kwargs['filename'] == (
        'adk_live_visual_storage_input_visual_1760000000001.zip'
    )

  @pytest.mark.asyncio
  async def test_flush_keeps_frames_when_not_selected(self):
    """Frames that aren't flushed stay cached for a later flush."""
    context = await _context_with_artifact_service()
    self.manager.cache_blob(context, _image(b'user_frame'), 'input')

    events = await self.manager.flush_caches(context, flush_user_visual=False)
    later_events = await self.manager.flush_caches(context)

    assert not events
    assert [event.author for event in later_events] == ['user']

  @pytest.mark.asyncio
  async def test_flush_orders_audio_before_visual_frames(self):
    """A flush saves user audio, model audio, then the user's frames."""
    context = await _context_with_artifact_service()
    self.manager.cache_blob(context, _image(b'user_frame'), 'input')
    self.manager.cache_blob(
        context,
        types.Blob(data=b'model_audio', mime_type='audio/pcm'),
        'output',
    )
    self.manager.cache_blob(
        context, types.Blob(data=b'user_audio', mime_type='audio/pcm'), 'input'
    )

    await self.manager.flush_caches(context)

    user_audio, model_audio, archive = _saved_blobs(context)
    assert (user_audio.data, model_audio.data) == (
        b'user_audio',
        b'model_audio',
    )
    frames, _ = unpack_visual_frames(archive.data)
    assert [frame.blob.data for frame in frames] == [b'user_frame']

  @pytest.mark.asyncio
  async def test_flush_keeps_frames_cached_while_saving(self):
    """Frames arriving during the save belong to the next flush."""
    context = await _context_with_artifact_service()

    def cache_while_saving(**kwargs):
      self.manager.cache_blob(context, _image(b'mid_flush'), 'input')
      return 0

    context.artifact_service.save_artifact.side_effect = cache_while_saving
    self.manager.cache_blob(context, _image(b'first_turn'), 'input')

    await self.manager.flush_caches(context)
    await self.manager.flush_caches(context)

    assert _saved_frames(context) == [[b'first_turn'], [b'mid_flush']]

  @pytest.mark.asyncio
  async def test_failed_visual_flush_drops_its_frames(self):
    """Frames whose save failed are not saved again with the next turn."""
    context = await _context_with_artifact_service()
    context.artifact_service.save_artifact.side_effect = [
        Exception('Artifact service error'),
        0,
    ]
    self.manager.cache_blob(context, _image(b'first_turn'), 'input')
    events = await self.manager.flush_caches(context)
    self.manager.cache_blob(context, _image(b'second_turn'), 'input')

    await self.manager.flush_caches(context)

    assert not events
    assert _saved_frames(context)[-1] == [b'second_turn']

  @pytest.mark.asyncio
  async def test_flush_without_artifact_service_drops_frames(self):
    """Frames that can't be saved are dropped rather than accumulated."""
    context = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )
    context.artifact_service = None
    self.manager.cache_blob(context, _image(b'first_turn'), 'input')
    events = await self.manager.flush_caches(context)
    context.artifact_service = AsyncMock()

    later_events = await self.manager.flush_caches(context)

    assert not events
    assert not later_events
    context.artifact_service.save_artifact.assert_not_called()

  @pytest.mark.asyncio
  async def test_flushed_frame_events_are_dropped_from_live_history(self):
    """Saved frame archives aren't replayed to the model."""
    context = await _context_with_artifact_service()
    self.manager.cache_blob(context, _image(b'frame_1'), 'input')
    self.manager.cache_blob(context, _image(b'frame_2'), 'input')

    events = await self.manager.flush_caches(context)

    assert len(events) == 1
    assert _filter_media_parts(events[0].content) is None
