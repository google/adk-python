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

import time
from unittest.mock import AsyncMock
from unittest.mock import Mock

from google.adk.live._cache_manager import CacheConfig
from google.adk.live._cache_manager import CacheManager
from google.adk.live._cache_manager import RealtimeCacheEntry
from google.genai import types
import pydantic
import pytest

from .. import testing_utils


class TestRealtimeCacheEntry:
  """Test RealtimeCacheEntry model validation."""

  def test_unknown_fields_are_rejected(self):
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
    assert entry.data.data == b'x'


class TestCacheConfig:
  """Test the CacheConfig class."""

  def test_default_values(self):
    """Test default configuration values."""
    config = CacheConfig()
    assert config.max_cache_size_bytes == 20 * 1024 * 1024
    assert config.max_cache_duration_seconds == 600.0
    assert config.auto_flush_threshold == 200
    assert config.max_media_cache_frames == 600
    assert config.max_media_cache_size_bytes == 100 * 1024 * 1024

  def test_custom_values(self):
    """Test custom configuration values."""
    config = CacheConfig(
        max_cache_size_bytes=5 * 1024 * 1024,
        max_cache_duration_seconds=120.0,
        auto_flush_threshold=50,
        max_media_cache_frames=100,
        max_media_cache_size_bytes=20 * 1024 * 1024,
    )
    assert config.max_cache_size_bytes == 5 * 1024 * 1024
    assert config.max_cache_duration_seconds == 120.0
    assert config.auto_flush_threshold == 50
    assert config.max_media_cache_frames == 100
    assert config.max_media_cache_size_bytes == 20 * 1024 * 1024


class TestCacheManager:
  """Test the multimodal CacheManager class."""

  def setup_method(self):
    self.config = CacheConfig()
    self.manager = CacheManager(self.config)

  @pytest.mark.asyncio
  async def test_cache_input_audio(self):
    ctx = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )
    audio_blob = types.Blob(data=b'test_audio_data', mime_type='audio/pcm')

    assert ctx.input_realtime_cache is None
    self.manager.cache_audio(ctx, audio_blob, 'input')

    assert ctx.input_realtime_cache is not None
    assert len(ctx.input_realtime_cache) == 1
    assert ctx.input_realtime_cache[0].role == 'user'
    assert ctx.input_realtime_cache[0].data == audio_blob
    assert isinstance(ctx.input_realtime_cache[0].timestamp, float)

  @pytest.mark.asyncio
  async def test_cache_input_media(self):
    ctx = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )
    media_blob = types.Blob(data=b'test_jpeg_data', mime_type='image/jpeg')

    assert ctx.input_media_realtime_cache is None
    self.manager.cache_media(ctx, media_blob, 'input')

    assert ctx.input_media_realtime_cache is not None
    assert len(ctx.input_media_realtime_cache) == 1
    assert ctx.input_media_realtime_cache[0].role == 'user'
    assert ctx.input_media_realtime_cache[0].data == media_blob

  @pytest.mark.asyncio
  async def test_cache_output_audio_and_media(self):
    ctx = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )
    audio_blob = types.Blob(data=b'model_audio', mime_type='audio/wav')
    media_blob = types.Blob(data=b'model_video_frame', mime_type='image/png')

    self.manager.cache_audio(ctx, audio_blob, 'output')
    self.manager.cache_media(ctx, media_blob, 'output')

    assert len(ctx.output_realtime_cache) == 1
    assert ctx.output_realtime_cache[0].role == 'model'
    assert len(ctx.output_media_realtime_cache) == 1
    assert ctx.output_media_realtime_cache[0].role == 'model'

  @pytest.mark.asyncio
  async def test_cache_blob_routing(self):
    ctx = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )
    audio_blob = types.Blob(data=b'audio_pcm', mime_type='audio/pcm')
    image_blob = types.Blob(data=b'image_jpg', mime_type='image/jpeg')
    video_blob = types.Blob(data=b'video_mp4', mime_type='video/mp4')

    self.manager.cache_blob(ctx, audio_blob, 'input')
    self.manager.cache_blob(ctx, image_blob, 'input')
    self.manager.cache_blob(ctx, video_blob, 'input')

    assert len(ctx.input_realtime_cache) == 1
    assert ctx.input_realtime_cache[0].data == audio_blob
    assert len(ctx.input_media_realtime_cache) == 2
    assert ctx.input_media_realtime_cache[0].data == image_blob
    assert ctx.input_media_realtime_cache[1].data == video_blob

  @pytest.mark.asyncio
  async def test_cache_audio_rejects_missing_byte_data(self):
    ctx = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )
    with pytest.raises(ValueError, match='must contain byte data'):
      self.manager.cache_audio(
          ctx, types.Blob(data=None, mime_type='audio/pcm'), 'input'
      )

  @pytest.mark.asyncio
  async def test_cache_media_rejects_missing_byte_data(self):
    ctx = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )
    with pytest.raises(ValueError, match='must contain byte data'):
      self.manager.cache_media(
          ctx, types.Blob(data=None, mime_type='image/jpeg'), 'input'
      )

  @pytest.mark.asyncio
  async def test_cache_invalid_cache_type(self):
    ctx = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )
    blob = types.Blob(data=b'123', mime_type='audio/pcm')
    with pytest.raises(
        ValueError, match="cache_type must be either 'input' or 'output'"
    ):
      self.manager.cache_audio(ctx, blob, 'invalid')

    with pytest.raises(
        ValueError, match="cache_type must be either 'input' or 'output'"
    ):
      self.manager.cache_media(ctx, blob, 'invalid')

  @pytest.mark.asyncio
  async def test_unsupported_mime_type_warning(self, caplog):
    ctx = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )
    text_blob = types.Blob(data=b'text', mime_type='text/plain')
    self.manager.cache_blob(ctx, text_blob, 'input')
    assert ctx.input_realtime_cache is None
    assert ctx.input_media_realtime_cache is None
    assert 'Unsupported MIME type' in caplog.text

  @pytest.mark.asyncio
  async def test_media_cache_frame_count_eviction(self):
    config = CacheConfig(max_media_cache_frames=3)
    manager = CacheManager(config)
    ctx = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )

    for i in range(5):
      blob = types.Blob(data=f'frame_{i}'.encode(), mime_type='image/jpeg')
      manager.cache_media(ctx, blob, 'input')

    assert len(ctx.input_media_realtime_cache) == 3
    # First two frames (0, 1) evicted, retaining 2, 3, 4
    assert ctx.input_media_realtime_cache[0].data.data == b'frame_2'
    assert ctx.input_media_realtime_cache[1].data.data == b'frame_3'
    assert ctx.input_media_realtime_cache[2].data.data == b'frame_4'

  @pytest.mark.asyncio
  async def test_media_cache_size_eviction(self):
    config = CacheConfig(max_media_cache_size_bytes=25)
    manager = CacheManager(config)
    ctx = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )

    # Cache two 10-byte frames (20 bytes total)
    manager.cache_media(
        ctx, types.Blob(data=b'0123456789', mime_type='image/jpeg'), 'input'
    )
    manager.cache_media(
        ctx, types.Blob(data=b'abcdefghij', mime_type='image/jpeg'), 'input'
    )
    assert len(ctx.input_media_realtime_cache) == 2

    # Cache another 10-byte frame (total would be 30 > 25, so oldest must be evicted)
    manager.cache_media(
        ctx, types.Blob(data=b'klmnopqrst', mime_type='image/jpeg'), 'input'
    )
    assert len(ctx.input_media_realtime_cache) == 2
    assert ctx.input_media_realtime_cache[0].data.data == b'abcdefghij'
    assert ctx.input_media_realtime_cache[1].data.data == b'klmnopqrst'

  @pytest.mark.asyncio
  async def test_flush_caches_parallel_audio_and_media(self):
    ctx = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )
    mock_artifact_service = AsyncMock()
    mock_artifact_service.save_artifact.return_value = 1
    mock_artifact_service.save_media_frames.return_value = 2
    ctx.artifact_service = mock_artifact_service

    self.manager.cache_audio(
        ctx, types.Blob(data=b'in_audio', mime_type='audio/pcm'), 'input'
    )
    self.manager.cache_audio(
        ctx, types.Blob(data=b'out_audio', mime_type='audio/pcm'), 'output'
    )
    self.manager.cache_media(
        ctx, types.Blob(data=b'in_media', mime_type='image/jpeg'), 'input'
    )
    self.manager.cache_media(
        ctx, types.Blob(data=b'out_media', mime_type='image/jpeg'), 'output'
    )

    events = await self.manager.flush_caches(ctx)

    assert len(events) == 4
    assert ctx.input_realtime_cache == []
    assert ctx.output_realtime_cache == []
    assert ctx.input_media_realtime_cache == []
    assert ctx.output_media_realtime_cache == []
    assert mock_artifact_service.save_artifact.call_count == 2
    assert mock_artifact_service.save_media_frames.call_count == 2

  @pytest.mark.asyncio
  async def test_flush_caches_audio_only(self):
    ctx = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )
    mock_artifact_service = AsyncMock()
    mock_artifact_service.save_artifact.return_value = 1
    ctx.artifact_service = mock_artifact_service

    self.manager.cache_audio(
        ctx, types.Blob(data=b'audio', mime_type='audio/pcm'), 'input'
    )
    self.manager.cache_media(
        ctx, types.Blob(data=b'media', mime_type='image/jpeg'), 'input'
    )

    events = await self.manager.flush_caches(
        ctx,
        flush_user_audio=True,
        flush_model_audio=False,
        flush_user_media=False,
        flush_model_media=False,
    )

    assert len(events) == 1
    assert ctx.input_realtime_cache == []
    assert len(ctx.input_media_realtime_cache) == 1

  @pytest.mark.asyncio
  async def test_flush_caches_media_only(self):
    ctx = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )
    mock_artifact_service = AsyncMock()
    mock_artifact_service.save_media_frames.return_value = 1
    ctx.artifact_service = mock_artifact_service

    self.manager.cache_audio(
        ctx, types.Blob(data=b'audio', mime_type='audio/pcm'), 'input'
    )
    self.manager.cache_media(
        ctx, types.Blob(data=b'media', mime_type='image/jpeg'), 'input'
    )

    events = await self.manager.flush_caches(
        ctx,
        flush_user_audio=False,
        flush_model_audio=False,
        flush_user_media=True,
        flush_model_media=False,
    )

    assert len(events) == 1
    assert len(ctx.input_realtime_cache) == 1
    assert ctx.input_media_realtime_cache == []

  @pytest.mark.asyncio
  async def test_flush_empty_caches(self):
    ctx = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )
    mock_artifact_service = AsyncMock()
    ctx.artifact_service = mock_artifact_service

    events = await self.manager.flush_caches(ctx)
    assert events == []
    mock_artifact_service.save_artifact.assert_not_called()
    mock_artifact_service.save_media_frames.assert_not_called()

  @pytest.mark.asyncio
  async def test_flush_without_artifact_service(self):
    ctx = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )
    ctx.artifact_service = None
    self.manager.cache_audio(
        ctx, types.Blob(data=b'audio', mime_type='audio/pcm'), 'input'
    )
    self.manager.cache_media(
        ctx, types.Blob(data=b'media', mime_type='image/jpeg'), 'input'
    )

    events = await self.manager.flush_caches(ctx)
    assert events == []
    assert len(ctx.input_realtime_cache) == 1
    assert len(ctx.input_media_realtime_cache) == 1

  @pytest.mark.asyncio
  async def test_error_handling_in_flush_partial_success(self):
    ctx = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )
    mock_artifact_service = AsyncMock()
    mock_artifact_service.save_artifact.side_effect = RuntimeError('Disk full')
    mock_artifact_service.save_media_frames.return_value = 1
    ctx.artifact_service = mock_artifact_service

    self.manager.cache_audio(
        ctx, types.Blob(data=b'audio', mime_type='audio/pcm'), 'input'
    )
    self.manager.cache_media(
        ctx, types.Blob(data=b'media', mime_type='image/jpeg'), 'input'
    )

    events = await self.manager.flush_caches(ctx)

    # Audio flush failed -> retained in cache; media flush succeeded -> cleared
    assert len(events) == 1
    assert len(ctx.input_realtime_cache) == 1
    assert ctx.input_media_realtime_cache == []

  @pytest.mark.asyncio
  async def test_collection_naming_and_frames_for_media(self):
    ctx = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )
    mock_artifact_service = AsyncMock()
    mock_artifact_service.save_media_frames.return_value = 10
    ctx.artifact_service = mock_artifact_service

    t0 = 1700000000.123
    t1 = 1700000001.456
    entry0 = RealtimeCacheEntry(
        role='user',
        data=types.Blob(data=b'f0', mime_type='image/jpeg'),
        timestamp=t0,
    )
    entry1 = RealtimeCacheEntry(
        role='user',
        data=types.Blob(data=b'f1', mime_type='image/jpeg'),
        timestamp=t1,
    )
    ctx.input_media_realtime_cache = [entry0, entry1]

    events = await self.manager.flush_caches(
        ctx, flush_user_audio=False, flush_model_audio=False
    )

    assert len(events) == 1
    mock_artifact_service.save_media_frames.assert_called_once()
    call_kwargs = mock_artifact_service.save_media_frames.call_args.kwargs
    expected_start_ms = int(t0 * 1000)
    assert (
        call_kwargs['collection_name']
        == f'adk_live_media_storage_input_media_{expected_start_ms}'
    )
    assert len(call_kwargs['frames']) == 2
    assert call_kwargs['frames'][0] == (entry0.data, t0)
    assert call_kwargs['frames'][1] == (entry1.data, t1)

  @pytest.mark.asyncio
  async def test_audio_filename_timestamp(self):
    ctx = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )
    mock_artifact_service = AsyncMock()
    mock_artifact_service.save_artifact.return_value = 1
    ctx.artifact_service = mock_artifact_service

    t0 = 1700000000.555
    entry = RealtimeCacheEntry(
        role='user',
        data=types.Blob(data=b'chunk', mime_type='audio/pcm'),
        timestamp=t0,
    )
    ctx.input_realtime_cache = [entry]

    await self.manager.flush_caches(
        ctx, flush_user_media=False, flush_model_media=False
    )

    mock_artifact_service.save_artifact.assert_called_once()
    call_kwargs = mock_artifact_service.save_artifact.call_args.kwargs
    expected_ms = int(t0 * 1000)
    assert (
        call_kwargs['filename']
        == f'adk_live_audio_storage_input_audio_{expected_ms}.pcm'
    )

  @pytest.mark.asyncio
  async def test_flush_event_author_for_user_and_model(self):
    agent = testing_utils.create_test_agent(name='my_bot')
    ctx = await testing_utils.create_invocation_context(agent)
    mock_artifact_service = AsyncMock()
    mock_artifact_service.save_artifact.return_value = 1
    mock_artifact_service.save_media_frames.return_value = 1
    ctx.artifact_service = mock_artifact_service

    self.manager.cache_audio(
        ctx, types.Blob(data=b'user_audio', mime_type='audio/pcm'), 'input'
    )
    self.manager.cache_media(
        ctx, types.Blob(data=b'model_frame', mime_type='image/jpeg'), 'output'
    )

    events = await self.manager.flush_caches(ctx)
    assert len(events) == 2
    user_event = next(e for e in events if e.author == 'user')
    model_event = next(e for e in events if e.author == 'my_bot')

    assert user_event.content.role == 'user'
    assert model_event.content.role == 'model'

  @pytest.mark.asyncio
  async def test_get_cache_stats(self):
    ctx = await testing_utils.create_invocation_context(
        testing_utils.create_test_agent()
    )

    # Empty stats
    stats = self.manager.get_cache_stats(ctx)
    assert stats['total_chunks'] == 0
    assert stats['total_bytes'] == 0
    assert stats['total_media_frames'] == 0

    # Populate caches
    self.manager.cache_audio(
        ctx, types.Blob(data=b'12345', mime_type='audio/pcm'), 'input'
    )
    self.manager.cache_audio(
        ctx, types.Blob(data=b'123', mime_type='audio/pcm'), 'output'
    )
    self.manager.cache_media(
        ctx, types.Blob(data=b'abcde', mime_type='image/jpeg'), 'input'
    )

    stats = self.manager.get_cache_stats(ctx)
    assert stats['input_chunks'] == 1
    assert stats['output_chunks'] == 1
    assert stats['input_bytes'] == 5
    assert stats['output_bytes'] == 3
    assert stats['total_chunks'] == 2
    assert stats['total_bytes'] == 8
    assert stats['input_media_frames'] == 1
    assert stats['output_media_frames'] == 0
    assert stats['input_media_bytes'] == 5
    assert stats['output_media_bytes'] == 0
    assert stats['total_media_frames'] == 1
    assert stats['total_media_bytes'] == 5
