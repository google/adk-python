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

"""Tests for the multimodal live cache manager (`CacheManager`)."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock
from unittest.mock import Mock

from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.live._cache_manager import CacheConfig
from google.adk.live._cache_manager import CacheManager
from google.adk.live._cache_manager import MEDIA_ZIP_MIME_TYPE
from google.adk.live._cache_manager import RealtimeCacheEntry
from google.adk.live._media_frames import unpack_media_frames
from google.genai import types
import pydantic
import pytest

from .. import testing_utils


def image_blob(data: bytes = b'frame', mime_type: str = 'image/jpeg'):
  return types.Blob(data=data, mime_type=mime_type)


def read_zip_archive(archive_bytes: bytes):
  """Returns `(payloads, manifest)` read back from a ZIP_STORED media archive."""
  frames, manifest = unpack_media_frames(archive_bytes)
  return [frame.blob.data for frame in frames], manifest


async def make_context():
  return await testing_utils.create_invocation_context(
      testing_utils.create_test_agent()
  )


def attach_artifact_service(invocation_context, revision_id: int = 7):
  service = AsyncMock()
  service.save_artifact.return_value = revision_id
  invocation_context.artifact_service = service
  return service


class TestRealtimeCacheEntry:
  """RealtimeCacheEntry schema validation."""

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


class TestCacheConfig:
  """Audio and media retention are bounded separately."""

  def test_default_values(self):
    config = CacheConfig()
    assert config.max_cache_size_bytes == 10 * 1024 * 1024
    assert config.max_cache_duration_seconds == 300.0
    assert config.auto_flush_threshold == 100
    assert config.max_media_cache_frames == 600
    assert config.max_media_cache_size_bytes == 100 * 1024 * 1024

  def test_custom_values(self):
    config = CacheConfig(
        max_cache_size_bytes=5 * 1024 * 1024,
        max_cache_duration_seconds=120.0,
        auto_flush_threshold=50,
        max_media_cache_frames=10,
        max_media_cache_size_bytes=2048,
    )
    assert config.max_cache_size_bytes == 5 * 1024 * 1024
    assert config.max_cache_duration_seconds == 120.0
    assert config.auto_flush_threshold == 50
    assert config.max_media_cache_frames == 10
    assert config.max_media_cache_size_bytes == 2048


class TestCacheAndFlushAudio:
  """Audio caching, flushing, and statistics on CacheManager."""

  def setup_method(self):
    self.config = CacheConfig()
    self.manager = CacheManager(self.config)

  @pytest.mark.asyncio
  async def test_cache_input_audio(self):
    invocation_context = await make_context()
    audio_blob = types.Blob(data=b'test_audio_data', mime_type='audio/pcm')

    assert invocation_context.input_realtime_cache is None
    self.manager.cache_audio(invocation_context, audio_blob, 'input')

    assert invocation_context.input_realtime_cache is not None
    assert len(invocation_context.input_realtime_cache) == 1
    entry = invocation_context.input_realtime_cache[0]
    assert entry.role == 'user'
    assert entry.data == audio_blob
    assert isinstance(entry.timestamp, float)

  @pytest.mark.asyncio
  async def test_cache_audio_rejects_missing_byte_data(self):
    invocation_context = await make_context()

    with pytest.raises(ValueError, match='must contain byte data'):
      self.manager.cache_audio(
          invocation_context,
          types.Blob(data=None, mime_type='audio/pcm'),
          'input',
      )

    assert invocation_context.input_realtime_cache is None

  @pytest.mark.asyncio
  async def test_cache_output_audio(self):
    invocation_context = await make_context()
    audio_blob = types.Blob(data=b'test_model_audio', mime_type='audio/wav')

    assert invocation_context.output_realtime_cache is None
    self.manager.cache_audio(invocation_context, audio_blob, 'output')

    assert invocation_context.output_realtime_cache is not None
    assert len(invocation_context.output_realtime_cache) == 1
    entry = invocation_context.output_realtime_cache[0]
    assert entry.role == 'model'
    assert entry.data == audio_blob
    assert isinstance(entry.timestamp, float)

  @pytest.mark.asyncio
  async def test_multiple_audio_caching(self):
    invocation_context = await make_context()

    for i in range(3):
      audio_blob = types.Blob(data=f'input_{i}'.encode(), mime_type='audio/pcm')
      self.manager.cache_audio(invocation_context, audio_blob, 'input')

    for i in range(2):
      audio_blob = types.Blob(
          data=f'output_{i}'.encode(), mime_type='audio/wav'
      )
      self.manager.cache_audio(invocation_context, audio_blob, 'output')

    assert len(invocation_context.input_realtime_cache) == 3
    assert len(invocation_context.output_realtime_cache) == 2

  @pytest.mark.asyncio
  async def test_flush_caches_both(self):
    invocation_context = await make_context()
    mock_artifact_service = attach_artifact_service(
        invocation_context, revision_id=123
    )

    input_blob = types.Blob(data=b'input_data', mime_type='audio/pcm')
    output_blob = types.Blob(data=b'output_data', mime_type='audio/wav')
    self.manager.cache_audio(invocation_context, input_blob, 'input')
    self.manager.cache_audio(invocation_context, output_blob, 'output')

    await self.manager.flush_caches(invocation_context)

    assert invocation_context.input_realtime_cache == []
    assert invocation_context.output_realtime_cache == []
    assert mock_artifact_service.save_artifact.call_count == 2

  @pytest.mark.asyncio
  async def test_flush_caches_selective(self):
    invocation_context = await make_context()
    mock_artifact_service = attach_artifact_service(
        invocation_context, revision_id=123
    )

    input_blob = types.Blob(data=b'input_data', mime_type='audio/pcm')
    output_blob = types.Blob(data=b'output_data', mime_type='audio/wav')
    self.manager.cache_audio(invocation_context, input_blob, 'input')
    self.manager.cache_audio(invocation_context, output_blob, 'output')

    await self.manager.flush_caches(
        invocation_context, flush_user_audio=True, flush_model_audio=False
    )

    assert invocation_context.input_realtime_cache == []
    assert len(invocation_context.output_realtime_cache) == 1
    assert mock_artifact_service.save_artifact.call_count == 1

  @pytest.mark.asyncio
  async def test_flush_empty_caches(self):
    invocation_context = await make_context()
    mock_artifact_service = attach_artifact_service(invocation_context)

    await self.manager.flush_caches(invocation_context)

    mock_artifact_service.save_artifact.assert_not_called()

  @pytest.mark.asyncio
  async def test_flush_without_artifact_service(self):
    invocation_context = await make_context()
    invocation_context.artifact_service = None

    input_blob = types.Blob(data=b'input_data', mime_type='audio/pcm')
    self.manager.cache_audio(invocation_context, input_blob, 'input')

    await self.manager.flush_caches(invocation_context)

    assert len(invocation_context.input_realtime_cache) == 1

  @pytest.mark.asyncio
  async def test_flush_artifact_creation(self):
    invocation_context = await make_context()
    mock_artifact_service = attach_artifact_service(
        invocation_context, revision_id=456
    )
    mock_session_service = AsyncMock()
    invocation_context.session_service = mock_session_service

    test_data = b'specific_test_audio_data'
    audio_blob = types.Blob(data=test_data, mime_type='audio/pcm')
    self.manager.cache_audio(invocation_context, audio_blob, 'input')

    await self.manager.flush_caches(invocation_context)

    mock_artifact_service.save_artifact.assert_called_once()
    saved_artifact = mock_artifact_service.save_artifact.call_args.kwargs[
        'artifact'
    ]
    assert saved_artifact.inline_data.data == test_data
    assert saved_artifact.inline_data.mime_type == 'audio/pcm'
    mock_session_service.append_event.assert_not_called()

  @pytest.mark.asyncio
  async def test_flush_audio_normalizes_uppercase_and_parameterized_extension(
      self,
  ):
    invocation_context = await make_context()
    mock_artifact_service = attach_artifact_service(invocation_context)

    audio_blob = types.Blob(data=b'pcm_bytes', mime_type='AUDIO/L16;rate=16000')
    self.manager.cache_audio(invocation_context, audio_blob, 'input')

    await self.manager.flush_caches(invocation_context)

    filename = mock_artifact_service.save_artifact.call_args.kwargs['filename']
    assert filename.endswith('.l16')
    assert ';' not in filename

  def test_legacy_flush_cache_to_services_alias_points_to_audio_flush(self):
    assert (
        self.manager._flush_cache_to_services
        == self.manager._flush_audio_cache_to_services
    )

  def test_get_cache_stats_empty(self):
    invocation_context = Mock()
    invocation_context.input_realtime_cache = None
    invocation_context.output_realtime_cache = None
    invocation_context.input_media_realtime_cache = None
    invocation_context.output_media_realtime_cache = None

    stats = self.manager.get_cache_stats(invocation_context)

    expected = {
        'input_chunks': 0,
        'output_chunks': 0,
        'input_bytes': 0,
        'output_bytes': 0,
        'total_chunks': 0,
        'total_bytes': 0,
        'input_media_frames': 0,
        'output_media_frames': 0,
        'input_media_bytes': 0,
        'output_media_bytes': 0,
        'total_media_frames': 0,
        'total_media_bytes': 0,
    }
    assert stats == expected

  @pytest.mark.asyncio
  async def test_get_cache_stats_with_data(self):
    invocation_context = await make_context()

    input_blob1 = types.Blob(data=b'12345', mime_type='audio/pcm')
    input_blob2 = types.Blob(data=b'1234567890', mime_type='audio/pcm')
    output_blob = types.Blob(data=b'abc', mime_type='audio/wav')

    self.manager.cache_audio(invocation_context, input_blob1, 'input')
    self.manager.cache_audio(invocation_context, input_blob2, 'input')
    self.manager.cache_audio(invocation_context, output_blob, 'output')

    stats = self.manager.get_cache_stats(invocation_context)

    expected = {
        'input_chunks': 2,
        'output_chunks': 1,
        'input_bytes': 15,
        'output_bytes': 3,
        'total_chunks': 3,
        'total_bytes': 18,
        'input_media_frames': 0,
        'output_media_frames': 0,
        'input_media_bytes': 0,
        'output_media_bytes': 0,
        'total_media_frames': 0,
        'total_media_bytes': 0,
    }
    assert stats == expected

  @pytest.mark.asyncio
  async def test_error_handling_in_flush(self):
    invocation_context = await make_context()
    mock_artifact_service = AsyncMock()
    mock_artifact_service.save_artifact.side_effect = Exception(
        'Artifact service error'
    )
    invocation_context.artifact_service = mock_artifact_service

    audio_blob = types.Blob(data=b'test_data', mime_type='audio/pcm')
    self.manager.cache_audio(invocation_context, audio_blob, 'input')

    await self.manager.flush_caches(invocation_context)

    assert len(invocation_context.input_realtime_cache) == 1

  @pytest.mark.asyncio
  async def test_filename_uses_first_chunk_timestamp(self):
    invocation_context = await make_context()
    mock_artifact_service = attach_artifact_service(
        invocation_context, revision_id=789
    )
    invocation_context.session_service = AsyncMock()

    first_timestamp = 1234567890.123
    second_timestamp = 1234567891.456
    invocation_context.input_realtime_cache = [
        RealtimeCacheEntry(
            role='user',
            data=types.Blob(data=b'first_chunk', mime_type='audio/pcm'),
            timestamp=first_timestamp,
        ),
        RealtimeCacheEntry(
            role='user',
            data=types.Blob(data=b'second_chunk', mime_type='audio/pcm'),
            timestamp=second_timestamp,
        ),
    ]

    time.sleep(0.01)
    await self.manager.flush_caches(invocation_context)

    mock_artifact_service.save_artifact.assert_called_once()
    filename = mock_artifact_service.save_artifact.call_args.kwargs['filename']
    expected_timestamp_ms = int(first_timestamp * 1000)
    assert (
        filename
        == f'adk_live_audio_storage_input_audio_{expected_timestamp_ms}.pcm'
    )
    current_timestamp_ms = int(time.time() * 1000)
    assert expected_timestamp_ms != current_timestamp_ms

  @pytest.mark.asyncio
  async def test_flush_event_author_for_user_audio(self):
    invocation_context = await make_context()
    attach_artifact_service(invocation_context, revision_id=123)

    input_blob = types.Blob(data=b'user_audio_data', mime_type='audio/pcm')
    self.manager.cache_audio(invocation_context, input_blob, 'input')

    events = await self.manager.flush_caches(
        invocation_context, flush_user_audio=True, flush_model_audio=False
    )

    assert len(events) == 1
    assert events[0].author == 'user'
    assert events[0].content.role == 'user'

  @pytest.mark.asyncio
  async def test_flush_event_author_for_model_audio(self):
    agent = testing_utils.create_test_agent(name='my_test_agent')
    invocation_context = await testing_utils.create_invocation_context(agent)
    attach_artifact_service(invocation_context, revision_id=123)

    output_blob = types.Blob(data=b'model_audio_data', mime_type='audio/wav')
    self.manager.cache_audio(invocation_context, output_blob, 'output')

    events = await self.manager.flush_caches(
        invocation_context, flush_user_audio=False, flush_model_audio=True
    )

    assert len(events) == 1
    assert events[0].author == 'my_test_agent'
    assert events[0].content.role == 'model'


class TestCacheMedia:
  """Frames go into their own cache, kept apart from audio."""

  def setup_method(self):
    self.manager = CacheManager()

  @pytest.mark.asyncio
  async def test_cache_input_media(self):
    invocation_context = await make_context()
    blob = image_blob(b'jpeg_bytes')

    assert invocation_context.input_media_realtime_cache is None

    self.manager.cache_media(invocation_context, blob, 'input')

    cache = invocation_context.input_media_realtime_cache
    assert len(cache) == 1
    assert cache[0].role == 'user'
    assert cache[0].data == blob
    assert isinstance(cache[0].timestamp, float)

  @pytest.mark.asyncio
  async def test_cache_output_media_is_attributed_to_the_model(self):
    invocation_context = await make_context()

    self.manager.cache_media(invocation_context, image_blob(), 'output')

    cache = invocation_context.output_media_realtime_cache
    assert len(cache) == 1
    assert cache[0].role == 'model'

  @pytest.mark.asyncio
  async def test_media_and_audio_caches_stay_separate(self):
    """Mixing them would splice video frames into the concatenated audio

    file, which is written as one continuous stream of samples.
    """
    invocation_context = await make_context()

    self.manager.cache_audio(
        invocation_context,
        types.Blob(data=b'pcm', mime_type='audio/pcm'),
        'input',
    )
    self.manager.cache_media(invocation_context, image_blob(), 'input')

    assert len(invocation_context.input_realtime_cache) == 1
    assert len(invocation_context.input_media_realtime_cache) == 1

  @pytest.mark.asyncio
  async def test_invalid_cache_type_is_rejected(self):
    invocation_context = await make_context()

    with pytest.raises(ValueError, match="either 'input' or 'output'"):
      self.manager.cache_media(invocation_context, image_blob(), 'sideways')

  @pytest.mark.asyncio
  async def test_blob_without_bytes_is_rejected(self):
    invocation_context = await make_context()

    with pytest.raises(ValueError, match='must contain byte data'):
      self.manager.cache_media(
          invocation_context,
          types.Blob(data=None, mime_type='image/jpeg'),
          'input',
      )


class TestMediaCacheEviction:
  """A live video stream is unbounded; the cache holding it is not."""

  @pytest.mark.asyncio
  async def test_frame_count_limit_drops_the_oldest_frames(self):
    manager = CacheManager(CacheConfig(max_media_cache_frames=3))
    invocation_context = await make_context()

    for index in range(5):
      manager.cache_media(
          invocation_context, image_blob(f'frame{index}'.encode()), 'input'
      )

    cache = invocation_context.input_media_realtime_cache
    assert len(cache) == 3
    assert [entry.data.data for entry in cache] == [
        b'frame2',
        b'frame3',
        b'frame4',
    ]

  @pytest.mark.asyncio
  async def test_byte_limit_drops_the_oldest_frames(self):
    """A handful of high-resolution frames can exceed the byte ceiling well

    before the frame ceiling, so size is capped independently.
    """
    manager = CacheManager(
        CacheConfig(max_media_cache_frames=1000, max_media_cache_size_bytes=25)
    )
    invocation_context = await make_context()

    for index in range(5):
      manager.cache_media(
          invocation_context,
          image_blob(b'x' * 10 + str(index).encode()),
          'input',
      )

    cache = invocation_context.input_media_realtime_cache
    total = sum(len(entry.data.data) for entry in cache)
    assert total <= 25
    # The most recent frame always survives: it is the one just handed over.
    assert cache[-1].data.data.endswith(b'4')

  @pytest.mark.asyncio
  async def test_a_frame_larger_than_the_whole_budget_is_still_kept(self):
    """Dropping it would mean silently storing nothing at all for a stream of

    large frames.
    """
    manager = CacheManager(CacheConfig(max_media_cache_size_bytes=10))
    invocation_context = await make_context()

    manager.cache_media(invocation_context, image_blob(b'x' * 100), 'input')

    assert len(invocation_context.input_media_realtime_cache) == 1

  @pytest.mark.asyncio
  async def test_eviction_does_not_touch_the_other_direction(self):
    manager = CacheManager(CacheConfig(max_media_cache_frames=2))
    invocation_context = await make_context()

    for index in range(4):
      manager.cache_media(
          invocation_context, image_blob(f'in{index}'.encode()), 'input'
      )
    manager.cache_media(invocation_context, image_blob(b'out'), 'output')

    assert len(invocation_context.input_media_realtime_cache) == 2
    assert len(invocation_context.output_media_realtime_cache) == 1


class TestCacheBlobRouting:
  """One entry point decides where a live blob belongs, by MIME type."""

  def setup_method(self):
    self.manager = CacheManager()

  @pytest.mark.asyncio
  @pytest.mark.parametrize(
      'mime_type',
      ['audio/pcm', 'audio/wav', 'AUDIO/PCM', 'audio/l16;rate=16000'],
  )
  async def test_audio_goes_to_the_audio_cache(self, mime_type):
    invocation_context = await make_context()

    self.manager.cache_blob(
        invocation_context, types.Blob(data=b'x', mime_type=mime_type), 'input'
    )

    assert len(invocation_context.input_realtime_cache) == 1
    assert not invocation_context.input_media_realtime_cache

  @pytest.mark.asyncio
  @pytest.mark.parametrize(
      'mime_type',
      ['image/jpeg', 'image/png', 'video/mp4', 'IMAGE/JPEG', 'video/webm'],
  )
  async def test_images_and_video_go_to_the_media_cache(self, mime_type):
    invocation_context = await make_context()

    self.manager.cache_blob(
        invocation_context, types.Blob(data=b'x', mime_type=mime_type), 'input'
    )

    assert len(invocation_context.input_media_realtime_cache) == 1
    assert not invocation_context.input_realtime_cache

  @pytest.mark.asyncio
  @pytest.mark.parametrize('mime_type', ['text/plain', 'application/pdf', ''])
  async def test_an_unsupported_type_is_dropped_with_a_warning(
      self, mime_type, caplog
  ):
    """Guessing a cache for an unknown type would corrupt whichever one it

    guessed; refusing it and saying so is the safe outcome.
    """
    invocation_context = await make_context()

    with caplog.at_level('WARNING'):
      self.manager.cache_blob(
          invocation_context,
          types.Blob(data=b'x', mime_type=mime_type),
          'input',
      )

    assert not invocation_context.input_realtime_cache
    assert not invocation_context.input_media_realtime_cache
    assert 'Unsupported MIME type' in caplog.text

  @pytest.mark.asyncio
  async def test_a_blob_with_no_mime_type_is_dropped_with_a_warning(
      self, caplog
  ):
    invocation_context = await make_context()

    with caplog.at_level('WARNING'):
      self.manager.cache_blob(
          invocation_context, types.Blob(data=b'x', mime_type=None), 'input'
      )

    assert not invocation_context.input_media_realtime_cache
    assert 'Unsupported MIME type' in caplog.text


class TestFlushMedia:
  """Single frames are stored natively; multi-frame sequences are packed into a ZIP."""

  def setup_method(self):
    self.manager = CacheManager()

  @pytest.mark.asyncio
  async def test_single_frame_is_stored_directly_with_native_mime_type(self):
    """When only 1 frame is cached, storing it directly as image/jpeg lets

    adk web render the image immediately without downloading and unzipping.
    """
    invocation_context = await make_context()
    service = attach_artifact_service(invocation_context, revision_id=9)
    self.manager.cache_media(
        invocation_context,
        image_blob(b'single_jpeg', mime_type='IMAGE/JPEG; charset=utf-8'),
        'input',
    )

    events = await self.manager.flush_caches(invocation_context)

    assert service.save_artifact.call_count == 1
    call_kwargs = service.save_artifact.call_args.kwargs
    assert call_kwargs['filename'].startswith(
        'adk_live_media_storage_input_media_'
    )
    assert call_kwargs['filename'].endswith('.jpeg')
    assert call_kwargs['artifact'].inline_data.data == b'single_jpeg'
    assert call_kwargs['artifact'].inline_data.mime_type == 'image/jpeg'
    assert call_kwargs['custom_metadata']['type'] == 'single_media_frame'
    assert call_kwargs['custom_metadata']['frameCount'] == 1
    assert call_kwargs['custom_metadata']['mimeTypes'] == ['image/jpeg']

    assert len(events) == 1
    file_data = events[0].content.parts[0].file_data
    assert file_data.mime_type == 'image/jpeg'
    assert file_data.file_uri.endswith('.jpeg#9')

  @pytest.mark.asyncio
  async def test_flush_saves_one_artifact_for_the_whole_sequence(self):
    """The reason for packing at all: a minute of video is hundreds of

    frames, and writing each as its own artifact is hundreds of round trips.
    """
    invocation_context = await make_context()
    service = attach_artifact_service(invocation_context)
    for index in range(50):
      self.manager.cache_media(
          invocation_context, image_blob(f'frame{index}'.encode()), 'input'
      )

    await self.manager.flush_caches(invocation_context)

    assert service.save_artifact.call_count == 1

  @pytest.mark.asyncio
  async def test_the_saved_artifact_is_an_archive_of_every_frame(self):
    invocation_context = await make_context()
    service = attach_artifact_service(invocation_context)
    payloads = [b'first', b'second', b'third']
    for payload in payloads:
      self.manager.cache_media(invocation_context, image_blob(payload), 'input')

    await self.manager.flush_caches(invocation_context)

    artifact = service.save_artifact.call_args.kwargs['artifact']
    assert artifact.inline_data.mime_type == MEDIA_ZIP_MIME_TYPE
    restored_payloads, manifest = read_zip_archive(artifact.inline_data.data)
    assert restored_payloads == payloads
    assert manifest['frameCount'] == 3

  @pytest.mark.asyncio
  async def test_stored_metadata_omits_the_per_frame_index(self):
    """Artifact metadata is size-capped on several backends, so only the

    fixed-size summary goes there; the per-frame index is in the archive.
    """
    invocation_context = await make_context()
    service = attach_artifact_service(invocation_context)
    for _ in range(20):
      self.manager.cache_media(invocation_context, image_blob(), 'input')

    await self.manager.flush_caches(invocation_context)

    metadata = service.save_artifact.call_args.kwargs['custom_metadata']
    assert 'frames' not in metadata
    assert metadata['frameCount'] == 20
    assert metadata['type'] == 'video_frame_sequence'

  @pytest.mark.asyncio
  async def test_multi_frame_filename_marks_it_as_a_media_archive(self):
    invocation_context = await make_context()
    service = attach_artifact_service(invocation_context)
    self.manager.cache_media(invocation_context, image_blob(b'f1'), 'input')
    self.manager.cache_media(invocation_context, image_blob(b'f2'), 'input')

    await self.manager.flush_caches(invocation_context)

    filename = service.save_artifact.call_args.kwargs['filename']
    assert filename.startswith('adk_live_media_storage_input_media_')
    assert filename.endswith('.zip')

  @pytest.mark.asyncio
  async def test_multi_frame_event_points_at_the_archive_and_declares_it_as_one(
      self,
  ):
    """The event's `file_data` is stored on the session, so its declared type

    has to match what `load_artifact` returns.
    """
    invocation_context = await make_context()
    attach_artifact_service(invocation_context, revision_id=42)
    self.manager.cache_media(invocation_context, image_blob(b'f1'), 'input')
    self.manager.cache_media(invocation_context, image_blob(b'f2'), 'input')

    events = await self.manager.flush_caches(invocation_context)

    assert len(events) == 1
    file_data = events[0].content.parts[0].file_data
    assert file_data.mime_type == MEDIA_ZIP_MIME_TYPE
    assert file_data.file_uri.startswith('artifact://')
    assert file_data.file_uri.endswith('.zip#42')
    assert '_adk_live/' in file_data.file_uri

  @pytest.mark.asyncio
  async def test_a_model_event_is_authored_by_the_agent(self):
    invocation_context = await make_context()
    attach_artifact_service(invocation_context)
    self.manager.cache_media(invocation_context, image_blob(), 'output')

    events = await self.manager.flush_caches(invocation_context)

    assert events[0].author == invocation_context.agent.name
    assert events[0].content.role == 'model'

  @pytest.mark.asyncio
  async def test_a_user_event_is_authored_by_the_user(self):
    invocation_context = await make_context()
    attach_artifact_service(invocation_context)
    self.manager.cache_media(invocation_context, image_blob(), 'input')

    events = await self.manager.flush_caches(invocation_context)

    assert events[0].author == 'user'
    assert events[0].content.role == 'user'

  @pytest.mark.asyncio
  async def test_a_flushed_cache_is_cleared(self):
    invocation_context = await make_context()
    attach_artifact_service(invocation_context)
    self.manager.cache_media(invocation_context, image_blob(), 'input')

    await self.manager.flush_caches(invocation_context)

    assert invocation_context.input_media_realtime_cache == []

  @pytest.mark.asyncio
  async def test_audio_and_media_flush_together(self):
    invocation_context = await make_context()
    service = attach_artifact_service(invocation_context)
    self.manager.cache_audio(
        invocation_context,
        types.Blob(data=b'pcm', mime_type='audio/pcm'),
        'input',
    )
    self.manager.cache_media(invocation_context, image_blob(), 'input')

    events = await self.manager.flush_caches(invocation_context)

    assert service.save_artifact.call_count == 2
    assert len(events) == 2
    assert invocation_context.input_realtime_cache == []
    assert invocation_context.input_media_realtime_cache == []

  @pytest.mark.asyncio
  async def test_media_flushing_can_be_skipped(self):
    """An interrupt flushes the model's completed output while the user is

    still mid-capture, so the flags have to be honoured independently.
    """
    invocation_context = await make_context()
    service = attach_artifact_service(invocation_context)
    self.manager.cache_media(invocation_context, image_blob(), 'input')
    self.manager.cache_media(invocation_context, image_blob(), 'output')

    await self.manager.flush_caches(
        invocation_context,
        flush_user_media=False,
        flush_model_media=True,
    )

    assert service.save_artifact.call_count == 1
    assert len(invocation_context.input_media_realtime_cache) == 1
    assert invocation_context.output_media_realtime_cache == []

  @pytest.mark.asyncio
  async def test_nothing_is_saved_without_an_artifact_service(self):
    invocation_context = await make_context()
    invocation_context.artifact_service = None
    self.manager.cache_media(invocation_context, image_blob(), 'input')

    events = await self.manager.flush_caches(invocation_context)

    assert events == []
    # The cache survives, so the frames are not lost to a misconfiguration.
    assert len(invocation_context.input_media_realtime_cache) == 1

  @pytest.mark.asyncio
  async def test_an_empty_media_cache_saves_nothing(self):
    invocation_context = await make_context()
    service = attach_artifact_service(invocation_context)

    events = await self.manager.flush_caches(invocation_context)

    assert events == []
    service.save_artifact.assert_not_called()

  @pytest.mark.asyncio
  async def test_a_failed_media_flush_keeps_the_frames(self):
    """Clearing a cache whose upload failed would discard the only copy."""
    invocation_context = await make_context()
    service = attach_artifact_service(invocation_context)
    service.save_artifact.side_effect = RuntimeError('backend down')
    self.manager.cache_media(invocation_context, image_blob(), 'input')

    events = await self.manager.flush_caches(invocation_context)

    assert events == []
    assert len(invocation_context.input_media_realtime_cache) == 1

  @pytest.mark.asyncio
  async def test_a_failed_media_flush_does_not_block_the_audio_flush(self):
    """The caches are flushed concurrently; one failing backend call must not

    take the other's data down with it.
    """
    invocation_context = await make_context()
    service = attach_artifact_service(invocation_context)

    async def fail_on_media(**kwargs):
      if kwargs['filename'].startswith('adk_live_media_storage_'):
        raise RuntimeError('media backend down')
      return 5

    service.save_artifact.side_effect = fail_on_media
    self.manager.cache_audio(
        invocation_context,
        types.Blob(data=b'pcm', mime_type='audio/pcm'),
        'input',
    )
    self.manager.cache_media(invocation_context, image_blob(), 'input')

    events = await self.manager.flush_caches(invocation_context)

    assert len(events) == 1
    assert invocation_context.input_realtime_cache == []
    assert len(invocation_context.input_media_realtime_cache) == 1


class TestGetCacheStats:
  """Stats report audio and media separately."""

  def setup_method(self):
    self.manager = CacheManager()

  @pytest.mark.asyncio
  async def test_media_frames_and_bytes_are_counted(self):
    invocation_context = await make_context()
    self.manager.cache_media(invocation_context, image_blob(b'12345'), 'input')
    self.manager.cache_media(
        invocation_context, image_blob(b'1234567890'), 'input'
    )
    self.manager.cache_media(invocation_context, image_blob(b'abc'), 'output')

    stats = self.manager.get_cache_stats(invocation_context)

    assert stats['input_media_frames'] == 2
    assert stats['output_media_frames'] == 1
    assert stats['input_media_bytes'] == 15
    assert stats['output_media_bytes'] == 3
    assert stats['total_media_frames'] == 3
    assert stats['total_media_bytes'] == 18
    # Media must not leak into the audio counters.
    assert stats['total_chunks'] == 0
    assert stats['total_bytes'] == 0

  def test_absent_caches_count_as_zero_even_on_bare_mock(self):
    invocation_context = Mock()
    invocation_context.input_realtime_cache = None
    invocation_context.output_realtime_cache = None

    stats = self.manager.get_cache_stats(invocation_context)

    assert stats['total_media_frames'] == 0
    assert stats['total_media_bytes'] == 0


class TestReadBackThroughTheArtifactService:
  """Everything written must come back out through the ordinary read API."""

  @pytest.mark.asyncio
  async def test_a_flushed_single_frame_is_recovered_with_load_artifact(self):
    manager = CacheManager()
    invocation_context = await make_context()
    service = InMemoryArtifactService()
    invocation_context.artifact_service = service
    manager.cache_media(
        invocation_context, image_blob(b'single_frame_bytes'), 'input'
    )

    events = await manager.flush_caches(invocation_context)

    file_uri = events[0].content.parts[0].file_data.file_uri
    filename = file_uri.rsplit('/', 1)[-1].split('#')[0]
    loaded = await service.load_artifact(
        app_name=invocation_context.app_name,
        user_id=invocation_context.user_id,
        session_id=invocation_context.session.id,
        filename=filename,
    )

    assert loaded is not None
    assert loaded.inline_data.mime_type == 'image/jpeg'
    assert loaded.inline_data.data == b'single_frame_bytes'

  @pytest.mark.asyncio
  async def test_a_flushed_sequence_is_recovered_with_load_artifact(self):
    manager = CacheManager()
    invocation_context = await make_context()
    service = InMemoryArtifactService()
    invocation_context.artifact_service = service
    payloads = [b'frame_one', b'frame_two', b'frame_three']
    for payload in payloads:
      manager.cache_media(invocation_context, image_blob(payload), 'input')

    events = await manager.flush_caches(invocation_context)

    # Read back the way any caller would: from the URI on the event, with no
    # knowledge that the artifact happens to hold frames.
    file_uri = events[0].content.parts[0].file_data.file_uri
    filename = file_uri.rsplit('/', 1)[-1].split('#')[0]
    loaded = await service.load_artifact(
        app_name=invocation_context.app_name,
        user_id=invocation_context.user_id,
        session_id=invocation_context.session.id,
        filename=filename,
    )

    assert loaded is not None
    assert loaded.inline_data.mime_type == MEDIA_ZIP_MIME_TYPE
    frames, manifest = manager.load_media_frames(loaded)
    assert [f.blob.data for f in frames] == payloads
    assert [f['mimeType'] for f in manifest['frames']] == ['image/jpeg'] * 3
    assert manifest['frameCount'] == 3

    read_only_manifest = manager.load_media_manifest(loaded)
    assert read_only_manifest == manifest

    second_frame = manager.load_media_frame(loaded, 1)
    assert second_frame.blob.data == b'frame_two'

    preview_frame = manager.load_preview_frame(loaded)
    assert preview_frame.blob.data == b'frame_one'

  @pytest.mark.asyncio
  async def test_single_frame_artifact_works_with_load_media_helpers(self):
    manager = CacheManager()
    invocation_context = await make_context()
    service = InMemoryArtifactService()
    invocation_context.artifact_service = service
    manager.cache_media(
        invocation_context, image_blob(b'single_preview'), 'input'
    )

    events = await manager.flush_caches(invocation_context)
    file_uri = events[0].content.parts[0].file_data.file_uri
    filename = file_uri.rsplit('/', 1)[-1].split('#')[0]
    loaded = await service.load_artifact(
        app_name=invocation_context.app_name,
        user_id=invocation_context.user_id,
        session_id=invocation_context.session.id,
        filename=filename,
    )

    assert loaded is not None
    frames, manifest = manager.load_media_frames(loaded)
    assert len(frames) == 1
    assert frames[0].blob.data == b'single_preview'
    assert manifest['type'] == 'single_media_frame'
    assert manager.load_media_manifest(loaded)['frameCount'] == 1
    assert manager.load_media_frame(loaded, 0).blob.data == b'single_preview'
    assert manager.load_preview_frame(loaded).blob.data == b'single_preview'
    with pytest.raises(ValueError, match='no frame at index 1'):
      manager.load_media_frame(loaded, 1)

  @pytest.mark.asyncio
  async def test_the_two_directions_are_stored_under_different_keys(self):
    """Input and output are flushed concurrently; colliding filenames would

    make one silently overwrite the other.
    """
    manager = CacheManager()
    invocation_context = await make_context()
    service = InMemoryArtifactService()
    invocation_context.artifact_service = service
    manager.cache_media(invocation_context, image_blob(b'u1'), 'input')
    manager.cache_media(invocation_context, image_blob(b'u2'), 'input')
    manager.cache_media(invocation_context, image_blob(b'm1'), 'output')
    manager.cache_media(invocation_context, image_blob(b'm2'), 'output')

    await manager.flush_caches(invocation_context)

    keys = await service.list_artifact_keys(
        app_name=invocation_context.app_name,
        user_id=invocation_context.user_id,
        session_id=invocation_context.session.id,
    )
    media_keys = [key for key in keys if key.endswith('.zip')]
    assert len(media_keys) == 2
