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

"""Tests for the generalized live cache manager (`CacheManager`)."""

from __future__ import annotations

from unittest import mock

from google.adk.live._cache_manager import AudioCacheConfig
from google.adk.live._cache_manager import AudioCacheManager
from google.adk.live._cache_manager import CacheConfig
from google.adk.live._cache_manager import CacheManager
from google.genai import types
import pytest

from .. import testing_utils


def test_cache_config_defaults_and_custom_values():
  """Verifies CacheConfig defaults, custom parameters, and AudioCacheConfig alias."""
  default_config = CacheConfig()
  assert default_config.max_cache_size_bytes == 10 * 1024 * 1024
  assert default_config.max_cache_duration_seconds == 300.0
  assert default_config.auto_flush_threshold == 100
  custom_config = CacheConfig(5 * 1024 * 1024, 120.0, 50)
  assert custom_config.max_cache_size_bytes == 5 * 1024 * 1024
  assert custom_config.max_cache_duration_seconds == 120.0
  assert custom_config.auto_flush_threshold == 50
  assert AudioCacheConfig is CacheConfig
  assert AudioCacheManager is CacheManager


@pytest.mark.asyncio
async def test_invocation_context_initializes_media_cache_fields_to_none():
  """Verifies input_media_realtime_cache and output_media_realtime_cache default to None."""
  ctx = await testing_utils.create_invocation_context(
      testing_utils.create_test_agent()
  )
  assert ctx.input_media_realtime_cache is None
  assert ctx.output_media_realtime_cache is None


@pytest.mark.asyncio
async def test_cache_blob_routes_audio_and_skips_non_audio():
  """Verifies cache_blob routes audio/* blobs and skips non-audio blobs."""
  manager = CacheManager()
  ctx = await testing_utils.create_invocation_context(
      testing_utils.create_test_agent()
  )
  in_audio = types.Blob(data=b'user_pcm', mime_type='audio/pcm')
  out_audio = types.Blob(data=b'model_wav', mime_type='AUDIO/WAV;rate=24000')
  manager.cache_blob(ctx, in_audio, 'input')
  manager.cache_blob(ctx, out_audio, 'output')
  manager.cache_blob(
      ctx, types.Blob(data=b'jpg', mime_type='image/jpeg'), 'input'
  )
  manager.cache_blob(
      ctx, types.Blob(data=b'mp4', mime_type='video/mp4'), 'output'
  )

  assert [e.data for e in ctx.input_realtime_cache] == [in_audio]
  assert [e.data for e in ctx.output_realtime_cache] == [out_audio]
  assert ctx.input_media_realtime_cache is None
  assert ctx.output_media_realtime_cache is None


@pytest.mark.asyncio
async def test_flush_normalizes_parameterized_audio_mime_extension():
  """Verifies parameterized MIME types like audio/pcm;rate=16000 produce a .pcm extension."""
  manager = CacheManager()
  ctx = await testing_utils.create_invocation_context(
      testing_utils.create_test_agent()
  )
  ctx.artifact_service = mock.AsyncMock(
      save_artifact=mock.AsyncMock(return_value=7)
  )
  manager.cache_blob(
      ctx, types.Blob(data=b'pcm', mime_type='audio/pcm;rate=16000'), 'input'
  )
  events = await manager.flush_caches(ctx)

  assert len(events) == 1
  saved_name = ctx.artifact_service.save_artifact.call_args.kwargs['filename']
  assert saved_name.endswith('.pcm') and ';' not in saved_name


@pytest.mark.asyncio
async def test_failed_input_flush_does_not_block_output_flush():
  """Verifies concurrent flushing keeps a failed cache while clearing the succeeded cache."""
  manager = CacheManager()
  ctx = await testing_utils.create_invocation_context(
      testing_utils.create_test_agent()
  )

  async def fail_on_input(**kwargs):
    if 'input_audio' in kwargs['filename']:
      raise RuntimeError('input storage failure')
    return 9

  ctx.artifact_service = mock.AsyncMock(
      save_artifact=mock.AsyncMock(side_effect=fail_on_input)
  )
  manager.cache_audio(
      ctx, types.Blob(data=b'in', mime_type='audio/pcm'), 'input'
  )
  manager.cache_audio(
      ctx, types.Blob(data=b'out', mime_type='audio/pcm'), 'output'
  )
  events = await manager.flush_caches(ctx)

  assert len(events) == 1 and events[0].content.role == 'model'
  assert len(ctx.input_realtime_cache) == 1
  assert not ctx.output_realtime_cache


def test_get_cache_stats_handles_bare_mock_invocation_context():
  """Verifies get_cache_stats returns zero counts on a bare Mock context."""
  stats = CacheManager().get_cache_stats(
      mock.Mock(input_realtime_cache=None, output_realtime_cache=None)
  )
  assert stats['total_chunks'] == 0 and stats['total_bytes'] == 0
