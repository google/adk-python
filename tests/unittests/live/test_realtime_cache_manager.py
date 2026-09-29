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

"""Tests for the live realtime cache manager (`RealtimeCacheManager`)."""

from __future__ import annotations

from google.adk.live._realtime_cache_manager import RealtimeCacheManager
from google.genai import types
import pytest

from .. import testing_utils


@pytest.mark.asyncio
async def test_cache_blob_routes_audio_and_skips_non_audio():
  """Verifies cache_blob routes audio/* blobs and skips non-audio blobs."""
  manager = RealtimeCacheManager()
  ctx = await testing_utils.create_invocation_context(
      testing_utils.create_test_agent()
  )
  in_audio = types.Blob(data=b'user_pcm', mime_type='audio/pcm')
  out_audio = types.Blob(data=b'model_wav', mime_type='audio/wav;rate=24000')
  manager.cache_blob(ctx, in_audio, 'input')
  manager.cache_blob(ctx, out_audio, 'output')
  manager.cache_blob(
      ctx, types.Blob(data=b'jpg', mime_type='image/jpeg'), 'input'
  )
  manager.cache_blob(
      ctx, types.Blob(data=b'mp4', mime_type='video/mp4'), 'output'
  )
  manager.cache_blob(ctx, types.Blob(data=b'raw'), 'input')

  assert [e.data for e in ctx.input_realtime_cache] == [in_audio]
  assert [e.data for e in ctx.output_realtime_cache] == [out_audio]
