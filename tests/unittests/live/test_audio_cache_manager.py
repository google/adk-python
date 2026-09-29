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

"""Backward-compatibility tests for `google.adk.live._audio_cache_manager`."""

from __future__ import annotations

from google.adk.live import _audio_cache_manager as compat_module
from google.adk.live._audio_cache_manager import AudioCacheConfig
from google.adk.live._audio_cache_manager import AudioCacheManager
from google.adk.live._audio_cache_manager import logger as compat_logger
from google.adk.live._audio_cache_manager import RealtimeCacheEntry as CompatRealtimeCacheEntry
from google.adk.live._cache_manager import CacheConfig
from google.adk.live._cache_manager import CacheManager
from google.adk.live._cache_manager import logger as canonical_logger
from google.adk.live._cache_manager import RealtimeCacheEntry


def test_audio_cache_manager_reexports_cache_manager_symbols():
  """The legacy `_audio_cache_manager` module must alias `_cache_manager`."""
  assert AudioCacheManager is CacheManager
  assert AudioCacheConfig is CacheConfig
  assert CompatRealtimeCacheEntry is RealtimeCacheEntry
  assert compat_logger is canonical_logger
  assert set(compat_module.__all__) == {
      'AudioCacheConfig',
      'AudioCacheManager',
      'RealtimeCacheEntry',
      'logger',
  }
