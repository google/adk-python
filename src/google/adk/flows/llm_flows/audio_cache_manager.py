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

"""Backward compatibility module for AudioCacheManager.

AudioCacheManager and AudioCacheConfig have been moved to
``google.adk.live.audio_cache_manager``.
"""

from __future__ import annotations

import logging

from ...live._audio_cache_manager import AudioCacheConfig as AudioCacheConfig
from ...live._audio_cache_manager import AudioCacheManager as AudioCacheManager
from ...live._audio_cache_manager import RealtimeCacheEntry as RealtimeCacheEntry

logger = logging.getLogger('google_adk.' + __name__)
