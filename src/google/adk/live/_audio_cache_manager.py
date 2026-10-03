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

"""Backward compatibility module for the audio-only cache manager names.

The cache manager now lives in ``_cache_manager`` under generalized names.
This module keeps the former names importable; it holds no implementation of
its own.
"""

from __future__ import annotations

from ._cache_manager import AudioCacheConfig
from ._cache_manager import AudioCacheManager
from ._cache_manager import logger
from ._cache_manager import RealtimeCacheEntry

__all__ = [
    'AudioCacheConfig',
    'AudioCacheManager',
    'RealtimeCacheEntry',
    'logger',
]
