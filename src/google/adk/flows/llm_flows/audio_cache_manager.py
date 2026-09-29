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

AudioCacheManager and AudioCacheConfig are no longer public; they live in
``google.adk.live._cache_manager`` and this module only keeps
existing imports working. RealtimeCacheEntry is public as
``google.adk.agents.invocation_context.RealtimeCacheEntry``.
"""

from __future__ import annotations

import warnings

from ...live._cache_manager import AudioCacheConfig
from ...live._cache_manager import AudioCacheManager
from ...live._cache_manager import CacheConfig
from ...live._cache_manager import CacheManager
from ...live._cache_manager import logger
from ...live._cache_manager import RealtimeCacheEntry

warnings.warn(
    'google.adk.flows.llm_flows.audio_cache_manager is deprecated; use'
    ' google.adk.live._cache_manager instead.',
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    'AudioCacheConfig',
    'AudioCacheManager',
    'CacheConfig',
    'CacheManager',
    'RealtimeCacheEntry',
    'logger',
]
