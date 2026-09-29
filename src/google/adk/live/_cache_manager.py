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

"""Multimodal cache manager for live streaming flows."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from google.adk.platform import time as platform_time
from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict

from ..events.event import Event

if TYPE_CHECKING:
  from ..agents.invocation_context import InvocationContext  # pylint: disable=g-import-not-at-top

logger = logging.getLogger('google_adk.' + __name__)


class RealtimeCacheEntry(BaseModel):
  """Store raw realtime chunk or frame data for caching before flushing."""

  model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')
  role: str
  data: types.Blob
  timestamp: float


def _require_blob_data(blob: types.Blob) -> bytes:
  data = blob.data
  if not isinstance(data, bytes):
    raise ValueError('Blobs must contain byte data.')
  return data


_require_audio_data = _require_blob_data


def _require_agent_name(invocation_context: InvocationContext) -> str:
  agent = invocation_context.agent
  if agent is None:
    raise TypeError('Live streaming requires an agent in InvocationContext.')
  return agent.name


def _normalize_mime_type(
    mime_type: str | None, default: str = 'application/octet-stream'
) -> str:
  """Returns the lowercase base MIME type with parameters stripped."""
  base = (mime_type or '').split(';', 1)[0].strip().lower()
  return base or default


def _get_cache_list(
    invocation_context: InvocationContext, attr_name: str
) -> list[RealtimeCacheEntry]:
  """Returns the cache list on ``invocation_context`` if initialized, else ``[]``."""
  cache = getattr(invocation_context, attr_name, None)
  return cache if isinstance(cache, list) else []


class CacheConfig:
  """Configuration for multimodal streaming cache behavior."""

  def __init__(
      self,
      max_cache_size_bytes: int = 10 * 1024 * 1024,
      max_cache_duration_seconds: float = 300.0,
      auto_flush_threshold: int = 100,
  ) -> None:
    self.max_cache_size_bytes = max_cache_size_bytes
    self.max_cache_duration_seconds = max_cache_duration_seconds
    self.auto_flush_threshold = auto_flush_threshold


class CacheManager:
  """Manages multimodal caching and flushing for live streaming flows."""

  def __init__(self, config: CacheConfig | None = None) -> None:
    self.config = config if config is not None else CacheConfig()

  def _resolve_cache_target(
      self, invocation_context: InvocationContext, cache_type: str
  ) -> tuple[list[RealtimeCacheEntry], str]:
    """Validates ``cache_type`` and returns ``(cache_list, role)``."""
    if cache_type == 'input':
      attr, role = 'input_realtime_cache', 'user'
    elif cache_type == 'output':
      attr, role = 'output_realtime_cache', 'model'
    else:
      raise ValueError("cache_type must be either 'input' or 'output'")

    cache = getattr(invocation_context, attr, None)
    if not isinstance(cache, list):
      cache = []
      setattr(invocation_context, attr, cache)
    return cache, role

  def cache_audio(
      self,
      invocation_context: InvocationContext,
      audio_blob: types.Blob,
      cache_type: str,
  ) -> None:
    """Cache incoming user or outgoing model audio data."""
    audio_data = _require_blob_data(audio_blob)
    cache, role = self._resolve_cache_target(invocation_context, cache_type)
    cache.append(
        RealtimeCacheEntry(
            role=role, data=audio_blob, timestamp=platform_time.get_time()
        )
    )
    logger.debug(
        'Cached %s audio chunk: %d bytes, cache size: %d',
        cache_type,
        len(audio_data),
        len(cache),
    )

  def cache_blob(
      self,
      invocation_context: InvocationContext,
      blob: types.Blob,
      cache_type: str,
  ) -> None:
    """Routes an incoming or outgoing blob to the appropriate cache by MIME type."""
    mime_type = _normalize_mime_type(blob.mime_type, default='')
    if mime_type.startswith('audio/'):
      self.cache_audio(invocation_context, blob, cache_type)
    else:
      logger.debug(
          'Skipping non-audio blob in cache_blob: %s (cache_type=%s)',
          mime_type,
          cache_type,
      )

  async def flush_caches(
      self,
      invocation_context: InvocationContext,
      flush_user_audio: bool = True,
      flush_model_audio: bool = True,
  ) -> list[Event]:
    """Flush caches concurrently to artifact services."""
    if not invocation_context.artifact_service:
      logger.debug('Skipping cache flush: no artifact service or empty cache')
      return []

    tasks = []
    targets: list[str] = []

    input_audio = _get_cache_list(invocation_context, 'input_realtime_cache')
    if flush_user_audio and input_audio:
      tasks.append(
          self._flush_audio_cache_to_services(
              invocation_context, input_audio, 'input_audio'
          )
      )
      targets.append('input_realtime_cache')

    output_audio = _get_cache_list(invocation_context, 'output_realtime_cache')
    if flush_model_audio and output_audio:
      tasks.append(
          self._flush_audio_cache_to_services(
              invocation_context, output_audio, 'output_audio'
          )
      )
      targets.append('output_realtime_cache')

    if not tasks:
      return []

    results = await asyncio.gather(*tasks, return_exceptions=True)
    flushed_events: list[Event] = []
    for attr_name, result in zip(targets, results):
      if isinstance(result, Exception):
        logger.error('Failed to flush %s: %s', attr_name, result)
      elif isinstance(result, Event):
        flushed_events.append(result)
        setattr(invocation_context, attr_name, [])
    return flushed_events

  async def _save_artifact_and_build_event(
      self,
      invocation_context: InvocationContext,
      *,
      filename: str,
      data: bytes,
      mime_type: str,
      role: str,
      timestamp: float,
  ) -> Event:
    """Saves an artifact and returns the corresponding ``_adk_live`` Event."""
    assert invocation_context.artifact_service is not None
    artifact = types.Part(
        inline_data=types.Blob(data=data, mime_type=mime_type)
    )
    revision_id = await invocation_context.artifact_service.save_artifact(
        app_name=invocation_context.app_name,
        user_id=invocation_context.user_id,
        session_id=invocation_context.session.id,
        filename=filename,
        artifact=artifact,
    )
    artifact_ref = (
        f'artifact://{invocation_context.app_name}/'
        f'{invocation_context.user_id}/{invocation_context.session.id}/'
        f'_adk_live/{filename}#{revision_id}'
    )
    author = (
        _require_agent_name(invocation_context) if role == 'model' else role
    )
    return Event(
        id=Event.new_id(),
        invocation_id=invocation_context.invocation_id,
        author=author,
        content=types.Content(
            role=role,
            parts=[
                types.Part(
                    file_data=types.FileData(
                        file_uri=artifact_ref, mime_type=mime_type
                    )
                )
            ],
        ),
        timestamp=timestamp,
    )

  async def _flush_audio_cache_to_services(
      self,
      invocation_context: InvocationContext,
      audio_cache: list[RealtimeCacheEntry],
      cache_type: str,
  ) -> Event | None:
    """Flush a list of audio cache entries to artifact services."""
    if not invocation_context.artifact_service or not audio_cache:
      logger.debug(
          'Skipping audio cache flush: no artifact service or empty cache'
      )
      return None

    try:
      mime_type = audio_cache[0].data.mime_type or 'audio/pcm'
      combined_audio_data = b''.join(
          entry.data.data or b'' for entry in audio_cache
      )
      timestamp = int(audio_cache[0].timestamp * 1000)
      raw_mime = _normalize_mime_type(mime_type, default='audio/pcm')
      ext = raw_mime.split('/')[-1] if '/' in raw_mime else 'pcm'
      filename = f'adk_live_audio_storage_{cache_type}_{timestamp}.{ext}'
      return await self._save_artifact_and_build_event(
          invocation_context,
          filename=filename,
          data=combined_audio_data,
          mime_type=mime_type,
          role=audio_cache[0].role,
          timestamp=audio_cache[0].timestamp,
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.error('Failed to flush %s cache: %s', cache_type, e)
      return None

  _flush_cache_to_services = _flush_audio_cache_to_services

  def get_cache_stats(
      self, invocation_context: InvocationContext
  ) -> dict[str, int]:
    """Get statistics about current cache state."""
    input_audio = _get_cache_list(invocation_context, 'input_realtime_cache')
    output_audio = _get_cache_list(invocation_context, 'output_realtime_cache')
    input_count = len(input_audio)
    output_count = len(output_audio)
    input_bytes = sum(
        len(_require_blob_data(entry.data)) for entry in input_audio
    )
    output_bytes = sum(
        len(_require_blob_data(entry.data)) for entry in output_audio
    )
    return {
        'input_chunks': input_count,
        'output_chunks': output_count,
        'input_bytes': input_bytes,
        'output_bytes': output_bytes,
        'total_chunks': input_count + output_count,
        'total_bytes': input_bytes + output_bytes,
    }


AudioCacheConfig = CacheConfig
AudioCacheManager = CacheManager
