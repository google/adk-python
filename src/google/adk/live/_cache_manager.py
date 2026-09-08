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

import asyncio
import logging
from typing import TYPE_CHECKING

from google.adk.platform import time as platform_time
from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict

from ..events.event import Event

if TYPE_CHECKING:
  from ..agents.invocation_context import InvocationContext

logger = logging.getLogger('google_adk.' + __name__)


class RealtimeCacheEntry(BaseModel):
  """Store raw realtime chunk/frame data for caching before flushing."""

  model_config = ConfigDict(
      arbitrary_types_allowed=True,
      extra='forbid',
  )
  """The pydantic model config."""

  role: str
  """The role that created this data, typically "user" or "model"."""

  data: types.Blob
  """The realtime data chunk or media frame."""

  timestamp: float
  """Timestamp when the data chunk was received."""


def _require_blob_data(blob: types.Blob) -> bytes:
  data = blob.data
  if not isinstance(data, bytes):
    raise ValueError('Blobs must contain byte data.')
  return data


# Deliberately duplicated from `flows.llm_flows._invocation_utils`
# rather than imported: `live` sits below `flows.llm_flows` in the
# layering, and importing upward would put a cycle back in.
def _require_agent_name(invocation_context: InvocationContext) -> str:
  agent = invocation_context.agent
  if agent is None:
    raise TypeError('Live streaming requires an agent in InvocationContext.')
  return agent.name


class CacheConfig:
  """Configuration for multimodal streaming cache behavior."""

  def __init__(
      self,
      max_cache_size_bytes: int = 20 * 1024 * 1024,  # 20MB
      max_cache_duration_seconds: float = 600.0,  # 10 minutes
      auto_flush_threshold: int = 200,  # Number of chunks
      max_media_cache_frames: int = 600,  # Max media frames per cache
      max_media_cache_size_bytes: int = 100 * 1024 * 1024,  # 100MB
  ) -> None:
    """Initialize cache configuration.

    Args:
      max_cache_size_bytes: Maximum audio cache size in bytes before auto-flush.
      max_cache_duration_seconds: Maximum duration to keep data in cache.
      auto_flush_threshold: Number of chunks that triggers auto-flush.
      max_media_cache_frames: Maximum frames retained in memory per media cache.
      max_media_cache_size_bytes: Maximum bytes retained in memory per media
        cache.
    """
    self.max_cache_size_bytes = max_cache_size_bytes
    self.max_cache_duration_seconds = max_cache_duration_seconds
    self.auto_flush_threshold = auto_flush_threshold
    self.max_media_cache_frames = max_media_cache_frames
    self.max_media_cache_size_bytes = max_media_cache_size_bytes


class CacheManager:
  """Manages multimodal caching and flushing for live streaming flows."""

  def __init__(self, config: CacheConfig | None = None) -> None:
    """Initialize the cache manager.

    Args:
      config: Configuration for caching behavior.
    """
    self.config = config or CacheConfig()

  def cache_audio(
      self,
      invocation_context: InvocationContext,
      audio_blob: types.Blob,
      cache_type: str,
  ) -> None:
    """Cache incoming user or outgoing model audio data.

    Args:
      invocation_context: The current invocation context.
      audio_blob: The audio data to cache.
      cache_type: Type of audio to cache, either 'input' or 'output'.

    Raises:
      ValueError: If cache_type is not 'input' or 'output'.
    """
    audio_data = _require_blob_data(audio_blob)
    if cache_type == 'input':
      if invocation_context.input_realtime_cache is None:
        invocation_context.input_realtime_cache = []
      cache = invocation_context.input_realtime_cache
      role = 'user'
    elif cache_type == 'output':
      if invocation_context.output_realtime_cache is None:
        invocation_context.output_realtime_cache = []
      cache = invocation_context.output_realtime_cache
      role = 'model'
    else:
      raise ValueError("cache_type must be either 'input' or 'output'")

    audio_entry = RealtimeCacheEntry(
        role=role, data=audio_blob, timestamp=platform_time.get_time()
    )
    cache.append(audio_entry)

    logger.debug(
        'Cached %s audio chunk: %d bytes, cache size: %d',
        cache_type,
        len(audio_data),
        len(cache),
    )

  def cache_media(
      self,
      invocation_context: InvocationContext,
      media_blob: types.Blob,
      cache_type: str,
  ) -> None:
    """Cache incoming user or outgoing model media (video/image) frame.

    Applies FIFO eviction if frame count or memory size thresholds are reached.

    Args:
      invocation_context: The current invocation context.
      media_blob: The media frame blob to cache.
      cache_type: Type of media to cache, either 'input' or 'output'.

    Raises:
      ValueError: If cache_type is not 'input' or 'output'.
    """
    media_data = _require_blob_data(media_blob)
    if cache_type == 'input':
      if invocation_context.input_media_realtime_cache is None:
        invocation_context.input_media_realtime_cache = []
      cache = invocation_context.input_media_realtime_cache
      role = 'user'
    elif cache_type == 'output':
      if invocation_context.output_media_realtime_cache is None:
        invocation_context.output_media_realtime_cache = []
      cache = invocation_context.output_media_realtime_cache
      role = 'model'
    else:
      raise ValueError("cache_type must be either 'input' or 'output'")

    # FIFO eviction based on frame count
    while cache and len(cache) >= self.config.max_media_cache_frames:
      cache.pop(0)

    # FIFO eviction based on total cached bytes
    new_bytes = len(media_data)
    while cache and (
        sum(len(entry.data.data or b'') for entry in cache) + new_bytes
        > self.config.max_media_cache_size_bytes
    ):
      cache.pop(0)

    media_entry = RealtimeCacheEntry(
        role=role, data=media_blob, timestamp=platform_time.get_time()
    )
    cache.append(media_entry)

    logger.debug(
        'Cached %s media frame: %d bytes, cache size: %d frames',
        cache_type,
        len(media_data),
        len(cache),
    )

  def cache_blob(
      self,
      invocation_context: InvocationContext,
      blob: types.Blob,
      cache_type: str,
  ) -> None:
    """Routes an incoming or outgoing blob to the appropriate cache by MIME type.

    Args:
      invocation_context: The current invocation context.
      blob: The blob containing data and mime_type.
      cache_type: 'input' or 'output'.
    """
    mime_type = (blob.mime_type or '').lower()
    if mime_type.startswith('audio/'):
      self.cache_audio(invocation_context, blob, cache_type)
    elif mime_type.startswith(('image/', 'video/')):
      self.cache_media(invocation_context, blob, cache_type)
    else:
      logger.warning(
          'Unsupported MIME type for caching: %s (cache_type=%s)',
          mime_type,
          cache_type,
      )

  async def flush_caches(
      self,
      invocation_context: InvocationContext,
      flush_user_audio: bool = True,
      flush_model_audio: bool = True,
      flush_user_media: bool = True,
      flush_model_media: bool = True,
  ) -> list[Event]:
    """Flush audio and media caches concurrently to artifact services.

    Args:
      invocation_context: The invocation context containing caches.
      flush_user_audio: Whether to flush the input audio cache.
      flush_model_audio: Whether to flush the output audio cache.
      flush_user_media: Whether to flush the input media cache.
      flush_model_media: Whether to flush the output media cache.

    Returns:
      A list of Event objects created from the successfully flushed caches.
    """
    if not invocation_context.artifact_service:
      logger.debug('Skipping cache flush: no artifact service or empty cache')
      return []

    tasks = []
    targets: list[str] = []

    if flush_user_audio and invocation_context.input_realtime_cache:
      tasks.append(
          self._flush_audio_cache_to_services(
              invocation_context,
              invocation_context.input_realtime_cache,
              'input_audio',
          )
      )
      targets.append('input_audio')

    if flush_model_audio and invocation_context.output_realtime_cache:
      tasks.append(
          self._flush_audio_cache_to_services(
              invocation_context,
              invocation_context.output_realtime_cache,
              'output_audio',
          )
      )
      targets.append('output_audio')

    if flush_user_media and invocation_context.input_media_realtime_cache:
      tasks.append(
          self._flush_media_cache_to_services(
              invocation_context,
              invocation_context.input_media_realtime_cache,
              'input_media',
          )
      )
      targets.append('input_media')

    if flush_model_media and invocation_context.output_media_realtime_cache:
      tasks.append(
          self._flush_media_cache_to_services(
              invocation_context,
              invocation_context.output_media_realtime_cache,
              'output_media',
          )
      )
      targets.append('output_media')

    if not tasks:
      return []

    results = await asyncio.gather(*tasks, return_exceptions=True)
    flushed_events: list[Event] = []

    for target, result in zip(targets, results):
      if isinstance(result, Exception):
        logger.error('Failed to flush %s cache: %s', target, result)
        continue
      if result is not None and isinstance(result, Event):
        flushed_events.append(result)
        if target == 'input_audio':
          invocation_context.input_realtime_cache = []
        elif target == 'output_audio':
          invocation_context.output_realtime_cache = []
        elif target == 'input_media':
          invocation_context.input_media_realtime_cache = []
        elif target == 'output_media':
          invocation_context.output_media_realtime_cache = []

    return flushed_events

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
      raw_mime = mime_type.split(';')[0].strip()
      ext = raw_mime.split('/')[-1] if '/' in raw_mime else 'pcm'
      filename = f'adk_live_audio_storage_{cache_type}_{timestamp}.{ext}'

      combined_audio_part = types.Part(
          inline_data=types.Blob(data=combined_audio_data, mime_type=mime_type)
      )

      revision_id = await invocation_context.artifact_service.save_artifact(
          app_name=invocation_context.app_name,
          user_id=invocation_context.user_id,
          session_id=invocation_context.session.id,
          filename=filename,
          artifact=combined_audio_part,
      )

      artifact_ref = f'artifact://{invocation_context.app_name}/{invocation_context.user_id}/{invocation_context.session.id}/_adk_live/{filename}#{revision_id}'

      author = (
          _require_agent_name(invocation_context)
          if audio_cache[0].role == 'model'
          else audio_cache[0].role
      )
      audio_event = Event(
          id=Event.new_id(),
          invocation_id=invocation_context.invocation_id,
          author=author,
          content=types.Content(
              role=audio_cache[0].role,
              parts=[
                  types.Part(
                      file_data=types.FileData(
                          file_uri=artifact_ref, mime_type=mime_type
                      )
                  )
              ],
          ),
          timestamp=audio_cache[0].timestamp,
      )

      logger.debug(
          'Successfully flushed %s cache: %d chunks, %d bytes, saved as %s',
          cache_type,
          len(audio_cache),
          len(combined_audio_data),
          filename,
      )
      return audio_event

    except Exception as e:
      logger.error('Failed to flush %s cache: %s', cache_type, e)
      return None

  async def _flush_media_cache_to_services(
      self,
      invocation_context: InvocationContext,
      media_cache: list[RealtimeCacheEntry],
      cache_type: str,
  ) -> Event | None:
    """Flush a list of media cache entries to artifact services as a frame sequence."""
    if not invocation_context.artifact_service or not media_cache:
      logger.debug(
          'Skipping media cache flush: no artifact service or empty cache'
      )
      return None

    try:
      start_timestamp_ms = int(media_cache[0].timestamp * 1000)
      collection_name = (
          f'adk_live_media_storage_{cache_type}_{start_timestamp_ms}'
      )
      frames = [(entry.data, entry.timestamp) for entry in media_cache]

      revision_id = await invocation_context.artifact_service.save_media_frames(
          app_name=invocation_context.app_name,
          user_id=invocation_context.user_id,
          session_id=invocation_context.session.id,
          collection_name=collection_name,
          frames=frames,
      )

      artifact_ref = f'artifact://{invocation_context.app_name}/{invocation_context.user_id}/{invocation_context.session.id}/_adk_live/{collection_name}#{revision_id}'
      mime_type = media_cache[0].data.mime_type or 'image/jpeg'

      author = (
          _require_agent_name(invocation_context)
          if media_cache[0].role == 'model'
          else media_cache[0].role
      )
      media_event = Event(
          id=Event.new_id(),
          invocation_id=invocation_context.invocation_id,
          author=author,
          content=types.Content(
              role=media_cache[0].role,
              parts=[
                  types.Part(
                      file_data=types.FileData(
                          file_uri=artifact_ref, mime_type=mime_type
                      )
                  )
              ],
          ),
          timestamp=media_cache[0].timestamp,
      )

      logger.debug(
          'Successfully flushed %s cache: %d frames, saved as collection %s',
          cache_type,
          len(media_cache),
          collection_name,
      )
      return media_event

    except Exception as e:
      logger.error('Failed to flush %s cache: %s', cache_type, e)
      return None

  def get_cache_stats(
      self, invocation_context: InvocationContext
  ) -> dict[str, int]:
    """Get statistics about current cache state.

    Args:
      invocation_context: The invocation context.

    Returns:
      Dictionary containing multimodal cache statistics.
    """
    input_audio_cache = invocation_context.input_realtime_cache or []
    output_audio_cache = invocation_context.output_realtime_cache or []
    input_media_cache = invocation_context.input_media_realtime_cache or []
    output_media_cache = invocation_context.output_media_realtime_cache or []

    input_chunks = len(input_audio_cache)
    output_chunks = len(output_audio_cache)
    input_bytes = sum(
        len(entry.data.data or b'') for entry in input_audio_cache
    )
    output_bytes = sum(
        len(entry.data.data or b'') for entry in output_audio_cache
    )

    input_media_frames = len(input_media_cache)
    output_media_frames = len(output_media_cache)
    input_media_bytes = sum(
        len(entry.data.data or b'') for entry in input_media_cache
    )
    output_media_bytes = sum(
        len(entry.data.data or b'') for entry in output_media_cache
    )

    return {
        'input_chunks': input_chunks,
        'output_chunks': output_chunks,
        'input_bytes': input_bytes,
        'output_bytes': output_bytes,
        'total_chunks': input_chunks + output_chunks,
        'total_bytes': input_bytes + output_bytes,
        'input_media_frames': input_media_frames,
        'output_media_frames': output_media_frames,
        'input_media_bytes': input_media_bytes,
        'output_media_bytes': output_media_bytes,
        'total_media_frames': input_media_frames + output_media_frames,
        'total_media_bytes': input_media_bytes + output_media_bytes,
    }
