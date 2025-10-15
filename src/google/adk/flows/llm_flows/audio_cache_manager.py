# Copyright 2025 Google LLC
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

import logging
import time
from typing import TYPE_CHECKING

from google.genai import types

from ...agents.invocation_context import RealtimeCacheEntry
from ...events.event import Event

if TYPE_CHECKING:
  from ...agents.invocation_context import InvocationContext

logger = logging.getLogger('google_adk.' + __name__)


class AudioCacheManager:
  """Manages audio caching and flushing for live streaming flows."""

  def __init__(self, config: AudioCacheConfig | None = None):
    """Initialize the audio cache manager.

    Args:
      config: Configuration for audio caching behavior.
    """
    self.config = config or AudioCacheConfig()

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
    if cache_type == 'input':
      if not invocation_context.input_realtime_cache:
        invocation_context.input_realtime_cache = []
      cache = invocation_context.input_realtime_cache
      role = 'user'
    elif cache_type == 'output':
      if not invocation_context.output_realtime_cache:
        invocation_context.output_realtime_cache = []
      cache = invocation_context.output_realtime_cache
      role = 'model'
    else:
      raise ValueError("cache_type must be either 'input' or 'output'")

    audio_entry = RealtimeCacheEntry(
        role=role, data=audio_blob, timestamp=time.time()
    )
    cache.append(audio_entry)

    logger.debug(
        'Cached %s audio chunk: %d bytes, cache size: %d',
        cache_type,
        len(audio_blob.data),
        len(cache),
    )

  async def flush_caches(
      self,
      invocation_context: InvocationContext,
      flush_user_audio: bool = True,
      flush_model_audio: bool = True,
  ) -> None:
    """Flush audio caches to artifact services.

    The multimodality data is saved in artifact service in the format of
    audio file. The file data reference is added to the session as an event.
    The audio file follows the naming convention: artifact_ref =
    f"artifact://{invocation_context.app_name}/{invocation_context.user_id}/
    {invocation_context.session.id}/_adk_live/{filename}#{revision_id}"

    Note: video data is not supported yet.

    Args:
      invocation_context: The invocation context containing audio caches.
      flush_user_audio: Whether to flush the input (user) audio cache.
      flush_model_audio: Whether to flush the output (model) audio cache.
    """
    if flush_user_audio and invocation_context.input_realtime_cache:
      flush_success = await self._flush_cache_to_services(
          invocation_context,
          invocation_context.input_realtime_cache,
          'input_audio',
      )
      if flush_success:
        invocation_context.input_realtime_cache = []

    if flush_model_audio and invocation_context.output_realtime_cache:
      logger.debug('Flushed output audio cache')
      flush_success = await self._flush_cache_to_services(
          invocation_context,
          invocation_context.output_realtime_cache,
          'output_audio',
      )
      if flush_success:
        invocation_context.output_realtime_cache = []

  async def _flush_cache_to_services(
      self,
      invocation_context: InvocationContext,
      audio_cache: list[RealtimeCacheEntry],
      cache_type: str,
  ) -> bool:
    """Flush a list of audio cache entries to artifact services.

    The artifact service stores the actual blob. The session stores the
    reference to the stored blob.

    Args:
      invocation_context: The invocation context.
      audio_cache: The audio cache to flush.
      cache_type: Type identifier for the cache ('input_audio' or 'output_audio').

    Returns:
      True if the cache was successfully flushed, False otherwise.
    """
    if not invocation_context.artifact_service or not audio_cache:
      logger.debug('Skipping cache flush: no artifact service or empty cache')
      return False

    try:
      # Combine audio chunks into a single file
      combined_audio_data = b''
      mime_type = audio_cache[0].data.mime_type if audio_cache else 'audio/pcm'

      for entry in audio_cache:
        combined_audio_data += entry.data.data

      # Generate filename with timestamp from first audio chunk (when recording started)
      timestamp = int(audio_cache[0].timestamp * 1000)  # milliseconds
      filename = f"adk_live_audio_storage_{cache_type}_{timestamp}.{mime_type.split('/')[-1]}"

      # Save to artifact service
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

      # Create artifact reference for session service
      artifact_ref = f'artifact://{invocation_context.app_name}/{invocation_context.user_id}/{invocation_context.session.id}/_adk_live/{filename}#{revision_id}'

      # Create event with file data reference to add to session
      audio_event = Event(
          id=Event.new_id(),
          invocation_id=invocation_context.invocation_id,
          author=audio_cache[0].role,
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
      return False

  def get_cache_stats(
      self, invocation_context: InvocationContext
  ) -> dict[str, int]:
    """Get statistics about current cache state.

    Args:
      invocation_context: The invocation context.

    Returns:
      Dictionary containing cache statistics.
    """
    input_count = len(invocation_context.input_realtime_cache or [])
    output_count = len(invocation_context.output_realtime_cache or [])

    input_bytes = sum(
        len(entry.data.data)
        for entry in (invocation_context.input_realtime_cache or [])
    )
    output_bytes = sum(
        len(entry.data.data)
        for entry in (invocation_context.output_realtime_cache or [])
    )

    return {
        'input_chunks': input_count,
        'output_chunks': output_count,
        'input_bytes': input_bytes,
        'output_bytes': output_bytes,
        'total_chunks': input_count + output_count,
        'total_bytes': input_bytes + output_bytes,
    }


class AudioCacheConfig:
  """Configuration for audio caching behavior."""

  def __init__(
      self,
      max_cache_size_bytes: int = 10 * 1024 * 1024,  # 10MB
      max_cache_duration_seconds: float = 300.0,  # 5 minutes
      auto_flush_threshold: int = 100,  # Number of chunks
  ):
    """Initialize audio cache configuration.

    Args:
      max_cache_size_bytes: Maximum cache size in bytes before auto-flush.
      max_cache_duration_seconds: Maximum duration to keep data in cache.
      auto_flush_threshold: Number of chunks that triggers auto-flush.
    """
    self.max_cache_size_bytes = max_cache_size_bytes
    self.max_cache_duration_seconds = max_cache_duration_seconds
    self.auto_flush_threshold = auto_flush_threshold
