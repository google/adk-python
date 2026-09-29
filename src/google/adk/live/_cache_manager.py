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

"""Multimodal cache manager for live streaming audio chunks and media frames."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from typing import TYPE_CHECKING

from google.adk.platform import time as platform_time
from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict

from ..events.event import Event
from ._media_frames import _extension_for_mime_type
from ._media_frames import build_manifest
from ._media_frames import DEFAULT_MIME_TYPE
from ._media_frames import extract_frame
from ._media_frames import extract_preview_frame
from ._media_frames import MEDIA_ZIP_MIME_TYPE
from ._media_frames import MediaFrame
from ._media_frames import pack_media_frames
from ._media_frames import read_manifest
from ._media_frames import summarize_manifest
from ._media_frames import unpack_media_frames

if TYPE_CHECKING:
  from ..agents.invocation_context import InvocationContext

logger = logging.getLogger('google_adk.' + __name__)


class RealtimeCacheEntry(BaseModel):
  """Store raw realtime chunk or frame data for caching before flushing."""

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


# Deliberately duplicated from `flows.llm_flows.core._utils`
# rather than imported: `live` sits below `flows.llm_flows` in the
# layering, and importing upward would put a cycle back in.
def _require_agent_name(invocation_context: InvocationContext) -> str:
  agent = invocation_context.agent
  if agent is None:
    raise TypeError('Live streaming requires an agent in InvocationContext.')
  return agent.name


def _normalize_mime_type(
    mime_type: str | None,
    default: str = DEFAULT_MIME_TYPE,
) -> str:
  """Returns the lowercase base MIME type with parameters stripped."""
  base = (mime_type or '').split(';', 1)[0].strip().lower()
  return base or default


def _get_cache_list(
    invocation_context: InvocationContext,
    attr_name: str,
) -> list[RealtimeCacheEntry]:
  """Returns the cache list on ``invocation_context`` if initialized, else ``[]``."""
  cache = getattr(invocation_context, attr_name, None)
  return cache if isinstance(cache, list) else []


class CacheConfig:
  """Configuration for multimodal streaming cache behavior."""

  def __init__(
      self,
      max_cache_size_bytes: int = 10 * 1024 * 1024,  # 10MB
      max_cache_duration_seconds: float = 300.0,  # 5 minutes
      auto_flush_threshold: int = 100,  # Number of chunks
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
    self.config = config if config is not None else CacheConfig()

  def _resolve_cache_target(
      self,
      invocation_context: InvocationContext,
      cache_type: str,
      *,
      is_media: bool,
  ) -> tuple[list[RealtimeCacheEntry], str]:
    """Validates ``cache_type`` and returns ``(cache_list, role)``."""
    if cache_type == 'input':
      attr = (
          'input_media_realtime_cache' if is_media else 'input_realtime_cache'
      )
      role = 'user'
    elif cache_type == 'output':
      attr = (
          'output_media_realtime_cache' if is_media else 'output_realtime_cache'
      )
      role = 'model'
    else:
      raise ValueError("cache_type must be either 'input' or 'output'")

    cache = getattr(invocation_context, attr, None)
    if not isinstance(cache, list):
      cache = []
      setattr(invocation_context, attr, cache)
    return cache, role

  def _enforce_media_cache_limits(
      self, cache: list[RealtimeCacheEntry]
  ) -> None:
    """Evicts oldest frames in O(n) so frame count and byte limits hold.

    Always retains at least the most recently appended frame even if a single
    frame exceeds ``max_media_cache_size_bytes``.
    """
    max_frames = max(1, self.config.max_media_cache_frames)
    if len(cache) > max_frames:
      del cache[: len(cache) - max_frames]

    max_bytes = self.config.max_media_cache_size_bytes
    total_bytes = sum(len(entry.data.data or b'') for entry in cache)
    if total_bytes <= max_bytes or len(cache) <= 1:
      return

    evict_count = 0
    max_evict = len(cache) - 1
    while evict_count < max_evict and total_bytes > max_bytes:
      total_bytes -= len(cache[evict_count].data.data or b'')
      evict_count += 1

    if evict_count > 0:
      del cache[:evict_count]

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
      ValueError: If cache_type is not 'input' or 'output', or if audio_blob has
        no byte data.
    """
    audio_data = _require_blob_data(audio_blob)
    cache, role = self._resolve_cache_target(
        invocation_context, cache_type, is_media=False
    )

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
      ValueError: If cache_type is not 'input' or 'output', or if media_blob has
        no byte data.
    """
    media_data = _require_blob_data(media_blob)
    cache, role = self._resolve_cache_target(
        invocation_context, cache_type, is_media=True
    )

    media_entry = RealtimeCacheEntry(
        role=role, data=media_blob, timestamp=platform_time.get_time()
    )
    cache.append(media_entry)
    self._enforce_media_cache_limits(cache)

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

    input_audio = _get_cache_list(invocation_context, 'input_realtime_cache')
    if flush_user_audio and input_audio:
      tasks.append(
          self._flush_audio_cache_to_services(
              invocation_context,
              input_audio,
              'input_audio',
          )
      )
      targets.append('input_realtime_cache')

    output_audio = _get_cache_list(invocation_context, 'output_realtime_cache')
    if flush_model_audio and output_audio:
      tasks.append(
          self._flush_audio_cache_to_services(
              invocation_context,
              output_audio,
              'output_audio',
          )
      )
      targets.append('output_realtime_cache')

    input_media = _get_cache_list(
        invocation_context, 'input_media_realtime_cache'
    )
    if flush_user_media and input_media:
      tasks.append(
          self._flush_media_cache_to_services(
              invocation_context,
              input_media,
              'input_media',
          )
      )
      targets.append('input_media_realtime_cache')

    output_media = _get_cache_list(
        invocation_context, 'output_media_realtime_cache'
    )
    if flush_model_media and output_media:
      tasks.append(
          self._flush_media_cache_to_services(
              invocation_context,
              output_media,
              'output_media',
          )
      )
      targets.append('output_media_realtime_cache')

    if not tasks:
      return []

    results = await asyncio.gather(*tasks, return_exceptions=True)
    flushed_events: list[Event] = []

    for attr_name, result in zip(targets, results):
      if isinstance(result, Exception):
        logger.error('Failed to flush %s: %s', attr_name, result)
        continue
      if isinstance(result, Event):
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
      custom_metadata: dict[str, Any] | None = None,
  ) -> Event:
    """Saves an artifact and returns the corresponding ``_adk_live`` Event."""
    assert invocation_context.artifact_service is not None
    artifact = types.Part(
        inline_data=types.Blob(data=data, mime_type=mime_type)
    )
    save_kwargs: dict[str, Any] = {
        'app_name': invocation_context.app_name,
        'user_id': invocation_context.user_id,
        'session_id': invocation_context.session.id,
        'filename': filename,
        'artifact': artifact,
    }
    if custom_metadata is not None:
      save_kwargs['custom_metadata'] = custom_metadata

    revision_id = await invocation_context.artifact_service.save_artifact(
        **save_kwargs
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
                        file_uri=artifact_ref,
                        mime_type=mime_type,
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

      audio_event = await self._save_artifact_and_build_event(
          invocation_context,
          filename=filename,
          data=combined_audio_data,
          mime_type=mime_type,
          role=audio_cache[0].role,
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

  # Backward-compatible alias for callers/subclasses referencing the old name.
  _flush_cache_to_services = _flush_audio_cache_to_services

  async def _flush_media_cache_to_services(
      self,
      invocation_context: InvocationContext,
      media_cache: list[RealtimeCacheEntry],
      cache_type: str,
  ) -> Event | None:
    """Flush cached media frames to artifact services.

    - Single-frame batches (`len(media_cache) == 1`) are stored directly with
      their native `image/*` or `video/*` MIME type so UI clients (`adk web`)
      can render the image immediately in the timeline and Artifact panel.
    - Multi-frame sequences (`len(media_cache) > 1`) are packed via
      `pack_media_frames()` into a single uncompressed (`ZIP_STORED`) `.zip`
      archive (`application/zip`) with a bounded `summarize_manifest()`
      `custom_metadata` summary (`type='video_frame_sequence'`).
    """
    if not invocation_context.artifact_service or not media_cache:
      logger.debug(
          'Skipping media cache flush: no artifact service or empty cache'
      )
      return None

    try:
      frames = [
          MediaFrame(blob=entry.data, timestamp=entry.timestamp)
          for entry in media_cache
      ]
      start_timestamp_ms = round(frames[0].timestamp * 1000)
      if len(frames) == 1:
        frame = frames[0]
        payload = _require_blob_data(frame.blob)
        mime_type = _normalize_mime_type(frame.blob.mime_type)
        ext = _extension_for_mime_type(mime_type)
        filename = (
            f'adk_live_media_storage_{cache_type}_{start_timestamp_ms}.{ext}'
        )
        manifest = build_manifest(
            frames,
            custom_metadata={
                'totalBytes': len(payload),
                'mimeTypes': [mime_type],
            },
        )
        manifest['type'] = 'single_media_frame'
        custom_metadata = summarize_manifest(manifest)
      else:
        filename = (
            f'adk_live_media_storage_{cache_type}_{start_timestamp_ms}.zip'
        )
        payload, manifest = pack_media_frames(frames)
        custom_metadata = summarize_manifest(manifest)
        mime_type = MEDIA_ZIP_MIME_TYPE

      media_event = await self._save_artifact_and_build_event(
          invocation_context,
          filename=filename,
          data=payload,
          mime_type=mime_type,
          role=media_cache[0].role,
          timestamp=media_cache[0].timestamp,
          custom_metadata=custom_metadata,
      )

      logger.debug(
          'Successfully flushed %s cache: %d frames, %d bytes, saved as %s',
          cache_type,
          len(media_cache),
          len(payload),
          filename,
      )
      return media_event

    except Exception as e:
      logger.error('Failed to flush %s cache: %s', cache_type, e)
      return None

  def _require_artifact_blob(self, artifact: types.Part) -> types.Blob:
    """Validates that ``artifact`` carries non-empty ``inline_data`` bytes."""
    blob = artifact.inline_data
    if blob is None or not isinstance(blob.data, bytes) or not blob.data:
      raise ValueError('Media artifact Part must contain non-empty byte data.')
    return blob

  def load_media_frames(
      self,
      artifact: types.Part,
  ) -> tuple[list[MediaFrame], dict[str, Any]]:
    """Restores frames and manifest from a flushed single-frame or ZIP artifact.

    Args:
      artifact: A ``types.Part`` returned by ``artifact_service.load_artifact``.

    Returns:
      A tuple ``(frames, manifest)`` whether the batch was stored as a single
      native ``image/*`` / ``video/*`` frame or a multi-frame ``.zip`` archive.
    """
    blob = self._require_artifact_blob(artifact)
    mime_type = _normalize_mime_type(blob.mime_type)
    if mime_type == MEDIA_ZIP_MIME_TYPE:
      return unpack_media_frames(blob.data or b'')
    frame = MediaFrame(
        blob=types.Blob(data=blob.data, mime_type=mime_type),
        timestamp=0.0,
    )
    manifest = build_manifest([frame])
    manifest['type'] = 'single_media_frame'
    return [frame], manifest

  def load_media_manifest(self, artifact: types.Part) -> dict[str, Any]:
    """Reads the manifest from a flushed single-frame or ZIP media artifact."""
    blob = self._require_artifact_blob(artifact)
    mime_type = _normalize_mime_type(blob.mime_type)
    if mime_type == MEDIA_ZIP_MIME_TYPE:
      return read_manifest(blob.data or b'')
    frame = MediaFrame(
        blob=types.Blob(data=blob.data, mime_type=mime_type),
        timestamp=0.0,
    )
    manifest = build_manifest([frame])
    manifest['type'] = 'single_media_frame'
    return manifest

  def load_media_frame(
      self,
      artifact: types.Part,
      index: int = 0,
  ) -> MediaFrame:
    """Extracts a single frame at ``index`` from a flushed media artifact."""
    blob = self._require_artifact_blob(artifact)
    mime_type = _normalize_mime_type(blob.mime_type)
    if mime_type == MEDIA_ZIP_MIME_TYPE:
      return extract_frame(blob.data or b'', index)
    if index != 0:
      raise ValueError(f'Media frame artifact has no frame at index {index}.')
    return MediaFrame(
        blob=types.Blob(data=blob.data, mime_type=mime_type),
        timestamp=0.0,
    )

  def load_preview_frame(self, artifact: types.Part) -> MediaFrame:
    """Extracts the first (preview) frame from a flushed media artifact."""
    blob = self._require_artifact_blob(artifact)
    mime_type = _normalize_mime_type(blob.mime_type)
    if mime_type == MEDIA_ZIP_MIME_TYPE:
      return extract_preview_frame(blob.data or b'')
    return MediaFrame(
        blob=types.Blob(data=blob.data, mime_type=mime_type),
        timestamp=0.0,
    )

  def get_cache_stats(
      self, invocation_context: InvocationContext
  ) -> dict[str, int]:
    """Get statistics about current cache state.

    Args:
      invocation_context: The invocation context.

    Returns:
      Dictionary containing multimodal cache statistics.
    """
    input_audio_cache = _get_cache_list(
        invocation_context, 'input_realtime_cache'
    )
    output_audio_cache = _get_cache_list(
        invocation_context, 'output_realtime_cache'
    )
    input_media_cache = _get_cache_list(
        invocation_context, 'input_media_realtime_cache'
    )
    output_media_cache = _get_cache_list(
        invocation_context, 'output_media_realtime_cache'
    )

    input_chunks = len(input_audio_cache)
    output_chunks = len(output_audio_cache)
    input_bytes = sum(
        len(_require_blob_data(entry.data)) for entry in input_audio_cache
    )
    output_bytes = sum(
        len(_require_blob_data(entry.data)) for entry in output_audio_cache
    )

    input_media_frames = len(input_media_cache)
    output_media_frames = len(output_media_cache)
    input_media_bytes = sum(
        len(_require_blob_data(entry.data)) for entry in input_media_cache
    )
    output_media_bytes = sum(
        len(_require_blob_data(entry.data)) for entry in output_media_cache
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
