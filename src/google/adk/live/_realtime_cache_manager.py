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
from ._visual_frames import pack_visual_frames
from ._visual_frames import VISUAL_FRAMES_ZIP_MIME_TYPE
from ._visual_frames import VisualFrame

if TYPE_CHECKING:
  from ..agents.invocation_context import InvocationContext

logger = logging.getLogger('google_adk.' + __name__)


class RealtimeCacheEntry(BaseModel):
  """A cached audio chunk or image frame, kept until the next flush."""

  model_config = ConfigDict(
      arbitrary_types_allowed=True,
      extra='forbid',
  )
  """The pydantic model config."""

  role: str
  """The role that created this data, typically "user" or "model"."""

  data: types.Blob
  """The audio chunk or image frame."""

  timestamp: float
  """Timestamp when the chunk or frame was received."""


def _require_audio_data(blob: types.Blob) -> bytes:
  data = blob.data
  if not isinstance(data, bytes):
    raise ValueError('Audio blobs must contain byte data.')
  return data


# Deliberately duplicated from `flows.llm_flows.core._utils`
# rather than imported: `live` sits below `flows.llm_flows` in the
# layering, and importing upward would put a cycle back in.
def _require_agent_name(invocation_context: InvocationContext) -> str:
  agent = invocation_context.agent
  if agent is None:
    raise TypeError('Live audio requires an agent in InvocationContext.')
  return agent.name


def _adk_live_artifact_event(
    invocation_context: InvocationContext,
    first_entry: RealtimeCacheEntry,
    filename: str,
    mime_type: str | None,
    revision_id: int,
) -> Event:
  """Builds the session event that references a saved live artifact."""
  # Create artifact reference for session service
  artifact_ref = f'artifact://{invocation_context.app_name}/{invocation_context.user_id}/{invocation_context.session.id}/_adk_live/{filename}#{revision_id}'

  # Create event with file data reference to add to session
  # For model events, author should be the agent name, not the role
  author = (
      _require_agent_name(invocation_context)
      if first_entry.role == 'model'
      else first_entry.role
  )
  return Event(
      id=Event.new_id(),
      invocation_id=invocation_context.invocation_id,
      author=author,
      content=types.Content(
          role=first_entry.role,
          parts=[
              types.Part(
                  file_data=types.FileData(
                      file_uri=artifact_ref, mime_type=mime_type
                  )
              )
          ],
      ),
      timestamp=first_entry.timestamp,
  )


def _non_decreasing_frames(
    entries: list[RealtimeCacheEntry],
) -> list[VisualFrame]:
  """Returns the cached frames with timestamps that never go backwards."""
  # Entries carry wall-clock timestamps, which can step back (for example on a
  # clock correction), and the archive rejects decreasing timestamps. A frame
  # stamped before the one ahead of it takes that frame's timestamp.
  frames: list[VisualFrame] = []
  latest = entries[0].timestamp
  for entry in entries:
    latest = max(latest, entry.timestamp)
    frames.append(VisualFrame(blob=entry.data, timestamp=latest))
  return frames


class RealtimeCacheManager:
  """Manages audio and image frame caching and flushing for live flows."""

  def __init__(self, config: RealtimeCacheConfig | None = None) -> None:
    """Initialize the realtime cache manager.

    Args:
      config: Configuration for caching behavior.
    """
    self.config = config or RealtimeCacheConfig()
    self._warned_unsaved_blob_kinds: set[tuple[str, str | None]] = set()

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
    audio_data = _require_audio_data(audio_blob)
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
        role=role, data=audio_blob, timestamp=platform_time.get_time()
    )
    cache.append(audio_entry)

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
    """Cache an incoming user or outgoing model blob if it is audio or an image.

    Audio chunks are concatenated into one audio file per flush. The user's
    image frames, such as camera or screen-share frames, are kept apart and
    saved as one ZIP archive per flush. Other blobs, such as `video/*` or image
    frames from the model, are not saved, and the first one of each kind logs a
    warning. Blobs are cached as copies, so later changes to the caller's blob
    do not change the cache.

    Args:
      invocation_context: The current invocation context.
      blob: The blob to cache.
      cache_type: Type of data to cache, either 'input' or 'output'.

    Raises:
      ValueError: If the blob is audio or an image and cache_type is not
        'input' or 'output'.
    """
    if blob.mime_type and blob.mime_type.startswith('audio/'):
      self.cache_audio(
          invocation_context,
          types.Blob(data=blob.data, mime_type=blob.mime_type),
          cache_type,
      )
    elif blob.mime_type and blob.mime_type.startswith('image/'):
      self._cache_visual_frame(
          invocation_context,
          types.Blob(data=blob.data, mime_type=blob.mime_type),
          cache_type,
      )
    else:
      self._warn_blob_not_saved(cache_type, blob.mime_type)

  def _warn_blob_not_saved(
      self, cache_type: str, mime_type: str | None
  ) -> None:
    """Logs a warning the first time a kind of blob is not saved.

    A live avatar, for example, streams dozens of video chunks a second, so a
    warning for every blob would flood the log.
    """
    kind = (cache_type, mime_type)
    if kind in self._warned_unsaved_blob_kinds:
      return
    self._warned_unsaved_blob_kinds.add(kind)
    logger.warning(
        'Not saving %s live blobs of type %r: save_live_blob saves audio and'
        " the user's image frames only. Later blobs of this kind are skipped"
        ' without a warning.',
        cache_type,
        mime_type,
    )

  def _cache_visual_frame(
      self,
      invocation_context: InvocationContext,
      frame: types.Blob,
      cache_type: str,
  ) -> None:
    """Caches one of the user's frames, dropping the oldest past the limit."""
    if cache_type == 'output':
      # Live models do not send image frames; a live avatar's picture arrives
      # as video.
      self._warn_blob_not_saved(cache_type, frame.mime_type)
      return
    if cache_type != 'input':
      raise ValueError("cache_type must be either 'input' or 'output'")
    if not frame.data:
      # The archive rejects empty frames; one must not cost the whole batch.
      logger.debug('Skipping image frame with no data')
      return

    cache = invocation_context._input_visual_cache
    excess = len(cache) + 1 - self.config.max_visual_cache_frames
    if excess > 0:
      del cache[:excess]
      logger.debug(
          'Dropped the %d oldest image frames to stay in limits', excess
      )
    cache.append(
        RealtimeCacheEntry(
            role='user', data=frame, timestamp=platform_time.get_time()
        )
    )

  async def flush_caches(
      self,
      invocation_context: InvocationContext,
      flush_user_audio: bool = True,
      flush_model_audio: bool = True,
      flush_user_visual: bool = True,
  ) -> list[Event]:
    """Flush the audio and image frame caches to artifact services.

    Each flushed cache is saved as one artifact: the audio as one audio file,
    the user's frames as one ZIP archive. The file data reference is added to
    the session as an event. The artifact reference follows the naming
    convention: artifact_ref =
    f"artifact://{invocation_context.app_name}/{invocation_context.user_id}/
    {invocation_context.session.id}/_adk_live/{filename}#{revision_id}"

    A flushed cache is emptied whether or not an artifact was written, so a
    cache the artifact service could not accept is dropped rather than kept
    for a later attempt.

    Args:
      invocation_context: The invocation context containing the caches.
      flush_user_audio: Whether to flush the input (user) audio cache.
      flush_model_audio: Whether to flush the output (model) audio cache.
      flush_user_visual: Whether to flush the user's image frame cache.

    Returns:
      A list of Event objects created from the flushed caches: user audio,
      model audio, then the user's frames.
    """
    flushed_events: list[Event] = []
    if flush_user_audio and invocation_context.input_realtime_cache:
      # Swap before the await so audio cached during it is kept for the next
      # flush rather than discarded by the unconditional drop.
      input_cache = invocation_context.input_realtime_cache
      invocation_context.input_realtime_cache = []
      audio_event = await self._flush_audio_cache_to_services(
          invocation_context,
          input_cache,
          'input_audio',
      )
      if audio_event:
        flushed_events.append(audio_event)

    if flush_model_audio and invocation_context.output_realtime_cache:
      logger.debug('Flushed output audio cache')
      output_cache = invocation_context.output_realtime_cache
      invocation_context.output_realtime_cache = []
      audio_event = await self._flush_audio_cache_to_services(
          invocation_context,
          output_cache,
          'output_audio',
      )
      if audio_event:
        flushed_events.append(audio_event)

    frame_cache = invocation_context._input_visual_cache
    if flush_user_visual and frame_cache:
      # Take the frames before the await, so frames cached during it are kept
      # for the next flush. The list is emptied in place because contexts
      # copied with `model_copy()` share it.
      frames = frame_cache[:]
      frame_cache.clear()
      frame_event = await self._flush_visual_cache_to_services(
          invocation_context, frames, 'input_visual'
      )
      if frame_event:
        flushed_events.append(frame_event)

    return flushed_events

  async def _flush_audio_cache_to_services(
      self,
      invocation_context: InvocationContext,
      audio_cache: list[RealtimeCacheEntry],
      cache_type: str,
  ) -> Event | None:
    """Flush a list of audio cache entries to artifact services.

    The artifact service stores the actual blob. The session stores the
    reference to the stored blob.

    Args:
      invocation_context: The invocation context.
      audio_cache: The audio cache to flush.
      cache_type: Type identifier for the cache ('input_audio' or
        'output_audio').

    Returns:
      The created Event if the cache was successfully flushed, None otherwise.
    """
    if not invocation_context.artifact_service or not audio_cache:
      logger.debug('Skipping cache flush: no artifact service or empty cache')
      return None

    try:
      # Combine audio chunks into a single file. Use join rather than repeated
      # `+=`, which is O(n^2) over the total audio size.
      mime_type = audio_cache[0].data.mime_type or 'audio/pcm'
      combined_audio_data = b''.join(
          entry.data.data or b'' for entry in audio_cache
      )

      # Generate filename with timestamp from first audio chunk (when recording started)
      timestamp = int(audio_cache[0].timestamp * 1000)  # milliseconds
      # Drop MIME parameters such as `;rate=24000` so they don't end up in the
      # file extension. The full MIME type is kept on the saved artifact.
      extension = mime_type.split(';', 1)[0].strip().split('/')[-1]
      filename = f'adk_live_audio_storage_{cache_type}_{timestamp}.{extension}'

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

      audio_event = _adk_live_artifact_event(
          invocation_context, audio_cache[0], filename, mime_type, revision_id
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

  async def _flush_visual_cache_to_services(
      self,
      invocation_context: InvocationContext,
      frames: list[RealtimeCacheEntry],
      cache_type: str,
  ) -> Event | None:
    """Flush a list of cached image frames to artifact services.

    The frames are packed into one uncompressed ZIP archive, even when there
    is only one, with the metadata `pack_visual_frames()` returns as the
    artifact's custom metadata.

    Args:
      invocation_context: The invocation context.
      frames: The cached frames, oldest first.
      cache_type: Type identifier for the cache ('input_visual').

    Returns:
      The created Event if the frames were saved, None otherwise.
    """
    if not invocation_context.artifact_service or not frames:
      logger.debug('Skipping cache flush: no artifact service or empty cache')
      return None

    try:
      # Packing copies every frame into the archive, so it runs in a worker
      # thread to keep the live stream flowing.
      archive_bytes, custom_metadata = await asyncio.to_thread(
          pack_visual_frames, _non_decreasing_frames(frames)
      )
      # Rounded the way the archive rounds its times, so an archive's name
      # carries the same millisecond as its `startTimestampMs`.
      timestamp = round(frames[0].timestamp * 1000)  # milliseconds
      filename = f'adk_live_visual_storage_{cache_type}_{timestamp}.zip'
      revision_id = await invocation_context.artifact_service.save_artifact(
          app_name=invocation_context.app_name,
          user_id=invocation_context.user_id,
          session_id=invocation_context.session.id,
          filename=filename,
          artifact=types.Part(
              inline_data=types.Blob(
                  data=archive_bytes, mime_type=VISUAL_FRAMES_ZIP_MIME_TYPE
              )
          ),
          custom_metadata=custom_metadata,
      )

      logger.debug(
          'Successfully flushed %s cache: %d frames, saved as %s',
          cache_type,
          len(frames),
          filename,
      )
      return _adk_live_artifact_event(
          invocation_context,
          frames[0],
          filename,
          VISUAL_FRAMES_ZIP_MIME_TYPE,
          revision_id,
      )

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
      Dictionary containing cache statistics.
    """
    input_count = len(invocation_context.input_realtime_cache or [])
    output_count = len(invocation_context.output_realtime_cache or [])

    input_bytes = sum(
        len(_require_audio_data(entry.data))
        for entry in invocation_context.input_realtime_cache or []
    )
    output_bytes = sum(
        len(_require_audio_data(entry.data))
        for entry in invocation_context.output_realtime_cache or []
    )

    return {
        'input_chunks': input_count,
        'output_chunks': output_count,
        'input_bytes': input_bytes,
        'output_bytes': output_bytes,
        'total_chunks': input_count + output_count,
        'total_bytes': input_bytes + output_bytes,
    }


class RealtimeCacheConfig:
  """Configuration for audio and image frame caching."""

  def __init__(
      self,
      max_cache_size_bytes: int = 10 * 1024 * 1024,  # 10MB
      max_cache_duration_seconds: float = 300.0,  # 5 minutes
      auto_flush_threshold: int = 100,  # Number of chunks
      max_visual_cache_frames: int = 600,
  ) -> None:
    """Initialize realtime cache configuration.

    Args:
      max_cache_size_bytes: Maximum cache size in bytes before auto-flush.
      max_cache_duration_seconds: Maximum duration to keep data in cache.
      auto_flush_threshold: Number of chunks that triggers auto-flush.
      max_visual_cache_frames: Maximum number of the user's image frames kept
        until the next flush. The oldest frames are dropped first.

    Raises:
      ValueError: If max_visual_cache_frames is less than 1.
    """
    if max_visual_cache_frames < 1:
      raise ValueError('max_visual_cache_frames must be at least 1.')
    self.max_cache_size_bytes = max_cache_size_bytes
    self.max_cache_duration_seconds = max_cache_duration_seconds
    self.auto_flush_threshold = auto_flush_threshold
    self.max_visual_cache_frames = max_visual_cache_frames
