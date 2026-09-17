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

"""Packs and unpacks live media frame sequences as a single ZIP artifact.

Live video and image frames arrive as many small blobs. Writing each one as
its own artifact would mean hundreds of round trips to the artifact backend
and would require a media-specific method on ``BaseArtifactService``, which
most backends (in-memory, filesystem, GCS) cannot implement without breaking
their flat ``{filename}/{version}`` layout.

Instead, the frames plus a ``metadata.json`` manifest are packed into one
uncompressed ZIP and stored with the ordinary ``save_artifact`` call, so
every backend works unchanged. ``ZIP_STORED`` is deliberate: the frames are
already JPEG/PNG so deflate buys nothing, and an uncompressed archive keeps
the central directory usable for reading one member without inflating the
rest.

Archive layout::

    <artifact>.zip
    ├── metadata.json
    └── frames/
        ├── frame_0000.jpeg
        ├── frame_0001.jpeg
        └── ...

Inside ``metadata.json``, each entry in ``frames`` records its archive
``member_name`` under the camelCase JSON key ``fileName`` (for example,
``"frames/frame_0000.jpeg"``).
"""

from __future__ import annotations

import io
import json
import logging
import math
import mimetypes
from typing import Any
import zipfile

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict

logger = logging.getLogger('google_adk.' + __name__)

DEFAULT_MIME_TYPE = 'image/jpeg'
"""MIME type assumed when a frame blob does not declare one."""

METADATA_FILENAME = 'metadata.json'
"""Name of the manifest member at the root of the archive."""

FRAMES_DIRECTORY = 'frames'
"""Name of the archive directory holding the frame members."""

MEDIA_ZIP_MIME_TYPE = 'application/zip'
"""MIME type of the packed archive, as stored in the artifact service."""

# Python 3.10's stdlib `mimetypes` tables do not include `image/webp` (added in
# Python 3.11). Register it on a private `MimeTypes` instance so Python 3.10
# containers without `/etc/mime.types` resolve `.webp` without mutating global
# `mimetypes` state.
_MIME_TYPES = mimetypes.MimeTypes()
_MIME_TYPES.add_type('image/webp', '.webp')


class MediaFrame(BaseModel):
  """A single media frame and the time it was captured."""

  model_config = ConfigDict(
      arbitrary_types_allowed=True,
      extra='forbid',
  )
  """The pydantic model config."""

  blob: types.Blob
  """The frame payload: its bytes and its MIME type."""

  timestamp: float
  """Wall-clock seconds at which the frame was received."""


def _extension_for_mime_type(mime_type: str | None) -> str:
  """Returns the archive file extension to use for a frame MIME type."""
  raw_mime = (mime_type or DEFAULT_MIME_TYPE).split(';')[0].strip().lower()
  guessed = _MIME_TYPES.guess_extension(raw_mime, strict=False)
  extension = guessed.lstrip('.') if guessed else ''
  return 'jpeg' if not extension or extension in ('jpg', 'jpe') else extension


def _frame_member_name(index: int, mime_type: str | None) -> str:
  """Returns the archive ``member_name`` for the frame at ``index``."""
  extension = _extension_for_mime_type(mime_type)
  return f'{FRAMES_DIRECTORY}/frame_{index:04d}.{extension}'


def _read_archive_manifest(
    archive: zipfile.ZipFile,
    member_names: set[str],
) -> dict[str, Any]:
  """Reads and validates ``metadata.json`` from an open archive."""
  if METADATA_FILENAME not in member_names:
    raise ValueError(f'Media frame archive is missing {METADATA_FILENAME}.')

  manifest = json.loads(archive.read(METADATA_FILENAME).decode('utf-8'))
  if not isinstance(manifest, dict):
    raise ValueError(f'{METADATA_FILENAME} must contain a JSON object.')
  entries = manifest.get('frames', [])
  if not isinstance(entries, list):
    raise ValueError(f'{METADATA_FILENAME} frames field must be a list.')
  start_ts_ms = manifest.get('startTimestampMs', 0)
  if (
      isinstance(start_ts_ms, bool)
      or not isinstance(start_ts_ms, (int, float))
      or not math.isfinite(start_ts_ms)
  ):
    raise ValueError(
        f'{METADATA_FILENAME} startTimestampMs must be a finite number.'
    )
  return manifest


def _read_frame_entry(
    archive: zipfile.ZipFile,
    member_names: set[str],
    entry: Any,
    start_timestamp: float,
) -> MediaFrame:
  """Validates a manifest frame entry and constructs its :class:`MediaFrame`."""
  if not isinstance(entry, dict):
    raise ValueError(f'{METADATA_FILENAME} frame entries must be objects.')
  member_name = entry.get('fileName')
  if (
      not isinstance(member_name, str)
      or not member_name
      or member_name not in member_names
  ):
    raise ValueError(
        f'Media frame archive is missing frame member {member_name!r}.'
    )
  offset_ms = entry.get('offsetMs', 0)
  if (
      isinstance(offset_ms, bool)
      or not isinstance(offset_ms, (int, float))
      or not math.isfinite(offset_ms)
  ):
    raise ValueError(
        f'{METADATA_FILENAME} frame offsetMs must be a finite number.'
    )
  mime_type = entry.get('mimeType', DEFAULT_MIME_TYPE)
  if not isinstance(mime_type, str) or not mime_type:
    mime_type = DEFAULT_MIME_TYPE
  return MediaFrame(
      blob=types.Blob(
          data=archive.read(member_name),
          mime_type=mime_type,
      ),
      timestamp=start_timestamp + offset_ms / 1000.0,
  )


def build_manifest(
    frames: list[MediaFrame],
    custom_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Builds the manifest describing a frame sequence.

  The full manifest (including the per-frame index) is written into the
  archive as ``metadata.json``, while :func:`summarize_manifest` strips the
  ``frames`` index so the bounded summary can be passed to ``save_artifact``
  as ``custom_metadata``.

  Args:
    frames: The frames, in capture order.
    custom_metadata: Extra caller-supplied keys merged into the manifest.
      Computed structural fields (such as ``startTimestampMs``, ``frameCount``,
      and ``frames``) always take precedence so the archive remains
      self-consistent and unpackable.

  Returns:
    The manifest dictionary.

  Raises:
    ValueError: If ``frames`` is empty or timestamps are non-finite or
      decreasing.
  """
  if not frames:
    raise ValueError('Cannot build a manifest for an empty frame list.')

  for index, frame in enumerate(frames):
    if not math.isfinite(frame.timestamp):
      raise ValueError(f'Frame {index} timestamp must be finite.')

  previous_timestamp = frames[0].timestamp
  for index, frame in enumerate(frames[1:], start=1):
    if frame.timestamp < previous_timestamp:
      raise ValueError(
          f'Frame timestamps must be non-decreasing (frame {index} at'
          f' {frame.timestamp} < {previous_timestamp}).'
      )
    previous_timestamp = frame.timestamp

  start_timestamp = frames[0].timestamp
  end_timestamp = frames[-1].timestamp
  elapsed_seconds = end_timestamp - start_timestamp
  # Rounded, not truncated. Seconds-to-milliseconds lands on values like
  # 599.9999999999999 for a timestamp that is exactly 0.6s after the start,
  # and `int()` would floor that to 599, making every offset up to a
  # millisecond short and the sequence fractionally shorter than it was.
  duration_ms = round(elapsed_seconds * 1000)
  frame_count = len(frames)
  # Rate over the gaps between frames, not the frames themselves, so a
  # single frame reports 0.0 rather than dividing by a zero interval.
  estimated_fps = (
      round((frame_count - 1) / elapsed_seconds, 2)
      if frame_count > 1 and elapsed_seconds > 0
      else 0.0
  )

  frame_entries: list[dict[str, Any]] = []
  for index, frame in enumerate(frames):
    data = frame.blob.data or b''
    mime_type = (frame.blob.mime_type or DEFAULT_MIME_TYPE).lower()
    frame_entries.append({
        'frameIndex': index,
        'offsetMs': round((frame.timestamp - start_timestamp) * 1000),
        'fileName': _frame_member_name(index, mime_type),
        'mimeType': mime_type,
        'sizeBytes': len(data),
    })

  manifest: dict[str, Any] = {
      **(custom_metadata or {}),
      'type': 'video_frame_sequence',
      'frameCount': frame_count,
      'startTimestampMs': round(start_timestamp * 1000),
      'endTimestampMs': round(end_timestamp * 1000),
      'durationMs': duration_ms,
      'estimatedFps': estimated_fps,
      'frames': frame_entries,
  }
  return manifest


def summarize_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
  """Returns the manifest without its per-frame index.

  The per-frame index grows with the frame count and belongs in the archive,
  not in the artifact's ``custom_metadata``, which several backends keep in
  object metadata with a size limit. The summary is what a caller needs to
  decide whether to download the archive at all.

  Args:
    manifest: A manifest as returned by :func:`build_manifest`.

  Returns:
    A copy of the manifest with the ``frames`` key removed.
  """
  return {key: value for key, value in manifest.items() if key != 'frames'}


def pack_media_frames(
    frames: list[MediaFrame],
    custom_metadata: dict[str, Any] | None = None,
) -> tuple[bytes, dict[str, Any]]:
  """Packs frames and their manifest into one uncompressed ZIP archive.

  Args:
    frames: The frames, in capture order. Must not be empty.
    custom_metadata: Extra keys merged into the manifest.

  Returns:
    A tuple of the archive bytes and the manifest that was written into it.

  Raises:
    ValueError: If ``frames`` is empty or a frame carries no bytes.
  """
  if not frames:
    raise ValueError('Cannot pack an empty frame list.')

  for index, frame in enumerate(frames):
    if not isinstance(frame.blob.data, bytes) or not frame.blob.data:
      raise ValueError(f'Frame {index} must contain non-empty byte data.')

  manifest = build_manifest(frames, custom_metadata)

  buffer = io.BytesIO()
  # ZIP_STORED: the frames are already compressed image data, so deflating
  # them costs CPU for no size win, and storing them keeps each member
  # individually readable from the central directory.
  with zipfile.ZipFile(
      buffer, mode='w', compression=zipfile.ZIP_STORED
  ) as archive:
    for index, frame in enumerate(frames):
      archive.writestr(
          _frame_member_name(index, frame.blob.mime_type),
          frame.blob.data or b'',
      )
    archive.writestr(METADATA_FILENAME, json.dumps(manifest, indent=2))

  archive_bytes = buffer.getvalue()
  logger.debug(
      'Packed %d media frames into a %d byte archive',
      len(frames),
      len(archive_bytes),
  )
  return archive_bytes, manifest


def unpack_media_frames(
    archive_bytes: bytes,
) -> tuple[list[MediaFrame], dict[str, Any]]:
  """Restores the frames and manifest from a packed archive.

  The inverse of :func:`pack_media_frames`: every frame comes back with the
  bytes and MIME type it went in with. Capture times are stored as
  whole-millisecond offsets from the first frame, so they come back rounded
  to the nearest millisecond rather than bit-for-bit.

  Args:
    archive_bytes: The archive body, as returned by ``load_artifact``.

  Returns:
    A tuple of the frames in capture order and the manifest.

  Raises:
    ValueError: If the archive is not a valid ZIP, has a corrupt member CRC,
      has no manifest, or names a frame member that is not present.
  """
  try:
    with zipfile.ZipFile(io.BytesIO(archive_bytes), mode='r') as archive:
      member_names = set(archive.namelist())
      manifest = _read_archive_manifest(archive, member_names)
      start_timestamp = manifest.get('startTimestampMs', 0) / 1000.0
      frames = [
          _read_frame_entry(archive, member_names, entry, start_timestamp)
          for entry in manifest.get('frames', [])
      ]
  except (
      zipfile.BadZipFile,
      json.JSONDecodeError,
      UnicodeDecodeError,
  ) as exc:
    raise ValueError('Media frame archive is not a valid ZIP.') from exc

  return frames, manifest


def read_manifest(archive_bytes: bytes) -> dict[str, Any]:
  """Reads only the manifest out of a packed archive.

  Args:
    archive_bytes: The archive body, as returned by ``load_artifact``.

  Returns:
    The manifest dictionary.

  Raises:
    ValueError: If the archive is not a valid ZIP, has a corrupt member CRC,
      or has no manifest.
  """
  try:
    with zipfile.ZipFile(io.BytesIO(archive_bytes), mode='r') as archive:
      return _read_archive_manifest(archive, set(archive.namelist()))
  except (
      zipfile.BadZipFile,
      json.JSONDecodeError,
      UnicodeDecodeError,
  ) as exc:
    raise ValueError('Media frame archive is not a valid ZIP.') from exc


def extract_frame(archive_bytes: bytes, index: int) -> MediaFrame:
  """Reads a single frame out of a packed archive.

  Only the requested member is read; the rest of the archive is located
  through the ZIP central directory and never decoded.

  Args:
    archive_bytes: The archive body, as returned by ``load_artifact``.
    index: Zero-based position of the frame in capture order.

  Returns:
    The requested frame.

  Raises:
    ValueError: If the archive is malformed, has no manifest, or has no frame
      at ``index``.
  """
  try:
    with zipfile.ZipFile(io.BytesIO(archive_bytes), mode='r') as archive:
      member_names = set(archive.namelist())
      manifest = _read_archive_manifest(archive, member_names)
      entries = manifest.get('frames', [])
      if index < 0 or index >= len(entries):
        raise ValueError(f'Media frame archive has no frame at index {index}.')
      start_timestamp = manifest.get('startTimestampMs', 0) / 1000.0
      return _read_frame_entry(
          archive, member_names, entries[index], start_timestamp
      )
  except (
      zipfile.BadZipFile,
      json.JSONDecodeError,
      UnicodeDecodeError,
  ) as exc:
    raise ValueError('Media frame archive is not a valid ZIP.') from exc


def extract_preview_frame(archive_bytes: bytes) -> MediaFrame:
  """Reads the first frame out of a packed archive.

  Args:
    archive_bytes: The archive body, as returned by ``load_artifact``.

  Returns:
    The first frame in capture order.

  Raises:
    ValueError: If the archive is malformed or holds no frames.
  """
  return extract_frame(archive_bytes, 0)
