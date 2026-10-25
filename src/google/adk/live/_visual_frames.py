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

"""Packs and unpacks live visual frame sequences as a single ZIP artifact.

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
from pydantic import Field

logger = logging.getLogger('google_adk.' + __name__)

_FORMAT_VERSION = 1
"""Archive format version, written to and required in ``metadata.json``."""

_DEFAULT_MIME_TYPE = 'image/jpeg'
"""MIME type assumed when a frame blob does not declare one."""

_METADATA_FILENAME = 'metadata.json'
"""Name of the manifest member at the root of the archive."""

_FRAMES_DIRECTORY = 'frames'
"""Name of the archive directory holding the frame members."""

_UNREADABLE_FLAG_BITS = 0x1 | 0x20 | 0x40
"""General-purpose flag bits that make `zipfile` refuse to read a member.

Bit 0 marks an encrypted member, which needs a password (`RuntimeError`); bits
5 and 6 mark compressed patched data and strong encryption, which `zipfile`
does not support (`NotImplementedError`).
"""

VISUAL_FRAMES_ZIP_MIME_TYPE = 'application/zip'
"""MIME type of the packed archive, as stored in the artifact service."""

# A private `MimeTypes` instance uses only the stdlib's built-in table, so the
# extensions don't depend on the host's mime.types files or on other code
# calling `mimetypes.add_type()`.
_MIME_TYPES = mimetypes.MimeTypes()


class VisualFrame(BaseModel):
  """A single visual frame and the time it was received."""

  model_config = ConfigDict(extra='forbid')
  """The pydantic model config."""

  blob: types.Blob
  """The frame payload: its bytes and its MIME type."""

  timestamp: float = Field(allow_inf_nan=False)
  """Wall-clock time the frame was received, in seconds since the Unix epoch."""


def _extension_for_mime_type(mime_type: str | None) -> str:
  """Returns the archive file extension to use for a frame MIME type."""
  raw_mime = (mime_type or _DEFAULT_MIME_TYPE).split(';')[0].strip().lower()
  guessed = _MIME_TYPES.guess_extension(raw_mime, strict=False)
  extension = guessed.lstrip('.') if guessed else ''
  return 'jpeg' if not extension or extension in ('jpg', 'jpe') else extension


def _frame_member_name(index: int, mime_type: str | None) -> str:
  """Returns the archive ``member_name`` for the frame at ``index``."""
  extension = _extension_for_mime_type(mime_type)
  return f'{_FRAMES_DIRECTORY}/frame_{index:04d}.{extension}'


def _check_members_are_stored(archive: zipfile.ZipFile) -> None:
  """Raises ``ValueError`` if any archive member is compressed or encrypted."""
  # Archives are written uncompressed and unencrypted. Refusing anything else
  # up front leaves `BadZipFile` as the only error a member read can raise,
  # and keeps a crafted archive from decompressing into far more memory than
  # it occupies.
  for info in archive.infolist():
    if info.compress_type != zipfile.ZIP_STORED or (
        info.flag_bits & _UNREADABLE_FLAG_BITS
    ):
      raise ValueError(
          f'Visual frame archive member {info.filename!r} must be stored'
          ' uncompressed and unencrypted.'
      )


def _read_archive_manifest(
    archive: zipfile.ZipFile,
    member_names: set[str],
) -> dict[str, Any]:
  """Reads and validates ``metadata.json`` from an open archive."""
  if _METADATA_FILENAME not in member_names:
    raise ValueError(f'Visual frame archive is missing {_METADATA_FILENAME}.')

  try:
    manifest = json.loads(archive.read(_METADATA_FILENAME).decode('utf-8'))
  except (json.JSONDecodeError, UnicodeDecodeError) as exc:
    raise ValueError(f'{_METADATA_FILENAME} is not valid JSON.') from exc
  if not isinstance(manifest, dict):
    raise ValueError(f'{_METADATA_FILENAME} must contain a JSON object.')
  format_version = manifest.get('formatVersion')
  if (
      isinstance(format_version, bool)
      or not isinstance(format_version, int)
      or format_version != _FORMAT_VERSION
  ):
    raise ValueError(
        f'Unsupported visual frame archive format version {format_version!r};'
        f' this reader supports {_FORMAT_VERSION}.'
    )
  entries = manifest.get('frames', [])
  if not isinstance(entries, list):
    raise ValueError(f'{_METADATA_FILENAME} frames field must be a list.')
  start_ts_ms = manifest.get('startTimestampMs', 0)
  if (
      isinstance(start_ts_ms, bool)
      or not isinstance(start_ts_ms, (int, float))
      or not math.isfinite(start_ts_ms)
  ):
    raise ValueError(
        f'{_METADATA_FILENAME} startTimestampMs must be a finite number.'
    )
  return manifest


def _read_frame_entry(
    archive: zipfile.ZipFile,
    member_names: set[str],
    entry: Any,
    start_timestamp_ms: float,
    read_members: set[str],
) -> VisualFrame:
  """Validates a manifest frame entry and builds its :class:`VisualFrame`.

  Records the entry's member in ``read_members`` and refuses one already there.
  """
  if not isinstance(entry, dict):
    raise ValueError(f'{_METADATA_FILENAME} frame entries must be objects.')
  member_name = entry.get('fileName')
  if (
      not isinstance(member_name, str)
      or not member_name
      or member_name not in member_names
  ):
    raise ValueError(
        f'Visual frame archive is missing frame member {member_name!r}.'
    )
  # Reading a member once per listing would let a small archive return one
  # large frame many times over.
  if member_name in read_members:
    raise ValueError(
        f'{_METADATA_FILENAME} lists frame member {member_name!r} more than'
        ' once.'
    )
  read_members.add(member_name)
  offset_ms = entry.get('offsetMs', 0)
  if (
      isinstance(offset_ms, bool)
      or not isinstance(offset_ms, (int, float))
      or not math.isfinite(offset_ms)
  ):
    raise ValueError(
        f'{_METADATA_FILENAME} frame offsetMs must be a finite number.'
    )
  mime_type = entry.get('mimeType', _DEFAULT_MIME_TYPE)
  if not isinstance(mime_type, str) or not mime_type:
    mime_type = _DEFAULT_MIME_TYPE
  return VisualFrame(
      blob=types.Blob(
          data=archive.read(member_name),
          mime_type=mime_type,
      ),
      timestamp=(start_timestamp_ms + offset_ms) / 1000.0,
  )


def _build_manifest(
    frames: list[VisualFrame],
    payloads: list[bytes],
) -> dict[str, Any]:
  """Builds the manifest written into the archive as ``metadata.json``.

  Args:
    frames: The checked frames, in the order they were received. Must not be
      empty.
    payloads: Each frame's checked bytes, in the same order as ``frames``.

  Returns:
    The manifest dictionary.
  """
  # Each time is rounded to a whole millisecond once and every field is
  # derived from those integers, so the fields always agree with each other
  # and unpacking restores each time to the nearest millisecond. Rounded, not
  # truncated: converting seconds to milliseconds lands on values such as
  # 599.9999999999999 for a time that is exactly 0.6s.
  timestamps_ms = [round(frame.timestamp * 1000) for frame in frames]
  start_ms = timestamps_ms[0]
  end_ms = timestamps_ms[-1]
  duration_ms = end_ms - start_ms
  frame_count = len(frames)
  # Rate over the gaps between frames, not the frames themselves, so frames
  # that all share one millisecond report 0.0 rather than dividing by zero.
  estimated_fps = (
      round((frame_count - 1) * 1000 / duration_ms, 2)
      if duration_ms > 0
      else 0.0
  )

  frame_entries: list[dict[str, Any]] = []
  for index, (frame, data, timestamp_ms) in enumerate(
      zip(frames, payloads, timestamps_ms)
  ):
    mime_type = (frame.blob.mime_type or _DEFAULT_MIME_TYPE).lower()
    frame_entries.append({
        'frameIndex': index,
        'offsetMs': timestamp_ms - start_ms,
        'fileName': _frame_member_name(index, mime_type),
        'mimeType': mime_type,
        'sizeBytes': len(data),
    })

  return {
      'formatVersion': _FORMAT_VERSION,
      'type': 'video_frame_sequence',
      'frameCount': frame_count,
      'startTimestampMs': start_ms,
      'endTimestampMs': end_ms,
      'durationMs': duration_ms,
      'estimatedFps': estimated_fps,
      'frames': frame_entries,
  }


def pack_visual_frames(
    frames: list[VisualFrame],
) -> tuple[bytes, dict[str, Any]]:
  """Packs frames and their manifest into one uncompressed ZIP archive.

  Args:
    frames: The frames, in the order they were received. Must not be empty.

  Returns:
    A tuple of the archive bytes and the metadata to store with the archive's
    artifact as its ``custom_metadata``. The metadata is the archive's
    manifest without the per-frame index, so its size does not grow with the
    frame count; several artifact backends keep ``custom_metadata`` in object
    metadata with a size limit.

  Raises:
    ValueError: If ``frames`` is empty, a frame carries no bytes, or the
      timestamps decrease.
  """
  if not frames:
    raise ValueError('Cannot pack an empty frame list.')

  payloads: list[bytes] = []
  previous_timestamp = frames[0].timestamp
  for index, frame in enumerate(frames):
    if not isinstance(frame.blob.data, bytes) or not frame.blob.data:
      raise ValueError(f'Frame {index} must contain non-empty byte data.')
    if frame.timestamp < previous_timestamp:
      raise ValueError(
          f'Frame timestamps must be non-decreasing (frame {index} at'
          f' {frame.timestamp} < {previous_timestamp}).'
      )
    previous_timestamp = frame.timestamp
    payloads.append(frame.blob.data)

  manifest = _build_manifest(frames, payloads)

  buffer = io.BytesIO()
  # ZIP_STORED: the frames are already compressed image data, so deflating
  # them costs CPU for no size win.
  with zipfile.ZipFile(
      buffer, mode='w', compression=zipfile.ZIP_STORED
  ) as archive:
    for entry, data in zip(manifest['frames'], payloads):
      archive.writestr(entry['fileName'], data)
    archive.writestr(_METADATA_FILENAME, json.dumps(manifest, indent=2))

  archive_bytes = buffer.getvalue()
  logger.debug(
      'Packed %d visual frames into a %d byte archive',
      len(frames),
      len(archive_bytes),
  )
  artifact_metadata = {
      key: value for key, value in manifest.items() if key != 'frames'
  }
  return archive_bytes, artifact_metadata


def unpack_visual_frames(
    archive_bytes: bytes,
) -> tuple[list[VisualFrame], dict[str, Any]]:
  """Restores the frames and manifest from a packed archive.

  The inverse of :func:`pack_visual_frames`: every frame comes back with the
  bytes and MIME type it went in with. Times are stored as whole
  milliseconds, so each one comes back as the nearest millisecond rather than
  bit-for-bit.

  Args:
    archive_bytes: The archive body, as returned by ``load_artifact``.

  Returns:
    A tuple of the frames, in the order they were received, and the manifest
    read from the archive's ``metadata.json``.

  Raises:
    ValueError: If the archive is not a valid ZIP or has a compressed,
      encrypted or corrupt member, or if ``metadata.json`` is missing, is not
      valid JSON, has an unsupported ``formatVersion`` or malformed fields, or
      lists a frame member that is missing or listed more than once.
  """
  try:
    with zipfile.ZipFile(io.BytesIO(archive_bytes), mode='r') as archive:
      _check_members_are_stored(archive)
      member_names = set(archive.namelist())
      manifest = _read_archive_manifest(archive, member_names)
      start_timestamp_ms = manifest.get('startTimestampMs', 0)
      read_members: set[str] = set()
      frames = [
          _read_frame_entry(
              archive, member_names, entry, start_timestamp_ms, read_members
          )
          for entry in manifest.get('frames', [])
      ]
  except (zipfile.BadZipFile, NotImplementedError) as exc:
    # Opening the archive raises NotImplementedError if a member needs a newer
    # ZIP version than `zipfile` supports, before any member can be checked.
    raise ValueError('Visual frame archive is not a valid ZIP.') from exc

  return frames, manifest
