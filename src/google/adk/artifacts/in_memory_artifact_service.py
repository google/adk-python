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

import dataclasses
import logging
from typing import Any
from typing import cast
from typing import Optional
from typing import Union

from google.genai import types
from pydantic import BaseModel
from pydantic import Field
from typing_extensions import override

from . import artifact_util
from ..errors.input_validation_error import InputValidationError
from .base_artifact_service import ArtifactVersion
from .base_artifact_service import BaseArtifactService
from .base_artifact_service import ensure_part

logger = logging.getLogger("google_adk." + __name__)


@dataclasses.dataclass
class _ArtifactEntry:
  """Represents a single version of an artifact stored in memory.

  Attributes:
    data: The actual data of the artifact.
    artifact_version: Metadata about this specific version of the artifact.
  """

  data: types.Part
  artifact_version: ArtifactVersion


class InMemoryArtifactService(BaseArtifactService, BaseModel):
  """An in-memory implementation of the artifact service.

  It is not suitable for multi-threaded production environments. Use it for
  testing and development only.
  """

  artifacts: dict[str, list[_ArtifactEntry]] = Field(default_factory=dict)

  def _file_has_user_namespace(self, filename: str) -> bool:
    """Checks if the filename has a user namespace.

    Args:
        filename: The filename to check.

    Returns:
        True if the filename has a user namespace (starts with "user:"),
        False otherwise.
    """
    return filename.startswith("user:")

  def _artifact_path(
      self,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str],
  ) -> str:
    """Constructs the artifact path.

    Args:
        app_name: The name of the application.
        user_id: The ID of the user.
        filename: The name of the artifact file.
        session_id: The ID of the session.

    Returns:
        The constructed artifact path.
    """
    artifact_util.validate_path_segment(app_name, "app_name")
    artifact_util.validate_path_segment(user_id, "user_id")
    if self._file_has_user_namespace(filename):
      return f"{app_name}/{user_id}/user/{filename}"

    if session_id is None:
      raise InputValidationError(
          "Session ID must be provided for session-scoped artifacts."
      )
    artifact_util.validate_path_segment(session_id, "session_id")
    return f"{app_name}/{user_id}/{session_id}/{filename}"

  @override
  async def save_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      artifact: Union[types.Part, dict[str, Any]],
      session_id: Optional[str] = None,
      custom_metadata: Optional[dict[str, Any]] = None,
  ) -> int:
    artifact = ensure_part(artifact)
    path = self._artifact_path(app_name, user_id, filename, session_id)
    if path not in self.artifacts:
      self.artifacts[path] = []
    version = len(self.artifacts[path])
    if self._file_has_user_namespace(filename):
      canonical_uri = f"memory://apps/{app_name}/users/{user_id}/artifacts/{filename}/versions/{version}"
    else:
      canonical_uri = f"memory://apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{filename}/versions/{version}"

    artifact_version = ArtifactVersion(
        version=version,
        canonical_uri=canonical_uri,
    )
    if custom_metadata:
      artifact_version.custom_metadata = custom_metadata

    if artifact.inline_data is not None:
      artifact_version.mime_type = artifact.inline_data.mime_type
    elif artifact.text is not None:
      artifact_version.mime_type = "text/plain"
    elif (file_data := artifact.file_data) is not None:
      if artifact_util.is_artifact_ref(artifact):
        parsed_uri = artifact_util.parse_artifact_uri(
            cast(str, file_data.file_uri)
        )
        if not parsed_uri:
          raise InputValidationError(
              f"Invalid artifact reference URI: {file_data.file_uri}"
          )
        artifact_util.validate_artifact_reference_scope(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            parsed_uri=parsed_uri,
        )
        # If it's a valid artifact URI, we store the artifact part as-is.
        # And we don't know the mime type until we load it.
      else:
        artifact_version.mime_type = file_data.mime_type
    else:
      raise InputValidationError("Not supported artifact type.")

    self.artifacts[path].append(
        _ArtifactEntry(data=artifact, artifact_version=artifact_version)
    )
    return version

  @override
  async def save_media_frames(
      self,
      *,
      app_name: str,
      user_id: str,
      collection_name: str,
      frames: list[tuple[types.Blob, float]],
      session_id: Optional[str] = None,
      custom_metadata: Optional[dict[str, Any]] = None,
  ) -> int:
    if not frames:
      raise InputValidationError("Cannot save empty frames list.")

    path = self._artifact_path(app_name, user_id, collection_name, session_id)
    if path not in self.artifacts:
      self.artifacts[path] = []
    version = len(self.artifacts[path])

    if self._file_has_user_namespace(collection_name):
      canonical_uri = f"memory://apps/{app_name}/users/{user_id}/artifacts/{collection_name}/versions/{version}"
    else:
      canonical_uri = f"memory://apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{collection_name}/versions/{version}"

    start_ts = frames[0][1]
    end_ts = frames[-1][1]
    duration_ms = int((end_ts - start_ts) * 1000)
    frame_count = len(frames)
    estimated_fps = (
        round((frame_count - 1) / (end_ts - start_ts), 2)
        if duration_ms > 0 and frame_count > 1
        else 0.0
    )

    frame_indices = []
    for idx, (blob, ts) in enumerate(frames):
      mime = (blob.mime_type or "image/jpeg").lower()
      ext = mime.split("/")[-1].split(";")[0].strip() or "jpeg"
      offset_ms = int((ts - start_ts) * 1000)
      frame_indices.append({
          "frameIndex": idx,
          "offsetMs": offset_ms,
          "fileName": f"frames/frame_{idx:04d}.{ext}",
          "mimeType": mime,
          "sizeBytes": len(blob.data or b""),
      })

    merged_custom_metadata = dict(custom_metadata or {})
    merged_custom_metadata.update({
        "type": "video_frame_sequence",
        "frameCount": frame_count,
        "startTimestampMs": int(start_ts * 1000),
        "endTimestampMs": int(end_ts * 1000),
        "durationMs": duration_ms,
        "estimatedFps": estimated_fps,
        "frames": frame_indices,
    })

    primary_mime_type = frames[0][0].mime_type or "image/jpeg"
    artifact_version = ArtifactVersion(
        version=version,
        canonical_uri=canonical_uri,
        custom_metadata=merged_custom_metadata,
        mime_type=primary_mime_type,
    )

    preview_part = types.Part(
        inline_data=types.Blob(
            data=frames[0][0].data,
            mime_type=primary_mime_type,
        )
    )

    self.artifacts[path].append(
        _ArtifactEntry(data=preview_part, artifact_version=artifact_version)
    )
    return version

  @override
  async def load_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
      version: Optional[int] = None,
  ) -> Optional[types.Part]:
    path = self._artifact_path(app_name, user_id, filename, session_id)
    versions = self.artifacts.get(path)
    if not versions:
      return None
    if version is None:
      version = -1

    try:
      artifact_entry = versions[version]
    except IndexError:
      return None

    if artifact_entry is None:
      return None

    # Resolve artifact reference if needed.
    artifact_data = artifact_entry.data
    if artifact_util.is_artifact_ref(artifact_data):
      file_data = artifact_data.file_data
      assert file_data is not None
      parsed_uri = artifact_util.parse_artifact_uri(
          cast(str, file_data.file_uri)
      )
      if not parsed_uri:
        raise InputValidationError(
            f"Invalid artifact reference URI: {file_data.file_uri}"
        )
      artifact_util.validate_artifact_reference_scope(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          parsed_uri=parsed_uri,
      )
      return await self.load_artifact(
          app_name=parsed_uri.app_name,
          user_id=parsed_uri.user_id,
          filename=parsed_uri.filename,
          session_id=parsed_uri.session_id,
          version=parsed_uri.version,
      )

    if (
        artifact_data == types.Part()
        or artifact_data == types.Part(text="")
        or (artifact_data.inline_data and not artifact_data.inline_data.data)
    ):
      return None
    return artifact_data

  @override
  async def list_artifact_keys(
      self, *, app_name: str, user_id: str, session_id: Optional[str] = None
  ) -> list[str]:
    artifact_util.validate_path_segment(app_name, "app_name")
    artifact_util.validate_path_segment(user_id, "user_id")
    if session_id is not None:
      artifact_util.validate_path_segment(session_id, "session_id")
    usernamespace_prefix = f"{app_name}/{user_id}/user/"
    session_prefix = (
        f"{app_name}/{user_id}/{session_id}/" if session_id else None
    )
    filenames = []
    for path in self.artifacts:
      if session_prefix and path.startswith(session_prefix):
        filename = path.removeprefix(session_prefix)
        filenames.append(filename)
      elif path.startswith(usernamespace_prefix):
        filename = path.removeprefix(usernamespace_prefix)
        filenames.append(filename)
    return sorted(filenames)

  @override
  async def delete_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
  ) -> None:
    path = self._artifact_path(app_name, user_id, filename, session_id)
    if not self.artifacts.get(path):
      return None
    self.artifacts.pop(path, None)

  @override
  async def list_versions(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
  ) -> list[int]:
    path = self._artifact_path(app_name, user_id, filename, session_id)
    versions = self.artifacts.get(path)
    if not versions:
      return []
    return list(range(len(versions)))

  @override
  async def list_artifact_versions(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
  ) -> list[ArtifactVersion]:
    path = self._artifact_path(app_name, user_id, filename, session_id)
    entries = self.artifacts.get(path)
    if not entries:
      return []
    return [entry.artifact_version for entry in entries]

  @override
  async def get_artifact_version(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
      version: Optional[int] = None,
  ) -> Optional[ArtifactVersion]:
    path = self._artifact_path(app_name, user_id, filename, session_id)
    entries = self.artifacts.get(path)
    if not entries:
      return None

    if version is None:
      version = -1
    try:
      return entries[version].artifact_version
    except IndexError:
      return None
