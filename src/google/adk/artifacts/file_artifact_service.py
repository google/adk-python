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

import asyncio
import logging
import os
from pathlib import Path
from pathlib import PurePosixPath
from pathlib import PureWindowsPath
import shutil
from typing import Any
from typing import Optional
from urllib.parse import unquote
from urllib.parse import urlparse

from google.genai import types
from pydantic import alias_generators
from pydantic import ConfigDict
from pydantic import Field
from pydantic import ValidationError
from typing_extensions import override

from ..errors.input_validation_error import InputValidationError
from .base_artifact_service import ArtifactVersion
from .base_artifact_service import BaseArtifactService

logger = logging.getLogger("google_adk." + __name__)


def _iter_artifact_dirs(root: Path) -> list[Path]:
  """Returns artifact directory paths beneath a root."""
  if not root.exists():
    return []
  artifact_dirs: list[Path] = []
  for dirpath, dirnames, _ in os.walk(root):
    current = Path(dirpath)
    if (current / "versions").exists():
      artifact_dirs.append(current)
      dirnames.clear()
  return artifact_dirs


def _file_uri_to_path(uri: str) -> Optional[Path]:
  """Converts a file:// URI to a filesystem path."""
  parsed = urlparse(uri)
  if parsed.scheme != "file":
    return None
  return Path(unquote(parsed.path))


_USER_NAMESPACE_PREFIX = "user:"


def _file_has_user_namespace(filename: str) -> bool:
  """Checks whether the file is scoped to the user namespace."""
  return filename.startswith(_USER_NAMESPACE_PREFIX)


def _strip_user_namespace(filename: str) -> str:
  """Removes the `user:` namespace prefix when present."""
  if _file_has_user_namespace(filename):
    return filename[len(_USER_NAMESPACE_PREFIX) :]
  return filename


def _to_posix_path(path_value: str) -> PurePosixPath:
  """Normalizes separators by converting to a `PurePosixPath`."""
  if "\\" in path_value:
    # Interpret Windows-style paths while still running on POSIX systems.
    path_value = PureWindowsPath(path_value).as_posix()
  return PurePosixPath(path_value)


def _resolve_scoped_artifact_path(
    scope_root: Path, filename: str
) -> tuple[Path, Path]:
  """Returns the absolute artifact directory and its relative path.

  The caller is expected to pass the scope root directory (user or session).
  This helper joins the filename under that root, resolves traversal segments,
  and guards against paths that escape the scope root.

  Args:
    scope_root: Directory that defines the storage scope.
    filename: Caller-supplied artifact name.

  Returns:
    A tuple containing the absolute artifact directory and its path relative
    to `scope_root`.

  Raises:
    InputValidationError: If `filename` resolves outside of `scope_root`.
  """
  stripped = _strip_user_namespace(filename).strip()
  pure_path = _to_posix_path(stripped)

  scope_root_resolved = scope_root.resolve(strict=False)
  if pure_path.is_absolute():
    raise InputValidationError(
        f"Absolute artifact filename {filename!r} is not permitted; "
        "provide a path relative to the storage scope."
    )
  candidate = scope_root_resolved / Path(pure_path)

  candidate = candidate.resolve(strict=False)

  try:
    relative = candidate.relative_to(scope_root_resolved)
  except ValueError as exc:
    raise InputValidationError(
        f"Artifact filename {filename!r} escapes storage directory "
        f"{scope_root_resolved}"
    ) from exc

  if relative == Path("."):
    relative = Path("artifact")
    candidate = scope_root_resolved / relative

  return candidate, relative


def _is_user_scoped(session_id: Optional[str], filename: str) -> bool:
  """Determines whether artifacts should be stored in the user namespace."""
  return session_id is None or _file_has_user_namespace(filename)


def _user_artifacts_dir(base_root: Path) -> Path:
  """Returns the path that stores user-scoped artifacts."""
  return base_root / "artifacts"


def _session_artifacts_dir(base_root: Path, session_id: str) -> Path:
  """Returns the path that stores session-scoped artifacts."""
  return base_root / "sessions" / session_id / "artifacts"


def _versions_dir(artifact_dir: Path) -> Path:
  """Returns the directory that contains versioned payloads."""
  return artifact_dir / "versions"


def _metadata_path(artifact_dir: Path, version: int) -> Path:
  """Returns the path to the metadata file for a specific version."""
  return _versions_dir(artifact_dir) / str(version) / "metadata.json"


def _list_versions_on_disk(artifact_dir: Path) -> list[int]:
  """Returns sorted versions discovered under the artifact directory."""
  versions_dir = _versions_dir(artifact_dir)
  if not versions_dir.exists():
    return []
  versions: list[int] = []
  for child in versions_dir.iterdir():
    if child.is_dir():
      try:
        versions.append(int(child.name))
      except ValueError:
        logger.debug("Skipping non-version directory %s", child)
  return sorted(versions)


class FileArtifactVersion(ArtifactVersion):
  """Represents persisted metadata for a file-backed artifact."""

  model_config = ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )

  file_name: str = Field(
      description="Original filename supplied by the caller."
  )


class FileArtifactService(BaseArtifactService):
  """Stores filesystem-backed artifacts beneath a configurable root directory."""

  # Storage layout matches the cloud and in-memory services:
  # root/
  # └── users/
  #     └── {user_id}/
  #         ├── sessions/
  #         │   └── {session_id}/
  #         │       └── artifacts/
  #         │           └── {artifact_path}/  # derived from filename
  #         │               └── versions/
  #         │                   └── {version}/
  #         │                       ├── {original_filename}
  #         │                       └── metadata.json
  #         └── artifacts/
  #             └── {artifact_path}/...
  #
  # Artifact paths are derived from the provided filenames: separators create
  # nested directories, and path traversal is rejected to keep the layout
  # portable across filesystems. `{artifact_path}` therefore mirrors the
  # sanitized, scope-relative path derived from each filename.

  def __init__(self, root_dir: Path | str):
    """Initializes the file-based artifact service.

    Args:
      root_dir: The directory that will contain artifact data.
    """
    self.root_dir = Path(root_dir).expanduser().resolve()
    self.root_dir.mkdir(parents=True, exist_ok=True)

  def _base_root(self, user_id: str, /) -> Path:
    """Returns the artifacts root directory for a user."""
    return self.root_dir / "users" / user_id

  def _scope_root(
      self,
      user_id: str,
      session_id: Optional[str],
      filename: str,
  ) -> Path:
    """Returns the directory that represents the artifact scope."""
    base = self._base_root(user_id)
    if _is_user_scoped(session_id, filename):
      return _user_artifacts_dir(base)
    if not session_id:
      raise InputValidationError(
          "Session ID must be provided for session-scoped artifacts."
      )
    return _session_artifacts_dir(base, session_id)

  def _artifact_dir(
      self,
      user_id: str,
      session_id: Optional[str],
      filename: str,
  ) -> Path:
    """Builds the directory path for an artifact."""
    scope_root = self._scope_root(
        user_id=user_id,
        session_id=session_id,
        filename=filename,
    )
    artifact_dir, _ = _resolve_scoped_artifact_path(scope_root, filename)
    return artifact_dir

  def _build_artifact_version(
      self,
      *,
      user_id: str,
      session_id: Optional[str],
      filename: str,
      version: int,
      metadata: Optional[FileArtifactVersion],
  ) -> ArtifactVersion:
    """Creates an ArtifactVersion payload using on-disk metadata."""
    canonical_uri = (
        metadata.canonical_uri
        if metadata and metadata.canonical_uri
        else self._canonical_uri(
            user_id=user_id,
            session_id=session_id,
            filename=filename,
            version=version,
        )
    )
    custom_metadata_val = metadata.custom_metadata if metadata else {}
    mime_type = metadata.mime_type if metadata else None
    return ArtifactVersion(
        version=version,
        canonical_uri=canonical_uri,
        custom_metadata=dict(custom_metadata_val),
        mime_type=mime_type,
    )

  def _canonical_uri(
      self,
      *,
      user_id: str,
      session_id: Optional[str],
      filename: str,
      version: int,
  ) -> str:
    """Builds the canonical file:// URI for an artifact payload."""
    artifact_dir = self._artifact_dir(
        user_id=user_id,
        session_id=session_id,
        filename=filename,
    )
    stored_filename = artifact_dir.name
    payload_path = _versions_dir(artifact_dir) / str(version) / stored_filename
    return payload_path.resolve().as_uri()

  def _latest_metadata(
      self, artifact_dir: Path
  ) -> Optional[FileArtifactVersion]:
    """Loads metadata for the most recent version."""
    versions = _list_versions_on_disk(artifact_dir)
    if not versions:
      return None
    return _read_metadata(_metadata_path(artifact_dir, versions[-1]))

  @override
  async def save_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      artifact: types.Part,
      session_id: Optional[str] = None,
      custom_metadata: Optional[dict[str, Any]] = None,
  ) -> int:
    """Persists an artifact to disk.

    Filenames may be simple (``"report.txt"``), nested
    (``"images/photo.png"``), or explicitly user-scoped
    (``"user:shared/diagram.png"``). All values are interpreted relative to the
    computed scope root; absolute paths or inputs that traverse outside that
    root (for example ``"../../secret.txt"``) raise ``ValueError``.
    """
    return await asyncio.to_thread(
        self._save_artifact_sync,
        user_id,
        filename,
        artifact,
        session_id,
        custom_metadata,
    )

  def _save_artifact_sync(
      self,
      user_id: str,
      filename: str,
      artifact: types.Part,
      session_id: Optional[str],
      custom_metadata: Optional[dict[str, Any]],
  ) -> int:
    """Saves an artifact to disk and returns its version."""
    artifact_dir = self._artifact_dir(
        user_id=user_id,
        session_id=session_id,
        filename=filename,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    versions = _list_versions_on_disk(artifact_dir)
    next_version = 0 if not versions else versions[-1] + 1
    versions_dir = _versions_dir(artifact_dir)
    versions_dir.mkdir(parents=True, exist_ok=True)
    version_dir = versions_dir / str(next_version)
    version_dir.mkdir()

    stored_filename = artifact_dir.name
    content_path = version_dir / stored_filename

    if artifact.inline_data:
      content_path.write_bytes(artifact.inline_data.data)
      mime_type = (
          artifact.inline_data.mime_type
          if artifact.inline_data.mime_type
          else "application/octet-stream"
      )
    elif artifact.text is not None:
      content_path.write_text(artifact.text, encoding="utf-8")
      mime_type = None
    else:
      raise InputValidationError(
          "Artifact must have either inline_data or text content."
      )

    canonical_uri = self._canonical_uri(
        user_id=user_id,
        session_id=session_id,
        filename=filename,
        version=next_version,
    )
    _write_metadata(
        version_dir / "metadata.json",
        filename=filename,
        mime_type=mime_type,
        version=next_version,
        canonical_uri=canonical_uri,
        custom_metadata=custom_metadata,
    )

    logger.debug(
        "Saved artifact %s version %d to %s",
        filename,
        next_version,
        version_dir,
    )
    return next_version

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
    return await asyncio.to_thread(
        self._load_artifact_sync,
        user_id,
        filename,
        session_id,
        version,
    )

  def _load_artifact_sync(
      self,
      user_id: str,
      filename: str,
      session_id: Optional[str],
      version: Optional[int],
  ) -> Optional[types.Part]:
    """Loads an artifact from disk."""
    artifact_dir = self._artifact_dir(
        user_id=user_id,
        session_id=session_id,
        filename=filename,
    )
    if not artifact_dir.exists():
      return None

    versions = _list_versions_on_disk(artifact_dir)
    if not versions:
      return None

    if version is None:
      version_to_load = versions[-1]
    else:
      if version not in versions:
        return None
      version_to_load = version

    version_dir = _versions_dir(artifact_dir) / str(version_to_load)
    metadata = _read_metadata(_metadata_path(artifact_dir, version_to_load))
    mime_type = metadata.mime_type if metadata else None
    stored_filename = artifact_dir.name
    content_path = version_dir / stored_filename
    if metadata and metadata.canonical_uri and not content_path.exists():
      uri_path = _file_uri_to_path(metadata.canonical_uri)
      if uri_path and uri_path.exists():
        content_path = uri_path

    if mime_type:
      if not content_path.exists():
        logger.warning(
            "Binary artifact %s missing at %s", filename, content_path
        )
        return None
      data = content_path.read_bytes()
      return types.Part(inline_data=types.Blob(mime_type=mime_type, data=data))

    if not content_path.exists():
      logger.warning("Text artifact %s missing at %s", filename, content_path)
      return None

    text = content_path.read_text(encoding="utf-8")
    return types.Part(text=text)

  @override
  async def list_artifact_keys(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: Optional[str] = None,
  ) -> list[str]:
    return await asyncio.to_thread(
        self._list_artifact_keys_sync,
        user_id,
        session_id,
    )

  def _list_artifact_keys_sync(
      self,
      user_id: str,
      session_id: Optional[str],
  ) -> list[str]:
    """Lists artifact filenames for the given session/user."""
    filenames: set[str] = set()

    base_root = self._base_root(user_id)

    if session_id:
      session_root = _session_artifacts_dir(base_root, session_id)
      for artifact_dir in _iter_artifact_dirs(session_root):
        metadata = self._latest_metadata(artifact_dir)
        if metadata and metadata.file_name:
          filenames.add(str(metadata.file_name))
        else:
          rel = artifact_dir.relative_to(session_root)
          filenames.add(rel.as_posix())

    user_root = _user_artifacts_dir(base_root)
    for artifact_dir in _iter_artifact_dirs(user_root):
      metadata = self._latest_metadata(artifact_dir)
      if metadata and metadata.file_name:
        filenames.add(str(metadata.file_name))
      else:
        rel = artifact_dir.relative_to(user_root)
        filenames.add(f"user:{rel.as_posix()}")

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
    """Deletes an artifact.

    Args:
        app_name: The name of the application.
        user_id: The ID of the user.
        filename: The name of the artifact file.
        session_id: The ID of the session. Leave unset for user-scoped
          artifacts.
    """
    await asyncio.to_thread(
        self._delete_artifact_sync,
        user_id,
        filename,
        session_id,
    )

  def _delete_artifact_sync(
      self,
      user_id: str,
      filename: str,
      session_id: Optional[str],
  ) -> None:
    artifact_dir = self._artifact_dir(
        user_id=user_id,
        session_id=session_id,
        filename=filename,
    )
    if artifact_dir.exists():
      shutil.rmtree(artifact_dir)
      logger.debug("Deleted artifact %s at %s", filename, artifact_dir)

  @override
  async def list_versions(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
  ) -> list[int]:
    """Lists all versions stored for an artifact."""
    return await asyncio.to_thread(
        self._list_versions_sync,
        user_id,
        filename,
        session_id,
    )

  def _list_versions_sync(
      self,
      user_id: str,
      filename: str,
      session_id: Optional[str],
  ) -> list[int]:
    artifact_dir = self._artifact_dir(
        user_id=user_id,
        session_id=session_id,
        filename=filename,
    )
    return _list_versions_on_disk(artifact_dir)

  @override
  async def list_artifact_versions(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
  ) -> list[ArtifactVersion]:
    """Lists metadata for each artifact version on disk."""
    return await asyncio.to_thread(
        self._list_artifact_versions_sync,
        user_id,
        filename,
        session_id,
    )

  def _list_artifact_versions_sync(
      self,
      user_id: str,
      filename: str,
      session_id: Optional[str],
  ) -> list[ArtifactVersion]:
    artifact_dir = self._artifact_dir(
        user_id=user_id,
        session_id=session_id,
        filename=filename,
    )
    versions = _list_versions_on_disk(artifact_dir)
    artifact_versions: list[ArtifactVersion] = []
    for version in versions:
      metadata_path = _metadata_path(artifact_dir, version)
      metadata = _read_metadata(metadata_path)
      artifact_versions.append(
          self._build_artifact_version(
              user_id=user_id,
              session_id=session_id,
              filename=filename,
              version=version,
              metadata=metadata,
          )
      )
    return artifact_versions

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
    """Gets metadata for a specific artifact version."""
    return await asyncio.to_thread(
        self._get_artifact_version_sync,
        user_id,
        filename,
        session_id,
        version,
    )

  def _get_artifact_version_sync(
      self,
      user_id: str,
      filename: str,
      session_id: Optional[str],
      version: Optional[int],
  ) -> Optional[ArtifactVersion]:
    artifact_dir = self._artifact_dir(
        user_id=user_id,
        session_id=session_id,
        filename=filename,
    )
    versions = _list_versions_on_disk(artifact_dir)
    if not versions:
      return None
    if version is None:
      version_to_read = versions[-1]
    else:
      if version not in versions:
        return None
      version_to_read = version

    metadata_path = _metadata_path(artifact_dir, version_to_read)
    metadata = _read_metadata(metadata_path)
    return self._build_artifact_version(
        user_id=user_id,
        session_id=session_id,
        filename=filename,
        version=version_to_read,
        metadata=metadata,
    )


def _write_metadata(
    path: Path,
    *,
    filename: str,
    mime_type: Optional[str],
    version: int,
    canonical_uri: str,
    custom_metadata: Optional[dict[str, Any]],
) -> None:
  """Persists metadata describing an artifact version."""
  metadata = FileArtifactVersion(
      file_name=filename,
      mime_type=mime_type,
      canonical_uri=canonical_uri,
      version=version,
      # Persist caller supplied metadata for feature parity with other
      # artifact services (e.g. GCS).
      custom_metadata=dict(custom_metadata or {}),
  )
  path.write_text(
      metadata.model_dump_json(by_alias=True, exclude_none=True),
      encoding="utf-8",
  )


def _read_metadata(path: Path) -> Optional[FileArtifactVersion]:
  """Loads a metadata payload from disk."""
  if not path.exists():
    return None
  try:
    return FileArtifactVersion.model_validate_json(
        path.read_text(encoding="utf-8")
    )
  except ValidationError as exc:
    logger.warning("Failed to parse metadata at %s: %s", path, exc)
    return None
  except ValueError as exc:
    logger.warning("Invalid metadata JSON at %s: %s", path, exc)
    return None
