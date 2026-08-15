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
import os
from pathlib import Path
from pathlib import PurePosixPath
from pathlib import PureWindowsPath
import shutil
import tempfile
from typing import Any
from typing import Optional
from typing import Union

from google.genai import types
from pydantic import alias_generators
from pydantic import ConfigDict
from pydantic import Field
from pydantic import ValidationError
from typing_extensions import override

from . import artifact_util
from ..errors.input_validation_error import InputValidationError
from .base_artifact_service import ArtifactVersion
from .base_artifact_service import BaseArtifactService
from .base_artifact_service import ensure_part

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
      # An artifact directory doubles as the parent of anything nested under
      # it ("doc" and "doc/nested"), so keep walking, skipping only the
      # stored versions of this artifact.
      dirnames[:] = [name for name in dirnames if name != "versions"]
  return artifact_dirs


def _read_bytes_if_present(path: Path) -> Optional[bytes]:
  """Reads a binary payload from disk.

  The read is attempted directly instead of being guarded by an `exists()`
  check so that a concurrent delete cannot be observed as a distinguishable
  state between the check and the read.

  Args:
    path: Location of the payload.

  Returns:
    The file contents, or None if it is not a readable file.
  """
  try:
    return path.read_bytes()
  except FileNotFoundError:
    return None
  except OSError as exc:
    logger.warning("Unreadable artifact payload at %s: %s", path, exc)
    return None


def _read_text_if_present(path: Path) -> Optional[str]:
  """Reads a UTF-8 text payload from disk.

  Args:
    path: Location of the payload.

  Returns:
    The decoded file contents, or None if it is not a readable file.
  """
  try:
    return path.read_text(encoding="utf-8")
  except FileNotFoundError:
    return None
  except OSError as exc:
    logger.warning("Unreadable artifact payload at %s: %s", path, exc)
    return None


def _umask_derived_file_mode() -> int:
  """Returns the mode a normally created file would get from the umask.

  Sampled once at import: reading the umask requires temporarily setting it,
  which is process-global and would race against concurrent writers if done
  per-write.

  Returns:
    The permission bits `open()` would produce for a new file.
  """
  umask = os.umask(0)
  os.umask(umask)
  return 0o666 & ~umask


# Payloads are written through `open()`, which applies the umask, but the
# metadata document is written through `tempfile.mkstemp`, which hardcodes
# 0600. Without this the two files in a version directory end up readable by
# different sets of principals.
_DEFAULT_FILE_MODE = _umask_derived_file_mode()

_USER_NAMESPACE_PREFIX = "user:"

# Name of the per-version metadata document. A payload is stored alongside it
# under the artifact directory's own name, so an artifact whose directory is
# named `metadata.json` would have its payload written over the metadata
# document. Callers may not use the name for that reason.
_METADATA_FILENAME = "metadata.json"


def _is_reserved_artifact_name(name: str) -> bool:
  """Checks whether an artifact directory name collides with the metadata doc.

  Compared caselessly because the collision is decided by the filesystem, and
  the case-insensitive ones ADK supports (APFS, NTFS) resolve `Metadata.json`
  and `metadata.json` to the same file.

  Args:
    name: The final path segment of the artifact directory.

  Returns:
    True if the name is reserved for internal use.
  """
  return name.casefold() == _METADATA_FILENAME.casefold()


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


def _is_rooted_or_drive_qualified(path_value: str) -> bool:
  """Checks POSIX and Windows rooted or drive-qualified path forms."""
  # A Windows root covers POSIX absolute paths, UNC and device prefixes alike;
  # only the drive-relative form (`C:name`) has no root of its own.
  if artifact_util._is_drive_qualified(path_value):
    return True
  return bool(PureWindowsPath(path_value).root)


def _has_parent_reference(path_value: str) -> bool:
  """Checks parent traversal using either platform's separators."""
  return (
      ".." in PurePosixPath(path_value).parts
      or ".." in PureWindowsPath(path_value).parts
  )


def _resolve_scoped_artifact_path(
    scope_root: Path, filename: str
) -> tuple[Path, Path]:
  """Returns the absolute artifact directory and its relative path.

  The caller is expected to pass the scope root directory (user or session).
  Filenames that are rooted, drive-qualified, or contain a parent reference are
  rejected outright, including parent references that would resolve back inside
  the scope root. Whatever remains is joined under the scope root.

  Args:
    scope_root: Directory that defines the storage scope.
    filename: Caller-supplied artifact name.

  Returns:
    A tuple containing the absolute artifact directory and its path relative
    to `scope_root`.

  Raises:
    InputValidationError: If `filename` is rooted, drive-qualified, contains a
      parent reference, or otherwise resolves outside of `scope_root`.
  """
  stripped = _strip_user_namespace(filename).strip()

  if _is_rooted_or_drive_qualified(stripped):
    raise InputValidationError(
        f"Rooted or drive-qualified artifact filename {filename!r} is not "
        "permitted; provide a path relative to the storage scope."
    )
  if _has_parent_reference(stripped):
    raise InputValidationError(
        f"Artifact filename {filename!r} must not contain parent traversal."
    )

  scope_root_resolved = scope_root.resolve(strict=False)
  pure_path = _to_posix_path(stripped)
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
  artifact_util.validate_path_segment(session_id, "session_id")
  return base_root / "sessions" / session_id / "artifacts"


def _versions_dir(artifact_dir: Path) -> Path:
  """Returns the directory that contains versioned payloads."""
  return artifact_dir / "versions"


def _metadata_path(artifact_dir: Path, version: int) -> Path:
  """Returns the path to the metadata file for a specific version."""
  return _versions_dir(artifact_dir) / str(version) / _METADATA_FILENAME


def _canonical_uri(artifact_dir: Path, version: int) -> str:
  """Builds the canonical file:// URI for an artifact payload."""
  payload_path = _versions_dir(artifact_dir) / str(version) / artifact_dir.name
  return payload_path.resolve().as_uri()


def _prune_empty_dirs(leaf: Path, stop_at: Path) -> None:
  """Removes `leaf` and any parents it leaves empty, stopping at `stop_at`.

  Filenames may contain "/", so the directory of an artifact doubles as the
  parent directory of every artifact nested under it: "doc" is stored at
  ``{scope}/doc`` and "doc/nested" at ``{scope}/doc/nested``. A directory may
  therefore only be removed once it holds nothing, or deleting "doc" would
  take "doc/nested" with it.

  Args:
    leaf: Directory to remove, if it is empty.
    stop_at: Scope root. It and everything above it are never removed.
  """
  current = leaf
  while current != stop_at and current.is_relative_to(stop_at):
    try:
      current.rmdir()  # Only succeeds on an empty directory.
    except OSError:
      return
    current = current.parent


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
  display_name: Optional[str] = Field(
      default=None,
      description=(
          "User-facing filename from inline_data.display_name when persisted."
      ),
  )


class FileArtifactService(BaseArtifactService):
  """Stores filesystem-backed artifacts beneath a configurable root directory."""

  # Storage layout matches the cloud and in-memory services:
  # root/
  # └── apps/
  #     └── {app_name}/
  #         └── users/
  #             └── {user_id}/
  #                 ├── sessions/
  #                 │   └── {session_id}/
  #                 │       └── artifacts/
  #                 │           └── {artifact_path}/  # from filename
  #                 │               └── versions/
  #                 │                   └── {version}/
  #                 │                       ├── {original_filename}
  #                 │                       └── metadata.json
  #                 └── artifacts/
  #                     └── {artifact_path}/...
  #
  # Releases that predate the `apps/{app_name}` level wrote the same tree
  # directly under `root/users`, which records no app name. A root can be
  # shared by several apps, so that tree cannot be attributed to one of them
  # and is never read from or deleted.
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

  def _base_root(self, app_name: str, user_id: str) -> Path:
    """Returns the app-scoped root holding a user's artifacts."""
    artifact_util.validate_path_segment(app_name, "app_name")
    artifact_util.validate_path_segment(user_id, "user_id")
    return self.root_dir / "apps" / app_name / "users" / user_id

  def _scope_root(
      self,
      base_root: Path,
      session_id: Optional[str],
      filename: str,
  ) -> Path:
    """Returns the directory that represents the artifact scope."""
    if _is_user_scoped(session_id, filename):
      return _user_artifacts_dir(base_root)
    if session_id is None:
      raise InputValidationError(
          "Session ID must be provided for session-scoped artifacts."
      )
    return _session_artifacts_dir(base_root, session_id)

  def _artifact_dir(
      self,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
      filename: str,
  ) -> Path:
    """Builds the directory that stores an artifact for an app."""
    base_root = self._base_root(app_name, user_id)
    return _resolve_scoped_artifact_path(
        self._scope_root(base_root, session_id, filename), filename
    )[0]

  def _build_artifact_version(
      self,
      *,
      artifact_dir: Path,
      version: int,
      metadata: Optional[FileArtifactVersion],
  ) -> ArtifactVersion:
    """Creates an ArtifactVersion payload using on-disk metadata."""
    # Always recomputed from the storage layout rather than read back from the
    # metadata document. For this service the two are equivalent for data this
    # service wrote, and recomputing means a tampered document cannot dictate
    # the URI handed to callers.
    canonical_uri = _canonical_uri(artifact_dir, version)
    custom_metadata_val = metadata.custom_metadata if metadata else {}
    mime_type = metadata.mime_type if metadata else None
    return ArtifactVersion(
        version=version,
        canonical_uri=canonical_uri,
        custom_metadata=dict(custom_metadata_val),
        mime_type=mime_type,
    )

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
      artifact: Union[types.Part, dict[str, Any]],
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
        app_name,
        user_id,
        filename,
        artifact,
        session_id,
        custom_metadata,
    )

  def _save_artifact_sync(
      self,
      app_name: str,
      user_id: str,
      filename: str,
      artifact: Union[types.Part, dict[str, Any]],
      session_id: Optional[str],
      custom_metadata: Optional[dict[str, Any]],
  ) -> int:
    """Saves an artifact to disk and returns its version."""
    artifact = ensure_part(artifact)
    artifact_dir = self._artifact_dir(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
    )
    # Enforced here rather than in `_artifact_dir`, which reads and deletes
    # share: an artifact stored under this name before the name was rejected
    # must stay readable and, above all, deletable.
    if _is_reserved_artifact_name(artifact_dir.name):
      raise InputValidationError(
          f"Artifact filename {filename!r} is reserved: an artifact may not be"
          f" named {_METADATA_FILENAME!r} (in any casing) because its payload"
          " is stored under the artifact's own name and would overwrite the"
          " metadata document."
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

    # A version directory is only ever observed complete or not at all. A
    # partially written version -- payload present, metadata missing or
    # truncated -- is indistinguishable from a valid one on the read path, so
    # any failure discards the whole directory instead of leaving it behind.
    try:
      display_name: Optional[str] = None
      if artifact.inline_data:
        data = artifact.inline_data.data
        if data is None:
          raise InputValidationError("Artifact inline_data must contain data.")
        content_path.write_bytes(data)
        mime_type = (
            artifact.inline_data.mime_type
            if artifact.inline_data.mime_type
            else "application/octet-stream"
        )
        display_name = artifact.inline_data.display_name
      elif artifact.text is not None:
        content_path.write_text(artifact.text, encoding="utf-8")
        mime_type = None
      else:
        raise InputValidationError(
            "Artifact must have either inline_data or text content."
        )

      canonical_uri = _canonical_uri(artifact_dir, next_version)
      _write_metadata(
          _metadata_path(artifact_dir, next_version),
          filename=filename,
          mime_type=mime_type,
          version=next_version,
          canonical_uri=canonical_uri,
          custom_metadata=custom_metadata,
          display_name=display_name,
      )
    except BaseException:
      shutil.rmtree(version_dir, ignore_errors=True)
      raise

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
        app_name,
        user_id,
        filename,
        session_id,
        version,
    )

  def _load_artifact_sync(
      self,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str],
      version: Optional[int],
  ) -> Optional[types.Part]:
    """Loads an artifact from disk."""
    artifact_dir = self._artifact_dir(
        app_name=app_name,
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
    # The payload location is derived exclusively from the storage layout. It
    # must never be taken from the metadata document: that document lives in
    # the artifact tree and is therefore attacker-influenced input, so honoring
    # a `canonical_uri` from it would turn this into an arbitrary file read.
    content_path = version_dir / stored_filename

    # Read without a preceding `exists()` check. A separate `delete_artifact`
    # can unlink the payload between the check and the read, and reacting to
    # that gap is what previously reached the metadata-supplied path.
    if mime_type:
      data = _read_bytes_if_present(content_path)
      if data is None:
        logger.warning(
            "Binary artifact %s missing at %s", filename, content_path
        )
        return None
      return types.Part(
          inline_data=types.Blob(
              mime_type=mime_type,
              data=data,
              display_name=metadata.display_name if metadata else None,
          )
      )

    text = _read_text_if_present(content_path)
    if text is None:
      logger.warning("Text artifact %s missing at %s", filename, content_path)
      return None
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
        app_name,
        user_id,
        session_id,
    )

  def _list_artifact_keys_sync(
      self,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
  ) -> list[str]:
    """Lists artifact filenames for the given session/user."""
    filenames: set[str] = set()
    base_root = self._base_root(app_name, user_id)

    if session_id is not None:
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
        app_name,
        user_id,
        filename,
        session_id,
    )

  def _delete_artifact_sync(
      self,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str],
  ) -> None:
    artifact_dir = self._artifact_dir(app_name, user_id, session_id, filename)
    versions_dir = _versions_dir(artifact_dir)
    if not versions_dir.exists():
      return
    # Only this artifact's own versions go. Its directory may also be the
    # parent of a nested artifact ("doc" vs "doc/nested"), so it is pruned
    # separately and only if nothing is left under it.
    shutil.rmtree(versions_dir)
    scope_root = self._scope_root(
        self._base_root(app_name, user_id), session_id, filename
    )
    _prune_empty_dirs(artifact_dir, scope_root)
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
        app_name,
        user_id,
        filename,
        session_id,
    )

  def _list_versions_sync(
      self,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str],
  ) -> list[int]:
    artifact_dir = self._artifact_dir(
        app_name=app_name,
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
        app_name,
        user_id,
        filename,
        session_id,
    )

  def _list_artifact_versions_sync(
      self,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str],
  ) -> list[ArtifactVersion]:
    artifact_dir = self._artifact_dir(
        app_name=app_name,
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
              artifact_dir=artifact_dir,
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
        app_name,
        user_id,
        filename,
        session_id,
        version,
    )

  def _get_artifact_version_sync(
      self,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str],
      version: Optional[int],
  ) -> Optional[ArtifactVersion]:
    artifact_dir = self._artifact_dir(
        app_name=app_name,
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
        artifact_dir=artifact_dir,
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
    display_name: Optional[str] = None,
) -> None:
  """Persists metadata describing an artifact version."""
  metadata = FileArtifactVersion(
      file_name=filename,
      mime_type=mime_type,
      canonical_uri=canonical_uri,
      version=version,
      display_name=display_name,
      # Persist caller supplied metadata for feature parity with other
      # artifact services (e.g. GCS).
      custom_metadata=dict(custom_metadata or {}),
  )
  # Serialize before touching the filesystem: serialization is caller-driven
  # (`custom_metadata` is arbitrary) and can fail, and it must not be able to
  # leave a truncated document behind.
  serialized = metadata.model_dump_json(by_alias=True, exclude_none=True)

  # Write via a uniquely named temporary file in the same directory and rename
  # it into place, so readers never observe a partial document.
  fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
  tmp_path = Path(tmp_name)
  try:
    with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
      tmp_file.write(serialized)
    # `os.replace` carries the temporary file's mode over to the destination,
    # and mkstemp made it 0600. Restore the mode the payload beside it got.
    os.chmod(tmp_path, _DEFAULT_FILE_MODE)
    os.replace(tmp_path, path)
  except BaseException:
    tmp_path.unlink(missing_ok=True)
    raise


def _read_metadata(path: Path) -> Optional[FileArtifactVersion]:
  """Loads a metadata payload from disk.

  The path is derived from a caller-supplied filename, so it can be made to
  name a directory rather than a file; that must degrade to "no metadata"
  instead of raising.

  Args:
    path: Location of the metadata document.

  Returns:
    The parsed metadata, or None for anything that is not a readable,
    well-formed metadata document.
  """
  try:
    raw = path.read_text(encoding="utf-8")
  except FileNotFoundError:
    return None
  except OSError as exc:
    logger.warning("Unreadable metadata at %s: %s", path, exc)
    return None
  try:
    return FileArtifactVersion.model_validate_json(raw)
  except ValidationError as exc:
    logger.warning("Failed to parse metadata at %s: %s", path, exc)
    return None
  except ValueError as exc:
    logger.warning("Invalid metadata JSON at %s: %s", path, exc)
    return None
