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

"""Artifact service backed by an fsspec filesystem."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import logging
import mimetypes
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

from google.genai import types
from typing_extensions import override

from ...artifacts import artifact_util
from ...artifacts.base_artifact_service import ArtifactVersion
from ...artifacts.base_artifact_service import BaseArtifactService
from ...errors.input_validation_error import InputValidationError
from ...features._feature_decorator import experimental
from ...features._feature_registry import FeatureName

if TYPE_CHECKING:
  import fsspec

logger = logging.getLogger("google_adk." + __name__)

_USER_NAMESPACE_PREFIX = "user:"
_USER_NAMESPACE_SEGMENT = "user"
_DEFAULT_MIME_TYPE = "application/octet-stream"

# Custom-metadata keys written by GcsArtifactService on save. They must stay
# identical so that objects written by either service are readable by both.
_ADK_DISPLAY_NAME_METADATA_KEY = "adkDisplayName"
_ADK_IS_TEXT_METADATA_KEY = "adkIsText"
_ADK_FILE_URI_METADATA_KEY = "adkFileUri"
_ADK_FILE_MIME_TYPE_METADATA_KEY = "adkFileMimeType"
# Only legacy key GcsArtifactService still reads; other snake_case names are
# indistinguishable from caller-supplied custom metadata and are not honored.
_LEGACY_FILE_URI_METADATA_KEY = "file_uri"


def _file_has_user_namespace(filename: str) -> bool:
  """Returns True when ``filename`` uses the ``user:`` cross-session scope."""
  return filename.startswith(_USER_NAMESPACE_PREFIX)


def _validate_filename(filename: str) -> None:
  """Rejects filenames that could escape their artifact prefix.

  Unlike ``artifact_util.validate_path_segment``, nested names may contain
  reserved words such as ``versions`` (``docs/versions/v2.md``), because
  GcsArtifactService accepts them and Fsspec must be able to read them back.

  Args:
    filename: The caller-supplied filename, optionally prefixed with ``user:``.

  Raises:
    InputValidationError: If the name (after any ``user:`` prefix) is empty,
      contains null bytes, is absolute or drive-qualified, or contains empty,
      ``.``, or ``..`` segments.
  """
  name = (
      filename[len(_USER_NAMESPACE_PREFIX) :]
      if _file_has_user_namespace(filename)
      else filename
  )
  if not name:
    raise InputValidationError("filename must not be empty.")
  if "\x00" in name:
    raise InputValidationError("filename must not contain null bytes.")
  if name.startswith(("/", "\\")) or artifact_util._is_drive_qualified(name):
    raise InputValidationError(
        f"filename {filename!r} must be a relative path."
    )
  if {"", ".", ".."} & set(name.replace("\\", "/").split("/")):
    raise InputValidationError(
        f"filename {filename!r} must not contain empty or traversal segments."
    )


def _parse_version(suffix: str) -> Optional[int]:
  """Returns ``suffix`` as a version number, or None if it is not one.

  Mirrors GcsArtifactService: only plain ASCII digits are accepted, so values
  ``int()`` would tolerate (``+1``, `` 1``, ``1_0``, non-ASCII digits) are
  skipped. Zero-padded values (``01``) are skipped too, since they would
  otherwise alias the canonical version and produce duplicates.
  """
  if not (suffix.isascii() and suffix.isdigit()):
    return None
  if len(suffix) > 1 and suffix.startswith("0"):
    return None
  return int(suffix)


def _guess_mime(filename: str) -> str:
  """Best-effort MIME type from ``filename`` with octet-stream fallback."""
  if _file_has_user_namespace(filename):
    filename = filename[len(_USER_NAMESPACE_PREFIX) :]
  guessed, _ = mimetypes.guess_type(filename)
  return guessed or _DEFAULT_MIME_TYPE


def _fs_protocols(fs: fsspec.AbstractFileSystem) -> tuple[str, ...]:
  """Returns the protocol names handled by ``fs``."""
  return (fs.protocol,) if isinstance(fs.protocol, str) else tuple(fs.protocol)


@experimental(FeatureName.FSSPEC_ARTIFACT_SERVICE)
class FsspecArtifactService(BaseArtifactService):
  """Artifact service backed by any ``fsspec``-compatible filesystem.

  Uses the same object layout and custom metadata keys as
  ``GcsArtifactService``::

      {root}/{app_name}/{user_id}/{session_id}/{filename}/{version}
      {root}/{app_name}/{user_id}/user/{filename}/{version}

  Object metadata is read from the ``fs.info()`` mapping: ``contentType`` (or
  ``ContentType``) for the MIME type and ``metadata`` for ADK custom metadata,
  which matches gcsfs. Filesystems without stored metadata (``file://``,
  ``memory://``) fall back to a MIME type guessed from the filename and load
  as bytes. Backend-specific metadata (S3, Azure) is not read yet.

  Requires ``fsspec`` (``pip install "google-adk[fsspec]"``) and the backend
  implementation for the target storage protocol, such as ``gcsfs`` for
  ``gs://``, ``s3fs`` for ``s3://``, or ``adlfs`` for ``abfs://``, which the
  user installs separately. Neither is installed by default with
  ``google-adk``.

  Args:
    base_url: Root URL for artifacts, e.g. ``"gs://my-bucket/prefix"``,
      ``"memory://test-root"``, or ``"file:///tmp/artifacts"``.
    fs: Optional pre-configured, synchronous ``fsspec.AbstractFileSystem``. When
      supplied, ``base_url`` only determines the root path on ``fs`` and its
      protocol (if any) must be one ``fs`` handles.
    **storage_options: Keyword arguments forwarded to ``fsspec.core.url_to_fs``
      when ``fs`` is not provided. ``use_listings_cache`` defaults to ``False``
      so directory listings reflect external mutations immediately.

  Raises:
    ImportError: If the ``fsspec`` package is not installed.
    ValueError: If ``base_url`` is empty, ``fs`` is combined with
      ``storage_options``, ``fs`` does not handle the ``base_url`` protocol, or
      ``fs`` was created with ``asynchronous=True``.
  """

  def __init__(
      self,
      base_url: str,
      *,
      fs: Optional[fsspec.AbstractFileSystem] = None,
      **storage_options: Any,
  ) -> None:
    try:
      import fsspec  # pylint: disable=g-import-not-at-top,redefined-outer-name
    except ImportError as e:
      raise ImportError(
          "The fsspec package is required to use FsspecArtifactService."
          ' Install it with `pip install "google-adk[fsspec]"` (plus the'
          " backend for your storage protocol, e.g. `pip install gcsfs` for"
          " `gs://`)."
      ) from e

    if not base_url or not isinstance(base_url, str):
      raise ValueError("base_url must be a non-empty string.")

    if fs is None:
      storage_options.setdefault("use_listings_cache", False)
      fs, root = fsspec.core.url_to_fs(base_url, **storage_options)
    else:
      if storage_options:
        raise ValueError("storage_options cannot be combined with fs.")
      protocol, _ = fsspec.core.split_protocol(base_url)
      if protocol is not None and protocol not in _fs_protocols(fs):
        raise ValueError(
            f"base_url protocol {protocol!r} is not handled by the provided fs"
            f" (protocols: {_fs_protocols(fs)})."
        )
      root = fs._strip_protocol(base_url)
    if fs.async_impl and getattr(fs, "asynchronous", False):
      raise ValueError(
          "fs must be synchronous; create it without asynchronous=True."
      )

    self._fs: fsspec.AbstractFileSystem = fs
    # Keep the root exactly as the filesystem normalizes it: object stores
    # return "bucket/prefix" while POSIX-like filesystems (file, sftp, hdfs)
    # return an absolute "/path" whose leading slash is significant. A bare
    # "/" root becomes "" so that joined paths stay absolute ("/app/...").
    self._root: str = str(root).rstrip("/")

  def _path(self, *parts: str) -> str:
    """Joins ``parts`` under the root."""
    return "/".join((self._root, *parts))

  def _scope_path(
      self,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
      filename: str,
  ) -> str:
    """Returns the session or user scope path that holds ``filename``."""
    if _file_has_user_namespace(filename):
      return self._path(app_name, user_id, _USER_NAMESPACE_SEGMENT)
    if session_id is None:
      raise InputValidationError(
          "Session ID must be provided for session-scoped artifacts."
      )
    return self._path(app_name, user_id, session_id)

  def _artifact_path(
      self,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
      filename: str,
  ) -> str:
    """Validates inputs and returns the path under which versions live."""
    artifact_util.validate_path_segment(app_name, "app_name")
    artifact_util.validate_path_segment(user_id, "user_id")
    if session_id is not None:
      artifact_util.validate_path_segment(session_id, "session_id")
    _validate_filename(filename)
    scope = self._scope_path(app_name, user_id, session_id, filename)
    return f"{scope}/{filename}"

  def _find_files(self, path: str) -> list[str]:
    """Returns paths of all files under directory ``path``, relative to it."""
    prefix = self._fs._strip_protocol(path) + "/"
    self._fs.invalidate_cache(path)
    try:
      detail = self._fs.find(prefix, detail=True)
    except FileNotFoundError:
      return []
    return [
        stripped[len(prefix) :]
        for name, info in detail.items()
        if info.get("type", "file") == "file"
        and (stripped := self._fs._strip_protocol(name)).startswith(prefix)
    ]

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
    raise NotImplementedError(
        "FsspecArtifactService.save_artifact is not implemented yet."
    )

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
        self._load_artifact,
        app_name,
        user_id,
        session_id,
        filename,
        version,
    )

  def _load_artifact(
      self,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
      filename: str,
      version: Optional[int],
      *,
      max_depth: int = artifact_util._MAX_ARTIFACT_REFERENCE_DEPTH,
  ) -> Optional[types.Part]:
    artifact_path = self._artifact_path(app_name, user_id, session_id, filename)
    if version is None:
      versions = self._versions_under(artifact_path)
      if not versions:
        return None
      version = max(versions)
    elif version < 0:
      return None
    key = f"{artifact_path}/{version}"

    self._fs.invalidate_cache(key)
    try:
      info = self._fs.info(key)
    except FileNotFoundError:
      return None
    if info.get("type", "file") != "file":
      return None
    content_type = (
        info.get("contentType")
        or info.get("ContentType")
        or _guess_mime(filename)
    )
    metadata: Mapping[str, Any] = info.get("metadata") or {}

    file_uri = metadata.get(_ADK_FILE_URI_METADATA_KEY) or metadata.get(
        _LEGACY_FILE_URI_METADATA_KEY
    )
    if file_uri:
      if file_uri.startswith("artifact://"):
        parsed_uri = artifact_util.resolve_artifact_reference(
            file_uri=file_uri,
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            remaining_depth=max_depth,
        )
        return self._load_artifact(
            parsed_uri.app_name,
            parsed_uri.user_id,
            parsed_uri.session_id,
            parsed_uri.filename,
            parsed_uri.version,
            max_depth=max_depth - 1,
        )
      return types.Part(
          file_data=types.FileData(
              file_uri=file_uri,
              mime_type=metadata.get(_ADK_FILE_MIME_TYPE_METADATA_KEY)
              or content_type,
          )
      )

    try:
      data = self._fs.cat_file(key)
    except FileNotFoundError:
      # Deleted between info() and cat_file().
      return None
    if metadata.get(_ADK_IS_TEXT_METADATA_KEY) == "true":
      return types.Part(text=data.decode("utf-8"))
    artifact = types.Part(
        inline_data=types.Blob(
            data=data,
            mime_type=content_type,
            display_name=metadata.get(_ADK_DISPLAY_NAME_METADATA_KEY) or None,
        )
    )
    # A session rewind removes an artifact by saving a marker part as its
    # newest version; like every other artifact service, load it as absent.
    if artifact_util._is_rewind_tombstone(artifact):
      return None
    return artifact

  @override
  async def list_artifact_keys(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: Optional[str] = None,
  ) -> list[str]:
    return await asyncio.to_thread(
        self._list_artifact_keys, app_name, user_id, session_id
    )

  def _list_artifact_keys(
      self,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
  ) -> list[str]:
    artifact_util.validate_path_segment(app_name, "app_name")
    artifact_util.validate_path_segment(user_id, "user_id")
    scopes = [self._path(app_name, user_id, _USER_NAMESPACE_SEGMENT)]
    if session_id is not None:
      artifact_util.validate_path_segment(session_id, "session_id")
      scopes.append(self._path(app_name, user_id, session_id))

    filenames: set[str] = set()
    for scope in scopes:
      for path in self._find_files(scope):
        filename, _, suffix = path.rpartition("/")
        if filename and _parse_version(suffix) is not None:
          filenames.add(filename)
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
    raise NotImplementedError(
        "FsspecArtifactService.delete_artifact is not implemented yet."
    )

  @override
  async def list_versions(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
  ) -> list[int]:
    return await asyncio.to_thread(
        self._list_versions, app_name, user_id, session_id, filename
    )

  def _list_versions(
      self,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
      filename: str,
  ) -> list[int]:
    return self._versions_under(
        self._artifact_path(app_name, user_id, session_id, filename)
    )

  def _versions_under(self, artifact_path: str) -> list[int]:
    """Returns the sorted versions stored directly under ``artifact_path``.

    Nested artifacts share the prefix (``a/3`` vs. ``a/b/3``), so only paths of
    the exact form ``{artifact_path}/{version}`` count.
    """
    versions = []
    for suffix in self._find_files(artifact_path):
      version = None if "/" in suffix else _parse_version(suffix)
      if version is None:
        logger.debug("Skipping %s/%s: not a version.", artifact_path, suffix)
        continue
      versions.append(version)
    return sorted(versions)

  @override
  async def list_artifact_versions(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
  ) -> list[ArtifactVersion]:
    raise NotImplementedError(
        "FsspecArtifactService.list_artifact_versions is not implemented yet."
    )

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
    raise NotImplementedError(
        "FsspecArtifactService.get_artifact_version is not implemented yet."
    )
