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
from ...artifacts.base_artifact_service import ensure_part
from ...errors.input_validation_error import InputValidationError
from ...features._feature_decorator import experimental
from ...features._feature_registry import FeatureName
from ._backends import _ADK_DISPLAY_NAME_METADATA_KEY
from ._backends import _ADK_FILE_MIME_TYPE_METADATA_KEY
from ._backends import _ADK_FILE_URI_METADATA_KEY
from ._backends import _ADK_IS_TEXT_METADATA_KEY
from ._backends import _BackendAdapter
from ._backends import _DEFAULT_MIME_TYPE
from ._backends import _fs_protocols
from ._backends import _ObjectMetadata
from ._backends import _user_metadata

if TYPE_CHECKING:
  import fsspec

logger = logging.getLogger("google_adk." + __name__)

_USER_NAMESPACE_PREFIX = "user:"
_USER_NAMESPACE_SEGMENT = "user"
_MAX_SAVE_VERSION_ATTEMPTS = 10


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

  Like GcsArtifactService, only plain ASCII digits are accepted, so values
  ``int()`` would tolerate (``+1``, `` 1``, ``1_0``, non-ASCII digits) are
  skipped. Zero-padded values (``01``), which GcsArtifactService never writes,
  are also rejected so they cannot alias canonical versions.
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


@experimental(FeatureName.FSSPEC_ARTIFACT_SERVICE)
class FsspecArtifactService(BaseArtifactService):
  """Artifact service backed by any ``fsspec``-compatible filesystem.

  Uses the same object layout and custom metadata keys as
  ``GcsArtifactService``::

      {root}/{app_name}/{user_id}/{session_id}/{filename}/{version}
      {root}/{app_name}/{user_id}/user/{filename}/{version}

  On GCS (``gcsfs``, including standard, hierarchical-namespace, and Rapid
  buckets), object ``contentType`` and custom ``metadata`` are persisted on
  save and restored on read, maintaining full payload and metadata parity with
  ``GcsArtifactService``. Filesystems without custom object metadata
  (``file://``, ``memory://``) persist raw ``inline_data`` payload bytes and
  fall back to a MIME type guessed from the filename on read: ``text`` parts,
  ``file_data`` URI references, and non-empty ``custom_metadata`` are rejected
  with ``InputValidationError``, ``inline_data.display_name`` is not persisted,
  and empty (``0``-byte) payloads with no stored content type load as ``None``
  so session-rewind tombstones work without object metadata.

  S3 (``s3fs``) also stores content type and custom metadata. Its
  ``custom_metadata`` keys come back lowercased with ``_`` turned into ``-``.
  ``custom_metadata`` keys must be HTTP header names, and caller keys that S3
  would read back as ADK's own keys or as each other are rejected.

  Azure (``adlfs``) also stores content type and custom metadata. Its
  ``custom_metadata`` keys must be C# identifiers that stay distinct and do not
  match ADK's or adlfs's own keys when case is ignored. On both stores, values
  must be printable ASCII without spaces at either end.

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

    protocol, _ = fsspec.core.split_protocol(base_url)
    if fs is None:
      storage_options.setdefault("use_listings_cache", False)
      fs, root = fsspec.core.url_to_fs(base_url, **storage_options)
    else:
      if storage_options:
        raise ValueError("storage_options cannot be combined with fs.")
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
    self._scheme: Optional[str] = protocol
    # Keep the root exactly as the filesystem normalizes it: object stores
    # return "bucket/prefix" while POSIX-like filesystems (file, sftp, hdfs)
    # return an absolute "/path" whose leading slash is significant. A bare
    # "/" root becomes "" so that joined paths stay absolute ("/app/...").
    self._root: str = str(root).rstrip("/")
    self._backend_adapter: _BackendAdapter = _BackendAdapter.for_filesystem(fs)

  def _canonical_uri(self, key: str) -> str:
    """Returns the canonical URI for ``key``, preserving ``base_url`` scheme."""
    uri = str(self._fs.unstrip_protocol(key))
    scheme, _, rest = uri.partition("://")
    effective_scheme = self._scheme or scheme
    if effective_scheme == "file" and artifact_util._is_drive_qualified(rest):
      rest = f"/{rest}"
    return f"{effective_scheme}://{rest}"

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
    artifact_util.validate_path_segment(session_id, "session_id")
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
    _validate_filename(filename)
    scope = self._scope_path(app_name, user_id, session_id, filename)
    return f"{scope}/{filename}"

  def _find_detail(self, path: str) -> dict[str, Mapping[str, Any]]:
    """Returns file detail mappings under ``path``, keyed by relative path."""
    prefix = self._fs._strip_protocol(path) + "/"
    self._fs.invalidate_cache(path)
    try:
      detail = self._backend_adapter.find_detail(prefix)
    except (FileNotFoundError, NotADirectoryError):
      return {}
    return {
        stripped[len(prefix) :]: info
        for name, info in detail.items()
        if info.get("type", "file") == "file"
        and (stripped := self._fs._strip_protocol(name)).startswith(prefix)
    }

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
    return await asyncio.to_thread(
        self._save_artifact,
        app_name,
        user_id,
        session_id,
        filename,
        artifact,
        custom_metadata,
    )

  def _save_artifact(
      self,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
      filename: str,
      artifact: Union[types.Part, dict[str, Any]],
      custom_metadata: Optional[dict[str, Any]],
  ) -> int:
    artifact_path = self._artifact_path(app_name, user_id, session_id, filename)
    if session_id is not None and not _file_has_user_namespace(filename):
      artifact_util._validate_session_id_for_flat_storage(session_id)

    part = ensure_part(artifact)
    blob_metadata = {
        k: str(v) for k, v in _user_metadata(custom_metadata).items()
    }

    data: bytes
    content_type: Optional[str]
    if part.inline_data is not None:
      if part.inline_data.data is None:
        raise InputValidationError("Artifact inline_data must contain data.")
      if blob_metadata and not self._backend_adapter.stores_custom_metadata:
        raise InputValidationError(
            "Saving custom_metadata requires a filesystem backend that persists"
            " custom object metadata (such as gs://, s3:// or abfs://)."
        )
      data = part.inline_data.data
      content_type = part.inline_data.mime_type
      if part.inline_data.display_name:
        blob_metadata[_ADK_DISPLAY_NAME_METADATA_KEY] = (
            part.inline_data.display_name
        )
    elif part.text is not None:
      if not self._backend_adapter.stores_custom_metadata:
        raise InputValidationError(
            "Saving text artifacts requires a filesystem backend that persists"
            " custom object metadata (such as gs://, s3:// or abfs://)."
        )
      data = part.text.encode("utf-8")
      content_type = "text/plain"
      blob_metadata[_ADK_IS_TEXT_METADATA_KEY] = "true"
    elif part.file_data is not None:
      file_uri = part.file_data.file_uri
      if not file_uri:
        raise InputValidationError("Artifact file_data must have a file_uri.")
      if not self._backend_adapter.stores_custom_metadata:
        raise InputValidationError(
            "Saving file_data URI references requires a filesystem backend that"
            " persists custom object metadata (such as gs://, s3:// or"
            " abfs://)."
        )
      if artifact_util.is_artifact_ref(part):
        parsed_uri = artifact_util.parse_artifact_uri(file_uri)
        if not parsed_uri:
          raise InputValidationError(
              f"Invalid artifact reference URI: {file_uri}"
          )
        artifact_util.validate_artifact_reference_scope(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            parsed_uri=parsed_uri,
        )
      blob_metadata[_ADK_FILE_URI_METADATA_KEY] = file_uri
      if part.file_data.mime_type:
        blob_metadata[_ADK_FILE_MIME_TYPE_METADATA_KEY] = (
            part.file_data.mime_type
        )
      if part.file_data.display_name:
        blob_metadata[_ADK_DISPLAY_NAME_METADATA_KEY] = (
            part.file_data.display_name
        )
      data = b""
      content_type = part.file_data.mime_type or None
    else:
      raise InputValidationError(
          "Artifact must have either inline_data, text, or file_data."
      )

    versions = self._versions_under(artifact_path)
    version = 0 if not versions else max(versions) + 1

    # Listing versions does not reserve one, so a concurrent save can choose the
    # same number. ``create_object`` is exclusive, so on FileExistsError,
    # advance and retry.
    attempts_left = _MAX_SAVE_VERSION_ATTEMPTS
    while True:
      key = f"{artifact_path}/{version}"
      try:
        self._backend_adapter.create_object(
            key, data, content_type=content_type, metadata=blob_metadata
        )
      except FileExistsError:
        attempts_left -= 1
        if not attempts_left:
          raise
        version += 1
        continue
      self._fs.invalidate_cache(artifact_path)
      self._fs.invalidate_cache(key)
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

    metadata = self._backend_adapter.read_metadata(key)
    if metadata is None:
      return None

    file_uri = metadata.file_uri
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
              mime_type=metadata.file_mime_type or metadata.content_type,
              display_name=metadata.display_name,
          )
      )

    try:
      data = self._fs.cat_file(key)
    except (FileNotFoundError, NotADirectoryError):
      # Deleted between info() and cat_file().
      return None
    if metadata.is_text:
      return types.Part(text=data.decode("utf-8"))
    artifact = types.Part(
        inline_data=types.Blob(
            data=data,
            mime_type=metadata.content_type or _guess_mime(filename),
            display_name=metadata.display_name,
        )
    )
    # A session rewind removes an artifact by saving a marker part as its
    # newest version; like every other artifact service, load it as absent.
    # Backends without custom object metadata cannot persist the marker's
    # octet-stream content type, so an empty payload with no stored content
    # type is also treated as a tombstone there.
    if artifact_util._is_rewind_tombstone(artifact) or (
        not self._backend_adapter.stores_custom_metadata
        and not data
        and metadata.content_type is None
    ):
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
      for path in self._find_detail(scope):
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
    await asyncio.to_thread(
        self._delete_artifact, app_name, user_id, session_id, filename
    )

  def _delete_artifact(
      self,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
      filename: str,
  ) -> None:
    artifact_path = self._artifact_path(app_name, user_id, session_id, filename)
    for version in self._versions_under(artifact_path):
      key = f"{artifact_path}/{version}"
      try:
        self._fs.rm_file(key)
      except (FileNotFoundError, NotADirectoryError):
        pass
      self._fs.invalidate_cache(key)
    self._fs.invalidate_cache(artifact_path)

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
    for suffix in self._find_detail(artifact_path):
      version = None if "/" in suffix else _parse_version(suffix)
      if version is None:
        logger.debug("Skipping %s/%s: not a version.", artifact_path, suffix)
        continue
      versions.append(version)
    return sorted(versions)

  def _build_artifact_version(
      self,
      version: int,
      key: str,
      filename: str,
      metadata: _ObjectMetadata,
  ) -> ArtifactVersion:
    """Constructs an ``ArtifactVersion`` from normalized object metadata."""
    if metadata.file_uri:
      mime_type = metadata.file_mime_type or metadata.content_type
    else:
      mime_type = metadata.content_type or _guess_mime(filename)
    return ArtifactVersion(
        version=version,
        canonical_uri=self._canonical_uri(key),
        create_time=metadata.create_time,
        mime_type=mime_type,
        custom_metadata=metadata.user_metadata,
    )

  @override
  async def list_artifact_versions(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
  ) -> list[ArtifactVersion]:
    return await asyncio.to_thread(
        self._list_artifact_versions, app_name, user_id, session_id, filename
    )

  def _list_artifact_versions(
      self,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
      filename: str,
  ) -> list[ArtifactVersion]:
    artifact_path = self._artifact_path(app_name, user_id, session_id, filename)
    entries: dict[str, Mapping[str, Any]] = {}
    versions: dict[str, int] = {}
    for suffix, info in self._find_detail(artifact_path).items():
      version = None if "/" in suffix else _parse_version(suffix)
      if version is not None:
        key = f"{artifact_path}/{version}"
        entries[key] = info
        versions[key] = version
    listed = self._backend_adapter.read_listing_metadata(entries)
    artifact_versions = [
        self._build_artifact_version(versions[key], key, filename, metadata)
        for key, metadata in listed.items()
    ]
    artifact_versions.sort(key=lambda av: av.version)
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
    return await asyncio.to_thread(
        self._get_artifact_version,
        app_name,
        user_id,
        session_id,
        filename,
        version,
    )

  def _get_artifact_version(
      self,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
      filename: str,
      version: Optional[int],
  ) -> Optional[ArtifactVersion]:
    artifact_path = self._artifact_path(app_name, user_id, session_id, filename)
    if version is None:
      versions = self._versions_under(artifact_path)
      if not versions:
        return None
      version = max(versions)
    elif version < 0:
      return None
    key = f"{artifact_path}/{version}"

    metadata = self._backend_adapter.read_metadata(key)
    if metadata is None:
      return None
    return self._build_artifact_version(version, key, filename, metadata)
