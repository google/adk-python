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
import datetime
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
from ...platform import time as platform_time

if TYPE_CHECKING:
  import fsspec

logger = logging.getLogger("google_adk." + __name__)

_USER_NAMESPACE_PREFIX = "user:"
_USER_NAMESPACE_SEGMENT = "user"
_DEFAULT_MIME_TYPE = "application/octet-stream"
_GCS_PROTOCOLS = frozenset({"gs", "gcs"})
_LOCAL_PROTOCOLS = frozenset({"file", "local"})
_RAPID_PRECONDITION_ERROR_NAMES = frozenset({
    "FailedPrecondition",
    "PreconditionFailed",
})
# Parameter names consumed by GCSFileSystem.setxattrs (plus fsspec's
# sync_wrapper), which would collide or be intercepted when unpacking
# **blob_metadata on Rapid buckets.
_RAPID_SETXATTRS_RESERVED_KEYS = frozenset({
    "self",
    "path",
    "content_type",
    "content_encoding",
    "fixed_key_metadata",
    "timeout",
})
_MAX_SAVE_VERSION_ATTEMPTS = 10

# Custom-metadata keys written by GcsArtifactService on save. They must stay
# identical so that objects written by either service are readable by both.
_ADK_DISPLAY_NAME_METADATA_KEY = "adkDisplayName"
_ADK_IS_TEXT_METADATA_KEY = "adkIsText"
_ADK_FILE_URI_METADATA_KEY = "adkFileUri"
_ADK_FILE_MIME_TYPE_METADATA_KEY = "adkFileMimeType"
# Only legacy key GcsArtifactService still reads; other snake_case names are
# indistinguishable from caller-supplied custom metadata and are not honored.
_LEGACY_FILE_URI_METADATA_KEY = "file_uri"
# Object metadata keys reserved for ADK's own bookkeeping. They are dropped
# from custom_metadata on save and hidden from it on read.
_INTERNAL_METADATA_KEYS = frozenset({
    _ADK_DISPLAY_NAME_METADATA_KEY,
    _ADK_IS_TEXT_METADATA_KEY,
    _ADK_FILE_URI_METADATA_KEY,
    _ADK_FILE_MIME_TYPE_METADATA_KEY,
    _LEGACY_FILE_URI_METADATA_KEY,
})


def _file_has_user_namespace(filename: str) -> bool:
  """Returns True when ``filename`` uses the ``user:`` cross-session scope."""
  return filename.startswith(_USER_NAMESPACE_PREFIX)


def _user_metadata(
    raw_metadata: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
  """Returns only caller-supplied metadata keys, hiding ADK internal markers."""
  if not raw_metadata:
    return {}
  return {
      key: value
      for key, value in raw_metadata.items()
      if key not in _INTERNAL_METADATA_KEYS
  }


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


def _fs_protocols(fs: fsspec.AbstractFileSystem) -> tuple[str, ...]:
  """Returns the protocol names handled by ``fs``."""
  return (fs.protocol,) if isinstance(fs.protocol, str) else tuple(fs.protocol)


def _parse_timestamp(info: Mapping[str, Any]) -> float:
  """Extracts a Unix creation timestamp from an fsspec info() mapping."""
  for field in ("timeCreated", "created", "mtime", "LastModified", "updated"):
    raw = info.get(field)
    if isinstance(raw, (int, float)):
      return float(raw)
    if isinstance(raw, datetime.datetime):
      if raw.tzinfo is None:
        raw = raw.replace(tzinfo=datetime.timezone.utc)
      return raw.timestamp()
    if isinstance(raw, str):
      iso = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
      try:
        parsed = datetime.datetime.fromisoformat(iso)
      except ValueError:
        continue
      if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
      return parsed.timestamp()
  return platform_time.get_time()


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
  so session-rewind tombstones work without object metadata. Backend-specific
  custom metadata normalization for S3 and Azure is not implemented yet.

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

  def _is_gcs(self) -> bool:
    """Returns True when ``self._fs`` handles GCS protocols."""
    return not _GCS_PROTOCOLS.isdisjoint(_fs_protocols(self._fs))

  def _is_local(self) -> bool:
    """Returns True when ``self._fs`` handles local filesystem protocols."""
    return not _LOCAL_PROTOCOLS.isdisjoint(_fs_protocols(self._fs))

  def _is_rapid_gcs_bucket(self, key: str) -> bool:
    """Returns True when ``self._fs`` is gcsfs targeting a Rapid bucket.

    Rapid buckets report the ``ZONAL_HIERARCHICAL`` bucket type in gcsfs.
    """
    lookup = getattr(self._fs, "_sync_lookup_bucket_type", None)
    split_path = getattr(self._fs, "split_path", None)
    if not (callable(lookup) and callable(split_path)):
      return False
    bucket, _, _ = split_path(key)
    return str(lookup(bucket)).endswith("ZONAL_HIERARCHICAL")

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

  def _find_detail(self, path: str) -> dict[str, dict[str, Any]]:
    """Returns file detail mappings under ``path``, keyed by relative path."""
    prefix = self._fs._strip_protocol(path) + "/"
    self._fs.invalidate_cache(path)
    try:
      detail = self._fs.find(prefix, detail=True)
    except (FileNotFoundError, NotADirectoryError):
      return {}
    return {
        stripped[len(prefix) :]: info
        for name, info in detail.items()
        if info.get("type", "file") == "file"
        and (stripped := self._fs._strip_protocol(name)).startswith(prefix)
    }

  def _remove_failed_version(self, key: str) -> None:
    """Best-effort removal of a version object left behind by a failed save.

    A failed save must not leave a partial or metadata-less object that later
    loads as a corrupt version. Removal errors are logged, not raised, so the
    original failure is what the caller sees.
    """
    try:
      self._fs.rm_file(key)
    except Exception:  # pylint: disable=broad-exception-caught
      logger.warning(
          "Could not remove artifact version %s after a failed save.",
          key,
          exc_info=True,
      )
    self._fs.invalidate_cache(key)

  def _write_version(
      self,
      key: str,
      data: bytes,
      content_type: Optional[str],
      blob_metadata: dict[str, str],
  ) -> None:
    """Writes one version object with exclusive-create semantics."""
    if not self._is_gcs():
      if self._is_local():
        with self._fs.open(key, "xb") as stream:
          stream.write(data)
      else:
        self._fs.pipe_file(key, data, mode="create")
      return

    if self._is_rapid_gcs_bucket(key):
      reserved = sorted(_RAPID_SETXATTRS_RESERVED_KEYS & blob_metadata.keys())
      if reserved:
        raise InputValidationError(
            "custom_metadata keys are not supported on Rapid GCS buckets:"
            f" {', '.join(reserved)}."
        )
      # ExtendedGcsFileSystem._pipe_file ignores mode="create", leaves objects
      # unfinalized, and drops content_type and metadata during gRPC upload.
      # Open with generation=0 (exclusive create) and finalize_on_close=True,
      # then patch contentType (including GCS's HTTP octet-stream default) and
      # metadata via setxattrs. The object is briefly visible without metadata
      # between the two steps; gcsfs offers no single-step alternative.
      stream = self._fs.open(key, "wb", generation=0, finalize_on_close=True)
      try:
        if data:
          stream.write(data)
        stream.close()
      except Exception as exc:
        # Prevent __del__() from calling close() to re-flush a failed stream.
        stream.closed = True
        if type(exc).__name__ in _RAPID_PRECONDITION_ERROR_NAMES:
          raise FileExistsError(key) from exc
        # The appendable writer is only set once the exclusive create
        # succeeded, so the (unfinalized) object is ours to remove.
        if getattr(stream, "aaow", None) is not None:
          self._remove_failed_version(key)
        raise
      try:
        self._fs.setxattrs(
            key,
            content_type=content_type or _DEFAULT_MIME_TYPE,
            **blob_metadata,
        )
      except Exception:
        self._remove_failed_version(key)
        raise
      return

    write_kwargs: dict[str, Any] = {"mode": "create"}
    if content_type is not None:
      write_kwargs["content_type"] = content_type
    if blob_metadata:
      write_kwargs["metadata"] = blob_metadata
    self._fs.pipe_file(key, data, **write_kwargs)

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
      if blob_metadata and not self._is_gcs():
        raise InputValidationError(
            "Saving custom_metadata requires a filesystem backend that"
            " persists custom object metadata (such as gs://)."
        )
      data = part.inline_data.data
      content_type = part.inline_data.mime_type
      if part.inline_data.display_name:
        blob_metadata[_ADK_DISPLAY_NAME_METADATA_KEY] = (
            part.inline_data.display_name
        )
    elif part.text is not None:
      if not self._is_gcs():
        raise InputValidationError(
            "Saving text artifacts requires a filesystem backend that"
            " persists custom object metadata (such as gs://)."
        )
      data = part.text.encode("utf-8")
      content_type = "text/plain"
      blob_metadata[_ADK_IS_TEXT_METADATA_KEY] = "true"
    elif part.file_data is not None:
      file_uri = part.file_data.file_uri
      if not file_uri:
        raise InputValidationError("Artifact file_data must have a file_uri.")
      if not self._is_gcs():
        raise InputValidationError(
            "Saving file_data URI references requires a filesystem backend"
            " that persists custom object metadata (such as gs://)."
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
      data = b""
      content_type = part.file_data.mime_type or None
    else:
      raise InputValidationError(
          "Artifact must have either inline_data, text, or file_data."
      )

    if not self._is_gcs() and getattr(self._fs, "auto_mkdir", True) is False:
      self._fs.makedirs(artifact_path, exist_ok=True)

    versions = self._versions_under(artifact_path)
    version = 0 if not versions else max(versions) + 1

    # Listing versions does not reserve one, so a concurrent save can choose the
    # same number. Exclusive create is atomic on GCS (generation=0), file://
    # (open with "xb"), and memory://; generic fsspec backends inherit
    # AbstractFileSystem.pipe_file(mode="create"), which uses a non-atomic
    # exists() check before open("wb"). On FileExistsError, advance and retry.
    attempts_left = _MAX_SAVE_VERSION_ATTEMPTS
    while True:
      key = f"{artifact_path}/{version}"
      try:
        self._write_version(key, data, content_type, blob_metadata)
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

    self._fs.invalidate_cache(key)
    try:
      info = self._fs.info(key)
    except (FileNotFoundError, NotADirectoryError):
      return None
    if info.get("type", "file") != "file":
      return None
    stored_content_type = (
        info.get("contentType") or info.get("ContentType") or None
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
              or stored_content_type,
          )
      )

    try:
      data = self._fs.cat_file(key)
    except (FileNotFoundError, NotADirectoryError):
      # Deleted between info() and cat_file().
      return None
    if metadata.get(_ADK_IS_TEXT_METADATA_KEY) == "true":
      return types.Part(text=data.decode("utf-8"))
    artifact = types.Part(
        inline_data=types.Blob(
            data=data,
            mime_type=stored_content_type or _guess_mime(filename),
            display_name=metadata.get(_ADK_DISPLAY_NAME_METADATA_KEY) or None,
        )
    )
    # A session rewind removes an artifact by saving a marker part as its
    # newest version; like every other artifact service, load it as absent.
    # Backends without custom object metadata cannot persist the marker's
    # octet-stream content type, so an empty payload with no stored content
    # type is also treated as a tombstone there.
    if artifact_util._is_rewind_tombstone(artifact) or (
        not self._is_gcs() and not data and stored_content_type is None
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
      info: Mapping[str, Any],
  ) -> ArtifactVersion:
    """Constructs an ``ArtifactVersion`` from an fsspec ``info()`` mapping."""
    stored_content_type = (
        info.get("contentType") or info.get("ContentType") or None
    )
    metadata: Mapping[str, Any] = info.get("metadata") or {}
    if metadata.get(_ADK_FILE_URI_METADATA_KEY) or metadata.get(
        _LEGACY_FILE_URI_METADATA_KEY
    ):
      mime_type = (
          metadata.get(_ADK_FILE_MIME_TYPE_METADATA_KEY) or stored_content_type
      )
    else:
      mime_type = stored_content_type or _guess_mime(filename)
    return ArtifactVersion(
        version=version,
        canonical_uri=self._canonical_uri(key),
        create_time=_parse_timestamp(info),
        mime_type=mime_type,
        custom_metadata=_user_metadata(metadata),
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
    artifact_versions = []
    for suffix, info in self._find_detail(artifact_path).items():
      version = None if "/" in suffix else _parse_version(suffix)
      if version is None:
        continue
      key = f"{artifact_path}/{version}"
      artifact_versions.append(
          self._build_artifact_version(version, key, filename, info)
      )
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

    self._fs.invalidate_cache(key)
    try:
      info = self._fs.info(key)
    except (FileNotFoundError, NotADirectoryError):
      return None
    if info.get("type", "file") != "file":
      return None
    return self._build_artifact_version(version, key, filename, info)
