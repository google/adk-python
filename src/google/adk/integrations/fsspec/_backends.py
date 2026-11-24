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

"""Per-backend object creation and metadata reads for FsspecArtifactService."""

from __future__ import annotations

import abc
from collections.abc import Callable
from collections.abc import Mapping
import concurrent.futures
import dataclasses
import datetime
import logging
import re
import sys
from typing import Any
from typing import ClassVar
from typing import Optional
from typing import TYPE_CHECKING
import urllib.parse

from packaging.version import InvalidVersion
from packaging.version import Version
from typing_extensions import override

from ...errors.input_validation_error import InputValidationError
from ...platform import time as platform_time

if TYPE_CHECKING:
  import fsspec

logger = logging.getLogger("google_adk." + __name__)


_DEFAULT_MIME_TYPE = "application/octet-stream"
_GCS_PROTOCOLS = frozenset({"gs", "gcs"})
_LOCAL_PROTOCOLS = frozenset({"file", "local"})
_S3_PROTOCOLS = frozenset({"s3", "s3a"})
_AZURE_PROTOCOLS = frozenset({"abfs", "abfss", "az"})
# Azure error codes that mean an exclusive create found an existing blob.
_AZURE_CONFLICT_ERROR_CODES = frozenset(
    {"BlobAlreadyExists", "ConditionNotMet"}
)
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
# Concurrent HeadObject calls when listing S3 versions.
_S3_MAX_HEAD_WORKERS = 8

# Custom-metadata keys written by GcsArtifactService on save. They must stay
# identical so that objects written by either service are readable by both.
_ADK_DISPLAY_NAME_METADATA_KEY = "adkDisplayName"
_ADK_IS_TEXT_METADATA_KEY = "adkIsText"
_ADK_FILE_URI_METADATA_KEY = "adkFileUri"
_ADK_FILE_MIME_TYPE_METADATA_KEY = "adkFileMimeType"
# Only legacy key GcsArtifactService still reads; other snake_case names are
# indistinguishable from caller-supplied custom metadata and are not honored.
# Every backend honors it, for backward compatibility.
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
# Maps ADK's keys as s3fs returns them (lowercased, "_" as "-") back.
_S3_INTERNAL_METADATA_KEYS = {
    key.lower().replace("_", "-"): key for key in _INTERNAL_METADATA_KEYS
}
# Azure ignores case in metadata keys. Maps ADK's keys, lowercased, back.
_AZURE_INTERNAL_METADATA_KEYS = {
    key.lower(): key for key in _INTERNAL_METADATA_KEYS
}
# Metadata keys adlfs uses to tell files from directories.
_ADLFS_DIRECTORY_METADATA_KEYS = frozenset({"is_directory", "hdi_isfolder"})
# Lowercased keys caller metadata may not use on Azure, mapped to their owner.
_AZURE_RESERVED_METADATA_KEYS = {
    **{key: key for key in _ADLFS_DIRECTORY_METADATA_KEYS},
    **_AZURE_INTERNAL_METADATA_KEYS,
}
# Azure metadata keys must be C# identifiers.
_AZURE_METADATA_KEY_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# S3 metadata keys are HTTP header names (RFC 9110 tokens).
_HTTP_TOKEN_PATTERN = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")
# Metadata values the stores keep as given: printable ASCII without spaces at
# either end, which stores and clients trim from header values.
_METADATA_VALUE_PATTERN = re.compile(r"(?:[!-~](?:[ -~]*[!-~])?)?")
# Printable ASCII except space and "%", left as is when percent-encoding ADK's
# own values. Spaces are encoded because stores trim them from header values.
_ASCII_METADATA_CHARS = "".join(
    chr(code) for code in range(0x21, 0x7F) if chr(code) != "%"
)


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


def _fs_protocols(fs: fsspec.AbstractFileSystem) -> tuple[str, ...]:
  """Returns the protocol names handled by ``fs``."""
  return (fs.protocol,) if isinstance(fs.protocol, str) else tuple(fs.protocol)


def _parse_timestamp(info: Mapping[str, Any]) -> float:
  """Extracts a Unix creation timestamp from an fsspec info() mapping."""
  for field in (
      "timeCreated",
      "created",
      "creation_time",
      "mtime",
      "LastModified",
      "updated",
  ):
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


def _encode_ascii_metadata(
    metadata: Mapping[str, str],
    *,
    store: str,
    key_rule: str,
    is_valid_key: Callable[[str], Any],
    store_key: Callable[[str], str],
    reserved: Mapping[str, str],
) -> dict[str, str]:
  """Returns ``metadata`` for stores that only keep ASCII metadata (S3, Azure).

  ADK's own values, such as display names and file URIs, are percent-encoded so
  they round-trip. Caller metadata that the store would reject or change is
  rejected instead.

  Args:
    metadata: Metadata to store, including ADK's own keys.
    store: Store name for error messages.
    key_rule: The store's rule for keys, for error messages.
    is_valid_key: Returns whether the store accepts a caller key.
    store_key: Returns a key in the form the store compares keys in.
    reserved: Keys, in store form, that callers may not use, mapped to the key
      that owns them.

  Raises:
    InputValidationError: If a caller key breaks ``is_valid_key``, a caller
      value is not printable ASCII or has spaces at either end, or a caller key
      is stored as the same key as a reserved key or another caller key.
  """
  encoded = {}
  # Store-form keys of caller metadata seen so far, mapped to the caller key.
  seen: dict[str, str] = {}
  for key, value in metadata.items():
    if key in _INTERNAL_METADATA_KEYS:
      encoded[key] = urllib.parse.quote(value, safe=_ASCII_METADATA_CHARS)
      continue
    if not (is_valid_key(key) and _METADATA_VALUE_PATTERN.fullmatch(value)):
      raise InputValidationError(
          f"custom_metadata on {store} needs {key_rule} keys and printable"
          " ASCII values without spaces at either end, got"
          f" {key!r}: {value!r}."
      )
    stored_as = store_key(key)
    owner = reserved.get(stored_as) or seen.get(stored_as)
    if owner is not None:
      raise InputValidationError(
          f"custom_metadata key {key!r} is stored on {store} as the same key"
          f" as {owner!r}."
      )
    seen[stored_as] = key
    encoded[key] = value
  return encoded


@dataclasses.dataclass(frozen=True)
class _ObjectMetadata:
  """Normalized storage metadata for a single stored artifact version.

  Attributes:
    raw_metadata: Raw key/value metadata attached to the object, including any
      ADK-internal bookkeeping keys.
    content_type: The object's declared MIME type, if the backend reports one.
    create_time: Unix timestamp (seconds) when the object was created.
  """

  raw_metadata: Mapping[str, Any]
  content_type: Optional[str]
  create_time: float

  def _get_reserved(self, key: str) -> Optional[str]:
    """Returns the value of an ADK-reserved metadata key, if present."""
    value = self.raw_metadata.get(key)
    return str(value) if value is not None else None

  @property
  def file_uri(self) -> Optional[str]:
    """Returns the external or artifact reference URI stored on the object."""
    return self._get_reserved(_ADK_FILE_URI_METADATA_KEY) or self._get_reserved(
        _LEGACY_FILE_URI_METADATA_KEY
    )

  @property
  def is_text(self) -> bool:
    """Returns True if the object was saved from a ``Part(text=...)``."""
    return self._get_reserved(_ADK_IS_TEXT_METADATA_KEY) == "true"

  @property
  def display_name(self) -> Optional[str]:
    """Returns the ``Blob.display_name`` recorded on the object, if any."""
    return self._get_reserved(_ADK_DISPLAY_NAME_METADATA_KEY) or None

  @property
  def file_mime_type(self) -> Optional[str]:
    """Returns the ``FileData.mime_type`` recorded on the object, if any."""
    return self._get_reserved(_ADK_FILE_MIME_TYPE_METADATA_KEY) or None

  @property
  def user_metadata(self) -> dict[str, Any]:
    """Returns caller-supplied metadata with ADK bookkeeping keys removed."""
    return _user_metadata(self.raw_metadata)


class _BackendAdapter(abc.ABC):
  """Creates version objects and reads their metadata for one fsspec backend.

  Operations that work the same on every backend, such as ``cat_file``,
  ``find`` and ``rm_file``, stay in ``FsspecArtifactService``. Subclasses
  implement ``create_object``. Backends whose ``info()`` or listings lack
  ``metadata`` and ``contentType`` also override ``from_info`` or
  ``read_listing_metadata``.
  """

  # Whether ``create_object`` stores content type and custom metadata. If not,
  # the service rejects saves that need them.
  stores_custom_metadata: ClassVar[bool] = False

  def __init__(self, fs: fsspec.AbstractFileSystem) -> None:
    self._fs = fs

  @staticmethod
  def for_filesystem(fs: fsspec.AbstractFileSystem) -> _BackendAdapter:
    """Returns the backend adapter matching ``fs``'s protocol."""
    protocols = _fs_protocols(fs)
    if not _GCS_PROTOCOLS.isdisjoint(protocols):
      return _GcsBackendAdapter(fs)
    if not _S3_PROTOCOLS.isdisjoint(protocols):
      return _S3BackendAdapter(fs)
    if not _AZURE_PROTOCOLS.isdisjoint(protocols):
      return _AzureBackendAdapter(fs)
    return _GenericBackendAdapter(fs)

  def read_metadata(self, key: str) -> Optional[_ObjectMetadata]:
    """Reads and normalizes metadata for a single file object at ``key``."""
    self._fs.invalidate_cache(key)
    # ``from_info`` may send its own request (S3), which can race a delete.
    try:
      info = self._fs.info(key)
      if info.get("type", "file") != "file":
        return None
      return self.from_info(key, info)
    except (FileNotFoundError, NotADirectoryError):
      return None

  def from_info(self, key: str, info: Mapping[str, Any]) -> _ObjectMetadata:
    """Normalizes an ``info()`` / ``find(detail=True)`` dict for ``key``."""
    del key
    return _ObjectMetadata(
        raw_metadata=dict(info.get("metadata") or {}),
        content_type=info.get("contentType") or info.get("ContentType") or None,
        create_time=_parse_timestamp(info),
    )

  def find_detail(self, prefix: str) -> Mapping[str, Mapping[str, Any]]:
    """Returns ``find(detail=True)`` entries for objects under ``prefix``."""
    entries: Mapping[str, Mapping[str, Any]] = self._fs.find(
        prefix, detail=True
    )
    return entries

  def read_listing_metadata(
      self, entries: Mapping[str, Mapping[str, Any]]
  ) -> dict[str, _ObjectMetadata]:
    """Normalizes ``find(detail=True)`` entries, keyed by object path.

    Backends whose listings omit metadata override this to fetch it, and leave
    out objects deleted in the meantime.
    """
    return {key: self.from_info(key, info) for key, info in entries.items()}

  @abc.abstractmethod
  def create_object(
      self,
      key: str,
      data: bytes,
      *,
      content_type: Optional[str],
      metadata: Mapping[str, str],
  ) -> None:
    """Creates the version object ``key``, failing if it already exists.

    Object stores only accept content type and custom metadata when an object
    is created, so backends that store metadata write it here too.

    Raises:
      FileExistsError: If ``key`` already exists.
    """


class _GcsBackendAdapter(_BackendAdapter):
  """Reads and writes object metadata on Google Cloud Storage (``gcsfs``).

  ``gcsfs`` returns ``metadata`` and ``contentType`` in both ``info()`` and
  listings, so no per-object calls are needed.
  """

  stores_custom_metadata = True

  @override
  def create_object(
      self,
      key: str,
      data: bytes,
      *,
      content_type: Optional[str],
      metadata: Mapping[str, str],
  ) -> None:
    if self._is_rapid_bucket(key):
      self._write_rapid(key, data, content_type, metadata)
      return
    write_kwargs: dict[str, Any] = {"mode": "create"}
    if content_type is not None:
      write_kwargs["content_type"] = content_type
    if metadata:
      write_kwargs["metadata"] = dict(metadata)
    self._fs.pipe_file(key, data, **write_kwargs)

  def _is_rapid_bucket(self, key: str) -> bool:
    """Returns True when ``key`` is in a Rapid bucket.

    Rapid buckets report the ``ZONAL_HIERARCHICAL`` bucket type in gcsfs.
    """
    lookup = getattr(self._fs, "_sync_lookup_bucket_type", None)
    split_path = getattr(self._fs, "split_path", None)
    if not (callable(lookup) and callable(split_path)):
      return False
    bucket, _, _ = split_path(key)
    return str(lookup(bucket)).endswith("ZONAL_HIERARCHICAL")

  def _write_rapid(
      self,
      key: str,
      data: bytes,
      content_type: Optional[str],
      metadata: Mapping[str, str],
  ) -> None:
    """Creates ``key`` on a Rapid bucket, then patches its metadata."""
    reserved = sorted(_RAPID_SETXATTRS_RESERVED_KEYS & metadata.keys())
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
          **metadata,
      )
    except Exception:
      self._remove_failed_version(key)
      raise

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


class _S3BackendAdapter(_BackendAdapter):
  """Reads and writes object metadata on Amazon S3 (``s3fs``).

  ``s3fs`` leaves user metadata out of ``info()`` and listings, so it is read
  with ``fs.metadata()`` (``HeadObject``). Keys come back lowercased with ``_``
  turned into ``-``. ADK's own keys are mapped back, caller keys are not.

  S3 metadata must be ASCII. ADK's own values, such as display names and file
  URIs, are percent-encoded so they round-trip. Caller keys must be HTTP header
  names, and caller values printable ASCII without spaces at either end. Caller
  keys that would read back as ADK's own keys or as each other are rejected.

  Reading one version takes two HeadObject requests, ``info()`` for content
  type and time and ``metadata()`` for user metadata, because s3fs returns them
  separately.

  Exclusive create uses ``pipe_file(mode="create")``, which s3fs sends as
  ``IfNoneMatch="*"``. s3fs only guarantees this on AWS S3 with botocore newer
  than 1.35.20. An endpoint that ignores the header overwrites the object, so
  concurrent saves could replace the same version. A 409
  ``ConditionalRequestConflict`` from a concurrent conditional write is retried
  at the next version, like ``FileExistsError``.
  """

  stores_custom_metadata = True

  @override
  def from_info(self, key: str, info: Mapping[str, Any]) -> _ObjectMetadata:
    raw_metadata = {}
    for s3_key, value in (self._fs.metadata(key) or {}).items():
      name = _S3_INTERNAL_METADATA_KEYS.get(s3_key, s3_key)
      if name in _INTERNAL_METADATA_KEYS:
        value = urllib.parse.unquote(value)
      raw_metadata[name] = value
    return _ObjectMetadata(
        raw_metadata=raw_metadata,
        content_type=info.get("ContentType") or info.get("contentType") or None,
        create_time=_parse_timestamp(info),
    )

  @override
  def read_listing_metadata(
      self, entries: Mapping[str, Mapping[str, Any]]
  ) -> dict[str, _ObjectMetadata]:
    keys = list(entries)
    if not keys:
      return {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(len(keys), _S3_MAX_HEAD_WORKERS)
    ) as pool:
      results = list(pool.map(self.read_metadata, keys))
    return {
        key: metadata
        for key, metadata in zip(keys, results)
        if metadata is not None
    }

  @override
  def create_object(
      self,
      key: str,
      data: bytes,
      *,
      content_type: Optional[str],
      metadata: Mapping[str, str],
  ) -> None:
    write_kwargs: dict[str, Any] = {
        "mode": "create",
        # S3 would default to binary/octet-stream. Match GCS and Azure.
        "ContentType": content_type or _DEFAULT_MIME_TYPE,
    }
    if metadata:
      write_kwargs["Metadata"] = _encode_ascii_metadata(
          metadata,
          store="S3",
          key_rule="HTTP header name",
          is_valid_key=_HTTP_TOKEN_PATTERN.fullmatch,
          store_key=lambda key: key.lower().replace("_", "-"),
          reserved=_S3_INTERNAL_METADATA_KEYS,
      )
    try:
      self._fs.pipe_file(key, data, **write_kwargs)
    except OSError as exc:
      # s3fs maps 412 PreconditionFailed to FileExistsError but leaves 409
      # ConditionalRequestConflict as a generic OSError.
      response = getattr(exc.__cause__, "response", None) or {}
      if response.get("Error", {}).get("Code") == "ConditionalRequestConflict":
        raise FileExistsError(key) from exc
      raise


class _AzureBackendAdapter(_BackendAdapter):
  """Reads and writes object metadata on Azure Blob Storage (``adlfs``).

  ``adlfs`` returns ``metadata``, ``content_settings`` and ``creation_time`` in
  both ``info()`` and listings. Its ``pipe_file`` cannot take metadata, so
  objects are created with ``open(key, "xb", metadata=, content_settings=)``,
  which commits with ``If-None-Match: *``. This needs adlfs 2026.8.0 or newer.
  Older versions drop ``content_settings``, so saving with them is rejected.
  adlfs treats a blob with a size as a file, so its ``is_directory`` marker is
  not written. It is hidden on read, since artifacts saved through adlfs's
  ``pipe_file`` carry it.

  Azure metadata must be ASCII, and keys must be C# identifiers that are
  compared without case. ADK's own values, such as display names and file
  URIs, are percent-encoded so they round-trip. Caller metadata is rejected
  if a key is not an identifier, a value is not printable ASCII or has spaces
  at either end, or a key matches ADK's or adlfs's own keys or another caller
  key, ignoring case.
  """

  stores_custom_metadata = True

  def __init__(self, fs: fsspec.AbstractFileSystem) -> None:
    super().__init__(fs)
    # adlfs before 2026.8.0 accepts content_settings but silently drops it.
    adlfs = sys.modules.get("adlfs")
    try:
      self._drops_content_type = adlfs is not None and (
          Version(adlfs.__version__) < Version("2026.8.0")
      )
    except InvalidVersion:  # adlfs reports "UNKNOWN" when built from source.
      self._drops_content_type = False

  @override
  def read_metadata(self, key: str) -> Optional[_ObjectMetadata]:
    # adlfs invalidate_cache(key) keeps a cached parent listing, which info()
    # would trust and report key as missing.
    self._fs.invalidate_cache(self._fs._parent(key))
    return super().read_metadata(key)

  @override
  def find_detail(self, prefix: str) -> Mapping[str, Mapping[str, Any]]:
    # adlfs find(path) returns [path] when path is itself a blob, which a flat
    # namespace allows next to blobs below it (a/0 and a/0/0). Listing from
    # the container with prefix= skips that check.
    container, _, path = prefix.partition("/")
    entries: Mapping[str, Mapping[str, Any]] = self._fs.find(
        container, prefix=path, detail=True
    )
    return entries

  @override
  def from_info(self, key: str, info: Mapping[str, Any]) -> _ObjectMetadata:
    del key
    raw_metadata = {}
    for name, value in (info.get("metadata") or {}).items():
      if name.lower() in _ADLFS_DIRECTORY_METADATA_KEYS:
        continue
      name = _AZURE_INTERNAL_METADATA_KEYS.get(name.lower(), name)
      if name in _INTERNAL_METADATA_KEYS:
        value = urllib.parse.unquote(value)
      raw_metadata[name] = value
    content_settings = info.get("content_settings") or {}
    return _ObjectMetadata(
        raw_metadata=raw_metadata,
        content_type=content_settings.get("content_type") or None,
        create_time=_parse_timestamp(info),
    )

  @override
  def create_object(
      self,
      key: str,
      data: bytes,
      *,
      content_type: Optional[str],
      metadata: Mapping[str, str],
  ) -> None:
    if self._drops_content_type:
      raise ImportError(
          "Saving artifacts on Azure needs adlfs 2026.8.0 or newer. Upgrade it"
          " with `pip install -U adlfs`."
      )
    open_kwargs: dict[str, Any] = {
        "metadata": _encode_ascii_metadata(
            metadata,
            store="Azure",
            key_rule="C# identifier",
            is_valid_key=_AZURE_METADATA_KEY_PATTERN.fullmatch,
            store_key=str.lower,
            reserved=_AZURE_RESERVED_METADATA_KEYS,
        )
    }
    if content_type is not None:
      open_kwargs["content_settings"] = {"content_type": content_type}
    stream = self._fs.open(key, "xb", **open_kwargs)
    try:
      stream.write(data)
      stream.close()
    except Exception as exc:
      # Keep __del__() from calling close() and committing a failed write.
      # discard() can't, since fsspec and adlfs implement it as a no-op.
      stream.closed = True
      # adlfs maps a conflicting commit to FileExistsError. Azure errors still
      # come through raw from its fallback upload of empty payloads, or wrapped
      # in RuntimeError, such as a 412.
      for error in (exc, exc.__cause__):
        code = getattr(error, "error_code", None)
        if getattr(code, "value", code) in _AZURE_CONFLICT_ERROR_CODES:
          raise FileExistsError(key) from exc
      raise


class _GenericBackendAdapter(_BackendAdapter):
  """Fallback for backends without custom object metadata.

  Stores payload bytes only. The service rejects saves that need metadata
  before calling ``create_object``.
  """

  @override
  def create_object(
      self,
      key: str,
      data: bytes,
      *,
      content_type: Optional[str],
      metadata: Mapping[str, str],
  ) -> None:
    del content_type, metadata
    if getattr(self._fs, "auto_mkdir", True) is False:
      self._fs.makedirs(key.rsplit("/", 1)[0], exist_ok=True)
    # Exclusive create is atomic on file:// (open with "xb") and memory://.
    # Other backends inherit AbstractFileSystem.pipe_file(mode="create"), which
    # uses a non-atomic exists() check before open("wb").
    if not _LOCAL_PROTOCOLS.isdisjoint(_fs_protocols(self._fs)):
      with self._fs.open(key, "xb") as stream:
        stream.write(data)
    else:
      self._fs.pipe_file(key, data, mode="create")
