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

"""Unit tests for FsspecArtifactService fsspec-specific behaviors.

Generic BaseArtifactService contract coverage runs through
ArtifactServiceType.FSSPEC and ArtifactServiceType.FSSPEC_MEMORY in
tests/unittests/artifacts/test_artifact_service.py. This module covers package
lazy exports, feature-flag gating, filesystem/URL normalization, GCS Rapid
uploads, backend-specific metadata/timestamp parsing, real memory:// and file://
backends, and bidirectional GcsArtifactService compatibility.
"""

import asyncio
from collections.abc import Iterator
import contextlib
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import io
import logging
from pathlib import Path
import threading
from typing import Any
from typing import Optional
from unittest import mock
import uuid

from google.adk.artifacts import artifact_util
from google.adk.errors.input_validation_error import InputValidationError
from google.adk.features import FeatureName
from google.adk.features._feature_registry import temporary_feature_override
from google.adk.integrations import fsspec as fsspec_pkg
from google.adk.integrations.fsspec import _fsspec_artifact_service
from google.adk.integrations.fsspec._fsspec_artifact_service import FsspecArtifactService
from google.genai import types
import pytest

from ...artifacts.test_artifact_service import FakeFsspecObjectStore
from ...artifacts.test_artifact_service import FIXED_DATETIME
from ...artifacts.test_artifact_service import mock_gcs_artifact_service

SCOPE = {"app_name": "app", "user_id": "u", "session_id": "s"}


class _FakePosixStore(FakeFsspecObjectStore):
  """POSIX-like filesystem whose absolute paths keep a leading slash."""

  protocol = "sftp"
  root_marker = "/"


class _FakeWindowsFileStore(FakeFsspecObjectStore):
  """LocalFileSystem stand-in on Windows (drive-qualified stripped paths)."""

  protocol = ("file", "local")

  @classmethod
  def _strip_protocol(cls, path: Any) -> str:
    stripped = super()._strip_protocol(path)
    if stripped.startswith("/") and artifact_util._is_drive_qualified(
        stripped[1:]
    ):
      stripped = stripped[1:]
    return stripped

  @contextlib.contextmanager
  def open(
      self, path: str, mode: str = "rb", **kwargs: Any
  ) -> Iterator[io.BytesIO]:
    assert mode == "xb"
    stream = io.BytesIO()
    yield stream
    self.pipe_file(path, stream.getvalue(), mode="create")


@pytest.fixture(autouse=True)
def _enable_feature() -> Iterator[None]:
  with temporary_feature_override(FeatureName.FSSPEC_ARTIFACT_SERVICE, True):
    yield


@pytest.fixture
def store() -> FakeFsspecObjectStore:
  return FakeFsspecObjectStore()


@pytest.fixture
def svc(store: FakeFsspecObjectStore) -> FsspecArtifactService:
  return FsspecArtifactService("gs://b", fs=store)


# ---------------------------------------------------------------------------
# Package export, feature flag, and constructor
# ---------------------------------------------------------------------------


def test_lazy_package_export_resolves_artifact_service() -> None:
  """Accessing FsspecArtifactService on the package resolves the class."""
  assert fsspec_pkg.FsspecArtifactService is FsspecArtifactService
  assert fsspec_pkg.__all__ == ["FsspecArtifactService"]
  assert "FsspecArtifactService" in dir(fsspec_pkg)


def test_lazy_package_export_rejects_unknown_attribute() -> None:
  """Accessing an undefined symbol on the package raises AttributeError."""
  with pytest.raises(AttributeError, match="has no attribute"):
    _ = fsspec_pkg.NonExistentSymbol  # type: ignore[attr-defined]


def test_instantiation_is_blocked_when_feature_disabled() -> None:
  """Constructing FsspecArtifactService fails when its feature flag is off."""
  with temporary_feature_override(FeatureName.FSSPEC_ARTIFACT_SERVICE, False):
    with pytest.raises(RuntimeError, match="is not enabled"):
      FsspecArtifactService("memory://root")


def test_init_explains_how_to_install_missing_fsspec() -> None:
  """Constructing without fsspec installed points to google-adk[fsspec]."""
  with mock.patch.dict("sys.modules", {"fsspec": None}):
    with pytest.raises(
        ImportError, match=r'pip install "google-adk\[fsspec\]"'
    ):
      FsspecArtifactService("memory://root")


@pytest.mark.parametrize("bad_url", ["", None, 123])
def test_init_rejects_empty_or_non_string_base_url(bad_url: Any) -> None:
  """Constructing with an empty or non-string base_url raises ValueError."""
  with pytest.raises(ValueError, match="base_url"):
    FsspecArtifactService(bad_url)


@pytest.mark.parametrize(
    ("options", "expected_options"),
    [
        ({"token": "anon"}, {"token": "anon", "use_listings_cache": False}),
        ({"use_listings_cache": True}, {"use_listings_cache": True}),
    ],
)
def test_init_forwards_storage_options_to_url_to_fs(
    options: dict[str, Any], expected_options: dict[str, Any]
) -> None:
  """Storage options default use_listings_cache=False unless overridden."""
  with mock.patch(
      "fsspec.core.url_to_fs", return_value=(FakeFsspecObjectStore(), "b")
  ) as url_to_fs:
    FsspecArtifactService("gs://b", **options)

  url_to_fs.assert_called_once_with("gs://b", **expected_options)


def test_init_rejects_storage_options_when_fs_provided(
    store: FakeFsspecObjectStore,
) -> None:
  """Passing both fs and storage_options raises ValueError."""
  with pytest.raises(ValueError, match="cannot be combined"):
    FsspecArtifactService("gs://b", fs=store, token="anon")


def test_init_rejects_mismatched_base_url_protocol() -> None:
  """Passing an fs that does not handle base_url's scheme raises ValueError."""
  with pytest.raises(ValueError, match="not handled by the provided fs"):
    FsspecArtifactService("gs://b", fs=_FakePosixStore())


def test_init_rejects_async_filesystem() -> None:
  """Passing an asynchronous=True filesystem raises ValueError."""

  class _FakeAsyncStore(FakeFsspecObjectStore):
    async_impl = True
    asynchronous = True

  with pytest.raises(ValueError, match="must be synchronous"):
    FsspecArtifactService("gs://b", fs=_FakeAsyncStore())


@pytest.mark.parametrize(
    ("make_fs", "base_url", "expected_key", "expected_uri"),
    [
        (
            FakeFsspecObjectStore,
            "gs://b/root/",
            "b/root/app/u/s/f.txt/0",
            "gs://b/root/app/u/s/f.txt/0",
        ),
        (
            FakeFsspecObjectStore,
            "gcs://b/root",
            "b/root/app/u/s/f.txt/0",
            "gcs://b/root/app/u/s/f.txt/0",
        ),
        (
            FakeFsspecObjectStore,
            "b/root",
            "b/root/app/u/s/f.txt/0",
            "gs://b/root/app/u/s/f.txt/0",
        ),
        (
            _FakePosixStore,
            "sftp:///data",
            "/data/app/u/s/f.txt/0",
            "sftp:///data/app/u/s/f.txt/0",
        ),
        (
            _FakePosixStore,
            "sftp:///",
            "/app/u/s/f.txt/0",
            "sftp:///app/u/s/f.txt/0",
        ),
        (
            _FakeWindowsFileStore,
            "file:///C:/data",
            "C:/data/app/u/s/f.txt/0",
            "file:///C:/data/app/u/s/f.txt/0",
        ),
        (
            _FakeWindowsFileStore,
            "C:/data",
            "C:/data/app/u/s/f.txt/0",
            "file:///C:/data/app/u/s/f.txt/0",
        ),
    ],
    ids=[
        "gs-bucket-prefix",
        "gcs-scheme",
        "no-protocol",
        "posix-absolute",
        "posix-root",
        "windows-file-uri",
        "windows-drive-path",
    ],
)
async def test_root_and_canonical_uri_preserve_filesystem_layout(
    make_fs: type[FakeFsspecObjectStore],
    base_url: str,
    expected_key: str,
    expected_uri: str,
) -> None:
  """Saved keys and canonical URIs preserve normalized roots and URL schemes."""
  fs = make_fs()
  service = FsspecArtifactService(base_url, fs=fs)

  await service.save_artifact(
      **SCOPE,
      filename="f.txt",
      artifact=types.Part.from_bytes(data=b"data", mime_type="text/plain"),
  )
  version_meta = await service.get_artifact_version(**SCOPE, filename="f.txt")

  assert expected_key in fs.objects
  assert version_meta is not None
  assert version_meta.canonical_uri == expected_uri


# ---------------------------------------------------------------------------
# Input validation and user-namespace scoping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "match"),
    [
        ("", "must not be empty"),
        ("user:", "must not be empty"),
        ("a\x00b", "null bytes"),
        ("/abs.txt", "relative path"),
        ("\\abs.txt", "relative path"),
        ("C:/abs.txt", "relative path"),
        ("user:/abs.txt", "relative path"),
        ("..", "empty or traversal"),
        ("a/../b", "empty or traversal"),
        ("a\\..\\b", "empty or traversal"),
        ("./a", "empty or traversal"),
        ("user:../escape.txt", "empty or traversal"),
        ("doc/", "empty or traversal"),
        ("doc\\", "empty or traversal"),
        ("a//b", "empty or traversal"),
        ("user:doc/", "empty or traversal"),
    ],
)
async def test_methods_reject_unsafe_filenames(
    svc: FsspecArtifactService, filename: str, match: str
) -> None:
  """Every filename-accepting method rejects empty, absolute, or '..' paths."""
  with pytest.raises(InputValidationError, match=match):
    await svc.save_artifact(
        **SCOPE, filename=filename, artifact=types.Part(text="x")
    )
  with pytest.raises(InputValidationError, match=match):
    await svc.load_artifact(**SCOPE, filename=filename)
  with pytest.raises(InputValidationError, match=match):
    await svc.list_versions(**SCOPE, filename=filename)
  with pytest.raises(InputValidationError, match=match):
    await svc.get_artifact_version(**SCOPE, filename=filename)
  with pytest.raises(InputValidationError, match=match):
    await svc.list_artifact_versions(**SCOPE, filename=filename)
  with pytest.raises(InputValidationError, match=match):
    await svc.delete_artifact(**SCOPE, filename=filename)


@pytest.mark.parametrize(
    "filename", ["docs/versions/v2.md", "user:apps/sessions.txt"]
)
async def test_nested_filenames_may_contain_reserved_words(
    svc: FsspecArtifactService, filename: str
) -> None:
  """Nested filenames containing 'versions' or 'sessions' round-trip cleanly."""
  await svc.save_artifact(
      **SCOPE, filename=filename, artifact=types.Part(text="x")
  )

  loaded = await svc.load_artifact(**SCOPE, filename=filename)

  assert loaded == types.Part(text="x")
  assert await svc.list_versions(**SCOPE, filename=filename) == [0]


async def test_session_scoped_methods_require_session_id(
    svc: FsspecArtifactService,
) -> None:
  """Session-scoped operations raise InputValidationError without session_id."""
  scope = {"app_name": "app", "user_id": "u", "session_id": None}

  with pytest.raises(InputValidationError, match="Session ID must be provided"):
    await svc.save_artifact(
        **scope, filename="f.txt", artifact=types.Part(text="x")
    )
  with pytest.raises(InputValidationError, match="Session ID must be provided"):
    await svc.load_artifact(**scope, filename="f.txt")
  with pytest.raises(InputValidationError, match="Session ID must be provided"):
    await svc.list_versions(**scope, filename="f.txt")
  with pytest.raises(InputValidationError, match="Session ID must be provided"):
    await svc.get_artifact_version(**scope, filename="f.txt")
  with pytest.raises(InputValidationError, match="Session ID must be provided"):
    await svc.list_artifact_versions(**scope, filename="f.txt")
  with pytest.raises(InputValidationError, match="Session ID must be provided"):
    await svc.delete_artifact(**scope, filename="f.txt")


@pytest.mark.parametrize("session_id", ["", "../ignored", "user", None])
async def test_user_scoped_methods_ignore_session_id(
    svc: FsspecArtifactService, session_id: Optional[str]
) -> None:
  """User-scoped operations ignore placeholder or invalid session_id values."""
  scope = {"app_name": "app", "user_id": "u", "session_id": session_id}

  version = await svc.save_artifact(
      **scope, filename="user:cfg.json", artifact=types.Part(text="{}")
  )
  loaded = await svc.load_artifact(**scope, filename="user:cfg.json")
  versions = await svc.list_versions(**scope, filename="user:cfg.json")
  version_meta = await svc.get_artifact_version(
      **scope, filename="user:cfg.json"
  )
  all_meta = await svc.list_artifact_versions(**scope, filename="user:cfg.json")
  await svc.delete_artifact(**scope, filename="user:cfg.json")

  assert version == 0
  assert loaded == types.Part(text="{}")
  assert versions == [0]
  assert version_meta is not None and version_meta.version == 0
  assert len(all_meta) == 1
  assert await svc.load_artifact(**scope, filename="user:cfg.json") is None


@pytest.mark.parametrize("session_id", ["user", "user/x"])
async def test_version_metadata_reads_allow_reserved_user_session_id(
    svc: FsspecArtifactService, session_id: str
) -> None:
  """Version metadata reads stay permissive when session_id starts with user."""
  scope = {**SCOPE, "session_id": session_id}

  assert await svc.get_artifact_version(**scope, filename="f.txt") is None
  assert await svc.list_artifact_versions(**scope, filename="f.txt") == []


# ---------------------------------------------------------------------------
# save_artifact validation, metadata sanitization, Rapid GCS, and contention
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bad_part", "match"),
    [
        (types.Part(), "inline_data, text, or file_data"),
        (
            types.Part(
                inline_data=types.Blob(mime_type="image/png", data=None)
            ),
            "must contain data",
        ),
        (
            types.Part(file_data=types.FileData(file_uri="")),
            "must have a file_uri",
        ),
        (
            types.Part(
                file_data=types.FileData(file_uri="artifact://apps/app/invalid")
            ),
            "Invalid artifact reference URI",
        ),
    ],
)
async def test_save_artifact_rejects_malformed_part(
    svc: FsspecArtifactService, bad_part: types.Part, match: str
) -> None:
  """Saving a Part without valid payload or URI raises InputValidationError."""
  with pytest.raises(InputValidationError, match=match):
    await svc.save_artifact(**SCOPE, filename="bad.bin", artifact=bad_part)


async def test_save_artifact_strips_internal_metadata_keys(
    svc: FsspecArtifactService,
) -> None:
  """Internal ADK metadata keys in custom_metadata are ignored on save."""
  part = types.Part.from_bytes(
      data=b"REAL BYTES", mime_type="application/octet-stream"
  )
  custom_metadata = {
      "adkIsText": "true",
      "adkFileUri": "gs://attacker/other.bin",
      "adkFileMimeType": "text/plain",
      "adkDisplayName": "spoofed",
      "file_uri": "gs://attacker/legacy.bin",
      "origin": "unit-test",
  }

  await svc.save_artifact(
      **SCOPE,
      filename="data.bin",
      artifact=part,
      custom_metadata=custom_metadata,
  )
  loaded = await svc.load_artifact(**SCOPE, filename="data.bin")
  version_meta = await svc.get_artifact_version(**SCOPE, filename="data.bin")

  assert loaded == part
  assert version_meta is not None
  assert version_meta.custom_metadata == {"origin": "unit-test"}


class _FakeRapidPreconditionError(Exception):
  """Stand-in for google.api_core.exceptions.FailedPrecondition."""


_FakeRapidPreconditionError.__name__ = "FailedPrecondition"


class _FakeRapidStream:
  """Minimal gcsfs ZonalFile (Rapid bucket) stand-in with generation=0 create.

  Like gcsfs, ``aaow`` stays None until the exclusive create succeeds; the
  object then exists (unfinalized) until ``close()`` finalizes it.
  """

  def __init__(
      self,
      store: FakeFsspecObjectStore,
      key: str,
      *,
      generation: Optional[int],
      error_on_write: Optional[Exception] = None,
      error_after_create: Optional[Exception] = None,
  ) -> None:
    self._store = store
    self._key = key
    self._generation = generation
    self._error_on_write = error_on_write
    self._error_after_create = error_after_create
    self._buf = bytearray()
    self.aaow: Optional[object] = None
    self.closed = False

  def _ensure_created(self) -> None:
    if self._error_on_write is not None:
      raise self._error_on_write
    if self.aaow is not None:
      return
    if self._generation == 0 and self._key in self._store.objects:
      raise _FakeRapidPreconditionError(self._key)
    self.aaow = object()
    self._store.put(self._key, b"")  # Unfinalized object now exists.
    if self._error_after_create is not None:
      raise self._error_after_create

  def write(self, data: bytes) -> int:
    self._ensure_created()
    self._buf.extend(data)
    return len(data)

  def close(self) -> None:
    if self.closed:
      return
    self._ensure_created()
    self.closed = True
    self._store.put(self._key, bytes(self._buf))


class _FakeRapidStore(FakeFsspecObjectStore):
  """Rapid GCS fake implementing gRPC-style open(generation=0) + setxattrs."""

  def __init__(
      self,
      *,
      error_on_write: Optional[Exception] = None,
      error_after_create: Optional[Exception] = None,
      error_on_setxattrs: Optional[Exception] = None,
  ) -> None:
    super().__init__()
    self.open_calls: list[tuple[str, str, dict[str, Any]]] = []
    self.streams: list[_FakeRapidStream] = []
    self.setxattrs_calls: list[tuple[str, Optional[str], dict[str, Any]]] = []
    self.removed_keys: list[str] = []
    self._error_on_write = error_on_write
    self._error_after_create = error_after_create
    self._error_on_setxattrs = error_on_setxattrs

  def split_path(self, path: str) -> tuple[str, str, None]:
    bucket, _, key = self._strip_protocol(path).partition("/")
    return bucket, key, None

  def _sync_lookup_bucket_type(self, bucket: str) -> str:
    return "ZONAL_HIERARCHICAL"

  def open(
      self, path: str, mode: str = "rb", **kwargs: Any
  ) -> _FakeRapidStream:
    key = self._strip_protocol(path)
    self.open_calls.append((key, mode, dict(kwargs)))
    stream = _FakeRapidStream(
        self,
        key,
        generation=kwargs.get("generation"),
        error_on_write=self._error_on_write,
        error_after_create=self._error_after_create,
    )
    self.streams.append(stream)
    return stream

  def setxattrs(
      self,
      path: str,
      content_type: Optional[str] = None,
      **metadata_kwargs: Any,
  ) -> dict[str, Any]:
    self.setxattrs_calls.append((path, content_type, dict(metadata_kwargs)))
    if self._error_on_setxattrs is not None:
      raise self._error_on_setxattrs
    return super().setxattrs(path, content_type=content_type, **metadata_kwargs)

  def rm_file(self, path: str) -> None:
    self.removed_keys.append(self._strip_protocol(path))
    super().rm_file(path)


async def test_save_artifact_rapid_gcs_applies_metadata_via_setxattrs() -> None:
  """Rapid GCS uses exclusive finalized open() and patches metadata/MIME."""
  rapid = _FakeRapidStore()
  service = FsspecArtifactService("gs://rapid-bucket", fs=rapid)

  await service.save_artifact(
      **SCOPE,
      filename="note.txt",
      artifact=types.Part(text="hello"),
      custom_metadata={"env": "rapid"},
  )
  await service.save_artifact(
      **SCOPE,
      filename="ref.json",
      artifact=types.Part(file_data=types.FileData(file_uri="gs://ext/opaque")),
  )
  loaded = await service.load_artifact(**SCOPE, filename="note.txt")
  loaded_ref = await service.load_artifact(**SCOPE, filename="ref.json")
  version_meta = await service.get_artifact_version(
      **SCOPE, filename="note.txt"
  )

  assert [call[2] for call in rapid.open_calls] == [
      {"generation": 0, "finalize_on_close": True},
      {"generation": 0, "finalize_on_close": True},
  ]
  assert rapid.setxattrs_calls == [
      (
          "rapid-bucket/app/u/s/note.txt/0",
          "text/plain",
          {"env": "rapid", "adkIsText": "true"},
      ),
      (
          "rapid-bucket/app/u/s/ref.json/0",
          "application/octet-stream",
          {"adkFileUri": "gs://ext/opaque"},
      ),
  ]
  assert loaded == types.Part(text="hello")
  assert loaded_ref == types.Part(
      file_data=types.FileData(
          file_uri="gs://ext/opaque", mime_type="application/octet-stream"
      )
  )
  assert version_meta is not None
  assert version_meta.mime_type == "text/plain"
  assert version_meta.custom_metadata == {"env": "rapid"}


async def test_save_artifact_rapid_gcs_retries_on_precondition_collision() -> (
    None
):
  """Rapid generation=0 collision maps to FileExistsError and retries."""
  rapid = _FakeRapidStore()
  rapid.put(
      "rapid-bucket/app/u/s/note.txt/0",
      b"existing",
      contentType="text/plain",
      metadata={"adkIsText": "true"},
  )
  service = FsspecArtifactService("gs://rapid-bucket", fs=rapid)

  with mock.patch.object(rapid, "find", return_value={}):
    version = await service.save_artifact(
        **SCOPE, filename="note.txt", artifact=types.Part(text="second")
    )

  assert version == 1
  assert [s.closed for s in rapid.streams] == [True, True]
  assert await service.load_artifact(
      **SCOPE, filename="note.txt", version=0
  ) == types.Part(text="existing")
  assert await service.load_artifact(
      **SCOPE, filename="note.txt", version=1
  ) == types.Part(text="second")


async def test_save_artifact_rapid_gcs_reraises_error_before_create() -> None:
  """Errors before the exclusive create propagate without touching the key.

  Another writer may own the key, so nothing is removed when our create never
  succeeded.
  """
  rapid = _FakeRapidStore(error_on_write=PermissionError("denied"))
  service = FsspecArtifactService("gs://rapid-bucket", fs=rapid)

  with pytest.raises(PermissionError, match="denied"):
    await service.save_artifact(
        **SCOPE, filename="note.txt", artifact=types.Part(text="fail")
    )

  assert rapid.streams[0].closed is True
  assert rapid.removed_keys == []


async def test_save_artifact_rapid_gcs_removes_partial_object_on_failure() -> (
    None
):
  """A write failing after the exclusive create removes the partial object."""
  rapid = _FakeRapidStore(error_after_create=TimeoutError("stream reset"))
  service = FsspecArtifactService("gs://rapid-bucket", fs=rapid)

  with pytest.raises(TimeoutError, match="stream reset"):
    await service.save_artifact(
        **SCOPE, filename="note.txt", artifact=types.Part(text="partial")
    )

  assert rapid.streams[0].closed is True
  assert rapid.removed_keys == ["rapid-bucket/app/u/s/note.txt/0"]
  assert await service.list_versions(**SCOPE, filename="note.txt") == []


async def test_save_artifact_rapid_gcs_removes_object_on_setxattrs_error() -> (
    None
):
  """A metadata patch failure removes the finalized object and propagates.

  Without this, the version would survive without ``adkIsText`` and later load
  as raw bytes instead of the text that was saved.
  """
  rapid = _FakeRapidStore(error_on_setxattrs=ConnectionError("patch failed"))
  service = FsspecArtifactService("gs://rapid-bucket", fs=rapid)

  with pytest.raises(ConnectionError, match="patch failed"):
    await service.save_artifact(
        **SCOPE, filename="note.txt", artifact=types.Part(text="orphan")
    )

  assert rapid.removed_keys == ["rapid-bucket/app/u/s/note.txt/0"]
  assert await service.load_artifact(**SCOPE, filename="note.txt") is None


async def test_save_artifact_rapid_gcs_cleanup_failure_keeps_original_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
  """If removing the failed version also fails, the save error still wins."""
  rapid = _FakeRapidStore(error_on_setxattrs=ConnectionError("patch failed"))
  service = FsspecArtifactService("gs://rapid-bucket", fs=rapid)

  with mock.patch.object(
      rapid, "rm_file", side_effect=PermissionError("no delete")
  ):
    with caplog.at_level(logging.WARNING):
      with pytest.raises(ConnectionError, match="patch failed"):
        await service.save_artifact(
            **SCOPE, filename="note.txt", artifact=types.Part(text="x")
        )

  assert any(
      "Could not remove artifact version" in record.getMessage()
      for record in caplog.records
  )


@pytest.mark.parametrize(
    "reserved_key",
    [
        "self",
        "path",
        "content_type",
        "content_encoding",
        "fixed_key_metadata",
        "timeout",
    ],
)
async def test_save_artifact_rapid_gcs_rejects_reserved_setxattrs_keys(
    reserved_key: str,
) -> None:
  """Rapid GCS rejects custom_metadata keys that collide with setxattrs."""
  rapid = _FakeRapidStore()
  service = FsspecArtifactService("gs://rapid-bucket", fs=rapid)

  with pytest.raises(InputValidationError, match=reserved_key):
    await service.save_artifact(
        **SCOPE,
        filename="note.txt",
        artifact=types.Part(text="hello"),
        custom_metadata={reserved_key: "collide"},
    )

  assert rapid.open_calls == []
  assert rapid.setxattrs_calls == []


async def test_save_artifact_standard_gcs_lookup_passes_upload_kwargs() -> None:
  """Non-Rapid gcsfs buckets pass content_type and metadata to pipe_file."""

  class _FakeRegionalGcsStore(FakeFsspecObjectStore):

    def split_path(self, path: str) -> tuple[str, str, None]:
      bucket, _, key = self._strip_protocol(path).partition("/")
      return bucket, key, None

    def _sync_lookup_bucket_type(self, bucket: str) -> str:
      return "REGIONAL"

  regional = _FakeRegionalGcsStore()
  service = FsspecArtifactService("gs://std-bucket", fs=regional)

  await service.save_artifact(
      **SCOPE,
      filename="raw.bin",
      artifact=types.Part(inline_data=types.Blob(data=b"bytes")),
  )
  loaded = await service.load_artifact(**SCOPE, filename="raw.bin")

  assert loaded == types.Part.from_bytes(
      data=b"bytes", mime_type="application/octet-stream"
  )


async def test_save_artifact_reserves_concurrent_versions(
    store: FakeFsspecObjectStore, svc: FsspecArtifactService
) -> None:
  """Overlapping saves retry on FileExistsError and pick distinct versions."""
  original_find = store.find
  first_reads = threading.Barrier(2)
  calls_lock = threading.Lock()
  synchronized_calls = 0

  def synchronize_initial_finds(
      path: str, detail: bool = False, **kwargs: Any
  ) -> Any:
    nonlocal synchronized_calls
    found = original_find(path, detail=detail, **kwargs)
    with calls_lock:
      synchronized_calls += 1
      should_wait = synchronized_calls <= 2
    if should_wait:
      first_reads.wait(timeout=5)
    return found

  with mock.patch.object(store, "find", side_effect=synchronize_initial_finds):
    saved_versions = await asyncio.gather(
        svc.save_artifact(
            **SCOPE, filename="report.txt", artifact=types.Part(text="first")
        ),
        svc.save_artifact(
            **SCOPE, filename="report.txt", artifact=types.Part(text="second")
        ),
    )

  loaded_parts = [
      await svc.load_artifact(**SCOPE, filename="report.txt", version=v)
      for v in saved_versions
  ]
  loaded_texts = {part.text for part in loaded_parts if part is not None}
  assert sorted(saved_versions) == [0, 1]
  assert loaded_texts == {"first", "second"}


async def test_save_artifact_bounds_version_retries(
    store: FakeFsspecObjectStore, svc: FsspecArtifactService
) -> None:
  """Sustained version collisions stop retrying and raise FileExistsError."""
  for version in range(20):
    store.put(f"b/app/u/s/report.txt/{version}", b"taken")

  with mock.patch.object(store, "find", return_value={}):
    with mock.patch.object(
        store, "pipe_file", wraps=store.pipe_file
    ) as pipe_calls:
      with pytest.raises(FileExistsError):
        await svc.save_artifact(
            **SCOPE, filename="report.txt", artifact=types.Part(text="new")
        )

  assert pipe_calls.call_count == 10


# ---------------------------------------------------------------------------
# Read, delete, and version-metadata edge cases
# ---------------------------------------------------------------------------


async def test_load_artifact_defaults_to_highest_version(
    store: FakeFsspecObjectStore, svc: FsspecArtifactService
) -> None:
  """Loading without an explicit version returns the highest version."""
  for version in (0, 2, 10):
    store.put(
        f"b/app/u/s/f.txt/{version}",
        f"v{version}".encode(),
        metadata={"adkIsText": "true"},
    )

  loaded = await svc.load_artifact(**SCOPE, filename="f.txt")

  assert loaded == types.Part(text="v10")


async def test_load_artifact_invalidates_cache_for_requested_version(
    store: FakeFsspecObjectStore, svc: FsspecArtifactService
) -> None:
  """Loading an explicit version invalidates its cache entry before reading."""
  store.put("b/app/u/s/f.txt/2", b"v2", metadata={"adkIsText": "true"})

  loaded = await svc.load_artifact(**SCOPE, filename="f.txt", version=2)

  assert loaded == types.Part(text="v2")
  assert "b/app/u/s/f.txt/2" in store.invalidated


async def test_untyped_file_data_preserves_none_mime_type(
    svc: FsspecArtifactService,
) -> None:
  """Saving file_data without mime_type keeps mime_type=None on read."""
  artifact = types.Part(
      file_data=types.FileData(file_uri="gs://my-bucket/data.bin")
  )

  await svc.save_artifact(**SCOPE, filename="data.bin", artifact=artifact)
  loaded = await svc.load_artifact(**SCOPE, filename="data.bin")
  version_meta = await svc.get_artifact_version(**SCOPE, filename="data.bin")

  assert loaded == types.Part(
      file_data=types.FileData(
          file_uri="gs://my-bucket/data.bin", mime_type=None
      )
  )
  assert version_meta is not None
  assert version_meta.mime_type is None


@pytest.mark.parametrize(
    ("filename", "data", "info", "expected_part", "expected_mime"),
    [
        pytest.param(
            "a.bin",
            b"\x01",
            {"ContentType": "image/gif"},
            types.Part.from_bytes(data=b"\x01", mime_type="image/gif"),
            "image/gif",
            id="capitalized-content-type",
        ),
        pytest.param(
            "a.json",
            b"{}",
            {},
            types.Part.from_bytes(data=b"{}", mime_type="application/json"),
            "application/json",
            id="no-metadata-guesses-mime",
        ),
        pytest.param(
            "user:a.json",
            b"{}",
            {},
            types.Part.from_bytes(data=b"{}", mime_type="application/json"),
            "application/json",
            id="user-namespace-guesses-mime",
        ),
        pytest.param(
            "a.unknownext",
            b"?",
            {},
            types.Part.from_bytes(
                data=b"?", mime_type="application/octet-stream"
            ),
            "application/octet-stream",
            id="unknown-extension-octet-stream",
        ),
        pytest.param(
            "a.bin",
            b"\x01",
            {
                "metadata": {
                    "is_text": "true",
                    "display_name": "x",
                    "adkIsText": "TRUE",
                }
            },
            types.Part.from_bytes(
                data=b"\x01", mime_type="application/octet-stream"
            ),
            "application/octet-stream",
            id="only-exact-adk-keys-are-honored",
        ),
        pytest.param(
            "a.mp4",
            b"",
            {
                "contentType": "text/plain",
                "metadata": {
                    "adkFileUri": "gs://x/v.mp4",
                    "adkFileMimeType": "video/mp4",
                    "adkDisplayName": "ignored",
                },
            },
            types.Part(
                file_data=types.FileData(
                    file_uri="gs://x/v.mp4", mime_type="video/mp4"
                )
            ),
            "video/mp4",
            id="file-reference-prefers-adk-mime",
        ),
        pytest.param(
            "a.pdf",
            b"",
            {
                "contentType": "application/pdf",
                "metadata": {"file_uri": "gs://x/y.pdf"},
            },
            types.Part(
                file_data=types.FileData(
                    file_uri="gs://x/y.pdf", mime_type="application/pdf"
                )
            ),
            "application/pdf",
            id="legacy-file-uri-uses-content-type",
        ),
    ],
)
async def test_load_and_version_metadata_reconstruct_from_object_metadata(
    store: FakeFsspecObjectStore,
    svc: FsspecArtifactService,
    filename: str,
    data: bytes,
    info: dict[str, Any],
    expected_part: types.Part,
    expected_mime: str,
) -> None:
  """Object metadata fields reconstruct matching Part and ArtifactVersion."""
  scope_segment = "user" if filename.startswith("user:") else "s"
  store.put(f"b/app/u/{scope_segment}/{filename}/0", data, **info)

  loaded = await svc.load_artifact(**SCOPE, filename=filename)
  version_meta = await svc.get_artifact_version(**SCOPE, filename=filename)

  assert loaded == expected_part
  assert version_meta is not None
  assert version_meta.mime_type == expected_mime


async def test_load_artifact_raises_when_text_is_not_utf8(
    store: FakeFsspecObjectStore, svc: FsspecArtifactService
) -> None:
  """Corrupt UTF-8 bytes marked as text surface UnicodeDecodeError."""
  store.put("b/app/u/s/bad.txt/0", b"\xff\xfe", metadata={"adkIsText": "true"})

  with pytest.raises(UnicodeDecodeError):
    await svc.load_artifact(**SCOPE, filename="bad.txt")


async def test_non_file_version_entries_are_ignored(
    store: FakeFsspecObjectStore, svc: FsspecArtifactService
) -> None:
  """Directory entries at version keys are ignored by read and list methods."""
  store.put("b/app/u/s/folder/0", type="directory")

  assert await svc.load_artifact(**SCOPE, filename="folder", version=0) is None
  assert (
      await svc.get_artifact_version(**SCOPE, filename="folder", version=0)
      is None
  )
  assert await svc.list_versions(**SCOPE, filename="folder") == []
  assert await svc.list_artifact_versions(**SCOPE, filename="folder") == []
  assert await svc.list_artifact_keys(**SCOPE) == []


async def test_non_canonical_version_suffixes_are_ignored(
    store: FakeFsspecObjectStore, svc: FsspecArtifactService
) -> None:
  """Zero-padded, signed, underscored, or non-ASCII suffixes are skipped."""
  for suffix in ("10", "0", "2", "01", "1_0", "+1", " 1", "x", "\u0661"):
    store.put(f"b/app/u/s/doc/{suffix}")
  store.put("b/app/u/s/no_version")

  versions = await svc.list_versions(**SCOPE, filename="doc")
  version_metas = await svc.list_artifact_versions(**SCOPE, filename="doc")
  keys = await svc.list_artifact_keys(**SCOPE)

  assert versions == [0, 2, 10]
  assert [v.version for v in version_metas] == [0, 2, 10]
  assert keys == ["doc"]


async def test_load_artifact_propagates_backend_errors(
    store: FakeFsspecObjectStore, svc: FsspecArtifactService
) -> None:
  """Permission or transport errors propagate instead of returning None."""
  store.put("b/app/u/s/f.txt/0", b"x")

  with mock.patch.object(store, "info", side_effect=PermissionError("denied")):
    with pytest.raises(PermissionError):
      await svc.load_artifact(**SCOPE, filename="f.txt", version=0)


async def test_load_artifact_returns_none_when_deleted_before_read(
    store: FakeFsspecObjectStore, svc: FsspecArtifactService
) -> None:
  """An object deleted between info() and cat_file() loads as None."""
  store.put("b/app/u/s/f.txt/0", b"x")

  with mock.patch.object(store, "cat_file", side_effect=FileNotFoundError):
    assert await svc.load_artifact(**SCOPE, filename="f.txt") is None


async def test_delete_artifact_tolerates_concurrent_removal(
    store: FakeFsspecObjectStore, svc: FsspecArtifactService
) -> None:
  """Deleting an artifact succeeds even if rm_file races with another delete."""
  store.put("b/app/u/s/f.txt/0", b"x")

  with mock.patch.object(store, "rm_file", side_effect=FileNotFoundError):
    await svc.delete_artifact(**SCOPE, filename="f.txt")


async def test_listing_missing_prefix_returns_empty(
    store: FakeFsspecObjectStore, svc: FsspecArtifactService
) -> None:
  """FileNotFoundError from find() produces empty listings and None reads."""
  with mock.patch.object(store, "find", side_effect=FileNotFoundError):
    assert await svc.list_artifact_keys(**SCOPE) == []
    assert await svc.list_versions(**SCOPE, filename="f.txt") == []
    assert await svc.list_artifact_versions(**SCOPE, filename="f.txt") == []
    assert await svc.load_artifact(**SCOPE, filename="f.txt") is None


@pytest.mark.parametrize(
    ("info_fields", "expected_timestamp"),
    [
        ({"mtime": 1700000000.5}, 1700000000.5),
        (
            {"created": datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)},
            datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp(),
        ),
        (
            {"LastModified": datetime(2025, 1, 1, 12, 0, 0)},
            datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp(),
        ),
        (
            {"timeCreated": "2025-01-01T12:00:00Z"},
            datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp(),
        ),
        (
            {"timeCreated": "2025-01-01T14:00:00+02:00"},
            datetime(
                2025, 1, 1, 14, 0, 0, tzinfo=timezone(timedelta(hours=2))
            ).timestamp(),
        ),
        (
            {"updated": "2025-01-01T12:00:00"},
            datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp(),
        ),
        (
            {"timeCreated": "not-a-timestamp", "mtime": 1234.0},
            1234.0,
        ),
        ({"timeCreated": None}, 9876.5),
    ],
)
async def test_get_artifact_version_parses_backend_timestamps(
    store: FakeFsspecObjectStore,
    svc: FsspecArtifactService,
    info_fields: dict[str, Any],
    expected_timestamp: float,
) -> None:
  """Backend timestamp formats normalize to Unix epoch seconds."""
  store.put("b/app/u/s/f.txt/0", b"x", timeCreated=None)
  store.objects["b/app/u/s/f.txt/0"][1].update(info_fields)

  with mock.patch.object(
      _fsspec_artifact_service.platform_time,
      "get_time",
      return_value=9876.5,
  ):
    version_meta = await svc.get_artifact_version(**SCOPE, filename="f.txt")

  assert version_meta is not None
  assert version_meta.create_time == expected_timestamp


# ---------------------------------------------------------------------------
# Real fsspec backends (memory:// and file://)
# ---------------------------------------------------------------------------


async def test_memory_backend_crud_and_version_metadata() -> None:
  """Real memory:// backend supports save, load, list, metadata, and delete."""
  root = f"adk-fsspec-{uuid.uuid4().hex}"
  service = FsspecArtifactService(f"memory://{root}")

  v0 = await service.save_artifact(
      **SCOPE,
      filename="notes.txt",
      artifact=types.Part.from_bytes(data=b"v0", mime_type="text/plain"),
  )
  v1 = await service.save_artifact(
      **SCOPE,
      filename="notes.txt",
      artifact=types.Part.from_bytes(data=b"v1", mime_type="text/plain"),
  )
  await service.save_artifact(
      **SCOPE,
      filename="user:cfg.json",
      artifact=types.Part.from_bytes(data=b"{}", mime_type="application/json"),
  )
  keys = await service.list_artifact_keys(**SCOPE)
  versions = await service.list_versions(**SCOPE, filename="notes.txt")
  latest = await service.load_artifact(**SCOPE, filename="notes.txt")
  latest_meta = await service.get_artifact_version(
      **SCOPE, filename="notes.txt"
  )
  all_metas = await service.list_artifact_versions(
      **SCOPE, filename="notes.txt"
  )
  await service.delete_artifact(**SCOPE, filename="notes.txt")
  after_delete = await service.load_artifact(**SCOPE, filename="notes.txt")

  assert (v0, v1) == (0, 1)
  assert keys == ["notes.txt", "user:cfg.json"]
  assert versions == [0, 1]
  assert latest == types.Part.from_bytes(data=b"v1", mime_type="text/plain")
  assert latest_meta is not None and latest_meta.version == 1
  assert latest_meta.canonical_uri.startswith("memory://")
  assert [av.version for av in all_metas] == [0, 1]
  assert after_delete is None


async def test_local_backend_crud_and_version_metadata(tmp_path: Path) -> None:
  """Real file:// backend creates parent dirs and supports full CRUD."""
  service = FsspecArtifactService(tmp_path.as_uri())

  v0 = await service.save_artifact(
      **SCOPE,
      filename="docs/data.json",
      artifact=types.Part.from_bytes(data=b"{}", mime_type="application/json"),
  )
  loaded = await service.load_artifact(**SCOPE, filename="docs/data.json")
  version_meta = await service.get_artifact_version(
      **SCOPE, filename="docs/data.json", version=0
  )
  all_metas = await service.list_artifact_versions(
      **SCOPE, filename="docs/data.json"
  )
  await service.delete_artifact(**SCOPE, filename="docs/data.json")
  after_delete = await service.load_artifact(**SCOPE, filename="docs/data.json")

  assert v0 == 0
  assert loaded == types.Part.from_bytes(
      data=b"{}", mime_type="application/json"
  )
  assert version_meta is not None
  assert version_meta.canonical_uri == (
      f"{tmp_path.as_uri()}/app/u/s/docs/data.json/0"
  )
  assert len(all_metas) == 1
  assert after_delete is None


@pytest.mark.parametrize(
    "descendant_name", ["docs/data.json/0", "docs/data.json/0/child"]
)
async def test_local_backend_handles_file_ancestor_as_absent(
    tmp_path: Path, descendant_name: str
) -> None:
  """Querying beneath a regular version file on file:// avoids ENOTDIR."""
  service = FsspecArtifactService(tmp_path.as_uri())
  await service.save_artifact(
      **SCOPE,
      filename="docs/data.json",
      artifact=types.Part.from_bytes(data=b"{}", mime_type="application/json"),
  )

  loaded = await service.load_artifact(
      **SCOPE, filename=descendant_name, version=0
  )
  version_meta = await service.get_artifact_version(
      **SCOPE, filename=descendant_name, version=0
  )
  versions = await service.list_versions(**SCOPE, filename=descendant_name)
  all_metas = await service.list_artifact_versions(
      **SCOPE, filename=descendant_name
  )
  await service.delete_artifact(**SCOPE, filename=descendant_name)
  parent_loaded = await service.load_artifact(
      **SCOPE, filename="docs/data.json", version=0
  )

  assert loaded is None
  assert version_meta is None
  assert versions == []
  assert all_metas == []
  assert parent_loaded == types.Part.from_bytes(
      data=b"{}", mime_type="application/json"
  )


async def test_memory_backend_rejects_metadata_and_honors_tombstone() -> None:
  """Real memory:// rejects metadata inputs and honors rewind tombstones."""
  root = f"adk-fsspec-{uuid.uuid4().hex}"
  service = FsspecArtifactService(f"memory://{root}")
  external_ref = types.Part(
      file_data=types.FileData(file_uri="gs://other/doc.pdf")
  )
  artifact_ref = types.Part(
      file_data=types.FileData(
          file_uri=(
              "artifact://apps/app/users/u/sessions/s/artifacts/"
              "notes.txt/versions/0"
          )
      )
  )

  for unsupported_name, unsupported_part, metadata in (
      ("notes.txt", types.Part(text="hello"), None),
      ("notes", types.Part(text=""), None),
      ("external.pdf", external_ref, None),
      ("ref.txt", artifact_ref, None),
      (
          "notes.txt",
          types.Part.from_bytes(data=b"before", mime_type="text/plain"),
          {"env": "memory"},
      ),
  ):
    with pytest.raises(
        InputValidationError, match="persists custom object metadata"
    ):
      await service.save_artifact(
          **SCOPE,
          filename=unsupported_name,
          artifact=unsupported_part,
          custom_metadata=metadata,
      )
  assert await service.list_artifact_keys(**SCOPE) == []

  await service.save_artifact(
      **SCOPE,
      filename="notes.txt",
      artifact=types.Part(
          inline_data=types.Blob(
              data=b"before",
              mime_type="text/plain",
              display_name="notes.txt",
          )
      ),
      custom_metadata={},
  )
  tombstone_version = await service.save_artifact(
      **SCOPE,
      filename="notes.txt",
      artifact=artifact_util._new_rewind_tombstone(),
  )
  latest = await service.load_artifact(**SCOPE, filename="notes.txt")
  prior = await service.load_artifact(**SCOPE, filename="notes.txt", version=0)
  version_meta = await service.get_artifact_version(
      **SCOPE, filename="notes.txt", version=0
  )

  assert tombstone_version == 1
  assert latest is None
  assert prior == types.Part.from_bytes(data=b"before", mime_type="text/plain")
  assert version_meta is not None
  assert version_meta.custom_metadata == {}


async def test_local_backend_rejects_metadata_and_honors_tombstone(
    tmp_path: Path,
) -> None:
  """Real file:// rejects metadata inputs and honors rewind tombstones."""
  service = FsspecArtifactService(tmp_path.as_uri())

  for unsupported_name, unsupported_part, metadata in (
      ("notes.txt", types.Part(text="hello"), None),
      ("notes", types.Part(text=""), None),
      (
          "external.pdf",
          types.Part(file_data=types.FileData(file_uri="gs://other/doc.pdf")),
          None,
      ),
      (
          "notes.txt",
          types.Part.from_bytes(data=b"before", mime_type="text/plain"),
          {"env": "local"},
      ),
  ):
    with pytest.raises(
        InputValidationError, match="persists custom object metadata"
    ):
      await service.save_artifact(
          **SCOPE,
          filename=unsupported_name,
          artifact=unsupported_part,
          custom_metadata=metadata,
      )
  assert await service.list_artifact_keys(**SCOPE) == []

  await service.save_artifact(
      **SCOPE,
      filename="notes.txt",
      artifact=types.Part(
          inline_data=types.Blob(
              data=b"before",
              mime_type="text/plain",
              display_name="notes.txt",
          )
      ),
      custom_metadata={},
  )
  tombstone_version = await service.save_artifact(
      **SCOPE,
      filename="notes.txt",
      artifact=artifact_util._new_rewind_tombstone(),
  )
  latest = await service.load_artifact(**SCOPE, filename="notes.txt")
  prior = await service.load_artifact(**SCOPE, filename="notes.txt", version=0)

  assert tombstone_version == 1
  assert latest is None
  assert prior == types.Part.from_bytes(data=b"before", mime_type="text/plain")


async def test_non_gcs_empty_object_with_stored_content_type_is_preserved() -> (
    None
):
  """Empty non-GCS object carrying an explicit non-tombstone MIME loads back."""
  posix = _FakePosixStore()
  posix.put("/data/app/u/s/empty.txt/0", b"", ContentType="text/plain")
  service = FsspecArtifactService("sftp:///data", fs=posix)

  loaded = await service.load_artifact(**SCOPE, filename="empty.txt")

  assert loaded == types.Part.from_bytes(data=b"", mime_type="text/plain")


async def test_local_backend_reserves_concurrent_versions_atomically(
    tmp_path: Path,
) -> None:
  """Overlapping file:// saves use O_EXCL open('xb') without overwriting."""
  service = FsspecArtifactService(tmp_path.as_uri())
  original_find = service._fs.find
  first_reads = threading.Barrier(2)
  calls_lock = threading.Lock()
  synchronized_calls = 0

  def synchronize_initial_finds(
      path: str, detail: bool = False, **kwargs: Any
  ) -> Any:
    nonlocal synchronized_calls
    found = original_find(path, detail=detail, **kwargs)
    with calls_lock:
      synchronized_calls += 1
      should_wait = synchronized_calls <= 2
    if should_wait:
      first_reads.wait(timeout=5)
    return found

  with mock.patch.object(
      service._fs, "find", side_effect=synchronize_initial_finds
  ):
    saved_versions = await asyncio.gather(
        service.save_artifact(
            **SCOPE,
            filename="report.txt",
            artifact=types.Part.from_bytes(
                data=b"first", mime_type="text/plain"
            ),
        ),
        service.save_artifact(
            **SCOPE,
            filename="report.txt",
            artifact=types.Part.from_bytes(
                data=b"second", mime_type="text/plain"
            ),
        ),
    )

  loaded_parts = [
      await service.load_artifact(**SCOPE, filename="report.txt", version=v)
      for v in saved_versions
  ]
  loaded_bytes = {
      part.inline_data.data
      for part in loaded_parts
      if part is not None and part.inline_data is not None
  }
  assert sorted(saved_versions) == [0, 1]
  assert loaded_bytes == {b"first", b"second"}


# ---------------------------------------------------------------------------
# Bidirectional compatibility with GcsArtifactService
# ---------------------------------------------------------------------------


def _compatibility_artifacts() -> dict[str, types.Part]:
  return {
      "notes.txt": types.Part(text="hello"),
      "photo.png": types.Part(
          inline_data=types.Blob(
              data=b"\x89PNG", mime_type="image/png", display_name="Diagram"
          )
      ),
      "user:shared.txt": types.Part(text="shared"),
      "docs/versions/v2.md": types.Part(text="nested"),
      "external.pdf": types.Part(
          file_data=types.FileData(
              file_uri="gs://other/x.pdf", mime_type="application/pdf"
          )
      ),
      "untyped.bin": types.Part(
          file_data=types.FileData(file_uri="gs://other/raw.bin")
      ),
      "ref.txt": types.Part(
          file_data=types.FileData(
              file_uri=(
                  "artifact://apps/app/users/u/sessions/s/artifacts/"
                  "notes.txt/versions/0"
              )
          )
      ),
  }


async def test_reads_and_versions_match_artifacts_written_by_gcs_service(
    store: FakeFsspecObjectStore,
) -> None:
  """Objects saved by GcsArtifactService load identically through fsspec."""
  gcs = mock_gcs_artifact_service()
  artifacts = _compatibility_artifacts()
  for filename, part in artifacts.items():
    for index in range(2):
      await gcs.save_artifact(
          **SCOPE,
          filename=filename,
          artifact=part,
          custom_metadata={"step": str(index)},
      )
  await gcs.save_artifact(
      **SCOPE, filename="rewound.txt", artifact=types.Part(text="before")
  )
  await gcs.save_artifact(
      **SCOPE,
      filename="rewound.txt",
      artifact=artifact_util._new_rewind_tombstone(),
  )
  for name, blob in gcs.bucket.blobs.items():
    info: dict[str, Any] = {}
    if blob.content_type is not None:
      info["contentType"] = blob.content_type
    if blob.metadata:
      info["metadata"] = dict(blob.metadata)
    store.put(f"test_bucket/{name}", blob.content or b"", **info)
  service = FsspecArtifactService("gs://test_bucket", fs=store)

  assert await service.list_artifact_keys(
      **SCOPE
  ) == await gcs.list_artifact_keys(**SCOPE)
  for filename in [*artifacts, "rewound.txt"]:
    assert await service.list_versions(
        **SCOPE, filename=filename
    ) == await gcs.list_versions(**SCOPE, filename=filename)
    assert await service.list_artifact_versions(
        **SCOPE, filename=filename
    ) == await gcs.list_artifact_versions(**SCOPE, filename=filename)
    for version in (None, 0):
      assert await service.load_artifact(
          **SCOPE, filename=filename, version=version
      ) == await gcs.load_artifact(**SCOPE, filename=filename, version=version)
      assert await service.get_artifact_version(
          **SCOPE, filename=filename, version=version
      ) == await gcs.get_artifact_version(
          **SCOPE, filename=filename, version=version
      )


async def test_gcs_service_reads_and_versions_match_artifacts_written_by_fsspec(
    store: FakeFsspecObjectStore,
) -> None:
  """Objects saved by FsspecArtifactService load identically through GCS."""
  service = FsspecArtifactService("gs://test_bucket", fs=store)
  artifacts = _compatibility_artifacts()
  for filename, part in artifacts.items():
    for index in range(2):
      await service.save_artifact(
          **SCOPE,
          filename=filename,
          artifact=part,
          custom_metadata={"step": str(index)},
      )
  gcs = mock_gcs_artifact_service()
  for key, (data, info) in store.objects.items():
    _, _, blob_name = key.partition("/")
    blob = gcs.bucket.blob(blob_name)
    blob.time_created = FIXED_DATETIME
    if info.get("metadata"):
      blob.metadata = dict(info["metadata"])
    blob.upload_from_string(data, content_type=info.get("contentType"))

  assert await gcs.list_artifact_keys(
      **SCOPE
  ) == await service.list_artifact_keys(**SCOPE)
  for filename in artifacts:
    assert await gcs.list_versions(
        **SCOPE, filename=filename
    ) == await service.list_versions(**SCOPE, filename=filename)
    assert await gcs.list_artifact_versions(
        **SCOPE, filename=filename
    ) == await service.list_artifact_versions(**SCOPE, filename=filename)
    for version in (None, 0):
      assert await gcs.load_artifact(
          **SCOPE, filename=filename, version=version
      ) == await service.load_artifact(
          **SCOPE, filename=filename, version=version
      )
      assert await gcs.get_artifact_version(
          **SCOPE, filename=filename, version=version
      ) == await service.get_artifact_version(
          **SCOPE, filename=filename, version=version
      )
