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

"""Unit tests for FsspecArtifactService read APIs and feature-flag gating.

Once save_artifact, delete_artifact, list_artifact_versions, and
get_artifact_version are implemented, FsspecArtifactService will be added into
ArtifactServiceType in test_artifact_service.py so the shared contract suite
runs uniformly across all services, keeping only fsspec-specific tests here.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from typing import Optional
from unittest import mock
import uuid

import fsspec
from google.adk.artifacts import artifact_util
from google.adk.errors.input_validation_error import InputValidationError
from google.adk.features import FeatureName
from google.adk.features._feature_registry import temporary_feature_override
from google.adk.integrations.fsspec._fsspec_artifact_service import FsspecArtifactService
from google.genai import types
import pytest

from ...artifacts.test_artifact_service import INVALID_PATH_SEGMENT_CASES
from ...artifacts.test_artifact_service import mock_gcs_artifact_service

SCOPE = {"app_name": "app", "user_id": "u", "session_id": "s"}


class _FakeObjectStore(fsspec.AbstractFileSystem):
  """In-memory filesystem whose info() mappings match gcsfs.

  Paths are normalized by fsspec's default ``_strip_protocol``, so object-store
  style roots look like ``bucket/prefix``.
  """

  protocol: Any = ("gs", "gcs")
  # fsspec's own spelling of its instance-caching switch.
  cachable = False  # codespell:ignore cachable

  def __init__(self, **kwargs: Any) -> None:
    super().__init__(**kwargs)
    self.objects: dict[str, tuple[bytes, dict[str, Any]]] = {}
    self.invalidated: list[Optional[str]] = []

  def put(self, key: str, data: bytes = b"", **info: Any) -> None:
    """Stores an object; ``info`` adds fields such as contentType/metadata."""
    self.objects[key] = (
        data,
        {"name": key, "size": len(data), "type": "file", **info},
    )

  def invalidate_cache(self, path: Optional[str] = None) -> None:
    self.invalidated.append(path)

  def info(self, path: str, **kwargs: Any) -> dict[str, Any]:
    try:
      return dict(self.objects[self._strip_protocol(path)][1])
    except KeyError:
      raise FileNotFoundError(path) from None

  def cat_file(self, path: str, start=None, end=None, **kwargs: Any) -> bytes:
    return self.objects[self._strip_protocol(path)][0]

  def find(self, path: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    stripped = self._strip_protocol(path)
    prefix = stripped + "/"
    return {
        key: dict(info)
        for key, (_, info) in sorted(self.objects.items())
        if key == stripped or key.startswith(prefix)
    }


class _FakePosixStore(_FakeObjectStore):
  """Like sftp/hdfs: absolute paths whose leading slash is significant."""

  protocol = "sftp"
  root_marker = "/"


class _FakeAsyncStore(_FakeObjectStore):
  """An async-capable filesystem created with asynchronous=True."""

  async_impl = True
  asynchronous = True


@pytest.fixture(autouse=True)
def _enable_feature() -> Iterator[None]:
  with temporary_feature_override(FeatureName.FSSPEC_ARTIFACT_SERVICE, True):
    yield


@pytest.fixture
def store() -> _FakeObjectStore:
  return _FakeObjectStore()


@pytest.fixture
def svc(store: _FakeObjectStore) -> FsspecArtifactService:
  return FsspecArtifactService("gs://b", fs=store)


def _text_meta() -> dict[str, str]:
  return {"adkIsText": "true"}


def _ref_meta(target: str) -> dict[str, str]:
  return {
      "adkFileUri": f"artifact://apps/app/users/u/sessions/s/artifacts/{target}"
  }


# ---------------------------------------------------------------------------
# Feature flag and constructor
# ---------------------------------------------------------------------------


def test_instantiation_is_blocked_when_feature_disabled() -> None:
  with temporary_feature_override(FeatureName.FSSPEC_ARTIFACT_SERVICE, False):
    with pytest.raises(RuntimeError, match="is not enabled"):
      FsspecArtifactService("memory://root")


def test_init_explains_how_to_install_missing_fsspec() -> None:
  with mock.patch.dict("sys.modules", {"fsspec": None}):
    with pytest.raises(
        ImportError, match=r'pip install "google-adk\[fsspec\]"'
    ):
      FsspecArtifactService("memory://root")


@pytest.mark.parametrize("bad_url", ["", None, 123])
def test_init_rejects_empty_or_non_string_base_url(bad_url: Any) -> None:
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
  with mock.patch(
      "fsspec.core.url_to_fs", return_value=(_FakeObjectStore(), "b")
  ) as url_to_fs:
    FsspecArtifactService("gs://b", **options)

  url_to_fs.assert_called_once_with("gs://b", **expected_options)


@pytest.mark.parametrize(
    ("make_fs", "kwargs", "match"),
    [
        (_FakeObjectStore, {"token": "anon"}, "cannot be combined"),
        (_FakePosixStore, {}, "not handled by the provided fs"),
        (_FakeAsyncStore, {}, "must be synchronous"),
    ],
)
def test_init_rejects_unusable_injected_fs(
    make_fs: type[_FakeObjectStore], kwargs: dict[str, Any], match: str
) -> None:
  with pytest.raises(ValueError, match=match):
    FsspecArtifactService("gs://b", fs=make_fs(), **kwargs)


@pytest.mark.parametrize(
    ("make_fs", "base_url", "stored_key"),
    [
        (_FakeObjectStore, "gs://b/root/", "b/root/app/u/s/f.txt/0"),
        (_FakeObjectStore, "b/root", "b/root/app/u/s/f.txt/0"),
        (_FakePosixStore, "sftp:///data", "/data/app/u/s/f.txt/0"),
        (_FakePosixStore, "sftp:///", "/app/u/s/f.txt/0"),
    ],
    ids=["bucket-prefix", "no-protocol", "posix-absolute", "posix-root"],
)
async def test_root_is_kept_as_the_filesystem_normalizes_it(
    make_fs: type[_FakeObjectStore], base_url: str, stored_key: str
) -> None:
  fs = make_fs()
  fs.put(stored_key, b"data")
  svc = FsspecArtifactService(base_url, fs=fs)

  part = await svc.load_artifact(**SCOPE, filename="f.txt")

  assert part is not None and part.inline_data is not None
  assert part.inline_data.data == b"data"
  assert await svc.list_artifact_keys(**SCOPE) == ["f.txt"]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["app_name", "user_id", "session_id"])
@pytest.mark.parametrize(("bad_value", "match"), INVALID_PATH_SEGMENT_CASES)
async def test_read_methods_reject_invalid_scope_segments(
    svc: FsspecArtifactService, field: str, bad_value: str, match: str
) -> None:
  scope = {**SCOPE, field: bad_value}

  with pytest.raises(InputValidationError, match=match):
    await svc.load_artifact(**scope, filename="f.txt")
  with pytest.raises(InputValidationError, match=match):
    await svc.list_versions(**scope, filename="f.txt")
  with pytest.raises(InputValidationError, match=match):
    await svc.list_artifact_keys(**scope)


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
async def test_read_methods_reject_unsafe_filenames(
    svc: FsspecArtifactService, filename: str, match: str
) -> None:
  with pytest.raises(InputValidationError, match=match):
    await svc.load_artifact(**SCOPE, filename=filename)
  with pytest.raises(InputValidationError, match=match):
    await svc.list_versions(**SCOPE, filename=filename)


@pytest.mark.parametrize(
    ("filename", "stored_key"),
    [
        ("docs/versions/v2.md", "b/app/u/s/docs/versions/v2.md/0"),
        ("user:apps/sessions.txt", "b/app/u/user/user:apps/sessions.txt/0"),
    ],
)
async def test_nested_filenames_may_contain_reserved_words(
    store: _FakeObjectStore,
    svc: FsspecArtifactService,
    filename: str,
    stored_key: str,
) -> None:
  """GcsArtifactService accepts these names, so they must stay readable."""
  store.put(stored_key, b"x", metadata=_text_meta())

  assert await svc.list_versions(**SCOPE, filename=filename) == [0]
  assert await svc.load_artifact(**SCOPE, filename=filename) == types.Part(
      text="x"
  )


async def test_session_scoped_reads_require_session_id(
    svc: FsspecArtifactService,
) -> None:
  scope = {**SCOPE, "session_id": None}

  with pytest.raises(InputValidationError, match="Session ID must be provided"):
    await svc.load_artifact(**scope, filename="f.txt")
  with pytest.raises(InputValidationError, match="Session ID must be provided"):
    await svc.list_versions(**scope, filename="f.txt")


async def test_user_scoped_reads_ignore_session_id(
    store: _FakeObjectStore, svc: FsspecArtifactService
) -> None:
  store.put("b/app/u/user/user:cfg.json/0", b"{}")

  for session_id in ("s", "other", None):
    part = await svc.load_artifact(
        app_name="app",
        user_id="u",
        session_id=session_id,
        filename="user:cfg.json",
    )
    assert part is not None and part.inline_data is not None


@pytest.mark.parametrize("session_id", ["user", "user/x"])
async def test_reads_allow_reserved_user_session_id(
    svc: FsspecArtifactService, session_id: str
) -> None:
  """Only saves reject a 'user' session; reads stay permissive like GCS."""
  scope = {**SCOPE, "session_id": session_id}

  assert await svc.load_artifact(**scope, filename="f.txt") is None
  assert await svc.list_versions(**scope, filename="f.txt") == []


# ---------------------------------------------------------------------------
# load_artifact
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stored_versions", "version"),
    [([], None), ([0], 1), ([0, -1], -1)],
    ids=["no-versions", "absent-version", "negative-version"],
)
async def test_load_artifact_returns_none_when_version_missing(
    store: _FakeObjectStore,
    svc: FsspecArtifactService,
    stored_versions: list[int],
    version: Optional[int],
) -> None:
  for v in stored_versions:
    store.put(f"b/app/u/s/f.txt/{v}", b"x")

  assert (
      await svc.load_artifact(**SCOPE, filename="f.txt", version=version)
      is None
  )


async def test_load_artifact_returns_none_when_version_key_is_a_directory(
    store: _FakeObjectStore, svc: FsspecArtifactService
) -> None:
  store.put("b/app/u/s/folder/0", type="directory")

  assert await svc.load_artifact(**SCOPE, filename="folder", version=0) is None


async def test_load_artifact_returns_latest_or_requested_version(
    store: _FakeObjectStore, svc: FsspecArtifactService
) -> None:
  for v in (0, 2, 10):
    store.put(f"b/app/u/s/f.txt/{v}", f"v{v}".encode(), metadata=_text_meta())

  assert await svc.load_artifact(**SCOPE, filename="f.txt") == types.Part(
      text="v10"
  )
  store.invalidated.clear()
  assert await svc.load_artifact(
      **SCOPE, filename="f.txt", version=2
  ) == types.Part(text="v2")
  assert "b/app/u/s/f.txt/2" in store.invalidated


@pytest.mark.parametrize(
    ("filename", "data", "info", "expected"),
    [
        pytest.param(
            "a.txt",
            b"hi",
            {"metadata": {"adkIsText": "true"}},
            types.Part(text="hi"),
            id="text",
        ),
        pytest.param(
            "a.txt",
            b"",
            {"metadata": {"adkIsText": "true"}},
            types.Part(text=""),
            id="empty-text",
        ),
        pytest.param(
            "a.bin",
            b"\x01",
            {"contentType": "image/png", "metadata": {"adkDisplayName": "Pic"}},
            types.Part(
                inline_data=types.Blob(
                    data=b"\x01", mime_type="image/png", display_name="Pic"
                )
            ),
            id="bytes-with-content-type-and-display-name",
        ),
        pytest.param(
            "a.bin",
            b"\x01",
            {"ContentType": "image/gif"},
            types.Part.from_bytes(data=b"\x01", mime_type="image/gif"),
            id="capitalized-content-type",
        ),
        pytest.param(
            "a.json",
            b"{}",
            {},
            types.Part.from_bytes(data=b"{}", mime_type="application/json"),
            id="no-metadata-guesses-mime",
        ),
        pytest.param(
            "a.unknownext",
            b"?",
            {},
            types.Part.from_bytes(
                data=b"?", mime_type="application/octet-stream"
            ),
            id="no-metadata-unknown-extension",
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
            id="file-reference",
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
            id="legacy-file-uri-uses-content-type",
        ),
    ],
)
async def test_load_artifact_reconstructs_part_from_object_metadata(
    store: _FakeObjectStore,
    svc: FsspecArtifactService,
    filename: str,
    data: bytes,
    info: dict[str, Any],
    expected: types.Part,
) -> None:
  store.put(f"b/app/u/s/{filename}/0", data, **info)

  assert await svc.load_artifact(**SCOPE, filename=filename) == expected


async def test_load_artifact_raises_when_text_is_not_utf8(
    store: _FakeObjectStore, svc: FsspecArtifactService
) -> None:
  """Matches GcsArtifactService, which also surfaces the decode error."""
  store.put("b/app/u/s/bad.txt/0", b"\xff\xfe", metadata=_text_meta())

  with pytest.raises(UnicodeDecodeError):
    await svc.load_artifact(**SCOPE, filename="bad.txt")


async def test_load_artifact_follows_artifact_references(
    store: _FakeObjectStore, svc: FsspecArtifactService
) -> None:
  store.put("b/app/u/s/target.txt/0", b"payload", metadata=_text_meta())
  store.put("b/app/u/s/ref/0", metadata=_ref_meta("target.txt/versions/0"))

  assert await svc.load_artifact(**SCOPE, filename="ref") == types.Part(
      text="payload"
  )


@pytest.mark.parametrize(
    ("uri", "match"),
    [
        (
            "artifact://apps/app/users/u/sessions/s2/artifacts/x/versions/0",
            "same session scope",
        ),
        (
            "artifact://apps/app/users/other/artifacts/user:x/versions/0",
            "same app and user scope",
        ),
    ],
)
async def test_load_artifact_rejects_cross_scope_references(
    store: _FakeObjectStore, svc: FsspecArtifactService, uri: str, match: str
) -> None:
  store.put("b/app/u/s/ref/0", metadata={"adkFileUri": uri})

  with pytest.raises(InputValidationError, match=match):
    await svc.load_artifact(**SCOPE, filename="ref")


async def test_load_artifact_limits_reference_chain_depth(
    store: _FakeObjectStore, svc: FsspecArtifactService
) -> None:
  max_depth = artifact_util._MAX_ARTIFACT_REFERENCE_DEPTH
  store.put("b/app/u/s/link0/0", b"leaf", metadata=_text_meta())
  for i in range(1, max_depth + 2):
    store.put(
        f"b/app/u/s/link{i}/0", metadata=_ref_meta(f"link{i - 1}/versions/0")
    )

  assert await svc.load_artifact(
      **SCOPE, filename=f"link{max_depth}"
  ) == types.Part(text="leaf")
  with pytest.raises(InputValidationError, match="maximum recursion depth"):
    await svc.load_artifact(**SCOPE, filename=f"link{max_depth + 1}")


async def test_load_artifact_propagates_backend_errors(
    store: _FakeObjectStore, svc: FsspecArtifactService
) -> None:
  """Permission or transport errors must not be reported as a missing artifact."""
  store.put("b/app/u/s/f.txt/0", b"x")

  with mock.patch.object(store, "info", side_effect=PermissionError("denied")):
    with pytest.raises(PermissionError):
      await svc.load_artifact(**SCOPE, filename="f.txt", version=0)


async def test_load_artifact_returns_none_when_deleted_before_read(
    store: _FakeObjectStore, svc: FsspecArtifactService
) -> None:
  store.put("b/app/u/s/f.txt/0", b"x")

  with mock.patch.object(store, "cat_file", side_effect=FileNotFoundError):
    assert await svc.load_artifact(**SCOPE, filename="f.txt") is None


# ---------------------------------------------------------------------------
# list_artifact_keys and list_versions
# ---------------------------------------------------------------------------


async def test_list_artifact_keys_merges_session_and_user_scopes(
    store: _FakeObjectStore, svc: FsspecArtifactService
) -> None:
  store.put("b/app/u/s/doc/0")
  store.put("b/app/u/s/doc/nested/3")
  store.put("b/app/u/s/no_version")
  store.put("b/app/u/s/bad/01")
  store.put("b/app/u/s/folder/0", type="directory")
  store.put("b/app/u/user/user:profile.json/0")
  store.put("b/app/u/s10/sibling_session/0")
  store.put("b/app/u/other/other_session/0")

  assert await svc.list_artifact_keys(**SCOPE) == [
      "doc",
      "doc/nested",
      "user:profile.json",
  ]
  assert await svc.list_artifact_keys(
      app_name="app", user_id="u", session_id=None
  ) == ["user:profile.json"]


async def test_list_versions_returns_only_canonical_versions_sorted(
    store: _FakeObjectStore, svc: FsspecArtifactService
) -> None:
  for suffix in ("10", "0", "2", "01", "1_0", "+1", " 1", "x", "\u0661"):
    store.put(f"b/app/u/s/doc/{suffix}")
  store.put("b/app/u/s/doc/nested/7")

  assert await svc.list_versions(**SCOPE, filename="doc") == [0, 2, 10]
  assert await svc.list_versions(**SCOPE, filename="doc/nested") == [7]
  assert await svc.list_versions(**SCOPE, filename="doc/0") == []
  assert "b/app/u/s/doc" in store.invalidated


async def test_listing_a_missing_directory_returns_empty(
    store: _FakeObjectStore, svc: FsspecArtifactService
) -> None:
  with mock.patch.object(store, "find", side_effect=FileNotFoundError):
    assert await svc.list_artifact_keys(**SCOPE) == []
    assert await svc.list_versions(**SCOPE, filename="f.txt") == []
    assert await svc.load_artifact(**SCOPE, filename="f.txt") is None


# ---------------------------------------------------------------------------
# Real fsspec backends
# ---------------------------------------------------------------------------


async def test_memory_backend_reads_and_lists() -> None:
  root = f"adk-fsspec-{uuid.uuid4().hex}"
  mem = fsspec.filesystem("memory")
  mem.pipe_file(f"{root}/app/u/s/notes.txt/0", b"v0")
  mem.pipe_file(f"{root}/app/u/s/notes.txt/1", b"v1")
  mem.pipe_file(f"{root}/app/u/s/nested/0/readme.txt/0", b"nested")
  mem.pipe_file(f"{root}/app/u/user/user:cfg.json/0", b"{}")
  svc = FsspecArtifactService(f"memory://{root}")

  assert await svc.list_artifact_keys(**SCOPE) == [
      "nested/0/readme.txt",
      "notes.txt",
      "user:cfg.json",
  ]
  assert await svc.list_versions(**SCOPE, filename="notes.txt") == [0, 1]
  assert await svc.list_versions(**SCOPE, filename="notes.txt/0") == []
  assert await svc.load_artifact(**SCOPE, filename="nested", version=0) is None
  assert await svc.load_artifact(
      **SCOPE, filename="notes.txt"
  ) == types.Part.from_bytes(data=b"v1", mime_type="text/plain")


async def test_local_backend_reads_and_lists(tmp_path: Path) -> None:
  artifact = tmp_path / "app" / "u" / "s" / "docs" / "data.json" / "0"
  artifact.parent.mkdir(parents=True)
  artifact.write_bytes(b"{}")
  svc = FsspecArtifactService(tmp_path.as_uri())

  assert await svc.list_artifact_keys(**SCOPE) == ["docs/data.json"]
  assert await svc.list_versions(**SCOPE, filename="docs/data.json") == [0]
  assert await svc.list_versions(**SCOPE, filename="docs/data.json/0") == []
  assert await svc.load_artifact(**SCOPE, filename="docs", version=0) is None
  assert await svc.load_artifact(
      **SCOPE, filename="docs/data.json"
  ) == types.Part.from_bytes(data=b"{}", mime_type="application/json")


# ---------------------------------------------------------------------------
# Compatibility with GcsArtifactService
# ---------------------------------------------------------------------------


async def test_reads_artifacts_written_by_gcs_artifact_service(
    store: _FakeObjectStore,
) -> None:
  gcs = mock_gcs_artifact_service()
  artifacts = {
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
      "ref.txt": types.Part(
          file_data=types.FileData(
              file_uri=(
                  "artifact://apps/app/users/u/sessions/s/artifacts/"
                  "notes.txt/versions/0"
              )
          )
      ),
  }
  for filename, part in artifacts.items():
    for _ in range(2):
      await gcs.save_artifact(**SCOPE, filename=filename, artifact=part)
  # A session rewind removes an artifact by saving a tombstone as its newest
  # version. Version 0 keeps its content; the latest version loads as absent.
  await gcs.save_artifact(
      **SCOPE, filename="rewound.txt", artifact=types.Part(text="before")
  )
  await gcs.save_artifact(
      **SCOPE,
      filename="rewound.txt",
      artifact=artifact_util._new_rewind_tombstone(),
  )
  for name, blob in gcs.bucket.blobs.items():
    store.put(
        f"test_bucket/{name}",
        blob.content,
        contentType=blob.content_type,
        metadata=blob.metadata,
    )
  svc = FsspecArtifactService("gs://test_bucket", fs=store)

  assert await svc.list_artifact_keys(**SCOPE) == await gcs.list_artifact_keys(
      **SCOPE
  )
  for filename in [*artifacts, "rewound.txt"]:
    assert await svc.list_versions(
        **SCOPE, filename=filename
    ) == await gcs.list_versions(**SCOPE, filename=filename)
    for version in (None, 0):
      assert await svc.load_artifact(
          **SCOPE, filename=filename, version=version
      ) == await gcs.load_artifact(**SCOPE, filename=filename, version=version)
  assert await svc.load_artifact(**SCOPE, filename="rewound.txt") is None
  assert await svc.load_artifact(
      **SCOPE, filename="rewound.txt", version=0
  ) == types.Part(text="before")


# ---------------------------------------------------------------------------
# Not implemented yet
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("save_artifact", {"filename": "f", "artifact": types.Part(text="x")}),
        ("delete_artifact", {"filename": "f"}),
        ("list_artifact_versions", {"filename": "f"}),
        ("get_artifact_version", {"filename": "f"}),
    ],
)
async def test_write_and_version_metadata_methods_are_not_implemented(
    svc: FsspecArtifactService, method: str, kwargs: dict[str, Any]
) -> None:
  with pytest.raises(NotImplementedError):
    await getattr(svc, method)(**SCOPE, **kwargs)
