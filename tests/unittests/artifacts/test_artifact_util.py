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

"""Tests for artifact_util."""

from google.adk.artifacts import artifact_util
from google.genai import types
import pytest


def test_parse_session_scoped_artifact_uri():
  """Tests parsing a valid session-scoped artifact URI."""
  uri = "artifact://apps/app1/users/user1/sessions/session1/artifacts/file1/versions/123"
  parsed = artifact_util.parse_artifact_uri(uri)
  assert parsed is not None
  assert parsed.app_name == "app1"
  assert parsed.user_id == "user1"
  assert parsed.session_id == "session1"
  assert parsed.filename == "file1"
  assert parsed.version == 123


def test_parse_user_scoped_artifact_uri():
  """Tests parsing a valid user-scoped artifact URI."""
  uri = "artifact://apps/app2/users/user2/artifacts/file2/versions/456"
  parsed = artifact_util.parse_artifact_uri(uri)
  assert parsed is not None
  assert parsed.app_name == "app2"
  assert parsed.user_id == "user2"
  assert parsed.session_id is None
  assert parsed.filename == "file2"
  assert parsed.version == 456


@pytest.mark.parametrize(
    "invalid_uri",
    [
        "http://example.com",
        "artifact://invalid",
        "artifact://app1/user1/sessions/session1/artifacts/file1",
        "artifact://apps/app1/users/user1/sessions/session1/artifacts/file1",
        "artifact://apps/app1/users/user1/artifacts/file1",
    ],
)
def test_parse_invalid_artifact_uri(invalid_uri):
  """Tests parsing invalid artifact URIs."""
  assert artifact_util.parse_artifact_uri(invalid_uri) is None


def test_get_session_scoped_artifact_uri():
  """Tests constructing a session-scoped artifact URI."""
  uri = artifact_util.get_artifact_uri(
      app_name="app1",
      user_id="user1",
      session_id="session1",
      filename="file1",
      version=123,
  )
  assert (
      uri
      == "artifact://apps/app1/users/user1/sessions/session1/artifacts/file1/versions/123"
  )


def test_get_user_scoped_artifact_uri():
  """Tests constructing a user-scoped artifact URI."""
  uri = artifact_util.get_artifact_uri(
      app_name="app2", user_id="user2", filename="file2", version=456
  )
  assert uri == "artifact://apps/app2/users/user2/artifacts/file2/versions/456"


def test_is_artifact_ref_true():
  """Tests is_artifact_ref with a valid artifact reference."""
  artifact = types.Part(
      file_data=types.FileData(
          file_uri="artifact://apps/a/u/s/f/v/1", mime_type="text/plain"
      )
  )
  assert artifact_util.is_artifact_ref(artifact) is True


@pytest.mark.parametrize(
    "part",
    [
        types.Part(text="hello"),
        types.Part(inline_data=types.Blob(data=b"123", mime_type="text/plain")),
        types.Part(
            file_data=types.FileData(
                file_uri="http://example.com", mime_type="text/plain"
            )
        ),
        types.Part(),
    ],
)
def test_is_artifact_ref_false(part):
  """Tests is_artifact_ref with non-reference parts."""
  assert artifact_util.is_artifact_ref(part) is False
