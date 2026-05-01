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

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from google.adk.artifacts import file_artifact_service


def test_file_uri_to_path_normalizes_windows_file_uri(monkeypatch):
  monkeypatch.setattr(file_artifact_service, "os", SimpleNamespace(name="nt"))
  mocked_url2pathname = mock.Mock(return_value=r"C:\tmp\adk artifacts")
  monkeypatch.setattr(
      file_artifact_service, "url2pathname", mocked_url2pathname
  )

  result = file_artifact_service._file_uri_to_path(
      "file:///C:/tmp/adk%20artifacts"
  )

  mocked_url2pathname.assert_called_once_with("/C:/tmp/adk artifacts")
  assert result == Path(r"C:\tmp\adk artifacts")


def test_file_uri_to_path_returns_none_for_non_file_uri():
  assert (
      file_artifact_service._file_uri_to_path("gs://bucket/adk_artifacts")
      is None
  )
