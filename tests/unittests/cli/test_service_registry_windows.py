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
from unittest import mock

import pytest

from google.adk.cli import service_registry


def test_file_artifact_factory_normalizes_windows_file_uri(monkeypatch):
  monkeypatch.setattr(service_registry.os, "name", "nt", raising=False)
  mocked_url2pathname = mock.Mock(return_value=r"C:\tmp\adk artifacts")
  monkeypatch.setattr(service_registry, "url2pathname", mocked_url2pathname)

  registry = service_registry.ServiceRegistry()
  service_registry._register_builtin_services(registry)

  with mock.patch(
      "google.adk.artifacts.file_artifact_service.FileArtifactService"
  ) as mock_file_artifact_service:
    registry.create_artifact_service("file:///C:/tmp/adk%20artifacts")

  mocked_url2pathname.assert_called_once_with("/C:/tmp/adk artifacts")
  mock_file_artifact_service.assert_called_once_with(
      root_dir=Path(r"C:\tmp\adk artifacts")
  )


def test_file_artifact_factory_rejects_non_local_authority():
  registry = service_registry.ServiceRegistry()
  service_registry._register_builtin_services(registry)

  with pytest.raises(ValueError, match="local filesystem"):
    registry.create_artifact_service("file://example.com/tmp/adk_artifacts")