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

import os
from typing import Optional
from unittest import mock

from google.adk.telemetry.google_cloud import get_gcp_exporters
from google.adk.telemetry.google_cloud import get_gcp_resource
import pytest


@pytest.mark.parametrize("enable_cloud_tracing", [True, False])
@pytest.mark.parametrize("enable_cloud_metrics", [True, False])
@pytest.mark.parametrize("enable_cloud_logging", [True, False])
def test_get_gcp_exporters(
    enable_cloud_tracing: bool,
    enable_cloud_metrics: bool,
    enable_cloud_logging: bool,
    monkeypatch: pytest.MonkeyPatch,
):
  """
  Test initializing correct providers in setup_otel
  when enabling telemetry via Google O11y.
  """
  # Arrange.
  # Mocking google.auth.default to improve the test time.
  auth_mock = mock.MagicMock()
  auth_mock.return_value = ("", "project-id")
  monkeypatch.setattr(
      "google.auth.default",
      auth_mock,
  )

  # Act.
  otel_hooks = get_gcp_exporters(
      enable_cloud_tracing=enable_cloud_tracing,
      enable_cloud_metrics=enable_cloud_metrics,
      enable_cloud_logging=enable_cloud_logging,
  )

  # Assert.
  # If given telemetry type was enabled,
  # the corresponding provider should be set.
  assert len(otel_hooks.span_processors) == (1 if enable_cloud_tracing else 0)
  assert len(otel_hooks.metric_readers) == (1 if enable_cloud_metrics else 0)
  assert len(otel_hooks.log_record_processors) == (
      1 if enable_cloud_logging else 0
  )


@pytest.mark.parametrize("project_id_in_arg", ["project_id_in_arg", None])
@pytest.mark.parametrize("project_id_on_env", ["project_id_on_env", None])
def test_get_gcp_resource(
    project_id_in_arg: Optional[str],
    project_id_on_env: Optional[str],
    monkeypatch: pytest.MonkeyPatch,
):
  # Arrange.
  if project_id_on_env is not None:
    monkeypatch.setenv(
        "OTEL_RESOURCE_ATTRIBUTES", f"gcp.project_id={project_id_on_env}"
    )

  # Act.
  otel_resource = get_gcp_resource(project_id_in_arg)

  # Assert.
  expected_project_id = (
      project_id_on_env
      if project_id_on_env is not None
      else project_id_in_arg
      if project_id_in_arg is not None
      else None
  )
  assert otel_resource is not None
  assert (
      otel_resource.attributes.get("gcp.project_id", None)
      == expected_project_id
  )
