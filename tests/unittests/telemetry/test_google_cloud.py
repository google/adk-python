# Copyright 2025 Google LLC
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

from unittest import mock

from google.adk.telemetry.google_cloud import get_gcp_exporters
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
