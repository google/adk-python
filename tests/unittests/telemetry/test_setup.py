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
from unittest import mock

from google.adk.telemetry.setup import maybe_set_otel_providers
import pytest


@pytest.fixture
def mock_os_environ():
  initial_env = os.environ.copy()
  with mock.patch.dict(os.environ, initial_env, clear=False) as m:
    yield m


@pytest.mark.parametrize(
    "env_vars, should_setup_trace, should_setup_metrics, should_setup_logs",
    [
        (
            {"OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "some-endpoint"},
            True,
            False,
            False,
        ),
        (
            {"OTEL_EXPORTER_OTLP_METRICS_ENDPOINT": "some-endpoint"},
            False,
            True,
            False,
        ),
        (
            {"OTEL_EXPORTER_OTLP_LOGS_ENDPOINT": "some-endpoint"},
            False,
            False,
            True,
        ),
        (
            {
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "some-endpoint",
                "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT": "some-endpoint",
                "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT": "some-endpoint",
            },
            True,
            True,
            True,
        ),
        (
            {"OTEL_EXPORTER_OTLP_ENDPOINT": "some-endpoint"},
            True,
            True,
            True,
        ),
    ],
)
def test_maybe_set_otel_providers(
    env_vars: dict[str, str],
    should_setup_trace: bool,
    should_setup_metrics: bool,
    should_setup_logs: bool,
    monkeypatch: pytest.MonkeyPatch,
    mock_os_environ,  # pylint: disable=unused-argument,redefined-outer-name
):
  """
  Test initializing correct providers in setup_otel
  when providing OTel env variables.
  """
  # Arrange.
  for k, v in env_vars.items():
    os.environ[k] = v
  trace_provider_mock = mock.MagicMock()
  monkeypatch.setattr(
      "opentelemetry.trace.set_tracer_provider",
      trace_provider_mock,
  )
  meter_provider_mock = mock.MagicMock()
  monkeypatch.setattr(
      "opentelemetry.metrics.set_meter_provider",
      meter_provider_mock,
  )
  logs_provider_mock = mock.MagicMock()
  monkeypatch.setattr(
      "opentelemetry._logs.set_logger_provider",
      logs_provider_mock,
  )

  # Act.
  maybe_set_otel_providers()

  # Assert.
  # If given telemetry type was enabled,
  # the corresponding provider should be set.
  assert trace_provider_mock.call_count == (1 if should_setup_trace else 0)
  assert meter_provider_mock.call_count == (1 if should_setup_metrics else 0)
  assert logs_provider_mock.call_count == (1 if should_setup_logs else 0)
