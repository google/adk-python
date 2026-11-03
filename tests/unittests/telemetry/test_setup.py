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

from google.adk.telemetry.setup import _get_otel_logs_exporter
from google.adk.telemetry.setup import _get_otel_metrics_exporter
from google.adk.telemetry.setup import _get_otel_span_exporter
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
    monkeypatch.setenv(k, v)
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
  monkeypatch.setattr(
      "google.adk.telemetry.setup._get_otel_span_exporter",
      lambda: mock.MagicMock(),
  )
  monkeypatch.setattr(
      "google.adk.telemetry.setup._get_otel_metrics_exporter",
      lambda: mock.MagicMock(),
  )
  monkeypatch.setattr(
      "google.adk.telemetry.setup._get_otel_logs_exporter",
      lambda: mock.MagicMock(),
  )

  # Act.
  maybe_set_otel_providers()

  # Assert.
  # If given telemetry type was enabled,
  # the corresponding provider should be set.
  assert trace_provider_mock.call_count == (1 if should_setup_trace else 0)
  assert meter_provider_mock.call_count == (1 if should_setup_metrics else 0)
  assert logs_provider_mock.call_count == (1 if should_setup_logs else 0)


@pytest.mark.parametrize(
    "getter, missing_module",
    [
        (
            _get_otel_span_exporter,
            "opentelemetry.exporter.otlp.proto.http.trace_exporter",
        ),
        (
            _get_otel_metrics_exporter,
            "opentelemetry.exporter.otlp.proto.http.metric_exporter",
        ),
        (
            _get_otel_logs_exporter,
            "opentelemetry.exporter.otlp.proto.http._log_exporter",
        ),
    ],
)
def test_get_otel_exporter_none_when_package_unavailable(
    getter, missing_module
):
  """A missing opentelemetry-exporter-otlp-proto-http disables the exporter
  instead of raising ModuleNotFoundError."""
  # Setting the module to None in sys.modules makes the `from ... import ...`
  # statement raise ImportError, simulating the package not being installed.
  with mock.patch.dict("sys.modules", {missing_module: None}):
    assert getter() is None


@pytest.mark.parametrize(
    "missing_module",
    [
        "opentelemetry.exporter.otlp.proto.http.trace_exporter",
        "opentelemetry.exporter.otlp.proto.http.metric_exporter",
        "opentelemetry.exporter.otlp.proto.http._log_exporter",
    ],
)
def test_maybe_set_otel_providers_does_not_raise_when_exporter_unavailable(
    missing_module: str,
    monkeypatch: pytest.MonkeyPatch,
    mock_os_environ,  # pylint: disable=unused-argument,redefined-outer-name
):
  """maybe_set_otel_providers degrades gracefully when an OTLP exporter package
  is missing rather than crashing telemetry setup."""
  monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "some-endpoint")

  with mock.patch.dict("sys.modules", {missing_module: None}):
    # Should not raise.
    maybe_set_otel_providers()
