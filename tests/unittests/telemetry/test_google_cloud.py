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

from google.adk.telemetry import _agent_engine
from google.adk.telemetry import google_cloud
from google.adk.telemetry._agent_engine import telemetry_user_agent_headers
from google.adk.telemetry._agent_engine_metric_exporter import MIN_EXPORT_INTERVAL_MS
from google.adk.telemetry.google_cloud import _DEFAULT_MTLS_TELEMETRY_METRICS_ENDPOINT
from google.adk.telemetry.google_cloud import _DEFAULT_MTLS_TELEMETRY_TRACES_ENPOINT
from google.adk.telemetry.google_cloud import _DEFAULT_TELEMETRY_METRICS_ENDPOINT
from google.adk.telemetry.google_cloud import _DEFAULT_TELEMETRY_TRACES_ENPOINT
from google.adk.telemetry.google_cloud import _get_api_endpoint
from google.adk.telemetry.google_cloud import _get_gcp_metrics_exporter
from google.adk.telemetry.google_cloud import _get_gcp_otlp_metric_exporter
from google.adk.telemetry.google_cloud import _get_gcp_span_exporter
from google.adk.telemetry.google_cloud import _use_client_cert_effective
from google.adk.telemetry.google_cloud import get_gcp_exporters
from google.adk.telemetry.google_cloud import get_gcp_resource
import google.auth.credentials
from google.auth.transport import mtls
from google.auth.transport import requests
from opentelemetry.exporter.otlp.proto.http import trace_exporter
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
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
  monkeypatch.setattr(
      "google.adk.telemetry.google_cloud._get_gcp_span_exporter",
      lambda credentials: mock.MagicMock(),
  )
  monkeypatch.setattr(
      "google.adk.telemetry.google_cloud._get_gcp_metrics_exporter",
      lambda google_auth: mock.MagicMock(),
  )
  monkeypatch.setattr(
      "google.adk.telemetry.google_cloud._get_gcp_logs_exporter",
      lambda project_id: mock.MagicMock(),
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


def test_get_gcp_resource_sets_standard_cloud_resource_id(
    monkeypatch: pytest.MonkeyPatch,
):
  # Arrange.
  monkeypatch.setenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", "1234567890")
  monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")

  # Act.
  otel_resource = get_gcp_resource("my-project")

  # Assert.
  # The Agent Engine dashboard filters on the OTel-standard key.
  assert otel_resource.attributes.get("cloud.resource_id") == (
      "//aiplatform.googleapis.com/projects/my-project"
      "/locations/us-central1/reasoningEngines/1234567890"
  )
  assert "cloud.resource.id" not in otel_resource.attributes


@mock.patch.object(mtls, "should_use_client_cert", autospec=True)
def test_use_client_cert_effective_from_mtls(mock_should_use):
  mock_should_use.return_value = True
  assert _use_client_cert_effective()

  mock_should_use.return_value = False
  assert not _use_client_cert_effective()


def test_use_client_cert_effective_from_env(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
  with mock.patch.object(
      mtls,
      "should_use_client_cert",
      autospec=True,
      side_effect=AttributeError,
  ):
    monkeypatch.setenv("GOOGLE_API_USE_CLIENT_CERTIFICATE", "true")
    assert _use_client_cert_effective()

    monkeypatch.setenv("GOOGLE_API_USE_CLIENT_CERTIFICATE", "false")
    assert not _use_client_cert_effective()

    # Test invalid value defaults to False
    monkeypatch.setenv("GOOGLE_API_USE_CLIENT_CERTIFICATE", "maybe")
    assert not _use_client_cert_effective()
    assert (
        "Environment variable `GOOGLE_API_USE_CLIENT_CERTIFICATE` must be"
        " either `true` or `false`"
        in caplog.text
    )


@pytest.mark.parametrize(
    "env_val, cert_source, expected",
    [
        ("auto", lambda: b"cert", _DEFAULT_MTLS_TELEMETRY_TRACES_ENPOINT),
        ("auto", None, _DEFAULT_TELEMETRY_TRACES_ENPOINT),
        ("always", None, _DEFAULT_MTLS_TELEMETRY_TRACES_ENPOINT),
        ("never", lambda: b"cert", _DEFAULT_TELEMETRY_TRACES_ENPOINT),
        ("invalid", None, _DEFAULT_TELEMETRY_TRACES_ENPOINT),
    ],
)
def test_get_api_endpoint(
    env_val,
    cert_source,
    expected,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
  monkeypatch.setenv("GOOGLE_API_USE_MTLS_ENDPOINT", env_val)
  if env_val == "invalid":
    assert _get_api_endpoint(cert_source) == expected
    assert (
        "Environment variable `GOOGLE_API_USE_MTLS_ENDPOINT` must be one of"
        in caplog.text
    )
  else:
    assert _get_api_endpoint(cert_source) == expected


@pytest.mark.parametrize(
    "env_val, cert_source, expected",
    [
        ("auto", lambda: b"cert", _DEFAULT_MTLS_TELEMETRY_METRICS_ENDPOINT),
        ("auto", None, _DEFAULT_TELEMETRY_METRICS_ENDPOINT),
        ("always", None, _DEFAULT_MTLS_TELEMETRY_METRICS_ENDPOINT),
        ("never", lambda: b"cert", _DEFAULT_TELEMETRY_METRICS_ENDPOINT),
    ],
)
def test_get_api_endpoint_for_metrics(
    env_val,
    cert_source,
    expected,
    monkeypatch: pytest.MonkeyPatch,
):
  """The same mTLS matrix, with the endpoints overridden for metrics."""
  monkeypatch.setenv("GOOGLE_API_USE_MTLS_ENDPOINT", env_val)

  assert (
      _get_api_endpoint(
          cert_source,
          default_endpoint=_DEFAULT_TELEMETRY_METRICS_ENDPOINT,
          mtls_endpoint=_DEFAULT_MTLS_TELEMETRY_METRICS_ENDPOINT,
      )
      == expected
  )


@mock.patch.object(requests, "AuthorizedSession", autospec=True)
@mock.patch(
    "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
    autospec=True,
)
@mock.patch(
    "google.adk.telemetry.google_cloud.BatchSpanProcessor", autospec=True
)
@mock.patch(
    "google.adk.telemetry.google_cloud._use_client_cert_effective",
    autospec=True,
)
@mock.patch(
    "google.auth.transport.mtls.has_default_client_cert_source", autospec=True
)
@mock.patch(
    "google.auth.transport.mtls.default_client_cert_source", autospec=True
)
def test_get_gcp_span_exporter_mtls(
    mock_default_cert: mock.MagicMock,
    mock_has_cert: mock.MagicMock,
    mock_use_cert: mock.MagicMock,
    mock_batch: mock.MagicMock,
    mock_exporter: mock.MagicMock,
    mock_session: mock.MagicMock,
):
  credentials = mock.create_autospec(
      google.auth.credentials.Credentials, instance=True
  )
  mock_use_cert.return_value = True
  mock_has_cert.return_value = True
  mock_default_cert.return_value = b"cert"

  _get_gcp_span_exporter(credentials)

  mock_session.assert_called_once_with(credentials=credentials)
  mock_session.return_value.configure_mtls_channel.assert_called_once()
  mock_exporter.assert_called_once_with(
      session=mock_session.return_value,
      endpoint=_DEFAULT_MTLS_TELEMETRY_TRACES_ENPOINT,
      headers=None,
  )


@mock.patch.object(requests, "AuthorizedSession", autospec=True)
@mock.patch(
    "opentelemetry.exporter.otlp.proto.http.metric_exporter.OTLPMetricExporter",
    autospec=True,
)
@mock.patch(
    "google.adk.telemetry.google_cloud._use_client_cert_effective",
    autospec=True,
)
@mock.patch(
    "google.auth.transport.mtls.has_default_client_cert_source", autospec=True
)
@mock.patch(
    "google.auth.transport.mtls.default_client_cert_source", autospec=True
)
def test_get_gcp_otlp_metric_exporter_mtls(
    mock_default_cert: mock.MagicMock,
    mock_has_cert: mock.MagicMock,
    mock_use_cert: mock.MagicMock,
    mock_exporter: mock.MagicMock,
    mock_session: mock.MagicMock,
):
  """Metrics take the mTLS branch onto the *metrics* endpoint, not traces'."""
  credentials = mock.create_autospec(
      google.auth.credentials.Credentials, instance=True
  )
  mock_use_cert.return_value = True
  mock_has_cert.return_value = True
  mock_default_cert.return_value = b"cert"

  _get_gcp_otlp_metric_exporter(google_auth=(credentials, "project-id"))

  mock_session.assert_called_once_with(credentials=credentials)
  mock_session.return_value.configure_mtls_channel.assert_called_once()
  mock_exporter.assert_called_once_with(
      session=mock_session.return_value,
      endpoint=_DEFAULT_MTLS_TELEMETRY_METRICS_ENDPOINT,
      headers=None,
  )


@mock.patch.object(requests, "AuthorizedSession", autospec=True)
@mock.patch(
    "opentelemetry.exporter.otlp.proto.http.metric_exporter.OTLPMetricExporter",
    autospec=True,
)
@mock.patch(
    "google.adk.telemetry.google_cloud._use_client_cert_effective",
    autospec=True,
)
def test_get_gcp_otlp_metric_exporter_no_mtls(
    mock_use_cert: mock.MagicMock,
    mock_exporter: mock.MagicMock,
    mock_session: mock.MagicMock,
):
  """Without a client cert, export goes to the plain metrics endpoint."""
  credentials = mock.create_autospec(
      google.auth.credentials.Credentials, instance=True
  )
  mock_use_cert.return_value = False

  _get_gcp_otlp_metric_exporter(google_auth=(credentials, "project-id"))

  mock_session.return_value.configure_mtls_channel.assert_not_called()
  mock_exporter.assert_called_once_with(
      session=mock_session.return_value,
      endpoint=_DEFAULT_TELEMETRY_METRICS_ENDPOINT,
      headers=None,
  )


@mock.patch.object(requests, "AuthorizedSession", autospec=True)
@mock.patch(
    "opentelemetry.exporter.otlp.proto.http.metric_exporter.OTLPMetricExporter",
    autospec=True,
)
@mock.patch(
    "google.adk.telemetry.google_cloud._use_client_cert_effective",
    autospec=True,
)
def test_get_gcp_otlp_metric_exporter_sends_agent_engine_user_agent(
    mock_use_cert: mock.MagicMock,
    mock_exporter: mock.MagicMock,
    mock_session: mock.MagicMock,
    monkeypatch: pytest.MonkeyPatch,
):
  """Agent Engine attributes metric traffic via the User-Agent header."""
  credentials = mock.create_autospec(
      google.auth.credentials.Credentials, instance=True
  )
  mock_use_cert.return_value = False
  monkeypatch.setenv("GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY", "1")

  _get_gcp_otlp_metric_exporter(google_auth=(credentials, "project-id"))

  headers = mock_exporter.call_args.kwargs["headers"]
  assert headers == telemetry_user_agent_headers()
  assert headers["User-Agent"].startswith("Vertex-Agent-Engine/")


def test_get_gcp_otlp_metric_exporter_uses_default_credentials(
    monkeypatch: pytest.MonkeyPatch,
):
  """Omitting google_auth falls back to google.auth.default()."""
  credentials = mock.create_autospec(
      google.auth.credentials.Credentials, instance=True
  )
  monkeypatch.setattr(
      "google.auth.default", lambda: (credentials, "project-id")
  )
  session = mock.MagicMock(name="session")
  monkeypatch.setattr(
      "google.auth.transport.requests.AuthorizedSession",
      lambda credentials: session,
  )
  monkeypatch.setattr(
      "google.adk.telemetry.google_cloud._use_client_cert_effective",
      lambda: False,
  )
  exporter = mock.MagicMock(name="exporter")
  monkeypatch.setattr(
      "opentelemetry.exporter.otlp.proto.http.metric_exporter.OTLPMetricExporter",
      lambda **kwargs: exporter,
  )

  assert _get_gcp_otlp_metric_exporter() is exporter


def test_get_gcp_metrics_exporter_wraps_otlp_in_periodic_reader(
    monkeypatch: pytest.MonkeyPatch,
):
  """Off Agent Engine, metrics go through a 5s periodic reader over OTLP."""
  exporter = mock.MagicMock(name="exporter")
  monkeypatch.setattr(
      "google.adk.telemetry.google_cloud._get_gcp_otlp_metric_exporter",
      lambda google_auth: exporter,
  )
  captured = {}

  def _reader(exp, export_interval_millis):
    captured["exporter"] = exp
    captured["interval"] = export_interval_millis
    return mock.MagicMock(spec=PeriodicExportingMetricReader)

  monkeypatch.setattr(
      "google.adk.telemetry.google_cloud.PeriodicExportingMetricReader", _reader
  )

  reader = _get_gcp_metrics_exporter(("credentials", "project-id"))

  assert reader is not None
  assert captured == {"exporter": exporter, "interval": MIN_EXPORT_INTERVAL_MS}


def test_get_gcp_metrics_exporter_none_when_otlp_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
  """A missing OTLP exporter package disables metrics instead of raising."""
  monkeypatch.setattr(
      "google.adk.telemetry.google_cloud._get_gcp_otlp_metric_exporter",
      lambda google_auth: None,
  )

  assert _get_gcp_metrics_exporter(("credentials", "project-id")) is None


@pytest.fixture(autouse=True)
def _clear_agent_engine_metrics_cache():
  """The memoized agent-engine metrics builder must not leak across tests."""
  _agent_engine._get_agent_engine_metrics_setup.cache_clear()
  yield
  _agent_engine._get_agent_engine_metrics_setup.cache_clear()


def test_agent_engine_uses_only_request_driven_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """On Agent Engine there must be exactly one metric reader: two exporters
  would double-report every point."""
  monkeypatch.delenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", raising=False)
  monkeypatch.setattr("google.auth.default", lambda: ("", "project-id"))
  fake_state = mock.MagicMock(name="metrics_state")
  monkeypatch.setattr(
      "google.adk.telemetry.google_cloud._get_agent_engine_metrics_setup",
      lambda: fake_state,
  )
  monkeypatch.setattr(
      "google.adk.telemetry.google_cloud._get_gcp_otlp_metric_exporter",
      lambda google_auth=None: mock.MagicMock(name="otlp_exporter"),
  )

  otel_hooks = get_gcp_exporters(enable_cloud_metrics=True)

  assert otel_hooks.metric_readers == [fake_state.reader]
  assert otel_hooks.span_processors == [fake_state.span_processor]
