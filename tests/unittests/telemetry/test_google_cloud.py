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

from google.adk.telemetry import google_cloud
from google.adk.telemetry.google_cloud import _DEFAULT_MTLS_TELEMETRY_TRACES_ENPOINT
from google.adk.telemetry.google_cloud import _DEFAULT_TELEMETRY_TRACES_ENPOINT
from google.adk.telemetry.google_cloud import _get_api_endpoint
from google.adk.telemetry.google_cloud import _get_gcp_span_exporter
from google.adk.telemetry.google_cloud import _use_client_cert_effective
from google.adk.telemetry.google_cloud import get_gcp_exporters
from google.adk.telemetry.google_cloud import get_gcp_resource
import google.auth.credentials
from google.auth.transport import mtls
from google.auth.transport import requests
from opentelemetry.exporter.otlp.proto.http import trace_exporter
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
      lambda project_id: mock.MagicMock(),
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
  )
