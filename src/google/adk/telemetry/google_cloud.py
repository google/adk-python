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

from __future__ import annotations

import logging
import os
from typing import cast
from typing import Optional
from typing import TYPE_CHECKING

import google.auth
from opentelemetry.sdk._logs import LogRecordProcessor
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics.export import MetricReader
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import OTELResourceDetector
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from ..utils.feature_decorator import experimental
from .setup import OTelHooks

if TYPE_CHECKING:
  from google.auth.credentials import Credentials

logger = logging.getLogger('google_adk.' + __name__)

_GCP_LOG_NAME_ENV_VARIABLE_NAME = 'GOOGLE_CLOUD_DEFAULT_LOG_NAME'
_DEFAULT_LOG_NAME = 'adk-otel'


@experimental
def get_gcp_exporters(
    enable_cloud_tracing: bool = False,
    enable_cloud_metrics: bool = False,
    enable_cloud_logging: bool = False,
    google_auth: Optional[tuple[Credentials, str]] = None,
) -> OTelHooks:
  """Returns GCP OTel exporters to be used in the app.

  Args:
    enable_tracing: whether to enable tracing to Cloud Trace.
    enable_metrics: whether to enable reporting metrics to Cloud Monitoring.
    enable_logging: whether to enable sending logs to Cloud Logging.
    google_auth: optional custom credentials and project_id. google.auth.default() used when this is omitted.
  """

  credentials, project_id = (
      google_auth if google_auth is not None else google.auth.default()
  )
  if TYPE_CHECKING:
    credentials = cast(Credentials, credentials)
    project_id = cast(str, project_id)

  if not project_id:
    logger.warning(
        'Cannot determine GCP Project. OTel GCP Exporters cannot be set up.'
        ' Please make sure to log into correct GCP Project.'
    )
    return OTelHooks()

  span_processors: list[SpanProcessor] = []
  if enable_cloud_tracing:
    exporter = _get_gcp_span_exporter(credentials)
    span_processors.append(exporter)

  metric_readers: list[MetricReader] = []
  if enable_cloud_metrics:
    exporter = _get_gcp_metrics_exporter(project_id)
    if exporter:
      metric_readers.append(exporter)

  log_record_processors: list[LogRecordProcessor] = []
  if enable_cloud_logging:
    exporter = _get_gcp_logs_exporter(project_id)
    if exporter:
      log_record_processors.append(exporter)

  return OTelHooks(
      span_processors=span_processors,
      metric_readers=metric_readers,
      log_record_processors=log_record_processors,
  )


def _get_gcp_span_exporter(credentials: Credentials) -> SpanProcessor:
  """Adds OTEL span exporter to telemetry.googleapis.com"""

  from google.auth.transport.requests import AuthorizedSession
  from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

  return BatchSpanProcessor(
      OTLPSpanExporter(
          session=AuthorizedSession(credentials=credentials),
          endpoint='https://telemetry.googleapis.com/v1/traces',
      )
  )


def _get_gcp_metrics_exporter(project_id: str) -> MetricReader:
  from opentelemetry.exporter.cloud_monitoring import CloudMonitoringMetricsExporter

  return PeriodicExportingMetricReader(
      CloudMonitoringMetricsExporter(project_id=project_id),
      export_interval_millis=5000,
  )


def _get_gcp_logs_exporter(project_id: str) -> LogRecordProcessor:
  from opentelemetry.exporter.cloud_logging import CloudLoggingExporter

  default_log_name = os.environ.get(
      _GCP_LOG_NAME_ENV_VARIABLE_NAME, _DEFAULT_LOG_NAME
  )
  return BatchLogRecordProcessor(
      CloudLoggingExporter(
          project_id=project_id, default_log_name=default_log_name
      ),
  )


def get_gcp_resource(project_id: Optional[str] = None) -> Resource:
  """Returns OTEL with attributes specified in the following order (attributes specified later, overwrite those specified earlier):
  1. Populates gcp.project_id attribute from the project_id argument if present.
  2. OTELResourceDetector populates resource labels from environment variables like OTEL_SERVICE_NAME and OTEL_RESOURCE_ATTRIBUTES.
  3. GCP detector adds attributes corresponding to a correct monitored resource if ADK runs on one of supported platforms (e.g. GCE, GKE, CloudRun).

  Args:
    project_id: project id to fill out as `gcp.project_id` on the OTEL resource.
    This may be overwritten by OTELResourceDetector, if `gcp.project_id` is present in `OTEL_RESOURCE_ATTRIBUTES` env var.
  """
  resource = Resource(
      attributes={'gcp.project_id': project_id}
      if project_id is not None
      else {}
  )
  resource = resource.merge(OTELResourceDetector().detect())
  try:
    from opentelemetry.resourcedetector.gcp_resource_detector import GoogleCloudResourceDetector

    resource = resource.merge(
        GoogleCloudResourceDetector(raise_on_error=False).detect()
    )
  except ImportError:
    logger.warning(
        'Cloud not import opentelemetry.resourcedetector.gcp_resource_detector'
        ' GCE, GKE or CloudRun related resource attributes may be missing'
    )
  return resource
