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

import google.auth
from opentelemetry.resourcedetector.gcp_resource_detector import GoogleCloudResourceDetector
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

logger = logging.getLogger('google_adk.' + __name__)


@experimental
def get_gcp_exporters(
    enable_cloud_tracing: bool = False,
    enable_cloud_metrics: bool = False,
    enable_cloud_logging: bool = False,
) -> OTelHooks:
  """Returns GCP OTel exporters to be used in the app.

  Args:
    enable_tracing: whether to enable tracing to Cloud Trace.
    enable_metrics: whether to enable raporting metrics to Cloud Monitoring.
    enable_logging: whether to enable sending logs to Cloud Logging.
  """
  _, project_id = google.auth.default()
  if not project_id:
    logger.warning(
        'Cannot determine GCP Project. OTel GCP Exporters cannot be set up.'
        ' Please make sure to log into correct GCP Project.'
    )
    return OTelHooks()

  span_processors = []
  if enable_cloud_tracing:
    exporter = _get_gcp_span_exporter(project_id)
    span_processors.append(exporter)

  metric_readers = []
  if enable_cloud_metrics:
    exporter = _get_gcp_metrics_exporter(project_id)
    if exporter:
      metric_readers.append(exporter)

  log_record_processors = []
  if enable_cloud_logging:
    exporter = _get_gcp_logs_exporter(project_id)
    if exporter:
      log_record_processors.append(exporter)

  return OTelHooks(
      span_processors=span_processors,
      metric_readers=metric_readers,
      log_record_processors=log_record_processors,
  )


def _get_gcp_span_exporter(project_id: str) -> SpanProcessor:
  from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

  return BatchSpanProcessor(CloudTraceSpanExporter(project_id=project_id))


def _get_gcp_metrics_exporter(project_id: str) -> MetricReader:
  from opentelemetry.exporter.cloud_monitoring import CloudMonitoringMetricsExporter

  return PeriodicExportingMetricReader(
      CloudMonitoringMetricsExporter(project_id=project_id),
      export_interval_millis=5000,
  )


def _get_gcp_logs_exporter(project_id: str) -> LogRecordProcessor:
  from opentelemetry.exporter.cloud_logging import CloudLoggingExporter

  return BatchLogRecordProcessor(
      # TODO(jawoszek) - add default_log_name once design is approved.
      CloudLoggingExporter(project_id=project_id)
  )


def get_gcp_resource() -> Resource:
  # The OTELResourceDetector populates resource labels from
  # environment variables like OTEL_SERVICE_NAME and OTEL_RESOURCE_ATTRIBUTES.
  # Then the GCP detector adds attributes corresponding to a correct
  # monitored resource if ADK runs on one of supported platforms
  # (e.g. GCE, GKE, CloudRun).
  return (
      OTELResourceDetector()
      .detect()
      .merge(GoogleCloudResourceDetector(raise_on_error=False).detect())
  )
