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

import logging
import os

from google.adk.telemetry.setup import credentials
from google.adk.telemetry.setup import env_utils
from google.adk.telemetry.setup import gcloud_project
from google.adk.telemetry.setup import otel_resource


try:
  from opentelemetry.sdk import metrics as optional_otel_metrics
except ImportError:
    optional_otel_metrics = None


try:
  from opentelemetry.exporter import cloud_monitoring as optional_cloud_monitoring_exporter
except ImportError:
  optional_cloud_monitoring_exporter = None


try:
  from opentelemetry.exporter.otlp.proto.grpc import metric_exporter as optional_otlp_metric_exporter
except ImportError:
  optional_otlp_metric_exporter = None


_logger = logging.getLogger(__name__)


def _create_gcp_exporter():
    if optional_otel_metrics is None:
        _logger.warning('Missing "opentelemetry-sdk" dependency; cannot initialize metrics export.')
        return None
    if optional_cloud_monitoring_exporter is None:
        _logger.warning('Missing "opentelemetry-exporter-gcp-monitoring" dependency; cannot initialize metrics export.')
        return None
    project_id = gcloud_project.get_metrics_project()
    if not project_id:
        _logger.warning(
            'Insufficient project information; cannot initialize metrics export. '
            'Set OTEL_GCLOUD_PROJECT_FOR_LOGS, OTEL_GCLOUD_PROJECT, or GOOGLE_CLOUD_PROJECT. '
            'Alternatively, setup Application Default Credentials with a Service Account associated with a project.')
        return None
    return optional_otlp_metric_exporter.CloudMonitoringMetricsExporter(
        project_id=project_id,
        add_unique_identifier=True)


def _create_otlp_exporter():
    if optional_otel_metrics is None:
        _logger.warning('Missing "opentelemetry-sdk" dependency; cannot initialize metrics export.')
        return None
    if optional_otlp_metric_exporter is None:
        _logger.warning('Missing "opentelemetry-exporter-otlp-proto-grpc" dependency; cannot initialize metrics export.')
        return None
    address = env_utils.get_metrics_otlp_endpoint()
    creds = credentials.create_credentials_for_address(address)
    return optional_otlp_metric_exporter.OTLPMetricExporter(credentials=creds)


_EXPORTER_FACTORIES = {
    'gcp': _create_gcp_exporter,
    'otlp': _create_otlp_exporter,
}


def _wrap_exporter_in_reader(exporter):
    if optional_otel_metrics is None:
        return None
    return optional_otel_metrics.PeriodicExportingMetricReader(exporter)


def _create_exporter_with_type(exporter_type):
    if not exporter_type:
        return None
    lowercase_name = exporter_type.lower()
    factory = _EXPORTER_FACTORIES.get(lowercase_name)
    if factory is None:
        _logger.warning('Unsupported exporter type: %s', exporter_type)
    return factory()


def _create_reader_from_exporter_type(exporter_type):
    exporter = _create_exporter_with_type(exporter_type)
    if exporter is None:
        return None
    return _wrap_exporter_in_reader(exporter)


def _setup_metrics_with_readers(readers):
    if optional_otel_metrics is None:
        _logger.warning('Missing "opentelemetry-sdk" dependency; cannot initialize metrics export.')
        return
    project_id = gcloud_project.get_metrics_project()
    resource = otel_resource.get_resource(project_id=project_id)
    meter_provider = optional_otel_metrics.MeterProvider(metric_readers=readers, resource=resource)
    optional_otel_metrics.set_meter_provider(meter_provider)


def setup_metrics_wiring(additional_readers = None) -> None:
    readers = []
    if additional_readers:
        readers.extend(additional_readers)
    requested_exporter_type = env_utils.get_metrics_exporter_type()
    reader = _create_reader_from_exporter_type(requested_exporter_type)
    if reader:
        readers.append(reader)
    if not readers:
        return
    _setup_metrics_with_readers(readers)
