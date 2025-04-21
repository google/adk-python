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
  from opentelemetry.sdk import _logs as optional_otel_logs
  from opentelemetry.sdk import _events as optional_otel_events
except ImportError:
    optional_otel_logs = None
    optional_otel_events = None


try:
  from opentelemetry.exporter import cloud_logging as optional_cloud_logging_exporter
except ImportError:
  optional_cloud_logging_exporter = None


try:
  from opentelemetry.exporter.otlp.proto.grpc import _log_exporter as optional_otlp_log_exporter
except ImportError:
  optional_otlp_log_exporter = None


_logger = logging.getLogger(__name__)



def _get_default_log_name():
    return env_utils.get_first_non_empty(
        [
            'GOOGLE_CLOUD_DEFAULT_LOG_NAME',
            'GCLOUD_DEFAULT_LOG_NAME',
            'GCP_DEFAULT_LOG_NAME'
        ],
        default_value='google-adk-python')


def _create_gcp_exporter():
    if optional_otel_logs is None:
        _logger.warning('Missing "opentelemetry-sdk" dependency; cannot initialize logs export.')
        return None
    if optional_cloud_logging_exporter is None:
        _logger.warning('Missing "opentelemetry-exporter-gcp-logging" dependency; cannot initialize logs export.')
        return None
    project_id = gcloud_project.get_logs_project()
    if not project_id:
        _logger.warning(
            'Insufficient project information; cannot initialize logs export. '
            'Set OTEL_GCLOUD_PROJECT_FOR_LOGS, OTEL_GCLOUD_PROJECT, or GOOGLE_CLOUD_PROJECT. '
            'Alternatively, setup Application Default Credentials with a Service Account associated with a project.')
        return None
    return optional_cloud_logging_exporter.CloudLoggingExporter(
        project_id=project_id,
        default_log_name=_get_default_log_name())


def _create_otlp_exporter():
    if optional_otel_logs is None:
        _logger.warning('Missing "opentelemetry-sdk" dependency; cannot initialize logs export.')
        return None
    if optional_otlp_log_exporter is None:
        _logger.warning('Missing "opentelemetry-exporter-otlp-proto-grpc" dependency; cannot initialize logs export.')
        return None
    address = env_utils.get_logs_otlp_endpoint()
    creds = credentials.create_credentials_for_address(address)
    return optional_otlp_log_exporter.OTLPLogExporter(credentials=creds)


_EXPORTER_FACTORIES = {
    'gcp': _create_gcp_exporter,
    'otlp': _create_otlp_exporter,
}


def _wrap_exporter_in_processor(exporter):
    if optional_otel_logs is None:
        return None
    return optional_otel_logs.BatchLogRecordProcessor(exporter)


def _create_exporter_with_type(exporter_type):
    if not exporter_type:
        return None
    lowercase_name = exporter_type.lower()
    factory = _EXPORTER_FACTORIES.get(lowercase_name)
    if factory is None:
        _logger.warning('Unsupported exporter type: %s', exporter_type)
    return factory()


def _create_processor_from_exporter_type(exporter_type):
    exporter = _create_exporter_with_type(exporter_type)
    if exporter is None:
        return None
    return _wrap_exporter_in_processor(exporter)


def _setup_logs_with_processors(processors):
    if optional_otel_logs is None:
        _logger.warning('Missing "opentelemetry-sdk" dependency; cannot initialize logs export.')
        return
    project_id = gcloud_project.get_logs_project()
    resource = otel_resource.get_resource(project_id=project_id)
    logger_provider = optional_otel_logs.LoggerProvider(resource=resource)
    for processor in processors:
        logger_provider.add_log_record_processor(processor)
    optional_otel_logs.set_logger_provider(logger_provider)
    optional_otel_events.set_event_logger_provider(optional_otel_events.EventLoggerProvider(
        logger_provider=logger_provider))


def setup_logs_wiring(additional_processors = None) -> None:
    processors = []
    if additional_processors:
        processors.extend(additional_processors)
    requested_exporter_type = env_utils.get_logs_exporter_type()
    processor = _create_processor_from_exporter_type(requested_exporter_type)
    if processor:
        processors.append(processor)
    if not processors:
        return
    _setup_logs_with_processors(processors)
