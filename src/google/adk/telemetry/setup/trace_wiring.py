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
  from opentelemetry.sdk import traces as optional_otel_traces
except ImportError:
    optional_otel_traces = None


try:
  from opentelemetry.exporter.otlp.proto.grpc import trace_exporter as optional_otlp_trace_exporter
except ImportError:
  optional_otlp_trace_exporter = None


_logger = logging.getLogger(__name__)


def _create_gcp_exporter():
    if optional_otel_traces is None:
        _logger.warning('Missing "opentelemetry-sdk" dependency; cannot initialize trace export.')
        return
    if optional_otlp_trace_exporter is None:
        _logger.warning('Missing "opentelemetry-exporter-otlp-proto-grpc" dependency; cannot initialize trace export.')
    address = env_utils.get_traces_otlp_endpoint()
    creds = credentials.create_credentials_for_address(address)
    return optional_otlp_trace_exporter.OTLPSpanExporter(credentials=creds)


def _create_otlp_exporter():
    if optional_otel_traces is None:
        _logger.warning('Missing "opentelemetry-sdk" dependency; cannot initialize trace export.')
        return None
    if optional_otlp_trace_exporter is None:
        _logger.warning('Missing "opentelemetry-exporter-otlp-proto-grpc" dependency; cannot initialize trace export.')
        return None
    address = env_utils.get_traces_otlp_endpoint()
    creds = credentials.create_credentials_for_address(address)
    return optional_otlp_trace_exporter.OTLPSpanExporter(credentials=creds)


def _create_gcp_exporter():
    if optional_otel_traces is None:
        _logger.warning('Missing "opentelemetry-sdk" dependency; cannot initialize trace export.')
        return None
    if optional_otlp_trace_exporter is None:
        _logger.warning('Missing "opentelemetry-exporter-otlp-proto-grpc" dependency; cannot initialize trace export.')
        return None
    creds = credentials.create_gcp_api_grpc_creds()
    return optional_otlp_trace_exporter.OTLPSpanExporter(
        endpoint='https://telemetry.googleapis.com/',
        credentials=creds)


_EXPORTER_FACTORIES = {
    'gcp': _create_gcp_exporter,
    'otlp': _create_otlp_exporter,
}


def _wrap_exporter_in_processor(exporter):
    if optional_otel_traces is None:
        return None
    return optional_otel_traces.BatchSpanProcessor(exporter)


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


def _setup_traces_with_processors(processors):
    if optional_otel_traces is None:
        _logger.warning('Missing "opentelemetry-sdk" dependency; cannot initialize trace export.')
        return
    project_id = gcloud_project.get_traces_project()
    resource = otel_resource.get_resource(project_id=project_id)
    tracer_provider = optional_otel_traces.TracerProvider(resource=resource)
    for processor in processors:
        tracer_provider.add_span_processor(processor)
    optional_otel_traces.set_tracer_provider(tracer_provider)


def setup_traces_wiring(additional_processors = None) -> None:
    processors = []
    if additional_processors:
        processors.extend(additional_processors)
    requested_exporter_type = env_utils.get_traces_exporter_type()
    processor = _create_processor_from_exporter_type(requested_exporter_type)
    if processor:
        processors.append(processor)
    if not processors:
        return
    _setup_traces_with_processors(processors)
