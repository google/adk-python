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

import os


def get_first_non_empty(env_vars, default_value=None):
    for env_var in env_vars:
        value = os.getenv(env_var)
        if value:
            return value
    return default_value


def env_to_bool(env_var, default_value=False):
    env_value = os.getenv(env_var) or ''
    lower_env = env_value.lower()
    if lower_env in ['1', 'true']:
        return True
    if lower_env in ['0', 'false']:
        return False
    return default_value


def get_logs_otlp_endpoint():
    return get_first_non_empty([
        # Based on https://opentelemetry.io/docs/languages/sdk-configuration/otlp-exporter/
        'OTEL_EXPORTER_OTLP_LOGS_ENDPOINT',
        'OTEL_EXPORTER_OTLP_ENDPOINT',
    ])


def get_metrics_otlp_endpoint():
    return get_first_non_empty([
        # Based on https://opentelemetry.io/docs/languages/sdk-configuration/otlp-exporter/
        'OTEL_EXPORTER_OTLP_METRICS_ENDPOINT',
        'OTEL_EXPORTER_OTLP_ENDPOINT',
    ])


def get_traces_otlp_endpoint():
    return get_first_non_empty([
        # Based on https://opentelemetry.io/docs/languages/sdk-configuration/otlp-exporter/
        'OTEL_EXPORTER_OTLP_TRACES_ENDPOINT',
        'OTEL_EXPORTER_OTLP_ENDPOINT',
    ])


def get_logs_exporter_type():
    exporter_type = get_first_non_empty([
        # Based on https://opentelemetry.io/docs/specs/otel/configuration/sdk-environment-variables/
        'OTLP_LOGS_EXPORTER',
        'OTLP_EXPORTER'
    ])
    if exporter_type:
        return exporter_type
    if env_to_bool('ADK_CLOUD_O11Y'):
        return 'gcp'
    if env_to_bool('ADK_LOG_TO_CLOUD'):
        return 'gcp'
    if get_logs_otlp_endpoint():
        return 'otlp'
    return None


def get_metrics_exporter_type():
    exporter_type = get_first_non_empty([
        # Based on https://opentelemetry.io/docs/specs/otel/configuration/sdk-environment-variables/
        'OTLP_METRICS_EXPORTER',
        'OTLP_EXPORTER'
    ])
    if exporter_type:
        return exporter_type
    if env_to_bool('ADK_CLOUD_O11Y'):
        return 'gcp'
    if env_to_bool('ADK_METRICS_TO_CLOUD'):
        return 'gcp'
    if get_metrics_otlp_endpoint():
        return 'otlp'
    return None


def get_trace_exporter_type():
    exporter_type = get_first_non_empty([
        # Based on https://opentelemetry.io/docs/specs/otel/configuration/sdk-environment-variables/
        'OTLP_TRACES_EXPORTER',
        'OTLP_EXPORTER'
    ])
    if exporter_type:
        return exporter_type
    if env_to_bool('ADK_CLOUD_O11Y'):
        return 'gcp'
    if env_to_bool('ADK_TRACE_TO_CLOUD'):
        return 'gcp'
    if get_traces_otlp_endpoint():
        return 'otlp'
    return None
