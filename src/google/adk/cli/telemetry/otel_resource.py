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

from typing import Optional

import functools
import os


try:
    from opentelemetry.sdk import resources as optional_otel_resources
except ImportError:
    optional_otel_resources = None

try:
    from google import auth as optional_google_auth
except ImportError:
    optional_google_auth = None


_PROJECT_ENV_VARS = [
    'OTEL_GCLOUD_PROJECT',
    'GOOGLE_CLOUD_PROJECT',
    'GCLOUD_PROJECT',
    'GCP_PROJECT',
]

@functools.cache
def _get_project_id() -> Optional[str]:
    for env_var in _PROJECT_ENV_VARS:
        from_env = os.getenv(env_var)
        if from_env:
            return from_env
    if optional_google_auth is not None:
        _, project = optional_google_auth.default()
        return project
    return None


def _get_project_with_override(override_env_var) -> Optional[str]:
    project_override = os.getenv(override_env_var)
    if project_override is not None:
        return project_override
    return _get_project_id()


def _get_metrics_project() -> Optional[str]:
    return _get_project_with_override('OTEL_GCLOUD_PROJECT_FOR_METRICS')


def _get_logs_project() -> Optional[str]:
    return _get_project_with_override('OTEL_GCLOUD_PROJECT_FOR_LOGS')


def _get_traces_project() -> Optional[str]:
    return _get_project_with_override('OTEL_GCLOUD_PROJECT_FOR_TRACES')


def _get_service_namespace() -> Optional[str]:
    return os.getenv('OTEL_SERVICE_NAMESPACE')


def _get_service_name() -> Optional[str]:
    return os.getenv('OTEL_SERVICE_NAME')


def _get_service_instance() -> Optional[str]:
    return os.getenv('OTEL_SERVICE_INSTANCE_ID')


def _get_service_attributes() -> dict[str, str]:
    result = {}
    service_namespace = _get_service_namespace()
    if service_namespace:
        result['service.namespace.name'] = service_namespace
    service_name = _get_service_name()
    if service_name:
        result['service.name'] = service_name
    service_instance = _get_service_instance()
    if service_instance:
        result['service.instance.id'] = service_instance
    return result


def _to_project_attributes(project_id: Optional[str]) -> dict[str, str]:
    result = {}
    if project_id:
        result['gcp.project_id'] = project_id
    return result


def _get_resource_detectors():
    if optional_otel_resources is None:
        return []
    return [
        optional_otel_resources.OTELResourceDetector(),
        optional_otel_resources.ProcessResourceDetector(),
        optional_otel_resources.OsResourceDetector()
    ]


def _create_resource(project_id: Optional[str]):
    resource_attributes = {}
    resource_attributes.update(_get_service_attributes())
    resource_attributes.update(_to_project_attributes(project_id=project_id))
    return optional_otel_resources.get_aggregated_resources(
        detectors=_get_resource_detectors,
        initial_resource=optional_otel_resources.Resource.create(
            attributes=resource_attributes,
        )
    )


def get_logs_resource():
    """Returns the Open Telemetry resource to use for logs."""
    if optional_otel_resources is None:
        return None
    return _create_resource(_get_logs_project())


def get_metrics_resource():
    """Returns the Open Telemetry resource to use for metrics."""
    if optional_otel_resources is None:
        return None
    return _create_resource(_get_metrics_project())


def get_trace_resource():
    """Returns the Open Telemetry resource to use for traces."""
    if optional_otel_resources is None:
        return None
    return _create_resource(_get_traces_project())

