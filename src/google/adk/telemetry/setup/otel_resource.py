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
import os


try:
    from opentelemetry.sdk import resources as optional_otel_resources
except ImportError:
    optional_otel_resources = None

try:
    from opentelemetry.resourcedetector import gcp_resource_detector as optional_gcp_resource_detector
except ImportError:
    optional_gcp_resource_detector = None


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
    result = [
        optional_otel_resources.OTELResourceDetector(),
        optional_otel_resources.ProcessResourceDetector(),
        optional_otel_resources.OsResourceDetector()
    ]
    if optional_gcp_resource_detector is not None:
        result.append(optional_gcp_resource_detector.GoogleCloudResourceDetector())
    return result


def get_resource(project_id: Optional[str] = None):
    """Returns the resource to use with Open Telemetry."""
    if optional_otel_resources is None:
        return None
    resource_attributes = {}
    resource_attributes.update(_get_service_attributes())
    resource_attributes.update(_to_project_attributes(project_id=project_id))
    return optional_otel_resources.get_aggregated_resources(
        detectors=_get_resource_detectors,
        initial_resource=optional_otel_resources.Resource.create(
            attributes=resource_attributes,
        )
    )
