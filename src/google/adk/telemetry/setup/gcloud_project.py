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
    from google import auth as optional_google_auth
    from google.auth import exceptions as optional_google_auth_exceptions
except ImportError:
    optional_google_auth = None
    optional_google_auth_exceptions = None


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
        try:
            _, project = optional_google_auth.default()
            return project
        except optional_google_auth_exceptions.DefaultCredentialsError:
            return None
    return None


def _get_project_with_override(override_env_var) -> Optional[str]:
    project_override = os.getenv(override_env_var)
    if project_override is not None:
        return project_override
    return _get_project_id()


@functools.cache
def get_metrics_project() -> Optional[str]:
    """Return the Google Cloud project to which to write metrics."""
    return _get_project_with_override('OTEL_GCLOUD_PROJECT_FOR_METRICS')


@functools.cache
def get_logs_project() -> Optional[str]:
    """Return the Google Cloud project to which to write logs."""
    return _get_project_with_override('OTEL_GCLOUD_PROJECT_FOR_LOGS')


@functools.cache
def get_traces_project() -> Optional[str]:
    """Return the Google Cloud project to which to write traces."""
    return _get_project_with_override('OTEL_GCLOUD_PROJECT_FOR_TRACES')
