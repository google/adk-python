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

"""GCP instance metadata helpers for runtime env defaults.

This module is for ADK internal use only.
Please do not rely on the implementation details.
"""

from __future__ import annotations

import logging
import os
from typing import Any
from typing import Final
from typing import TypedDict
import warnings

from google.auth import _cloud_sdk
from google.auth import exceptions as auth_exceptions

from . import env_utils

logger = logging.getLogger('google_adk.' + __name__)

_PROJECT_ENV: Final[str] = 'GOOGLE_CLOUD_PROJECT'
_ENTERPRISE_ENV: Final[str] = 'GOOGLE_GENAI_USE_ENTERPRISE'
_VERTEXAI_ENV: Final[str] = 'GOOGLE_GENAI_USE_VERTEXAI'
_API_KEY_ENVS: Final[tuple[str, ...]] = (
    'GOOGLE_API_KEY',
    'GEMINI_API_KEY',
)
_CREDENTIALS_ENV: Final[str] = 'GOOGLE_APPLICATION_CREDENTIALS'
_METADATA_TIMEOUT_SECONDS: Final[float] = 1.0

# Cached project ID from MDS probe (None if not yet probed).
_cached_project_id: str | None = None


class GcpClientDefaults(TypedDict, total=False):
  """Default keyword arguments for google.genai.Client on GCP."""

  project: str


def get_project_id_from_metadata() -> str | None:
  """Returns the GCP project id from the instance metadata server, if available."""
  global _cached_project_id
  if _cached_project_id is not None:
    return _cached_project_id or None

  # Imported here: the transport pulls in requests, urllib3 and cryptography,
  # which `google_llm` would otherwise load for every ADK process at startup.
  # google-auth does not expose a public API for direct Compute Engine metadata
  # server probes (google.auth.default() requires credentials configuration).
  # _metadata is the standard internal module used across Google Cloud SDKs.
  from google.auth.compute_engine import _metadata
  from google.auth.transport import requests as auth_requests

  try:
    request = auth_requests.Request()
    if not _metadata.ping(
        request, timeout=_METADATA_TIMEOUT_SECONDS, retry_count=1
    ):
      return None

    project = _metadata.get(
        request,
        'project/project-id',
        timeout=_METADATA_TIMEOUT_SECONDS,
        retry_count=1,
    )
    project = str(project or '').strip()
    _cached_project_id = project
    return project or None
  except auth_exceptions.TransportError as e:
    logger.debug('GCP metadata project-id lookup transport error: %s', e)
    return None
  except Exception as e:  # pylint: disable=broad-except
    logger.debug('GCP metadata project-id lookup failed: %s', e)
    return None


def _has_api_key(client_kwargs: dict[str, Any] | None = None) -> bool:
  if client_kwargs and client_kwargs.get('api_key'):
    return True
  return any(os.environ.get(name) for name in _API_KEY_ENVS)


def _has_credentials(client_kwargs: dict[str, Any] | None = None) -> bool:
  if client_kwargs and client_kwargs.get('credentials') is not None:
    return True
  if os.environ.get(_CREDENTIALS_ENV):
    return True
  try:
    adc_path = _cloud_sdk.get_application_default_credentials_path()
    return bool(adc_path and os.path.isfile(adc_path))
  except Exception:  # pylint: disable=broad-except
    return False


def _get_enterprise_setting(
    client_kwargs: dict[str, Any] | None = None,
) -> bool | None:
  """Returns the enterprise setting (True/False if explicitly set, None if unset).

  Args:
    client_kwargs: Optional keyword arguments passed to the client.

  Returns:
    True if enterprise or vertexai mode is explicitly enabled (via
    client_kwargs or environment variables), False if explicitly disabled,
    or None if unset.
  """
  if client_kwargs:
    if (
        client_kwargs.get('enterprise') is False
        or client_kwargs.get('vertexai') is False
    ):
      return False
    if client_kwargs.get('enterprise') or client_kwargs.get('vertexai'):
      return True
  if _ENTERPRISE_ENV in os.environ:
    return env_utils.is_env_enabled(_ENTERPRISE_ENV)
  if _VERTEXAI_ENV in os.environ:
    warnings.warn(
        'GOOGLE_GENAI_USE_VERTEXAI is deprecated, please use'
        ' GOOGLE_GENAI_USE_ENTERPRISE instead',
        DeprecationWarning,
        stacklevel=2,
    )
    return env_utils.is_env_enabled(_VERTEXAI_ENV)
  return None


def get_gcp_client_defaults(
    client_kwargs: dict[str, Any] | None = None,
) -> GcpClientDefaults:
  """Returns GCP runtime defaults (project) to pass to Client on Vertex AI.

  Defaults are applied only when running on GCP with enterprise (Vertex AI) mode
  enabled, and only for options that are not already set (explicit client_kwargs,
  shell env, and `.env` values win). When enterprise mode is not enabled, or an
  API key is configured, defaults are skipped so that Google AI Studio / Gemini
  Developer API / custom endpoints are preserved without flipping the backend.

  Args:
    client_kwargs: Optional keyword arguments passed to the client.

  Returns:
    Mapping of keyword arguments to pass to the google.genai.Client constructor.
  """
  if _has_api_key(client_kwargs):
    return {}

  if _has_credentials(client_kwargs):
    return {}

  enterprise_setting = _get_enterprise_setting(client_kwargs)
  if enterprise_setting is not True:
    return {}

  has_project = bool(
      (client_kwargs and 'project' in client_kwargs)
      or os.environ.get(_PROJECT_ENV)
  )
  if has_project:
    return {}

  project = get_project_id_from_metadata()
  if not project:
    return {}

  logger.debug(
      'Applying GCP metadata client defaults for unset options: project'
  )
  return {'project': project}
