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
from typing import Final
from typing import Optional
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen

logger = logging.getLogger('google_adk.' + __name__)

_METADATA_ROOT: Final[str] = (
    'http://metadata.google.internal/computeMetadata/v1'
)
_METADATA_FLAVOR_HEADER: Final[str] = 'Metadata-Flavor'
_METADATA_TIMEOUT_SECONDS: Final[float] = 0.35

_PROJECT_ENV: Final[str] = 'GOOGLE_CLOUD_PROJECT'
_LOCATION_ENV: Final[str] = 'GOOGLE_CLOUD_LOCATION'
_ENTERPRISE_ENV: Final[str] = 'GOOGLE_GENAI_USE_ENTERPRISE'
_VERTEXAI_ENV: Final[str] = 'GOOGLE_GENAI_USE_VERTEXAI'
_API_KEY_ENVS: Final[tuple[str, ...]] = (
    'GOOGLE_API_KEY',
    'GOOGLE_GENAI_API_KEY',
)

# Once we have confirmed the metadata server is unreachable, skip further
# probes for the life of the process (avoids repeated 0.35s timeouts locally).
_off_gcp_cached: bool = False


def _metadata_get(path: str) -> Optional[str]:
  """Fetches a metadata server attribute, or None when unavailable."""
  url = f'{_METADATA_ROOT}/{path.lstrip("/")}'
  request = Request(
      url,
      headers={_METADATA_FLAVOR_HEADER: 'Google'},
      method='GET',
  )
  try:
    with urlopen(request, timeout=_METADATA_TIMEOUT_SECONDS) as response:
      body = response.read().decode('utf-8').strip()
      return body or None
  except (URLError, TimeoutError, OSError, ValueError) as e:
    logger.debug('GCP metadata lookup failed for %s: %s', path, e)
    return None


def is_running_on_gcp() -> bool:
  """Returns True when the GCP instance metadata server is reachable."""
  return _metadata_get('project/project-id') is not None


def get_project_id_from_metadata() -> Optional[str]:
  """Returns the GCP project id from the metadata server, if available."""
  return _metadata_get('project/project-id')


def get_location_from_metadata() -> Optional[str]:
  """Returns a GCP region from the metadata server, if available.

  Prefers Cloud Run's ``instance/region`` attribute
  (``projects/NUM/regions/REGION``). Falls back to deriving the region from
  ``instance/zone`` (``projects/NUM/zones/ZONE``) on GCE-like runtimes.
  """
  region_path = _metadata_get('instance/region')
  if region_path:
    # projects/123/regions/us-central1
    parts = region_path.strip('/').split('/')
    if len(parts) >= 4 and parts[-2] == 'regions':
      return parts[-1]
    if '/' not in region_path:
      return region_path

  zone_path = _metadata_get('instance/zone')
  if not zone_path:
    return None
  # projects/123/zones/us-central1-a -> us-central1
  parts = zone_path.strip('/').split('/')
  zone = parts[-1] if parts else zone_path
  if zone.count('-') >= 2:
    return zone.rsplit('-', 1)[0]
  return zone or None


def _has_api_key() -> bool:
  return any(os.environ.get(name) for name in _API_KEY_ENVS)


def _has_enterprise_or_vertex_flag() -> bool:
  return _ENTERPRISE_ENV in os.environ or _VERTEXAI_ENV in os.environ


def apply_gcp_runtime_defaults() -> dict[str, str]:
  """Fills unset Vertex/GCP env vars from the instance metadata server.

  Defaults are applied only when running on GCP and only for variables that
  are not already set (explicit shell env and `.env` values win). When no API
  key is configured, enterprise/Vertex mode is enabled by default on GCP.

  Returns:
    Mapping of environment variable names to values that were applied.
  """
  global _off_gcp_cached

  if _off_gcp_cached:
    return {}

  needs_project = not os.environ.get(_PROJECT_ENV)
  needs_location = not os.environ.get(_LOCATION_ENV)
  needs_enterprise = not _has_enterprise_or_vertex_flag() and not _has_api_key()

  if not (needs_project or needs_location or needs_enterprise):
    return {}

  # Probe metadata only when we would actually use it. A successful project-id
  # lookup is the on-GCP signal (and reuses the value when project is needed).
  project = get_project_id_from_metadata()
  if project is None:
    _off_gcp_cached = True
    return {}

  applied: dict[str, str] = {}

  if needs_project:
    os.environ[_PROJECT_ENV] = project
    applied[_PROJECT_ENV] = project

  if needs_location:
    location = get_location_from_metadata()
    if location:
      os.environ[_LOCATION_ENV] = location
      applied[_LOCATION_ENV] = location

  if needs_enterprise:
    # Prefer the modern flag; google-genai and ADK both honor it.
    os.environ[_ENTERPRISE_ENV] = 'true'
    applied[_ENTERPRISE_ENV] = 'true'

  if applied:
    logger.info(
        'Applied GCP metadata runtime defaults for unset env vars: %s',
        ', '.join(sorted(applied)),
    )
  return applied
