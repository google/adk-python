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

"""Configuration and Secret Manager credential resolution for Google Maps."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import logging
import os

import google.auth
from google.auth import impersonated_credentials

logger = logging.getLogger("google_adk." + __name__)

_SECRET_MANAGER_REQUIRED = (
    "Secret Manager support requires google-cloud-secret-manager. "
    "Install it with: pip install 'google-adk[gcp]'."
)


def _clean_env(value: str | None) -> str | None:
  """Returns None if a variable is unset or still holds a template token."""
  if not value:
    return None
  stripped = value.strip()
  if not stripped or stripped.startswith(("<TODO", "<YOUR_")):
    return None
  return stripped


class GoogleMapsConfig:
  """Manages credentials and default API settings for Google Maps Platform.

  Resolution priority for the Google Maps API key:
    1. Explicit ``api_key`` passed to ``GoogleMapsConfig(api_key=...)``.
    2. ``MAPS_API_KEY`` environment variable.
    3. Google Cloud Secret Manager secret specified via
       ``secret_resource_name`` or ``MAPS_API_KEY_SECRET_RESOURCE`` (supports
       optional Service Account impersonation via ``service_account_email`` or
       ``GOOGLE_SA_EMAIL``).
  """

  def __init__(
      self,
      api_key: str | None = None,
      secret_resource_name: str | None = None,
      service_account_email: str | None = None,
      default_pitch: int = 10,
      default_fov: int = 120,
      default_size: str = "600x400",
      timeout: float = 10.0,
  ) -> None:
    self._explicit_api_key = _clean_env(api_key)
    self.secret_resource_name = _clean_env(secret_resource_name) or _clean_env(
        os.environ.get("MAPS_API_KEY_SECRET_RESOURCE")
    )
    self.service_account_email = _clean_env(
        service_account_email
    ) or _clean_env(os.environ.get("GOOGLE_SA_EMAIL"))
    self.default_pitch = default_pitch
    self.default_fov = default_fov
    self.default_size = default_size
    self.timeout = timeout
    self._cached_api_key: str | None = None

  def get_api_key(self) -> str | None:
    """Resolves Maps API key from memory, environment, or Secret Manager."""
    if self._explicit_api_key:
      return self._explicit_api_key
    if self._cached_api_key:
      return self._cached_api_key

    env_key = _clean_env(os.environ.get("MAPS_API_KEY"))
    if env_key:
      self._cached_api_key = env_key
      return self._cached_api_key

    secret_res = self.secret_resource_name or _clean_env(
        os.environ.get("MAPS_API_KEY_SECRET_RESOURCE")
    )
    if not secret_res:
      return None

    try:
      secretmanager = importlib.import_module("google.cloud.secretmanager")
    except ImportError as exc:
      raise ImportError(_SECRET_MANAGER_REQUIRED) from exc

    sa_email = self.service_account_email or _clean_env(
        os.environ.get("GOOGLE_SA_EMAIL")
    )
    agent_engine_id = _clean_env(os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID"))

    try:
      credentials, _ = google.auth.default()
      if (
          sa_email
          and not agent_engine_id
          and getattr(credentials, "service_account_email", None) != sa_email
      ):
        credentials = impersonated_credentials.Credentials(
            source_credentials=credentials,
            target_principal=sa_email,
            target_scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
      sm_client = secretmanager.SecretManagerServiceClient(
          credentials=credentials
      )
      resp = sm_client.access_secret_version(request={"name": secret_res})
      key = _clean_env(resp.payload.data.decode("UTF-8"))
      if key:
        self._cached_api_key = key
        return self._cached_api_key
    except Exception as exc:  # pylint: disable=broad-except
      logger.warning(
          "Secret Manager lookup failed for Maps API key (%s): %s",
          secret_res,
          type(exc).__name__,
      )

    return None

  async def get_api_key_async(self) -> str | None:
    """Async ``get_api_key`` that runs Secret Manager lookups off the loop."""
    if self._explicit_api_key:
      return self._explicit_api_key
    if self._cached_api_key:
      return self._cached_api_key
    return await asyncio.to_thread(self.get_api_key)

  def get_key_fingerprint(self) -> str:
    """Returns a non-reversible SHA-256 prefix for cache partitioning."""
    api_key = self.get_api_key()
    if not api_key:
      return "unconfigured"
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]


default_config = GoogleMapsConfig()
