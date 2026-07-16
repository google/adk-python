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

"""GCP Skill Registry implementation."""

from __future__ import annotations

import asyncio
import os
import ssl
import tempfile
from typing import Any

from google.adk.skills import _utils
from google.adk.skills import models
from google.adk.skills.skill_registry import SkillRegistry
from google.adk.utils import _mtls_utils
import google.auth
import google.auth.credentials
from google.auth.credentials import Credentials
import google.auth.exceptions
from google.auth.transport import mtls
from google.auth.transport import requests as auth_requests
import httpx


class GCPSkillRegistry(SkillRegistry):
  """GCP implementation of SkillRegistry using GCP Skill Registry API."""

  def __init__(
      self,
      *,
      project_id: str | None = None,
      location: str | None = None,
      credentials: Credentials | None = None,
  ):
    """Initializes the GCP Skill Registry.

    Args:
      project_id: Optional GCP project ID. If omitted, loads from environment.
      location: Optional GCP location. If omitted, loads from environment.
      credentials: Optional credentials to use for the client.
    """
    self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
    self.location = location or os.environ.get("GOOGLE_CLOUD_LOCATION")
    # Set up SSL context for mTLS if needed
    self._ssl_context = None
    use_client_cert = _mtls_utils.use_client_cert_effective()
    if use_client_cert and mtls.has_default_client_cert_source():
      try:
        client_cert_source = mtls.default_client_cert_source()
        cert_bytes, key_bytes = client_cert_source()
        fd_cert, cert_path = tempfile.mkstemp()
        fd_key, key_path = tempfile.mkstemp()
        try:
          with os.fdopen(fd_cert, "wb") as f:
            f.write(cert_bytes)
          with os.fdopen(fd_key, "wb") as f:
            f.write(key_bytes)
          self._ssl_context = ssl.create_default_context()
          self._ssl_context.load_cert_chain(
              certfile=cert_path, keyfile=key_path
          )
        finally:
          try:
            os.remove(cert_path)
          except OSError:
            pass
          try:
            os.remove(key_path)
          except OSError:
            pass
      except Exception:  # pylint: disable=broad-exception-caught
        # Fallback to default ssl configuration if cert source is broken
        pass

    self.base_url = os.environ.get(
        "AGENT_REGISTRY_ENDPOINT",
        _mtls_utils.get_api_endpoint(
            location="",
            default_template="https://agentregistry.googleapis.com/v1alpha",
            mtls_template="https://agentregistry.mtls.googleapis.com/v1alpha",
        ),
    )

    if not self.project_id or not self.location:
      raise ValueError(
          "project_id and location must be specified or set via environment"
          " variables."
      )
    self._credentials: Credentials | None = credentials

  async def _get_headers(self) -> dict[str, str]:
    """Refreshes credentials and returns authorization headers."""
    if self._credentials is None:
      try:
        self._credentials, _ = google.auth.default()
      except google.auth.exceptions.DefaultCredentialsError as e:
        raise RuntimeError(
            f"Failed to get default Google Cloud credentials: {e}"
        ) from e

    if not self._credentials.valid:
      # google.auth.credentials.Credentials.refresh is a blocking call,
      # so run it in a separate thread.
      request = auth_requests.Request()
      await asyncio.to_thread(self._credentials.refresh, request)

    quota_project_id = (
        getattr(self._credentials, "quota_project_id", None) or self.project_id
    )
    headers = {
        "Authorization": f"Bearer {self._credentials.token}",
        "Content-Type": "application/json",
    }
    if quota_project_id:
      headers["x-goog-user-project"] = quota_project_id
    return headers

  async def _make_request(
      self,
      client: httpx.AsyncClient,
      url: str,
      params: dict[str, Any] | None = None,
  ) -> httpx.Response:
    """Helper function to make GET requests to the Agent Registry API."""
    headers = await self._get_headers()
    try:
      response = await client.get(url, headers=headers, params=params)
      response.raise_for_status()
      return response
    except httpx.HTTPStatusError as e:
      raise RuntimeError(
          f"API request failed with status {e.response.status_code}:"
          f" {e.response.text}"
      ) from e
    except httpx.RequestError as e:
      raise RuntimeError(f"API request failed (network error): {e}") from e
    except Exception as e:
      raise RuntimeError(f"API request failed: {e}") from e

  def _create_httpx_client(self) -> httpx.AsyncClient:
    """Creates a new httpx.AsyncClient with appropriate SSL/mTLS configuration."""
    if self._ssl_context is not None:
      return httpx.AsyncClient(verify=self._ssl_context)
    return httpx.AsyncClient()

  async def get_skill(self, *, name: str) -> models.Skill:
    """Fetches a skill from the registry.

    Args:
      name: The name of the skill.

    Returns:
      A Skill object.
    """
    async with self._create_httpx_client() as client:
      # 1. Fetch the logical Skill metadata
      skill_url = (
          f"{self.base_url}/projects/{self.project_id}/"
          f"locations/{self.location}/skills/{name}"
      )
      response = await self._make_request(client, skill_url)
      skill_data = response.json()

      default_revision = skill_data.get("defaultRevision") or skill_data.get(
          "default_revision"
      )
      if not default_revision:
        raise ValueError(f"Skill '{name}' does not contain default revision.")

      # 2. Fetch the zipped filesystem via direct media download of default
      # revision
      revision_url = f"{self.base_url}/{default_revision}"
      media_response = await self._make_request(
          client, revision_url, params={"alt": "media"}
      )
      zip_bytes = media_response.content

    # pylint: disable=protected-access
    return await asyncio.to_thread(_utils._load_skill_from_zip_bytes, zip_bytes)

  async def search_skills(self, *, query: str) -> list[models.Frontmatter]:
    """Searches for skills in the registry.

    Args:
      query: The search query.

    Returns:
      A list of Frontmatter objects for discovery.
    """
    async with self._create_httpx_client() as client:
      url = (
          f"{self.base_url}/projects/{self.project_id}/"
          f"locations/{self.location}/skills:search"
      )
      params = {
          "search_string": query,
      }
      response = await self._make_request(client, url, params=params)
      response_data = response.json()

      results = []
      for s in response_data.get("skills", []):
        results.append(
            models.Frontmatter(
                name=s.get("name", "").split("/")[-1],
                description=s.get("description", "") or "",
            )
        )
      return results
