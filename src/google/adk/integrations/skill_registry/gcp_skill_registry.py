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
import base64
import os

from google.adk.dependencies.vertexai import vertexai
from google.adk.skills import _utils
from google.adk.skills import models
from google.adk.skills.skill_registry import SkillRegistry


class GCPSkillRegistry(SkillRegistry):
  """GCP implementation of SkillRegistry using GCP Skill Registry API."""

  def __init__(
      self, *, project_id: str | None = None, location: str | None = None
  ):
    """Initializes the GCP Skill Registry.

    Args:
      project_id: Optional GCP project ID. If omitted, loads from environment.
      location: Optional GCP location. If omitted, loads from environment.
    """
    self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
    self.location = location or os.environ.get("GOOGLE_CLOUD_LOCATION")
    self._lazy_client: vertexai.AsyncClient | None = None

  @property
  def _client(self) -> vertexai.AsyncClient:
    if self._lazy_client is None:
      self._lazy_client = vertexai.Client(
          project=self.project_id,
          location=self.location,
          http_options={
              "api_version": "v1beta1",
          },
      ).aio
    return self._lazy_client

  async def get_skill(self, *, name: str) -> models.Skill:
    """Fetches a skill from the registry.

    Args:
      name: The name of the skill.

    Returns:
      A Skill object.
    """
    full_name = (
        f"projects/{self.project_id}/locations/{self.location}/skills/{name}"
    )
    skill_resource = await self._client.skills.get(name=full_name)

    zip_bytes_base64 = skill_resource.zipped_filesystem
    if not zip_bytes_base64:
      raise ValueError(f"Skill '{name}' does not contain zipped filesystem.")

    zip_bytes = base64.b64decode(zip_bytes_base64)

    return await asyncio.to_thread(_utils._load_skill_from_zip_bytes, zip_bytes)

  async def search_skills(self, *, query: str) -> list[models.Frontmatter]:
    """Searches for skills in the registry.

    Args:
      query: The search query.

    Returns:
      A list of Frontmatter objects for discovery.
    """
    response = await self._client.skills.retrieve(query=query)

    results = []
    if response.retrieved_skills:
      for s in response.retrieved_skills:
        results.append(
            models.Frontmatter(
                name=s.skill_name.split("/")[-1] if s.skill_name else "",
                description=s.description or "",
            )
        )
    return results
