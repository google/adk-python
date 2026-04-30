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

"""Interface for skill registry."""

from __future__ import annotations

import abc
from typing import Any
from typing import Dict
from typing import List

from . import models


class SkillRegistry(abc.ABC):
  """Interface for a skill registry."""

  @abc.abstractmethod
  async def get_skill(
      self, *, name: str, version: str | None = None
  ) -> models.Skill:
    """Fetches a skill from the registry.

    Args:
        name: The name of the skill.
        version: Optional version of the skill.

    Returns:
        A Skill object.
    """
    pass

  @abc.abstractmethod
  async def search_skills(
      self,
      *,
      query: str,
      filters: Dict[str, Any] | None = None,
      **kwargs,
  ) -> List[models.Frontmatter]:
    """Searches for skills in the registry.

    Args:
        query: The search query.
        filters: Optional filters.
        **kwargs: Additional implementation-specific arguments.

    Returns:
        A list of Frontmatter objects for discovery.
    """
    pass

  @abc.abstractmethod
  def get_filter_schema(self) -> Dict[str, Any] | None:
    """Returns the JSON schema for the filters supported by this registry.

    Returns:
        A JSON schema dict or None if filters are not supported
    """
    pass

  def get_search_description(self) -> str:
    """Returns the description for the search_skills tool.

    Registries can override this to provide specialized instructions to the
    model on how to use their specific search capabilities.
    """
    return (
        "Searches for relevant skills in the registry based on a semantic or"
        " keyword query."
    )
