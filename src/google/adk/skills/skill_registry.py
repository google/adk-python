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

"""Interface for a Skill Registry in ADK."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from .models import Frontmatter
from .models import Skill


class SkillRegistry(ABC):
  """Interface for a skill registry."""

  @abstractmethod
  async def get_skill(self, *, name: str) -> Skill:
    """Fetches a skill from the registry.

    Args:
        name: The name of the skill.

    Returns:
        A Skill object.

    Raises:
        Exception: If the skill with the specified name does not exist.
    """
    pass

  @abstractmethod
  async def search_skills(self, *, query: str) -> list[Frontmatter]:
    """Searches for skills in the registry.

    Args:
        query: The search query.

    Returns:
        A list of Frontmatter objects for discovery.
    """
    pass

  def search_tool_description(self) -> str | None:
    """Returns the description for the search_skills tool.

    Registries can define this to provide specialized instructions to the model
    on how to use their specific search capabilities.
    """
    return None
