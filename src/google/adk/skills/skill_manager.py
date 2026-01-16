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

"""Skills manager for ADK.

Manages skill registration, discovery, and execution.
"""

from __future__ import annotations

import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from ..utils.feature_decorator import experimental
from .base_skill import BaseSkill


class SkillInvocationResult(BaseModel):
  """Result from a skill invocation."""

  model_config = ConfigDict(extra="forbid")

  success: bool = Field(
      description="Whether the skill execution was successful.",
  )
  result: Any = Field(
      default=None,
      description="The result of the skill execution.",
  )
  error: Optional[str] = Field(
      default=None,
      description="Error message if execution failed.",
  )
  call_log: List[dict] = Field(
      default_factory=list,
      description="Log of tool calls made during execution.",
  )
  execution_time_ms: float = Field(
      default=0.0,
      description="Execution time in milliseconds.",
  )


@experimental
class SkillsManager:
  """Manages skill registration, discovery, and execution.

  The SkillsManager provides:
  - Skill registration and lookup
  - Skill discovery for agents
  - Execution coordination

  Example:
    ```python
    manager = SkillsManager()
    manager.register_skill(my_skill)

    skill = manager.get_skill("my_skill")
    all_skills = manager.get_all_skills()
    ```
  """

  def __init__(self):
    """Initialize the skills manager."""
    self._skills: Dict[str, BaseSkill] = {}

  def register_skill(self, skill: BaseSkill) -> None:
    """Register a skill.

    Args:
      skill: The skill to register.

    Raises:
      ValueError: If a skill with the same name is already registered.
    """
    if skill.name in self._skills:
      raise ValueError(f"Skill '{skill.name}' already registered")
    self._skills[skill.name] = skill

  def unregister_skill(self, name: str) -> bool:
    """Unregister a skill by name.

    Args:
      name: The name of the skill to unregister.

    Returns:
      True if the skill was unregistered, False if it wasn't found.
    """
    if name in self._skills:
      del self._skills[name]
      return True
    return False

  def get_skill(self, name: str) -> Optional[BaseSkill]:
    """Get a registered skill by name.

    Args:
      name: The name of the skill to retrieve.

    Returns:
      The skill if found, None otherwise.
    """
    return self._skills.get(name)

  def get_all_skills(self) -> List[BaseSkill]:
    """Get all registered skills.

    Returns:
      List of all registered skills.
    """
    return list(self._skills.values())

  def get_skill_names(self) -> List[str]:
    """Get names of all registered skills.

    Returns:
      List of skill names.
    """
    return list(self._skills.keys())

  def has_skill(self, name: str) -> bool:
    """Check if a skill is registered.

    Args:
      name: The name of the skill to check.

    Returns:
      True if the skill is registered.
    """
    return name in self._skills

  def clear(self) -> None:
    """Remove all registered skills."""
    self._skills.clear()

  @staticmethod
  async def execute_skill(
      skill: BaseSkill,
      tool_results: Dict[str, Any],
  ) -> SkillInvocationResult:
    """Execute a skill with pre-computed tool results.

    This is a simplified execution that doesn't use PTC code generation.
    It applies the skill's result filtering to the provided results.

    Args:
      skill: The skill to execute.
      tool_results: Dictionary of tool results to filter.

    Returns:
      SkillInvocationResult with filtered results.
    """
    start_time = time.time()

    try:
      filtered_result = skill.filter_result(tool_results)
      execution_time = (time.time() - start_time) * 1000

      return SkillInvocationResult(
          success=True,
          result=filtered_result,
          execution_time_ms=execution_time,
      )
    except Exception as e:
      execution_time = (time.time() - start_time) * 1000
      return SkillInvocationResult(
          success=False,
          error=str(e),
          execution_time_ms=execution_time,
      )
