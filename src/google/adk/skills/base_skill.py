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

"""Base skill class for ADK skills.

Skills bundle related tools with orchestration logic for efficient
programmatic tool calling (PTC).
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from ..utils.feature_decorator import experimental


class SkillConfig(BaseModel):
  """Configuration for skill execution."""

  model_config = ConfigDict(extra="forbid")

  max_parallel_calls: int = Field(
      default=10,
      description="Maximum concurrent tool invocations in orchestration code.",
  )
  timeout_seconds: float = Field(
      default=60.0,
      description="Execution timeout for the skill.",
  )
  allow_network: bool = Field(
      default=False,
      description="Whether to allow network access in sandbox execution.",
  )
  memory_limit_mb: int = Field(
      default=256,
      description="Memory limit for execution in megabytes.",
  )


@experimental
class BaseSkill(ABC, BaseModel):
  """Abstract base class for all skills.

  Skills bundle related tools with orchestration logic for efficient
  programmatic tool calling (PTC). A skill provides:

  - A set of related tools that work together
  - Semantic grouping with descriptions for LLM understanding
  - Orchestration templates showing how to use the tools
  - Result filtering to reduce context size
  - Configuration for execution limits

  Example:
    ```python
    class DatabaseSkill(BaseSkill):
        name = "database"
        description = "Query and manage database records"

        def get_tool_declarations(self):
            return [
                {"name": "query", "description": "Execute SQL query"},
                {"name": "insert", "description": "Insert records"},
            ]

        def get_orchestration_template(self):
            return '''
    async def db_operation(tools):
        results = await tools.query(sql="SELECT * FROM users LIMIT 10")
        return {"users": results, "count": len(results)}
    '''

        def filter_result(self, result):
            # Remove sensitive fields
            for user in result.get("users", []):
                user.pop("password_hash", None)
            return result
    ```
  """

  model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

  name: str = Field(
      description="Unique identifier for the skill.",
  )
  description: str = Field(
      description="LLM-readable description of what this skill does.",
  )
  config: SkillConfig = Field(
      default_factory=SkillConfig,
      description="Execution configuration for this skill.",
  )
  allowed_callers: List[str] = Field(
      default_factory=lambda: ["code_execution_20250825"],
      description=(
          "Who can invoke this skill programmatically. "
          "Set to ['code_execution_20250825'] to enable PTC."
      ),
  )

  @abstractmethod
  def get_orchestration_template(self) -> str:
    """Return example orchestration code for this skill.

    The template should demonstrate how to use the skill's tools
    effectively. This helps the LLM understand usage patterns.

    Returns:
      A Python async function as a string that shows example usage.
    """

  @abstractmethod
  def get_tool_declarations(self) -> List[dict[str, Any]]:
    """Return tool declarations for LLM code generation.

    Each declaration should include at minimum:
    - name: The tool name
    - description: What the tool does

    Optionally include:
    - parameters: Dict of parameter names to descriptions

    Returns:
      List of tool declaration dictionaries.
    """

  def filter_result(self, result: Any) -> Any:
    """Filter results before returning to LLM context.

    Override this method to implement custom filtering logic.
    This is useful for:
    - Removing sensitive data (passwords, tokens)
    - Truncating large responses
    - Summarizing verbose output

    Args:
      result: The raw result from skill execution.

    Returns:
      The filtered result to be added to LLM context.
    """
    return result

  def get_skill_prompt(self) -> str:
    """Generate a prompt describing this skill for the LLM.

    Returns:
      A formatted string describing the skill and its tools.
    """
    tool_list = "\n".join(
        f"  - {t['name']}: {t.get('description', '')}"
        for t in self.get_tool_declarations()
    )
    return (
        f"Skill: {self.name}\n"
        f"Description: {self.description}\n\n"
        f"Available tools:\n{tool_list}\n\n"
        f"Example orchestration:\n{self.get_orchestration_template()}"
    )

  def is_programmatically_callable(self) -> bool:
    """Check if this skill can be called programmatically via PTC.

    Returns:
      True if the skill is enabled for programmatic tool calling.
    """
    return (
        self.allowed_callers is not None
        and "code_execution_20250825" in self.allowed_callers
    )
