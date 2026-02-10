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

"""Toolset for discovering, viewing, and executing agent skills."""

from __future__ import annotations

from typing import Any

from google.genai import types

from ..agents.readonly_context import ReadonlyContext
from ..features import experimental
from ..features import FeatureName
from ..models.llm_request import LlmRequest
from ..skills import models
from ..skills import prompt
from .base_tool import BaseTool
from .base_toolset import BaseToolset
from .tool_context import ToolContext


@experimental(FeatureName.SKILL_TOOLSET)
class LoadSkillTool(BaseTool):
  """Tool to load a skill's instructions."""

  def __init__(self, toolset: "SkillToolset"):
    super().__init__(
        name="load_skill",
        description="Loads the SKILL.md instructions for a given skill.",
    )
    self._toolset = toolset

  def _get_declaration(self) -> types.FunctionDeclaration | None:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters_json_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name of the skill to load.",
                },
            },
            "required": ["name"],
        },
    )

  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    skill_name = args.get("name")
    if not skill_name:
      return {
          "error": "Skill name is required.",
          "error_code": "MISSING_SKILL_NAME",
      }

    skill = self._toolset._get_skill(skill_name)
    if not skill:
      return {
          "error": f"Skill '{skill_name}' not found.",
          "error_code": "SKILL_NOT_FOUND",
      }

    return {
        "skill_name": skill_name,
        "instructions": skill.instructions,
        "frontmatter": skill.frontmatter.model_dump(),
    }


@experimental(FeatureName.SKILL_TOOLSET)
class LoadSkillResourceTool(BaseTool):
  """Tool to load resources (references or assets) from a skill."""

  def __init__(self, toolset: "SkillToolset"):
    super().__init__(
        name="load_skill_resource",
        description=(
            "Loads a resource file (from references/ or assets/) from within a"
            " skill."
        ),
    )
    self._toolset = toolset

  def _get_declaration(self) -> types.FunctionDeclaration | None:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters_json_schema={
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "The name of the skill.",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "The relative path to the resource (e.g.,"
                        " 'references/my_doc.md' or 'assets/template.txt')."
                    ),
                },
            },
            "required": ["skill_name", "path"],
        },
    )

  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    skill_name = args.get("skill_name")
    resource_path = args.get("path")

    if not skill_name:
      return {
          "error": "Skill name is required.",
          "error_code": "MISSING_SKILL_NAME",
      }
    if not resource_path:
      return {
          "error": "Resource path is required.",
          "error_code": "MISSING_RESOURCE_PATH",
      }

    skill = self._toolset._get_skill(skill_name)
    if not skill:
      return {
          "error": f"Skill '{skill_name}' not found.",
          "error_code": "SKILL_NOT_FOUND",
      }

    content = None
    if resource_path.startswith("references/"):
      ref_name = resource_path[len("references/") :]
      content = skill.resources.get_reference(ref_name)
    elif resource_path.startswith("assets/"):
      asset_name = resource_path[len("assets/") :]
      content = skill.resources.get_asset(asset_name)
    else:
      return {
          "error": "Path must start with 'references/' or 'assets/'.",
          "error_code": "INVALID_RESOURCE_PATH",
      }

    if content is None:
      return {
          "error": (
              f"Resource '{resource_path}' not found in skill '{skill_name}'."
          ),
          "error_code": "RESOURCE_NOT_FOUND",
      }

    return {
        "skill_name": skill_name,
        "path": resource_path,
        "content": content,
    }


@experimental(FeatureName.SKILL_TOOLSET)
class SkillToolset(BaseToolset):
  """A toolset for managing and interacting with agent skills."""

  def __init__(self, skills: list[models.Skill]):
    super().__init__()
    self._skills = {skill.name: skill for skill in skills}
    self._tools = [
        LoadSkillTool(self),
        LoadSkillResourceTool(self),
    ]

  async def get_tools(
      self, readonly_context: ReadonlyContext | None = None
  ) -> list[BaseTool]:
    """Returns the list of tools in this toolset."""
    return self._tools

  def _get_skill(self, name: str) -> models.Skill | None:
    """Retrieves a skill by name."""
    return self._skills.get(name)

  def _list_skills(self) -> list[models.Frontmatter]:
    """Lists the frontmatter of all available skills."""
    return [s.frontmatter for s in self._skills.values()]

  async def process_llm_request(
      self,
      *,
      tool_context: ToolContext,
      llm_request: LlmRequest,
  ) -> None:
    """Adds available skills to the system instruction."""

    skill_frontmatters = self._list_skills()

    # Append the skill instruction into the system instruction
    skills_xml = prompt.format_skills_as_xml(skill_frontmatters)
    skill_si = f"""
You can use specialized 'skills' to help you with complex tasks. Each skill has a name and a description listed below:
{skills_xml}

Skills are folders of instructions and resources that extend your capabilities for specialized tasks. Each skill folder contains:
- **SKILL.md** (required): The main instruction file with skill metadata and detailed markdown instructions.
- **references/** (Optional): Additional documentation or examples for skill usage.
- **assets/** (Optional): Templates, scripts or other resources used by the skill.

This is very important:

1. If a skill seems relevant to the current user query, you MUST use the `load_skill` tool with `name="<SKILL_NAME>"` to read its full instructions before proceeding.
2. Once you have read the instructions, follow them exactly as documented before replying to the user. For example, If the instruction lists multiple steps, please make sure you complete all of them in order.
3. The `load_skill_resource` tool is for viewing files within a skill's directory (e.g., `references/*`, `assets/*`). Do NOT use other tools to access these files.
"""

    llm_request.append_instructions([skill_si])
