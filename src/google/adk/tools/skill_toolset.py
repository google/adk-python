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

import logging
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types

from ..agents.readonly_context import ReadonlyContext
from ..code_executors.base_code_executor import BaseCodeExecutor
from ..code_executors.code_execution_utils import CodeExecutionInput
from ..features import experimental
from ..features import FeatureName
from ..skills import models
from ..skills import prompt
from .base_tool import BaseTool
from .base_toolset import BaseToolset
from .tool_context import ToolContext

if TYPE_CHECKING:
  from ..models.llm_request import LlmRequest

logger = logging.getLogger("google_adk." + __name__)

DEFAULT_SKILL_SYSTEM_INSTRUCTION = """You can use specialized 'skills' to help you with complex tasks. You MUST use the skill tools to interact with these skills.

Skills are folders of instructions and resources that extend your capabilities for specialized tasks. Each skill folder contains:
- **SKILL.md** (required): The main instruction file with skill metadata and detailed markdown instructions.
- **references/** (Optional): Additional documentation or examples for skill usage.
- **assets/** (Optional): Templates, scripts or other resources used by the skill.
- **scripts/** (Optional): Executable scripts that can be run via bash.

This is very important:

1. If a skill seems relevant to the current user query, you MUST use the `load_skill` tool with `name="<SKILL_NAME>"` to read its full instructions before proceeding.
2. Once you have read the instructions, follow them exactly as documented before replying to the user. For example, If the instruction lists multiple steps, please make sure you complete all of them in order.
3. The `load_skill_resource` tool is for viewing files within a skill's directory (e.g., `references/*`, `assets/*`, `scripts/*`). Do NOT use other tools to access these files.
4. Use `execute_skill_script` to run scripts from a skill's `scripts/` directory. Use `load_skill_resource` to view script content first if needed.
"""


@experimental(FeatureName.SKILL_TOOLSET)
class ListSkillsTool(BaseTool):
  """Tool to list all available skills."""

  def __init__(self, toolset: "SkillToolset"):
    super().__init__(
        name="list_skills",
        description=(
            "Lists all available skills with their names and descriptions."
        ),
    )
    self._toolset = toolset

  def _get_declaration(self) -> types.FunctionDeclaration | None:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    )

  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    skills = self._toolset._list_skills()
    return prompt.format_skills_as_xml(skills)


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
  """Tool to load resources (references, assets, or scripts) from a skill."""

  def __init__(self, toolset: "SkillToolset"):
    super().__init__(
        name="load_skill_resource",
        description=(
            "Loads a resource file (from references/, assets/, or"
            " scripts/) from within a skill."
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
                        " 'references/my_doc.md', 'assets/template.txt',"
                        " or 'scripts/setup.sh')."
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
    elif resource_path.startswith("scripts/"):
      script_name = resource_path[len("scripts/") :]
      script = skill.resources.get_script(script_name)
      if script is not None:
        content = script.src
    else:
      return {
          "error": (
              "Path must start with 'references/', 'assets/', or 'scripts/'."
          ),
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
class ExecuteSkillScriptTool(BaseTool):
  """Tool to execute scripts from a skill's scripts/ directory."""

  def __init__(self, toolset: "SkillToolset"):
    super().__init__(
        name="execute_skill_script",
        description=(
            "Executes a script from a skill's scripts/ directory"
            " and returns its output."
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
                "script_name": {
                    "type": "string",
                    "description": (
                        "The name of the script to execute (e.g.,"
                        " 'setup.sh' or 'scripts/setup.sh')."
                    ),
                },
                "input_args": {
                    "type": "string",
                    "description": (
                        "Optional space-separated arguments to pass"
                        " to the script."
                    ),
                },
            },
            "required": ["skill_name", "script_name"],
        },
    )

  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    skill_name = args.get("skill_name")
    script_name = args.get("script_name")
    input_args = args.get("input_args", "")

    if not skill_name:
      return {
          "error": "Skill name is required.",
          "error_code": "MISSING_SKILL_NAME",
      }
    if not script_name:
      return {
          "error": "Script name is required.",
          "error_code": "MISSING_SCRIPT_NAME",
      }

    # Strip scripts/ prefix for consistency
    if script_name.startswith("scripts/"):
      script_name = script_name[len("scripts/") :]

    skill = self._toolset._get_skill(skill_name)
    if not skill:
      return {
          "error": f"Skill '{skill_name}' not found.",
          "error_code": "SKILL_NOT_FOUND",
      }

    script = skill.resources.get_script(script_name)
    if script is None:
      return {
          "error": f"Script '{script_name}' not found in skill '{skill_name}'.",
          "error_code": "SCRIPT_NOT_FOUND",
      }

    # Resolve code executor: toolset-level first, then agent fallback
    code_executor = self._toolset._code_executor
    if code_executor is None:
      agent = tool_context._invocation_context.agent
      if hasattr(agent, "code_executor"):
        code_executor = agent.code_executor
    if code_executor is None:
      return {
          "error": (
              "No code executor configured. A code executor is"
              " required to run scripts."
          ),
          "error_code": "NO_CODE_EXECUTOR",
      }

    # Prepare code based on script extension
    code = self._prepare_code(script_name, script.src, input_args)
    if code is None:
      ext = script_name.rsplit(".", 1)[-1] if "." in script_name else ""
      return {
          "error": (
              f"Unsupported script type '.{ext}'. Supported"
              " types: .py, .sh, .bash"
          ),
          "error_code": "UNSUPPORTED_SCRIPT_TYPE",
      }

    try:
      result = code_executor.execute_code(
          tool_context._invocation_context,
          CodeExecutionInput(code=code),
      )
      return {
          "skill_name": skill_name,
          "script_name": script_name,
          "stdout": result.stdout,
          "stderr": result.stderr,
          "status": "error" if result.stderr else "success",
      }
    except Exception as e:
      logger.exception(
          "Error executing script '%s' from skill '%s'",
          script_name,
          skill_name,
      )
      # Keep the error message short for the LLM; full trace is logged above.
      short_msg = str(e)
      if len(short_msg) > 200:
        short_msg = short_msg[:200] + "..."
      return {
          "error": f"Failed to execute script '{script_name}': {short_msg}",
          "error_code": "EXECUTION_ERROR",
      }

  def _prepare_code(
      self,
      script_name: str,
      script_src: str,
      input_args: str,
  ) -> str | None:
    """Prepares Python code to execute the script.

    Args:
      script_name: The script filename.
      script_src: The script source content.
      input_args: Optional arguments string.

    Returns:
      Python code string to execute, or None if unsupported type.
    """
    ext = ""
    if "." in script_name:
      ext = script_name.rsplit(".", 1)[-1].lower()

    if ext in ("py", ""):
      # Python script: execute directly, inject sys.argv if args
      if input_args:
        return (
            "import sys\n"
            f"sys.argv = [{script_name!r}] + {input_args!r}.split()\n"
            + script_src
        )
      return script_src
    elif ext in ("sh", "bash"):
      # Shell script: wrap in subprocess.run
      return (
          "import subprocess\n"
          "_result = subprocess.run(\n"
          f"    ['bash', '-c', {script_src!r}"
          + (f" + ' ' + {input_args!r}" if input_args else "")
          + f"],\n"
          f"    capture_output=True, text=True,\n"
          f")\n"
          f"print(_result.stdout, end='')\n"
          f"if _result.stderr:\n"
          f"    import sys\n"
          f"    print(_result.stderr, end='', file=sys.stderr)\n"
      )
    return None


@experimental(FeatureName.SKILL_TOOLSET)
class SkillToolset(BaseToolset):
  """A toolset for managing and interacting with agent skills."""

  def __init__(
      self,
      skills: list[models.Skill],
      *,
      code_executor: Optional[BaseCodeExecutor] = None,
  ):
    super().__init__()

    # Check for duplicate skill names
    seen: set[str] = set()
    for skill in skills:
      if skill.name in seen:
        raise ValueError(f"Duplicate skill name '{skill.name}'.")
      seen.add(skill.name)

    self._skills = {skill.name: skill for skill in skills}
    self._code_executor = code_executor
    self._tools = [
        ListSkillsTool(self),
        LoadSkillTool(self),
        LoadSkillResourceTool(self),
        ExecuteSkillScriptTool(self),
    ]

  async def get_tools(
      self, readonly_context: ReadonlyContext | None = None
  ) -> list[BaseTool]:
    """Returns the list of tools in this toolset."""
    return self._tools

  def _get_skill(self, name: str) -> models.Skill | None:
    """Retrieves a skill by name."""
    return self._skills.get(name)

  def _list_skills(self) -> list[models.Skill]:
    """Lists all available skills."""
    return list(self._skills.values())

  async def process_llm_request(
      self, *, tool_context: ToolContext, llm_request: LlmRequest
  ) -> None:
    """Processes the outgoing LLM request to include available skills."""
    skills = self._list_skills()
    skills_xml = prompt.format_skills_as_xml(skills)
    instructions = []
    instructions.append(DEFAULT_SKILL_SYSTEM_INSTRUCTION)
    instructions.append(skills_xml)
    llm_request.append_instructions(instructions)
