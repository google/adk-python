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

"""SkillTool - Wraps a Skill as a BaseTool for LLM invocation.

This module provides the SkillTool class which exposes skills as ADK tools,
enabling LLMs to interact with Agent Skills standard skills through a
unified tool interface.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Union

from google.genai import types
from typing_extensions import override

from ..tools.base_tool import BaseTool
from ..tools.tool_context import ToolContext
from ..utils.feature_decorator import experimental
from .base_skill import BaseSkill
from .markdown_skill import MarkdownSkill
from .script_executor import ScriptExecutionResult
from .script_executor import ScriptExecutor

logger = logging.getLogger("google_adk." + __name__)


@experimental
class SkillTool(BaseTool):
  """Wraps a Skill as a BaseTool for LLM invocation.

  Provides three action types:
  - "activate": Load full skill instructions (Stage 2)
  - "run_script": Execute a bundled script (Stage 3)
  - "load_reference": Load a reference document (Stage 3)

  This tool enables LLMs to interact with Agent Skills standard skills
  through the standard ADK tool interface.

  Example:
    ```python
    from google.adk.skills import MarkdownSkill, SkillTool

    # Load a skill
    skill = MarkdownSkill.from_directory("/path/to/pdf-processing")

    # Create tool wrapper
    tool = SkillTool(skill)

    # LLM can now invoke:
    # - {"action": "activate"} → Returns full instructions
    # - {"action": "run_script", "script": "extract.py", "args": ["file.pdf"]}
    # - {"action": "load_reference", "reference": "FORMS.md"}
    ```
  """

  def __init__(
      self,
      skill: BaseSkill,
      script_executor: Optional[ScriptExecutor] = None,
      working_dir: Optional[Union[str, Path]] = None,
  ):
    """Initialize the SkillTool.

    Args:
      skill: The skill to wrap.
      script_executor: Optional custom script executor. If not provided,
        a default executor will be created.
      working_dir: Default working directory for script execution.
        If not provided, uses the skill's directory.
    """
    # Create safe tool name from skill name
    safe_name = skill.name.replace("-", "_").replace(".", "_")

    super().__init__(
        name=f"skill_{safe_name}",
        description=self._build_description(skill),
    )
    self._skill = skill
    self._script_executor = script_executor or ScriptExecutor()

    # Set working directory
    if working_dir:
      self._working_dir = Path(working_dir).resolve()
    elif isinstance(skill, MarkdownSkill):
      self._working_dir = skill.skill_path
    else:
      self._working_dir = Path.cwd()

  def _build_description(self, skill: BaseSkill) -> str:
    """Build tool description from skill metadata.

    Args:
      skill: The skill to describe.

    Returns:
      Formatted description string.
    """
    lines = [
        skill.description,
        "",
        "Actions:",
        "- activate: Load full skill instructions",
    ]

    if isinstance(skill, MarkdownSkill):
      scripts = skill.list_scripts()
      if scripts:
        lines.append(
            f"- run_script: Execute bundled scripts ({', '.join(scripts[:5])})"
        )
        if len(scripts) > 5:
          lines[-1] += f" and {len(scripts) - 5} more"

      refs = skill.list_references()
      if refs:
        lines.append(
            "- load_reference: Load reference documents"
            f" ({', '.join(refs[:5])})"
        )
        if len(refs) > 5:
          lines[-1] += f" and {len(refs) - 5} more"

    return "\n".join(lines)

  @override
  def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    """Get function declaration for LLM.

    Returns:
      FunctionDeclaration describing the tool's parameters.
    """
    # Build action enum based on skill capabilities
    actions = ["activate"]
    if isinstance(self._skill, MarkdownSkill):
      if self._skill.has_scripts():
        actions.append("run_script")
      if self._skill.has_references():
        actions.append("load_reference")

    properties = {
        "action": types.Schema(
            type="STRING",
            description=f"Action to perform: {', '.join(actions)}",
            enum=actions,
        ),
    }

    # Add script-related parameters if skill has scripts
    if isinstance(self._skill, MarkdownSkill) and self._skill.has_scripts():
      scripts = self._skill.list_scripts()
      properties["script"] = types.Schema(
          type="STRING",
          description=(
              "Script name for run_script action. Available:"
              f" {', '.join(scripts)}"
          ),
      )
      properties["args"] = types.Schema(
          type="ARRAY",
          items=types.Schema(type="STRING"),
          description="Command-line arguments for the script",
      )

    # Add reference parameter if skill has references
    if isinstance(self._skill, MarkdownSkill) and self._skill.has_references():
      refs = self._skill.list_references()
      properties["reference"] = types.Schema(
          type="STRING",
          description=(
              "Reference file for load_reference action. Available:"
              f" {', '.join(refs)}"
          ),
      )

    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters=types.Schema(
            type="OBJECT",
            properties=properties,
            required=["action"],
        ),
    )

  @override
  async def run_async(
      self,
      *,
      args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Any:
    """Execute the skill action.

    Args:
      args: Arguments from the LLM containing action and parameters.
      tool_context: The tool execution context.

    Returns:
      Result dictionary with action-specific content.
    """
    action = args.get("action", "activate")

    logger.debug(
        "SkillTool %s executing action: %s",
        self._skill.name,
        action,
    )

    if action == "activate":
      return self._handle_activate()

    elif action == "run_script":
      return await self._handle_run_script(args)

    elif action == "load_reference":
      return self._handle_load_reference(args)

    else:
      return {
          "error": f"Unknown action: {action}",
          "available_actions": ["activate", "run_script", "load_reference"],
      }

  def _handle_activate(self) -> dict[str, Any]:
    """Handle skill activation (Stage 2).

    Returns:
      Dictionary with skill instructions and available resources.
    """
    if isinstance(self._skill, MarkdownSkill):
      instructions = self._skill.get_instructions()
      return {
          "status": "activated",
          "skill": self._skill.name,
          "instructions": instructions,
          "available_scripts": self._skill.list_scripts(),
          "available_references": self._skill.list_references(),
          "available_assets": self._skill.list_assets()[:20],  # Limit assets
      }
    else:
      # For non-markdown skills, return the skill prompt
      return {
          "status": "activated",
          "skill": self._skill.name,
          "prompt": self._skill.get_skill_prompt(),
          "tool_declarations": self._skill.get_tool_declarations(),
      }

  async def _handle_run_script(self, args: dict[str, Any]) -> dict[str, Any]:
    """Handle script execution (Stage 3).

    Args:
      args: Arguments containing script name and args.

    Returns:
      Dictionary with script execution results.
    """
    if not isinstance(self._skill, MarkdownSkill):
      return {"error": "This skill does not support scripts"}

    script_name = args.get("script")
    if not script_name:
      return {
          "error": "Script name required",
          "available_scripts": self._skill.list_scripts(),
      }

    script_args = args.get("args", [])
    if isinstance(script_args, str):
      # Handle case where args is passed as a single string
      script_args = script_args.split() if script_args else []

    # Get script path
    script_path = self._skill.get_script_path(script_name)
    if not script_path:
      return {
          "error": f"Script not found: {script_name}",
          "available_scripts": self._skill.list_scripts(),
      }

    logger.info(
        "Executing script: %s with args: %s",
        script_name,
        script_args,
    )

    # Execute script
    try:
      result: ScriptExecutionResult = (
          await self._script_executor.execute_script(
              script_path=script_path,
              args=script_args,
              working_dir=self._working_dir,
          )
      )

      return {
          "script": script_name,
          "success": result.success,
          "stdout": result.stdout,
          "stderr": result.stderr,
          "return_code": result.return_code,
          "execution_time_ms": result.execution_time_ms,
          "timed_out": result.timed_out,
      }
    except Exception as e:
      logger.error("Script execution failed: %s", e)
      return {
          "script": script_name,
          "success": False,
          "error": str(e),
      }

  def _handle_load_reference(self, args: dict[str, Any]) -> dict[str, Any]:
    """Handle reference loading (Stage 3).

    Args:
      args: Arguments containing reference name.

    Returns:
      Dictionary with reference content.
    """
    if not isinstance(self._skill, MarkdownSkill):
      return {"error": "This skill does not support references"}

    ref_name = args.get("reference")
    if not ref_name:
      return {
          "error": "Reference name required",
          "available_references": self._skill.list_references(),
      }

    content = self._skill.get_reference(ref_name)
    if content is None:
      return {
          "error": f"Reference not found: {ref_name}",
          "available_references": self._skill.list_references(),
      }

    return {
        "reference": ref_name,
        "content": content,
    }

  @property
  def skill(self) -> BaseSkill:
    """Get the wrapped skill."""
    return self._skill

  @property
  def working_dir(self) -> Path:
    """Get the working directory for script execution."""
    return self._working_dir

  @working_dir.setter
  def working_dir(self, value: Union[str, Path]) -> None:
    """Set the working directory for script execution."""
    self._working_dir = Path(value).resolve()


def create_skill_tools(
    skills: list[BaseSkill],
    script_executor: Optional[ScriptExecutor] = None,
    working_dir: Optional[Union[str, Path]] = None,
) -> list[SkillTool]:
  """Create SkillTool instances for a list of skills.

  Convenience function to create tool wrappers for multiple skills.

  Args:
    skills: List of skills to wrap.
    script_executor: Optional shared script executor.
    working_dir: Optional default working directory.

  Returns:
    List of SkillTool instances.

  Example:
    ```python
    from google.adk.skills import AgentSkillLoader, create_skill_tools

    loader = AgentSkillLoader()
    loader.add_skill_directory("./skills")

    tools = create_skill_tools(loader.get_all_skills())
    ```
  """
  return [
      SkillTool(
          skill=skill,
          script_executor=script_executor,
          working_dir=working_dir,
      )
      for skill in skills
  ]
