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

# pylint: disable=g-import-not-at-top,protected-access

"""Toolset for discovering, viewing, and executing agent skills."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING
import warnings

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

_DEFAULT_SCRIPT_TIMEOUT = 300
_MAX_SKILL_PAYLOAD_BYTES = 16 * 1024 * 1024  # 16 MB

_DEFAULT_SKILL_SYSTEM_INSTRUCTION = """You can use specialized 'skills' to help you with complex tasks. You MUST use the skill tools to interact with these skills.

Skills are folders of instructions and resources that extend your capabilities for specialized tasks. Each skill folder contains:
- **SKILL.md** (required): The main instruction file with skill metadata and detailed markdown instructions.
- **references/** (Optional): Additional documentation or examples for skill usage.
- **assets/** (Optional): Templates, scripts or other resources used by the skill.
- **scripts/** (Optional): Executable scripts that can be run via bash.

This is very important:

1. If a skill seems relevant to the current user query, you MUST use the `load_skill` tool with `name="<SKILL_NAME>"` to read its full instructions before proceeding.
2. Once you have read the instructions, follow them exactly as documented before replying to the user. For example, If the instruction lists multiple steps, please make sure you complete all of them in order.
3. The `load_skill_resource` tool is for viewing files within a skill's directory (e.g., `references/*`, `assets/*`, `scripts/*`). Do NOT use other tools to access these files.
4. Use `run_skill_script` to run scripts from a skill's `scripts/` directory. Use `load_skill_resource` to view script content first if needed.
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


class _SkillScriptCodeExecutor:
  """A helper that materializes skill files and executes scripts."""

  _base_executor: BaseCodeExecutor
  _script_timeout: int

  def __init__(self, base_executor: BaseCodeExecutor, script_timeout: int):
    self._base_executor = base_executor
    self._script_timeout = script_timeout

  async def execute_script_async(
      self,
      invocation_context: Any,
      skill: models.Skill,
      script_path: str,
      script_args: dict[str, Any],
  ) -> dict[str, Any]:
    """Prepares and executes the script using the base executor."""
    code = self._build_wrapper_code(skill, script_path, script_args)
    if code is None:
      if "." in script_path:
        ext_msg = f"'.{script_path.rsplit('.', 1)[-1]}'"
      else:
        ext_msg = "(no extension)"
      return {
          "error": (
              f"Unsupported script type {ext_msg}."
              " Supported types: .py, .sh, .bash"
          ),
          "error_code": "UNSUPPORTED_SCRIPT_TYPE",
      }

    try:
      # Execute the self-contained script using the underlying executor
      result = await asyncio.to_thread(
          self._base_executor.execute_code,
          invocation_context,
          CodeExecutionInput(code=code),
      )

      stdout = result.stdout
      stderr = result.stderr

      # Shell scripts serialize both streams as JSON
      # through stdout; parse the envelope if present.
      rc = 0
      is_shell = "." in script_path and script_path.rsplit(".", 1)[
          -1
      ].lower() in ("sh", "bash")
      if is_shell and stdout:
        try:
          parsed = json.loads(stdout)
          if isinstance(parsed, dict) and parsed.get("__shell_result__"):
            stdout = parsed.get("stdout", "")
            stderr = parsed.get("stderr", "")
            rc = parsed.get("returncode", 0)
            if rc != 0 and not stderr:
              stderr = f"Exit code {rc}"
        except (json.JSONDecodeError, ValueError):
          pass

      status = "success"
      if rc != 0:
        status = "error"
      elif stderr and not stdout:
        status = "error"
      elif stderr:
        status = "warning"

      return {
          "skill_name": skill.name,
          "script_path": script_path,
          "stdout": stdout,
          "stderr": stderr,
          "status": status,
      }
    except SystemExit as e:
      if e.code in (None, 0):
        return {
            "skill_name": skill.name,
            "script_path": script_path,
            "stdout": "",
            "stderr": "",
            "status": "success",
        }
      return {
          "error": (
              f"Failed to execute script '{script_path}':"
              f" exited with code {e.code}"
          ),
          "error_code": "EXECUTION_ERROR",
      }
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.exception(
          "Error executing script '%s' from skill '%s'",
          script_path,
          skill.name,
      )
      short_msg = str(e)
      if len(short_msg) > 200:
        short_msg = short_msg[:200] + "..."
      return {
          "error": (
              "Failed to execute script"
              f" '{script_path}':\n{type(e).__name__}:"
              f" {short_msg}"
          ),
          "error_code": "EXECUTION_ERROR",
      }

  def _build_wrapper_code(
      self,
      skill: models.Skill,
      script_path: str,
      script_args: dict[str, Any],
  ) -> str | None:
    """Builds a self-extracting Python script."""
    ext = ""
    if "." in script_path:
      ext = script_path.rsplit(".", 1)[-1].lower()

    if not script_path.startswith("scripts/"):
      script_path = f"scripts/{script_path}"

    files_dict = {}
    for ref_name in skill.resources.list_references():
      content = skill.resources.get_reference(ref_name)
      if content is not None:
        files_dict[f"references/{ref_name}"] = content

    for asset_name in skill.resources.list_assets():
      content = skill.resources.get_asset(asset_name)
      if content is not None:
        files_dict[f"assets/{asset_name}"] = content

    for scr_name in skill.resources.list_scripts():
      scr = skill.resources.get_script(scr_name)
      if scr is not None and scr.src is not None:
        files_dict[f"scripts/{scr_name}"] = scr.src

    total_size = sum(
        len(v) if isinstance(v, (str, bytes)) else 0
        for v in files_dict.values()
    )
    if total_size > _MAX_SKILL_PAYLOAD_BYTES:
      logger.warning(
          "Skill '%s' resources total %d bytes, exceeding"
          " the recommended limit of %d bytes.",
          skill.name,
          total_size,
          _MAX_SKILL_PAYLOAD_BYTES,
      )

    # Build the boilerplate extract string
    code_lines = [
        "import os",
        "import tempfile",
        "import sys",
        "import json as _json",
        "import subprocess",
        "import runpy",
        f"_files = {files_dict!r}",
        "def _materialize_and_run():",
        "  _orig_cwd = os.getcwd()",
        "  with tempfile.TemporaryDirectory() as td:",
        "    for rel_path, content in _files.items():",
        "      full_path = os.path.join(td, rel_path)",
        "      os.makedirs(os.path.dirname(full_path), exist_ok=True)",
        "      mode = 'wb' if isinstance(content, bytes) else 'w'",
        "      with open(full_path, mode) as f:",
        "        f.write(content)",
        "    os.chdir(td)",
        "    try:",
    ]

    if ext == "py":
      argv_list = [script_path]
      for k, v in script_args.items():
        argv_list.extend([f"--{k}", str(v)])
      code_lines.extend([
          f"      sys.argv = {argv_list!r}",
          "      try:",
          f"        runpy.run_path({script_path!r}, run_name='__main__')",
          "      except SystemExit as e:",
          "        if e.code is not None and e.code != 0:",
          "          raise e",
      ])
    elif ext in ("sh", "bash"):
      arr = ["bash", script_path]
      for k, v in script_args.items():
        arr.extend([f"--{k}", str(v)])
      timeout = self._script_timeout
      code_lines.extend([
          "      try:",
          "        _r = subprocess.run(",
          f"          {arr!r},",
          "          capture_output=True, text=True,",
          f"          timeout={timeout!r}, cwd=td,",
          "        )",
          "        print(_json.dumps({",
          "            '__shell_result__': True,",
          "            'stdout': _r.stdout,",
          "            'stderr': _r.stderr,",
          "            'returncode': _r.returncode,",
          "        }))",
          "      except subprocess.TimeoutExpired as _e:",
          "        print(_json.dumps({",
          "            '__shell_result__': True,",
          "            'stdout': _e.stdout or '',",
          f"            'stderr': 'Timed out after {timeout}s',",
          "            'returncode': -1,",
          "        }))",
      ])
    else:
      return None

    code_lines.extend([
        "    finally:",
        "      os.chdir(_orig_cwd)",
    ])

    code_lines.append("_materialize_and_run()")
    return "\n".join(code_lines)


@experimental(FeatureName.SKILL_TOOLSET)
class RunSkillScriptTool(BaseTool):
  """Tool to execute scripts from a skill's scripts/ directory."""

  def __init__(self, toolset: "SkillToolset"):
    super().__init__(
        name="run_skill_script",
        description="Executes a script from a skill's scripts/ directory.",
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
                "script_path": {
                    "type": "string",
                    "description": (
                        "The relative path to the script (e.g.,"
                        " 'scripts/setup.py')."
                    ),
                },
                "args": {
                    "type": "object",
                    "description": (
                        "Optional arguments to pass to the script as key-value"
                        " pairs."
                    ),
                },
            },
            "required": ["skill_name", "script_path"],
        },
    )

  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    skill_name = args.get("skill_name")
    script_path = args.get("script_path")
    script_args = args.get("args", {})
    if not isinstance(script_args, dict):
      return {
          "error": (
              "'args' must be a JSON object (key-value pairs),"
              f" got {type(script_args).__name__}."
          ),
          "error_code": "INVALID_ARGS_TYPE",
      }

    if not skill_name:
      return {
          "error": "Skill name is required.",
          "error_code": "MISSING_SKILL_NAME",
      }
    if not script_path:
      return {
          "error": "Script path is required.",
          "error_code": "MISSING_SCRIPT_PATH",
      }

    skill = self._toolset._get_skill(skill_name)
    if not skill:
      return {
          "error": f"Skill '{skill_name}' not found.",
          "error_code": "SKILL_NOT_FOUND",
      }

    script = None
    if script_path.startswith("scripts/"):
      script = skill.resources.get_script(script_path[len("scripts/") :])
    else:
      script = skill.resources.get_script(script_path)

    if script is None:
      return {
          "error": f"Script '{script_path}' not found in skill '{skill_name}'.",
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

    script_executor = _SkillScriptCodeExecutor(
        code_executor, self._toolset._script_timeout  # pylint: disable=protected-access
    )
    return await script_executor.execute_script_async(
        tool_context._invocation_context, skill, script_path, script_args  # pylint: disable=protected-access
    )


@experimental(FeatureName.SKILL_TOOLSET)
class SkillToolset(BaseToolset):
  """A toolset for managing and interacting with agent skills."""

  def __init__(
      self,
      skills: list[models.Skill],
      *,
      code_executor: Optional[BaseCodeExecutor] = None,
      script_timeout: int = _DEFAULT_SCRIPT_TIMEOUT,
  ):
    """Initializes the SkillToolset.

    Args:
      skills: List of skills to register.
      code_executor: Optional code executor for script execution.
      script_timeout: Timeout in seconds for shell script execution via
        subprocess.run. Defaults to 300 seconds. Does not apply to Python
        scripts executed via exec().
    """
    super().__init__()

    # Check for duplicate skill names
    seen: set[str] = set()
    for skill in skills:
      if skill.name in seen:
        raise ValueError(f"Duplicate skill name '{skill.name}'.")
      seen.add(skill.name)

    self._skills = {skill.name: skill for skill in skills}
    self._code_executor = code_executor
    self._script_timeout = script_timeout

    # Initialize core skill tools
    self._tools = [
        ListSkillsTool(self),
        LoadSkillTool(self),
        LoadSkillResourceTool(self),
    ]
    # Always add RunSkillScriptTool, relies on invocation_context fallback if _code_executor is None
    self._tools.append(RunSkillScriptTool(self))

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
    instructions.append(_DEFAULT_SKILL_SYSTEM_INSTRUCTION)
    instructions.append(skills_xml)
    llm_request.append_instructions(instructions)


def __getattr__(name: str) -> Any:
  if name == "DEFAULT_SKILL_SYSTEM_INSTRUCTION":
    warnings.warn(
        "DEFAULT_SKILL_SYSTEM_INSTRUCTION is experimental. Its content "
        "is internal implementation and will change in minor/patch releases "
        "to tune agent performance.",
        UserWarning,
        stacklevel=2,
    )
    return _DEFAULT_SKILL_SYSTEM_INSTRUCTION
  raise AttributeError(f"module {__name__} has no attribute {name}")
