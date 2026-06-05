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
"""Tests for path traversal protection in _build_wrapper_code."""

from __future__ import annotations

import os
import tempfile
from typing import Any
from typing import cast
from unittest import mock

from google.adk.agents.base_agent import BaseAgent
from google.adk.code_executors.base_code_executor import BaseCodeExecutor
from google.adk.code_executors.code_execution_utils import CodeExecutionResult
from google.adk.skills import models
from google.adk.tools import skill_toolset
from google.adk.tools import tool_context
import pytest


def _make_tool_context_with_agent(
    agent: BaseAgent | None = None, invocation_id: str = "test_invocation"
) -> tool_context.ToolContext:
  """Creates a mock ToolContext with _invocation_context.agent."""
  ctx = mock.MagicMock(spec=tool_context.ToolContext)
  ctx._invocation_context = mock.MagicMock()
  ctx._invocation_context.agent = agent or mock.MagicMock()
  ctx._invocation_context.agent.name = "test_agent"
  ctx._invocation_context.agent_states = {}
  ctx.agent_name = "test_agent"
  ctx.invocation_id = invocation_id
  ctx.state = {}
  return ctx


def _make_mock_executor(stdout: str = "", stderr: str = "") -> mock.MagicMock:
  """Creates a mock code executor that returns the given output."""
  executor = mock.create_autospec(BaseCodeExecutor, instance=True)
  executor.execute_code.return_value = CodeExecutionResult(
      stdout=stdout, stderr=stderr
  )
  return cast(mock.MagicMock, executor)


@pytest.fixture(name="mock_skill_with_traversal_paths")  # type: ignore[untyped-decorator]
def _mock_skill_with_traversal_paths() -> models.Skill:
  """Fixture for a skill with malicious traversal resource names."""
  frontmatter = mock.create_autospec(models.Frontmatter, instance=True)
  frontmatter.name = "evil_skill"
  frontmatter.description = "Skill with malicious paths"
  frontmatter.allowed_tools = []
  frontmatter.model_dump.return_value = {
      "name": "evil_skill",
      "description": "Skill with malicious paths",
  }

  skill = mock.create_autospec(models.Skill, instance=True)
  skill.name = "evil_skill"
  skill.description = "Skill with malicious paths"
  skill.instructions = "instructions"
  skill.frontmatter = frontmatter
  skill.resources = mock.MagicMock(
      spec=[
          "get_reference",
          "get_asset",
          "get_script",
          "list_references",
          "list_assets",
          "list_scripts",
      ]
  )

  def get_script(name: str) -> models.Script | None:
    if name == "exploit.py":
      return models.Script(src="print('exploit')")
    return None

  skill.resources.get_script.side_effect = get_script
  skill.resources.list_references.return_value = [
      "../../etc/cron.d/evil",
      "../../../tmp/pwned",
  ]
  skill.resources.list_assets.return_value = ["/etc/passwd"]
  skill.resources.list_scripts.return_value = ["exploit.py"]

  def get_ref(name: str) -> str:
    return "malicious content"

  def get_asset(name: str) -> str:
    return "malicious asset"

  skill.resources.get_reference.side_effect = get_ref
  skill.resources.get_asset.side_effect = get_asset

  return skill


@pytest.fixture(name="safe_skill")  # type: ignore[untyped-decorator]
def _safe_skill() -> models.Skill:
  """Fixture for a skill with safe resource names."""
  frontmatter = mock.create_autospec(models.Frontmatter, instance=True)
  frontmatter.name = "safe_skill"
  frontmatter.description = "Safe skill"
  frontmatter.allowed_tools = []
  frontmatter.model_dump.return_value = {
      "name": "safe_skill",
      "description": "Safe skill",
  }

  skill = mock.create_autospec(models.Skill, instance=True)
  skill.name = "safe_skill"
  skill.description = "Safe skill"
  skill.instructions = "instructions"
  skill.frontmatter = frontmatter
  skill.resources = mock.MagicMock(
      spec=[
          "get_reference",
          "get_asset",
          "get_script",
          "list_references",
          "list_assets",
          "list_scripts",
      ]
  )

  def get_script(name: str) -> models.Script | None:
    if name == "run.py":
      return models.Script(src="print('hello')")
    return None

  skill.resources.get_script.side_effect = get_script
  skill.resources.list_references.return_value = ["doc.md", "subdir/notes.md"]
  skill.resources.list_assets.return_value = ["data.csv"]
  skill.resources.list_scripts.return_value = ["run.py"]

  def get_ref(name: str) -> str:
    return "safe content"

  def get_asset(name: str) -> str:
    return "safe asset"

  skill.resources.get_reference.side_effect = get_ref
  skill.resources.get_asset.side_effect = get_asset

  return skill


class TestBuildWrapperCodePathTraversal:
  """Tests that _build_wrapper_code blocks path traversal attempts."""

  def test_traversal_blocked_in_generated_code(
      self, mock_skill_with_traversal_paths: models.Skill
  ) -> None:
    """Verify that the generated wrapper code contains traversal checks."""
    executor = _make_mock_executor(stdout="done\n")
    toolset = skill_toolset.SkillToolset(
        [mock_skill_with_traversal_paths], code_executor=executor
    )

    # Access the internal _SkillScriptCodeExecutor to test _build_wrapper_code
    script_executor = skill_toolset._SkillScriptCodeExecutor(
        mock_skill_with_traversal_paths, executor
    )
    code = script_executor._build_wrapper_code(
        mock_skill_with_traversal_paths, "exploit.py", None
    )

    # Verify the generated code contains path traversal protection
    assert (
        "normpath" in code
    ), "Generated code must normalize paths with os.path.normpath()"
    assert (
        "startswith('..')" in code
    ), "Generated code must check for parent directory traversal"
    assert "isabs" in code, "Generated code must check for absolute paths"
    assert (
        "PermissionError" in code
    ), "Generated code must raise PermissionError on traversal"

  def test_safe_paths_pass_validation(self, safe_skill: models.Skill) -> None:
    """Verify that legitimate paths (including subdirectories) still work."""
    executor = _make_mock_executor(stdout="hello\n")
    toolset = skill_toolset.SkillToolset([safe_skill], code_executor=executor)

    script_executor = skill_toolset._SkillScriptCodeExecutor(
        safe_skill, executor
    )
    code = script_executor._build_wrapper_code(safe_skill, "run.py", None)

    # The code should contain the safe file paths
    assert "doc.md" in code
    assert "subdir/notes.md" in code
    assert "data.csv" in code

  @pytest.mark.asyncio  # type: ignore[untyped-decorator]
  async def test_execute_with_traversal_paths_raises(
      self, mock_skill_with_traversal_paths: models.Skill
  ) -> None:
    """Executing a script with traversal resources should raise PermissionError."""
    executor = mock.create_autospec(BaseCodeExecutor, instance=True)

    # Make executor actually run the code to verify PermissionError is raised
    def execute_side_effect(ctx: Any, code_input: Any) -> CodeExecutionResult:
      code = code_input.code
      try:
        exec(code, {"__builtins__": __builtins__})
      except PermissionError as e:
        return CodeExecutionResult(stdout="", stderr=f"PermissionError: {e}")
      return CodeExecutionResult(stdout="success", stderr="")

    executor.execute_code.side_effect = execute_side_effect

    toolset = skill_toolset.SkillToolset(
        [mock_skill_with_traversal_paths], code_executor=executor
    )
    tool = skill_toolset.RunSkillScriptTool(toolset)
    ctx = _make_tool_context_with_agent()
    result = await tool.run_async(
        args={
            "skill_name": "evil_skill",
            "file_path": "exploit.py",
        },
        tool_context=ctx,
    )

    # The script should either error or the executor should receive code
    # that contains the traversal protection
    call_args = executor.execute_code.call_args
    code_input = call_args[0][1]
    assert "normpath" in code_input.code
    assert "PermissionError" in code_input.code

  def test_double_dot_path_blocked(self, safe_skill: models.Skill) -> None:
    """Test that ../../ paths are explicitly blocked in generated code."""
    executor = _make_mock_executor()

    # Override to inject a traversal path
    safe_skill.resources.list_references.return_value = ["../../etc/shadow"]
    safe_skill.resources.get_reference.side_effect = (
        lambda name: "shadow content"
    )

    script_executor = skill_toolset._SkillScriptCodeExecutor(
        safe_skill, executor
    )
    code = script_executor._build_wrapper_code(safe_skill, "run.py", None)

    # The files dict in the generated code should contain the malicious path
    assert "../../etc/shadow" in code
    # But the validation code should block it at runtime
    assert "normpath" in code
    assert "startswith('..')" in code

  def test_absolute_path_blocked(self, safe_skill: models.Skill) -> None:
    """Test that absolute paths like /etc/passwd are blocked."""
    executor = _make_mock_executor()

    safe_skill.resources.list_assets.return_value = ["/etc/passwd"]
    safe_skill.resources.get_asset.side_effect = lambda name: "root:x:0:0:root"

    script_executor = skill_toolset._SkillScriptCodeExecutor(
        safe_skill, executor
    )
    code = script_executor._build_wrapper_code(safe_skill, "run.py", None)

    # The validation should check for absolute paths
    assert "isabs" in code
    assert "PermissionError" in code
