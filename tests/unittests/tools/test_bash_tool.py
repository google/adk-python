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

from unittest import mock

from google.adk.tools import bash_tool
from google.adk.tools import tool_context
from google.adk.tools.tool_confirmation import ToolConfirmation
import pytest


@pytest.fixture
def workspace(tmp_path):
  """Creates a workspace mirroring the anthropics/skills PDF skill layout."""
  # Skill: pdf/
  skill_dir = tmp_path / "pdf"
  skill_dir.mkdir()
  (skill_dir / "SKILL.md").write_text(
      "---\nname: pdf\n"
      "description: Use this skill whenever the user wants to do"
      " anything with PDF files.\n"
      "---\n# PDF Processing Guide\n\n## Overview\n"
      "This guide covers PDF processing operations."
  )
  scripts = skill_dir / "scripts"
  scripts.mkdir()
  (scripts / "extract_form_structure.py").write_text(
      "import sys; print(f'extracting from {sys.argv[1]}')"
  )
  (scripts / "fill_pdf_form_with_annotations.py").write_text(
      "print('filling form')"
  )
  references = skill_dir / "references"
  references.mkdir()
  (references / "REFERENCE.md").write_text("# Reference\nDetailed docs.")
  # A loose file at workspace root (not inside a skill).
  (tmp_path / "sample.pdf").write_bytes(b"%PDF-1.4 fake")
  return tmp_path


@pytest.fixture
def tool_context_no_confirmation():
  """ToolContext with no confirmation (initial call)."""
  ctx = mock.create_autospec(tool_context.ToolContext, instance=True)
  ctx.tool_confirmation = None
  ctx.actions = mock.MagicMock()
  return ctx


@pytest.fixture
def tool_context_confirmed():
  """ToolContext with confirmation approved."""
  ctx = mock.create_autospec(tool_context.ToolContext, instance=True)
  confirmation = mock.create_autospec(ToolConfirmation, instance=True)
  confirmation.confirmed = True
  ctx.tool_confirmation = confirmation
  ctx.actions = mock.MagicMock()
  return ctx


@pytest.fixture
def tool_context_rejected():
  """ToolContext with confirmation rejected."""
  ctx = mock.create_autospec(tool_context.ToolContext, instance=True)
  confirmation = mock.create_autospec(ToolConfirmation, instance=True)
  confirmation.confirmed = False
  ctx.tool_confirmation = confirmation
  ctx.actions = mock.MagicMock()
  return ctx


# --- _validate_command tests ---


class TestValidateCommand:

  def test_empty_command(self):
    policy = bash_tool.BashToolPolicy()
    assert bash_tool._validate_command("", policy) is not None
    assert bash_tool._validate_command("   ", policy) is not None

  def test_default_policy_allows_everything(self):
    policy = bash_tool.BashToolPolicy()
    assert bash_tool._validate_command("rm -rf /", policy) is None
    assert bash_tool._validate_command("cat /etc/passwd", policy) is None
    assert bash_tool._validate_command("sudo curl", policy) is None

  def test_restricted_policy_allows_prefixes(self):
    policy = bash_tool.BashToolPolicy(allowed_command_prefixes=("ls", "cat"))
    assert bash_tool._validate_command("ls -la", policy) is None
    assert bash_tool._validate_command("cat file.txt", policy) is None

  def test_restricted_policy_blocks_others(self):
    policy = bash_tool.BashToolPolicy(allowed_command_prefixes=("ls", "cat"))
    assert bash_tool._validate_command("rm -rf .", policy) is not None
    assert bash_tool._validate_command("tree", policy) is not None
    assert "Permitted prefixes are: ls, cat" in bash_tool._validate_command(
        "tree", policy
    )


class TestExecuteBashTool:

  @pytest.mark.asyncio
  async def test_requests_confirmation(
      self, workspace, tool_context_no_confirmation
  ):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(
        args={"command": "ls"},
        tool_context=tool_context_no_confirmation,
    )
    assert "error" in result
    assert "requires confirmation" in result["error"]
    tool_context_no_confirmation.request_confirmation.assert_called_once()

  @pytest.mark.asyncio
  async def test_rejected(self, workspace, tool_context_rejected):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(
        args={"command": "ls"}, tool_context=tool_context_rejected
    )
    assert result == {"error": "This tool call is rejected."}

  @pytest.mark.asyncio
  async def test_executes_when_confirmed(
      self, workspace, tool_context_confirmed
  ):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(
        args={"command": "ls"},
        tool_context=tool_context_confirmed,
    )
    assert result["returncode"] == 0
    assert "pdf" in result["stdout"]

  @pytest.mark.asyncio
  async def test_cat_skill_md(self, workspace, tool_context_confirmed):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(
        args={"command": "cat pdf/SKILL.md"},
        tool_context=tool_context_confirmed,
    )
    assert "PDF Processing Guide" in result["stdout"]

  @pytest.mark.asyncio
  async def test_python_script(self, workspace, tool_context_confirmed):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(
        args={
            "command": "python3 pdf/scripts/extract_form_structure.py test.pdf"
        },
        tool_context=tool_context_confirmed,
    )
    assert "extracting from test.pdf" in result["stdout"]
    assert result["returncode"] == 0

  @pytest.mark.asyncio
  async def test_blocks_disallowed_by_policy(
      self, workspace, tool_context_no_confirmation
  ):
    policy = bash_tool.BashToolPolicy(allowed_command_prefixes=("ls",))
    tool = bash_tool.ExecuteBashTool(workspace=workspace, policy=policy)
    result = await tool.run_async(
        args={"command": "rm -rf ."},
        tool_context=tool_context_no_confirmation,
    )
    assert "error" in result
    assert "Permitted prefixes are: ls" in result["error"]
    tool_context_no_confirmation.request_confirmation.assert_not_called()

  @pytest.mark.asyncio
  async def test_captures_stderr(self, workspace, tool_context_confirmed):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(
        args={"command": "python3 -c 'import sys; sys.stderr.write(\"err\")'"},
        tool_context=tool_context_confirmed,
    )
    assert "err" in result["stderr"]

  @pytest.mark.asyncio
  async def test_nonzero_returncode(self, workspace, tool_context_confirmed):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(
        args={"command": "python3 -c 'exit(42)'"},
        tool_context=tool_context_confirmed,
    )
    assert result["returncode"] == 42

  @pytest.mark.asyncio
  async def test_timeout(self, workspace, tool_context_confirmed):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    with mock.patch(
        "google.adk.tools.bash_tool.subprocess.run",
        side_effect=__import__("subprocess").TimeoutExpired("cmd", 30),
    ):
      result = await tool.run_async(
          args={"command": "python scripts/do_thing.py"},
          tool_context=tool_context_confirmed,
      )
    assert "error" in result
    assert "timed out" in result["error"].lower()

  @pytest.mark.asyncio
  async def test_cwd_is_workspace(self, workspace, tool_context_confirmed):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(
        args={"command": "python3 -c 'import os; print(os.getcwd())'"},
        tool_context=tool_context_confirmed,
    )
    assert result["stdout"].strip() == str(workspace)

  @pytest.mark.asyncio
  async def test_no_command(self, workspace, tool_context_confirmed):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(args={}, tool_context=tool_context_confirmed)
    assert "error" in result
    assert "required" in result["error"].lower()
