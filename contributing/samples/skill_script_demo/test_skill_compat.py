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

"""Compatibility test: Anthropic mcp-builder skill vs ADK SkillToolset.

Tests every tool in SkillToolset against the real Anthropic mcp-builder
skill to verify end-to-end compatibility with the public Agent Skills
spec (agentskills.io/specification).

Run:
    pytest contributing/samples/skill_script_demo/test_skill_compat.py -v
"""

import pathlib
from unittest import mock

from google.adk.code_executors.base_code_executor import BaseCodeExecutor
from google.adk.code_executors.code_execution_utils import CodeExecutionResult
from google.adk.skills import load_skill_from_dir
from google.adk.tools import skill_toolset
from google.adk.tools import tool_context
import pytest

_SKILLS_DIR = pathlib.Path(__file__).parent / "skills"


@pytest.fixture
def mcp_builder_skill():
  return load_skill_from_dir(_SKILLS_DIR / "mcp-builder")


@pytest.fixture
def toolset_with_executor(mcp_builder_skill):
  executor = mock.create_autospec(BaseCodeExecutor, instance=True)
  executor.execute_code.return_value = CodeExecutionResult(
      stdout="ok\n", stderr=""
  )
  return skill_toolset.SkillToolset(
      skills=[mcp_builder_skill], code_executor=executor
  )


@pytest.fixture
def mock_ctx():
  ctx = mock.MagicMock(spec=tool_context.ToolContext)
  ctx._invocation_context = mock.MagicMock()
  ctx._invocation_context.agent = mock.MagicMock()
  return ctx


# ── 1. Skill loading ──────────────────────────────────────────────


def test_skill_loads_from_disk(mcp_builder_skill):
  """Verify the skill loads successfully from disk."""
  assert mcp_builder_skill.name == "mcp-builder"
  assert "MCP" in mcp_builder_skill.description
  assert len(mcp_builder_skill.instructions) > 0


def test_skill_scripts_loaded(mcp_builder_skill):
  """All scripts/ files should be loaded."""
  scripts = mcp_builder_skill.resources.list_scripts()
  assert "connections.py" in scripts
  assert "evaluation.py" in scripts
  assert "requirements.txt" in scripts
  assert "example_evaluation.xml" in scripts


def test_skill_references_loaded(mcp_builder_skill):
  """All references/ files should be loaded per spec."""
  refs = mcp_builder_skill.resources.list_references()
  assert "evaluation.md" in refs
  assert "mcp_best_practices.md" in refs
  assert "node_mcp_server.md" in refs
  assert "python_mcp_server.md" in refs


# ── 2. list_skills tool ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_skills_shows_mcp_builder(toolset_with_executor, mock_ctx):
  tool = skill_toolset.ListSkillsTool(toolset_with_executor)
  result = await tool.run_async(args={}, tool_context=mock_ctx)
  assert "mcp-builder" in result


# ── 3. load_skill tool ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_load_skill_returns_instructions(toolset_with_executor, mock_ctx):
  tool = skill_toolset.LoadSkillTool(toolset_with_executor)
  result = await tool.run_async(
      args={"name": "mcp-builder"}, tool_context=mock_ctx
  )
  assert result["skill_name"] == "mcp-builder"
  assert "MCP Server Development Guide" in result["instructions"]
  assert result["frontmatter"]["name"] == "mcp-builder"


# ── 4. load_skill_resource tool ───────────────────────────────────


@pytest.mark.asyncio
async def test_load_script_content(toolset_with_executor, mock_ctx):
  """Can read a Python script via load_skill_resource."""
  tool = skill_toolset.LoadSkillResourceTool(toolset_with_executor)
  result = await tool.run_async(
      args={
          "skill_name": "mcp-builder",
          "path": "scripts/connections.py",
      },
      tool_context=mock_ctx,
  )
  assert result["content"] is not None
  assert "MCPConnection" in result["content"]


@pytest.mark.asyncio
async def test_load_requirements_txt(toolset_with_executor, mock_ctx):
  """Can read non-executable script files like requirements.txt."""
  tool = skill_toolset.LoadSkillResourceTool(toolset_with_executor)
  result = await tool.run_async(
      args={
          "skill_name": "mcp-builder",
          "path": "scripts/requirements.txt",
      },
      tool_context=mock_ctx,
  )
  assert "anthropic" in result["content"]


@pytest.mark.asyncio
async def test_load_reference_content(toolset_with_executor, mock_ctx):
  """Can read reference files via load_skill_resource."""
  tool = skill_toolset.LoadSkillResourceTool(toolset_with_executor)
  result = await tool.run_async(
      args={
          "skill_name": "mcp-builder",
          "path": "references/evaluation.md",
      },
      tool_context=mock_ctx,
  )
  assert result["content"] is not None
  assert "evaluation" in result["content"].lower()


# ── 5. execute_skill_script tool ──────────────────────────────────


@pytest.mark.asyncio
async def test_execute_python_script(toolset_with_executor, mock_ctx):
  """Can execute a .py script from the skill."""
  tool = skill_toolset.ExecuteSkillScriptTool(toolset_with_executor)
  result = await tool.run_async(
      args={
          "skill_name": "mcp-builder",
          "script_name": "connections.py",
      },
      tool_context=mock_ctx,
  )
  assert result["status"] == "success"
  assert result["script_name"] == "connections.py"

  # Verify the executor received the actual script source
  call_args = toolset_with_executor._code_executor.execute_code.call_args
  code_input = call_args[0][1]
  assert "MCPConnection" in code_input.code


@pytest.mark.asyncio
async def test_execute_with_scripts_prefix(toolset_with_executor, mock_ctx):
  """scripts/ prefix is stripped automatically."""
  tool = skill_toolset.ExecuteSkillScriptTool(toolset_with_executor)
  result = await tool.run_async(
      args={
          "skill_name": "mcp-builder",
          "script_name": "scripts/evaluation.py",
      },
      tool_context=mock_ctx,
  )
  assert result["status"] == "success"
  assert result["script_name"] == "evaluation.py"


@pytest.mark.asyncio
async def test_execute_unsupported_txt(toolset_with_executor, mock_ctx):
  """requirements.txt is not executable — returns UNSUPPORTED_SCRIPT_TYPE."""
  tool = skill_toolset.ExecuteSkillScriptTool(toolset_with_executor)
  result = await tool.run_async(
      args={
          "skill_name": "mcp-builder",
          "script_name": "requirements.txt",
      },
      tool_context=mock_ctx,
  )
  assert result["error_code"] == "UNSUPPORTED_SCRIPT_TYPE"


@pytest.mark.asyncio
async def test_execute_unsupported_xml(toolset_with_executor, mock_ctx):
  """example_evaluation.xml is not executable — returns UNSUPPORTED_SCRIPT_TYPE."""
  tool = skill_toolset.ExecuteSkillScriptTool(toolset_with_executor)
  result = await tool.run_async(
      args={
          "skill_name": "mcp-builder",
          "script_name": "example_evaluation.xml",
      },
      tool_context=mock_ctx,
  )
  assert result["error_code"] == "UNSUPPORTED_SCRIPT_TYPE"


@pytest.mark.asyncio
async def test_execute_with_input_args(toolset_with_executor, mock_ctx):
  """input_args are injected into sys.argv for Python scripts."""
  tool = skill_toolset.ExecuteSkillScriptTool(toolset_with_executor)
  result = await tool.run_async(
      args={
          "skill_name": "mcp-builder",
          "script_name": "evaluation.py",
          "input_args": "-t stdio -c python eval.xml",
      },
      tool_context=mock_ctx,
  )
  assert result["status"] == "success"

  call_args = toolset_with_executor._code_executor.execute_code.call_args
  code_input = call_args[0][1]
  assert "sys.argv" in code_input.code
  assert "shlex.split" in code_input.code
  assert "-t stdio -c python eval.xml" in code_input.code


# ── 6. Tool count ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_toolset_provides_4_tools(toolset_with_executor):
  tools = await toolset_with_executor.get_tools()
  assert len(tools) == 4
  names = [t.name for t in tools]
  assert "list_skills" in names
  assert "load_skill" in names
  assert "load_skill_resource" in names
  assert "execute_skill_script" in names
