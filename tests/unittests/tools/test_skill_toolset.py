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

from google.adk.code_executors.base_code_executor import BaseCodeExecutor
from google.adk.code_executors.code_execution_utils import CodeExecutionResult
from google.adk.models import llm_request as llm_request_model
from google.adk.skills import models
from google.adk.tools import skill_toolset
from google.adk.tools import tool_context
import pytest


@pytest.fixture
def mock_skill1_frontmatter():
  """Fixture for skill1 frontmatter."""
  frontmatter = mock.create_autospec(models.Frontmatter, instance=True)
  frontmatter.name = "skill1"
  frontmatter.description = "Skill 1 description"
  frontmatter.model_dump.return_value = {
      "name": "skill1",
      "description": "Skill 1 description",
  }
  return frontmatter


@pytest.fixture
def mock_skill1(mock_skill1_frontmatter):
  """Fixture for skill1."""
  skill = mock.create_autospec(models.Skill, instance=True)
  skill.name = "skill1"
  skill.description = "Skill 1 description"
  skill.instructions = "instructions for skill1"
  skill.frontmatter = mock_skill1_frontmatter
  skill.resources = mock.MagicMock(
      spec=["get_reference", "get_asset", "get_script"]
  )

  def get_ref(name):
    if name == "ref1.md":
      return "ref content 1"
    return None

  def get_asset(name):
    if name == "asset1.txt":
      return "asset content 1"
    return None

  def get_script(name):
    if name == "setup.sh":
      return models.Script(src="echo setup")
    if name == "run.py":
      return models.Script(src="print('hello')")
    if name == "build.rb":
      return models.Script(src="puts 'hello'")
    return None

  skill.resources.get_reference.side_effect = get_ref
  skill.resources.get_asset.side_effect = get_asset
  skill.resources.get_script.side_effect = get_script
  return skill


@pytest.fixture
def mock_skill2_frontmatter():
  """Fixture for skill2 frontmatter."""
  frontmatter = mock.create_autospec(models.Frontmatter, instance=True)
  frontmatter.name = "skill2"
  frontmatter.description = "Skill 2 description"
  frontmatter.model_dump.return_value = {
      "name": "skill2",
      "description": "Skill 2 description",
  }
  return frontmatter


@pytest.fixture
def mock_skill2(mock_skill2_frontmatter):
  """Fixture for skill2."""
  skill = mock.create_autospec(models.Skill, instance=True)
  skill.name = "skill2"
  skill.description = "Skill 2 description"
  skill.instructions = "instructions for skill2"
  skill.frontmatter = mock_skill2_frontmatter
  skill.resources = mock.MagicMock(
      spec=["get_reference", "get_asset", "get_script"]
  )

  def get_ref(name):
    if name == "ref2.md":
      return "ref content 2"
    return None

  def get_asset(name):
    if name == "asset2.txt":
      return "asset content 2"
    return None

  skill.resources.get_reference.side_effect = get_ref
  skill.resources.get_asset.side_effect = get_asset
  return skill


@pytest.fixture
def tool_context_instance():
  """Fixture for tool context."""
  return mock.create_autospec(tool_context.ToolContext, instance=True)


# SkillToolset tests
def test_get_skill(mock_skill1, mock_skill2):
  toolset = skill_toolset.SkillToolset([mock_skill1, mock_skill2])
  assert toolset._get_skill("skill1") == mock_skill1
  assert toolset._get_skill("nonexistent") is None


def test_list_skills(mock_skill1, mock_skill2):
  toolset = skill_toolset.SkillToolset([mock_skill1, mock_skill2])
  skills = toolset._list_skills()
  assert len(skills) == 2
  assert mock_skill1 in skills
  assert mock_skill2 in skills


@pytest.mark.asyncio
async def test_get_tools(mock_skill1, mock_skill2):
  toolset = skill_toolset.SkillToolset([mock_skill1, mock_skill2])
  tools = await toolset.get_tools()
  assert len(tools) == 4
  assert isinstance(tools[0], skill_toolset.ListSkillsTool)
  assert isinstance(tools[1], skill_toolset.LoadSkillTool)
  assert isinstance(tools[2], skill_toolset.LoadSkillResourceTool)
  assert isinstance(tools[3], skill_toolset.ExecuteSkillScriptTool)


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_list_skills_tool(
    mock_skill1, mock_skill2, tool_context_instance
):
  toolset = skill_toolset.SkillToolset([mock_skill1, mock_skill2])
  tool = skill_toolset.ListSkillsTool(toolset)
  result = await tool.run_async(args={}, tool_context=tool_context_instance)
  assert "<available_skills>" in result
  assert "skill1" in result
  assert "skill2" in result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args, expected_result",
    [
        (
            {"name": "skill1"},
            {
                "skill_name": "skill1",
                "instructions": "instructions for skill1",
                "frontmatter": {
                    "name": "skill1",
                    "description": "Skill 1 description",
                },
            },
        ),
        (
            {"name": "nonexistent"},
            {
                "error": "Skill 'nonexistent' not found.",
                "error_code": "SKILL_NOT_FOUND",
            },
        ),
        (
            {},
            {
                "error": "Skill name is required.",
                "error_code": "MISSING_SKILL_NAME",
            },
        ),
    ],
)
async def test_load_skill_run_async(
    mock_skill1, tool_context_instance, args, expected_result
):
  toolset = skill_toolset.SkillToolset([mock_skill1])
  tool = skill_toolset.LoadSkillTool(toolset)
  result = await tool.run_async(args=args, tool_context=tool_context_instance)
  assert result == expected_result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args, expected_result",
    [
        (
            {"skill_name": "skill1", "path": "references/ref1.md"},
            {
                "skill_name": "skill1",
                "path": "references/ref1.md",
                "content": "ref content 1",
            },
        ),
        (
            {"skill_name": "skill1", "path": "assets/asset1.txt"},
            {
                "skill_name": "skill1",
                "path": "assets/asset1.txt",
                "content": "asset content 1",
            },
        ),
        (
            {"skill_name": "skill1", "path": "scripts/setup.sh"},
            {
                "skill_name": "skill1",
                "path": "scripts/setup.sh",
                "content": "echo setup",
            },
        ),
        (
            {"skill_name": "nonexistent", "path": "references/ref1.md"},
            {
                "error": "Skill 'nonexistent' not found.",
                "error_code": "SKILL_NOT_FOUND",
            },
        ),
        (
            {"skill_name": "skill1", "path": "references/other.md"},
            {
                "error": (
                    "Resource 'references/other.md' not found in skill"
                    " 'skill1'."
                ),
                "error_code": "RESOURCE_NOT_FOUND",
            },
        ),
        (
            {"skill_name": "skill1", "path": "invalid/path.txt"},
            {
                "error": (
                    "Path must start with 'references/', 'assets/',"
                    " or 'scripts/'."
                ),
                "error_code": "INVALID_RESOURCE_PATH",
            },
        ),
        (
            {"path": "references/ref1.md"},
            {
                "error": "Skill name is required.",
                "error_code": "MISSING_SKILL_NAME",
            },
        ),
        (
            {"skill_name": "skill1"},
            {
                "error": "Resource path is required.",
                "error_code": "MISSING_RESOURCE_PATH",
            },
        ),
    ],
)
async def test_load_resource_run_async(
    mock_skill1, tool_context_instance, args, expected_result
):
  toolset = skill_toolset.SkillToolset([mock_skill1])
  tool = skill_toolset.LoadSkillResourceTool(toolset)
  result = await tool.run_async(args=args, tool_context=tool_context_instance)
  assert result == expected_result


@pytest.mark.asyncio
async def test_process_llm_request(
    mock_skill1, mock_skill2, tool_context_instance
):
  toolset = skill_toolset.SkillToolset([mock_skill1, mock_skill2])
  llm_req = mock.create_autospec(llm_request_model.LlmRequest, instance=True)

  await toolset.process_llm_request(
      tool_context=tool_context_instance, llm_request=llm_req
  )

  llm_req.append_instructions.assert_called_once()
  args, _ = llm_req.append_instructions.call_args
  instructions = args[0]
  assert len(instructions) == 2
  assert instructions[0] == skill_toolset.DEFAULT_SKILL_SYSTEM_INSTRUCTION
  assert "<available_skills>" in instructions[1]
  assert "skill1" in instructions[1]
  assert "skill2" in instructions[1]


def test_duplicate_skill_name_raises(mock_skill1):
  skill_dup = mock.create_autospec(models.Skill, instance=True)
  skill_dup.name = "skill1"
  with pytest.raises(ValueError, match="Duplicate skill name"):
    skill_toolset.SkillToolset([mock_skill1, skill_dup])


@pytest.mark.asyncio
async def test_scripts_resource_not_found(mock_skill1, tool_context_instance):
  toolset = skill_toolset.SkillToolset([mock_skill1])
  tool = skill_toolset.LoadSkillResourceTool(toolset)
  result = await tool.run_async(
      args={"skill_name": "skill1", "path": "scripts/nonexistent.sh"},
      tool_context=tool_context_instance,
  )
  assert result["error_code"] == "RESOURCE_NOT_FOUND"


# ExecuteSkillScriptTool tests


def _make_tool_context_with_agent(agent=None):
  """Creates a mock ToolContext with _invocation_context.agent."""
  ctx = mock.MagicMock(spec=tool_context.ToolContext)
  ctx._invocation_context = mock.MagicMock()
  ctx._invocation_context.agent = agent or mock.MagicMock()
  return ctx


def _make_mock_executor(stdout="", stderr=""):
  """Creates a mock code executor that returns the given output."""
  executor = mock.create_autospec(BaseCodeExecutor, instance=True)
  executor.execute_code.return_value = CodeExecutionResult(
      stdout=stdout, stderr=stderr
  )
  return executor


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args, expected_error_code",
    [
        (
            {"script_name": "setup.sh"},
            "MISSING_SKILL_NAME",
        ),
        (
            {"skill_name": "skill1"},
            "MISSING_SCRIPT_NAME",
        ),
        (
            {"skill_name": "", "script_name": "setup.sh"},
            "MISSING_SKILL_NAME",
        ),
        (
            {"skill_name": "skill1", "script_name": ""},
            "MISSING_SCRIPT_NAME",
        ),
    ],
)
async def test_execute_script_missing_params(
    mock_skill1, args, expected_error_code
):
  executor = _make_mock_executor()
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.ExecuteSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(args=args, tool_context=ctx)
  assert result["error_code"] == expected_error_code


@pytest.mark.asyncio
async def test_execute_script_skill_not_found(mock_skill1):
  executor = _make_mock_executor()
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.ExecuteSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={"skill_name": "nonexistent", "script_name": "setup.sh"},
      tool_context=ctx,
  )
  assert result["error_code"] == "SKILL_NOT_FOUND"


@pytest.mark.asyncio
async def test_execute_script_script_not_found(mock_skill1):
  executor = _make_mock_executor()
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.ExecuteSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_name": "nonexistent.py"},
      tool_context=ctx,
  )
  assert result["error_code"] == "SCRIPT_NOT_FOUND"


@pytest.mark.asyncio
async def test_execute_script_no_code_executor(mock_skill1):
  toolset = skill_toolset.SkillToolset([mock_skill1])
  tool = skill_toolset.ExecuteSkillScriptTool(toolset)
  # Agent without code_executor attribute
  agent = mock.MagicMock(spec=[])
  ctx = _make_tool_context_with_agent(agent=agent)
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_name": "setup.sh"},
      tool_context=ctx,
  )
  assert result["error_code"] == "NO_CODE_EXECUTOR"


@pytest.mark.asyncio
async def test_execute_script_agent_code_executor_none(mock_skill1):
  """Agent has code_executor attr but it's None."""
  toolset = skill_toolset.SkillToolset([mock_skill1])
  tool = skill_toolset.ExecuteSkillScriptTool(toolset)
  agent = mock.MagicMock()
  agent.code_executor = None
  ctx = _make_tool_context_with_agent(agent=agent)
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_name": "setup.sh"},
      tool_context=ctx,
  )
  assert result["error_code"] == "NO_CODE_EXECUTOR"


@pytest.mark.asyncio
async def test_execute_script_unsupported_type(mock_skill1):
  executor = _make_mock_executor()
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.ExecuteSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_name": "build.rb"},
      tool_context=ctx,
  )
  assert result["error_code"] == "UNSUPPORTED_SCRIPT_TYPE"


@pytest.mark.asyncio
async def test_execute_script_python_success(mock_skill1):
  executor = _make_mock_executor(stdout="hello\n")
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.ExecuteSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_name": "run.py"},
      tool_context=ctx,
  )
  assert result["status"] == "success"
  assert result["stdout"] == "hello\n"
  assert result["stderr"] == ""
  assert result["skill_name"] == "skill1"
  assert result["script_name"] == "run.py"

  # Verify the code passed to executor is the raw script
  call_args = executor.execute_code.call_args
  code_input = call_args[0][1]
  assert code_input.code == "print('hello')"


@pytest.mark.asyncio
async def test_execute_script_shell_success(mock_skill1):
  executor = _make_mock_executor(stdout="setup\n")
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.ExecuteSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_name": "setup.sh"},
      tool_context=ctx,
  )
  assert result["status"] == "success"
  assert result["stdout"] == "setup\n"

  # Verify the code wraps in subprocess.run with check=True
  call_args = executor.execute_code.call_args
  code_input = call_args[0][1]
  assert "subprocess.run" in code_input.code
  assert "bash" in code_input.code
  assert "check=True" in code_input.code


@pytest.mark.asyncio
async def test_execute_script_with_input_args_python(mock_skill1):
  executor = _make_mock_executor(stdout="done\n")
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.ExecuteSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={
          "skill_name": "skill1",
          "script_name": "run.py",
          "input_args": "--verbose --count 3",
      },
      tool_context=ctx,
  )
  assert result["status"] == "success"

  call_args = executor.execute_code.call_args
  code_input = call_args[0][1]
  assert "sys.argv" in code_input.code
  assert "shlex.split" in code_input.code
  assert "--verbose --count 3" in code_input.code


@pytest.mark.asyncio
async def test_execute_script_with_input_args_shell(mock_skill1):
  executor = _make_mock_executor(stdout="done\n")
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.ExecuteSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={
          "skill_name": "skill1",
          "script_name": "setup.sh",
          "input_args": "--force",
      },
      tool_context=ctx,
  )
  assert result["status"] == "success"

  call_args = executor.execute_code.call_args
  code_input = call_args[0][1]
  assert "shlex.split" in code_input.code
  assert "--force" in code_input.code


@pytest.mark.asyncio
async def test_execute_script_scripts_prefix_stripping(mock_skill1):
  executor = _make_mock_executor(stdout="setup\n")
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.ExecuteSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={
          "skill_name": "skill1",
          "script_name": "scripts/setup.sh",
      },
      tool_context=ctx,
  )
  assert result["status"] == "success"
  assert result["script_name"] == "setup.sh"


@pytest.mark.asyncio
async def test_execute_script_toolset_executor_priority(mock_skill1):
  """Toolset-level executor takes priority over agent's."""
  toolset_executor = _make_mock_executor(stdout="from toolset\n")
  agent_executor = _make_mock_executor(stdout="from agent\n")
  toolset = skill_toolset.SkillToolset(
      [mock_skill1], code_executor=toolset_executor
  )
  tool = skill_toolset.ExecuteSkillScriptTool(toolset)
  agent = mock.MagicMock()
  agent.code_executor = agent_executor
  ctx = _make_tool_context_with_agent(agent=agent)
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_name": "run.py"},
      tool_context=ctx,
  )
  assert result["stdout"] == "from toolset\n"
  toolset_executor.execute_code.assert_called_once()
  agent_executor.execute_code.assert_not_called()


@pytest.mark.asyncio
async def test_execute_script_agent_executor_fallback(mock_skill1):
  """Falls back to agent's code executor when toolset has none."""
  agent_executor = _make_mock_executor(stdout="from agent\n")
  toolset = skill_toolset.SkillToolset([mock_skill1])
  tool = skill_toolset.ExecuteSkillScriptTool(toolset)
  agent = mock.MagicMock()
  agent.code_executor = agent_executor
  ctx = _make_tool_context_with_agent(agent=agent)
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_name": "run.py"},
      tool_context=ctx,
  )
  assert result["stdout"] == "from agent\n"
  agent_executor.execute_code.assert_called_once()


@pytest.mark.asyncio
async def test_execute_script_execution_error(mock_skill1):
  executor = _make_mock_executor()
  executor.execute_code.side_effect = RuntimeError("boom")
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.ExecuteSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_name": "run.py"},
      tool_context=ctx,
  )
  assert result["error_code"] == "EXECUTION_ERROR"
  assert "boom" in result["error"]
  assert result["error"].startswith("Failed to execute script 'run.py':")


@pytest.mark.asyncio
async def test_execute_script_stderr_sets_error_status(mock_skill1):
  executor = _make_mock_executor(stdout="", stderr="warning\n")
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.ExecuteSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_name": "run.py"},
      tool_context=ctx,
  )
  assert result["status"] == "error"
  assert result["stderr"] == "warning\n"


@pytest.mark.asyncio
async def test_execute_script_execution_error_truncated(mock_skill1):
  """Long exception messages are truncated to avoid wasting LLM tokens."""
  executor = _make_mock_executor()
  executor.execute_code.side_effect = RuntimeError("x" * 300)
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.ExecuteSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_name": "run.py"},
      tool_context=ctx,
  )
  assert result["error_code"] == "EXECUTION_ERROR"
  # 200 chars of the message + "..." suffix + the prefix
  assert result["error"].endswith("...")
  assert len(result["error"]) < 300
