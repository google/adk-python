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

# pylint: disable=redefined-outer-name,g-import-not-at-top,protected-access


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
  frontmatter.allowed_tools = ["test_tool"]
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
      spec=[
          "get_reference",
          "get_asset",
          "get_script",
          "list_references",
          "list_assets",
          "list_scripts",
      ]
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
  skill.resources.list_references.return_value = ["ref1.md"]
  skill.resources.list_assets.return_value = ["asset1.txt"]
  skill.resources.list_scripts.return_value = [
      "setup.sh",
      "run.py",
      "build.rb",
  ]
  return skill


@pytest.fixture
def mock_skill2_frontmatter():
  """Fixture for skill2 frontmatter."""
  frontmatter = mock.create_autospec(models.Frontmatter, instance=True)
  frontmatter.name = "skill2"
  frontmatter.description = "Skill 2 description"
  frontmatter.allowed_tools = []
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
      spec=[
          "get_reference",
          "get_asset",
          "get_script",
          "list_references",
          "list_assets",
          "list_scripts",
      ]
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
  skill.resources.list_references.return_value = ["ref2.md"]
  skill.resources.list_assets.return_value = ["asset2.txt"]
  skill.resources.list_scripts.return_value = []
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
  assert isinstance(tools[3], skill_toolset.RunSkillScriptTool)


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


# RunSkillScriptTool tests


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
            {"script_path": "setup.sh"},
            "MISSING_SKILL_NAME",
        ),
        (
            {"skill_name": "skill1"},
            "MISSING_SCRIPT_PATH",
        ),
        (
            {"skill_name": "", "script_path": "setup.sh"},
            "MISSING_SKILL_NAME",
        ),
        (
            {"skill_name": "skill1", "script_path": ""},
            "MISSING_SCRIPT_PATH",
        ),
    ],
)
async def test_execute_script_missing_params(
    mock_skill1, args, expected_error_code
):
  executor = _make_mock_executor()
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(args=args, tool_context=ctx)
  assert result["error_code"] == expected_error_code


@pytest.mark.asyncio
async def test_execute_script_skill_not_found(mock_skill1):
  executor = _make_mock_executor()
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={"skill_name": "nonexistent", "script_path": "setup.sh"},
      tool_context=ctx,
  )
  assert result["error_code"] == "SKILL_NOT_FOUND"


@pytest.mark.asyncio
async def test_execute_script_script_not_found(mock_skill1):
  executor = _make_mock_executor()
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_path": "nonexistent.py"},
      tool_context=ctx,
  )
  assert result["error_code"] == "SCRIPT_NOT_FOUND"


@pytest.mark.asyncio
async def test_execute_script_no_code_executor(mock_skill1):
  toolset = skill_toolset.SkillToolset([mock_skill1])
  tool = skill_toolset.RunSkillScriptTool(toolset)
  # Agent without code_executor attribute
  agent = mock.MagicMock(spec=[])
  ctx = _make_tool_context_with_agent(agent=agent)
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_path": "setup.sh"},
      tool_context=ctx,
  )
  assert result["error_code"] == "NO_CODE_EXECUTOR"


@pytest.mark.asyncio
async def test_execute_script_agent_code_executor_none(mock_skill1):
  """Agent has code_executor attr but it's None."""
  toolset = skill_toolset.SkillToolset([mock_skill1])
  tool = skill_toolset.RunSkillScriptTool(toolset)
  agent = mock.MagicMock()
  agent.code_executor = None
  ctx = _make_tool_context_with_agent(agent=agent)
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_path": "setup.sh"},
      tool_context=ctx,
  )
  assert result["error_code"] == "NO_CODE_EXECUTOR"


@pytest.mark.asyncio
async def test_execute_script_unsupported_type(mock_skill1):
  executor = _make_mock_executor()
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_path": "build.rb"},
      tool_context=ctx,
  )
  assert result["error_code"] == "UNSUPPORTED_SCRIPT_TYPE"


@pytest.mark.asyncio
async def test_execute_script_python_success(mock_skill1):
  executor = _make_mock_executor(stdout="hello\n")
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_path": "run.py"},
      tool_context=ctx,
  )
  assert result["status"] == "success"
  assert result["stdout"] == "hello\n"
  assert result["stderr"] == ""
  assert result["skill_name"] == "skill1"
  assert result["script_path"] == "run.py"

  # Verify the code passed to executor runs the python scripts
  call_args = executor.execute_code.call_args
  code_input = call_args[0][1]
  assert "_materialize_and_run()" in code_input.code
  assert "import runpy" in code_input.code
  assert "sys.argv = ['scripts/run.py']" in code_input.code
  assert (
      "runpy.run_path('scripts/run.py', run_name='__main__')" in code_input.code
  )


@pytest.mark.asyncio
async def test_execute_script_shell_success(mock_skill1):
  executor = _make_mock_executor(stdout="setup\n")
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_path": "setup.sh"},
      tool_context=ctx,
  )
  assert result["status"] == "success"
  assert result["stdout"] == "setup\n"

  # Verify the code wraps in subprocess.run with JSON envelope
  call_args = executor.execute_code.call_args
  code_input = call_args[0][1]
  assert "subprocess.run" in code_input.code
  assert "bash" in code_input.code
  assert "__shell_result__" in code_input.code


@pytest.mark.asyncio
async def test_execute_script_with_input_args_python(mock_skill1):
  executor = _make_mock_executor(stdout="done\n")
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={
          "skill_name": "skill1",
          "script_path": "run.py",
          "args": {"verbose": True, "count": "3"},
      },
      tool_context=ctx,
  )
  assert result["status"] == "success"

  call_args = executor.execute_code.call_args
  code_input = call_args[0][1]
  assert (
      "['scripts/run.py', '--verbose', 'True', '--count', '3']"
      in code_input.code
  )


@pytest.mark.asyncio
async def test_execute_script_with_input_args_shell(mock_skill1):
  executor = _make_mock_executor(stdout="done\n")
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={
          "skill_name": "skill1",
          "script_path": "setup.sh",
          "args": {"force": True},
      },
      tool_context=ctx,
  )
  assert result["status"] == "success"

  call_args = executor.execute_code.call_args
  code_input = call_args[0][1]
  assert "['bash', 'scripts/setup.sh', '--force', 'True']" in code_input.code


@pytest.mark.asyncio
async def test_execute_script_scripts_prefix_stripping(mock_skill1):
  executor = _make_mock_executor(stdout="setup\n")
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={
          "skill_name": "skill1",
          "script_path": "scripts/setup.sh",
      },
      tool_context=ctx,
  )
  assert result["status"] == "success"
  assert result["script_path"] == "scripts/setup.sh"


@pytest.mark.asyncio
async def test_execute_script_toolset_executor_priority(mock_skill1):
  """Toolset-level executor takes priority over agent's."""
  toolset_executor = _make_mock_executor(stdout="from toolset\n")
  agent_executor = _make_mock_executor(stdout="from agent\n")
  toolset = skill_toolset.SkillToolset(
      [mock_skill1], code_executor=toolset_executor
  )
  tool = skill_toolset.RunSkillScriptTool(toolset)
  agent = mock.MagicMock()
  agent.code_executor = agent_executor
  ctx = _make_tool_context_with_agent(agent=agent)
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_path": "run.py"},
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
  tool = skill_toolset.RunSkillScriptTool(toolset)
  agent = mock.MagicMock()
  agent.code_executor = agent_executor
  ctx = _make_tool_context_with_agent(agent=agent)
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_path": "run.py"},
      tool_context=ctx,
  )
  assert result["stdout"] == "from agent\n"
  agent_executor.execute_code.assert_called_once()


@pytest.mark.asyncio
async def test_execute_script_execution_error(mock_skill1):
  executor = _make_mock_executor()
  executor.execute_code.side_effect = RuntimeError("boom")
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_path": "run.py"},
      tool_context=ctx,
  )
  assert result["error_code"] == "EXECUTION_ERROR"
  assert "boom" in result["error"]
  assert result["error"].startswith("Failed to execute script 'run.py':")


@pytest.mark.asyncio
async def test_execute_script_stderr_only_sets_error_status(mock_skill1):
  """stderr with no stdout should report error status."""
  executor = _make_mock_executor(stdout="", stderr="fatal error\n")
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_path": "run.py"},
      tool_context=ctx,
  )
  assert result["status"] == "error"
  assert result["stderr"] == "fatal error\n"


@pytest.mark.asyncio
async def test_execute_script_stderr_with_stdout_sets_warning(mock_skill1):
  """stderr alongside stdout should report warning status."""
  executor = _make_mock_executor(stdout="output\n", stderr="deprecation\n")
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_path": "run.py"},
      tool_context=ctx,
  )
  assert result["status"] == "warning"
  assert result["stdout"] == "output\n"
  assert result["stderr"] == "deprecation\n"


@pytest.mark.asyncio
async def test_execute_script_execution_error_truncated(mock_skill1):
  """Long exception messages are truncated to avoid wasting LLM tokens."""
  executor = _make_mock_executor()
  executor.execute_code.side_effect = RuntimeError("x" * 300)
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_path": "run.py"},
      tool_context=ctx,
  )
  assert result["error_code"] == "EXECUTION_ERROR"
  # 200 chars of the message + "..." suffix + the prefix
  assert result["error"].endswith("...")
  assert len(result["error"]) < 300


@pytest.mark.asyncio
async def test_execute_script_system_exit_caught(mock_skill1):
  """sys.exit() in a script should not terminate the process."""
  executor = _make_mock_executor()
  executor.execute_code.side_effect = SystemExit(1)
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_path": "run.py"},
      tool_context=ctx,
  )
  assert result["error_code"] == "EXECUTION_ERROR"
  assert "exited with code 1" in result["error"]


@pytest.mark.asyncio
async def test_execute_script_system_exit_zero_is_success(mock_skill1):
  """sys.exit(0) is a normal termination and should report success."""
  executor = _make_mock_executor()
  executor.execute_code.side_effect = SystemExit(0)
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()

  result = await tool.run_async(
      args={"skill_name": "skill1", "script_path": "run.py"},
      tool_context=ctx,
  )
  assert result["status"] == "success"


@pytest.mark.asyncio
async def test_execute_script_system_exit_none_is_success(mock_skill1):
  """sys.exit() with no arg (None) should report success."""
  executor = _make_mock_executor()
  executor.execute_code.side_effect = SystemExit(None)
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_path": "run.py"},
      tool_context=ctx,
  )
  assert result["status"] == "success"


@pytest.mark.asyncio
async def test_execute_script_shell_includes_timeout(mock_skill1):
  """Shell wrapper includes timeout in subprocess.run."""
  executor = _make_mock_executor(stdout="ok\n")
  toolset = skill_toolset.SkillToolset(
      [mock_skill1], code_executor=executor, script_timeout=60
  )
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_path": "setup.sh"},
      tool_context=ctx,
  )
  assert result["status"] == "success"
  call_args = executor.execute_code.call_args
  code_input = call_args[0][1]
  assert "timeout=60" in code_input.code


@pytest.mark.asyncio
async def test_execute_script_extensionless_unsupported(mock_skill1):
  """Files without extensions should return UNSUPPORTED_SCRIPT_TYPE."""
  # Add a script with no extension to the mock
  original_side_effect = mock_skill1.resources.get_script.side_effect

  def get_script_extended(name):
    if name == "noext":
      return models.Script(src="print('hi')")
    return original_side_effect(name)

  mock_skill1.resources.get_script.side_effect = get_script_extended

  executor = _make_mock_executor()
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_path": "noext"},
      tool_context=ctx,
  )
  assert result["error_code"] == "UNSUPPORTED_SCRIPT_TYPE"


# ── Integration tests using real UnsafeLocalCodeExecutor ──


def _make_skill_with_script(skill_name, script_name, script):
  """Creates a minimal mock Skill with a single script."""
  skill = mock.create_autospec(models.Skill, instance=True)
  skill.name = skill_name
  skill.description = f"Test skill {skill_name}"
  skill.instructions = "test instructions"
  fm = mock.create_autospec(models.Frontmatter, instance=True)
  fm.name = skill_name
  fm.description = f"Test skill {skill_name}"
  skill.frontmatter = fm
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

  def get_script(name):
    if name == script_name:
      return script
    return None

  skill.resources.get_script.side_effect = get_script
  skill.resources.get_reference.return_value = None
  skill.resources.get_asset.return_value = None
  skill.resources.list_references.return_value = []
  skill.resources.list_assets.return_value = []
  skill.resources.list_scripts.return_value = [script_name]
  return skill


def _make_real_executor_toolset(skills, **kwargs):
  """Creates a SkillToolset with a real UnsafeLocalCodeExecutor."""
  from google.adk.code_executors.unsafe_local_code_executor import UnsafeLocalCodeExecutor

  executor = UnsafeLocalCodeExecutor()
  return skill_toolset.SkillToolset(skills, code_executor=executor, **kwargs)


@pytest.mark.asyncio
async def test_integration_python_stdout():
  """Real executor: Python script stdout is captured."""
  script = models.Script(src="print('hello world')")
  skill = _make_skill_with_script("test_skill", "hello.py", script)
  toolset = _make_real_executor_toolset([skill])
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={
          "skill_name": "test_skill",
          "script_path": "hello.py",
      },
      tool_context=ctx,
  )
  assert result["status"] == "success"
  assert result["stdout"] == "hello world\n"
  assert result["stderr"] == ""


@pytest.mark.asyncio
async def test_integration_python_sys_exit_zero():
  """Real executor: sys.exit(0) is treated as success."""
  script = models.Script(src="import sys; sys.exit(0)")
  skill = _make_skill_with_script("test_skill", "exit_zero.py", script)
  toolset = _make_real_executor_toolset([skill])
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={
          "skill_name": "test_skill",
          "script_path": "exit_zero.py",
      },
      tool_context=ctx,
  )
  assert result["status"] == "success"


@pytest.mark.asyncio
async def test_integration_shell_stdout_and_stderr():
  """Real executor: shell script preserves both stdout and stderr."""
  script = models.Script(src="echo output; echo warning >&2")
  skill = _make_skill_with_script("test_skill", "both.sh", script)
  toolset = _make_real_executor_toolset([skill])
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={
          "skill_name": "test_skill",
          "script_path": "both.sh",
      },
      tool_context=ctx,
  )
  assert result["status"] == "warning"
  assert "output" in result["stdout"]
  assert "warning" in result["stderr"]


@pytest.mark.asyncio
async def test_integration_shell_stderr_only():
  """Real executor: shell script with only stderr reports error."""
  script = models.Script(src="echo failure >&2")
  skill = _make_skill_with_script("test_skill", "err.sh", script)
  toolset = _make_real_executor_toolset([skill])
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={
          "skill_name": "test_skill",
          "script_path": "err.sh",
      },
      tool_context=ctx,
  )
  assert result["status"] == "error"
  assert "failure" in result["stderr"]


# ── Shell JSON envelope parsing (unit tests with mock executor) ──


@pytest.mark.asyncio
async def test_shell_json_envelope_parsed(mock_skill1):
  """Shell JSON envelope is correctly unpacked by run_async."""
  import json

  envelope = json.dumps({
      "__shell_result__": True,
      "stdout": "hello from shell\n",
      "stderr": "",
      "returncode": 0,
  })
  executor = _make_mock_executor(stdout=envelope)
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_path": "setup.sh"},
      tool_context=ctx,
  )
  assert result["status"] == "success"
  assert result["stdout"] == "hello from shell\n"
  assert result["stderr"] == ""


@pytest.mark.asyncio
async def test_shell_json_envelope_nonzero_returncode(mock_skill1):
  """Non-zero returncode in shell envelope sets stderr."""
  import json

  envelope = json.dumps({
      "__shell_result__": True,
      "stdout": "",
      "stderr": "",
      "returncode": 2,
  })
  executor = _make_mock_executor(stdout=envelope)
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_path": "setup.sh"},
      tool_context=ctx,
  )
  assert result["status"] == "error"
  assert "Exit code 2" in result["stderr"]


@pytest.mark.asyncio
async def test_shell_json_envelope_with_stderr(mock_skill1):
  """Shell envelope with both stdout and stderr reports warning."""
  import json

  envelope = json.dumps({
      "__shell_result__": True,
      "stdout": "data\n",
      "stderr": "deprecation warning\n",
      "returncode": 0,
  })
  executor = _make_mock_executor(stdout=envelope)
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_path": "setup.sh"},
      tool_context=ctx,
  )
  assert result["status"] == "warning"
  assert result["stdout"] == "data\n"
  assert result["stderr"] == "deprecation warning\n"


@pytest.mark.asyncio
async def test_shell_json_envelope_timeout(mock_skill1):
  """Shell envelope from TimeoutExpired reports error status."""
  import json

  envelope = json.dumps({
      "__shell_result__": True,
      "stdout": "partial output\n",
      "stderr": "Timed out after 300s",
      "returncode": -1,
  })
  executor = _make_mock_executor(stdout=envelope)
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_path": "setup.sh"},
      tool_context=ctx,
  )
  assert result["status"] == "error"
  assert result["stdout"] == "partial output\n"
  assert "Timed out" in result["stderr"]


@pytest.mark.asyncio
async def test_shell_non_json_stdout_passthrough(mock_skill1):
  """Non-JSON shell stdout is passed through without parsing."""
  executor = _make_mock_executor(stdout="plain text output\n")
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={"skill_name": "skill1", "script_path": "setup.sh"},
      tool_context=ctx,
  )
  assert result["status"] == "success"
  assert result["stdout"] == "plain text output\n"


# ── input_files packaging ──


@pytest.mark.asyncio
async def test_execute_script_input_files_packaged(mock_skill1):
  """Verify references, assets, and scripts are packaged inside the wrapper code."""
  executor = _make_mock_executor(stdout="ok\n")
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  await tool.run_async(
      args={"skill_name": "skill1", "script_path": "run.py"},
      tool_context=ctx,
  )

  call_args = executor.execute_code.call_args
  code_input = call_args[0][1]

  # input_files is no longer populated; it's serialized inside the script
  assert code_input.input_files is None or len(code_input.input_files) == 0

  # Ensure the extracted literal contains our fake files
  assert "references/ref1.md" in code_input.code
  assert "assets/asset1.txt" in code_input.code
  assert "scripts/setup.sh" in code_input.code
  assert "scripts/run.py" in code_input.code
  assert "scripts/build.rb" in code_input.code

  # Verify content mappings exist in the string
  assert "'references/ref1.md': 'ref content 1'" in code_input.code
  assert "'assets/asset1.txt': 'asset content 1'" in code_input.code


# ── Integration: shell non-zero exit ──


@pytest.mark.asyncio
async def test_integration_shell_nonzero_exit():
  """Real executor: shell script with non-zero exit via JSON envelope."""
  script = models.Script(src="exit 42")
  skill = _make_skill_with_script("test_skill", "fail.sh", script)
  toolset = _make_real_executor_toolset([skill])
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={
          "skill_name": "test_skill",
          "script_path": "fail.sh",
      },
      tool_context=ctx,
  )
  assert result["status"] == "error"
  assert "42" in result["stderr"]


# ── Finding 1: system instruction references correct tool name ──


def test_system_instruction_references_run_skill_script():
  """System instruction must reference the actual tool name."""
  assert "run_skill_script" in skill_toolset.DEFAULT_SKILL_SYSTEM_INSTRUCTION
  assert (
      "execute_skill_script"
      not in skill_toolset.DEFAULT_SKILL_SYSTEM_INSTRUCTION
  )


# ── Finding 2: empty files are mounted (not silently dropped) ──


@pytest.mark.asyncio
async def test_execute_script_empty_files_mounted():
  """Verify empty files are included in wrapper code, not dropped."""
  skill = mock.create_autospec(models.Skill, instance=True)
  skill.name = "skill_empty"
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
  skill.resources.get_reference.side_effect = (
      lambda n: "" if n == "empty.md" else None
  )
  skill.resources.get_asset.side_effect = (
      lambda n: "" if n == "empty.cfg" else None
  )
  skill.resources.get_script.side_effect = (
      lambda n: models.Script(src="") if n == "run.py" else None
  )
  skill.resources.list_references.return_value = ["empty.md"]
  skill.resources.list_assets.return_value = ["empty.cfg"]
  skill.resources.list_scripts.return_value = ["run.py"]

  executor = _make_mock_executor(stdout="ok\n")
  toolset = skill_toolset.SkillToolset([skill], code_executor=executor)
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  await tool.run_async(
      args={"skill_name": "skill_empty", "script_path": "run.py"},
      tool_context=ctx,
  )

  call_args = executor.execute_code.call_args
  code_input = call_args[0][1]
  assert "'references/empty.md': ''" in code_input.code
  assert "'assets/empty.cfg': ''" in code_input.code
  assert "'scripts/run.py': ''" in code_input.code


# ── Finding 3: invalid args type returns clear error ──


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_args",
    [
        "not a dict",
        ["a", "list"],
        42,
        True,
    ],
)
async def test_execute_script_invalid_args_type(mock_skill1, bad_args):
  """Non-dict args should return INVALID_ARGS_TYPE, not crash."""
  executor = _make_mock_executor()
  toolset = skill_toolset.SkillToolset([mock_skill1], code_executor=executor)
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await tool.run_async(
      args={
          "skill_name": "skill1",
          "script_path": "run.py",
          "args": bad_args,
      },
      tool_context=ctx,
  )
  assert result["error_code"] == "INVALID_ARGS_TYPE"
  executor.execute_code.assert_not_called()


# ── Finding 4: binary file content is handled in wrapper ──


@pytest.mark.asyncio
async def test_execute_script_binary_content_packaged():
  """Verify binary asset content uses 'wb' mode in wrapper code."""
  skill = mock.create_autospec(models.Skill, instance=True)
  skill.name = "skill_bin"
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
  skill.resources.get_reference.side_effect = (
      lambda n: b"\x00\x01\x02" if n == "data.bin" else None
  )
  skill.resources.get_asset.return_value = None
  skill.resources.get_script.side_effect = lambda n: (
      models.Script(src="print('ok')") if n == "run.py" else None
  )
  skill.resources.list_references.return_value = ["data.bin"]
  skill.resources.list_assets.return_value = []
  skill.resources.list_scripts.return_value = ["run.py"]

  executor = _make_mock_executor(stdout="ok\n")
  toolset = skill_toolset.SkillToolset([skill], code_executor=executor)
  tool = skill_toolset.RunSkillScriptTool(toolset)
  ctx = _make_tool_context_with_agent()
  await tool.run_async(
      args={"skill_name": "skill_bin", "script_path": "run.py"},
      tool_context=ctx,
  )

  call_args = executor.execute_code.call_args
  code_input = call_args[0][1]
  # Binary content should appear as bytes literal
  assert "b'\\x00\\x01\\x02'" in code_input.code
  # Wrapper code handles binary with 'wb' mode
  assert "'wb' if isinstance(content, bytes)" in code_input.code
