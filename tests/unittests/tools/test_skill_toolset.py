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

from google.adk.models import llm_request
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

  skill.resources.get_reference.side_effect = get_ref
  skill.resources.get_asset.side_effect = get_asset
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
  frontmatters = toolset._list_skills()
  assert len(frontmatters) == 2
  assert mock_skill1.frontmatter in frontmatters
  assert mock_skill2.frontmatter in frontmatters


@pytest.mark.asyncio
async def test_get_tools(mock_skill1, mock_skill2):
  toolset = skill_toolset.SkillToolset([mock_skill1, mock_skill2])
  tools = await toolset.get_tools()
  assert len(tools) == 2
  assert isinstance(tools[0], skill_toolset.LoadSkillTool)
  assert isinstance(tools[1], skill_toolset.LoadSkillResourceTool)


@pytest.mark.asyncio
async def test_process_llm_request(
    mock_skill1, mock_skill2, tool_context_instance
):
  toolset = skill_toolset.SkillToolset([mock_skill1, mock_skill2])
  mock_llm_request = llm_request.LlmRequest()
  mock_llm_request.config.system_instruction = "existing instruction"
  await toolset.process_llm_request(
      tool_context=tool_context_instance, llm_request=mock_llm_request
  )
  assert "<available_skills>" in mock_llm_request.config.system_instruction
  assert (
      "You can use specialized 'skills'"
      in mock_llm_request.config.system_instruction
  )
  assert (
      "skills are folders" in mock_llm_request.config.system_instruction.lower()
  )
  assert mock_llm_request.config.system_instruction.startswith(
      "existing instruction"
  )


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
                "error": "Path must start with 'references/' or 'assets/'.",
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
