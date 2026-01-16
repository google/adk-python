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

"""Unit tests for SkillTool class."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any
from typing import List
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from google.adk.skills import BaseSkill
from google.adk.skills import create_skill_tools
from google.adk.skills import MarkdownSkill
from google.adk.skills import ScriptExecutor
from google.adk.skills import SkillTool
import pytest


class ConcreteSkill(BaseSkill):
  """Concrete skill for testing."""

  def get_tool_declarations(self) -> List[dict[str, Any]]:
    return [{"name": "test_tool", "description": "A test tool"}]

  def get_orchestration_template(self) -> str:
    return "async def test(tools): pass"


class TestSkillTool:
  """Tests for SkillTool."""

  @pytest.fixture
  def skill_dir(self):
    """Create a temporary skill directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
      skill_path = Path(tmpdir) / "test-skill"
      skill_path.mkdir()

      # Create SKILL.md
      skill_md = skill_path / "SKILL.md"
      skill_md.write_text("""---
name: test-skill
description: A test skill for unit testing.
---

# Test Skill

## Instructions
Follow these instructions to use the skill.
""")

      # Create scripts directory
      scripts_dir = skill_path / "scripts"
      scripts_dir.mkdir()

      # Create a Python script
      (scripts_dir / "hello.py").write_text('''"""Say hello."""
print("Hello, World!")
''')

      # Create references directory
      refs_dir = skill_path / "references"
      refs_dir.mkdir()
      (refs_dir / "GUIDE.md").write_text("# Guide\n\nReference guide content.")

      yield skill_path

  @pytest.fixture
  def markdown_skill(self, skill_dir):
    """Create a MarkdownSkill from the test directory."""
    return MarkdownSkill.from_directory(skill_dir)

  @pytest.fixture
  def concrete_skill(self):
    """Create a concrete BaseSkill."""
    return ConcreteSkill(
        name="concrete-skill",
        description="A concrete test skill",
    )

  @pytest.fixture
  def tool_context(self):
    """Create a mock tool context."""
    context = MagicMock()
    context.invocation_context = MagicMock()
    return context

  def test_create_from_markdown_skill(self, markdown_skill):
    """Test creating SkillTool from MarkdownSkill."""
    tool = SkillTool(markdown_skill)

    assert tool.name == "skill_test_skill"
    assert "test skill for unit testing" in tool.description
    assert tool.skill == markdown_skill

  def test_create_from_concrete_skill(self, concrete_skill):
    """Test creating SkillTool from concrete BaseSkill."""
    tool = SkillTool(concrete_skill)

    assert tool.name == "skill_concrete_skill"
    assert "concrete test skill" in tool.description

  def test_description_includes_actions(self, markdown_skill):
    """Test that description includes available actions."""
    tool = SkillTool(markdown_skill)

    assert "activate" in tool.description
    assert "run_script" in tool.description
    assert "load_reference" in tool.description

  def test_get_declaration(self, markdown_skill):
    """Test getting function declaration."""
    tool = SkillTool(markdown_skill)

    declaration = tool._get_declaration()

    assert declaration is not None
    assert declaration.name == "skill_test_skill"
    assert "action" in declaration.parameters.properties

  @pytest.mark.asyncio
  async def test_activate_action(self, markdown_skill, tool_context):
    """Test activate action."""
    tool = SkillTool(markdown_skill)

    result = await tool.run_async(
        args={"action": "activate"},
        tool_context=tool_context,
    )

    assert result["status"] == "activated"
    assert result["skill"] == "test-skill"
    assert "Instructions" in result["instructions"]
    assert "hello.py" in result["available_scripts"]
    assert "GUIDE.md" in result["available_references"]

  @pytest.mark.asyncio
  async def test_activate_action_concrete_skill(
      self, concrete_skill, tool_context
  ):
    """Test activate action with concrete skill."""
    tool = SkillTool(concrete_skill)

    result = await tool.run_async(
        args={"action": "activate"},
        tool_context=tool_context,
    )

    assert result["status"] == "activated"
    assert result["skill"] == "concrete-skill"
    assert "prompt" in result
    assert "tool_declarations" in result

  @pytest.mark.asyncio
  async def test_run_script_action(self, markdown_skill, tool_context):
    """Test run_script action."""
    tool = SkillTool(markdown_skill)

    result = await tool.run_async(
        args={"action": "run_script", "script": "hello.py"},
        tool_context=tool_context,
    )

    assert result["script"] == "hello.py"
    assert result["success"] is True
    assert "Hello, World!" in result["stdout"]

  @pytest.mark.asyncio
  async def test_run_script_with_args(self, skill_dir, tool_context):
    """Test run_script action with arguments."""
    # Create a script that uses arguments
    scripts_dir = skill_dir / "scripts"
    (scripts_dir / "greet.py").write_text('''"""Greet someone."""
import sys
name = sys.argv[1] if len(sys.argv) > 1 else "World"
print(f"Hello, {name}!")
''')

    skill = MarkdownSkill.from_directory(skill_dir)
    tool = SkillTool(skill)

    result = await tool.run_async(
        args={"action": "run_script", "script": "greet.py", "args": ["Alice"]},
        tool_context=tool_context,
    )

    assert result["success"] is True
    assert "Hello, Alice!" in result["stdout"]

  @pytest.mark.asyncio
  async def test_run_script_not_found(self, markdown_skill, tool_context):
    """Test run_script action with non-existent script."""
    tool = SkillTool(markdown_skill)

    result = await tool.run_async(
        args={"action": "run_script", "script": "nonexistent.py"},
        tool_context=tool_context,
    )

    assert "error" in result
    assert "not found" in result["error"].lower()
    assert "available_scripts" in result

  @pytest.mark.asyncio
  async def test_run_script_no_script_name(self, markdown_skill, tool_context):
    """Test run_script action without script name."""
    tool = SkillTool(markdown_skill)

    result = await tool.run_async(
        args={"action": "run_script"},
        tool_context=tool_context,
    )

    assert "error" in result
    assert "Script name required" in result["error"]

  @pytest.mark.asyncio
  async def test_run_script_concrete_skill(self, concrete_skill, tool_context):
    """Test run_script action with non-markdown skill."""
    tool = SkillTool(concrete_skill)

    result = await tool.run_async(
        args={"action": "run_script", "script": "test.py"},
        tool_context=tool_context,
    )

    assert "error" in result
    assert "does not support scripts" in result["error"]

  @pytest.mark.asyncio
  async def test_load_reference_action(self, markdown_skill, tool_context):
    """Test load_reference action."""
    tool = SkillTool(markdown_skill)

    result = await tool.run_async(
        args={"action": "load_reference", "reference": "GUIDE.md"},
        tool_context=tool_context,
    )

    assert result["reference"] == "GUIDE.md"
    assert "Reference guide content" in result["content"]

  @pytest.mark.asyncio
  async def test_load_reference_not_found(self, markdown_skill, tool_context):
    """Test load_reference action with non-existent reference."""
    tool = SkillTool(markdown_skill)

    result = await tool.run_async(
        args={"action": "load_reference", "reference": "nonexistent.md"},
        tool_context=tool_context,
    )

    assert "error" in result
    assert "not found" in result["error"].lower()
    assert "available_references" in result

  @pytest.mark.asyncio
  async def test_load_reference_no_name(self, markdown_skill, tool_context):
    """Test load_reference action without reference name."""
    tool = SkillTool(markdown_skill)

    result = await tool.run_async(
        args={"action": "load_reference"},
        tool_context=tool_context,
    )

    assert "error" in result
    assert "Reference name required" in result["error"]

  @pytest.mark.asyncio
  async def test_unknown_action(self, markdown_skill, tool_context):
    """Test unknown action."""
    tool = SkillTool(markdown_skill)

    result = await tool.run_async(
        args={"action": "unknown"},
        tool_context=tool_context,
    )

    assert "error" in result
    assert "Unknown action" in result["error"]
    assert "available_actions" in result

  @pytest.mark.asyncio
  async def test_default_action_is_activate(self, markdown_skill, tool_context):
    """Test that default action is activate."""
    tool = SkillTool(markdown_skill)

    result = await tool.run_async(
        args={},
        tool_context=tool_context,
    )

    assert result["status"] == "activated"

  def test_custom_script_executor(self, markdown_skill):
    """Test using custom script executor."""
    executor = ScriptExecutor(timeout_seconds=30.0)
    tool = SkillTool(markdown_skill, script_executor=executor)

    assert tool._script_executor == executor

  def test_custom_working_dir(self, markdown_skill):
    """Test using custom working directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
      tool = SkillTool(markdown_skill, working_dir=tmpdir)

      # Resolve both paths to handle symlinks (e.g., /var -> /private/var on macOS)
      assert tool.working_dir == Path(tmpdir).resolve()

  def test_working_dir_defaults_to_skill_path(self, markdown_skill):
    """Test that working dir defaults to skill path."""
    tool = SkillTool(markdown_skill)

    assert tool.working_dir == markdown_skill.skill_path

  def test_working_dir_setter(self, markdown_skill):
    """Test setting working directory."""
    tool = SkillTool(markdown_skill)

    with tempfile.TemporaryDirectory() as tmpdir:
      tool.working_dir = tmpdir
      # Resolve both paths to handle symlinks (e.g., /var -> /private/var on macOS)
      assert tool.working_dir == Path(tmpdir).resolve()


class TestCreateSkillTools:
  """Tests for create_skill_tools function."""

  def test_create_tools_from_skills(self):
    """Test creating tools from multiple skills."""
    skills = [
        ConcreteSkill(name="skill-1", description="First skill"),
        ConcreteSkill(name="skill-2", description="Second skill"),
    ]

    tools = create_skill_tools(skills)

    assert len(tools) == 2
    assert all(isinstance(t, SkillTool) for t in tools)
    assert tools[0].name == "skill_skill_1"
    assert tools[1].name == "skill_skill_2"

  def test_create_tools_with_shared_executor(self):
    """Test creating tools with shared executor."""
    skills = [
        ConcreteSkill(name="skill-1", description="First skill"),
        ConcreteSkill(name="skill-2", description="Second skill"),
    ]
    executor = ScriptExecutor(timeout_seconds=30.0)

    tools = create_skill_tools(skills, script_executor=executor)

    assert all(t._script_executor == executor for t in tools)

  def test_create_tools_with_working_dir(self):
    """Test creating tools with shared working directory."""
    skills = [
        ConcreteSkill(name="skill-1", description="First skill"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
      tools = create_skill_tools(skills, working_dir=tmpdir)

      # Resolve both paths to handle symlinks (e.g., /var -> /private/var on macOS)
      assert tools[0].working_dir == Path(tmpdir).resolve()

  def test_create_tools_empty_list(self):
    """Test creating tools from empty list."""
    tools = create_skill_tools([])

    assert len(tools) == 0


class TestSkillToolNameSanitization:
  """Tests for skill name sanitization in tool names."""

  def test_hyphens_replaced(self):
    """Test that hyphens are replaced with underscores."""
    skill = ConcreteSkill(name="my-skill-name", description="Test")
    tool = SkillTool(skill)

    assert tool.name == "skill_my_skill_name"

  def test_dots_replaced(self):
    """Test that dots are replaced with underscores."""
    skill = ConcreteSkill(name="my.skill.name", description="Test")
    tool = SkillTool(skill)

    assert tool.name == "skill_my_skill_name"

  def test_mixed_special_chars(self):
    """Test mixed special characters."""
    skill = ConcreteSkill(name="my-skill.name", description="Test")
    tool = SkillTool(skill)

    assert tool.name == "skill_my_skill_name"
