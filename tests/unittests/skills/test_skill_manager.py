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

"""Unit tests for skills manager."""

from __future__ import annotations

from typing import Any
from typing import List

from google.adk.skills import BaseSkill
from google.adk.skills import SkillInvocationResult
from google.adk.skills import SkillsManager
import pytest


class MockSkill(BaseSkill):
  """Mock skill for testing."""

  name: str = "mock_skill"
  description: str = "A mock skill for testing"

  def get_tool_declarations(self) -> List[dict[str, Any]]:
    return [{"name": "mock_tool", "description": "A mock tool"}]

  def get_orchestration_template(self) -> str:
    return "async def mock(tools): pass"


class MockSkill2(BaseSkill):
  """Second mock skill for testing."""

  name: str = "mock_skill_2"
  description: str = "Another mock skill"

  def get_tool_declarations(self) -> List[dict[str, Any]]:
    return [{"name": "mock_tool_2", "description": "Another mock tool"}]

  def get_orchestration_template(self) -> str:
    return "async def mock2(tools): pass"


class TestSkillsManager:
  """Tests for SkillsManager."""

  def test_register_skill(self):
    """Test registering a skill."""
    manager = SkillsManager()
    skill = MockSkill()

    manager.register_skill(skill)

    assert manager.has_skill("mock_skill")
    assert manager.get_skill("mock_skill") == skill

  def test_register_duplicate_skill_raises(self):
    """Test that registering a duplicate skill raises ValueError."""
    manager = SkillsManager()
    skill1 = MockSkill()
    skill2 = MockSkill()

    manager.register_skill(skill1)

    with pytest.raises(ValueError, match="already registered"):
      manager.register_skill(skill2)

  def test_unregister_skill(self):
    """Test unregistering a skill."""
    manager = SkillsManager()
    skill = MockSkill()

    manager.register_skill(skill)
    result = manager.unregister_skill("mock_skill")

    assert result is True
    assert not manager.has_skill("mock_skill")

  def test_unregister_nonexistent_skill(self):
    """Test unregistering a skill that doesn't exist."""
    manager = SkillsManager()

    result = manager.unregister_skill("nonexistent")

    assert result is False

  def test_get_skill(self):
    """Test getting a skill by name."""
    manager = SkillsManager()
    skill = MockSkill()
    manager.register_skill(skill)

    retrieved = manager.get_skill("mock_skill")

    assert retrieved == skill

  def test_get_nonexistent_skill(self):
    """Test getting a skill that doesn't exist."""
    manager = SkillsManager()

    result = manager.get_skill("nonexistent")

    assert result is None

  def test_get_all_skills(self):
    """Test getting all registered skills."""
    manager = SkillsManager()
    skill1 = MockSkill()
    skill2 = MockSkill2()

    manager.register_skill(skill1)
    manager.register_skill(skill2)

    skills = manager.get_all_skills()

    assert len(skills) == 2
    assert skill1 in skills
    assert skill2 in skills

  def test_get_skill_names(self):
    """Test getting all skill names."""
    manager = SkillsManager()
    skill1 = MockSkill()
    skill2 = MockSkill2()

    manager.register_skill(skill1)
    manager.register_skill(skill2)

    names = manager.get_skill_names()

    assert set(names) == {"mock_skill", "mock_skill_2"}

  def test_has_skill(self):
    """Test checking if a skill exists."""
    manager = SkillsManager()
    skill = MockSkill()

    assert not manager.has_skill("mock_skill")

    manager.register_skill(skill)

    assert manager.has_skill("mock_skill")

  def test_clear(self):
    """Test clearing all skills."""
    manager = SkillsManager()
    manager.register_skill(MockSkill())
    manager.register_skill(MockSkill2())

    manager.clear()

    assert len(manager.get_all_skills()) == 0


class TestSkillInvocationResult:
  """Tests for SkillInvocationResult."""

  def test_success_result(self):
    """Test creating a success result."""
    result = SkillInvocationResult(
        success=True,
        result={"data": "value"},
        execution_time_ms=100.5,
    )

    assert result.success is True
    assert result.result == {"data": "value"}
    assert result.error is None
    assert result.execution_time_ms == 100.5

  def test_error_result(self):
    """Test creating an error result."""
    result = SkillInvocationResult(
        success=False,
        error="Something went wrong",
        execution_time_ms=50.0,
    )

    assert result.success is False
    assert result.result is None
    assert result.error == "Something went wrong"

  def test_default_values(self):
    """Test default values."""
    result = SkillInvocationResult(success=True)

    assert result.result is None
    assert result.error is None
    assert result.call_log == []
    assert result.execution_time_ms == 0.0


class FilteringSkill(BaseSkill):
  """Skill with custom filtering for testing."""

  name: str = "filtering_skill"
  description: str = "A skill that filters results"

  def get_tool_declarations(self) -> List[dict[str, Any]]:
    return []

  def get_orchestration_template(self) -> str:
    return "async def f(tools): pass"

  def filter_result(self, result: Any) -> Any:
    if isinstance(result, dict):
      return {k: v for k, v in result.items() if not k.startswith("_")}
    return result


class TestSkillsManagerExecution:
  """Tests for SkillsManager.execute_skill."""

  @pytest.mark.asyncio
  async def test_execute_skill_success(self):
    """Test executing a skill successfully."""
    skill = FilteringSkill()
    tool_results = {"public": "data", "_private": "secret"}

    result = await SkillsManager.execute_skill(skill, tool_results)

    assert result.success is True
    assert result.result == {"public": "data"}
    assert "_private" not in result.result
    assert result.execution_time_ms > 0

  @pytest.mark.asyncio
  async def test_execute_skill_with_error(self):
    """Test executing a skill that raises an error."""

    class ErrorSkill(BaseSkill):
      name: str = "error_skill"
      description: str = "A skill that errors"

      def get_tool_declarations(self) -> List[dict[str, Any]]:
        return []

      def get_orchestration_template(self) -> str:
        return ""

      def filter_result(self, result: Any) -> Any:
        raise ValueError("Filter error")

    skill = ErrorSkill()

    result = await SkillsManager.execute_skill(skill, {"data": "value"})

    assert result.success is False
    assert "Filter error" in result.error
