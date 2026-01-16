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

"""Unit tests for base skill classes."""

from __future__ import annotations

from typing import Any
from typing import List

from google.adk.skills import BaseSkill
from google.adk.skills import SkillConfig
import pytest


class TestSkillConfig:
  """Tests for SkillConfig."""

  def test_default_values(self):
    """Test default configuration values."""
    config = SkillConfig()

    assert config.max_parallel_calls == 10
    assert config.timeout_seconds == 60.0
    assert config.allow_network is False
    assert config.memory_limit_mb == 256

  def test_custom_values(self):
    """Test custom configuration values."""
    config = SkillConfig(
        max_parallel_calls=20,
        timeout_seconds=120.0,
        allow_network=True,
        memory_limit_mb=512,
    )

    assert config.max_parallel_calls == 20
    assert config.timeout_seconds == 120.0
    assert config.allow_network is True
    assert config.memory_limit_mb == 512

  def test_forbid_extra_fields(self):
    """Test that extra fields are forbidden."""
    with pytest.raises(ValueError):
      SkillConfig(unknown_field="value")


class ConcreteSkill(BaseSkill):
  """Concrete implementation for testing."""

  name: str = "test_skill"
  description: str = "A test skill"

  def get_tool_declarations(self) -> List[dict[str, Any]]:
    return [
        {"name": "tool1", "description": "First tool"},
        {"name": "tool2", "description": "Second tool"},
    ]

  def get_orchestration_template(self) -> str:
    return """
async def test_func(tools):
    result = await tools.tool1()
    return result
"""


class TestBaseSkill:
  """Tests for BaseSkill."""

  def test_create_skill(self):
    """Test creating a skill with default values."""
    skill = ConcreteSkill()

    assert skill.name == "test_skill"
    assert skill.description == "A test skill"
    assert skill.config.max_parallel_calls == 10
    assert skill.allowed_callers == ["code_execution_20250825"]

  def test_create_skill_with_custom_config(self):
    """Test creating a skill with custom config."""
    config = SkillConfig(timeout_seconds=300.0)
    skill = ConcreteSkill(config=config)

    assert skill.config.timeout_seconds == 300.0

  def test_get_tool_declarations(self):
    """Test getting tool declarations."""
    skill = ConcreteSkill()
    declarations = skill.get_tool_declarations()

    assert len(declarations) == 2
    assert declarations[0]["name"] == "tool1"
    assert declarations[1]["name"] == "tool2"

  def test_get_orchestration_template(self):
    """Test getting orchestration template."""
    skill = ConcreteSkill()
    template = skill.get_orchestration_template()

    assert "async def test_func" in template
    assert "tools.tool1()" in template

  def test_filter_result_default(self):
    """Test default filter_result returns unchanged data."""
    skill = ConcreteSkill()
    data = {"key": "value", "nested": {"inner": 123}}

    result = skill.filter_result(data)

    assert result == data

  def test_is_programmatically_callable(self):
    """Test checking if skill is programmatically callable."""
    skill = ConcreteSkill()

    assert skill.is_programmatically_callable() is True

  def test_is_not_programmatically_callable(self):
    """Test skill with empty allowed_callers."""
    skill = ConcreteSkill(allowed_callers=[])

    assert skill.is_programmatically_callable() is False

  def test_get_skill_prompt(self):
    """Test generating skill prompt."""
    skill = ConcreteSkill()
    prompt = skill.get_skill_prompt()

    assert "test_skill" in prompt
    assert "A test skill" in prompt
    assert "tool1" in prompt
    assert "tool2" in prompt
    assert "async def test_func" in prompt


class CustomFilterSkill(ConcreteSkill):
  """Skill with custom filter logic."""

  def filter_result(self, result: Any) -> Any:
    if isinstance(result, dict) and "sensitive" in result:
      result = result.copy()
      del result["sensitive"]
    return result


class TestSkillFiltering:
  """Tests for skill result filtering."""

  def test_custom_filter(self):
    """Test custom filtering logic."""
    skill = CustomFilterSkill()
    data = {"public": "info", "sensitive": "secret"}

    result = skill.filter_result(data)

    assert "public" in result
    assert "sensitive" not in result

  def test_filter_preserves_original(self):
    """Test that filtering doesn't modify original data."""
    skill = CustomFilterSkill()
    original = {"public": "info", "sensitive": "secret"}
    data = original.copy()

    skill.filter_result(data)

    assert "sensitive" in original
