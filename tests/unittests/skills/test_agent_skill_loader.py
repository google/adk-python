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

"""Unit tests for AgentSkillLoader class."""

from __future__ import annotations

from pathlib import Path
import tempfile

from google.adk.skills import AgentSkillLoader
from google.adk.skills import MarkdownSkill
from google.adk.skills import SkillsManager
import pytest


def create_skill_in_dir(base_dir: Path, name: str, description: str) -> Path:
  """Helper to create a skill directory with SKILL.md."""
  skill_path = base_dir / name
  skill_path.mkdir(parents=True, exist_ok=True)

  skill_md = skill_path / "SKILL.md"
  skill_md.write_text(f"""---
name: {name}
description: {description}
---

# {name}

Instructions for {name}.
""")

  return skill_path


class TestAgentSkillLoader:
  """Tests for AgentSkillLoader."""

  @pytest.fixture
  def skills_dir(self):
    """Create a temporary directory with multiple skills."""
    with tempfile.TemporaryDirectory() as tmpdir:
      base = Path(tmpdir)

      # Create multiple skills
      create_skill_in_dir(base, "skill-one", "First test skill")
      create_skill_in_dir(base, "skill-two", "Second test skill")
      create_skill_in_dir(base, "skill-three", "Third test skill")

      # Create a non-skill directory
      (base / "not-a-skill").mkdir()

      # Create a hidden directory (should be skipped)
      hidden = base / ".hidden-skill"
      hidden.mkdir()
      (hidden / "SKILL.md").write_text(
          "---\nname: hidden\ndescription: Hidden\n---\nContent"
      )

      yield base

  def test_add_skill_directory(self, skills_dir):
    """Test adding a skill directory."""
    loader = AgentSkillLoader()

    count = loader.add_skill_directory(skills_dir)

    assert count == 3
    assert len(loader) == 3

  def test_add_skill_directory_not_found(self):
    """Test adding non-existent directory."""
    loader = AgentSkillLoader()

    with pytest.raises(FileNotFoundError):
      loader.add_skill_directory("/nonexistent/path")

  def test_add_skill_directory_not_a_directory(self):
    """Test adding a file instead of directory."""
    with tempfile.NamedTemporaryFile() as f:
      loader = AgentSkillLoader()

      with pytest.raises(ValueError, match="not a directory"):
        loader.add_skill_directory(f.name)

  def test_add_single_skill(self, skills_dir):
    """Test adding a single skill."""
    loader = AgentSkillLoader()

    result = loader.add_skill(skills_dir / "skill-one")

    assert result is True
    assert len(loader) == 1
    assert "skill-one" in loader

  def test_add_single_skill_not_found(self):
    """Test adding non-existent skill."""
    loader = AgentSkillLoader()

    result = loader.add_skill("/nonexistent/path")

    assert result is False
    assert len(loader.get_load_errors()) == 1

  def test_get_skill(self, skills_dir):
    """Test getting a skill by name."""
    loader = AgentSkillLoader()
    loader.add_skill_directory(skills_dir)

    skill = loader.get_skill("skill-one")

    assert skill is not None
    assert skill.name == "skill-one"
    assert isinstance(skill, MarkdownSkill)

  def test_get_skill_not_found(self, skills_dir):
    """Test getting non-existent skill."""
    loader = AgentSkillLoader()
    loader.add_skill_directory(skills_dir)

    skill = loader.get_skill("nonexistent")

    assert skill is None

  def test_get_all_skills(self, skills_dir):
    """Test getting all skills."""
    loader = AgentSkillLoader()
    loader.add_skill_directory(skills_dir)

    skills = loader.get_all_skills()

    assert len(skills) == 3
    assert all(isinstance(s, MarkdownSkill) for s in skills)

  def test_get_skill_names(self, skills_dir):
    """Test getting skill names."""
    loader = AgentSkillLoader()
    loader.add_skill_directory(skills_dir)

    names = loader.get_skill_names()

    assert sorted(names) == ["skill-one", "skill-three", "skill-two"]

  def test_has_skill(self, skills_dir):
    """Test checking if skill exists."""
    loader = AgentSkillLoader()
    loader.add_skill_directory(skills_dir)

    assert loader.has_skill("skill-one") is True
    assert loader.has_skill("nonexistent") is False

  def test_contains(self, skills_dir):
    """Test __contains__ method."""
    loader = AgentSkillLoader()
    loader.add_skill_directory(skills_dir)

    assert "skill-one" in loader
    assert "nonexistent" not in loader

  def test_iter(self, skills_dir):
    """Test __iter__ method."""
    loader = AgentSkillLoader()
    loader.add_skill_directory(skills_dir)

    skills = list(loader)

    assert len(skills) == 3

  def test_clear(self, skills_dir):
    """Test clearing all skills."""
    loader = AgentSkillLoader()
    loader.add_skill_directory(skills_dir)

    loader.clear()

    assert len(loader) == 0

  def test_register_all(self, skills_dir):
    """Test registering all skills with manager."""
    loader = AgentSkillLoader()
    loader.add_skill_directory(skills_dir)

    manager = SkillsManager()
    count = loader.register_all(manager)

    assert count == 3
    assert manager.has_skill("skill-one")
    assert manager.has_skill("skill-two")
    assert manager.has_skill("skill-three")

  def test_generate_discovery_prompt(self, skills_dir):
    """Test generating discovery prompt."""
    loader = AgentSkillLoader()
    loader.add_skill_directory(skills_dir)

    prompt = loader.generate_discovery_prompt()

    assert "<available_skills>" in prompt
    assert "</available_skills>" in prompt
    assert "skill-one" in prompt
    assert "skill-two" in prompt
    assert "skill-three" in prompt
    assert "First test skill" in prompt

  def test_generate_discovery_prompt_empty(self):
    """Test generating prompt with no skills."""
    loader = AgentSkillLoader()

    prompt = loader.generate_discovery_prompt()

    assert prompt == "<available_skills></available_skills>"

  def test_generate_discovery_prompt_without_resources(self, skills_dir):
    """Test generating prompt without resource hints."""
    loader = AgentSkillLoader()
    loader.add_skill_directory(skills_dir)

    prompt = loader.generate_discovery_prompt(include_resources=False)

    assert "<available_skills>" in prompt
    assert "has_scripts" not in prompt
    assert "has_references" not in prompt

  def test_generate_activation_prompt(self, skills_dir):
    """Test generating activation prompt."""
    loader = AgentSkillLoader()
    loader.add_skill_directory(skills_dir)

    prompt = loader.generate_activation_prompt("skill-one")

    assert prompt is not None
    assert "# Skill: skill-one" in prompt
    assert "Instructions for skill-one" in prompt

  def test_generate_activation_prompt_not_found(self, skills_dir):
    """Test generating activation prompt for non-existent skill."""
    loader = AgentSkillLoader()
    loader.add_skill_directory(skills_dir)

    prompt = loader.generate_activation_prompt("nonexistent")

    assert prompt is None

  def test_generate_summary(self, skills_dir):
    """Test generating summary."""
    loader = AgentSkillLoader()
    loader.add_skill_directory(skills_dir)

    summary = loader.generate_summary()

    assert "Agent Skills Loader Summary" in summary
    assert "Skills discovered: 3" in summary
    assert "skill-one" in summary

  def test_load_errors(self):
    """Test tracking load errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
      base = Path(tmpdir)

      # Create a skill with invalid SKILL.md
      invalid_skill = base / "invalid-skill"
      invalid_skill.mkdir()
      (invalid_skill / "SKILL.md").write_text(
          "Invalid content without frontmatter"
      )

      loader = AgentSkillLoader()
      loader.add_skill_directory(base)

      errors = loader.get_load_errors()

      assert len(errors) == 1
      assert str(invalid_skill) in list(errors.keys())[0]

  def test_multiple_directories(self):
    """Test loading from multiple directories."""
    with (
        tempfile.TemporaryDirectory() as tmpdir1,
        tempfile.TemporaryDirectory() as tmpdir2,
    ):

      base1 = Path(tmpdir1)
      base2 = Path(tmpdir2)

      create_skill_in_dir(base1, "skill-a", "Skill A")
      create_skill_in_dir(base2, "skill-b", "Skill B")

      loader = AgentSkillLoader()
      loader.add_skill_directory(base1)
      loader.add_skill_directory(base2)

      assert len(loader) == 2
      assert "skill-a" in loader
      assert "skill-b" in loader

  def test_duplicate_skill_names(self):
    """Test handling duplicate skill names."""
    with (
        tempfile.TemporaryDirectory() as tmpdir1,
        tempfile.TemporaryDirectory() as tmpdir2,
    ):

      base1 = Path(tmpdir1)
      base2 = Path(tmpdir2)

      create_skill_in_dir(base1, "duplicate-skill", "First version")
      create_skill_in_dir(base2, "duplicate-skill", "Second version")

      loader = AgentSkillLoader()
      loader.add_skill_directory(base1)
      loader.add_skill_directory(base2)

      # Should have only one skill (second shadows first)
      skill = loader.get_skill("duplicate-skill")
      assert skill.description == "Second version"

  def test_xml_escaping(self):
    """Test that XML special characters are escaped."""
    with tempfile.TemporaryDirectory() as tmpdir:
      base = Path(tmpdir)

      # Create skill with special characters in description
      skill_path = base / "special-chars"
      skill_path.mkdir()
      (skill_path / "SKILL.md").write_text("""---
name: special-chars
description: Handle <special> & "characters"
---

Content
""")

      loader = AgentSkillLoader()
      loader.add_skill_directory(base)

      prompt = loader.generate_discovery_prompt()

      assert "&lt;special&gt;" in prompt
      assert "&amp;" in prompt
      assert "&quot;characters&quot;" in prompt

  def test_validate_names_disabled(self):
    """Test with name validation disabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
      base = Path(tmpdir)

      # Create skill with mismatched name
      skill_path = base / "directory-name"
      skill_path.mkdir()
      (skill_path / "SKILL.md").write_text("""---
name: different-name
description: Test skill
---

Content
""")

      # With validation enabled, should not load
      loader1 = AgentSkillLoader(validate_names=True)
      loader1.add_skill_directory(base)
      assert len(loader1) == 0

      # With validation disabled, should load
      loader2 = AgentSkillLoader(validate_names=False)
      loader2.add_skill_directory(base)
      assert len(loader2) == 1
      assert "different-name" in loader2
