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

"""Unit tests for MarkdownSkill class."""

from __future__ import annotations

from pathlib import Path
import tempfile

from google.adk.skills import MarkdownSkill
from google.adk.skills import MarkdownSkillMetadata
import pytest


class TestMarkdownSkillMetadata:
  """Tests for MarkdownSkillMetadata."""

  def test_required_fields(self):
    """Test that name and description are required."""
    metadata = MarkdownSkillMetadata(
        name="test-skill",
        description="A test skill",
    )

    assert metadata.name == "test-skill"
    assert metadata.description == "A test skill"

  def test_optional_fields(self):
    """Test optional fields."""
    metadata = MarkdownSkillMetadata(
        name="test-skill",
        description="A test skill",
        license="Apache-2.0",
        compatibility="Python 3.8+",
        metadata={"author": "test", "version": "1.0"},
    )

    assert metadata.license == "Apache-2.0"
    assert metadata.compatibility == "Python 3.8+"
    assert metadata.metadata["author"] == "test"

  def test_adk_extensions(self):
    """Test ADK-specific extensions."""
    metadata = MarkdownSkillMetadata(
        name="test-skill",
        description="A test skill",
        adk={
            "config": {
                "timeout_seconds": 120,
                "allow_network": True,
            },
            "allowed_callers": ["custom_caller"],
        },
    )

    assert metadata.adk["config"]["timeout_seconds"] == 120
    assert metadata.adk["config"]["allow_network"] is True


class TestMarkdownSkill:
  """Tests for MarkdownSkill."""

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
license: Apache-2.0
compatibility: Python 3.8+
metadata:
  author: test-author
  version: "1.0"
---

# Test Skill

## When to use this skill
Use this skill for testing.

## Instructions
1. Step one
2. Step two

## Examples
Example usage here.
""")

      # Create scripts directory
      scripts_dir = skill_path / "scripts"
      scripts_dir.mkdir()

      # Create a Python script
      (scripts_dir / "process.py").write_text('''"""Process data."""
import sys
print("Processing...")
''')

      # Create a shell script
      (scripts_dir / "run.sh").write_text("""# Run something
echo "Running"
""")

      # Create references directory
      refs_dir = skill_path / "references"
      refs_dir.mkdir()
      (refs_dir / "REFERENCE.md").write_text(
          "# Reference\n\nReference content."
      )

      # Create assets directory
      assets_dir = skill_path / "assets"
      assets_dir.mkdir()
      (assets_dir / "template.txt").write_text("Template content")

      yield skill_path

  def test_from_directory(self, skill_dir):
    """Test loading a skill from directory."""
    skill = MarkdownSkill.from_directory(skill_dir)

    assert skill.name == "test-skill"
    assert skill.description == "A test skill for unit testing."
    assert skill.skill_metadata.license == "Apache-2.0"
    assert skill.skill_metadata.metadata["author"] == "test-author"

  def test_from_directory_not_found(self):
    """Test loading from non-existent directory."""
    with pytest.raises(FileNotFoundError):
      MarkdownSkill.from_directory("/nonexistent/path")

  def test_from_directory_no_skill_md(self):
    """Test loading from directory without SKILL.md."""
    with tempfile.TemporaryDirectory() as tmpdir:
      skill_path = Path(tmpdir) / "empty-skill"
      skill_path.mkdir()

      with pytest.raises(FileNotFoundError, match="SKILL.md not found"):
        MarkdownSkill.from_directory(skill_path)

  def test_from_directory_name_mismatch(self):
    """Test that name must match directory name."""
    with tempfile.TemporaryDirectory() as tmpdir:
      skill_path = Path(tmpdir) / "wrong-name"
      skill_path.mkdir()

      skill_md = skill_path / "SKILL.md"
      skill_md.write_text("""---
name: correct-name
description: Test skill
---

Content
""")

      with pytest.raises(ValueError, match="must match directory name"):
        MarkdownSkill.from_directory(skill_path)

  def test_from_directory_skip_name_validation(self):
    """Test skipping name validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
      skill_path = Path(tmpdir) / "wrong-name"
      skill_path.mkdir()

      skill_md = skill_path / "SKILL.md"
      skill_md.write_text("""---
name: correct-name
description: Test skill
---

Content
""")

      # Should not raise when validation is disabled
      skill = MarkdownSkill.from_directory(skill_path, validate_name=False)
      assert skill.name == "correct-name"

  def test_get_instructions(self, skill_dir):
    """Test getting full instructions (Stage 2)."""
    skill = MarkdownSkill.from_directory(skill_dir)

    instructions = skill.get_instructions()

    assert "# Test Skill" in instructions
    assert "## When to use this skill" in instructions
    assert "## Instructions" in instructions
    assert skill.current_stage == 2

  def test_get_script(self, skill_dir):
    """Test getting script content (Stage 3)."""
    skill = MarkdownSkill.from_directory(skill_dir)

    script = skill.get_script("process.py")

    assert script is not None
    assert "Process data" in script
    assert skill.current_stage == 3

  def test_get_script_not_found(self, skill_dir):
    """Test getting non-existent script."""
    skill = MarkdownSkill.from_directory(skill_dir)

    script = skill.get_script("nonexistent.py")

    assert script is None

  def test_get_script_path(self, skill_dir):
    """Test getting script path."""
    skill = MarkdownSkill.from_directory(skill_dir)

    path = skill.get_script_path("process.py")

    assert path is not None
    assert path.exists()
    assert path.name == "process.py"

  def test_get_reference(self, skill_dir):
    """Test getting reference content (Stage 3)."""
    skill = MarkdownSkill.from_directory(skill_dir)

    ref = skill.get_reference("REFERENCE.md")

    assert ref is not None
    assert "Reference content" in ref
    assert skill.current_stage == 3

  def test_get_reference_not_found(self, skill_dir):
    """Test getting non-existent reference."""
    skill = MarkdownSkill.from_directory(skill_dir)

    ref = skill.get_reference("nonexistent.md")

    assert ref is None

  def test_get_asset_path(self, skill_dir):
    """Test getting asset path."""
    skill = MarkdownSkill.from_directory(skill_dir)

    path = skill.get_asset_path("template.txt")

    assert path is not None
    assert path.exists()

  def test_list_scripts(self, skill_dir):
    """Test listing available scripts."""
    skill = MarkdownSkill.from_directory(skill_dir)

    scripts = skill.list_scripts()

    assert "process.py" in scripts
    assert "run.sh" in scripts

  def test_list_references(self, skill_dir):
    """Test listing available references."""
    skill = MarkdownSkill.from_directory(skill_dir)

    refs = skill.list_references()

    assert "REFERENCE.md" in refs

  def test_list_assets(self, skill_dir):
    """Test listing available assets."""
    skill = MarkdownSkill.from_directory(skill_dir)

    assets = skill.list_assets()

    assert "template.txt" in assets

  def test_has_scripts(self, skill_dir):
    """Test checking for scripts."""
    skill = MarkdownSkill.from_directory(skill_dir)

    assert skill.has_scripts() is True

  def test_has_references(self, skill_dir):
    """Test checking for references."""
    skill = MarkdownSkill.from_directory(skill_dir)

    assert skill.has_references() is True

  def test_has_assets(self, skill_dir):
    """Test checking for assets."""
    skill = MarkdownSkill.from_directory(skill_dir)

    assert skill.has_assets() is True

  def test_get_tool_declarations(self, skill_dir):
    """Test getting tool declarations."""
    skill = MarkdownSkill.from_directory(skill_dir)

    declarations = skill.get_tool_declarations()

    # Should have declarations for scripts
    names = [d["name"] for d in declarations]
    assert any("process" in name for name in names)
    assert any("run" in name for name in names)

  def test_get_orchestration_template(self, skill_dir):
    """Test getting orchestration template."""
    skill = MarkdownSkill.from_directory(skill_dir)

    template = skill.get_orchestration_template()

    assert "async def" in template
    assert "tools" in template

  def test_get_skill_prompt(self, skill_dir):
    """Test getting skill prompt."""
    skill = MarkdownSkill.from_directory(skill_dir)

    prompt = skill.get_skill_prompt()

    assert skill.name in prompt
    assert skill.description in prompt
    assert "Scripts" in prompt
    assert "References" in prompt

  def test_progressive_disclosure_stages(self, skill_dir):
    """Test progressive disclosure stages."""
    skill = MarkdownSkill.from_directory(skill_dir)

    # Stage 1: Only metadata loaded
    assert skill.current_stage == 1

    # Stage 2: Instructions loaded
    skill.get_instructions()
    assert skill.current_stage == 2

    # Stage 3: Resources loaded
    skill.get_script("process.py")
    assert skill.current_stage == 3


class TestMarkdownSkillFrontmatter:
  """Tests for frontmatter parsing."""

  def test_missing_frontmatter(self):
    """Test handling missing frontmatter."""
    with tempfile.TemporaryDirectory() as tmpdir:
      skill_path = Path(tmpdir) / "no-frontmatter"
      skill_path.mkdir()

      skill_md = skill_path / "SKILL.md"
      skill_md.write_text("# No Frontmatter\n\nContent without frontmatter.")

      with pytest.raises(ValueError, match="must start with YAML frontmatter"):
        MarkdownSkill.from_directory(skill_path)

  def test_missing_name(self):
    """Test handling missing name field."""
    with tempfile.TemporaryDirectory() as tmpdir:
      skill_path = Path(tmpdir) / "missing-name"
      skill_path.mkdir()

      skill_md = skill_path / "SKILL.md"
      skill_md.write_text("""---
description: Test skill
---

Content
""")

      with pytest.raises(ValueError, match="Missing required field 'name'"):
        MarkdownSkill.from_directory(skill_path)

  def test_missing_description(self):
    """Test handling missing description field."""
    with tempfile.TemporaryDirectory() as tmpdir:
      skill_path = Path(tmpdir) / "missing-desc"
      skill_path.mkdir()

      skill_md = skill_path / "SKILL.md"
      skill_md.write_text("""---
name: missing-desc
---

Content
""")

      with pytest.raises(
          ValueError, match="Missing required field 'description'"
      ):
        MarkdownSkill.from_directory(skill_path)

  def test_invalid_yaml(self):
    """Test handling invalid YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
      skill_path = Path(tmpdir) / "invalid-yaml"
      skill_path.mkdir()

      skill_md = skill_path / "SKILL.md"
      skill_md.write_text("""---
name: [invalid
description: yaml
---

Content
""")

      with pytest.raises(ValueError, match="Invalid YAML"):
        MarkdownSkill.from_directory(skill_path)

  def test_adk_config_extension(self):
    """Test ADK config extension in frontmatter."""
    with tempfile.TemporaryDirectory() as tmpdir:
      skill_path = Path(tmpdir) / "adk-config"
      skill_path.mkdir()

      skill_md = skill_path / "SKILL.md"
      skill_md.write_text("""---
name: adk-config
description: Test ADK config
adk:
  config:
    timeout_seconds: 120
    allow_network: true
    memory_limit_mb: 512
  allowed_callers:
    - custom_caller
---

Content
""")

      skill = MarkdownSkill.from_directory(skill_path)

      assert skill.config.timeout_seconds == 120
      assert skill.config.allow_network is True
      assert skill.config.memory_limit_mb == 512
      assert "custom_caller" in skill.allowed_callers
