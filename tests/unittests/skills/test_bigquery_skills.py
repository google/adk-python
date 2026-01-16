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

"""Unit tests for BigQuery MarkdownSkill implementations."""

from __future__ import annotations

from pathlib import Path

from google.adk.skills import MarkdownSkill
from google.adk.skills import SKILLS_DIR
import pytest


class TestBigQueryDataManagementSkill:
  """Tests for bigquery-data-management skill."""

  @pytest.fixture
  def skill(self):
    """Load the skill from the skills directory."""
    skill_path = SKILLS_DIR / "bigquery-data-management"
    return MarkdownSkill.from_directory(skill_path)

  def test_skill_loads_successfully(self, skill):
    """Test that the skill loads without errors."""
    assert skill is not None
    assert skill.name == "bigquery-data-management"

  def test_skill_has_description(self, skill):
    """Test that the skill has a proper description."""
    assert "data" in skill.description.lower()
    assert len(skill.description) > 50

  def test_skill_has_metadata(self, skill):
    """Test that skill metadata is properly set."""
    assert skill.skill_metadata.license == "Apache-2.0"
    assert skill.skill_metadata.metadata.get("category") == "data-management"

  def test_skill_has_instructions(self, skill):
    """Test that skill instructions are available."""
    instructions = skill.get_instructions()
    assert len(instructions) > 100
    assert "LOAD DATA" in instructions
    assert "partitioning" in instructions.lower()

  def test_skill_has_references(self, skill):
    """Test that skill has reference documents."""
    refs = skill.list_references()
    assert len(refs) >= 2
    assert "DATA_FORMATS.md" in refs
    assert "PARTITIONING.md" in refs

  def test_skill_has_scripts(self, skill):
    """Test that skill has helper scripts."""
    scripts = skill.list_scripts()
    assert len(scripts) >= 1
    assert "validate_schema.py" in scripts

  def test_reference_content_loads(self, skill):
    """Test that reference content can be loaded."""
    content = skill.get_reference("DATA_FORMATS.md")
    assert content is not None
    assert "Parquet" in content
    assert "CSV" in content

  def test_progressive_disclosure_stages(self, skill):
    """Test progressive disclosure works correctly."""
    # Stage 1: Just loaded
    assert skill.current_stage == 1

    # Stage 2: Get instructions
    skill.get_instructions()
    assert skill.current_stage == 2

    # Stage 3: Access scripts/references
    skill.get_script("validate_schema.py")
    assert skill.current_stage == 3


class TestBigQueryAnalyticsSkill:
  """Tests for bigquery-analytics skill."""

  @pytest.fixture
  def skill(self):
    """Load the skill from the skills directory."""
    skill_path = SKILLS_DIR / "bigquery-analytics"
    return MarkdownSkill.from_directory(skill_path)

  def test_skill_loads_successfully(self, skill):
    """Test that the skill loads without errors."""
    assert skill is not None
    assert skill.name == "bigquery-analytics"

  def test_skill_has_description(self, skill):
    """Test that the skill has a proper description."""
    assert "analytics" in skill.description.lower()
    assert "SQL" in skill.description

  def test_skill_has_instructions(self, skill):
    """Test that skill instructions contain expected content."""
    instructions = skill.get_instructions()
    assert "window function" in instructions.lower()
    assert "aggregation" in instructions.lower()
    assert "geospatial" in instructions.lower()

  def test_skill_has_references(self, skill):
    """Test that skill has reference documents."""
    refs = skill.list_references()
    assert "WINDOW_FUNCTIONS.md" in refs
    assert "GEOSPATIAL.md" in refs

  def test_window_functions_reference(self, skill):
    """Test window functions reference content."""
    content = skill.get_reference("WINDOW_FUNCTIONS.md")
    assert "ROW_NUMBER" in content
    assert "RANK" in content
    assert "LAG" in content

  def test_geospatial_reference(self, skill):
    """Test geospatial reference content."""
    content = skill.get_reference("GEOSPATIAL.md")
    assert "ST_DISTANCE" in content
    assert "ST_CONTAINS" in content


class TestBigQueryStorageSkill:
  """Tests for bigquery-storage skill."""

  @pytest.fixture
  def skill(self):
    """Load the skill from the skills directory."""
    skill_path = SKILLS_DIR / "bigquery-storage"
    return MarkdownSkill.from_directory(skill_path)

  def test_skill_loads_successfully(self, skill):
    """Test that the skill loads without errors."""
    assert skill is not None
    assert skill.name == "bigquery-storage"

  def test_skill_has_description(self, skill):
    """Test that the skill has a proper description."""
    assert "storage" in skill.description.lower()

  def test_skill_has_instructions(self, skill):
    """Test that skill instructions contain expected content."""
    instructions = skill.get_instructions()
    assert "CREATE TABLE" in instructions
    assert "schema" in instructions.lower()
    assert "time travel" in instructions.lower()

  def test_skill_has_scripts(self, skill):
    """Test that skill has helper scripts."""
    scripts = skill.list_scripts()
    assert "storage_report.py" in scripts


class TestBigQueryGovernanceSkill:
  """Tests for bigquery-governance skill."""

  @pytest.fixture
  def skill(self):
    """Load the skill from the skills directory."""
    skill_path = SKILLS_DIR / "bigquery-governance"
    return MarkdownSkill.from_directory(skill_path)

  def test_skill_loads_successfully(self, skill):
    """Test that the skill loads without errors."""
    assert skill is not None
    assert skill.name == "bigquery-governance"

  def test_skill_has_description(self, skill):
    """Test that the skill has a proper description."""
    assert "governance" in skill.description.lower()
    assert (
        "access" in skill.description.lower()
        or "security" in skill.description.lower()
    )

  def test_skill_has_instructions(self, skill):
    """Test that skill instructions contain expected content."""
    instructions = skill.get_instructions()
    assert "IAM" in instructions
    assert "row" in instructions.lower() and "security" in instructions.lower()
    assert "masking" in instructions.lower()

  def test_skill_has_scripts(self, skill):
    """Test that skill has helper scripts."""
    scripts = skill.list_scripts()
    assert "audit_report.py" in scripts


class TestBigQueryAdminSkill:
  """Tests for bigquery-admin skill."""

  @pytest.fixture
  def skill(self):
    """Load the skill from the skills directory."""
    skill_path = SKILLS_DIR / "bigquery-admin"
    return MarkdownSkill.from_directory(skill_path)

  def test_skill_loads_successfully(self, skill):
    """Test that the skill loads without errors."""
    assert skill is not None
    assert skill.name == "bigquery-admin"

  def test_skill_has_description(self, skill):
    """Test that the skill has a proper description."""
    assert "admin" in skill.description.lower()

  def test_skill_has_instructions(self, skill):
    """Test that skill instructions contain expected content."""
    instructions = skill.get_instructions()
    assert "reservation" in instructions.lower()
    assert "BI Engine" in instructions
    assert "monitoring" in instructions.lower()

  def test_skill_has_scripts(self, skill):
    """Test that skill has helper scripts."""
    scripts = skill.list_scripts()
    assert "cost_report.py" in scripts


class TestBigQueryIntegrationSkill:
  """Tests for bigquery-integration skill."""

  @pytest.fixture
  def skill(self):
    """Load the skill from the skills directory."""
    skill_path = SKILLS_DIR / "bigquery-integration"
    return MarkdownSkill.from_directory(skill_path)

  def test_skill_loads_successfully(self, skill):
    """Test that the skill loads without errors."""
    assert skill is not None
    assert skill.name == "bigquery-integration"

  def test_skill_has_description(self, skill):
    """Test that the skill has a proper description."""
    assert "integrate" in skill.description.lower()

  def test_skill_has_instructions(self, skill):
    """Test that skill instructions contain expected content."""
    instructions = skill.get_instructions()
    assert "Python" in instructions
    assert "JDBC" in instructions
    assert "REST API" in instructions

  def test_skill_has_scripts(self, skill):
    """Test that skill has helper scripts."""
    scripts = skill.list_scripts()
    assert "connection_test.py" in scripts


class TestBigQuerySkillsIntegration:
  """Integration tests for all BigQuery skills."""

  @pytest.fixture
  def skill_names(self):
    """List of all BigQuery skill directory names."""
    return [
        "bigquery-data-management",
        "bigquery-analytics",
        "bigquery-storage",
        "bigquery-governance",
        "bigquery-admin",
        "bigquery-integration",
        "bqml",
        "bigquery-ai",
    ]

  def test_all_skills_exist(self, skill_names):
    """Test that all expected skill directories exist."""
    for name in skill_names:
      skill_path = SKILLS_DIR / name
      assert skill_path.exists(), f"Skill directory {name} not found"
      assert (skill_path / "SKILL.md").exists(), f"SKILL.md not found in {name}"

  def test_all_skills_load(self, skill_names):
    """Test that all skills can be loaded."""
    for name in skill_names:
      skill_path = SKILLS_DIR / name
      skill = MarkdownSkill.from_directory(skill_path)
      assert skill.name == name

  def test_all_skills_have_unique_names(self, skill_names):
    """Test that all skills have unique names."""
    loaded_names = []
    for name in skill_names:
      skill_path = SKILLS_DIR / name
      skill = MarkdownSkill.from_directory(skill_path)
      loaded_names.append(skill.name)

    assert len(loaded_names) == len(
        set(loaded_names)
    ), "Duplicate skill names found"

  def test_all_skills_have_valid_metadata(self, skill_names):
    """Test that all skills have valid metadata."""
    for name in skill_names:
      skill_path = SKILLS_DIR / name
      skill = MarkdownSkill.from_directory(skill_path)

      assert skill.name, f"Skill {name} missing name"
      assert skill.description, f"Skill {name} missing description"
      assert len(skill.description) >= 50, f"Skill {name} description too short"

  def test_skill_prompt_generation(self, skill_names):
    """Test that all skills can generate prompts."""
    for name in skill_names:
      skill_path = SKILLS_DIR / name
      skill = MarkdownSkill.from_directory(skill_path)
      prompt = skill.get_skill_prompt()

      assert skill.name in prompt
      assert len(prompt) > 50


class TestSkillToolDeclarations:
  """Tests for skill tool declarations."""

  def test_data_management_tool_declarations(self):
    """Test data management skill tool declarations."""
    skill = MarkdownSkill.from_directory(
        SKILLS_DIR / "bigquery-data-management"
    )
    declarations = skill.get_tool_declarations()

    # Should have auto-generated declarations from scripts
    tool_names = [d["name"] for d in declarations]
    assert any("validate_schema" in name for name in tool_names)

  def test_analytics_tool_declarations(self):
    """Test analytics skill tool declarations."""
    skill = MarkdownSkill.from_directory(SKILLS_DIR / "bigquery-analytics")
    declarations = skill.get_tool_declarations()

    tool_names = [d["name"] for d in declarations]
    assert any("query_analyzer" in name for name in tool_names)

  def test_declarations_have_descriptions(self):
    """Test that all tool declarations have descriptions."""
    skill_names = [
        "bigquery-data-management",
        "bigquery-analytics",
        "bigquery-admin",
        "bigquery-governance",
        "bigquery-integration",
    ]

    for name in skill_names:
      skill = MarkdownSkill.from_directory(SKILLS_DIR / name)
      declarations = skill.get_tool_declarations()

      for decl in declarations:
        assert "name" in decl, f"Tool in {name} missing name"
        assert (
            "description" in decl
        ), f"Tool {decl.get('name')} in {name} missing description"
