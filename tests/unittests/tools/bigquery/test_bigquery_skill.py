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

"""Tests for the pre-packaged BigQuery skill."""

from __future__ import annotations

import re

from google.adk.skills._utils import _validate_skill_dir
from google.adk.tools.bigquery.bigquery_skill import _SKILL_DIR
from google.adk.tools.bigquery.bigquery_skill import get_bigquery_skill
from google.adk.tools.skill_toolset import ListSkillsTool
from google.adk.tools.skill_toolset import LoadSkillResourceTool
from google.adk.tools.skill_toolset import LoadSkillTool
from google.adk.tools.skill_toolset import RunSkillScriptTool
from google.adk.tools.skill_toolset import SkillToolset
import pytest


def test_get_bigquery_skill_returns_valid_skill():
  """Verify get_bigquery_skill returns a Skill with expected fields."""
  skill = get_bigquery_skill()

  assert skill.name == "bigquery-ai-ml"
  assert skill.description
  assert len(skill.description) > 0
  assert skill.instructions
  assert len(skill.instructions) > 0


def test_skill_name_matches_spec():
  """Verify skill name is kebab-case and matches directory name."""
  skill = get_bigquery_skill()

  # Name must be kebab-case
  assert re.fullmatch(r"[a-z][a-z0-9]*(-[a-z0-9]+)*", skill.name)

  # Name must match the directory name
  assert skill.name == _SKILL_DIR.name


def test_skill_has_expected_references():
  """Verify all expected reference files are present and non-empty."""
  skill = get_bigquery_skill()

  expected_refs = {
      "bigquery_ai_classify.md",
      "bigquery_ai_detect_anomalies.md",
      "bigquery_ai_forecast.md",
      "bigquery_ai_generate.md",
      "bigquery_ai_generate_bool.md",
      "bigquery_ai_generate_double.md",
      "bigquery_ai_generate_int.md",
      "bigquery_ai_if.md",
      "bigquery_ai_score.md",
      "bigquery_ai_search.md",
      "bigquery_ai_similarity.md",
  }
  actual_refs = set(skill.resources.list_references())

  assert expected_refs == actual_refs

  for ref_name in expected_refs:
    content = skill.resources.get_reference(ref_name)
    assert content is not None, f"Reference {ref_name} returned None"
    assert len(content) > 0, f"Reference {ref_name} is empty"


@pytest.mark.asyncio
async def test_skill_works_with_skill_toolset():
  """Verify the skill integrates with SkillToolset and produces 4 tools."""
  skill = get_bigquery_skill()
  toolset = SkillToolset(skills=[skill])

  tools = await toolset.get_tools()
  assert len(tools) == 4

  tool_types = {type(t) for t in tools}
  expected_types = {
      ListSkillsTool,
      LoadSkillTool,
      LoadSkillResourceTool,
      RunSkillScriptTool,
  }
  assert tool_types == expected_types


def test_skill_passes_validation():
  """Verify the skill directory passes ADK's built-in validator."""
  problems = _validate_skill_dir(_SKILL_DIR)
  assert not problems, f"Validation problems: {problems}"


def test_skill_frontmatter_has_license():
  """Verify the skill includes a license field."""
  skill = get_bigquery_skill()
  assert skill.frontmatter.license == "Apache-2.0"


def test_skill_frontmatter_has_metadata():
  """Verify the skill includes author and version metadata."""
  skill = get_bigquery_skill()
  assert "author" in skill.frontmatter.metadata
  assert "version" in skill.frontmatter.metadata
