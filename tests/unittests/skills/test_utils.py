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

"""Unit tests for skill utilities."""

from google.adk.skills import load_skill_from_dir
import pytest


def test_load_skill_from_dir(tmp_path):
  """Tests loading a skill from a directory."""
  skill_dir = tmp_path / "test-skill"
  skill_dir.mkdir()

  skill_md_content = """---
name: test-skill
description: Test description
---
Test instructions
"""
  (skill_dir / "SKILL.md").write_text(skill_md_content)

  # Create references
  ref_dir = skill_dir / "references"
  ref_dir.mkdir()
  (ref_dir / "ref1.md").write_text("ref1 content")

  # Create assets
  assets_dir = skill_dir / "assets"
  assets_dir.mkdir()
  (assets_dir / "asset1.txt").write_text("asset1 content")

  # Create scripts
  scripts_dir = skill_dir / "scripts"
  scripts_dir.mkdir()
  (scripts_dir / "script1.sh").write_text("echo hello")

  skill = load_skill_from_dir(skill_dir)

  assert skill.name == "test-skill"
  assert skill.description == "Test description"
  assert skill.instructions == "Test instructions"
  assert skill.resources.get_reference("ref1.md") == "ref1 content"
  assert skill.resources.get_asset("asset1.txt") == "asset1 content"
  assert skill.resources.get_script("script1.sh").src == "echo hello"
