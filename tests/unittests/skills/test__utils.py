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

from google.adk.skills import load_skill_from_dir as _load_skill_from_dir
from google.adk.skills._utils import _read_skill_properties
from google.adk.skills._utils import _validate_skill_dir
import pytest


def test__load_skill_from_dir(tmp_path):
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

  skill = _load_skill_from_dir(skill_dir)

  assert skill.name == "test-skill"
  assert skill.description == "Test description"
  assert skill.instructions == "Test instructions"
  assert skill.resources.get_reference("ref1.md") == "ref1 content"
  assert skill.resources.get_asset("asset1.txt") == "asset1 content"
  assert skill.resources.get_script("script1.sh").src == "echo hello"


def test_allowed_tools_yaml_key(tmp_path):
  """Tests that allowed-tools YAML key loads correctly."""
  skill_dir = tmp_path / "my-skill"
  skill_dir.mkdir()

  skill_md = """---
name: my-skill
description: A skill
allowed-tools: "some-tool-*"
---
Instructions here
"""
  (skill_dir / "SKILL.md").write_text(skill_md)

  skill = _load_skill_from_dir(skill_dir)
  assert skill.frontmatter.allowed_tools == "some-tool-*"


def test_name_directory_mismatch(tmp_path):
  """Tests that name-directory mismatch raises ValueError."""
  skill_dir = tmp_path / "wrong-dir"
  skill_dir.mkdir()

  skill_md = """---
name: my-skill
description: A skill
---
Body
"""
  (skill_dir / "SKILL.md").write_text(skill_md)

  with pytest.raises(ValueError, match="does not match directory"):
    _load_skill_from_dir(skill_dir)


def test_validate_skill_dir_valid(tmp_path):
  """Tests validate_skill_dir with a valid skill."""
  skill_dir = tmp_path / "my-skill"
  skill_dir.mkdir()

  skill_md = """---
name: my-skill
description: A skill
---
Body
"""
  (skill_dir / "SKILL.md").write_text(skill_md)

  problems = _validate_skill_dir(skill_dir)
  assert problems == []


def test_validate_skill_dir_missing_dir(tmp_path):
  """Tests validate_skill_dir with missing directory."""
  problems = _validate_skill_dir(tmp_path / "nonexistent")
  assert len(problems) == 1
  assert "does not exist" in problems[0]


def test_validate_skill_dir_missing_skill_md(tmp_path):
  """Tests validate_skill_dir with missing SKILL.md."""
  skill_dir = tmp_path / "my-skill"
  skill_dir.mkdir()

  problems = _validate_skill_dir(skill_dir)
  assert len(problems) == 1
  assert "SKILL.md not found" in problems[0]


def test_validate_skill_dir_name_mismatch(tmp_path):
  """Tests validate_skill_dir catches name-directory mismatch."""
  skill_dir = tmp_path / "wrong-dir"
  skill_dir.mkdir()

  skill_md = """---
name: my-skill
description: A skill
---
Body
"""
  (skill_dir / "SKILL.md").write_text(skill_md)

  problems = _validate_skill_dir(skill_dir)
  assert any("does not match" in p for p in problems)


def test_validate_skill_dir_unknown_fields(tmp_path):
  """Tests validate_skill_dir detects unknown frontmatter fields."""
  skill_dir = tmp_path / "my-skill"
  skill_dir.mkdir()

  skill_md = """---
name: my-skill
description: A skill
unknown-field: something
---
Body
"""
  (skill_dir / "SKILL.md").write_text(skill_md)

  problems = _validate_skill_dir(skill_dir)
  assert any("Unknown frontmatter" in p for p in problems)


def test__read_skill_properties(tmp_path):
  """Tests read_skill_properties basic usage."""
  skill_dir = tmp_path / "my-skill"
  skill_dir.mkdir()

  skill_md = """---
name: my-skill
description: A cool skill
license: MIT
---
Body content
"""
  (skill_dir / "SKILL.md").write_text(skill_md)

  fm = _read_skill_properties(skill_dir)
  assert fm.name == "my-skill"
  assert fm.description == "A cool skill"
  assert fm.license == "MIT"
