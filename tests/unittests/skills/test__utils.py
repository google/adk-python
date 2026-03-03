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
from google.adk.skills._utils import _resolve_skill_dir
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


# ---- _resolve_skill_dir / relative_to tests ----

_SKILL_MD = """\
---
name: {name}
description: A skill
---
Body
"""


def _make_skill(parent: "pathlib.Path", name: str = "my-skill"):
  """Helper to create a minimal skill directory."""
  import pathlib  # noqa: F811 – local re-import for helper

  skill_dir = parent / name
  skill_dir.mkdir(parents=True, exist_ok=True)
  (skill_dir / "SKILL.md").write_text(_SKILL_MD.format(name=name))
  return skill_dir


def test_resolve_skill_dir_helper(tmp_path):
  """Unit test _resolve_skill_dir directly."""
  import pathlib

  # Absolute path is unchanged regardless of relative_to.
  abs_path = tmp_path / "my-skill"
  result = _resolve_skill_dir(abs_path, relative_to="/some/file.py")
  assert result == abs_path.resolve()

  # Relative path + relative_to resolves against parent of relative_to.
  ref_file = tmp_path / "agent.py"
  ref_file.touch()
  result = _resolve_skill_dir("skills/my-skill", relative_to=ref_file)
  assert result == (tmp_path / "skills" / "my-skill").resolve()

  # Relative path + no relative_to resolves against CWD.
  result = _resolve_skill_dir("some-dir")
  assert result == pathlib.Path("some-dir").resolve()


def test_load_skill_with_relative_to(tmp_path):
  """load_skill_from_dir resolves a relative path via relative_to."""
  _make_skill(tmp_path / "skills", "my-skill")

  # Simulate a __file__ sitting in tmp_path.
  ref_file = tmp_path / "agent.py"
  ref_file.touch()

  skill = _load_skill_from_dir("skills/my-skill", relative_to=ref_file)
  assert skill.name == "my-skill"


def test_load_skill_relative_to_none_uses_cwd(tmp_path):
  """When relative_to is None, CWD behaviour is preserved."""
  _make_skill(tmp_path, "my-skill")

  # Passing the absolute tmp_path still works with relative_to=None.
  skill = _load_skill_from_dir(tmp_path / "my-skill")
  assert skill.name == "my-skill"


def test_load_skill_absolute_path_ignores_relative_to(tmp_path):
  """Absolute skill_dir ignores relative_to entirely."""
  skill_dir = _make_skill(tmp_path, "my-skill")

  skill = _load_skill_from_dir(skill_dir, relative_to="/nonexistent/agent.py")
  assert skill.name == "my-skill"


def test_validate_skill_dir_with_relative_to(tmp_path):
  """_validate_skill_dir works with relative_to."""
  _make_skill(tmp_path / "skills", "my-skill")

  ref_file = tmp_path / "agent.py"
  ref_file.touch()

  problems = _validate_skill_dir("skills/my-skill", relative_to=ref_file)
  assert problems == []


def test_read_skill_properties_with_relative_to(tmp_path):
  """_read_skill_properties works with relative_to."""
  _make_skill(tmp_path / "skills", "my-skill")

  ref_file = tmp_path / "agent.py"
  ref_file.touch()

  fm = _read_skill_properties("skills/my-skill", relative_to=ref_file)
  assert fm.name == "my-skill"


def test_resolve_skill_dir_with_directory_relative_to(tmp_path):
  """relative_to accepts a directory and uses it directly as anchor."""
  _make_skill(tmp_path / "skills", "my-skill")

  # When relative_to is a directory, it should be used as-is (no .parent).
  result = _resolve_skill_dir("skills/my-skill", relative_to=tmp_path)
  assert result == (tmp_path / "skills" / "my-skill").resolve()

  # End-to-end: load_skill_from_dir with a directory-valued relative_to.
  skill = _load_skill_from_dir("skills/my-skill", relative_to=tmp_path)
  assert skill.name == "my-skill"
