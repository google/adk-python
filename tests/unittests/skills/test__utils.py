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

from unittest import mock

from google.adk.skills import list_skills_in_dir
from google.adk.skills import list_skills_in_gcs_dir as _list_skills_in_gcs_dir
from google.adk.skills import load_skill_from_dir as _load_skill_from_dir
from google.adk.skills import load_skill_from_gcs_dir as _load_skill_from_gcs_dir
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


@mock.patch("google.cloud.storage.Client")
def test__list_skills_in_gcs_dir(mock_client_class):

  mock_client = mock.MagicMock()
  mock_client_class.return_value = mock_client
  mock_bucket = mock.MagicMock()
  mock_client.bucket.return_value = mock_bucket

  mock_iterator = mock.MagicMock()
  mock_iterator.prefixes = ["skills/my-skill/"]
  mock_bucket.list_blobs.return_value = mock_iterator

  mock_blob = mock.MagicMock()
  mock_blob.exists.return_value = True
  mock_blob.download_as_text.return_value = (
      "---\nname: my-skill\ndescription: A skill\n---\nBody"
  )
  mock_bucket.blob.return_value = mock_blob

  skills = _list_skills_in_gcs_dir("my-bucket", "skills/")
  assert "my-skill" in skills
  assert skills["my-skill"].name == "my-skill"


@mock.patch("google.cloud.storage.Client")
@mock.patch("logging.warning")
def test__list_skills_in_gcs_dir_skips_invalid(
    mock_logging_warning, mock_client_class
):
  mock_client = mock.MagicMock()
  mock_client_class.return_value = mock_client
  mock_bucket = mock.MagicMock()
  mock_client.bucket.return_value = mock_bucket

  mock_iterator = mock.MagicMock()
  mock_iterator.prefixes = ["skills/invalid-skill/", "skills/valid-skill/"]
  mock_bucket.list_blobs.return_value = mock_iterator

  def mock_blob_side_effect(path):
    m = mock.MagicMock()
    m.exists.return_value = True
    if "invalid-skill" in path:
      m.download_as_text.return_value = "invalid yaml content"
    else:
      m.download_as_text.return_value = (
          "---\nname: valid-skill\ndescription: A skill\n---\nBody"
      )
    return m

  mock_bucket.blob.side_effect = mock_blob_side_effect

  skills = _list_skills_in_gcs_dir("my-bucket", "skills/")
  assert "valid-skill" in skills
  assert "invalid-skill" not in skills

  # Verify warning was logged for the invalid skill
  mock_logging_warning.assert_called_once()
  args, _ = mock_logging_warning.call_args
  assert "Skipping invalid skill" in args[0]
  assert args[1] == "invalid-skill"
  assert args[2] == "my-bucket"


@mock.patch("google.cloud.storage.Client")
def test__load_skill_from_gcs_dir(mock_client_class):

  mock_client = mock.MagicMock()
  mock_client_class.return_value = mock_client
  mock_bucket = mock.MagicMock()
  mock_client.bucket.return_value = mock_bucket

  def mock_blob_side_effect(path):
    m = mock.MagicMock()
    if path.endswith("SKILL.md"):
      m.exists.return_value = True
      m.download_as_text.return_value = (
          "---\nname: my-skill\ndescription: Test description\n---\nTest"
          " instructions"
      )
    else:
      m.exists.return_value = False
    return m

  mock_bucket.blob.side_effect = mock_blob_side_effect

  # For resources
  def list_blobs_side_effect(prefix=None):
    if prefix.endswith("references/"):
      m = mock.MagicMock()
      m.name = prefix + "ref1.md"
      m.download_as_text.return_value = "ref1 content"
      return [m]
    return []

  mock_bucket.list_blobs.side_effect = list_blobs_side_effect

  skill = _load_skill_from_gcs_dir("my-bucket", "skills/my-skill/")

  assert skill.name == "my-skill"
  assert skill.description == "Test description"
  assert skill.instructions == "Test instructions"
  # Using dict access for reference
  assert skill.resources.get_reference("ref1.md") == "ref1 content"


def test_list_skills_in_dir(tmp_path):
  """Tests listing skills in a directory."""
  skills_dir = tmp_path / "skills"
  skills_dir.mkdir()

  # Valid skill 1
  skill1_dir = skills_dir / "skill1"
  skill1_dir.mkdir()
  (skill1_dir / "SKILL.md").write_text(
      "---\nname: skill1\ndescription: desc1\n---\nbody"
  )

  # Valid skill 2
  skill2_dir = skills_dir / "skill2"
  skill2_dir.mkdir()
  (skill2_dir / "SKILL.md").write_text(
      "---\nname: skill2\ndescription: desc2\n---\nbody"
  )

  # Invalid skill: missing SKILL.md
  (skills_dir / "invalid-no-md").mkdir()

  # Invalid skill: invalid YAML
  invalid_yaml_dir = skills_dir / "invalid-yaml"
  invalid_yaml_dir.mkdir()
  (invalid_yaml_dir / "SKILL.md").write_text("---\ninvalid: yaml: :\n---\nbody")

  # Invalid skill: name mismatch
  mismatch_dir = skills_dir / "mismatch"
  mismatch_dir.mkdir()
  (mismatch_dir / "SKILL.md").write_text(
      "---\nname: other-name\ndescription: desc\n---\nbody"
  )

  skills = list_skills_in_dir(skills_dir)

  assert len(skills) == 2
  assert "skill1" in skills
  assert "skill2" in skills
  assert skills["skill1"].name == "skill1"
  assert skills["skill2"].name == "skill2"


def test_list_skills_in_dir_missing_base_path(tmp_path):
  """Tests list_skills_in_dir with missing base directory."""

  skills = list_skills_in_dir(tmp_path / "nonexistent")
  assert skills == {}
