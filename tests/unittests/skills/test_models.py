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

"""Unit tests for skill models."""

from google.adk.skills import models
from pydantic import ValidationError
import pytest


def test_frontmatter():
  """Tests Frontmatter model."""
  frontmatter = models.Frontmatter(
      name="test-skill",
      description="Test description",
      license="Apache 2.0",
      compatibility="test",
      allowed_tools="test",
      metadata={"key": "value"},
  )
  assert frontmatter.name == "test-skill"
  assert frontmatter.description == "Test description"
  assert frontmatter.license == "Apache 2.0"
  assert frontmatter.compatibility == "test"
  assert frontmatter.allowed_tools == "test"
  assert frontmatter.metadata == {"key": "value"}


def test_resources():
  """Tests Resources model."""
  resources = models.Resources(
      references={"ref1": "ref content"},
      assets={"asset1": "asset content"},
      scripts={"script1": models.Script(src="print('hello')")},
  )
  assert resources.get_reference("ref1") == "ref content"
  assert resources.get_asset("asset1") == "asset content"
  assert resources.get_script("script1").src == "print('hello')"
  assert resources.get_reference("ref2") is None
  assert resources.get_asset("asset2") is None
  assert resources.get_script("script2") is None
  assert resources.list_references() == ["ref1"]
  assert resources.list_assets() == ["asset1"]
  assert resources.list_scripts() == ["script1"]


def test_skill_properties():
  """Tests Skill model."""
  frontmatter = models.Frontmatter(
      name="my-skill", description="my description"
  )
  skill = models.Skill(frontmatter=frontmatter, instructions="do this")
  assert skill.name == "my-skill"
  assert skill.description == "my description"


def test_script_to_string():
  """Tests Script model."""
  script = models.Script(src="print('hello')")
  assert str(script) == "print('hello')"


# --- Name validation tests ---


def test_name_too_long():
  with pytest.raises(ValidationError, match="at most 64 characters"):
    models.Frontmatter(name="a" * 65, description="desc")


def test_name_uppercase_rejected():
  with pytest.raises(ValidationError, match="lowercase kebab-case"):
    models.Frontmatter(name="My-Skill", description="desc")


def test_name_leading_hyphen():
  with pytest.raises(ValidationError, match="lowercase kebab-case"):
    models.Frontmatter(name="-my-skill", description="desc")


def test_name_trailing_hyphen():
  with pytest.raises(ValidationError, match="lowercase kebab-case"):
    models.Frontmatter(name="my-skill-", description="desc")


def test_name_consecutive_hyphens():
  with pytest.raises(ValidationError, match="lowercase kebab-case"):
    models.Frontmatter(name="my--skill", description="desc")


def test_name_invalid_chars_underscore():
  with pytest.raises(ValidationError, match="lowercase kebab-case"):
    models.Frontmatter(name="my_skill", description="desc")


def test_name_invalid_chars_ampersand():
  with pytest.raises(ValidationError, match="lowercase kebab-case"):
    models.Frontmatter(name="skill&name", description="desc")


def test_name_valid_passes():
  fm = models.Frontmatter(name="my-skill-2", description="desc")
  assert fm.name == "my-skill-2"


def test_name_single_word():
  fm = models.Frontmatter(name="skill", description="desc")
  assert fm.name == "skill"


# --- Description validation tests ---


def test_description_empty():
  with pytest.raises(ValidationError, match="must not be empty"):
    models.Frontmatter(name="my-skill", description="")


def test_description_too_long():
  with pytest.raises(ValidationError, match="at most 1024 characters"):
    models.Frontmatter(name="my-skill", description="x" * 1025)


# --- Compatibility validation tests ---


def test_compatibility_too_long():
  with pytest.raises(ValidationError, match="at most 500 characters"):
    models.Frontmatter(
        name="my-skill", description="desc", compatibility="c" * 501
    )


# --- Extra field rejected ---


def test_extra_field_allowed():
  fm = models.Frontmatter.model_validate({
      "name": "my-skill",
      "description": "desc",
      "unknown_field": "value",
  })
  assert fm.name == "my-skill"


# --- allowed-tools alias ---


def test_allowed_tools_alias_via_model_validate():
  fm = models.Frontmatter.model_validate({
      "name": "my-skill",
      "description": "desc",
      "allowed-tools": "tool-pattern",
  })
  assert fm.allowed_tools == "tool-pattern"


def test_allowed_tools_serialization_alias():
  fm = models.Frontmatter(
      name="my-skill", description="desc", allowed_tools="tool-pattern"
  )
  dumped = fm.model_dump(by_alias=True)
  assert "allowed-tools" in dumped
  assert dumped["allowed-tools"] == "tool-pattern"
