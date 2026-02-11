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
