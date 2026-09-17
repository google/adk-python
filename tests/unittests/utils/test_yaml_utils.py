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

"""Tests for YAML utility functions."""

from pathlib import Path
from typing import Optional

from google.adk.utils.yaml_utils import dump_pydantic_to_yaml
from google.adk.utils.yaml_utils import load_yaml_file
from google.genai import types
from pydantic import BaseModel
import pytest
import yaml


class SimpleModel(BaseModel):
  """Simple test model."""

  name: str
  age: int
  active: bool
  finish_reason: Optional[types.FinishReason] = None
  multiline_text: Optional[str] = None
  items: Optional[list[str]] = None


def test_yaml_file_generation(tmp_path: Path):
  """Test that YAML file is correctly generated."""
  model = SimpleModel(
      name="Alice",
      age=30,
      active=True,
      finish_reason=types.FinishReason.STOP,
  )
  yaml_file = tmp_path / "test.yaml"

  dump_pydantic_to_yaml(model, yaml_file)

  assert yaml_file.read_text(encoding="utf-8") == """\
active: true
age: 30
finish_reason: STOP
name: Alice
"""


def test_multiline_string_pipe_style(tmp_path: Path):
  """Test that multiline strings use | style."""
  multiline_text = """\
This is a long description
that spans multiple lines
and should be formatted with pipe style"""
  model = SimpleModel(
      name="Test",
      age=25,
      active=False,
      multiline_text=multiline_text,
  )
  yaml_file = tmp_path / "test.yaml"

  dump_pydantic_to_yaml(model, yaml_file)

  assert yaml_file.read_text(encoding="utf-8") == """\
active: false
age: 25
multiline_text: |-
  This is a long description
  that spans multiple lines
  and should be formatted with pipe style
name: Test
"""


def test_list_indentation(tmp_path: Path):
  """Test that lists in mappings are properly indented."""
  model = SimpleModel(
      name="Test",
      age=25,
      active=True,
      items=["item1", "item2", "item3"],
  )
  yaml_file = tmp_path / "test.yaml"

  dump_pydantic_to_yaml(model, yaml_file)

  expected = """\
active: true
age: 25
items:
  - item1
  - item2
  - item3
name: Test
"""
  assert yaml_file.read_text(encoding="utf-8") == expected


def test_empty_list_formatting(tmp_path: Path):
  """Test that empty lists are formatted properly."""
  model = SimpleModel(
      name="Test",
      age=25,
      active=True,
      items=[],
  )
  yaml_file = tmp_path / "test.yaml"

  dump_pydantic_to_yaml(model, yaml_file)

  expected = """\
active: true
age: 25
items: []
name: Test
"""
  assert yaml_file.read_text(encoding="utf-8") == expected


def test_non_ascii_character_preservation(tmp_path: Path):
  """Test that non-ASCII characters are preserved in YAML output."""
  model = SimpleModel(
      name="你好世界",  # Chinese
      age=30,
      active=True,
      multiline_text="🌍 Hello World 🌏\nこんにちは世界\nHola Mundo 🌎",
      items=["Château", "naïve", "café", "🎉"],
  )
  yaml_file = tmp_path / "test.yaml"

  dump_pydantic_to_yaml(model, yaml_file)

  assert yaml_file.read_text(encoding="utf-8") == """\
active: true
age: 30
items:
  - Château
  - naïve
  - café
  - 🎉
multiline_text: |-
  🌍 Hello World 🌏
  こんにちは世界
  Hola Mundo 🌎
name: 你好世界
"""


def test_load_yaml_file_missing_file_raises_file_not_found(tmp_path: Path):
  missing = tmp_path / "absent.yaml"
  with pytest.raises(FileNotFoundError, match=str(missing)):
    load_yaml_file(missing)


def test_load_yaml_file_directory_raises_file_not_found(tmp_path: Path):
  """A directory is not a loadable config, even though the path exists."""
  with pytest.raises(FileNotFoundError):
    load_yaml_file(tmp_path)


def test_load_yaml_file_parses_scalars_with_their_yaml_types(tmp_path: Path):
  yaml_file = tmp_path / "config.yaml"
  yaml_file.write_text(
      "name: agent\nage: 30\nactive: true\nmissing: null\n", encoding="utf-8"
  )

  loaded = load_yaml_file(yaml_file)

  assert loaded == {
      "name": "agent",
      "age": 30,
      "active": True,
      "missing": None,
  }


def test_load_yaml_file_parses_nested_structures(tmp_path: Path):
  yaml_file = tmp_path / "config.yaml"
  yaml_file.write_text(
      "agent:\n  name: root\n  tools:\n    - one\n    - two\n",
      encoding="utf-8",
  )

  assert load_yaml_file(yaml_file) == {
      "agent": {"name": "root", "tools": ["one", "two"]}
  }


def test_load_yaml_file_accepts_a_string_path(tmp_path: Path):
  yaml_file = tmp_path / "config.yaml"
  yaml_file.write_text("name: agent\n", encoding="utf-8")

  assert load_yaml_file(str(yaml_file)) == {"name": "agent"}


def test_load_yaml_file_empty_file_returns_none(tmp_path: Path):
  """An empty config parses to None, not to an empty dict."""
  yaml_file = tmp_path / "config.yaml"
  yaml_file.write_text("", encoding="utf-8")

  assert load_yaml_file(yaml_file) is None


def test_load_yaml_file_decodes_utf8(tmp_path: Path):
  yaml_file = tmp_path / "config.yaml"
  yaml_file.write_text("name: 你好世界\n", encoding="utf-8")

  assert load_yaml_file(yaml_file) == {"name": "你好世界"}


def test_load_yaml_file_refuses_arbitrary_python_tags(tmp_path: Path):
  """Config files are untrusted input, so object construction must not run."""
  yaml_file = tmp_path / "config.yaml"
  yaml_file.write_text(
      "value: !!python/object/apply:os.getcwd []\n", encoding="utf-8"
  )

  with pytest.raises(yaml.YAMLError):
    load_yaml_file(yaml_file)
