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

"""Tests for locating the ADK source folder and loading its config schema."""

from __future__ import annotations

import json
from pathlib import Path

from google.adk.cli.built_in_agents.utils import adk_source_utils
from google.adk.cli.built_in_agents.utils.adk_source_utils import clear_schema_cache
from google.adk.cli.built_in_agents.utils.adk_source_utils import find_adk_source_folder
from google.adk.cli.built_in_agents.utils.adk_source_utils import get_adk_schema_path
from google.adk.cli.built_in_agents.utils.adk_source_utils import load_agent_config_schema
import pytest

_SCHEMA_RELPATH = 'agents/config_schemas/AgentConfig.json'


@pytest.fixture(autouse=True)
def _isolated_schema_cache():
  """Keeps the module-level schema cache from leaking across tests."""
  clear_schema_cache()
  yield
  clear_schema_cache()


def _make_adk_source(root: Path, layout: str = 'src/google/adk') -> Path:
  """Creates a directory that looks like an ADK source tree."""
  adk_dir = root / layout
  schema_path = adk_dir / _SCHEMA_RELPATH
  schema_path.parent.mkdir(parents=True, exist_ok=True)
  schema_path.write_text('{}', encoding='utf-8')
  return adk_dir


def test_find_adk_source_folder_finds_src_layout_from_a_nested_start_dir(
    tmp_path,
):
  adk_dir = _make_adk_source(tmp_path)
  nested = tmp_path / 'deep' / 'nested' / 'cwd'
  nested.mkdir(parents=True)

  assert find_adk_source_folder(str(nested)) == str(adk_dir)


def test_find_adk_source_folder_finds_flat_layout_without_a_src_dir(tmp_path):
  adk_dir = _make_adk_source(tmp_path, layout='google/adk')

  assert find_adk_source_folder(str(tmp_path)) == str(adk_dir)


def test_find_adk_source_folder_returns_none_when_marker_schema_is_missing(
    tmp_path,
):
  # Right directory shape, but no AgentConfig.json: not an ADK source tree.
  (tmp_path / 'src' / 'google' / 'adk' / 'agents').mkdir(parents=True)

  assert find_adk_source_folder(str(tmp_path)) is None


def test_find_adk_source_folder_returns_the_nearest_ancestor_match(tmp_path):
  outer = _make_adk_source(tmp_path)
  inner_root = tmp_path / 'vendored'
  inner = _make_adk_source(inner_root)
  start = inner_root / 'scripts'
  start.mkdir()

  found = find_adk_source_folder(str(start))

  assert found == str(inner)
  assert found != str(outer)


def test_get_adk_schema_path_points_at_the_config_schema_file(tmp_path):
  adk_dir = _make_adk_source(tmp_path)

  assert get_adk_schema_path(str(tmp_path)) == str(adk_dir / _SCHEMA_RELPATH)


def test_get_adk_schema_path_returns_none_when_no_adk_source_above_start(
    tmp_path,
):
  empty = tmp_path / 'empty'
  empty.mkdir()

  assert get_adk_schema_path(str(empty)) is None


def _point_loader_at(monkeypatch, schema_path: Path) -> None:
  monkeypatch.setattr(
      adk_source_utils,
      'get_adk_schema_path',
      lambda *args, **kwargs: str(schema_path),
  )


def test_load_agent_config_schema_returns_the_parsed_dict_by_default(
    tmp_path, monkeypatch
):
  schema_path = tmp_path / 'AgentConfig.json'
  schema_path.write_text('{"title": "AgentConfig"}', encoding='utf-8')
  _point_loader_at(monkeypatch, schema_path)

  assert load_agent_config_schema() == {'title': 'AgentConfig'}


def test_load_agent_config_schema_caches_the_file_until_cache_is_cleared(
    tmp_path, monkeypatch
):
  schema_path = tmp_path / 'AgentConfig.json'
  schema_path.write_text('{"title": "first"}', encoding='utf-8')
  _point_loader_at(monkeypatch, schema_path)

  first = load_agent_config_schema()
  schema_path.write_text('{"title": "second"}', encoding='utf-8')

  assert load_agent_config_schema() == first == {'title': 'first'}

  clear_schema_cache()

  assert load_agent_config_schema() == {'title': 'second'}


def test_load_agent_config_schema_raw_format_returns_indented_json(
    tmp_path, monkeypatch
):
  schema = {'title': 'AgentConfig', 'properties': {'name': {'type': 'string'}}}
  schema_path = tmp_path / 'AgentConfig.json'
  schema_path.write_text(json.dumps(schema), encoding='utf-8')
  _point_loader_at(monkeypatch, schema_path)

  raw = load_agent_config_schema(raw_format=True)

  assert isinstance(raw, str)
  assert json.loads(raw) == schema
  assert '\n  "title": "AgentConfig"' in raw


def test_load_agent_config_schema_escaped_braces_survive_str_format(
    tmp_path, monkeypatch
):
  schema = {'title': 'AgentConfig', 'properties': {'name': {'type': 'string'}}}
  schema_path = tmp_path / 'AgentConfig.json'
  schema_path.write_text(json.dumps(schema), encoding='utf-8')
  _point_loader_at(monkeypatch, schema_path)

  raw = load_agent_config_schema(raw_format=True)
  escaped = load_agent_config_schema(raw_format=True, escape_braces=True)

  # The point of escaping is that the result can be embedded in a prompt
  # template and survive str.format() with its braces intact.
  assert escaped != raw
  assert escaped.format() == raw


def test_load_agent_config_schema_ignores_escape_braces_for_dict_output(
    tmp_path, monkeypatch
):
  schema_path = tmp_path / 'AgentConfig.json'
  schema_path.write_text('{"title": "AgentConfig"}', encoding='utf-8')
  _point_loader_at(monkeypatch, schema_path)

  assert load_agent_config_schema(escape_braces=True) == {
      'title': 'AgentConfig'
  }


def test_load_agent_config_schema_raises_when_the_schema_is_not_found(
    monkeypatch,
):
  monkeypatch.setattr(
      adk_source_utils, 'get_adk_schema_path', lambda *args, **kwargs: None
  )

  with pytest.raises(FileNotFoundError, match='AgentConfig.json schema'):
    load_agent_config_schema()
