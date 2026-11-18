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

"""Tests for the ADK source search tool used by Agent Builder."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Iterator
from unittest import mock

from google.adk.cli.built_in_agents.tools.search_adk_source import search_adk_source
import pytest

# The tools package re-exports the function under the module's own name, so the
# module object has to be looked up directly to stub its source-folder lookup.
_module = importlib.import_module(
    "google.adk.cli.built_in_agents.tools.search_adk_source"
)


@pytest.fixture
def source_tree(tmp_path: Path) -> Iterator[Path]:
  """Point the tool at a throwaway source tree with one file beside it."""
  source = tmp_path / "src"
  source.mkdir()
  (source / "agent.py").write_text("class FunctionTool:\n")
  (tmp_path / "private.py").write_text("hidden value\n")
  with mock.patch.object(
      _module, "find_adk_source_folder", autospec=True, return_value=str(source)
  ):
    yield source


@pytest.mark.usefixtures("source_tree")
async def test_search_adk_source_returns_matches_from_the_source_tree():
  result = await search_adk_source("class FunctionTool")

  assert result["success"]
  assert result["errors"] == []
  assert result["total_matches"] == 1
  assert result["results"][0]["file_path"] == "agent.py"


@pytest.mark.usefixtures("source_tree")
async def test_search_adk_source_rejects_a_pattern_leaving_the_source_tree():
  """A glob pattern cannot read files outside the ADK source directory."""
  result = await search_adk_source("hidden", file_patterns=["../*.py"])

  assert result["total_matches"] == 0
  assert result["results"] == []
  assert result["errors"] == [
      "File pattern must stay within the ADK source directory: ../*.py"
  ]


async def test_search_adk_source_searches_files_symlinked_into_the_tree(
    source_tree,
):
  """An install that links each file in from a cache is still searched."""
  cache = source_tree.parent / "cache"
  cache.mkdir()
  (cache / "tool.py").write_text("class LinkedTool:\n")
  (source_tree / "tool.py").symlink_to(cache / "tool.py")

  result = await search_adk_source("class LinkedTool")

  assert result["success"]
  assert result["total_matches"] == 1
  assert result["results"][0]["file_path"] == "tool.py"
