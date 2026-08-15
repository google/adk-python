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

"""Tests for the unused-file scanner used by Agent Builder."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from google.adk.cli.built_in_agents.tools.cleanup_unused_files import cleanup_unused_files


def _tool_context(root: Path) -> mock.MagicMock:
  tool_context = mock.MagicMock()
  tool_context.state = {"root_directory": str(root)}
  return tool_context


def _populate(root: Path, names: list[str]) -> None:
  for name in names:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")


def _unused(result, root: Path) -> list[str]:
  return sorted(
      str(Path(p).relative_to(root.resolve())) for p in result["unused_files"]
  )


async def test_cleanup_unused_files_reports_python_files_not_in_use(tmp_path):
  _populate(tmp_path, ["used.py", "orphan.py", "pkg/nested_orphan.py"])

  result = await cleanup_unused_files(
      used_files=["used.py"], tool_context=_tool_context(tmp_path)
  )

  assert result["success"]
  assert result["errors"] == []
  assert _unused(result, tmp_path) == ["orphan.py", "pkg/nested_orphan.py"]
  # Identification only; nothing is removed by this tool.
  assert result["deleted_files"] == []
  assert result["total_freed_space"] == 0


async def test_cleanup_unused_files_applies_the_default_exclusions(tmp_path):
  _populate(
      tmp_path,
      [
          "orphan.py",
          "__init__.py",
          "widget_test.py",
          "test_widget.py",
          "notes.txt",
      ],
  )

  result = await cleanup_unused_files(
      used_files=[], tool_context=_tool_context(tmp_path)
  )

  # Package markers, both test-file conventions, and non-Python files are
  # never reported as orphans.
  assert _unused(result, tmp_path) == ["orphan.py"]


async def test_cleanup_unused_files_honors_custom_patterns(tmp_path):
  _populate(tmp_path, ["a.yaml", "b.yaml", "keep.py", "__init__.py"])

  result = await cleanup_unused_files(
      used_files=["a.yaml"],
      tool_context=_tool_context(tmp_path),
      file_patterns=["*.yaml"],
      exclude_patterns=[],
  )

  assert _unused(result, tmp_path) == ["b.yaml"]


async def test_cleanup_unused_files_matches_used_files_after_resolution(
    tmp_path,
):
  """A used file written a different way is still recognised as used."""
  _populate(tmp_path, ["pkg/tool.py"])

  result = await cleanup_unused_files(
      used_files=["./pkg/../pkg/tool.py"], tool_context=_tool_context(tmp_path)
  )

  assert result["success"]
  assert result["unused_files"] == []


async def test_cleanup_unused_files_reports_a_missing_root_directory(tmp_path):
  missing = tmp_path / "does_not_exist"

  result = await cleanup_unused_files(
      used_files=[], tool_context=_tool_context(missing)
  )

  assert not result["success"]
  assert len(result["errors"]) == 1
  assert "Root directory does not exist" in result["errors"][0]
  assert result["unused_files"] == []


async def test_cleanup_unused_files_fails_closed_on_a_used_file_escape(
    tmp_path,
):
  """A used_files entry outside the root aborts the scan instead of listing

  everything under the root as unused.
  """
  _populate(tmp_path, ["orphan.py"])

  result = await cleanup_unused_files(
      used_files=["../outside.py"], tool_context=_tool_context(tmp_path)
  )

  assert not result["success"]
  assert result["unused_files"] == []
  assert len(result["errors"]) == 1
  assert result["errors"][0].startswith("Cleanup scan failed:")
