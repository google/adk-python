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

"""Path-traversal containment tests for Agent Builder file tools."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

from google.adk.cli.built_in_agents.tools.delete_files import delete_files
from google.adk.cli.built_in_agents.tools.read_files import read_files
from google.adk.cli.built_in_agents.tools.write_files import write_files
from google.adk.cli.built_in_agents.utils.resolve_root_directory import resolve_file_path
import pytest


def _tool_context(root: Path) -> mock.MagicMock:
  tool_context = mock.MagicMock()
  tool_context._invocation_context.session.state = {"root_directory": str(root)}
  return tool_context


def test_resolve_file_path_allows_path_within_root(tmp_path):
  resolved = resolve_file_path(
      "sub/dir/file.txt", {"root_directory": str(tmp_path)}
  )
  assert resolved == (tmp_path / "sub" / "dir" / "file.txt").resolve()


def test_resolve_file_path_allows_dot(tmp_path):
  resolved = resolve_file_path(".", {"root_directory": str(tmp_path)})
  assert resolved == tmp_path.resolve()


def test_resolve_file_path_allows_interior_dotdot_within_root(tmp_path):
  resolved = resolve_file_path(
      "sub/../file.txt", {"root_directory": str(tmp_path)}
  )
  assert resolved == (tmp_path / "file.txt").resolve()


def test_resolve_file_path_allows_absolute_within_root(tmp_path):
  target = tmp_path / "nested" / "ok.txt"
  resolved = resolve_file_path(str(target), {"root_directory": str(tmp_path)})
  assert resolved == target.resolve()


def test_resolve_file_path_rejects_relative_traversal(tmp_path):
  with pytest.raises(ValueError):
    resolve_file_path("../../escape.txt", {"root_directory": str(tmp_path)})


def test_resolve_file_path_rejects_absolute_outside_root(tmp_path):
  with pytest.raises(ValueError):
    resolve_file_path("/etc/passwd", {"root_directory": str(tmp_path)})


async def test_write_files_blocks_relative_traversal(
    tmp_path, tmp_path_factory
):
  outside = tmp_path_factory.mktemp("outside")
  payload = os.path.relpath(outside / "pwned.txt", tmp_path)

  result = await write_files(
      files={payload: "PWNED"}, tool_context=_tool_context(tmp_path)
  )

  assert not result["success"]
  assert not (outside / "pwned.txt").exists()


async def test_write_files_blocks_absolute_outside_root(
    tmp_path, tmp_path_factory
):
  outside = tmp_path_factory.mktemp("outside")
  target = outside / "abs.txt"

  result = await write_files(
      files={str(target): "PWNED"}, tool_context=_tool_context(tmp_path)
  )

  assert not result["success"]
  assert not target.exists()


async def test_write_files_allows_path_within_root(tmp_path):
  result = await write_files(
      files={"sub/ok.txt": "hello"}, tool_context=_tool_context(tmp_path)
  )

  assert result["success"]
  assert (tmp_path / "sub" / "ok.txt").read_text() == "hello"


async def test_read_files_blocks_relative_traversal(tmp_path, tmp_path_factory):
  outside = tmp_path_factory.mktemp("outside")
  secret = outside / "secret.txt"
  secret.write_text("TOKEN=abc")
  payload = os.path.relpath(secret, tmp_path)

  result = await read_files(
      file_paths=[payload], tool_context=_tool_context(tmp_path)
  )

  assert not result["success"]
  assert all(
      "TOKEN=abc" not in info.get("content", "")
      for info in result["files"].values()
  )


async def test_delete_files_blocks_relative_traversal(
    tmp_path, tmp_path_factory
):
  outside = tmp_path_factory.mktemp("outside")
  victim = outside / "victim.txt"
  victim.write_text("bye")
  payload = os.path.relpath(victim, tmp_path)

  result = await delete_files(
      file_paths=[payload],
      tool_context=_tool_context(tmp_path),
      confirm_deletion=True,
  )

  assert not result["success"]
  assert victim.exists()
