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

from __future__ import annotations

from types import SimpleNamespace

import pytest

from google.adk.cli.built_in_agents.tools.delete_files import delete_files
from google.adk.cli.built_in_agents.tools.read_files import read_files
from google.adk.cli.built_in_agents.tools.write_files import write_files
from google.adk.cli.built_in_agents.utils.resolve_root_directory import (
    resolve_file_path,
)


def _tool_context(root_directory):
  return SimpleNamespace(
      _invocation_context=SimpleNamespace(
          session=SimpleNamespace(state={"root_directory": str(root_directory)})
      )
  )


def test_resolve_file_path_allows_paths_inside_project_root(tmp_path):
  project_root = tmp_path / "project"
  project_root.mkdir()

  assert (
      resolve_file_path("tools/agent.py", {"root_directory": str(project_root)})
      == project_root / "tools" / "agent.py"
  )
  assert (
      resolve_file_path(
          str(project_root / "root_agent.yaml"),
          {"root_directory": str(project_root)},
      )
      == project_root / "root_agent.yaml"
  )


@pytest.mark.parametrize("path", ["../outside.txt", "/tmp/outside.txt"])
def test_resolve_file_path_rejects_paths_outside_project_root(tmp_path, path):
  project_root = tmp_path / "project"
  project_root.mkdir()

  with pytest.raises(ValueError, match="escapes project root"):
    resolve_file_path(path, {"root_directory": str(project_root)})


def test_resolve_file_path_rejects_symlink_escape(tmp_path):
  project_root = tmp_path / "project"
  outside_dir = tmp_path / "outside"
  project_root.mkdir()
  outside_dir.mkdir()
  (project_root / "linked").symlink_to(outside_dir, target_is_directory=True)

  with pytest.raises(ValueError, match="escapes project root"):
    resolve_file_path(
        "linked/agent.py",
        {"root_directory": str(project_root)},
    )


@pytest.mark.asyncio
async def test_builder_file_tools_do_not_access_paths_outside_project_root(
    tmp_path,
):
  project_root = tmp_path / "project"
  project_root.mkdir()
  outside_file = tmp_path / "outside.txt"
  outside_file.write_text("secret", encoding="utf-8")
  tool_context = _tool_context(project_root)

  read_result = await read_files([str(outside_file)], tool_context)
  write_result = await write_files({str(outside_file): "changed"}, tool_context)
  delete_result = await delete_files([str(outside_file)], tool_context)

  assert not read_result["success"]
  assert not write_result["success"]
  assert not delete_result["success"]
  assert outside_file.read_text(encoding="utf-8") == "secret"
