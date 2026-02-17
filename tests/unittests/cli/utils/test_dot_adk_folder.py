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

from pathlib import Path

from google.adk.cli.utils.dot_adk_folder import dot_adk_folder_for_agent
from google.adk.cli.utils.dot_adk_folder import DotAdkFolder
import pytest


def test_paths_are_relative_to_agent_dir(tmp_path: Path):
  agent_dir = tmp_path / "agent_a"
  folder = DotAdkFolder(agent_dir)

  assert folder.dot_adk_dir == agent_dir.resolve() / ".adk"
  assert folder.artifacts_dir == folder.dot_adk_dir / "artifacts"
  assert folder.session_db_path == folder.dot_adk_dir / "session.db"


def test_for_agent_validates_app_name(tmp_path: Path):
  agents_root = tmp_path / "agents"
  agents_root.mkdir()

  with pytest.raises(ValueError):
    dot_adk_folder_for_agent(
        agents_root=agents_root, app_name="../escape_attempt"
    )

  folder = dot_adk_folder_for_agent(
      agents_root=agents_root, app_name="valid_agent"
  )

  expected_dir = (agents_root / "valid_agent").resolve()
  assert folder.agent_dir == expected_dir
