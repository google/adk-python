# Copyright 2025 Google LLC
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
"""Helpers for managing an agent's `.adk` folder."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path


def _resolve_agent_dir(*, agents_root: Path | str, app_name: str) -> Path:
  """Resolves the agent directory with safety checks."""
  agents_root_path = Path(agents_root).resolve()
  agent_dir = (agents_root_path / app_name).resolve()
  if not str(agent_dir).startswith(str(agents_root_path)):
    raise ValueError(
        f"Invalid app_name '{app_name}': resolves outside base directory"
    )

  return agent_dir


class DotAdkFolder:
  """Manages the lifecycle of the `.adk` folder for a single agent."""

  def __init__(self, agent_dir: Path | str):
    self._agent_dir = Path(agent_dir).resolve()

  @property
  def agent_dir(self) -> Path:
    return self._agent_dir

  @cached_property
  def dot_adk_dir(self) -> Path:
    return self._agent_dir / ".adk"

  @cached_property
  def artifacts_dir(self) -> Path:
    return self.dot_adk_dir / "artifacts"

  @cached_property
  def session_db_path(self) -> Path:
    return self.dot_adk_dir / "session.db"


def dot_adk_folder_for_agent(
    *, agents_root: Path | str, app_name: str
) -> DotAdkFolder:
  """Creates a manager for an agent rooted under `agents_root`.

  Args:
    agents_root: Directory that contains all agents.
    app_name: Name of the agent directory.

  Returns:
    A `DotAdkFolder` scoped to the given agent.

  Raises:
    ValueError: If `app_name` traverses outside of `agents_root`.
  """
  return DotAdkFolder(
      _resolve_agent_dir(agents_root=agents_root, app_name=app_name)
  )
