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
"""File system event handler for agent changes to trigger hot reload for agents."""

from __future__ import annotations

import logging
from pathlib import Path
import warnings

from watchdog.events import FileSystemEventHandler

from .agent_loader import AgentLoader
from .agent_loader import is_single_agent_directory
from .shared_value import SharedValue

logger = logging.getLogger("google_adk." + __name__)


class AgentChangeEventHandler(FileSystemEventHandler):

  def __init__(
      self,
      agent_loader: AgentLoader,
      runners_to_clean: set[str],
      current_app_name_ref: SharedValue[str] | None = None,
      agents_dir: str | None = None,
  ):
    if isinstance(current_app_name_ref, (str, Path)):
      agents_dir = str(current_app_name_ref)
      current_app_name_ref = None
    elif isinstance(agents_dir, SharedValue):
      current_app_name_ref = agents_dir
      agents_dir = None
    self.agent_loader = agent_loader
    self.runners_to_clean = runners_to_clean
    self.agents_dir = Path(agents_dir).resolve() if agents_dir else None
    self.current_app_name_ref = current_app_name_ref
    if current_app_name_ref is not None:
      warnings.warn(
          "current_app_name_ref is deprecated and will be removed in a future"
          " release. Use agents_dir instead.",
          DeprecationWarning,
          stacklevel=2,
      )

  def on_modified(self, event):
    if not event.src_path.endswith((".py", ".yaml", ".yml")):
      return
    logger.info("Change detected in agents directory: %s", event.src_path)
    if self.agents_dir is None:
      if self.current_app_name_ref is not None:
        self.agent_loader.remove_agent_from_cache(
            self.current_app_name_ref.value
        )
        self.runners_to_clean.add(self.current_app_name_ref.value)
      return

    file_path = Path(event.src_path).resolve()
    try:
      rel_path = file_path.relative_to(self.agents_dir)
    except ValueError:
      return

    agent_names: list[str] = []
    if not rel_path.parent.parts:
      agent_names.append(rel_path.stem)
    else:
      curr = file_path.parent
      while curr != self.agents_dir and curr != curr.parent:
        if is_single_agent_directory(curr):
          agent_names.append(".".join(curr.relative_to(self.agents_dir).parts))
        curr = curr.parent

      top_level_name = rel_path.parts[0]
      top_level_dir = self.agents_dir / top_level_name
      if (top_level_dir / "__init__.py").is_file() or agent_names:
        if top_level_name not in agent_names:
          agent_names.append(top_level_name)

    for agent_name in agent_names:
      self.agent_loader.remove_agent_from_cache(agent_name)
      self.runners_to_clean.add(agent_name)
