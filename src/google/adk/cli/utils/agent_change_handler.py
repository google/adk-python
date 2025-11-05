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
"""File system event handler for agent changes to trigger hot reload for agents."""

from __future__ import annotations

import logging

from watchdog.events import FileSystemEventHandler

from .agent_loader import AgentLoader
from .shared_value import SharedValue

logger = logging.getLogger("google_adk." + __name__)


class AgentChangeEventHandler(FileSystemEventHandler):

  def __init__(
      self,
      agent_loader: AgentLoader,
      runners_to_clean: set[str],
      current_app_name_ref: SharedValue[str],
  ):
    self.agent_loader = agent_loader
    self.runners_to_clean = runners_to_clean
    self.current_app_name_ref = current_app_name_ref

  def on_modified(self, event):
    if not (event.src_path.endswith(".py") or event.src_path.endswith(".yaml")):
      return
    logger.info("Change detected in agents directory: %s", event.src_path)
    self.agent_loader.remove_agent_from_cache(self.current_app_name_ref.value)
    self.runners_to_clean.add(self.current_app_name_ref.value)
