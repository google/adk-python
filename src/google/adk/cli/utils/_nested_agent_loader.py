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

import importlib
import importlib.util
import logging
import os
from pathlib import Path
import sys
from typing import Literal
from typing import Optional
from typing import Union

from typing_extensions import override

from . import envs
from ...agents.base_agent import BaseAgent
from ...apps.app import App
from .agent_loader import AgentLoader
from .agent_loader import SPECIAL_AGENTS_DIR

logger = logging.getLogger("google_adk." + __name__)


class NestedAgentLoader(AgentLoader):
  """Subclass of AgentLoader that supports recursive nested directory discovery and dot-nested namespaces for dev environments."""

  @staticmethod
  def _is_valid_agent_dir(path: Path) -> bool:
    """Returns True if the directory is a valid agent directory."""
    if not path.is_dir():
      return False
    if (path / "agent.py").is_file():
      return True
    if (path / "root_agent.yaml").is_file():
      return True

    init_py = path / "__init__.py"
    if init_py.is_file():
      try:
        content = init_py.read_text(encoding="utf-8")
        if "root_agent" in content:
          return True
      except Exception as e:
        logger.warning("Error reading %s: %s", init_py, e)

    return False

  def _has_nested_agents(self, agents_path: Path) -> bool:
    """Returns True if there are any nested agents within the directory (up to max depth)."""
    max_depth = 5
    for root, dirs, _ in os.walk(agents_path):
      rel_path = os.path.relpath(root, agents_path)
      depth = 0 if rel_path == "." else len(Path(rel_path).parts)

      if depth >= max_depth:
        dirs[:] = []
      else:
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".") and d != "__pycache__" and d != "tmp"
        ]

      if root == str(agents_path):
        continue

      if self._is_valid_agent_dir(Path(root)):
        return True
    return False

  @override
  def _init_agent_mode(self, agents_path: Path) -> None:
    if agents_path.is_file():
      # Explicit file-based single-agent mode
      self._is_single_agent = True
      self._single_agent_name = agents_path.stem
      self.agents_dir = str(agents_path.parent)
    else:
      # It is a directory. Check if it contains any nested agents.
      if self._has_nested_agents(agents_path):
        # Force multi-agent (nested) mode even if the root directory itself
        # contains an agent.py, to allow discovering the nested agents.
        self._is_single_agent = False
        self._single_agent_name = None
        self.agents_dir = str(agents_path)
      else:
        # Fall back to parent class behavior
        super()._init_agent_mode(agents_path)

  @override
  def list_agents(self) -> list[str]:
    """Lists all agents recursively across subdirectories (sorted alphabetically)."""
    if self._is_single_agent:
      return [self._single_agent_name]
    base_path = Path(self.agents_dir)
    if not base_path.exists() or not base_path.is_dir():
      return []

    apps = []
    max_depth = 5
    # Walk the directory recursively to find all apps
    for root, dirs, _ in os.walk(base_path):
      rel_path = os.path.relpath(root, base_path)
      depth = 0 if rel_path == "." else len(Path(rel_path).parts)

      if depth >= max_depth:
        dirs[:] = []
      else:
        # Avoid hidden directories, pycache, and tmp
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".") and d != "__pycache__" and d != "tmp"
        ]

      if self._is_valid_agent_dir(Path(root)):
        if rel_path and rel_path != ".":
          apps.append(rel_path.replace("\\", ".").replace("/", "."))
    apps.sort()
    return apps

  @override
  def _validate_agent_name(self, full_agent_name: str) -> None:
    """Validate agent name allowing dot-separated paths."""
    if full_agent_name.startswith("__"):
      agent_relative_path = full_agent_name[2:]
      check_dir = os.path.abspath(SPECIAL_AGENTS_DIR)
    else:
      agent_relative_path = full_agent_name
      check_dir = self.agents_dir

    if self._is_single_agent and not full_agent_name.startswith("__"):
      if full_agent_name != self._single_agent_name:
        raise ValueError(
            f"Agent not found: {full_agent_name!r}. In single agent mode, only "
            f"'{self._single_agent_name}' is accessible."
        )

    normalized_path = agent_relative_path.replace(".", "/")
    parts = normalized_path.split("/")
    for part in parts:
      if not part or not part.isidentifier():
        raise ValueError(
            f"Invalid agent name: {full_agent_name!r}. Agent names must be"
            " valid Python identifiers or paths separated by dots (letters,"
            " digits, underscores, and dots)."
        )

    # Verify the agent exists on disk before allowing import
    agent_path = Path(check_dir) / normalized_path
    agent_file = Path(check_dir) / f"{normalized_path}.py"
    if not (agent_path.is_dir() or agent_file.is_file()):
      raise ValueError(
          f"Agent not found: {full_agent_name!r}. No matching directory or"
          f" module exists in '{os.path.join(check_dir, normalized_path)}'."
      )

  @override
  def load_agent(self, agent_name: str) -> Union[BaseAgent, App]:
    """Load an agent module (with caching & .env) and return its root_agent.

    Args:
        agent_name: The dot-delimited full agent name (e.g. 'folder_name.app_name').
    """
    if agent_name in self._agent_cache:
      logger.debug("Returning cached agent for %s (async)", agent_name)
      return self._agent_cache[agent_name]

    logger.debug("Loading agent %s - not in cache.", agent_name)
    agent_or_app = self._perform_load(agent_name)
    self._agent_cache[agent_name] = agent_or_app
    return agent_or_app

  @override
  def _perform_load(self, agent_path: str) -> Union[BaseAgent, App]:
    """Internal logic to load an agent allowing slash-separated paths."""
    self._validate_agent_name(agent_path)
    # Determine the directory to use for loading
    if agent_path.startswith("__"):
      agents_dir = os.path.abspath(SPECIAL_AGENTS_DIR)
      actual_agent_name = agent_path[2:]
      module_base_name = actual_agent_name
      package_parts: list[str] = []
      package_root: Optional[Path] = None
      current_dir = Path(agents_dir).resolve()
      while True:
        if not (current_dir / "__init__.py").is_file():
          package_root = current_dir
          break
        package_parts.append(current_dir.name)
        current_dir = current_dir.parent
      if package_parts:
        package_parts.reverse()
        module_base_name = ".".join(package_parts + [actual_agent_name])
        if str(package_root) not in sys.path:
          sys.path.insert(0, str(package_root))
    else:
      agents_dir = self.agents_dir
      actual_agent_name = agent_path.replace(".", "/")
      module_base_name = agent_path.replace("/", ".")

    if agents_dir not in sys.path:
      sys.path.insert(0, agents_dir)

    logger.debug("Loading .env for agent %s from %s", agent_path, agents_dir)
    envs.load_dotenv_for_agent(actual_agent_name, str(agents_dir))

    if root_agent := self._load_from_module_or_package(module_base_name):
      self._record_origin_metadata(
          loaded=root_agent,
          expected_app_name=agent_path,
          module_name=module_base_name,
          agents_dir=agents_dir,
      )
      return root_agent

    if root_agent := self._load_from_submodule(module_base_name):
      self._record_origin_metadata(
          loaded=root_agent,
          expected_app_name=agent_path,
          module_name=f"{module_base_name}.agent",
          agents_dir=agents_dir,
      )
      return root_agent

    if root_agent := self._load_from_yaml_config(actual_agent_name, agents_dir):
      self._record_origin_metadata(
          loaded=root_agent,
          expected_app_name=actual_agent_name,
          module_name=None,
          agents_dir=agents_dir,
      )
      return root_agent

    hint = ""
    agents_path = Path(agents_dir)
    if (
        agents_path.joinpath("agent.py").is_file()
        or agents_path.joinpath("root_agent.yaml").is_file()
    ):
      hint = (
          "\n\nHINT: It looks like this command might be running from inside an"
          " agent directory. Run it from the parent directory that contains"
          " your agent folder (for example the project root) so the loader can"
          " locate your agents."
      )

    raise ValueError(
        f"No root_agent found for '{agent_path}'. Searched in"
        f" '{actual_agent_name}.agent.root_agent',"
        f" '{actual_agent_name}.root_agent' and"
        f" '{actual_agent_name}{os.sep}root_agent.yaml'.\n\nExpected directory"
        f" structure:\n  <agents_dir>{os.sep}\n   "
        f" {actual_agent_name}{os.sep}\n      agent.py (with root_agent) OR\n  "
        "    root_agent.yaml\n\nThen run: adk web <agents_dir>\n\nEnsure"
        f" '{os.path.join(agents_dir, actual_agent_name)}' is structured"
        " correctly, an .env file can be loaded if present, and a root_agent"
        f" is exposed.{hint}"
    )

  @override
  def _determine_agent_language(
      self, agent_name: str
  ) -> Literal["yaml", "python"]:
    agent_path = agent_name.replace(".", "/")
    base_path = Path(self.agents_dir) / agent_path

    if (base_path / "root_agent.yaml").exists():
      return "yaml"
    elif (base_path / "agent.py").exists():
      return "python"
    elif (base_path / "__init__.py").exists() and self._is_valid_agent_dir(
        base_path
    ):
      return "python"

    raise ValueError(f"Could not determine agent type for '{agent_name}'.")

  @override
  def remove_agent_from_cache(self, agent_name: str):
    agent_dot_path = agent_name.replace("/", ".")
    keys_to_delete = [
        module_name
        for module_name in sys.modules
        if module_name == agent_dot_path
        or module_name.startswith(f"{agent_dot_path}.")
    ]
    for key in keys_to_delete:
      logger.debug("Deleting module %s", key)
      del sys.modules[key]
    self._agent_cache.pop(agent_name, None)

  @override
  def _record_origin_metadata(
      self,
      *,
      loaded: Union[BaseAgent, App],
      expected_app_name: str,
      module_name: Optional[str],
      agents_dir: str,
  ) -> None:
    expected_full_app_name = expected_app_name

    # Do not attach metadata for built-in agents (double underscore names).
    if expected_full_app_name.startswith("__"):
      return

    origin_path: Optional[Path] = None
    if module_name:
      spec = importlib.util.find_spec(module_name)
      if spec and spec.origin:
        module_origin = Path(spec.origin).resolve()
        origin_path = (
            module_origin.parent if module_origin.is_file() else module_origin
        )

    if origin_path is None:
      candidate = Path(agents_dir, expected_full_app_name.replace(".", "/"))
      origin_path = candidate if candidate.exists() else Path(agents_dir)

    def _attach_metadata(target: Union[BaseAgent, App]) -> None:
      setattr(target, "_adk_origin_app_name", expected_full_app_name)
      setattr(target, "_adk_origin_path", origin_path)

    if isinstance(loaded, App):
      _attach_metadata(loaded)
      if loaded.root_agent is not None:
        _attach_metadata(loaded.root_agent)
    else:
      _attach_metadata(loaded)
