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

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from pathlib import Path
import sys
from typing import Any
from typing import Literal
from typing import Optional
from typing import Union

from pydantic import ValidationError
from typing_extensions import override

from . import envs
from ...agents import config_agent_utils
from ...agents.base_agent import BaseAgent
from ...apps.app import App
from ...utils.feature_decorator import experimental
from .base_agent_loader import BaseAgentLoader

logger = logging.getLogger("google_adk." + __name__)

# Special agents directory for agents with names starting with double underscore
SPECIAL_AGENTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "built_in_agents"
)


class AgentLoader(BaseAgentLoader):
  """Centralized agent loading with proper isolation, caching, and .env loading.
  Support loading agents from below folder/file structures:
  a)  {agent_name}.agent as a module name:
      agents_dir/{agent_name}/agent.py (with root_agent defined in the module)
  b)  {agent_name} as a module name
      agents_dir/{agent_name}.py (with root_agent defined in the module)
  c)  {agent_name} as a package name
      agents_dir/{agent_name}/__init__.py (with root_agent in the package)
  d)  {agent_name} as a YAML config folder:
      agents_dir/{agent_name}/root_agent.yaml defines the root agent

  """

  def __init__(self, agents_dir: str):
    self.agents_dir = str(Path(agents_dir))
    self._original_sys_path = None
    self._agent_cache: dict[str, Union[BaseAgent, App]] = {}

  def _load_from_module_or_package(
      self, agent_name: str
  ) -> Optional[Union[BaseAgent, App]]:
    # Load for case: Import "{agent_name}" (as a package or module)
    # Covers structures:
    #   a) agents_dir/{agent_name}.py (with root_agent in the module)
    #   b) agents_dir/{agent_name}/__init__.py (with root_agent in the package)
    try:
      module_candidate = importlib.import_module(agent_name)
      # Check for "app" first, then "root_agent"
      if hasattr(module_candidate, "app") and isinstance(
          module_candidate.app, App
      ):
        logger.debug("Found app in %s", agent_name)
        return module_candidate.app
      # Check for "root_agent" directly in "{agent_name}" module/package
      elif hasattr(module_candidate, "root_agent"):
        logger.debug("Found root_agent directly in %s", agent_name)
        if isinstance(module_candidate.root_agent, BaseAgent):
          return module_candidate.root_agent
        else:
          logger.warning(
              "Root agent found is not an instance of BaseAgent. But a type %s",
              type(module_candidate.root_agent),
          )
      else:
        logger.debug(
            "Module %s has no root_agent. Trying next pattern.",
            agent_name,
        )

    except ModuleNotFoundError as e:
      if e.name == agent_name:
        logger.debug("Module %s itself not found.", agent_name)
      else:
        # the module imported by {agent_name}.agent module is not
        # found
        e.msg = f"Fail to load '{agent_name}' module. " + e.msg
        raise e
    except Exception as e:
      if hasattr(e, "msg"):
        e.msg = f"Fail to load '{agent_name}' module. " + e.msg
        raise e
      e.args = (
          f"Fail to load '{agent_name}' module. {e.args[0] if e.args else ''}",
      ) + e.args[1:]
      raise e

    return None

  def _load_from_submodule(
      self, agent_name: str
  ) -> Optional[Union[BaseAgent], App]:
    # Load for case: Import "{agent_name}.agent" and look for "root_agent"
    # Covers structure: agents_dir/{agent_name}/agent.py (with root_agent defined in the module)
    try:
      module_candidate = importlib.import_module(f"{agent_name}.agent")
      # Check for "app" first, then "root_agent"
      if hasattr(module_candidate, "app") and isinstance(
          module_candidate.app, App
      ):
        logger.debug("Found app in %s.agent", agent_name)
        return module_candidate.app
      elif hasattr(module_candidate, "root_agent"):
        logger.info("Found root_agent in %s.agent", agent_name)
        if isinstance(module_candidate.root_agent, BaseAgent):
          return module_candidate.root_agent
        else:
          logger.warning(
              "Root agent found is not an instance of BaseAgent. But a type %s",
              type(module_candidate.root_agent),
          )
      else:
        logger.debug(
            "Module %s.agent has no root_agent.",
            agent_name,
        )
    except ModuleNotFoundError as e:
      # if it's agent module not found, it's fine, search for next pattern
      if e.name == f"{agent_name}.agent" or e.name == agent_name:
        logger.debug("Module %s.agent not found.", agent_name)
      else:
        # the module imported by {agent_name}.agent module is not found
        e.msg = f"Fail to load '{agent_name}.agent' module. " + e.msg
        raise e
    except Exception as e:
      if hasattr(e, "msg"):
        e.msg = f"Fail to load '{agent_name}.agent' module. " + e.msg
        raise e
      e.args = (
          (
              f"Fail to load '{agent_name}.agent' module."
              f" {e.args[0] if e.args else ''}"
          ),
      ) + e.args[1:]
      raise e

    return None

  @experimental
  def _load_from_yaml_config(
      self, agent_name: str, agents_dir: str
  ) -> Optional[BaseAgent]:
    # Load from the config file at agents_dir/{agent_name}/root_agent.yaml
    config_path = os.path.join(agents_dir, agent_name, "root_agent.yaml")
    try:
      agent = config_agent_utils.from_config(config_path)
      logger.info("Loaded root agent for %s from %s", agent_name, config_path)
      return agent
    except FileNotFoundError:
      logger.debug("Config file %s not found.", config_path)
      return None
    except ValidationError as e:
      logger.error("Config file %s is invalid YAML.", config_path)
      raise e
    except Exception as e:
      if hasattr(e, "msg"):
        e.msg = f"Fail to load '{config_path}' config. " + e.msg
        raise e
      e.args = (
          f"Fail to load '{config_path}' config. {e.args[0] if e.args else ''}",
      ) + e.args[1:]
      raise e

  def _perform_load(self, agent_name: str) -> Union[BaseAgent, App]:
    """Internal logic to load an agent"""
    # Determine the directory to use for loading
    if agent_name.startswith("__"):
      # Special agent: use special agents directory
      agents_dir = os.path.abspath(SPECIAL_AGENTS_DIR)
      # Remove the double underscore prefix for the actual agent name
      actual_agent_name = agent_name[2:]
      # If this special agents directory is part of a package (has __init__.py
      # up the tree), build a fully-qualified module path so the built-in agent
      # can continue to use relative imports. Otherwise, fall back to importing
      # by module name relative to agents_dir.
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
      # Regular agent: use the configured agents directory
      agents_dir = self.agents_dir
      actual_agent_name = agent_name
      module_base_name = actual_agent_name

    # Add agents_dir to sys.path
    if agents_dir not in sys.path:
      sys.path.insert(0, agents_dir)

    logger.debug("Loading .env for agent %s from %s", agent_name, agents_dir)
    envs.load_dotenv_for_agent(actual_agent_name, str(agents_dir))

    if root_agent := self._load_from_module_or_package(module_base_name):
      self._record_origin_metadata(
          loaded=root_agent,
          expected_app_name=agent_name,
          module_name=module_base_name,
          agents_dir=agents_dir,
      )
      return root_agent

    if root_agent := self._load_from_submodule(module_base_name):
      self._record_origin_metadata(
          loaded=root_agent,
          expected_app_name=agent_name,
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

    # If no root_agent was found by any pattern
    # Check if user might be in the wrong directory
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
        f"No root_agent found for '{agent_name}'. Searched in"
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

  def _record_origin_metadata(
      self,
      *,
      loaded: Union[BaseAgent, App],
      expected_app_name: str,
      module_name: Optional[str],
      agents_dir: str,
  ) -> None:
    """Annotates loaded agent/App with its origin for later diagnostics."""

    # Do not attach metadata for built-in agents (double underscore names).
    if expected_app_name.startswith("__"):
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
      candidate = Path(agents_dir, expected_app_name)
      origin_path = candidate if candidate.exists() else Path(agents_dir)

    def _attach_metadata(target: Union[BaseAgent, App]) -> None:
      setattr(target, "_adk_origin_app_name", expected_app_name)
      setattr(target, "_adk_origin_path", origin_path)

    if isinstance(loaded, App):
      _attach_metadata(loaded)
      _attach_metadata(loaded.root_agent)
    else:
      _attach_metadata(loaded)

  @override
  def load_agent(self, agent_name: str) -> Union[BaseAgent, App]:
    """Load an agent module (with caching & .env) and return its root_agent."""
    if agent_name in self._agent_cache:
      logger.debug("Returning cached agent for %s (async)", agent_name)
      return self._agent_cache[agent_name]

    logger.debug("Loading agent %s - not in cache.", agent_name)
    agent_or_app = self._perform_load(agent_name)
    self._agent_cache[agent_name] = agent_or_app
    return agent_or_app

  @override
  def list_agents(self) -> list[str]:
    """Lists all agents available in the agent loader (sorted alphabetically)."""
    base_path = Path.cwd() / self.agents_dir
    agent_names = [
        x
        for x in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, x))
        and not x.startswith(".")
        and x != "__pycache__"
    ]
    agent_names.sort()
    return agent_names

  def list_agents_detailed(self) -> list[dict[str, Any]]:
    """Lists all agents with detailed metadata (name, description, type)."""
    agent_names = self.list_agents()
    apps_info = []

    for agent_name in agent_names:
      try:
        loaded = self.load_agent(agent_name)
        if isinstance(loaded, App):
          agent = loaded.root_agent
        else:
          agent = loaded

        language = self._determine_agent_language(agent_name)

        app_info = {
            "name": agent_name,
            "root_agent_name": agent.name,
            "description": agent.description,
            "language": language,
        }
        apps_info.append(app_info)

      except Exception as e:
        logger.error("Failed to load agent '%s': %s", agent_name, e)
        continue

    return apps_info

  def _determine_agent_language(
      self, agent_name: str
  ) -> Literal["yaml", "python"]:
    """Determine the type of agent based on file structure."""
    base_path = Path.cwd() / self.agents_dir / agent_name

    if (base_path / "root_agent.yaml").exists():
      return "yaml"
    elif (base_path / "agent.py").exists():
      return "python"
    elif (base_path / "__init__.py").exists():
      return "python"

    raise ValueError(f"Could not determine agent type for '{agent_name}'.")

  def remove_agent_from_cache(self, agent_name: str):
    # Clear module cache for the agent and its submodules
    keys_to_delete = [
        module_name
        for module_name in sys.modules
        if module_name == agent_name or module_name.startswith(f"{agent_name}.")
    ]
    for key in keys_to_delete:
      logger.debug("Deleting module %s", key)
      del sys.modules[key]
    self._agent_cache.pop(agent_name, None)
