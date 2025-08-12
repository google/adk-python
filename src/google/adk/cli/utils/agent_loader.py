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
import logging
import os
from pathlib import Path
import sys
from typing import Optional

from pydantic import ValidationError
from typing_extensions import override

from . import envs
from ...agents import config_agent_utils
from ...agents.base_agent import BaseAgent
from ...utils.feature_decorator import experimental
from .base_agent_loader import BaseAgentLoader

logger = logging.getLogger("google_adk." + __name__)


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
    self.agents_dir = agents_dir.rstrip("/")
    self._original_sys_path = None
    self._agent_cache: dict[str, BaseAgent] = {}

  def _load_from_module_or_package(
      self, agent_name: str
  ) -> Optional[BaseAgent]:
    # Load for case: Import "{agent_name}" (as a package or module)
    # Covers structures:
    #   a) agents_dir/{agent_name}.py (with root_agent in the module)
    #   b) agents_dir/{agent_name}/__init__.py (with root_agent in the package)
    try:
      module_candidate = importlib.import_module(agent_name)
      # Check for "root_agent" directly in "{agent_name}" module/package
      if hasattr(module_candidate, "root_agent"):
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
        # it's the case the module imported by {agent_name}.agent module is not
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

  def _load_from_submodule(self, agent_name: str) -> Optional[BaseAgent]:
    # Load for case: Import "{agent_name}.agent" and look for "root_agent"
    # Covers structure: agents_dir/{agent_name}/agent.py (with root_agent defined in the module)
    try:
      module_candidate = importlib.import_module(f"{agent_name}.agent")
      if hasattr(module_candidate, "root_agent"):
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
        # it's the case the module imported by {agent_name}.agent module is not
        # found
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
  def _load_from_yaml_config(self, agent_name: str) -> Optional[BaseAgent]:
    # Load from the config file at agents_dir/{agent_name}/root_agent.yaml
    config_path = os.path.join(self.agents_dir, agent_name, "root_agent.yaml")
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

  def _perform_load(self, agent_name: str) -> BaseAgent:
    """Internal logic to load an agent"""
    # Add self.agents_dir to sys.path
    if self.agents_dir not in sys.path:
      sys.path.insert(0, self.agents_dir)

    logger.debug(
        "Loading .env for agent %s from %s", agent_name, self.agents_dir
    )
    envs.load_dotenv_for_agent(agent_name, str(self.agents_dir))

    if root_agent := self._load_from_module_or_package(agent_name):
      return root_agent

    if root_agent := self._load_from_submodule(agent_name):
      return root_agent

    if root_agent := self._load_from_yaml_config(agent_name):
      return root_agent

    # If no root_agent was found by any pattern
    raise ValueError(
        f"No root_agent found for '{agent_name}'. Searched in"
        f" '{agent_name}.agent.root_agent', '{agent_name}.root_agent' and"
        f" '{agent_name}/root_agent.yaml'."
        f" Ensure '{self.agents_dir}/{agent_name}' is structured correctly,"
        " an .env file can be loaded if present, and a root_agent is"
        " exposed."
    )

  @override
  def load_agent(self, agent_name: str) -> BaseAgent:
    """Load an agent module (with caching & .env) and return its root_agent."""
    if agent_name in self._agent_cache:
      logger.debug("Returning cached agent for %s (async)", agent_name)
      return self._agent_cache[agent_name]

    logger.debug("Loading agent %s - not in cache.", agent_name)
    agent = self._perform_load(agent_name)
    self._agent_cache[agent_name] = agent
    return agent

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
