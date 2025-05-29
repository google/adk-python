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
import sys

from . import envs
from ...agents.base_agent import BaseAgent

logger = logging.getLogger("google_adk." + __name__)


class AgentLoader:
  """Centralized agent loading with proper isolation, caching, and .env loading.
  Support loading agents from below folder/file structures:
  a) agents_dir/agent_name.py (with root_agent or agent.root_agent in it)
  b) agents_dir/agent_name_folder/__init__.py (with root_agent or agent.root_agent in the package)
  c) agents_dir/agent_name_folder/agent.py (where agent.py has root_agent)
  """

  def __init__(self, agents_dir: str):
    self.agents_dir = agents_dir.rstrip("/")
    self._original_sys_path = None
    self._agent_cache: dict[str, BaseAgent] = {}

  def _load_from_module_or_package(self, agent_name: str) -> BaseAgent:
    # Load for case: Import "<agent_name>" (as a package or module)
    # Covers structures:
    #   a) agents_dir/agent_name.py (with root_agent or agent.root_agent in it)
    #   b) agents_dir/agent_name_folder/__init__.py (with root_agent or agent.root_agent in the package)
    try:
      module_candidate = importlib.import_module(agent_name)
      # Check for "root_agent" directly in "<agent_name>" module/package
      if hasattr(module_candidate, "root_agent"):
        logger.debug("Found root_agent directly in %s", agent_name)
        return module_candidate.root_agent
      # Check for "<agent_name>.agent.root_agent" structure (e.g. agent_name is a package,
      # and it has an 'agent' submodule/attribute which in turn has 'root_agent')
      if hasattr(module_candidate, "agent") and hasattr(
          module_candidate.agent, "root_agent"
      ):
        logger.debug("Found root_agent in %s.agent attribute", agent_name)
        if isinstance(module_candidate.agent, BaseAgent):
          return module_candidate.agent.root_agent
        else:
          logger.warning(
              "Root agent found is not an instance of BaseAgent. But a type %s",
              type(module_candidate.agent),
          )
    except ModuleNotFoundError:
      logger.debug("Module %s itself not found.", agent_name)
      # Re-raise as ValueError to be caught by the final error message construction
      raise ValueError(
          f"Module {agent_name} not found during import attempts."
      ) from None
    except ImportError as e:
      logger.warning("Error importing %s: %s", agent_name, e)

    return None

  def _load_from_submodule(self, agent_name: str) -> BaseAgent:
    # Load for case: Import "<agent_name>.agent" and look for "root_agent"
    # Covers structure: agents_dir/agent_name_folder/agent.py (where agent.py has root_agent)
    try:
      module_candidate = importlib.import_module(f"{agent_name}.agent")
      if hasattr(module_candidate, "root_agent"):
        logger.debug("Found root_agent in %s.agent", agent_name)
        if isinstance(module_candidate.root_agent, BaseAgent):
          return module_candidate.root_agent
        else:
          logger.warning(
              "Root agent found is not an instance of BaseAgent. But a type %s",
              type(module_candidate.root_agent),
          )
    except ModuleNotFoundError:
      logger.debug(
          "Module %s.agent not found, trying next pattern.", agent_name
      )
    except ImportError as e:
      logger.warning("Error importing %s.agent: %s", agent_name, e)

    return None

  def _perform_load(self, agent_name: str) -> BaseAgent:
    """Internal logic to load an agent"""
    # Add self.agents_dir to sys.path
    if self.agents_dir not in sys.path:
      sys.path.insert(0, self.agents_dir)

    logger.debug(
        "Loading .env for agent %s from %s", agent_name, self.agents_dir
    )
    envs.load_dotenv_for_agent(agent_name, str(self.agents_dir))

    root_agent = self._load_from_module_or_package(agent_name)
    if root_agent:
      return root_agent

    root_agent = self._load_from_submodule(agent_name)
    if root_agent:
      return root_agent

    # If no root_agent was found by any pattern
    raise ValueError(
        f"No root_agent found for '{agent_name}'. Searched in"
        f" '{agent_name}.agent.root_agent', '{agent_name}.root_agent', and"
        f" via an 'agent' attribute within the '{agent_name}' module/package."
        f" Ensure '{self.agents_dir}/{agent_name}' is structured correctly,"
        " an .env file can be loaded if present, and a root_agent is"
        " exposed."
    )

  def load_agent(self, agent_name: str) -> BaseAgent:
    """Load an agent module (with caching & .env) and return its root_agent (asynchronously)."""
    if agent_name in self._agent_cache:
      logger.debug("Returning cached agent for %s (async)", agent_name)
      return self._agent_cache[agent_name]

    logger.debug("Loading agent %s - not in cache.", agent_name)
    # Assumes this method is called when the context manager (`with self:`) is active
    agent = self._perform_load(agent_name)
    self._agent_cache[agent_name] = agent
    return agent
