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

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from ..agents.base_agent import BaseAgent
  from ..runners import Runner


class AgentToolManager:
  """Manages the relationship between runners and agents used by AgentTool.

  This class prevents premature cleanup of agent toolsets when multiple
  AgentTools using the same agent are running concurrently. It tracks
  active runners per agent and ensures that agent toolsets are only cleaned
  up when no runners are using that agent.

  The manager uses a lock to ensure thread-safe registration and
  unregistration of runners. When unregistering a runner, the lock is held
  until the returned async generator is fully consumed, ensuring that cleanup
  operations can complete safely without race conditions.
  """

  def __init__(self):
    """Initializes the AgentToolManager."""
    # Maps agent to a set of active runners using that agent
    self._runners_by_agent: dict[int, set[Runner]] = {}
    # Lock to ensure thread-safe access to _runners_by_agent
    self._lock = asyncio.Lock()

  async def register_runner(self, agent: BaseAgent, runner: Runner) -> None:
    """Registers a runner for the given agent.

    This method should be called at the start of AgentTool.run_async()
    when a runner is created. The runner is tracked to prevent premature
    cleanup of the agent's toolsets.

    Args:
      agent: The agent instance used by the runner.
      runner: The runner instance to register.
    """
    async with self._lock:
      # TODO: can we use the name of the agent as the key?
      if id(agent) not in self._runners_by_agent:
        self._runners_by_agent[id(agent)] = set()
      self._runners_by_agent[id(agent)].add(runner)

  @asynccontextmanager
  async def unregister_runner(self, agent: BaseAgent, runner: Runner):
    """Unregisters a runner for the given agent.

    This method should be called before cleaning up a runner at the end
    of AgentTool.run_async(). It returns an async context manager that yields
    whether the runner should be cleaned up (i.e., if it's the last runner
    using the agent). The lock is held until the context manager is fully consumed,
    ensuring that cleanup operations can complete safely.

    Usage:
      async with manager.unregister_runner(agent, runner) as should_cleanup:
        if should_cleanup:
          await runner.close()

    Args:
      agent: The agent instance used by the runner.
      runner: The runner instance to unregister.

    Yields:
      True if this was the last runner using the agent and cleanup should
      proceed, False if other runners are still using the agent and cleanup
      should be skipped.
    """
    async with self._lock:
      yield self._unregister(agent, runner)

  def _unregister(self, agent: BaseAgent, runner: Runner) -> bool:
    """Unregisters a runner and determines if cleanup should proceed.

    Args:
      agent: The agent instance used by the runner.
      runner: The runner instance to unregister.

    Returns:
      True if cleanup should proceed (no other runners using the agent),
      False if cleanup should be skipped (other runners still using the agent).
    """
    if id(agent) not in self._runners_by_agent:
      # Agent not registered, safe to cleanup
      return True

    runners = self._runners_by_agent[id(agent)]
    if runner not in runners:
      # Runner not registered, safe to cleanup
      return True

    runners.remove(runner)

    # If no runners left for this agent, cleanup is safe
    if not runners:
      del self._runners_by_agent[id(agent)]
      return True

    # Other runners still using this agent, skip cleanup
    return False


_agent_tool_manager_instance: AgentToolManager | None = None


def get_agent_tool_manager() -> AgentToolManager:
  """Gets the singleton AgentToolManager instance, initializing it if needed."""
  global _agent_tool_manager_instance
  if _agent_tool_manager_instance is None:
    _agent_tool_manager_instance = AgentToolManager()
  return _agent_tool_manager_instance
