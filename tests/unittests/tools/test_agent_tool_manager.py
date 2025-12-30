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
from unittest import mock

from google.adk.agents.llm_agent import Agent
from google.adk.runners import Runner
from google.adk.tools._agent_tool_manager import AgentToolManager
from google.adk.tools._agent_tool_manager import get_agent_tool_manager
import pytest

from .. import testing_utils


@pytest.fixture
def manager():
  """Creates a fresh AgentToolManager instance for each test."""
  return AgentToolManager()


@pytest.fixture
def agent():
  """Creates a test agent."""
  return Agent(
      name='test_agent',
      model=testing_utils.MockModel.create(responses=['test']),
  )


@pytest.fixture
def runner(agent):
  """Creates a test runner."""
  return testing_utils.InMemoryRunner(agent)


@pytest.mark.asyncio
async def test_register_runner(manager, agent, runner):
  """Test basic runner registration."""
  await manager.register_runner(agent, runner)

  # Verify runner is registered
  async with manager._lock:
    assert id(agent) in manager._runners_by_agent
    assert runner in manager._runners_by_agent[id(agent)]


@pytest.mark.asyncio
async def test_unregister_runner_single_runner(manager, agent, runner):
  """Test unregistering the only runner for an agent."""
  await manager.register_runner(agent, runner)

  async with manager.unregister_runner(agent, runner) as should_cleanup:
    assert should_cleanup is True

  # Verify runner is removed
  async with manager._lock:
    assert id(agent) not in manager._runners_by_agent


@pytest.mark.asyncio
async def test_unregister_runner_multiple_runners(manager, agent):
  """Test unregistering one runner when multiple runners exist."""
  runner1 = testing_utils.InMemoryRunner(agent)
  runner2 = testing_utils.InMemoryRunner(agent)

  await manager.register_runner(agent, runner1)
  await manager.register_runner(agent, runner2)

  # Unregister first runner - should not cleanup
  async with manager.unregister_runner(agent, runner1) as should_cleanup:
    assert should_cleanup is False

  # Verify runner1 is removed but runner2 remains
  async with manager._lock:
    assert id(agent) in manager._runners_by_agent
    assert runner1 not in manager._runners_by_agent[id(agent)]
    assert runner2 in manager._runners_by_agent[id(agent)]

  # Unregister second runner - should cleanup
  async with manager.unregister_runner(agent, runner2) as should_cleanup:
    assert should_cleanup is True

  # Verify agent is completely removed
  async with manager._lock:
    assert id(agent) not in manager._runners_by_agent


@pytest.mark.asyncio
async def test_unregister_unregistered_runner(manager, agent, runner):
  """Test unregistering a runner that was never registered."""
  async with manager.unregister_runner(agent, runner) as should_cleanup:
    # Should allow cleanup for unregistered runner
    assert should_cleanup is True


@pytest.mark.asyncio
async def test_unregister_unregistered_agent(manager, agent, runner):
  """Test unregistering from an agent that was never registered."""
  # Register runner for a different agent
  other_agent = Agent(
      name='other_agent',
      model=testing_utils.MockModel.create(responses=['test']),
  )
  await manager.register_runner(other_agent, runner)

  # Try to unregister from unregistered agent
  async with manager.unregister_runner(agent, runner) as should_cleanup:
    assert should_cleanup is True

  # Verify other agent is still registered
  async with manager._lock:
    assert id(other_agent) in manager._runners_by_agent
    assert runner in manager._runners_by_agent[id(other_agent)]


@pytest.mark.asyncio
async def test_multiple_agents(manager):
  """Test managing runners for multiple different agents."""
  agent1 = Agent(
      name='agent1', model=testing_utils.MockModel.create(responses=['test'])
  )
  agent2 = Agent(
      name='agent2', model=testing_utils.MockModel.create(responses=['test'])
  )
  runner1 = testing_utils.InMemoryRunner(agent1)
  runner2 = testing_utils.InMemoryRunner(agent2)

  await manager.register_runner(agent1, runner1)
  await manager.register_runner(agent2, runner2)

  # Verify both agents are tracked separately
  async with manager._lock:
    assert id(agent1) in manager._runners_by_agent
    assert id(agent2) in manager._runners_by_agent
    assert runner1 in manager._runners_by_agent[id(agent1)]
    assert runner2 in manager._runners_by_agent[id(agent2)]

  # Unregister one agent
  async with manager.unregister_runner(agent1, runner1) as should_cleanup:
    assert should_cleanup is True

  # Verify only agent2 remains
  async with manager._lock:
    assert id(agent1) not in manager._runners_by_agent
    assert id(agent2) in manager._runners_by_agent


@pytest.mark.asyncio
async def test_concurrent_registration(manager, agent):
  """Test concurrent registration of multiple runners."""
  num_runners = 10
  runners = [testing_utils.InMemoryRunner(agent) for _ in range(num_runners)]

  # Register all runners concurrently
  await asyncio.gather(
      *[manager.register_runner(agent, runner) for runner in runners]
  )

  # Verify all runners are registered
  async with manager._lock:
    assert id(agent) in manager._runners_by_agent
    assert len(manager._runners_by_agent[id(agent)]) == num_runners
    for runner in runners:
      assert runner in manager._runners_by_agent[id(agent)]


@pytest.mark.asyncio
async def test_concurrent_unregistration(manager, agent):
  """Test concurrent unregistration of multiple runners."""
  num_runners = 10
  runners = [testing_utils.InMemoryRunner(agent) for _ in range(num_runners)]

  # Register all runners
  await asyncio.gather(
      *[manager.register_runner(agent, runner) for runner in runners]
  )

  # Unregister all runners concurrently
  async def unregister_runner(runner):
    async with manager.unregister_runner(agent, runner) as should_cleanup:
      return should_cleanup

  cleanup_results = await asyncio.gather(
      *[unregister_runner(runner) for runner in runners]
  )

  # Only the last runner should trigger cleanup
  cleanup_count = sum(1 for result in cleanup_results if result is True)
  assert cleanup_count == 1

  # Verify agent is removed
  async with manager._lock:
    assert id(agent) not in manager._runners_by_agent


@pytest.mark.asyncio
async def test_concurrent_register_and_unregister(manager, agent):
  """Test concurrent registration and unregistration."""
  num_operations = 20
  runners = [testing_utils.InMemoryRunner(agent) for _ in range(num_operations)]

  async def register_and_unregister(runner):
    await manager.register_runner(agent, runner)
    async with manager.unregister_runner(agent, runner) as should_cleanup:
      return should_cleanup

  # Run register/unregister operations concurrently
  results = await asyncio.gather(
      *[register_and_unregister(runner) for runner in runners]
  )

  # All operations should complete without errors
  # The cleanup results depend on timing, but at least one should be True
  assert any(results)

  # Verify final state - agent should be removed
  async with manager._lock:
    assert id(agent) not in manager._runners_by_agent


@pytest.mark.asyncio
async def test_lock_prevents_race_condition(manager, agent):
  """Test that the lock prevents race conditions during unregistration."""
  runner1 = testing_utils.InMemoryRunner(agent)
  runner2 = testing_utils.InMemoryRunner(agent)

  await manager.register_runner(agent, runner1)
  await manager.register_runner(agent, runner2)

  # Create a barrier to synchronize unregistration attempts
  barrier = asyncio.Barrier(2)

  async def unregister_with_barrier(runner):
    await barrier.wait()  # Wait for both to reach this point
    async with manager.unregister_runner(agent, runner) as should_cleanup:
      return should_cleanup

  # Unregister both runners concurrently
  results = await asyncio.gather(
      unregister_with_barrier(runner1), unregister_with_barrier(runner2)
  )

  # Exactly one should return True (the last one to complete)
  cleanup_count = sum(1 for result in results if result is True)
  assert cleanup_count == 1

  # Verify agent is removed
  async with manager._lock:
    assert id(agent) not in manager._runners_by_agent


@pytest.mark.asyncio
async def test_unregister_runner_context_manager_holds_lock(manager, agent):
  """Test that unregister_runner context manager holds lock until exit."""
  runner = testing_utils.InMemoryRunner(agent)
  await manager.register_runner(agent, runner)

  lock_acquired_during_context = False

  async def try_acquire_lock():
    nonlocal lock_acquired_during_context
    try:
      # Try to acquire lock with timeout
      await asyncio.wait_for(manager._lock.acquire(), timeout=0.1)
      lock_acquired_during_context = True
      manager._lock.release()
    except asyncio.TimeoutError:
      # Lock is held, which is expected
      pass

  async with manager.unregister_runner(agent, runner) as should_cleanup:
    # Try to acquire lock from another task
    await try_acquire_lock()
    assert should_cleanup is True

  # After context exits, lock should be released
  async with manager._lock:
    assert id(agent) not in manager._runners_by_agent


@pytest.mark.asyncio
async def test_get_agent_tool_manager_singleton():
  """Test that get_agent_tool_manager returns a singleton."""
  manager1 = get_agent_tool_manager()
  manager2 = get_agent_tool_manager()

  assert manager1 is manager2
  assert isinstance(manager1, AgentToolManager)


@pytest.mark.asyncio
async def test_register_same_runner_twice(manager, agent, runner):
  """Test registering the same runner twice for the same agent."""
  await manager.register_runner(agent, runner)
  await manager.register_runner(agent, runner)

  # Runner should only appear once in the set
  async with manager._lock:
    assert id(agent) in manager._runners_by_agent
    assert runner in manager._runners_by_agent[id(agent)]
    assert len(manager._runners_by_agent[id(agent)]) == 1


@pytest.mark.asyncio
async def test_unregister_same_runner_twice(manager, agent, runner):
  """Test unregistering the same runner twice."""
  await manager.register_runner(agent, runner)

  # First unregistration should return True
  async with manager.unregister_runner(agent, runner) as should_cleanup:
    assert should_cleanup is True

  # Second unregistration should also return True (runner not found)
  async with manager.unregister_runner(agent, runner) as should_cleanup:
    assert should_cleanup is True

  # Verify agent is removed
  async with manager._lock:
    assert id(agent) not in manager._runners_by_agent
