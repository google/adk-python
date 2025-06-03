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

"""Parallel agent implementation."""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator

from typing_extensions import override

from ..agents.invocation_context import InvocationContext
from ..events.event import Event
from .base_agent import BaseAgent


def _create_branch_ctx_for_sub_agent(
    agent: BaseAgent,
    sub_agent: BaseAgent,
    invocation_context: InvocationContext,
) -> InvocationContext:
  """Create isolated branch for every sub-agent."""
  invocation_context = invocation_context.model_copy()
  branch_suffix = f"{agent.name}.{sub_agent.name}"
  invocation_context.branch = (
      f"{invocation_context.branch}.{branch_suffix}"
      if invocation_context.branch
      else branch_suffix
  )
  return invocation_context


async def _merge_agent_run(
    agent_runs: list[AsyncGenerator[Event, None]],
) -> AsyncGenerator[Event, None]:
  """Merges the agent run event generator.

  This implementation guarantees for each agent, it won't move on until the
  generated event is processed by upstream runner.

  Args:
      agent_runs: A list of async generators that yield events from each agent.

  Yields:
      Event: The next event from the merged generator.
  """
  tasks = [
      asyncio.create_task(events_for_one_agent.__anext__())
      for events_for_one_agent in agent_runs
  ]
  pending_tasks = set(tasks)

  while pending_tasks:
    done, pending_tasks = await asyncio.wait(
        pending_tasks, return_when=asyncio.FIRST_COMPLETED
    )
    for task in done:
      try:
        yield task.result()

        # Find the generator that produced this event and move it on.
        for i, original_task in enumerate(tasks):
          if task == original_task:
            new_task = asyncio.create_task(agent_runs[i].__anext__())
            tasks[i] = new_task
            pending_tasks.add(new_task)
            break  # stop iterating once found

      except StopAsyncIteration:
        continue


class ParallelAgent(BaseAgent):
  """A shell agent that run its sub-agents in parallel in isolated manner.

  This approach is beneficial for scenarios requiring multiple perspectives or
  attempts on a single task, such as:

  - Running different algorithms simultaneously.
  - Generating multiple responses for review by a subsequent evaluation agent.
  """

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    agent_runs = [
        sub_agent.run_async(
            _create_branch_ctx_for_sub_agent(self, sub_agent, ctx)
        )
        for sub_agent in self.sub_agents
    ]
    async for event in _merge_agent_run(agent_runs):
      yield event

  @override
  async def _run_live_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    raise NotImplementedError("This is not supported yet for ParallelAgent.")
    yield  # AsyncGenerator requires having at least one yield statement
