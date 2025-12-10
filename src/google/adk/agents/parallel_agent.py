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
import sys
from typing import AsyncGenerator
from typing import ClassVar

from typing_extensions import override

from ..events.event import Event
from ..utils.context_utils import Aclosing
from .base_agent import BaseAgent
from .base_agent import BaseAgentState
from .base_agent_config import BaseAgentConfig
from .invocation_context import InvocationContext
from .parallel_agent_config import ParallelAgentConfig


def _create_branch_ctx_for_sub_agent(
    agent: BaseAgent,
    sub_agent: BaseAgent,
    invocation_context: InvocationContext,
) -> InvocationContext:
  """Create isolated branch for every sub-agent."""
  invocation_context = invocation_context.model_copy()
  branch_suffix = f'{agent.name}.{sub_agent.name}'
  invocation_context.branch = (
      f'{invocation_context.branch}.{branch_suffix}'
      if invocation_context.branch
      else branch_suffix
  )
  return invocation_context


async def _merge_agent_run(
    agent_runs: list[AsyncGenerator[Event, None]],
) -> AsyncGenerator[Event, None]:
  """Merges agent runs using asyncio.TaskGroup on Python 3.11+."""
  sentinel = object()
  queue = asyncio.Queue()

  # Agents are processed in parallel.
  # Events for each agent are put on queue sequentially.
  async def process_an_agent(events_for_one_agent):
    try:
      async for event in events_for_one_agent:
        resume_signal = asyncio.Event()
        await queue.put((event, resume_signal))
        # Wait for upstream to consume event before generating new events.
        await resume_signal.wait()
    finally:
      # Mark agent as finished.
      await queue.put((sentinel, None))

  async with asyncio.TaskGroup() as tg:
    for events_for_one_agent in agent_runs:
      tg.create_task(process_an_agent(events_for_one_agent))

    sentinel_count = 0
    # Run until all agents finished processing.
    while sentinel_count < len(agent_runs):
      event, resume_signal = await queue.get()
      # Agent finished processing.
      if event is sentinel:
        sentinel_count += 1
      else:
        yield event
        # Signal to agent that it should generate next event.
        resume_signal.set()


# TODO - remove once Python <3.11 is no longer supported.
async def _merge_agent_run_pre_3_11(
    agent_runs: list[AsyncGenerator[Event, None]],
) -> AsyncGenerator[Event, None]:
  """Merges agent runs for Python 3.10 without asyncio.TaskGroup.

  Uses custom cancellation and exception handling to mirror TaskGroup
  semantics. Each agent waits until the runner processes emitted events.

  Args:
      agent_runs: Async generators that yield events from each agent.

  Yields:
      Event: The next event from the merged generator.
  """
  sentinel = object()
  queue = asyncio.Queue()

  def propagate_exceptions(tasks):
    # Propagate exceptions and errors from tasks.
    for task in tasks:
      if task.done():
        # Ignore the result (None) of correctly finished tasks and re-raise
        # exceptions and errors.
        task.result()

  # Agents are processed in parallel.
  # Events for each agent are put on queue sequentially.
  async def process_an_agent(events_for_one_agent):
    try:
      async for event in events_for_one_agent:
        resume_signal = asyncio.Event()
        await queue.put((event, resume_signal))
        # Wait for upstream to consume event before generating new events.
        await resume_signal.wait()
    finally:
      # Mark agent as finished.
      await queue.put((sentinel, None))

  tasks = []
  try:
    for events_for_one_agent in agent_runs:
      tasks.append(asyncio.create_task(process_an_agent(events_for_one_agent)))

    sentinel_count = 0
    # Run until all agents finished processing.
    while sentinel_count < len(agent_runs):
      propagate_exceptions(tasks)
      event, resume_signal = await queue.get()
      # Agent finished processing.
      if event is sentinel:
        sentinel_count += 1
      else:
        yield event
        # Signal to agent that event has been processed by runner and it can
        # continue now.
        resume_signal.set()
  finally:
    for task in tasks:
      task.cancel()


class ParallelAgent(BaseAgent):
  """A shell agent that runs its sub-agents in parallel in an isolated manner.

  This approach is beneficial for scenarios requiring multiple perspectives or
  attempts on a single task, such as:

  - Running different algorithms simultaneously.
  - Generating multiple responses for review by a subsequent evaluation agent.
  """

  config_type: ClassVar[type[BaseAgentConfig]] = ParallelAgentConfig
  """The config type for this agent."""

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    if not self.sub_agents:
      return

    agent_state = self._load_agent_state(ctx, BaseAgentState)
    if ctx.is_resumable and agent_state is None:
      ctx.set_agent_state(self.name, agent_state=BaseAgentState())
      yield self._create_agent_state_event(ctx)

    agent_runs = []
    # Prepare and collect async generators for each sub-agent.
    for sub_agent in self.sub_agents:
      sub_agent_ctx = _create_branch_ctx_for_sub_agent(self, sub_agent, ctx)

      # Only include sub-agents that haven't finished in a previous run.
      if not sub_agent_ctx.end_of_agents.get(sub_agent.name):
        agent_runs.append(sub_agent.run_async(sub_agent_ctx))

    pause_invocation = False
    try:
      merge_func = (
          _merge_agent_run
          if sys.version_info >= (3, 11)
          else _merge_agent_run_pre_3_11
      )
      async with Aclosing(merge_func(agent_runs)) as agen:
        async for event in agen:
          yield event
          if ctx.should_pause_invocation(event):
            pause_invocation = True

      if pause_invocation:
        return

      # Once all sub-agents are done, mark the ParallelAgent as final.
      if ctx.is_resumable and all(
          ctx.end_of_agents.get(sub_agent.name) for sub_agent in self.sub_agents
      ):
        ctx.set_agent_state(self.name, end_of_agent=True)
        yield self._create_agent_state_event(ctx)

    finally:
      for sub_agent_run in agent_runs:
        await sub_agent_run.aclose()

  @override
  async def _run_live_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    raise NotImplementedError('This is not supported yet for ParallelAgent.')
    yield  # AsyncGenerator requires having at least one yield statement
