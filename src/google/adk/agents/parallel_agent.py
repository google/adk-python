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

"""Parallel agent implementation."""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import AsyncGenerator
from typing import ClassVar

from typing_extensions import deprecated
from typing_extensions import override

from ..events._branch_path import _BranchPath
from ..events.event import Event
from ..utils.context_utils import Aclosing
from .base_agent import BaseAgent
from .base_agent import BaseAgentState
from .base_agent_config import BaseAgentConfig
from .invocation_context import InvocationContext
from .parallel_agent_config import ParallelAgentConfig

logger = logging.getLogger('google_adk.' + __name__)


class _AgentRunComplete:
  """Queue marker emitted after one parallel agent finishes."""


def _create_branch_ctx_for_sub_agent(
    agent: BaseAgent,
    sub_agent: BaseAgent,
    invocation_context: InvocationContext,
) -> InvocationContext:
  """Create isolated branch for every sub-agent."""
  invocation_context = invocation_context.model_copy()
  branch_suffix = f'{agent.name}.{sub_agent.name}'
  invocation_context.branch = _BranchPath.create_sub_branch(
      invocation_context.branch, name=branch_suffix
  )
  return invocation_context


def _has_escalate_action(event: Event) -> bool:
  """Returns whether the event asks the parent workflow to exit early."""
  return bool(event.actions.escalate)


def _cancel_tasks(tasks: list[asyncio.Task[None]]) -> None:
  """Cancels still-running merge tasks."""
  for task in tasks:
    if not task.done():
      task.cancel()


async def _merge_agent_run(
    agent_runs: list[AsyncGenerator[Event, None]],
) -> AsyncGenerator[Event, None]:
  """Merges agent runs using asyncio.TaskGroup on Python 3.11+."""
  sentinel = _AgentRunComplete()
  queue: asyncio.Queue[
      tuple[Event | _AgentRunComplete, asyncio.Event | BaseException | None]
  ] = asyncio.Queue()
  tasks: list[asyncio.Task[None]] = []

  # Agents are processed in parallel.
  # Events for each agent are put on queue sequentially.
  async def process_an_agent(
      events_for_one_agent: AsyncGenerator[Event, None],
  ) -> None:
    error: BaseException | None = None
    try:
      async with Aclosing(events_for_one_agent):
        async for event in events_for_one_agent:
          resume_signal = asyncio.Event()
          await queue.put((event, resume_signal))
          # Wait for upstream to consume event before generating new events.
          await resume_signal.wait()
    except asyncio.CancelledError:
      logger.info('Agent run cancelled.')
      raise
    except Exception as e:
      # Reported through the queue rather than by failing the task: this
      # generator stays suspended at `yield` for as long as the caller is busy,
      # and a task group aborting around a suspended frame cancels the caller
      # instead of handing it the error.
      error = e
    finally:
      # Mark agent as finished, carrying the failure that ended it, if any.
      try:
        await queue.put((sentinel, error))
      except Exception as e:
        logger.warning('Failed to put sentinel on queue: %s', e)

  async with asyncio.TaskGroup() as tg:
    for events_for_one_agent in agent_runs:
      tasks.append(tg.create_task(process_an_agent(events_for_one_agent)))

    sentinel_count = 0
    # Run until all agents finished processing.
    while sentinel_count < len(agent_runs):
      event, payload = await queue.get()
      # Agent finished processing.
      if isinstance(event, _AgentRunComplete):
        sentinel_count += 1
        if isinstance(payload, BaseException):
          raise payload
      else:
        yield event
        if _has_escalate_action(event):
          _cancel_tasks(tasks)
          return
        # Signal to agent that it should generate next event.
        if not isinstance(payload, asyncio.Event):
          raise RuntimeError(
              'Parallel-agent event is missing its resume signal.'
          )
        payload.set()


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
  sentinel = _AgentRunComplete()
  queue: asyncio.Queue[
      tuple[Event | _AgentRunComplete, asyncio.Event | BaseException | None]
  ] = asyncio.Queue()

  # Agents are processed in parallel.
  # Events for each agent are put on queue sequentially.
  async def process_an_agent(
      events_for_one_agent: AsyncGenerator[Event, None],
  ) -> None:
    error: BaseException | None = None
    try:
      async with Aclosing(events_for_one_agent):
        async for event in events_for_one_agent:
          resume_signal = asyncio.Event()
          await queue.put((event, resume_signal))
          # Wait for upstream to consume event before generating new events.
          await resume_signal.wait()
    except asyncio.CancelledError:
      # Cancellation is not a branch failure; keep it off the queue.
      raise
    except Exception as e:
      error = e
    finally:
      # Mark agent as finished, carrying the failure that ended it, if any.
      await queue.put((sentinel, error))

  tasks: list[asyncio.Task[None]] = []
  try:
    for events_for_one_agent in agent_runs:
      tasks.append(asyncio.create_task(process_an_agent(events_for_one_agent)))

    sentinel_count = 0
    # Run until all agents finished processing.
    while sentinel_count < len(agent_runs):
      event, payload = await queue.get()
      # Agent finished processing.
      if isinstance(event, _AgentRunComplete):
        sentinel_count += 1
        if isinstance(payload, BaseException):
          raise payload
      else:
        yield event
        if _has_escalate_action(event):
          _cancel_tasks(tasks)
          return
        # Signal to agent that event has been processed by runner and it can
        # continue now.
        if not isinstance(payload, asyncio.Event):
          raise RuntimeError(
              'Parallel-agent event is missing its resume signal.'
          )
        payload.set()
  finally:
    _cancel_tasks(tasks)
    if tasks:
      # Await cancellation so siblings are no longer mid-iteration when the
      # caller `aclose()`s them (else "generator is already running").
      await asyncio.gather(*tasks, return_exceptions=True)


@deprecated(
    'ParallelAgent is deprecated in favor of Workflow and will be removed in'
    ' a future version. Workflow cannot yet be used as an LlmAgent sub-agent.'
)
class ParallelAgent(BaseAgent):
  """A shell agent that runs its sub-agents in parallel in an isolated manner.

  This approach is beneficial for scenarios requiring multiple perspectives or
  attempts on a single task, such as:

  - Running different algorithms simultaneously.
  - Generating multiple responses for review by a subsequent evaluation agent.

  .. deprecated::
    ParallelAgent is deprecated in favor of Workflow and will be removed in a
    future version. Workflow cannot yet be used as an LlmAgent sub-agent.
  """

  config_type: ClassVar[type[BaseAgentConfig]] = ParallelAgentConfig
  """The config type for this agent.

  DEPRECATED: This attribute is deprecated and will be removed in a future
  version, along with the AgentConfig YAML loader.
  """

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

    escalated = False
    pause_invocation = False
    merge_func = (
        _merge_agent_run
        if sys.version_info >= (3, 11)
        else _merge_agent_run_pre_3_11
    )
    async with Aclosing(merge_func(agent_runs)) as agen:
      async for event in agen:
        yield event
        if _has_escalate_action(event):
          escalated = True
        if ctx.should_pause_invocation(event):
          pause_invocation = True

    if pause_invocation:
      return

    # Once all sub-agents are done, mark the ParallelAgent as final.
    if ctx.is_resumable and (
        escalated
        or all(
            ctx.end_of_agents.get(sub_agent.name)
            for sub_agent in self.sub_agents
        )
    ):
      ctx.set_agent_state(self.name, end_of_agent=True)
      yield self._create_agent_state_event(ctx)

  @override
  async def _run_live_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    raise NotImplementedError('This is not supported yet for ParallelAgent.')
    yield  # AsyncGenerator requires having at least one yield statement
