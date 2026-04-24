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
from typing import Optional

from typing_extensions import deprecated
from typing_extensions import override

from ..events.event import Event
from ..utils.context_utils import Aclosing
from .base_agent import BaseAgent
from .base_agent import BaseAgentState
from .base_agent_config import BaseAgentConfig
from .invocation_context import InvocationContext
from .parallel_agent_config import ParallelAgentConfig

logger = logging.getLogger('google_adk.' + __name__)


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


async def _iter_with_idle_timeout(
    gen: AsyncGenerator[Event, None],
    timeout_secs: float,
    branch_name: str,
) -> AsyncGenerator[Event, None]:
  """Wrap *gen*, raising TimeoutError if no event arrives within *timeout_secs*.

  Uses asyncio.wait_for on each __anext__ call so that a branch whose upstream
  model stream silently stalls (connection open, no chunks) is detected and
  cancelled rather than hanging the parent ParallelAgent indefinitely.
  """
  while True:
    try:
      event = await asyncio.wait_for(
          gen.__anext__(),
          timeout=timeout_secs,
      )
    except StopAsyncIteration:
      return
    except asyncio.TimeoutError as exc:
      logger.warning(
          'ParallelAgent branch %r has not produced an event for %.1fs. '
          'The upstream model stream may have stalled. Cancelling the branch '
          'so the parent agent is not blocked indefinitely.',
          branch_name,
          timeout_secs,
      )
      raise asyncio.TimeoutError(
          f'Branch {branch_name!r} idle for >{timeout_secs}s without an event'
      ) from exc
    yield event


async def _merge_agent_run(
    agent_runs: list[AsyncGenerator[Event, None]],
    *,
    branch_names: list[str] | None = None,
    branch_idle_timeout_secs: float | None = None,
) -> AsyncGenerator[Event, None]:
  """Merges agent runs using asyncio.TaskGroup on Python 3.11+."""
  sentinel = object()
  queue: asyncio.Queue = asyncio.Queue()
  names = branch_names or [f'branch-{i}' for i in range(len(agent_runs))]

  # Agents are processed in parallel.
  # Events for each agent are put on queue sequentially.
  async def process_an_agent(events_for_one_agent, branch_name: str):
    try:
      gen = (
          _iter_with_idle_timeout(
              events_for_one_agent, branch_idle_timeout_secs, branch_name
          )
          if branch_idle_timeout_secs is not None
          else events_for_one_agent
      )
      async for event in gen:
        resume_signal = asyncio.Event()
        # put_nowait: the queue is unbounded so this never blocks, and it is
        # safe to call from a finally block that may run during cancellation.
        queue.put_nowait((event, resume_signal))
        # Wait for upstream to consume event before generating new events.
        await resume_signal.wait()
    finally:
      # Mark agent as finished. put_nowait is cancellation-safe (see above).
      queue.put_nowait((sentinel, None))

  async with asyncio.TaskGroup() as tg:
    for events_for_one_agent, name in zip(agent_runs, names):
      tg.create_task(process_an_agent(events_for_one_agent, name))

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
    *,
    branch_names: list[str] | None = None,
    branch_idle_timeout_secs: float | None = None,
) -> AsyncGenerator[Event, None]:
  """Merges agent runs for Python 3.10 without asyncio.TaskGroup.

  Uses custom cancellation and exception handling to mirror TaskGroup
  semantics. Each agent waits until the runner processes emitted events.

  Args:
      agent_runs: Async generators that yield events from each agent.
      branch_names: Optional names for each branch, used in log messages.
      branch_idle_timeout_secs: If set, cancel a branch that produces no event
          for this many seconds (guards against silently stalled model streams).

  Yields:
      Event: The next event from the merged generator.
  """
  sentinel = object()
  queue: asyncio.Queue = asyncio.Queue()
  names = branch_names or [f'branch-{i}' for i in range(len(agent_runs))]

  def propagate_exceptions(tasks):
    # Propagate exceptions and errors from tasks.
    for task in tasks:
      if task.done():
        # Ignore the result (None) of correctly finished tasks and re-raise
        # exceptions and errors.
        task.result()

  # Agents are processed in parallel.
  # Events for each agent are put on queue sequentially.
  async def process_an_agent(events_for_one_agent, branch_name: str):
    try:
      gen = (
          _iter_with_idle_timeout(
              events_for_one_agent, branch_idle_timeout_secs, branch_name
          )
          if branch_idle_timeout_secs is not None
          else events_for_one_agent
      )
      async for event in gen:
        resume_signal = asyncio.Event()
        queue.put_nowait((event, resume_signal))
        # Wait for upstream to consume event before generating new events.
        await resume_signal.wait()
    finally:
      # put_nowait is cancellation-safe: the queue is unbounded so it never
      # blocks, and it will not raise even if the task is being cancelled.
      queue.put_nowait((sentinel, None))

  tasks = []
  try:
    for events_for_one_agent, name in zip(agent_runs, names):
      tasks.append(
          asyncio.create_task(process_an_agent(events_for_one_agent, name))
      )

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
    # A task may have put its sentinel AND raised an exception; if the sentinel
    # was consumed before propagate_exceptions ran in the loop body, the error
    # would be silently lost.  Check once more after the loop to surface it.
    propagate_exceptions(tasks)
  finally:
    for task in tasks:
      task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


@deprecated(
    'ParallelAgent is deprecated and will be removed in future versions.'
    ' Please use Workflow instead.'
)
class ParallelAgent(BaseAgent):
  """A shell agent that runs its sub-agents in parallel in an isolated manner.

  This approach is beneficial for scenarios requiring multiple perspectives or
  attempts on a single task, such as:

  - Running different algorithms simultaneously.
  - Generating multiple responses for review by a subsequent evaluation agent.

  .. deprecated::
    ParallelAgent is deprecated and will be removed in future versions.
    Please use Workflow instead.

  Attributes:
      branch_idle_timeout_secs: Optional per-branch idle timeout in seconds.
          When set, any branch that produces no event for this many seconds is
          cancelled and raises ``asyncio.TimeoutError``, which unblocks the
          parent agent instead of hanging indefinitely.  This guards against
          upstream model streams that stall silently (connection open, no
          chunks arriving).  ``None`` (the default) disables the timeout and
          preserves the original unbounded-wait behaviour.
  """

  config_type: ClassVar[type[BaseAgentConfig]] = ParallelAgentConfig
  """The config type for this agent.

  DEPRECATED: This attribute is deprecated and will be removed in a future
  version, along with the AgentConfig YAML loader.
  """

  branch_idle_timeout_secs: Optional[float] = None
  """Per-branch idle timeout in seconds; None disables the guard."""

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
    branch_names = []
    # Prepare and collect async generators for each sub-agent.
    for sub_agent in self.sub_agents:
      sub_agent_ctx = _create_branch_ctx_for_sub_agent(self, sub_agent, ctx)

      # Only include sub-agents that haven't finished in a previous run.
      if not sub_agent_ctx.end_of_agents.get(sub_agent.name):
        agent_runs.append(sub_agent.run_async(sub_agent_ctx))
        branch_names.append(sub_agent.name)

    pause_invocation = False
    try:
      merge_func = (
          _merge_agent_run
          if sys.version_info >= (3, 11)
          else _merge_agent_run_pre_3_11
      )
      async with Aclosing(
          merge_func(
              agent_runs,
              branch_names=branch_names,
              branch_idle_timeout_secs=self.branch_idle_timeout_secs,
          )
      ) as agen:
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
