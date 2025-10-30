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

"""Loop agent implementation."""

from __future__ import annotations

import logging
from typing import Any
from typing import AsyncGenerator
from typing import ClassVar
from typing import Dict
from typing import Optional

from typing_extensions import override

from ..events.event import Event
from ..utils.context_utils import Aclosing
from ..utils.feature_decorator import experimental
from .base_agent import BaseAgent
from .base_agent import BaseAgentState
from .base_agent_config import BaseAgentConfig
from .invocation_context import InvocationContext
from .loop_agent_config import LoopAgentConfig

logger = logging.getLogger('google_adk.' + __name__)


@experimental
class LoopAgentState(BaseAgentState):
  """State for LoopAgent."""

  current_sub_agent: str = ''
  """The name of the current sub-agent to run in the loop."""

  times_looped: int = 0
  """The number of times the loop agent has looped."""


class LoopAgent(BaseAgent):
  """A shell agent that run its sub-agents in a loop.

  When sub-agent generates an event with escalate or max_iterations are
  reached, the loop agent will stop.
  """

  config_type: ClassVar[type[BaseAgentConfig]] = LoopAgentConfig
  """The config type for this agent."""

  max_iterations: Optional[int] = None
  """The maximum number of iterations to run the loop agent.

  If not set, the loop agent will run indefinitely until a sub-agent
  escalates.
  """

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    if not self.sub_agents:
      return

    agent_state = self._load_agent_state(ctx, LoopAgentState)
    is_resuming_at_current_agent = agent_state is not None
    times_looped, start_index = self._get_start_state(agent_state)

    should_exit = False
    pause_invocation = False
    while (
        not self.max_iterations or times_looped < self.max_iterations
    ) and not (should_exit or pause_invocation):
      for i in range(start_index, len(self.sub_agents)):
        sub_agent = self.sub_agents[i]

        if ctx.is_resumable and not is_resuming_at_current_agent:
          # If we are resuming from the current event, it means the same event
          # has already been logged, so we should avoid yielding it again.
          agent_state = LoopAgentState(
              current_sub_agent=sub_agent.name,
              times_looped=times_looped,
          )
          ctx.set_agent_state(self.name, agent_state=agent_state)
          yield self._create_agent_state_event(ctx)

        is_resuming_at_current_agent = False

        async with Aclosing(sub_agent.run_async(ctx)) as agen:
          async for event in agen:
            yield event
            if event.actions.escalate:
              should_exit = True
            if ctx.should_pause_invocation(event):
              pause_invocation = True

        if should_exit or pause_invocation:
          break  # break inner for loop

      # Restart from the beginning of the loop.
      start_index = 0
      times_looped += 1
      # Reset the state of all sub-agents in the loop.
      ctx.reset_sub_agent_states(self.name)

    # If the invocation is paused, we should not yield the end of agent event.
    if pause_invocation:
      return

    if ctx.is_resumable:
      ctx.set_agent_state(self.name, end_of_agent=True)
      yield self._create_agent_state_event(ctx)

  def _get_start_state(
      self,
      agent_state: Optional[LoopAgentState],
  ) -> tuple[int, int]:
    """Computes the start state of the loop agent from the agent state."""
    if not agent_state:
      return 0, 0

    times_looped = agent_state.times_looped
    start_index = 0
    if agent_state.current_sub_agent:
      try:
        sub_agent_names = [sub_agent.name for sub_agent in self.sub_agents]
        start_index = sub_agent_names.index(agent_state.current_sub_agent)
      except ValueError:
        # A sub-agent was removed so the agent name is not found.
        # For now, we restart from the beginning.
        logger.warning(
            'Sub-agent %s was not found. Restarting from the beginning.',
            agent_state.current_sub_agent,
        )
    return times_looped, start_index

  @override
  async def _run_live_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    raise NotImplementedError('This is not supported yet for LoopAgent.')
    yield  # AsyncGenerator requires having at least one yield statement

  @override
  @classmethod
  @experimental
  def _parse_config(
      cls: type[LoopAgent],
      config: LoopAgentConfig,
      config_abs_path: str,
      kwargs: Dict[str, Any],
  ) -> Dict[str, Any]:
    if config.max_iterations:
      kwargs['max_iterations'] = config.max_iterations
    return kwargs
