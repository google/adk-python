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

from typing import AsyncGenerator
from typing import ClassVar
from typing import Optional
from typing import Type

from typing_extensions import override

from ..agents.invocation_context import InvocationContext
from ..events.event import Event
from ..utils.feature_decorator import working_in_progress
from .base_agent import BaseAgent
from .base_agent_config import BaseAgentConfig
from .loop_agent_config import LoopAgentConfig


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
    times_looped = 0
    while not self.max_iterations or times_looped < self.max_iterations:
      for sub_agent in self.sub_agents:
        should_exit = False
        async for event in sub_agent.run_async(ctx):
          yield event
          if event.actions.escalate:
            should_exit = True

        if should_exit:
          return

      times_looped += 1
    return

  @override
  async def _run_live_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    raise NotImplementedError('This is not supported yet for LoopAgent.')
    yield  # AsyncGenerator requires having at least one yield statement

  @classmethod
  @override
  @working_in_progress('LoopAgent.from_config is not ready for use.')
  def from_config(
      cls: Type[LoopAgent],
      config: LoopAgentConfig,
      config_abs_path: str,
  ) -> LoopAgent:
    agent = super().from_config(config, config_abs_path)
    if config.max_iterations:
      agent.max_iterations = config.max_iterations
    return agent
