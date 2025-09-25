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

from typing import Any
from typing import AsyncGenerator
from typing import ClassVar
from typing import Dict
from typing import Optional

from typing_extensions import override

from ..agents.invocation_context import InvocationContext
from ..events.event import Event
from ..utils.context_utils import Aclosing
from ..utils.feature_decorator import experimental
from .base_agent import BaseAgent
from .base_agent import BaseAgentState
from .base_agent_config import BaseAgentConfig
from .loop_agent_config import LoopAgentConfig


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
    times_looped = 0
    while not self.max_iterations or times_looped < self.max_iterations:
      for sub_agent in self.sub_agents:
        should_exit = False
        async with Aclosing(sub_agent.run_async(ctx)) as agen:
          async for event in agen:
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
