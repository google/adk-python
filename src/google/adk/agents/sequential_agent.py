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

"""Sequential agent implementation."""

from __future__ import annotations

from typing import AsyncGenerator
from typing import ClassVar
from typing import Type

from typing_extensions import override

from ..events.event import Event
from ..utils.context_utils import Aclosing
from ..utils.feature_decorator import experimental
from .base_agent import BaseAgent
from .base_agent import BaseAgentState
from .base_agent_config import BaseAgentConfig
from .invocation_context import InvocationContext
from .llm_agent import LlmAgent
from .sequential_agent_config import SequentialAgentConfig


@experimental
class SequentialAgentState(BaseAgentState):
  """State for SequentialAgent."""

  current_sub_agent: str = ''
  """The name of the current sub-agent to run."""


class SequentialAgent(BaseAgent):
  """A shell agent that runs its sub-agents in sequence."""

  config_type: ClassVar[Type[BaseAgentConfig]] = SequentialAgentConfig
  """The config type for this agent."""

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    for sub_agent in self.sub_agents:
      async with Aclosing(sub_agent.run_async(ctx)) as agen:
        async for event in agen:
          yield event

  @override
  async def _run_live_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    """Implementation for live SequentialAgent.

    Compared to the non-live case, live agents process a continuous stream of audio
    or video, so there is no way to tell if it's finished and should pass
    to the next agent or not. So we introduce a task_completed() function so the
    model can call this function to signal that it's finished the task and we
    can move on to the next agent.

    Args:
      ctx: The invocation context of the agent.
    """
    # There is no way to know if it's using live during init phase so we have to init it here
    for sub_agent in self.sub_agents:
      # add tool
      def task_completed():
        """
        Signals that the agent has successfully completed the user's question
        or task.
        """
        return 'Task completion signaled.'

      if isinstance(sub_agent, LlmAgent):
        # Use function name to dedupe.
        if task_completed.__name__ not in sub_agent.tools:
          sub_agent.tools.append(task_completed)
          sub_agent.instruction += f"""If you finished the user's request
          according to its description, call the {task_completed.__name__} function
          to exit so the next agents can take over. When calling this function,
          do not generate any text other than the function call."""

    for sub_agent in self.sub_agents:
      async with Aclosing(sub_agent.run_live(ctx)) as agen:
        async for event in agen:
          yield event
