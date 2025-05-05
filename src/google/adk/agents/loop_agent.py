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

from google.genai import types
from typing import AsyncGenerator
from typing import Optional

from typing_extensions import override

from ..agents.invocation_context import InvocationContext
from ..events.event import Event
from .base_agent import BaseAgent


class LoopAgent(BaseAgent):
  """A shell agent that run its sub-agents in a loop.

  When sub-agent generates an event with escalate or max_iterations are
  reached, the loop agent will stop.
  """

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
        async for event in sub_agent.run_async(ctx):
          yield event  # Yield the event immediately
          if event.actions.escalate:
            # Ensure the escalation message is processed immediately
            if event.content and event.content.parts:
              for part in event.content.parts:
                if part.function_response.response:
                  # Yield the escalation message as a new event
                  yield Event(
                      invocation_id=ctx.invocation_id,
                      author=sub_agent.name,
                      branch=ctx.branch,
                      content=types.Content(
                          role='assistant',
                          parts=[
                              types.Part(
                                  text=part.function_response.response.get(
                                      'message'
                                  ),
                              )
                          ],
                      ),
                  )
            return  # Exit the loop after escalation
      times_looped += 1
    return

  @override
  async def _run_live_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    raise NotImplementedError('The behavior for run_live is not defined yet.')
    yield  # AsyncGenerator requires having at least one yield statement
