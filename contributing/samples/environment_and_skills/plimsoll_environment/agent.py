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

"""An agent that runs generated code in a self-hosted Plimsoll sandbox."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.environment import BaseEnvironment
from google.adk.integrations.plimsoll import PlimsollEnvironment
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.environment import EnvironmentToolset


class SandboxPerSession(BaseToolset):
  """The environment tools, with a sandbox of their own for each ADK session.

  ADK closes an agent's toolsets whenever a Runner that holds the agent
  closes, and `adk eval` runs each eval case in a Runner of its own, several
  at once, over this one agent. A single PlimsollEnvironment would be closed
  when the first case ends, and it refuses to reopen, so the next case would
  fail. Here each session gets an environment of its own, and a close spares
  the sessions whose agent run has not ended.
  """

  def __init__(self, new_environment: Callable[[], BaseEnvironment]):
    super().__init__()
    self._new_environment = new_environment
    self._toolsets: dict[str, EnvironmentToolset] = {}
    self._running: set[str] = set()

  def _toolset(self, session_id: str) -> EnvironmentToolset:
    if session_id not in self._toolsets:
      self._toolsets[session_id] = EnvironmentToolset(
          environment=self._new_environment()
      )
    return self._toolsets[session_id]

  async def get_tools(self, readonly_context=None):
    if readonly_context is None:
      # Listing the tools only, as `adk web` does to draw the agent graph.
      toolset = EnvironmentToolset(environment=self._new_environment())
    else:
      toolset = self._toolset(readonly_context.session.id)
    return await toolset.get_tools(readonly_context)

  async def process_llm_request(self, *, tool_context, llm_request) -> None:
    await self._toolset(tool_context.session.id).process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

  def run_started(self, callback_context: CallbackContext) -> None:
    self._running.add(callback_context.session.id)

  def run_ended(self, callback_context: CallbackContext) -> None:
    self._running.discard(callback_context.session.id)

  async def close(self) -> None:
    idle = [
        toolset
        for session_id, toolset in self._toolsets.items()
        if session_id not in self._running
    ]
    results = await asyncio.gather(
        *(toolset.close() for toolset in idle), return_exceptions=True
    )
    for result in results:
      if isinstance(result, BaseException):
        raise result


# Reads PLIMSOLL_URL and PLIMSOLL_TOKEN. Refuses a sandbox weaker than the
# kernel tier, such as Docker without gVisor.
sandboxes = SandboxPerSession(
    lambda: PlimsollEnvironment(minimum_isolation="kernel")
)

root_agent = Agent(
    name="plimsoll_coding_agent",
    description=(
        "A coding agent that executes commands and edits files in a"
        " self-hosted Plimsoll sandbox."
    ),
    instruction="""\
You are a coding assistant. Work only through the environment tools, which run
inside a Plimsoll sandbox rather than on the user's machine. The sandbox has
no network access.

For each request:
1. Inspect relevant files before changing them.
2. Write the smallest script or data file needed to solve the request.
3. Execute the script and use its actual output in your answer.
4. If a command fails, read stderr, fix the problem, and retry.
""",
    tools=[sandboxes],
    before_agent_callback=sandboxes.run_started,
    after_agent_callback=sandboxes.run_ended,
)
