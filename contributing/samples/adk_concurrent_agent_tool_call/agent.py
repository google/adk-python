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

# pylint: disable=g-importing-member

import asyncio
from typing import Any

from google.adk import Agent
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.tool_context import ToolContext
from google.genai.types import FunctionDeclaration


class MockTool(BaseTool):
  """A mock tool that waits for a closed event before completing."""

  def __init__(
      self,
      name: str,
      closed_event: asyncio.Event,
  ):
    super().__init__(name=name, description=f"Mock tool {name}")
    self.closed_event = closed_event
    self.done_event = asyncio.Event()

  def _get_declaration(self) -> FunctionDeclaration:
    return FunctionDeclaration(
        name=self.name,
        description=self.description,
    )

  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> dict[str, str]:
    """Runs the tool, checking if toolset is closed during execution."""
    # Check if toolset is closed before starting
    if self.closed_event.is_set():
      raise RuntimeError(f"Tool {self.name} cannot run: toolset is closed")

    closed_event_task = asyncio.create_task(self.closed_event.wait())
    done_event_task = asyncio.create_task(self.done_event.wait())

    await asyncio.wait(
        [closed_event_task, done_event_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # Check if toolset was closed during execution
    if self.closed_event.is_set():
      raise RuntimeError(
          f"Tool {self.name} interrupted: toolset was closed during execution"
      )

    # Tool completed successfully
    return {"result": f"Tool {self.name} completed successfully"}


class MockMcpToolset(BaseToolset):
  """A mock MCP toolset with a closed event.
  This toolset is used to test the runner close behavior when a MCP toolset is used.
  """

  def __init__(self):
    super().__init__()
    self.closed_event = asyncio.Event()

  async def get_tools(self, readonly_context=None) -> list[BaseTool]:
    """Returns a single mock tool."""
    # Note that if you cache the tool, there is no such issue since the tool is reused.
    # e.g. `return [self._tool]`
    # However, MCP is a stateful protocol, so the tool should not be reused.
    return [MockTool(
        name="mcp_tool",
        closed_event=self.closed_event,
    )]

  async def close(self) -> None:
    """Closes the toolset by setting the closed event."""
    print(f"Closing toolset {self.__hash__()}")
    self.closed_event.set()


# Create a MCP toolset
mcp_toolset = MockMcpToolset()

system_prompt = """
You are a helpful assistant that can use tools to help users.
When asked to use the mcp_tool, you should call it.
"""

root_agent = Agent(
    model="gemini-2.5-flash",
    name="parallel_agent",
    description="An agent that uses a MCP toolset for testing runner close behavior.",
    instruction=system_prompt,
    tools=[mcp_toolset],
)
