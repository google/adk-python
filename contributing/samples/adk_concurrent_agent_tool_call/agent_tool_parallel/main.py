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

import os
import sys

SAMPLES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if SAMPLES_DIR not in sys.path:
  sys.path.append(SAMPLES_DIR)

import asyncio
import time
from typing import Any

from adk_concurrent_agent_tool_call.agent_tool_parallel import agent
from adk_concurrent_agent_tool_call.mock_tools import MockMcpTool
from google.adk.agents.run_config import RunConfig
from google.adk.events.event import Event
from google.adk.runners import InMemoryRunner
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

# Track running tools using monkey patch
running_tools: dict[str, MockMcpTool] = {}

# Track running tasks using monkey patch
running_tasks: dict[str, asyncio.Task[Any]] = {}

async def main():
  """Tests parallel AgentTool call behavior with shared agents.

  This test verifies the scenario where:
  1. Root agent calls sub-agent via AgentTool multiple times in parallel
  2. Each AgentTool call creates a runner that uses the same sub-agent
  3. Each sub-agent runner calls tools from the shared toolset concurrently
  4. When one AgentTool call completes and its runner closes, other parallel
     calls should not be interrupted

  This demonstrates that AgentToolManager properly handles parallel execution
  of AgentTool calls that share the same agent.
  """
  app_name = "adk_agent_tool_parallel_app"
  user_id = "adk_agent_tool_parallel_user"

  trigger_count = 0

  # Event to wait for both tool call requests to be made
  tool_call_request_event = asyncio.Event()

  def trigger_tool_call_request():
    """Trigger the tool call request event."""
    nonlocal trigger_count
    trigger_count += 1
    if trigger_count >= 2:
      tool_call_request_event.set()

  # Create runner with root agent
  runner = InMemoryRunner(
      agent=agent.root_agent,
      app_name=app_name,
  )

  session = await runner.session_service.create_session(
      app_name=app_name, user_id=user_id
  )

  # Monkey patch __call_tool_async to track running tools
  from google.adk.flows.llm_flows import functions
  original_call_tool_async = functions.__call_tool_async
  async def patched_call_tool_async(
      tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    """Patched version that tracks running tools."""
    if isinstance(tool, MockMcpTool):
      running_tools[tool_context.state['task_id']] = tool
      print(f"Tool {tool.name} started for session {tool_context.session.id}")
      trigger_tool_call_request()
    return await original_call_tool_async(tool, args, tool_context)

  functions.__call_tool_async = patched_call_tool_async

  # Monkey patch AgentTool.run_async to track running task
  from google.adk.tools.agent_tool import AgentTool
  original_run_async = AgentTool.run_async
  async def patched_run_async(self: AgentTool, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
    """Patched version that tracks running task."""
    task = asyncio.create_task(original_run_async(self, args=args, tool_context=tool_context))
    
    task_id = task.__hash__()
    tool_context.state['task_id'] = task_id
    running_tasks[task_id] = task
    return await task

  AgentTool.run_async = patched_run_async

  events: list[Event] = []
  try:
    async def run_agent():
      nonlocal events
      # Run agent with a prompt that triggers parallel AgentTool calls
      print("Starting agent with parallel AgentTool calls...")
      content = types.Content(
          role="user",
          parts=[
              types.Part.from_text(
                  text="Please call the sub_agent tool twice in parallel to help me."
              )
          ],
      )

      async for event in runner.run_async(
          user_id=session.user_id,
          session_id=session.id,
          new_message=content,
          run_config=RunConfig(save_input_blobs_as_artifacts=False),
      ):
        events.append(event)
        if event.content and event.content.parts:
          for part in event.content.parts:
            if part.text:
              print(f"Session {session.id}: {event.author}: {part.text}")
            if part.function_call:
              print(
                  f"Session {session.id}: {event.author}: function_call"
                  f" {part.function_call.name}"
              )
            if part.function_response:
              print(
                  f"Session {session.id}: {event.author}: function_response"
                  f" {part.function_response.name}"
              )
      return events
  
    runner_task = asyncio.create_task(run_agent())

    # Wait for both tools to start if they haven't already
    assert not runner_task.done(), "Runner should not be done"
    await tool_call_request_event.wait()

    print(f"Running tools: {list(running_tools.keys())}")

    # Get the running tools
    tool1_tuple = list(running_tools.items())[0]
    tool2_tuple = list(running_tools.items())[1]

    tool1_task = running_tasks[tool1_tuple[0]]
    tool2_task = running_tasks[tool2_tuple[0]]

    # Complete tool1
    print("Waiting for agent tool 1 to complete...")
    tool1_tuple[1].done_event.set()
    await tool1_task
    print("Tool1 completed ✓")

    await asyncio.sleep(0.1)
    
    print("Waiting for agent tool 2 to complete...")
    tool2_tuple[1].done_event.set()
    await tool2_task
    print("Tool2 completed ✓")

    await runner_task
    print(f"Agent completed with {len(events)} events ✓")

    # Check if any tools were interrupted
    has_error = any(
        event.content
        and event.content.parts
        and any(
            "interrupted" in str(part.function_response)
            or "interrupted" in str(part.text)
            for part in event.content.parts
        )
        for event in events
    )

    if has_error:
      print("⚠️ Some tools were interrupted during parallel execution")
    else:
      print("✅ All parallel AgentTool calls completed successfully")

  finally:
    # Restore original function
    functions.__call_tool_async = original_call_tool_async
    AgentTool.run_async = original_run_async
    print("Monkey patch restored ✓")


if __name__ == "__main__":
  start_time = time.time()
  print(
      "Script start time:",
      time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_time)),
  )
  print("=" * 50)
  print("Testing parallel AgentTool calls with shared agents")
  print("=" * 50)
  asyncio.run(main())
  end_time = time.time()
  print("=" * 50)
  print(
      "Script end time:",
      time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(end_time)),
  )
  print("Total script execution time:", f"{end_time - start_time:.2f} seconds")

