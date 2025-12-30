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

from adk_concurrent_agent_tool_call.mock_tools import MockMcpTool
from adk_concurrent_agent_tool_call.runner_shared_toolset import agent
from google.adk.agents.run_config import RunConfig
from google.adk.runners import InMemoryRunner
from google.adk.sessions.session import Session
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

# Track running tools using monkey patch
running_tools: dict[str, MockMcpTool] = {}


async def main():
  """Tests runner close behavior with shared toolsets.

  This test verifies the scenario where:
  1. Runner1 and Runner2 both use the same agent with a shared toolset
  2. Both runners call tools concurrently
  3. Runner1's tool completes and runner1 closes (which closes the shared toolset)
  4. Runner2's tool should not be interrupted when toolset is closed

  This demonstrates the issue: when a toolset is closed, all tools using that
  toolset are affected, even if they're being used by different runners.
  """
  app_name = "adk_parallel_agent_app"
  user_id_1 = "adk_parallel_user_1"
  user_id_2 = "adk_parallel_user_2"

  trigger_count = 0

  # Event to wait for both tool call requests to be made
  tool_call_request_event = asyncio.Event()

  def trigger_tool_call_request():
    """Trigger the tool call request event."""
    nonlocal trigger_count
    trigger_count += 1
    if trigger_count >= 2:
      tool_call_request_event.set()

  # Create two runners with the same agent (sharing the same toolset)
  runner1 = InMemoryRunner(
      agent=agent.root_agent,
      app_name=app_name,
  )
  runner2 = InMemoryRunner(
      agent=agent.root_agent,
      app_name=app_name,
  )

  session_1 = await runner1.session_service.create_session(
      app_name=app_name, user_id=user_id_1
  )
  session_2 = await runner2.session_service.create_session(
      app_name=app_name, user_id=user_id_2
  )

  # Monkey patch __call_tool_async to track running tools
  from google.adk.flows.llm_flows import functions

  original_call_tool_async = functions.__call_tool_async

  async def patched_call_tool_async(
      tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    """Patched version that tracks running tools."""
    if isinstance(tool, MockMcpTool):
      running_tools[tool_context.session.id] = tool
      print(f"Tool {tool.name} started for session {tool_context.session.id}")
      trigger_tool_call_request()
    return await original_call_tool_async(tool, args, tool_context)

  functions.__call_tool_async = patched_call_tool_async

  try:

    async def run_agent_prompt(
        runner: InMemoryRunner, session: Session, prompt_text: str
    ):
      """Run agent with a prompt and collect events."""
      content = types.Content(
          role="user", parts=[types.Part.from_text(text=prompt_text)]
      )
      events = []
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
              print(
                  f"Runner {runner.__hash__()} Session {session.id}:"
                  f" {event.author}: {part.text}"
              )
            if part.function_call:
              print(
                  f"Runner {runner.__hash__()} Session {session.id}:"
                  f" {event.author}: function_call {part.function_call.name}"
              )
            if part.function_response:
              print(
                  f"Runner {runner.__hash__()} Session {session.id}:"
                  f" {event.author}: function_response"
                  f" {part.function_response.name}"
              )
      return events

    # Start tool execution in runner1
    print("Starting runner tool execution...")
    runner1_task = asyncio.create_task(
        run_agent_prompt(
            runner1, session_1, "Please use the mcp_tool to help me."
        )
    )

    # Start tool execution in runner2
    runner2_task = asyncio.create_task(
        run_agent_prompt(
            runner2, session_2, "Please use the mcp_tool to help me."
        )
    )

    # Verify both runners are running
    assert not runner1_task.done(), "Runner1 should not be done"
    assert not runner2_task.done(), "Runner2 should not be done"

    # Wait to both tools are running
    await tool_call_request_event.wait()

    print(f"Running tools: {list(running_tools.keys())}")

    # Get the running tools
    runner1_tool = running_tools.get(session_1.id)
    runner2_tool = running_tools.get(session_2.id)

    if runner1_tool:
      print(f"Completing runner1's tool (session {session_1.id})...")
      runner1_tool.done_event.set()

    # Verify runner1 completed
    print("Waiting for runner1 to complete...")
    runner1_events = await runner1_task
    print(f"Runner1 completed with {len(runner1_events)} events ✓")

    # We are closing the runner1 here, this will close the toolset and interrupt the runner2's tool.
    # This may happen when you call 2 concurrent AgentTools of which origins are the same agent.
    await runner1.close()

    # Verify toolset was closed
    # assert agent.mcp_toolset.closed_event.is_set()
    # print("Toolset closed event is set ✓")

    # Complete runner2's tool if it's still running
    if runner2_tool:
      print(f"Completing runner2's tool (session {session_2.id})...")
      runner2_tool.done_event.set()

    # Wait for runner2's task to complete
    print("Waiting for runner2 to complete...")
    runner2_events = await runner2_task
    print(f"Runner2 completed with {len(runner2_events)} events")

    # Check if runner2's tool was interrupted
    has_error = any(
        event.content
        and event.content.parts
        and any(
            "interrupted" in str(part.function_response)
            or "interrupted" in str(part.text)
            for part in event.content.parts
        )
        for event in runner2_events
    )

    if has_error:
      print("Runner2's tool was interrupted by toolset close")
    else:
      print(
          "Runner2's tool completed normally (may have finished before close) ✓"
      )

    # Clean up runner2
    await runner2.close()
    print("All runners closed ✓")

  finally:
    # Restore original function
    functions.__call_tool_async = original_call_tool_async
    print("Monkey patch restored ✓")


if __name__ == "__main__":
  start_time = time.time()
  print(
      "Script start time:",
      time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_time)),
  )
  print("=" * 50)
  print("Testing runner close with shared toolsets")
  print("=" * 50)
  asyncio.run(main())
  end_time = time.time()
  print("=" * 50)
  print(
      "Script end time:",
      time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(end_time)),
  )
  print("Total script execution time:", f"{end_time - start_time:.2f} seconds")
