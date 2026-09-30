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

import asyncio
from typing import Any
from typing import Dict

from google.adk.agents.llm_agent import Agent
from google.adk.tools.function_tool import FunctionTool
from google.genai import types


async def slow_background_task(task_description: str) -> Dict[str, Any]:
  """Performs a long-running background computation or data retrieval.

  Args:
    task_description: Description of the task to run in the background.

  Returns:
    A dictionary containing the completion status and task result.
  """
  print(f"[Tool] Starting slow background task: {task_description}")
  # Simulate a 10-second non-blocking background operation
  await asyncio.sleep(10)
  print(f"[Tool] Completed slow background task: {task_description}")
  return {
      "status": "completed",
      "task": task_description,
      "result": "Background task finished successfully after 10 seconds.",
  }


# Create a FunctionTool wrapping the long-running async function
non_blocking_tool = FunctionTool(slow_background_task)

# Configure response_scheduling to indicate non-blocking behavior for Live mode.
# Options: WHEN_IDLE, SILENT, or INTERRUPT.
non_blocking_tool.response_scheduling = (
    types.FunctionResponseScheduling.WHEN_IDLE
)


root_agent = Agent(
    model="gemini-live-2.5-flash-native-audio",
    name="non_blocking_tool_agent",
    description=(
        "Agent demonstrating non-blocking tool execution in ADK Live mode."
    ),
    instruction="""
      You are a helpful assistant for testing live mode non-blocking tool execution.

      You have access to a tool `slow_background_task` which is configured with
      NON_BLOCKING response scheduling (WHEN_IDLE).

      When the user asks you to run a long-running or background task, call the `slow_background_task` tool.
      Inform the user that the task has started and continue conversing with them normally while the task runs in the background.
    """,
    tools=[
        non_blocking_tool,
    ],
)
