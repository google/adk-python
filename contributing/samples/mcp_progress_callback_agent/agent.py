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

"""Sample agent demonstrating MCP progress callback feature.

This sample shows how to use the progress_callback parameter in McpToolset
to receive progress notifications from MCP servers during long-running tool
executions.

There are two ways to use progress callbacks:

1. Simple callback (shared by all tools):
   Pass a ProgressFnT callback that receives (progress, total, message).

2. Factory function (per-tool callbacks with runtime context):
   Pass a ProgressCallbackFactory that takes (tool_name, callback_context, **kwargs)
   and returns a ProgressFnT or None. This allows different tools to have different
   progress handling logic, and the factory can access and modify session state
   via the CallbackContext. The **kwargs ensures forward compatibility for future
   parameters.

IMPORTANT: Progress callbacks only work when the MCP server actually sends
progress notifications. Most simple MCP servers (like the filesystem server)
do not send progress updates. This sample uses a mock server that demonstrates
progress reporting.

Usage:
  adk run contributing/samples/mcp_progress_callback_agent

Then try:
  "Run the long running task with 5 steps"
  "Process these items: apple, banana, cherry"
"""

import os
import sys
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool import StdioConnectionParams
from mcp import StdioServerParameters
from mcp.shared.session import ProgressFnT

_current_dir = os.path.dirname(os.path.abspath(__file__))
_mock_server_path = os.path.join(_current_dir, "mock_progress_server.py")


# Option 1: Simple shared callback
async def simple_progress_callback(
    progress: float,
    total: float | None,
    message: str | None,
) -> None:
  """Handle progress notifications from MCP server.

  This callback is shared by all tools in the toolset.
  """
  if total is not None:
    percentage = (progress / total) * 100
    bar_length = 20
    filled = int(bar_length * progress / total)
    bar = "=" * filled + "-" * (bar_length - filled)
    print(f"[{bar}] {percentage:.0f}% ({progress}/{total}) {message or ''}")
  else:
    print(f"Progress: {progress} {f'- {message}' if message else ''}")


# Option 2: Factory function for per-tool callbacks with runtime context
def progress_callback_factory(
    tool_name: str,
    *,
    callback_context: CallbackContext | None = None,
    **kwargs: Any,
) -> ProgressFnT | None:
  """Create a progress callback for a specific tool.

  This factory allows different tools to have different progress handling.
  It receives a CallbackContext for accessing and modifying runtime information
  like session state. The **kwargs parameter ensures forward compatibility.

  Args:
    tool_name: The name of the MCP tool.
    callback_context: The callback context providing access to session,
      state, artifacts, and other runtime information. Allows modifying
      state via ctx.state['key'] = value. May be None if not available.
    **kwargs: Additional keyword arguments for future extensibility.

  Returns:
    A progress callback function, or None if no callback is needed.
  """
  # Example: Access session info from context (if available)
  session_id = "unknown"
  if callback_context and callback_context.session:
    session_id = callback_context.session.id

  async def callback(
      progress: float,
      total: float | None,
      message: str | None,
  ) -> None:
    # Include tool name and session info in the progress output
    prefix = f"[{tool_name}][session:{session_id}]"
    if total is not None:
      percentage = (progress / total) * 100
      bar_length = 20
      filled = int(bar_length * progress / total)
      bar = "=" * filled + "-" * (bar_length - filled)
      print(f"{prefix} [{bar}] {percentage:.0f}% {message or ''}")
      # Example: Store progress in state (callback_context allows modification)
      if callback_context:
        callback_context.state["last_progress"] = progress
        callback_context.state["last_total"] = total
    else:
      print(
          f"{prefix} Progress: {progress} {f'- {message}' if message else ''}"
      )

  return callback


root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="progress_demo_agent",
    instruction="""\
You are a helpful assistant that can run long-running tasks.

Available tools:
- long_running_task: Simulates a task with multiple steps. You can specify
  the number of steps and delay between them.
- process_items: Processes a list of items one by one with progress updates.

When the user asks you to run a task, use these tools and the progress
will be logged automatically.

Example requests:
- "Run a long task with 5 steps"
- "Process these items: apple, banana, cherry, date"
    """,
    tools=[
        McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command=sys.executable,  # Use current Python interpreter
                    args=[_mock_server_path],
                ),
                timeout=60,
            ),
            # Use factory function for per-tool callbacks (Option 2)
            # Or use simple_progress_callback for shared callback (Option 1)
            progress_callback=progress_callback_factory,
        )
    ],
)
