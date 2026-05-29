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

"""Mock MCP server that sends progress notifications.

This server demonstrates how MCP servers can send progress updates
during long-running tool execution.

Run this server directly:
  python mock_progress_server.py

Or use it with the sample agent:
  See agent_with_mock_server.py
"""

import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent
from mcp.types import Tool

server = Server("mock-progress-server")


@server.list_tools()
async def list_tools() -> list[Tool]:
  """List available tools."""
  return [
      Tool(
          name="long_running_task",
          description=(
              "A simulated long-running task that reports progress. "
              "Use this to test progress callback functionality."
          ),
          inputSchema={
              "type": "object",
              "properties": {
                  "steps": {
                      "type": "integer",
                      "description": "Number of steps to simulate (default: 5)",
                      "default": 5,
                  },
                  "delay": {
                      "type": "number",
                      "description": (
                          "Delay in seconds between steps (default: 0.5)"
                      ),
                      "default": 0.5,
                  },
              },
          },
      ),
      Tool(
          name="process_items",
          description="Process a list of items with progress reporting.",
          inputSchema={
              "type": "object",
              "properties": {
                  "items": {
                      "type": "array",
                      "items": {"type": "string"},
                      "description": "List of items to process",
                  },
              },
              "required": ["items"],
          },
      ),
  ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
  """Handle tool calls with progress reporting."""
  ctx = server.request_context

  if name == "long_running_task":
    steps = arguments.get("steps", 5)
    delay = arguments.get("delay", 0.5)

    # Get progress token from request metadata
    progress_token = None
    if ctx.meta and hasattr(ctx.meta, "progressToken"):
      progress_token = ctx.meta.progressToken

    for i in range(steps):
      # Simulate work
      await asyncio.sleep(delay)

      # Send progress notification if client supports it
      if progress_token is not None:
        await ctx.session.send_progress_notification(
            progress_token=progress_token,
            progress=i + 1,
            total=steps,
            message=f"Completed step {i + 1} of {steps}",
        )

    return [
        TextContent(
            type="text",
            text=f"Successfully completed {steps} steps!",
        )
    ]

  elif name == "process_items":
    items = arguments.get("items", [])
    total = len(items)

    progress_token = None
    if ctx.meta and hasattr(ctx.meta, "progressToken"):
      progress_token = ctx.meta.progressToken

    results = []
    for i, item in enumerate(items):
      # Simulate processing
      await asyncio.sleep(0.3)
      results.append(f"Processed: {item}")

      # Send progress
      if progress_token is not None:
        await ctx.session.send_progress_notification(
            progress_token=progress_token,
            progress=i + 1,
            total=total,
            message=f"Processing item: {item}",
        )

    return [
        TextContent(
            type="text",
            text="\n".join(results),
        )
    ]

  return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
  """Run the MCP server."""
  async with stdio_server() as (read_stream, write_stream):
    await server.run(
        read_stream,
        write_stream,
        server.create_initialization_options(),
    )


if __name__ == "__main__":
  asyncio.run(main())
