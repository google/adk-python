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

import asyncio
import os
from pathlib import Path
import sys

from mcp.server.fastmcp import FastMCP

# Create an MCP server with a name
mcp = FastMCP("Filesystem Server", host="localhost", port=3000)


# Add a tool to read file contents
@mcp.tool(description="Read contents of a file")
def read_file(filepath: str) -> str:
  """Read and return the contents of a file."""
  with open(filepath, "r") as f:
    return f.read()


# Add a tool to list directory contents
@mcp.tool(description="List contents of a directory")
def list_directory(dirpath: str) -> list:
  """List all files and directories in the given directory."""
  return os.listdir(dirpath)


# Add a tool to get current working directory
@mcp.tool(description="Get current working directory")
def get_cwd() -> str:
  """Return the current working directory."""
  return str(Path.cwd())


# Graceful shutdown handler
async def shutdown(signal, loop):
  """Cleanup tasks tied to the service's shutdown."""
  print(f"\nReceived exit signal {signal.name}...")

  # Get all running tasks
  tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

  # Cancel all tasks
  for task in tasks:
    task.cancel()

  print(f"Cancelling {len(tasks)} outstanding tasks")
  await asyncio.gather(*tasks, return_exceptions=True)

  # Stop the loop
  loop.stop()
  print("Shutdown complete!")


# Main entry point with graceful shutdown handling
if __name__ == "__main__":
  try:
    # The MCP run function ultimately uses asyncio.run() internally
    mcp.run(transport="sse")
  except KeyboardInterrupt:
    print("\nServer shutting down gracefully...")
    # The asyncio event loop has already been stopped by the KeyboardInterrupt
    print("Server has been shut down.")
  except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(1)
  finally:
    print("Thank you for using the Filesystem MCP Server!")
