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

"""Per-event-loop thread pool executors for synchronous tool execution."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
from typing import Any

from ....tools.base_tool import BaseTool
from ....tools.function_tool import FunctionTool
from ....tools.tool_context import ToolContext
from ....utils._sync_runner import _get_tool_thread_pool
from ....utils._sync_runner import _shutdown_closed_loop_pools  # noqa: F401
from ....utils._sync_runner import _TOOL_THREAD_POOL_LOCK  # noqa: F401
from ....utils._sync_runner import _TOOL_THREAD_POOLS  # noqa: F401
from ....utils._sync_runner import _use_executor_for_sync_callables


def _is_sync_tool(tool: BaseTool) -> bool:
  """Checks if a tool has synchronous callables that should run on the pool."""
  from ....tools._node_tool import NodeTool
  from ....workflow._function_node import FunctionNode

  if isinstance(tool, NodeTool):
    if isinstance(tool.node, FunctionNode):
      unwrapped = tool.node._unwrapped_func
      return not (
          inspect.iscoroutinefunction(unwrapped)
          or inspect.isasyncgenfunction(unwrapped)
      )
    return True
  if not hasattr(tool, 'func'):
    return False
  func = getattr(tool, 'func')
  return not (
      inspect.iscoroutinefunction(func)
      or inspect.isasyncgenfunction(func)
      or (
          hasattr(func, '__call__')
          and inspect.iscoroutinefunction(func.__call__)
      )
  )


async def _call_tool_in_thread_pool(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    max_workers: int = 4,
) -> object:
  """Runs a tool in a thread pool to avoid blocking the event loop.

  The complete ``BaseTool.run_async`` contract is preserved. For synchronous
  ``FunctionTool`` callables and ``NodeTool``s (such as synchronous
  ``FunctionNode``s and ``Workflow``s), tool-owned validation, authentication,
  and confirmation stay on the caller loop while only synchronous callables
  enter the pool. A ``NodeTool`` wrapping an async ``FunctionNode`` runs on the
  caller loop as it does without the pool, because its events must be enqueued
  from that loop. Other tools run their complete async contract in a worker
  loop.

  Note: Due to Python's GIL, this does NOT help with pure Python CPU-bound code.
  Thread pool only helps when the GIL is released (blocking I/O, C extensions).

  Args:
    tool: The tool to execute.
    args: Arguments to pass to the tool.
    tool_context: The tool context.
    max_workers: Maximum number of worker threads in the pool.

  Returns:
    The result of running the tool.
  """
  from ....tools._node_tool import NodeTool

  loop = asyncio.get_running_loop()
  executor = _get_tool_thread_pool(max_workers)

  if _is_sync_tool(tool) and isinstance(tool, (FunctionTool, NodeTool)):
    with _use_executor_for_sync_callables(executor):
      return await tool.run_async(args=args, tool_context=tool_context)
  if isinstance(tool, NodeTool):
    return await tool.run_async(args=args, tool_context=tool_context)

  ctx = contextvars.copy_context()

  def run_tool_in_new_loop() -> Any:
    return asyncio.run(tool.run_async(args=args, tool_context=tool_context))

  return await loop.run_in_executor(
      executor, lambda: ctx.run(run_tool_in_new_loop)
  )
