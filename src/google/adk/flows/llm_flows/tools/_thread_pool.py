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
from collections.abc import Callable
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
import contextlib
import contextvars
import inspect
import threading
from typing import Any
import weakref

from ....tools.base_tool import BaseTool
from ....tools.function_tool import _use_sync_callable_runner
from ....tools.function_tool import FunctionTool
from ....tools.tool_context import ToolContext

# Thread pool executors for running tools in background threads, keyed by the
# event loop they serve and then by max_workers. A pool dedicated to tools keeps
# blocking tools from blocking the event loop in Live API mode without competing
# with the loop's own default executor. A pool is released when its loop is
# closed, or when the loop is collected, whichever happens first -- see
# _shutdown_closed_loop_pools for why collection alone is not enough.
_TOOL_THREAD_POOLS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[int, ThreadPoolExecutor]
] = weakref.WeakKeyDictionary()
# Loops on other threads reach this registry concurrently.
_TOOL_THREAD_POOL_LOCK = threading.Lock()


def _shutdown_closed_loop_pools() -> None:
  """Releases the pools of loops that are closed but not yet collected.

  The registry is weak-keyed, so a pool is normally released when its loop is
  collected. Collection is not guaranteed to follow closure: a coroutine that
  raises leaves a traceback that references its frames, the frames reference
  the loop, and the traceback can be held for as long as the caller keeps the
  exception -- ``logging.exception`` keeps it on the log record. The loop is
  already closed and will never run anything again, but its idle tool threads
  stay alive with it.

  A server that runs ``asyncio.run`` per request therefore accumulates threads
  in proportion to the number of *failed* requests, not the number served, and
  the accumulation does not stop until the process restarts. Sweeping on every
  acquisition bounds it to the loops closed since the previous call.

  The caller must hold ``_TOOL_THREAD_POOL_LOCK``.
  """
  # Materialize the keys first: popping inside the loop would mutate the
  # mapping that is being iterated.
  for closed_loop in [loop for loop in _TOOL_THREAD_POOLS if loop.is_closed()]:
    for pool in _TOOL_THREAD_POOLS.pop(closed_loop, {}).values():
      # wait=False so that whichever caller happens to sweep is not made to
      # join another loop's tool threads. Work already submitted still runs;
      # shutdown only stops new work and lets the threads exit after it.
      pool.shutdown(wait=False)


def _get_tool_thread_pool(max_workers: int = 4) -> ThreadPoolExecutor:
  """Gets or creates the running loop's thread pool executor for tool execution.

  The pool is only used for tool calls, so a blocking tool cannot starve work
  the loop itself submits to its default executor, such as name resolution.

  Args:
    max_workers: Maximum number of worker threads in the pool.

  Returns:
    A ThreadPoolExecutor with the specified max_workers, shut down when the
    event loop that created it is closed or collected.
  """
  loop = asyncio.get_running_loop()
  with _TOOL_THREAD_POOL_LOCK:
    _shutdown_closed_loop_pools()
    pools = _TOOL_THREAD_POOLS.setdefault(loop, {})
    pool = pools.get(max_workers)
    if pool is None:
      pool = ThreadPoolExecutor(
          max_workers=max_workers, thread_name_prefix='adk_tool_executor'
      )
      pools[max_workers] = pool
      weakref.finalize(loop, pool.shutdown, wait=False)
    return pool


def _is_sync_tool(tool: BaseTool) -> bool:
  """Checks if a tool's underlying function is synchronous."""
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


@contextlib.contextmanager
def _use_executor_for_sync_callables(
    executor: ThreadPoolExecutor,
) -> Iterator[None]:
  """Binds a sync callable runner that calls each callable on ``executor``.

  The callable runs with a copy of the caller's context variables and with no
  runner bound, so a nested call it makes runs inline on its worker thread.
  """

  async def run_sync_callable(
      target: Callable[..., Any], call_args: dict[str, Any]
  ) -> Any:
    call_context = contextvars.copy_context()

    def invoke() -> Any:
      with _use_sync_callable_runner(None):
        return target(**call_args)

    return await asyncio.get_running_loop().run_in_executor(
        executor,
        lambda: call_context.run(invoke),
    )

  with _use_sync_callable_runner(run_sync_callable):
    yield


async def _call_tool_in_thread_pool(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    max_workers: int = 4,
) -> object:
  """Runs a tool in a thread pool to avoid blocking the event loop.

  The complete ``BaseTool.run_async`` contract is preserved. For synchronous
  ``FunctionTool`` callables, tool-owned validation, authentication, and
  confirmation stay on the caller loop while only synchronous callables enter
  the pool. Other tools run their complete async contract in a worker loop.

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
  loop = asyncio.get_running_loop()
  executor = _get_tool_thread_pool(max_workers)

  if _is_sync_tool(tool) and isinstance(tool, FunctionTool):
    with _use_executor_for_sync_callables(executor):
      return await tool.run_async(args=args, tool_context=tool_context)

  ctx = contextvars.copy_context()

  def run_tool_in_new_loop() -> Any:
    return asyncio.run(tool.run_async(args=args, tool_context=tool_context))

  return await loop.run_in_executor(
      executor, lambda: ctx.run(run_tool_in_new_loop)
  )
