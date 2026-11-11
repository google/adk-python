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

"""Binds the executor that runs synchronous user callables.

Synchronous callables in ``FunctionTool`` and ``FunctionNode`` resolve their
runner from an explicit ``_use_sync_callable_runner`` scope when active, or
otherwise from ``RunConfig.tool_thread_pool_config`` on the execution context.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import contextvars
import inspect
import threading
from typing import Any
from typing import TYPE_CHECKING
import weakref

if TYPE_CHECKING:
  from ..agents.context import Context

_SyncCallableRunner = Callable[
    [Callable[..., Any], dict[str, Any]], Awaitable[Any]
]
_SYNC_CALLABLE_RUNNER: contextvars.ContextVar[_SyncCallableRunner | None] = (
    contextvars.ContextVar("adk_sync_callable_runner", default=None)
)
_SYNC_CALLABLE_INLINE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "adk_sync_callable_inline", default=False
)
_SYNC_CALLABLE_CONTEXT: contextvars.ContextVar[Context | None] = (
    contextvars.ContextVar("adk_sync_callable_context", default=None)
)

# Thread pool executors for running synchronous tools and nodes in background
# threads, keyed by the event loop they serve and then by max_workers. A pool
# dedicated to tools keeps blocking callables from starving the loop's own
# default executor. A pool is released when its loop is closed or collected,
# whichever happens first -- see _shutdown_closed_loop_pools.
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

  The pool is only used for tool and node calls, so a blocking call cannot
  starve work the loop itself submits to its default executor, such as name
  resolution.

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
          max_workers=max_workers, thread_name_prefix="adk_tool_executor"
      )
      pools[max_workers] = pool
      weakref.finalize(loop, pool.shutdown, wait=False)
    return pool


@contextmanager
def _use_sync_callable_runner(
    runner: _SyncCallableRunner | None = None,
) -> Iterator[None]:
  """Binds the runner used for synchronous callables.

  Passing ``None`` clears the binding, which stops a worker-owned nested call
  from reusing the caller's runner or re-entering the thread pool.
  """
  token = _SYNC_CALLABLE_RUNNER.set(runner)
  inline_token = _SYNC_CALLABLE_INLINE.set(runner is None)
  try:
    yield
  finally:
    _SYNC_CALLABLE_INLINE.reset(inline_token)
    _SYNC_CALLABLE_RUNNER.reset(token)


def _make_executor_sync_callable_runner(
    executor: ThreadPoolExecutor,
) -> _SyncCallableRunner:
  """Creates a sync callable runner backed by ``executor``."""

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

  return run_sync_callable


@contextmanager
def _use_executor_for_sync_callables(
    executor: ThreadPoolExecutor,
) -> Iterator[None]:
  """Binds a sync callable runner that calls each callable on ``executor``.

  The callable runs with a copy of the caller's context variables and with no
  runner bound, so a nested call it makes runs inline on its worker thread.
  """
  with _use_sync_callable_runner(_make_executor_sync_callable_runner(executor)):
    yield


@contextmanager
def _use_sync_callable_context(
    context: Context | None,
) -> Iterator[None]:
  """Binds fallback ``Context`` for nested ``_invoke_callable`` calls."""
  token = _SYNC_CALLABLE_CONTEXT.set(context)
  try:
    yield
  finally:
    _SYNC_CALLABLE_CONTEXT.reset(token)


def _resolve_sync_callable_runner() -> _SyncCallableRunner | None:
  """Returns the active sync callable runner, falling back to bound context."""
  runner = _SYNC_CALLABLE_RUNNER.get()
  if runner is not None or _SYNC_CALLABLE_INLINE.get():
    return runner
  context = _SYNC_CALLABLE_CONTEXT.get()
  # Tool and node tests often pass MagicMock / create_autospec contexts whose
  # attributes either raise AttributeError or return child mocks instead of int.
  run_config = getattr(context, "run_config", None)
  thread_pool_config = getattr(run_config, "tool_thread_pool_config", None)
  max_workers = getattr(thread_pool_config, "max_workers", None)
  if isinstance(max_workers, int) and not isinstance(max_workers, bool):
    return _make_executor_sync_callable_runner(
        _get_tool_thread_pool(max_workers)
    )
  return None


def _is_async_callable(target: Any) -> bool:
  """Returns whether calling ``target`` returns a coroutine."""
  return inspect.iscoroutinefunction(target) or (
      hasattr(target, "__call__")
      and inspect.iscoroutinefunction(target.__call__)
  )


async def _invoke_callable(
    target: Callable[..., Any],
    kwargs: dict[str, Any],
    *,
    unwrapped: Callable[..., Any] | None = None,
) -> Any:
  """Invokes a sync or async callable, using the sync runner when configured.

  Note:
    The function name `_invoke_callable` is part of the traceback frame contract
    checked by `FunctionTool._is_invocation_type_error` to detect call-site
    TypeErrors for callables without Python code objects (e.g. C builtins).
  """
  check_target = unwrapped or target
  if _is_async_callable(check_target):
    return await target(**kwargs)
  runner = _resolve_sync_callable_runner()
  if runner is not None:
    result = await runner(target, kwargs)
  else:
    result = target(**kwargs)
  if inspect.isawaitable(result):
    return await result
  return result
