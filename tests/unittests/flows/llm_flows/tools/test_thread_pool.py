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

"""Unit tests for flows.llm_flows.tools._thread_pool."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from collections.abc import Callable
import concurrent.futures

from google.adk.flows.llm_flows.tools import _thread_pool as _tool_thread_pool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool


def _run_with_own_loop(
    coro_fn: Callable[[], Awaitable[None]], *, raise_after: bool
) -> dict[str, object]:
  """Runs coro_fn on a fresh loop; returns that loop and its tool pool.

  Mirrors a server that calls asyncio.run per request. The returned dict keeps
  the loop referenced after asyncio.run has closed it, standing in for whatever
  holds it in production -- a traceback on a log record, most often.

  Args:
    coro_fn: Awaited on the fresh loop, after its tool pool is acquired.
    raise_after: Whether the coroutine should raise once coro_fn returns, so
      that the run ends the way a failed request does.

  Returns:
    A dict with the run's event loop under 'loop' and the tool pool it
    acquired under 'pool'.
  """
  captured: dict[str, object] = {}

  async def main() -> None:
    captured['loop'] = asyncio.get_running_loop()
    captured['pool'] = _tool_thread_pool._get_tool_thread_pool()
    await coro_fn()
    if raise_after:
      raise RuntimeError('request failed')

  try:
    asyncio.run(main())
  except RuntimeError:
    pass
  return captured


async def _noop() -> None:
  await asyncio.sleep(0)


def _is_shut_down(pool: concurrent.futures.ThreadPoolExecutor) -> bool:
  """Whether the pool refuses new work, via public API rather than _shutdown."""
  try:
    pool.submit(bool).cancel()
  except RuntimeError:
    return True
  return False


def test_tool_thread_pool_is_released_when_its_loop_closes() -> None:
  """A closed-but-uncollected loop must not keep its tool threads alive."""
  failed = _run_with_own_loop(_noop, raise_after=True)
  stranded_loop = failed['loop']
  stranded_pool = failed['pool']
  assert isinstance(stranded_loop, asyncio.AbstractEventLoop)
  assert isinstance(stranded_pool, concurrent.futures.ThreadPoolExecutor)

  assert stranded_loop.is_closed()
  assert stranded_loop in _tool_thread_pool._TOOL_THREAD_POOLS
  assert not _is_shut_down(stranded_pool)

  # A later acquisition sweeps it.
  _run_with_own_loop(_noop, raise_after=False)

  assert stranded_loop not in _tool_thread_pool._TOOL_THREAD_POOLS
  assert _is_shut_down(stranded_pool)


def test_tool_thread_pool_is_reused_within_one_loop() -> None:
  """Sweeping must not disturb the pool of the loop that is still running."""

  async def main() -> None:
    first = _tool_thread_pool._get_tool_thread_pool()
    second = _tool_thread_pool._get_tool_thread_pool()
    assert first is second
    assert not _is_shut_down(first)
    assert _tool_thread_pool._get_tool_thread_pool(max_workers=2) is not first
    assert not _is_shut_down(first)

  asyncio.run(main())


def test_functions_shim_shares_thread_pool_registry() -> None:
  """Compatibility re-exports on functions must reference the same pool registry."""
  from google.adk.flows.llm_flows import functions

  assert functions._TOOL_THREAD_POOLS is _tool_thread_pool._TOOL_THREAD_POOLS
  assert (
      functions._TOOL_THREAD_POOL_LOCK
      is _tool_thread_pool._TOOL_THREAD_POOL_LOCK
  )


def test_is_sync_tool_distinguishes_sync_and_async_callables() -> None:
  """Sync FunctionTools report True while async FunctionTools and plain BaseTools report False."""

  def sync_fn(x: int) -> int:
    return x + 1

  async def async_fn(x: int) -> int:
    return x + 1

  assert _tool_thread_pool._is_sync_tool(FunctionTool(sync_fn)) is True
  assert _tool_thread_pool._is_sync_tool(FunctionTool(async_fn)) is False
  assert (
      _tool_thread_pool._is_sync_tool(BaseTool(name='base', description='desc'))
      is False
  )
