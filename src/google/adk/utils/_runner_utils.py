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

"""Private runner and execution utility functions."""

from __future__ import annotations

import contextvars
import logging
from typing import Any
from typing import AsyncGenerator
from typing import Awaitable
from typing import cast
from typing import Generator
from typing import TYPE_CHECKING
from typing import TypeVar

if TYPE_CHECKING:
  from ..agents.invocation_context import InvocationContext
  from ..plugins.plugin_manager import PluginManager

logger = logging.getLogger("google_adk." + __name__)

_T = TypeVar("_T")


async def _stream_in_caller_scope(
    agen: AsyncGenerator[_T, None],
) -> AsyncGenerator[_T, None]:
  """Iterates `agen`, running it only in its own copy of the caller context.

  An async generator borrows the context of whoever resumes it, so what it
  attaches across a yield leaks into its consumer, and a detach run by a
  different resumer (another task, the GC) fails. Running it in its own
  context, like a Task, fixes both.
  """
  ctx = contextvars.copy_context()
  try:
    while True:
      try:
        item = await _InContext(ctx, agen.__anext__())
      except StopAsyncIteration:
        return
      yield item
  finally:
    await _InContext(ctx, agen.aclose())


class _InContext(Awaitable[_T]):
  """`await _InContext(ctx, x)` is `await x`, but every piece of x runs in `ctx`.

  `ctx.run` is synchronous, so it cannot stay entered across an await without
  every other task running in `ctx` too. Instead each piece of `x`, up to the
  next future it waits on, runs in `ctx`, as asyncio does for a Task.
  """

  def __init__(self, ctx: contextvars.Context, awaitable: Awaitable[_T]):
    self._ctx = ctx
    self._awaitable = awaitable

  def __await__(self) -> Generator[Any, Any, _T]:
    coro = self._awaitable.__await__()
    value: Any = None
    error: BaseException | None = None
    while True:
      try:
        if error is None:
          waiting_on = self._ctx.run(coro.send, value)
        else:
          waiting_on = self._ctx.run(coro.throw, error)
      except StopIteration as done:
        return cast(_T, done.value)
      try:
        value, error = (yield waiting_on), None
      except BaseException as e:  # pylint: disable=broad-exception-caught
        # Cancelled while waiting: deliver it inside `ctx`.
        value, error = None, e


async def _notify_run_error(
    plugin_manager: PluginManager,
    invocation_context: InvocationContext,
    error: Exception,
) -> None:
  """Best-effort on_run_error notification; never masks the original error.

  on_run_error_callback is notification-only: the triggering exception is
  always re-raised by the caller, so any exception from the callback itself
  (or from a test double that does not implement it) is logged and suppressed.
  """
  try:
    await plugin_manager.run_on_run_error_callback(
        invocation_context=invocation_context, error=error
    )
  except Exception:  # pylint: disable=broad-except
    logger.exception(
        "on_run_error_callback raised; suppressing so the original run error"
        " propagates."
    )
