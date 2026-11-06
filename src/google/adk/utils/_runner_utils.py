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

import asyncio
import contextlib
import contextvars
import logging
from typing import Any
from typing import AsyncGenerator
from typing import AsyncIterator
from typing import Awaitable
from typing import Callable
from typing import cast
from typing import Generator
from typing import TYPE_CHECKING
from typing import TypeVar

from ..errors._stale_session_error import StaleSessionError

if TYPE_CHECKING:
  from ..agents.invocation_context import InvocationContext
  from ..plugins.plugin_manager import PluginManager

logger = logging.getLogger("google_adk." + __name__)

_T = TypeVar("_T")

# Sentinel cancellation message indicating synchronous run() caller stopped
# iterating early.
_CALLER_CLOSED_EARLY_MSG = "adk-runner-caller-closed-early"


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


async def _seal_aborted_invocation(
    invocation_context: InvocationContext,
    synthesize_abort_events: Callable[[InvocationContext], Awaitable[Any]],
) -> None:
  """Seals dangling calls on abort, skipping if a newer turn updated session."""
  try:
    await synthesize_abort_events(invocation_context)
  except StaleSessionError:
    # Match Runner._run_post_invocation_compaction: if a newer turn has already
    # updated the session while the aborted turn wound down, skip sealing.
    logger.info(
        "Skipping abort seal for invocation %s; a newer turn updated the"
        " session before abort cleanup completed.",
        invocation_context.invocation_id,
    )
  except Exception:  # pylint: disable=broad-exception-caught
    logger.error(
        "Failed to seal aborted invocation %s.",
        invocation_context.invocation_id,
        exc_info=True,
    )


@contextlib.asynccontextmanager
async def _run_invocation_lifecycle(
    *,
    invocation_context: InvocationContext,
    plugin_manager: PluginManager,
    synthesize_abort_events: Callable[[InvocationContext], Awaitable[Any]],
    is_live_call: bool = False,
    post_run_compaction: (
        Callable[[InvocationContext], Awaitable[None]] | None
    ) = None,
) -> AsyncIterator[None]:
  """Manages runner cancellation, abort sealing, after_run, and compaction.

  Exit policy across the five completion paths:
    - Normal return: runs `after_run_callback`, then `post_run_compaction`
      (unless aborted).
    - `GeneratorExit` or `_CALLER_CLOSED_EARLY_MSG`: no abort mark, no seal
      (keeps paused long-running calls resumable), runs `after_run_callback`,
      no compaction.
    - Other `asyncio.CancelledError`: marks `_abort_state.aborted = True` and
      seals; `after_run_callback` runs only if `is_aborted` was already `True`
      before cancellation, no compaction.
    - `Exception`: runs `on_run_error_callback`, seals if aborted, no
      `after_run_callback`.
    - Other `BaseException`: seals if aborted, no `after_run_callback`.
  If sealing raises `StaleSessionError` because a newer turn updated the
  session while the aborted turn wound down, sealing is skipped at `INFO`.

  Args:
    invocation_context: Active invocation context for the run.
    plugin_manager: Plugin manager notified for error and post-run hooks.
    synthesize_abort_events: Callback that seals dangling calls on abort.
    is_live_call: Whether this lifecycle wraps a live streaming run.
    post_run_compaction: Optional post-invocation compaction coroutine.

  Yields:
    Control to the wrapped execution block.
  """
  run_error: BaseException | None = None
  closing_early = False
  try:
    yield
  except GeneratorExit:
    closing_early = True
    raise
  except Exception as e:
    run_error = e
    await _notify_run_error(plugin_manager, invocation_context, e)
    raise
  except asyncio.CancelledError as e:
    if e.args and e.args[0] == _CALLER_CLOSED_EARLY_MSG:
      closing_early = True
    else:
      was_aborted = invocation_context.is_aborted
      # Mark this invocation (and its model_copy clones) aborted for sealing
      # without tripping _abort_state.signal, which may be a caller-owned
      # Event or a parent AgentTool's _abort_signal.
      invocation_context._abort_state.aborted = True  # pylint: disable=protected-access
      # Only cooperative aborts (signal set before cancel) run after_run;
      # unrequested external task cancellation skips after_run_callback.
      if was_aborted:
        closing_early = True
      else:
        run_error = e
    raise
  except BaseException as e:
    run_error = e
    raise
  finally:
    if not is_live_call and invocation_context.is_aborted:
      await _seal_aborted_invocation(
          invocation_context, synthesize_abort_events
      )
    if run_error is None:
      try:
        await plugin_manager.run_after_run_callback(
            invocation_context=invocation_context
        )
        if (
            post_run_compaction is not None
            and not invocation_context.is_aborted
            and not closing_early
        ):
          await post_run_compaction(invocation_context)
      except Exception as e:  # pylint: disable=broad-exception-caught
        await _notify_run_error(plugin_manager, invocation_context, e)
        if closing_early:
          logger.error(
              "after_run callback failed while closing invocation %s early.",
              invocation_context.invocation_id,
              exc_info=True,
          )
        else:
          raise
