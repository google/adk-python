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
from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from collections.abc import Callable
import dataclasses
import inspect
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

from a2a.server.agent_execution.context import RequestContext
from a2a.server.events import Event as A2AEvent
from a2a.server.events.event_queue import EventQueue
from a2a.types import Message
from a2a.types import TaskStatusUpdateEvent

from .. import _compat
from ...events._abort_events import _INVOCATION_ABORTED
from ...events.event import Event
from ...utils.context_utils import Aclosing
from .config import ExecuteInterceptor
from .executor_context import ExecutorContext

if TYPE_CHECKING:
  from ..converters.request_converter import AgentRunRequest


def _is_abort_adk_event(event: Any) -> bool:
  return getattr(event, 'error_code', None) == _INVOCATION_ABORTED


@dataclasses.dataclass
class _ActiveTaskRun:
  """Tracks in-flight state for a single A2A task execution."""

  task_id: Optional[str]
  event_queue: EventQueue
  abort_signal: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)
  done_event: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)
  terminal_cancel_emitted: bool = False


class _ActiveTaskRegistry:
  """Coordinates active A2A task runs and cancellation signals."""

  def __init__(self) -> None:
    self._runs: dict[str, _ActiveTaskRun] = {}

  def start(
      self, task_id: Optional[str], event_queue: EventQueue
  ) -> _ActiveTaskRun:
    """Registers an active run for ``task_id`` if present."""
    run = _ActiveTaskRun(task_id=task_id, event_queue=event_queue)
    if task_id:
      self._runs[task_id] = run
    return run

  def get(self, task_id: Optional[str]) -> _ActiveTaskRun | None:
    """Returns the active run for ``task_id`` if registered."""
    return self._runs.get(task_id) if task_id else None

  def finish(self, run: _ActiveTaskRun) -> None:
    """Marks ``run`` complete and unregisters it."""
    if run.task_id and self._runs.get(run.task_id) is run:
      del self._runs[run.task_id]
    run.done_event.set()

  async def cancel(
      self, context: RequestContext, event_queue: EventQueue
  ) -> None:
    """Signals an active task run to abort, or emits a canceled event."""
    if not context.task_id:
      raise ValueError('A2A cancellation must have a task ID')
    run = self._runs.get(context.task_id)
    if run is None:
      await _enqueue_canceled_task_event(context, event_queue)
      return
    run.abort_signal.set()
    await run.done_event.wait()
    tapped = getattr(run.event_queue, '_children', ())
    same_queue = event_queue is run.event_queue or event_queue in tapped
    if not run.terminal_cancel_emitted or not same_queue:
      await _enqueue_canceled_task_event(context, event_queue)


async def _consume_runner_with_abort(
    run_async_fn: Callable[..., Any],
    run_request: AgentRunRequest,
    abort_signal: asyncio.Event,
    on_event: Callable[[Event], Awaitable[None]],
) -> None:
  """Drives ``run_async_fn`` in one task and cancels it on ``abort_signal``."""
  if abort_signal.is_set():
    return
  run_kwargs = dict(vars(run_request))
  params = inspect.signature(run_async_fn).parameters
  if 'abort_signal' in params or any(
      p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
  ):
    run_kwargs['abort_signal'] = abort_signal

  async def _drive() -> None:
    async with Aclosing(run_async_fn(**run_kwargs)) as agen:
      async for adk_event in agen:
        await on_event(adk_event)

  abort_task = asyncio.create_task(abort_signal.wait())
  drive_task = asyncio.create_task(_drive())
  try:
    await asyncio.wait(
        {drive_task, abort_task}, return_when=asyncio.FIRST_COMPLETED
    )
    if abort_signal.is_set() and not drive_task.done():
      drive_task.cancel()
    await drive_task
  except asyncio.CancelledError:
    curr = asyncio.current_task()
    abort_signal.set()
    drive_task.cancel()
    await asyncio.gather(drive_task, return_exceptions=True)
    if curr is not None and curr.cancelling() > 0:
      raise
  finally:
    abort_task.cancel()
    await asyncio.gather(abort_task, return_exceptions=True)


async def _enqueue_canceled_task_event(
    context: RequestContext,
    event_queue: EventQueue,
) -> None:
  """Publishes the terminal event required by the A2A cancellation contract."""
  if not context.task_id:
    raise ValueError('A2A cancellation must have a task ID')

  # ``final`` is load-bearing on a2a-sdk 0.3.x, which keeps consuming the queue
  # until it sees a status update with it set; 1.x has no such field.
  await event_queue.enqueue_event(
      _compat.make_task_status_update_event(
          task_id=context.task_id,
          context_id=context.context_id,
          status=_compat.make_task_status(_compat.TS_CANCELED),
          final=True,
      )
  )


def _require_request_context(
    context: RequestContext,
) -> tuple[Message, str, str]:
  """Return the values guaranteed for an executor-ready A2A request."""
  message = context.message
  if message is None:
    raise ValueError('A2A request must have a message')
  if not context.task_id:
    raise ValueError('A2A request must have a task ID')
  if not context.context_id:
    raise ValueError('A2A request must have a context ID')
  return message, context.task_id, context.context_id


def _require_user_id(run_request: AgentRunRequest) -> str:
  """Return the request's user ID, which every session service requires."""
  if run_request.user_id is None:
    raise ValueError('The A2A request converter returned no user ID')
  return run_request.user_id


async def execute_before_agent_interceptors(
    context: RequestContext,
    execute_interceptors: Optional[list[ExecuteInterceptor]],
) -> RequestContext:
  if execute_interceptors:
    for interceptor in execute_interceptors:
      if interceptor.before_agent:
        context = await interceptor.before_agent(context)
  return context


async def execute_after_event_interceptors(
    a2a_event: A2AEvent,
    executor_context: ExecutorContext,
    adk_event: Event,
    execute_interceptors: Optional[list[ExecuteInterceptor]],
) -> list[A2AEvent]:
  events = [a2a_event]
  if execute_interceptors:
    for interceptor in execute_interceptors:
      if interceptor.after_event:
        next_events: list[A2AEvent] = []
        for e in events:
          res = await interceptor.after_event(executor_context, e, adk_event)
          if res is None:
            continue
          if isinstance(res, list):
            next_events.extend(res)
          else:
            next_events.append(res)
        events = next_events
        if not events:
          return []
  return events


async def execute_after_agent_interceptors(
    executor_context: ExecutorContext,
    final_event: TaskStatusUpdateEvent,
    execute_interceptors: Optional[list[ExecuteInterceptor]],
) -> TaskStatusUpdateEvent:
  if execute_interceptors:
    for interceptor in reversed(execute_interceptors):
      if interceptor.after_agent:
        final_event = await interceptor.after_agent(
            executor_context, final_event
        )
  return final_event
