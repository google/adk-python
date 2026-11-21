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
from ...events._branch_path import _BranchPath
from ...events._node_path_builder import _NodePathBuilder
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


def _is_error_event(event: Event) -> bool:
  """Returns whether the ADK event reports an error."""
  # Fields are read defensively: the v1 executor only hands events to a
  # (possibly custom) converter, so it must not assume a full ADK event.
  return bool(
      getattr(event, 'error_code', None)
      or getattr(event, 'error_message', None)
  )


def _is_root_agent_event(event: Event) -> bool:
  """Returns whether the event was emitted by the root agent or node itself.

  The root's own events carry a single-segment node path (e.g. ``'root@1'``),
  while events from sub-agents and child nodes are nested under it (e.g.
  ``'root@1/helper@1'``). Events with no ``node_info.path`` (from the legacy
  ``BaseAgent`` runner branch) also count as root events.
  """
  path = getattr(getattr(event, 'node_info', None), 'path', None)
  if not path or not isinstance(path, str):
    return True
  return _NodePathBuilder.from_string(path).parent is None


def _has_result_payload(event: Event) -> bool:
  """Returns whether the event carries a node output or user-visible content."""
  if getattr(event, 'output', None) is not None:
    return True
  content = getattr(event, 'content', None)
  if not content or not content.parts:
    return False
  for part in content.parts:
    if part.thought:
      continue
    if (part.text and part.text.strip()) or part.inline_data or part.file_data:
      return True
  return False


def _is_successful_result(event: Event) -> bool:
  """Returns whether the event is a complete, error-free result.

  A result is a final, non-partial, agent-authored response that carries a
  node output or user-visible content (non-thought text, inline data or file
  data). Function calls and responses, thought-only and state-only events are
  not results.
  """
  if (
      getattr(event, 'author', None) == 'user'
      or getattr(event, 'partial', None)
      or _is_error_event(event)
      or not _has_result_payload(event)
  ):
    return False
  return event.is_final_response()


def _is_in_branch_scope(branch: str | None, scope: str | None) -> bool:
  """Returns whether ``branch`` is ``scope`` itself or nested under it."""
  branch_path = _BranchPath.from_string(
      branch if isinstance(branch, str) else None
  )
  scope_path = _BranchPath.from_string(
      scope if isinstance(scope, str) else None
  )
  return branch_path == scope_path or branch_path.is_descendant_of(scope_path)


class _ErrorRecoveryTracker:
  """Tracks the error events of a run and whether the run recovered from them.

  An error event no longer means the run is over: a sub-agent or a workflow
  node can fail, and the run continues and produces an answer. An error is
  recovered from once a later successful result is emitted from the same or an
  enclosing branch, so a sibling branch succeeding does not hide a failure.
  Errors emitted by the root agent itself are never recovered from.
  """

  def __init__(self) -> None:
    self._unresolved_errors: list[Event] = []

  def observe(self, event: Event) -> None:
    """Records an event emitted by the run, in emission order."""
    if _is_error_event(event):
      self._unresolved_errors.append(event)
      return
    if not _is_successful_result(event):
      return
    self._unresolved_errors = [
        error
        for error in self._unresolved_errors
        if _is_root_agent_event(error)
        or not _is_in_branch_scope(
            getattr(error, 'branch', None), getattr(event, 'branch', None)
        )
    ]

  def is_unresolved(self, event: Event) -> bool:
    """Returns whether ``event`` is an error the run has not recovered from."""
    return any(error is event for error in self._unresolved_errors)

  @property
  def unresolved_error(self) -> Event | None:
    """The latest error event the run has not recovered from, if any."""
    return self._unresolved_errors[-1] if self._unresolved_errors else None
