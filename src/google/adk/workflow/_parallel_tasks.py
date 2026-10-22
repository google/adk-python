# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Owned parallel node runs using the workflow's existing scheduler."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from contextlib import nullcontext
import copy
import inspect
import logging
import math
from typing import Any
from typing import Literal
from typing import TYPE_CHECKING

from ..events._internal_metadata import INTERNAL_METADATA_PREFIX
from ..events._node_path_builder import _NodePathBuilder
from ..events._rewind_events import _apply_rewinds
from ..events.event import Event
from ..events.event import NodeInfo
from ..events.event_actions import EventActions
from ._errors import DynamicNodeFailError
from ._errors import NodeInterruptedError
from ._parallel_task_context import _bind_parallel_task
from ._parallel_task_context import _ParallelTaskControl
from .utils._workflow_graph_utils import build_node

if TYPE_CHECKING:
  from ..agents.context import Context
  from ._base_node import BaseNode
  from ._graph import NodeLike

logger = logging.getLogger('google_adk.' + __name__)
_CONTROL_KEY = INTERNAL_METADATA_PREFIX + 'parallel_tasks'
_MISSING = object()


def _same_value(left: Any, right: Any) -> bool:
  """Compare state values without treating booleans as integers."""
  if type(left) is not type(right):
    return False
  if isinstance(left, dict):
    return left.keys() == right.keys() and all(
        _same_value(value, right[key]) for key, value in left.items()
    )
  if isinstance(left, (list, tuple)):
    return len(left) == len(right) and all(
        _same_value(a, b) for a, b in zip(left, right)
    )
  return bool(left == right)


class ParallelTask:
  """A stable handle to one node run owned by a ``ParallelTasks`` scope.

  Cancelling an await of ``result()`` only stops that observer. Use ``cancel``
  to request cancellation of the work, or await ``group.cancel(task)`` to also
  wait for it to stop. Messages are delivered at the next LLM request boundary.
  """

  def __init__(
      self,
      group: ParallelTasks,
      node: BaseNode,
      node_input: Any,
      run_id: str,
      index: int,
  ) -> None:
    self._group = group
    self._node = node
    self._input = node_input
    self._run_id = run_id
    self._index = index
    self._path = str(
        _NodePathBuilder.from_string(group._ctx.node_path).append(
            node.name, run_id
        )
    )
    self._status = 'queued'
    self._output: Any = None
    self._error: BaseException | None = None
    self._runner: asyncio.Task[None] | None = None
    self._done: asyncio.Future[None] = (
        asyncio.get_running_loop().create_future()
    )
    self._cancel_requested = False
    self._cancellation_sent = False
    self._cancel_deadline: float | None = None
    self._interrupt_ids: set[str] = set()
    self._control = _ParallelTaskControl(
        task_id=self.id, root_node_path=self.path
    )

  @property
  def id(self) -> str:
    """Stable node path, including the run ID reserved at spawn time."""
    return self._path

  @property
  def path(self) -> str:
    """Node path used by native workflow events."""
    return self._path

  @property
  def name(self) -> str:
    """Name of the scheduled node."""
    return self._node.name

  @property
  def status(self) -> str:
    """Queued, running, completed, failed, waiting, or cancelled."""
    return self._status

  @property
  def output(self) -> Any:
    """Completed output, or None when no output is available."""
    return self._output

  @property
  def error(self) -> BaseException | None:
    """Execution failure, if any."""
    return self._error

  async def result(self) -> Any:
    """Wait for this run without transferring cancellation ownership."""
    await asyncio.shield(self._done)
    if self.status == 'cancelled':
      raise asyncio.CancelledError()
    if self._error is not None:
      raise self._error
    return self._output

  def cancel(self) -> bool:
    """Request deliberate cancellation; returns whether it was requested."""
    if self.status in {'completed', 'failed', 'cancelled'}:
      return False
    if self._group._closed:
      raise RuntimeError('The parallel task scope is closed.')
    self._cancel_requested = True
    self._group._wake.set()
    return True

  def send_message(self, content: str) -> dict[str, str]:
    """Queue a message for the worker's next LLM request."""
    return self._control.send_message(content)

  def message_receipt(self, message_id: str) -> dict[str, str]:
    """Return the delivery status of a queued message."""
    return self._control.message_receipt(message_id)

  @property
  def message_receipts(self) -> list[dict[str, str]]:
    """Detached snapshots of this worker's message receipts."""
    return self._control.receipts

  def _finish(self, status: str, error: BaseException | None = None) -> None:
    self._status = status
    self._error = error
    self._control.finish(status)
    if not self._done.done():
      self._done.set_result(None)
    self._group._changed.set()
    self._group._wake.set()


class ParallelTasks:
  """An owned, bounded group of native workflow node runs.

  Use through ``async with ctx.parallel_tasks(...) as tasks``. Worker state
  remains normal shared session state. Return mapping proposals and explicitly
  call ``merge_result`` when only selected results should update parent state.
  Names and spawn order must be stable when a workflow resumes.
  """

  def __init__(
      self,
      ctx: Context,
      *,
      name: str = 'parallel',
      max_parallel_workers: int | None = None,
      cancellation_timeout: float = 5.0,
      _legacy_run_ids: bool = False,
  ) -> None:
    if not name or any(char in name for char in '/@'):
      raise ValueError(
          'Group name must be nonempty and contain neither / nor @.'
      )
    if max_parallel_workers is not None and (
        isinstance(max_parallel_workers, bool)
        or not isinstance(max_parallel_workers, int)
        or max_parallel_workers < 1
    ):
      raise ValueError(
          'max_parallel_workers must be a positive integer or None.'
      )
    if not math.isfinite(cancellation_timeout) or cancellation_timeout <= 0:
      raise ValueError('cancellation_timeout must be positive and finite.')
    self._ctx = ctx
    self._name = name
    self._limit = max_parallel_workers
    self._timeout = cancellation_timeout
    self._legacy_run_ids = _legacy_run_ids
    ordinal = ctx._parallel_group_counters.get(name, 0) + 1
    ctx._parallel_group_counters[name] = ordinal
    self._prefix = f'{name}-{ordinal}'
    self._call_scope: str | None = None
    group_run_id = str(ordinal)
    if not _legacy_run_ids and ctx.function_call_id:
      self._call_scope = (
          base64.urlsafe_b64encode(ctx.function_call_id.encode('utf-8'))
          .decode('ascii')
          .rstrip('=')
      )
      self._prefix += f'-call-{self._call_scope}'
      group_run_id += f'-call-{self._call_scope}'
    self._id = str(
        _NodePathBuilder.from_string(ctx.node_path).append(name, group_run_id)
    )
    self._tasks: list[ParallelTask] = []
    self._next_index = 0
    self._wake = asyncio.Event()
    self._changed = asyncio.Event()
    self._decision_lock = asyncio.Lock()
    self._pump_task: asyncio.Task[None] | None = None
    self._pump_error: BaseException | None = None
    self._entered = False
    self._closed = False
    self._paused = False
    self._fail_fast = False
    self._first_mode = False
    self._selecting = False
    self._decision_pending = False
    self._cancelled_ids: set[str] = set()
    self._winner_id: str | None = None
    self._winner: ParallelTask | None = None
    self._completed_ids: set[str] = set()
    self._state_snapshot: dict[str, Any] = {}
    self._merged: set[tuple[str, str]] = set()

  async def __aenter__(self) -> ParallelTasks:
    if self._entered:
      raise RuntimeError('A parallel task scope can only be entered once.')
    if not self._ctx._node_rerun_on_resume:
      raise ValueError('Parallel tasks require rerun_on_resume=True.')
    self._entered = True
    if not self._legacy_run_ids:
      self._state_snapshot = copy.deepcopy(self._ctx.state.to_dict())
    for event in _apply_rewinds(self._ctx.session.events):
      if event.invocation_id != self._ctx._invocation_context.invocation_id:
        continue
      record = (event.custom_metadata or {}).get(_CONTROL_KEY)
      if not isinstance(record, dict) or record.get('group') != self._id:
        continue
      self._cancelled_ids.update(record.get('cancelled', []))
      self._completed_ids.update(record.get('completed', []))
      for task_id, keys in record.get('merged', {}).items():
        self._merged.update((task_id, key) for key in keys)
      if record.get('winner') is not None:
        self._winner_id = record['winner']
    self._pump_task = asyncio.create_task(self._pump())
    return self

  async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
    active_error = exc
    try:
      if exc is None and self._winner is None:
        await self.gather()
    except BaseException as error:
      active_error = error
      raise
    finally:
      self._closed = True
      self._wake.set()
      if self._pump_task is not None:
        self._pump_task.cancel()
      try:
        await self._drain(
            self._tasks,
            deliberate=False,
            preserve_error=active_error is not None,
        )
      finally:
        for task in self._tasks:
          task._control.close_event_delivery()
        if self._pump_task is not None:
          await asyncio.gather(self._pump_task, return_exceptions=True)

  def spawn(
      self,
      node: NodeLike,
      node_input: Any = None,
      *,
      run_id: str | None = None,
  ) -> ParallelTask:
    """Reserve a stable identity and queue a child without waiting for a slot."""
    if not self._entered or self._closed:
      raise RuntimeError('Spawn requires an open parallel task scope.')
    if self._winner is not None:
      raise RuntimeError('Cannot spawn after this group selected its result.')
    built_node = build_node(node)
    index = self._next_index
    self._next_index += 1
    if run_id is not None and (
        not run_id or run_id.isdigit() or any(c in run_id for c in '/@')
    ):
      raise ValueError(
          'Explicit run_id must be nonnumeric and contain neither / nor @.'
      )
    if run_id is None:
      scheduler = self._ctx._workflow_scheduler
      if self._legacy_run_ids and scheduler is not None:
        run_id = scheduler._state.next_run_id(
            built_node.name, parent_path=self._ctx.node_path
        )
      elif self._legacy_run_ids:
        run_id = str(index + 1)
      else:
        run_id = f'{self._prefix}-{index + 1}'
    elif self._call_scope is not None:
      run_id = f'call-{self._call_scope}-{run_id}'
    task = ParallelTask(self, built_node, node_input, run_id, index)
    if not self._legacy_run_ids:
      if task.id in self._ctx._parallel_task_paths:
        raise ValueError(f'Duplicate parallel task ID: {task.id}')
      self._ctx._parallel_task_paths.add(task.id)
    self._tasks.append(task)
    if task.id in self._cancelled_ids:
      task._finish('cancelled')
    self._wake.set()
    return task

  async def wait(
      self,
      *,
      mode: Literal['any', 'all'] = 'all',
  ) -> tuple[list[ParallelTask], list[ParallelTask]]:
    """Return completed and pending handles in spawn order.

    A worker awaiting human input pauses admission. Once running siblings stop,
    ``all`` returns with queued handles pending so the parent can suspend.
    """
    if mode not in {'any', 'all'}:
      raise ValueError('mode must be any or all.')
    while True:
      self._changed.clear()
      self._raise_pump_error()
      done = [task for task in self._tasks if task._done.done()]
      pending = [task for task in self._tasks if not task._done.done()]
      running = any(task.status == 'running' for task in pending)
      if (
          not pending
          or (mode == 'any' and done)
          or (self._paused and not running)
      ):
        return done, pending
      await self._changed.wait()

  async def gather(self) -> list[Any]:
    """Gather outputs in spawn order, stopping admission on failure or HITL."""
    self._fail_fast = True
    if any(task.status in {'failed', 'waiting'} for task in self._tasks):
      self._paused = True
    while True:
      self._changed.clear()
      self._raise_pump_error()
      failed = next(
          (task for task in self._tasks if task.status == 'failed'), None
      )
      if failed is not None:
        await self._drain(self._tasks, deliberate=False, preserve_error=True)
        await failed.result()
      running = any(task.status == 'running' for task in self._tasks)
      if self._paused and not running:
        waiting = next(
            (task for task in self._tasks if task.status == 'waiting'), None
        )
        if waiting is not None:
          await waiting.result()
      if all(task._done.done() for task in self._tasks):
        for task in self._tasks:
          if task.status == 'cancelled' and task.id not in self._cancelled_ids:
            await task.result()
        return [task.output for task in self._tasks]
      await self._changed.wait()

  async def _gather_items(self, node: NodeLike, items: list[Any]) -> list[Any]:
    """Legacy list fan-out retaining only active handles and ordered outputs."""
    if not self._legacy_run_ids or self._tasks:
      raise RuntimeError('List fan-out requires an empty legacy worker group.')
    self._fail_fast = True
    results: list[Any] = [None] * len(items)
    next_input = 0
    interrupted: NodeInterruptedError | None = None
    while next_input < len(items) or self._tasks:
      while (
          not self._paused
          and next_input < len(items)
          and (self._limit is None or len(self._tasks) < self._limit)
      ):
        self.spawn(node, items[next_input])
        next_input += 1
      if not self._tasks:
        break
      done, _ = await self.wait(mode='any')
      for task in done:
        if task.status == 'waiting':
          interrupted = NodeInterruptedError()
        elif task.status in {'failed', 'cancelled'}:
          await self._drain(self._tasks, deliberate=False, preserve_error=True)
          await task.result()
        else:
          results[task._index] = task.output
        self._tasks.remove(task)
    if interrupted is not None:
      raise interrupted
    return results

  async def first(
      self,
      accept: Callable[[Any], Any] | None = None,
  ) -> ParallelTask:
    """Return an acceptable result after cancelling unfinished siblings.

    The predicate receives the output and may be asynchronous. Rejections and
    worker failures allow remaining candidates to continue. Completion ties are
    resolved in spawn order. Raises ValueError when no result is acceptable.
    Already completed or failed siblings retain their native replay behavior.
    Retrying after a cleanup timeout still requires unfinished siblings to stop.
    """
    if self._selecting:
      raise RuntimeError('Only one first() selection may run at a time.')
    if self._closed:
      raise RuntimeError('The parallel task scope is closed.')
    if self._winner is not None:
      await self._drain(
          [task for task in self._tasks if task.id in self._cancelled_ids],
          deliberate=True,
      )
      await self._restore_completed()
      return self._winner
    self._selecting = True
    try:
      return await self._first(accept)
    finally:
      self._selecting = False
      self._first_mode = False
      self._decision_pending = False
      self._wake.set()

  async def _first(self, accept: Callable[[Any], Any] | None) -> ParallelTask:
    self._fail_fast = False
    self._first_mode = True
    self._decision_pending = any(
        task.status == 'completed' for task in self._tasks
    )
    seen: set[str] = set()
    try:
      if self._winner_id is not None:
        # A failed sibling can retry before the saved winner is admitted. Stop
        # that retry first so it cannot occupy the winner's bounded slot.
        await self._record_selection(self._winner_id)
        await self._drain(
            [task for task in self._tasks if task.id in self._cancelled_ids],
            deliberate=True,
        )
        self._wake.set()
      while True:
        self._changed.clear()
        self._raise_pump_error()
        for task in self._tasks:
          if task.id in seen or not task._done.done():
            continue
          seen.add(task.id)
          if task.status != 'completed':
            continue
          accepted = self._winner_id == task.id
          if self._winner_id is None:
            self._decision_pending = True
            accepted = True if accept is None else accept(task.output)
            if inspect.isawaitable(accepted):
              accepted = await accepted
          if accepted:
            await self._record_selection(task.id)
            self._winner = task
            await self._drain(
                [
                    other
                    for other in self._tasks
                    if other.id in self._cancelled_ids
                ],
                deliberate=True,
            )
            await self._restore_completed()
            return task
        self._decision_pending = any(
            task.status == 'completed' and task.id not in seen
            for task in self._tasks
        )
        self._wake.set()
        if self._decision_pending:
          continue
        running = any(task.status == 'running' for task in self._tasks)
        restoring = self._winner_id is not None and any(
            not task._done.done()
            and (task.id == self._winner_id or task.id in self._completed_ids)
            for task in self._tasks
        )
        if all(task._done.done() for task in self._tasks) or (
            self._paused and not running and not restoring
        ):
          waiting = next(
              (task for task in self._tasks if task.status == 'waiting'), None
          )
          if waiting is not None:
            await waiting.result()
          failed = next(
              (task for task in self._tasks if task.status == 'failed'), None
          )
          if failed is not None:
            await failed.result()
          raise ValueError('No parallel task produced an acceptable result.')
        await self._changed.wait()
    except BaseException:
      await self._drain(
          [
              task
              for task in self._tasks
              if self._winner_id is None or task.id in self._cancelled_ids
          ],
          deliberate=False,
          preserve_error=True,
      )
      raise

  async def _record_selection(self, winner_id: str) -> None:
    # Pause admission before persistence yields to the Runner. Saved completed
    # siblings may still hydrate through the same bounded scheduler.
    self._paused = True
    async with self._decision_lock:
      losers = [
          task.id
          for task in self._tasks
          if task.id != winner_id
          and task.status in {'queued', 'running', 'waiting'}
          and task.id not in self._completed_ids
      ]
      completed = [
          task.id
          for task in self._tasks
          if task.id != winner_id and task.status == 'completed'
      ]
      if (
          self._winner_id is None
          or any(task_id not in self._cancelled_ids for task_id in losers)
          or any(task_id not in self._completed_ids for task_id in completed)
      ):
        await self._record(
            cancelled=losers, winner=winner_id, completed=completed
        )
        self._winner_id = winner_id
        self._cancelled_ids.update(losers)
        self._completed_ids.update(completed)

  async def _restore_completed(self) -> None:
    """Finish native replay of siblings completed before the saved selection."""
    while True:
      self._changed.clear()
      self._raise_pump_error()
      for task in self._tasks:
        if task.id in self._completed_ids and task.status == 'cancelled':
          await task.result()
      if all(
          task._done.done()
          for task in self._tasks
          if task.id in self._completed_ids
      ):
        return
      self._wake.set()
      await self._changed.wait()

  async def cancel(self, task: ParallelTask) -> None:
    """Deliberately cancel one owned task and wait for it to stop."""
    self._require_owned(task)
    if task.status in {'completed', 'failed', 'cancelled'}:
      return
    if self._closed:
      raise RuntimeError('The parallel task scope is closed.')
    async with self._decision_lock:
      # Completion may have won while this caller waited for another decision.
      if task.status in {'completed', 'failed', 'cancelled'}:
        return
      if task.id not in self._cancelled_ids:
        await self._record(cancelled=[task.id])
        self._cancelled_ids.add(task.id)
    # A finishing worker needs the decision lock to publish its terminal state.
    await self._drain([task], deliberate=True)
    if self._winner_id is None:
      self._paused = any(
          other.status == 'waiting'
          or (self._fail_fast and other.status == 'failed')
          for other in self._tasks
      )
    self._wake.set()

  async def merge_result(
      self,
      task: ParallelTask,
      *,
      keys: Iterable[str] | None = None,
      expected: Mapping[str, Any] | None = None,
  ) -> None:
    """Persist a completed mapping proposal after validating every key.

    Selected keys must still match their values when the group was entered,
    unless ``expected`` explicitly supplies another expected value. The whole
    proposal is validated before writing, and a repeated merge of the same task
    and keys is a no-op, including after a resume. The merge disposition and
    state delta are persisted in one native event. Other worker writes remain
    shared; this operation does not roll back their side effects.
    """
    async with self._decision_lock:
      await self._merge_result(task, keys=keys, expected=expected)

  async def _merge_result(
      self,
      task: ParallelTask,
      *,
      keys: Iterable[str] | None,
      expected: Mapping[str, Any] | None,
  ) -> None:
    from ..sessions.state import _validate_state_entry

    if self._closed:
      raise RuntimeError('The parallel task scope is closed.')
    self._require_owned(task)
    if task.status != 'completed' or not isinstance(task.output, Mapping):
      raise ValueError('merge_result requires a completed mapping output.')
    proposal = copy.deepcopy(dict(task.output))
    if any(not isinstance(key, str) for key in proposal):
      raise ValueError('State proposal keys must be strings.')
    if isinstance(keys, str):
      raise ValueError('keys must be an iterable of state keys, not a string.')
    selected = list(proposal) if keys is None else list(dict.fromkeys(keys))
    if any(not isinstance(key, str) or key not in proposal for key in selected):
      raise ValueError('Every selected key must exist in the mapping output.')
    selected = [key for key in selected if (task.id, key) not in self._merged]
    current = self._ctx.state.to_dict()
    for key in selected:
      original = (
          expected[key]
          if expected is not None and key in expected
          else self._state_snapshot.get(key, _MISSING)
      )
      if not _same_value(current.get(key, _MISSING), original):
        raise ValueError(f'State changed before merging key {key!r}.')
    schema = self._ctx.state._schema
    if schema is not None:
      for key in selected:
        _validate_state_entry(schema, key, proposal[key])
    for key in selected:
      self._ctx.state[key] = proposal[key]
    # State's check/write remains synchronous. This event owns these deltas,
    # so they must not be flushed again with the parent's later output event.
    for key in selected:
      self._ctx.actions.state_delta.pop(key, None)
    if not selected:
      return
    await self._record(
        cancelled=[],
        merged={task.id: selected},
        state_delta={key: proposal[key] for key in selected},
    )
    for key in selected:
      self._merged.add((task.id, key))

  def _require_owned(self, task: ParallelTask) -> None:
    if task._group is not self:
      raise ValueError('The task belongs to a different parallel scope.')

  def _raise_pump_error(self) -> None:
    if self._pump_error is not None:
      raise self._pump_error

  async def _record(
      self,
      *,
      cancelled: list[str],
      winner: str | None = None,
      completed: list[str] | None = None,
      merged: dict[str, list[str]] | None = None,
      state_delta: dict[str, Any] | None = None,
  ) -> None:
    # A synthetic child path keeps records in the replay manager's subtree
    # indexes without making them look like completion events for the owner.
    record: dict[str, Any] = {'group': self._id, 'cancelled': cancelled}
    if winner is not None:
      record['winner'] = winner
    if completed:
      record['completed'] = completed
    if merged is not None:
      record['merged'] = merged
    await self._ctx._invocation_context._enqueue_event(
        Event(
            author=self._ctx.node.name
            if self._ctx.node is not None
            else 'parallel',
            invocation_id=self._ctx._invocation_context.invocation_id,
            node_info=NodeInfo(
                path=str(
                    _NodePathBuilder.from_string(self._ctx.node_path).append(
                        '__parallel_tasks_control', self._prefix
                    )
                )
            ),
            actions=EventActions(state_delta=state_delta or {}),
            custom_metadata={_CONTROL_KEY: record},
        )
    )

  async def _pump(self) -> None:
    try:
      while not self._closed:
        self._wake.clear()
        for task in self._tasks:
          if task._cancel_requested and task.id not in self._cancelled_ids:
            await self.cancel(task)
          if task.status == 'cancelled':
            scheduler = self._ctx._workflow_scheduler
            if scheduler is not None:
              scheduler._replay_manager.prepare_parent_sequence_barrier(
                  self._ctx, self._ctx.node_path
              )
              await scheduler._replay_manager.advance_sequence(
                  self._ctx.node_path, f'{task.name}@{task._run_id}'
              )
        active = sum(task.status == 'running' for task in self._tasks)
        for task in self._tasks:
          if self._limit is not None and active >= self._limit:
            break
          if task.status != 'queued':
            continue
          if self._paused or self._decision_pending:
            if self._winner_id is None:
              break
            if (
                task.id != self._winner_id
                and task.id not in self._completed_ids
            ):
              continue
          task._status = 'running'
          task._control.status = 'running'
          task._runner = asyncio.create_task(self._execute(task))
          active += 1
        await self._wake.wait()
    except asyncio.CancelledError:
      raise
    except BaseException as error:
      self._pump_error = error
      self._changed.set()

  async def _execute(self, task: ParallelTask) -> None:
    child: Context | None = None
    execution_error: BaseException | None = None
    try:
      try:
        # Legacy workers preserve their event format while inheriting any outer
        # public task's ownership, including cancellation and transfer tracking.
        with (
            nullcontext()
            if self._legacy_run_ids
            else _bind_parallel_task(task._control)
        ):
          child = await self._ctx._run_node_internal(
              task._node,
              task._input,
              run_id=task._run_id,
              use_sub_branch=True,
              return_ctx=True,
              skip_run_id_validation=self._legacy_run_ids,
          )
        # Cancellation can interrupt the following lock acquisition. Expose
        # returned HITL IDs first so deliberate drain can always remove them.
        task._interrupt_ids = set(child.interrupt_ids)
      except asyncio.CancelledError:
        raise
      except BaseException as error:
        execution_error = error
      # Persistence yields to the Runner. Serialize terminal publication with
      # cancellation decisions so observers and replay see the same outcome.
      async with self._decision_lock:
        if task.id in self._cancelled_ids:
          if child is not None:
            self._ctx._interrupt_ids.difference_update(child.interrupt_ids)
            scheduler = self._ctx._workflow_scheduler
            if scheduler is not None:
              scheduler._state.interrupt_ids.difference_update(
                  child.interrupt_ids
              )
          task._finish('cancelled')
          return
        if execution_error is not None:
          raise execution_error
        assert child is not None
        if child.error is not None:
          raise DynamicNodeFailError(
              message=f'Dynamic node {task.name} failed',
              error=child.error,
              error_node_path=child.error_node_path,
          )
        if child.interrupt_ids:
          self._ctx._interrupt_ids.update(task._interrupt_ids)
          raise NodeInterruptedError()
        task._output = child.output
        if self._first_mode:
          self._decision_pending = True
        task._finish('completed')
    except asyncio.CancelledError:
      task._finish('cancelled')
      raise
    except NodeInterruptedError as error:
      self._paused = True
      task._finish('waiting', error)
    except BaseException as error:
      if self._fail_fast:
        self._paused = True
      task._finish('failed', error)

  async def _drain(
      self,
      tasks: list[ParallelTask],
      *,
      deliberate: bool,
      preserve_error: bool = False,
  ) -> None:
    running: list[asyncio.Task[None]] = []
    deadlines: list[float] = []
    loop = asyncio.get_running_loop()
    for task in tasks:
      if deliberate:
        task._control.cancellation_requested = True
      if deliberate and task._interrupt_ids:
        self._ctx._interrupt_ids.difference_update(task._interrupt_ids)
        scheduler = self._ctx._workflow_scheduler
        if scheduler is not None:
          scheduler._state.interrupt_ids.difference_update(task._interrupt_ids)
      if task.status == 'queued' or (deliberate and task.status == 'waiting'):
        task._finish('cancelled')
      if task._runner is not None and not task._runner.done():
        if not task._cancellation_sent:
          task._runner.cancel()
          task._cancellation_sent = True
        if task._cancel_deadline is None:
          task._cancel_deadline = loop.time() + self._timeout
        deadlines.append(task._cancel_deadline)
        running.append(task._runner)
    if not running:
      return
    done, pending = await asyncio.wait(
        running, timeout=max(0.0, max(deadlines) - loop.time())
    )
    for runner in done:
      if not runner.cancelled():
        runner.exception()
    if pending:
      message = (
          f'Parallel group {self._id}: {len(pending)} task(s) did not stop'
          f' within {self._timeout}s.'
      )
      if preserve_error:
        logger.warning(message)
      else:
        raise TimeoutError(message)
