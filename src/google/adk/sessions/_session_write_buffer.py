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

"""Per-session writers, ordering and failure reports for BufferedSessionService.

Everything here runs on one event loop and is not thread-safe. Code between two
awaits runs without interruption, so the buffer needs no lock.
"""

from __future__ import annotations

import asyncio
import collections
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
import copy
import logging
from typing import NamedTuple
import weakref

from ..errors._session_persistence_error import SessionPersistenceError
from ..errors._stale_session_error import StaleSessionError
from ..events.event import Event
from .session import Session
from .state import State

logger = logging.getLogger('google_adk.' + __name__)

MAX_UNREPORTED_FAILURES = 1024
"""How many sessions keep a failure to report before the oldest is forgotten."""


class SessionKey(NamedTuple):
  """Identifies one session."""

  app_name: str
  user_id: str
  session_id: str


class _Scope(NamedTuple):
  """State shared across sessions: `app:` state (user_id None) or `user:`."""

  app_name: str
  user_id: str | None


_Version = tuple[int, str | None, float, str | None]
"""A session's event count, last event id, update time and storage marker."""


def changed_scopes(
    key: SessionKey, state_keys: Iterable[str]
) -> tuple[_Scope, ...]:
  """Returns the shared scopes that a change to `state_keys` touches."""
  keys = list(state_keys)
  scopes = []
  if any(k.startswith(State.USER_PREFIX) for k in keys):
    scopes.append(_Scope(key.app_name, key.user_id))
  if any(k.startswith(State.APP_PREFIX) for k in keys):
    scopes.append(_Scope(key.app_name, None))
  return tuple(scopes)


def _sees(key: SessionKey, scope: _Scope) -> bool:
  """Returns whether a read of the session `key` sees `scope`."""
  return scope.app_name == key.app_name and scope.user_id in (
      None,
      key.user_id,
  )


PersistFn = Callable[[Session, Event], Awaitable[BaseException | None]]
"""Writes one event through a store session.

Returns the error that stopped the write once retries ran out, or None.
"""


class _Write:
  """An event appended to a live session, on its way to the store."""

  __slots__ = (
      'key',
      'event',
      'scopes',
      'after',
      'writer',
      'store_session',
      'live_session',
      '_result',
  )

  def __init__(
      self,
      *,
      key: SessionKey,
      event: Event,
      scopes: tuple[_Scope, ...],
      after: Sequence[_Write],
      writer: _SessionWriter,
      store_session: Session,
      live_session: Session,
  ) -> None:
    self.key = key
    self.event = event
    # Taken at append time, since the store may change the event.
    self.scopes = scopes
    # Earlier writes of other sessions to the same shared scopes.
    self.after = tuple(after)
    self.writer = writer
    self.store_session: Session | None = store_session
    self.live_session: Session | None = live_session
    # Holds the failure, or None, as a result rather than an exception, so a
    # failure nobody waited for is not logged as never retrieved.
    self._result: asyncio.Future[BaseException | None] = (
        asyncio.get_running_loop().create_future()
    )

  @property
  def done(self) -> bool:
    return self._result.done()

  @property
  def failure(self) -> BaseException | None:
    """The error that stopped the write, or None. Only valid once done."""
    return self._result.result()

  async def join(self) -> None:
    """Waits for the write. Canceling the caller leaves the write running."""
    await asyncio.wait([self._result])

  def release(self) -> None:
    """Drops the references a completed or abandoned write no longer needs."""
    self.after = ()
    self.store_session = None
    self.live_session = None

  def complete(self, failure: BaseException | None) -> None:
    if not self._result.done():
      self._result.set_result(failure)
    self.release()


class _SessionWriter:
  """Writes one session's events in order.

  Each write goes through a store session, which carries the state and version
  the store needs. The appends of one session object, and of copies with the
  same history, share one store session, so they continue each other's writes.
  """

  def __init__(self, key: SessionKey, buffer: SessionWriteBuffer) -> None:
    self.key = key
    self._buffer = buffer
    self._queue: collections.deque[_Write] = collections.deque()
    # The write being sent or dropped, which has left the queue.
    self._current: _Write | None = None
    self.last_write: _Write | None = None
    self._task: asyncio.Task[None] | None = None
    self._task_started = False
    # Once a non-stale write fails, every later write of this writer fails too.
    self._failure: BaseException | None = None
    # Chains that failed with StaleSessionError, by id(store_session).
    self._failed_chains: dict[int, tuple[Session, BaseException]] = {}
    # Set when the writer's loop closed; it must not touch the buffer again.
    self.abandoned = False
    # Each chain's store session, by the version its last append left on the
    # live session.
    self._chains: dict[_Version, Session] = {}

  def chain_failure(
      self, store_session: Session | None
  ) -> BaseException | None:
    """Returns the error that stopped `store_session`'s chain, if any."""
    if self._failure is not None:
      return self._failure
    if store_session is None:
      return None
    entry = self._failed_chains.get(id(store_session))
    if entry is not None and entry[0] is store_session:
      return entry[1]
    return None

  def store_session_for(self, live_session: Session) -> Session:
    """Returns the store session for the next append from `live_session`.

    Called before the append changes the live session.
    """
    store_session = self._chains.get(_version_of(live_session))
    if store_session is not None:
      return store_session
    return _new_store_session(live_session)

  def write(
      self,
      event: Event,
      *,
      scopes: tuple[_Scope, ...],
      after: Sequence[_Write],
      store_session: Session,
      live_session: Session,
      version_before: _Version,
  ) -> _Write:
    """Queues the event that was just applied to `live_session`."""
    write = _Write(
        key=self.key,
        event=event,
        scopes=scopes,
        after=after,
        writer=self,
        store_session=store_session,
        live_session=live_session,
    )
    if self._chains.get(version_before) is store_session:
      del self._chains[version_before]
    self._chains[_version_of(live_session)] = store_session
    self.last_write = write
    self._queue.append(write)
    chain_failure = self.chain_failure(store_session)
    if chain_failure is not None and write.live_session is not None:
      self._buffer.mark_failed_live_session(write.live_session, chain_failure)
    if self._task is None or self._task.done():
      if self._task is not None and not self._task_started:
        # The last task was canceled before it ran, and the writes it left
        # still wait. They fail, and so does this one, in order.
        self._fail(asyncio.CancelledError())
      self._start_task()
    return write

  @property
  def failed(self) -> bool:
    """Whether a write failed, so the writer only drops writes now."""
    return self._failure is not None

  def pending_writes(self) -> list[_Write]:
    """Returns the writes not completed yet, in order."""
    current = [self._current] if self._current is not None else []
    return current + list(self._queue)

  def _start_task(self) -> None:
    self._task_started = False
    self._task = asyncio.get_running_loop().create_task(self._send_queued())
    self._task.add_done_callback(self._on_task_done)
    self._buffer.track_task(self._task, self)

  def _fail(self, failure: BaseException) -> None:
    """Drops pending and later writes, and marks their sessions failed."""
    if self._failure is None:
      self._failure = failure
    for write in self.pending_writes():
      if write.live_session is not None:
        self._buffer.mark_failed_live_session(write.live_session, self._failure)

  def _fail_chain(
      self, store_session: Session | None, failure: BaseException
  ) -> None:
    """Drops pending and later writes of `store_session` only."""
    if store_session is None:
      self._fail(failure)
      return
    sid = id(store_session)
    entry = self._failed_chains.get(sid)
    if entry is None or entry[0] is not store_session:
      self._failed_chains[sid] = (store_session, failure)
      chain_failure = failure
    else:
      chain_failure = entry[1]
    for write in self.pending_writes():
      if (
          write.store_session is store_session
          and write.live_session is not None
      ):
        self._buffer.mark_failed_live_session(write.live_session, chain_failure)

  async def _send_queued(self) -> None:
    self._task_started = True
    try:
      while self._queue and not self.abandoned:
        write = self._current = self._queue.popleft()
        # Even a write that will be dropped waits, so that it completes, and
        # releases the shared scopes it changes, only after the writes of
        # other sessions it follows. Otherwise a later change to those scopes
        # could overtake them.
        await self._wait_for_earlier_shared_state_writes(write)
        write_failure = self.chain_failure(write.store_session)
        if write_failure is None:
          write_failure = await self._persist(write)
          if write_failure is not None:
            if isinstance(write_failure, StaleSessionError):
              self._fail_chain(write.store_session, write_failure)
            else:
              self._fail(write_failure)
        if self.abandoned:
          return
        self._current = None
        self._complete(write, write_failure)
    except GeneratorExit:
      # The loop closed with this task pending and is being collected. The
      # buffer failed these writes when it moved to the next loop.
      raise
    except BaseException as e:  # pylint: disable=broad-exception-caught
      # Canceled, an error that is not an Exception, which retries skip, or a
      # bug outside the store call. Only the second stops every writer.
      if self.abandoned:
        raise
      self._fail(e)
      if not isinstance(e, (Exception, asyncio.CancelledError)):
        self._buffer.stop_on(e)
      await self._drop_pending_ignoring_cancellation()
      if isinstance(e, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
        raise
      # Not raised further: nothing awaits this task, and every caller that
      # waits for one of its writes gets the failure instead.

  async def _persist(self, write: _Write) -> BaseException | None:
    """Sends the write and returns its failure, or None."""
    store_session = write.store_session
    if store_session is None:
      # Only a write abandoned with a closed loop has none.
      return RuntimeError('The write was abandoned.')
    failure = await self._buffer.persist(store_session, write.event)
    # The store adds each event it persists to the store session, after any
    # events of other writers that it merged in on a conflict. The live
    # session holds its own events already, so the store session keeps none.
    merged = [e for e in store_session.events if e.id != write.event.id]
    store_session.events.clear()
    if failure is not None:
      logger.warning(
          'A buffered session write failed; later ones are dropped (%s).',
          type(failure).__name__,
      )
    else:
      if merged:
        self._merge_into_live_session(write, store_session, merged)
      self._sync_live_session(write, store_session)
    return failure

  def _merge_into_live_session(
      self, write: _Write, store_session: Session, merged: list[Event]
  ) -> None:
    """Copies what a store merged in on a conflict to the caller's session.

    The merged events go before the write's event, as the store put them. The
    merged state replaces the caller's, except for `temp:` state and keys that
    the caller's queued writes still change.
    """
    live = write.live_session
    if live is None:
      return
    version_before = _version_of(live)
    known = {e.id for e in live.events}
    new_events = [e for e in merged if e.id not in known]
    at = next(
        (i for i, e in enumerate(live.events) if e.id == write.event.id),
        len(live.events),
    )
    live.events[at:at] = new_events
    queued_keys = {
        k
        for w in self._queue
        if w.live_session is live
        for k in w.event.actions.state_delta
    }
    for k, v in store_session.state.items():
      if not k.startswith(State.TEMP_PREFIX) and k not in queued_keys:
        live.state[k] = copy.deepcopy(v)
    if self._chains.get(version_before) is store_session:
      del self._chains[version_before]
      self._chains[_version_of(live)] = store_session

  def _sync_live_session(self, write: _Write, store_session: Session) -> None:
    """Copies the stored version to the caller's session."""
    live = write.live_session
    if live is None or any(w.live_session is live for w in self._queue):
      return
    version_before = _version_of(live)
    _copy_version(store_session, live)
    if self._chains.get(version_before) is store_session:
      del self._chains[version_before]
      self._chains[_version_of(live)] = store_session

  async def _wait_for_earlier_shared_state_writes(self, write: _Write) -> None:
    # A failed write of another session does not drop this one: the shared
    # state change is this session's own, so it is still sent.
    for earlier in write.after:
      await earlier.join()

  async def _drop_pending_ignoring_cancellation(self) -> None:
    """Fails the pending writes in order, after the writes they wait for."""
    while not self.abandoned and (self._current or self._queue):
      write = self._current or self._queue.popleft()
      self._current = write
      for earlier in write.after:
        while not earlier.done:
          try:
            await earlier.join()
          except asyncio.CancelledError:
            pass
      if self.abandoned:
        return
      self._current = None
      self._complete(write, self._failure)

  def _on_task_done(self, task: asyncio.Task[None]) -> None:
    self._buffer.untrack_task(task)
    if self.abandoned or task is not self._task or self._task_started:
      return
    if self.pending_writes():
      # Canceled before its first step, so its body never ran. A new task
      # drops the writes it left, in order.
      self._fail(asyncio.CancelledError())
      self._start_task()

  def _complete(self, write: _Write, failure: BaseException | None) -> None:
    self._buffer.before_complete(self, write, failure)
    write.complete(failure)


def _version_of(session: Session) -> _Version:
  """Returns the session's history and storage version.

  Two session objects with the same version hold the same events, so their
  appends can share a chain.
  """
  last_event_id = session.events[-1].id if session.events else None
  return (
      len(session.events),
      last_event_id,
      session.last_update_time,
      _marker(session),
  )


def _marker(session: Session) -> str | None:
  """Returns the session's storage marker.

  A store may delete it from a session, to have it read again.
  """
  return getattr(session, '_storage_update_marker', None)


def _copy_version(source: Session, target: Session) -> None:
  target.last_update_time = source.last_update_time
  target._storage_update_marker = _marker(source)  # pylint: disable=protected-access


def _new_store_session(live: Session) -> Session:
  """Returns a store session for a new chain, from the caller's session.

  It leaves out `temp:` state, which the store never has.
  """
  store_session = Session(
      id=live.id,
      app_name=live.app_name,
      user_id=live.user_id,
      state={
          k: copy.deepcopy(v)
          for k, v in live.state.items()
          if not k.startswith(State.TEMP_PREFIX)
      },
      last_update_time=live.last_update_time,
  )
  _copy_version(live, store_session)
  return store_session


class SessionWriteBuffer:
  """Buffers the writes of many sessions until the store has them."""

  def __init__(
      self,
      persist: PersistFn,
      *,
      max_unreported_failures: int = MAX_UNREPORTED_FAILURES,
  ) -> None:
    self.persist = persist
    self._max_unreported_failures = max_unreported_failures
    self._loop: asyncio.AbstractEventLoop | None = None
    self._is_closed = False
    # Set once the first `aclose` has ended.
    self._closed: asyncio.Event | None = None
    # The error, not an Exception, that stopped every writer.
    self._stopped_by: BaseException | None = None
    # Only sessions with writes in flight have a writer.
    self._writers: dict[SessionKey, _SessionWriter] = {}
    # Every writer task still running, with its writer, including the tasks
    # of writers retired while they drop their writes.
    self._tasks: dict[asyncio.Task[None], _SessionWriter] = {}
    self._last_write_by_scope: dict[_Scope, _Write] = {}
    # Each session's first unreported failure, oldest first.
    self._failures: collections.OrderedDict[SessionKey, BaseException] = (
        collections.OrderedDict()
    )
    # The failures that a flush of every session already raised.
    self._failures_raised_by_flush_all: set[SessionKey] = set()
    # Session objects holding an event whose write failed, by id().
    self._failed_live_sessions: dict[
        int, tuple[weakref.ref[Session], BaseException]
    ] = {}

  def check_loop(self) -> None:
    """Binds the buffer to the running loop; raises on any other loop."""
    loop = asyncio.get_running_loop()
    if self._loop is loop:
      return
    closing = self._closed is not None and not self._closed.is_set()
    if self._loop is not None and (self._writers or self._tasks or closing):
      if not self._loop.is_closed():
        raise RuntimeError(
            'BufferedSessionService has writes in flight on another event'
            ' loop. Use one instance for each event loop, or flush before'
            ' switching.'
        )
      self._drop_writes_of_closed_loop()
    self._loop = loop

  def _drop_writes_of_closed_loop(self) -> None:
    """Fails the writes left behind by a loop that closed."""
    failure = RuntimeError(
        'The event loop closed before the buffered session writes completed.'
    )
    writers = dict(self._writers)
    abandoned = {id(w): w for w in [*writers.values(), *self._tasks.values()]}
    self._writers.clear()
    self._tasks.clear()
    if self._closed is not None and not self._closed.is_set():
      # A close that the loop cut off is redone by the next one, on this loop.
      self._closed = None
    self._last_write_by_scope.clear()
    for writer in abandoned.values():
      writer.abandoned = True
      for write in writer.pending_writes():
        if write.live_session is not None:
          self.mark_failed_live_session(write.live_session, failure)
        write.release()
    for key in writers:
      self._record_failure(key, failure)

  def append(
      self, session: Session, snapshot: Event, apply: Callable[[], Event]
  ) -> Event:
    """Applies an event with `apply` and queues the write of `snapshot`."""
    self.check_loop()
    if self._is_closed:
      raise RuntimeError('BufferedSessionService is closed.')
    if self._stopped_by is not None:
      raise RuntimeError(
          'BufferedSessionService has stopped.'
      ) from self._stopped_by
    key = SessionKey(session.app_name, session.user_id, session.id)
    if not isinstance(self._failures.get(key), StaleSessionError):
      self._report_failure(key)
    self._check_live_session(session)
    writer = self._writers.get(key)
    version_before = _version_of(session)
    # Taken before `apply`, so a new store session starts from the state the
    # store has.
    store_session = (
        writer.store_session_for(session)
        if writer
        else _new_store_session(session)
    )
    if (
        writer is not None
        and (chain_failure := writer.chain_failure(store_session)) is not None
    ):
      self.mark_failed_live_session(session, chain_failure)
      self._check_live_session(session)
    appended = apply()
    scopes = changed_scopes(key, snapshot.actions.state_delta)
    if writer is None:
      writer = _SessionWriter(key, self)
      self._writers[key] = writer
    after = [
        self._last_write_by_scope[s]
        for s in scopes
        if s in self._last_write_by_scope
    ]
    write = writer.write(
        snapshot,
        scopes=scopes,
        after=after,
        store_session=store_session,
        live_session=session,
        version_before=version_before,
    )
    for scope in scopes:
      self._last_write_by_scope[scope] = write
    return appended

  async def await_writes_seen_by(self, key: SessionKey) -> None:
    """Waits for the session's writes and the writes to the scopes it sees."""
    writer = self._writers.get(key)
    writes = [writer.last_write] if writer else []
    writes += [w for s, w in self._last_write_by_scope.items() if _sees(key, s)]
    await self._await(writes)

  async def await_writes_listed_for(
      self, app_name: str, user_id: str | None
  ) -> None:
    """Waits for the writes a listing would see."""
    writes = [
        w.last_write
        for k, w in self._writers.items()
        if k.app_name == app_name and user_id in (None, k.user_id)
    ]
    writes += [
        w
        for s, w in self._last_write_by_scope.items()
        if s.app_name == app_name
        and (user_id is None or s.user_id in (None, user_id))
    ]
    await self._await(writes)

  async def await_user_state_writes(self, app_name: str, user_id: str) -> None:
    """Waits for the writes that change the user's `user:` state."""
    write = self._last_write_by_scope.get(_Scope(app_name, user_id))
    await self._await([write])

  async def flush(
      self,
      key: SessionKey,
      before_report: Callable[[], Awaitable[None]],
  ) -> None:
    """Waits for the session's writes, runs `before_report`, then reports."""
    writer = self._writers.get(key)
    waited = writer.last_write if writer else None
    waited_failure = None
    if waited is not None:
      await waited.join()
      waited_failure = waited.failure
    await before_report()
    self._report_failure(key, waited_failure)

  async def flush_all(
      self, before_report: Callable[[], Awaitable[None]]
  ) -> None:
    """Waits for every write, runs `before_report`, then reports failures."""
    await self._await([w.last_write for w in self._writers.values()])
    await before_report()
    keys = [
        key
        for key in self._failures
        if key not in self._failures_raised_by_flush_all
    ]
    if not keys:
      return
    self._failures_raised_by_flush_all.update(keys)
    causes = [self._failures[key] for key in keys]
    error = SessionPersistenceError(
        f'Buffered writes of {len(causes)} session(s) could not be persisted.'
    )
    for cause in causes[1:]:
      error.add_note(
          f'Another session failed with {type(cause).__name__}: {cause}'
      )
    raise error from causes[0]

  async def aclose(self, before_report: Callable[[], Awaitable[None]]) -> None:
    """Rejects later appends, flushes, then stops the writers."""
    if self._closed is not None:
      await self._closed.wait()
      await self._stop_writers()
      return
    self._is_closed = True
    self._closed = asyncio.Event()
    try:
      await self.flush_all(before_report)
    finally:
      try:
        await self._stop_writers()
      finally:
        self._closed.set()

  def stop_on(self, error: BaseException) -> None:
    """Stops every writer after an error that is not an Exception."""
    if self._stopped_by is not None:
      return
    self._stopped_by = error
    current = asyncio.current_task()
    for task in list(self._tasks):
      if task is not current:
        task.cancel()

  def track_task(
      self, task: asyncio.Task[None], writer: _SessionWriter
  ) -> None:
    self._tasks[task] = writer

  def untrack_task(self, task: asyncio.Task[None]) -> None:
    self._tasks.pop(task, None)

  async def _stop_writers(self) -> None:
    # Repeats until no writer task runs, and defers the caller's cancellation
    # so that it does not reach tasks that are cleaning up.
    canceled: asyncio.CancelledError | None = None
    stopped: set[asyncio.Task[None]] = set()
    while tasks := [t for t in self._tasks if not t.done()]:
      for task in tasks:
        if task not in stopped and not self._tasks[task].failed:
          task.cancel()
          stopped.add(task)
      try:
        await asyncio.wait(tasks)
        # Lets the done callbacks run, which may start such tasks.
        await asyncio.sleep(0)
      except asyncio.CancelledError as e:
        canceled = e
    if canceled is not None:
      raise canceled

  async def _await(self, writes: Sequence[_Write | None]) -> None:
    for w in writes:
      if w is not None and not w.done:
        await w.join()

  def before_complete(
      self,
      writer: _SessionWriter,
      write: _Write,
      failure: BaseException | None,
  ) -> None:
    """Runs before the write's waiters see its result."""
    for scope in write.scopes:
      if self._last_write_by_scope.get(scope) is write:
        del self._last_write_by_scope[scope]
    if failure is not None and write.live_session is not None:
      self.mark_failed_live_session(write.live_session, failure)
    # A retired writer's failure was reported or forgotten.
    if self._writers.get(writer.key) is not writer:
      return
    if writer.last_write is write:
      # The writer is idle. A failure stays in `_failures` for the session to
      # report, so no writer holds on to a loop that may be gone.
      del self._writers[writer.key]
    if failure is not None:
      self._record_failure(writer.key, failure)

  def _record_failure(self, key: SessionKey, failure: BaseException) -> None:
    existing = self._failures.get(key)
    if existing is not None:
      if isinstance(existing, StaleSessionError) and not isinstance(
          failure, StaleSessionError
      ):
        self._failures[key] = failure
        self._failures_raised_by_flush_all.discard(key)
      return
    self._failures[key] = failure
    # Once the buffer stopped, the cascade of failed writes must not push out
    # the failure of the error that stopped it.
    if (
        len(self._failures) > self._max_unreported_failures
        and self._stopped_by is None
    ):
      self._retire(next(iter(self._failures)))
      # No key: a session id can carry user data.
      logger.warning(
          'Forgot the oldest unreported buffered-write failure; its session'
          ' will not report it.'
      )

  def mark_failed_live_session(
      self, session: Session, failure: BaseException
  ) -> None:
    """Rejects later appends to `session`, which holds a dropped event."""
    sid = id(session)
    entry = self._failed_live_sessions.get(sid)
    if entry is not None and entry[0]() is session:
      return
    self._failed_live_sessions[sid] = (
        weakref.ref(session, self._forget_failed_live_session(sid)),
        failure,
    )

  def _forget_failed_live_session(
      self, sid: int
  ) -> Callable[[weakref.ref[Session]], None]:
    def forget(ref: weakref.ref[Session]) -> None:
      entry = self._failed_live_sessions.get(sid)
      if entry is not None and entry[0] is ref:
        del self._failed_live_sessions[sid]

    return forget

  def _check_live_session(self, session: Session) -> None:
    entry = self._failed_live_sessions.get(id(session))
    if entry is not None and entry[0]() is session:
      key = SessionKey(session.app_name, session.user_id, session.id)
      if self._failures.get(key) is entry[1]:
        self._retire(key)
      raise SessionPersistenceError(
          'This session object holds an event whose buffered write could not'
          ' be persisted. Read the session again before appending to it.'
      ) from entry[1]

  def _report_failure(
      self, key: SessionKey, waited_failure: BaseException | None = None
  ) -> None:
    failure = self._failures.get(key)
    if failure is not None:
      self._retire(key)
    cause = failure or waited_failure
    if cause is not None:
      raise SessionPersistenceError(
          'A buffered write of this session could not be persisted. Read the'
          ' session again before appending to it.'
      ) from cause

  def _retire(self, key: SessionKey) -> None:
    """Clears the session's recorded failure, and drops a failed writer."""
    writer = self._writers.get(key)
    if writer is not None and writer.failed:
      del self._writers[key]
    self._failures.pop(key, None)
    self._failures_raised_by_flush_all.discard(key)
