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

"""Tests for BufferedSessionService.

Verifies that appends return before the store has them, that writes reach the
store in order, that reads wait for the writes they can see, and that failed
writes are retried and then reported. The shared session service contract
tests in test_session_service.py also run against BufferedSessionService.
"""

from __future__ import annotations

import asyncio
import collections
import copy
import gc
import logging
import random
import threading
from typing import Any

from google.adk import platform as adk_platform
from google.adk.errors import SessionPersistenceError
from google.adk.errors import StaleSessionError
from google.adk.errors.session_not_found_error import SessionNotFoundError
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions import BufferedSessionService
from google.adk.sessions import BufferedSessionServiceConfig
from google.adk.sessions import DatabaseSessionService
from google.adk.sessions import InMemorySessionService
from google.adk.sessions.base_session_service import GetSessionConfig
from google.adk.sessions.session import Session
from google.adk.workflow import RetryConfig
from google.genai import types
import pytest

_APP = 'app'
_USER = 'user'


class _Store(InMemorySessionService):
  """An in-memory store whose writes a test can hold, fail and observe."""

  def __init__(self) -> None:
    super().__init__()
    self.gates: dict[str, asyncio.Event] = {}
    self.started: collections.defaultdict[str, asyncio.Event] = (
        collections.defaultdict(asyncio.Event)
    )
    # Raised by the next append attempts, in order, before the store commits.
    self.failures: list[BaseException] = []
    self.attempts: list[str] = []
    self.flushes: list[tuple[str | None, str | None, str | None]] = []

  def hold(self, session_id: str = 's1') -> asyncio.Event:
    """Holds the session's writes until the returned event is set."""
    gate = asyncio.Event()
    self.gates[session_id] = gate
    return gate

  async def append_event(self, session: Session, event: Event) -> Event:
    self.attempts.append(event.id)
    self.started[session.id].set()
    gate = self.gates.get(session.id)
    if gate is not None:
      await gate.wait()
    if self.failures:
      raise self.failures.pop(0)
    return await super().append_event(session, event)

  async def flush_session(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> None:
    self.flushes.append((app_name, user_id, session_id))

  async def flush(self) -> None:
    self.flushes.append((None, None, None))

  def stored_ids(
      self, session_id: str = 's1', user_id: str = _USER
  ) -> list[str]:
    return [e.id for e in self.sessions[_APP][user_id][session_id].events]


class _MergingStore(InMemorySessionService):
  """Merges newer stored events and state into a stale session, as Orcas does.

  On a version conflict, Orcas reads what other writers committed into the
  session it was given and retries the write.
  """

  async def append_event(self, session: Session, event: Event) -> Event:
    stored = self.sessions[session.app_name][session.user_id][session.id]
    if session.last_update_time < stored.last_update_time:
      known = {e.id for e in session.events}
      session.events[:0] = [
          copy.deepcopy(e) for e in stored.events if e.id not in known
      ]
      session.state = copy.deepcopy(stored.state)
      session.last_update_time = stored.last_update_time
    return await super().append_event(session, event)


def _event(event_id: str, **state_delta: Any) -> Event:
  return Event(
      id=event_id,
      author='agent',
      invocation_id='inv',
      content=types.Content(role='model', parts=[types.Part(text=event_id)]),
      actions=EventActions(state_delta=state_delta),
  )


def _service(
    store: _Store | None = None, **retry: Any
) -> BufferedSessionService:
  retry.setdefault('initial_delay', 0.0)
  return BufferedSessionService(
      store or _Store(),
      config=BufferedSessionServiceConfig(retry_config=RetryConfig(**retry)),
  )


async def _create(
    service: BufferedSessionService,
    session_id: str = 's1',
    user_id: str = _USER,
    **state: Any,
) -> Session:
  return await service.create_session(
      app_name=_APP, user_id=user_id, session_id=session_id, state=state
  )


async def _flush(
    service: BufferedSessionService, session_id: str = 's1', user_id=_USER
) -> None:
  await service.flush_session(
      app_name=_APP, user_id=user_id, session_id=session_id
  )


async def _started(task: asyncio.Task[Any]) -> None:
  """Lets `task` run until it blocks, and checks that it blocked.

  A task created before this call runs up to its first wait before the caller
  resumes, since the loop runs ready tasks in order.
  """
  await asyncio.sleep(0)
  assert not task.done()


# --- Appending and writing ---


async def test_append_event_store_blocked_returns_before_the_store_has_it():
  """An append updates the live session and returns before the store has it."""
  store = _Store()
  service = _service(store)
  session = await _create(service)
  gate = store.hold()

  await service.append_event(session, _event('e1'))

  assert [e.id for e in session.events] == ['e1']
  assert store.stored_ids() == []
  gate.set()
  await _flush(service)
  assert store.stored_ids() == ['e1']


async def test_flush_several_appends_persists_them_in_append_order():
  """Several appends to one session reach the store in append order."""
  store = _Store()
  service = _service(store)
  session = await _create(service)

  for i in range(5):
    await service.append_event(session, _event(f'e{i}'))
  await _flush(service)

  assert store.stored_ids() == ['e0', 'e1', 'e2', 'e3', 'e4']


async def test_append_event_changed_afterward_persists_the_appended_version():
  """Changing the appended event afterward does not change the stored one."""
  store = _Store()
  service = _service(store)
  session = await _create(service)
  gate = store.hold()
  appended = await service.append_event(session, _event('e1', k='v'))

  appended.actions.state_delta['k'] = 'changed'
  appended.content.parts[0].text = 'changed'
  appended.custom_metadata = {'changed': True}
  gate.set()
  await _flush(service)

  stored = await store.get_session(
      app_name=_APP, user_id=_USER, session_id='s1'
  )
  assert stored.state['k'] == 'v'
  assert stored.events[0].content.parts[0].text == 'e1'
  assert stored.events[0].custom_metadata is None


async def test_append_event_partial_never_reaches_the_store():
  """A partial event is neither applied nor written."""
  store = _Store()
  service = _service(store)
  session = await _create(service)
  event = _event('e1')
  event.partial = True

  returned = await service.append_event(session, event)
  await _flush(service)

  assert returned is event
  assert session.events == []
  assert store.attempts == []


async def test_append_event_without_session_id_raises():
  """An append to a session without an id raises ValueError."""
  service = _service()
  session = Session(id='', app_name=_APP, user_id=_USER)

  with pytest.raises(ValueError, match='session id'):
    await service.append_event(session, _event('e1'))


async def test_append_event_temp_state_stays_on_the_live_session_only():
  """`temp:` state reaches the live session, but not the store."""
  store = _Store()
  service = _service(store)
  session = await _create(service)

  await service.append_event(session, _event('e1', **{'temp:t': 1, 'k': 2}))
  await _flush(service)

  stored = await store.get_session(
      app_name=_APP, user_id=_USER, session_id='s1'
  )
  assert session.state == {'temp:t': 1, 'k': 2}
  assert stored.state == {'k': 2}


async def test_append_event_store_reads_session_state_gets_the_stored_state():
  """A store that writes from the session's state gets the stored state."""
  seen_states = []

  class _StateReadingStore(_Store):

    async def append_event(self, session: Session, event: Event) -> Event:
      seen_states.append(dict(session.state))
      return await super().append_event(session, event)

  service = _service(_StateReadingStore())
  session = await _create(service, k='v0')
  session.state['temp:t'] = 'live only'

  await service.append_event(session, _event('e1', k='v1'))
  await service.append_event(session, _event('e2'))
  await _flush(service)

  assert seen_states == [{'k': 'v0'}, {'k': 'v1'}]


async def test_flush_store_with_its_own_version_accepts_each_write():
  """Each write sends the version the store set after the previous write."""

  class _VersionedStore(_Store):
    """Rejects a write that does not send the store's current version."""

    version = 0.0

    async def create_session(self, **kwargs: Any) -> Session:
      session = await super().create_session(**kwargs)
      self.version = session.last_update_time
      return session

    async def append_event(self, session: Session, event: Event) -> Event:
      if session.last_update_time != self.version:
        raise StaleSessionError('stale')
      await super().append_event(session, event)
      self.version += 1000.0
      session.last_update_time = self.version
      return event

  store = _VersionedStore()
  service = _service(store)
  session = await _create(service)

  for i in range(3):
    await service.append_event(session, _event(f'e{i}'))
  await _flush(service)
  await service.append_event(session, _event('e3'))
  await _flush(service)

  assert store.stored_ids() == ['e0', 'e1', 'e2', 'e3']
  assert session.last_update_time == store.version


async def test_append_event_that_cannot_be_copied_leaves_the_session_unchanged():
  """An event that cannot be copied raises before the session changes."""
  service = _service()
  session = await _create(service)
  event = _event('e1', lock=threading.Lock())

  with pytest.raises(TypeError):
    await service.append_event(session, event)

  assert session.events == []
  assert session.state == {}


async def test_append_event_last_update_time_never_moves_back():
  """A session's `last_update_time` stays at its newest event while writing."""
  store = _Store()
  service = _service(store)
  session = await _create(service)
  first_gate = store.hold()
  await service.append_event(session, _event('e1'))
  await store.started['s1'].wait()
  second = _event('e2')
  second.timestamp = session.last_update_time + 100
  await service.append_event(session, second)
  times = []

  store.gates['s1'] = asyncio.Event()
  first_gate.set()
  await asyncio.sleep(0)
  times.append(session.last_update_time)
  store.gates['s1'].set()
  await _flush(service)

  assert times == [second.timestamp]
  assert session.last_update_time == second.timestamp


async def test_append_event_session_read_before_another_writer_is_rejected():
  """A session object behind another object's buffered writes is stale."""
  database = DatabaseSessionService('sqlite+aiosqlite:///:memory:')
  service = BufferedSessionService(
      database,
      config=BufferedSessionServiceConfig(
          retry_config=RetryConfig(max_attempts=1)
      ),
  )
  await _create(service)
  first = await service.get_session(
      app_name=_APP, user_id=_USER, session_id='s1'
  )
  behind = await service.get_session(
      app_name=_APP, user_id=_USER, session_id='s1'
  )

  await service.append_event(first, _event('e1'))
  await service.append_event(behind, _event('e2'))
  with pytest.raises(SessionPersistenceError) as raised:
    await _flush(service)

  assert isinstance(raised.value.__cause__, StaleSessionError)
  stored = await database.get_session(
      app_name=_APP, user_id=_USER, session_id='s1'
  )
  assert [e.id for e in stored.events] == ['e1']
  await database.close()


async def test_stale_session_object_leaves_newer_session_objects_writes_intact():
  """A stale append from an older session object leaves newer writes working."""
  database = DatabaseSessionService('sqlite+aiosqlite:///:memory:')
  service = BufferedSessionService(
      database,
      config=BufferedSessionServiceConfig(
          retry_config=RetryConfig(max_attempts=1)
      ),
  )
  await _create(service)
  older = await service.get_session(
      app_name=_APP, user_id=_USER, session_id='s1'
  )
  newer = await service.get_session(
      app_name=_APP, user_id=_USER, session_id='s1'
  )

  # Newer handle advances storage; older handle then appends before and after
  # newer handle's subsequent writes.
  await service.append_event(newer, _event('e1'))
  await _flush(service)

  older_copy = older.model_copy(deep=True)
  await service.append_event(older, _event('stale1'))
  await service.append_event(older_copy, _event('stale2'))
  await service.append_event(newer, _event('e2'))
  with pytest.raises(SessionPersistenceError) as flush_err:
    await _flush(service)
  assert isinstance(flush_err.value.__cause__, StaleSessionError)

  # Stale handles stay rejected; newer handle keeps appending cleanly.
  with pytest.raises(SessionPersistenceError) as append_err:
    await service.append_event(older, _event('stale3'))
  assert isinstance(append_err.value.__cause__, StaleSessionError)

  await service.append_event(newer, _event('e3'))
  await _flush(service)

  # Also when the stale write already failed in the background before newer's
  # next append (without an intervening flush).
  stale_again = await database.get_session(
      app_name=_APP, user_id=_USER, session_id='s1'
  )
  await service.append_event(newer, _event('e4'))
  await _flush(service)
  await service.append_event(stale_again, _event('stale4'))
  await asyncio.sleep(0.05)
  await service.append_event(newer, _event('e5'))
  with pytest.raises(SessionPersistenceError) as second_flush_err:
    await _flush(service)
  assert isinstance(second_flush_err.value.__cause__, StaleSessionError)

  stored = await database.get_session(
      app_name=_APP, user_id=_USER, session_id='s1'
  )
  assert [e.id for e in stored.events] == ['e1', 'e2', 'e3', 'e4', 'e5']
  await database.close()


async def test_flush_store_merged_another_writer_updates_the_session():
  """Events and state a store merges in on a conflict reach the session."""
  store = _MergingStore()
  service = BufferedSessionService(store)
  session = await _create(service, shared='mine')
  other = await store.get_session(app_name=_APP, user_id=_USER, session_id='s1')
  theirs = _event('theirs', shared='theirs', extra='x')
  theirs.timestamp = session.last_update_time + 1
  await store.append_event(other, theirs)
  mine = _event('mine')
  mine.timestamp = session.last_update_time + 2

  await service.append_event(session, mine)
  await _flush(service)
  later = _event('later')
  later.timestamp = session.last_update_time + 1
  await service.append_event(session, later)
  await _flush(service)

  assert [e.id for e in session.events] == ['theirs', 'mine', 'later']
  assert session.state['shared'] == 'theirs'
  assert session.state['extra'] == 'x'
  stored = await store.get_session(
      app_name=_APP, user_id=_USER, session_id='s1'
  )
  assert [e.id for e in stored.events] == ['theirs', 'mine', 'later']


async def test_flush_store_merged_while_writes_queued_keeps_their_state():
  """A merge keeps the state that the session's queued writes change."""
  store = _MergingStore()
  service = BufferedSessionService(store)
  session = await _create(service)
  other = await store.get_session(app_name=_APP, user_id=_USER, session_id='s1')
  theirs = _event('theirs', shared='theirs', extra='x')
  theirs.timestamp = session.last_update_time + 1
  await store.append_event(other, theirs)
  first = _event('first')
  first.timestamp = session.last_update_time + 2
  second = _event('second', shared='second')
  second.timestamp = session.last_update_time + 3

  await service.append_event(session, first)
  await service.append_event(session, second)
  await _flush(service)

  assert [e.id for e in session.events] == ['theirs', 'first', 'second']
  assert session.state['shared'] == 'second'
  assert session.state['extra'] == 'x'
  stored = await store.get_session(
      app_name=_APP, user_id=_USER, session_id='s1'
  )
  assert [e.id for e in stored.events] == ['theirs', 'first', 'second']
  assert stored.state['shared'] == 'second'


async def test_append_event_stale_session_with_a_tied_timestamp_is_rejected():
  """A stale session object is told apart even when event times tie."""

  class _RevisionStore(_Store):
    """Rejects a write whose revision marker is not the stored one."""

    revision = 0

    async def get_session(self, **kwargs: Any) -> Session | None:
      session = await super().get_session(**kwargs)
      if session is not None:
        session._storage_update_marker = str(self.revision)
      return session

    async def append_event(self, session: Session, event: Event) -> Event:
      if session._storage_update_marker != str(self.revision):
        raise StaleSessionError('stale')
      await super().append_event(session, event)
      self.revision += 1
      session._storage_update_marker = str(self.revision)
      return event

  service = _service(_RevisionStore(), max_attempts=1)
  await _create(service)
  first = await service.get_session(
      app_name=_APP, user_id=_USER, session_id='s1'
  )
  behind = await service.get_session(
      app_name=_APP, user_id=_USER, session_id='s1'
  )
  tied = _event('e1')
  tied.timestamp = first.last_update_time

  await service.append_event(first, tied)
  await service.append_event(behind, _event('e2'))
  with pytest.raises(SessionPersistenceError) as raised:
    await _flush(service)

  assert isinstance(raised.value.__cause__, StaleSessionError)


async def test_flush_retry_sends_no_event_a_failed_attempt_added():
  """A retry gets a store session without the events of a failed attempt."""
  seen = []

  class _AppliesThenFails(_Store):
    fail = True

    async def append_event(self, session: Session, event: Event) -> Event:
      seen.append((len(session.events), dict(session.state)))
      if self.fail:
        self.fail = False
        await InMemorySessionService.append_event(self, session, event)
        raise RuntimeError('lost after applying')
      return await super().append_event(session, event)

  service = _service(_AppliesThenFails())
  session = await _create(service, k='v0')

  await service.append_event(session, _event('e1', k='v1'))
  await _flush(service)

  assert [event_count for event_count, _ in seen] == [0, 0]


async def test_append_event_temp_value_that_cannot_be_copied_is_accepted():
  """A `temp:` value need not be copyable, since it never reaches the store."""
  store = _Store()
  service = _service(store)
  session = await _create(service)
  lock = threading.Lock()

  await service.append_event(session, _event('e1', **{'temp:lock': lock}))
  await _flush(service)

  assert session.state['temp:lock'] is lock
  assert store.stored_ids() == ['e1']


async def test_append_event_copy_with_the_same_history_continues_its_writes():
  """A copy of a session object with the same events shares its writes."""
  database = DatabaseSessionService('sqlite+aiosqlite:///:memory:')
  service = BufferedSessionService(
      database,
      config=BufferedSessionServiceConfig(
          retry_config=RetryConfig(max_attempts=1)
      ),
  )
  session = await _create(service)
  await service.append_event(session, _event('e1'))
  same_history = session.model_copy(deep=True)

  await service.append_event(same_history, _event('e2'))
  await _flush(service)

  stored = await database.get_session(
      app_name=_APP, user_id=_USER, session_id='s1'
  )
  assert [e.id for e in stored.events] == ['e1', 'e2']
  await database.close()


async def test_append_event_session_holding_a_dropped_event_is_rejected_at_once():
  """A session object whose event will be dropped rejects appends right away."""
  store = _Store()
  service = _service(store, max_attempts=1)
  first = await _create(service, 's1')
  await service.create_session(app_name=_APP, user_id=_USER, session_id='s2')
  s2_a = await service.get_session(
      app_name=_APP, user_id=_USER, session_id='s2'
  )
  s2_b = s2_a.model_copy(deep=True)
  gate = store.hold('s1')
  await service.append_event(first, _event('e1', **{'user:k': 1}))
  store.failures = [RuntimeError('unavailable')]
  await service.append_event(s2_a, _event('e2a'))
  await service.append_event(s2_b, _event('e2b', **{'user:k': 2}))
  await asyncio.sleep(0)
  with pytest.raises(SessionPersistenceError):
    await service.append_event(
        Session(id='s2', app_name=_APP, user_id=_USER), _event('report')
    )

  with pytest.raises(SessionPersistenceError, match='Read the session again'):
    await service.append_event(s2_b, _event('e2c'))

  gate.set()
  await service.flush()


# --- Flushing ---


async def test_flush_session_and_flush_forward_to_the_store():
  """`flush_session` passes the key to the store; `flush` calls global flush."""
  store = _Store()
  service = _service(store)

  await service.flush_session(app_name=_APP, user_id=_USER, session_id='s1')
  await service.flush()

  assert store.flushes == [(_APP, _USER, 's1'), (None, None, None)]


async def test_flush_session_falls_back_to_store_flush_when_not_overridden():
  """A store or wrapper implementing only `flush()` still runs on `flush_session`."""
  calls = []

  class _GlobalFlushWrapper(InMemorySessionService):

    async def flush(self) -> None:
      calls.append('flushed')

  service = BufferedSessionService(_GlobalFlushWrapper())

  await _flush(service)

  assert calls == ['flushed']


@pytest.mark.parametrize(
    'blank',
    ['', '  '],
    ids=['empty_session_id', 'blank_session_id'],
)
async def test_flush_session_blank_session_id_raises(blank: str):
  """`flush_session` with an empty or blank session id raises."""
  service = _service()

  with pytest.raises(ValueError):
    await service.flush_session(app_name=_APP, user_id=_USER, session_id=blank)


async def test_flush_caller_canceled_the_write_continues():
  """Canceling a flush stops the caller's wait, not the write."""
  store = _Store()
  service = _service(store)
  session = await _create(service)
  gate = store.hold()
  await service.append_event(session, _event('e1'))
  waiter = asyncio.create_task(_flush(service))
  await _started(waiter)

  waiter.cancel()
  with pytest.raises(asyncio.CancelledError):
    await waiter
  gate.set()
  await _flush(service)

  assert store.stored_ids() == ['e1']


async def test_flush_after_writes_persisted_leaves_no_task_running():
  """Once the writes are persisted, the service runs no background task."""
  service = _service()
  session = await _create(service)
  before = asyncio.all_tasks()

  await service.append_event(session, _event('e1'))
  await _flush(service)
  await asyncio.sleep(0)

  assert asyncio.all_tasks() == before


async def test_flush_without_key_waits_for_every_session():
  """A global flush waits for the writes of every session."""
  store = _Store()
  service = _service(store)
  s1 = await _create(service, 's1')
  s2 = await _create(service, 's2', user_id='other')
  gates = [store.hold('s1'), store.hold('s2')]
  await service.append_event(s1, _event('e1'))
  await service.append_event(s2, _event('e2'))
  flush = asyncio.create_task(service.flush())
  await _started(flush)

  for gate in gates:
    gate.set()
  await flush

  assert store.stored_ids('s1') == ['e1']
  assert store.stored_ids('s2', user_id='other') == ['e2']


# --- Reads wait for the writes they can see ---


async def test_get_session_writes_buffered_waits_for_them():
  """A read of a session waits for its buffered writes."""
  store = _Store()
  service = _service(store)
  session = await _create(service)
  gate = store.hold()
  await service.append_event(session, _event('e1'))
  read = asyncio.create_task(
      service.get_session(app_name=_APP, user_id=_USER, session_id='s1')
  )
  await _started(read)

  gate.set()

  assert [e.id for e in (await read).events] == ['e1']


async def test_get_session_with_config_passes_it_on_to_the_store():
  """A filtered read waits for the writes and passes the filter on."""
  service = _service()
  session = await _create(service)
  await service.append_event(session, _event('e1'))
  await service.append_event(session, _event('e2'))

  read = await service.get_session(
      app_name=_APP,
      user_id=_USER,
      session_id='s1',
      config=GetSessionConfig(num_recent_events=1),
  )

  assert [e.id for e in read.events] == ['e2']


@pytest.mark.parametrize(
    ('writer_user', 'waits'), [(_USER, True), ('other', False)]
)
async def test_get_session_user_state_changed_in_another_session(
    writer_user: str, waits: bool
):
  """A read waits for a `user:` change by the same user only."""
  store = _Store()
  service = _service(store)
  await _create(service, 's1')
  writer = await _create(service, 's2', user_id=writer_user)
  store.hold('s2')
  await service.append_event(writer, _event('e1', **{'user:k': 'v'}))

  read = asyncio.create_task(
      service.get_session(app_name=_APP, user_id=_USER, session_id='s1')
  )
  await asyncio.sleep(0)

  assert read.done() is not waits
  store.gates['s2'].set()
  await read


async def test_create_session_app_state_changed_by_another_user_waits_for_it():
  """Creating a session waits for another user's `app:` change."""
  store = _Store()
  service = _service(store)
  writer = await _create(service, 's1', user_id='other')
  gate = store.hold('s1')
  await service.append_event(writer, _event('e1', **{'app:k': 'v'}))
  create = asyncio.create_task(_create(service, 's2'))
  await _started(create)

  gate.set()

  assert (await create).state['app:k'] == 'v'


async def test_create_session_passes_extra_arguments_on_to_the_store():
  """Keyword arguments beyond the base ones reach the store."""
  received = {}

  class _OptionsStore(_Store):

    async def create_session(self, **kwargs: Any) -> Session:
      received.update(kwargs)
      kwargs.pop('ttl')
      return await super().create_session(**kwargs)

  service = _service(_OptionsStore())

  await service.create_session(app_name=_APP, user_id=_USER, ttl='60s')

  assert received['ttl'] == '60s'


@pytest.mark.parametrize(
    ('writer_user', 'state_key', 'listed_user', 'waits'),
    [
        ('other', 'app:k', _USER, True),
        (_USER, 'k', _USER, True),
        ('other', 'k', _USER, False),
        ('other', 'k', None, True),
    ],
    ids=[
        'app_state_of_another_user',
        'session_state_of_the_user',
        'session_state_of_another_user',
        'any_session_when_listing_every_user',
    ],
)
async def test_list_sessions_waits_for_the_writes_the_listing_sees(
    writer_user: str, state_key: str, listed_user: str | None, waits: bool
):
  """A listing waits for the buffered writes it can see, and no others."""
  store = _Store()
  service = _service(store)
  writer = await _create(service, 's1', user_id=writer_user)
  gate = store.hold('s1')
  await service.append_event(writer, _event('e1', **{state_key: 'v'}))

  listing = asyncio.create_task(
      service.list_sessions(app_name=_APP, user_id=listed_user)
  )
  await asyncio.sleep(0)

  assert listing.done() is not waits
  gate.set()
  await listing


async def test_get_user_state_user_state_change_buffered_waits_for_it():
  """Reading user state waits for a buffered `user:` change."""
  store = _Store()
  service = _service(store)
  session = await _create(service)
  gate = store.hold()
  await service.append_event(session, _event('e1', **{'user:k': 'v'}))
  read = asyncio.create_task(
      service.get_user_state(app_name=_APP, user_id=_USER)
  )
  await _started(read)

  gate.set()

  assert await read == {'k': 'v'}


async def test_get_session_app_state_changed_by_another_user_waits_for_it():
  """A read of a session waits for another user's `app:` change."""
  store = _Store()
  service = _service(store)
  await _create(service, 's1')
  writer = await _create(service, 's2', user_id='other')
  gate = store.hold('s2')
  await service.append_event(writer, _event('e1', **{'app:k': 'v'}))
  read = asyncio.create_task(
      service.get_session(app_name=_APP, user_id=_USER, session_id='s1')
  )
  await _started(read)

  gate.set()

  assert (await read).state['app:k'] == 'v'


async def test_create_session_user_state_changed_by_the_user_waits_for_it():
  """Creating a session waits for the same user's `user:` change."""
  store = _Store()
  service = _service(store)
  writer = await _create(service, 's1')
  gate = store.hold('s1')
  await service.append_event(writer, _event('e1', **{'user:k': 'v'}))
  create = asyncio.create_task(_create(service, 's2'))
  await _started(create)

  gate.set()

  assert (await create).state['user:k'] == 'v'


@pytest.mark.parametrize(
    ('writer_user', 'state_key'),
    [('other', 'user:k'), ('other', 'app:k'), (_USER, 'k')],
    ids=['user_state_of_another_user', 'app_state', 'session_state'],
)
async def test_get_user_state_does_not_wait_for_other_changes(
    writer_user: str, state_key: str
):
  """Reading user state waits for no change but the user's `user:` state."""
  store = _Store()
  service = _service(store)
  writer = await _create(service, 's1', user_id=writer_user)
  gate = store.hold('s1')
  await service.append_event(writer, _event('e1', **{state_key: 'v'}))

  state = await service.get_user_state(app_name=_APP, user_id=_USER)

  assert state == {}
  gate.set()
  await service.flush()


# --- Retries ---


async def test_flush_default_config_does_not_retry_a_failed_write():
  """Retries are off unless configured, so a failed write is tried once."""
  store = _Store()
  service = BufferedSessionService(store)
  session = await _create(service)
  store.failures.append(RuntimeError('unavailable'))

  await service.append_event(session, _event('e1'))
  with pytest.raises(SessionPersistenceError):
    await _flush(service)

  assert store.attempts == ['e1']


async def test_flush_write_fails_then_succeeds_persists_the_event_once():
  """A failed attempt is retried, and the event is stored once."""
  store = _Store()
  service = _service(store)
  session = await _create(service)
  store.failures = [RuntimeError('unavailable'), RuntimeError('unavailable')]

  await service.append_event(session, _event('e1'))
  await _flush(service)

  assert store.stored_ids() == ['e1']
  assert store.attempts == ['e1', 'e1', 'e1']


async def test_flush_write_fails_after_the_store_commits_it_stores_it_once():
  """A retry after a committed attempt does not store the event twice."""

  class _LosesResponses(_Store):
    lost = 1

    async def append_event(self, session: Session, event: Event) -> Event:
      result = await super().append_event(session, event)
      if self.lost:
        self.lost -= 1
        raise RuntimeError('response lost')
      return result

  store = _LosesResponses()
  service = _service(store)
  session = await _create(service)

  await service.append_event(session, _event('e1'))
  await service.append_event(session, _event('e2'))
  await _flush(service)

  assert store.stored_ids() == ['e1', 'e2']


@pytest.mark.parametrize(
    ('retry', 'expected_delays'),
    [
        (dict(jitter=0.0), [1.0, 2.0, 4.0, 8.0]),
        (dict(jitter=0.0, max_delay=3.0), [1.0, 2.0, 3.0, 3.0]),
        (dict(jitter=0.0, initial_delay=0.5, backoff_factor=3.0), [0.5, 1.5]),
    ],
    ids=['defaults', 'capped', 'custom_growth'],
)
async def test_flush_retries_wait_the_configured_delays(
    monkeypatch: pytest.MonkeyPatch,
    retry: dict[str, Any],
    expected_delays: list[float],
):
  """Retries wait the exponentially growing, capped delays configured."""
  delays = []
  real_sleep = asyncio.sleep

  async def record_sleep(delay: float, *args: Any, **kwargs: Any) -> Any:
    if delay:
      delays.append(delay)
    return await real_sleep(0, *args, **kwargs)

  monkeypatch.setattr(asyncio, 'sleep', record_sleep)
  retry.setdefault('initial_delay', 1.0)
  store = _Store()
  service = _service(store, **retry)
  session = await _create(service)
  store.failures = [RuntimeError('unavailable')] * len(expected_delays)

  await service.append_event(session, _event('e1'))
  await _flush(service)

  assert delays == pytest.approx(expected_delays)


class _FixedDraw(random.Random):
  """Draws the low end, the middle or the high end of every range."""

  def __init__(self, where: str) -> None:
    super().__init__()
    self._where = where

  def uniform(self, a: float, b: float) -> float:
    return {'low': a, 'middle': (a + b) / 2, 'high': b}[self._where]


@pytest.mark.parametrize(
    ('draw', 'retry', 'expected_delays'),
    [
        ('high', dict(jitter=0.5), [1.5, 3.0]),
        ('low', dict(jitter=0.5), [0.5, 1.0]),
        ('high', dict(), [2.0, 4.0]),
        ('middle', dict(initial_delay=10.0, max_delay=4.0), [2.0, 2.0]),
        ('high', dict(initial_delay=10.0, max_delay=4.0), [4.0, 4.0]),
    ],
    ids=[
        'spread_up',
        'spread_down',
        'default_jitter_doubles_at_most',
        'capped_before_jitter',
        'never_above_max_delay',
    ],
)
async def test_flush_retries_jitter_each_delay(
    monkeypatch: pytest.MonkeyPatch,
    draw: str,
    retry: dict[str, Any],
    expected_delays: list[float],
):
  """Jitter spreads each delay around its capped base, within `max_delay`."""
  delays = []
  real_sleep = asyncio.sleep

  async def record_sleep(delay: float, *args: Any, **kwargs: Any) -> Any:
    delays.append(delay)
    return await real_sleep(0, *args, **kwargs)

  monkeypatch.setattr(asyncio, 'sleep', record_sleep)
  adk_platform.set_random_provider(lambda: _FixedDraw(draw))
  retry.setdefault('initial_delay', 1.0)
  store = _Store()
  service = _service(store, **retry)
  session = await _create(service)
  store.failures = [RuntimeError('unavailable')] * len(expected_delays)

  try:
    await service.append_event(session, _event('e1'))
    await _flush(service)
  finally:
    adk_platform.reset_random_provider()

  assert delays == pytest.approx(expected_delays)


async def test_flush_overflowing_backoff_caps_the_delay(
    monkeypatch: pytest.MonkeyPatch,
):
  """A backoff too large for a float waits the maximum delay."""
  delays = []
  real_sleep = asyncio.sleep

  async def record_sleep(delay: float, *args: Any, **kwargs: Any) -> Any:
    delays.append(delay)
    return await real_sleep(0, *args, **kwargs)

  monkeypatch.setattr(asyncio, 'sleep', record_sleep)
  store = _Store()
  service = _service(
      store,
      max_attempts=400,
      initial_delay=1.0,
      backoff_factor=10.0,
      max_delay=5.0,
      jitter=0.0,
  )
  session = await _create(service)
  store.failures = [RuntimeError('unavailable')] * 399

  await service.append_event(session, _event('e1'))
  await _flush(service)

  assert delays[-1] == 5.0
  assert store.stored_ids() == ['e1']


@pytest.mark.parametrize(
    'error', [StaleSessionError('stale'), SessionNotFoundError()]
)
async def test_flush_stale_or_missing_session_is_not_retried(
    error: Exception,
):
  """A stale or missing session fails at once, since a retry fails the same."""
  store = _Store()
  service = _service(store)
  session = await _create(service)
  store.failures = [error]

  await service.append_event(session, _event('e1'))
  with pytest.raises(SessionPersistenceError) as raised:
    await _flush(service)

  assert raised.value.__cause__ is error
  assert store.attempts == ['e1']


async def test_flush_error_outside_the_configured_exceptions_is_not_retried():
  """With `exceptions` set, other errors fail without a retry."""
  store = _Store()
  service = _service(store, exceptions=['ConnectionError'])
  session = await _create(service)
  store.failures = [ConnectionResetError('reset'), RuntimeError('bug')]

  await service.append_event(session, _event('e1'))
  with pytest.raises(SessionPersistenceError) as raised:
    await _flush(service)

  assert isinstance(raised.value.__cause__, RuntimeError)
  assert store.attempts == ['e1', 'e1']


async def test_flush_store_raises_cancelled_error_retries_the_write():
  """A CancelledError the store raises is retried like any other failure."""
  store = _Store()
  service = _service(store)
  session = await _create(service)
  store.failures = [asyncio.CancelledError()]

  await service.append_event(session, _event('e1'))
  await _flush(service)

  assert store.stored_ids() == ['e1']


# --- Failures ---


async def test_flush_retries_run_out_raises_once_with_the_last_error():
  """A failed write raises once, chained to the last attempt's error."""
  store = _Store()
  service = _service(store, max_attempts=2)
  session = await _create(service)
  last = RuntimeError('second attempt')
  store.failures = [RuntimeError('first attempt'), last]
  await service.append_event(session, _event('e1'))

  with pytest.raises(SessionPersistenceError) as raised:
    await _flush(service)
  await _flush(service)

  assert raised.value.__cause__ is last


async def test_append_event_session_read_again_after_a_failure_persists():
  """A session read again after a reported failure appends normally."""
  store = _Store()
  service = _service(store, max_attempts=1)
  session = await _create(service)
  store.failures = [RuntimeError('unavailable')]
  await service.append_event(session, _event('e1'))
  with pytest.raises(SessionPersistenceError):
    await _flush(service)
  reloaded = await service.get_session(
      app_name=_APP, user_id=_USER, session_id='s1'
  )

  await service.append_event(reloaded, _event('e2'))
  await _flush(service)

  assert store.stored_ids() == ['e2']


async def test_flush_failed_write_drops_the_later_writes_of_the_session():
  """The writes queued behind a failed write are dropped with it."""
  store = _Store()
  service = _service(store, max_attempts=1)
  session = await _create(service)
  store.failures = [RuntimeError('unavailable')]

  await service.append_event(session, _event('e1'))
  await service.append_event(session, _event('e2'))
  with pytest.raises(SessionPersistenceError):
    await _flush(service)

  assert store.attempts == ['e1']
  assert store.stored_ids() == []


async def test_flush_reported_failure_is_not_reported_by_a_global_flush():
  """A failure a keyed flush reported is not raised again by a global one."""
  store = _Store()
  service = _service(store, max_attempts=1)
  session = await _create(service)
  store.failures = [RuntimeError('unavailable')]
  await service.append_event(session, _event('e1'))

  with pytest.raises(SessionPersistenceError):
    await _flush(service)
  await service.flush()


async def test_append_event_after_retries_ran_out_raises_without_applying():
  """The next append reports the failure and leaves the session unchanged."""
  store = _Store()
  service = _service(store, max_attempts=1)
  session = await _create(service)
  store.failures = [RuntimeError('unavailable')]
  await service.append_event(session, _event('e1'))
  reloaded = await service.get_session(
      app_name=_APP, user_id=_USER, session_id='s1'
  )

  with pytest.raises(SessionPersistenceError):
    await service.append_event(reloaded, _event('e2'))

  assert reloaded.events == []


async def test_append_event_after_the_failure_was_reported_persists():
  """Once an append reported the failure, the next append persists."""
  store = _Store()
  service = _service(store, max_attempts=1)
  session = await _create(service)
  store.failures = [RuntimeError('unavailable')]
  await service.append_event(session, _event('e1'))
  reloaded = await service.get_session(
      app_name=_APP, user_id=_USER, session_id='s1'
  )
  with pytest.raises(SessionPersistenceError):
    await service.append_event(reloaded, _event('e2'))

  await service.append_event(reloaded, _event('e3'))
  await _flush(service)

  assert store.stored_ids() == ['e3']


async def test_append_event_to_a_session_holding_a_failed_event_raises():
  """A live session that holds an event whose write failed rejects appends."""
  store = _Store()
  service = _service(store, max_attempts=1)
  session = await _create(service)
  store.failures = [RuntimeError('unavailable')]
  await service.append_event(session, _event('e1'))
  with pytest.raises(SessionPersistenceError):
    await _flush(service)

  with pytest.raises(
      SessionPersistenceError, match='Read the session again'
  ) as raised:
    await service.append_event(session, _event('e2'))

  assert isinstance(raised.value.__cause__, RuntimeError)
  assert [e.id for e in session.events] == ['e1']


async def test_get_session_after_retries_ran_out_returns_what_the_store_holds():
  """Reads after a failure return the store's contents without raising."""
  store = _Store()
  service = _service(store, max_attempts=1)
  session = await _create(service)
  store.failures = [RuntimeError('unavailable')]
  await service.append_event(session, _event('e1'))

  read = await service.get_session(
      app_name=_APP, user_id=_USER, session_id='s1'
  )

  assert read.events == []
  with pytest.raises(SessionPersistenceError):
    await _flush(service)


async def test_flush_store_raises_a_base_exception_reports_it_and_stops():
  """An error that is not an Exception stops the service and is reported."""

  class _Fatal(BaseException):
    pass

  store = _Store()
  service = _service(store)
  session = await _create(service)
  store.failures = [_Fatal()]
  await service.append_event(session, _event('e1'))

  with pytest.raises(SessionPersistenceError) as raised:
    await _flush(service)
  with pytest.raises(RuntimeError, match='stopped'):
    await service.append_event(session, _event('e2'))

  assert isinstance(raised.value.__cause__, _Fatal)
  assert store.attempts == ['e1']


async def test_flush_unreported_failures_past_the_cap_forget_the_oldest(
    caplog: pytest.LogCaptureFixture,
):
  """Past 1024 sessions with a failure to report, the oldest is forgotten."""
  store = _Store()
  service = _service(store, max_attempts=1)
  sessions = [await _create(service, f's{i}') for i in range(1025)]
  store.failures = [RuntimeError('unavailable')] * 1025
  for session in sessions:
    await service.append_event(session, _event(f'e-{session.id}'))
  await service.list_sessions(app_name=_APP, user_id=_USER)

  oldest = await service.get_session(
      app_name=_APP, user_id=_USER, session_id='s0'
  )
  await service.append_event(oldest, _event('again'))
  newest = await service.get_session(
      app_name=_APP, user_id=_USER, session_id='s1024'
  )

  with pytest.raises(SessionPersistenceError):
    await service.append_event(newest, _event('again'))
  assert 'Forgot the oldest unreported' in caplog.text


async def test_flush_waiting_on_a_failed_write_forgotten_by_the_cap_still_raises():
  """A flush waiting on a write still raises its failure after the cap."""
  store = _Store()
  service = _service(store, max_attempts=1)
  session = await _create(service, 'waited')
  others = [await _create(service, f's{i}') for i in range(1024)]
  gate = store.hold('waited')
  store.failures = [RuntimeError('unavailable')] * 1025
  await service.append_event(session, _event('e1'))
  flush = asyncio.create_task(_flush(service, 'waited'))
  await _started(flush)

  gate.set()
  for other in others:
    await service.append_event(other, _event(f'e-{other.id}'))

  with pytest.raises(SessionPersistenceError):
    await flush


async def test_flush_error_that_stopped_the_service_is_never_forgotten():
  """The error that stopped the service stays reportable past the cap."""

  class _Fatal(BaseException):
    pass

  class _FatalStore(_Store):

    async def append_event(self, session: Session, event: Event) -> Event:
      if session.id == 'fatal':
        raise _Fatal()
      return await super().append_event(session, event)

  store = _FatalStore()
  service = _service(store)
  fatal = await _create(service, 'fatal')
  held = [await _create(service, f's{i}') for i in range(1030)]
  for session in held:
    store.hold(session.id)
    await service.append_event(session, _event(f'e-{session.id}'))
  await service.append_event(fatal, _event('e-fatal'))

  with pytest.raises(SessionPersistenceError):
    await service.flush()
  with pytest.raises(SessionPersistenceError) as raised:
    await _flush(service, 'fatal')

  assert isinstance(raised.value.__cause__, _Fatal)


async def test_flush_writer_stopped_before_it_started_reports_its_writes():
  """Writes whose writer is stopped before it ever ran are still reported."""

  class _Fatal(BaseException):
    pass

  class _FatalOnS0(_Store):

    async def append_event(self, session: Session, event: Event) -> Event:
      if session.id == 's0':
        await self.gates['s0'].wait()
        raise _Fatal()
      return await super().append_event(session, event)

  store = _FatalOnS0()
  service = _service(store)
  s0 = await _create(service, 's0')
  s1 = await _create(service, 's1')
  gate = store.hold('s0')
  await service.append_event(s0, _event('e0'))
  await asyncio.sleep(0)

  # The s0 writer wakes first and stops every writer, before the s1 writer,
  # queued behind it, runs at all.
  gate.set()
  await service.append_event(s1, _event('e1'))
  with pytest.raises(SessionPersistenceError):
    await asyncio.wait_for(_flush(service, 's1'), timeout=5)


# --- Shared state ordering ---


@pytest.mark.parametrize(
    ('state_key', 'second_user'),
    [('user:k', _USER), ('app:k', 'other')],
    ids=['user_state_of_one_user', 'app_state_of_two_users'],
)
async def test_flush_shared_state_changed_in_two_sessions_persists_in_append_order(
    state_key: str, second_user: str
):
  """Shared state changes of two sessions reach the store in append order."""
  store = _Store()
  service = _service(store)
  first = await _create(service, 's1')
  second = await _create(service, 's2', user_id=second_user)
  gate = store.hold('s1')
  await service.append_event(first, _event('first', **{state_key: 1}))
  await service.append_event(second, _event('second', **{state_key: 2}))
  await store.started['s1'].wait()
  await asyncio.sleep(0)

  assert 'second' not in store.attempts
  gate.set()
  await service.flush()
  assert store.attempts == ['first', 'second']
  read = await service.get_session(
      app_name=_APP, user_id=_USER, session_id='s1'
  )
  assert read.state[state_key] == 2


@pytest.mark.parametrize(
    ('first_key', 'second_key', 'second_user'),
    [
        ('user:k', 'user:k', 'other'),
        ('app:k', 'user:k', _USER),
        ('user:k', 'app:k', _USER),
    ],
    ids=[
        'user_state_of_another_user',
        'user_state_after_app_state',
        'app_state_after_user_state',
    ],
)
async def test_flush_different_shared_scopes_do_not_wait_for_each_other(
    first_key: str, second_key: str, second_user: str
):
  """Changes to different shared scopes do not wait for each other."""
  store = _Store()
  service = _service(store)
  first = await _create(service, 's1')
  second = await _create(service, 's2', user_id=second_user)
  gate = store.hold('s1')
  await service.append_event(first, _event('first', **{first_key: 1}))
  await service.append_event(second, _event('second', **{second_key: 2}))

  await _flush(service, 's2', user_id=second_user)

  assert store.stored_ids('s2', user_id=second_user) == ['second']
  gate.set()
  await service.flush()


async def test_flush_earlier_shared_state_change_fails_persists_the_later_one():
  """A failed shared state write of one session does not drop another's."""
  store = _Store()
  service = _service(store, max_attempts=1)
  first = await _create(service, 's1')
  second = await _create(service, 's2')
  store.failures = [RuntimeError('unavailable')]

  await service.append_event(first, _event('first', **{'user:k': 1}))
  await service.append_event(second, _event('second', **{'user:k': 2}))
  await _flush(service, 's2')

  assert store.stored_ids('s2') == ['second']
  with pytest.raises(SessionPersistenceError):
    await _flush(service, 's1')


async def test_flush_dropped_shared_state_write_keeps_later_changes_in_order():
  """A dropped shared state write does not let a later change overtake."""
  store = _Store()
  service = _service(store, max_attempts=1)
  s1 = await _create(service, 's1')
  s2 = await _create(service, 's2')
  s3 = await _create(service, 's3')
  gate = store.hold('s1')
  await service.append_event(s1, _event('e1', **{'user:k': 1}))
  # s2's first write fails, which drops its shared state change behind it.
  store.failures = [RuntimeError('unavailable')]
  await service.append_event(s2, _event('e2a'))
  await service.append_event(s2, _event('e2b', **{'user:k': 2}))
  await service.append_event(s3, _event('e3', **{'user:k': 3}))
  await asyncio.sleep(0)

  attempts_while_held = list(store.attempts)
  gate.set()
  with pytest.raises(SessionPersistenceError):
    await service.flush()

  assert 'e3' not in attempts_while_held
  assert await store.get_user_state(app_name=_APP, user_id=_USER) == {'k': 3}


async def test_get_user_state_waits_behind_a_dropped_shared_state_write():
  """A read still waits for the write a dropped shared state write followed."""
  store = _Store()
  service = _service(store, max_attempts=1)
  s1 = await _create(service, 's1')
  s2 = await _create(service, 's2')
  gate = store.hold('s1')
  await service.append_event(s1, _event('e1', **{'user:k': 1}))
  store.failures = [RuntimeError('unavailable')]
  await service.append_event(s2, _event('e2a'))
  await service.append_event(s2, _event('e2b', **{'user:k': 2}))
  read = asyncio.create_task(
      service.get_user_state(app_name=_APP, user_id=_USER)
  )
  await _started(read)

  gate.set()

  assert await read == {'k': 1}
  with pytest.raises(SessionPersistenceError):
    await service.flush()


# --- Deleting ---


async def test_delete_session_writes_buffered_waits_for_them():
  """Deleting a session waits for its buffered writes first."""
  store = _Store()
  service = _service(store)
  session = await _create(service)
  gate = store.hold()
  await service.append_event(session, _event('e1'))
  delete = asyncio.create_task(
      service.delete_session(app_name=_APP, user_id=_USER, session_id='s1')
  )
  await _started(delete)

  gate.set()
  await delete

  assert store.attempts == ['e1']
  assert (
      await store.get_session(app_name=_APP, user_id=_USER, session_id='s1')
      is None
  )


async def test_delete_session_write_fails_raises_and_keeps_the_session():
  """A failed write makes the delete raise once and keep the session."""
  store = _Store()
  service = _service(store, max_attempts=1)
  session = await _create(service)
  store.failures = [RuntimeError('unavailable')]
  await service.append_event(session, _event('e1'))

  with pytest.raises(SessionPersistenceError):
    await service.delete_session(app_name=_APP, user_id=_USER, session_id='s1')
  kept = await store.get_session(app_name=_APP, user_id=_USER, session_id='s1')
  await service.delete_session(app_name=_APP, user_id=_USER, session_id='s1')

  assert kept is not None
  assert (
      await store.get_session(app_name=_APP, user_id=_USER, session_id='s1')
      is None
  )


async def test_delete_session_user_state_change_buffered_persists_it_first():
  """A buffered `user:` change of a deleted session reaches the store first."""
  store = _Store()
  service = _service(store)
  session = await _create(service)

  await service.append_event(session, _event('e1', **{'user:k': 'v'}))
  await service.delete_session(app_name=_APP, user_id=_USER, session_id='s1')

  assert await store.get_user_state(app_name=_APP, user_id=_USER) == {'k': 'v'}


async def test_delete_session_while_a_flush_waits_on_a_failing_write_both_raise():
  """A flush and a delete waiting on one failing write both raise."""
  store = _Store()
  service = _service(store, max_attempts=1)
  session = await _create(service)
  gate = store.hold()
  store.failures = [RuntimeError('unavailable')]
  await service.append_event(session, _event('e1'))
  flush = asyncio.create_task(_flush(service))
  delete = asyncio.create_task(
      service.delete_session(app_name=_APP, user_id=_USER, session_id='s1')
  )
  await _started(flush)

  gate.set()
  results = await asyncio.gather(flush, delete, return_exceptions=True)

  assert all(isinstance(r, SessionPersistenceError) for r in results)
  assert (
      await store.get_session(app_name=_APP, user_id=_USER, session_id='s1')
      is not None
  )


# --- Flushing every session ---


async def test_flush_without_key_reports_a_failure_once_and_leaves_it_for_the_session():
  """A global flush raises a failure once; the session still reports it."""
  store = _Store()
  service = _service(store, max_attempts=1)
  session = await _create(service)
  store.failures = [RuntimeError('unavailable')]
  await service.append_event(session, _event('e1'))

  with pytest.raises(SessionPersistenceError):
    await service.flush()
  await service.flush()

  with pytest.raises(SessionPersistenceError):
    await _flush(service)


async def test_flush_without_key_reports_every_failed_session_in_one_error():
  """A global flush raises one error that names every failed session."""
  store = _Store()
  service = _service(store, max_attempts=1)
  s1 = await _create(service, 's1')
  s2 = await _create(service, 's2')
  first, second = RuntimeError('first'), ValueError('second')
  store.failures = [first, second]
  await service.append_event(s1, _event('e1'))
  await service.append_event(s2, _event('e2'))

  with pytest.raises(SessionPersistenceError, match='2 session') as raised:
    await service.flush()

  assert raised.value.__cause__ is first
  assert raised.value.__notes__ == [
      'Another session failed with ValueError: second'
  ]


# --- Closing ---


async def test_aclose_writes_buffered_persists_them_and_rejects_later_appends():
  """Closing persists the buffered writes and rejects later appends."""
  store = _Store()
  service = _service(store)
  session = await _create(service)
  await service.append_event(session, _event('e1'))

  await service.aclose()

  assert store.stored_ids() == ['e1']
  assert store.flushes == [(None, None, None)]
  with pytest.raises(RuntimeError, match='closed'):
    await service.append_event(session, _event('e2'))


async def test_aclose_write_fails_raises():
  """Closing raises when a buffered write failed."""
  store = _Store()
  service = _service(store, max_attempts=1)
  session = await _create(service)
  store.failures = [RuntimeError('unavailable')]
  await service.append_event(session, _event('e1'))

  with pytest.raises(SessionPersistenceError):
    await service.close()


async def test_aclose_called_twice_flushes_the_store_once():
  """A second close does nothing."""
  store = _Store()
  service = _service(store)

  async with service:
    pass
  await service.aclose()

  assert store.flushes == [(None, None, None)]


async def test_aclose_leaves_the_wrapped_store_open():
  """Closing the service does not close the store it wraps."""
  closed = []

  class _ClosableStore(_Store):

    async def close(self) -> None:
      closed.append(True)

  store = _ClosableStore()
  service = _service(store)

  await service.aclose()

  assert closed == []


async def test_aclose_canceled_stops_the_writers_before_it_ends():
  """A canceled close waits for every background writer to stop."""
  unwound = []

  class _SlowToCancel(_Store):

    async def append_event(self, session: Session, event: Event) -> Event:
      try:
        return await super().append_event(session, event)
      except asyncio.CancelledError:
        for _ in range(3):
          await asyncio.sleep(0)
        unwound.append(event.id)
        raise

  store = _SlowToCancel()
  service = _service(store)
  session = await _create(service)
  store.hold()
  await service.append_event(session, _event('e1'))
  close = asyncio.create_task(service.aclose())
  await _started(close)

  close.cancel()
  with pytest.raises(asyncio.CancelledError):
    await close

  assert unwound == ['e1']


async def test_aclose_canceled_reads_wait_for_writes_still_finishing():
  """A canceled close does not release reads before earlier writes finish."""
  read_done_before_commit = []

  class _CommitsWhileCanceled(_Store):

    async def append_event(self, session: Session, event: Event) -> Event:
      if session.id != 's1':
        return await super().append_event(session, event)
      self.started['s1'].set()
      try:
        await asyncio.Event().wait()
      except asyncio.CancelledError:
        for _ in range(3):
          await asyncio.sleep(0)
        read_done_before_commit.append(reader.done())
        await InMemorySessionService.append_event(self, session, event)
        raise
      return event

  store = _CommitsWhileCanceled()
  service = _service(store)
  s1 = await _create(service, 's1')
  s2 = await _create(service, 's2')
  await service.append_event(s1, _event('e1', **{'user:k': 1}))
  await service.append_event(s2, _event('e2', **{'user:k': 2}))
  await store.started['s1'].wait()
  reader = asyncio.create_task(
      service.get_user_state(app_name=_APP, user_id=_USER)
  )
  close = asyncio.create_task(service.aclose())
  await _started(close)

  close.cancel()
  with pytest.raises(asyncio.CancelledError):
    await close

  assert read_done_before_commit == [False]
  await reader


async def test_aclose_second_call_waits_for_the_first():
  """A close called while another is flushing returns after it, not before."""
  store = _Store()
  service = _service(store)
  session = await _create(service)
  gate = store.hold()
  await service.append_event(session, _event('e1'))
  first = asyncio.create_task(service.aclose())
  await _started(first)

  second = asyncio.create_task(service.aclose())
  await _started(second)
  gate.set()
  await asyncio.gather(first, second)

  assert store.stored_ids() == ['e1']


async def test_aclose_canceled_again_lets_the_store_clean_up_once():
  """A close canceled a second time still waits for the writers to stop, and
  does not cancel the store again while it cleans up."""
  cancellations = []
  cleanup_gate = asyncio.Event()

  class _SlowCleanup(_Store):

    async def append_event(self, session: Session, event: Event) -> Event:
      self.started[session.id].set()
      try:
        await asyncio.Event().wait()
      except asyncio.CancelledError:
        cancellations.append(event.id)
        try:
          await cleanup_gate.wait()
        except asyncio.CancelledError:
          cancellations.append('cleanup')
        raise
      return event

  store = _SlowCleanup()
  service = _service(store)
  session = await _create(service)
  await service.append_event(session, _event('e1'))
  await store.started['s1'].wait()
  close = asyncio.create_task(service.aclose())
  await _started(close)
  close.cancel()
  while not cancellations:
    await asyncio.sleep(0)

  close.cancel()
  for _ in range(5):
    await asyncio.sleep(0)
  assert not close.done()
  cleanup_gate.set()
  with pytest.raises(asyncio.CancelledError):
    await close

  assert cancellations == ['e1']
  with pytest.raises(SessionPersistenceError):
    await _flush(service)


async def test_flush_store_that_deletes_the_marker_keeps_the_service_running():
  """A store may delete the storage marker from the session it writes."""

  class _DeletesMarker(_Store):

    async def append_event(self, session: Session, event: Event) -> Event:
      await super().append_event(session, event)
      delattr(session, '_storage_update_marker')
      return event

  store = _DeletesMarker()
  service = _service(store)
  first = await _create(service, 's1')
  second = await _create(service, 's2')

  await service.append_event(first, _event('e1'))
  await _flush(service, 's1')
  await service.append_event(second, _event('e2'))
  await _flush(service, 's2')

  assert store.stored_ids('s1') == ['e1']
  assert store.stored_ids('s2') == ['e2']


# --- Event loops ---


def test_idle_service_moves_to_the_running_loop():
  """A service with no writes in flight works on a new event loop."""
  store = _Store()
  service = _service(store)

  async def append(event_id: str) -> None:
    session = await service.get_session(
        app_name=_APP, user_id=_USER, session_id='s1'
    ) or await _create(service)
    await service.append_event(session, _event(event_id))
    await _flush(service)

  asyncio.run(append('e1'))
  asyncio.run(append('e2'))

  assert store.stored_ids() == ['e1', 'e2']


def test_writes_cut_off_by_the_loop_ending_are_reported_on_the_next_loop():
  """Writes canceled when their loop ends are reported on the next loop."""
  store = _Store()
  service = _service(store)

  async def append_and_leave() -> None:
    session = await _create(service)
    store.hold()
    await service.append_event(session, _event('e1'))

  async def flush_then_append() -> None:
    with pytest.raises(SessionPersistenceError):
      await _flush(service)
    session = await service.get_session(
        app_name=_APP, user_id=_USER, session_id='s1'
    )
    await service.append_event(session, _event('e2'))
    await _flush(service)

  asyncio.run(append_and_leave())
  store.gates.clear()
  asyncio.run(flush_then_append())

  assert store.stored_ids() == ['e2']


def test_writes_left_on_a_closed_loop_are_reported_on_the_next_loop():
  """Writes stranded on a loop closed without canceling them are reported."""
  store = _Store()
  service = _service(store)
  loop = asyncio.new_event_loop()

  async def append_and_leave() -> None:
    session = await _create(service)
    store.hold()
    await service.append_event(session, _event('e1'))

  loop.run_until_complete(append_and_leave())
  loop.close()
  store.gates.clear()

  async def flush() -> None:
    with pytest.raises(SessionPersistenceError) as raised:
      await _flush(service)
    assert 'event loop closed' in str(raised.value.__cause__)

  asyncio.run(flush())


def test_aclose_cut_off_by_its_loop_closing_is_redone_on_the_next_loop():
  """A close that its loop cut off is redone, not waited for, on the next."""
  store = _Store()
  service = _service(store)
  loop = asyncio.new_event_loop()

  async def append_and_start_closing() -> None:
    session = await _create(service)
    store.hold()
    await service.append_event(session, _event('e1'))
    closing = asyncio.create_task(service.aclose())
    await asyncio.sleep(0)
    assert not closing.done()

  loop.run_until_complete(append_and_start_closing())
  loop.close()
  store.gates.clear()

  async def close() -> None:
    with pytest.raises(SessionPersistenceError):
      await asyncio.wait_for(service.aclose(), timeout=5)

  asyncio.run(close())


def test_aclose_cut_off_without_writes_left_is_redone_on_the_next_loop():
  """A close cut off while flushing the store, its writes done, is redone."""
  flushing = []

  class _SlowFlush(_Store):

    async def flush(self, **kwargs: Any) -> None:
      flushing.append(True)
      if len(flushing) == 1:
        await asyncio.Event().wait()

  service = _service(_SlowFlush())
  loop = asyncio.new_event_loop()

  async def start_closing() -> None:
    closing = asyncio.create_task(service.aclose())
    while not flushing:
      await asyncio.sleep(0)
    assert not closing.done()

  loop.run_until_complete(start_closing())
  loop.close()

  asyncio.run(asyncio.wait_for(service.aclose(), timeout=5))

  assert len(flushing) == 2


def test_closed_loop_collected_later_leaves_the_service_working():
  """Collecting a closed loop's abandoned writer does not stop the service."""
  store = _Store()
  service = _service(store)
  loop = asyncio.new_event_loop()

  async def append_and_leave() -> None:
    session = await _create(service)
    store.hold()
    await service.append_event(session, _event('e1'))
    await asyncio.sleep(0)

  loop.run_until_complete(append_and_leave())
  loop.close()
  store.gates.clear()

  async def report_then_append() -> None:
    with pytest.raises(SessionPersistenceError):
      await _flush(service)
    del loop_ref[:]
    gc.collect()
    session = await service.get_session(
        app_name=_APP, user_id=_USER, session_id='s1'
    )
    await service.append_event(session, _event('e2'))
    await _flush(service)

  loop_ref = [loop]
  del loop
  asyncio.run(report_then_append())

  assert store.stored_ids() == ['e2']


def test_closed_loop_collected_during_store_cleanup_leaves_the_service_working():
  """A store cleaning up after its loop closed cannot stop the service."""

  class _CleansUpOnAbort(_Store):

    async def append_event(self, session: Session, event: Event) -> Event:
      if self.gates.get(session.id) is not None:
        loop = asyncio.get_running_loop()
        try:
          await loop.create_future()
        except BaseException:
          # Like a database rollback, which fails on the closed loop.
          await loop.run_in_executor(None, lambda: None)
          raise
      return await super().append_event(session, event)

  store = _CleansUpOnAbort()
  service = _service(store, max_attempts=1)
  loop = asyncio.new_event_loop()

  async def append_and_leave() -> None:
    session = await _create(service)
    store.hold()
    await service.append_event(session, _event('e1'))
    reader = asyncio.get_running_loop().create_task(
        service.get_session(app_name=_APP, user_id=_USER, session_id='s1')
    )
    await asyncio.sleep(0)
    assert not reader.done()

  loop.run_until_complete(append_and_leave())
  loop.close()
  store.gates.clear()
  loop_ref = [loop]
  del loop

  async def collect_then_append() -> None:
    with pytest.raises(SessionPersistenceError):
      await _flush(service)
    del loop_ref[:]
    gc.collect()
    session = await service.get_session(
        app_name=_APP, user_id=_USER, session_id='s1'
    )
    await service.append_event(session, _event('e2'))
    await _flush(service)

  asyncio.run(collect_then_append())

  assert store.stored_ids() == ['e2']


def test_writes_of_many_sessions_left_on_a_closed_loop_are_reported():
  """More stranded sessions than the failure cap still move to a new loop."""
  store = _Store()
  service = _service(store)
  loop = asyncio.new_event_loop()

  async def append_and_leave() -> None:
    for i in range(1030):
      session = await _create(service, f's{i}')
      store.hold(session.id)
      await service.append_event(session, _event(f'e{i}'))

  loop.run_until_complete(append_and_leave())
  loop.close()
  store.gates.clear()

  async def flush_the_newest() -> None:
    with pytest.raises(SessionPersistenceError):
      await _flush(service, 's1029')

  asyncio.run(flush_the_newest())


def test_session_holding_a_write_left_on_a_closed_loop_is_rejected():
  """A session object whose write a closed loop abandoned rejects appends."""
  store = _Store()
  service = _service(store)
  loop = asyncio.new_event_loop()
  sessions = []

  async def append_and_leave() -> None:
    session = await _create(service)
    sessions.append(session)
    store.hold()
    await service.append_event(session, _event('e1'))

  loop.run_until_complete(append_and_leave())
  loop.close()
  store.gates.clear()

  async def append_to_the_old_session() -> None:
    with pytest.raises(SessionPersistenceError):
      await _flush(service)
    with pytest.raises(SessionPersistenceError, match='Read the session again'):
      await service.append_event(sessions[0], _event('e2'))

  asyncio.run(append_to_the_old_session())


async def test_failed_write_does_not_hold_the_service_on_its_loop():
  """After a write fails, an idle service works on another running loop."""
  store = _Store()
  service = _service(store, max_attempts=1)
  session = await _create(service)
  store.failures = [RuntimeError('unavailable')]
  await service.append_event(session, _event('e1'))
  await service.get_session(app_name=_APP, user_id=_USER, session_id='s1')

  async def flush_on_another_loop() -> None:
    with pytest.raises(SessionPersistenceError):
      await _flush(service)

  await asyncio.to_thread(asyncio.run, flush_on_another_loop())


async def test_call_from_another_loop_while_writes_are_in_flight_raises():
  """A call from another loop raises while writes are in flight."""
  store = _Store()
  service = _service(store)
  session = await _create(service)
  gate = store.hold()
  await service.append_event(session, _event('e1'))

  with pytest.raises(RuntimeError, match='another event loop'):
    await asyncio.to_thread(
        asyncio.run,
        service.get_session(app_name=_APP, user_id=_USER, session_id='s1'),
    )

  gate.set()
  await _flush(service)


# --- Configuration ---


def test_config_rejects_unknown_fields():
  """A misspelled config field raises instead of being ignored."""
  with pytest.raises(ValueError):
    BufferedSessionServiceConfig(retry=RetryConfig())


async def test_logs_name_only_the_error_type(caplog: pytest.LogCaptureFixture):
  """A failed write is logged by error type, without its message."""
  store = _Store()
  service = _service(store, max_attempts=1)
  session = await _create(service)
  store.failures = [RuntimeError('secret user data')]
  caplog.set_level(logging.WARNING)

  await service.append_event(session, _event('e1'))
  with pytest.raises(SessionPersistenceError):
    await _flush(service)

  assert 'RuntimeError' in caplog.text
  assert 'secret user data' not in caplog.text
