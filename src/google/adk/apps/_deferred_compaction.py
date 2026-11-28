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
"""Durable sliding-window work shared by serving and worker runners."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import uuid

from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.pool import StaticPool

from ..agents.base_agent import BaseAgent
from ..events._rewind_events import get_active_events
from ..events.event import Event
from ..events.event_actions import EventActions
from ..sessions.base_session_service import BaseSessionService
from ..sessions.database_session_service import DatabaseSessionService
from ..sessions.schemas.v1 import PendingCompaction as Job
from ..sessions.schemas.v1 import StorageEvent
from ..sessions.schemas.v1 import StorageSession
from ..sessions.session import Session
from ._configs import EventsCompactionConfig
from .compaction import _ensure_compaction_summarizer
from .compaction import _longest_self_contained_prefix
from .compaction import _sliding_window
from .compaction import _summarize_events_with_trace

_LEASE_SECONDS = 120


def validate_backend(service: BaseSessionService) -> DatabaseSessionService:
  if not isinstance(service, DatabaseSessionService):
    raise ValueError('Deferred compaction requires DatabaseSessionService.')
  if service.db_engine.dialect.name not in ('postgresql', 'sqlite'):
    raise ValueError('Deferred compaction requires PostgreSQL or SQLite.')
  if service.db_engine.dialect.name == 'sqlite' and isinstance(
      service.db_engine.pool, StaticPool
  ):
    raise ValueError(
        'Deferred compaction requires a shared, file-backed SQLite database.'
    )
  return service


async def _database(service: BaseSessionService) -> DatabaseSessionService:
  service = validate_backend(service)
  await service.prepare_tables()
  if service._get_schema_classes().StorageEvent is not StorageEvent:
    raise ValueError('Deferred compaction requires database schema v1.')
  return service


@asynccontextmanager
async def _transaction(
    service: DatabaseSessionService,
) -> AsyncIterator[AsyncSession]:
  async with service.database_session_factory() as db, db.begin():
    if service.db_engine.dialect.name == 'sqlite':
      # Injected engines may lack the service's BEGIN IMMEDIATE hook. Start
      # with a write so concurrent claimants cannot both read the same row.
      await db.execute(update(Job).where(False).values(available_at=0))
    yield db


async def enqueue_window(
    service: BaseSessionService,
    session: Session,
    config: EventsCompactionConfig,
) -> None:
  service = await _database(service)
  live = [
      e for e in get_active_events(session.events) if e.isolation_scope is None
  ]
  window = _sliding_window(config, live)
  if not window:
    return
  last = next(e for e in reversed(live) if not e.actions.compaction)
  insert = (
      postgres_insert
      if service.db_engine.dialect.name == 'postgresql'
      else sqlite_insert
  )
  stmt = insert(Job).values(
      app_name=session.app_name,
      user_id=session.user_id,
      session_id=session.id,
      through=last.id,
      through_time=last.timestamp,
      available_at=0,
  )
  async with _transaction(service) as db:
    await db.execute(
        stmt.on_conflict_do_update(
            index_elements=[Job.app_name, Job.user_id, Job.session_id],
            set_={'through': last.id, 'through_time': last.timestamp},
            where=Job.through_time <= last.timestamp,
        )
    )


def _window(
    events: list[Event], through: str, config: EventsCompactionConfig
) -> list[Event]:
  live = [e for e in get_active_events(events) if e.isolation_scope is None]
  for i, event in enumerate(live):
    if event.id == through:
      break
  else:
    return []
  # Summaries may be published after the raw events they cover. A subsequent
  # worker must see those summaries even when replaying an older watermark.
  later = live[i + 1 :]
  if any(
      e.actions.compaction
      and e.actions.compaction.end_timestamp is not None
      and e.actions.compaction.end_timestamp >= event.timestamp
      for e in live
  ):
    return []
  if any(
      not e.actions.compaction and e.timestamp <= event.timestamp for e in later
  ):
    return []
  history = live[: i + 1] + [e for e in later if e.actions.compaction]
  window = _sliding_window(config, history)
  window = _longest_self_contained_prefix(window, all_events=live)
  if window:
    # The prompt processor applies inclusive timestamp ranges. Tied timestamps
    # at a boundary must not hide a raw event the summarizer never received.
    covered = [
        e.id
        for e in live
        if not e.actions.compaction
        and window[0].timestamp <= e.timestamp <= window[-1].timestamp
    ]
    if covered != [e.id for e in window]:
      return []
  return window


async def _load_events(db: AsyncSession, session: Session) -> list[Event]:
  rows = await db.scalars(
      select(StorageEvent)
      .filter_by(
          app_name=session.app_name,
          user_id=session.user_id,
          session_id=session.id,
      )
      .order_by(StorageEvent.timestamp, StorageEvent.id)
  )
  return [row.to_event() for row in rows]


def _now(service: DatabaseSessionService):
  # All instances compare deadlines against the same database clock.
  if service.db_engine.dialect.name == 'sqlite':
    return (func.julianday('now') - 2440587.5) * 86400
  return func.extract('epoch', func.now())


async def _renew(
    service: DatabaseSessionService, key: tuple[str, str, str], owner: str
) -> None:
  while True:
    await asyncio.sleep(_LEASE_SECONDS / 3)
    async with _transaction(service) as db:
      result = await db.execute(
          update(Job)
          .where(
              Job.app_name == key[0],
              Job.user_id == key[1],
              Job.session_id == key[2],
              Job.owner == owner,
          )
          .values(available_at=_now(service) + _LEASE_SECONDS)
      )
      if not result.rowcount:
        return


async def wait_for_compaction(
    service: BaseSessionService, key: tuple[str, str, str]
) -> None:
  service = await _database(service)
  while True:
    async with service._get_database_session_factory(read_only=True)() as db:
      job = await db.get(Job, key)
      if job is None:
        return
      if job.last_error:
        raise RuntimeError(f'Last compaction attempt failed: {job.last_error}')
    await asyncio.sleep(0.1)


async def process_next_window(
    service: BaseSessionService,
    app_name: str,
    agent: BaseAgent,
    config: EventsCompactionConfig,
) -> bool:
  service = await _database(service)
  owner = str(uuid.uuid4())
  now = _now(service)
  async with _transaction(service) as db:
    job = await db.scalar(
        select(Job)
        .where(Job.app_name == app_name, Job.available_at <= now)
        .order_by(Job.available_at, Job.through_time)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    if job is None:
      return False
    job.owner = owner
    job.available_at = now + _LEASE_SECONDS
    key = (job.app_name, job.user_id, job.session_id)
    through = job.through

  heartbeat = asyncio.create_task(_renew(service, key, owner))
  try:
    await _process_claim(service, key, owner, through, agent, config)
  except Exception as exc:
    async with _transaction(service) as db:
      job = await db.get(Job, key, with_for_update=True)
      if job is not None and job.owner == owner:
        job.last_error = type(exc).__name__[:255]
    raise
  finally:
    heartbeat.cancel()
    await asyncio.gather(heartbeat, return_exceptions=True)
  return True


async def _process_claim(
    service: DatabaseSessionService,
    key: tuple[str, str, str],
    owner: str,
    through: str,
    agent: BaseAgent,
    config: EventsCompactionConfig,
) -> None:
  session = Session(app_name=key[0], user_id=key[1], id=key[2])
  async with service._get_database_session_factory(read_only=True)() as db:
    session.events = await _load_events(db, session)
  window = _window(session.events, through, config)
  summary = None
  if window:
    _ensure_compaction_summarizer(config=config, agent=agent)
    summary = await _summarize_events_with_trace(
        session=session,
        config=config,
        events=session.events,
        events_to_compact=window,
        trigger='sliding_window',
        propagate_errors=True,
    )
    if summary is None:
      raise RuntimeError('Summarizer returned no event for a non-empty window.')
    bounds = summary.actions.compaction
    if not bounds or (bounds.start_timestamp, bounds.end_timestamp) != (
        window[0].timestamp,
        window[-1].timestamp,
    ):
      raise ValueError('Summary range differs from the queued events.')

  async with _transaction(service) as db:
    # Serialize publication with foreground writes, but do not advance the
    # session state revision: summaries are derived, and a reader of the raw
    # events can still append its answer.
    if await db.get(StorageSession, key, with_for_update=True) is None:
      return
    job = await db.get(Job, key, with_for_update=True)
    if job is None or job.owner != owner:
      return
    if summary is not None:
      current = _window(await _load_events(db, session), through, config)
      if current and [e.id for e in current] != [e.id for e in window]:
        # The window is still eligible but its inputs changed. Leave it for
        # a fresh attempt, even if the serving runner hasn't moved the target.
        job.owner = None
        job.available_at = 0
        job.last_error = None
        return
      if current:
        derived = Event(
            author=summary.author,
            invocation_id=summary.invocation_id,
            usage_metadata=summary.usage_metadata,
            actions=EventActions(compaction=summary.actions.compaction),
        )
        db.add(StorageEvent.from_event(session, derived))
    if job.through == through:
      await db.delete(job)
    else:
      job.owner = None
      job.last_error = None
      job.available_at = 0
