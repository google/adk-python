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
"""Recovery and publication of queued summaries across database connections."""

import asyncio

from google.adk.agents import LlmAgent
from google.adk.apps import _deferred_compaction as queue
from google.adk.apps.app import App
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.base_events_summarizer import BaseEventsSummarizer
from google.adk.events import Event
from google.adk.events import EventActions
from google.adk.events.event_actions import EventCompaction
from google.adk.runners import Runner
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.adk.sessions.schemas.v1 import PendingCompaction
from google.genai import types
import pytest
import pytest_asyncio
from sqlalchemy import update


class ControlledSummary(BaseEventsSummarizer):

  def __init__(self):
    self.started = asyncio.Event()
    self.release = asyncio.Event()
    self.fail = False

  async def maybe_summarize_events(self, *, events):
    self.started.set()
    await self.release.wait()
    if self.fail:
      raise TimeoutError('provider timeout')
    return Event(
        author='user',
        invocation_id=Event.new_id(),
        actions=EventActions(
            compaction=EventCompaction(
                start_timestamp=events[0].timestamp,
                end_timestamp=events[-1].timestamp,
                compacted_content=types.Content(
                    role='model', parts=[types.Part(text='summary')]
                ),
            )
        ),
    )


def message(turn):
  return Event(
      author='user',
      invocation_id=turn,
      content=types.Content(role='user', parts=[types.Part(text=turn * 120)]),
  )


@pytest_asyncio.fixture
async def runners(tmp_path):
  url = f'sqlite+aiosqlite:///{tmp_path / "sessions.db"}'
  services = [DatabaseSessionService(url) for _ in range(2)]
  summary = ControlledSummary()
  config = EventsCompactionConfig(
      defer_sliding_window=True,
      summarizer=summary,
      compaction_interval=2,
      overlap_size=0,
  )
  app = App(
      name='orders',
      root_agent=LlmAgent(name='clerk'),
      events_compaction_config=config,
  )
  first, second = [Runner(app=app, session_service=s) for s in services]
  try:
    session = await services[0].create_session(
        app_name=app.name, user_id='u', session_id='s'
    )
    for turn in ('one', 'two'):
      await services[0].append_event(session=session, event=message(turn))
    await queue.enqueue_window(services[0], session, config)
    yield first, second, session, summary
  finally:
    for service in services:
      await service.close()


async def loaded(runner):
  return await runner.session_service.get_session(
      app_name='orders', user_id='u', session_id='s'
  )


async def expire(runner):
  async with (
      runner.session_service.database_session_factory() as db,
      db.begin(),
  ):
    await db.execute(update(PendingCompaction).values(available_at=0))


@pytest.mark.asyncio
async def test_competing_worker_and_followup(runners):
  first, second, session, summary = runners
  task = asyncio.create_task(first.compact_next_session())
  await asyncio.wait_for(summary.started.wait(), 5)
  assert not await second.compact_next_session()
  followup = await loaded(second)
  assert [e.invocation_id for e in followup.events] == ['one', 'two']
  await second.session_service.append_event(
      session=followup, event=message('three')
  )
  await queue.enqueue_window(
      second.session_service, followup, second.app.events_compaction_config
  )
  summary.release.set()
  assert await task
  await second.session_service.append_event(
      session=followup, event=message('four')
  )
  assert await second.compact_next_session()
  assert not await second.compact_next_session()
  result = await loaded(second)
  assert [
      e.invocation_id for e in result.events if not e.actions.compaction
  ] == ['one', 'two', 'three', 'four']
  compacted = [
      e.actions.compaction for e in result.events if e.actions.compaction
  ]
  assert len(compacted) == 1
  assert compacted[0].end_timestamp == session.events[-1].timestamp


@pytest.mark.asyncio
async def test_expired_claim_cannot_publish_twice(runners):
  first, second, _, summary = runners
  stale = asyncio.create_task(first.compact_next_session())
  await asyncio.wait_for(summary.started.wait(), 5)
  await expire(second)
  replacement = ControlledSummary()
  replacement.release.set()
  second.app.events_compaction_config.summarizer = replacement
  assert await second.compact_next_session()
  assert replacement.started.is_set()
  summary.release.set()
  assert await stale
  assert (
      sum(bool(e.actions.compaction) for e in (await loaded(second)).events)
      == 1
  )


@pytest.mark.asyncio
async def test_failure_survives_until_retry(runners):
  first, second, _, summary = runners
  summary.release.set()
  summary.fail = True
  with pytest.raises(TimeoutError):
    await first.compact_next_session()
  with pytest.raises(RuntimeError, match='TimeoutError'):
    await second.wait_for_pending_compaction(user_id='u', session_id='s')
  assert len((await loaded(second)).events) == 2
  await expire(second)
  summary.fail = False
  assert await second.compact_next_session()
  await second.wait_for_pending_compaction(user_id='u', session_id='s')
  assert len((await loaded(second)).events) == 3


@pytest.mark.asyncio
async def test_rewind_while_summarizing(runners):
  first, second, session, summary = runners
  pending = asyncio.create_task(first.compact_next_session())
  await asyncio.wait_for(summary.started.wait(), 5)
  await second.session_service.append_event(
      session=session,
      event=Event(
          author='user',
          invocation_id='undo',
          actions=EventActions(rewind_before_invocation_id='two'),
      ),
  )
  summary.release.set()
  assert await pending
  assert not any(e.actions.compaction for e in (await loaded(second)).events)


@pytest.mark.asyncio
async def test_offline_worker_coalesces_turns(runners):
  first, second, session, summary = runners
  for index in range(20):
    await first.session_service.append_event(
        session=session, event=message(f'followup-{index}')
    )
    await queue.enqueue_window(
        first.session_service, session, first.app.events_compaction_config
    )
  assert not summary.started.is_set()
  assert len((await loaded(second)).events) == 22
  summary.release.set()
  assert await second.compact_next_session()
  assert not await second.compact_next_session()
  stored = await loaded(second)
  assert [e.id for e in stored.events if not e.actions.compaction] == [
      e.id for e in session.events
  ]
  assert (
      stored.events[-1].actions.compaction.end_timestamp
      == session.events[-1].timestamp
  )


@pytest.mark.asyncio
async def test_deleted_session_cannot_receive_an_old_summary(runners):
  first, second, _, summary = runners
  pending = asyncio.create_task(first.compact_next_session())
  await asyncio.wait_for(summary.started.wait(), 5)
  await second.session_service.delete_session('orders', 'u', 's')
  await second.session_service.create_session(
      app_name='orders', user_id='u', session_id='s'
  )
  summary.release.set()
  assert await pending
  assert (await loaded(second)).events == []
  assert not await second.compact_next_session()
