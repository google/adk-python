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

"""Tests for SqliteSessionService append_events batching implementation."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from datetime import timezone

from google.adk.errors._stale_session_error import StaleSessionError
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.genai import types
import pytest


def _batch_events(start: float) -> list[Event]:
    return [
        Event(
            invocation_id="inv1",
            author="agent",
            timestamp=start + 1,
            actions=EventActions(
                state_delta={"k": 1, "user:u": "a", "app:a": 1, "temp:t": 1}
            ),
            content=types.Content(parts=[types.Part(text="a")]),
        ),
        Event(
            invocation_id="inv1",
            author="agent",
            timestamp=start + 2,
            actions=EventActions(state_delta={"k": 2, "user:u": None, "other": "x"}),
            content=types.Content(parts=[types.Part(text="b")]),
        ),
        Event(
            invocation_id="inv1",
            author="agent",
            timestamp=start + 3,
            content=types.Content(parts=[types.Part(text="c")]),
        ),
    ]


def test_sqlite_append_events_uses_one_transaction(tmp_path) -> None:
    """Batch appending events must execute in a single BEGIN IMMEDIATE transaction."""

    async def _run():
        db_file = str(tmp_path / "test_tx.db")
        service = SqliteSessionService(db_file)
        session = await service.create_session(app_name="app", user_id="u")
        start = datetime.now().astimezone(timezone.utc).timestamp()

        # Check that SqliteSessionService explicitly overrides append_events
        assert "append_events" in SqliteSessionService.__dict__, (
            "SqliteSessionService must override append_events to batch writes in"
            " one transaction"
        )

        commit_count = 0
        orig_get_db = service._get_db_connection

        @asynccontextmanager
        async def counting_get_db():
            nonlocal commit_count
            async with orig_get_db() as db:
                orig_commit = db.commit

                async def tracked_commit():
                    nonlocal commit_count
                    commit_count += 1
                    return await orig_commit()

                db.commit = tracked_commit
                yield db

        service._get_db_connection = counting_get_db
        events = _batch_events(start)
        returned = await service.append_events(session, events)

        assert len(returned) == 3
        assert commit_count == 1, (
            f"Expected 1 transaction commit for batch of 3 events, got {commit_count}"
        )
        await service.close()

    asyncio.run(_run())


def test_sqlite_append_events_failed_write_stores_nothing(tmp_path) -> None:
    """A failure while writing the batch rolls back every event in it."""

    async def _run():
        db_file = str(tmp_path / "test_rollback.db")
        service = SqliteSessionService(db_file)
        session = await service.create_session(app_name="app", user_id="u")
        start = datetime.now().astimezone(timezone.utc).timestamp()
        events = _batch_events(start)

        # First check that append_events is overridden
        assert "append_events" in SqliteSessionService.__dict__, (
            "SqliteSessionService must override append_events"
        )

        orig_get_db = service._get_db_connection

        @asynccontextmanager
        async def failing_get_db():
            async with orig_get_db() as db:

                async def failing_executemany(*args, **kwargs):
                    raise RuntimeError("write failed")

                db.executemany = failing_executemany
                yield db

        service._get_db_connection = failing_get_db

        with pytest.raises(RuntimeError, match="write failed"):
            await service.append_events(session, events)

        stored = await service.get_session(
            app_name="app", user_id="u", session_id=session.id
        )
        assert stored.events == []
        assert "k" not in stored.state
        assert session.events == []
        await service.close()

    asyncio.run(_run())


def test_sqlite_append_events_matches_one_by_one(tmp_path) -> None:
    """Batch appending events produces identical state and events to one-by-one append."""

    async def _run():
        assert "append_events" in SqliteSessionService.__dict__, (
            "SqliteSessionService must override append_events"
        )
        start = datetime.now().astimezone(timezone.utc).timestamp()
        outcomes = {}

        for mode in ("one_by_one", "batched"):
            db_file = str(tmp_path / f"test_{mode}.db")
            service = SqliteSessionService(db_file)
            session = await service.create_session(app_name="app", user_id="u")

            if mode == "one_by_one":
                for event in _batch_events(start):
                    await service.append_event(session, event)
            else:
                await service.append_events(session, _batch_events(start))

            stored = await service.get_session(
                app_name="app", user_id="u", session_id=session.id
            )
            outcomes[mode] = (
                [e.content.parts[0].text for e in stored.events],
                dict(session.state),
                session.last_update_time,
                dict(stored.state),
                stored.last_update_time,
            )
            await service.close()

        assert outcomes["batched"] == outcomes["one_by_one"]
        texts, live_state, live_time, stored_state, stored_time = outcomes["batched"]
        assert texts == ["a", "b", "c"]
        assert stored_state == {"k": 2, "user:u": None, "app:a": 1, "other": "x"}
        assert live_state["temp:t"] == 1
        assert live_time == stored_time == start + 3

    asyncio.run(_run())


def test_sqlite_append_events_to_stale_session_stores_nothing(tmp_path) -> None:
    """Appending to a stale session raises StaleSessionError and writes nothing."""

    async def _run():
        assert "append_events" in SqliteSessionService.__dict__, (
            "SqliteSessionService must override append_events"
        )
        db_file = str(tmp_path / "test_stale.db")
        service = SqliteSessionService(db_file)
        stale = await service.create_session(app_name="app", user_id="u")
        fresh = await service.get_session(
            app_name="app", user_id="u", session_id=stale.id
        )
        # Ensure fresh event timestamp is strictly ahead of stale.last_update_time
        newer_time = stale.last_update_time + 10.0
        await service.append_event(
            fresh, Event(invocation_id="inv0", author="user", timestamp=newer_time)
        )

        with pytest.raises(StaleSessionError):
            await service.append_events(stale, _batch_events(newer_time + 1.0))

        stored = await service.get_session(
            app_name="app", user_id="u", session_id=stale.id
        )
        assert [e.invocation_id for e in stored.events] == ["inv0"]
        assert "k" not in stored.state
        assert stale.events == []
        await service.close()

    asyncio.run(_run())


def test_sqlite_append_events_skips_partial_events(tmp_path) -> None:
    """Partial events are returned but not persisted to disk."""

    async def _run():
        assert "append_events" in SqliteSessionService.__dict__, (
            "SqliteSessionService must override append_events"
        )
        db_file = str(tmp_path / "test_partial.db")
        service = SqliteSessionService(db_file)
        session = await service.create_session(app_name="app", user_id="u")
        start = datetime.now().astimezone(timezone.utc).timestamp()
        partial = Event(
            invocation_id="inv1", author="agent", timestamp=start, partial=True
        )
        full = Event(invocation_id="inv1", author="agent", timestamp=start + 1)

        assert await service.append_events(session, [partial, full]) == [
            partial,
            full,
        ]
        assert await service.append_events(session, [partial]) == [partial]
        assert await service.append_events(session, []) == []

        stored = await service.get_session(
            app_name="app", user_id="u", session_id=session.id
        )
        assert [e.id for e in stored.events] == [full.id]
        assert [e.id for e in session.events] == [full.id]
        await service.close()

    asyncio.run(_run())
