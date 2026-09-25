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
from pathlib import Path
import sqlite3

import aiosqlite
from google.adk.events.event import Event
from google.adk.memory import SqliteMemoryService
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.adk.sessions.migration._schema_check_utils import get_db_schema_version
from google.adk.sessions.session import Session
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.genai import types
import pytest

_HAS_FTS5 = True
try:
  with sqlite3.connect(":memory:") as _conn:
    _conn.execute("CREATE VIRTUAL TABLE _test_fts USING fts5(col)")
except sqlite3.OperationalError:
  _HAS_FTS5 = False


@pytest.fixture
def require_fts5():
  if not _HAS_FTS5:
    pytest.skip("FTS5 not available in this SQLite build.")


@pytest.fixture(params=["off", "on"])
def fts_mode(request):
  if request.param == "on" and not _HAS_FTS5:
    pytest.skip("FTS5 not available in this SQLite build.")
  return request.param


def _make_event(
    author: str,
    text: str,
    timestamp: float,
    *,
    event_id: str | None = None,
) -> Event:
  event = Event(
      author=author,
      timestamp=timestamp,
      content=types.Content(
          role="user",
          parts=[types.Part(text=text)],
      ),
  )
  if event_id is not None:
    event.id = event_id
  return event


def _make_session(
    events: list[Event],
    *,
    session_id: str = "session-1",
    app_name: str = "app",
    user_id: str = "user",
) -> Session:
  return Session(
      id=session_id,
      app_name=app_name,
      user_id=user_id,
      events=events,
      last_update_time=0.0,
  )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "make_uri",
    [
        lambda p: f"sqlite:///{p / 'abs.db'}",
        lambda p: f"sqlite+aiosqlite:///{p / 'abs_aio.db'}",
        lambda p: str(p / "plain.db"),
        lambda p: p / "path_obj.db",
    ],
)
async def test_sqlite_uri_variants_store_and_search_memories(
    tmp_path, make_uri
):
  """SQLite URI format variants store and retrieve memories through the public interface."""
  uri = make_uri(tmp_path)
  service = SqliteMemoryService(uri, fts="off")
  event = _make_event("user", "test content", 1.0)
  await service.add_events_to_memory(
      app_name="app", user_id="user", events=[event], session_id="s1"
  )
  resp = await service.search_memory(
      app_name="app", user_id="user", query="test"
  )
  assert len(resp.memories) == 1
  assert resp.memories[0].content.parts[0].text == "test content"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "uri_scheme",
    [
        "sqlite:///rel.db",
        "sqlite+aiosqlite:///rel_aio.db",
    ],
)
async def test_relative_path_uri_variants(tmp_path, monkeypatch, uri_scheme):
  """Relative SQLite URIs resolve relative to the current working directory."""
  monkeypatch.chdir(tmp_path)
  service = SqliteMemoryService(uri_scheme, fts="off")
  event = _make_event("user", "relative content", 1.0)
  await service.add_events_to_memory(
      app_name="app", user_id="user", events=[event], session_id="s1"
  )
  resp = await service.search_memory(
      app_name="app", user_id="user", query="relative"
  )
  assert len(resp.memories) == 1
  assert resp.memories[0].content.parts[0].text == "relative content"


@pytest.mark.asyncio
@pytest.mark.parametrize("uri", [":memory:", "sqlite:///:memory:"])
async def test_in_memory_uri_variants(uri):
  """In-memory SQLite URI variants store and search memories without creating files on disk."""
  service = SqliteMemoryService(uri, fts="off")
  try:
    event = _make_event("user", "in-memory content", 1.0)
    await service.add_events_to_memory(
        app_name="app", user_id="user", events=[event], session_id="s1"
    )
    resp = await service.search_memory(
        app_name="app", user_id="user", query="in-memory"
    )
    assert len(resp.memories) == 1
    assert resp.memories[0].content.parts[0].text == "in-memory content"
  finally:
    await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ro_uri_fn",
    [
        lambda p: f"sqlite:///{p / 'ro_abs.db'}?mode=ro",
        lambda p: f"sqlite+aiosqlite:///{p / 'ro_abs.db'}?mode=ro",
    ],
)
async def test_readonly_uri_mode_restricts_writes(tmp_path, ro_uri_fn):
  """Read-only URI query parameter enables searching existing rows and rejects new writes."""
  db_path = tmp_path / "ro_abs.db"
  rw_service = SqliteMemoryService(db_path=db_path, fts="off")
  event = _make_event("user", "readonly test content", 1.0)
  await rw_service.add_events_to_memory(
      app_name="app", user_id="user", events=[event], session_id="s1"
  )

  ro_uri = ro_uri_fn(tmp_path)
  ro_service = SqliteMemoryService(ro_uri, fts="off")
  resp = await ro_service.search_memory(
      app_name="app", user_id="user", query="readonly"
  )
  assert len(resp.memories) == 1
  assert resp.memories[0].content.parts[0].text == "readonly test content"

  with pytest.raises((sqlite3.OperationalError, aiosqlite.OperationalError)):
    new_event = _make_event("user", "cannot write", 2.0)
    await ro_service.add_events_to_memory(
        app_name="app", user_id="user", events=[new_event], session_id="s1"
    )


def test_sqlite_uri_does_not_create_sqlite_colon_directory(tmp_path):
  """Initializing with a sqlite:/// URI does not create a literal sqlite: directory."""
  uri = f"sqlite:///{tmp_path}/nested/memory.db"
  SqliteMemoryService(uri)
  assert not Path("sqlite:").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "batches, expected_rows",
    [
        (
            [
                [_make_event("user", "Hello memory", 1.0, event_id="e1")],
                [_make_event("user", "Hello memory", 1.0, event_id="e1")],
            ],
            1,
        ),
        (
            [
                [_make_event("user", "First event", 1.0, event_id="e1")],
                [
                    _make_event("user", "First event", 1.0, event_id="e1"),
                    _make_event(
                        "assistant", "Second event", 2.0, event_id="e2"
                    ),
                ],
            ],
            2,
        ),
        (
            [
                [
                    _make_event(
                        "user", "Favorite color is blue", 1.0, event_id="e1"
                    )
                ],
                [
                    _make_event(
                        "assistant", "Noted blue color", 2.0, event_id="e2"
                    )
                ],
            ],
            2,
        ),
    ],
)
async def test_add_events_row_counts(tmp_path, batches, expected_rows):
  """Adding event batches deduplicates identical events and inserts new ones."""
  db_path = tmp_path / "memory.db"
  service = SqliteMemoryService(db_path=db_path, fts="off")
  for batch in batches:
    await service.add_events_to_memory(
        app_name="app", user_id="user", events=batch, session_id="s1"
    )
  with sqlite3.connect(db_path) as conn:
    count = conn.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0]
    assert count == expected_rows


@pytest.mark.asyncio
async def test_add_event_without_id_uses_deterministic_hash_deduplication(
    tmp_path,
):
  """Events with empty IDs use a deterministic content hash for deduplication."""
  db_path = tmp_path / "memory.db"
  service = SqliteMemoryService(db_path=db_path, fts="off")
  event1 = _make_event("user", "Event with empty id", 1.0, event_id="")
  event2 = _make_event("user", "Event with empty id", 1.0, event_id="")
  await service.add_events_to_memory(
      app_name="app", user_id="user", events=[event1], session_id="s1"
  )
  await service.add_events_to_memory(
      app_name="app", user_id="user", events=[event2], session_id="s1"
  )
  with sqlite3.connect(db_path) as conn:
    count = conn.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0]
    assert count == 1


@pytest.mark.asyncio
async def test_coexistence_with_sqlite_session_service(tmp_path, fts_mode):
  """SqliteMemoryService and SqliteSessionService share a database without table collisions."""
  db_path = tmp_path / "shared.db"
  session_service = SqliteSessionService(db_path=str(db_path))
  memory_service = SqliteMemoryService(db_path=str(db_path), fts=fts_mode)

  session = await session_service.create_session(
      app_name="test_app", user_id="test_user", session_id="s1"
  )
  event = _make_event("user", "Hello shared database", 1.0)
  await session_service.append_event(session, event)
  await memory_service.add_session_to_memory(session)

  resp = await memory_service.search_memory(
      app_name="test_app", user_id="test_user", query="shared"
  )
  assert len(resp.memories) == 1
  assert resp.memories[0].author == "user"
  assert resp.memories[0].content.parts[0].text == "Hello shared database"


@pytest.mark.asyncio
async def test_preload_memory_tool_compatibility(tmp_path, fts_mode):
  """Natural language queries return relevant stored memories with timestamps."""
  db_path = tmp_path / "memory.db"
  service = SqliteMemoryService(db_path=db_path, fts=fts_mode)
  event = _make_event("user", "My favorite fruit is apple", 1.0)
  session = _make_session([event])
  await service.add_session_to_memory(session)

  resp = await service.search_memory(
      app_name="app", user_id="user", query="What is my favorite fruit?"
  )
  assert len(resp.memories) == 1
  memory = resp.memories[0]
  assert memory.author == "user"
  assert memory.content.parts[0].text == "My favorite fruit is apple"
  assert memory.timestamp is not None


@pytest.mark.asyncio
async def test_search_ranking_relevance(tmp_path, fts_mode):
  """Search results rank events with higher query token overlap first."""
  db_path = tmp_path / "memory.db"
  service = SqliteMemoryService(db_path=db_path, fts=fts_mode)
  low_relevance_event = _make_event(
      "user", "The weather is very sunny today", 10.0
  )
  high_relevance_event = _make_event(
      "user", "My favorite fruit is apple and sweet orange", 1.0
  )
  unrelated_event = _make_event("user", "Completely different topic", 5.0)

  session = _make_session(
      [low_relevance_event, high_relevance_event, unrelated_event]
  )
  await service.add_session_to_memory(session)

  response = await service.search_memory(
      app_name="app", user_id="user", query="What is my favorite fruit?"
  )
  assert len(response.memories) >= 2
  assert (
      response.memories[0].content.parts[0].text
      == "My favorite fruit is apple and sweet orange"
  )


@pytest.mark.asyncio
async def test_persistence_across_restarts(tmp_path, fts_mode):
  """Memories persist across separate SqliteMemoryService instances on the same database."""
  db_path = tmp_path / "memory.db"
  service = SqliteMemoryService(db_path=db_path, fts=fts_mode)
  session = _make_session([_make_event("user", "Remember me", 1.0)])
  await service.add_session_to_memory(session)

  new_service = SqliteMemoryService(db_path=db_path, fts=fts_mode)
  response = await new_service.search_memory(
      app_name="app", user_id="user", query="Remember"
  )
  assert response.memories
  assert response.memories[0].custom_metadata["session_id"] == session.id


@pytest.mark.asyncio
async def test_reopen_database_with_fts_finds_earlier_rows_once(
    tmp_path, require_fts5
):
  """Reopening an FTS-enabled database finds previously inserted rows without duplicates."""
  db_path = tmp_path / "memory.db"
  service1 = SqliteMemoryService(db_path=db_path, fts="on")
  session = _make_session([_make_event("user", "test persistent memory", 1.0)])
  await service1.add_session_to_memory(session)

  service2 = SqliteMemoryService(db_path=db_path, fts="on")
  response = await service2.search_memory(
      app_name="app", user_id="user", query="persistent"
  )
  assert len(response.memories) == 1
  assert response.memories[0].content.parts[0].text == "test persistent memory"


@pytest.mark.asyncio
async def test_in_memory_database_multiple_operations(fts_mode):
  """An in-memory database retains memories across multiple operations until closed."""
  service = SqliteMemoryService(":memory:", fts=fts_mode)
  session1 = _make_session(
      [_make_event("user", "first session info", 1.0)], session_id="s1"
  )
  session2 = _make_session(
      [_make_event("user", "second session info", 2.0)], session_id="s2"
  )
  await service.add_session_to_memory(session1)
  await service.add_session_to_memory(session2)

  resp1 = await service.search_memory(
      app_name="app", user_id="user", query="first"
  )
  assert len(resp1.memories) == 1
  assert resp1.memories[0].custom_metadata["session_id"] == "s1"

  resp2 = await service.search_memory(
      app_name="app", user_id="user", query="second"
  )
  assert len(resp2.memories) == 1
  assert resp2.memories[0].custom_metadata["session_id"] == "s2"
  await service.close()


@pytest.mark.asyncio
async def test_search_like_escaping_wildcards(tmp_path):
  """LIKE search fallback escapes percent and underscore wildcard characters in queries."""
  db_path = tmp_path / "memory.db"
  service = SqliteMemoryService(db_path=db_path, fts="off")
  session1 = _make_session(
      [_make_event("user", "Discount is 100% complete", 1.0)], session_id="s1"
  )
  session2 = _make_session(
      [_make_event("user", "Discount is 1000 complete", 2.0)], session_id="s2"
  )
  session3 = _make_session(
      [_make_event("user", "variable foo_bar here", 3.0)], session_id="s3"
  )
  session4 = _make_session(
      [_make_event("user", "variable foo1bar here", 4.0)], session_id="s4"
  )
  await service.add_session_to_memory(session1)
  await service.add_session_to_memory(session2)
  await service.add_session_to_memory(session3)
  await service.add_session_to_memory(session4)

  resp_percent = await service.search_memory(
      app_name="app", user_id="user", query="100%"
  )
  assert len(resp_percent.memories) == 1
  assert resp_percent.memories[0].custom_metadata["session_id"] == "s1"

  resp_underscore = await service.search_memory(
      app_name="app", user_id="user", query="foo_bar"
  )
  assert len(resp_underscore.memories) == 1
  assert resp_underscore.memories[0].custom_metadata["session_id"] == "s3"


@pytest.mark.asyncio
async def test_fts_rebuild_indexes_preexisting_rows(tmp_path, require_fts5):
  """Enabling FTS on a database with existing rows indexes them for search."""
  db_path = tmp_path / "memory.db"
  service_off = SqliteMemoryService(db_path=db_path, fts="off")
  session = _make_session(
      [_make_event("user", "preexisting documentation entry", 1.0)]
  )
  await service_off.add_session_to_memory(session)

  service_on = SqliteMemoryService(db_path=db_path, fts="on")
  response = await service_on.search_memory(
      app_name="app", user_id="user", query="preexisting"
  )
  assert len(response.memories) == 1
  assert response.memories[0].custom_metadata["session_id"] == session.id


@pytest.mark.asyncio
async def test_memory_events_schema(tmp_path):
  """Database schema creates the expected memory tables and columns without sessions table."""
  db_path = tmp_path / "memory.db"
  service = SqliteMemoryService(db_path=db_path, fts="off")
  session = _make_session([_make_event("user", "test content", 1.0)])
  await service.add_session_to_memory(session)

  with sqlite3.connect(db_path) as conn:
    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    ]
    assert "memory_events" in tables
    assert "adk_memory_metadata" in tables
    assert "adk_internal_metadata" not in tables
    assert "sessions" not in tables
    columns = [
        row[1]
        for row in conn.execute("PRAGMA table_info(memory_events)").fetchall()
    ]
    assert "content_json" in columns
    assert "search_text" in columns
    assert "author" in columns
    assert "timestamp" in columns


@pytest.mark.asyncio
async def test_cross_user_and_app_scoping(tmp_path, fts_mode):
  """Search queries only return memories matching the requested app_name and user_id."""
  db_path = tmp_path / "memory.db"
  service = SqliteMemoryService(db_path=db_path, fts=fts_mode)
  session_app1_user1 = _make_session(
      [_make_event("user", "apple fruit entry", 1.0)],
      session_id="s1",
      app_name="app1",
      user_id="user1",
  )
  session_app1_user2 = _make_session(
      [_make_event("user", "apple pie entry", 2.0)],
      session_id="s2",
      app_name="app1",
      user_id="user2",
  )
  session_app2_user1 = _make_session(
      [_make_event("user", "apple cider entry", 3.0)],
      session_id="s3",
      app_name="app2",
      user_id="user1",
  )
  await service.add_session_to_memory(session_app1_user1)
  await service.add_session_to_memory(session_app1_user2)
  await service.add_session_to_memory(session_app2_user1)

  resp_a1_u1 = await service.search_memory(
      app_name="app1", user_id="user1", query="apple"
  )
  assert len(resp_a1_u1.memories) == 1
  assert resp_a1_u1.memories[0].custom_metadata["session_id"] == "s1"

  resp_a1_u2 = await service.search_memory(
      app_name="app1", user_id="user2", query="apple"
  )
  assert len(resp_a1_u2.memories) == 1
  assert resp_a1_u2.memories[0].custom_metadata["session_id"] == "s2"

  resp_a2_u1 = await service.search_memory(
      app_name="app2", user_id="user1", query="apple"
  )
  assert len(resp_a2_u1.memories) == 1
  assert resp_a2_u1.memories[0].custom_metadata["session_id"] == "s3"

  resp_a2_u2 = await service.search_memory(
      app_name="app2", user_id="user2", query="apple"
  )
  assert len(resp_a2_u2.memories) == 0


@pytest.mark.asyncio
async def test_add_session_truncates_large_payload(tmp_path):
  """Search index text is truncated to max_event_bytes while preserving row insertion."""
  db_path = tmp_path / "memory.db"
  service = SqliteMemoryService(db_path=db_path, fts="off", max_event_bytes=200)
  large_text = "x" * 1000
  session = _make_session([_make_event("user", large_text, 1.0)])
  await service.add_session_to_memory(session)

  with sqlite3.connect(db_path) as conn:
    row = conn.execute(
        "SELECT LENGTH(search_text), search_text FROM memory_events"
    ).fetchone()
    assert row[0] <= 200


@pytest.mark.asyncio
async def test_concurrent_upsert_across_instances(tmp_path):
  """Concurrent writes to the same event from separate instances resolve cleanly."""
  db_path = tmp_path / "memory.db"
  service1 = SqliteMemoryService(db_path=db_path, fts="off")
  service2 = SqliteMemoryService(db_path=db_path, fts="off")
  session = _make_session([_make_event("user", "concurrent write", 1.0)])

  await asyncio.gather(
      service1.add_session_to_memory(session),
      service2.add_session_to_memory(session),
  )

  with sqlite3.connect(db_path) as conn:
    row = conn.execute(
        "SELECT COUNT(*), search_text FROM memory_events"
    ).fetchone()
    assert row[0] == 1
    assert "concurrent write" in row[1]


@pytest.mark.asyncio
async def test_search_like_fallback_matches_punctuated_query(
    tmp_path, fts_mode
):
  """LIKE fallback tokenizes punctuated queries and matches stored events."""
  db_path = tmp_path / "memory.db"
  service = SqliteMemoryService(db_path=db_path, fts=fts_mode)
  session = _make_session(
      [_make_event("user", "My favorite fruit is apple.", 1.0)]
  )
  await service.add_session_to_memory(session)

  response = await service.search_memory(
      app_name="app", user_id="user", query="Which fruit?"
  )
  assert len(response.memories) == 1
  assert (
      response.memories[0].content.parts[0].text
      == "My favorite fruit is apple."
  )
  await service.close()


@pytest.mark.asyncio
async def test_add_events_updates_content_and_timestamp_when_search_text_unchanged(
    tmp_path,
):
  """Updating an event with identical search text updates content_json and timestamp."""
  db_path = tmp_path / "memory.db"
  service = SqliteMemoryService(db_path=db_path, fts="off")
  event = _make_event("user", "Stable search text", 1.0)
  await service.add_events_to_memory(
      app_name="app", user_id="user", events=[event], session_id="s1"
  )

  updated_event = Event(
      id=event.id,
      author="user",
      timestamp=5.0,
      content=types.Content(
          role="model", parts=[types.Part(text="Stable search text")]
      ),
  )
  await service.add_events_to_memory(
      app_name="app", user_id="user", events=[updated_event], session_id="s1"
  )

  response = await service.search_memory(
      app_name="app", user_id="user", query="Stable"
  )
  assert len(response.memories) == 1
  assert response.memories[0].content.role == "model"
  await service.close()


@pytest.mark.asyncio
async def test_concurrent_in_memory_initialization(fts_mode):
  """Concurrent operations on an in-memory database share a single connection safely."""
  service = SqliteMemoryService(":memory:", fts=fts_mode)
  event1 = _make_event("user", "first concurrent event", 1.0)
  event2 = _make_event("user", "second concurrent event", 2.0)
  try:
    await asyncio.gather(
        service.add_events_to_memory(
            app_name="app", user_id="user", events=[event1]
        ),
        service.add_events_to_memory(
            app_name="app", user_id="user", events=[event2]
        ),
    )
    resp1 = await service.search_memory(
        app_name="app", user_id="user", query="first"
    )
    resp2 = await service.search_memory(
        app_name="app", user_id="user", query="second"
    )
    assert len(resp1.memories) == 1
    assert len(resp2.memories) == 1
  finally:
    await service.close()


@pytest.mark.asyncio
async def test_search_memory_with_readonly_mode(tmp_path):
  """Opening a database with mode=ro allows searching pre-existing memories."""
  db_path = tmp_path / "memory.db"
  service_rw = SqliteMemoryService(db_path=db_path)
  session = _make_session([_make_event("user", "readonly test entry", 1.0)])
  await service_rw.add_session_to_memory(session)

  ro_uri = f"sqlite:///{db_path}?mode=ro"
  service_ro = SqliteMemoryService(ro_uri)
  resp = await service_ro.search_memory(
      app_name="app", user_id="user", query="readonly"
  )
  assert len(resp.memories) == 1
  assert resp.memories[0].content.parts[0].text == "readonly test entry"


@pytest.mark.asyncio
async def test_search_fts_tied_relevance_ranks_by_timestamp_desc(
    tmp_path, require_fts5
):
  """FTS search breaks BM25 score ties by sorting by timestamp descending."""
  db_path = tmp_path / "memory.db"
  service = SqliteMemoryService(db_path=db_path, fts="on")
  older_event = _make_event(
      "user", "My favorite fruit is apple", 1.0, event_id="e1"
  )
  newer_event = _make_event(
      "user", "My favorite fruit is orange", 2.0, event_id="e2"
  )
  await service.add_events_to_memory(
      app_name="app",
      user_id="user",
      events=[older_event, newer_event],
      session_id="s1",
  )
  response = await service.search_memory(
      app_name="app", user_id="user", query="favorite"
  )
  assert len(response.memories) == 2
  assert response.memories[0].custom_metadata["event_id"] == "e2"
  assert response.memories[1].custom_metadata["event_id"] == "e1"
  await service.close()


@pytest.mark.asyncio
async def test_search_like_fallback_unicode_case_folding(tmp_path):
  """LIKE fallback matches accented characters using unicode case folding."""
  db_path = tmp_path / "memory.db"
  service = SqliteMemoryService(db_path=db_path, fts="off")
  session = _make_session(
      [_make_event("user", "J'aime l'École de musique", 1.0)]
  )
  await service.add_session_to_memory(session)
  response = await service.search_memory(
      app_name="app", user_id="user", query="école"
  )
  assert len(response.memories) == 1
  assert (
      response.memories[0].content.parts[0].text == "J'aime l'École de musique"
  )
  await service.close()


@pytest.mark.asyncio
async def test_readonly_db_with_fts_on_raises_when_table_missing(tmp_path):
  """Opening a read-only database with fts='on' raises when FTS table is missing."""
  db_path = tmp_path / "memory.db"
  service_rw = SqliteMemoryService(db_path=db_path, fts="off")
  session = _make_session([_make_event("user", "hello world", 1.0)])
  await service_rw.add_session_to_memory(session)
  ro_uri = f"sqlite:///{db_path}?mode=ro"
  service_ro = SqliteMemoryService(ro_uri, fts="on")
  with pytest.raises(
      RuntimeError, match="FTS5 table memory_events_fts not found"
  ):
    await service_ro.search_memory(
        app_name="app", user_id="user", query="hello"
    )


@pytest.mark.asyncio
async def test_search_cjk_mixed_latin_query(tmp_path, fts_mode):
  """Search matches mixed CJK and Latin queries across FTS and LIKE modes."""
  db_path = tmp_path / "memory.db"
  service = SqliteMemoryService(db_path=db_path, fts=fts_mode)
  session = _make_session([_make_event("user", "私はPythonを使う", 1.0)])
  await service.add_session_to_memory(session)
  response = await service.search_memory(
      app_name="app", user_id="user", query="Python"
  )
  assert len(response.memories) == 1
  assert response.memories[0].content.parts[0].text == "私はPythonを使う"
  await service.close()


@pytest.mark.asyncio
async def test_incompatible_schema_version_raises(tmp_path):
  """Initializing with an incompatible schema version raises a RuntimeError."""
  db_path = tmp_path / "memory.db"
  service = SqliteMemoryService(db_path=db_path, fts="off")
  session = _make_session([_make_event("user", "hello world", 1.0)])
  await service.add_session_to_memory(session)
  with sqlite3.connect(db_path) as conn:
    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    ]
    assert "adk_memory_metadata" in tables
    assert "adk_internal_metadata" not in tables
    assert "schema_meta" not in tables
    conn.execute(
        "UPDATE adk_memory_metadata SET value = '999' WHERE key ="
        " 'schema_version'"
    )
  service2 = SqliteMemoryService(db_path=db_path, fts="off")
  with pytest.raises(RuntimeError, match="Unsupported schema version: 999"):
    await service2.search_memory(app_name="app", user_id="user", query="hello")


@pytest.mark.asyncio
async def test_shared_db_with_v0_session_schema_preserves_session_migration_check(
    tmp_path,
):
  """Initializing memory service on a shared legacy session database preserves v0 migration detection."""
  db_path = tmp_path / "shared_legacy.db"
  with sqlite3.connect(db_path) as conn:
    conn.execute("""
        CREATE TABLE events (
          id TEXT PRIMARY KEY,
          session_id TEXT NOT NULL,
          app_name TEXT NOT NULL,
          user_id TEXT NOT NULL,
          timestamp REAL NOT NULL,
          actions BLOB NOT NULL
        )
        """)
    conn.commit()

  memory_service = SqliteMemoryService(db_path=db_path, fts="off")
  event = _make_event("user", "memory in shared legacy db", 1.0)
  await memory_service.add_events_to_memory(
      app_name="app", user_id="user", events=[event], session_id="s1"
  )

  with sqlite3.connect(db_path) as conn:
    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    ]
    assert "adk_memory_metadata" in tables
    assert "adk_internal_metadata" not in tables

  version = get_db_schema_version(f"sqlite:///{db_path}")
  assert version == "0"

  session_service = DatabaseSessionService(
      db_url=f"sqlite+aiosqlite:///{db_path}"
  )
  await session_service.prepare_tables()
  assert session_service._db_schema_version == "0"

  resp = await memory_service.search_memory(
      app_name="app", user_id="user", query="memory"
  )
  assert len(resp.memories) == 1
  assert resp.memories[0].content.parts[0].text == "memory in shared legacy db"
