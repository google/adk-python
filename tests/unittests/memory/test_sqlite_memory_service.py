# Copyright 2025 Google LLC
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

import sqlite3

from google.adk.events.event import Event
from google.adk.memory.sqlite_memory_service import SqliteMemoryService
from google.adk.sessions.session import Session
from google.genai import types
import pytest


def _make_event(author: str, text: str, timestamp: float) -> Event:
  return Event(
      author=author,
      timestamp=timestamp,
      content=types.Content(
          role="user",
          parts=[types.Part(text=text)],
      ),
  )


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
async def test_add_session_is_idempotent(tmp_path, monkeypatch):
  db_path = tmp_path / "memory.db"
  service = SqliteMemoryService(db_path=db_path, fts="off")
  session = _make_session([_make_event("user", "Hello memory", 1.0)])

  monkeypatch.setattr(
      "google.adk.memory.sqlite_memory_service._now_ms", lambda: 1000
  )
  await service.add_session_to_memory(session)

  monkeypatch.setattr(
      "google.adk.memory.sqlite_memory_service._now_ms", lambda: 2000
  )
  await service.add_session_to_memory(session)

  with sqlite3.connect(db_path) as conn:
    row = conn.execute(
        "SELECT COUNT(*), updated_at_ms FROM sessions"
    ).fetchone()
    assert row[0] == 1
    assert row[1] == 1000


@pytest.mark.asyncio
async def test_add_session_updates_when_changed(tmp_path, monkeypatch):
  db_path = tmp_path / "memory.db"
  service = SqliteMemoryService(db_path=db_path, fts="off")
  session = _make_session([_make_event("user", "First event", 1.0)])

  monkeypatch.setattr(
      "google.adk.memory.sqlite_memory_service._now_ms", lambda: 1000
  )
  await service.add_session_to_memory(session)

  session.events.append(_make_event("assistant", "Second event", 2.0))

  monkeypatch.setattr(
      "google.adk.memory.sqlite_memory_service._now_ms", lambda: 2000
  )
  await service.add_session_to_memory(session)

  with sqlite3.connect(db_path) as conn:
    row = conn.execute(
        "SELECT updated_at_ms, search_text FROM sessions"
    ).fetchone()
    assert row[0] == 2000
    assert "Second event" in row[1]


@pytest.mark.asyncio
async def test_persistence_across_restarts(tmp_path):
  db_path = tmp_path / "memory.db"
  service = SqliteMemoryService(db_path=db_path, fts="off")
  session = _make_session([_make_event("user", "Remember me", 1.0)])
  await service.add_session_to_memory(session)

  new_service = SqliteMemoryService(db_path=db_path, fts="off")
  response = await new_service.search_memory(
      app_name="app", user_id="user", query="Remember"
  )
  assert response.memories
  assert response.memories[0].custom_metadata["session_id"] == session.id


@pytest.mark.asyncio
async def test_search_with_fts_when_available(tmp_path):
  db_path = tmp_path / "memory.db"
  service = SqliteMemoryService(db_path=db_path, fts="on")
  session = _make_session([_make_event("user", "FTS sample text", 1.0)])

  try:
    await service.add_session_to_memory(session)
  except RuntimeError:
    pytest.skip("FTS5 not available in this SQLite build.")

  response = await service.search_memory(
      app_name="app", user_id="user", query="sample"
  )
  assert response.memories
