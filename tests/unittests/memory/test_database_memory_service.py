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

"""Unit tests for DatabaseMemoryService."""

from __future__ import annotations

from collections.abc import Sequence
import time
from typing import Any
from unittest.mock import MagicMock

from google.adk.events.event import Event
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.database_memory_service import DatabaseMemoryService
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.memory.memory_search_backend import MemorySearchBackend
from google.adk.memory.schemas.memory_schema import StorageMemoryEntry
from google.adk.sessions.session import Session
from google.genai import types
import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DB_URL = "sqlite+aiosqlite:///:memory:"
_APP = "test_app"
_USER = "user_1"
_SESSION = "session_1"


def _make_content(text: str) -> types.Content:
  return types.Content(role="user", parts=[types.Part(text=text)])


def _make_event(
    text: str, event_id: str = "ev1", author: str = "user"
) -> Event:
  return Event(
      id=event_id,
      author=author,
      content=_make_content(text),
      timestamp=time.time(),
      invocation_id="inv1",
  )


def _make_session(events: list[Event], session_id: str = _SESSION) -> Session:
  return Session(
      id=session_id,
      app_name=_APP,
      user_id=_USER,
      events=events,
  )


@pytest.fixture
def svc() -> DatabaseMemoryService:
  return DatabaseMemoryService(_DB_URL)


# ---------------------------------------------------------------------------
# 1. add_session_to_memory — filters empty events, persists content/author/ts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_session_to_memory_persists_text_events(svc):
  session = _make_session([_make_event("hello world")])
  await svc.add_session_to_memory(session)

  resp = await svc.search_memory(app_name=_APP, user_id=_USER, query="hello")
  assert len(resp.memories) == 1
  assert resp.memories[0].author == "user"
  assert resp.memories[0].timestamp is not None


@pytest.mark.asyncio
async def test_add_session_to_memory_skips_empty_events(svc):
  empty_event = Event(
      id="empty",
      author="user",
      content=types.Content(role="user", parts=[]),
      timestamp=time.time(),
      invocation_id="inv1",
  )
  session = _make_session([empty_event])
  await svc.add_session_to_memory(session)

  resp = await svc.search_memory(app_name=_APP, user_id=_USER, query="anything")
  assert resp.memories == []


# ---------------------------------------------------------------------------
# 2. Re-ingest same session → idempotent (no duplicates)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_session_to_memory_idempotent(svc):
  session = _make_session([_make_event("idempotent test")])
  await svc.add_session_to_memory(session)
  await svc.add_session_to_memory(session)

  resp = await svc.search_memory(
      app_name=_APP, user_id=_USER, query="idempotent"
  )
  assert len(resp.memories) == 1


# ---------------------------------------------------------------------------
# 3. add_events_to_memory — delta, skips duplicate event_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_events_to_memory_delta(svc):
  ev = _make_event("delta event", event_id="ev_delta")
  await svc.add_events_to_memory(
      app_name=_APP,
      user_id=_USER,
      events=[ev],
      session_id=_SESSION,
  )
  # Second call with same event_id should not create a duplicate
  await svc.add_events_to_memory(
      app_name=_APP,
      user_id=_USER,
      events=[ev],
      session_id=_SESSION,
  )

  resp = await svc.search_memory(app_name=_APP, user_id=_USER, query="delta")
  assert len(resp.memories) == 1


@pytest.mark.asyncio
async def test_add_events_to_memory_skips_empty(svc):
  empty = Event(
      id="empty2",
      author="agent",
      content=types.Content(role="model", parts=[]),
      timestamp=time.time(),
      invocation_id="inv1",
  )
  await svc.add_events_to_memory(
      app_name=_APP, user_id=_USER, events=[empty], session_id=_SESSION
  )
  resp = await svc.search_memory(app_name=_APP, user_id=_USER, query="anything")
  assert resp.memories == []


# ---------------------------------------------------------------------------
# 4. add_memory — direct MemoryEntry persist, auto-UUID
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_memory_direct(svc):
  entry = MemoryEntry(
      content=_make_content("direct memory fact"),
      author="system",
  )
  await svc.add_memory(app_name=_APP, user_id=_USER, memories=[entry])

  resp = await svc.search_memory(app_name=_APP, user_id=_USER, query="direct")
  assert len(resp.memories) == 1
  assert resp.memories[0].author == "system"
  assert resp.memories[0].id is not None  # auto-UUID


@pytest.mark.asyncio
async def test_add_memory_preserves_explicit_id(svc):
  entry = MemoryEntry(
      id="explicit-id-123",
      content=_make_content("explicit id memory"),
  )
  await svc.add_memory(app_name=_APP, user_id=_USER, memories=[entry])
  resp = await svc.search_memory(app_name=_APP, user_id=_USER, query="explicit")
  assert resp.memories[0].id == "explicit-id-123"


# ---------------------------------------------------------------------------
# 5. search_memory — AND match, OR fallback, no results for empty query
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_and_match(svc):
  await svc.add_memory(
      app_name=_APP,
      user_id=_USER,
      memories=[MemoryEntry(content=_make_content("cats and dogs"))],
  )
  # Both tokens present → AND match
  resp = await svc.search_memory(
      app_name=_APP, user_id=_USER, query="cats dogs"
  )
  assert len(resp.memories) == 1


@pytest.mark.asyncio
async def test_search_or_fallback(svc):
  await svc.add_memory(
      app_name=_APP,
      user_id=_USER,
      memories=[MemoryEntry(content=_make_content("cats are great"))],
  )
  # Only one token matches → OR fallback should still find it
  resp = await svc.search_memory(
      app_name=_APP, user_id=_USER, query="cats fish"
  )
  assert len(resp.memories) == 1


@pytest.mark.asyncio
async def test_search_empty_query_returns_empty(svc):
  await svc.add_memory(
      app_name=_APP,
      user_id=_USER,
      memories=[MemoryEntry(content=_make_content("something"))],
  )
  resp = await svc.search_memory(app_name=_APP, user_id=_USER, query="")
  assert resp.memories == []


@pytest.mark.asyncio
async def test_search_no_match(svc):
  await svc.add_memory(
      app_name=_APP,
      user_id=_USER,
      memories=[MemoryEntry(content=_make_content("hello world"))],
  )
  resp = await svc.search_memory(
      app_name=_APP, user_id=_USER, query="zzznomatch"
  )
  assert resp.memories == []


# ---------------------------------------------------------------------------
# 6. Scratchpad KV: set/get/overwrite/delete/list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scratchpad_kv_set_get(svc):
  await svc.set_scratchpad(
      app_name=_APP, user_id=_USER, session_id=_SESSION, key="k1", value="v1"
  )
  val = await svc.get_scratchpad(
      app_name=_APP, user_id=_USER, session_id=_SESSION, key="k1"
  )
  assert val == "v1"


@pytest.mark.asyncio
async def test_scratchpad_kv_overwrite(svc):
  await svc.set_scratchpad(
      app_name=_APP, user_id=_USER, session_id=_SESSION, key="k2", value="old"
  )
  await svc.set_scratchpad(
      app_name=_APP, user_id=_USER, session_id=_SESSION, key="k2", value="new"
  )
  val = await svc.get_scratchpad(
      app_name=_APP, user_id=_USER, session_id=_SESSION, key="k2"
  )
  assert val == "new"


@pytest.mark.asyncio
async def test_scratchpad_kv_missing_returns_none(svc):
  val = await svc.get_scratchpad(
      app_name=_APP, user_id=_USER, session_id=_SESSION, key="nonexistent"
  )
  assert val is None


@pytest.mark.asyncio
async def test_scratchpad_kv_delete(svc):
  await svc.set_scratchpad(
      app_name=_APP, user_id=_USER, session_id=_SESSION, key="k3", value="v3"
  )
  await svc.delete_scratchpad(
      app_name=_APP, user_id=_USER, session_id=_SESSION, key="k3"
  )
  val = await svc.get_scratchpad(
      app_name=_APP, user_id=_USER, session_id=_SESSION, key="k3"
  )
  assert val is None


@pytest.mark.asyncio
async def test_scratchpad_kv_list_keys(svc):
  for k in ("a", "b", "c"):
    await svc.set_scratchpad(
        app_name=_APP,
        user_id=_USER,
        session_id=_SESSION,
        key=k,
        value=k,
    )
  keys = await svc.list_scratchpad_keys(
      app_name=_APP, user_id=_USER, session_id=_SESSION
  )
  assert set(keys) == {"a", "b", "c"}


@pytest.mark.asyncio
async def test_scratchpad_kv_json_types(svc):
  payload = {"nested": [1, 2, 3], "flag": True}
  await svc.set_scratchpad(
      app_name=_APP,
      user_id=_USER,
      session_id=_SESSION,
      key="json_key",
      value=payload,
  )
  val = await svc.get_scratchpad(
      app_name=_APP, user_id=_USER, session_id=_SESSION, key="json_key"
  )
  assert val == payload


# ---------------------------------------------------------------------------
# 7. Scratchpad log: append/get, filter by tag, limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scratchpad_log_append_get(svc):
  await svc.append_log(
      app_name=_APP, user_id=_USER, session_id=_SESSION, content="entry 1"
  )
  await svc.append_log(
      app_name=_APP, user_id=_USER, session_id=_SESSION, content="entry 2"
  )
  entries = await svc.get_log(app_name=_APP, user_id=_USER, session_id=_SESSION)
  assert len(entries) == 2
  assert entries[0]["content"] == "entry 1"
  assert entries[1]["content"] == "entry 2"


@pytest.mark.asyncio
async def test_scratchpad_log_filter_by_tag(svc):
  await svc.append_log(
      app_name=_APP,
      user_id=_USER,
      session_id=_SESSION,
      content="tagged",
      tag="mytag",
  )
  await svc.append_log(
      app_name=_APP, user_id=_USER, session_id=_SESSION, content="untagged"
  )
  tagged = await svc.get_log(
      app_name=_APP, user_id=_USER, session_id=_SESSION, tag="mytag"
  )
  assert len(tagged) == 1
  assert tagged[0]["content"] == "tagged"


@pytest.mark.asyncio
async def test_scratchpad_log_limit(svc):
  for i in range(10):
    await svc.append_log(
        app_name=_APP,
        user_id=_USER,
        session_id=_SESSION,
        content=f"msg {i}",
    )
  entries = await svc.get_log(
      app_name=_APP, user_id=_USER, session_id=_SESSION, limit=3
  )
  assert len(entries) == 3


# ---------------------------------------------------------------------------
# 8. Custom search backend
# ---------------------------------------------------------------------------


class _AlwaysReturnOneBackend(MemorySearchBackend):
  """Stub backend that always returns a single hard-coded row."""

  async def search(
      self,
      *,
      sql_session,
      app_name,
      user_id,
      query,
      limit=10,
  ) -> Sequence[StorageMemoryEntry]:
    row = StorageMemoryEntry(
        id="stub-id",
        app_name=app_name,
        user_id=user_id,
        content_json={"role": "user", "parts": [{"text": "stub result"}]},
        author="stub",
        timestamp=None,
        custom_metadata={},
    )
    return [row]


@pytest.mark.asyncio
async def test_custom_search_backend():
  svc = DatabaseMemoryService(_DB_URL, search_backend=_AlwaysReturnOneBackend())
  resp = await svc.search_memory(app_name=_APP, user_id=_USER, query="anything")
  assert len(resp.memories) == 1
  assert resp.memories[0].id == "stub-id"
  assert resp.memories[0].author == "stub"


# ---------------------------------------------------------------------------
# 9. Engine construction errors raise ValueError
# ---------------------------------------------------------------------------


def test_bad_url_raises_value_error():
  with pytest.raises(ValueError, match="Invalid database URL"):
    DatabaseMemoryService("not_a_valid_url://")


def test_missing_driver_raises_value_error():
  with pytest.raises(ValueError):
    # Use a driver that definitely isn't installed
    DatabaseMemoryService("sqlite+nonexistentdriver:///:memory:")


# ---------------------------------------------------------------------------
# 10. Multi-user isolation — user A results must not leak to user B
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_user_isolation(svc):
  """User B's search must not return user A's memories even on matching text."""
  await svc.add_memory(
      app_name=_APP,
      user_id="user_a",
      memories=[MemoryEntry(content=_make_content("secret data alpha"))],
  )

  # User B has no memories; searching the same text should yield nothing.
  resp = await svc.search_memory(
      app_name=_APP, user_id="user_b", query="secret"
  )
  assert resp.memories == [], "User B should not see user A's memories"


@pytest.mark.asyncio
async def test_add_session_user_isolation(svc):
  """Session ingestion from user A must not appear in user B's search."""
  session_a = Session(
      id="sess_a",
      app_name=_APP,
      user_id="user_a",
      events=[_make_event("shared keyword")],
  )
  await svc.add_session_to_memory(session_a)

  resp = await svc.search_memory(
      app_name=_APP, user_id="user_b", query="shared"
  )
  assert resp.memories == []


# ---------------------------------------------------------------------------
# 11. Scratchpad KV scoping — session A key invisible in session B or user-level
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scratchpad_kv_session_scoping(svc):
  """A key written for s1 must not be visible under s2 or the user-level scope."""
  await svc.set_scratchpad(
      app_name=_APP, user_id=_USER, session_id="s1", key="scoped", value="yes"
  )

  val_s2 = await svc.get_scratchpad(
      app_name=_APP, user_id=_USER, session_id="s2", key="scoped"
  )
  assert val_s2 is None, "Key from s1 must not appear in s2"

  val_user = await svc.get_scratchpad(
      app_name=_APP, user_id=_USER, session_id="", key="scoped"
  )
  assert val_user is None, "Key from s1 must not appear in user-level scope"


@pytest.mark.asyncio
async def test_scratchpad_log_session_scoping(svc):
  """Log entries appended to s1 must not appear when querying s2."""
  await svc.append_log(
      app_name=_APP,
      user_id=_USER,
      session_id="s1",
      content="session-one log",
  )

  entries = await svc.get_log(app_name=_APP, user_id=_USER, session_id="s2")
  assert entries == [], "Log from s1 must not appear in s2"


# ---------------------------------------------------------------------------
# 12. add_memory with custom_metadata — verify merge with entry.custom_metadata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_memory_custom_metadata_merge(svc):
  """custom_metadata passed to add_memory should merge with entry.custom_metadata."""
  entry = MemoryEntry(
      content=_make_content("metadata test"),
      author="agent",
      custom_metadata={"entry_key": "entry_val"},
  )
  await svc.add_memory(
      app_name=_APP,
      user_id=_USER,
      memories=[entry],
      custom_metadata={"call_key": "call_val"},
  )
  resp = await svc.search_memory(app_name=_APP, user_id=_USER, query="metadata")
  assert len(resp.memories) == 1
  meta = resp.memories[0].custom_metadata
  assert (
      meta.get("entry_key") == "entry_val"
  ), "entry.custom_metadata must survive"
  assert (
      meta.get("call_key") == "call_val"
  ), "call-site custom_metadata must survive"


# ---------------------------------------------------------------------------
# 13. delete_scratchpad no-op — deleting a non-existent key must not raise
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scratchpad_delete_noop(svc):
  """Deleting a key that does not exist must be a silent no-op."""
  # Should not raise any exception.
  await svc.delete_scratchpad(
      app_name=_APP, user_id=_USER, session_id=_SESSION, key="ghost"
  )
  val = await svc.get_scratchpad(
      app_name=_APP, user_id=_USER, session_id=_SESSION, key="ghost"
  )
  assert val is None


# ---------------------------------------------------------------------------
# 14. list_scratchpad_keys on empty scope returns []
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scratchpad_list_keys_empty_scope(svc):
  """list_scratchpad_keys on a brand-new scope returns an empty list."""
  keys = await svc.list_scratchpad_keys(
      app_name=_APP, user_id=_USER, session_id="brand_new_session"
  )
  assert keys == []


# ---------------------------------------------------------------------------
# 15. Scratchpad tool tests — all 4 BaseTool subclasses
# ---------------------------------------------------------------------------

# Helper: build a minimal mock ToolContext backed by a real DatabaseMemoryService.


def _make_tool_context(svc: DatabaseMemoryService, session_id: str = _SESSION):
  """Return a ToolContext mock wired to *svc*."""
  session_mock = MagicMock()
  session_mock.user_id = _USER
  session_mock.id = session_id

  ic_mock = MagicMock()
  ic_mock.app_name = _APP
  ic_mock.session = session_mock
  ic_mock.memory_service = svc

  ctx = MagicMock()
  ctx._invocation_context = ic_mock
  ctx.agent_name = "test_agent"
  return ctx


@pytest.mark.asyncio
async def test_scratchpad_set_tool_happy_path(svc):
  from google.adk.tools.scratchpad_tool import ScratchpadSetTool

  tool = ScratchpadSetTool()
  ctx = _make_tool_context(svc)
  result = await tool.run_async(
      args={"key": "tool_key", "value": "tool_value"}, tool_context=ctx
  )
  assert result == "ok"
  # Verify persistence.
  val = await svc.get_scratchpad(
      app_name=_APP, user_id=_USER, session_id=_SESSION, key="tool_key"
  )
  assert val == "tool_value"


@pytest.mark.asyncio
async def test_scratchpad_get_tool_happy_path(svc):
  from google.adk.tools.scratchpad_tool import ScratchpadGetTool

  await svc.set_scratchpad(
      app_name=_APP, user_id=_USER, session_id=_SESSION, key="gt_key", value=42
  )
  tool = ScratchpadGetTool()
  ctx = _make_tool_context(svc)
  val = await tool.run_async(args={"key": "gt_key"}, tool_context=ctx)
  assert val == 42


@pytest.mark.asyncio
async def test_scratchpad_append_log_tool_happy_path(svc):
  from google.adk.tools.scratchpad_tool import ScratchpadAppendLogTool

  tool = ScratchpadAppendLogTool()
  ctx = _make_tool_context(svc)
  result = await tool.run_async(
      args={"content": "observation logged", "tag": "obs"}, tool_context=ctx
  )
  assert result == "ok"
  entries = await svc.get_log(
      app_name=_APP, user_id=_USER, session_id=_SESSION, tag="obs"
  )
  assert len(entries) == 1
  assert entries[0]["content"] == "observation logged"
  assert entries[0]["agent_name"] == "test_agent"


@pytest.mark.asyncio
async def test_scratchpad_get_log_tool_happy_path(svc):
  from google.adk.tools.scratchpad_tool import ScratchpadGetLogTool

  for i in range(5):
    await svc.append_log(
        app_name=_APP,
        user_id=_USER,
        session_id=_SESSION,
        content=f"log {i}",
    )
  tool = ScratchpadGetLogTool()
  ctx = _make_tool_context(svc)
  entries = await tool.run_async(args={"limit": 3}, tool_context=ctx)
  assert len(entries) == 3


# ---------------------------------------------------------------------------
# 15b. Wrong-service-type error for all 4 tools
# ---------------------------------------------------------------------------


def _make_wrong_service_context():
  """Return a ToolContext backed by a plain InMemoryMemoryService."""
  from google.adk.memory.in_memory_memory_service import InMemoryMemoryService

  session_mock = MagicMock()
  session_mock.user_id = _USER
  session_mock.id = _SESSION

  ic_mock = MagicMock()
  ic_mock.app_name = _APP
  ic_mock.session = session_mock
  ic_mock.memory_service = InMemoryMemoryService()

  ctx = MagicMock()
  ctx._invocation_context = ic_mock
  ctx.agent_name = "test_agent"
  return ctx


@pytest.mark.asyncio
async def test_scratchpad_get_tool_wrong_service():
  from google.adk.tools.scratchpad_tool import ScratchpadGetTool

  tool = ScratchpadGetTool()
  with pytest.raises(ValueError, match="DatabaseMemoryService"):
    await tool.run_async(
        args={"key": "x"}, tool_context=_make_wrong_service_context()
    )


@pytest.mark.asyncio
async def test_scratchpad_set_tool_wrong_service():
  from google.adk.tools.scratchpad_tool import ScratchpadSetTool

  tool = ScratchpadSetTool()
  with pytest.raises(ValueError, match="DatabaseMemoryService"):
    await tool.run_async(
        args={"key": "x", "value": 1},
        tool_context=_make_wrong_service_context(),
    )


@pytest.mark.asyncio
async def test_scratchpad_append_log_tool_wrong_service():
  from google.adk.tools.scratchpad_tool import ScratchpadAppendLogTool

  tool = ScratchpadAppendLogTool()
  with pytest.raises(ValueError, match="DatabaseMemoryService"):
    await tool.run_async(
        args={"content": "x"}, tool_context=_make_wrong_service_context()
    )


@pytest.mark.asyncio
async def test_scratchpad_get_log_tool_wrong_service():
  from google.adk.tools.scratchpad_tool import ScratchpadGetLogTool

  tool = ScratchpadGetLogTool()
  with pytest.raises(ValueError, match="DatabaseMemoryService"):
    await tool.run_async(args={}, tool_context=_make_wrong_service_context())
