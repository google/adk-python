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
"""Integration tests for SessionService implementations.

This test suite verifies the behavioral contracts defined in BaseSessionService
across different storage backends. It can be used to validate any new
SessionService implementation (e.g., Redis, PostgreSQL, Spanner).

To add a new SessionService implementation:
1. Create a new SessionServiceType enum value
2. Add it to the `params` list in the `session_service` fixture
3. Implement `get_session_service` to return an instance of your service

All tests in this file will automatically run against the new implementation.
"""

from __future__ import annotations

import asyncio
import enum
from typing import Any

from google.adk.errors.already_exists_error import AlreadyExistsError
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.sqlite_session_service import SqliteSessionService
import pytest


APP_NAME = "test_app"
USER_ID = "test_user"


class SessionServiceType(enum.Enum):
    IN_MEMORY = "IN_MEMORY"
    DATABASE = "DATABASE"
    SQLITE = "SQLITE"


def get_session_service(
    service_type: SessionServiceType,
    tmp_path,
) -> BaseSessionService:
    """Creates a fresh session service instance for testing.

    Args:
        service_type: The type of session service to create.
        tmp_path: Pytest's temporary path fixture for file-based storage.

    Returns:
        A new instance of the specified session service.
    """
    if service_type == SessionServiceType.IN_MEMORY:
        return InMemorySessionService()
    if service_type == SessionServiceType.DATABASE:
        return DatabaseSessionService("sqlite+aiosqlite:///:memory:")
    if service_type == SessionServiceType.SQLITE:
        return SqliteSessionService(str(tmp_path / "sessions.db"))
    raise ValueError(f"Unknown service type: {service_type}")


@pytest.fixture(
    params=[
        SessionServiceType.IN_MEMORY,
        SessionServiceType.DATABASE,
        SessionServiceType.SQLITE,
    ]
)
async def session_service(request, tmp_path):
    """Parametrized fixture providing fresh SessionService instances.

    This fixture creates a new session service for each test to ensure
    isolation. For database-backed services, it handles proper cleanup.

    Yields:
        A fresh BaseSessionService instance ready for testing.
    """
    service = get_session_service(request.param, tmp_path)
    try:
        yield service
    finally:
        if isinstance(service, DatabaseSessionService):
            await service.close()


async def _create_event(
    invocation_id: str,
    author: str,
    timestamp: float,
    state_delta: dict[str, Any] | None = None,
) -> Event:
    """Helper to create an Event with consistent parameters."""
    return Event(
        invocation_id=invocation_id,
        author=author,
        timestamp=timestamp,
        actions=EventActions(state_delta=state_delta or {}),
    )


@pytest.mark.asyncio
async def test_multiple_sessions_same_user_are_isolated(session_service):
    """Sessions with different IDs for the same user must be isolated.

    Each session maintains its own independent state and event history.
    Modifying one session must not affect any other session, even if they
    belong to the same user and app.

    Contract verification:
    - Session isolation by (app_name, user_id, session_id) tuple
    - State and events are completely independent between sessions
    """
    session1 = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id="session_1",
        state={"counter": 1, "shared": "initial"},
    )
    session2 = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id="session_2",
        state={"counter": 100, "shared": "initial"},
    )

    event1 = await _create_event("inv1", "agent1", 1000.0, {"counter": 2})
    await session_service.append_event(session1, event1)

    event2 = await _create_event("inv2", "agent2", 2000.0, {"counter": 200})
    await session_service.append_event(session2, event2)

    session1_refreshed = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id="session_1"
    )
    session2_refreshed = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id="session_2"
    )

    assert session1_refreshed is not None
    assert session2_refreshed is not None

    assert session1_refreshed.state.get("counter") == 2
    assert len(session1_refreshed.events) == 1
    assert session1_refreshed.events[0].invocation_id == "inv1"

    assert session2_refreshed.state.get("counter") == 200
    assert len(session2_refreshed.events) == 1
    assert session2_refreshed.events[0].invocation_id == "inv2"


@pytest.mark.asyncio
async def test_state_persists_across_turns(session_service):
    """Session state modifications must persist across multiple invocations.

    When an event modifies the session state via state_delta, those changes
    must be visible in subsequent get_session calls and available for the
    next invocation (turn).

    Contract verification:
    - State updates are persisted after append_event
    - Subsequent get_session calls return the updated state
    """
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id="persistent_session",
        state={"page": 1, "filters": {"category": "all"}},
    )

    assert session.state.get("page") == 1

    event1 = await _create_event(
        "inv1", "agent", 1000.0, {"page": 2, "filters": {"category": "books"}}
    )
    await session_service.append_event(session, event1)

    session_after_turn1 = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id="persistent_session"
    )
    assert session_after_turn1 is not None
    assert session_after_turn1.state.get("page") == 2
    assert session_after_turn1.state.get("filters") == {"category": "books"}

    event2 = await _create_event(
        "inv2", "agent", 2000.0, {"page": 3, "view_mode": "list"}
    )
    await session_service.append_event(session_after_turn1, event2)

    session_after_turn2 = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id="persistent_session"
    )
    assert session_after_turn2 is not None
    assert session_after_turn2.state.get("page") == 3
    assert session_after_turn2.state.get("view_mode") == "list"
    assert session_after_turn2.state.get("filters") == {"category": "books"}


@pytest.mark.asyncio
async def test_events_append_order_and_timestamp_monotonic(session_service):
    """Events must maintain append order and timestamps must be monotonic.

    Events are appended to session.events in the order they are received.
    Each new event's timestamp must be greater than or equal to the previous
    event's timestamp, and the session's last_update_time is updated to the
    event's timestamp after each append.

    Contract verification:
    - Events are appended in order (list index reflects append order)
    - Timestamps are preserved as provided
    - last_update_time is updated to event.timestamp
    """
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id="ordered_events",
    )

    base_ts = 1000.0
    events_to_append = []
    for i in range(5):
        ts = base_ts + float(i) * 100.0
        event = await _create_event(f"inv_{i}", f"agent_{i}", ts, {"seq": i})
        events_to_append.append(event)
        await session_service.append_event(session, event)

    refreshed = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id="ordered_events"
    )
    assert refreshed is not None

    assert len(refreshed.events) == 5

    for i, event in enumerate(refreshed.events):
        assert event.invocation_id == f"inv_{i}"
        assert event.timestamp == base_ts + float(i) * 100.0
        state_delta = event.actions.state_delta
        assert state_delta.get("seq") == i

    for i in range(1, len(refreshed.events)):
        assert refreshed.events[i].timestamp >= refreshed.events[i - 1].timestamp

    assert refreshed.last_update_time == refreshed.events[-1].timestamp


@pytest.mark.asyncio
async def test_concurrent_appends_race_condition(session_service):
    """Concurrent appends to the same session must detect stale sessions.

    When two concurrent operations try to append events to the same session
    using stale session objects (with outdated last_update_time), at most
    one should succeed. The other should raise ValueError indicating the
    session was "modified in storage".

    This prevents lost writes in concurrent scenarios.

    Note: Only DatabaseSessionService provides full concurrency control with
    session-level locking. InMemorySessionService is designed for single-threaded
    use only, and SqliteSessionService relies on SQLite's transaction isolation
    which may not catch all race conditions in this test scenario.

    Contract verification (for implementations that support it):
    - Concurrency control via last_update_time or storage marker
    - Exactly one success, one failure when concurrent writers use stale sessions
    - No lost writes
    """
    if not isinstance(session_service, DatabaseSessionService):
        pytest.skip(
            "This test requires session-level locking for concurrent append "
            "detection, which is only provided by DatabaseSessionService"
        )

    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id="concurrent_session",
    )

    base_ts = session.last_update_time + 100.0

    stale_session_1 = session.model_copy(deep=True)
    stale_session_2 = session.model_copy(deep=True)

    event1 = await _create_event("inv_concurrent_1", "agent_a", base_ts, {"a": 1})
    event2 = await _create_event(
        "inv_concurrent_2", "agent_b", base_ts + 50.0, {"b": 2}
    )

    results = await asyncio.gather(
        session_service.append_event(stale_session_1, event1),
        session_service.append_event(stale_session_2, event2),
        return_exceptions=True,
    )

    errors = [r for r in results if isinstance(r, Exception)]
    successes = [r for r in results if not isinstance(r, Exception)]

    assert len(successes) == 1, "Expected exactly one successful append"
    assert len(errors) == 1, "Expected exactly one failed append"

    assert isinstance(errors[0], ValueError)
    error_msg = str(errors[0]).lower()
    assert "modified" in error_msg or "stale" in error_msg or "storage" in error_msg

    final_session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id="concurrent_session"
    )
    assert final_session is not None
    assert len(final_session.events) == 1

    state = final_session.state
    has_a = state.get("a") == 1
    has_b = state.get("b") == 2
    assert has_a ^ has_b, "Expected exactly one of the state updates to persist"


@pytest.mark.asyncio
async def test_delete_session_is_idempotent(session_service):
    """Deleting a session must be safe and idempotent.

    - delete_session must not raise an exception for non-existent sessions
    - After deletion, get_session must return None
    - Re-deleting an already-deleted session must not raise

    Contract verification:
    - Idempotent deletion (no exception on missing session)
    - get_session returns None after deletion
    """
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id="to_delete",
        state={"will_be_gone": True},
    )

    event = await _create_event("inv1", "agent", 1000.0, {"extra": "data"})
    await session_service.append_event(session, event)

    before_delete = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id="to_delete"
    )
    assert before_delete is not None

    await session_service.delete_session(
        app_name=APP_NAME, user_id=USER_ID, session_id="to_delete"
    )

    after_delete = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id="to_delete"
    )
    assert after_delete is None, "get_session must return None after delete"

    await session_service.delete_session(
        app_name=APP_NAME, user_id=USER_ID, session_id="to_delete"
    )

    await session_service.delete_session(
        app_name=APP_NAME, user_id=USER_ID, session_id="non_existent"
    )


@pytest.mark.asyncio
async def test_state_delta_merge_semantics(session_service):
    """State updates use merge semantics, not full replacement.

    When appending an event with state_delta, the delta is merged into the
    existing session state. This means:
    - New keys are added to the state
    - Existing keys have their values updated
    - Keys not present in the delta remain unchanged

    This test verifies that state updates do NOT replace the entire state
    dictionary, but instead merge in only the changes.

    Note: Different implementations may have different semantics for nested
    dictionaries. InMemorySessionService and DatabaseSessionService use
    shallow merge (top-level keys only), while SqliteSessionService uses
    RFC 7396 JSON Merge Patch (recursive merge for nested dicts). This test
    focuses on simple values where all implementations behave consistently.

    Contract verification:
    - Keys not in delta are preserved
    - Existing keys are updated with new values
    - New keys are added to the state
    - The entire state dict is NOT replaced
    """
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id="merge_test",
        state={
            "unchanged_key": "original_value",
            "key_to_update": "old_value",
            "counter": 1,
        },
    )

    event1 = await _create_event(
        "inv1",
        "agent",
        1000.0,
        {
            "key_to_update": "new_value",
            "new_key": "just_added",
            "counter": 2,
        },
    )
    await session_service.append_event(session, event1)

    refreshed1 = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id="merge_test"
    )
    assert refreshed1 is not None

    assert refreshed1.state.get("unchanged_key") == "original_value"

    assert refreshed1.state.get("key_to_update") == "new_value"

    assert refreshed1.state.get("new_key") == "just_added"

    assert refreshed1.state.get("counter") == 2

    event2 = await _create_event(
        "inv2",
        "agent",
        2000.0,
        {
            "another_new_key": "added_later",
            "counter": 3,
        },
    )
    await session_service.append_event(refreshed1, event2)

    refreshed2 = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id="merge_test"
    )
    assert refreshed2 is not None

    assert refreshed2.state.get("unchanged_key") == "original_value"
    assert refreshed2.state.get("key_to_update") == "new_value"
    assert refreshed2.state.get("new_key") == "just_added"
    assert refreshed2.state.get("another_new_key") == "added_later"
    assert refreshed2.state.get("counter") == 3


@pytest.mark.asyncio
async def test_create_session_with_existing_id_raises_already_exists(session_service):
    """Creating a session with duplicate ID must raise AlreadyExistsError.

    If create_session is called with a session_id that already exists for
    the same (app_name, user_id), it must raise AlreadyExistsError.

    Contract verification:
    - Atomic creation with duplicate detection
    - AlreadyExistsError raised on duplicate session_id
    """
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id="duplicate_id",
        state={"first": True},
    )

    with pytest.raises(AlreadyExistsError, match="already exists"):
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id="duplicate_id",
            state={"second": True},
        )


@pytest.mark.asyncio
async def test_different_users_same_session_id_are_isolated(session_service):
    """Sessions with same ID but different users must be isolated.

    The session identity is (app_name, user_id, session_id). Two sessions
    with the same session_id but different user_id are completely separate.

    Contract verification:
    - Full tuple identity: (app_name, user_id, session_id)
    - User boundaries are respected
    """
    await session_service.create_session(
        app_name=APP_NAME,
        user_id="user_alice",
        session_id="shared_session_id",
        state={"owner": "alice", "secret": "alice_secret"},
    )

    await session_service.create_session(
        app_name=APP_NAME,
        user_id="user_bob",
        session_id="shared_session_id",
        state={"owner": "bob", "secret": "bob_secret"},
    )

    alice_session = await session_service.get_session(
        app_name=APP_NAME, user_id="user_alice", session_id="shared_session_id"
    )
    bob_session = await session_service.get_session(
        app_name=APP_NAME, user_id="user_bob", session_id="shared_session_id"
    )

    assert alice_session is not None
    assert bob_session is not None

    assert alice_session.state.get("owner") == "alice"
    assert alice_session.state.get("secret") == "alice_secret"

    assert bob_session.state.get("owner") == "bob"
    assert bob_session.state.get("secret") == "bob_secret"


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason="State merge semantics are inconsistent across implementations. "
           "Sqlite uses RFC 7396 recursive merge; InMemory/Database use shallow merge. "
           "See contributing/dev/session_state_merge_semantics.md for details."
)
async def test_nested_state_merge_is_recursive(session_service):
    """Nested dict updates should use RFC 7396 recursive merge semantics.

    This test documents the EXPECTED behavior (RFC 7396 JSON Merge Patch),
    which is currently only implemented by SqliteSessionService.

    RFC 7396 rules:
    - If patch value is null, delete key from target
    - If patch value is dict AND target value is dict → RECURSIVELY MERGE
    - Otherwise → REPLACE

    Bug: InMemorySessionService and DatabaseSessionService use shallow merge
    (dict.update() and dict | operator), which replaces nested dicts entirely.

    Expected behavior (RFC 7396):
    - Update nested.inner_a → nested becomes {inner_a: new, inner_b: preserved}
    - NOT shallow merge where nested = {inner_a: new} (inner_b lost!)

    Contract verification (once fixed):
    - Nested dicts are recursively merged
    - Unspecified nested keys are preserved
    - See RFC 7396 for the standard
    """
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id="nested_merge_test",
        state={
            "top_level": "unchanged",
            "nested": {
                "inner_a": 1,
                "inner_b": 2,
                "inner_c": 3,
            },
        },
    )

    event = await _create_event(
        "inv1",
        "agent",
        1000.0,
        {
            "nested": {
                "inner_a": 100,
                "inner_d": 400,
            },
        },
    )
    await session_service.append_event(session, event)

    refreshed = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id="nested_merge_test"
    )
    assert refreshed is not None

    assert refreshed.state.get("top_level") == "unchanged"

    assert refreshed.state.get("nested") == {
        "inner_a": 100,
        "inner_b": 2,
        "inner_c": 3,
        "inner_d": 400,
    }

    assert "inner_b" in refreshed.state["nested"]
    assert "inner_c" in refreshed.state["nested"]


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason="Concurrent append behavior is implementation-defined. "
           "Only DatabaseSessionService provides guaranteed serialization via locks. "
           "InMemory and Sqlite have undefined behavior under concurrent writes."
)
async def test_concurrent_appends_document_undefined_behavior(session_service):
    """Document the actual behavior of concurrent appends (for debugging).

    This test is marked xfail because:
    - DatabaseSessionService: Uses per-session locks → one success, one error
    - InMemorySessionService: No locking → both may succeed (lost writes possible)
    - SqliteSessionService: Transaction-based → behavior depends on timing

    This test exists to DOCUMENT the current behavior, not to enforce a contract.
    For guaranteed serialization, use DatabaseSessionService.

    Note: See BaseSessionService docstring which states:
    "Subclass may or may not serialize concurrent appends; use DatabaseSessionService
    for guaranteed serialization."
    """
    if isinstance(session_service, DatabaseSessionService):
        pytest.skip("DatabaseSessionService has guaranteed serialization")

    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id="concurrent_behavior",
    )

    base_ts = session.last_update_time + 100.0

    stale_session_1 = session.model_copy(deep=True)
    stale_session_2 = session.model_copy(deep=True)

    event1 = await _create_event("inv_concurrent_1", "agent_a", base_ts, {"a": 1})
    event2 = await _create_event(
        "inv_concurrent_2", "agent_b", base_ts + 50.0, {"b": 2}
    )

    results = await asyncio.gather(
        session_service.append_event(stale_session_1, event1),
        session_service.append_event(stale_session_2, event2),
        return_exceptions=True,
    )

    final_session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id="concurrent_behavior"
    )
    assert final_session is not None

    print(f"\n=== Concurrent append observation for {type(session_service).__name__} ===")
    print(f"Results: {[type(r).__name__ if isinstance(r, Exception) else 'SUCCESS' for r in results]}")
    print(f"Final events count: {len(final_session.events)}")
    print(f"Final state: {final_session.state}")

    assert False, (
        f"This test documents behavior, not enforces contract. "
        f"Use DatabaseSessionService for guaranteed serialization."
    )


@pytest.mark.asyncio
async def test_cross_app_same_user_isolation(session_service):
    """Sessions must be isolated across different apps for the same user.

    Session identity is (app_name, user_id, session_id). Two sessions with
    the same user_id and session_id but different app_name must be completely
    isolated.

    This is important for:
    - Multi-tenant applications
    - Different apps sharing the same user base
    - Security boundaries

    Contract verification:
    - Full tuple identity: (app_name, user_id, session_id)
    - App boundaries are respected
    - State and events are isolated
    """
    APP_1 = "shopping_app"
    APP_2 = "banking_app"
    SHARED_USER = "user_123"
    SHARED_SESSION_ID = "session_abc"

    await session_service.create_session(
        app_name=APP_1,
        user_id=SHARED_USER,
        session_id=SHARED_SESSION_ID,
        state={
            "cart": ["item1", "item2"],
            "total": 99.99,
        },
    )

    await session_service.create_session(
        app_name=APP_2,
        user_id=SHARED_USER,
        session_id=SHARED_SESSION_ID,
        state={
            "balance": 10000.00,
            "account_id": "ACC-001",
        },
    )

    shopping_session = await session_service.get_session(
        app_name=APP_1, user_id=SHARED_USER, session_id=SHARED_SESSION_ID
    )
    banking_session = await session_service.get_session(
        app_name=APP_2, user_id=SHARED_USER, session_id=SHARED_SESSION_ID
    )

    assert shopping_session is not None
    assert banking_session is not None

    assert shopping_session.app_name == APP_1
    assert shopping_session.state.get("cart") == ["item1", "item2"]
    assert shopping_session.state.get("total") == 99.99
    assert "balance" not in shopping_session.state
    assert "account_id" not in shopping_session.state

    assert banking_session.app_name == APP_2
    assert banking_session.state.get("balance") == 10000.00
    assert banking_session.state.get("account_id") == "ACC-001"
    assert "cart" not in banking_session.state
    assert "total" not in banking_session.state


@pytest.mark.asyncio
async def test_cross_user_same_session_id_isolation(session_service):
    """Sessions must be isolated across different users with the same session_id.

    This is a stricter version of test_different_users_same_session_id_are_isolated,
    also testing that list_sessions respects user boundaries.

    Contract verification:
    - list_sessions with user_id filter only returns that user's sessions
    - list_sessions without user_id returns all sessions for the app
    - State isolation is maintained
    """
    SESSION_ID = "shared_session_001"

    await session_service.create_session(
        app_name=APP_NAME,
        user_id="alice",
        session_id=SESSION_ID,
        state={"role": "admin", "permissions": ["read", "write", "delete"]},
    )

    await session_service.create_session(
        app_name=APP_NAME,
        user_id="bob",
        session_id=SESSION_ID,
        state={"role": "guest", "permissions": ["read"]},
    )

    await session_service.create_session(
        app_name=APP_NAME,
        user_id="charlie",
        session_id=SESSION_ID,
        state={"role": "editor", "permissions": ["read", "write"]},
    )

    alice_list = await session_service.list_sessions(
        app_name=APP_NAME, user_id="alice"
    )
    assert len(alice_list.sessions) == 1
    assert alice_list.sessions[0].user_id == "alice"
    assert alice_list.sessions[0].state.get("role") == "admin"

    all_sessions = await session_service.list_sessions(
        app_name=APP_NAME, user_id=None
    )
    assert len(all_sessions.sessions) == 3

    users_found = {s.user_id for s in all_sessions.sessions}
    assert users_found == {"alice", "bob", "charlie"}

    alice_session = await session_service.get_session(
        app_name=APP_NAME, user_id="alice", session_id=SESSION_ID
    )
    bob_session = await session_service.get_session(
        app_name=APP_NAME, user_id="bob", session_id=SESSION_ID
    )
    charlie_session = await session_service.get_session(
        app_name=APP_NAME, user_id="charlie", session_id=SESSION_ID
    )

    assert alice_session.state.get("permissions") == ["read", "write", "delete"]
    assert bob_session.state.get("permissions") == ["read"]
    assert charlie_session.state.get("permissions") == ["read", "write"]


@pytest.mark.asyncio
async def test_returned_session_is_deep_copy_not_reference(session_service):
    """Sessions returned from get_session must be copies, not references.

    Modifying a returned session object should NOT affect:
    1. Subsequent get_session calls (persisted state)
    2. The in-memory state of the service (if any)

    This is a critical safety feature to prevent:
    - Accidental modifications from affecting persisted state
    - Race conditions between concurrent readers
    - Side effects in caller code

    Contract verification:
    - get_session returns a copy (not reference to internal storage)
    - Modifying returned session doesn't affect storage
    - list_sessions also returns copies
    """
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id="copy_test",
        state={
            "counter": 0,
            "config": {"theme": "light", "notifications": True},
        },
    )

    read_1 = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id="copy_test"
    )
    assert read_1 is not None

    read_1.state["counter"] = 999
    read_1.state["config"]["theme"] = "dark"
    read_1.state["new_field"] = "accidentally_added"

    read_2 = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id="copy_test"
    )
    assert read_2 is not None

    assert read_2.state.get("counter") == 0
    assert read_2.state.get("config") == {"theme": "light", "notifications": True}
    assert "new_field" not in read_2.state

    list_result = await session_service.list_sessions(
        app_name=APP_NAME, user_id=USER_ID
    )
    assert len(list_result.sessions) == 1

    listed_session = list_result.sessions[0]
    listed_session.state["counter"] = 888

    read_3 = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id="copy_test"
    )
    assert read_3 is not None
    assert read_3.state.get("counter") == 0
