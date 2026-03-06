"""Tests for StorageEvent serialization with non-serializable types.

Regression test for https://github.com/google/adk-python/issues/4724
"""

import time

import pytest

from google.adk.events.event import Event
from google.adk.sessions import Session
from google.adk.sessions.schemas.v1 import StorageEvent


def _make_session() -> Session:
    return Session(
        app_name="test-app",
        user_id="test-user",
        id="test-session",
        state={},
    )


def _make_event(**kwargs) -> Event:
    defaults = dict(
        invocation_id="inv-1",
        author="agent",
        timestamp=time.time(),
    )
    defaults.update(kwargs)
    return Event(**defaults)


class TestStorageEventSerialization:
    """Test that StorageEvent.from_event handles non-serializable types."""

    def test_basic_event_roundtrip(self):
        """Normal events should serialize and deserialize correctly."""
        session = _make_session()
        event = _make_event()
        storage = StorageEvent.from_event(session, event)
        assert storage.id == event.id
        assert storage.session_id == session.id

    def test_event_with_function_in_state_delta(self):
        """Events with function objects in state_delta should not crash.

        This is the core regression test for #4724: when tools attach
        non-serializable function references to events, model_dump()
        should gracefully degrade instead of raising
        PydanticSerializationError.
        """
        session = _make_session()
        event = _make_event()
        # Simulate a function object being attached to state_delta
        # (this happens when MCP tools resolve their function references)
        event.actions.state_delta["callback"] = lambda x: x

        # This should NOT raise PydanticSerializationError
        storage = StorageEvent.from_event(session, event)
        assert storage.event_data is not None
        # The function should be serialized as a placeholder string
        actions = storage.event_data.get("actions", {})
        state_delta = actions.get("state_delta", actions.get("stateDelta", {}))
        assert "non-serializable" in str(state_delta.get("callback", ""))

    def test_roundtrip_preserves_serializable_fields(self):
        """Non-serializable fields are replaced but other fields survive."""
        session = _make_session()
        event = _make_event()
        event.actions.state_delta["normal_key"] = "normal_value"
        event.actions.state_delta["func_key"] = lambda: None

        storage = StorageEvent.from_event(session, event)
        restored = storage.to_event()

        assert restored.actions.state_delta["normal_key"] == "normal_value"
        assert "non-serializable" in str(restored.actions.state_delta.get("func_key", ""))
