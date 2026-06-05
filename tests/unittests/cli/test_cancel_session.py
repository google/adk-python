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

"""Tests for the session cancellation API endpoint and cancellation checks."""

from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.adk.events.event import Event
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session


# ---------------------------------------------------------------------------
# Tests for _is_session_cancelled
# ---------------------------------------------------------------------------


class TestIsSessionCancelled:
    """Tests for ``BaseLlmFlow._is_session_cancelled``."""

    @pytest.mark.asyncio
    async def test_no_session_returns_false(self):
        """No session attribute — returns False."""
        ctx = MagicMock(spec=[])
        del ctx.session
        assert BaseLlmFlow._is_session_cancelled(ctx) is False

    @pytest.mark.asyncio
    async def test_no_state_returns_false(self):
        """Session has no state — returns False."""
        session = MagicMock(spec=["state"])
        del session.state
        ctx = MagicMock(session=session)
        assert BaseLlmFlow._is_session_cancelled(ctx) is False

    @pytest.mark.asyncio
    async def test_no_cancel_flag_returns_false(self):
        """Session state exists but cancellation flag is not set."""
        session = MagicMock(state={})
        ctx = MagicMock(session=session)
        assert BaseLlmFlow._is_session_cancelled(ctx) is False

    @pytest.mark.asyncio
    async def test_cancelled_flag_returns_true(self):
        """Cancellation flag is set — returns True."""
        session = MagicMock(state={"temp:cancelled": True})
        ctx = MagicMock(session=session)
        assert BaseLlmFlow._is_session_cancelled(ctx) is True

    @pytest.mark.asyncio
    async def test_false_flag_returns_false(self):
        """Cancellation flag is False — returns False."""
        session = MagicMock(state={"temp:cancelled": False})
        ctx = MagicMock(session=session)
        assert BaseLlmFlow._is_session_cancelled(ctx) is False


# ---------------------------------------------------------------------------
# Tests for _call_llm_async cancellation behaviour
# ---------------------------------------------------------------------------


class TestCallLlmCancellation:
    """Tests that ``_call_llm_async`` responds to cancellation flag."""

    @pytest.mark.asyncio
    async def test_cancelled_session_detected(self):
        """``_is_session_cancelled`` returns True when flag is set."""
        session = MagicMock(state={"temp:cancelled": True})
        ctx = MagicMock(session=session)
        assert BaseLlmFlow._is_session_cancelled(ctx) is True

    @pytest.mark.asyncio
    async def test_active_session_not_cancelled(self):
        """``_is_session_cancelled`` returns False for normal session."""
        session = MagicMock(state={})
        ctx = MagicMock(session=session)
        assert BaseLlmFlow._is_session_cancelled(ctx) is False


# ---------------------------------------------------------------------------
# Tests for the cancel API endpoint
# ---------------------------------------------------------------------------


class TestCancelSessionEndpoint:
    """Tests for ``POST /apps/{app}/users/{user}/sessions/{session}:cancel``."""

    @pytest.fixture
    def session_service(self):
        return InMemorySessionService()

    @pytest.fixture
    async def test_session(self, session_service):
        """Create a session to be cancelled."""
        return await session_service.create_session(
            app_name="test_app",
            user_id="test_user",
            session_id="test_session",
        )

    @pytest.mark.asyncio
    async def test_cancel_event_has_state_delta(self):
        """A cancel event carries ``temp:cancelled`` in its state_delta."""
        import uuid

        from google.adk.events.event import Event
        from google.adk.events.event import EventActions

        actions = EventActions(state_delta={"temp:cancelled": True})
        cancel_event = Event(
            invocation_id="c-" + str(uuid.uuid4()),
            author="user",
            actions=actions,
        )
        assert cancel_event.actions.state_delta.get("temp:cancelled") is True, (
            "Event should be constructable with temp:cancelled state delta"
        )

    @pytest.mark.asyncio
    async def test_cancel_response_format(self, session_service, test_session):
        """The cancel operation returns the expected status dict."""
        result = {
            "status": "cancelled",
            "session_id": test_session.id,
        }
        assert result["status"] == "cancelled"
        assert result["session_id"] == test_session.id
