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

"""Integration tests for on_run_error_callback in the Runner lifecycle.

These tests verify that:
  - on_run_error_callback is called when the execution generator raises.
  - after_run_callback is NOT called on the error path.
  - The original exception is re-raised unchanged.
  - on_run_error_callback receives the correct invocation_context and error.
  - on_run_error_callback is NOT called on successful runs.
"""

from __future__ import annotations

from typing import AsyncGenerator
from typing import ClassVar
from unittest.mock import AsyncMock
from unittest.mock import Mock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.plugins.plugin_manager import PluginManager
from google.adk.runners import InMemoryRunner
from google.genai import types
import pytest
from typing_extensions import override

from .. import testing_utils


# ---------------------------------------------------------------------------
# Concrete agent implementations for testing
# ---------------------------------------------------------------------------

class _SuccessAgent(BaseAgent):
    """Agent that yields one event then completes successfully."""

    def __init__(self):
        super().__init__(name="success_agent")

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=types.Content(
                parts=[types.Part.from_text(text="ok")]
            ),
        )


class _FailingAgent(BaseAgent):
    """Agent that raises a RuntimeError during execution."""

    BOOM: ClassVar[RuntimeError] = RuntimeError("agent exploded mid-run")

    def __init__(self):
        super().__init__(name="failing_agent")

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        raise _FailingAgent.BOOM
        yield  # make this an async generator


# ---------------------------------------------------------------------------
# Tracking plugin
# ---------------------------------------------------------------------------

class TrackingPlugin(BasePlugin):
    """Records lifecycle callback invocations for assertions."""

    __test__ = False

    def __init__(self, name: str = "tracker"):
        super().__init__(name)
        self.after_run_called = False
        self.run_error_calls: list[dict] = []

    async def after_run_callback(self, *, invocation_context, **kwargs) -> None:
        self.after_run_called = True

    async def on_run_error_callback(self, *, invocation_context, error, **kwargs) -> None:
        self.run_error_calls.append(
            {"invocation_context": invocation_context, "error": error}
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunnerOnRunErrorCallback:

    def _make_runner(self, agent: BaseAgent, plugins: list[BasePlugin]):
        return InMemoryRunner(agent=agent, plugins=plugins)

    @pytest.mark.asyncio
    async def test_on_run_error_callback_called_when_agent_raises(self):
        """on_run_error_callback must fire when the agent execution raises."""
        tracker = TrackingPlugin()
        runner = self._make_runner(_FailingAgent(), [tracker])

        session = await runner.session_service.create_session(
            app_name=runner.app_name, user_id="u1"
        )
        user_msg = types.Content(
            parts=[types.Part.from_text(text="hello")], role="user"
        )

        with pytest.raises(RuntimeError, match="agent exploded mid-run"):
            async for _ in runner.run_async(
                user_id="u1",
                session_id=session.id,
                new_message=user_msg,
            ):
                pass

        assert len(tracker.run_error_calls) == 1
        assert tracker.run_error_calls[0]["error"] is _FailingAgent.BOOM

    @pytest.mark.asyncio
    async def test_after_run_callback_not_called_on_error(self):
        """after_run_callback must NOT be called when execution raises."""
        tracker = TrackingPlugin()
        runner = self._make_runner(_FailingAgent(), [tracker])

        session = await runner.session_service.create_session(
            app_name=runner.app_name, user_id="u2"
        )
        user_msg = types.Content(
            parts=[types.Part.from_text(text="hello")], role="user"
        )

        with pytest.raises(RuntimeError):
            async for _ in runner.run_async(
                user_id="u2",
                session_id=session.id,
                new_message=user_msg,
            ):
                pass

        assert not tracker.after_run_called

    @pytest.mark.asyncio
    async def test_original_exception_is_reraised_unchanged(self):
        """The exact original exception must propagate to the caller."""
        tracker = TrackingPlugin()
        runner = self._make_runner(_FailingAgent(), [tracker])

        session = await runner.session_service.create_session(
            app_name=runner.app_name, user_id="u3"
        )
        user_msg = types.Content(
            parts=[types.Part.from_text(text="hello")], role="user"
        )

        with pytest.raises(RuntimeError) as exc_info:
            async for _ in runner.run_async(
                user_id="u3",
                session_id=session.id,
                new_message=user_msg,
            ):
                pass

        assert exc_info.value is _FailingAgent.BOOM

    @pytest.mark.asyncio
    async def test_on_run_error_callback_receives_correct_invocation_context(self):
        """The invocation_context passed to on_run_error_callback must be the
        same one used for the run."""
        tracker = TrackingPlugin()
        runner = self._make_runner(_FailingAgent(), [tracker])

        session = await runner.session_service.create_session(
            app_name=runner.app_name, user_id="u4"
        )
        user_msg = types.Content(
            parts=[types.Part.from_text(text="hello")], role="user"
        )

        with pytest.raises(RuntimeError):
            async for _ in runner.run_async(
                user_id="u4",
                session_id=session.id,
                new_message=user_msg,
            ):
                pass

        inv_ctx = tracker.run_error_calls[0]["invocation_context"]
        assert isinstance(inv_ctx, InvocationContext)
        assert inv_ctx.session.id == session.id

    @pytest.mark.asyncio
    async def test_on_run_error_callback_not_called_on_success(self):
        """on_run_error_callback must NOT be called for a successful run."""
        tracker = TrackingPlugin()
        runner = self._make_runner(_SuccessAgent(), [tracker])

        session = await runner.session_service.create_session(
            app_name=runner.app_name, user_id="u5"
        )
        user_msg = types.Content(
            parts=[types.Part.from_text(text="hello")], role="user"
        )

        async for _ in runner.run_async(
            user_id="u5",
            session_id=session.id,
            new_message=user_msg,
        ):
            pass

        assert len(tracker.run_error_calls) == 0

    @pytest.mark.asyncio
    async def test_after_run_callback_called_on_success(self):
        """after_run_callback must still be called for a successful run."""
        tracker = TrackingPlugin()
        runner = self._make_runner(_SuccessAgent(), [tracker])

        session = await runner.session_service.create_session(
            app_name=runner.app_name, user_id="u6"
        )
        user_msg = types.Content(
            parts=[types.Part.from_text(text="hello")], role="user"
        )

        async for _ in runner.run_async(
            user_id="u6",
            session_id=session.id,
            new_message=user_msg,
        ):
            pass

        assert tracker.after_run_called

    @pytest.mark.asyncio
    async def test_multiple_plugins_all_notified_on_error(self):
        """All registered plugins must receive on_run_error_callback on failure."""
        tracker_a = TrackingPlugin("tracker_a")
        tracker_b = TrackingPlugin("tracker_b")
        runner = self._make_runner(_FailingAgent(), [tracker_a, tracker_b])

        session = await runner.session_service.create_session(
            app_name=runner.app_name, user_id="u7"
        )
        user_msg = types.Content(
            parts=[types.Part.from_text(text="hello")], role="user"
        )

        with pytest.raises(RuntimeError):
            async for _ in runner.run_async(
                user_id="u7",
                session_id=session.id,
                new_message=user_msg,
            ):
                pass

        assert len(tracker_a.run_error_calls) == 1
        assert len(tracker_b.run_error_calls) == 1
