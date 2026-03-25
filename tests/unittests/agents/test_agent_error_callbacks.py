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

"""Integration tests for on_agent_error_callback in BaseAgent lifecycle.

These tests verify that:
  - on_agent_error_callback is called when _run_async_impl raises.
  - on_agent_error_callback is called when _run_live_impl raises.
  - after_agent_callback is NOT called on the error path.
  - The original exception is re-raised unchanged after notification.
  - on_agent_error_callback receives the agent, callback_context, and error.
  - on_agent_error_callback is NOT called on a successful run.
"""

from __future__ import annotations

from typing import AsyncGenerator
from typing import ClassVar
from unittest.mock import Mock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.plugins.plugin_manager import PluginManager
from google.genai import types
import pytest
from typing_extensions import override

from .. import testing_utils

# ---------------------------------------------------------------------------
# Concrete agent implementations
# ---------------------------------------------------------------------------


class _SuccessAgent(BaseAgent):

  def __init__(self):
    super().__init__(name="success_agent")

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield Event(
        invocation_id=ctx.invocation_id,
        author=self.name,
        content=types.Content(parts=[types.Part.from_text(text="done")]),
    )


class _FailingAgent(BaseAgent):
  BOOM: ClassVar[RuntimeError] = RuntimeError("agent impl exploded")

  def __init__(self):
    super().__init__(name="failing_agent")

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    raise _FailingAgent.BOOM
    yield  # pragma: no cover


class _FailingLiveAgent(BaseAgent):
  BOOM: ClassVar[RuntimeError] = RuntimeError("live agent impl exploded")

  def __init__(self):
    super().__init__(name="failing_live_agent")

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield  # pragma: no cover

  @override
  async def _run_live_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    raise _FailingLiveAgent.BOOM
    yield  # pragma: no cover


# ---------------------------------------------------------------------------
# Tracking plugin
# ---------------------------------------------------------------------------


class TrackingPlugin(BasePlugin):
  __test__ = False

  def __init__(self, name: str = "tracker"):
    super().__init__(name)
    self.after_agent_called = False
    self.agent_error_calls: list[dict] = []

  async def after_agent_callback(self, *, agent, callback_context, **kwargs):
    self.after_agent_called = True

  async def on_agent_error_callback(
      self, *, agent, callback_context, error, **kwargs
  ) -> None:
    self.agent_error_calls.append(
        {"agent": agent, "callback_context": callback_context, "error": error}
    )


# ---------------------------------------------------------------------------
# Helper to drive run_async
# ---------------------------------------------------------------------------


async def _collect_events(
    agent: BaseAgent, plugins: list[BasePlugin]
) -> list[Event]:
  inv_ctx = await testing_utils.create_invocation_context(
      agent=agent, plugins=plugins
  )
  events = []
  async for event in agent.run_async(inv_ctx):
    events.append(event)
  return events


async def _collect_live_events(
    agent: BaseAgent, plugins: list[BasePlugin]
) -> list[Event]:
  inv_ctx = await testing_utils.create_invocation_context(
      agent=agent, plugins=plugins
  )
  events = []
  async for event in agent.run_live(inv_ctx):
    events.append(event)
  return events


# ---------------------------------------------------------------------------
# Tests — run_async path
# ---------------------------------------------------------------------------


class TestAgentOnAgentErrorCallbackAsync:

  @pytest.mark.asyncio
  async def test_on_agent_error_callback_called_when_impl_raises(self):
    tracker = TrackingPlugin()
    with pytest.raises(RuntimeError, match="agent impl exploded"):
      await _collect_events(_FailingAgent(), [tracker])

    assert len(tracker.agent_error_calls) == 1
    assert tracker.agent_error_calls[0]["error"] is _FailingAgent.BOOM

  @pytest.mark.asyncio
  async def test_on_agent_error_callback_receives_correct_agent(self):
    tracker = TrackingPlugin()
    agent = _FailingAgent()

    with pytest.raises(RuntimeError):
      await _collect_events(agent, [tracker])

    assert tracker.agent_error_calls[0]["agent"] is agent

  @pytest.mark.asyncio
  async def test_on_agent_error_callback_receives_callback_context(self):
    tracker = TrackingPlugin()

    with pytest.raises(RuntimeError):
      await _collect_events(_FailingAgent(), [tracker])

    cb_ctx = tracker.agent_error_calls[0]["callback_context"]
    assert isinstance(cb_ctx, CallbackContext)

  @pytest.mark.asyncio
  async def test_original_exception_reraised_after_notification(self):
    tracker = TrackingPlugin()

    with pytest.raises(RuntimeError) as exc_info:
      await _collect_events(_FailingAgent(), [tracker])

    assert exc_info.value is _FailingAgent.BOOM

  @pytest.mark.asyncio
  async def test_after_agent_callback_not_called_on_error(self):
    tracker = TrackingPlugin()

    with pytest.raises(RuntimeError):
      await _collect_events(_FailingAgent(), [tracker])

    assert not tracker.after_agent_called

  @pytest.mark.asyncio
  async def test_on_agent_error_callback_not_called_on_success(self):
    tracker = TrackingPlugin()
    events = await _collect_events(_SuccessAgent(), [tracker])

    assert len(events) >= 1
    assert len(tracker.agent_error_calls) == 0

  @pytest.mark.asyncio
  async def test_after_agent_callback_still_called_on_success(self):
    tracker = TrackingPlugin()
    await _collect_events(_SuccessAgent(), [tracker])

    assert tracker.after_agent_called

  @pytest.mark.asyncio
  async def test_multiple_plugins_all_notified_on_agent_error(self):
    tracker_a = TrackingPlugin("a")
    tracker_b = TrackingPlugin("b")

    with pytest.raises(RuntimeError):
      await _collect_events(_FailingAgent(), [tracker_a, tracker_b])

    assert len(tracker_a.agent_error_calls) == 1
    assert len(tracker_b.agent_error_calls) == 1


# ---------------------------------------------------------------------------
# Tests — run_live path
# ---------------------------------------------------------------------------


class TestAgentOnAgentErrorCallbackLive:

  @pytest.mark.asyncio
  async def test_on_agent_error_callback_called_when_live_impl_raises(self):
    tracker = TrackingPlugin()

    with pytest.raises(RuntimeError, match="live agent impl exploded"):
      await _collect_live_events(_FailingLiveAgent(), [tracker])

    assert len(tracker.agent_error_calls) == 1
    assert tracker.agent_error_calls[0]["error"] is _FailingLiveAgent.BOOM

  @pytest.mark.asyncio
  async def test_after_agent_callback_not_called_on_live_error(self):
    tracker = TrackingPlugin()

    with pytest.raises(RuntimeError):
      await _collect_live_events(_FailingLiveAgent(), [tracker])

    assert not tracker.after_agent_called

  @pytest.mark.asyncio
  async def test_original_live_exception_reraised_unchanged(self):
    tracker = TrackingPlugin()

    with pytest.raises(RuntimeError) as exc_info:
      await _collect_live_events(_FailingLiveAgent(), [tracker])

    assert exc_info.value is _FailingLiveAgent.BOOM
