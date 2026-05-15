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

"""Tests for on_agent_error_callback and on_run_error_callback.

Validates RFC #5044: agent-level and runner-level error callbacks.
"""

import asyncio
from typing import AsyncGenerator
from typing import Optional
from unittest.mock import AsyncMock
from unittest.mock import Mock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.plugins.plugin_manager import PluginManager
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
import pytest
from typing_extensions import override

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CrashingAgent(BaseAgent):
  """Agent whose _run_async_impl always raises."""

  crash_error: Exception = RuntimeError("agent crashed")

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    raise self.crash_error
    yield  # make it an async generator

  @override
  async def _run_live_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    raise self.crash_error
    yield


class _SuccessAgent(BaseAgent):
  """Agent that completes successfully."""

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield Event(
        author=self.name,
        branch=ctx.branch,
        invocation_id=ctx.invocation_id,
        content=types.Content(parts=[types.Part(text="ok")]),
    )

  @override
  async def _run_live_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield Event(
        author=self.name,
        branch=ctx.branch,
        invocation_id=ctx.invocation_id,
        content=types.Content(parts=[types.Part(text="ok live")]),
    )


class _ErrorTrackingPlugin(BasePlugin):
  """Plugin that records which error callbacks were called."""

  __test__ = False

  def __init__(self, name: str = "error_tracker"):
    super().__init__(name)
    self.agent_errors: list[tuple[str, Exception]] = []
    self.run_errors: list[Exception] = []
    self.after_agent_called = False
    self.after_run_called = False

  async def on_agent_error_callback(
      self,
      *,
      agent: BaseAgent,
      callback_context: CallbackContext,
      error: Exception,
  ) -> None:
    self.agent_errors.append((agent.name, error))

  async def on_run_error_callback(
      self,
      *,
      invocation_context: InvocationContext,
      error: Exception,
  ) -> None:
    self.run_errors.append(error)

  async def after_agent_callback(
      self,
      *,
      agent: BaseAgent,
      callback_context: CallbackContext,
  ) -> Optional[types.Content]:
    self.after_agent_called = True
    return None

  async def after_run_callback(
      self,
      *,
      invocation_context: InvocationContext,
  ) -> None:
    self.after_run_called = True


async def _create_ctx(
    agent: BaseAgent,
    plugins: list[BasePlugin] | None = None,
) -> InvocationContext:
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name="test_app", user_id="test_user"
  )
  return InvocationContext(
      invocation_id="test_invocation",
      agent=agent,
      session=session,
      session_service=session_service,
      plugin_manager=PluginManager(plugins=plugins or []),
  )


# ---------------------------------------------------------------------------
# Agent-level error callback tests
# ---------------------------------------------------------------------------


class TestAgentErrorCallback:
  """Tests for on_agent_error_callback in base_agent.py."""

  @pytest.mark.asyncio
  async def test_agent_error_callback_fires_on_crash(self):
    """Error callback fires when _run_async_impl raises."""
    plugin = _ErrorTrackingPlugin()
    agent = _CrashingAgent(name="crash_agent")
    ctx = await _create_ctx(agent, plugins=[plugin])

    with pytest.raises(RuntimeError, match="agent crashed"):
      _ = [e async for e in agent.run_async(ctx)]

    assert len(plugin.agent_errors) == 1
    assert plugin.agent_errors[0][0] == "crash_agent"
    assert str(plugin.agent_errors[0][1]) == "agent crashed"

  @pytest.mark.asyncio
  async def test_agent_error_callback_fires_on_live_crash(self):
    """Error callback fires when _run_live_impl raises."""
    plugin = _ErrorTrackingPlugin()
    agent = _CrashingAgent(name="crash_agent")
    ctx = await _create_ctx(agent, plugins=[plugin])

    with pytest.raises(RuntimeError, match="agent crashed"):
      _ = [e async for e in agent.run_live(ctx)]

    assert len(plugin.agent_errors) == 1
    assert plugin.agent_errors[0][0] == "crash_agent"

  @pytest.mark.asyncio
  async def test_after_agent_not_called_on_crash(self):
    """after_agent_callback (success-only) is NOT called on failure."""
    plugin = _ErrorTrackingPlugin()
    agent = _CrashingAgent(name="crash_agent")
    ctx = await _create_ctx(agent, plugins=[plugin])

    with pytest.raises(RuntimeError):
      _ = [e async for e in agent.run_async(ctx)]

    assert not plugin.after_agent_called

  @pytest.mark.asyncio
  async def test_exception_is_reraised_after_agent_error_callback(self):
    """The original exception propagates after the error callback."""
    plugin = _ErrorTrackingPlugin()
    err = ValueError("specific error")
    agent = _CrashingAgent(name="crash_agent", crash_error=err)
    ctx = await _create_ctx(agent, plugins=[plugin])

    with pytest.raises(ValueError, match="specific error"):
      _ = [e async for e in agent.run_async(ctx)]

  @pytest.mark.asyncio
  async def test_agent_error_callback_not_fired_on_success(self):
    """Error callback does NOT fire when agent succeeds."""
    plugin = _ErrorTrackingPlugin()
    agent = _SuccessAgent(name="good_agent")
    ctx = await _create_ctx(agent, plugins=[plugin])

    events = [e async for e in agent.run_async(ctx)]

    assert len(events) > 0
    assert len(plugin.agent_errors) == 0
    # after_agent_callback should still fire on success
    assert plugin.after_agent_called

  @pytest.mark.asyncio
  async def test_cancelled_error_does_not_trigger_agent_error_callback(
      self,
  ):
    """asyncio.CancelledError (BaseException) does NOT trigger error callback."""

    class _CancellingAgent(BaseAgent):

      @override
      async def _run_async_impl(self, ctx):
        raise asyncio.CancelledError()
        yield

      @override
      async def _run_live_impl(self, ctx):
        raise asyncio.CancelledError()
        yield

    plugin = _ErrorTrackingPlugin()
    agent = _CancellingAgent(name="cancel_agent")
    ctx = await _create_ctx(agent, plugins=[plugin])

    with pytest.raises(asyncio.CancelledError):
      _ = [e async for e in agent.run_async(ctx)]

    assert len(plugin.agent_errors) == 0


# ---------------------------------------------------------------------------
# Runner-level error callback tests
# ---------------------------------------------------------------------------


class TestRunErrorCallback:
  """Tests for on_run_error_callback in runners.py."""

  @pytest.mark.asyncio
  async def test_run_error_callback_fires_on_crash(self):
    """on_run_error_callback fires when execute_fn raises."""
    from google.adk.runners import Runner

    plugin = _ErrorTrackingPlugin()
    agent = _CrashingAgent(name="crash_agent")
    runner = Runner(
        agent=agent,
        app_name="test_app",
        session_service=InMemorySessionService(),
        plugins=[plugin],
    )
    session = await runner.session_service.create_session(
        app_name="test_app", user_id="test_user"
    )

    with pytest.raises(RuntimeError, match="agent crashed"):
      _ = [
          e
          async for e in runner.run_async(
              user_id="test_user",
              session_id=session.id,
              new_message=types.Content(parts=[types.Part(text="hello")]),
          )
      ]

    assert len(plugin.run_errors) == 1
    assert str(plugin.run_errors[0]) == "agent crashed"

  @pytest.mark.asyncio
  async def test_after_run_not_called_on_crash(self):
    """after_run_callback (success-only) is NOT called on failure."""
    from google.adk.runners import Runner

    plugin = _ErrorTrackingPlugin()
    agent = _CrashingAgent(name="crash_agent")
    runner = Runner(
        agent=agent,
        app_name="test_app",
        session_service=InMemorySessionService(),
        plugins=[plugin],
    )
    session = await runner.session_service.create_session(
        app_name="test_app", user_id="test_user"
    )

    with pytest.raises(RuntimeError):
      _ = [
          e
          async for e in runner.run_async(
              user_id="test_user",
              session_id=session.id,
              new_message=types.Content(parts=[types.Part(text="hello")]),
          )
      ]

    assert not plugin.after_run_called

  @pytest.mark.asyncio
  async def test_run_error_callback_not_fired_on_success(self):
    """on_run_error_callback does NOT fire on success."""
    from google.adk.runners import Runner

    plugin = _ErrorTrackingPlugin()
    agent = _SuccessAgent(name="good_agent")
    runner = Runner(
        agent=agent,
        app_name="test_app",
        session_service=InMemorySessionService(),
        plugins=[plugin],
    )
    session = await runner.session_service.create_session(
        app_name="test_app", user_id="test_user"
    )

    events = [
        e
        async for e in runner.run_async(
            user_id="test_user",
            session_id=session.id,
            new_message=types.Content(parts=[types.Part(text="hello")]),
        )
    ]

    assert len(events) > 0
    assert len(plugin.run_errors) == 0
    assert plugin.after_run_called


# ---------------------------------------------------------------------------
# Exactly-once-per-layer tests
# ---------------------------------------------------------------------------


class TestExactlyOncePerLayer:
  """Verify each error callback fires exactly once at its own layer."""

  @pytest.mark.asyncio
  async def test_agent_crash_fires_both_callbacks_once_each(self):
    """A crashing agent fires on_agent_error_callback once AND
    on_run_error_callback once (the re-raised exception propagates)."""
    from google.adk.runners import Runner

    plugin = _ErrorTrackingPlugin()
    agent = _CrashingAgent(name="crash_agent")
    runner = Runner(
        agent=agent,
        app_name="test_app",
        session_service=InMemorySessionService(),
        plugins=[plugin],
    )
    session = await runner.session_service.create_session(
        app_name="test_app", user_id="test_user"
    )

    with pytest.raises(RuntimeError, match="agent crashed"):
      _ = [
          e
          async for e in runner.run_async(
              user_id="test_user",
              session_id=session.id,
              new_message=types.Content(parts=[types.Part(text="hello")]),
          )
      ]

    # Agent error callback: exactly 1 call
    assert len(plugin.agent_errors) == 1
    assert plugin.agent_errors[0][0] == "crash_agent"

    # Run error callback: exactly 1 call (same exception bubbled up)
    assert len(plugin.run_errors) == 1
    assert plugin.run_errors[0] is plugin.agent_errors[0][1]

    # Neither after callback should fire
    assert not plugin.after_agent_called
    assert not plugin.after_run_called


# ---------------------------------------------------------------------------
# PluginManager dispatch tests
# ---------------------------------------------------------------------------


class TestPluginManagerErrorCallbackDispatch:
  """Test PluginManager correctly dispatches the new error callbacks."""

  @pytest.mark.asyncio
  async def test_run_on_agent_error_callback_dispatches(self):
    """run_on_agent_error_callback calls all plugins."""
    plugin1 = _ErrorTrackingPlugin(name="p1")
    plugin2 = _ErrorTrackingPlugin(name="p2")
    pm = PluginManager(plugins=[plugin1, plugin2])

    mock_agent = Mock(spec=BaseAgent)
    mock_agent.name = "test_agent"
    mock_ctx = Mock(spec=CallbackContext)
    err = RuntimeError("boom")

    await pm.run_on_agent_error_callback(
        agent=mock_agent,
        callback_context=mock_ctx,
        error=err,
    )

    assert len(plugin1.agent_errors) == 1
    assert len(plugin2.agent_errors) == 1

  @pytest.mark.asyncio
  async def test_run_on_run_error_callback_dispatches(self):
    """run_on_run_error_callback calls all plugins."""
    plugin1 = _ErrorTrackingPlugin(name="p1")
    plugin2 = _ErrorTrackingPlugin(name="p2")
    pm = PluginManager(plugins=[plugin1, plugin2])

    mock_ctx = Mock(spec=InvocationContext)
    err = RuntimeError("boom")

    await pm.run_on_run_error_callback(
        invocation_context=mock_ctx,
        error=err,
    )

    assert len(plugin1.run_errors) == 1
    assert len(plugin2.run_errors) == 1

  @pytest.mark.asyncio
  async def test_agent_error_callback_does_not_short_circuit(self):
    """on_agent_error_callback is notification-only: a non-None return
    from one plugin does NOT skip subsequent plugins."""

    class _ReturningPlugin(BasePlugin):
      __test__ = False

      def __init__(self, name):
        super().__init__(name)
        self.agent_error_called = False

      async def on_agent_error_callback(self, **kwargs):
        self.agent_error_called = True
        return "should be ignored"

    p1 = _ReturningPlugin(name="p1")
    p2 = _ReturningPlugin(name="p2")
    pm = PluginManager(plugins=[p1, p2])

    await pm.run_on_agent_error_callback(
        agent=Mock(spec=BaseAgent),
        callback_context=Mock(spec=CallbackContext),
        error=RuntimeError("x"),
    )

    # Both plugins must be called even though p1 returns non-None.
    assert p1.agent_error_called
    assert p2.agent_error_called

  @pytest.mark.asyncio
  async def test_run_error_callback_does_not_short_circuit(self):
    """on_run_error_callback is notification-only: a non-None return
    from one plugin does NOT skip subsequent plugins."""

    class _ReturningPlugin(BasePlugin):
      __test__ = False

      def __init__(self, name):
        super().__init__(name)
        self.run_error_called = False

      async def on_run_error_callback(self, **kwargs):
        self.run_error_called = True
        return "should be ignored"

    p1 = _ReturningPlugin(name="p1")
    p2 = _ReturningPlugin(name="p2")
    pm = PluginManager(plugins=[p1, p2])

    await pm.run_on_run_error_callback(
        invocation_context=Mock(spec=InvocationContext),
        error=RuntimeError("x"),
    )

    # Both plugins must be called even though p1 returns non-None.
    assert p1.run_error_called
    assert p2.run_error_called

  @pytest.mark.asyncio
  async def test_plugin_callback_failure_does_not_mask_app_error(self):
    """When a plugin's error callback raises, iteration continues
    and the original application exception is what the caller sees."""

    class _FailingPlugin(BasePlugin):
      __test__ = False

      def __init__(self, name):
        super().__init__(name)
        self.agent_error_called = False
        self.run_error_called = False

      async def on_agent_error_callback(self, **kwargs):
        self.agent_error_called = True
        raise ValueError("plugin boom")

      async def on_run_error_callback(self, **kwargs):
        self.run_error_called = True
        raise ValueError("plugin boom")

    p1 = _FailingPlugin(name="p1")
    p2 = _ErrorTrackingPlugin(name="p2")
    pm = PluginManager(plugins=[p1, p2])

    # Agent error callback: p1 raises, p2 must still be notified.
    mock_agent = Mock(spec=BaseAgent)
    mock_agent.name = "test_agent"
    await pm.run_on_agent_error_callback(
        agent=mock_agent,
        callback_context=Mock(spec=CallbackContext),
        error=RuntimeError("app crash"),
    )
    assert p1.agent_error_called
    assert len(p2.agent_errors) == 1

    # Run error callback: same behavior.
    await pm.run_on_run_error_callback(
        invocation_context=Mock(spec=InvocationContext),
        error=RuntimeError("app crash"),
    )
    assert p1.run_error_called
    assert len(p2.run_errors) == 1

  @pytest.mark.asyncio
  async def test_original_exception_propagates_despite_plugin_failure(
      self,
  ):
    """End-to-end: a crashing plugin error callback does not mask
    the original agent exception seen by the caller."""

    class _FailingPlugin(BasePlugin):
      __test__ = False

      def __init__(self, name):
        super().__init__(name)

      async def on_agent_error_callback(self, **kwargs):
        raise ValueError("plugin internal error")

    plugin = _FailingPlugin(name="bad_plugin")
    agent = _CrashingAgent(name="crash_agent")
    ctx = await _create_ctx(agent, plugins=[plugin])

    # The caller must see the original RuntimeError("agent crashed"),
    # NOT the plugin's ValueError.
    with pytest.raises(RuntimeError, match="agent crashed"):
      _ = [e async for e in agent.run_async(ctx)]

  @pytest.mark.asyncio
  async def test_original_exception_propagates_despite_run_plugin_failure(
      self,
  ):
    """End-to-end: a crashing plugin on_run_error_callback does not mask
    the original agent exception seen by the runner caller."""
    from google.adk.runners import Runner

    class _FailingRunPlugin(BasePlugin):
      __test__ = False

      def __init__(self, name):
        super().__init__(name)

      async def on_run_error_callback(self, **kwargs):
        raise ValueError("plugin internal error")

    plugin = _FailingRunPlugin(name="bad_plugin")
    agent = _CrashingAgent(name="crash_agent")
    runner = Runner(
        agent=agent,
        app_name="test_app",
        session_service=InMemorySessionService(),
        plugins=[plugin],
    )
    session = await runner.session_service.create_session(
        app_name="test_app", user_id="test_user"
    )

    # The caller must see the original RuntimeError("agent crashed"),
    # NOT the plugin's ValueError.
    with pytest.raises(RuntimeError, match="agent crashed"):
      _ = [
          e
          async for e in runner.run_async(
              user_id="test_user",
              session_id=session.id,
              new_message=types.Content(parts=[types.Part(text="hello")]),
          )
      ]
