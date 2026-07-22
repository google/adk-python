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
from unittest.mock import Mock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.plugins.plugin_manager import PluginManager
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow._base_node import BaseNode
from google.genai import types
import pytest
from typing_extensions import override

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CrashingNode(BaseNode):
  """A workflow node whose _run_impl always raises.

  A root ``BaseNode`` (that is not a ``BaseAgent``) is executed through the
  node runtime (``Runner._run_node_async``), so this exercises the
  ``on_run_error_callback`` dispatch site added there.
  """

  __test__ = False

  @override
  async def _run_impl(self, *, ctx, node_input):
    raise RuntimeError("node crashed")
    yield  # pragma: no cover - makes this an async generator


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
  async def test_agent_error_callback_fires_on_before_callback_failure(self):
    """Error callback fires when before_agent_callback raises.

    The error handler wraps the full agent lifecycle, so lifecycle-callback
    failures (not just _run_async_impl) are surfaced to on_agent_error_callback.
    """
    plugin = _ErrorTrackingPlugin()

    def _boom(callback_context):
      raise RuntimeError("before boom")

    agent = _SuccessAgent(name="good_agent", before_agent_callback=_boom)
    ctx = await _create_ctx(agent, plugins=[plugin])

    with pytest.raises(RuntimeError, match="before boom"):
      _ = [e async for e in agent.run_async(ctx)]

    assert len(plugin.agent_errors) == 1
    assert plugin.agent_errors[0][0] == "good_agent"
    assert not plugin.after_agent_called

  @pytest.mark.asyncio
  async def test_agent_error_callback_fires_on_after_callback_failure(self):
    """Error callback fires when after_agent_callback raises."""
    plugin = _ErrorTrackingPlugin()

    def _boom(callback_context):
      raise RuntimeError("after boom")

    agent = _SuccessAgent(name="good_agent", after_agent_callback=_boom)
    ctx = await _create_ctx(agent, plugins=[plugin])

    with pytest.raises(RuntimeError, match="after boom"):
      _ = [e async for e in agent.run_async(ctx)]

    assert len(plugin.agent_errors) == 1
    assert plugin.agent_errors[0][0] == "good_agent"

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

  @pytest.mark.asyncio
  async def test_run_error_callback_fires_on_after_run_failure(self):
    """An after_run_callback failure notifies on_run_error and re-raises.

    after_run runs on the success path, after the main-execution catch. A
    failing after_run plugin (which PluginManager surfaces as a RuntimeError)
    is an unhandled runner error, so it must still notify on_run_error_callback
    exactly once while the original error propagates.
    """
    from google.adk.runners import Runner

    class _FailingAfterRunPlugin(BasePlugin):

      async def after_run_callback(
          self, *, invocation_context: InvocationContext
      ) -> None:
        raise RuntimeError("after_run failed")

    tracker = _ErrorTrackingPlugin()
    agent = _SuccessAgent(name="good_agent")
    runner = Runner(
        agent=agent,
        app_name="test_app",
        session_service=InMemorySessionService(),
        plugins=[_FailingAfterRunPlugin(name="failing_after_run"), tracker],
    )
    session = await runner.session_service.create_session(
        app_name="test_app", user_id="test_user"
    )

    with pytest.raises(RuntimeError, match="after_run failed"):
      _ = [
          e
          async for e in runner.run_async(
              user_id="test_user",
              session_id=session.id,
              new_message=types.Content(parts=[types.Part(text="hello")]),
          )
      ]

    # Exactly one run-error notification for the after_run failure.
    assert len(tracker.run_errors) == 1
    # PluginManager wraps plugin exceptions with plugin/callback context.
    assert "after_run failed" in str(tracker.run_errors[0])


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
  async def test_original_exception_propagates_despite_agent_plugin_failure(
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


# ---------------------------------------------------------------------------
# Node-runtime run-error coverage
# ---------------------------------------------------------------------------


class TestNodeRuntimeRunErrorCallback:
  """on_run_error_callback fires for the node runtime path (_run_node_async).

  Runner(node=...) with a non-agent BaseNode root routes through
  _run_node_async rather than the legacy _exec_with_plugin path.
  """

  @pytest.mark.asyncio
  async def test_run_error_callback_fires_via_node_runtime(self):
    from google.adk.runners import Runner

    plugin = _ErrorTrackingPlugin()
    node = _CrashingNode(name="crash_node")
    runner = Runner(
        app_name="test_app",
        node=node,
        session_service=InMemorySessionService(),
        plugins=[plugin],
    )
    session = await runner.session_service.create_session(
        app_name="test_app", user_id="test_user"
    )

    with pytest.raises(RuntimeError, match="node crashed"):
      _ = [
          e
          async for e in runner.run_async(
              user_id="test_user",
              session_id=session.id,
              new_message=types.Content(parts=[types.Part(text="hello")]),
          )
      ]

    # Exactly one run-error notification from the node runtime path.
    assert len(plugin.run_errors) == 1
    assert str(plugin.run_errors[0]) == "node crashed"
    # after_run stays success-only.
    assert not plugin.after_run_called

  @pytest.mark.asyncio
  async def test_run_error_callback_not_fired_on_node_success(self):
    """No run-error notification for a successful node-runtime run."""
    from google.adk.runners import Runner

    plugin = _ErrorTrackingPlugin()

    class _OkNode(BaseNode):
      __test__ = False

      @override
      async def _run_impl(self, *, ctx, node_input):
        yield Event(
            author=self.name,
            invocation_id=ctx.get_invocation_context().invocation_id,
            content=types.Content(parts=[types.Part(text="ok")]),
        )

    runner = Runner(
        app_name="test_app",
        node=_OkNode(name="ok_node"),
        session_service=InMemorySessionService(),
        plugins=[plugin],
    )
    session = await runner.session_service.create_session(
        app_name="test_app", user_id="test_user"
    )

    _ = [
        e
        async for e in runner.run_async(
            user_id="test_user",
            session_id=session.id,
            new_message=types.Content(parts=[types.Part(text="hello")]),
        )
    ]

    assert len(plugin.run_errors) == 0
    assert plugin.after_run_called

  @pytest.mark.asyncio
  async def test_run_error_callback_fires_on_node_before_run_failure(self):
    """A before_run_callback failure on the node path notifies on_run_error.

    The setup hooks in _run_node_async run before the main event loop; their
    failures must still be surfaced to on_run_error_callback.
    """
    from google.adk.runners import Runner

    class _FailingBeforeRunPlugin(BasePlugin):

      async def before_run_callback(
          self, *, invocation_context: InvocationContext
      ) -> Optional[types.Content]:
        raise RuntimeError("before_run failed")

    tracker = _ErrorTrackingPlugin()
    runner = Runner(
        app_name="test_app",
        node=_CrashingNode(name="never_runs"),
        session_service=InMemorySessionService(),
        plugins=[_FailingBeforeRunPlugin(name="failing_before_run"), tracker],
    )
    session = await runner.session_service.create_session(
        app_name="test_app", user_id="test_user"
    )

    with pytest.raises(RuntimeError, match="before_run failed"):
      _ = [
          e
          async for e in runner.run_async(
              user_id="test_user",
              session_id=session.id,
              new_message=types.Content(parts=[types.Part(text="hello")]),
          )
      ]

    assert len(tracker.run_errors) == 1
    # PluginManager wraps plugin exceptions with plugin/callback context.
    assert "before_run failed" in str(tracker.run_errors[0])
    # after_run stays success-only.
    assert not tracker.after_run_called

  @pytest.mark.asyncio
  async def test_run_error_callback_fires_on_node_user_message_failure(self):
    """An on_user_message_callback failure on the node path notifies on_run_error."""
    from google.adk.runners import Runner

    class _FailingUserMessagePlugin(BasePlugin):

      async def on_user_message_callback(
          self,
          *,
          invocation_context: InvocationContext,
          user_message: types.Content,
      ) -> Optional[types.Content]:
        raise RuntimeError("user_message failed")

    tracker = _ErrorTrackingPlugin()
    runner = Runner(
        app_name="test_app",
        node=_CrashingNode(name="never_runs"),
        session_service=InMemorySessionService(),
        plugins=[_FailingUserMessagePlugin(name="failing_user_msg"), tracker],
    )
    session = await runner.session_service.create_session(
        app_name="test_app", user_id="test_user"
    )

    with pytest.raises(RuntimeError, match="user_message failed"):
      _ = [
          e
          async for e in runner.run_async(
              user_id="test_user",
              session_id=session.id,
              new_message=types.Content(parts=[types.Part(text="hello")]),
          )
      ]

    assert len(tracker.run_errors) == 1
    # PluginManager wraps plugin exceptions with plugin/callback context.
    assert "user_message failed" in str(tracker.run_errors[0])
    assert not tracker.after_run_called

  @pytest.mark.asyncio
  async def test_run_error_callback_fires_on_node_after_run_failure(self):
    """An after_run_callback failure on the node path notifies on_run_error.

    after_run runs on the node-runtime success path, outside the main-loop
    catch. A failing after_run plugin (which PluginManager surfaces as a
    RuntimeError) is an unhandled runner error, so it must still notify
    on_run_error_callback exactly once while the original error propagates.
    """
    from google.adk.runners import Runner

    class _OkNode(BaseNode):
      __test__ = False

      @override
      async def _run_impl(self, *, ctx, node_input):
        yield Event(
            author=self.name,
            invocation_id=ctx.get_invocation_context().invocation_id,
            content=types.Content(parts=[types.Part(text="ok")]),
        )

    class _FailingAfterRunPlugin(BasePlugin):

      async def after_run_callback(
          self, *, invocation_context: InvocationContext
      ) -> None:
        raise RuntimeError("after_run failed")

    tracker = _ErrorTrackingPlugin()
    runner = Runner(
        app_name="test_app",
        node=_OkNode(name="ok_node"),
        session_service=InMemorySessionService(),
        plugins=[_FailingAfterRunPlugin(name="failing_after_run"), tracker],
    )
    session = await runner.session_service.create_session(
        app_name="test_app", user_id="test_user"
    )

    with pytest.raises(RuntimeError, match="after_run failed"):
      _ = [
          e
          async for e in runner.run_async(
              user_id="test_user",
              session_id=session.id,
              new_message=types.Content(parts=[types.Part(text="hello")]),
          )
      ]

    # Exactly one run-error notification for the after_run failure.
    assert len(tracker.run_errors) == 1
    # PluginManager wraps plugin exceptions with plugin/callback context.
    assert "after_run failed" in str(tracker.run_errors[0])
