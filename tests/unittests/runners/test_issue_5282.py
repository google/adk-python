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

"""Regression tests for google/adk-python#5282.

Runner._run_node_async (the dispatch path for Workflow / BaseNode roots)
dispatches on_user_message_callback, before_run_callback, and
on_event_callback, but does not dispatch run_after_run_callback
(runners.py:427 TODO).

Three tests:
  (a) Baseline — pre-run and event hooks DO fire on a Workflow root.
  (b) Regression anchor — after_run_callback does NOT fire (strict xfail).
      Remove the xfail when the TODO at runners.py:427 is wired.
  (c) Workaround proof — Runner subclass wrapping run_async restores
      after_run_callback dispatch without touching ADK source.

Concurrency note for the WorkaroundRunner pattern: under concurrent
run_async calls on a shared Runner, the _last_ic stash should live on a
contextvars.ContextVar rather than self. The instance attribute is safe
for single-invocation tests but will race under concurrent load.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncGenerator
from typing import Optional

from google.adk.agents.invocation_context import InvocationContext
from google.adk.apps.app import App
from google.adk.events.event import Event
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow import Workflow
from google.genai import types
import pytest

APP_NAME = "issue_5282_repro"
USER_ID = "u1"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class CallbackCounts:
  on_user_message_callback: int = 0
  before_run_callback: int = 0
  on_event_callback: int = 0
  after_run_callback: int = 0


class TracerPlugin(BasePlugin):
  """Counts every Plugin lifecycle callback the Runner dispatches."""

  __test__ = False

  def __init__(self) -> None:
    super().__init__(name="tracer")
    self.counts = CallbackCounts()

  async def on_user_message_callback(
      self,
      *,
      invocation_context: InvocationContext,
      user_message: types.Content,
  ) -> Optional[types.Content]:
    self.counts.on_user_message_callback += 1
    return None

  async def before_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> Optional[types.Content]:
    self.counts.before_run_callback += 1
    return None

  async def on_event_callback(
      self, *, invocation_context: InvocationContext, event: Event
  ) -> Optional[Event]:
    self.counts.on_event_callback += 1
    return None

  async def after_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> None:
    self.counts.after_run_callback += 1
    return None


async def _terminal_node(ctx) -> Event:
  """Minimal terminal node yielding a content-bearing Event.

  Content (not just state) ensures _consume_event_queue runs the
  on_event_callback path — the canonical case the plugin hook targets.
  """
  return Event(
      content=types.Content(
          parts=[types.Part(text="done")],
          role="model",
      )
  )


def _build_runner(
    plugin: TracerPlugin, *, runner_cls: type[Runner] = Runner
) -> Runner:
  workflow = Workflow(
      name="Issue5282Repro", edges=[("START", _terminal_node)]
  )
  app = App(name=APP_NAME, root_agent=workflow, plugins=[plugin])
  return runner_cls(
      app_name=APP_NAME,
      app=app,
      session_service=InMemorySessionService(),
      memory_service=InMemoryMemoryService(),
  )


async def _drive_one_invocation(runner: Runner) -> None:
  session = await runner.session_service.create_session(
      app_name=APP_NAME, user_id=USER_ID
  )
  async for _ in runner.run_async(
      user_id=USER_ID,
      session_id=session.id,
      new_message=types.Content(
          parts=[types.Part(text="hi")], role="user"
      ),
  ):
    pass


# ---------------------------------------------------------------------------
# Workaround: Runner subclass dispatching run_after_run_callback post-drain
# ---------------------------------------------------------------------------


class WorkaroundRunner(Runner):
  """Interim workaround for #5282.

  Wraps run_async to dispatch plugin_manager.run_after_run_callback once
  the inner generator drains. Captures the active InvocationContext via
  _new_invocation_context (called once at runners.py:446).

  Drop this class when the runners.py:427 TODO is resolved — the stock
  Runner will dispatch after_run_callback natively, and
  test_workflow_root_after_run_callback_not_dispatched will flip green
  as the signal.
  """

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self._last_ic: Optional[InvocationContext] = None

  def _new_invocation_context(self, session, **kwargs) -> InvocationContext:
    ic = super()._new_invocation_context(session, **kwargs)
    self._last_ic = ic
    return ic

  async def run_async(self, **kwargs) -> AsyncGenerator[Event, None]:
    async for event in super().run_async(**kwargs):
      yield event
    ic = self._last_ic
    if ic is not None:
      await ic.plugin_manager.run_after_run_callback(
          invocation_context=ic
      )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workflow_root_dispatches_pre_run_and_event_hooks():
  """Baseline: pre-run and event hooks fire on a Workflow (BaseNode) root."""
  plugin = TracerPlugin()
  runner = _build_runner(plugin)

  await _drive_one_invocation(runner)

  assert plugin.counts.on_user_message_callback == 1
  assert plugin.counts.before_run_callback == 1
  assert plugin.counts.on_event_callback >= 1, (
      "on_event_callback should fire via _consume_event_queue "
      "(runners.py:619) for the content-bearing terminal event"
  )


@pytest.mark.xfail(
    reason=(
        "#5282: runners.py:427 TODO — _run_node_async does not dispatch "
        "plugin_manager.run_after_run_callback on the BaseNode path. "
        "Remove this xfail when the TODO lands."
    ),
    strict=True,
)
@pytest.mark.asyncio
async def test_workflow_root_after_run_callback_not_dispatched():
  """Regression anchor: stock Runner does NOT fire after_run_callback.

  Strict xfail — passes (as xfail) while the bug exists, fails loudly if
  after_run_callback starts firing unexpectedly. When the fix lands, delete
  the @xfail decorator and the test becomes a green regression guard.
  """
  plugin = TracerPlugin()
  runner = _build_runner(plugin)

  await _drive_one_invocation(runner)

  assert plugin.counts.after_run_callback == 1


@pytest.mark.asyncio
async def test_workaround_runner_dispatches_after_run_callback():
  """WorkaroundRunner restores after_run_callback without touching ADK source."""
  plugin = TracerPlugin()
  runner = _build_runner(plugin, runner_cls=WorkaroundRunner)

  await _drive_one_invocation(runner)

  assert plugin.counts.on_user_message_callback == 1
  assert plugin.counts.before_run_callback == 1
  assert plugin.counts.on_event_callback >= 1
  assert plugin.counts.after_run_callback == 1
