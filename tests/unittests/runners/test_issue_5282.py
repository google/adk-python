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

"""Regression test for google/adk-python#5282.

Before the fix, Runner._run_node_async never dispatched
plugin_manager.run_after_run_callback on the BaseNode path (runners.py:427
TODO), so Plugin.after_run_callback silently no-op'd for Workflow roots.

These tests pin the post-fix behavior: both the pre-run/event hooks AND
after_run_callback must fire on a Workflow root.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from google.adk import Event
from google.adk import Workflow
from google.adk.agents.invocation_context import InvocationContext
from google.adk.apps.app import App
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
import pytest


@dataclass
class CallbackCounts:
  on_user_message_callback: int = 0
  before_run_callback: int = 0
  on_event_callback: int = 0
  after_run_callback: int = 0


class TracerPlugin(BasePlugin):
  """Counts every Plugin lifecycle callback the Runner actually dispatches."""

  def __init__(self) -> None:
    super().__init__(name='tracer')
    self.counts = CallbackCounts()

  async def on_user_message_callback(
      self,
      *,
      invocation_context: InvocationContext,
      user_message: types.Content,
  ) -> Optional[types.Content]:
    del invocation_context, user_message
    self.counts.on_user_message_callback += 1
    return None

  async def before_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> Optional[types.Content]:
    del invocation_context
    self.counts.before_run_callback += 1
    return None

  async def on_event_callback(
      self, *, invocation_context: InvocationContext, event: Event
  ) -> Optional[Event]:
    del invocation_context, event
    self.counts.on_event_callback += 1
    return None

  async def after_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> None:
    del invocation_context
    self.counts.after_run_callback += 1
    return None


async def _terminal_node(ctx) -> Event:
  """A single terminal node that yields a content-bearing Event.

  Using content (not just state) ensures _consume_event_queue actually runs
  the on_event_callback path -- state-only events still flow through the
  queue, but content is the canonical case the plugin hook was designed for.
  """
  del ctx
  return Event(
      content=types.Content(
          parts=[types.Part(text='done')],
          role='model',
      )
  )


def _build_runner(plugin: TracerPlugin) -> Runner:
  workflow = Workflow(
      name='Issue5282Repro',
      edges=[('START', _terminal_node)],
  )
  app = App(name='issue_5282_repro', root_agent=workflow, plugins=[plugin])
  return Runner(
      app_name='issue_5282_repro',
      app=app,
      session_service=InMemorySessionService(),
      memory_service=InMemoryMemoryService(),
  )


async def _drive_one_invocation(runner: Runner) -> None:
  session = await runner.session_service.create_session(
      app_name='issue_5282_repro', user_id='u1'
  )
  async for _ in runner.run_async(
      user_id='u1',
      session_id=session.id,
      new_message=types.Content(parts=[types.Part(text='hi')], role='user'),
  ):
    pass


@pytest.mark.asyncio
async def test_workflow_root_dispatches_pre_run_and_event_hooks():
  """Baseline: pre-run and per-event hooks fire on a Workflow root."""
  plugin = TracerPlugin()
  runner = _build_runner(plugin)

  await _drive_one_invocation(runner)

  assert plugin.counts.on_user_message_callback == 1
  assert plugin.counts.before_run_callback == 1
  assert plugin.counts.on_event_callback >= 1, (
      'on_event_callback should fire at least once via _consume_event_queue '
      'for the content-bearing terminal event'
  )


@pytest.mark.asyncio
async def test_workflow_root_dispatches_after_run_callback():
  """Regression anchor for #5282: after_run_callback fires on Workflow roots.

  Pre-fix, _run_node_async never dispatched run_after_run_callback on the
  BaseNode path, so this would have asserted 0 and been xfail-anchored.
  Post-fix, the dispatch mirrors _exec_with_plugin's legacy path and fires
  exactly once per successful invocation.
  """
  plugin = TracerPlugin()
  runner = _build_runner(plugin)

  await _drive_one_invocation(runner)

  assert plugin.counts.after_run_callback == 1
