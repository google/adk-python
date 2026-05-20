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

"""Tests for Runner state_delta handling.

Verifies that state deltas supplied to Runner.run_async are applied to the
initial user event before NodeRunner executes nodes or agents.
"""

from __future__ import annotations

from typing import Any
from typing import AsyncGenerator

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.context import Context
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow._base_node import BaseNode
from google.genai import types
import pytest


def _user_message(text: str = "hello") -> types.Content:
  return types.Content(parts=[types.Part(text=text)], role="user")


@pytest.mark.asyncio
async def test_node_runner_applies_state_delta_before_base_node_runs():
  """A BaseNode sees run_async state_delta as session state."""

  class _StateReaderNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield f"state:{ctx.state['test_state']}"

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="test",
      node=_StateReaderNode(name="reader"),
      session_service=session_service,
  )
  session = await session_service.create_session(app_name="test", user_id="u")

  events: list[Event] = []
  async for event in runner.run_async(
      user_id="u",
      session_id=session.id,
      new_message=_user_message(),
      state_delta={"test_state": "must_change"},
  ):
    events.append(event)

  updated = await session_service.get_session(
      app_name="test", user_id="u", session_id=session.id
  )
  user_events = [event for event in updated.events if event.author == "user"]

  assert [event.output for event in events if event.output is not None] == [
      "state:must_change"
  ]
  assert updated.state["test_state"] == "must_change"
  assert user_events[0].actions.state_delta == {"test_state": "must_change"}


@pytest.mark.asyncio
async def test_node_runner_yields_user_event_with_state_delta():
  """yield_user_message=True yields the user event with state_delta."""

  class _NoopNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield "done"

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="test",
      node=_NoopNode(name="noop"),
      session_service=session_service,
  )
  session = await session_service.create_session(app_name="test", user_id="u")

  events: list[Event] = []
  async for event in runner.run_async(
      user_id="u",
      session_id=session.id,
      new_message=_user_message(),
      state_delta={"test_state": "must_change"},
      yield_user_message=True,
  ):
    events.append(event)

  assert events[0].author == "user"
  assert events[0].actions.state_delta == {"test_state": "must_change"}


@pytest.mark.asyncio
async def test_node_runner_applies_state_delta_before_llm_agent_runs():
  """An LlmAgent callback sees run_async state_delta before model execution."""

  captured_state_value = None

  def _before_agent_callback(
      callback_context: CallbackContext,
  ) -> types.Content:
    nonlocal captured_state_value
    captured_state_value = callback_context.state["test_state"]
    return types.Content(
        role="model",
        parts=[types.Part(text=f"state:{captured_state_value}")],
    )

  session_service = InMemorySessionService()
  agent = LlmAgent(
      name="state_agent",
      before_agent_callback=_before_agent_callback,
  )
  runner = Runner(app_name="test", agent=agent, session_service=session_service)
  session = await session_service.create_session(app_name="test", user_id="u")

  events: list[Event] = []
  async for event in runner.run_async(
      user_id="u",
      session_id=session.id,
      new_message=_user_message(),
      state_delta={"test_state": "must_change"},
  ):
    events.append(event)

  updated = await session_service.get_session(
      app_name="test", user_id="u", session_id=session.id
  )
  user_events = [event for event in updated.events if event.author == "user"]
  response_texts = [
      part.text
      for event in events
      if event.content
      for part in event.content.parts
      if part.text
  ]

  assert captured_state_value == "must_change"
  assert "state:must_change" in response_texts
  assert user_events[0].actions.state_delta == {"test_state": "must_change"}
