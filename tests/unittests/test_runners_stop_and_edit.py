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

"""Stop a running agent and edit a previous prompt (#3849)."""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
import pytest

TEST_APP_ID = "test_app"
TEST_USER_ID = "test_user"
TEST_SESSION_ID = "test_session"


def _user_message(text: str) -> types.Content:
  return types.Content(role="user", parts=[types.Part(text=text)])


def _user_text(invocation_context: InvocationContext) -> str:
  if invocation_context.user_content and invocation_context.user_content.parts:
    return invocation_context.user_content.parts[0].text or ""
  return ""


def _make_slow_agent(kind: str) -> tuple[BaseAgent, asyncio.Event]:
  """kind is 'base' (legacy path) or 'llm' (node path)."""
  started = asyncio.Event()

  class SlowBaseAgent(BaseAgent):

    async def _run_async_impl(
        self, invocation_context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
      yield Event(
          invocation_id=invocation_context.invocation_id,
          author=self.name,
          content=types.Content(
              role="model", parts=[types.Part(text="thinking")]
          ),
      )
      started.set()
      try:
        await asyncio.sleep(30)
        yield Event(
            invocation_id=invocation_context.invocation_id,
            author=self.name,
            content=types.Content(
                role="model", parts=[types.Part(text="should not appear")]
            ),
        )
      except (asyncio.CancelledError, GeneratorExit):
        raise

  class SlowLlmAgent(LlmAgent):

    async def _run_async_impl(
        self, invocation_context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
      yield Event(
          invocation_id=invocation_context.invocation_id,
          author=self.name,
          content=types.Content(
              role="model", parts=[types.Part(text="thinking")]
          ),
      )
      started.set()
      try:
        await asyncio.sleep(30)
        yield Event(
            invocation_id=invocation_context.invocation_id,
            author=self.name,
            content=types.Content(
                role="model", parts=[types.Part(text="should not appear")]
            ),
        )
      except (asyncio.CancelledError, GeneratorExit):
        raise

  if kind == "llm":
    return SlowLlmAgent(name="slow_agent", model="gemini-1.5-pro"), started
  return SlowBaseAgent(name="slow_agent"), started


def _make_slow_then_echo_agent() -> tuple[BaseAgent, asyncio.Event]:
  started = asyncio.Event()
  calls = {"n": 0}

  class SlowThenEchoAgent(BaseAgent):

    async def _run_async_impl(
        self, invocation_context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
      calls["n"] += 1
      text = _user_text(invocation_context)
      if calls["n"] == 1:
        yield Event(
            invocation_id=invocation_context.invocation_id,
            author=self.name,
            content=types.Content(
                role="model", parts=[types.Part(text="thinking")]
            ),
        )
        started.set()
        await asyncio.sleep(30)
        return
      yield Event(
          invocation_id=invocation_context.invocation_id,
          author=self.name,
          content=types.Content(
              role="model", parts=[types.Part(text=f"echo:{text}")]
          ),
      )

  return SlowThenEchoAgent(name="slow_echo"), started


class EchoAgent(BaseAgent):
  """Replies with the latest user text so edits are observable."""

  def __init__(self, name: str):
    super().__init__(name=name, sub_agents=[])

  async def _run_async_impl(
      self, invocation_context: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    text = ""
    if (
        invocation_context.user_content
        and invocation_context.user_content.parts
    ):
      text = invocation_context.user_content.parts[0].text or ""
    yield Event(
        invocation_id=invocation_context.invocation_id,
        author=self.name,
        content=types.Content(
            role="model", parts=[types.Part(text=f"echo:{text}")]
        ),
    )


def _runner(agent: BaseAgent) -> tuple[Runner, InMemorySessionService]:
  session_service = InMemorySessionService()
  runner = Runner(
      app_name=TEST_APP_ID,
      agent=agent,
      session_service=session_service,
      artifact_service=InMemoryArtifactService(),
      auto_create_session=True,
  )
  return runner, session_service


async def _drain(agen) -> list[Event]:
  events: list[Event] = []
  try:
    async for event in agen:
      events.append(event)
  except asyncio.CancelledError:
    pass
  return events


@pytest.mark.parametrize("kind", ["base", "llm"])
async def test_cancel_async_stops_a_running_agent(kind):
  """cancel_async aborts a blocked agent and records interrupted=True."""
  agent, started = _make_slow_agent(kind)
  runner, session_service = _runner(agent)
  first_invocation = asyncio.get_running_loop().create_future()

  async def _run() -> list[Event]:
    events: list[Event] = []
    async for event in runner.run_async(
        user_id=TEST_USER_ID,
        session_id=TEST_SESSION_ID,
        new_message=_user_message("go"),
    ):
      events.append(event)
      if not first_invocation.done() and event.invocation_id:
        first_invocation.set_result(event.invocation_id)
    return events

  run_task = asyncio.create_task(_run())
  await asyncio.wait_for(started.wait(), timeout=5)
  cancelled_ids = await runner.cancel_async(
      user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
  )
  try:
    run_events = await asyncio.wait_for(run_task, timeout=5)
  except asyncio.CancelledError:
    run_events = []

  assert cancelled_ids
  assert first_invocation.result() in cancelled_ids
  session = await session_service.get_session(
      app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
  )
  assert session is not None
  assert any(event.interrupted for event in session.events)
  assert all(
      event.content is None
      or not event.content.parts
      or event.content.parts[0].text != "should not appear"
      for event in run_events + session.events
  )


async def test_cancel_async_is_idempotent_when_nothing_is_running():
  """Stopping a session with no live run returns an empty list."""
  runner, _ = _runner(EchoAgent("echo"))
  async for _ in runner.run_async(
      user_id=TEST_USER_ID,
      session_id=TEST_SESSION_ID,
      new_message=_user_message("done"),
  ):
    pass

  cancelled_ids = await runner.cancel_async(
      user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
  )
  assert cancelled_ids == []


async def test_edit_message_async_regenerates_from_the_edited_prompt():
  """Editing a prior user turn rewinds history and reruns with the new text."""
  runner, session_service = _runner(EchoAgent("echo"))

  async for _ in runner.run_async(
      user_id=TEST_USER_ID,
      session_id=TEST_SESSION_ID,
      new_message=_user_message("first"),
  ):
    pass
  session = await session_service.get_session(
      app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
  )
  assert session is not None
  first_invocation_id = session.events[0].invocation_id

  async for _ in runner.run_async(
      user_id=TEST_USER_ID,
      session_id=TEST_SESSION_ID,
      new_message=_user_message("second"),
  ):
    pass

  edited = []
  async for event in runner.edit_message_async(
      user_id=TEST_USER_ID,
      session_id=TEST_SESSION_ID,
      invocation_id=first_invocation_id,
      new_message=_user_message("first edited"),
  ):
    edited.append(event)

  assert any(
      event.content
      and event.content.parts
      and event.content.parts[0].text == "echo:first edited"
      for event in edited
  )

  session = await session_service.get_session(
      app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
  )
  assert session is not None
  assert any(
      event.actions.rewind_before_invocation_id == first_invocation_id
      for event in session.events
  )
  user_texts = [
      event.content.parts[0].text
      for event in session.events
      if event.author == "user"
      and event.content
      and event.content.parts
      and event.content.parts[0].text
  ]
  assert "first edited" in user_texts


async def test_edit_message_async_cancels_a_live_turn_then_regenerates():
  """Editing the in-flight prompt stops it, then reruns with the new text."""
  agent, started = _make_slow_then_echo_agent()
  runner, session_service = _runner(agent)
  first_invocation = asyncio.get_running_loop().create_future()

  async def _run() -> None:
    async for event in runner.run_async(
        user_id=TEST_USER_ID,
        session_id=TEST_SESSION_ID,
        new_message=_user_message("original"),
    ):
      if not first_invocation.done() and event.invocation_id:
        first_invocation.set_result(event.invocation_id)

  run_task = asyncio.create_task(_run())
  await asyncio.wait_for(started.wait(), timeout=5)
  live_invocation_id = await first_invocation

  edited_events = await asyncio.wait_for(
      _drain(
          runner.edit_message_async(
              user_id=TEST_USER_ID,
              session_id=TEST_SESSION_ID,
              invocation_id=live_invocation_id,
              new_message=_user_message("edited live"),
          )
      ),
      timeout=5,
  )
  try:
    await asyncio.wait_for(run_task, timeout=5)
  except asyncio.CancelledError:
    pass

  session = await session_service.get_session(
      app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
  )
  assert session is not None
  assert any(event.interrupted for event in session.events)
  assert any(
      event.actions.rewind_before_invocation_id == live_invocation_id
      for event in session.events
  )
  assert any(
      event.content
      and event.content.parts
      and event.content.parts[0].text == "echo:edited live"
      for event in edited_events
  )


async def test_edit_message_async_rejects_unknown_invocation():
  """Editing a turn that does not exist raises ValueError."""
  runner, _ = _runner(EchoAgent("echo"))
  async for _ in runner.run_async(
      user_id=TEST_USER_ID,
      session_id=TEST_SESSION_ID,
      new_message=_user_message("hello"),
  ):
    pass

  with pytest.raises(ValueError, match="No user message found"):
    agen = runner.edit_message_async(
        user_id=TEST_USER_ID,
        session_id=TEST_SESSION_ID,
        invocation_id="missing-inv",
        new_message=_user_message("nope"),
    )
    await agen.__anext__()
