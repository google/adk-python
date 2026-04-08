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

"""Tests for Runner.run_async with termination_condition."""

from __future__ import annotations

from typing import AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.termination.max_iterations_termination import MaxIterationsTermination
from google.adk.termination.text_mention_termination import TextMentionTermination
from google.genai import types
import pytest
from typing_extensions import override


class _TextAgent(BaseAgent):
  """A simple agent that yields a fixed text event and then stops."""

  text: str = 'hello'

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield Event(
        invocation_id=ctx.invocation_id,
        author=self.name,
        content=types.Content(
            role='model',
            parts=[types.Part(text=self.text)],
        ),
    )


async def _run_with_termination(
    agent: BaseAgent,
    termination_condition=None,
) -> list[Event]:
  """Creates a runner with an in-memory session and collects all events."""
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )
  runner = Runner(
      app_name='test_app',
      agent=agent,
      session_service=session_service,
      artifact_service=InMemoryArtifactService(),
  )
  events = []
  async for event in runner.run_async(
      user_id=session.user_id,
      session_id=session.id,
      new_message=types.Content(role='user', parts=[types.Part(text='go')]),
      termination_condition=termination_condition,
  ):
    events.append(event)
  return events


@pytest.mark.asyncio
async def test_run_async_without_termination_condition():
  """Baseline: runner emits the agent event with no condition attached."""
  agent = _TextAgent(name='agent')
  events = await _run_with_termination(agent)
  assert any(e.author == 'agent' for e in events)
  assert not any(
      e.actions.termination_reason for e in events
  ), 'No termination event should exist without a condition'


@pytest.mark.asyncio
async def test_run_async_termination_condition_stops_run():
  """Termination condition fires after the first event and stops the run."""
  agent = _TextAgent(name='agent', text='hello world')
  condition = MaxIterationsTermination(1)

  events = await _run_with_termination(agent, termination_condition=condition)

  # The last event must be a synthetic termination event.
  termination_event = events[-1]
  assert termination_event.actions.escalate is True
  assert termination_event.actions.termination_reason is not None


@pytest.mark.asyncio
async def test_run_async_text_mention_termination():
  """TextMentionTermination fires when the keyword is found in an event."""
  agent = _TextAgent(name='agent', text='STOP now')
  condition = TextMentionTermination('STOP')

  events = await _run_with_termination(agent, termination_condition=condition)

  termination_event = events[-1]
  assert termination_event.actions.escalate is True
  assert 'STOP' in termination_event.actions.termination_reason


@pytest.mark.asyncio
async def test_run_async_text_mention_termination_keyword_absent():
  """Run completes normally when the keyword is not present in any event."""
  agent = _TextAgent(name='agent', text='everything is fine')
  condition = TextMentionTermination('STOP')

  events = await _run_with_termination(agent, termination_condition=condition)

  assert not any(
      e.actions.termination_reason for e in events
  ), 'No termination event expected when keyword is absent'


@pytest.mark.asyncio
async def test_run_async_termination_condition_resets_between_runs():
  """The condition is reset at the start of each run_async call."""
  agent = _TextAgent(name='agent', text='hello')
  condition = MaxIterationsTermination(1)

  session_service = InMemorySessionService()
  runner = Runner(
      app_name='test_app',
      agent=agent,
      session_service=session_service,
      artifact_service=InMemoryArtifactService(),
  )

  for _ in range(2):
    session = await session_service.create_session(
        app_name='test_app', user_id='test_user'
    )
    events = []
    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=types.Content(role='user', parts=[types.Part(text='go')]),
        termination_condition=condition,
    ):
      events.append(event)

    # Each run should emit a termination event, proving the reset happened.
    assert (
        events[-1].actions.termination_reason is not None
    ), 'Expected a termination event on every run'
