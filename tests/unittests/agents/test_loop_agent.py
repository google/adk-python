# Copyright 2025 Google LLC
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

"""Testings for the SequentialAgent."""

from typing import AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.loop_agent import LoopAgent
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
import pytest
from typing_extensions import override


class _TestingAgent(BaseAgent):

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield Event(
        author=self.name,
        invocation_id=ctx.invocation_id,
        content=types.Content(
            parts=[types.Part(text=f'Hello, async {self.name}!')]
        ),
    )

  @override
  async def _run_live_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield Event(
        author=self.name,
        invocation_id=ctx.invocation_id,
        content=types.Content(
            parts=[types.Part(text=f'Hello, live {self.name}!')]
        ),
    )


class _TestingAgentWithEscalateAction(BaseAgent):

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield Event(
        author=self.name,
        invocation_id=ctx.invocation_id,
        content=types.Content(
            parts=[types.Part(text=f'Hello, async {self.name}!')]
        ),
        actions=EventActions(escalate=True),
    )
    yield Event(
        author=self.name,
        invocation_id=ctx.invocation_id,
        content=types.Content(
            parts=[types.Part(text=f'I have done my job after escalation!!')]
        ),
    )


async def _create_parent_invocation_context(
    test_name: str, agent: BaseAgent
) -> InvocationContext:
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )
  return InvocationContext(
      invocation_id=f'{test_name}_invocation_id',
      agent=agent,
      session=session,
      session_service=session_service,
  )


@pytest.mark.asyncio
async def test_run_async(request: pytest.FixtureRequest):
  agent = _TestingAgent(name=f'{request.function.__name__}_test_agent')
  loop_agent = LoopAgent(
      name=f'{request.function.__name__}_test_loop_agent',
      max_iterations=2,
      sub_agents=[
          agent,
      ],
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, loop_agent
  )
  events = [e async for e in loop_agent.run_async(parent_ctx)]

  assert len(events) == 2
  assert events[0].author == agent.name
  assert events[1].author == agent.name
  assert events[0].content.parts[0].text == f'Hello, async {agent.name}!'
  assert events[1].content.parts[0].text == f'Hello, async {agent.name}!'


@pytest.mark.asyncio
async def test_run_async_with_escalate_action(request: pytest.FixtureRequest):
  non_escalating_agent = _TestingAgent(
      name=f'{request.function.__name__}_test_non_escalating_agent'
  )
  escalating_agent = _TestingAgentWithEscalateAction(
      name=f'{request.function.__name__}_test_escalating_agent'
  )
  ignored_agent = _TestingAgent(
      name=f'{request.function.__name__}_test_ignored_agent'
  )
  loop_agent = LoopAgent(
      name=f'{request.function.__name__}_test_loop_agent',
      sub_agents=[non_escalating_agent, escalating_agent, ignored_agent],
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, loop_agent
  )
  events = [e async for e in loop_agent.run_async(parent_ctx)]

  # Only two events are generated because the sub escalating_agent escalates.
  assert len(events) == 3
  assert events[0].author == non_escalating_agent.name
  assert events[1].author == escalating_agent.name
  assert events[0].content.parts[0].text == (
      f'Hello, async {non_escalating_agent.name}!'
  )
  assert events[1].content.parts[0].text == (
      f'Hello, async {escalating_agent.name}!'
  )
  assert (
      events[2].content.parts[0].text == 'I have done my job after escalation!!'
  )
