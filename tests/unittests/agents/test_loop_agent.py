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

"""Testings for the SequentialAgent."""

from typing import AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.loop_agent import LoopAgent
from google.adk.agents.loop_agent import LoopAgentState
from google.adk.apps import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.termination.max_iterations_termination import MaxIterationsTermination
from google.adk.termination.text_mention_termination import TextMentionTermination
from google.genai import types
import pytest
from typing_extensions import override

from .. import testing_utils

END_OF_AGENT = testing_utils.END_OF_AGENT


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
            parts=[types.Part(text='I have done my job after escalation!!')]
        ),
    )


async def _create_parent_invocation_context(
    test_name: str, agent: BaseAgent, resumable: bool = False
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
      resumability_config=ResumabilityConfig(is_resumable=resumable),
  )


@pytest.mark.asyncio
@pytest.mark.parametrize('resumable', [True, False])
async def test_run_async(request: pytest.FixtureRequest, resumable: bool):
  agent = _TestingAgent(name=f'{request.function.__name__}_test_agent')
  loop_agent = LoopAgent(
      name=f'{request.function.__name__}_test_loop_agent',
      max_iterations=2,
      sub_agents=[
          agent,
      ],
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, loop_agent, resumable=resumable
  )
  events = [e async for e in loop_agent.run_async(parent_ctx)]

  simplified_events = testing_utils.simplify_resumable_app_events(events)
  if resumable:
    expected_events = [
        (
            loop_agent.name,
            {'current_sub_agent': agent.name, 'times_looped': 0},
        ),
        (agent.name, f'Hello, async {agent.name}!'),
        (
            loop_agent.name,
            {'current_sub_agent': agent.name, 'times_looped': 1},
        ),
        (agent.name, f'Hello, async {agent.name}!'),
        (loop_agent.name, END_OF_AGENT),
    ]
  else:
    expected_events = [
        (agent.name, f'Hello, async {agent.name}!'),
        (agent.name, f'Hello, async {agent.name}!'),
    ]
  assert simplified_events == expected_events


@pytest.mark.asyncio
async def test_resume_async(request: pytest.FixtureRequest):
  agent_1 = _TestingAgent(name=f'{request.function.__name__}_test_agent_1')
  agent_2 = _TestingAgent(name=f'{request.function.__name__}_test_agent_2')
  loop_agent = LoopAgent(
      name=f'{request.function.__name__}_test_loop_agent',
      max_iterations=2,
      sub_agents=[
          agent_1,
          agent_2,
      ],
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, loop_agent, resumable=True
  )
  parent_ctx.agent_states[loop_agent.name] = LoopAgentState(
      current_sub_agent=agent_2.name, times_looped=1
  ).model_dump(mode='json')

  events = [e async for e in loop_agent.run_async(parent_ctx)]

  simplified_events = testing_utils.simplify_resumable_app_events(events)
  expected_events = [
      (agent_2.name, f'Hello, async {agent_2.name}!'),
      (loop_agent.name, END_OF_AGENT),
  ]
  assert simplified_events == expected_events


@pytest.mark.asyncio
async def test_run_async_skip_if_no_sub_agent(request: pytest.FixtureRequest):
  loop_agent = LoopAgent(
      name=f'{request.function.__name__}_test_loop_agent',
      max_iterations=2,
      sub_agents=[],
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, loop_agent
  )
  events = [e async for e in loop_agent.run_async(parent_ctx)]
  assert not events


@pytest.mark.asyncio
@pytest.mark.parametrize('resumable', [True, False])
async def test_run_async_with_escalate_action(
    request: pytest.FixtureRequest, resumable: bool
):
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
      request.function.__name__, loop_agent, resumable=resumable
  )
  events = [e async for e in loop_agent.run_async(parent_ctx)]

  simplified_events = testing_utils.simplify_resumable_app_events(events)

  if resumable:
    expected_events = [
        (
            loop_agent.name,
            {
                'current_sub_agent': non_escalating_agent.name,
                'times_looped': 0,
            },
        ),
        (
            non_escalating_agent.name,
            f'Hello, async {non_escalating_agent.name}!',
        ),
        (
            loop_agent.name,
            {'current_sub_agent': escalating_agent.name, 'times_looped': 0},
        ),
        (
            escalating_agent.name,
            f'Hello, async {escalating_agent.name}!',
        ),
        (
            escalating_agent.name,
            'I have done my job after escalation!!',
        ),
        (loop_agent.name, END_OF_AGENT),
    ]
  else:
    expected_events = [
        (
            non_escalating_agent.name,
            f'Hello, async {non_escalating_agent.name}!',
        ),
        (
            escalating_agent.name,
            f'Hello, async {escalating_agent.name}!',
        ),
        (
            escalating_agent.name,
            'I have done my job after escalation!!',
        ),
    ]
  assert simplified_events == expected_events


@pytest.mark.asyncio
async def test_run_async_with_termination_condition_stops_loop(
    request: pytest.FixtureRequest,
):
  """Termination condition fires after first event and stops the loop."""
  agent = _TestingAgent(name=f'{request.function.__name__}_agent')
  loop_agent = LoopAgent(
      name=f'{request.function.__name__}_loop',
      max_iterations=5,
      sub_agents=[agent],
      termination_condition=MaxIterationsTermination(1),
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, loop_agent
  )

  events = [e async for e in loop_agent.run_async(parent_ctx)]

  # The sub-agent emits one text event, then the loop emits a termination event.
  assert len(events) == 2
  text_event, termination_event = events
  assert text_event.author == agent.name
  assert termination_event.author == loop_agent.name
  assert termination_event.actions.escalate is True
  assert termination_event.actions.termination_reason is not None


@pytest.mark.asyncio
async def test_run_async_with_termination_condition_text_mention(
    request: pytest.FixtureRequest,
):
  """TextMentionTermination fires when the keyword appears in an event."""
  agent = _TestingAgent(name=f'{request.function.__name__}_agent')
  loop_agent = LoopAgent(
      name=f'{request.function.__name__}_loop',
      max_iterations=10,
      sub_agents=[agent],
      # 'Hello' appears in the sub-agent's response, so it should fire at once.
      termination_condition=TextMentionTermination('Hello'),
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, loop_agent
  )

  events = [e async for e in loop_agent.run_async(parent_ctx)]

  termination_event = events[-1]
  assert termination_event.author == loop_agent.name
  assert termination_event.actions.escalate is True
  assert termination_event.actions.termination_reason is not None


@pytest.mark.asyncio
async def test_run_async_termination_condition_resets_between_runs(
    request: pytest.FixtureRequest,
):
  """The termination condition is reset at the start of each run."""
  agent = _TestingAgent(name=f'{request.function.__name__}_agent')
  condition = MaxIterationsTermination(1)
  loop_agent = LoopAgent(
      name=f'{request.function.__name__}_loop',
      max_iterations=5,
      sub_agents=[agent],
      termination_condition=condition,
  )

  # First run – condition fires and gets reset automatically.
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, loop_agent
  )
  events_first = [e async for e in loop_agent.run_async(parent_ctx)]
  assert events_first[-1].actions.termination_reason is not None

  # Second run – condition must have been reset; should fire again identically.
  parent_ctx2 = await _create_parent_invocation_context(
      request.function.__name__ + '_2', loop_agent
  )
  events_second = [e async for e in loop_agent.run_async(parent_ctx2)]
  assert events_second[-1].actions.termination_reason is not None
