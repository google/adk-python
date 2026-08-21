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
from unittest.mock import patch

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.loop_agent import LoopAgent
from google.adk.agents.loop_agent import LoopAgentState
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.apps import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.event_actions import EscalationContext
from google.adk.events.event_actions import EventActions
from google.adk.sessions.in_memory_session_service import InMemorySessionService
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
async def test_run_async_with_pause_preserves_sub_agent_state(
    request: pytest.FixtureRequest,
):
  """Test that the sub-agent state is preserved when the loop agent pauses."""
  agent = _TestingAgent(name=f'{request.function.__name__}_test_agent')
  loop_agent = LoopAgent(
      name=f'{request.function.__name__}_test_loop_agent',
      max_iterations=2,
      sub_agents=[agent],
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, loop_agent, resumable=True
  )

  # Set some dummy state for the sub-agent
  parent_ctx.agent_states[agent.name] = {'some_key': 'some_value'}

  # Mock should_pause_invocation to return True for the agent's event
  def mock_should_pause(event):
    return event.author == agent.name

  with patch.object(
      InvocationContext,
      'should_pause_invocation',
      side_effect=mock_should_pause,
  ):
    async for _ in loop_agent.run_async(parent_ctx):
      pass  # Consume the async generator

  # Verify that the sub-agent state was NOT reset
  assert agent.name in parent_ctx.agent_states
  assert parent_ctx.agent_states[agent.name] == {'some_key': 'some_value'}


def test_deprecation_mentions_sub_agent_limitation():
  with pytest.warns(DeprecationWarning, match='sub-agent'):
    LoopAgent(name='deprecated_loop', sub_agents=[])


class _CountingAgent(BaseAgent):

  def __init__(self, name: str, bucket: list[str]):
    super().__init__(name=name)
    object.__setattr__(self, '_bucket', bucket)

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    self._bucket.append(self.name)
    yield Event(
        author=self.name,
        invocation_id=ctx.invocation_id,
        content=types.Content(parts=[types.Part(text=self.name)]),
    )


class _ParentEscalateAgent(BaseAgent):

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield Event(
        author=self.name,
        invocation_id=ctx.invocation_id,
        content=types.Content(parts=[types.Part(text='exit inner')]),
        actions=EventActions(
            escalate=True,
            escalation_context=EscalationContext(type='parent'),
        ),
    )


class _TargetedEscalateAgent(BaseAgent):

  def __init__(self, name: str, target_agent: str):
    super().__init__(name=name)
    object.__setattr__(self, '_target_agent', target_agent)

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield Event(
        author=self.name,
        invocation_id=ctx.invocation_id,
        content=types.Content(parts=[types.Part(text='exit named')]),
        actions=EventActions(
            escalate=True,
            escalation_context=EscalationContext(
                type='parent', target_agent=self._target_agent
            ),
        ),
    )


@pytest.mark.asyncio
async def test_nested_loop_root_escalate_stops_every_loop(
    request: pytest.FixtureRequest,
):
  """Bare escalate=True still exits inner and outer loops (issue #2808 default)."""
  inner_runs: list[str] = []
  outer_runs: list[str] = []
  inner = LoopAgent(
      name=f'{request.function.__name__}_inner',
      sub_agents=[
          _CountingAgent(f'{request.function.__name__}_inner_work', inner_runs),
          _TestingAgentWithEscalateAction(
              name=f'{request.function.__name__}_root_exit'
          ),
      ],
  )
  outer = LoopAgent(
      name=f'{request.function.__name__}_outer',
      max_iterations=3,
      sub_agents=[
          inner,
          _CountingAgent(f'{request.function.__name__}_outer_work', outer_runs),
      ],
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, outer
  )
  _ = [e async for e in outer.run_async(parent_ctx)]
  assert inner_runs == [f'{request.function.__name__}_inner_work']
  assert outer_runs == []


@pytest.mark.asyncio
async def test_nested_loop_parent_escalate_exits_only_inner(
    request: pytest.FixtureRequest,
):
  """EscalationContext(type='parent') lets the outer loop keep iterating."""
  inner_runs: list[str] = []
  outer_runs: list[str] = []
  inner_name = f'{request.function.__name__}_inner'
  inner = LoopAgent(
      name=inner_name,
      sub_agents=[
          _CountingAgent(f'{request.function.__name__}_inner_work', inner_runs),
          _ParentEscalateAgent(name=f'{request.function.__name__}_parent_exit'),
      ],
  )
  outer = LoopAgent(
      name=f'{request.function.__name__}_outer',
      max_iterations=3,
      sub_agents=[
          inner,
          _CountingAgent(f'{request.function.__name__}_outer_work', outer_runs),
      ],
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, outer
  )
  events = [e async for e in outer.run_async(parent_ctx)]

  assert inner_runs == [f'{request.function.__name__}_inner_work'] * 3
  assert outer_runs == [f'{request.function.__name__}_outer_work'] * 3
  parent_events = [
      event
      for event in events
      if event.actions.escalation_context
      and event.actions.escalation_context.type == 'parent'
  ]
  assert parent_events
  assert parent_events[0].actions.escalation_context.handled_by == [inner_name]


class _EscalateAfterNAgent(BaseAgent):
  """Escalates with parent scope after this instance has run ``limit`` times."""

  def __init__(self, name: str, limit: int):
    super().__init__(name=name)
    object.__setattr__(self, '_limit', limit)
    object.__setattr__(self, '_calls', 0)

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    self._calls += 1
    if self._calls >= self._limit:
      yield Event(
          author=self.name,
          invocation_id=ctx.invocation_id,
          content=types.Content(parts=[types.Part(text='break')]),
          actions=EventActions(
              escalate=True,
              escalation_context=EscalationContext(type='parent'),
          ),
      )
      return
    yield Event(
        author=self.name,
        invocation_id=ctx.invocation_id,
        content=types.Content(parts=[types.Part(text='continue')]),
    )


class _EscalateEveryNCountsAgent(BaseAgent):
  """Parent-escalates whenever ``bucket`` length is a multiple of ``n``."""

  def __init__(self, name: str, n: int, bucket: list[str]):
    super().__init__(name=name)
    object.__setattr__(self, '_n', n)
    object.__setattr__(self, '_bucket', bucket)

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    if self._bucket and len(self._bucket) % self._n == 0:
      yield Event(
          author=self.name,
          invocation_id=ctx.invocation_id,
          content=types.Content(parts=[types.Part(text='break inner')]),
          actions=EventActions(
              escalate=True,
              escalation_context=EscalationContext(type='parent'),
          ),
      )
      return
    yield Event(
        author=self.name,
        invocation_id=ctx.invocation_id,
        content=types.Content(parts=[types.Part(text='continue')]),
    )


@pytest.mark.asyncio
async def test_nested_loop_parent_escalate_runs_inner_times_outer(
    request: pytest.FixtureRequest,
):
  """Issue #2808: inner 5 × outer 5 should be 25 inner steps, not 5."""
  inner_runs: list[str] = []
  inner = LoopAgent(
      name=f'{request.function.__name__}_inner',
      sub_agents=[
          _CountingAgent(f'{request.function.__name__}_inner_work', inner_runs),
          _EscalateEveryNCountsAgent(
              name=f'{request.function.__name__}_inner_break',
              n=5,
              bucket=inner_runs,
          ),
      ],
  )
  outer = LoopAgent(
      name=f'{request.function.__name__}_outer',
      sub_agents=[
          inner,
          _EscalateAfterNAgent(
              name=f'{request.function.__name__}_outer_break', limit=5
          ),
      ],
  )
  root = SequentialAgent(
      name=f'{request.function.__name__}_root',
      sub_agents=[outer],
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, root
  )
  _ = [e async for e in root.run_async(parent_ctx)]
  assert inner_runs == [f'{request.function.__name__}_inner_work'] * 25


@pytest.mark.asyncio
async def test_nested_loop_target_agent_skips_inner(
    request: pytest.FixtureRequest,
):
  """target_agent on the outer loop leaves the inner loop running until max."""
  inner_runs: list[str] = []
  outer_name = f'{request.function.__name__}_outer'
  inner = LoopAgent(
      name=f'{request.function.__name__}_inner',
      max_iterations=2,
      sub_agents=[
          _CountingAgent(f'{request.function.__name__}_inner_work', inner_runs),
          _TargetedEscalateAgent(
              name=f'{request.function.__name__}_aim_outer',
              target_agent=outer_name,
          ),
      ],
  )
  outer = LoopAgent(
      name=outer_name,
      max_iterations=5,
      sub_agents=[inner],
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, outer
  )
  _ = [e async for e in outer.run_async(parent_ctx)]
  assert inner_runs == [f'{request.function.__name__}_inner_work'] * 2
