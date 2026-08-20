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

"""Tests for the ParallelAgent."""

import asyncio
from types import SimpleNamespace
from typing import AsyncGenerator
from unittest.mock import patch

from google.adk.agents import parallel_agent as parallel_agent_module
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.base_agent import BaseAgentState
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.parallel_agent import _merge_agent_run_pre_3_11
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.sequential_agent import SequentialAgentState
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.telemetry.node_tracing import TelemetryContext
from google.genai import types
from opentelemetry import context as context_api
import pytest
from typing_extensions import override


class _TestingAgent(BaseAgent):

  delay: float = 0
  """The delay before the agent generates an event."""

  def event(
      self,
      ctx: InvocationContext,
      *,
      text: str | None = None,
      actions: EventActions | None = None,
  ):
    return Event(
        author=self.name,
        branch=ctx.branch,
        invocation_id=ctx.invocation_id,
        content=types.Content(
            parts=[types.Part(text=text or f'Hello, async {self.name}!')]
        ),
        actions=actions if actions is not None else EventActions(),
    )

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    await asyncio.sleep(self.delay)
    yield self.event(ctx)
    if ctx.is_resumable:
      ctx.set_agent_state(self.name, end_of_agent=True)


async def _create_parent_invocation_context(
    test_name: str, agent: BaseAgent, is_resumable: bool = False
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
      resumability_config=ResumabilityConfig(is_resumable=is_resumable),
  )


@pytest.mark.asyncio
@pytest.mark.parametrize('is_resumable', [True, False])
async def test_run_async(request: pytest.FixtureRequest, is_resumable: bool):
  agent1 = _TestingAgent(
      name=f'{request.function.__name__}_test_agent_1',
      delay=0.5,
  )
  agent2 = _TestingAgent(name=f'{request.function.__name__}_test_agent_2')
  parallel_agent = ParallelAgent(
      name=f'{request.function.__name__}_test_parallel_agent',
      sub_agents=[
          agent1,
          agent2,
      ],
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, parallel_agent, is_resumable=is_resumable
  )
  events = [e async for e in parallel_agent.run_async(parent_ctx)]

  if is_resumable:
    assert len(events) == 4

    assert events[0].author == parallel_agent.name
    assert not events[0].actions.end_of_agent

    # agent2 generates an event first, then agent1. Because they run in parallel
    # and agent1 has a delay.
    assert events[1].author == agent2.name
    assert events[2].author == agent1.name
    assert events[1].branch == f'{parallel_agent.name}.{agent2.name}'
    assert events[2].branch == f'{parallel_agent.name}.{agent1.name}'
    assert events[1].content.parts[0].text == f'Hello, async {agent2.name}!'
    assert events[2].content.parts[0].text == f'Hello, async {agent1.name}!'

    assert events[3].author == parallel_agent.name
    assert events[3].actions.end_of_agent
  else:
    assert len(events) == 2

    assert events[0].author == agent2.name
    assert events[1].author == agent1.name
    assert events[0].branch == f'{parallel_agent.name}.{agent2.name}'
    assert events[1].branch == f'{parallel_agent.name}.{agent1.name}'
    assert events[0].content.parts[0].text == f'Hello, async {agent2.name}!'
    assert events[1].content.parts[0].text == f'Hello, async {agent1.name}!'


@pytest.mark.asyncio
@pytest.mark.parametrize('is_resumable', [True, False])
async def test_run_async_branches(
    request: pytest.FixtureRequest, is_resumable: bool
):
  agent1 = _TestingAgent(
      name=f'{request.function.__name__}_test_agent_1',
      delay=0.5,
  )
  agent2 = _TestingAgent(name=f'{request.function.__name__}_test_agent_2')
  agent3 = _TestingAgent(name=f'{request.function.__name__}_test_agent_3')
  sequential_agent = SequentialAgent(
      name=f'{request.function.__name__}_test_sequential_agent',
      sub_agents=[agent2, agent3],
  )
  parallel_agent = ParallelAgent(
      name=f'{request.function.__name__}_test_parallel_agent',
      sub_agents=[
          sequential_agent,
          agent1,
      ],
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, parallel_agent, is_resumable=is_resumable
  )
  events = [e async for e in parallel_agent.run_async(parent_ctx)]

  if is_resumable:
    assert len(events) == 8

    # 1. parallel agent checkpoint
    assert events[0].author == parallel_agent.name
    assert not events[0].actions.end_of_agent

    # 2. sequential agent checkpoint
    assert events[1].author == sequential_agent.name
    assert not events[1].actions.end_of_agent
    assert events[1].actions.agent_state['current_sub_agent'] == agent2.name
    assert events[1].branch == f'{parallel_agent.name}.{sequential_agent.name}'

    # 3. agent 2 event
    assert events[2].author == agent2.name
    assert events[2].branch == f'{parallel_agent.name}.{sequential_agent.name}'

    # 4. sequential agent checkpoint
    assert events[3].author == sequential_agent.name
    assert not events[3].actions.end_of_agent
    assert events[3].actions.agent_state['current_sub_agent'] == agent3.name
    assert events[3].branch == f'{parallel_agent.name}.{sequential_agent.name}'

    # 5. agent 3 event
    assert events[4].author == agent3.name
    assert events[4].branch == f'{parallel_agent.name}.{sequential_agent.name}'

    # 6. sequential agent checkpoint (end)
    assert events[5].author == sequential_agent.name
    assert events[5].actions.end_of_agent
    assert events[5].branch == f'{parallel_agent.name}.{sequential_agent.name}'

    # Descendants of the same sub-agent should have the same branch.
    assert events[1].branch == events[2].branch
    assert events[2].branch == events[3].branch
    assert events[3].branch == events[4].branch
    assert events[4].branch == events[5].branch

    # 7. agent 1 event
    assert events[6].author == agent1.name
    assert events[6].branch == f'{parallel_agent.name}.{agent1.name}'

    # Sub-agents should have different branches.
    assert events[6].branch != events[1].branch

    # 8. parallel agent checkpoint (end)
    assert events[7].author == parallel_agent.name
    assert events[7].actions.end_of_agent
  else:
    assert len(events) == 3

    # 1. agent 2 event
    assert events[0].author == agent2.name
    assert events[0].branch == f'{parallel_agent.name}.{sequential_agent.name}'

    # 2. agent 3 event
    assert events[1].author == agent3.name
    assert events[1].branch == f'{parallel_agent.name}.{sequential_agent.name}'

    # 3. agent 1 event
    assert events[2].author == agent1.name
    assert events[2].branch == f'{parallel_agent.name}.{agent1.name}'


@pytest.mark.asyncio
async def test_resume_async_branches(request: pytest.FixtureRequest):
  agent1 = _TestingAgent(
      name=f'{request.function.__name__}_test_agent_1', delay=0.5
  )
  agent2 = _TestingAgent(name=f'{request.function.__name__}_test_agent_2')
  agent3 = _TestingAgent(name=f'{request.function.__name__}_test_agent_3')
  sequential_agent = SequentialAgent(
      name=f'{request.function.__name__}_test_sequential_agent',
      sub_agents=[agent2, agent3],
  )
  parallel_agent = ParallelAgent(
      name=f'{request.function.__name__}_test_parallel_agent',
      sub_agents=[
          sequential_agent,
          agent1,
      ],
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, parallel_agent, is_resumable=True
  )
  parent_ctx.agent_states[parallel_agent.name] = BaseAgentState().model_dump(
      mode='json'
  )
  parent_ctx.agent_states[sequential_agent.name] = SequentialAgentState(
      current_sub_agent=agent3.name
  ).model_dump(mode='json')

  events = [e async for e in parallel_agent.run_async(parent_ctx)]

  assert len(events) == 4

  # The sequential agent resumes from agent3.
  # 1. Agent 3 event
  assert events[0].author == agent3.name
  assert events[0].branch == f'{parallel_agent.name}.{sequential_agent.name}'

  # 2. Sequential agent checkpoint (end)
  assert events[1].author == sequential_agent.name
  assert events[1].actions.end_of_agent
  assert events[1].branch == f'{parallel_agent.name}.{sequential_agent.name}'

  # Agent 1 runs in parallel but has a delay.
  # 3. Agent 1 event
  assert events[2].author == agent1.name
  assert events[2].branch == f'{parallel_agent.name}.{agent1.name}'

  # 4. Parallel agent checkpoint (end)
  assert events[3].author == parallel_agent.name
  assert events[3].actions.end_of_agent


class _TestingAgentWithMultipleEvents(_TestingAgent):
  """Mock agent for testing."""

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    for _ in range(0, 3):
      event = self.event(ctx)
      yield event
      # Check that the event was processed by the consumer.
      assert event.custom_metadata is not None
      assert event.custom_metadata['processed']


@pytest.mark.asyncio
async def test_generating_one_event_per_agent_at_once(
    request: pytest.FixtureRequest,
):
  # This test is to verify that the parallel agent won't generate more than one
  # event per agent at a time.
  agent1 = _TestingAgentWithMultipleEvents(
      name=f'{request.function.__name__}_test_agent_1'
  )
  agent2 = _TestingAgentWithMultipleEvents(
      name=f'{request.function.__name__}_test_agent_2'
  )
  parallel_agent = ParallelAgent(
      name=f'{request.function.__name__}_test_parallel_agent',
      sub_agents=[
          agent1,
          agent2,
      ],
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, parallel_agent
  )

  agen = parallel_agent.run_async(parent_ctx)
  async for event in agen:
    event.custom_metadata = {'processed': True}
    # Asserts on event are done in _TestingAgentWithMultipleEvents.


@pytest.mark.asyncio
async def test_run_async_skip_if_no_sub_agent(request: pytest.FixtureRequest):
  parallel_agent = ParallelAgent(
      name=f'{request.function.__name__}_test_parallel_agent',
      sub_agents=[],
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, parallel_agent
  )
  events = [e async for e in parallel_agent.run_async(parent_ctx)]
  assert not events


class _TestingAgentWithException(_TestingAgent):
  """Mock agent for testing."""

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield self.event(ctx)
    raise Exception()


class _TestingAgentInfiniteEvents(_TestingAgent):
  """Mock agent for testing."""

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    while True:
      yield self.event(ctx)


@pytest.mark.asyncio
async def test_stop_agent_if_sub_agent_fails(
    request: pytest.FixtureRequest,
):
  # This test is to verify that the parallel agent and subagents will all stop
  # processing and throw exception to top level runner in case of exception.
  agent1 = _TestingAgentWithException(
      name=f'{request.function.__name__}_test_agent_1'
  )
  agent2 = _TestingAgentInfiniteEvents(
      name=f'{request.function.__name__}_test_agent_2'
  )
  parallel_agent = ParallelAgent(
      name=f'{request.function.__name__}_test_parallel_agent',
      sub_agents=[
          agent1,
          agent2,
      ],
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, parallel_agent
  )

  agen = parallel_agent.run_async(parent_ctx)
  # We expect to receive an exception from one of subagents.
  # The exception should be propagated to root agent and other subagents.
  # Otherwise we'll have an infinite loop.
  with pytest.raises(Exception):
    async for _ in agen:
      # The infinite agent could iterate a few times depending on scheduling.
      pass


async def _slow_agent_with_cleanup_delay():
  """Async generator that sleeps in its finally block to simulate cleanup."""
  try:
    await asyncio.sleep(10)
    yield 'slow-event'
  finally:
    await asyncio.sleep(0.05)


async def _failing_agent():
  """Async generator that raises after a short delay."""
  await asyncio.sleep(0.01)
  raise ValueError('simulated sub-agent failure')
  yield  # pragma: no cover


@pytest.mark.asyncio
async def test_merge_agent_run_pre_3_11_no_aclose_error_on_failure():
  """Regression test for Python 3.10 RuntimeError: aclose() already running.

  _merge_agent_run_pre_3_11 must await all cancelled tasks before returning so
  that generators are fully released before the caller invokes aclose() on them.
  """
  agent_runs = [_slow_agent_with_cleanup_delay(), _failing_agent()]

  with pytest.raises(ValueError, match='simulated sub-agent failure'):
    async for _ in _merge_agent_run_pre_3_11(agent_runs):
      pass

  # If tasks were not properly awaited, aclose() on a still-running generator
  # would raise RuntimeError here.
  for agen in agent_runs:
    await agen.aclose()


def test_deprecation_mentions_sub_agent_limitation():
  with pytest.warns(DeprecationWarning, match='sub-agent'):
    ParallelAgent(name='deprecated_parallel', sub_agents=[])


class _TestingAgentWithOpenTelemetryContext(_TestingAgent):
  """Mock agent for testing opentelemetry contextvars termination."""

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    token = context_api.attach(context_api.set_value('test_key', 'test_val'))
    try:
      yield self.event(ctx)
      yield self.event(ctx)
    finally:
      context_api.detach(token)


@pytest.mark.asyncio
async def test_parallel_agent_early_exit_context_cleanup(caplog):
  """Verify that early breaking out of a ParallelAgent run doesn't crash contextvars."""
  agent1 = _TestingAgentWithOpenTelemetryContext(name='otel_agent_1')
  parallel_agent = ParallelAgent(
      name='otel_parallel_agent',
      sub_agents=[agent1],
  )
  parent_ctx = await _create_parent_invocation_context(
      'otel_test', parallel_agent
  )

  agen = parallel_agent.run_async(parent_ctx)
  # Break early to trigger GeneratorExit and cleanup
  async for _ in agen:
    break

  # Check it exited successfully without logging open telemetry detachment errors
  # "ValueError: Token was created in a different Context" usually logged by opentelemetry.
  assert 'Failed to detach context' not in caplog.text
  assert 'test_key' not in context_api.get_current()


@pytest.mark.asyncio
@pytest.mark.parametrize('is_resumable', [True, False])
@pytest.mark.parametrize('use_pre_3_11_merge', [False, True])
async def test_run_async_short_circuits_other_agents_on_escalate_action(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    is_resumable: bool,
    use_pre_3_11_merge: bool,
):
  """ParallelAgent stops sibling agents and finishes when a sub-agent escalates."""

  class _TestingAgentWithEscalateAction(_TestingAgent):
    """Mock agent for testing escalation short-circuit behavior."""

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
      await asyncio.sleep(self.delay)
      yield self.event(
          ctx,
          text=f'Escalating from {self.name}!',
          actions=EventActions(escalate=True),
      )
      yield self.event(
          ctx, text='This event should be cancelled after escalation.'
      )

  if use_pre_3_11_merge:
    monkeypatch.setattr(
        parallel_agent_module,
        'sys',
        SimpleNamespace(version_info=(3, 10)),
    )

  fast_agent = _TestingAgent(
      name=f'{request.function.__name__}_test_fast_agent',
      delay=0.05,
  )
  escalating_agent = _TestingAgentWithEscalateAction(
      name=f'{request.function.__name__}_test_escalating_agent',
      delay=0.1,
  )
  slow_agent = _TestingAgent(
      name=f'{request.function.__name__}_test_slow_agent',
      delay=0.5,
  )
  parallel_agent = ParallelAgent(
      name=f'{request.function.__name__}_test_parallel_agent',
      sub_agents=[fast_agent, escalating_agent, slow_agent],
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, parallel_agent, is_resumable=is_resumable
  )

  events = [e async for e in parallel_agent.run_async(parent_ctx)]

  assert all(event.author != slow_agent.name for event in events)
  assert all(
      not event.content
      or not event.content.parts
      or event.content.parts[0].text
      != 'This event should be cancelled after escalation.'
      for event in events
  )

  if is_resumable:
    assert len(events) == 4

    assert events[0].author == parallel_agent.name
    assert not events[0].actions.end_of_agent

    assert events[1].author == fast_agent.name
    assert events[1].branch == f'{parallel_agent.name}.{fast_agent.name}'
    assert events[1].content.parts[0].text == f'Hello, async {fast_agent.name}!'

    assert events[2].author == escalating_agent.name
    assert events[2].branch == f'{parallel_agent.name}.{escalating_agent.name}'
    assert events[2].content.parts[0].text == (
        f'Escalating from {escalating_agent.name}!'
    )
    assert events[2].actions.escalate

    assert events[3].author == parallel_agent.name
    assert events[3].actions.end_of_agent
  else:
    assert len(events) == 2

    assert events[0].author == fast_agent.name
    assert events[0].branch == f'{parallel_agent.name}.{fast_agent.name}'
    assert events[0].content.parts[0].text == f'Hello, async {fast_agent.name}!'

    assert events[1].author == escalating_agent.name
    assert events[1].branch == f'{parallel_agent.name}.{escalating_agent.name}'
    assert events[1].content.parts[0].text == (
        f'Escalating from {escalating_agent.name}!'
    )
    assert events[1].actions.escalate


@pytest.mark.asyncio
@pytest.mark.parametrize('is_resumable', [True, False])
@pytest.mark.parametrize('use_pre_3_11_merge', [False, True])
@pytest.mark.parametrize('pause_target', ['fast', 'escalating'])
async def test_run_async_with_escalate_and_pause_does_not_finalize_agent(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    is_resumable: bool,
    use_pre_3_11_merge: bool,
    pause_target: str,
):
  """ParallelAgent does not emit end_of_agent when an escalating run is paused."""

  class _TestingAgentWithEscalateAction(_TestingAgent):
    """Mock agent for testing escalation short-circuit behavior."""

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
      await asyncio.sleep(self.delay)
      yield self.event(
          ctx,
          text=f'Escalating from {self.name}!',
          actions=EventActions(escalate=True),
      )

  if use_pre_3_11_merge:
    monkeypatch.setattr(
        parallel_agent_module,
        'sys',
        SimpleNamespace(version_info=(3, 10)),
    )

  fast_agent = _TestingAgent(
      name=f'{request.function.__name__}_test_fast_agent',
      delay=0.05,
  )
  escalating_agent = _TestingAgentWithEscalateAction(
      name=f'{request.function.__name__}_test_escalating_agent',
      delay=0.1,
  )
  parallel_agent = ParallelAgent(
      name=f'{request.function.__name__}_test_parallel_agent',
      sub_agents=[fast_agent, escalating_agent],
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, parallel_agent, is_resumable=is_resumable
  )

  target_agent_name = (
      fast_agent.name if pause_target == 'fast' else escalating_agent.name
  )

  def mock_should_pause(event: Event) -> bool:
    return event.author == target_agent_name

  with patch.object(
      InvocationContext,
      'should_pause_invocation',
      side_effect=mock_should_pause,
  ):
    events = [e async for e in parallel_agent.run_async(parent_ctx)]

  assert not any(event.actions.end_of_agent for event in events)
  assert any(event.actions.escalate for event in events)
  if is_resumable:
    assert len(events) == 3
    assert events[0].author == parallel_agent.name
    assert not events[0].actions.end_of_agent
    assert events[1].author == fast_agent.name
    assert events[2].author == escalating_agent.name
    assert events[2].actions.escalate
  else:
    assert len(events) == 2
    assert events[0].author == fast_agent.name
    assert events[1].author == escalating_agent.name
    assert events[1].actions.escalate
