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

"""Tests for agent timeout mechanism."""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator
from typing import Optional
from unittest import mock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.llm_agent import Agent
from google.adk.errors.agent_timeout_error import AgentTimeoutError
from google.adk.errors.agent_timeout_error import TimeoutTrigger
from google.adk.errors.agent_timeout_error import TimeoutType
from google.adk.events.event import Event
from google.adk.plugins.plugin_manager import PluginManager
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
import pytest
from typing_extensions import override

from .. import testing_utils


async def _create_test_invocation_context(
    agent: BaseAgent,
) -> InvocationContext:
  """Create a test invocation context for timeout tests."""
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )
  return InvocationContext(
      invocation_id='test_invocation',
      agent=agent,
      session=session,
      session_service=session_service,
      plugin_manager=PluginManager(plugins=[]),
  )


_sleep_seconds_by_agent: dict[str, float] = {}
_started_events: dict[str, asyncio.Event] = {}
_cancelled_events: dict[str, asyncio.Event] = {}
_sub_agent_by_parent: dict[str, str] = {}


class _SlowTestingAgent(BaseAgent):
  """A testing agent that simulates slow execution."""

  def __init__(self, name: str, sleep_seconds: float = 5.0):
    super().__init__(name=name)
    _sleep_seconds_by_agent[name] = sleep_seconds
    _started_events[name] = asyncio.Event()
    _cancelled_events[name] = asyncio.Event()

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    _started_events[self.name].set()
    sleep_seconds = _sleep_seconds_by_agent.get(self.name, 5.0)
    try:
      await asyncio.sleep(sleep_seconds)
    except asyncio.CancelledError:
      _cancelled_events[self.name].set()
      raise
    yield Event(
        author=self.name,
        branch=ctx.branch,
        invocation_id=ctx.invocation_id,
        content=types.Content(parts=[types.Part(text='Done')]),
    )

  @override
  async def _run_live_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    _started_events[self.name].set()
    sleep_seconds = _sleep_seconds_by_agent.get(self.name, 5.0)
    try:
      await asyncio.sleep(sleep_seconds)
    except asyncio.CancelledError:
      _cancelled_events[self.name].set()
      raise
    yield Event(
        author=self.name,
        invocation_id=ctx.invocation_id,
        branch=ctx.branch,
        content=types.Content(parts=[types.Part(text='Done')]),
    )


class _AgentWithSubAgent(BaseAgent):
  """An agent that calls a sub-agent."""

  def __init__(self, name: str, sub_agent_name: str):
    super().__init__(name=name)
    _sub_agent_by_parent[name] = sub_agent_name
    _started_events[name] = asyncio.Event()

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    _started_events[self.name].set()
    sub_agent_name = _sub_agent_by_parent.get(self.name)
    sub_agent = self.find_sub_agent(sub_agent_name)
    if sub_agent:
      async for event in sub_agent.run_async(ctx):
        yield event

  @override
  async def _run_live_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    _started_events[self.name].set()
    sub_agent_name = _sub_agent_by_parent.get(self.name)
    sub_agent = self.find_sub_agent(sub_agent_name)
    if sub_agent:
      async for event in sub_agent.run_live(ctx):
        yield event


@pytest.mark.asyncio
async def test_run_async_without_timeout_backward_compatible():
  """Test that agents without timeout work as before (backward compatibility)."""
  agent = _SlowTestingAgent(name='test_agent_bc', sleep_seconds=0.1)
  parent_ctx = await _create_test_invocation_context(agent)

  events = [e async for e in agent.run_async(parent_ctx)]

  assert len(events) == 1
  assert events[0].content.parts[0].text == 'Done'


@pytest.mark.asyncio
async def test_run_async_total_timeout_triggers():
  """Test that total_timeout triggers when execution takes too long."""
  agent = _SlowTestingAgent(name='test_agent_timeout', sleep_seconds=5.0)
  agent.total_timeout = 0.1
  parent_ctx = await _create_test_invocation_context(agent)

  with pytest.raises(AgentTimeoutError) as exc_info:
    [e async for e in agent.run_async(parent_ctx)]

  error = exc_info.value
  assert error.timeout_type == TimeoutType.TOTAL
  assert error.trigger == TimeoutTrigger.USER_INPUT
  assert error.agent_name == 'test_agent_timeout'
  assert error.elapsed_time >= 0.1
  assert 'total' in str(error).lower()


@pytest.mark.asyncio
async def test_run_live_total_timeout_triggers():
  """Test that total_timeout triggers in run_live when execution takes too long."""
  agent = _SlowTestingAgent(name='test_agent_live_timeout', sleep_seconds=5.0)
  agent.total_timeout = 0.1
  parent_ctx = await _create_test_invocation_context(agent)

  with pytest.raises(AgentTimeoutError) as exc_info:
    [e async for e in agent.run_live(parent_ctx)]

  error = exc_info.value
  assert error.timeout_type == TimeoutType.TOTAL
  assert error.trigger == TimeoutTrigger.USER_INPUT
  assert error.agent_name == 'test_agent_live_timeout'


@pytest.mark.asyncio
async def test_sub_agent_cascade_cancellation():
  """Test that when parent agent times out, sub-agent is also cancelled."""
  sub_agent = _SlowTestingAgent(name='sub_agent_cascade', sleep_seconds=5.0)
  parent_agent = _AgentWithSubAgent(
      name='parent_agent_cascade', sub_agent_name='sub_agent_cascade'
  )
  parent_agent.sub_agents = [sub_agent]
  parent_agent.total_timeout = 0.1

  parent_ctx = await _create_test_invocation_context(parent_agent)

  with pytest.raises(AgentTimeoutError):
    [e async for e in parent_agent.run_async(parent_ctx)]

  await asyncio.sleep(0.1)
  assert _cancelled_events['sub_agent_cascade'].is_set()


@pytest.mark.asyncio
async def test_agent_timeout_error_message():
  """Test that AgentTimeoutError has a descriptive message."""
  agent = _SlowTestingAgent(name='my_agent_msg', sleep_seconds=5.0)
  agent.total_timeout = 0.1
  parent_ctx = await _create_test_invocation_context(agent)

  with pytest.raises(AgentTimeoutError) as exc_info:
    [e async for e in agent.run_async(parent_ctx)]

  error = exc_info.value
  message = str(error)

  assert 'my_agent_msg' in message
  assert 'total' in message.lower() or 'TOTAL' in message
  assert 'USER_INPUT' in message or 'user_input' in message


def test_agent_timeout_fields():
  """Test that AgentTimeoutError has all required fields."""
  error = AgentTimeoutError(
      message='Test timeout',
      timeout_type=TimeoutType.SINGLE_TURN,
      elapsed_time=5.5,
      trigger=TimeoutTrigger.LLM_CALL,
      agent_name='test_agent',
  )

  assert error.timeout_type == TimeoutType.SINGLE_TURN
  assert error.elapsed_time == 5.5
  assert error.trigger == TimeoutTrigger.LLM_CALL
  assert error.agent_name == 'test_agent'
  assert isinstance(error, TimeoutError)


def test_agent_timeout_with_str_parameters():
  """Test that AgentTimeoutError accepts string parameters."""
  error = AgentTimeoutError(
      message='Test',
      timeout_type='single_turn',
      elapsed_time=10.0,
      trigger='tool_call',
      agent_name='agent',
  )

  assert error.timeout_type == 'single_turn'
  assert error.trigger == 'tool_call'


@pytest.mark.asyncio
async def test_llm_agent_with_timeout_fields():
  """Test that LlmAgent inherits timeout fields from BaseAgent."""
  agent = Agent(
      name='test_agent_fields',
      model='mock',
      single_turn_timeout=30.0,
      total_timeout=300.0,
  )

  assert agent.single_turn_timeout == 30.0
  assert agent.total_timeout == 300.0


@pytest.mark.asyncio
async def test_llm_agent_timeout_fields_none_by_default():
  """Test that timeout fields are None by default (backward compatible)."""
  agent = Agent(
      name='test_agent_default',
      model='mock',
  )

  assert agent.single_turn_timeout is None
  assert agent.total_timeout is None


@pytest.mark.asyncio
async def test_run_async_total_timeout_not_triggered_if_fast_enough():
  """Test that timeout is not triggered if execution finishes within timeout."""
  agent = _SlowTestingAgent(name='test_agent_fast', sleep_seconds=0.1)
  agent.total_timeout = 5.0
  parent_ctx = await _create_test_invocation_context(agent)

  events = [e async for e in agent.run_async(parent_ctx)]

  assert len(events) == 1
  assert events[0].content.parts[0].text == 'Done'


@pytest.mark.asyncio
async def test_run_async_cancelled_properly():
  """Test that internal task is properly cancelled on timeout."""
  agent = _SlowTestingAgent(name='test_agent_cancel', sleep_seconds=5.0)
  agent.total_timeout = 0.1
  parent_ctx = await _create_test_invocation_context(agent)

  with pytest.raises(AgentTimeoutError):
    [e async for e in agent.run_async(parent_ctx)]

  await asyncio.sleep(0.1)
  assert _cancelled_events['test_agent_cancel'].is_set()


class _AgentWithMultipleSubAgents(BaseAgent):
  """An agent that calls multiple sub-agents."""

  def __init__(self, name: str, sub_agent_names: list[str]):
    super().__init__(name=name)
    _sub_agent_by_parent[name] = sub_agent_names[0] if sub_agent_names else None
    _started_events[name] = asyncio.Event()
    self._sub_agent_names = sub_agent_names

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    _started_events[self.name].set()
    for sub_agent_name in self._sub_agent_names:
      sub_agent = self.find_sub_agent(sub_agent_name)
      if sub_agent:
        async for event in sub_agent.run_async(ctx):
          yield event


@pytest.mark.asyncio
async def test_parent_timeout_cascades_cancel_to_children():
  """Test that when parent agent times out, all sub-agent tasks are cancelled.

  This test verifies that the cascade cancellation works correctly:
  1. Parent agent starts a sub-agent
  2. Parent agent times out
  3. Parent's internal task is cancelled
  4. Sub-agent's run_async generator is exited via Aclosing
  5. Sub-agent's internal task receives the cancellation
  """
  sub_agent_1 = _SlowTestingAgent(name='child_1', sleep_seconds=5.0)
  sub_agent_2 = _SlowTestingAgent(name='child_2', sleep_seconds=5.0)
  parent_agent = _AgentWithMultipleSubAgents(
      name='parent_multi', sub_agent_names=['child_1', 'child_2']
  )
  parent_agent.sub_agents = [sub_agent_1, sub_agent_2]
  parent_agent.total_timeout = 0.1

  parent_ctx = await _create_test_invocation_context(parent_agent)

  with pytest.raises(AgentTimeoutError):
    [e async for e in parent_agent.run_async(parent_ctx)]

  await asyncio.sleep(0.1)
  assert _started_events['parent_multi'].is_set()
  assert _cancelled_events['child_1'].is_set()
