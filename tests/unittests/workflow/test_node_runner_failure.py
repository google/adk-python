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

"""Tests for NodeRunner retry logic on failures."""

import asyncio
import sys
from typing import Any
from typing import AsyncGenerator
from unittest import mock

from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow import BaseNode
from google.adk.workflow import Edge
from google.adk.workflow import START
from google.adk.workflow._errors import NodeTimeoutError
from google.adk.workflow._graph import Graph
from google.adk.workflow._node import node
from google.adk.workflow._node import Node
from google.adk.workflow._node_status import NodeStatus
from google.adk.workflow._retry_config import RetryConfig
from google.adk.workflow._workflow import Workflow
from google.genai import types
from pydantic import ConfigDict
from pydantic import Field
import pytest
from typing_extensions import override

from .workflow_testing_utils import _FlakyNode
from .workflow_testing_utils import create_parent_invocation_context
from .workflow_testing_utils import CustomNonRetryableError
from .workflow_testing_utils import CustomRetryableError
from .workflow_testing_utils import simplify_events_with_node
from .workflow_testing_utils import TestingNode


async def _run_workflow(wf, message='start'):
  """Run a Workflow through Runner, return collected events."""
  ss = InMemorySessionService()
  runner = Runner(app_name=wf.name, node=wf, session_service=ss)
  session = await ss.create_session(app_name=wf.name, user_id='u')
  msg = types.Content(parts=[types.Part(text=message)], role='user')
  events = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg
  ):
    events.append(event)
  return events, ss, session


@pytest.mark.asyncio
async def test_node_retries_on_matched_exception_string(
    request: pytest.FixtureRequest,
):
  """A node retries when raised exception matches a string name in RetryConfig.exceptions.

  Setup: Workflow with NodeA -> FlakyNode -> NodeC. FlakyNode fails twice with CustomRetryableError.
  Act: Run the workflow.
  Assert: FlakyNode succeeds on 3rd attempt, full workflow completes.
  """

  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  # Node will fail 2 times, then succeed on 3rd attempt
  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=3,
      tracker=tracker,
      exception_to_raise=CustomRetryableError('Simulated failure'),
      retry_config=RetryConfig(
          initial_delay=0.0,
          exceptions=['CustomRetryableError'],
      ),
  )
  node_c = TestingNode(name='NodeC', output='Executing C')
  agent = Workflow(
      name='test_workflow_agent_retry',
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=flaky_node),
          Edge(from_node=flaky_node, to_node=node_c),
      ],
  )

  events, _, _ = await _run_workflow(agent)

  results = simplify_events_with_node(events)
  filtered_results = [
      r
      for r in results
      if not (isinstance(r[1], str) and 'Retrying in' in r[1])
  ]
  assert filtered_results == [
      (
          'test_workflow_agent_retry@1/NodeA@1',
          {'output': 'Executing A'},
      ),
      (
          'test_workflow_agent_retry@1/FlakyNode@1',
          {'output': 'Executing B'},
      ),
      (
          'test_workflow_agent_retry@1/NodeC@1',
          {'output': 'Executing C'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 3


@pytest.mark.asyncio
async def test_node_fails_immediately_on_unmatched_exception_string(
    request: pytest.FixtureRequest,
):
  """A node fails immediately when raised exception does not match configured string names.

  Setup: Workflow with NodeA -> FlakyNode -> NodeC. FlakyNode fails with CustomNonRetryableError.
  Act: Run the workflow.
  Assert: Execution completes normally and emits CustomNonRetryableError event immediately without retry.
  """

  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  # Node will fail 1 time
  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=2,
      tracker=tracker,
      exception_to_raise=CustomNonRetryableError('Unexpected failure'),
      retry_config=RetryConfig(
          initial_delay=0.0,
          exceptions=['CustomRetryableError'],
      ),
  )
  node_c = TestingNode(name='NodeC', output='Executing C')
  agent = Workflow(
      name='test_workflow_agent_no_retry',
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=flaky_node),
          Edge(from_node=flaky_node, to_node=node_c),
      ],
  )

  ss = InMemorySessionService()
  runner = Runner(app_name=agent.name, node=agent, session_service=ss)
  session = await ss.create_session(app_name=agent.name, user_id='u')
  msg = types.Content(parts=[types.Part(text='start')], role='user')
  events = []

  # When the workflow is executed
  with pytest.raises(CustomNonRetryableError):
    async for event in runner.run_async(
        user_id='u', session_id=session.id, new_message=msg
    ):
      events.append(event)

  # Assert that the node error is persisted in session as an event
  error_events = [
      e
      for e in events
      if isinstance(e, Event) and e.error_code == 'CustomNonRetryableError'
  ]
  assert len(error_events) == 1
  assert error_events[0].error_message == 'Unexpected failure'

  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_no_retry@1/NodeA@1',
          {'output': 'Executing A'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 1


@pytest.mark.asyncio
async def test_retry_occurs_for_any_exception_when_exceptions_not_specified(
    request: pytest.FixtureRequest,
):
  """A node retries on any exception when RetryConfig.exceptions is not specified.

  Setup: Workflow with NodeA -> FlakyNode. FlakyNode fails once with ValueError.
  Act: Run the workflow with exceptions=None in RetryConfig.
  Assert: FlakyNode succeeds on 2nd attempt, workflow completes.
  """
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  # Node will fail 1 time, then succeed
  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=2,
      tracker=tracker,
      exception_to_raise=ValueError('Any failure'),
      retry_config=RetryConfig(
          initial_delay=0.0,
          exceptions=None,
      ),
  )
  agent = Workflow(
      name='test_workflow_agent_retry_all',
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=flaky_node),
      ],
  )

  events, _, _ = await _run_workflow(agent)

  results = simplify_events_with_node(events)
  filtered_results = [
      r
      for r in results
      if not (isinstance(r[1], str) and 'Retrying in' in r[1])
  ]
  assert filtered_results == [
      (
          'test_workflow_agent_retry_all@1/NodeA@1',
          {'output': 'Executing A'},
      ),
      (
          'test_workflow_agent_retry_all@1/FlakyNode@1',
          {'output': 'Executing B'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 2


@pytest.mark.asyncio
async def test_node_receives_incrementing_attempt_counts(
    request: pytest.FixtureRequest,
):
  """A node receives the current attempt count in its context for each attempt.

  Setup: Workflow with NodeA -> FlakyNode -> NodeC. FlakyNode fails twice before success.
  Act: Run the workflow.
  Assert: FlakyNode observes attempt_counts [1, 2, 3] in context across attempts.
  """
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  # Node will fail 2 times, then succeed on 3rd attempt
  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=3,
      tracker=tracker,
      exception_to_raise=CustomRetryableError('Simulated failure'),
      retry_config=RetryConfig(
          initial_delay=0.0, exceptions=['CustomRetryableError']
      ),
  )
  node_c = TestingNode(name='NodeC', output='Executing C')
  agent = Workflow(
      name='test_retry_count_populated_correctly',
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=flaky_node),
          Edge(from_node=flaky_node, to_node=node_c),
      ],
  )

  events, _, _ = await _run_workflow(agent)

  results = simplify_events_with_node(events)
  filtered_results = [
      r
      for r in results
      if not (isinstance(r[1], str) and 'Retrying in' in r[1])
  ]
  assert filtered_results == [
      (
          'test_retry_count_populated_correctly@1/NodeA@1',
          {'output': 'Executing A'},
      ),
      (
          'test_retry_count_populated_correctly@1/FlakyNode@1',
          {'output': 'Executing B'},
      ),
      (
          'test_retry_count_populated_correctly@1/NodeC@1',
          {'output': 'Executing C'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 3
  assert flaky_node_in_agent.tracker['attempt_counts'] == [1, 2, 3]


@pytest.mark.asyncio
async def test_node_stops_retrying_after_max_attempts(
    request: pytest.FixtureRequest,
):
  """A node fails with the original exception after exceeding max_attempts.

  Setup: Workflow with NodeA -> FlakyNode -> NodeC. FlakyNode fails persistently, max_attempts=3.
  Act: Run the workflow.
  Assert: Execution completes normally and emits CustomRetryableError event after 3 total attempts.
  """
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  # Node will fail 4 times, but max_attempts is 3.
  # Total attempts = 3 (1 initial + 2 retries).
  # Attempt 1: retry_count = 0, fails.
  # Attempt 2: retry_count = 1, fails.
  # Attempt 3: retry_count = 2, fails. Now _should_retry_node returns False.
  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=5,
      tracker=tracker,
      exception_to_raise=CustomRetryableError('Persisted failure'),
      retry_config=RetryConfig(
          initial_delay=0.0,
          max_attempts=3,
          exceptions=['CustomRetryableError'],
      ),
  )
  node_c = TestingNode(name='NodeC', output='Executing C')
  agent = Workflow(
      name='test_workflow_agent_max_attempts',
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=flaky_node),
          Edge(from_node=flaky_node, to_node=node_c),
      ],
  )

  ss = InMemorySessionService()
  runner = Runner(app_name=agent.name, node=agent, session_service=ss)
  session = await ss.create_session(app_name=agent.name, user_id='u')
  msg = types.Content(parts=[types.Part(text='start')], role='user')
  events = []

  # When the workflow is executed
  with pytest.raises(CustomRetryableError):
    async for event in runner.run_async(
        user_id='u', session_id=session.id, new_message=msg
    ):
      events.append(event)

  # Assert that the node error is persisted in session as an event after max attempts
  error_events = [
      e
      for e in events
      if isinstance(e, Event) and e.error_code == 'CustomRetryableError'
  ]
  assert len(error_events) == 3
  for err in error_events:
    assert err.error_message == 'Persisted failure'

  results = simplify_events_with_node(events)
  filtered_results = [
      r
      for r in results
      if not (isinstance(r[1], str) and 'Retrying in' in r[1])
  ]
  assert filtered_results == [
      (
          'test_workflow_agent_max_attempts@1/NodeA@1',
          {'output': 'Executing A'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 3


@pytest.mark.asyncio
async def test_node_fails_immediately_without_retry_config(
    request: pytest.FixtureRequest,
):
  """A node fails immediately on exception when it has no retry configuration.

  Setup: Workflow with NodeA -> FlakyNode. FlakyNode has retry_config=None.
  Act: Run the workflow.
  Assert: Execution completes normally and emits ValueError event immediately on first failure.
  """
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  # Node will fail 1 time
  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=2,
      tracker=tracker,
      exception_to_raise=ValueError('Any failure'),
      retry_config=None,
  )
  agent = Workflow(
      name='test_workflow_agent_fails_without_retry_config',
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=flaky_node),
      ],
  )

  ss = InMemorySessionService()
  runner = Runner(app_name=agent.name, node=agent, session_service=ss)
  session = await ss.create_session(app_name=agent.name, user_id='u')
  msg = types.Content(parts=[types.Part(text='start')], role='user')
  events = []

  # When the workflow is executed
  with pytest.raises(ValueError):
    async for event in runner.run_async(
        user_id='u', session_id=session.id, new_message=msg
    ):
      events.append(event)

  # Assert that the node error is persisted in session as an event
  error_events = [
      e for e in events if isinstance(e, Event) and e.error_code == 'ValueError'
  ]
  assert len(error_events) == 1
  assert error_events[0].error_message == 'Any failure'

  results = simplify_events_with_node(events)
  filtered_results = [
      r
      for r in results
      if not (isinstance(r[1], str) and 'Retrying in' in r[1])
  ]
  assert filtered_results == [
      (
          'test_workflow_agent_fails_without_retry_config@1/NodeA@1',
          {'output': 'Executing A'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 1


@pytest.mark.asyncio
async def test_node_retries_with_default_config_when_empty(
    request: pytest.FixtureRequest,
):
  """A node uses default retry settings when provided with an empty RetryConfig.

  Setup: Workflow with NodeA -> FlakyNode. FlakyNode has empty RetryConfig().
  Act: Run the workflow.
  Assert: FlakyNode succeeds on 2nd attempt using default retry behavior.
  """
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  # Node will fail 1 time, then succeed
  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=2,
      tracker=tracker,
      exception_to_raise=ValueError('Another failure'),
      retry_config=RetryConfig(),
  )
  agent = Workflow(
      name='test_workflow_agent_retries_with_empty_retry_config',
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=flaky_node),
      ],
  )

  events, _, _ = await _run_workflow(agent)

  results = simplify_events_with_node(events)
  filtered_results = [
      r
      for r in results
      if not (isinstance(r[1], str) and 'Retrying in' in r[1])
  ]
  assert filtered_results == [
      (
          'test_workflow_agent_retries_with_empty_retry_config@1/NodeA@1',
          {'output': 'Executing A'},
      ),
      (
          'test_workflow_agent_retries_with_empty_retry_config@1/FlakyNode@1',
          {'output': 'Executing B'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 2


@pytest.mark.asyncio
async def test_node_waits_for_initial_delay_before_retry(
    request: pytest.FixtureRequest,
):
  """A node sleeps for the specified initial delay before attempting a retry.

  Setup: Workflow with NodeA -> FlakyNode -> NodeC. FlakyNode has initial_delay=5.0.
  Act: Run the workflow, mocking asyncio.sleep.
  Assert: FlakyNode succeeds on 2nd attempt, sleep called with 5.0.
  """
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=2,
      tracker=tracker,
      exception_to_raise=CustomRetryableError('Sleep test failure'),
      retry_config=RetryConfig(
          initial_delay=5.0,
          max_attempts=3,
          jitter=0.0,
          exceptions=['CustomRetryableError'],
      ),
  )
  node_c = TestingNode(name='NodeC', output='Executing C')
  agent = Workflow(
      name='test_workflow_agent_retry_delay',
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=flaky_node),
          Edge(from_node=flaky_node, to_node=node_c),
      ],
  )

  ss = InMemorySessionService()
  runner = Runner(app_name=agent.name, node=agent, session_service=ss)
  session = await ss.create_session(app_name=agent.name, user_id='u')
  msg = types.Content(parts=[types.Part(text='start')], role='user')

  with mock.patch.object(
      asyncio, 'sleep', new_callable=mock.AsyncMock
  ) as mock_sleep:
    events = []
    async for event in runner.run_async(
        user_id='u', session_id=session.id, new_message=msg
    ):
      events.append(event)
    mock_sleep.assert_any_await(5.0)

  results = simplify_events_with_node(events)
  filtered_results = [
      r
      for r in results
      if not (isinstance(r[1], str) and 'Retrying in' in r[1])
  ]
  assert filtered_results == [
      (
          'test_workflow_agent_retry_delay@1/NodeA@1',
          {'output': 'Executing A'},
      ),
      (
          'test_workflow_agent_retry_delay@1/FlakyNode@1',
          {'output': 'Executing B'},
      ),
      (
          'test_workflow_agent_retry_delay@1/NodeC@1',
          {'output': 'Executing C'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 2


@pytest.mark.asyncio
async def test_retry_applies_backoff_strategy(request: pytest.FixtureRequest):
  """A node increases sleep delay on subsequent retries according to the backoff factor.

  Setup: Workflow with NodeA -> FlakyNode -> NodeC. initial_delay=2.0, backoff_factor=3.0.
  Act: Run the workflow, mocking asyncio.sleep.
  Assert: Sleep called with delays [2.0, 2.0, 6.0] matching backoff math.
  """
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=4,  # Fails 3 times
      tracker=tracker,
      exception_to_raise=CustomRetryableError('Backoff test failure'),
      retry_config=RetryConfig(
          initial_delay=2.0,
          max_attempts=5,
          backoff_factor=3.0,
          jitter=0.0,
          exceptions=['CustomRetryableError'],
      ),
  )
  node_c = TestingNode(name='NodeC', output='Executing C')
  agent = Workflow(
      name='test_workflow_agent_retry_backoff',
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=flaky_node),
          Edge(from_node=flaky_node, to_node=node_c),
      ],
  )

  ss = InMemorySessionService()
  runner = Runner(app_name=agent.name, node=agent, session_service=ss)
  session = await ss.create_session(app_name=agent.name, user_id='u')
  msg = types.Content(parts=[types.Part(text='start')], role='user')

  with mock.patch('asyncio.sleep', new_callable=mock.AsyncMock) as mock_sleep:
    events = []
    async for event in runner.run_async(
        user_id='u', session_id=session.id, new_message=msg
    ):
      events.append(event)
    # Attempt 1 (First Retry): fails, delay = 2.0 * (3.0 ** 0) = 2.0
    # Attempt 2 (Second Retry): fails, delay = 2.0 * (3.0 ** 1) = 6.0
    # Attempt 3 (Third Retry): fails, delay = 2.0 * (3.0 ** 2) = 18.0
    mock_sleep.assert_has_awaits(
        [mock.call(2.0), mock.call(6.0), mock.call(18.0)]
    )

  results = simplify_events_with_node(events)
  filtered_results = [
      r
      for r in results
      if not (isinstance(r[1], str) and 'Retrying in' in r[1])
  ]
  assert filtered_results == [
      (
          'test_workflow_agent_retry_backoff@1/NodeA@1',
          {'output': 'Executing A'},
      ),
      (
          'test_workflow_agent_retry_backoff@1/FlakyNode@1',
          {'output': 'Executing B'},
      ),
      (
          'test_workflow_agent_retry_backoff@1/NodeC@1',
          {'output': 'Executing C'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 4


@pytest.mark.asyncio
async def test_retry_applies_random_jitter(request: pytest.FixtureRequest):
  """A node adjusts retry delay with random jitter when configured.

  Setup: Workflow with NodeA -> FlakyNode -> NodeC. jitter=0.5, initial_delay=4.0.
  Act: Run the workflow, mocking random.uniform to return -1.0.
  Assert: Sleep called with 3.0 (4.0 + -1.0).
  """
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=2,
      tracker=tracker,
      exception_to_raise=CustomRetryableError('Jitter test failure'),
      retry_config=RetryConfig(
          initial_delay=4.0,
          max_attempts=3,
          backoff_factor=1.0,
          jitter=0.5,
          exceptions=['CustomRetryableError'],
      ),
  )
  node_c = TestingNode(name='NodeC', output='Executing C')
  agent = Workflow(
      name='test_workflow_agent_retry_jitter',
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=flaky_node),
          Edge(from_node=flaky_node, to_node=node_c),
      ],
  )

  ss = InMemorySessionService()
  runner = Runner(app_name=agent.name, node=agent, session_service=ss)
  session = await ss.create_session(app_name=agent.name, user_id='u')
  msg = types.Content(parts=[types.Part(text='start')], role='user')

  with (
      mock.patch('asyncio.sleep', new_callable=mock.AsyncMock) as mock_sleep,
      mock.patch('random.uniform', return_value=-1.0) as mock_random,
  ):
    events = []
    async for event in runner.run_async(
        user_id='u', session_id=session.id, new_message=msg
    ):
      events.append(event)

    # 4.0 + (-1.0) = 3.0
    mock_sleep.assert_any_await(3.0)
    # Called with -0.5 * 4.0, 0.5 * 4.0
    mock_random.assert_called_once_with(-2.0, 2.0)

  results = simplify_events_with_node(events)
  filtered_results = [
      r
      for r in results
      if not (isinstance(r[1], str) and 'Retrying in' in r[1])
  ]
  assert filtered_results == [
      (
          'test_workflow_agent_retry_jitter@1/NodeA@1',
          {'output': 'Executing A'},
      ),
      (
          'test_workflow_agent_retry_jitter@1/FlakyNode@1',
          {'output': 'Executing B'},
      ),
      (
          'test_workflow_agent_retry_jitter@1/NodeC@1',
          {'output': 'Executing C'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 2


@pytest.mark.asyncio
async def test_node_retries_on_exception_class_match(
    request: pytest.FixtureRequest,
):
  """A node retries when raised exception matches a class type in RetryConfig.exceptions.

  Setup: Workflow with NodeA -> FlakyNode -> NodeC. exceptions=[CustomRetryableError] (class).
  Act: Run the workflow.
  Assert: FlakyNode succeeds on 3rd attempt after matching exception class.
  """

  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=3,
      tracker=tracker,
      exception_to_raise=CustomRetryableError('Simulated failure'),
      retry_config=RetryConfig(
          initial_delay=0.0,
          exceptions=[CustomRetryableError],  # class, not string
      ),
  )
  node_c = TestingNode(name='NodeC', output='Executing C')
  agent = Workflow(
      name='test_retry_exception_classes',
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=flaky_node),
          Edge(from_node=flaky_node, to_node=node_c),
      ],
  )

  events, _, _ = await _run_workflow(agent)

  results = simplify_events_with_node(events)
  filtered_results = [
      r
      for r in results
      if not (isinstance(r[1], str) and 'Retrying in' in r[1])
  ]
  assert filtered_results == [
      (
          'test_retry_exception_classes@1/NodeA@1',
          {'output': 'Executing A'},
      ),
      (
          'test_retry_exception_classes@1/FlakyNode@1',
          {'output': 'Executing B'},
      ),
      (
          'test_retry_exception_classes@1/NodeC@1',
          {'output': 'Executing C'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 3


@pytest.mark.asyncio
async def test_node_retries_on_mixed_exception_types(
    request: pytest.FixtureRequest,
):
  """A node retries when exception matches either string name or class type in config.

  Setup: Workflow with NodeA -> FlakyNode -> NodeC. exceptions=[CustomRetryableError, 'ValueError'].
  Act: Run the workflow.
  Assert: FlakyNode succeeds on 2nd attempt after matching mixed types.
  """

  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=2,
      tracker=tracker,
      exception_to_raise=CustomRetryableError('Simulated failure'),
      retry_config=RetryConfig(
          initial_delay=0.0,
          exceptions=[CustomRetryableError, 'ValueError'],  # mixed
      ),
  )
  node_c = TestingNode(name='NodeC', output='Executing C')
  agent = Workflow(
      name='test_retry_mixed_exceptions',
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=flaky_node),
          Edge(from_node=flaky_node, to_node=node_c),
      ],
  )

  events, _, _ = await _run_workflow(agent)

  results = simplify_events_with_node(events)
  filtered_results = [
      r
      for r in results
      if not (isinstance(r[1], str) and 'Retrying in' in r[1])
  ]
  assert filtered_results == [
      (
          'test_retry_mixed_exceptions@1/NodeA@1',
          {'output': 'Executing A'},
      ),
      (
          'test_retry_mixed_exceptions@1/FlakyNode@1',
          {'output': 'Executing B'},
      ),
      (
          'test_retry_mixed_exceptions@1/NodeC@1',
          {'output': 'Executing C'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 2


@pytest.mark.asyncio
async def test_node_fails_immediately_on_unmatched_exception_class(
    request: pytest.FixtureRequest,
):
  """A node fails immediately when raised exception does not match configured class types.

  Setup: Workflow with NodeA -> FlakyNode. exceptions=[CustomRetryableError] (class).
  Act: Run the workflow. FlakyNode raises CustomNonRetryableError.
  Assert: Execution completes normally and emits CustomNonRetryableError event immediately.
  """

  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=3,
      tracker=tracker,
      exception_to_raise=CustomNonRetryableError('Unexpected failure'),
      retry_config=RetryConfig(
          initial_delay=0.0,
          exceptions=[CustomRetryableError],  # class, won't match
      ),
  )
  agent = Workflow(
      name='test_retry_exception_class_no_match',
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=flaky_node),
      ],
  )

  ss = InMemorySessionService()
  runner = Runner(app_name=agent.name, node=agent, session_service=ss)
  session = await ss.create_session(app_name=agent.name, user_id='u')
  msg = types.Content(parts=[types.Part(text='start')], role='user')
  events = []

  # When the workflow is executed
  with pytest.raises(CustomNonRetryableError):
    async for event in runner.run_async(
        user_id='u', session_id=session.id, new_message=msg
    ):
      events.append(event)

  # Assert that the node error is persisted in session as an event
  error_events = [
      e
      for e in events
      if isinstance(e, Event) and e.error_code == 'CustomNonRetryableError'
  ]
  assert len(error_events) == 1
  assert error_events[0].error_message == 'Unexpected failure'

  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 1


def test_retry_config_rejects_invalid_exception_types():
  """Tests that RetryConfig rejects non-string, non-class exception entries."""
  with pytest.raises(ValueError, match='exception class names'):
    RetryConfig(exceptions=[42])


def test_retry_config_normalizes_classes_to_strings():
  """Tests that exception classes are normalized to their names."""
  config = RetryConfig(exceptions=[ValueError, 'KeyError'])
  assert config.exceptions == ['ValueError', 'KeyError']


@pytest.mark.asyncio
async def test_error_event_emitted_on_failure(
    request: pytest.FixtureRequest,
):
  """Tests that an error event is emitted when a node raises an exception.

  Setup: Workflow with NodeA -> FlakyNode. FlakyNode fails with ValueError.
  Act: Run the workflow.
  Assert: Execution completes normally and emits ValueError event.
  """
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=999,
      tracker=tracker,
      exception_to_raise=ValueError('Something went wrong'),
      retry_config=None,
  )
  agent = Workflow(
      name='test_error_event',
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=flaky_node),
      ],
  )

  ss = InMemorySessionService()
  runner = Runner(app_name=agent.name, node=agent, session_service=ss)
  session = await ss.create_session(app_name=agent.name, user_id='u')
  msg = types.Content(parts=[types.Part(text='start')], role='user')
  events = []

  # When the workflow is executed
  with pytest.raises(ValueError):
    async for event in runner.run_async(
        user_id='u', session_id=session.id, new_message=msg
    ):
      events.append(event)

  # Find the error event emitted by the failed node.
  error_events = [
      e
      for e in events
      if isinstance(e, Event)
      and e.error_code is not None
      and e.node_name == 'FlakyNode'
  ]
  assert len(error_events) == 1
  assert error_events[0].error_code == 'ValueError'
  assert error_events[0].error_message == 'Something went wrong'


@pytest.mark.asyncio
async def test_error_event_emitted_on_each_retry(
    request: pytest.FixtureRequest,
):
  """Tests that an error event is emitted for each failed retry attempt."""
  tracker = {'iteration_count': 0}

  # Node will fail 2 times, then succeed on 3rd attempt
  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Success',
      succeed_on_iteration=3,
      tracker=tracker,
      exception_to_raise=CustomRetryableError('Transient error'),
      retry_config=RetryConfig(
          initial_delay=0.0,
          exceptions=['CustomRetryableError'],
      ),
  )
  agent = Workflow(
      name='test_error_event_retry',
      edges=[
          Edge(from_node=START, to_node=flaky_node),
      ],
  )

  ss = InMemorySessionService()
  runner = Runner(app_name=agent.name, node=agent, session_service=ss)
  session = await ss.create_session(app_name=agent.name, user_id='u')
  msg = types.Content(parts=[types.Part(text='start')], role='user')
  events = []

  # When the workflow is executed
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg
  ):
    events.append(event)

  # Two failures before success → two error events.
  error_events = [
      e
      for e in events
      if isinstance(e, Event)
      and e.error_code is not None
      and e.node_name == 'FlakyNode'
  ]
  assert len(error_events) == 2
  for err in error_events:
    assert err.error_code == 'CustomRetryableError'
    assert err.error_message == 'Transient error'

  # The node should still produce its output after retries.
  results = simplify_events_with_node(events)
  filtered_results = [
      r
      for r in results
      if not (isinstance(r[1], str) and 'Retrying in' in r[1])
  ]
  assert filtered_results == [
      (
          'test_error_event_retry@1/FlakyNode@1',
          {'output': 'Success'},
      ),
  ]


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason='asyncio.timeout requires Python 3.11+'
)
async def test_node_runner_timeout():
  async def slow_route(ctx, node_input):
    await asyncio.sleep(2)
    return 'done'

  node = TestingNode(name='SlowNode', route=slow_route, timeout=0.1)

  agent = Workflow(
      name='test_timeout',
      edges=[(START, node)],
  )

  # Workflow should yield a timeout error event
  with pytest.raises(NodeTimeoutError) as exc_info:
    await _run_workflow(agent)
  assert 'SlowNode' in str(exc_info.value)
  assert 'timed out' in str(exc_info.value)


@pytest.mark.skip(
    reason='Timeout is now supported in Python 3.10 via asyncio.wait_for',
)
async def test_node_runner_timeout_warning(caplog):
  async def slow_route(ctx, node_input):
    await asyncio.sleep(0.5)
    return 'done'

  node = TestingNode(name='SlowNode', route=slow_route, timeout=0.1)

  agent = Workflow(
      name='test_timeout_warning',
      edges=[(START, node)],
  )

  await _run_workflow(agent)

  assert (
      'timeout 0.10 seconds is ignored because Python version is < 3.11'
      in caplog.text
  )
