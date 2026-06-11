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

"""Tests for Workflow error handling, graceful shutdown, and retry logic."""

import asyncio
from typing import Any
from typing import AsyncGenerator
from unittest import mock

from google.adk.agents.context import Context
from google.adk.apps.app import App
from google.adk.events.event import Event
# Added for the moved test
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow import BaseNode
from google.adk.workflow import Edge
from google.adk.workflow import START
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

from .. import testing_utils
from .workflow_testing_utils import create_parent_invocation_context
from .workflow_testing_utils import simplify_events_with_node
from .workflow_testing_utils import TestingNode


class CustomError(Exception):
  """A custom error for testing."""


class CustomRetryableError(Exception):
  """A custom error meant to be retried."""


class CustomNonRetryableError(Exception):
  """A custom error not meant to be retried."""


class _FlakyNode(BaseNode):
  model_config = ConfigDict(arbitrary_types_allowed=True)

  message: str = Field(default='')
  succeed_on_iteration: int = Field(default=0)
  tracker: dict[str, Any] = Field(default_factory=dict)
  exception_to_raise: Exception = Field(...)

  @override
  async def run(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    iteration_count = self.tracker.get('iteration_count', 0) + 1
    self.tracker['iteration_count'] = iteration_count
    self.tracker.setdefault('attempt_counts', []).append(ctx.attempt_count)

    if iteration_count < self.succeed_on_iteration:
      raise self.exception_to_raise

    yield Event(
        output=self.message,
    )


async def _run_workflow(wf, message='start'):
  """Run a Workflow through Runner, return collected events."""
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')
  msg = types.Content(parts=[types.Part(text=message)], role='user')
  events = []
  try:
    async for event in runner.run_async(
        user_id='u', session_id=session.id, new_message=msg
    ):
      events.append(event)
  except CustomError:
    pass
  return events, ss, session


# --- Tests originally in test_workflow_agent_failures.py ---


@pytest.mark.asyncio
async def test_retry_on_matching_exception(request: pytest.FixtureRequest):
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
          exceptions=['CustomRetryableError'],
      ),
  )
  node_c = TestingNode(name='NodeC', output='Executing C')
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=flaky_node),
          Edge(from_node=flaky_node, to_node=node_c),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_retry',
      graph=graph,
  )

  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  assert simplify_events_with_node(events) == [
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
async def test_no_retry_on_non_matching_exception(
    request: pytest.FixtureRequest,
):
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

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
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=flaky_node),
          Edge(from_node=flaky_node, to_node=node_c),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_no_retry',
      graph=graph,
  )

  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)

  with pytest.raises(CustomNonRetryableError, match='Unexpected failure'):
    await runner.run_async(testing_utils.get_user_content('start'))

  events = runner.session.events

  assert simplify_events_with_node(events) == [
      ('user', 'start'),
      (
          'test_workflow_agent_no_retry@1/NodeA@1',
          {'output': 'Executing A'},
      ),
  ]


@pytest.mark.asyncio
async def test_retry_on_all_exceptions_if_not_specified(
    request: pytest.FixtureRequest,
):
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

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
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=flaky_node),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_retry_all',
      graph=graph,
  )

  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_retry_all@1/NodeA@1',
          {'output': 'Executing A'},
      ),
      (
          'test_workflow_agent_retry_all@1/FlakyNode@1',
          {'output': 'Executing B'},
      ),
  ]


@pytest.mark.asyncio
async def test_attempt_count_populated_correctly(
    request: pytest.FixtureRequest,
):
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

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
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=flaky_node),
          Edge(from_node=flaky_node, to_node=node_c),
      ],
  )
  agent = Workflow(
      name='test_retry_count_populated_correctly',
      graph=graph,
  )

  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  assert simplify_events_with_node(events) == [
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
async def test_retry_max_attempts_exceeded(
    request: pytest.FixtureRequest,
):
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

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
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=flaky_node),
          Edge(from_node=flaky_node, to_node=node_c),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_max_attempts',
      graph=graph,
  )

  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)

  with pytest.raises(CustomRetryableError, match='Persisted failure'):
    await runner.run_async(testing_utils.get_user_content('start'))

  events = runner.session.events

  assert simplify_events_with_node(events) == [
      ('user', 'start'),
      (
          'test_workflow_agent_max_attempts@1/NodeA@1',
          {'output': 'Executing A'},
      ),
  ]


@pytest.mark.asyncio
async def test_fails_without_retry_config(
    request: pytest.FixtureRequest,
):
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=2,
      tracker=tracker,
      exception_to_raise=ValueError('Any failure'),
      retry_config=None,
  )
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=flaky_node),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_fails_without_retry_config',
      graph=graph,
  )

  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)
  with pytest.raises(ValueError, match='Any failure'):
    await runner.run_async(testing_utils.get_user_content('start'))
  events = runner.session.events

  assert simplify_events_with_node(events) == [
      ('user', 'start'),
      (
          'test_workflow_agent_fails_without_retry_config@1/NodeA@1',
          {'output': 'Executing A'},
      ),
  ]


@pytest.mark.asyncio
async def test_retries_with_empty_retry_config(
    request: pytest.FixtureRequest,
):
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=2,
      tracker=tracker,
      exception_to_raise=ValueError('Another failure'),
      retry_config=RetryConfig(),
  )
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=flaky_node),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_retries_with_empty_retry_config',
      graph=graph,
  )

  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_retries_with_empty_retry_config@1/NodeA@1',
          {'output': 'Executing A'},
      ),
      (
          'test_workflow_agent_retries_with_empty_retry_config@1/FlakyNode@1',
          {'output': 'Executing B'},
      ),
  ]


@pytest.mark.asyncio
async def test_retry_with_delay(request: pytest.FixtureRequest):
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
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=flaky_node),
          Edge(from_node=flaky_node, to_node=node_c),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_retry_delay',
      graph=graph,
  )

  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)

  with mock.patch.object(
      asyncio, 'sleep', new_callable=mock.AsyncMock
  ) as mock_sleep:
    events = await runner.run_async(testing_utils.get_user_content('start'))
    mock_sleep.assert_any_await(5.0)

  assert simplify_events_with_node(events) == [
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


@pytest.mark.asyncio
async def test_retry_with_backoff_and_jitter(request: pytest.FixtureRequest):
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=4,
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
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=flaky_node),
          Edge(from_node=flaky_node, to_node=node_c),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_retry_backoff',
      graph=graph,
  )

  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)

  with mock.patch('asyncio.sleep', new_callable=mock.AsyncMock) as mock_sleep:
    events = await runner.run_async(testing_utils.get_user_content('start'))
    mock_sleep.assert_has_awaits(
        [mock.call(2.0), mock.call(6.0), mock.call(18.0)]
    )

  assert simplify_events_with_node(events) == [
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


@pytest.mark.asyncio
async def test_retry_with_jitter(request: pytest.FixtureRequest):
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
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=flaky_node),
          Edge(from_node=flaky_node, to_node=node_c),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_retry_jitter',
      graph=graph,
  )

  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)

  with (
      mock.patch('asyncio.sleep', new_callable=mock.AsyncMock) as mock_sleep,
      mock.patch('random.uniform', return_value=-1.0) as mock_random,
  ):
    events = await runner.run_async(testing_utils.get_user_content('start'))
    mock_sleep.assert_any_await(3.0)
    mock_random.assert_called_once_with(-2.0, 2.0)

  assert simplify_events_with_node(events) == [
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


@pytest.mark.asyncio
async def test_retry_with_exception_classes(request: pytest.FixtureRequest):
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
          exceptions=[CustomRetryableError],
      ),
  )
  node_c = TestingNode(name='NodeC', output='Executing C')
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=flaky_node),
          Edge(from_node=flaky_node, to_node=node_c),
      ],
  )
  agent = Workflow(
      name='test_retry_exception_classes',
      graph=graph,
  )

  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  assert simplify_events_with_node(events) == [
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


@pytest.mark.asyncio
async def test_retry_with_mixed_exception_types(request: pytest.FixtureRequest):
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
          exceptions=[CustomRetryableError, 'ValueError'],
      ),
  )
  node_c = TestingNode(name='NodeC', output='Executing C')
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=flaky_node),
          Edge(from_node=flaky_node, to_node=node_c),
      ],
  )
  agent = Workflow(
      name='test_retry_mixed_exceptions',
      graph=graph,
  )

  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  assert simplify_events_with_node(events) == [
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


@pytest.mark.asyncio
async def test_retry_exception_class_no_match(request: pytest.FixtureRequest):
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
          exceptions=[CustomRetryableError],
      ),
  )
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=flaky_node),
      ],
  )
  agent = Workflow(
      name='test_retry_exception_class_no_match',
      graph=graph,
  )

  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)

  with pytest.raises(CustomNonRetryableError, match='Unexpected failure'):
    await runner.run_async(testing_utils.get_user_content('start'))

  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 1


def test_retry_config_rejects_invalid_exception_types():
  with pytest.raises(ValueError, match='exception class names'):
    RetryConfig(exceptions=[42])


def test_retry_config_normalizes_classes_to_strings():
  config = RetryConfig(exceptions=[ValueError, 'KeyError'])
  assert config.exceptions == ['ValueError', 'KeyError']


@pytest.mark.asyncio
async def test_node_cancellation_on_sibling_failure(
    request: pytest.FixtureRequest,
):
  slow_node_started = False
  slow_node_cancelled = False

  async def slow_node():
    nonlocal slow_node_started, slow_node_cancelled
    slow_node_started = True
    try:
      await asyncio.sleep(10)
    except asyncio.CancelledError:
      slow_node_cancelled = True
      raise
    yield 'Slow'

  async def fail_node():
    await asyncio.sleep(0.1)
    raise ValueError('Fail')

  agent = Workflow(
      name='test_workflow_cancellation_sibling',
      edges=[
          (START, slow_node),
          (START, fail_node),
      ],
  )

  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)
  with pytest.raises(ValueError, match='Fail'):
    await runner.run_async(testing_utils.get_user_content('start'))

  assert slow_node_started is True
  assert slow_node_cancelled is True


@pytest.mark.asyncio
async def test_parallel_worker_cancellation_on_sibling_failure(
    request: pytest.FixtureRequest,
):
  slow_node_started = False
  slow_node_cancelled = False

  async def slow_node_impl(ctx: Context, node_input: Any):
    nonlocal slow_node_started, slow_node_cancelled
    slow_node_started = True
    try:
      await asyncio.sleep(10)
    except asyncio.CancelledError:
      slow_node_cancelled = True
      raise
    yield f'Slow {node_input}'

  async def fail_node():
    await asyncio.sleep(0.1)
    raise ValueError('Fail')

  node_parallel = node(
      slow_node_impl, name='node_parallel', parallel_worker=True
  )

  agent = Workflow(
      name='test_workflow_parallel_cancellation_sibling',
      edges=[
          (START, node_parallel),
          (START, fail_node),
      ],
  )

  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)
  with pytest.raises(ValueError, match='Fail'):
    await runner.run_async(testing_utils.get_user_content('start'))

  assert slow_node_started is True
  assert slow_node_cancelled is True


@pytest.mark.asyncio
async def test_parallel_worker_cancellation_on_worker_failure(
    request: pytest.FixtureRequest,
):
  slow_worker_started = False
  slow_worker_cancelled = False

  async def worker_node_impl(ctx: Context, node_input: Any):
    nonlocal slow_worker_started, slow_worker_cancelled
    if node_input == 'fail':
      await asyncio.sleep(0.1)
      raise ValueError('Worker Fail')
    else:
      slow_worker_started = True
      try:
        await asyncio.sleep(10)
      except asyncio.CancelledError:
        slow_worker_cancelled = True
        raise
      yield f'Success {node_input}'

  from tests.unittests.workflow.workflow_testing_utils import TestingNode

  node_list = TestingNode(name='NodeList', output=['fail', 'slow'])
  node_parallel = node(
      worker_node_impl, name='node_parallel', parallel_worker=True
  )

  agent = Workflow(
      name='test_workflow_parallel_cancellation_worker',
      edges=[
          (START, node_list),
          (node_list, node_parallel),
      ],
  )

  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)
  with pytest.raises(ValueError, match='Worker Fail'):
    await runner.run_async(testing_utils.get_user_content('start'))

  assert slow_worker_started is True
  assert slow_worker_cancelled is True


@pytest.mark.asyncio
async def test_nested_workflow_cancellation_on_sibling_failure(
    request: pytest.FixtureRequest,
):
  inner_node_started = False
  inner_node_cancelled = False

  async def inner_slow_node():
    nonlocal inner_node_started, inner_node_cancelled
    inner_node_started = True
    try:
      await asyncio.sleep(10)
    except asyncio.CancelledError:
      inner_node_cancelled = True
      raise
    yield 'Inner Slow'

  inner_agent = Workflow(
      name='inner_workflow',
      edges=[
          (START, inner_slow_node),
      ],
  )

  async def fail_node():
    await asyncio.sleep(0.1)
    raise ValueError('Fail')

  outer_agent = Workflow(
      name='outer_workflow',
      edges=[
          (START, inner_agent),
          (START, fail_node),
      ],
  )

  app = App(name=request.function.__name__, root_agent=outer_agent)
  runner = testing_utils.InMemoryRunner(app=app)
  with pytest.raises(ValueError, match='Fail'):
    await runner.run_async(testing_utils.get_user_content('start'))

  assert inner_node_started is True
  assert inner_node_cancelled is True


@pytest.mark.asyncio
async def test_error_event_emitted_on_failure(
    request: pytest.FixtureRequest,
):
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
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=flaky_node),
      ],
  )
  agent = Workflow(
      name='test_error_event',
      graph=graph,
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)
  with pytest.raises(ValueError, match='Something went wrong'):
    await runner.run_async(testing_utils.get_user_content('start'))
  events = runner.session.events

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
  tracker = {'iteration_count': 0}

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
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=flaky_node),
      ],
  )
  agent = Workflow(
      name='test_error_event_retry',
      graph=graph,
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

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

  assert simplify_events_with_node(events) == [
      (
          'test_error_event_retry@1/FlakyNode@1',
          {'output': 'Success'},
      ),
  ]


# --- Moved from test_workflow_failure.py ---


@pytest.mark.asyncio
async def test_workflow_returns_normally_on_node_failure():
  """Workflow returns normally when a node fails, without duplicate error events."""

  @node()
  def failing_node(ctx: Context):
    raise CustomError('Node failed')
    yield 'output'

  wf = Workflow(
      name='test_error_workflow',
      edges=[
          (START, failing_node),
      ],
  )

  events, ss, session = await _run_workflow(wf)

  error_events = [
      e
      for e in events
      if isinstance(e, Event) and e.error_code == 'CustomError'
  ]
  assert len(error_events) == 1
  assert error_events[0].error_message == 'Node failed'

  workflow_error_events = [
      e
      for e in events
      if isinstance(e, Event)
      and e.error_code is not None
      and e.node_info
      and e.node_info.path == 'test_error_workflow@1'
  ]
  assert len(workflow_error_events) == 0
