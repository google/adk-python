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
import logging
from typing import Any
from typing import AsyncGenerator
from unittest import mock

from google.adk import platform as adk_platform
from google.adk.agents.context import Context
from google.adk.agents.invocation_context import InvocationContext
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.plugins.base_plugin import BasePlugin
# Added for the moved test
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow import BaseNode
from google.adk.workflow import Edge
from google.adk.workflow import START
from google.adk.workflow._errors import NodeTimeoutError
from google.adk.workflow._graph import Graph
from google.adk.workflow._node import Node
from google.adk.workflow._node import node
from google.adk.workflow._node_status import NodeStatus
from google.adk.workflow._retry_config import RetryConfig
from google.adk.workflow._workflow import Workflow
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
from google.adk.workflow.utils._workflow_hitl_utils import get_request_input_interrupt_ids
from google.genai import types
from pydantic import ConfigDict
from pydantic import Field
import pytest
from typing_extensions import override

from .. import testing_utils
from .workflow_testing_utils import create_parent_invocation_context
from .workflow_testing_utils import get_request_input_events
from .workflow_testing_utils import RequestInputNode
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

  mock_random = mock.Mock()
  mock_random.uniform = mock.Mock(return_value=-1.0)
  adk_platform.set_random_provider(lambda: mock_random)
  try:
    with mock.patch('asyncio.sleep', new_callable=mock.AsyncMock) as mock_sleep:
      events = await runner.run_async(testing_utils.get_user_content('start'))
      mock_sleep.assert_any_await(3.0)
      mock_random.uniform.assert_called_once_with(-2.0, 2.0)
  finally:
    adk_platform.reset_random_provider()

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


@pytest.mark.asyncio
async def test_retry_config_on_nested_workflow_retries_failing_child():
  """A sub-workflow's retry_config re-runs a child node that raises.

  A Workflow reports a failing child by setting an error on its context
  instead of raising, so this only works if the node runner treats a
  reported failure the same as a raised one.
  """
  attempts = 0
  downstream_input = None

  @node()
  def flaky_node(ctx: Context):
    nonlocal attempts
    attempts += 1
    if attempts < 3:
      raise CustomError('Node failed')
    yield 'recovered'

  @node()
  def downstream_node(ctx: Context, node_input: Any):
    nonlocal downstream_input
    downstream_input = node_input
    yield 'downstream'

  inner_wf = Workflow(
      name='inner_wf',
      edges=[(START, flaky_node)],
      retry_config=RetryConfig(max_attempts=3, initial_delay=0.0, jitter=0.0),
  )
  outer_wf = Workflow(
      name='outer_wf',
      edges=[(START, inner_wf, downstream_node)],
  )

  await _run_workflow(outer_wf)

  assert attempts == 3
  assert downstream_input == 'recovered'


@pytest.mark.asyncio
async def test_retry_config_on_nested_workflow_gives_up_after_max_attempts():
  """A child that keeps failing still fails the workflow once retries run out.

  The failure must keep propagating: it must not be replayed as a completed
  node and fast-forwarded into a false success.
  """
  attempts = 0
  downstream_ran = False

  @node()
  def failing_node(ctx: Context):
    nonlocal attempts
    attempts += 1
    raise CustomError('Node failed')
    yield 'output'

  @node()
  def downstream_node(ctx: Context):
    nonlocal downstream_ran
    downstream_ran = True
    yield 'downstream'

  inner_wf = Workflow(
      name='inner_wf',
      edges=[(START, failing_node)],
      retry_config=RetryConfig(max_attempts=3, initial_delay=0.0, jitter=0.0),
  )
  outer_wf = Workflow(
      name='outer_wf',
      edges=[(START, inner_wf, downstream_node)],
  )

  events, _, _ = await _run_workflow(outer_wf)

  assert attempts == 3
  assert not downstream_ran
  error_events = [
      e
      for e in events
      if isinstance(e, Event) and e.error_code == 'CustomError'
  ]
  assert error_events


@pytest.mark.asyncio
async def test_retry_config_on_nested_workflow_replays_completed_children():
  """A retried sub-workflow does not redo a child that already produced output."""
  completed_attempts = 0
  flaky_attempts = 0

  @node()
  def completed_node(ctx: Context):
    nonlocal completed_attempts
    completed_attempts += 1
    yield 'done'

  @node()
  def flaky_node(ctx: Context, node_input: Any):
    nonlocal flaky_attempts
    flaky_attempts += 1
    if flaky_attempts < 3:
      raise CustomError('Node failed')
    yield 'recovered'

  inner_wf = Workflow(
      name='inner_wf',
      edges=[(START, completed_node, flaky_node)],
      retry_config=RetryConfig(max_attempts=3, initial_delay=0.0, jitter=0.0),
  )
  outer_wf = Workflow(name='outer_wf', edges=[(START, inner_wf)])

  await _run_workflow(outer_wf)

  assert flaky_attempts == 3
  assert completed_attempts == 1


@pytest.mark.asyncio
async def test_retry_config_on_outer_workflow_retries_nested_failure():
  """A failure inside a sub-workflow is a failure of the outer workflow too."""
  attempts = 0

  @node()
  def flaky_node(ctx: Context):
    nonlocal attempts
    attempts += 1
    if attempts < 3:
      raise CustomError('Node failed')
    yield 'recovered'

  inner_wf = Workflow(name='inner_wf', edges=[(START, flaky_node)])
  outer_wf = Workflow(
      name='outer_wf',
      edges=[(START, inner_wf)],
      retry_config=RetryConfig(max_attempts=3, initial_delay=0.0, jitter=0.0),
  )

  await _run_workflow(outer_wf)

  assert attempts == 3


@pytest.mark.asyncio
async def test_retry_config_on_nested_workflow_honors_exceptions_filter():
  """The exceptions filter matches the child's error, not the workflow."""
  retryable_attempts = 0
  non_retryable_attempts = 0

  @node()
  def retryable_node(ctx: Context):
    nonlocal retryable_attempts
    retryable_attempts += 1
    raise CustomRetryableError('Node failed')
    yield 'output'

  @node()
  def non_retryable_node(ctx: Context):
    nonlocal non_retryable_attempts
    non_retryable_attempts += 1
    raise CustomNonRetryableError('Node failed')
    yield 'output'

  retry_config = RetryConfig(
      max_attempts=3,
      initial_delay=0.0,
      jitter=0.0,
      exceptions=[CustomRetryableError],
  )

  retryable_wf = Workflow(
      name='retryable_wf',
      edges=[(START, retryable_node)],
      retry_config=retry_config,
  )
  with pytest.raises(CustomRetryableError):
    await _run_workflow(Workflow(name='outer', edges=[(START, retryable_wf)]))

  non_retryable_wf = Workflow(
      name='non_retryable_wf',
      edges=[(START, non_retryable_node)],
      retry_config=retry_config,
  )
  with pytest.raises(CustomNonRetryableError):
    await _run_workflow(
        Workflow(name='outer2', edges=[(START, non_retryable_wf)])
    )

  assert retryable_attempts == 3
  assert non_retryable_attempts == 1


@pytest.mark.asyncio
async def test_retry_config_on_nested_workflow_retries_its_own_timeout():
  """A sub-workflow's retry_config also applies when it times out as a unit."""
  attempts = 0

  @node()
  async def slow_node():
    nonlocal attempts
    attempts += 1
    await asyncio.sleep(1.0)
    return 'done'

  inner_wf = Workflow(
      name='inner_wf',
      edges=[(START, slow_node)],
      timeout=0.05,
      retry_config=RetryConfig(max_attempts=3, initial_delay=0.0, jitter=0.0),
  )
  outer_wf = Workflow(name='outer_wf', edges=[(START, inner_wf)])

  with pytest.raises(NodeTimeoutError):
    await _run_workflow(outer_wf)

  assert attempts == 3


@pytest.mark.asyncio
async def test_fail_fast_preserves_completed_siblings(
    request: pytest.FixtureRequest,
):
  """Tests that when one node fails, other sibling nodes completed in the same tick still have their outputs preserved."""
  node_success_started = False
  node_success_completed = False

  @node()
  async def succeeding_node(ctx: Context):
    nonlocal node_success_started, node_success_completed
    node_success_started = True
    await asyncio.sleep(0)
    node_success_completed = True
    return 'success_output'

  @node()
  async def failing_node(ctx: Context):
    await asyncio.sleep(0)
    raise ValueError('Fail')

  wf = Workflow(
      name='test_fail_fast_workflow',
      edges=[
          (START, failing_node),
          (START, succeeding_node),
      ],
  )

  original_handle_completion = Workflow._handle_completion
  handle_completion_calls = []

  def spy_handle_completion(
      self, loop_state, node_name, node_obj, child_ctx, ctx
  ):
    handle_completion_calls.append(node_name)
    return original_handle_completion(
        self, loop_state, node_name, node_obj, child_ctx, ctx
    )

  app = App(name=request.function.__name__, root_agent=wf)
  runner = testing_utils.InMemoryRunner(app=app)

  with mock.patch.object(
      Workflow, '_handle_completion', new=spy_handle_completion
  ):
    with pytest.raises(ValueError, match='Fail'):
      await runner.run_async(testing_utils.get_user_content('start'))

  # The succeeding_node should have successfully completed.
  assert node_success_started is True
  assert node_success_completed is True

  # Under the bug, succeeding_node's completion handler was skipped.
  # With the fix, succeeding_node's completion is handled.
  assert 'failing_node' not in handle_completion_calls
  assert 'succeeding_node' in handle_completion_calls


@pytest.mark.asyncio
async def test_multiple_failures_first_error_wins(
    request: pytest.FixtureRequest,
):
  """Tests that when multiple parallel nodes fail in the same tick, the first error is preserved."""

  @node()
  async def failing_node_1(ctx: Context):
    await asyncio.sleep(0.1)
    raise ValueError('Fail 1')

  @node()
  async def failing_node_2(ctx: Context):
    await asyncio.sleep(0.1)
    raise ValueError('Fail 2')

  wf = Workflow(
      name='test_multiple_failures_workflow',
      edges=[
          (START, failing_node_1),
          (START, failing_node_2),
      ],
  )

  app = App(name=request.function.__name__, root_agent=wf)
  runner = testing_utils.InMemoryRunner(app=app)

  with pytest.raises(ValueError, match='Fail 1'):
    await runner.run_async(testing_utils.get_user_content('start'))


@pytest.mark.asyncio
async def test_workflow_halts_when_before_run_callback_returns_content():
  """Regression for #6013: a plugin before_run_callback returning Content must
  halt the workflow run with that content and skip node execution."""

  ran = {'node': False}

  class _RecordingNode(BaseNode):

    @override
    async def run(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      ran['node'] = True
      yield Event(output='should not run')

  class _HaltPlugin(BasePlugin):

    def __init__(self):
      super().__init__(name='halt_plugin')

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> types.Content:
      return types.Content(
          role='model', parts=[types.Part(text='halted by plugin')]
      )

  graph = Graph(edges=[Edge(from_node=START, to_node=_RecordingNode(name='A'))])
  wf = Workflow(name='halt_wf', graph=graph)

  ss = InMemorySessionService()
  app = App(name='test', root_agent=wf, plugins=[_HaltPlugin()])
  runner = Runner(app=app, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')
  msg = types.Content(parts=[types.Part(text='start')], role='user')
  events = [
      event
      async for event in runner.run_async(
          user_id='u', session_id=session.id, new_message=msg
      )
  ]

  # The run halts with the plugin's content and the node never executes.
  assert ran['node'] is False
  assert any(
      e.content
      and e.content.parts
      and e.content.parts[0].text == 'halted by plugin'
      for e in events
  )


@pytest.mark.asyncio
async def test_workflow_dispatches_after_run_callback_on_before_run_early_exit(
    monkeypatch: pytest.MonkeyPatch,
):
  """A plugin after_run_callback and post-invocation compaction must be dispatched even when before_run_callback early exits with Content."""

  ran = {'node': False, 'after_run': False, 'compaction': False}

  class _RecordingNode(BaseNode):

    @override
    async def run(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      ran['node'] = True
      yield Event(output='should not run')

  class _HaltWithAfterRunPlugin(BasePlugin):

    def __init__(self):
      super().__init__(name='halt_with_after_run_plugin')

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> types.Content:
      return types.Content(
          role='model', parts=[types.Part(text='halted by plugin')]
      )

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
      ran['after_run'] = True

  graph = Graph(edges=[Edge(from_node=START, to_node=_RecordingNode(name='A'))])
  wf = Workflow(name='halt_wf', graph=graph)

  ss = InMemorySessionService()
  app = App(name='test', root_agent=wf, plugins=[_HaltWithAfterRunPlugin()])
  runner = Runner(app=app, session_service=ss)

  original_compaction = runner._run_post_invocation_compaction

  async def _mock_compaction(*args, **kwargs):
    ran['compaction'] = True
    await original_compaction(*args, **kwargs)

  monkeypatch.setattr(
      runner, '_run_post_invocation_compaction', _mock_compaction
  )

  session = await ss.create_session(app_name='test', user_id='u')
  msg = types.Content(parts=[types.Part(text='start')], role='user')
  events = [
      event
      async for event in runner.run_async(
          user_id='u', session_id=session.id, new_message=msg
      )
  ]

  assert ran['node'] is False
  assert ran['after_run'] is True
  assert ran['compaction'] is True


# --- Resuming an invocation that failed ---


def _resumable_runner(
    wf: Workflow, request: pytest.FixtureRequest
) -> testing_utils.InMemoryRunner:
  app = App(
      name=request.function.__name__,
      root_agent=wf,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  return testing_utils.InMemoryRunner(app=app)


@pytest.mark.asyncio
async def test_failed_node_reruns_on_resume(request: pytest.FixtureRequest):
  """A node that raised is rerun on resume, and feeds its real output down."""
  upstream_runs = 0
  flaky_runs = 0
  downstream_inputs = []

  @node()
  async def upstream_node(ctx: Context):
    nonlocal upstream_runs
    upstream_runs += 1
    return 'upstream output'

  @node()
  async def flaky_node(ctx: Context, node_input: Any):
    nonlocal flaky_runs
    flaky_runs += 1
    if flaky_runs == 1:
      raise CustomError('Node failed')
    return 'flaky output'

  @node()
  async def downstream_node(ctx: Context, node_input: Any):
    downstream_inputs.append(node_input)
    return 'downstream output'

  wf = Workflow(
      name='test_resume_after_failure',
      edges=[
          (START, upstream_node),
          (upstream_node, flaky_node),
          (flaky_node, downstream_node),
      ],
  )
  runner = _resumable_runner(wf, request)

  with pytest.raises(CustomError, match='Node failed'):
    await runner.run_async(testing_utils.get_user_content('start'))
  invocation_id = runner.session.events[0].invocation_id

  assert downstream_inputs == []

  await runner.run_async(invocation_id=invocation_id)

  assert flaky_runs == 2
  # The downstream node sees the rerun's output, not the failure's None.
  assert downstream_inputs == ['flaky output']
  # The node that already succeeded is still fast-forwarded.
  assert upstream_runs == 1


@pytest.mark.asyncio
async def test_completed_node_fast_forwarded_on_resume(
    request: pytest.FixtureRequest,
):
  """A node that completed is replayed from history, output and all."""
  completed_runs = 0
  flaky_runs = 0
  flaky_inputs = []

  @node()
  async def completed_node(ctx: Context):
    nonlocal completed_runs
    completed_runs += 1
    return 'completed output'

  @node()
  async def flaky_node(ctx: Context, node_input: Any):
    nonlocal flaky_runs
    flaky_runs += 1
    flaky_inputs.append(node_input)
    if flaky_runs == 1:
      raise CustomError('Node failed')
    return 'flaky output'

  wf = Workflow(
      name='test_resume_fast_forward',
      edges=[
          (START, completed_node),
          (completed_node, flaky_node),
      ],
  )
  runner = _resumable_runner(wf, request)

  with pytest.raises(CustomError, match='Node failed'):
    await runner.run_async(testing_utils.get_user_content('start'))
  invocation_id = runner.session.events[0].invocation_id

  await runner.run_async(invocation_id=invocation_id)

  assert completed_runs == 1
  # The rerun is fed the fast-forwarded output rather than a fresh one.
  assert flaky_inputs == ['completed output', 'completed output']


@pytest.mark.asyncio
async def test_retried_node_not_rerun_on_resume(
    request: pytest.FixtureRequest,
):
  """A node that failed and then succeeded in the same turn is not rerun."""
  tracker = {'iteration_count': 0}
  flaky = _FlakyNode(
      name='FlakyNode',
      message='flaky output',
      succeed_on_iteration=2,
      tracker=tracker,
      exception_to_raise=CustomRetryableError('Transient error'),
      retry_config=RetryConfig(
          initial_delay=0.0,
          exceptions=['CustomRetryableError'],
      ),
  )
  pause = RequestInputNode(name='PauseNode', message='approve?')

  wf = Workflow(
      name='test_resume_after_retry',
      edges=[
          (START, flaky),
          (flaky, pause),
      ],
  )
  runner = _resumable_runner(wf, request)

  events = await runner.run_async(testing_utils.get_user_content('start'))
  flaky_in_graph = next(n for n in wf.graph.nodes if n.name == 'FlakyNode')
  # One failed attempt then one that produced output, both in this turn.
  assert flaky_in_graph.tracker['iteration_count'] == 2

  req_events = get_request_input_events(events)
  interrupt_id = get_request_input_interrupt_ids(req_events[0])[0]
  user_input = create_request_input_response(interrupt_id, {'ok': True})
  await runner.run_async(
      new_message=testing_utils.UserContent(user_input),
      invocation_id=events[0].invocation_id,
  )

  assert flaky_in_graph.tracker['iteration_count'] == 2


@pytest.mark.asyncio
async def test_failed_node_in_nested_workflow_reruns_on_resume(
    request: pytest.FixtureRequest,
):
  """A failure inside a nested workflow reruns that node, not its parent's."""
  inner_flaky_runs = 0
  inner_downstream_inputs = []

  @node()
  async def inner_flaky_node(ctx: Context):
    nonlocal inner_flaky_runs
    inner_flaky_runs += 1
    if inner_flaky_runs == 1:
      raise CustomError('Node failed')
    return 'inner output'

  @node()
  async def inner_downstream_node(ctx: Context, node_input: Any):
    inner_downstream_inputs.append(node_input)
    return 'inner downstream output'

  inner_wf = Workflow(
      name='inner_workflow',
      edges=[
          (START, inner_flaky_node),
          (inner_flaky_node, inner_downstream_node),
      ],
  )

  outer_upstream_runs = 0

  @node()
  async def outer_upstream_node(ctx: Context):
    nonlocal outer_upstream_runs
    outer_upstream_runs += 1
    return 'outer upstream output'

  wf = Workflow(
      name='outer_workflow',
      edges=[
          (START, outer_upstream_node),
          (outer_upstream_node, inner_wf),
      ],
  )
  runner = _resumable_runner(wf, request)

  with pytest.raises(CustomError, match='Node failed'):
    await runner.run_async(testing_utils.get_user_content('start'))
  invocation_id = runner.session.events[0].invocation_id

  await runner.run_async(invocation_id=invocation_id)

  assert inner_flaky_runs == 2
  assert inner_downstream_inputs == ['inner output']
  assert outer_upstream_runs == 1
