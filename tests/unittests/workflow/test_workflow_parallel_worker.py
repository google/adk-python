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

import asyncio
from typing import Any
from typing import AsyncGenerator

from google.adk.agents.context import Context
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.workflow import BaseNode
from google.adk.workflow import START
from google.adk.workflow._node import node
from google.adk.workflow._parallel_worker import _ParallelWorker as ParallelWorker
from google.adk.workflow._workflow import Workflow
from google.adk.workflow.utils._workflow_hitl_utils import get_request_input_interrupt_ids
from google.adk.workflow.utils._workflow_hitl_utils import has_request_input_function_call
from google.genai import types
from pydantic import ConfigDict
from pydantic import Field
import pytest
from typing_extensions import override

from . import testing_utils
from .workflow_testing_utils import simplify_events_with_node


class _ProducerNode(BaseNode):
  """A node that produces a list of items."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  items: list[Any] = Field(default_factory=list)
  name: str = Field(default='Producer')

  def __init__(self, items: list[Any], name: str = 'Producer'):
    super().__init__()
    object.__setattr__(self, 'items', items)
    object.__setattr__(self, 'name', name)

  @override
  async def run(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    yield Event(output=self.items)


class _SingleItemProducerNode(BaseNode):
  """A node that produces a single item."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  item: Any = None
  name: str = Field(default='Producer')

  def __init__(self, item: Any, name: str = 'Producer'):
    super().__init__()
    object.__setattr__(self, 'item', item)
    object.__setattr__(self, 'name', name)

  @override
  async def run(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    yield Event(output=self.item)


class _WorkerNode(BaseNode):
  """A node that processes an item, with an optional delay."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  name: str = Field(default='Worker')

  def __init__(self, name: str = 'Worker'):
    super().__init__()
    object.__setattr__(self, 'name', name)

  @override
  async def run(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    if isinstance(node_input, dict) and 'delay' in node_input:
      await asyncio.sleep(node_input['delay'])
      val = node_input['val']
    else:
      val = node_input
    yield Event(output=f'{val}_processed')


@pytest.mark.asyncio
async def test_parallel_worker_processes_list_ordered(
    request: pytest.FixtureRequest,
):
  """ParallelWorker processes a list of items and returns ordered results."""
  # Given a workflow with a producer and a parallel worker.
  # Delays are used to ensure deterministic output order and prevent race conditions in event ordering.
  items = [{'val': 'item1', 'delay': 0}, {'val': 'item2', 'delay': 0.1}]
  node_a = _ProducerNode(items=items, name='NodeA')
  worker = ParallelWorker(node=_WorkerNode(name='Worker'))

  agent = Workflow(
      name='test_agent',
      edges=[
          (START, node_a),
          (node_a, worker),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # When the workflow is run
  events = await runner.run_async(testing_utils.get_user_content('start'))

  # Then results should be ordered and match expected events
  simplified_events = simplify_events_with_node(events)

  assert simplified_events == [
      (
          'test_agent@1/NodeA@1',
          {
              'output': [
                  {'val': 'item1', 'delay': 0},
                  {'val': 'item2', 'delay': 0.1},
              ],
          },
      ),
      (
          'test_agent@1/Worker@1/Worker@1',
          {'output': 'item1_processed'},
      ),
      (
          'test_agent@1/Worker@1/Worker@2',
          {'output': 'item2_processed'},
      ),
      (
          'test_agent@1/Worker@1',
          {
              'output': ['item1_processed', 'item2_processed'],
          },
      ),
  ]


@pytest.mark.asyncio
async def test_parallel_worker_with_empty_input_returns_empty_list(
    request: pytest.FixtureRequest,
):
  """ParallelWorker with empty input returns an empty list."""
  # Given a workflow with a producer yielding empty list
  items = []
  node_a = _ProducerNode(items=items, name='NodeA')
  worker = ParallelWorker(node=_WorkerNode(name='Worker'))

  agent = Workflow(
      name='test_empty_agent',
      edges=[
          (START, node_a),
          (node_a, worker),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # When the workflow is run
  events = await runner.run_async(testing_utils.get_user_content('start'))

  # Then output aggregator should return empty list
  simplified_events = simplify_events_with_node(events)

  assert simplified_events == [
      (
          'test_empty_agent@1/NodeA@1',
          {
              'output': [],
          },
      ),
      (
          'test_empty_agent@1/Worker@1',
          {
              'output': [],
          },
      ),
  ]


@pytest.mark.asyncio
async def test_parallel_worker_wraps_single_item_in_list(
    request: pytest.FixtureRequest,
):
  """ParallelWorker wraps a single non-list item into a one-element list."""
  # Given a workflow with a producer yielding a single item (not a list)
  item = {'val': 'item1', 'delay': 0}
  node_a = _SingleItemProducerNode(item=item, name='NodeA')
  worker = ParallelWorker(node=_WorkerNode(name='Worker'))

  agent = Workflow(
      name='test_single_item_agent',
      edges=[
          (START, node_a),
          (node_a, worker),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # When the workflow is run
  events = await runner.run_async(testing_utils.get_user_content('start'))

  # Then it should be wrapped and processed as a single-element list
  simplified_events = simplify_events_with_node(events)

  assert simplified_events == [
      (
          'test_single_item_agent@1/NodeA@1',
          {
              'output': {'val': 'item1', 'delay': 0},
          },
      ),
      (
          'test_single_item_agent@1/Worker@1/Worker@1',
          {'output': 'item1_processed'},
      ),
      (
          'test_single_item_agent@1/Worker@1',
          {
              'output': ['item1_processed'],
          },
      ),
  ]


async def _worker_func(node_input: dict[str, Any]) -> AsyncGenerator[Any, None]:
  if isinstance(node_input, dict) and 'delay' in node_input:
    await asyncio.sleep(node_input['delay'])
    val = node_input['val']
  else:
    val = node_input
  yield f'{val}_processed'


@pytest.mark.asyncio
async def test_parallel_worker_accepts_plain_function(
    request: pytest.FixtureRequest,
):
  """ParallelWorker accepts a plain function as the wrapped node."""
  # Given a workflow with a producer and a parallel worker wrapping a function
  items = [{'val': 'item1', 'delay': 0}, {'val': 'item2', 'delay': 0.1}]
  node_a = _ProducerNode(items=items, name='NodeA')
  worker = ParallelWorker(node=_worker_func)

  agent = Workflow(
      name='test_agent',
      edges=[
          (START, node_a),
          (node_a, worker),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # When the workflow is run
  events = await runner.run_async(testing_utils.get_user_content('start'))

  # Then it should process items correctly
  simplified_events = simplify_events_with_node(events)

  assert simplified_events == [
      (
          'test_agent@1/NodeA@1',
          {
              'output': [
                  {'val': 'item1', 'delay': 0},
                  {'val': 'item2', 'delay': 0.1},
              ],
          },
      ),
      (
          'test_agent@1/_worker_func@1/_worker_func@1',
          {
              'output': 'item1_processed',
          },
      ),
      (
          'test_agent@1/_worker_func@1/_worker_func@2',
          {
              'output': 'item2_processed',
          },
      ),
      (
          'test_agent@1/_worker_func@1',
          {
              'output': ['item1_processed', 'item2_processed'],
          },
      ),
  ]


@pytest.mark.asyncio
async def test_parallel_worker_failure_propagates_and_cancels_others(
    request: pytest.FixtureRequest,
):
  """One worker failure cancels remaining workers and propagates the exception.

  Setup: 3 items — task-1 completes fast, task-2 fails after delay,
    task-3 is slow.
  Assert:
    - task-1 finishes before the failure.
    - task-2's ValueError propagates to the runner.
    - task-3 is cancelled (never finishes).
  """
  # Given a worker that fails on task-2
  items = ['task-1', 'task-2', 'task-3']
  node_a = _ProducerNode(items=items, name='NodeA')

  tracker = {}
  task_3_done_cancelled = False

  async def _worker_failable_func(node_input: str) -> AsyncGenerator[Any, None]:
    if node_input == 'task-1':
      yield f'{node_input}_processed'
    elif node_input == 'task-2':
      await asyncio.sleep(0.05)
      raise ValueError(f'{node_input} failed')
    elif node_input == 'task-3':
      try:
        await asyncio.sleep(0.1)
      except asyncio.CancelledError:
        nonlocal task_3_done_cancelled
        task_3_done_cancelled = True
        raise
      yield f'{node_input}_processed'

    tracker[node_input] = True

  worker = ParallelWorker(node=_worker_failable_func)

  agent = Workflow(
      name='test_agent_fail',
      edges=[
          (START, node_a),
          (node_a, worker),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  events = []

  # When running the agent, expect it to raise the worker exception
  with pytest.raises(ValueError, match='task-2 failed'):
    async for event in runner.runner.run_async(
        user_id=runner.session.user_id,
        session_id=runner.session.id,
        new_message=testing_utils.get_user_content('start'),
    ):
      events.append(event)

  # Then assert that the error was captured in an event and state is correct
  assert any(
      event.error_code == 'ValueError'
      and event.error_message == 'task-2 failed'
      for event in events
  )
  assert tracker == {'task-1': True}
  assert task_3_done_cancelled

  simplified_events = simplify_events_with_node(events)

  assert simplified_events == [
      (
          'test_agent_fail@1/NodeA@1',
          {
              'output': ['task-1', 'task-2', 'task-3'],
          },
      ),
      (
          'test_agent_fail@1/_worker_failable_func@1/_worker_failable_func@1',
          {
              'output': 'task-1_processed',
          },
      ),
  ]


class _HitlWorkerNode(BaseNode):
  """A worker node that can request human input."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  rerun_on_resume: bool = Field(default=True)
  name: str = Field(default='Worker')

  def __init__(self, name: str = 'Worker'):
    super().__init__()
    object.__setattr__(self, 'name', name)

  @override
  def get_name(self) -> str:
    return self.name

  @override
  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    val = node_input['val']
    ask = node_input['ask']

    if ask:
      interrupt_id = f'req_{val}'
      if resume_input := ctx.resume_inputs.get(interrupt_id):
        yield Event(output=f"{val}_{resume_input['text']}")
      else:
        yield RequestInput(
            interrupt_id=interrupt_id, message=f'Input for {val}'
        )
    else:
      yield Event(output=f'{val}_processed')


@pytest.mark.asyncio
@pytest.mark.xfail(reason='ctx.run_node needs barrier for parallel HITL')
async def test_parallel_worker_pauses_for_human_input(
    request: pytest.FixtureRequest,
):
  """Worker requesting input pauses the workflow; resume completes all workers.

  Setup: 2 items — item1 completes, item2 requests input.
  Act:
    - Run 1: item1 completes, item2 interrupts.
    - Run 2: resume item2 with FR.
  Assert:
    - Run 1: item1 output emitted, RequestInput for item2.
    - Run 2: item2 output emitted, parent returns full list.
  """
  # Given a workflow with a worker that requests input for item2
  items = [{'val': 'item1', 'ask': False}, {'val': 'item2', 'ask': True}]
  node_a = _ProducerNode(items=items, name='NodeA')
  worker = ParallelWorker(node=_HitlWorkerNode(name='Worker'))

  agent = Workflow(
      name='parallel_worker_hitl_agent',
      edges=[
          (START, node_a),
          (node_a, worker),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # When the workflow is run for the first time
  events1 = await runner.run_async(testing_utils.get_user_content('start'))

  # Then item1 should complete and item2 should interrupt
  req_events = [e for e in events1 if has_request_input_function_call(e)]
  assert len(req_events) == 1
  interrupt_id = get_request_input_interrupt_ids(req_events[0])[0]
  assert interrupt_id == 'req_item2'
  invocation_id = events1[0].invocation_id

  simplified_events1 = simplify_events_with_node(events1)
  assert simplified_events1 == [
      (
          'parallel_worker_hitl_agent@1/NodeA@1',
          {
              'output': [
                  {'val': 'item1', 'ask': False},
                  {'val': 'item2', 'ask': True},
              ],
          },
      ),
      (
          'parallel_worker_hitl_agent@1/Worker@1',
          {'output': 'item1_processed'},
      ),
      (
          'parallel_worker_hitl_agent',
          testing_utils.simplify_content(req_events[0].content),
      ),
  ]

  # When resuming with human input for item2
  user_input = types.Part(
      function_response=types.FunctionResponse(
          id=interrupt_id,
          name='user_input',
          response={'text': 'resumed'},
      )
  )
  events2 = await runner.run_async(
      new_message=testing_utils.UserContent(user_input),
      invocation_id=invocation_id,
  )

  # Then item2 should complete and the full list should be returned
  simplified_events2 = simplify_events_with_node(events2)

  assert simplified_events2 == [
      (
          'parallel_worker_hitl_agent@1/Worker@1',
          {'output': 'item2_resumed'},
      ),
      (
          'parallel_worker_hitl_agent@1/Worker@1',
          {
              'output': ['item1_processed', 'item2_resumed'],
          },
      ),
  ]


class _AsyncWorkerNode(BaseNode):
  """A worker node that waits for an asyncio event before processing an item."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  name: str = Field(default='Worker')
  events: dict[str, asyncio.Event] = Field(default_factory=dict)

  def __init__(
      self,
      name: str = 'Worker',
      events: dict[str, asyncio.Event] | None = None,
  ):
    super().__init__()
    object.__setattr__(self, 'name', name)
    object.__setattr__(self, 'events', events or {})

  @override
  def get_name(self) -> str:
    return self.name

  @override
  async def run(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    if node_input in self.events:
      await self.events[node_input].wait()
    yield Event(output=f'{node_input}_res')


@pytest.mark.asyncio
async def test_parallel_worker_preserves_input_order_regardless_of_completion_order(
    request: pytest.FixtureRequest,
):
  """Final output list preserves input order regardless of worker completion order."""
  # Given items and events to control completion order
  item1 = 'item1'
  item2 = 'item2'
  events_map = {
      item1: asyncio.Event(),
      item2: asyncio.Event(),
  }

  node_a = _ProducerNode(items=[item1, item2], name='NodeA')
  worker = ParallelWorker(
      node=_AsyncWorkerNode(name='Worker', events=events_map)
  )

  agent = Workflow(
      name='out_of_order_agent',
      edges=[
          (START, node_a),
          (node_a, worker),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # When running the workflow (starting in background)
  run_task = asyncio.create_task(
      runner.run_async(testing_utils.get_user_content('start'))
  )

  # Allow the runner to reach the wait point.
  await asyncio.sleep(0.1)

  # Finish item2 first
  events_map[item2].set()
  await asyncio.sleep(0.1)

  # Finish item1 second
  events_map[item1].set()

  events = await run_task

  # Then output should preserve input order
  simplified_events = simplify_events_with_node(events)

  assert simplified_events == [
      (
          'out_of_order_agent@1/NodeA@1',
          {'output': [item1, item2]},
      ),
      (
          'out_of_order_agent@1/Worker@1/Worker@2',
          {'output': 'item2_res'},
      ),
      (
          'out_of_order_agent@1/Worker@1/Worker@1',
          {'output': 'item1_res'},
      ),
      (
          'out_of_order_agent@1/Worker@1',
          {'output': ['item1_res', 'item2_res']},
      ),
  ]


@pytest.mark.asyncio
async def test_parallel_worker_can_wrap_nested_workflow(
    request: pytest.FixtureRequest,
):
  """Nested Workflow wrapped in ParallelWorker processes items through its graph."""
  # Given a workflow wrapping a nested workflow in ParallelWorker
  items = ['item1', 'item2']
  node_a = _ProducerNode(items=items, name='NodeA')

  async def worker_func(node_input: Any):
    return f'{node_input}_processed'

  nested_agent = Workflow(
      name='nested_agent',
      edges=[(START, worker_func)],
  )

  parallel_nested = node(nested_agent, parallel_worker=True)

  outer_agent = Workflow(
      name='outer_agent',
      edges=[
          (START, node_a),
          (node_a, parallel_nested),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # When the workflow is run
  events = await runner.run_async(testing_utils.get_user_content('start'))

  # Then results should be processed by the nested workflow
  simplified_events = simplify_events_with_node(events)

  assert simplified_events == [
      (
          'outer_agent@1/NodeA@1',
          {
              'output': ['item1', 'item2'],
          },
      ),
      (
          'outer_agent@1/nested_agent@1/nested_agent@1/worker_func@1',
          {'output': 'item1_processed'},
      ),
      (
          'outer_agent@1/nested_agent@1/nested_agent@2/worker_func@1',
          {'output': 'item2_processed'},
      ),
      (
          'outer_agent@1/nested_agent@1',
          {
              'output': [
                  'item1_processed',
                  'item2_processed',
              ],
          },
      ),
  ]


@pytest.mark.asyncio
@pytest.mark.xfail(reason='New Workflow has no parallel_worker field')
async def test_workflow_auto_wraps_parallel_worker_when_flag_set(
    request: pytest.FixtureRequest,
):
  """Workflow with parallel_worker=True auto-wraps in ParallelWorker."""

  # Given a nested workflow with parallel_worker=True
  async def producer_func():
    return ['item1', 'item2']

  async def worker_func(node_input: Any):
    return f'{node_input}_processed'

  nested_agent = Workflow(
      name='nested_agent',
      edges=[('START', worker_func)],
      parallel_worker=True,
  )

  outer_agent = Workflow(
      name='outer_agent',
      edges=[
          ('START', producer_func),
          (producer_func, nested_agent),
      ],
  )

  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # When the workflow is run
  events = await runner.run_async(testing_utils.get_user_content('start'))

  # Then it should process items in parallel
  simplified_events = simplify_events_with_node(events)

  assert simplified_events == [
      (
          'outer_agent@1/producer_func@1',
          {
              'output': ['item1', 'item2'],
          },
      ),
      (
          'nested_agent@1/worker_func@1',
          {'output': 'item1_processed'},
      ),
      (
          'nested_agent@1/worker_func@1',
          {'output': 'item2_processed'},
      ),
      (
          'outer_agent@1/nested_agent@1',
          {
              'output': [
                  'item1_processed',
                  'item2_processed',
              ],
          },
      ),
  ]


@pytest.mark.asyncio
async def test_parallel_worker_limits_concurrency(
    request: pytest.FixtureRequest,
):
  """max_concurrency limits the number of concurrent workers at any time."""
  # Given items and events to control concurrency
  items = ['item1', 'item2', 'item3', 'item4']
  started_events = {item: asyncio.Event() for item in items}
  finish_events = {item: asyncio.Event() for item in items}

  async def _concurrency_worker_func(
      node_input: str,
  ) -> AsyncGenerator[Any, None]:
    started_events[node_input].set()
    await finish_events[node_input].wait()
    yield f'{node_input}_processed'

  node_a = _ProducerNode(items=items, name='NodeA')
  worker = ParallelWorker(node=_concurrency_worker_func, max_concurrency=2)

  agent = Workflow(
      name='max_concurrency_agent',
      edges=[
          (START, node_a),
          (node_a, worker),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  events = []

  async def _run_agent():
    async for event in runner.runner.run_async(
        user_id=runner.session.user_id,
        session_id=runner.session.id,
        new_message=testing_utils.get_user_content('start'),
    ):
      events.append(event)

  # When running the agent (starting in background)
  run_task = asyncio.create_task(_run_agent())

  # Then verify that initially only two workers are started
  await asyncio.gather(
      started_events['item1'].wait(),
      started_events['item2'].wait(),
  )

  assert started_events['item1'].is_set()
  assert started_events['item2'].is_set()
  assert not started_events['item3'].is_set()
  assert not started_events['item4'].is_set()

  # Signal second worker to finish
  finish_events['item2'].set()

  # Check that the next worker (third one) is scheduled
  await started_events['item3'].wait()
  assert started_events['item3'].is_set()
  assert not started_events['item4'].is_set()

  # Signal the third worker to be finished. Check 4th worker is scheduled
  finish_events['item3'].set()
  await started_events['item4'].wait()
  assert started_events['item4'].is_set()

  # Finish all workers, and assert the workflow finishes
  finish_events['item1'].set()
  finish_events['item4'].set()

  await run_task

  simplified_events = simplify_events_with_node(events)

  assert simplified_events == [
      (
          'max_concurrency_agent@1/NodeA@1',
          {'output': items},
      ),
      (
          'max_concurrency_agent@1/_concurrency_worker_func@1/_concurrency_worker_func@2',
          {
              'output': 'item2_processed',
          },
      ),
      (
          'max_concurrency_agent@1/_concurrency_worker_func@1/_concurrency_worker_func@3',
          {
              'output': 'item3_processed',
          },
      ),
      (
          'max_concurrency_agent@1/_concurrency_worker_func@1/_concurrency_worker_func@1',
          {
              'output': 'item1_processed',
          },
      ),
      (
          'max_concurrency_agent@1/_concurrency_worker_func@1/_concurrency_worker_func@4',
          {
              'output': 'item4_processed',
          },
      ),
      (
          'max_concurrency_agent@1/_concurrency_worker_func@1',
          {
              'output': [
                  'item1_processed',
                  'item2_processed',
                  'item3_processed',
                  'item4_processed',
              ],
          },
      ),
  ]


@pytest.mark.asyncio
@pytest.mark.skip(reason='Hangs: ctx.run_node needs barrier for parallel HITL')
async def test_parallel_worker_hitl_respects_concurrency_limits(
    request: pytest.FixtureRequest,
):
  """HITL resume under max_concurrency schedules next worker after resolution.

  Setup: 3 items, max_concurrency=2. item1 waits, item2 does HITL,
    item3 does HITL.
  Act:
    - Run 1: item1 and item2 start. item2 interrupts. Signal item1 to finish.
    - Run 2: resume item2. item3 starts and interrupts.
    - Run 3: resume item3. All complete.
  Assert:
    - Run 1: item2 RequestInput, item1 output.
    - Run 2: item2 output, item3 RequestInput.
    - Run 3: item3 output, parent returns full list.
  """
  # Given items and events to control concurrency and HITL
  items = [
      {'val': 'item1', 'ask': False},
      {'val': 'item2', 'ask': True},
      {'val': 'item3', 'ask': True},
  ]
  started_events = {item['val']: asyncio.Event() for item in items}
  finish_events = {item['val']: asyncio.Event() for item in items}

  @node(name='Worker', rerun_on_resume=True)
  async def hitl_concurrency_worker(
      ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    val = node_input['val']
    ask = node_input['ask']

    started_events[val].set()

    if ask:
      interrupt_id = f'req_{val}'
      if resume_input := ctx.resume_inputs.get(interrupt_id):
        yield Event(output=f"{val}_{resume_input['text']}")
      else:
        yield RequestInput(
            interrupt_id=interrupt_id, message=f'Input for {val}'
        )
    else:
      await finish_events[val].wait()
      yield Event(output=f'{val}_processed')

  node_a = _ProducerNode(items=items, name='NodeA')
  worker = ParallelWorker(node=hitl_concurrency_worker, max_concurrency=2)

  agent = Workflow(
      name='max_concurrency_hitl_agent',
      edges=[
          (START, node_a),
          (node_a, worker),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  events = []

  # When running the agent for the first time
  run_task = asyncio.create_task(
      runner.run_async(testing_utils.get_user_content('start'))
  )

  # Then Node 1 and Node 2 should start
  await asyncio.gather(
      started_events['item1'].wait(),
      started_events['item2'].wait(),
  )

  assert started_events['item1'].is_set()
  assert started_events['item2'].is_set()
  assert not started_events['item3'].is_set()

  # Signal node 1 to finish
  finish_events['item1'].set()

  events1 = await run_task
  events.extend(events1)

  req_events = [e for e in events1 if has_request_input_function_call(e)]
  assert len(req_events) == 1
  interrupt_id_2 = get_request_input_interrupt_ids(req_events[0])[0]
  assert interrupt_id_2 == 'req_item2'
  invocation_id_1 = events1[0].invocation_id

  simplified_events1 = simplify_events_with_node(events1)
  assert simplified_events1 == [
      (
          'max_concurrency_hitl_agent@1/NodeA@1',
          {
              'output': items,
          },
      ),
      (
          'max_concurrency_hitl_agent',
          testing_utils.simplify_content(req_events[0].content),
      ),
      (
          'max_concurrency_hitl_agent@1/Worker__0@1',
          {'output': 'item1_processed'},
      ),
  ]

  # When resuming Node 2
  user_input_2 = types.Part(
      function_response=types.FunctionResponse(
          id=interrupt_id_2,
          name='user_input',
          response={'text': 'resumed'},
      )
  )

  run_task_2 = asyncio.create_task(
      runner.run_async(
          new_message=testing_utils.UserContent(user_input_2),
          invocation_id=invocation_id_1,
      )
  )

  # Then node 3 should start and yield RequestInput
  await asyncio.sleep(0.1)
  await started_events['item3'].wait()
  assert started_events['item3'].is_set()

  events2 = await run_task_2
  events.extend(events2)

  req_events_2 = [e for e in events2 if has_request_input_function_call(e)]
  assert len(req_events_2) == 1
  interrupt_id_3 = get_request_input_interrupt_ids(req_events_2[0])[0]
  assert interrupt_id_3 == 'req_item3'
  invocation_id_2 = events2[0].invocation_id

  simplified_events2 = simplify_events_with_node(events2)
  assert simplified_events2 == [
      (
          'max_concurrency_hitl_agent@1/Worker__1@1',
          {'output': 'item2_resumed'},
      ),
      (
          'max_concurrency_hitl_agent',
          testing_utils.simplify_content(req_events_2[0].content),
      ),
  ]

  # When resuming Node 3
  user_input_3 = types.Part(
      function_response=types.FunctionResponse(
          id=interrupt_id_3,
          name='user_input',
          response={'text': 'resumed'},
      )
  )

  run_task_3 = asyncio.create_task(
      runner.run_async(
          new_message=testing_utils.UserContent(user_input_3),
          invocation_id=invocation_id_2,
      )
  )

  events3 = await run_task_3
  events.extend(events3)

  # Then all should complete
  simplified_events3 = simplify_events_with_node(events3)

  assert simplified_events3 == [
      (
          'max_concurrency_hitl_agent@1/Worker__2@1',
          {'output': 'item3_resumed'},
      ),
      (
          'max_concurrency_hitl_agent@1/Worker@1',
          {
              'output': ['item1_processed', 'item2_resumed', 'item3_resumed'],
          },
      ),
  ]
