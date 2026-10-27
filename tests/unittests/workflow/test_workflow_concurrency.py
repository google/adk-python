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
import functools
from typing import Any
from typing import AsyncGenerator

from google.adk.agents.context import Context
from google.adk.apps.app import App
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow import BaseNode
from google.adk.workflow import FunctionNode
from google.adk.workflow import JoinNode
from google.adk.workflow import START
from google.adk.workflow._parallel_worker import _ParallelWorker as ParallelWorker
from google.adk.workflow._workflow import Workflow
from google.genai import types
from pydantic import Field
import pytest
from typing_extensions import override

from .. import testing_utils
from .workflow_testing_utils import create_parent_invocation_context


@pytest.mark.asyncio
async def test_max_concurrency_limits_running_nodes(
    request: pytest.FixtureRequest,
):
  """Max concurrency limits the number of parallel graph-scheduled nodes.

  Setup:
    Workflow with 4 parallel nodes and max_concurrency=2.
  Act:
    - Start workflow in background.
    - Release nodes one by one.
  Assert:
    - Initially only 2 nodes start.
    - Releasing one node allows another to start.
    - All nodes eventually complete.
  """

  class ConcurrencyWorkerNode(BaseNode):
    """A node that signals when it starts and waits for a signal to finish."""

    started_event: asyncio.Event
    finish_event: asyncio.Event

    @override
    async def _run_impl(
        self,
        *,
        ctx: Context,
        node_input: Any,
    ) -> AsyncGenerator[Any, None]:
      self.started_event.set()
      await self.finish_event.wait()
      yield f'{self.name}_done'

  class TerminalNode(BaseNode):

    @override
    async def _run_impl(
        self, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield 'workflow_done'

  # Given a workflow with 4 parallel nodes and max_concurrency=2
  num_nodes = 4
  max_concurrency = 2
  started_events = [asyncio.Event() for _ in range(num_nodes)]
  finish_events = [asyncio.Event() for _ in range(num_nodes)]

  nodes = [
      ConcurrencyWorkerNode(
          name=f'Node{i}',
          started_event=started_events[i],
          finish_event=finish_events[i],
      )
      for i in range(num_nodes)
  ]

  terminal_node = TerminalNode(name='Terminal')
  edges = [(START, tuple(nodes), terminal_node)]

  agent = Workflow(
      name='concurrency_agent',
      max_concurrency=max_concurrency,
      edges=edges,
  )

  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)

  # When the workflow is run in background
  async def run_agent():
    return await runner.run_async(testing_utils.get_user_content('start'))

  run_task = asyncio.create_task(run_agent())

  # Then initially only max_concurrency nodes should start
  await asyncio.sleep(0.1)
  started_count = sum(1 for e in started_events if e.is_set())
  assert started_count == max_concurrency

  # When one node is released
  for i in range(num_nodes):
    if started_events[i].is_set():
      finish_events[i].set()
      break

  # Then another node should start, bringing total to max_concurrency + 1
  await asyncio.sleep(0.1)
  started_count = sum(1 for e in started_events if e.is_set())
  assert started_count == max_concurrency + 1

  # When all remaining nodes are released
  for e in finish_events:
    e.set()

  # Then all nodes should eventually complete
  await run_task
  started_count = sum(1 for e in started_events if e.is_set())
  assert started_count == num_nodes


@pytest.mark.parametrize('max_concurrency', [-1, 0])
@pytest.mark.asyncio
async def test_non_positive_max_concurrency_treated_as_unlimited(
    request: pytest.FixtureRequest,
    max_concurrency: int,
):
  """A max_concurrency of 0 or less is unlimited and does not wedge."""

  class SimpleWorkerNode(BaseNode):

    @override
    async def _run_impl(
        self,
        *,
        ctx: Context,
        node_input: Any,
    ) -> AsyncGenerator[Any, None]:
      yield f'{self.name}_done'

  class TerminalNode(BaseNode):

    @override
    async def _run_impl(
        self, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield 'workflow_done'

  nodes = [SimpleWorkerNode(name=f'Node{i}') for i in range(3)]
  terminal_node = TerminalNode(name='Terminal')
  agent = Workflow(
      name='negative_concurrency_agent',
      max_concurrency=max_concurrency,
      edges=[(START, tuple(nodes), terminal_node)],
  )

  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))
  outputs = [e.output for e in events if e.output is not None]
  assert 'workflow_done' in outputs


@pytest.mark.parametrize('max_concurrency', [1, 2, None])
@pytest.mark.asyncio
async def test_max_concurrency_resume_does_not_deadlock_on_replay(
    max_concurrency: int | None,
):
  """Replayed nodes waiting on the replay barrier must not hold a slot.

  Setup:
    A loop that requests input twice, beside a three-node sibling chain, both
    meeting in a JoinNode.
  Act:
    Run, then resume twice by answering each RequestInput.
  Assert:
    - The workflow completes instead of timing out on the replay barrier.
    - Nodes that really execute never exceed max_concurrency.
  """
  rounds = 2
  running = 0
  peak = 0

  def tracked(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
      nonlocal running, peak
      running += 1
      peak = max(peak, running)
      try:
        await asyncio.sleep(0.01)
        return func(*args, **kwargs)
      finally:
        running -= 1

    return wrapper

  def start(node_input: dict):
    return {'round': 0}

  async def ask(ctx, node_input: dict):
    ctx.state['round'] = node_input['round']
    yield RequestInput(
        interrupt_id=f"ask-{node_input['round']}", message='approve?'
    )

  def answered(ctx, node_input):
    return {'round': ctx.state['round']}

  def again(node_input: dict):
    state = {'round': node_input['round'] + 1}
    return Event(route=state['round'] < rounds, output=state)

  def other(node_input: dict):
    return node_input

  start_node = FunctionNode(func=tracked(start), name='start')
  answered_node = FunctionNode(func=tracked(answered), name='answered')
  again_node = FunctionNode(func=tracked(again), name='again')
  chain = [
      FunctionNode(func=tracked(other), name=f'other{i}') for i in range(3)
  ]
  asking = FunctionNode(func=ask, name='ask', rerun_on_resume=False)
  join = JoinNode(name='join')
  workflow = Workflow(
      name='root',
      edges=[
          (START, (start_node, chain[0])),
          (start_node, asking),
          (asking, answered_node),
          (answered_node, again_node),
          (again_node, {True: asking, False: join}),
          *zip(chain, chain[1:]),
          (chain[-1], join),
      ],
      max_concurrency=max_concurrency,
  )

  sessions = InMemorySessionService()
  runner = Runner(app_name='repro', node=workflow, session_service=sessions)
  session = await sessions.create_session(app_name='repro', user_id='user')
  message = types.Content(role='user', parts=[types.Part(text='{}')])
  answered_ids = []
  while True:
    pending = []
    async for event in runner.run_async(
        user_id='user', session_id=session.id, new_message=message
    ):
      if event.content:
        pending += [
            part.function_call.id
            for part in event.content.parts or []
            if part.function_call
        ]
    if not pending:
      break
    answered_ids += pending
    message = types.Content(
        role='user',
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=call, name='adk_request_input', response={'ok': True}
                )
            )
            for call in pending
        ],
    )

  assert answered_ids == ['ask-0', 'ask-1']
  if max_concurrency is not None:
    assert peak <= max_concurrency
  if max_concurrency == 1:
    assert peak == 1
