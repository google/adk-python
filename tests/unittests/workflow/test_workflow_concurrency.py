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

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.context import Context
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.workflow import BaseNode
from google.adk.workflow import FunctionNode
from google.adk.workflow import JoinNode
from google.adk.workflow import START
from google.adk.workflow._parallel_worker import _ParallelWorker as ParallelWorker
from google.adk.workflow._workflow import Workflow
from google.adk.workflow.utils import _replay_manager
from google.adk.workflow.utils import _replay_sequence_barrier
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


@pytest.mark.asyncio
async def test_max_concurrency_counts_released_transfers(
    request: pytest.FixtureRequest,
):
  """Resolved-transfer fast-forwards neither deadlock nor overrun the limit.

  Setup:
    Resumable workflow with max_concurrency=1: LlmAgents `a1` and `a2` (each
    with its own sub-agent and a long-running tool) beside a FunctionNode `b`
    that requests input and is followed by FunctionNode `c`, all meeting in a
    JoinNode.
  Act:
    - Turn 1: `a1`, `a2` and `b` pause on their interrupts.
    - Turns 2 and 3: answer `a1`'s, then `a2`'s, interrupt so each transfers
      to its sub-agent, which pauses (placing `a1@1` and `a2@1` after `b@1` on
      the replay barrier).
    - Turn 4: answer `b`'s and both sub-agents' interrupts together.
  Assert:
    - The workflow completes instead of timing out on the replay barrier.
    - The two transfer targets and `c` never run at the same time.
    - Every branch completes and `JoinNode` fires.
  """

  def wait_for_approval() -> None:
    return None

  async def ask_b(ctx: Context, node_input: Any) -> AsyncGenerator[Any, None]:
    del node_input
    if ctx.resume_inputs and 'ask-b' in ctx.resume_inputs:
      yield 'b_done'
      return
    yield RequestInput(interrupt_id='ask-b', message='approve?')

  running = 0
  peak = 0

  async def hold_model(
      callback_context: CallbackContext, llm_request: LlmRequest
  ) -> None:
    nonlocal running, peak
    del callback_context, llm_request
    running += 1
    peak = max(peak, running)
    await asyncio.sleep(0.05)

  def release_model(
      callback_context: CallbackContext, llm_response: LlmResponse
  ) -> None:
    nonlocal running
    del callback_context, llm_response
    running -= 1

  async def record_c(node_input: Any) -> str:
    nonlocal running, peak
    del node_input
    running += 1
    peak = max(peak, running)
    await asyncio.sleep(0.05)
    running -= 1
    return 'c_done'

  def find_call_id(events: list[Event], name: str, author: str) -> str:
    for event in events:
      if event.author != author or not event.content:
        continue
      for part in event.content.parts or []:
        if part.function_call and part.function_call.name == name:
          return part.function_call.id or ''
    raise AssertionError(f'Missing function call {name!r} from {author!r}')

  def answer(call_id: str, name: str) -> types.Part:
    return types.Part(
        function_response=types.FunctionResponse(
            id=call_id, name=name, response={'result': 'ok'}
        )
    )

  tool = LongRunningFunctionTool(func=wait_for_approval)

  def transfer_agent(index: int) -> LlmAgent:
    sub = LlmAgent(
        name=f'sub{index}',
        model=testing_utils.MockModel.create(
            responses=[
                types.Part.from_function_call(
                    name='wait_for_approval', args={}
                ),
                types.Part.from_text(text=f'sub{index}_done'),
            ]
        ),
        tools=[tool],
        before_model_callback=hold_model,
        after_model_callback=release_model,
    )
    return LlmAgent(
        name=f'a{index}',
        model=testing_utils.MockModel.create(
            responses=[
                types.Part.from_function_call(
                    name='wait_for_approval', args={}
                ),
                types.Part.from_function_call(
                    name='transfer_to_agent', args={'agent_name': sub.name}
                ),
            ]
        ),
        sub_agents=[sub],
        tools=[tool],
    )

  a1 = transfer_agent(1)
  a2 = transfer_agent(2)
  b = FunctionNode(func=ask_b, name='b', rerun_on_resume=True)
  c = FunctionNode(func=record_c, name='c')
  join = JoinNode(name='join')
  workflow = Workflow(
      name='root',
      edges=[(START, (a1, a2, b)), (b, c), ((a1, a2, c), join)],
      max_concurrency=1,
  )
  app = App(
      name=request.function.__name__,
      root_agent=workflow,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  events1 = await runner.run_async(testing_utils.get_user_content('start'))
  invocation_id = events1[0].invocation_id

  sub_call_ids = []
  for index in (1, 2):
    events = await runner.run_async(
        new_message=types.Content(
            role='user',
            parts=[
                answer(
                    find_call_id(events1, 'wait_for_approval', f'a{index}'),
                    'wait_for_approval',
                )
            ],
        ),
        invocation_id=invocation_id,
    )
    sub_call_ids.append(
        find_call_id(events, 'wait_for_approval', f'sub{index}')
    )

  events_last = await runner.run_async(
      new_message=types.Content(
          role='user',
          parts=[
              answer('ask-b', 'adk_request_input'),
              *(answer(i, 'wait_for_approval') for i in sub_call_ids),
          ],
      ),
      invocation_id=invocation_id,
  )
  texts = [
      part.text
      for event in events_last
      if event.content and event.content.parts
      for part in event.content.parts
      if part.text
  ]
  outputs = [e.output for e in events_last if e.output is not None]
  assert peak == 1
  assert 'sub1_done' in texts
  assert 'sub2_done' in texts
  assert {'a1': None, 'a2': None, 'c': 'c_done'} in outputs


@pytest.mark.asyncio
async def test_max_concurrency_held_transfer_reports_divergence(
    monkeypatch: pytest.MonkeyPatch,
):
  """A held transfer nothing can release still fails on the replay barrier.

  Setup:
    Resumable workflow with max_concurrency=1: LlmAgent `a` (with a sub-agent
    and a long-running tool) beside a FunctionNode `b` that requests input.
    The barrier timeout is shortened.
  Act:
    - Turn 1: `a` and `b` pause on their interrupts.
    - Turn 2: answer `a`'s interrupt so it transfers to its sub-agent, which
      pauses (placing `a@1` after `b@1` on the replay barrier).
    - Turn 3: resume a workflow without `b`, answering the sub-agent.
  Assert:
    - The run raises a replay divergence instead of finishing without `a`.
  """
  monkeypatch.setattr(
      _replay_manager,
      'ReplaySequenceBarrier',
      functools.partial(
          _replay_sequence_barrier.ReplaySequenceBarrier, timeout_sec=0.1
      ),
  )

  def wait_for_approval() -> None:
    return None

  async def ask_b(ctx: Context, node_input: Any) -> AsyncGenerator[Any, None]:
    del ctx, node_input
    yield RequestInput(interrupt_id='ask-b', message='approve?')

  def build_runner(*, with_b: bool) -> Runner:
    tool = LongRunningFunctionTool(func=wait_for_approval)
    call = types.Part.from_function_call(name='wait_for_approval', args={})
    sub = LlmAgent(
        name='sub',
        model=testing_utils.MockModel.create(responses=[call]),
        tools=[tool],
    )
    a = LlmAgent(
        name='a',
        model=testing_utils.MockModel.create(
            responses=[
                call,
                types.Part.from_function_call(
                    name='transfer_to_agent', args={'agent_name': 'sub'}
                ),
            ]
        ),
        sub_agents=[sub],
        tools=[tool],
    )
    b = FunctionNode(func=ask_b, name='b', rerun_on_resume=True)
    workflow = Workflow(
        name='root',
        edges=[(START, (a, b) if with_b else (a,))],
        max_concurrency=1,
    )
    app = App(
        name='diverge',
        root_agent=workflow,
        resumability_config=ResumabilityConfig(is_resumable=True),
    )
    return Runner(app=app, session_service=sessions)

  def answer(events: list[Event], author: str) -> types.Content:
    call_id = next(
        part.function_call.id
        for event in events
        if event.author == author and event.content
        for part in event.content.parts or []
        if part.function_call and part.function_call.name == 'wait_for_approval'
    )
    return types.Content(
        role='user',
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=call_id, name='wait_for_approval', response={}
                )
            )
        ],
    )

  sessions = InMemorySessionService()
  session = await sessions.create_session(app_name='diverge', user_id='user')

  async def run(
      runner: Runner, message: types.Content, invocation_id: str | None
  ) -> list[Event]:
    return [
        event
        async for event in runner.run_async(
            user_id='user',
            session_id=session.id,
            new_message=message,
            invocation_id=invocation_id,
        )
    ]

  runner = build_runner(with_b=True)
  events1 = await run(runner, testing_utils.get_user_content('start'), None)
  invocation_id = events1[0].invocation_id
  events2 = await run(runner, answer(events1, 'a'), invocation_id)

  with pytest.raises(RuntimeError, match='Replay divergence detected'):
    await run(build_runner(with_b=False), answer(events2, 'sub'), invocation_id)
