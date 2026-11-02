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

"""Tests for ctx.run_node(use_as_output=True) dynamic terminal paths."""

from __future__ import annotations

import asyncio
from typing import Any
from typing import AsyncGenerator

from google.adk.agents.context import Context
from google.adk.agents.llm_agent import Agent
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.workflow import BaseNode
from google.adk.workflow import FunctionNode
from google.adk.workflow import node
from google.adk.workflow import START
from google.adk.workflow._workflow import Workflow
from google.genai import types
from pydantic import ConfigDict
import pytest
from typing_extensions import override

from .. import testing_utils
from .workflow_testing_utils import get_outputs as _get_outputs


def _make_app(name: str, agent: Workflow, resumable: bool) -> App:
  return App(
      name=name,
      root_agent=agent,
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )


# ---------------------------------------------------------------------------

# FunctionNode delegates to FunctionNode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('resumable', [True, False])
@pytest.mark.asyncio
async def test_use_as_output_function_to_function(
    request: pytest.FixtureRequest, resumable: bool
):
  """Node A delegates output to dynamic child B via use_as_output=True.

  Only B's output event should appear; A should not emit a duplicate.
  """

  def func_b() -> str:
    return 'from_b'

  node_b = FunctionNode(func=func_b)

  async def func_a(ctx: Context) -> str:
    return await ctx.run_node(node_b, use_as_output=True)

  node_a = FunctionNode(func=func_a, rerun_on_resume=True)

  wf_name = request.node.name.replace('[', '_').replace(']', '')
  agent = Workflow(name=wf_name, edges=[(START, node_a)])
  app = _make_app(wf_name, agent, resumable)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  # Only one output event (from B). A's output is suppressed.
  outputs = _get_outputs(events)
  assert outputs == ['from_b']


# ---------------------------------------------------------------------------
# FunctionNode delegates to Workflow
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('resumable', [True, False])
@pytest.mark.asyncio
async def test_use_as_output_function_to_workflow(
    request: pytest.FixtureRequest, resumable: bool
):
  """Node A delegates output to a Workflow B via use_as_output=True.

  The Workflow's terminal node output should be used as A's output.
  """

  def step_1() -> str:
    return 'step_1_done'

  def step_2(node_input: str) -> str:
    return f'final:{node_input}'

  inner_wf = Workflow(
      name='inner_wf',
      edges=[
          (START, step_1),
          (step_1, step_2),
      ],
  )

  async def func_a(ctx: Context):
    return await ctx.run_node(inner_wf, use_as_output=True)

  node_a = FunctionNode(func=func_a, rerun_on_resume=True)

  wf_name = request.node.name.replace('[', '_').replace(']', '')
  agent = Workflow(name=wf_name, edges=[(START, node_a)])
  app = _make_app(wf_name, agent, resumable)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  # step_2 is the terminal node; its output is func_a's output.
  # step_1's output is intermediate (not terminal), so only step_2 appears.
  outputs = _get_outputs(events)
  assert 'final:step_1_done' in outputs
  # func_a should NOT have a duplicate output event.
  assert outputs.count('final:step_1_done') == 1


# ---------------------------------------------------------------------------
# Custom BaseNode delegates to FunctionNode
# ---------------------------------------------------------------------------


class _DelegatingNode(BaseNode):
  """A custom BaseNode that delegates output via use_as_output."""

  model_config = ConfigDict(arbitrary_types_allowed=True)
  delegate: BaseNode

  @override
  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    await ctx.run_node(self.delegate, use_as_output=True)
    # Intentionally yield nothing — output comes from delegate.
    return
    yield  # Make this an async generator


@pytest.mark.parametrize('resumable', [True, False])
@pytest.mark.asyncio
async def test_use_as_output_custom_node_to_function(
    request: pytest.FixtureRequest, resumable: bool
):
  """Custom BaseNode delegates output to dynamic child via use_as_output."""

  def func_b() -> str:
    return 'delegated_output'

  node_b = FunctionNode(func=func_b)
  node_a = _DelegatingNode(
      name='delegator', delegate=node_b, rerun_on_resume=True
  )

  wf_name = request.node.name.replace('[', '_').replace(']', '')
  agent = Workflow(name=wf_name, edges=[(START, node_a)])
  app = _make_app(wf_name, agent, resumable)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  outputs = _get_outputs(events)
  assert outputs == ['delegated_output']


# ---------------------------------------------------------------------------
# Multiple use_as_output=True calls (fan-out delegation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('resumable', [True, False])
@pytest.mark.asyncio
async def test_use_as_output_multiple_disallowed(
    request: pytest.FixtureRequest, resumable: bool
):
  """V2 engine forbids calling use_as_output=True multiple times from the same node."""

  def func_b() -> str:
    return 'from_b'

  def func_c() -> str:
    return 'from_c'

  node_b = FunctionNode(func=func_b)
  node_c = FunctionNode(func=func_c)

  async def func_a(ctx: Context):
    await ctx.run_node(node_b, use_as_output=True)
    # V2 engine should throw ValueError on second call
    with pytest.raises(
        ValueError, match='already has a use_as_output delegate'
    ):
      await ctx.run_node(node_c, use_as_output=True)
    return 'failure_as_expected'

  node_a = FunctionNode(func=func_a, rerun_on_resume=True)

  wf_name = request.node.name.replace('[', '_').replace(']', '')
  agent = Workflow(name=wf_name, edges=[(START, node_a)])
  app = _make_app(wf_name, agent, resumable)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  outputs = _get_outputs(events)
  # func_a's output is suppressed because it successfully delegated to func_b
  # before failing on func_c. So only func_b's output appears.
  assert outputs == ['from_b']


# ---------------------------------------------------------------------------
# Nested dynamic delegation (A → B → C, all use_as_output=True)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('resumable', [True, False])
@pytest.mark.asyncio
async def test_use_as_output_nested_delegation(
    request: pytest.FixtureRequest, resumable: bool
):
  """Chained delegation: A delegates to B, B delegates to C.

  Only C's output should appear; both A's and B's outputs are suppressed.
  """

  def func_c() -> str:
    return 'from_c'

  node_c = FunctionNode(func=func_c)

  async def func_b(ctx: Context) -> str:
    return await ctx.run_node(node_c, use_as_output=True)

  node_b = FunctionNode(func=func_b, rerun_on_resume=True)

  async def func_a(ctx: Context) -> str:
    return await ctx.run_node(node_b, use_as_output=True)

  node_a = FunctionNode(func=func_a, rerun_on_resume=True)

  wf_name = request.node.name.replace('[', '_').replace(']', '')
  agent = Workflow(name=wf_name, edges=[(START, node_a)])
  app = _make_app(wf_name, agent, resumable)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  outputs = _get_outputs(events)
  assert outputs == ['from_c']


@pytest.mark.parametrize('resumable', [True, False])
@pytest.mark.asyncio
async def test_use_as_output_nested_delegation_with_downstream(
    request: pytest.FixtureRequest, resumable: bool
):
  """Chained delegation with a downstream node that consumes the output.

  A delegates to B, B delegates to C. D is downstream of A and should
  receive C's output as its node_input.
  """

  def func_c() -> str:
    return 'from_c'

  node_c = FunctionNode(func=func_c)

  async def func_b(ctx: Context) -> str:
    return await ctx.run_node(node_c, use_as_output=True)

  node_b = FunctionNode(func=func_b, rerun_on_resume=True)

  async def func_a(ctx: Context) -> str:
    return await ctx.run_node(node_b, use_as_output=True)

  node_a = FunctionNode(func=func_a, rerun_on_resume=True)

  def func_d(node_input: str) -> str:
    return f'received:{node_input}'

  wf_name = request.node.name.replace('[', '_').replace(']', '')
  agent = Workflow(
      name=wf_name,
      edges=[(START, node_a), (node_a, func_d)],
  )
  app = _make_app(wf_name, agent, resumable)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  outputs = _get_outputs(events)
  assert 'received:from_c' in outputs


# ---------------------------------------------------------------------------
# use_as_output=False (default) still emits both events
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('resumable', [True, False])
@pytest.mark.asyncio
async def test_without_use_as_output_emits_both(
    request: pytest.FixtureRequest, resumable: bool
):
  """Without use_as_output, both parent and child emit output events."""

  def func_b() -> str:
    return 'from_b'

  node_b = FunctionNode(func=func_b)

  async def func_a(ctx: Context) -> str:
    return await ctx.run_node(node_b)

  node_a = FunctionNode(func=func_a, rerun_on_resume=True)

  wf_name = request.node.name.replace('[', '_').replace(']', '')
  agent = Workflow(name=wf_name, edges=[(START, node_a)])
  app = _make_app(wf_name, agent, resumable)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  # Both B and A emit output events (no dedup).
  outputs = _get_outputs(events)
  assert outputs == ['from_b', 'from_b']


# ---------------------------------------------------------------------------
# use_as_output with Workflow containing HITL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_use_as_output_workflow_with_hitl(
    request: pytest.FixtureRequest,
):
  """Dynamic Workflow child with HITL node pauses and resumes correctly.

  Requires resumable=True because the inner Workflow's internal node state
  (hitl_step in WAITING) must be persisted across invocations. Without
  resumability, re-spawning the dynamic Workflow starts its graph fresh.

  Flow:
    1. func_a calls ctx.run_node(inner_wf, use_as_output=True)
    2. inner_wf's hitl_step yields RequestInput → workflow pauses
    3. User responds → workflow resumes
    4. inner_wf completes, func_a re-runs with cached result
    5. Only inner_wf's terminal node output appears
  """
  resumable = True

  async def hitl_step():
    yield RequestInput(
        interrupt_id='wf_req1',
        message='enter value',
    )

  def final_step(node_input: Any) -> str:
    text = node_input['text'] if isinstance(node_input, dict) else node_input
    return f'final:{text}'

  inner_wf = Workflow(
      name='inner_wf',
      edges=[
          (START, hitl_step),
          (hitl_step, final_step),
      ],
  )

  async def func_a(ctx: Context):
    return await ctx.run_node(inner_wf, use_as_output=True)

  node_a = FunctionNode(func=func_a, rerun_on_resume=True)

  wf_name = request.node.name.replace('[', '_').replace(']', '')
  agent = Workflow(name=wf_name, edges=[(START, node_a)])
  app = _make_app(wf_name, agent, resumable)
  runner = testing_utils.InMemoryRunner(app=app)

  # Run 1: Should pause at hitl_step.
  events1 = await runner.run_async(testing_utils.get_user_content('start'))
  outputs1 = _get_outputs(events1)
  assert outputs1 == []

  # Resume with user response.
  invocation_id = events1[0].invocation_id
  resume_payload = testing_utils.UserContent(
      types.Part(
          function_response=types.FunctionResponse(
              id='wf_req1',
              name='user_input',
              response={'text': 'hello'},
          )
      )
  )
  events2 = await runner.run_async(
      new_message=resume_payload, invocation_id=invocation_id
  )

  outputs2 = _get_outputs(events2)
  assert 'final:hello' in outputs2
  assert outputs2.count('final:hello') == 1


@pytest.mark.asyncio
async def test_use_as_output_static_node_not_rerun_on_resume(
    request: pytest.FixtureRequest,
):
  """Static node delegating output to dynamic child is not re-run on resume.

  Setup:
    - wf with node_a (FunctionNode) and hitl_step.
    - node_a calls ctx.run_node(node_b, use_as_output=True).
    - hitl_step yields RequestInput to pause.
  Act:
    - Run 1: Run workflow, node_a executes, hitl_step pauses.
    - Run 2: Resume with user response for hitl_step.
  Assert:
    - Run 1: node_b output is emitted as node_a's output.
    - Run 2: node_a is NOT re-run (run_count remains 1).
  """

  run_count = 0

  @node
  def node_b() -> str:
    return 'from_b'

  @node(rerun_on_resume=True)
  async def node_a(ctx: Context):
    nonlocal run_count
    run_count += 1
    return await ctx.run_node(node_b, use_as_output=True)

  node_a.wait_for_output = True

  @node
  async def hitl_step():
    yield RequestInput(
        interrupt_id='pause_req',
        message='pause',
    )

  @node
  def final_step(node_input: Any) -> None:
    return None

  wf_name = request.node.name.replace('[', '_').replace(']', '')
  agent = Workflow(
      name=wf_name,
      edges=[
          (START, node_a),
          (START, hitl_step),
          (hitl_step, final_step),
      ],
  )

  runner = testing_utils.InMemoryRunner(root_agent=agent)

  events1 = await runner.run_async(testing_utils.get_user_content('start'))

  assert run_count == 1
  outputs1 = [e.output for e in events1 if e.output]
  assert 'from_b' in outputs1

  invocation_id = events1[0].invocation_id
  resume_payload = testing_utils.UserContent(
      types.Part(
          function_response=types.FunctionResponse(
              id='pause_req',
              name='user_input',
              response={'text': 'continue'},
          )
      )
  )

  events2 = await runner.run_async(
      new_message=resume_payload, invocation_id=invocation_id
  )

  assert run_count == 1


@pytest.mark.asyncio
async def test_use_as_output_instance_isolation(request: pytest.FixtureRequest):
  """Output delegation is isolated to specific dynamic instances.

  Setup:
    - wf with node_a calling node_b twice in parallel with different run_ids
    ('b1', 'b2').
    - node_b yields RequestInput to pause.
  Act:
    - Run 1: Run workflow, both node_b instances pause.
    - Run 2: Resume only the 'b1' instance.
  Assert:
    - Run 1: Both instances run (run_count_b is 2).
    - Run 2: Only instance 'b1' completes and delegates output.
      Instance 'b2' remains paused and does not emit output.
  """
  run_count_b = 0

  @node
  def node_c() -> str:
    return 'from_c'

  @node(rerun_on_resume=True)
  async def node_b(ctx: Context):
    nonlocal run_count_b

    interrupt_id = f'pause_{ctx.node_path}'
    if ctx.resume_inputs and interrupt_id in ctx.resume_inputs:
      response = ctx.resume_inputs[interrupt_id]
      if (
          response
          and isinstance(response, dict)
          and response.get('text') == 'continue'
      ):
        await ctx.run_node(node_c, use_as_output=True)
        return

    run_count_b += 1
    yield RequestInput(
        interrupt_id=interrupt_id,
        message='pause',
    )

  @node(rerun_on_resume=True)
  async def node_a(ctx: Context):
    task1 = ctx.run_node(node_b, run_id='b1')
    task2 = ctx.run_node(node_b, run_id='b2')
    await asyncio.gather(task1, task2)

  wf_name = request.node.name.replace('[', '_').replace(']', '')
  agent = Workflow(name=wf_name, edges=[(START, node_a)])

  runner = testing_utils.InMemoryRunner(root_agent=agent)

  # Given the workflow is started with parallel instances of node_b
  events1 = await runner.run_async(testing_utils.get_user_content('start'))

  # Then both instances execute and pause
  assert run_count_b == 2

  # Find the interrupt ID for instance b1
  interrupt_id_b1 = None
  for event in events1:
    if event.long_running_tool_ids:
      for interrupt_id in event.long_running_tool_ids:
        if 'b1' in interrupt_id:
          interrupt_id_b1 = interrupt_id
          break

  assert interrupt_id_b1 is not None

  # When resuming only instance b1
  invocation_id = events1[0].invocation_id
  resume_payload = testing_utils.UserContent(
      types.Part(
          function_response=types.FunctionResponse(
              id=interrupt_id_b1,
              name='user_input',
              response={'text': 'continue'},
          )
      )
  )

  events2 = await runner.run_async(
      new_message=resume_payload, invocation_id=invocation_id
  )

  # Then node_b is not re-run, and only one output is emitted from node_c
  assert run_count_b == 2

  outputs2 = [e.output for e in events2 if e.output]
  assert outputs2.count('from_c') == 1


# ---------------------------------------------------------------------------
# use_as_output + use_sub_branch: the delegated answer is visible to later
# delegations under the same claimer, the child's tool traffic is not.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('nested', [False, True], ids=['root', 'rerun-parent'])
@pytest.mark.parametrize('use_as_output', [False, True])
@pytest.mark.asyncio
async def test_claimed_sub_branch_answer_is_visible_to_later_delegations(
    use_as_output: bool, nested: bool
):
  """A dispatcher that claims a sub-branch answer shares it, not the tools.

  Turn 1 goes to the single-turn ``bookings`` (tool call + private thought +
  answer), turn 2 to the chat-mode ``guide``. Both run on their own
  sub-branch. ``guide`` sees the answer only when the dispatcher took it as
  its output, and never sees the tool result or the thought. The session
  keeps both.
  """
  tool_calls: list[str] = []

  def reservation_lookup() -> dict:
    """Looks up the user's reservation."""
    tool_calls.append('lookup')
    return {'pnr': 'PRIVATE_PNR', 'arrival': 'TUE 14:05'}

  booking_model = testing_utils.MockModel.create([
      types.Part(
          function_call=types.FunctionCall(name='reservation_lookup', args={})
      ),
      [
          types.Part(text='PRIVATE_THOUGHT', thought=True),
          types.Part(text='You land on Tuesday at 14:05.'),
      ],
  ])
  guide_model = testing_utils.MockModel.create(['A restaurant suggestion.'])
  bookings = Agent(
      name='bookings', model=booking_model, tools=[reservation_lookup]
  )
  guide = Agent(name='guide', model=guide_model, mode='chat')

  @node(rerun_on_resume=True)
  async def dispatcher(ctx: Context, node_input: Any):
    if ctx.state.get('booked'):
      target, target_input = guide, None
    else:
      target, target_input = bookings, node_input
    answer = await ctx.run_node(
        target,
        node_input=target_input,
        use_sub_branch=True,
        use_as_output=use_as_output,
    )
    ctx.state['booked'] = True
    return answer if use_as_output else None

  @node(rerun_on_resume=True)
  async def turn(ctx: Context, node_input: Any):
    run_id = 'r2' if ctx.state.get('booked') else 'r1'
    return await ctx.run_node(
        dispatcher,
        node_input=node_input,
        run_id=run_id,
        use_sub_branch=True,
        use_as_output=use_as_output,
    )

  agent = Workflow(
      name='dispatch_wf', edges=[(START, turn if nested else dispatcher)]
  )
  runner = testing_utils.InMemoryRunner(
      app=_make_app('dispatch_wf', agent, False)
  )

  await runner.run_async(testing_utils.get_user_content('When do I land?'))
  await runner.run_async(testing_utils.get_user_content('Dinner that evening?'))

  assert tool_calls == ['lookup']
  assert len(guide_model.requests) == 1
  request_text = '\n'.join(
      part.text or ''
      for content in guide_model.requests[0].contents
      for part in content.parts or []
  )
  assert ('[bookings] said:' in request_text) == use_as_output
  assert ('You land on Tuesday at 14:05.' in request_text) == use_as_output
  assert 'When do I land?' in request_text
  assert 'PRIVATE_PNR' not in request_text
  assert 'PRIVATE_THOUGHT' not in request_text

  saved = runner.session
  assert 'PRIVATE_PNR' in saved.model_dump_json()
  assert 'PRIVATE_THOUGHT' in saved.model_dump_json()
  lookup = next(e for e in saved.events if e.get_function_responses())
  answer = next(
      e
      for e in saved.events
      if e.author == 'bookings'
      and e.content
      and (e.content.parts[-1].text or '').startswith('You land')
  )
  assert '.bookings@' in lookup.branch or lookup.branch.startswith('bookings@')
  if use_as_output:
    # Published on the outermost claimer, the workflow root, not the lane.
    assert answer.branch is None
  else:
    assert answer.branch == lookup.branch


@pytest.mark.asyncio
async def test_claimed_task_sub_branch_keeps_finish_task_paired():
  """Claiming a task sub-branch keeps finish_task call/response paired on it."""
  worker_model = testing_utils.MockModel.create([
      types.Part(
          function_call=types.FunctionCall(
              id='fc_finish_1',
              name='finish_task',
              args={'result': 'You land on Tuesday at 14:05.'},
          )
      ),
  ])
  guide_model = testing_utils.MockModel.create(['A restaurant suggestion.'])
  worker = Agent(name='worker', model=worker_model, mode='task')
  guide = Agent(name='guide', model=guide_model, mode='chat')

  @node(rerun_on_resume=True)
  async def dispatcher(ctx: Context, node_input: Any):
    if ctx.state.get('done'):
      return await ctx.run_node(guide, use_sub_branch=True, use_as_output=True)
    answer = await ctx.run_node(
        worker,
        node_input=node_input,
        use_sub_branch=True,
        use_as_output=True,
        override_isolation_scope='',
    )
    ctx.state['done'] = True
    return answer

  agent = Workflow(name='task_dispatch_wf', edges=[(START, dispatcher)])
  runner = testing_utils.InMemoryRunner(
      app=_make_app('task_dispatch_wf', agent, False)
  )

  events1 = await runner.run_async(
      testing_utils.get_user_content('When do I land?')
  )
  await runner.run_async(testing_utils.get_user_content('Dinner that evening?'))

  outputs1 = [e.output for e in events1 if e.output is not None]
  assert outputs1 == [{'result': 'You land on Tuesday at 14:05.'}]

  saved = runner.session
  finish_fc = next(e for e in saved.events if e.get_function_calls())
  finish_fr = next(e for e in saved.events if e.get_function_responses())
  assert finish_fc.branch is not None
  assert finish_fr.branch == finish_fc.branch

  assert len(guide_model.requests) == 1
  request_text = '\n'.join(
      part.text or ''
      for content in guide_model.requests[0].contents
      for part in content.parts or []
  )
  assert 'finish_task' not in request_text


@pytest.mark.parametrize('delegated_in_lane', [False, True])
@pytest.mark.asyncio
async def test_terminal_fan_out_lane_keeps_sub_branch(
    delegated_in_lane: bool,
) -> None:
  """A static terminal fan-out node stays on its sub-branch rather than root."""
  lane_a_done = asyncio.Event()

  @node(rerun_on_resume=True)
  async def lane_a(ctx: Context) -> Any:
    if delegated_in_lane:
      result = await ctx.run_node(
          FunctionNode(func=lambda: 'SECRET_LANE_A', name='helper'),
          use_sub_branch=True,
          use_as_output=True,
      )
    else:
      result = 'SECRET_LANE_A'
    lane_a_done.set()
    return result

  sibling_model = testing_utils.MockModel.create(['sibling note'])

  @node(rerun_on_resume=True)
  async def lane_b(ctx: Context) -> None:
    await lane_a_done.wait()
    await ctx.run_node(Agent(name='reader', model=sibling_model, mode='chat'))
    return None

  agent = Workflow(
      name='fan_out_wf',
      edges=[(START, lane_a), (START, lane_b)],
  )
  runner = testing_utils.InMemoryRunner(
      app=_make_app('fan_out_wf', agent, False)
  )
  events = await runner.run_async(testing_utils.get_user_content('go'))

  output_events = [e for e in events if e.output == 'SECRET_LANE_A']
  assert len(output_events) == 1
  assert output_events[0].branch == 'lane_a@1'

  sibling_parts: list[str] = []
  for content in sibling_model.requests[0].contents or []:
    for part in content.parts or []:
      sibling_parts.append(part.text or '')
  assert 'SECRET_LANE_A' not in '\n'.join(sibling_parts)


@pytest.mark.asyncio
async def test_fast_forwarded_claimed_child_keeps_sub_branch_on_resume() -> (
    None
):
  """Fast-forwarding a claimed sub-branch child restores its own sub-branch."""
  observed_branches: list[str | None] = []

  @node
  def fn_child() -> str:
    return 'from_child'

  @node(rerun_on_resume=True)
  async def claimer(ctx: Context) -> AsyncGenerator[Any, None]:
    child_ctx = await ctx._run_node_internal(
        fn_child,
        use_sub_branch=True,
        use_as_output=True,
        return_ctx=True,
    )
    observed_branches.append(child_ctx.branch)
    if not (ctx.resume_inputs and 'resume_gate' in ctx.resume_inputs):
      yield RequestInput(interrupt_id='resume_gate', message='continue?')

  agent = Workflow(
      name='resume_branch_wf',
      edges=[
          (START, claimer),
          (START, FunctionNode(func=lambda: None, name='side')),
      ],
  )
  runner = testing_utils.InMemoryRunner(
      app=_make_app('resume_branch_wf', agent, False)
  )

  events1 = await runner.run_async(testing_utils.get_user_content('start'))
  claimed_ev = next(e for e in events1 if e.output == 'from_child')
  assert claimed_ev.branch == 'claimer@1'
  assert observed_branches == ['claimer@1.fn_child@1']

  resume_payload = types.Content(
      role='user',
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  id='resume_gate',
                  name='user_input',
                  response={'text': 'ok'},
              )
          )
      ],
  )
  await runner.run_async(
      new_message=resume_payload, invocation_id=events1[0].invocation_id
  )

  assert observed_branches == ['claimer@1.fn_child@1', 'claimer@1.fn_child@1']
