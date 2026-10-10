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

"""Testings for the JoinNode."""

from google.adk import workflow
from google.adk.apps import app
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.workflow import _base_node as base_node
from google.adk.workflow import _graph as workflow_graph
from google.adk.workflow import _join_node as join_node
from google.adk.workflow import START
from google.adk.workflow._workflow import Workflow
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
from google.adk.workflow.utils._workflow_hitl_utils import get_request_input_interrupt_ids
from pydantic import BaseModel
import pytest

from . import workflow_testing_utils
from .. import testing_utils


def _build_join_node_workflow(
    request: pytest.FixtureRequest,
) -> tuple[
    workflow_testing_utils.InputCapturingNode, testing_utils.InMemoryRunner
]:
  """Builds a workflow with a JoinNode."""
  node_a = workflow_testing_utils.TestingNode(
      name='NodeA', output={'a': 1, 'b': 1}
  )
  node_b = workflow_testing_utils.TestingNode(name='NodeB', output={'b': 2})
  node_join = join_node.JoinNode(name='NodeJoin')
  node_capture = workflow_testing_utils.InputCapturingNode(name='NodeCapture')
  agent = workflow.Workflow(
      name='test_join_node',
      edges=[
          workflow_graph.Edge(from_node=base_node.START, to_node=node_a),
          workflow_graph.Edge(from_node=base_node.START, to_node=node_b),
          workflow_graph.Edge(from_node=node_a, to_node=node_join),
          workflow_graph.Edge(from_node=node_b, to_node=node_join),
          workflow_graph.Edge(from_node=node_join, to_node=node_capture),
      ],
  )
  app_instance = app.App(
      name=request.function.__name__,
      root_agent=agent,
  )
  return node_capture, testing_utils.InMemoryRunner(app=app_instance)


@pytest.mark.asyncio
async def test_join_node_waits_for_all_inputs(request: pytest.FixtureRequest):
  """Tests JoinNode with fan-in."""
  node_capture, runner = _build_join_node_workflow(request)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  assert node_capture.received_inputs == [{
      'NodeA': {'a': 1, 'b': 1},
      'NodeB': {'b': 2},
  }]


@pytest.mark.asyncio
async def test_join_node_waits_when_start_is_a_predecessor(
    request: pytest.FixtureRequest,
):
  """Tests JoinNode with a direct START edge alongside parallel branches."""
  node_a = workflow_testing_utils.TestingNode(name='NodeA', output={'a': 1})
  node_b = workflow_testing_utils.TestingNode(name='NodeB', output={'b': 2})
  node_join = join_node.JoinNode(name='NodeJoin')
  node_capture = workflow_testing_utils.InputCapturingNode(name='NodeCapture')
  agent = workflow.Workflow(
      name='test_join_node_start_predecessor',
      edges=[
          workflow_graph.Edge(from_node=base_node.START, to_node=node_a),
          workflow_graph.Edge(from_node=base_node.START, to_node=node_b),
          workflow_graph.Edge(from_node=base_node.START, to_node=node_join),
          workflow_graph.Edge(from_node=node_a, to_node=node_join),
          workflow_graph.Edge(from_node=node_b, to_node=node_join),
          workflow_graph.Edge(from_node=node_join, to_node=node_capture),
      ],
  )
  app_instance = app.App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app_instance)

  user_content = testing_utils.get_user_content('start')
  await runner.run_async(user_content)

  assert node_capture.received_inputs == [{
      base_node.START.name: user_content,
      'NodeA': {'a': 1},
      'NodeB': {'b': 2},
  }]


@pytest.mark.asyncio
async def test_join_node_start_predecessor_keeps_nested_branch(
    request: pytest.FixtureRequest,
):
  """Tests a START-fed JoinNode emits on its own workflow's branch."""
  node_a = workflow_testing_utils.TestingNode(name='NodeA', output={'a': 1})
  node_b = workflow_testing_utils.TestingNode(name='NodeB', output={'b': 2})
  node_join = join_node.JoinNode(name='NodeJoin')
  node_capture = workflow_testing_utils.InputCapturingNode(name='NodeCapture')
  inner = workflow.Workflow(
      name='Inner',
      edges=[
          workflow_graph.Edge(from_node=base_node.START, to_node=node_a),
          workflow_graph.Edge(from_node=base_node.START, to_node=node_b),
          workflow_graph.Edge(from_node=base_node.START, to_node=node_join),
          workflow_graph.Edge(from_node=node_a, to_node=node_join),
          workflow_graph.Edge(from_node=node_b, to_node=node_join),
          workflow_graph.Edge(from_node=node_join, to_node=node_capture),
      ],
  )
  # A second START edge in the outer workflow gives Inner a sub-branch of its
  # own, so the join's predecessors run under 'Inner@1.<node>@1'.
  node_sibling = workflow_testing_utils.TestingNode(name='Sibling')
  outer = workflow.Workflow(
      name='Outer',
      edges=[
          workflow_graph.Edge(from_node=base_node.START, to_node=inner),
          workflow_graph.Edge(from_node=base_node.START, to_node=node_sibling),
      ],
  )
  app_instance = app.App(name=request.function.__name__, root_agent=outer)
  runner = testing_utils.InMemoryRunner(app=app_instance)

  events = await runner.run_async(testing_utils.get_user_content('start'))

  a_events = [e for e in events if 'NodeA' in e.node_info.path]
  assert any(e.branch == 'Inner@1.NodeA@1' for e in a_events)

  # START contributes Inner's own branch, so the common prefix of
  # ['Inner@1', 'Inner@1.NodeA@1', 'Inner@1.NodeB@1'] is still 'Inner@1'.
  # Recording an empty branch for START would collapse it to the root.
  join_events = [
      e
      for e in events
      if 'NodeJoin' in e.node_info.path and e.output is not None
  ]
  assert len(join_events) == 1
  assert join_events[0].branch == 'Inner@1'


@pytest.mark.asyncio
async def test_join_node_with_none_state(request: pytest.FixtureRequest):
  """Tests JoinNode with fan-in when node state is None."""
  node_capture, runner = _build_join_node_workflow(request)
  # Run once to set state to None
  await runner.run_async(testing_utils.get_user_content('start'))
  # Run again to trigger join_node with state=None
  await runner.run_async(testing_utils.get_user_content('start'))

  assert node_capture.received_inputs == [
      {'NodeA': {'a': 1, 'b': 1}, 'NodeB': {'b': 2}},
      {'NodeA': {'a': 1, 'b': 1}, 'NodeB': {'b': 2}},
  ]


@pytest.mark.asyncio
async def test_join_node_with_none_inputs(request: pytest.FixtureRequest):
  """Tests JoinNode with fan-in when incoming edges have None output."""
  node_a = workflow_testing_utils.TestingNode(
      name='NodeA', output=None, route='NodeJoin'
  )
  node_b = workflow_testing_utils.TestingNode(
      name='NodeB', output=None, route='NodeJoin'
  )
  node_join = join_node.JoinNode(name='NodeJoin')
  node_capture = workflow_testing_utils.InputCapturingNode(name='NodeCapture')
  agent = workflow.Workflow(
      name='test_join_node_none_inputs',
      edges=[
          workflow_graph.Edge(from_node=base_node.START, to_node=node_a),
          workflow_graph.Edge(from_node=base_node.START, to_node=node_b),
          workflow_graph.Edge(from_node=node_a, to_node=node_join),
          workflow_graph.Edge(from_node=node_b, to_node=node_join),
          workflow_graph.Edge(from_node=node_join, to_node=node_capture),
      ],
  )
  app_instance = app.App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app_instance)

  await runner.run_async(testing_utils.get_user_content('start'))

  assert node_capture.received_inputs == [
      {'NodeA': None, 'NodeB': None},
  ]


# ── JoinNode input_schema ──────────────────────────────────────
# input_schema on JoinNode validates each trigger input individually
# (each predecessor's output), not the joined dict.


class _TriggerInput(BaseModel):
  key: str
  value: int


@pytest.mark.asyncio
async def test_join_node_input_schema_validates_per_trigger(
    request: pytest.FixtureRequest,
):
  """JoinNode input_schema validates each trigger input individually."""

  def node_a() -> dict:
    return {'key': 'a', 'value': 1}

  def node_b() -> dict:
    return {'key': 'b', 'value': 2}

  join = join_node.JoinNode(name='join', input_schema=_TriggerInput)
  capture = workflow_testing_utils.InputCapturingNode(name='capture')

  agent = Workflow(
      name='wf',
      edges=[
          (START, node_a),
          (START, node_b),
          (node_a, join),
          (node_b, join),
          (join, capture),
      ],
  )
  app_instance = app.App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app_instance)
  await runner.run_async(testing_utils.get_user_content('start'))

  assert capture.received_inputs == [{
      'node_a': {'key': 'a', 'value': 1},
      'node_b': {'key': 'b', 'value': 2},
  }]


@pytest.mark.asyncio
async def test_join_node_input_schema_rejects_invalid_trigger(
    request: pytest.FixtureRequest,
):
  """JoinNode input_schema rejects invalid trigger input early."""

  def node_a() -> dict:
    return {'key': 'a', 'value': 1}

  def node_b() -> dict:
    return {'wrong': 'shape'}  # missing required fields

  join = join_node.JoinNode(name='join', input_schema=_TriggerInput)
  capture = workflow_testing_utils.InputCapturingNode(name='capture')

  agent = Workflow(
      name='wf',
      edges=[
          (START, node_a),
          (START, node_b),
          (node_a, join),
          (node_b, join),
          (join, capture),
      ],
  )
  app_instance = app.App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app_instance)
  with pytest.raises(Exception):
    await runner.run_async(testing_utils.get_user_content('start'))


@pytest.mark.asyncio
async def test_join_node_input_schema_none_trigger_passes(
    request: pytest.FixtureRequest,
):
  """JoinNode input_schema skips validation for None trigger input."""
  # Given
  node_a_fn = workflow_testing_utils.TestingNode(
      name='NodeA', output=None, route='join'
  )
  node_b_fn = workflow_testing_utils.TestingNode(
      name='NodeB', output={'key': 'b', 'value': 2}
  )
  join = join_node.JoinNode(name='join', input_schema=_TriggerInput)
  capture = workflow_testing_utils.InputCapturingNode(name='capture')

  agent = Workflow(
      name='wf',
      edges=[
          (START, node_a_fn),
          (START, node_b_fn),
          (node_a_fn, join),
          (node_b_fn, join),
          (join, capture),
      ],
  )
  app_instance = app.App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app_instance)

  # When
  await runner.run_async(testing_utils.get_user_content('start'))

  # Then
  assert capture.received_inputs == [{
      'NodeA': None,
      'NodeB': {'key': 'b', 'value': 2},
  }]


@pytest.mark.asyncio
async def test_join_node_computes_common_branch_prefix(
    request: pytest.FixtureRequest,
):
  """Tests JoinNode computes common branch prefix for final output."""
  node_capture, runner = _build_join_node_workflow(request)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  # Find the final output event from JoinNode
  join_events = [
      e
      for e in events
      if 'NodeJoin' in e.node_info.path and e.output is not None
  ]
  assert len(join_events) == 1
  join_event = join_events[0]

  # NodeA and NodeB run in parallel, so they should have branches like 'NodeA@1' and 'NodeB@1'.
  a_events = [e for e in events if 'NodeA' in e.node_info.path]
  b_events = [e for e in events if 'NodeB' in e.node_info.path]

  assert any('NodeA@' in e.branch for e in a_events if e.branch)
  assert any('NodeB@' in e.branch for e in b_events if e.branch)

  # The common prefix of 'NodeA@1' and 'NodeB@1' is empty string.
  # So JoinNode should set branch to empty string (which is converted to None).
  assert join_event.branch is None

  # The node after JoinNode (NodeCapture) should also have branch=None
  capture_events = [e for e in events if 'NodeCapture' in e.node_info.path]
  assert len(capture_events) > 0
  for e in capture_events:
    assert e.branch is None


@pytest.mark.asyncio
async def test_join_node_in_loop_waits_for_each_iteration(
    request: pytest.FixtureRequest,
):
  """Tests JoinNode in a loop waits for every predecessor's new run.

  NodeB sits one hop further from START than NodeA, so on the second
  iteration NodeA completes first. The join must not fire with NodeB's
  output left over from the first iteration.
  """
  iteration = 0
  join_inputs = []

  def node_a(node_input):
    nonlocal iteration
    iteration += 1
    return f'A{iteration}'

  def node_x(node_input):
    return 'x'

  def node_b(node_input):
    return f'B{iteration}'

  def node_check(node_input):
    join_inputs.append(node_input)
    if len(join_inputs) < 2:
      yield Event(route='again')
    else:
      yield Event(route='done', output='finished')

  def node_end(node_input):
    return node_input

  a = workflow.node(node_a, name='NodeA')
  x = workflow.node(node_x, name='NodeX')
  b = workflow.node(node_b, name='NodeB')
  check = workflow.node(node_check, name='NodeCheck')
  end = workflow.node(node_end, name='NodeEnd')
  join = join_node.JoinNode(name='NodeJoin')
  agent = workflow.Workflow(
      name='test_join_node_loop',
      edges=[
          workflow_graph.Edge(from_node=START, to_node=a),
          workflow_graph.Edge(from_node=START, to_node=x),
          workflow_graph.Edge(from_node=x, to_node=b),
          workflow_graph.Edge(from_node=a, to_node=join),
          workflow_graph.Edge(from_node=b, to_node=join),
          workflow_graph.Edge(from_node=join, to_node=check),
          workflow_graph.Edge(from_node=check, to_node=a, route='again'),
          workflow_graph.Edge(from_node=check, to_node=x, route='again'),
          workflow_graph.Edge(from_node=check, to_node=end, route='done'),
      ],
  )
  runner = testing_utils.InMemoryRunner(
      app=app.App(name=request.function.__name__, root_agent=agent)
  )

  await runner.run_async(testing_utils.get_user_content('start'))

  assert join_inputs == [
      {'NodeA': 'A1', 'NodeB': 'B1'},
      {'NodeA': 'A2', 'NodeB': 'B2'},
  ]


@pytest.mark.asyncio
@pytest.mark.parametrize('is_resumable', [False, True])
async def test_join_node_in_loop_waits_for_predecessor_paused_for_input(
    request: pytest.FixtureRequest, is_resumable: bool
):
  """Tests JoinNode in a loop waits for a predecessor that asks the user.

  On the second iteration NodeB pauses for input. The join must not fire
  with NodeB's first-iteration output while it waits, and must fire once
  with both second-iteration outputs after the resume.
  """
  a_runs = 0
  b_runs = 0
  join_inputs = []

  def node_a(node_input):
    nonlocal a_runs
    a_runs += 1
    return f'A{a_runs}'

  def node_x(node_input):
    return 'x'

  def node_b(ctx, node_input):
    nonlocal b_runs
    if b_runs == 1 and 'ask' not in ctx.resume_inputs:
      return RequestInput(interrupt_id='ask', message='Continue?')
    b_runs += 1
    return f'B{b_runs}'

  def node_check(node_input):
    join_inputs.append(node_input)
    if len(join_inputs) < 2:
      yield Event(route='again')
    else:
      yield Event(route='done', output='finished')

  def node_end(node_input):
    return node_input

  a = workflow.node(node_a, name='NodeA')
  x = workflow.node(node_x, name='NodeX')
  b = workflow.node(node_b, name='NodeB', rerun_on_resume=True)
  check = workflow.node(node_check, name='NodeCheck')
  end = workflow.node(node_end, name='NodeEnd')
  join = join_node.JoinNode(name='NodeJoin')
  agent = workflow.Workflow(
      name='test_join_node_loop_hitl',
      edges=[
          workflow_graph.Edge(from_node=START, to_node=a),
          workflow_graph.Edge(from_node=START, to_node=x),
          workflow_graph.Edge(from_node=x, to_node=b),
          workflow_graph.Edge(from_node=a, to_node=join),
          workflow_graph.Edge(from_node=b, to_node=join),
          workflow_graph.Edge(from_node=join, to_node=check),
          workflow_graph.Edge(from_node=check, to_node=a, route='again'),
          workflow_graph.Edge(from_node=check, to_node=x, route='again'),
          workflow_graph.Edge(from_node=check, to_node=end, route='done'),
      ],
  )
  runner = testing_utils.InMemoryRunner(
      app=app.App(
          name=request.function.__name__,
          root_agent=agent,
          resumability_config=ResumabilityConfig(is_resumable=is_resumable),
      )
  )

  events = await runner.run_async(testing_utils.get_user_content('start'))

  assert join_inputs == [{'NodeA': 'A1', 'NodeB': 'B1'}]

  request_events = workflow_testing_utils.get_request_input_events(events)
  interrupt_id = get_request_input_interrupt_ids(request_events[0])[0]
  await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(interrupt_id, {'ok': True})
      ),
      invocation_id=events[0].invocation_id,
  )

  assert join_inputs == [
      {'NodeA': 'A1', 'NodeB': 'B1'},
      {'NodeA': 'A2', 'NodeB': 'B2'},
  ]
