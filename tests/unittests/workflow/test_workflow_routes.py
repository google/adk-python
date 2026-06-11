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

"""Testings for the Workflow routes."""

from typing import Any
from typing import Dict

from google.adk.agents.context import Context
from google.adk.apps.app import App
from google.adk.workflow import Edge
from google.adk.workflow import START
from google.adk.workflow._graph import DEFAULT_ROUTE
from google.adk.workflow._graph import Graph
from google.adk.workflow._join_node import JoinNode
from google.adk.workflow._workflow import Workflow
import pytest

from .. import testing_utils
from .workflow_testing_utils import create_parent_invocation_context
from .workflow_testing_utils import simplify_events_with_node
from .workflow_testing_utils import TestingNode


@pytest.mark.asyncio
async def test_run_async_with_edge_routes(request: pytest.FixtureRequest):
  route_holder = {'route': 'route_b'}

  def dynamic_router(ctx: Context, node_input: Any):
    return route_holder['route']

  node_a = TestingNode(name='NodeA', output='A', route=dynamic_router)
  node_b = TestingNode(name='NodeB', output='B')
  node_c = TestingNode(name='NodeC', output='C')
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(
              from_node=node_a,
              to_node=node_b,
              route='route_b',
          ),
          Edge(
              from_node=node_a,
              to_node=node_c,
              route='route_c',
          ),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent',
      graph=graph,
  )

  # Test case for route_b
  route_holder['route'] = 'route_b'
  app = App(name=request.function.__name__ + '_b', root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)
  events_b = await runner.run_async(testing_utils.get_user_content('start'))
  assert simplify_events_with_node(events_b) == [
      ('test_workflow_agent@1/NodeA@1', {'output': 'A'}),
      ('test_workflow_agent@1/NodeB@1', {'output': 'B'}),
  ]

  # Test case for route_c
  route_holder['route'] = 'route_c'
  app_c = App(name=request.function.__name__ + '_c', root_agent=agent)
  runner_c = testing_utils.InMemoryRunner(app=app_c)
  events_c = await runner_c.run_async(testing_utils.get_user_content('start'))
  assert simplify_events_with_node(events_c) == [
      ('test_workflow_agent@1/NodeA@1', {'output': 'A'}),
      ('test_workflow_agent@1/NodeC@1', {'output': 'C'}),
  ]


@pytest.mark.asyncio
async def test_output_route_int(request: pytest.FixtureRequest):
  node_a = TestingNode(name='NodeA', route=1)
  node_b = TestingNode(name='NodeB', output='B')
  node_c = TestingNode(name='NodeC', output='C')
  agent = Workflow(
      name='test_workflow_agent_route_int',
      edges=[
          (START, node_a),
          (node_a, {1: node_b, 2: node_c}),
      ],
  )
  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_route_int@1/NodeA@1',
          {'output': None},
      ),
      (
          'test_workflow_agent_route_int@1/NodeB@1',
          {'output': 'B'},
      ),
  ]


@pytest.mark.asyncio
async def test_output_route_bool(request: pytest.FixtureRequest):
  node_a = TestingNode(name='NodeA', route=True)
  node_b = TestingNode(name='NodeB', output='B')
  node_c = TestingNode(name='NodeC', output='C')
  agent = Workflow(
      name='test_workflow_agent_route_bool',
      edges=[
          (START, node_a),
          (node_a, {True: node_b, False: node_c}),
      ],
  )
  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_route_bool@1/NodeA@1',
          {'output': None},
      ),
      (
          'test_workflow_agent_route_bool@1/NodeB@1',
          {'output': 'B'},
      ),
  ]


@pytest.mark.asyncio
async def test_output_route_no_data(request: pytest.FixtureRequest):
  node_a = TestingNode(name='NodeA', route='route_b')
  node_b = TestingNode(name='NodeB', output='B')
  node_c = TestingNode(name='NodeC', output='C')
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(
              from_node=node_a,
              to_node=node_b,
              route='route_b',
          ),
          Edge(
              from_node=node_a,
              to_node=node_c,
              route='route_c',
          ),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_route_no_data',
      graph=graph,
  )
  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_route_no_data@1/NodeA@1',
          {'output': None},
      ),
      (
          'test_workflow_agent_route_no_data@1/NodeB@1',
          {'output': 'B'},
      ),
  ]


@pytest.mark.asyncio
async def test_run_async_with_list_of_routes(request: pytest.FixtureRequest):
  node_a = TestingNode(name='NodeA', output='A', route=['route_b', 'route_c'])
  node_b = TestingNode(name='NodeB', output='B')
  node_c = TestingNode(name='NodeC', output='C')
  node_d = TestingNode(name='NodeD', output='D')
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(
              from_node=node_a,
              to_node=node_b,
              route='route_b',
          ),
          Edge(
              from_node=node_a,
              to_node=node_c,
              route='route_c',
          ),
          Edge(
              from_node=node_a,
              to_node=node_d,
              route='route_d',
          ),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_list_routes',
      graph=graph,
  )

  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  simplified_events = simplify_events_with_node(events)

  assert len(simplified_events) == 3
  assert simplified_events[0] == (
      'test_workflow_agent_list_routes@1/NodeA@1',
      {'output': 'A'},
  )

  # Check that the other two events are from NodeB and NodeC, in any order.
  other_events = simplified_events[1:]
  expected_other_events = [
      (
          'test_workflow_agent_list_routes@1/NodeB@1',
          {'output': 'B'},
      ),
      (
          'test_workflow_agent_list_routes@1/NodeC@1',
          {'output': 'C'},
      ),
  ]
  assert len(other_events) == len(expected_other_events)
  assert all(item in other_events for item in expected_other_events)


@pytest.mark.asyncio
async def test_run_async_with_default_route(request: pytest.FixtureRequest):
  node_a = TestingNode(name='NodeA', output='A', route='unmatched_route')
  node_b = TestingNode(name='NodeB', output='B')
  node_c = TestingNode(name='NodeC', output='C')
  node_d = TestingNode(name='NodeD', output='D')

  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(
              from_node=node_a,
              to_node=node_b,
              route='route_b',
          ),
          # This edge has the DEFAULT_ROUTE tag.
          Edge(
              from_node=node_a,
              to_node=node_c,
              route=DEFAULT_ROUTE,
          ),
          # This edge has no route tag, so it should always be triggered.
          Edge(
              from_node=node_a,
              to_node=node_d,
          ),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_default_route',
      graph=graph,
  )

  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  simplified_events = simplify_events_with_node(events)

  assert len(simplified_events) == 3
  assert simplified_events[0] == (
      'test_workflow_agent_default_route@1/NodeA@1',
      {'output': 'A'},
  )

  # Check that NodeC (default route) and NodeD (untagged) are triggered.
  other_events = simplified_events[1:]
  expected_other_events = [
      (
          'test_workflow_agent_default_route@1/NodeC@1',
          {'output': 'C'},
      ),
      (
          'test_workflow_agent_default_route@1/NodeD@1',
          {'output': 'D'},
      ),
  ]
  assert len(other_events) == len(expected_other_events)
  assert all(item in other_events for item in expected_other_events)


@pytest.mark.asyncio
async def test_run_async_default_route_not_triggered_if_match(
    request: pytest.FixtureRequest,
):
  node_a = TestingNode(name='NodeA', output='A', route='route_b')
  node_b = TestingNode(name='NodeB', output='B')
  node_c = TestingNode(name='NodeC', output='C')

  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(
              from_node=node_a,
              to_node=node_b,
              route='route_b',
          ),
          # This edge has the DEFAULT_ROUTE tag.
          Edge(
              from_node=node_a,
              to_node=node_c,
              route=DEFAULT_ROUTE,
          ),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_default_route_not_triggered',
      graph=graph,
  )

  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  simplified_events = simplify_events_with_node(events)

  assert simplified_events == [
      (
          'test_workflow_agent_default_route_not_triggered@1/NodeA@1',
          {'output': 'A'},
      ),
      (
          'test_workflow_agent_default_route_not_triggered@1/NodeB@1',
          {'output': 'B'},
      ),
  ]


@pytest.mark.asyncio
async def test_run_async_with_untagged_edges(request: pytest.FixtureRequest):
  node_a = TestingNode(name='NodeA', output='A', route='route_b')
  node_b = TestingNode(name='NodeB', output='B')
  node_c = TestingNode(name='NodeC', output='C')
  node_d = TestingNode(name='NodeD', output='D')
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(
              from_node=node_a,
              to_node=node_b,
              route='route_b',
          ),
          Edge(
              from_node=node_a,
              to_node=node_c,
              route='route_c',
          ),
          # This edge has no route tag, so it should always be triggered.
          Edge(
              from_node=node_a,
              to_node=node_d,
          ),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_untagged_edges',
      graph=graph,
  )

  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  simplified_events = simplify_events_with_node(events)

  assert len(simplified_events) == 3
  assert simplified_events[0] == (
      'test_workflow_agent_untagged_edges@1/NodeA@1',
      {'output': 'A'},
  )

  # Check that NodeB and NodeD are triggered.
  other_events = simplified_events[1:]
  expected_other_events = [
      (
          'test_workflow_agent_untagged_edges@1/NodeB@1',
          {'output': 'B'},
      ),
      (
          'test_workflow_agent_untagged_edges@1/NodeD@1',
          {'output': 'D'},
      ),
  ]
  assert len(other_events) == len(expected_other_events)
  assert all(item in other_events for item in expected_other_events)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'emitted_route, expected_target',
    [
        ('route_a', 'Target'),
        ('route_b', 'Target'),
        ('route_c', 'Other'),
    ],
    ids=['route_a_matches_list', 'route_b_matches_list', 'route_c_single'],
)
async def test_edge_with_multiple_routes(
    request: pytest.FixtureRequest, emitted_route, expected_target
):
  """Tests that an edge with a list of routes matches any of them."""
  node_router = TestingNode(name='Router', output='R', route=emitted_route)
  node_target = TestingNode(name='Target', output='T')
  node_other = TestingNode(name='Other', output='O')

  agent = Workflow(
      name='test_multi_route',
      edges=[
          (START, node_router),
          Edge(
              from_node=node_router,
              to_node=node_target,
              route=['route_a', 'route_b'],
          ),
          (node_router, {'route_c': node_other}),
      ],
  )

  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))
  assert simplify_events_with_node(events) == [
      ('test_multi_route@1/Router@1', {'output': 'R'}),
      (
          f'test_multi_route@1/{expected_target}@1',
          {
              'output': 'T' if expected_target == 'Target' else 'O',
          },
      ),
  ]


# --- Routing map integration tests ---


@pytest.mark.asyncio
async def test_routing_map_with_default_route(
    request: pytest.FixtureRequest,
):
  """Tests that DEFAULT_ROUTE works as a fallback in routing maps."""
  node_a = TestingNode(name='NodeA', output='A', route='unmatched_route')
  node_b = TestingNode(name='NodeB', output='B')
  node_c = TestingNode(name='NodeC', output='C')

  agent = Workflow(
      name='test_routing_map_default',
      edges=[
          (START, node_a),
          (node_a, {'route_b': node_b, DEFAULT_ROUTE: node_c}),
      ],
  )

  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))
  assert simplify_events_with_node(events) == [
      (
          'test_routing_map_default@1/NodeA@1',
          {'output': 'A'},
      ),
      (
          'test_routing_map_default@1/NodeC@1',
          {'output': 'C'},
      ),
  ]


@pytest.mark.asyncio
async def test_routing_map_mixed_with_other_formats(
    request: pytest.FixtureRequest,
):
  """Tests that routing maps coexist with tuples and Edge objects."""
  node_a = TestingNode(name='NodeA', output='A', route='route_b')
  node_b = TestingNode(name='NodeB', output='B')
  node_c = TestingNode(name='NodeC', output='C')
  node_d = TestingNode(name='NodeD', output='D')

  agent = Workflow(
      name='test_routing_map_mixed',
      edges=[
          (START, node_a),
          (node_a, {'route_b': node_b, 'route_c': node_c}),
          (node_b, node_d),  # unconditional 2-tuple
          Edge(from_node=node_c, to_node=node_d),  # Edge object
      ],
  )

  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))
  assert simplify_events_with_node(events) == [
      (
          'test_routing_map_mixed@1/NodeA@1',
          {'output': 'A'},
      ),
      (
          'test_routing_map_mixed@1/NodeB@1',
          {'output': 'B'},
      ),
      (
          'test_routing_map_mixed@1/NodeD@1',
          {'output': 'D'},
      ),
  ]


@pytest.mark.asyncio
async def test_routing_map_fan_out_runs_both_targets(
    request: pytest.FixtureRequest,
):
  """Tests that fan-out in routing maps triggers both targets at runtime."""
  node_a = TestingNode(name='NodeA', output='A', route='route_x')
  node_b = TestingNode(name='NodeB', output='B')
  node_c = TestingNode(name='NodeC', output='C')
  gate = JoinNode(name='Gate')

  agent = Workflow(
      name='test_routing_map_fan_out',
      edges=[
          (START, node_a),
          (node_a, {'route_x': (node_b, node_c)}),
          (node_b, gate),
          (node_c, gate),
      ],
  )

  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))
  simplified = simplify_events_with_node(events)

  # NodeB and NodeC should both be triggered, in any order.
  # Gate node also produces output combining the two.
  outputs = [
      e
      for e in simplified
      if isinstance(e[1], dict) and e[1].get('output') is not None
  ]

  assert len(outputs) == 4
  assert outputs[0] == (
      'test_routing_map_fan_out@1/NodeA@1',
      {'output': 'A'},
  )

  # Gate should be last
  assert outputs[-1] == (
      'test_routing_map_fan_out@1/Gate@1',
      {'output': {'NodeB': 'B', 'NodeC': 'C'}},
  )

  other = outputs[1:3]
  expected = [
      (
          'test_routing_map_fan_out@1/NodeB@1',
          {'output': 'B'},
      ),
      (
          'test_routing_map_fan_out@1/NodeC@1',
          {'output': 'C'},
      ),
  ]
  assert len(other) == len(expected)
  assert all(item in other for item in expected)


@pytest.mark.asyncio
async def test_fan_in_with_route(request: pytest.FixtureRequest):
  """Fan-in with conditional routes — both route to same target."""
  a = TestingNode(name='a', output='A', route='route1')
  b = TestingNode(name='b', output='B', route='route1')
  c = TestingNode(name='c', output='C')
  wf = Workflow(
      name='wf',
      edges=[
          (START, a),
          (START, b),
          ((a, b), {'route1': c}),
      ],
  )

  app = App(name=request.function.__name__, root_agent=wf)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))
  simplified = simplify_events_with_node(events)

  # 'c' should receive from both 'a' and 'b' and run twice.
  c_events = [e for e in simplified if e[0].split('/')[-1].split('@')[0] == 'c']
  assert len(c_events) == 2


@pytest.mark.asyncio
async def test_fan_out_with_route(request: pytest.FixtureRequest):
  """Fan-out via route to multiple terminals raises ValueError."""
  router = TestingNode(name='r', output='R', route='route1')
  b = TestingNode(name='b', output='B')
  c = TestingNode(name='c', output='C')
  wf = Workflow(
      name='wf',
      edges=[
          (START, router),
          (router, {'route1': (b, c)}),
      ],
  )

  app = App(name=request.function.__name__, root_agent=wf)
  runner = testing_utils.InMemoryRunner(app=app)

  with pytest.raises(ValueError, match='multiple terminal nodes'):
    await runner.run_async(testing_utils.get_user_content('start'))


@pytest.mark.asyncio
async def test_fan_in_out_with_route(request: pytest.FixtureRequest):
  """Fan-in/out with routes to multiple terminals raises ValueError."""
  a = TestingNode(name='a', output='A', route='route1')
  b = TestingNode(name='b', output='B', route='route1')
  c = TestingNode(name='c', output='C')
  d = TestingNode(name='d', output='D')
  wf = Workflow(
      name='wf',
      edges=[
          (START, a),
          (START, b),
          ((a, b), {'route1': (c, d)}),
      ],
  )

  app = App(name=request.function.__name__, root_agent=wf)
  runner = testing_utils.InMemoryRunner(app=app)

  with pytest.raises(ValueError, match='multiple terminal nodes'):
    await runner.run_async(testing_utils.get_user_content('start'))
