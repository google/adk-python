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

"""Tests for Graph validation."""

from google.adk.workflow import Edge
from google.adk.workflow import FunctionNode
from google.adk.workflow import START
from google.adk.workflow._graph import DEFAULT_ROUTE
from google.adk.workflow._graph import Graph
from pydantic import BaseModel
import pytest

from .workflow_testing_utils import TestingNode


def test_valid_graph() -> None:
  """Tests that a valid graph passes validation."""
  node_a = TestingNode(name='NodeA')
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
      ],
  )
  graph.validate_graph()  # Should not raise


def test_missing_start_node() -> None:
  """Tests that a graph missing the START node fails validation."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  graph = Graph(
      edges=[Edge(from_node=node_a, to_node=node_b)],
  )
  with pytest.raises(
      ValueError,
      match=(
          r"Graph validation failed\. START node \(name: '__START__'\) not"
          r' found in graph nodes\.'
      ),
  ):
    graph.validate_graph()


def test_unreachable_node() -> None:
  """Tests that a graph with an unreachable node fails validation."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')  # Unreachable
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_b, to_node=node_a),
      ],
  )
  with pytest.raises(
      ValueError,
      match=(
          r'Graph validation failed\. The following nodes are unreachable'
          r" from START: \['NodeB'\]"
      ),
  ):
    graph.validate_graph()


def test_disconnected_routed_subgraph_is_unreachable() -> None:
  """Tests that a disconnected subgraph with routed edges fails validation.

  Even though B and C each appear as a to_node in some edge, neither is
  reachable from START.  The old "has incoming edge" heuristic would let
  this pass; true reachability from START catches it.
  """
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  node_c = TestingNode(name='NodeC')
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_b, to_node=node_c, route='x'),
          Edge(from_node=node_c, to_node=node_b, route='y'),
      ],
  )
  with pytest.raises(
      ValueError,
      match=(
          r'Graph validation failed\. The following nodes are unreachable'
          r" from START: \['NodeB', 'NodeC'\]"
      ),
  ):
    graph.validate_graph()


@pytest.mark.parametrize(
    'routes',
    [
        (None, None),
        ('route1', 'route1'),
        ('route1', 'route2'),
        ('route1', None),
    ],
)
def test_duplicate_edges_fail_validation(
    routes: tuple[str | None, str | None],
) -> None:
  """Tests that duplicate edges fail validation, regardless of routes."""
  node_a = TestingNode(name='NodeA')
  graph = Graph(
      edges=[
          Edge(
              from_node=START,
              to_node=node_a,
              route=routes[0],
          ),
          Edge(
              from_node=START,
              to_node=node_a,
              route=routes[1],
          ),
      ],
  )
  with pytest.raises(
      ValueError,
      match=(
          r'Graph validation failed\. Duplicate edge found: from=__START__,'
          r' to=NodeA'
      ),
  ):
    graph.validate_graph()


def test_start_node_with_incoming_edge() -> None:
  """Tests graph with incoming edge to START node fails validation."""
  node_a = TestingNode(name='NodeA')
  graph = Graph(
      edges=[
          Edge(from_node=node_a, to_node=START),
          Edge(from_node=START, to_node=node_a),
      ],
  )
  with pytest.raises(
      ValueError,
      match=(
          r'Graph validation failed\. START node must not have incoming edges\.'
      ),
  ):
    graph.validate_graph()


def test_multiple_default_routes_fail_validation() -> None:
  """Tests that multiple DEFAULT_ROUTE edges from a node fail validation."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  node_c = TestingNode(name='NodeC')
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=node_b, route=DEFAULT_ROUTE),
          Edge(from_node=node_a, to_node=node_c, route=DEFAULT_ROUTE),
      ],
  )
  with pytest.raises(
      ValueError,
      match=(
          r'Graph validation failed\. Multiple DEFAULT_ROUTE edges found from'
          r' node NodeA to NodeB and NodeC'
      ),
  ):
    graph.validate_graph()


def test_single_default_route_passes_validation() -> None:
  """Tests that a single DEFAULT_ROUTE edge from a node passes validation."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  node_c = TestingNode(name='NodeC')
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=node_b, route=DEFAULT_ROUTE),
          Edge(from_node=node_a, to_node=node_c, route='another_route'),
      ],
  )
  graph.validate_graph()  # Should not raise


def test_duplicate_node_names_fail_validation() -> None:
  """Tests that duplicate nodes raise error."""

  node_a1 = TestingNode(name='NodeA')
  node_a2 = TestingNode(name='NodeA')
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a1),
          Edge(from_node=node_a1, to_node=node_a2),
      ],
  )
  with pytest.raises(
      ValueError,
      match=(
          r"Graph validation failed\. Duplicate node names found: \['NodeA'\]\."
          r' This means multiple distinct node objects have the same name\. If'
          r' you intended to reuse the same node, ensure you pass the exact'
          r' same object instance\. If you intended to have distinct nodes,'
          r' ensure they have unique names\.'
      ),
  ):
    graph.validate_graph()


def test_from_edge_items_with_node_reuse_passes_validation() -> None:
  """Tests that node reuse with from_edge_items passes validation.

  The same my_node_func instance is used in the graph multiple times, and
  the workflow graph should recognize it as the same instance and not throw
  an error during validation.
  """

  def my_node_func() -> None:
    pass

  node_b = TestingNode(name='NodeB')
  graph = Graph.from_edge_items([
      (START, my_node_func),
      (my_node_func, node_b),
  ])
  graph.validate_graph()  # Should not raise duplicate name error

  node_names = {n.name for n in graph.nodes}
  assert node_names == {'__START__', 'my_node_func', 'NodeB'}
  assert len(graph.nodes) == 3
  # Check that my_node_func was wrapped and deduplicated.
  func_node = next(n for n in graph.nodes if n.name == 'my_node_func')
  assert isinstance(func_node, FunctionNode)


def test_unconditional_cycle_fails_validation() -> None:
  """Tests that a cycle of unconditional edges (route=None) fails."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=node_b),
          Edge(from_node=node_b, to_node=node_a),
      ],
  )
  with pytest.raises(
      ValueError,
      match=r'Graph validation failed\. Unconditional cycle detected:',
  ):
    graph.validate_graph()


def test_unconditional_self_loop_fails_validation() -> None:
  """Tests that an unconditional self-loop (A -> A) fails."""
  node_a = TestingNode(name='NodeA')
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=node_a),
      ],
  )
  with pytest.raises(
      ValueError,
      match=r'Graph validation failed\. Unconditional cycle detected:',
  ):
    graph.validate_graph()


def test_longer_unconditional_cycle_fails_validation() -> None:
  """Tests that a longer unconditional cycle (A -> B -> C -> A) fails."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  node_c = TestingNode(name='NodeC')
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=node_b),
          Edge(from_node=node_b, to_node=node_c),
          Edge(from_node=node_c, to_node=node_a),
      ],
  )
  with pytest.raises(
      ValueError,
      match=r'Graph validation failed\. Unconditional cycle detected:',
  ):
    graph.validate_graph()


def test_conditional_cycle_passes_validation() -> None:
  """Tests that a cycle with a routed edge (loop pattern) passes."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=node_b),
          Edge(from_node=node_b, to_node=node_a, route='retry'),
      ],
  )
  graph.validate_graph()  # Should not raise — routed back-edge


def test_conditional_self_loop_passes_validation() -> None:
  """Tests that a self-loop with a route passes validation."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=node_a, route='continue'),
          Edge(from_node=node_a, to_node=node_b, route='done'),
      ],
  )
  graph.validate_graph()  # Should not raise — routed self-loop


def test_dag_with_diamond_passes_validation() -> None:
  """Tests that a DAG with a diamond shape passes validation."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  node_c = TestingNode(name='NodeC')
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=START, to_node=node_b),
          Edge(from_node=node_a, to_node=node_c),
          Edge(from_node=node_b, to_node=node_c),
      ],
  )
  graph.validate_graph()  # Should not raise


# --- Routing map tests ---


def test_routing_map_basic() -> None:
  """Tests that a string-keyed routing map expands to correct edges."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  node_c = TestingNode(name='NodeC')
  graph = Graph.from_edge_items([
      (START, node_a),
      (node_a, {'route_b': node_b, 'route_c': node_c}),
  ])
  graph.validate_graph()

  assert len(graph.edges) == 3  # START->A, A->B(route_b), A->C(route_c)

  routed_edges = [e for e in graph.edges if e.route is not None]
  assert len(routed_edges) == 2

  routes_and_targets = {(e.route, e.to_node.name) for e in routed_edges}
  assert routes_and_targets == {('route_b', 'NodeB'), ('route_c', 'NodeC')}

  for e in routed_edges:
    assert e.from_node.name == 'NodeA'


def test_routing_map_int_keys() -> None:
  """Tests that integer route keys work in routing maps."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  node_c = TestingNode(name='NodeC')
  graph = Graph.from_edge_items([
      (START, node_a),
      (node_a, {1: node_b, 2: node_c}),
  ])
  graph.validate_graph()

  routed_edges = [e for e in graph.edges if e.route is not None]
  assert len(routed_edges) == 2
  routes = [e.route for e in routed_edges]
  assert 1 in routes
  assert 2 in routes


def test_routing_map_bool_keys() -> None:
  """Tests that boolean route keys work in routing maps."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  node_c = TestingNode(name='NodeC')
  graph = Graph.from_edge_items([
      (START, node_a),
      (node_a, {True: node_b, False: node_c}),
  ])
  graph.validate_graph()

  routed_edges = [e for e in graph.edges if e.route is not None]
  assert len(routed_edges) == 2
  routes = [e.route for e in routed_edges]
  assert True in routes
  assert False in routes


def test_routing_map_with_fan_in_source() -> None:
  """Tests that fan-in on the source side works with routing maps."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  node_c = TestingNode(name='NodeC')
  node_d = TestingNode(name='NodeD')
  graph = Graph.from_edge_items([
      (START, node_a),
      (START, node_b),
      ((node_a, node_b), {'route_x': node_c, 'route_y': node_d}),
  ])
  graph.validate_graph()

  # 2 from START + 4 from fan-in (A->C, A->D, B->C, B->D)
  assert len(graph.edges) == 6

  fan_in_edges = [
      e for e in graph.edges if e.from_node.name in ('NodeA', 'NodeB')
  ]
  assert len(fan_in_edges) == 4

  combos = {(e.from_node.name, e.to_node.name, e.route) for e in fan_in_edges}
  assert combos == {
      ('NodeA', 'NodeC', 'route_x'),
      ('NodeA', 'NodeD', 'route_y'),
      ('NodeB', 'NodeC', 'route_x'),
      ('NodeB', 'NodeD', 'route_y'),
  }


def test_routing_map_with_callable_target() -> None:
  """Tests that callable values in routing maps get wrapped via build_node."""
  node_a = TestingNode(name='NodeA')

  def my_target_func() -> None:
    pass

  graph = Graph.from_edge_items([
      (START, node_a),
      (node_a, {'route_x': my_target_func}),
  ])
  graph.validate_graph()

  target_edge = next(e for e in graph.edges if e.route == 'route_x')
  assert isinstance(target_edge.to_node, FunctionNode)
  assert target_edge.to_node.name == 'my_target_func'


def test_routing_map_node_reuse() -> None:
  """Tests that the same callable used in a map and elsewhere is deduplicated."""

  def my_func() -> None:
    pass

  node_b = TestingNode(name='NodeB')
  graph = Graph.from_edge_items([
      (START, my_func),
      (my_func, {'route_x': node_b}),
  ])
  graph.validate_graph()

  # my_func should be wrapped once and reused.
  func_nodes = [n for n in graph.nodes if n.name == 'my_func']
  assert len(func_nodes) == 1
  assert isinstance(func_nodes[0], FunctionNode)


def test_routing_map_empty_dict_raises() -> None:
  """Tests that an empty routing map raises ValueError."""
  node_a = TestingNode(name='NodeA')
  with pytest.raises(
      ValueError,
      match=r'Routing map must not be empty',
  ):
    Graph.from_edge_items([
        (START, node_a),
        (node_a, {}),
    ])


def test_routing_map_invalid_key_raises() -> None:
  """Tests that a non-RouteValue key raises ValueError."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  with pytest.raises(
      ValueError,
      match=r'Invalid routing map key',
  ):
    Graph.from_edge_items([
        (START, node_a),
        (node_a, {1.5: node_b}),
    ])


def test_routing_map_invalid_value_raises() -> None:
  """Tests that a non-NodeLike value raises ValueError."""
  node_a = TestingNode(name='NodeA')
  with pytest.raises(
      ValueError,
      match=r'Invalid routing map value',
  ):
    Graph.from_edge_items([
        (START, node_a),
        (node_a, {'route_x': 42}),
    ])


def test_routing_map_fan_out_target() -> None:
  """Tests that a tuple value in a routing map creates fan-out edges."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  node_c = TestingNode(name='NodeC')
  graph = Graph.from_edge_items([
      (START, node_a),
      (node_a, {'route_x': (node_b, node_c)}),
  ])
  graph.validate_graph()

  # START->A, A->B(route_x), A->C(route_x)
  assert len(graph.edges) == 3

  routed_edges = [e for e in graph.edges if e.route is not None]
  assert len(routed_edges) == 2

  # Both fan-out edges share the same route and source.
  for e in routed_edges:
    assert e.from_node.name == 'NodeA'
    assert e.route == 'route_x'

  targets = {e.to_node.name for e in routed_edges}
  assert targets == {'NodeB', 'NodeC'}


def test_routing_map_fan_out_invalid_element_raises() -> None:
  """Tests that a non-NodeLike element inside a fan-out tuple raises."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  with pytest.raises(
      ValueError,
      match=r'Invalid node in fan-out tuple',
  ):
    Graph.from_edge_items([
        (START, node_a),
        (node_a, {'route_x': (node_b, 42)}),
    ])


# --- Routing map as chain element tests ---


def test_routing_map_chain_ending_with_dict() -> None:
  """Tests a chain ending with a routing map creates correct edges."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  node_c = TestingNode(name='NodeC')
  graph = Graph.from_edge_items([
      (START, node_a, {'r1': node_b, 'r2': node_c}),
  ])
  graph.validate_graph()

  # START->A (None), A->B (r1), A->C (r2)
  assert len(graph.edges) == 3

  start_edge = next(e for e in graph.edges if e.from_node.name == '__START__')
  assert start_edge.to_node.name == 'NodeA'
  assert start_edge.route is None

  routed_edges = [e for e in graph.edges if e.route is not None]
  assert len(routed_edges) == 2
  routes_and_targets = {(e.route, e.to_node.name) for e in routed_edges}
  assert routes_and_targets == {('r1', 'NodeB'), ('r2', 'NodeC')}
  for e in routed_edges:
    assert e.from_node.name == 'NodeA'


def test_routing_map_mid_chain_with_fan_in() -> None:
  """Tests routing map mid-chain with fan-in to the next element."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  node_c = TestingNode(name='NodeC')
  node_d = TestingNode(name='NodeD')
  graph = Graph.from_edge_items([
      (START, node_a, {'r1': node_b, 'r2': node_c}, node_d),
  ])
  graph.validate_graph()

  # START->A (None), A->B (r1), A->C (r2), B->D (None), C->D (None)
  assert len(graph.edges) == 5

  routed_edges = sorted(
      [e for e in graph.edges if e.route is not None],
      key=lambda e: e.to_node.name,
  )
  assert len(routed_edges) == 2
  assert routed_edges[0].from_node.name == 'NodeA'
  assert routed_edges[0].to_node.name == 'NodeB'
  assert routed_edges[0].route == 'r1'
  assert routed_edges[1].from_node.name == 'NodeA'
  assert routed_edges[1].to_node.name == 'NodeC'
  assert routed_edges[1].route == 'r2'

  fan_in_edges = sorted(
      [e for e in graph.edges if e.to_node.name == 'NodeD'],
      key=lambda e: e.from_node.name,
  )
  assert len(fan_in_edges) == 2
  assert fan_in_edges[0].from_node.name == 'NodeB'
  assert fan_in_edges[0].route is None
  assert fan_in_edges[1].from_node.name == 'NodeC'
  assert fan_in_edges[1].route is None


def test_routing_map_mid_chain_fan_out_values() -> None:
  """Tests routing map with fan-out tuple values, followed by fan-in."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  node_c = TestingNode(name='NodeC')
  node_d = TestingNode(name='NodeD')
  graph = Graph.from_edge_items([
      (START, node_a, {'r1': (node_b, node_c)}, node_d),
  ])
  graph.validate_graph()

  # START->A (None), A->B (r1), A->C (r1), B->D (None), C->D (None)
  assert len(graph.edges) == 5

  routed_edges = [e for e in graph.edges if e.route is not None]
  assert len(routed_edges) == 2
  for e in routed_edges:
    assert e.from_node.name == 'NodeA'
    assert e.route == 'r1'

  fan_in_edges = [e for e in graph.edges if e.to_node.name == 'NodeD']
  assert len(fan_in_edges) == 2
  fan_in_sources = {e.from_node.name for e in fan_in_edges}
  assert fan_in_sources == {'NodeB', 'NodeC'}


def test_routing_map_consecutive_dicts_raises() -> None:
  """Tests that consecutive routing maps in a chain are rejected."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  node_c = TestingNode(name='NodeC')
  node_d = TestingNode(name='NodeD')
  with pytest.raises(
      ValueError, match=r'Consecutive routing maps are not allowed'
  ):
    Graph.from_edge_items([
        (START, node_a, {'r1': node_b, 'r2': node_c}, {'r3': node_d}),
    ])


def test_routing_map_empty_dict_in_chain_raises() -> None:
  """Tests that an empty routing map in a chain raises ValueError."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  with pytest.raises(ValueError, match=r'Routing map must not be empty'):
    Graph.from_edge_items([
        (START, node_a, {}, node_b),
    ])


def test_routing_map_invalid_key_in_chain_raises() -> None:
  """Tests that invalid routing map keys in a chain raise ValueError."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  with pytest.raises(ValueError, match=r'Invalid routing map key'):
    Graph.from_edge_items([
        (START, node_a, {1.5: node_b}),
    ])


def test_routing_map_2_tuple_backward_compat() -> None:
  """Ensures existing 2-tuple routing map syntax still works."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  node_c = TestingNode(name='NodeC')
  graph = Graph.from_edge_items([
      (START, node_a),
      (node_a, {'r1': node_b, 'r2': node_c}),
  ])
  graph.validate_graph()
  assert len(graph.edges) == 3


class ModelA(BaseModel):
  x: int


class ModelB(BaseModel):
  x: int


def test_schema_match_passes() -> None:
  """Tests that edges with matching schemas pass validation."""
  node_a = TestingNode(name='NodeA', output_schema=ModelA)
  node_b = TestingNode(name='NodeB', input_schema=ModelA)
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=node_b),
      ],
  )
  graph.validate_graph()  # Should not raise


def test_schema_mismatch_raises() -> None:
  """Tests that edges with mismatching schemas fail validation."""
  node_a = TestingNode(name='NodeA', output_schema=ModelA)
  node_b = TestingNode(name='NodeB', input_schema=ModelB)
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=node_b),
      ],
  )
  with pytest.raises(
      ValueError,
      match=r'Graph validation failed\. Schema mismatch on edge',
  ):
    graph.validate_graph()


def test_schema_missing_passes() -> None:
  """Tests that edges with missing schemas pass validation."""
  node_a = TestingNode(name='NodeA', output_schema=ModelA)
  node_b = TestingNode(name='NodeB')  # No input schema
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=node_b),
      ],
  )
  graph.validate_graph()  # Should not raise


def test_get_next_pending_nodes() -> None:
  """Tests that get_next_pending_nodes returns correct nodes based on routes."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  node_c = TestingNode(name='NodeC')
  node_d = TestingNode(name='NodeD')

  graph = Graph(
      edges=[
          Edge(from_node=node_a, to_node=node_b),  # Unconditional
          Edge(from_node=node_a, to_node=node_c, route='route1'),  # Conditional
          Edge(
              from_node=node_a, to_node=node_d, route=DEFAULT_ROUTE
          ),  # Default
      ],
  )

  # Test unconditional edge triggered
  next_nodes = graph.get_next_pending_nodes('NodeA', routes_to_match=None)
  assert set(next_nodes) == {'NodeB', 'NodeD'}

  # Test specific route matched
  next_nodes = graph.get_next_pending_nodes('NodeA', routes_to_match='route1')
  assert set(next_nodes) == {'NodeB', 'NodeC'}

  # Test unmatched route falls back to default
  next_nodes = graph.get_next_pending_nodes(
      'NodeA', routes_to_match='unknown_route'
  )
  assert set(next_nodes) == {'NodeB', 'NodeD'}

  # Test list of routes to match
  next_nodes = graph.get_next_pending_nodes(
      'NodeA', routes_to_match=['route1', 'unknown_route']
  )
  assert set(next_nodes) == {'NodeB', 'NodeC'}


def test_chat_agent_wiring_validation_only_runs_on_llm_agent() -> None:
  """Tests that _validate_chat_agent_wiring checks non-LlmAgent nodes safely."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  # Set mode='chat' on a non-LlmAgent node
  object.__setattr__(node_b, 'mode', 'chat')

  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=node_b),
      ],
  )
  graph.validate_graph()  # Should not raise because node_b is a TestingNode, not LlmAgent
