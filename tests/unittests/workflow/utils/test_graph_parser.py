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

"""Tests for Graph parser utility."""

from google.adk.workflow import Edge
from google.adk.workflow import FunctionNode
from google.adk.workflow import START
from google.adk.workflow._graph import DEFAULT_ROUTE
from google.adk.workflow.utils._graph_parser import parse_edge_items
import pytest

from ..workflow_testing_utils import TestingNode


def test_parse_edge_items_with_node_reuse() -> None:
  """Tests that node reuse during parsing works and deduplicates wrapped nodes."""

  def my_node_func() -> None:
    pass

  node_b = TestingNode(name='NodeB')
  edges = parse_edge_items([
      (START, my_node_func),
      (my_node_func, node_b),
  ])

  assert len(edges) == 2
  edge1, edge2 = edges
  assert edge1.from_node == START
  assert isinstance(edge1.to_node, FunctionNode)
  assert edge1.to_node.name == 'my_node_func'

  assert isinstance(edge2.from_node, FunctionNode)
  assert edge2.from_node.name == 'my_node_func'
  assert edge2.to_node == node_b

  # Verify exact same object instance was reused
  assert edge1.to_node is edge2.from_node


def test_routing_map_basic() -> None:
  """Tests that a string-keyed routing map expands to correct edges."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  node_c = TestingNode(name='NodeC')
  edges = parse_edge_items([
      (START, node_a),
      (node_a, {'route_b': node_b, 'route_c': node_c}),
  ])

  assert len(edges) == 3  # START->A, A->B(route_b), A->C(route_c)

  routed_edges = [e for e in edges if e.route is not None]
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
  edges = parse_edge_items([
      (START, node_a),
      (node_a, {1: node_b, 2: node_c}),
  ])

  routed_edges = [e for e in edges if e.route is not None]
  assert len(routed_edges) == 2
  routes = [e.route for e in routed_edges]
  assert 1 in routes
  assert 2 in routes


def test_routing_map_bool_keys() -> None:
  """Tests that boolean route keys work in routing maps."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  node_c = TestingNode(name='NodeC')
  edges = parse_edge_items([
      (START, node_a),
      (node_a, {True: node_b, False: node_c}),
  ])

  routed_edges = [e for e in edges if e.route is not None]
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
  edges = parse_edge_items([
      (START, node_a),
      (START, node_b),
      ((node_a, node_b), {'route_x': node_c, 'route_y': node_d}),
  ])

  # 2 from START + 4 from fan-in (A->C, A->D, B->C, B->D)
  assert len(edges) == 6

  fan_in_edges = [e for e in edges if e.from_node.name in ('NodeA', 'NodeB')]
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

  edges = parse_edge_items([
      (START, node_a),
      (node_a, {'route_x': my_target_func}),
  ])

  target_edge = next(e for e in edges if e.route == 'route_x')
  assert isinstance(target_edge.to_node, FunctionNode)
  assert target_edge.to_node.name == 'my_target_func'


def test_routing_map_node_reuse() -> None:
  """Tests that the same callable used in a map and elsewhere is deduplicated."""

  def my_func() -> None:
    pass

  node_b = TestingNode(name='NodeB')
  edges = parse_edge_items([
      (START, my_func),
      (my_func, {'route_x': node_b}),
  ])

  # my_func should be wrapped once and reused.
  assert len(edges) == 2
  assert edges[0].to_node is edges[1].from_node
  assert isinstance(edges[0].to_node, FunctionNode)


def test_routing_map_empty_dict_raises() -> None:
  """Tests that an empty routing map raises ValueError."""
  node_a = TestingNode(name='NodeA')
  with pytest.raises(
      ValueError,
      match=r'Routing map must not be empty',
  ):
    parse_edge_items([
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
    parse_edge_items([
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
    parse_edge_items([
        (START, node_a),
        (node_a, {'route_x': 42}),
    ])


def test_routing_map_fan_out_target() -> None:
  """Tests that a tuple value in a routing map creates fan-out edges."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  node_c = TestingNode(name='NodeC')
  edges = parse_edge_items([
      (START, node_a),
      (node_a, {'route_x': (node_b, node_c)}),
  ])

  # START->A, A->B(route_x), A->C(route_x)
  assert len(edges) == 3

  routed_edges = [e for e in edges if e.route is not None]
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
    parse_edge_items([
        (START, node_a),
        (node_a, {'route_x': (node_b, 42)}),
    ])


# --- Routing map as chain element tests ---


def test_routing_map_chain_ending_with_dict() -> None:
  """Tests a chain ending with a routing map creates correct edges."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  node_c = TestingNode(name='NodeC')
  edges = parse_edge_items([
      (START, node_a, {'r1': node_b, 'r2': node_c}),
  ])

  # START->A (None), A->B (r1), A->C (r2)
  assert len(edges) == 3

  start_edge = next(e for e in edges if e.from_node.name == '__START__')
  assert start_edge.to_node.name == 'NodeA'
  assert start_edge.route is None

  routed_edges = [e for e in edges if e.route is not None]
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
  edges = parse_edge_items([
      (START, node_a, {'r1': node_b, 'r2': node_c}, node_d),
  ])

  # START->A (None), A->B (r1), A->C (r2), B->D (None), C->D (None)
  assert len(edges) == 5

  routed_edges = sorted(
      [e for e in edges if e.route is not None],
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
      [e for e in edges if e.to_node.name == 'NodeD'],
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
  edges = parse_edge_items([
      (START, node_a, {'r1': (node_b, node_c)}, node_d),
  ])

  # START->A (None), A->B (r1), A->C (r1), B->D (None), C->D (None)
  assert len(edges) == 5

  routed_edges = [e for e in edges if e.route is not None]
  assert len(routed_edges) == 2
  for e in routed_edges:
    assert e.from_node.name == 'NodeA'
    assert e.route == 'r1'

  fan_in_edges = [e for e in edges if e.to_node.name == 'NodeD']
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
    parse_edge_items([
        (START, node_a, {'r1': node_b, 'r2': node_c}, {'r3': node_d}),
    ])


def test_routing_map_empty_dict_in_chain_raises() -> None:
  """Tests that an empty routing map in a chain raises ValueError."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  with pytest.raises(ValueError, match=r'Routing map must not be empty'):
    parse_edge_items([
        (START, node_a, {}, node_b),
    ])


def test_routing_map_invalid_key_in_chain_raises() -> None:
  """Tests that invalid routing map keys in a chain raise ValueError."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  with pytest.raises(ValueError, match=r'Invalid routing map key'):
    parse_edge_items([
        (START, node_a, {1.5: node_b}),
    ])


def test_routing_map_2_tuple_backward_compat() -> None:
  """Ensures existing 2-tuple routing map syntax still works."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  node_c = TestingNode(name='NodeC')
  edges = parse_edge_items([
      (START, node_a),
      (node_a, {'r1': node_b, 'r2': node_c}),
  ])
  assert len(edges) == 3
