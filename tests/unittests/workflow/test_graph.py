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

"""Tests for Graph validation and routing."""

import logging

from google.adk.workflow import Edge
from google.adk.workflow import START
from google.adk.workflow._graph import DEFAULT_ROUTE
from google.adk.workflow._graph import Graph

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


def test_get_next_pending_nodes_unmatched_route_warning(caplog) -> None:
  """Tests that a warning is logged when a route is unmatched and there's no DEFAULT_ROUTE."""
  node_a = TestingNode(name='NodeA')
  node_c = TestingNode(name='NodeC')

  graph = Graph(
      edges=[
          Edge(from_node=node_a, to_node=node_c, route='route1'),
      ],
  )

  with caplog.at_level(logging.WARNING):
    next_nodes = graph.get_next_pending_nodes(
        'NodeA', routes_to_match='unknown_route'
    )

  assert not next_nodes
  assert any(
      'has conditional/DEFAULT edges but none were matched' in record.message
      for record in caplog.records
  )
