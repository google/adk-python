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

"""Tests for Graph validation utility."""

import logging

from google.adk.workflow import Edge
from google.adk.workflow import START
from google.adk.workflow._graph import DEFAULT_ROUTE
from google.adk.workflow._graph import Graph
from google.adk.workflow.utils._graph_validation import validate_graph
from pydantic import BaseModel
import pytest

from ..workflow_testing_utils import TestingNode


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
    validate_graph(graph.nodes, graph.edges)


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
    validate_graph(graph.nodes, graph.edges)


def test_disconnected_routed_subgraph_is_unreachable() -> None:
  """Tests that a disconnected subgraph with routed edges fails validation."""
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
    validate_graph(graph.nodes, graph.edges)


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
  node_b = TestingNode(name='NodeB')
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(
              from_node=node_a,
              to_node=node_b,
              route=routes[0],
          ),
          Edge(
              from_node=node_a,
              to_node=node_b,
              route=routes[1],
          ),
      ],
  )
  with pytest.raises(
      ValueError,
      match=(
          r'Graph validation failed\. Duplicate edge found: from=NodeA,'
          r' to=NodeB'
      ),
  ):
    validate_graph(graph.nodes, graph.edges)


def test_routed_start_edge_fails_validation() -> None:
  """Tests that routed edges from START node fail validation."""
  node_a = TestingNode(name='NodeA')
  graph = Graph(
      edges=[
          Edge(from_node=START, to_node=node_a, route='route1'),
      ],
  )
  with pytest.raises(
      ValueError,
      match=r'Graph validation failed\. Edges from START must not have routes',
  ):
    validate_graph(graph.nodes, graph.edges)


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
    validate_graph(graph.nodes, graph.edges)


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
    validate_graph(graph.nodes, graph.edges)


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
  validate_graph(graph.nodes, graph.edges)  # Should not raise


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
    validate_graph(graph.nodes, graph.edges)


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
    validate_graph(graph.nodes, graph.edges)


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
    validate_graph(graph.nodes, graph.edges)


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
    validate_graph(graph.nodes, graph.edges)


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
  validate_graph(
      graph.nodes, graph.edges
  )  # Should not raise — routed back-edge


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
  validate_graph(
      graph.nodes, graph.edges
  )  # Should not raise — routed self-loop


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
  validate_graph(graph.nodes, graph.edges)  # Should not raise


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
  validate_graph(graph.nodes, graph.edges)  # Should not raise


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
    validate_graph(graph.nodes, graph.edges)


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
  validate_graph(graph.nodes, graph.edges)  # Should not raise


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
  validate_graph(
      graph.nodes, graph.edges
  )  # Should not raise because node_b is a TestingNode, not LlmAgent
