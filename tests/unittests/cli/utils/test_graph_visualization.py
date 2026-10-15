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

"""Tests for plot_workflow_graph DOT output."""

import re
from typing import Any

from google.adk.cli.utils.graph_visualization import _Digraph
from google.adk.cli.utils.graph_visualization import plot_workflow_graph
from google.adk.workflow._node_status import NodeStatus
import graphviz

_TOKEN = r'"(?:[^"\\]|\\.)*"|[^\s"\[]+'
_NODE_LINE = re.compile(rf'^\t({_TOKEN}) \[', re.MULTILINE)
_EDGE_LINE = re.compile(rf'^\t({_TOKEN}) -> ({_TOKEN})', re.MULTILINE)


def _workflow_info(
    nodes: list[dict[str, Any]], edges: list[tuple[str, str]]
) -> dict[str, Any]:
  return {
      'root_agent': {
          'name': 'wf',
          'graph': {
              'nodes': nodes,
              'edges': [
                  {'from_node': {'name': a}, 'to_node': {'name': b}}
                  for a, b in edges
              ],
          },
      }
  }


def _plot(info: dict[str, Any], state: dict[str, Any] | None = None) -> str:
  dot = plot_workflow_graph(info, state, format='dot')
  assert isinstance(dot, str)
  return dot


def _node_line(dot: str, name: str) -> str:
  return next(
      line for line in dot.splitlines() if f' {name}>' in line and '[' in line
  )


def test_edges_attach_to_nodes_whose_names_contain_colons():
  """Edges to colon-named nodes hit the real nodes, not phantom ports.

  The real name must stay the DOT id because adk-web reads the SVG title as
  the node name for click handling.
  """
  info = _workflow_info(
      [
          {'name': '__START__'},
          {
              'name': 'a',
              'type': 'agent',
              'tools': [{'name': 'srv:tool', 'type': 'tool'}],
          },
          {'name': 'step:two', 'type': 'function'},
      ],
      [('__START__', 'a'), ('a', 'step:two')],
  )

  dot = _plot(info)

  declared = set(_NODE_LINE.findall(dot))
  endpoints = {end for edge in _EDGE_LINE.findall(dot) for end in edge}
  assert {'"srv:tool"', '"step:two"'} <= declared
  assert endpoints <= declared
  assert not {'srv', 'step'} & endpoints


def _assert_colon_node_is_wired(dot: str, edge: str) -> None:
  """Checks `edge` is present and the colon node "b:x" has no phantom port."""
  declared = set(_NODE_LINE.findall(dot))
  endpoints = {end for pair in _EDGE_LINE.findall(dot) for end in pair}
  assert '"b:x"' in declared
  assert endpoints <= declared
  assert 'b' not in endpoints
  assert f'\t{edge}' in dot


def test_workflow_node_with_colon_can_be_an_edge_target():
  """An edge into a node named "b:x" lands on the declared node."""
  info = _workflow_info(
      [
          {'name': 'a', 'type': 'agent'},
          {'name': 'b:x', 'type': 'function'},
      ],
      [('a', 'b:x')],
  )

  _assert_colon_node_is_wired(_plot(info), 'a -> "b:x"')


def test_workflow_node_with_colon_can_be_an_edge_source():
  """An edge out of a node named "b:x" starts at the declared node."""
  info = _workflow_info(
      [
          {'name': 'b:x', 'type': 'function'},
          {'name': 'c', 'type': 'agent'},
      ],
      [('b:x', 'c')],
  )

  _assert_colon_node_is_wired(_plot(info), '"b:x" -> c')


def test_workflow_terminal_node_with_colon_is_linked_to_end():
  """A terminal node named "b:x" gets its edge to END from the real node."""
  info = _workflow_info(
      [
          {'name': 'a', 'type': 'agent'},
          {'name': 'b:x', 'type': 'function'},
      ],
      [('a', 'b:x')],
  )

  dot = _plot(info)

  _assert_colon_node_is_wired(dot, '"b:x" -> __END__')
  assert '\t__END__ [label=END' in dot


def test_digraph_edge_hook_quotes_colon_endpoints():
  """The graphviz hook that keeps colon names whole still exists and works."""
  assert hasattr(graphviz.Digraph, '_quote_edge'), (
      'graphviz renamed Digraph._quote_edge; update _Digraph in'
      ' graph_visualization.py'
  )

  dot = _Digraph()
  dot.edge('a', 'srv:tool')

  assert 'a -> "srv:tool"' in dot.source


def test_start_and_end_nodes_keep_their_identifiers():
  """START and END keep the __START__ and __END__ identifiers."""
  info = _workflow_info(
      [{'name': '__START__'}, {'name': 'a', 'type': 'agent'}],
      [('__START__', 'a')],
  )

  dot = _plot(info)

  assert '\t__START__ [label=START' in dot
  assert '\t__START__ -> a' in dot
  assert '\t__END__ [label=END' in dot
  assert '\ta -> __END__' in dot


def test_tools_are_injected_as_dashed_nodes_with_edges_from_their_agent():
  """A node's tools show up as dashed tool nodes linked from the node."""
  info = _workflow_info(
      [{'name': 'a', 'type': 'agent', 'tools': [{'name': 'search'}]}], []
  )

  dot = _plot(info)

  assert 'rounded,filled,dashed' in _node_line(dot, 'search')
  assert '\ta -> search [' in dot
  assert 'style=dashed' in dot


def test_completed_node_is_filled_with_completed_status_color():
  """A completed node is filled with the completed color, others are not."""
  info = _workflow_info(
      [
          {'name': 'a', 'type': 'agent'},
          {'name': 'b', 'type': 'agent'},
      ],
      [('a', 'b')],
  )
  state = {'nodes': {'a': {'status': NodeStatus.COMPLETED.value}}}

  dot = _plot(info, state)

  assert 'fillcolor="#16A34A"' in _node_line(dot, 'a')
  assert 'fillcolor="#16A34A"' not in _node_line(dot, 'b')
