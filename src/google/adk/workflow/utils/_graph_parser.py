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

"""Utility functions for parsing workflow edges and chains."""

from __future__ import annotations

from typing import Any
from typing import get_args

from .._base_node import BaseNode
from .._base_node import START
from .._graph import ChainElement
from .._graph import Edge
from .._graph import EdgeItem
from .._graph import NodeLike
from .._graph import RouteValue
from .._graph import RoutingMap
from ._workflow_graph_utils import build_node
from ._workflow_graph_utils import is_node_like


def _expand_routing_map(
    from_element: ChainElement,
    routing_map: RoutingMap,
) -> list[tuple[ChainElement, NodeLike | tuple[NodeLike, ...], RouteValue]]:
  """Expands a routing map into individual (from, to, route) triples."""
  if not routing_map:
    raise ValueError(
        "Routing map must not be empty. Provide at least one route -> node"
        " mapping."
    )

  route_value_types = get_args(RouteValue)
  expanded: list[
      tuple[ChainElement, NodeLike | tuple[NodeLike, ...], RouteValue]
  ] = []

  for route_key, target in routing_map.items():
    if not isinstance(route_key, route_value_types):
      raise ValueError(
          f"Invalid routing map key: {route_key!r} (type"
          f" {type(route_key).__name__}). Keys must be RouteValue"
          " (str, int, or bool)."
      )
    if isinstance(target, tuple):
      for node in target:
        if not is_node_like(node):
          raise ValueError(
              f"Invalid node in fan-out tuple for route {route_key!r}:"
              f" {node!r} (type {type(node).__name__})."
              " Values must be NodeLike (BaseNode, BaseAgent, BaseTool,"
              " callable, or 'START')."
          )
    elif not is_node_like(target):
      raise ValueError(
          f"Invalid routing map value for route {route_key!r}:"
          f" {target!r} (type {type(target).__name__})."
          " Values must be NodeLike (BaseNode, BaseAgent, BaseTool,"
          " callable, or 'START')."
      )
    expanded.append((from_element, target, route_key))

  return expanded


def _nodes_from_routing_map(
    routing_map: RoutingMap,
) -> list[NodeLike]:
  """Extracts all target nodes from a routing map, flattening fan-out tuples."""
  nodes: list[NodeLike] = []
  for target in routing_map.values():
    if isinstance(target, tuple):
      nodes.extend(target)
    else:
      nodes.append(target)
  return nodes


def _flatten_element(
    element: NodeLike | tuple[NodeLike, ...] | RoutingMap,
) -> list[NodeLike]:
  """Flattens a chain element into a list of individual nodes."""
  if isinstance(element, dict):
    return _nodes_from_routing_map(element)
  if isinstance(element, tuple):
    return list(element)
  return [element]


def _get_or_build_node(
    node_like: NodeLike, node_map: dict[int, BaseNode]
) -> BaseNode:
  """Gets a node from the map or builds it if not present."""
  if node_like == "START":
    return START

  node_id = id(node_like)
  if node_id in node_map:
    return node_map[node_id]

  if isinstance(node_like, BaseNode):
    wrapped = build_node(node_like)
    if wrapped is not node_like:
      node_map[node_id] = wrapped
      return wrapped
    return node_like

  node = build_node(node_like)
  node_map[node_id] = node
  return node


def _process_explicit_edge(
    edge: Edge, node_map: dict[int, BaseNode], graph_edges: list[Edge]
) -> None:
  """Processes an explicit Edge object."""
  graph_edges.append(
      Edge(
          from_node=_get_or_build_node(edge.from_node, node_map),
          to_node=_get_or_build_node(edge.to_node, node_map),
          route=edge.route,
      )
  )


def _process_routing_map_edge(
    from_el: Any,
    to_el: RoutingMap,
    node_map: dict[int, BaseNode],
    graph_edges: list[Edge],
) -> None:
  """Processes edges where the destination is a routing map."""
  if isinstance(from_el, dict):
    raise ValueError(
        "Consecutive routing maps are not allowed in a chain."
        " Split them into separate edge items."
    )

  for exp_from, exp_to, route in _expand_routing_map(from_el, to_el):
    for from_node in _flatten_element(exp_from):
      for to_node in _flatten_element(exp_to):
        graph_edges.append(
            Edge(
                from_node=_get_or_build_node(from_node, node_map),
                to_node=_get_or_build_node(to_node, node_map),
                route=route,
            )
        )


def _process_unconditional_edge(
    from_el: Any,
    to_el: Any,
    node_map: dict[int, BaseNode],
    graph_edges: list[Edge],
) -> None:
  """Processes unconditional edges between elements."""
  for from_node in _flatten_element(from_el):
    for to_node in _flatten_element(to_el):
      graph_edges.append(
          Edge(
              from_node=_get_or_build_node(from_node, node_map),
              to_node=_get_or_build_node(to_node, node_map),
              route=None,
          )
      )


def _process_chain(
    chain: tuple[Any, ...],
    node_map: dict[int, BaseNode],
    graph_edges: list[Edge],
) -> None:
  """Processes a chain of elements (tuple)."""
  for i in range(len(chain) - 1):
    from_el = chain[i]
    to_el = chain[i + 1]

    if isinstance(to_el, dict):
      _process_routing_map_edge(from_el, to_el, node_map, graph_edges)
    else:
      _process_unconditional_edge(from_el, to_el, node_map, graph_edges)


def parse_edge_items(edge_items: list[EdgeItem]) -> list[Edge]:
  """Parses a list of edge items into a flat list of Edge objects."""
  node_map: dict[int, BaseNode] = {}
  graph_edges: list[Edge] = []

  for item in edge_items:
    if isinstance(item, Edge):
      _process_explicit_edge(item, node_map, graph_edges)
    elif isinstance(item, tuple):
      _process_chain(item, node_map, graph_edges)
    else:
      raise ValueError(f"Invalid edge type: {type(item)}")

  return graph_edges
