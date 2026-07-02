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

"""Utility functions for validating workflow graphs."""

from __future__ import annotations

from collections import Counter

from .._base_node import BaseNode
from .._base_node import START
from .._graph import DEFAULT_ROUTE
from .._graph import Edge


def _detect_unconditional_cycles(
    edges: list[Edge], node_names: Set[str]
) -> None:
  """Detects unconditional cycles in the graph."""
  unconditional_adj: dict[str, list[str]] = {name: [] for name in node_names}
  for edge in edges:
    if edge.route is None:
      unconditional_adj[edge.from_node.name].append(edge.to_node.name)

  in_stack: set[str] = set()
  done: set[str] = set()

  def _dfs(node: str, path: list[str]) -> None:
    in_stack.add(node)
    path.append(node)
    for neighbor in unconditional_adj[node]:
      if neighbor in in_stack:
        cycle_start = path.index(neighbor)
        cycle = path[cycle_start:] + [neighbor]
        raise ValueError(
            "Graph validation failed. Unconditional cycle detected:"
            f" {' -> '.join(cycle)}. Cycles must include at"
            " least one conditional (routed) edge to avoid"
            " infinite loops."
        )
      if neighbor not in done:
        _dfs(neighbor, path)
    path.pop()
    in_stack.remove(node)
    done.add(node)

  for name in node_names:
    if name not in done:
      _dfs(name, [])


def _validate_duplicate_node_names(nodes: list[BaseNode]) -> set[str]:
  """Checks for duplicate node names."""
  names = [node.name for node in nodes]
  duplicates = sorted(
      name for name, count in Counter(names).items() if count > 1
  )

  if duplicates:
    raise ValueError(
        "Graph validation failed. Duplicate node names found:"
        f" {duplicates}. This means multiple distinct node objects"
        " have the same name. If you intended to reuse the same node, ensure"
        " you pass the exact same object instance. If you intended to have"
        " distinct nodes, ensure they have unique names."
    )
  return set(names)


def _validate_start_node(node_names: set[str]) -> None:
  """Checks for existence of START node."""
  if START.name not in node_names:
    raise ValueError(
        "Graph validation failed. START node (name: "
        f"'{START.name}') not found in graph nodes."
    )


def _validate_connectivity(edges: list[Edge], node_names: set[str]) -> None:
  """Checks connectivity and reachability from START."""
  to_nodes: set[str] = set()
  adj: dict[str, set[str]] = {name: set() for name in node_names}
  for edge in edges:
    adj[edge.from_node.name].add(edge.to_node.name)
    to_nodes.add(edge.to_node.name)

  reachable: set[str] = set()
  stack = [START.name]
  while stack:
    node = stack.pop()
    if node in reachable:
      continue
    reachable.add(node)
    stack.extend(adj[node] - reachable)

  unreachable_nodes = node_names - reachable
  if unreachable_nodes:
    raise ValueError(
        "Graph validation failed. The following nodes are unreachable"
        f" from START: {sorted(unreachable_nodes)}"
    )
  if START.name in to_nodes:
    raise ValueError(
        "Graph validation failed. START node must not have incoming edges."
    )


def _validate_duplicate_edges(edges: list[Edge]) -> None:
  """Checks for duplicate edges."""
  seen_edges = set()
  for edge in edges:
    edge_tuple = (edge.from_node.name, edge.to_node.name)
    if edge_tuple in seen_edges:
      raise ValueError(
          "Graph validation failed. Duplicate edge found: from="
          f"{edge.from_node.name}, to={edge.to_node.name}"
      )
    seen_edges.add(edge_tuple)


def _validate_default_routes(edges: list[Edge]) -> None:
  """Checks constraints on DEFAULT_ROUTE."""
  default_route_edges: dict[str, str] = {}
  for edge in edges:
    if isinstance(edge.route, list) and DEFAULT_ROUTE in edge.route:
      raise ValueError(
          "Graph validation failed. DEFAULT_ROUTE cannot be combined"
          " with other routes in a list (edge from="
          f"{edge.from_node.name}, to={edge.to_node.name})."
          " Use a separate edge for DEFAULT_ROUTE."
      )
    if edge.route == DEFAULT_ROUTE:
      from_node_name = edge.from_node.name
      if from_node_name in default_route_edges:
        raise ValueError(
            "Graph validation failed. Multiple DEFAULT_ROUTE edges found"
            f" from node {from_node_name} to"
            f" {default_route_edges[from_node_name]} and"
            f" {edge.to_node.name}"
        )
      default_route_edges[from_node_name] = edge.to_node.name


def _validate_static_schemas(edges: list[Edge]) -> None:
  """Validates static schemas on edges."""
  for edge in edges:
    from_node = edge.from_node
    to_node = edge.to_node
    if from_node.output_schema and to_node.input_schema:
      if from_node.output_schema != to_node.input_schema:
        raise ValueError(
            "Graph validation failed. Schema mismatch on edge"
            f" {from_node.name} -> {to_node.name}."
            f" Output schema {from_node.output_schema} does not match"
            f" input schema {to_node.input_schema}."
        )


def _validate_chat_agent_wiring(edges: list[Edge]) -> None:
  """Validates that chat-mode agents do not have incoming edges from non-START nodes."""
  from ...agents.llm_agent import LlmAgent

  for edge in edges:
    to_node = edge.to_node
    if (
        isinstance(to_node, LlmAgent)
        and getattr(to_node, "mode", None) == "chat"
    ):
      if edge.from_node.name != START.name:
        raise ValueError(
            f"The agent '{to_node.name}' has been added to the workflow with"
            f" mode='chat' following node '{edge.from_node.name}'. This is"
            " not supported because chat-mode agents rely on conversational"
            " history (session events) and cannot consume direct node inputs"
            " from preceding nodes. Please change the agent's mode to"
            " 'single_turn'"
        )


def _compute_terminal_nodes(
    nodes: list[BaseNode], edges: list[Edge]
) -> set[str]:
  """Computes terminal nodes (no outgoing edges)."""
  from_names = {edge.from_node.name for edge in edges}
  return {
      n.name for n in nodes if n.name != START.name and n.name not in from_names
  }


def validate_graph(nodes: list[BaseNode], edges: list[Edge]) -> set[str]:
  """Validates the workflow graph and returns terminal node names."""
  node_names = _validate_duplicate_node_names(nodes)
  _validate_start_node(node_names)
  _validate_connectivity(edges, node_names)
  _validate_duplicate_edges(edges)
  _validate_default_routes(edges)
  _detect_unconditional_cycles(edges, node_names)
  _validate_static_schemas(edges)
  _validate_chat_agent_wiring(edges)
  return _compute_terminal_nodes(nodes, edges)
