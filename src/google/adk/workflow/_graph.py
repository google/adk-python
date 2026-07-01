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

"""Defines the graph and edges in the Workflow."""

from __future__ import annotations

from collections.abc import Callable
import logging

logger = logging.getLogger("google_adk." + __name__)
from typing import Annotated
from typing import Any
from typing import Literal
from typing import TypeAlias

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import PrivateAttr
from pydantic import SerializeAsAny

from ..tools.base_tool import BaseTool
from ._base_node import BaseNode

RouteValue: TypeAlias = bool | int | str
"""Type alias for valid routing values used in conditional graph edges."""

NodeLike: TypeAlias = (
    BaseNode | BaseTool | Callable[..., Any] | Literal["START"]
)
"""Type alias for objects that can be converted to a workflow node."""

RoutingMap: TypeAlias = dict[RouteValue, NodeLike | tuple[NodeLike, ...]]
"""A mapping from route values to destination nodes.

Syntactic sugar for declaring multiple routed edges from a single source.
Values can be a single node or a tuple of nodes (fan-out).

Examples::

    {"route_a": node_a, "route_b": node_b}
    {"route_x": (node_a, node_b)}  # fan-out: both triggered
"""


class Edge(BaseModel):
  """An edge in the workflow graph."""

  model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

  from_node: Annotated[BaseNode, SerializeAsAny()]
  """The from node."""

  to_node: Annotated[BaseNode, SerializeAsAny()]
  """The to node."""

  route: RouteValue | list[RouteValue] | None = Field(
      description=(
          "The route(s) that this edge is associated with."
          " A single value or a list of values. The edge is followed when the"
          " emitted route matches any value in the list."
      ),
      default=None,
  )


ChainElement: TypeAlias = NodeLike | tuple[NodeLike, ...] | RoutingMap
"""Type alias for an element in a workflow chain.

Can be a single NodeLike, a tuple of NodeLike (fan-out), or a RoutingMap.
"""

EdgeItem: TypeAlias = Edge | tuple[ChainElement, ...]
"""Type alias for an item that can be parsed into workflow edges.

Can be an explicit Edge object, or a tuple representing a chain of elements.
"""
DEFAULT_ROUTE = "__DEFAULT__"

# --- Graph ---


class Graph(BaseModel):
  """A workflow graph."""

  model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

  nodes: list[Annotated[BaseNode, SerializeAsAny()]] = Field(
      default_factory=list
  )
  """The nodes in the workflow graph."""

  edges: list[Edge] = Field(default_factory=list)
  """The edges in the workflow graph."""

  _terminal_node_names: set[str] = PrivateAttr(default_factory=set)
  """Nodes with no outgoing edges. Computed by validate_graph."""

  @classmethod
  def from_edge_items(cls, edge_items: list[EdgeItem]) -> Graph:
    """Creates a Graph from a list of edge items."""
    from .utils._graph_parser import parse_edge_items

    return Graph(edges=parse_edge_items(edge_items))

  def model_post_init(self, context: Any) -> None:
    """Populates nodes from edges."""
    if "nodes" in self.model_fields_set and self.nodes:
      raise ValueError(
          "Nodes are inferred from edges, do not set nodes explicitly."
      )
    if self.edges:
      # Use a dictionary to preserve order and deduplicate nodes by object id.
      nodes = {
          id(node): node
          for edge in self.edges
          for node in [edge.from_node, edge.to_node]
      }
      self.nodes = list(nodes.values())

  def get_next_pending_nodes(
      self,
      node_name: str,
      routes_to_match: RouteValue | list[RouteValue] | None,
  ) -> list[str]:
    """Determines the next nodes to transition to PENDING state based on routes."""
    next_pending_nodes: list[str] = []
    matched_specific_route = False
    default_route_node: str | None = None
    has_routing_edges = False

    for edge in self.edges:
      if edge.from_node.name == node_name:
        if edge.route is None:
          # Edges with no route tag are always triggered.
          next_pending_nodes.append(edge.to_node.name)
          continue

        has_routing_edges = True
        if edge.route == DEFAULT_ROUTE:
          default_route_node = edge.to_node.name
          continue

        # Normalize edge routes to a set for matching.
        edge_routes = (
            set(edge.route) if isinstance(edge.route, list) else {edge.route}
        )

        edge_matched = False
        if isinstance(routes_to_match, list):
          if edge_routes & set(routes_to_match):
            edge_matched = True
        elif routes_to_match in edge_routes:
          edge_matched = True

        if edge_matched:
          next_pending_nodes.append(edge.to_node.name)
          matched_specific_route = True

    if not matched_specific_route and default_route_node:
      next_pending_nodes.append(default_route_node)

    if has_routing_edges and not next_pending_nodes:
      logger.warning(
          "Node '%s' has conditional/DEFAULT edges but none were matched by the"
          " emitted route(s): %s. The branch will end.",
          node_name,
          routes_to_match,
      )

    return next_pending_nodes

  def validate_graph(self) -> None:
    """Validates the workflow graph."""
    from .utils._graph_validation import validate_graph

    self._terminal_node_names = validate_graph(self.nodes, self.edges)
