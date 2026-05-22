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

"""Utility functions for visualizing agent graphs."""

from __future__ import annotations

import html
from typing import Any

import graphviz

from ...workflow._node_status import NodeStatus


def plot_workflow_graph(
    app_info: dict[str, Any],
    agent_state: dict[str, Any] | None = None,
    format: str = "svg",
    dark_mode: bool = True,
) -> str | bytes:
  """Plots the workflow graph with node statuses."""
  agent_state = agent_state or {}
  root_agent = app_info.get("root_agent", {})
  graph = root_agent.get("graph", {})
  is_workflow = bool(graph)

  if not graph:
    root_name = root_agent.get("name", "root_agent")
    sub_agents = root_agent.get("sub_agents", [])
    tools = root_agent.get("tools", [])

    nodes = [{"name": root_name, "type": "agent", "tools": tools}]
    edges = []

    def _traverse_sub_agents(agent_dict, parent_name):
      for sub in agent_dict.get("sub_agents", []):
        sub_name = sub.get("name")
        if sub_name:
          nodes.append(
              {"name": sub_name, "type": "agent", "tools": sub.get("tools", [])}
          )
          edges.append({
              "from_node": {"name": parent_name},
              "to_node": {"name": sub_name},
          })
          _traverse_sub_agents(sub, sub_name)

    _traverse_sub_agents(root_agent, root_name)
    graph = {"nodes": nodes, "edges": edges}

  nodes_state = agent_state.get("nodes", {})
  dot = graphviz.Digraph(comment="Workflow Visualization")

  if dark_mode:
    graph_bgcolor = "#0F172A"
    node_fillcolor = "#1E293B"
    node_color = "#475569"
    node_fontcolor = "#F8FAFC"
    edge_color = "#94A3B8"
    edge_fontcolor = "#CBD5E1"
    start_fillcolor = "#059669"
    start_color = "#047857"
    end_fillcolor = "#DC2626"
    end_color = "#B91C1C"
    status_colors = {
        NodeStatus.COMPLETED: "#16A34A",
        NodeStatus.RUNNING: "#D97706",
        NodeStatus.FAILED: "#EF4444",
        NodeStatus.INACTIVE: "#1E293B",
        NodeStatus.WAITING: "#9333EA",
        NodeStatus.CANCELLED: "#475569",
    }
  else:
    graph_bgcolor = "#F8FAFC"
    node_fillcolor = "#FFFFFF"
    node_color = "#94A3B8"
    node_fontcolor = "#0F172A"
    edge_color = "#64748B"
    edge_fontcolor = "#475569"
    start_fillcolor = "#10B981"
    start_color = "#059669"
    end_fillcolor = "#EF4444"
    end_color = "#DC2626"
    status_colors = {
        NodeStatus.COMPLETED: "#69CB87",
        NodeStatus.RUNNING: "#e8b589",
        NodeStatus.FAILED: "salmon",
        NodeStatus.INACTIVE: "#FFFFFF",
        NodeStatus.WAITING: "#d2a6e0",
        NodeStatus.CANCELLED: "lightgray",
    }

  dot.attr(
      "graph",
      bgcolor=graph_bgcolor,
      pad="0.5",
      nodesep="0.5",
      ranksep="0.8",
      fontname="Helvetica",
      splines="spline",
  )

  dot.attr(
      "node",
      shape="rect",
      style="rounded,filled",
      fillcolor=node_fillcolor,
      color=node_color,
      penwidth="1.5",
      fontname="Helvetica",
      fontcolor=node_fontcolor,
      fontsize="12",
      margin="0.25,0.15",
  )

  dot.attr(
      "edge",
      color=edge_color,
      penwidth="1.2",
      fontname="Helvetica",
      fontcolor=edge_fontcolor,
      fontsize="10",
      arrowhead="vee",
      arrowsize="0.7",
  )

  # Get nodes and edges
  nodes = list(graph.get("nodes", []))
  edges = list(graph.get("edges", []))

  # Inject tools as nodes
  tool_nodes = {}
  tool_edges = []
  for node in nodes:
    node_name = node.get("name")
    if not node_name or node_name == "__START__":
      continue

    tools = node.get("tools", [])
    for tool in tools:
      tool_name = tool.get("name") if isinstance(tool, dict) else str(tool)
      if tool_name:
        if tool_name not in tool_nodes:
          tool_type = (
              tool.get("type", "tool") if isinstance(tool, dict) else "tool"
          )
          tool_nodes[tool_name] = {"name": tool_name, "type": tool_type}
        tool_edges.append({
            "from_node": {"name": node_name},
            "to_node": {"name": tool_name},
            "is_tool_edge": True,
        })

  for n in tool_nodes.values():
    if not any(on.get("name") == n["name"] for on in nodes):
      nodes.append(n)
  edges.extend(tool_edges)

  for node in nodes:
    node_name = node.get("name")
    if not node_name or node_name == "__START__":
      continue

    outgoing_edges = [
        e for e in edges if e.get("from_node", {}).get("name") == node_name
    ]
    is_conditional = any(e.get("route") for e in outgoing_edges)

    node_data = nodes_state.get(node_name, {})
    status_val = node_data.get("status", NodeStatus.INACTIVE.value)
    if isinstance(status_val, NodeStatus):
      status = status_val
    else:
      try:
        status = NodeStatus(status_val)
      except (ValueError, KeyError):
        status = NodeStatus.INACTIVE

    fillcolor = status_colors.get(status, node_fillcolor)

    node_type = node.get("type", "node")
    icons = {
        "agent": ("✦", "#42A5F5"),
        "workflow": ("⊷", "#9333EA"),
        "function": ("ƒ", "#10B981"),
        "join": ("⌵", "#F59E0B"),
        "tool": ("🔧", "#6B7280"),
    }
    icon_data = icons.get(node_type)
    type_display = node_type.title()

    if icon_data:
      icon, color = icon_data
      escaped_name = html.escape(node_name)
      node_label = (
          f'<<FONT COLOR="{color}" POINT-SIZE="14">{icon}</FONT>'
          f" {escaped_name}>"
      )
    else:
      node_label = node_name

    if is_conditional:
      has_default = any(
          not e.get("route") or e.get("route") == "__DEFAULT__"
          for e in outgoing_edges
          if not e.get("is_tool_edge")
      )
      if not has_default:
        if icon_data:
          icon, color = icon_data
          escaped_name = html.escape(node_name)
          node_label = (
              f'<<FONT COLOR="{color}" POINT-SIZE="14">{icon}</FONT>'
              f' {escaped_name}<br/><br/><FONT POINT-SIZE="10">⚠️ [NO'
              " DEFAULT]</FONT>>"
          )
        else:
          escaped_label = html.escape(node_label)
          node_label = (
              f"<{escaped_label}<br/><br/><font point-size='10'>⚠️ [NO"
              " DEFAULT]</font>>"
          )

      dot.node(
          node_name,
          node_label,
          tooltip=type_display,
          shape="diamond",
          style="filled",
          fillcolor=fillcolor,
          height="1.2",
          width="0.8",
          margin="0.0,0.0",
      )
    elif node_type == "join":
      dot.node(
          node_name,
          node_label,
          tooltip=type_display,
          shape="oval",
          style="filled",
          fillcolor=fillcolor,
          margin="0.05,0.05",
      )
    elif node_type == "tool":
      dot.node(
          node_name,
          node_label,
          tooltip=type_display,
          style="rounded,filled,dashed",
          fillcolor=fillcolor,
      )
    else:
      dot.node(
          node_name,
          node_label,
          tooltip=type_display,
          style="rounded,filled",
          fillcolor=fillcolor,
      )

  # Add edges
  for edge in edges:
    from_node_obj = edge.get("from_node", {})
    to_node_obj = edge.get("to_node", {})

    from_node = from_node_obj.get("name")
    to_node = to_node_obj.get("name")

    if from_node == "__START__":
      dot.node(
          "__START__",
          "START",
          shape="oval",
          style="filled",
          fillcolor=start_fillcolor,
          color=start_color,
          fontcolor=node_fontcolor,
          fontname="Helvetica-Bold",
          width="0.9",
          fixedsize="true",
      )

    if from_node and to_node:
      if edge.get("is_tool_edge"):
        dot.edge(from_node, to_node, style="dashed", color=edge_color)
      else:
        label = f"  {edge.get('route')}" if edge.get("route") else ""
        dot.edge(from_node, to_node, label=label)

  terminal_nodes = []
  for node in nodes:
    node_name = node.get("name")
    if not node_name or node_name in ("__START__", "__END__"):
      continue

    if node.get("type") == "tool":
      continue

    outgoing_edges = [
        e
        for e in edges
        if e.get("from_node", {}).get("name") == node_name
        and not e.get("is_tool_edge")
    ]

    is_terminal = False
    if not outgoing_edges:
      is_terminal = True

    if is_terminal:
      terminal_nodes.append(node_name)

  if is_workflow and terminal_nodes:
    dot.node(
        "__END__",
        "END",
        shape="oval",
        style="filled",
        fillcolor=end_fillcolor,
        color=end_color,
        fontcolor=node_fontcolor,
        fontname="Helvetica-Bold",
        width="0.9",
        fixedsize="true",
    )
    for t_node in terminal_nodes:
      dot.edge(t_node, "__END__")

  if format == "dot":
    return dot.source
  if format == "svg":
    return dot.pipe(format="svg").decode("utf-8")
  return dot.pipe(format=format)
