"""Graph export functions for visualization.

Standalone functions that export graph structure and execution data
in D3-compatible JSON format. Separated from GraphAgent for
single-responsibility.
"""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from .graph_agent import GraphAgent


def export_graph_structure(graph: GraphAgent) -> Dict[str, Any]:
  """Export graph structure in D3-compatible JSON format.

  Args:
      graph: GraphAgent instance to export

  Returns:
      Dictionary with nodes, links, and metadata suitable for
      D3.js or other graph visualization tools.
  """
  nodes = []
  links = []

  for node_name, node in graph.nodes.items():
    node_data = {
        "id": node_name,
        "type": "agent" if node.agent else "function",
        "name": node.name,
    }
    nodes.append(node_data)

  for node_name, node in graph.nodes.items():
    for edge in node.edges:
      link_data = {
          "source": node_name,
          "target": edge.target_node,
          "conditional": edge.has_condition,
      }
      links.append(link_data)

  metadata = {
      "start_node": graph.start_node,
      "end_nodes": graph.end_nodes,
      "checkpointing": graph.checkpointing,
      "max_iterations": graph.max_iterations,
  }

  return {
      "nodes": nodes,
      "links": links,
      "metadata": metadata,
      "directed": True,
  }


def export_graph_with_execution(
    graph: GraphAgent,
    execution_history: Optional[List[Dict[str, Any]]] = None,
    state_history: Optional[List[Dict[str, Any]]] = None,
    interrupt_markers: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
  """Export graph with execution history, state evolution, and interrupts.

  Enhanced D3-compatible format including runtime information for
  interactive visualization and replay.

  Args:
      graph: GraphAgent instance to export
      execution_history: List of executed nodes with timestamps
      state_history: List of state snapshots after each node
      interrupt_markers: List of interrupt events

  Returns:
      Enhanced dictionary with execution data overlaid on graph structure.
  """
  base_structure = export_graph_structure(graph)
  nodes = base_structure["nodes"]
  links = base_structure["links"]

  if execution_history:
    node_executions: Dict[str, List[Dict[str, Any]]] = {}
    for exec_record in execution_history:
      node_name = exec_record["node"]
      if node_name not in node_executions:
        node_executions[node_name] = []
      node_executions[node_name].append(exec_record)

    for node in nodes:
      node_id = node["id"]
      if node_id in node_executions:
        node["executions"] = node_executions[node_id]
        node["execution_count"] = len(node_executions[node_id])
        statuses = [
            e.get("status", "unknown") for e in node_executions[node_id]
        ]
        node["status_summary"] = {
            "success": statuses.count("success"),
            "error": statuses.count("error"),
            "interrupted": statuses.count("interrupted"),
        }
      else:
        node["executions"] = []
        node["execution_count"] = 0

  if execution_history:
    link_traversals: Dict[Tuple[str, str], int] = {}
    for i in range(len(execution_history) - 1):
      source = execution_history[i]["node"]
      target = execution_history[i + 1]["node"]
      link_key = (source, target)
      link_traversals[link_key] = link_traversals.get(link_key, 0) + 1

    for link in links:
      link_key = (link["source"], link["target"])
      link["traversals"] = link_traversals.get(link_key, 0)

  if interrupt_markers:
    node_interrupts: Dict[str, List[Dict[str, Any]]] = {}
    for interrupt in interrupt_markers:
      node_name = interrupt.get("node")
      if node_name:
        if node_name not in node_interrupts:
          node_interrupts[node_name] = []
        node_interrupts[node_name].append(interrupt)

    for node in nodes:
      node_id = node["id"]
      if node_id in node_interrupts:
        node["interrupt_markers"] = node_interrupts[node_id]
        node["interrupt_count"] = len(node_interrupts[node_id])

  return {
      "nodes": nodes,
      "links": links,
      "metadata": base_structure["metadata"],
      "execution_history": execution_history or [],
      "state_history": state_history or [],
      "interrupt_markers": interrupt_markers or [],
      "directed": True,
      "enhanced": True,
  }


def export_execution_timeline(
    execution_history: List[Dict[str, Any]],
    state_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
  """Export execution timeline for temporal visualization.

  Creates a timeline view of graph execution suitable for
  Gantt charts, timeline visualizations, or replay UIs.

  Args:
      execution_history: List of executed nodes with timestamps
      state_history: Optional state snapshots at each step

  Returns:
      Dictionary with timeline, total_duration, total_steps, iterations.
  """
  if not execution_history:
    return {
        "timeline": [],
        "total_duration": 0,
        "total_steps": 0,
        "iterations": 0,
    }

  timeline = []
  for i, exec_record in enumerate(execution_history):
    step_data = {
        "step": i,
        "node": exec_record["node"],
        "timestamp": exec_record.get("timestamp", 0),
        "iteration": exec_record.get("iteration", 1),
        "status": exec_record.get("status", "unknown"),
    }

    if i < len(execution_history) - 1:
      next_timestamp = execution_history[i + 1].get("timestamp", 0)
      step_data["duration"] = next_timestamp - step_data["timestamp"]
    else:
      step_data["duration"] = 0

    if state_history and i < len(state_history):
      step_data["state"] = state_history[i].get("state", {})

    timeline.append(step_data)

  total_duration = 0
  if len(timeline) > 1:
    total_duration = timeline[-1]["timestamp"] - timeline[0]["timestamp"]

  max_iteration = max((step["iteration"] for step in timeline), default=0)

  return {
      "timeline": timeline,
      "total_duration": total_duration,
      "total_steps": len(timeline),
      "iterations": max_iteration,
  }
