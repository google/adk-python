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

from __future__ import annotations

"""Utility functions for serializing agent graphs for the web UI."""

import logging
from typing import Any

logger = logging.getLogger("google_adk." + __name__)

from ...agents.base_agent import BaseAgent
from ...tools.base_toolset import BaseToolset

# Node type mapping for cleaner lookup
NODE_TYPE_MAP = {
    "FunctionNode": "function",
    "ToolNode": "tool",
    "JoinNode": "join",
}

# Fields to skip during agent serialization
SKIP_FIELDS = {
    "parent_agent",
    "before_agent_callback",
    "after_agent_callback",
    "before_model_callback",
    "after_model_callback",
    "on_model_error_callback",
    "before_tool_callback",
    "after_tool_callback",
    "on_tool_error_callback",
}


def _get_node_field(node: Any, field_name: str) -> Any:
  """Safely get a node field using object.__getattribute__."""
  return object.__getattribute__(node, field_name)


def serialize_node_like(item: Any) -> Any:
  """Serialize a NodeLike object (str, BaseAgent, BaseTool, Callable, BaseNode)."""
  if item == "START":
    return "START"
  # Handle primitives
  if isinstance(item, (str, int, float, bool)):
    return item
  # Handle BaseAgent
  class_name = type(item).__name__
  if "Agent" in class_name and hasattr(item, "model_fields"):
    return serialize_agent(item)
  # Handle BaseNode
  if "Node" in class_name and hasattr(item, "get_name"):
    return serialize_node(item)
  # Handle callable
  if callable(item):
    return {"name": getattr(item, "__name__", str(item)), "type": "function"}
  return str(item)


def serialize_node(node: Any) -> dict[str, Any]:
  """Serialize a node (BaseNode subclasses like FunctionNode, AgentNode, etc.)."""
  class_name = type(node).__name__
  node_name = _get_node_field(node, "name")

  # Handle START node
  if node_name == "__START__":
    return {
        "name": "__START__",
        "type": "start",
        "rerun_on_resume": _get_node_field(node, "rerun_on_resume"),
    }

  if hasattr(node, "model_fields"):
    result = serialize_agent(node)
    if "type" not in result:
      if getattr(node, "graph", None) is not None:
        result["type"] = "workflow"
      else:
        result["type"] = NODE_TYPE_MAP.get(
            class_name, "agent" if "Agent" in class_name else "node"
        )
    return result

  # Get node type from mapping or default to 'node'
  node_type = NODE_TYPE_MAP.get(class_name, "node")

  return {
      "name": node_name,
      "type": node_type,
      "rerun_on_resume": _get_node_field(node, "rerun_on_resume"),
  }


def serialize_agent(agent: BaseAgent) -> dict[str, Any]:
  """Recursively serialize an agent, excluding non-serializable fields."""
  agent_dict = {}

  for field_name, field_info in agent.__class__.model_fields.items():
    if field_name in SKIP_FIELDS:
      continue

    value = getattr(agent, field_name, None)

    if value is None:
      continue

    # Handle sub_agents recursively
    if field_name == "sub_agents":
      agent_dict[field_name] = [
          serialize_agent(sub_agent) for sub_agent in value
      ]
    # Handle nodes field (for _Mesh/LlmAgent)
    elif field_name == "nodes":
      try:
        serialized_nodes = []
        for node in value:
          if hasattr(node, "model_fields"):
            serialized_nodes.append(serialize_agent(node))
          else:
            serialized_nodes.append(serialize_node(node))
        agent_dict[field_name] = serialized_nodes
      except Exception as e:
        logger.warning("Error serializing nodes field: %s", e)
    # Handle graph field (Graph with nodes and edges)
    elif field_name == "graph":
      try:
        graph_dict = {}
        # Serialize nodes
        if hasattr(value, "nodes") and value.nodes:
          graph_dict["nodes"] = [serialize_node(node) for node in value.nodes]
        # Serialize edges
        if hasattr(value, "edges") and value.edges:
          serialized_edges = []
          for edge in value.edges:
            edge_dict = {}
            if hasattr(edge, "from_node"):
              edge_dict["from_node"] = serialize_node(edge.from_node)
            if hasattr(edge, "to_node"):
              edge_dict["to_node"] = serialize_node(edge.to_node)
            if hasattr(edge, "route") and edge.route is not None:
              edge_dict["route"] = edge.route
            serialized_edges.append(edge_dict)
          graph_dict["edges"] = serialized_edges
        agent_dict[field_name] = graph_dict
      except Exception:
        pass
    # Handle edges field (list of EdgeItems)
    elif field_name == "edges":
      try:
        serialized_edges = []
        for edge_item in value:
          if isinstance(edge_item, tuple):
            serialized = []
            for elem in edge_item:
              if isinstance(elem, dict):
                serialized.append(
                    {str(k): serialize_node_like(v) for k, v in elem.items()}
                )
              else:
                serialized.append(serialize_node_like(elem))
            serialized_edges.append(serialized)
          elif hasattr(edge_item, "from_node") and hasattr(
              edge_item, "to_node"
          ):
            edge_dict = {
                "from_node": serialize_node(edge_item.from_node),
                "to_node": serialize_node(edge_item.to_node),
            }
            if hasattr(edge_item, "route") and edge_item.route is not None:
              edge_dict["route"] = edge_item.route
            serialized_edges.append(edge_dict)
          else:
            serialized_edges.append(str(edge_item))
        agent_dict[field_name] = serialized_edges
      except Exception:
        pass
    # Handle tools field
    elif field_name == "tools":
      try:
        sub_agents = getattr(agent, "sub_agents", []) or []
        sub_agent_names = {
            getattr(sa, "name", None)
            for sa in sub_agents
            if getattr(sa, "name", None)
        }

        serialized_tools = []
        for tool in value:
          tool_name = None
          if callable(tool):
            tool_name = getattr(tool, "__name__", str(tool))
          elif hasattr(tool, "name"):
            tool_name = tool.name
          elif isinstance(tool, BaseToolset):
            tool_name = type(tool).__name__

          if tool_name and tool_name in sub_agent_names:
            continue

          if tool_name is not None:
            serialized_tools.append({
                "name": tool_name,
                "type": "tool",
            })
          else:
            serialized_tools.append(str(tool))
        agent_dict[field_name] = serialized_tools
      except Exception:
        pass
    else:
      try:
        if callable(value):
          continue
        # Handle nested agents
        if isinstance(value, BaseAgent):
          agent_dict[field_name] = serialize_agent(value)
        # Handle simple types and collections
        elif isinstance(value, (str, int, float, bool, list, dict)):
          agent_dict[field_name] = value
        elif hasattr(value, "model_dump"):
          agent_dict[field_name] = value.model_dump(
              mode="python", exclude_none=True
          )
        else:
          agent_dict[field_name] = str(value)
      except Exception as e:
        logger.warning(
            "Error serializing field '%s' of agent %s: %s",
            field_name,
            type(agent).__name__,
            e,
        )

  return agent_dict


def serialize_app_info(app: Any, readme: str | None = None) -> dict[str, Any]:
  """Serialize app information for the build_graph endpoint."""
  root = app.root_agent
  try:
    root_agent_data = serialize_agent(root)
  except Exception as e:
    logger.error("Error serializing root agent/node: %s", e, exc_info=True)
    raise

  app_info = {
      "name": app.name,
      "root_agent": root_agent_data,
  }

  # Add optional fields if present
  if app.plugins:
    app_info["plugins"] = [
        {"name": getattr(plugin, "name", type(plugin).__name__)}
        for plugin in app.plugins
    ]

  if app.context_cache_config:
    try:
      app_info["context_cache_config"] = app.context_cache_config.model_dump(
          mode="python", exclude_none=True
      )
    except Exception:
      pass

  if app.resumability_config:
    try:
      app_info["resumability_config"] = app.resumability_config.model_dump(
          mode="python", exclude_none=True
      )
    except Exception:
      pass

  # Include README content if provided
  if readme:
    app_info["readme"] = readme

  return app_info
