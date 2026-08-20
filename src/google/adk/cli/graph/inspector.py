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

import inspect
import textwrap
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from pydantic import BaseModel
from pydantic import Field

from ...agents.base_agent import BaseAgent
from ...agents.llm_agent import LlmAgent
from ...agents.loop_agent import LoopAgent
from ...agents.parallel_agent import ParallelAgent
from ...agents.sequential_agent import SequentialAgent
from ...apps.app import App
from ...tools.base_tool import BaseTool
from ...workflow._workflow import Workflow


class GraphNode(BaseModel):
  id: str
  type: str  # Agent, tool, plugin, workflow, or custom component type.
  label: str
  description: Optional[str] = None
  sub_agents: List[str] = Field(default_factory=list)
  tools: List[str] = Field(default_factory=list)
  parent_id: Optional[str] = None
  config: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
  id: str
  source: str
  target: str
  type: str  # 'sub_agent', 'tool_binding', or 'app_plugin'
  label: Optional[str] = None


class GraphTopology(BaseModel):
  root_id: str
  nodes: List[GraphNode] = Field(default_factory=list)
  edges: List[GraphEdge] = Field(default_factory=list)


class AgentInspector:
  """Inspects Google ADK Agent instances and converts them to a visual GraphTopology."""

  def __init__(self, root: Union[BaseAgent, App, Any]):
    self.root = root
    self.nodes: Dict[str, GraphNode] = {}
    self.edges: List[GraphEdge] = []

  def inspect(self) -> GraphTopology:
    if isinstance(self.root, App):
      root_agent = self.root.root_agent
    else:
      root_agent = self.root

    root_id = self._inspect_agent(root_agent)
    if isinstance(self.root, App):
      for index, plugin in enumerate(self.root.plugins):
        plugin_name = getattr(plugin, "name", type(plugin).__name__)
        plugin_id = f"plugin_{index}_{plugin_name}"
        self.nodes[plugin_id] = GraphNode(
            id=plugin_id,
            type="plugin",
            label=plugin_name,
            description=(type(plugin).__doc__ or "").strip(),
            config={"class": type(plugin).__name__, "read_only": True},
        )
        self.edges.append(
            GraphEdge(
                id=f"edge_plugin_{plugin_id}_{root_id}",
                source=plugin_id,
                target=root_id,
                type="app_plugin",
            )
        )
    return GraphTopology(
        root_id=root_id,
        nodes=list(self.nodes.values()),
        edges=self.edges,
    )

  def _inspect_agent(self, agent: Any, parent_id: Optional[str] = None) -> str:
    if isinstance(agent, Workflow):
      return self._inspect_workflow(agent, parent_id=parent_id)
    agent_id = getattr(agent, "name", str(id(agent)))

    # Determine type
    if isinstance(agent, SequentialAgent):
      node_type = "sequential"
    elif isinstance(agent, ParallelAgent):
      node_type = "parallel"
    elif isinstance(agent, LoopAgent):
      node_type = "loop"
    elif isinstance(agent, LlmAgent):
      node_type = "llm_agent"
    else:
      node_type = "base_agent"

    config: Dict[str, Any] = {"class": type(agent).__name__}
    if hasattr(agent, "model") and agent.model:
      config["model"] = str(agent.model)
    if hasattr(agent, "instruction") and agent.instruction:
      config["instruction"] = str(agent.instruction)
    if hasattr(agent, "max_iterations") and agent.max_iterations is not None:
      config["max_iterations"] = agent.max_iterations
    if node_type == "base_agent":
      config["read_only"] = True

    node = GraphNode(
        id=agent_id,
        type=node_type,
        label=getattr(agent, "name", "Agent"),
        description=getattr(agent, "description", None),
        parent_id=parent_id,
        config=config,
    )
    self.nodes[agent_id] = node

    # Inspect Tools
    if hasattr(agent, "tools") and agent.tools:
      for tool in agent.tools:
        tool_id = self._inspect_tool(tool, agent_id)
        node.tools.append(tool_id)

    # Inspect Sub-Agents
    sub_agents = getattr(agent, "sub_agents", [])
    if sub_agents:
      for sub in sub_agents:
        sub_id = self._inspect_agent(sub, parent_id=agent_id)
        node.sub_agents.append(sub_id)
        # Add edge
        self.edges.append(
            GraphEdge(
                id=f"edge_sub_{agent_id}_{sub_id}",
                source=agent_id,
                target=sub_id,
                type="sub_agent",
            )
        )

    return agent_id

  def _inspect_workflow(
      self, workflow: Workflow, parent_id: Optional[str] = None
  ) -> str:
    """Exposes workflow nodes and routes without claiming they are editable."""
    workflow_id = workflow.name
    self.nodes[workflow_id] = GraphNode(
        id=workflow_id,
        type="workflow",
        label=workflow.name,
        description=workflow.description,
        parent_id=parent_id,
        config={"class": type(workflow).__name__, "read_only": True},
    )
    if not workflow.graph:
      return workflow_id

    node_ids: dict[int, str] = {}
    for workflow_node in workflow.graph.nodes:
      node_id = f"{workflow_id}:{workflow_node.name}"
      node_ids[id(workflow_node)] = node_id
      self.nodes[node_id] = GraphNode(
          id=node_id,
          type="workflow_node",
          label=workflow_node.name,
          description=workflow_node.description,
          parent_id=workflow_id,
          config={"class": type(workflow_node).__name__, "read_only": True},
      )
      self.edges.append(
          GraphEdge(
              id=f"edge_workflow_contains_{workflow_id}_{node_id}",
              source=workflow_id,
              target=node_id,
              type="workflow_contains",
          )
      )

    for index, edge in enumerate(workflow.graph.edges):
      source_id = node_ids.get(id(edge.from_node))
      target_id = node_ids.get(id(edge.to_node))
      if not source_id or not target_id:
        continue
      self.edges.append(
          GraphEdge(
              id=f"edge_workflow_route_{workflow_id}_{index}",
              source=source_id,
              target=target_id,
              type="workflow_route",
              label=str(edge.route) if edge.route is not None else None,
          )
      )
    return workflow_id

  def _inspect_tool(self, tool: Union[BaseTool, Any], agent_id: str) -> str:
    tool_name = getattr(tool, "name", getattr(tool, "__name__", str(id(tool))))
    tool_id = f"tool_{agent_id}_{tool_name}"

    if tool_id not in self.nodes:
      doc = getattr(tool, "description", getattr(tool, "__doc__", None))
      config: Dict[str, Any] = {"class": type(tool).__name__}
      implementation = self._tool_implementation(tool)
      if implementation:
        config["implementation"] = implementation
      else:
        # The graph can still show tools backed by OpenAPI, MCP, toolsets, or
        # dynamically-created callables.  They are intentionally read-only:
        # serializing an invented implementation would make generated code lie.
        config["read_only"] = True
        config["generation_hint"] = (
            "This tool's implementation is not available as a Python function. "
            "Add a Python implementation before generating managed source."
        )
      self.nodes[tool_id] = GraphNode(
          id=tool_id,
          type="tool",
          label=tool_name,
          description=doc,
          parent_id=agent_id,
          config=config,
      )

    self.edges.append(
        GraphEdge(
            id=f"edge_tool_{tool_id}_{agent_id}",
            source=tool_id,
            target=agent_id,
            type="tool_binding",
        )
    )

    return tool_id

  @staticmethod
  def _tool_implementation(tool: Union[BaseTool, Any]) -> Optional[str]:
    """Returns inspectable Python function source for a FunctionTool.

    ADK converts callables passed to ``tools`` into ``FunctionTool`` objects.
    Keeping their original function source allows the graph generator to emit
    a real implementation instead of a deceptive placeholder.  Native tools
    and dynamically-created functions are left read-only when source is not
    available.
    """
    function = getattr(tool, "func", tool)
    if not inspect.isfunction(function):
      return None
    try:
      source = textwrap.dedent(inspect.getsource(function)).strip()
      parsed = compile(source, "<tool>", "exec")
    except (OSError, TypeError, SyntaxError):
      return None
    del parsed
    return source or None
