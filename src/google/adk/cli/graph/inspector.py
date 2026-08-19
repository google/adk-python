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

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from ...agents.base_agent import BaseAgent
from ...agents.llm_agent import LlmAgent
from ...agents.sequential_agent import SequentialAgent
from ...agents.parallel_agent import ParallelAgent
from ...agents.loop_agent import LoopAgent
from ...tools.base_tool import BaseTool
from ...apps.app import App


class GraphNode(BaseModel):
  id: str
  type: str  # 'llm_agent', 'sequential', 'parallel', 'loop', 'tool'
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
  type: str  # 'sub_agent', 'tool_binding'
  label: Optional[str] = None


class GraphTopology(BaseModel):
  root_id: str
  nodes: List[GraphNode] = Field(default_factory=list)
  edges: List[GraphEdge] = Field(default_factory=list)


class AgentInspector:
  """Inspects Google ADK Agent instances and converts them to a visual GraphTopology."""

  def __init__(self, root: Union[BaseAgent, App]):
    self.root = root
    self.nodes: Dict[str, GraphNode] = {}
    self.edges: List[GraphEdge] = []

  def inspect(self) -> GraphTopology:
    if isinstance(self.root, App):
      root_agent = self.root.root_agent
    else:
      root_agent = self.root

    root_id = self._inspect_agent(root_agent)
    return GraphTopology(
        root_id=root_id,
        nodes=list(self.nodes.values()),
        edges=self.edges,
    )

  def _inspect_agent(self, agent: BaseAgent, parent_id: Optional[str] = None) -> str:
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

    config: Dict[str, Any] = {}
    if hasattr(agent, "model") and agent.model:
      config["model"] = str(agent.model)
    if hasattr(agent, "instruction") and agent.instruction:
      config["instruction"] = str(agent.instruction)

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

  def _inspect_tool(self, tool: Union[BaseTool, Any], agent_id: str) -> str:
    tool_name = getattr(tool, "name", getattr(tool, "__name__", str(id(tool))))
    tool_id = f"tool_{agent_id}_{tool_name}"

    if tool_id not in self.nodes:
      doc = getattr(tool, "description", getattr(tool, "__doc__", None))
      self.nodes[tool_id] = GraphNode(
          id=tool_id,
          type="tool",
          label=tool_name,
          description=doc,
          parent_id=agent_id,
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
