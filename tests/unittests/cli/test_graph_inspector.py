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

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.apps.app import App
from google.adk.cli.graph.inspector import AgentInspector
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.workflow import START
from google.adk.workflow import Workflow


def dummy_tool(query: str) -> str:
  """Dummy tool function for testing."""
  return f"Result: {query}"


def test_agent_inspector_topology():
  agent_a = LlmAgent(
      name="Researcher", instruction="Search web", tools=[dummy_tool]
  )
  agent_b = LlmAgent(name="Writer", instruction="Write report")

  pipeline = SequentialAgent(name="Pipeline", sub_agents=[agent_a, agent_b])

  inspector = AgentInspector(pipeline)
  topology = inspector.inspect()

  assert topology.root_id == "Pipeline"
  assert len(topology.nodes) == 4  # Pipeline, Researcher, Writer, dummy_tool
  assert (
      len(topology.edges) == 3
  )  # Pipeline->Researcher, Pipeline->Writer, dummy_tool->Researcher
  tool_node = next(
      node for node in topology.nodes if node.label == "dummy_tool"
  )
  assert "def dummy_tool" in tool_node.config["implementation"]


def test_agent_inspector_includes_application_plugins():
  """Application-wide plugins are visible as read-only graph capabilities."""
  root_agent = LlmAgent(name="Root")
  plugin = BasePlugin(name="audit")

  topology = AgentInspector(
      App(name="sample", root_agent=root_agent, plugins=[plugin])
  ).inspect()

  plugin_node = next(node for node in topology.nodes if node.type == "plugin")
  plugin_edge = next(
      edge for edge in topology.edges if edge.type == "app_plugin"
  )
  assert plugin_node.label == "audit"
  assert plugin_node.config["read_only"] is True
  assert plugin_edge.target == "Root"


def test_agent_inspector_includes_workflow_nodes_and_routes():
  """Workflow scheduling nodes and route connections remain visible."""
  first = LlmAgent(name="first")
  second = LlmAgent(name="second")
  workflow = Workflow(name="pipeline", edges=[(START, first), (first, second)])

  topology = AgentInspector(workflow).inspect()

  assert any(node.type == "workflow" for node in topology.nodes)
  assert {"pipeline:first", "pipeline:second"}.issubset(
      {node.id for node in topology.nodes}
  )
  assert any(edge.type == "workflow_route" for edge in topology.edges)
