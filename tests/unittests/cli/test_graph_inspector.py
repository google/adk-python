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

import pytest
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.cli.graph.inspector import AgentInspector


def dummy_tool(query: str) -> str:
  """Dummy tool function for testing."""
  return f"Result: {query}"


def test_agent_inspector_topology():
  agent_a = LlmAgent(name="Researcher", instruction="Search web", tools=[dummy_tool])
  agent_b = LlmAgent(name="Writer", instruction="Write report")
  
  pipeline = SequentialAgent(name="Pipeline", sub_agents=[agent_a, agent_b])

  inspector = AgentInspector(pipeline)
  topology = inspector.inspect()

  assert topology.root_id == "Pipeline"
  assert len(topology.nodes) == 4  # Pipeline, Researcher, Writer, dummy_tool
  assert len(topology.edges) == 3  # Pipeline->Researcher, Pipeline->Writer, dummy_tool->Researcher
