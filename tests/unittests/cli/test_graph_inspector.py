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
