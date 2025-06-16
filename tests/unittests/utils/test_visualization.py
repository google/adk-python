"""
Tests for src/google/adk/utils/visualization.py
"""

from __future__ import annotations

from google.adk.agents import Agent
from google.adk.agents import LoopAgent
from google.adk.agents import ParallelAgent
from google.adk.agents import SequentialAgent
from google.adk.utils.visualization import build_mermaid


def test_build_mermaid():
  """
  Tests build_mermaid function.

  We build an agent workflow and then pass the
  root_agent to build_mermaid, which builds
  a mermaid diagram.
  """

  agent1 = Agent(
      model="nonexistent-model",
      name="agent1",
      description="Example",
      instruction="""
    Example
    """,
  )

  agent2 = Agent(
      model="nonexistent-model",
      name="agent2",
      description="Example",
      instruction="""
    Example
    """,
  )

  agent3 = Agent(
      model="nonexistent-model",
      name="agent3",
      description="Example",
      instruction=f"""
    Example
    """,
  )

  agent4 = Agent(
      model="nonexistent-model",
      name="agent4",
      description="Example",
      instruction=f"""
    Example
    """,
  )

  agent5 = Agent(
      model="nonexistent-model",
      name="agent5",
      description="Example",
      instruction=f"""
    Example
    """,
  )

  agent6 = Agent(
      model="nonexistent-model",
      name="agent6",
      description="Example",
      instruction=f"""
    Example
    """,
  )

  agent7 = Agent(
      model="nonexistent-model",
      name="agent7",
      description="Example",
      instruction=f"""
    Example
    """,
  )

  # example sequence
  sequence_1 = SequentialAgent(
      name="ExampleSequence",
      sub_agents=[agent1, agent2],
  )

  # example loop
  loop_1 = LoopAgent(
      name="ExampleLoop",
      sub_agents=[agent6, agent7],
      max_iterations=10,
  )

  # example parallel
  parallel_1 = ParallelAgent(
      name="ExampleParallel",
      sub_agents=[agent3, agent4, agent5],
  )

  # sequence for orchestrating everything together
  root_agent = SequentialAgent(
      name="root_agent",
      sub_agents=[sequence_1, loop_1, parallel_1],
      description="Example",
  )

  mermaid_src, png_display_bytes = build_mermaid(root_agent)

  assert isinstance(mermaid_src, str)
  assert mermaid_src.startswith("flowchart LR")

  assert isinstance(png_display_bytes, bytes)
  assert png_display_bytes.startswith(b"\x89PNG\r\n\x1a\n")
