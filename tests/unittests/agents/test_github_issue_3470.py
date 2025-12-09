# Copyright 2025 Google LLC
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

"""Integration tests for GitHub issue #3470.

Tests two problematic architectures where reducers couldn't see outputs
from parallel agents:

1. Nested Parallel + Reduce:
   Sequential[Parallel[A,B,C], Reducer1] in parallel with
   Sequential[Parallel[D,E,F], Reducer2], followed by Reducer3

2. Simple Sequence of Parallels:
   Sequential[Parallel1[A,B,C], Parallel2[D,E,F], Parallel3[G,H,I]]
"""

from __future__ import annotations

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent
import pytest

from tests.unittests import testing_utils


def test_nested_parallel_reduce_architecture():
  """Test the nested parallel + reduce architecture from GitHub issue #3470.

  Architecture:
    Sequential1 = Parallel[A, B, C] -> Reducer1
    Sequential2 = Parallel[D, E, F] -> Reducer2
    Final = Parallel[Sequential1, Sequential2] -> Reducer3

  The bug was that:
  - Reducer1 couldn't see outputs from A, B, C
  - Reducer2 couldn't see outputs from D, E, F
  - Reducer3 couldn't see outputs from Reducer1 and Reducer2

  With BranchContext fix:
  - A, B, C get tokens {1}, {2}, {3}
  - Parallel1 joins to {1,2,3}
  - Reducer1 gets {1,2,3} and can see all events from {1}, {2}, {3}
  - Same for D, E, F in Sequential2
  - Final reducer can see all previous events
  """
  # Group 1 agents
  agent_a = LlmAgent(
      name="Alice",
      description="Agent A",
      instruction="Say: I am Alice",
      model=testing_utils.MockModel.create(responses=["I am Alice"]),
  )
  agent_b = LlmAgent(
      name="Bob",
      description="Agent B",
      instruction="Say: I am Bob",
      model=testing_utils.MockModel.create(responses=["I am Bob"]),
  )
  agent_c = LlmAgent(
      name="Charlie",
      description="Agent C",
      instruction="Say: I am Charlie",
      model=testing_utils.MockModel.create(responses=["I am Charlie"]),
  )

  # Group 2 agents
  agent_d = LlmAgent(
      name="David",
      description="Agent D",
      instruction="Say: I am David",
      model=testing_utils.MockModel.create(responses=["I am David"]),
  )
  agent_e = LlmAgent(
      name="Eve",
      description="Agent E",
      instruction="Say: I am Eve",
      model=testing_utils.MockModel.create(responses=["I am Eve"]),
  )
  agent_f = LlmAgent(
      name="Frank",
      description="Agent F",
      instruction="Say: I am Frank",
      model=testing_utils.MockModel.create(responses=["I am Frank"]),
  )

  # Parallel groups
  parallel_abc = ParallelAgent(
      name="ABC_Parallel",
      description="Parallel group ABC",
      sub_agents=[agent_a, agent_b, agent_c],
  )

  parallel_def = ParallelAgent(
      name="DEF_Parallel",
      description="Parallel group DEF",
      sub_agents=[agent_d, agent_e, agent_f],
  )

  # Reducers with models that track requests
  reducer1_model = testing_utils.MockModel.create(responses=["Summary of ABC"])
  reducer1 = LlmAgent(
      name="Reducer1",
      description="Reducer for ABC",
      instruction="Summarize responses from A, B, and C",
      model=reducer1_model,
  )

  reducer2_model = testing_utils.MockModel.create(responses=["Summary of DEF"])
  reducer2 = LlmAgent(
      name="Reducer2",
      description="Reducer for DEF",
      instruction="Summarize responses from D, E, and F",
      model=reducer2_model,
  )

  # Sequential groups (Parallel -> Reducer)
  sequential1 = SequentialAgent(
      name="Group1_Sequential",
      description="Sequential ABC -> Reducer1",
      sub_agents=[parallel_abc, reducer1],
  )

  sequential2 = SequentialAgent(
      name="Group2_Sequential",
      description="Sequential DEF -> Reducer2",
      sub_agents=[parallel_def, reducer2],
  )

  # Run both sequential groups in parallel
  final_parallel = ParallelAgent(
      name="Final_Parallel",
      description="Run both groups in parallel",
      sub_agents=[sequential1, sequential2],
  )

  # Final reducer with model that tracks requests
  final_reducer_model = testing_utils.MockModel.create(
      responses=["Final summary"]
  )
  final_reducer = LlmAgent(
      name="Final_Reducer",
      description="Final reducer",
      instruction="Summarize all outputs",
      model=final_reducer_model,
  )

  # Top-level sequential
  root_agent = SequentialAgent(
      name="Root_Sequential",
      description="Root sequential agent",
      sub_agents=[final_parallel, final_reducer],
  )

  # Run the agent
  runner = testing_utils.InMemoryRunner(root_agent=root_agent)
  runner.run("Start")
  session = runner.session

  # Debug: print all events and their branches
  print("\n=== Token Distribution (Nested Parallel) ===")
  for event in session.events:
    if event.author and event.branch:
      print(f"{event.author:15} | tokens={event.branch.tokens}")

  # Verify all agents ran
  agent_names = {event.author for event in session.events if event.author}
  expected_agents = {
      "Alice",
      "Bob",
      "Charlie",
      "David",
      "Eve",
      "Frank",
      "Reducer1",
      "Reducer2",
      "Final_Reducer",
  }
  assert expected_agents.issubset(
      agent_names
  ), f"Missing agents: {expected_agents - agent_names}"

  # Verify event visibility using branch tokens
  # Get reducer events
  reducer1_events = [e for e in session.events if e.author == "Reducer1"]
  reducer2_events = [e for e in session.events if e.author == "Reducer2"]
  final_reducer_events = [
      e for e in session.events if e.author == "Final_Reducer"
  ]

  assert len(reducer1_events) > 0, "Reducer1 should have events"
  assert len(reducer2_events) > 0, "Reducer2 should have events"
  assert len(final_reducer_events) > 0, "Final_Reducer should have events"

  # Check that reducers can see their parallel group outputs
  # Reducer1 should see A, B, C
  abc_events = [
      e
      for e in session.events
      if e.author in ["Alice", "Bob", "Charlie"] and e.branch
  ]
  for abc_event in abc_events:
    for reducer1_event in reducer1_events:
      if reducer1_event.branch:
        # Reducer1's tokens should be a superset of ABC tokens
        assert reducer1_event.branch.can_see(abc_event.branch), (
            f"Reducer1 (tokens={reducer1_event.branch.tokens}) should see"
            f" {abc_event.author} (tokens={abc_event.branch.tokens})"
        )

  # Reducer2 should see D, E, F
  def_events = [
      e
      for e in session.events
      if e.author in ["David", "Eve", "Frank"] and e.branch
  ]
  for def_event in def_events:
    for reducer2_event in reducer2_events:
      if reducer2_event.branch:
        # Reducer2's tokens should be a superset of DEF tokens
        assert reducer2_event.branch.can_see(def_event.branch), (
            f"Reducer2 (tokens={reducer2_event.branch.tokens}) should see"
            f" {def_event.author} (tokens={def_event.branch.tokens})"
        )

  # Final reducer should see all reducers
  all_reducer_events = reducer1_events + reducer2_events
  for reducer_event in all_reducer_events:
    if reducer_event.branch:
      for final_event in final_reducer_events:
        if final_event.branch:
          assert final_event.branch.can_see(reducer_event.branch), (
              f"Final_Reducer (tokens={final_event.branch.tokens}) should see"
              f" {reducer_event.author} (tokens={reducer_event.branch.tokens})"
          )

  # Verify LLM request contents - the actual text sent to the model
  # This is the critical test: does the reducer actually receive the parallel agents' outputs?

  # Helper to extract text from simplified contents
  def extract_text(contents):
    """Extract all text from simplified contents."""
    texts = []
    for role, content in contents:
      if isinstance(content, str):
        texts.append(content)
      elif isinstance(content, list):
        for part in content:
          if hasattr(part, "text") and part.text:
            texts.append(part.text)
      elif hasattr(content, "text") and content.text:
        texts.append(content.text)
    return " ".join(texts)

  # Reducer1 should receive outputs from A, B, C in its LLM request
  assert (
      len(reducer1_model.requests) > 0
  ), "Reducer1 should have made LLM requests"
  reducer1_contents = testing_utils.simplify_contents(
      reducer1_model.requests[0].contents
  )
  reducer1_text = extract_text(reducer1_contents)

  # Check that A, B, C outputs are in the context
  assert "Alice" in reducer1_text or "I am Alice" in reducer1_text, (
      "Reducer1 should see Alice's output in LLM request. Got:"
      f" {reducer1_text[:200]}"
  )
  assert "Bob" in reducer1_text or "I am Bob" in reducer1_text, (
      "Reducer1 should see Bob's output in LLM request. Got:"
      f" {reducer1_text[:200]}"
  )
  assert "Charlie" in reducer1_text or "I am Charlie" in reducer1_text, (
      "Reducer1 should see Charlie's output in LLM request. Got:"
      f" {reducer1_text[:200]}"
  )

  # Reducer2 should receive outputs from D, E, F in its LLM request
  assert (
      len(reducer2_model.requests) > 0
  ), "Reducer2 should have made LLM requests"
  reducer2_contents = testing_utils.simplify_contents(
      reducer2_model.requests[0].contents
  )
  reducer2_text = extract_text(reducer2_contents)

  assert "David" in reducer2_text or "I am David" in reducer2_text, (
      "Reducer2 should see David's output in LLM request. Got:"
      f" {reducer2_text[:200]}"
  )
  assert "Eve" in reducer2_text or "I am Eve" in reducer2_text, (
      "Reducer2 should see Eve's output in LLM request. Got:"
      f" {reducer2_text[:200]}"
  )
  assert "Frank" in reducer2_text or "I am Frank" in reducer2_text, (
      "Reducer2 should see Frank's output in LLM request. Got:"
      f" {reducer2_text[:200]}"
  )

  # Final reducer should receive outputs from both reducers AND nested agents
  assert (
      len(final_reducer_model.requests) > 0
  ), "Final_Reducer should have made LLM requests"
  final_contents = testing_utils.simplify_contents(
      final_reducer_model.requests[0].contents
  )
  final_text = extract_text(final_contents)

  # Should see the reducer summaries
  assert "Summary of ABC" in final_text, (
      "Final_Reducer should see Reducer1's summary in LLM request. Got:"
      f" {final_text[:200]}"
  )
  assert "Summary of DEF" in final_text, (
      "Final_Reducer should see Reducer2's summary in LLM request. Got:"
      f" {final_text[:200]}"
  )

  # Should also see the original agent outputs (nested visibility)
  assert "Alice" in final_text or "I am Alice" in final_text, (
      "Final_Reducer should see Alice's output in LLM request. Got:"
      f" {final_text[:200]}"
  )
  assert "David" in final_text or "I am David" in final_text, (
      "Final_Reducer should see David's output in LLM request. Got:"
      f" {final_text[:200]}"
  )


def test_sequence_of_parallel_agents():
  """Test sequence of parallel agents from GitHub issue #3470.

  Architecture:
    Sequential[Parallel1[A,B,C], Parallel2[D,E,F], Parallel3[G,H,I]]

  The bug was that agents in Parallel2 and Parallel3 couldn't see outputs
  from previous parallel groups.

  With BranchContext fix:
  - Parallel1: A={1}, B={2}, C={3}, joins to {1,2,3}
  - Parallel2 forks from {1,2,3}: D={1,2,3,4}, E={1,2,3,5}, F={1,2,3,6}
  - D, E, F can all see A, B, C because {1}⊆{1,2,3,4}
  - Parallel3 forks from joined tokens and can see all previous events
  """
  # Group 1
  agent_a_model = testing_utils.MockModel.create(responses=["I am Alice"])
  agent_a = LlmAgent(
      name="Alice",
      description="Agent A",
      instruction="Say: I am Alice",
      model=agent_a_model,
  )
  agent_b = LlmAgent(
      name="Bob",
      description="Agent B",
      instruction="Say: I am Bob",
      model=testing_utils.MockModel.create(responses=["I am Bob"]),
  )
  agent_c = LlmAgent(
      name="Charlie",
      description="Agent C",
      instruction="Say: I am Charlie",
      model=testing_utils.MockModel.create(responses=["I am Charlie"]),
  )

  # Group 2 - track David's model to check it sees Group 1
  agent_d_model = testing_utils.MockModel.create(responses=["I am David"])
  agent_d = LlmAgent(
      name="David",
      description="Agent D",
      instruction="Say: I am David",
      model=agent_d_model,
  )
  agent_e = LlmAgent(
      name="Eve",
      description="Agent E",
      instruction="Say: I am Eve",
      model=testing_utils.MockModel.create(responses=["I am Eve"]),
  )
  agent_f = LlmAgent(
      name="Frank",
      description="Agent F",
      instruction="Say: I am Frank",
      model=testing_utils.MockModel.create(responses=["I am Frank"]),
  )

  # Group 3 - track Grace's model to check it sees Groups 1 and 2
  agent_g_model = testing_utils.MockModel.create(responses=["I am Grace"])
  agent_g = LlmAgent(
      name="Grace",
      description="Agent G",
      instruction="Say: I am Grace",
      model=agent_g_model,
  )
  agent_h = LlmAgent(
      name="Henry",
      description="Agent H",
      instruction="Say: I am Henry",
      model=testing_utils.MockModel.create(responses=["I am Henry"]),
  )
  agent_i = LlmAgent(
      name="Iris",
      description="Agent I",
      instruction="Say: I am Iris",
      model=testing_utils.MockModel.create(responses=["I am Iris"]),
  )

  # Create parallel groups
  parallel1 = ParallelAgent(
      name="Parallel1",
      description="First parallel group",
      sub_agents=[agent_a, agent_b, agent_c],
  )

  parallel2 = ParallelAgent(
      name="Parallel2",
      description="Second parallel group",
      sub_agents=[agent_d, agent_e, agent_f],
  )

  parallel3 = ParallelAgent(
      name="Parallel3",
      description="Third parallel group",
      sub_agents=[agent_g, agent_h, agent_i],
  )

  # Create sequential agent
  root_agent = SequentialAgent(
      name="Root_Sequential",
      description="Sequential of parallels",
      sub_agents=[parallel1, parallel2, parallel3],
  )

  # Run the agent
  runner = testing_utils.InMemoryRunner(root_agent=root_agent)
  runner.run("Start")
  session = runner.session

  # Verify all agents ran
  agent_names = {event.author for event in session.events if event.author}
  expected_agents = {
      "Alice",
      "Bob",
      "Charlie",
      "David",
      "Eve",
      "Frank",
      "Grace",
      "Henry",
      "Iris",
  }
  assert expected_agents.issubset(
      agent_names
  ), f"Missing agents: {expected_agents - agent_names}"

  # Get events by agent group
  parallel1_events = [
      e
      for e in session.events
      if e.author in ["Alice", "Bob", "Charlie"] and e.branch
  ]
  parallel2_events = [
      e
      for e in session.events
      if e.author in ["David", "Eve", "Frank"] and e.branch
  ]
  parallel3_events = [
      e
      for e in session.events
      if e.author in ["Grace", "Henry", "Iris"] and e.branch
  ]

  assert len(parallel1_events) > 0, "Parallel1 should have events"
  assert len(parallel2_events) > 0, "Parallel2 should have events"
  assert len(parallel3_events) > 0, "Parallel3 should have events"

  # Verify visibility: Parallel2 should see Parallel1
  for p1_event in parallel1_events:
    for p2_event in parallel2_events:
      # Parallel2 tokens should be superset of Parallel1 tokens
      assert p2_event.branch.can_see(p1_event.branch), (
          f"{p2_event.author} (tokens={p2_event.branch.tokens}) should see"
          f" {p1_event.author} (tokens={p1_event.branch.tokens})"
      )

  # Verify visibility: Parallel3 should see Parallel1 and Parallel2
  for p1_event in parallel1_events:
    for p3_event in parallel3_events:
      assert p3_event.branch.can_see(p1_event.branch), (
          f"{p3_event.author} (tokens={p3_event.branch.tokens}) should see"
          f" {p1_event.author} (tokens={p1_event.branch.tokens})"
      )

  for p2_event in parallel2_events:
    for p3_event in parallel3_events:
      assert p3_event.branch.can_see(p2_event.branch), (
          f"{p3_event.author} (tokens={p3_event.branch.tokens}) should see"
          f" {p2_event.author} (tokens={p2_event.branch.tokens})"
      )

  # Print token sets for verification
  print("\n=== Token Distribution ===")
  for event in session.events:
    if event.author and event.branch:
      print(f"{event.author:15} | tokens={event.branch.tokens}")

  # Verify LLM request contents - the actual text sent to the models
  # This is the critical test from the GitHub issue: does each parallel group
  # actually receive the previous groups' outputs in their LLM context?

  # Helper to extract text from simplified contents
  def extract_text(contents):
    """Extract all text from simplified contents."""
    texts = []
    for role, content in contents:
      if isinstance(content, str):
        texts.append(content)
      elif isinstance(content, list):
        for part in content:
          if hasattr(part, "text") and part.text:
            texts.append(part.text)
      elif hasattr(content, "text") and content.text:
        texts.append(content.text)
    return " ".join(texts)

  # David (in Parallel2) should see Alice, Bob, Charlie from Parallel1
  assert len(agent_d_model.requests) > 0, "David should have made LLM requests"
  david_contents = testing_utils.simplify_contents(
      agent_d_model.requests[0].contents
  )
  david_text = extract_text(david_contents)

  assert "Alice" in david_text or "I am Alice" in david_text, (
      "David should see Alice's output in LLM request (Parallel2 seeing"
      f" Parallel1). Got: {david_text[:200]}"
  )
  assert "Bob" in david_text or "I am Bob" in david_text, (
      "David should see Bob's output in LLM request (Parallel2 seeing"
      f" Parallel1). Got: {david_text[:200]}"
  )
  assert "Charlie" in david_text or "I am Charlie" in david_text, (
      "David should see Charlie's output in LLM request (Parallel2 seeing"
      f" Parallel1). Got: {david_text[:200]}"
  )

  # Grace (in Parallel3) should see all previous agents
  assert len(agent_g_model.requests) > 0, "Grace should have made LLM requests"
  grace_contents = testing_utils.simplify_contents(
      agent_g_model.requests[0].contents
  )
  grace_text = extract_text(grace_contents)

  # Should see Parallel1 agents
  assert "Alice" in grace_text or "I am Alice" in grace_text, (
      "Grace should see Alice's output in LLM request (Parallel3 seeing"
      f" Parallel1). Got: {grace_text[:200]}"
  )
  assert "Bob" in grace_text or "I am Bob" in grace_text, (
      "Grace should see Bob's output in LLM request (Parallel3 seeing"
      f" Parallel1). Got: {grace_text[:200]}"
  )

  # Should see Parallel2 agents
  assert "David" in grace_text or "I am David" in grace_text, (
      "Grace should see David's output in LLM request (Parallel3 seeing"
      f" Parallel2). Got: {grace_text[:200]}"
  )
  assert "Eve" in grace_text or "I am Eve" in grace_text, (
      "Grace should see Eve's output in LLM request (Parallel3 seeing"
      f" Parallel2). Got: {grace_text[:200]}"
  )
