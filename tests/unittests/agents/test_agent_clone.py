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

"""Testings for the clone functionality of agents."""

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.loop_agent import LoopAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent
import pytest


def test_llm_agent_clone():
  """Test cloning an LLM agent."""
  # Create an LLM agent
  original = LlmAgent(
      name="llm_agent",
      description="An LLM agent",
      instruction="You are a helpful assistant.",
  )

  # Clone it with name update
  cloned = original.clone(update={"name": "cloned_llm_agent"})

  # Verify the clone
  assert cloned.name == "cloned_llm_agent"
  assert cloned.description == "An LLM agent"
  assert cloned.instruction == "You are a helpful assistant."
  assert cloned.parent_agent is None
  assert len(cloned.sub_agents) == 0
  assert isinstance(cloned, LlmAgent)

  # Verify the original is unchanged
  assert original.name == "llm_agent"
  assert original.instruction == "You are a helpful assistant."


def test_agent_with_sub_agents():
  """Test cloning an agent that has sub-agents."""
  # Create sub-agents
  sub_agent1 = LlmAgent(name="sub_agent1", description="First sub-agent")
  sub_agent2 = LlmAgent(name="sub_agent2", description="Second sub-agent")

  # Create a parent agent with sub-agents
  original = SequentialAgent(
      name="parent_agent",
      description="Parent agent with sub-agents",
      sub_agents=[sub_agent1, sub_agent2],
  )

  # Clone it with name update
  cloned = original.clone(update={"name": "cloned_parent"})

  # Verify the clone has sub-agents (deep copy behavior)
  assert cloned.name == "cloned_parent"
  assert cloned.description == "Parent agent with sub-agents"
  assert cloned.parent_agent is None
  assert len(cloned.sub_agents) == 2

  # Sub-agents should be cloned with their original names
  assert cloned.sub_agents[0].name == "sub_agent1"
  assert cloned.sub_agents[1].name == "sub_agent2"

  # Sub-agents should have the cloned agent as their parent
  assert cloned.sub_agents[0].parent_agent == cloned
  assert cloned.sub_agents[1].parent_agent == cloned

  # Sub-agents should be different objects from the original
  assert cloned.sub_agents[0] is not original.sub_agents[0]
  assert cloned.sub_agents[1] is not original.sub_agents[1]

  # Verify the original still has sub-agents
  assert original.name == "parent_agent"
  assert len(original.sub_agents) == 2
  assert original.sub_agents[0].name == "sub_agent1"
  assert original.sub_agents[1].name == "sub_agent2"
  assert original.sub_agents[0].parent_agent == original
  assert original.sub_agents[1].parent_agent == original


def test_three_level_nested_agent():
  """Test cloning a three-level nested agent to verify recursive cloning logic."""
  # Create third-level agents (leaf nodes)
  leaf_agent1 = LlmAgent(name="leaf1", description="First leaf agent")
  leaf_agent2 = LlmAgent(name="leaf2", description="Second leaf agent")

  # Create second-level agents
  middle_agent1 = SequentialAgent(
      name="middle1", description="First middle agent", sub_agents=[leaf_agent1]
  )
  middle_agent2 = ParallelAgent(
      name="middle2",
      description="Second middle agent",
      sub_agents=[leaf_agent2],
  )

  # Create top-level agent
  root_agent = LoopAgent(
      name="root_agent",
      description="Root agent with three levels",
      max_iterations=5,
      sub_agents=[middle_agent1, middle_agent2],
  )

  # Clone the root agent
  cloned_root = root_agent.clone(update={"name": "cloned_root"})

  # Verify root level
  assert cloned_root.name == "cloned_root"
  assert cloned_root.description == "Root agent with three levels"
  assert cloned_root.max_iterations == 5
  assert cloned_root.parent_agent is None
  assert len(cloned_root.sub_agents) == 2
  assert isinstance(cloned_root, LoopAgent)

  # Verify middle level
  cloned_middle1 = cloned_root.sub_agents[0]
  cloned_middle2 = cloned_root.sub_agents[1]

  assert cloned_middle1.name == "middle1"
  assert cloned_middle1.description == "First middle agent"
  assert cloned_middle1.parent_agent == cloned_root
  assert len(cloned_middle1.sub_agents) == 1
  assert isinstance(cloned_middle1, SequentialAgent)

  assert cloned_middle2.name == "middle2"
  assert cloned_middle2.description == "Second middle agent"
  assert cloned_middle2.parent_agent == cloned_root
  assert len(cloned_middle2.sub_agents) == 1
  assert isinstance(cloned_middle2, ParallelAgent)

  # Verify leaf level
  cloned_leaf1 = cloned_middle1.sub_agents[0]
  cloned_leaf2 = cloned_middle2.sub_agents[0]

  assert cloned_leaf1.name == "leaf1"
  assert cloned_leaf1.description == "First leaf agent"
  assert cloned_leaf1.parent_agent == cloned_middle1
  assert len(cloned_leaf1.sub_agents) == 0
  assert isinstance(cloned_leaf1, LlmAgent)

  assert cloned_leaf2.name == "leaf2"
  assert cloned_leaf2.description == "Second leaf agent"
  assert cloned_leaf2.parent_agent == cloned_middle2
  assert len(cloned_leaf2.sub_agents) == 0
  assert isinstance(cloned_leaf2, LlmAgent)

  # Verify all objects are different from originals
  assert cloned_root is not root_agent
  assert cloned_middle1 is not middle_agent1
  assert cloned_middle2 is not middle_agent2
  assert cloned_leaf1 is not leaf_agent1
  assert cloned_leaf2 is not leaf_agent2

  # Verify original structure is unchanged
  assert root_agent.name == "root_agent"
  assert root_agent.sub_agents[0].name == "middle1"
  assert root_agent.sub_agents[1].name == "middle2"
  assert root_agent.sub_agents[0].sub_agents[0].name == "leaf1"
  assert root_agent.sub_agents[1].sub_agents[0].name == "leaf2"


def test_multiple_clones():
  """Test creating multiple clones with automatic naming."""
  # Create multiple agents and clone each one
  original = LlmAgent(
      name="original_agent", description="Agent for multiple cloning"
  )

  # Test multiple clones from the same original
  clone1 = original.clone(update={"name": "clone1"})
  clone2 = original.clone(update={"name": "clone2"})

  assert clone1.name == "clone1"
  assert clone2.name == "clone2"
  assert clone1 is not clone2


def test_clone_with_complex_configuration():
  """Test cloning an agent with complex configuration."""
  # Create an LLM agent with various configurations
  original = LlmAgent(
      name="complex_agent",
      description="A complex agent with many settings",
      instruction="You are a specialized assistant.",
      global_instruction="Always be helpful and accurate.",
      disallow_transfer_to_parent=True,
      disallow_transfer_to_peers=True,
      include_contents="none",
  )

  # Clone it with name update
  cloned = original.clone(update={"name": "complex_clone"})

  # Verify all configurations are preserved
  assert cloned.name == "complex_clone"
  assert cloned.description == "A complex agent with many settings"
  assert cloned.instruction == "You are a specialized assistant."
  assert cloned.global_instruction == "Always be helpful and accurate."
  assert cloned.disallow_transfer_to_parent is True
  assert cloned.disallow_transfer_to_peers is True
  assert cloned.include_contents == "none"

  # Verify parent and sub-agents are set
  assert cloned.parent_agent is None
  assert len(cloned.sub_agents) == 0


def test_clone_without_updates():
  """Test cloning without providing updates (should use original values)."""
  original = LlmAgent(name="test_agent", description="Test agent")

  cloned = original.clone()

  assert cloned.name == "test_agent"
  assert cloned.description == "Test agent"


def test_clone_with_multiple_updates():
  """Test cloning with multiple field updates."""
  original = LlmAgent(
      name="original_agent",
      description="Original description",
      instruction="Original instruction",
  )

  cloned = original.clone(
      update={
          "name": "updated_agent",
          "description": "Updated description",
          "instruction": "Updated instruction",
      }
  )

  assert cloned.name == "updated_agent"
  assert cloned.description == "Updated description"
  assert cloned.instruction == "Updated instruction"


def test_clone_with_sub_agents_deep_copy():
  """Test cloning with deep copy of sub-agents."""
  # Create an agent with sub-agents
  sub_agent = LlmAgent(name="sub_agent", description="Sub agent")
  original = LlmAgent(
      name="root_agent",
      description="Root agent",
      sub_agents=[sub_agent],
  )

  # Clone with deep copy
  cloned = original.clone(update={"name": "cloned_root_agent"})
  assert cloned.name == "cloned_root_agent"
  assert cloned.sub_agents[0].name == "sub_agent"
  assert cloned.sub_agents[0].parent_agent == cloned
  assert cloned.sub_agents[0] is not original.sub_agents[0]


def test_clone_invalid_field():
  """Test that cloning with invalid fields raises an error."""
  original = LlmAgent(name="test_agent", description="Test agent")

  with pytest.raises(ValueError, match="Cannot update non-existent fields"):
    original.clone(update={"invalid_field": "value"})


def test_clone_parent_agent_field():
  """Test that cloning with parent_agent field raises an error."""
  original = LlmAgent(name="test_agent", description="Test agent")

  with pytest.raises(
      ValueError, match="Cannot update `parent_agent` field in clone"
  ):
    original.clone(update={"parent_agent": None})


def test_clone_preserves_agent_type():
  """Test that cloning preserves the specific agent type."""
  # Test LlmAgent
  llm_original = LlmAgent(name="llm_test")
  llm_cloned = llm_original.clone()
  assert isinstance(llm_cloned, LlmAgent)

  # Test SequentialAgent
  seq_original = SequentialAgent(name="seq_test")
  seq_cloned = seq_original.clone()
  assert isinstance(seq_cloned, SequentialAgent)

  # Test ParallelAgent
  par_original = ParallelAgent(name="par_test")
  par_cloned = par_original.clone()
  assert isinstance(par_cloned, ParallelAgent)

  # Test LoopAgent
  loop_original = LoopAgent(name="loop_test")
  loop_cloned = loop_original.clone()
  assert isinstance(loop_cloned, LoopAgent)


def test_clone_with_agent_specific_fields():
  # Test LoopAgent
  loop_original = LoopAgent(name="loop_test")
  loop_cloned = loop_original.clone({"max_iterations": 10})
  assert isinstance(loop_cloned, LoopAgent)
  assert loop_cloned.max_iterations == 10


def test_clone_with_none_update():
  """Test cloning with explicit None update parameter."""
  original = LlmAgent(name="test_agent", description="Test agent")

  cloned = original.clone(update=None)

  assert cloned.name == "test_agent"
  assert cloned.description == "Test agent"
  assert cloned is not original


def test_clone_with_empty_update():
  """Test cloning with empty update dictionary."""
  original = LlmAgent(name="test_agent", description="Test agent")

  cloned = original.clone(update={})

  assert cloned.name == "test_agent"
  assert cloned.description == "Test agent"
  assert cloned is not original


def test_clone_with_sub_agents_update():
  """Test cloning with sub_agents provided in update."""
  # Create original sub-agents
  original_sub1 = LlmAgent(name="original_sub1", description="Original sub 1")
  original_sub2 = LlmAgent(name="original_sub2", description="Original sub 2")

  # Create new sub-agents for the update
  new_sub1 = LlmAgent(name="new_sub1", description="New sub 1")
  new_sub2 = LlmAgent(name="new_sub2", description="New sub 2")

  # Create original agent with sub-agents
  original = SequentialAgent(
      name="original_agent",
      description="Original agent",
      sub_agents=[original_sub1, original_sub2],
  )

  # Clone with sub_agents update
  cloned = original.clone(
      update={"name": "cloned_agent", "sub_agents": [new_sub1, new_sub2]}
  )

  # Verify the clone uses the new sub-agents
  assert cloned.name == "cloned_agent"
  assert len(cloned.sub_agents) == 2
  assert cloned.sub_agents[0].name == "new_sub1"
  assert cloned.sub_agents[1].name == "new_sub2"
  assert cloned.sub_agents[0].parent_agent == cloned
  assert cloned.sub_agents[1].parent_agent == cloned

  # Verify original is unchanged
  assert original.name == "original_agent"
  assert len(original.sub_agents) == 2
  assert original.sub_agents[0].name == "original_sub1"
  assert original.sub_agents[1].name == "original_sub2"


if __name__ == "__main__":
  # Run a specific test for debugging
  test_three_level_nested_agent()
