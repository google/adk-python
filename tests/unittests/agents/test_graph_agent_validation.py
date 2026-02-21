"""Test suite for GraphAgent validation features.

Tests:
- Duplicate node name validation
- Duplicate edge validation
- Auto-defaulting output_key for LlmAgent with output_schema
- Warning emissions for auto-defaulted output_key
"""

import logging

from google.adk.agents import LlmAgent
from google.adk.agents.graph import EdgeCondition
from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import GraphNode
from pydantic import BaseModel
import pytest


# Test schema for output_schema tests
class TestOutput(BaseModel):
  """Test output schema."""

  result: str


def test_add_node_duplicate_name_raises_error():
  """Adding node with duplicate name raises ValueError."""
  graph = GraphAgent(name="test_graph")
  agent1 = LlmAgent(name="agent1", model="gemini-2.0-flash")
  agent2 = LlmAgent(name="agent2", model="gemini-2.0-flash")

  # Add first node
  graph.add_node(GraphNode(name="duplicate", agent=agent1))

  # Adding second node with same name should raise
  with pytest.raises(ValueError, match="already exists"):
    graph.add_node(GraphNode(name="duplicate", agent=agent2))


def test_add_node_duplicate_name_convenience_api():
  """Duplicate node validation works with convenience API."""
  graph = GraphAgent(name="test_graph")
  agent1 = LlmAgent(name="agent1", model="gemini-2.0-flash")
  agent2 = LlmAgent(name="agent2", model="gemini-2.0-flash")

  # Add first node using convenience API
  graph.add_node("duplicate", agent=agent1)

  # Adding second node with same name should raise
  with pytest.raises(ValueError, match="already exists"):
    graph.add_node("duplicate", agent=agent2)


def test_add_edge_duplicate_raises_error():
  """Adding edge with duplicate source→target raises ValueError."""
  graph = GraphAgent(name="test_graph")

  # Add nodes
  graph.add_node(
      "start", agent=LlmAgent(name="start", model="gemini-2.0-flash")
  )
  graph.add_node("end", agent=LlmAgent(name="end", model="gemini-2.0-flash"))

  # Add first edge
  graph.add_edge("start", "end")

  # Adding second edge with same source→target should raise
  with pytest.raises(ValueError, match="already exists"):
    graph.add_edge("start", "end")


def test_add_edge_duplicate_with_conditions():
  """Duplicate edge validation works even with different conditions."""
  graph = GraphAgent(name="test_graph")

  # Add nodes
  graph.add_node(
      "start", agent=LlmAgent(name="start", model="gemini-2.0-flash")
  )
  graph.add_node("end", agent=LlmAgent(name="end", model="gemini-2.0-flash"))

  # Add first edge with condition
  graph.add_edge("start", "end", condition=lambda s: s.data.get("foo"))

  # Adding second edge to same target should raise even with different condition
  with pytest.raises(ValueError, match="already exists"):
    graph.add_edge("start", "end", condition=lambda s: s.data.get("bar"))


def test_output_key_auto_defaults_to_agent_name():
  """GraphNode auto-defaults output_key to agent.name when output_schema is set.

  model_copy() is used so the original agent is NOT mutated; the copy stored
  in node.agent has the defaulted output_key.
  """
  agent = LlmAgent(
      name="analyzer",
      model="gemini-2.0-flash",
      output_schema=TestOutput,
      # output_key NOT SET
  )

  # Before wrapping
  assert agent.output_key is None

  # After wrapping in GraphNode
  node = GraphNode(name="test_node", agent=agent)

  # node.agent is a copy with the auto-defaulted output_key
  assert node.agent.output_key == "analyzer"
  # Original agent is NOT mutated (model_copy creates an isolated copy)
  assert agent.output_key is None


def test_explicit_output_key_not_overridden():
  """Explicit output_key is not overridden by auto-defaulting."""
  agent = LlmAgent(
      name="analyzer",
      model="gemini-2.0-flash",
      output_schema=TestOutput,
      output_key="custom_key",  # Explicit
  )

  node = GraphNode(name="test_node", agent=agent)

  # Should keep explicit value
  assert agent.output_key == "custom_key"


def test_no_auto_default_without_output_schema():
  """output_key is not auto-defaulted if output_schema is not set."""
  agent = LlmAgent(
      name="analyzer",
      model="gemini-2.0-flash",
      # No output_schema
  )

  # Before wrapping
  assert agent.output_key is None

  # After wrapping in GraphNode
  node = GraphNode(name="test_node", agent=agent)

  # output_key should still be None (no auto-default)
  assert agent.output_key is None


def test_warning_for_auto_defaulted_output_key(caplog):
  """Warning emitted when output_key is auto-defaulted."""
  agent = LlmAgent(
      name="analyzer",
      model="gemini-2.0-flash",
      output_schema=TestOutput,
  )
  graph = GraphAgent(name="test_graph")

  with caplog.at_level(logging.WARNING):
    graph.add_node(GraphNode(name="test_node", agent=agent))

  # Verify warning about auto-defaulting
  assert any("auto-defaulted" in rec.message.lower() for rec in caplog.records)


def test_no_warning_for_explicit_output_key(caplog):
  """No warning emitted when output_key is explicitly set."""
  agent = LlmAgent(
      name="analyzer",
      model="gemini-2.0-flash",
      output_schema=TestOutput,
      output_key="custom_key",
  )
  graph = GraphAgent(name="test_graph")

  with caplog.at_level(logging.WARNING):
    graph.add_node(GraphNode(name="test_node", agent=agent))

  # No warning should be emitted
  assert not any(
      "auto-defaulted" in rec.message.lower() for rec in caplog.records
  )


def test_add_edge_duplicate_edge_condition_raises():
  """Duplicate EdgeCondition (Pattern 1) to same target raises ValueError."""
  graph = GraphAgent(name="test_graph")
  graph.add_node(
      "start", agent=LlmAgent(name="start", model="gemini-2.0-flash")
  )
  graph.add_node("end", agent=LlmAgent(name="end", model="gemini-2.0-flash"))

  edge = EdgeCondition(target_node="end", priority=10)
  graph.add_edge("start", edge)

  # Second EdgeCondition to the same target must raise
  with pytest.raises(ValueError, match="already exists"):
    graph.add_edge("start", EdgeCondition(target_node="end", priority=5))


def test_agent_name_collides_with_graph_name_raises():
  """Agent name matching GraphAgent name raises ValueError."""
  graph = GraphAgent(name="my_graph")
  agent = LlmAgent(name="my_graph", model="gemini-2.0-flash")

  with pytest.raises(ValueError, match="collides with GraphAgent name"):
    graph.add_node("node1", agent=agent)


def test_ast_rejects_dunder_attribute_access():
  """AST validator blocks dunder attribute access (sandbox escape prevention)."""
  from google.adk.agents.graph.graph_agent import _validate_condition_ast
  import ast

  # Safe attribute access should pass
  tree = ast.parse("data.get('x')", mode="eval")
  _validate_condition_ast(tree.body)

  # Dunder attribute access should be rejected
  tree = ast.parse("data.__class__", mode="eval")
  with pytest.raises(ValueError, match="Unsafe attribute access.*__class__"):
    _validate_condition_ast(tree.body)

  # Nested dunder chain should be rejected (outermost attr checked first)
  tree = ast.parse("data.__class__.__init__", mode="eval")
  with pytest.raises(ValueError, match="Unsafe attribute access.*__init__"):
    _validate_condition_ast(tree.body)

  # Single underscore prefix should also be rejected
  tree = ast.parse("data._private", mode="eval")
  with pytest.raises(ValueError, match="Unsafe attribute access.*_private"):
    _validate_condition_ast(tree.body)
