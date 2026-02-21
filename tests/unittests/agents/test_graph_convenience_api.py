"""Tests for GraphAgent convenience API methods.

Tests the convenience methods for add_node() and add_edge() that provide
simpler syntax alternatives to the explicit GraphNode/EdgeCondition patterns.
"""

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import GraphNode
from google.adk.agents.graph import GraphState
from google.adk.agents.graph import StateReducer
from google.adk.agents.graph.graph_edge import EdgeCondition
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.genai import types
import pytest


class SimpleAgent(BaseAgent):
  """Simple test agent."""

  async def _run_async_impl(self, ctx: InvocationContext):
    yield Event(
        author=self.name,
        content=types.Content(parts=[types.Part(text=f"{self.name} output")]),
    )


def simple_function(state: GraphState, ctx: InvocationContext) -> str:
  """Simple test function."""
  return "function output"


class TestAddNodeConvenience:
  """Test add_node() convenience patterns."""

  def test_add_node_with_graphnode(self):
    """Test traditional GraphNode pattern still works."""
    graph = GraphAgent(name="test")
    agent = SimpleAgent(name="test_agent")

    graph.add_node(GraphNode(name="node1", agent=agent))

    assert "node1" in graph.nodes
    assert graph.nodes["node1"].agent == agent

  def test_add_node_convenience_with_agent(self):
    """Test convenience pattern: add_node(name, agent=...)"""
    graph = GraphAgent(name="test")
    agent = SimpleAgent(name="test_agent")

    graph.add_node("node1", agent=agent)

    assert "node1" in graph.nodes
    assert graph.nodes["node1"].agent == agent

  def test_add_node_convenience_with_function(self):
    """Test convenience pattern: add_node(name, function=...)"""
    graph = GraphAgent(name="test")

    graph.add_node("node1", function=simple_function)

    assert "node1" in graph.nodes
    assert graph.nodes["node1"].function == simple_function

  def test_add_node_convenience_with_kwargs(self):
    """Test convenience pattern with additional kwargs."""
    graph = GraphAgent(name="test")
    agent = SimpleAgent(name="test_agent")

    graph.add_node(
        "node1",
        agent=agent,
        reducer=StateReducer.APPEND,
    )

    assert "node1" in graph.nodes
    assert graph.nodes["node1"].agent == agent
    assert graph.nodes["node1"].reducer == StateReducer.APPEND

  def test_add_node_error_graphnode_with_kwargs(self):
    """Test error when passing GraphNode with kwargs."""
    graph = GraphAgent(name="test")
    agent = SimpleAgent(name="test_agent")
    node = GraphNode(name="node1", agent=agent)

    with pytest.raises(ValueError, match="When passing a GraphNode"):
      graph.add_node(node, agent=agent)

  def test_add_node_error_string_without_agent_or_function(self):
    """Test error when passing string name without agent or function."""
    graph = GraphAgent(name="test")

    with pytest.raises(ValueError, match="must specify agent or function"):
      graph.add_node("node1")

  def test_add_node_error_both_agent_and_function(self):
    """Test error when passing both agent and function."""
    graph = GraphAgent(name="test")
    agent = SimpleAgent(name="test_agent")

    with pytest.raises(ValueError, match="Cannot specify both"):
      graph.add_node("node1", agent=agent, function=simple_function)

  def test_add_node_error_invalid_type(self):
    """Test error when passing invalid node type."""
    graph = GraphAgent(name="test")

    with pytest.raises(TypeError, match="node must be GraphNode or str"):
      graph.add_node(123)  # Invalid type

  def test_add_node_chaining(self):
    """Test that add_node returns self for chaining."""
    graph = GraphAgent(name="test")
    agent = SimpleAgent(name="test_agent")

    result = graph.add_node("node1", agent=agent).add_node("node2", agent=agent)

    assert result is graph
    assert "node1" in graph.nodes
    assert "node2" in graph.nodes


class TestAddEdgeConvenience:
  """Test add_edge() convenience patterns."""

  def test_add_edge_simple(self):
    """Test simple unconditional edge."""
    graph = GraphAgent(name="test")
    agent = SimpleAgent(name="test_agent")

    graph.add_node("node1", agent=agent)
    graph.add_node("node2", agent=agent)
    graph.add_edge("node1", "node2")

    # Edge should be added to node1
    assert len(graph.nodes["node1"].edges) == 1
    assert graph.nodes["node1"].edges[0].target_node == "node2"

  def test_add_edge_with_condition(self):
    """Test conditional edge."""
    graph = GraphAgent(name="test")
    agent = SimpleAgent(name="test_agent")

    graph.add_node("node1", agent=agent)
    graph.add_node("node2", agent=agent)

    condition = lambda s: s.data.get("valid", False)
    graph.add_edge("node1", "node2", condition=condition)

    assert len(graph.nodes["node1"].edges) == 1
    assert graph.nodes["node1"].edges[0].condition == condition

  def test_add_edge_with_priority(self):
    """Test priority-based edge."""
    graph = GraphAgent(name="test")
    agent = SimpleAgent(name="test_agent")

    graph.add_node("node1", agent=agent)
    graph.add_node("node2", agent=agent)
    graph.add_node("node3", agent=agent)

    # Add edges with priority
    graph.add_edge(
        "node1",
        "node2",
        condition=lambda s: s.data.get("score", 0) > 0.5,
        priority=10,
    )
    graph.add_edge("node1", "node3", priority=0)  # Fallback

    # Should create EdgeCondition objects
    assert hasattr(graph.nodes["node1"], "edges")
    assert len(graph.nodes["node1"].edges) == 2
    assert graph.nodes["node1"].edges[0].priority == 10
    assert graph.nodes["node1"].edges[1].priority == 0

  def test_add_edge_with_weight(self):
    """Test weighted random edge."""
    graph = GraphAgent(name="test")
    agent = SimpleAgent(name="test_agent")

    graph.add_node("node1", agent=agent)
    graph.add_node("node2", agent=agent)

    graph.add_edge(
        "node1", "node2", condition=lambda s: True, priority=1, weight=0.5
    )

    assert len(graph.nodes["node1"].edges) == 1
    assert graph.nodes["node1"].edges[0].weight == 0.5

  def test_add_edge_error_source_not_found(self):
    """Test error when source node not found."""
    graph = GraphAgent(name="test")
    agent = SimpleAgent(name="test_agent")

    graph.add_node("node2", agent=agent)

    with pytest.raises(ValueError, match="Source node node1 not found"):
      graph.add_edge("node1", "node2")

  def test_add_edge_error_target_not_found(self):
    """Test error when target node not found."""
    graph = GraphAgent(name="test")
    agent = SimpleAgent(name="test_agent")

    graph.add_node("node1", agent=agent)

    with pytest.raises(ValueError, match="Target node node2 not found"):
      graph.add_edge("node1", "node2")

  def test_add_edge_chaining(self):
    """Test that add_edge returns self for chaining."""
    graph = GraphAgent(name="test")
    agent = SimpleAgent(name="test_agent")

    graph.add_node("node1", agent=agent)
    graph.add_node("node2", agent=agent)
    graph.add_node("node3", agent=agent)

    result = graph.add_edge("node1", "node2").add_edge("node2", "node3")

    assert result is graph

  def test_add_edge_mixed_simple_and_priority(self):
    """Test mixing simple edges and priority edges on same node."""
    graph = GraphAgent(name="test")
    agent = SimpleAgent(name="test_agent")

    graph.add_node("node1", agent=agent)
    graph.add_node("node2", agent=agent)
    graph.add_node("node3", agent=agent)

    # Add simple edge first
    graph.add_edge("node1", "node2")

    # Add priority edge
    graph.add_edge("node1", "node3", priority=10)

    # Both edges should be in edges list
    assert len(graph.nodes["node1"].edges) == 2
    # First is simple edge (added via add_edge with no priority)
    assert graph.nodes["node1"].edges[0].target_node == "node2"
    # Second is priority edge
    assert graph.nodes["node1"].edges[1].target_node == "node3"
    assert graph.nodes["node1"].edges[1].priority == 10


class TestConvenienceAPIIntegration:
  """Integration tests using both convenience patterns together."""

  def test_full_graph_with_convenience_api(self):
    """Test building complete graph using convenience API."""
    graph = GraphAgent(name="validation_pipeline")

    # Create agents
    validator = SimpleAgent(name="validator")
    processor = SimpleAgent(name="processor")
    error_handler = SimpleAgent(name="error_handler")

    # Build graph using convenience API
    (
        graph.add_node("validate", agent=validator)
        .add_node("process", agent=processor)
        .add_node("error", agent=error_handler)
        .add_edge(
            "validate", "process", condition=lambda s: s.data.get("valid")
        )
        .add_edge(
            "validate", "error", condition=lambda s: not s.data.get("valid")
        )
        .set_start("validate")
        .set_end("process")
        .set_end("error")
    )

    assert len(graph.nodes) == 3
    assert graph.start_node == "validate"
    assert set(graph.end_nodes) == {"process", "error"}

  def test_priority_routing_with_convenience_api(self):
    """Test priority routing using convenience API."""
    graph = GraphAgent(name="router")
    agent = SimpleAgent(name="test_agent")

    (
        graph.add_node("check", agent=agent)
        .add_node("critical", agent=agent)
        .add_node("warning", agent=agent)
        .add_node("normal", agent=agent)
        .add_edge(
            "check",
            "critical",
            condition=lambda s: s.data.get("score", 0) > 0.9,
            priority=10,
        )
        .add_edge(
            "check",
            "warning",
            condition=lambda s: s.data.get("score", 0) > 0.5,
            priority=5,
        )
        .add_edge("check", "normal", priority=0)  # Fallback
        .set_start("check")
    )

    # Verify priority edges created
    assert len(graph.nodes["check"].edges) == 3
    priorities = [e.priority for e in graph.nodes["check"].edges]
    assert priorities == [10, 5, 0]


class TestAddEdgeWithEdgeCondition:
  """Test add_edge() with EdgeCondition objects (Pattern 1: Explicit)."""

  def test_add_edge_with_edge_condition(self):
    """Test add_edge with EdgeCondition object."""
    graph = GraphAgent(name="test")
    agent = SimpleAgent(name="test_agent")

    graph.add_node("source", agent=agent)
    graph.add_node("target", agent=agent)

    edge = EdgeCondition(
        target_node="target",
        condition=lambda s: s.data.get("valid"),
        priority=10,
        weight=0.5,
    )
    graph.add_edge("source", edge)

    assert len(graph.nodes["source"].edges) == 1
    assert graph.nodes["source"].edges[0] is edge
    assert graph.nodes["source"].edges[0].priority == 10
    assert graph.nodes["source"].edges[0].weight == 0.5

  def test_add_edge_error_edge_condition_with_params(self):
    """Test error when passing EdgeCondition with extra params."""
    graph = GraphAgent(name="test")
    agent = SimpleAgent(name="test_agent")

    graph.add_node("source", agent=agent)
    graph.add_node("target", agent=agent)

    edge = EdgeCondition(target_node="target", priority=10)

    with pytest.raises(ValueError, match="do not specify condition"):
      graph.add_edge("source", edge, condition=lambda s: True)

    with pytest.raises(ValueError, match="do not specify"):
      graph.add_edge("source", edge, priority=5)

    with pytest.raises(ValueError, match="do not specify"):
      graph.add_edge("source", edge, weight=0.5)

  def test_add_edge_edge_condition_target_not_found(self):
    """Test error when EdgeCondition references non-existent target."""
    graph = GraphAgent(name="test")
    agent = SimpleAgent(name="test_agent")

    graph.add_node("source", agent=agent)

    edge = EdgeCondition(target_node="nonexistent", priority=10)

    with pytest.raises(ValueError, match="Target node nonexistent not found"):
      graph.add_edge("source", edge)

  def test_add_edge_invalid_type(self):
    """Test error when passing invalid target type."""
    graph = GraphAgent(name="test")
    agent = SimpleAgent(name="test_agent")

    graph.add_node("source", agent=agent)

    with pytest.raises(
        TypeError, match="target_node must be str or EdgeCondition"
    ):
      graph.add_edge("source", 123)  # Invalid type

  def test_add_edge_chaining_with_edge_condition(self):
    """Test that add_edge returns self for chaining when using EdgeCondition."""
    graph = GraphAgent(name="test")
    agent = SimpleAgent(name="test_agent")

    graph.add_node("node1", agent=agent)
    graph.add_node("node2", agent=agent)
    graph.add_node("node3", agent=agent)

    result = graph.add_edge(
        "node1", EdgeCondition(target_node="node2", priority=10)
    ).add_edge("node2", "node3")

    assert result is graph
    assert len(graph.nodes["node1"].edges) == 1
    assert len(graph.nodes["node2"].edges) == 1

  def test_add_edge_mixed_edge_condition_and_convenience(self):
    """Test mixing EdgeCondition and convenience patterns on same node."""
    graph = GraphAgent(name="test")
    agent = SimpleAgent(name="test_agent")

    graph.add_node("source", agent=agent)
    graph.add_node("target1", agent=agent)
    graph.add_node("target2", agent=agent)

    # Add EdgeCondition first
    graph.add_edge(
        "source",
        EdgeCondition(
            target_node="target1",
            condition=lambda s: s.data.get("score", 0) > 0.9,
            priority=10,
        ),
    )

    # Add convenience edge
    graph.add_edge("source", "target2", priority=5)

    # Both edges should be in edges list
    assert len(graph.nodes["source"].edges) == 2
    assert graph.nodes["source"].edges[0].target_node == "target1"
    assert graph.nodes["source"].edges[0].priority == 10
    assert graph.nodes["source"].edges[1].target_node == "target2"
    assert graph.nodes["source"].edges[1].priority == 5
