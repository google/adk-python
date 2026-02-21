"""Tests for enhanced graph routing (priority, weight, fallback)."""

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import EdgeCondition
from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import GraphNode
from google.adk.agents.graph import GraphState
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import pytest


class SimpleAgent(BaseAgent):
  """Simple test agent that returns predictable output."""

  def __init__(self, name: str, output: str):
    super().__init__(name=name)
    self._test_output = output

  async def _run_async_impl(self, ctx):
    yield Event(
        author=self.name,
        content=types.Content(parts=[types.Part(text=self._test_output)]),
    )


@pytest.fixture
async def session_service():
  """Create InMemorySessionService for tests."""
  return InMemorySessionService()


@pytest.mark.asyncio
async def test_priority_routing_basic(session_service):
  """Test that higher priority edges are evaluated first."""
  graph = GraphAgent(name="test_graph")

  start = SimpleAgent(name="start", output="starting")
  high_priority = SimpleAgent(name="high_priority", output="high_priority_path")
  low_priority = SimpleAgent(name="low_priority", output="low_priority_path")

  graph.add_node(GraphNode(name="start", agent=start))
  graph.add_node(GraphNode(name="high_priority", agent=high_priority))
  graph.add_node(GraphNode(name="low_priority", agent=low_priority))

  # Both edges have conditions that would match, but high priority should win
  start_node = graph.nodes["start"]
  start_node.edges = [
      EdgeCondition(
          target_node="low_priority",
          condition=lambda s: True,  # Always matches
          priority=1,  # Lower priority
      ),
      EdgeCondition(
          target_node="high_priority",
          condition=lambda s: True,  # Always matches
          priority=10,  # Higher priority - should be chosen
      ),
  ]

  graph.set_start("start")
  graph.set_end("high_priority")
  graph.set_end("low_priority")

  runner = Runner(
      app_name="test_app",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  events = []
  async for event in runner.run_async(
      user_id="u1",
      session_id="s1",
      new_message=types.Content(parts=[types.Part(text="Start")]),
  ):
    events.append(event)

  event_texts = [
      e.content.parts[0].text for e in events if e.content and e.content.parts
  ]

  # Should route to high_priority path (priority=10 > priority=1)
  assert "high_priority_path" in event_texts
  assert "low_priority_path" not in event_texts


@pytest.mark.asyncio
async def test_fallback_edge_priority_zero(session_service):
  """Test that priority=0 edges act as fallbacks."""
  graph = GraphAgent(name="test_graph")

  start = SimpleAgent(name="start", output="starting")
  conditional = SimpleAgent(name="conditional", output="conditional_path")
  fallback = SimpleAgent(name="fallback", output="fallback_path")

  graph.add_node(GraphNode(name="start", agent=start))
  graph.add_node(GraphNode(name="conditional", agent=conditional))
  graph.add_node(GraphNode(name="fallback", agent=fallback))

  # Add conditional edge (won't match) and fallback edge
  start_node = graph.nodes["start"]
  start_node.edges = [
      EdgeCondition(
          target_node="conditional",
          condition=lambda s: s.data.get("trigger_condition", False),
          priority=5,
      ),
      EdgeCondition(
          target_node="fallback",
          priority=0,  # Fallback - always matches if no higher priority matched
      ),
  ]

  graph.set_start("start")
  graph.set_end("conditional")
  graph.set_end("fallback")

  runner = Runner(
      app_name="test_app",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  events = []
  async for event in runner.run_async(
      user_id="u1",
      session_id="s1",
      new_message=types.Content(parts=[types.Part(text="Start")]),
  ):
    events.append(event)

  event_texts = [
      e.content.parts[0].text for e in events if e.content and e.content.parts
  ]

  # Should route to fallback (conditional doesn't match)
  assert "fallback_path" in event_texts
  assert "conditional_path" not in event_texts


@pytest.mark.asyncio
async def test_weighted_routing(session_service):
  """Test weighted random selection among matching edges."""
  graph = GraphAgent(name="test_graph")

  start = SimpleAgent(name="start", output="starting")
  path_a = SimpleAgent(name="path_a", output="path_a")
  path_b = SimpleAgent(name="path_b", output="path_b")

  graph.add_node(GraphNode(name="start", agent=start))
  graph.add_node(GraphNode(name="path_a", agent=path_a))
  graph.add_node(GraphNode(name="path_b", agent=path_b))

  # Both edges match, but path_a has much higher weight
  start_node = graph.nodes["start"]
  start_node.edges = [
      EdgeCondition(
          target_node="path_a",
          condition=lambda s: True,
          priority=1,
          weight=0.9,  # 90% probability
      ),
      EdgeCondition(
          target_node="path_b",
          condition=lambda s: True,
          priority=1,  # Same priority
          weight=0.1,  # 10% probability
      ),
  ]

  graph.set_start("start")
  graph.set_end("path_a")
  graph.set_end("path_b")

  runner = Runner(
      app_name="test_app",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  # Run multiple times to verify weighted distribution
  path_a_count = 0
  path_b_count = 0
  trials = 50

  for i in range(trials):
    events = []
    async for event in runner.run_async(
        user_id="u1",
        session_id=f"s_{i}",
        new_message=types.Content(parts=[types.Part(text="Start")]),
    ):
      events.append(event)

    event_texts = [
        e.content.parts[0].text for e in events if e.content and e.content.parts
    ]

    if "path_a" in event_texts:
      path_a_count += 1
    if "path_b" in event_texts:
      path_b_count += 1

  # With 0.9/0.1 weights, expect roughly 45/5 split (90%/10%)
  # Allow some variance (at least 35/50 = 70% for path_a)
  assert (
      path_a_count > trials * 0.7
  ), f"Expected path_a > 70%, got {path_a_count}/{trials}"
  assert (
      path_b_count < trials * 0.3
  ), f"Expected path_b < 30%, got {path_b_count}/{trials}"


@pytest.mark.asyncio
async def test_priority_and_condition(session_service):
  """Test that priority works correctly with conditions."""
  graph = GraphAgent(name="test_graph")

  start = SimpleAgent(name="start", output="starting")
  high_match = SimpleAgent(name="high_match", output="high_match_path")
  low_no_match = SimpleAgent(name="low_no_match", output="low_no_match_path")
  fallback = SimpleAgent(name="fallback", output="fallback_path")

  graph.add_node(GraphNode(name="start", agent=start))
  graph.add_node(GraphNode(name="high_match", agent=high_match))
  graph.add_node(GraphNode(name="low_no_match", agent=low_no_match))
  graph.add_node(GraphNode(name="fallback", agent=fallback))

  # Set state with score=0.5
  start_node = graph.nodes["start"]

  # Define output mapper to set score in state
  def set_score(output, state):
    new_state = GraphState(data=state.data.copy().copy())
    new_state.data["score"] = 0.5
    return new_state

  start_node.output_mapper = set_score

  start_node.edges = [
      EdgeCondition(
          target_node="low_no_match",
          condition=lambda s: s.data.get("score", 0)
          > 0.8,  # Won't match (0.5 < 0.8)
          priority=20,  # Highest priority but won't match
      ),
      EdgeCondition(
          target_node="high_match",
          condition=lambda s: s.data.get("score", 0)
          > 0.3,  # Will match (0.5 > 0.3)
          priority=10,  # Medium priority and matches
      ),
      EdgeCondition(
          target_node="fallback",
          priority=0,  # Fallback
      ),
  ]

  graph.set_start("start")
  graph.set_end("high_match")
  graph.set_end("low_no_match")
  graph.set_end("fallback")

  runner = Runner(
      app_name="test_app",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  events = []
  async for event in runner.run_async(
      user_id="u1",
      session_id="s1",
      new_message=types.Content(parts=[types.Part(text="Start")]),
  ):
    events.append(event)

  event_texts = [
      e.content.parts[0].text for e in events if e.content and e.content.parts
  ]

  # Should route to high_match (highest priority that matches)
  assert "high_match_path" in event_texts
  assert "low_no_match_path" not in event_texts
  assert "fallback_path" not in event_texts


@pytest.mark.asyncio
async def test_backward_compatibility(session_service):
  """Test that existing code without priority/weight still works."""
  graph = GraphAgent(name="test_graph")

  start = SimpleAgent(name="start", output="starting")
  next_node = SimpleAgent(name="next", output="next_output")

  graph.add_node(GraphNode(name="start", agent=start))
  graph.add_node(GraphNode(name="next", agent=next_node))

  # Old style: just target_node and condition (no priority/weight)
  start_node = graph.nodes["start"]
  start_node.edges = [
      EdgeCondition(target_node="next", condition=lambda s: True),
  ]

  graph.set_start("start")
  graph.set_end("next")

  runner = Runner(
      app_name="test_app",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  events = []
  async for event in runner.run_async(
      user_id="u1",
      session_id="s1",
      new_message=types.Content(parts=[types.Part(text="Start")]),
  ):
    events.append(event)

  event_texts = [
      e.content.parts[0].text for e in events if e.content and e.content.parts
  ]

  # Should work exactly as before
  assert "next_output" in event_texts


@pytest.mark.asyncio
async def test_edge_repr():
  """Test EdgeCondition string representation."""
  edge = EdgeCondition(
      target_node="test_target",
      condition=lambda s: True,
      priority=5,
      weight=0.7,
  )

  repr_str = repr(edge)
  assert "test_target" in repr_str
  assert "priority=5" in repr_str
  assert "weight=0.7" in repr_str
  assert "has_condition=True" in repr_str


# ---------------------------------------------------------------------------
# Edge-weight selection edge cases (graph_node.py lines 176, 187)
# ---------------------------------------------------------------------------


def test_weighted_edges_all_zero_weight():
  """Line 176: when all matching edges have weight=0.0, pick the first one.

  all_same_weight is False (weights differ: -0.5 vs 0.5) but total_weight == 0,
  so the code falls into the total_weight == 0 branch and picks matching_edges[0].
  """
  from google.adk.agents.graph.graph_state import GraphState

  node = GraphNode(name="n", function=lambda s, c: "x")
  # Two edges at same priority that both match, but weights are asymmetric around 0
  node.edges = [
      EdgeCondition(
          target_node="first", condition=lambda s: True, priority=1, weight=-0.5
      ),
      EdgeCondition(
          target_node="second", condition=lambda s: True, priority=1, weight=0.5
      ),
  ]

  state = GraphState()
  # With different weights but total == 0, should always pick "first"
  for _ in range(20):
    result = node.get_next_node(state)
    assert (
        result == "first"
    ), f"Expected 'first' (total_weight=0 branch), got '{result}'"


def test_weighted_edges_float_fallback():
  """Line 187: safety fallback when rand_value > cumulative after loop.

  Achieved by mocking random.random() to return 2.0 (> 1.0), making
  rand_value = 2.0 * total_weight > total_weight so no edge matches in
  the weighted-choice loop and we reach the fallback return.
  """
  from unittest.mock import patch

  from google.adk.agents.graph.graph_state import GraphState

  node = GraphNode(name="n", function=lambda s, c: "x")
  # Two edges at same priority, different weights (so weighted path is used)
  node.edges = [
      EdgeCondition(
          target_node="a", condition=lambda s: True, priority=1, weight=0.4
      ),
      EdgeCondition(
          target_node="b", condition=lambda s: True, priority=1, weight=0.6
      ),
  ]

  state = GraphState()
  # random.random() returns 2.0 → rand_value = 2.0 * 1.0 = 2.0 > cumulative(1.0)
  # The for-loop completes without returning → fallback at line 187 returns last edge
  with patch("random.random", return_value=2.0):
    result = node.get_next_node(state)
  assert result == "b", f"Expected fallback to last edge 'b', got '{result}'"


class TestConditionalRouting:
  """Unit tests for EdgeCondition.should_route and GraphNode.get_next_node."""

  def test_unconditional_edge(self):
    """Edge without condition always routes."""
    edge = EdgeCondition(target_node="next")
    state = GraphState(data={})

    assert edge.should_route(state) is True

  def test_conditional_edge_true(self):
    """Conditional edge routes when condition is true."""
    edge = EdgeCondition(
        target_node="next", condition=lambda s: s.data.get("value") > 10
    )
    state = GraphState(data={"value": 15})

    assert edge.should_route(state) is True

  def test_conditional_edge_false(self):
    """Conditional edge does not route when condition is false."""
    edge = EdgeCondition(
        target_node="next", condition=lambda s: s.data.get("value") > 10
    )
    state = GraphState(data={"value": 5})

    assert edge.should_route(state) is False

  def test_node_routing_multiple_edges_picks_first_match(self):
    """Node with multiple conditional edges selects the first matching edge."""
    node = GraphNode(name="router", agent=SimpleAgent("agent", "ok"))

    node.add_edge("path1", condition=lambda s: s.data.get("type") == "A")
    node.add_edge("path2", condition=lambda s: s.data.get("type") == "B")
    node.add_edge("default", condition=lambda s: True)

    assert node.get_next_node(GraphState(data={"type": "A"})) == "path1"
    assert node.get_next_node(GraphState(data={"type": "B"})) == "path2"
    assert node.get_next_node(GraphState(data={"type": "C"})) == "default"


def test_edge_sort_cache_invalidation():
  """Edge sort cache is built once and invalidated on add_edge."""
  node = GraphNode(
      name="test",
      agent=SimpleAgent(name="agent", output="out"),
  )

  assert node._sorted_edges_cache is None

  node.add_edge("path1", condition=lambda s: True)
  assert node._sorted_edges_cache is None

  node.get_next_node(GraphState(data={}))
  assert node._sorted_edges_cache is not None
  cached = node._sorted_edges_cache

  node.get_next_node(GraphState(data={}))
  assert node._sorted_edges_cache is cached

  node.add_edge("path2", condition=lambda s: False)
  assert node._sorted_edges_cache is None

  node.get_next_node(GraphState(data={}))
  assert node._sorted_edges_cache is not None
  assert node._sorted_edges_cache is not cached
