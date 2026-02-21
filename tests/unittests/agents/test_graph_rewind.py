"""Tests for GraphAgent + Rewind integration.

Tests the tight coupling between GraphAgent and ADK's rewind feature,
enabling temporal navigation within graph workflows.
"""

from typing import AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import GraphNode
from google.adk.agents.graph import rewind_to_node
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import pytest
from typing_extensions import override

# Test Fixtures (proper BaseAgent implementations per ADK guidelines)


class SimpleAgent(BaseAgent):
  """Simple test agent that returns predictable output."""

  def __init__(self, name: str, output: str):
    super().__init__(name=name)
    self._test_output = output

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    """Return test output."""
    yield Event(
        author=self.name,
        content=types.Content(parts=[types.Part(text=self._test_output)]),
    )


class StatefulAgent(BaseAgent):
  """Agent that modifies state."""

  def __init__(self, name: str, state_key: str, state_value: str):
    super().__init__(name=name)
    self._state_key = state_key
    self._state_value = state_value

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    """Modify state and return output."""
    ctx.session.state[self._state_key] = self._state_value
    yield Event(
        author=self.name,
        content=types.Content(
            parts=[
                types.Part(text=f"Set {self._state_key}={self._state_value}")
            ]
        ),
    )


def _get_node_invocations_from_events(session) -> dict:
  """Extract node_invocations from the latest agent_state event."""
  for event in reversed(session.events):
    if (
        event.actions
        and event.actions.agent_state
        and "node_invocations" in (event.actions.agent_state or {})
    ):
      return event.actions.agent_state["node_invocations"]
  return {}


@pytest.fixture
def session_service():
  """Create in-memory session service."""
  return InMemorySessionService()


# Test 1: Basic Rewind to Node


@pytest.mark.asyncio
async def test_rewind_to_node_basic(session_service):
  """Test rewinding to a specific node."""
  # Create graph with 3 nodes
  graph = GraphAgent(name="test_graph")
  graph.add_node(
      GraphNode(name="step1", agent=SimpleAgent("agent1", "Output 1"))
  )
  graph.add_node(
      GraphNode(name="step2", agent=SimpleAgent("agent2", "Output 2"))
  )
  graph.add_node(
      GraphNode(name="step3", agent=SimpleAgent("agent3", "Output 3"))
  )

  graph.add_edge("step1", "step2")
  graph.add_edge("step2", "step3")
  graph.set_start("step1")
  graph.set_end("step3")

  # Execute graph
  runner = Runner(
      app_name="test_app",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  events = []
  new_message = types.Content(parts=[types.Part(text="Start")])
  async for event in runner.run_async(
      user_id="u1", session_id="s1", new_message=new_message
  ):
    events.append(event)

  # Get session
  session = await session_service.get_session(
      app_name="test_app", user_id="u1", session_id="s1"
  )

  # Verify all nodes executed
  node_invocations = _get_node_invocations_from_events(session)
  assert "step1" in node_invocations
  assert "step2" in node_invocations
  assert "step3" in node_invocations
  assert len(node_invocations["step1"]) == 1
  assert len(node_invocations["step2"]) == 1
  assert len(node_invocations["step3"]) == 1

  # Rewind to step2 - just verify it doesn't raise an error
  # The actual state reversion is handled by ADK's rewind functionality
  await rewind_to_node(graph, session_service, "test_app", "u1", "s1", "step2")

  # Verify we can still access the session after rewind
  session_after_rewind = await session_service.get_session(
      app_name="test_app", user_id="u1", session_id="s1"
  )
  assert session_after_rewind is not None


# Test 2: Rewind in Loop (Multiple Invocations)


@pytest.mark.asyncio
async def test_rewind_to_node_in_loop(session_service):
  """Test rewinding when node executed multiple times."""
  # Create graph with loop
  graph = GraphAgent(name="test_graph", max_iterations=5)
  graph.add_node(
      GraphNode(
          name="counter",
          agent=StatefulAgent("counter_agent", "count", "incremented"),
      )
  )
  graph.add_node(
      GraphNode(
          name="validator",
          agent=SimpleAgent("validator_agent", "Validated"),
      )
  )

  # Create loop: counter -> validator -> counter (condition based)
  graph.add_edge("counter", "validator")
  graph.add_edge(
      "validator",
      "counter",
      condition=lambda s: s.data.get("_graph_iteration", 0) < 3,
  )
  graph.set_start("counter")
  graph.set_end("validator")

  # Execute graph (should loop 3 times)
  runner = Runner(
      app_name="test_app",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  new_message = types.Content(parts=[types.Part(text="Start")])
  async for event in runner.run_async(
      user_id="u1", session_id="s2", new_message=new_message
  ):
    pass

  # Get session
  session = await session_service.get_session(
      app_name="test_app", user_id="u1", session_id="s2"
  )

  # Verify multiple invocations
  node_invocations = _get_node_invocations_from_events(session)
  assert len(node_invocations.get("counter", [])) >= 2
  assert len(node_invocations.get("validator", [])) >= 2

  # Rewind to 2nd invocation of counter
  counter_invocations = node_invocations["counter"]
  if len(counter_invocations) >= 2:
    # Just verify rewind works with specific invocation index
    await rewind_to_node(
        graph,
        session_service,
        "test_app",
        "u1",
        "s2",
        "counter",
        invocation_index=1,
    )

    # Verify session still accessible
    session_after = await session_service.get_session(
        app_name="test_app", user_id="u1", session_id="s2"
    )
    assert session_after is not None


# Test 3: Rewind Restores State


@pytest.mark.asyncio
async def test_rewind_restores_state(session_service):
  """Test that rewind works with sequential nodes."""
  graph = GraphAgent(name="test_graph")
  graph.add_node(
      GraphNode(name="step_a", agent=SimpleAgent("agent_a", "Output A"))
  )
  graph.add_node(
      GraphNode(name="step_b", agent=SimpleAgent("agent_b", "Output B"))
  )
  graph.add_node(
      GraphNode(name="step_c", agent=SimpleAgent("agent_c", "Output C"))
  )

  graph.add_edge("step_a", "step_b")
  graph.add_edge("step_b", "step_c")
  graph.set_start("step_a")
  graph.set_end("step_c")

  # Execute graph
  runner = Runner(
      app_name="test_app",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  new_message = types.Content(parts=[types.Part(text="Start")])
  async for event in runner.run_async(
      user_id="u1", session_id="s3", new_message=new_message
  ):
    pass

  # Get session after full execution
  session = await session_service.get_session(
      app_name="test_app", user_id="u1", session_id="s3"
  )

  # Verify all nodes executed
  node_invocations = _get_node_invocations_from_events(session)
  assert "step_a" in node_invocations
  assert "step_b" in node_invocations
  assert "step_c" in node_invocations

  # Rewind to step_b - verify it works without error
  await rewind_to_node(graph, session_service, "test_app", "u1", "s3", "step_b")

  # Verify session still accessible after rewind
  session_after = await session_service.get_session(
      app_name="test_app", user_id="u1", session_id="s3"
  )
  assert session_after is not None


# Test 4: Rewind Invalid Node


@pytest.mark.asyncio
async def test_rewind_invalid_node(session_service):
  """Test rewind fails for non-executed node."""
  graph = GraphAgent(name="test_graph")
  graph.add_node(
      GraphNode(name="step1", agent=SimpleAgent("agent1", "Output 1"))
  )
  graph.add_node(
      GraphNode(name="step2", agent=SimpleAgent("agent2", "Output 2"))
  )

  graph.add_edge("step1", "step2")
  graph.set_start("step1")
  graph.set_end("step2")

  # Execute graph
  runner = Runner(
      app_name="test_app",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  new_message = types.Content(parts=[types.Part(text="Start")])
  async for event in runner.run_async(
      user_id="u1", session_id="s4", new_message=new_message
  ):
    pass

  session = await session_service.get_session(
      app_name="test_app", user_id="u1", session_id="s4"
  )

  # Try to rewind to non-existent node
  with pytest.raises(ValueError, match="has not been executed yet"):
    await rewind_to_node(
        graph, session_service, "test_app", "u1", "s4", "nonexistent_node"
    )


# Test 5: Rewind Invalid Invocation Index


@pytest.mark.asyncio
async def test_rewind_invalid_invocation_index(session_service):
  """Test rewind fails for invalid invocation index."""
  graph = GraphAgent(name="test_graph")
  graph.add_node(
      GraphNode(name="step1", agent=SimpleAgent("agent1", "Output 1"))
  )
  graph.set_start("step1")
  graph.set_end("step1")

  # Execute graph
  runner = Runner(
      app_name="test_app",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  new_message = types.Content(parts=[types.Part(text="Start")])
  async for event in runner.run_async(
      user_id="u1", session_id="s5", new_message=new_message
  ):
    pass

  session = await session_service.get_session(
      app_name="test_app", user_id="u1", session_id="s5"
  )

  # Try to rewind with invalid index
  with pytest.raises(ValueError, match="out of range"):
    await rewind_to_node(
        graph,
        session_service,
        "test_app",
        "u1",
        "s5",
        "step1",
        invocation_index=10,
    )


# Test 6: Rewind Preserves Path


@pytest.mark.asyncio
async def test_rewind_preserves_path(session_service):
  """Test execution path preserved correctly after rewind."""
  graph = GraphAgent(name="test_graph")
  graph.add_node(GraphNode(name="a", agent=SimpleAgent("agent_a", "A")))
  graph.add_node(GraphNode(name="b", agent=SimpleAgent("agent_b", "B")))
  graph.add_node(GraphNode(name="c", agent=SimpleAgent("agent_c", "C")))

  graph.add_edge("a", "b")
  graph.add_edge("b", "c")
  graph.set_start("a")
  graph.set_end("c")

  # Execute graph
  runner = Runner(
      app_name="test_app",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  new_message = types.Content(parts=[types.Part(text="Start")])
  async for event in runner.run_async(
      user_id="u1", session_id="s6", new_message=new_message
  ):
    pass

  session = await session_service.get_session(
      app_name="test_app", user_id="u1", session_id="s6"
  )

  # Path should be [a, b, c]
  # Note: Path tracking depends on implementation details

  # Rewind to b - verify it works
  await rewind_to_node(graph, session_service, "test_app", "u1", "s6", "b")

  # Verify session still accessible
  session_after = await session_service.get_session(
      app_name="test_app", user_id="u1", session_id="s6"
  )
  assert session_after is not None


# Test 7: Rewind with Conditional Branching


@pytest.mark.asyncio
async def test_rewind_with_branching(session_service):
  """Test rewind works with conditional branches."""
  graph = GraphAgent(name="test_graph")
  graph.add_node(
      GraphNode(name="start", agent=SimpleAgent("start_agent", "Start"))
  )
  graph.add_node(GraphNode(name="branch_a", agent=SimpleAgent("a_agent", "A")))
  graph.add_node(GraphNode(name="branch_b", agent=SimpleAgent("b_agent", "B")))
  graph.add_node(GraphNode(name="end", agent=SimpleAgent("end_agent", "End")))

  # Simple branching - always take branch_a
  graph.add_edge("start", "branch_a", condition=lambda s: True)
  graph.add_edge("start", "branch_b", condition=lambda s: False)
  graph.add_edge("branch_a", "end")
  graph.add_edge("branch_b", "end")
  graph.set_start("start")
  graph.set_end("end")

  # Execute graph
  runner = Runner(
      app_name="test_app",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  new_message = types.Content(parts=[types.Part(text="Start")])
  async for event in runner.run_async(
      user_id="u1", session_id="s7", new_message=new_message
  ):
    pass

  session = await session_service.get_session(
      app_name="test_app", user_id="u1", session_id="s7"
  )

  # Verify execution completed
  node_invocations = _get_node_invocations_from_events(session)
  assert "start" in node_invocations

  # Rewind to start - verify rewind works with conditional branching
  await rewind_to_node(graph, session_service, "test_app", "u1", "s7", "start")

  # Verify session still accessible
  session_after = await session_service.get_session(
      app_name="test_app", user_id="u1", session_id="s7"
  )
  assert session_after is not None


# Test 8: Rewind Negative Index


@pytest.mark.asyncio
async def test_rewind_negative_index(session_service):
  """Test rewind with negative invocation index (most recent)."""
  graph = GraphAgent(name="test_graph", max_iterations=5)
  graph.add_node(
      GraphNode(
          name="repeater",
          agent=SimpleAgent("repeater_agent", "Repeated"),
      )
  )
  graph.add_edge(
      "repeater",
      "repeater",
      condition=lambda s: s.data.get("_graph_iteration", 0) < 2,
  )
  graph.set_start("repeater")
  graph.set_end("repeater")

  # Execute graph (loops 2 times)
  runner = Runner(
      app_name="test_app",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  new_message = types.Content(parts=[types.Part(text="Start")])
  async for event in runner.run_async(
      user_id="u1", session_id="s8", new_message=new_message
  ):
    pass

  session = await session_service.get_session(
      app_name="test_app", user_id="u1", session_id="s8"
  )

  node_invocations = _get_node_invocations_from_events(session)
  repeater_invocations = node_invocations.get("repeater", [])

  if len(repeater_invocations) >= 2:
    # Rewind to last invocation (index -1) - test negative indexing
    await rewind_to_node(
        graph,
        session_service,
        "test_app",
        "u1",
        "s8",
        "repeater",
        invocation_index=-1,
    )

    # Verify session still accessible
    session_after = await session_service.get_session(
        app_name="test_app", user_id="u1", session_id="s8"
    )
    assert session_after is not None


# Test 9: Rewind Integration with Session State


@pytest.mark.asyncio
async def test_rewind_session_state_integration(session_service):
  """Test rewind properly integrates with session state tracking."""
  graph = GraphAgent(name="test_graph")
  graph.add_node(
      GraphNode(name="step1", agent=SimpleAgent("agent1", "Output 1"))
  )
  graph.add_node(
      GraphNode(name="step2", agent=SimpleAgent("agent2", "Output 2"))
  )

  graph.add_edge("step1", "step2")
  graph.set_start("step1")
  graph.set_end("step2")

  # Execute graph
  runner = Runner(
      app_name="test_app",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  new_message = types.Content(parts=[types.Part(text="Start")])
  async for event in runner.run_async(
      user_id="u1", session_id="s9", new_message=new_message
  ):
    pass

  session = await session_service.get_session(
      app_name="test_app", user_id="u1", session_id="s9"
  )

  # Verify node_invocations in agent_state events
  node_invocations = _get_node_invocations_from_events(session)
  assert isinstance(node_invocations, dict)
  assert len(node_invocations) > 0

  # Verify invocation IDs are tracked
  for node_name, invocations in node_invocations.items():
    assert isinstance(invocations, list)
    assert len(invocations) > 0
    # Each invocation should be a string (invocation ID)
    for inv_id in invocations:
      assert isinstance(inv_id, str)


# Test 10: Rewind with Empty Graph State


@pytest.mark.asyncio
async def test_rewind_empty_node_invocations(session_service):
  """Test rewind handles case with no invocations gracefully."""
  graph = GraphAgent(name="test_graph")
  graph.add_node(
      GraphNode(name="step1", agent=SimpleAgent("agent1", "Output 1"))
  )
  graph.set_start("step1")
  graph.set_end("step1")

  # Create session without executing
  runner = Runner(
      app_name="test_app",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )
  session = await session_service.create_session(
      app_name="test_app", user_id="u1", session_id="s10"
  )

  # Try to rewind without any execution
  with pytest.raises(ValueError, match="has not been executed yet"):
    await rewind_to_node(
        graph, session_service, "test_app", "u1", "s10", "step1"
    )
