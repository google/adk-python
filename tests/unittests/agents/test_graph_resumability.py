"""Tests for GraphAgent ADK resumability integration.

Verifies that GraphAgent properly integrates with ADK's built-in
resumability pattern: ctx.is_resumable guards, ctx.should_pause_invocation,
resume from saved state, and end_of_agent lifecycle.
"""

from __future__ import annotations

from typing import AsyncGenerator
from unittest.mock import patch

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph.graph_agent import GraphAgent
from google.adk.agents.graph.graph_agent_state import GraphAgentState
from google.adk.agents.graph.graph_node import GraphNode
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.run_config import RunConfig
from google.adk.apps import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.genai import types
import pytest


# ============================================================================
# Test Agents
# ============================================================================


class SimpleTestAgent(BaseAgent):
  """Test agent that yields predetermined responses."""

  model_config = {"extra": "allow", "arbitrary_types_allowed": True}

  def __init__(self, name: str, responses: list[str]):
    super().__init__(name=name)
    object.__setattr__(self, "_responses", responses)
    object.__setattr__(self, "_call_count", 0)

  async def _run_async_impl(self, ctx) -> AsyncGenerator[Event, None]:
    call_count = object.__getattribute__(self, "_call_count")
    responses = object.__getattribute__(self, "_responses")
    response = responses[min(call_count, len(responses) - 1)]
    object.__setattr__(self, "_call_count", call_count + 1)
    yield Event(
        author=self.name,
        content=types.Content(parts=[types.Part(text=response)]),
    )

  @property
  def call_count(self):
    return object.__getattribute__(self, "_call_count")


class PausingAgent(BaseAgent):
  """Agent that yields a long-running tool event to trigger pause."""

  model_config = {"extra": "allow", "arbitrary_types_allowed": True}

  def __init__(self, name: str):
    super().__init__(name=name)

  async def _run_async_impl(self, ctx) -> AsyncGenerator[Event, None]:
    # Yield an event with long_running_tool_ids to trigger pause
    fc = types.FunctionCall(
        id="tool_call_1",
        name="long_running_tool",
        args={},
    )
    yield Event(
        author=self.name,
        content=types.Content(parts=[types.Part(function_call=fc)]),
        long_running_tool_ids=["tool_call_1"],
    )


# ============================================================================
# Helpers
# ============================================================================


def _build_linear_graph(
    name: str,
    node_agents: list[BaseAgent],
    node_names: list[str] | None = None,
) -> GraphAgent:
  """Build a linear graph: node0 -> node1 -> ... -> nodeN."""
  if node_names is None:
    node_names = [f"n{i}" for i in range(len(node_agents))]
  graph = GraphAgent(name=name)
  for nname, agent in zip(node_names, node_agents):
    graph.add_node(GraphNode(name=nname, agent=agent))
  for i in range(len(node_names) - 1):
    graph.add_edge(node_names[i], node_names[i + 1])
  graph.set_start(node_names[0])
  graph.set_end(node_names[-1])
  return graph


def _make_ctx(
    agent: BaseAgent,
    *,
    resumable: bool = False,
    agent_states: dict | None = None,
) -> InvocationContext:
  """Create InvocationContext for testing."""
  svc = InMemorySessionService()
  session = Session(id="test-session", appName="test", userId="test-user")
  ctx = InvocationContext(
      session=session,
      session_service=svc,
      invocation_id="inv-1",
      agent=agent,
      user_content=types.Content(
          role="user", parts=[types.Part(text="test")]
      ),
  )
  if resumable:
    ctx.resumability_config = ResumabilityConfig(is_resumable=True)
    ctx.run_config = RunConfig()
  if agent_states:
    ctx.agent_states = agent_states
  return ctx


async def _collect_events(graph: GraphAgent, ctx: InvocationContext) -> list[Event]:
  """Collect all events from graph execution."""
  events = []
  async for event in graph._run_async_impl(ctx):
    events.append(event)
  return events


# ============================================================================
# Tests: Fix 1 — Resume from saved node
# ============================================================================


@pytest.mark.asyncio
class TestResumeFromSavedNode:
  """Verify graph resumes from agent_state.current_node, not start_node."""

  async def test_resume_from_saved_node(self):
    """After pause at node B, resume starts from B (not A)."""
    agent_a = SimpleTestAgent("a", ["output_a"])
    agent_b = SimpleTestAgent("b", ["output_b"])
    agent_c = SimpleTestAgent("c", ["output_c"])
    graph = _build_linear_graph("g", [agent_a, agent_b, agent_c], ["nA", "nB", "nC"])

    # Simulate resumed context: agent_state says we're at nB, iteration 1
    saved_state = GraphAgentState(current_node="nB", iteration=1, path=["nA"])
    ctx = _make_ctx(
        graph,
        resumable=True,
        agent_states={"g": saved_state.model_dump(mode="json")},
    )

    events = await _collect_events(graph, ctx)

    # Agent A should NOT have been called (we resumed past it)
    assert agent_a.call_count == 0
    # Agents B and C should have been called
    assert agent_b.call_count == 1
    assert agent_c.call_count == 1

  async def test_resume_with_removed_node_restarts(self):
    """If saved node no longer exists, restart from start_node."""
    agent_a = SimpleTestAgent("a", ["output_a"])
    agent_b = SimpleTestAgent("b", ["output_b"])
    graph = _build_linear_graph("g", [agent_a, agent_b], ["nA", "nB"])

    # Saved state references a node that doesn't exist
    saved_state = GraphAgentState(current_node="nX_removed", iteration=3)
    ctx = _make_ctx(
        graph,
        resumable=True,
        agent_states={"g": saved_state.model_dump(mode="json")},
    )

    events = await _collect_events(graph, ctx)

    # Should restart from beginning
    assert agent_a.call_count == 1
    assert agent_b.call_count == 1


# ============================================================================
# Tests: Fix 2 — Guard state events with is_resumable
# ============================================================================


@pytest.mark.asyncio
class TestStateEventGuards:
  """Verify state events are only emitted when ctx.is_resumable=True."""

  async def test_non_resumable_context_no_end_of_agent(self):
    """When ctx.is_resumable=False, no end_of_agent event emitted.

    Per-iteration state events are always emitted (they serve rewind,
    interrupts, telemetry — not just resumability). Only end_of_agent
    is guarded by is_resumable.
    """
    agent_a = SimpleTestAgent("a", ["out"])
    graph = _build_linear_graph("g", [agent_a], ["nA"])
    ctx = _make_ctx(graph, resumable=False)

    events = await _collect_events(graph, ctx)

    # end_of_agent should NOT be emitted for non-resumable
    end_events = [
        e for e in events
        if e.actions and e.actions.end_of_agent
    ]
    assert len(end_events) == 0

    # But per-iteration state events ARE emitted (they serve other consumers)
    state_events = [
        e for e in events
        if e.actions and e.actions.agent_state is not None
    ]
    assert len(state_events) > 0

  async def test_resumable_context_emits_state_events(self):
    """When ctx.is_resumable=True, agent_state events are emitted."""
    agent_a = SimpleTestAgent("a", ["out"])
    agent_b = SimpleTestAgent("b", ["out"])
    graph = _build_linear_graph("g", [agent_a, agent_b], ["nA", "nB"])
    ctx = _make_ctx(graph, resumable=True)

    events = await _collect_events(graph, ctx)

    # Should have state events (at least end_of_agent)
    state_events = [
        e for e in events
        if e.actions and (
            e.actions.agent_state is not None or e.actions.end_of_agent
        )
    ]
    assert len(state_events) > 0

  async def test_resume_skips_duplicate_state_event(self):
    """First iteration after resume doesn't emit duplicate state event."""
    agent_b = SimpleTestAgent("b", ["output_b"])
    agent_c = SimpleTestAgent("c", ["output_c"])
    graph = _build_linear_graph(
        "g",
        [SimpleTestAgent("a", ["x"]), agent_b, agent_c],
        ["nA", "nB", "nC"],
    )

    # Resume at nB
    saved_state = GraphAgentState(current_node="nB", iteration=1, path=["nA"])
    ctx = _make_ctx(
        graph,
        resumable=True,
        agent_states={"g": saved_state.model_dump(mode="json")},
    )

    events = await _collect_events(graph, ctx)

    # Count state events for graph "g" (not end_of_agent, just agent_state)
    state_events = [
        e for e in events
        if e.author == "g"
        and e.actions
        and e.actions.agent_state is not None
        and not e.actions.end_of_agent
    ]
    # For a 2-node execution (B, C) with resume skipping first:
    # Only nC should emit a state event (nB is skipped as resume iteration)
    assert len(state_events) == 1


# ============================================================================
# Tests: Fix 3 — Pause on long-running tool
# ============================================================================


@pytest.mark.asyncio
class TestPauseOnLongRunningTool:
  """Verify should_pause_invocation triggers pause in graph execution."""

  async def test_pause_on_long_running_tool(self):
    """should_pause_invocation triggers, execution stops, state preserved."""
    agent_a = SimpleTestAgent("a", ["output_a"])
    pausing_agent = PausingAgent("pauser")
    agent_c = SimpleTestAgent("c", ["output_c"])
    graph = _build_linear_graph(
        "g", [agent_a, pausing_agent, agent_c], ["nA", "nB", "nC"]
    )

    ctx = _make_ctx(graph, resumable=True)

    events = await _collect_events(graph, ctx)

    # Agent A should have run
    assert agent_a.call_count == 1
    # Agent C should NOT have run (paused at B)
    assert agent_c.call_count == 0

    # The pause event (long_running_tool_ids) should be in events
    pause_events = [
        e for e in events if e.long_running_tool_ids
    ]
    assert len(pause_events) == 1

    # No end_of_agent should be emitted (we paused)
    end_events = [
        e for e in events
        if e.actions and e.actions.end_of_agent
    ]
    assert len(end_events) == 0

  async def test_resume_after_pause_continues(self):
    """Full pause->resume roundtrip: A runs, B pauses, resume->B runs, C runs."""
    agent_a = SimpleTestAgent("a", ["output_a"])
    pausing_agent = PausingAgent("pauser")
    agent_b_resumed = SimpleTestAgent("b_resumed", ["output_b"])
    agent_c = SimpleTestAgent("c", ["output_c"])

    # First run: A -> B(pauses)
    graph1 = _build_linear_graph(
        "g", [agent_a, pausing_agent, agent_c], ["nA", "nB", "nC"]
    )
    ctx1 = _make_ctx(graph1, resumable=True)
    events1 = await _collect_events(graph1, ctx1)

    # Verify paused at B
    assert agent_a.call_count == 1
    assert agent_c.call_count == 0

    # Build a new graph for resume where B won't pause anymore
    agent_a2 = SimpleTestAgent("a2", ["output_a2"])
    agent_b2 = SimpleTestAgent("b2", ["output_b2"])
    agent_c2 = SimpleTestAgent("c2", ["output_c2"])
    graph2 = _build_linear_graph(
        "g", [agent_a2, agent_b2, agent_c2], ["nA", "nB", "nC"]
    )

    # Resume from nB
    saved_state = GraphAgentState(current_node="nB", iteration=1, path=["nA"])
    ctx2 = _make_ctx(
        graph2,
        resumable=True,
        agent_states={"g": saved_state.model_dump(mode="json")},
    )
    events2 = await _collect_events(graph2, ctx2)

    # A should NOT run (resumed past it), B and C should run
    assert agent_a2.call_count == 0
    assert agent_b2.call_count == 1
    assert agent_c2.call_count == 1

    # end_of_agent should be emitted on completed run
    end_events = [
        e for e in events2
        if e.actions and e.actions.end_of_agent
    ]
    assert len(end_events) == 1


# ============================================================================
# Tests: Fix 4 — end_of_agent guarded
# ============================================================================


@pytest.mark.asyncio
class TestEndOfAgentGuard:
  """Verify end_of_agent lifecycle events."""

  async def test_end_of_agent_guarded(self):
    """end_of_agent only emitted when ctx.is_resumable=True."""
    agent_a = SimpleTestAgent("a", ["out"])
    graph = _build_linear_graph("g", [agent_a], ["nA"])

    # Non-resumable: no end_of_agent
    ctx_nr = _make_ctx(graph, resumable=False)
    events_nr = await _collect_events(graph, ctx_nr)
    end_nr = [e for e in events_nr if e.actions and e.actions.end_of_agent]
    assert len(end_nr) == 0

    # Resumable: has end_of_agent
    agent_a2 = SimpleTestAgent("a2", ["out"])
    graph2 = _build_linear_graph("g", [agent_a2], ["nA"])
    ctx_r = _make_ctx(graph2, resumable=True)
    events_r = await _collect_events(graph2, ctx_r)
    end_r = [e for e in events_r if e.actions and e.actions.end_of_agent]
    assert len(end_r) == 1

  async def test_end_of_agent_skipped_on_pause(self):
    """No end_of_agent when paused mid-graph."""
    agent_a = SimpleTestAgent("a", ["out"])
    pausing = PausingAgent("pauser")
    graph = _build_linear_graph("g", [agent_a, pausing], ["nA", "nB"])
    ctx = _make_ctx(graph, resumable=True)

    events = await _collect_events(graph, ctx)

    end_events = [e for e in events if e.actions and e.actions.end_of_agent]
    assert len(end_events) == 0


# ============================================================================
# Tests: Fix 5 — Cycle resets sub-agent states
# ============================================================================


@pytest.mark.asyncio
class TestCycleReset:
  """Verify reset_sub_agent_states on cycle revisit."""

  async def test_cycle_resets_sub_agent_state(self):
    """Back-edge to visited node calls reset_sub_agent_states."""
    # Create a graph with a cycle: nA -> nB -> nA (conditional)
    agent_a = SimpleTestAgent("a", ["go", "done"])
    agent_b = SimpleTestAgent("b", ["loop_back", "final"])

    graph = GraphAgent(name="g", max_iterations=5)
    graph.add_node(GraphNode(name="nA", agent=agent_a))
    graph.add_node(GraphNode(name="nB", agent=agent_b))

    graph.add_edge("nA", "nB")
    # B -> A (back-edge, always taken except when max_iterations)
    graph.add_edge("nB", "nA")
    graph.set_start("nA")
    # No end node — relies on max_iterations

    ctx = _make_ctx(graph, resumable=True)

    # Track reset calls via patch
    reset_calls = []
    original_reset = ctx.reset_sub_agent_states

    def tracking_reset(agent_name: str):
      reset_calls.append(agent_name)
      return original_reset(agent_name)

    with patch.object(
        type(ctx), "reset_sub_agent_states", side_effect=tracking_reset
    ):
      events = await _collect_events(graph, ctx)

    # Agent A should be visited at least twice (nA -> nB -> nA)
    assert agent_a.call_count >= 2
    # reset_sub_agent_states should have been called for the agent
    # when revisiting nA (the second visit)
    assert len(reset_calls) > 0
    # The reset should be for agent "a" (the node agent)
    assert "a" in reset_calls


# ============================================================================
# Tests: _get_resume_state method
# ============================================================================


class TestGetResumeState:
  """Unit tests for _get_resume_state helper."""

  def test_fresh_start(self):
    """No saved state returns start_node."""
    graph = _build_linear_graph("g", [SimpleTestAgent("a", ["x"])], ["nA"])
    state = GraphAgentState()
    node, iteration, resuming = graph._get_resume_state(state)
    assert node == "nA"
    assert iteration == 0
    assert resuming is False

  def test_resume_from_valid_node(self):
    """Saved state with valid node returns that node."""
    agent_a = SimpleTestAgent("a", ["x"])
    agent_b = SimpleTestAgent("b", ["x"])
    graph = _build_linear_graph("g", [agent_a, agent_b], ["nA", "nB"])
    state = GraphAgentState(current_node="nB", iteration=3)
    node, iteration, resuming = graph._get_resume_state(state)
    assert node == "nB"
    assert iteration == 3
    assert resuming is True

  def test_resume_from_invalid_node_falls_back(self):
    """Saved state with removed node falls back to start_node."""
    graph = _build_linear_graph("g", [SimpleTestAgent("a", ["x"])], ["nA"])
    state = GraphAgentState(current_node="nRemoved", iteration=5)
    node, iteration, resuming = graph._get_resume_state(state)
    assert node == "nA"
    assert iteration == 0
    assert resuming is False


# ============================================================================
# Tests: Integration — pause_invocation flag, state integrity, rewind compat
# ============================================================================


@pytest.mark.asyncio
class TestPauseInvocationFlag:
  """Verify pause_invocation=True/False is reflected in emitted events."""

  async def test_pause_sets_flag_and_skips_final_events(self):
    """When pause_invocation=True, no final response or end_of_agent."""
    agent_a = SimpleTestAgent("a", ["out"])
    pausing = PausingAgent("pauser")
    agent_c = SimpleTestAgent("c", ["out"])
    graph = _build_linear_graph(
        "g", [agent_a, pausing, agent_c], ["nA", "nB", "nC"]
    )
    ctx = _make_ctx(graph, resumable=True)
    events = await _collect_events(graph, ctx)

    # No final graph response (author=graph, state_delta with graph_data)
    final_responses = [
        e for e in events
        if e.author == "g"
        and e.content
        and e.actions
        and e.actions.state_delta
        and "graph_data" in (e.actions.state_delta or {})
    ]
    assert len(final_responses) == 0

    # No end_of_agent
    end_events = [e for e in events if e.actions and e.actions.end_of_agent]
    assert len(end_events) == 0

  async def test_no_pause_emits_final_events(self):
    """When pause_invocation=False (normal run), final response emitted."""
    agent_a = SimpleTestAgent("a", ["out"])
    graph = _build_linear_graph("g", [agent_a], ["nA"])
    ctx = _make_ctx(graph, resumable=True)
    events = await _collect_events(graph, ctx)

    # Final graph response present
    final_responses = [
        e for e in events
        if e.author == "g"
        and e.content
        and e.actions
        and e.actions.state_delta
        and "graph_data" in (e.actions.state_delta or {})
    ]
    assert len(final_responses) == 1

    # end_of_agent present
    end_events = [e for e in events if e.actions and e.actions.end_of_agent]
    assert len(end_events) == 1


@pytest.mark.asyncio
class TestStateIntegrity:
  """Verify agent_state is consistent through pause/resume cycle."""

  async def test_agent_state_tracks_current_node_on_pause(self):
    """After pause, ctx.agent_states has current_node pointing to paused node."""
    agent_a = SimpleTestAgent("a", ["out"])
    pausing = PausingAgent("pauser")
    graph = _build_linear_graph(
        "g", [agent_a, pausing], ["nA", "nB"]
    )
    ctx = _make_ctx(graph, resumable=True)
    events = await _collect_events(graph, ctx)

    # The last agent_state event should have current_node = "nB"
    state_events = [
        e for e in events
        if e.author == "g"
        and e.actions
        and e.actions.agent_state is not None
        and not e.actions.end_of_agent
    ]
    assert len(state_events) >= 1
    last_state = state_events[-1].actions.agent_state
    assert last_state["current_node"] == "nB"
    assert "nA" in last_state["path"]

  async def test_load_agent_state_roundtrip(self):
    """State saved during run can be loaded back via _load_agent_state."""
    agent_a = SimpleTestAgent("a", ["out"])
    agent_b = SimpleTestAgent("b", ["out"])
    graph = _build_linear_graph("g", [agent_a, agent_b], ["nA", "nB"])
    ctx = _make_ctx(graph, resumable=True)
    events = await _collect_events(graph, ctx)

    # Get the last state event's agent_state dict
    state_events = [
        e for e in events
        if e.author == "g"
        and e.actions
        and e.actions.agent_state is not None
        and not e.actions.end_of_agent
    ]
    last_state_dict = state_events[-1].actions.agent_state

    # Simulate loading it back (as _load_agent_state does)
    loaded = GraphAgentState.model_validate(last_state_dict)
    assert loaded.current_node == "nB"
    assert loaded.iteration == 2
    assert loaded.path == ["nA", "nB"]

  async def test_function_node_pause_is_falsy(self):
    """Function nodes never set pause in output_holder — safe path."""
    def my_func(state, ctx):
      return "func_output"

    graph = GraphAgent(name="g")
    graph.add_node(GraphNode(name="nA", function=my_func))
    graph.set_start("nA")
    graph.set_end("nA")

    ctx = _make_ctx(graph, resumable=True)
    events = await _collect_events(graph, ctx)

    # Should complete normally with end_of_agent
    end_events = [e for e in events if e.actions and e.actions.end_of_agent]
    assert len(end_events) == 1


@pytest.mark.asyncio
class TestRewindCompatibility:
  """Verify rewind works with resumability state events."""

  async def test_state_events_contain_node_invocations(self):
    """State events include node_invocations needed by rewind_to_node."""
    agent_a = SimpleTestAgent("a", ["out"])
    agent_b = SimpleTestAgent("b", ["out"])
    graph = _build_linear_graph("g", [agent_a, agent_b], ["nA", "nB"])
    ctx = _make_ctx(graph, resumable=True)
    events = await _collect_events(graph, ctx)

    # Find state events with node_invocations
    state_events = [
        e for e in events
        if e.author == "g"
        and e.actions
        and e.actions.agent_state is not None
        and not e.actions.end_of_agent
    ]
    # Last state event should have node_invocations for both nodes
    last_state = state_events[-1].actions.agent_state
    assert "node_invocations" in last_state
    assert "nA" in last_state["node_invocations"]
    assert "nB" in last_state["node_invocations"]

  async def test_rewind_to_node_with_runner(self):
    """Full integration: run graph via Runner, then rewind_to_node."""
    from google.adk.agents.graph import rewind_to_node
    from google.adk.runners import Runner

    agent_a = SimpleTestAgent("step1", ["a_out"])
    agent_b = SimpleTestAgent("step2", ["b_out"])
    graph = _build_linear_graph("g", [agent_a, agent_b], ["step1", "step2"])

    svc = InMemorySessionService()
    runner = Runner(app_name="test", agent=graph, session_service=svc)
    await svc.create_session(app_name="test", user_id="u", session_id="s")

    # Execute graph
    events = []
    async for event in runner.run_async(
        user_id="u",
        session_id="s",
        new_message=types.Content(
            role="user", parts=[types.Part(text="go")]
        ),
    ):
      events.append(event)

    # Both agents should have run
    assert agent_a.call_count == 1
    assert agent_b.call_count == 1

    # Rewind to step1 — should not raise
    await rewind_to_node(
        graph=graph,
        session_service=svc,
        app_name="test",
        user_id="u",
        session_id="s",
        node_name="step1",
    )
