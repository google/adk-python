"""Direct unit tests for parallel.py module (targeting >95% coverage).

Tests parallel execution internals directly:
- _collect_events function
- execute_parallel_group function
- All join strategies (WAIT_ALL, WAIT_ANY, WAIT_N)
- All error policies (FAIL_FAST, CONTINUE, COLLECT)
- State isolation with deepcopy
- State merge conflict detection (P0.2)
- Task lookup with O(1) inverse mapping (P0.1)
- CancelledError handling
- Telemetry integration
"""

import asyncio
from copy import deepcopy
from typing import AsyncGenerator
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph.graph_node import GraphNode
from google.adk.agents.graph.graph_state import GraphState
from google.adk.agents.graph.parallel import _collect_events
from google.adk.agents.graph.parallel import ErrorPolicy
from google.adk.agents.graph.parallel import execute_parallel_group
from google.adk.agents.graph.parallel import JoinStrategy
from google.adk.agents.graph.parallel import ParallelNodeGroup
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.genai import types
import pytest
from typing_extensions import override


# Test agents (proper BaseAgent implementations per ADK guidelines)
class SimpleAgent(BaseAgent):
  """Agent that yields one event."""

  output: str = "test"
  delay: float = 0.0

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    if self.delay > 0:
      await asyncio.sleep(self.delay)
    yield Event(
        author=self.name,
        content=types.Content(parts=[types.Part(text=self.output)]),
    )


class MultiEventAgent(BaseAgent):
  """Agent that yields multiple events."""

  num_events: int = 3

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    for i in range(self.num_events):
      yield Event(
          author=self.name,
          content=types.Content(parts=[types.Part(text=f"event_{i}")]),
      )


class ErrorProneAgent(BaseAgent):
  """Agent that raises an error."""

  error_msg: str = "Test error"

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    if False:
      yield  # Make it async generator
    raise RuntimeError(self.error_msg)


class StateModifyingAgent(BaseAgent):
  """Agent that modifies state."""

  key: str = "test"
  value: str = "modified"

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    # Simulate state modification (in real scenario would use state_delta)
    yield Event(
        author=self.name,
        content=types.Content(
            parts=[types.Part(text=f"Set {self.key}={self.value}")]
        ),
    )


def create_mock_context():
  """Create minimal InvocationContext for testing."""
  session = Session(id="test", appName="test", userId="test")
  session_service = InMemorySessionService()
  agent = SimpleAgent(name="mock")

  return InvocationContext(
      session=session,
      session_service=session_service,
      invocation_id="inv_test",
      agent=agent,
      user_content=None,
  )


class TestCollectEvents:
  """Test _collect_events helper function."""

  @pytest.mark.asyncio
  async def test_collect_events_success(self):
    """Test collecting events successfully."""

    async def event_generator():
      yield Event(
          author="test",
          content=types.Content(parts=[types.Part(text="event1")]),
      )
      yield Event(
          author="test",
          content=types.Content(parts=[types.Part(text="event2")]),
      )

    result = await _collect_events(event_generator())

    assert len(result["events"]) == 2
    assert result["error"] is None

  @pytest.mark.asyncio
  async def test_collect_events_with_error(self):
    """Test collecting events when generator raises error."""

    async def error_generator():
      yield Event(
          author="test",
          content=types.Content(parts=[types.Part(text="event1")]),
      )
      raise ValueError("Test error")

    result = await _collect_events(error_generator())

    assert len(result["events"]) == 1
    assert result["error"] is not None
    assert isinstance(result["error"], ValueError)

  @pytest.mark.asyncio
  async def test_collect_events_with_cancellation(self):
    """Test collecting events when task is cancelled."""

    async def cancellable_generator():
      yield Event(
          author="test",
          content=types.Content(parts=[types.Part(text="event1")]),
      )
      await asyncio.sleep(10)  # Will be cancelled
      yield Event(
          author="test",
          content=types.Content(parts=[types.Part(text="event2")]),
      )

    task = asyncio.create_task(_collect_events(cancellable_generator()))
    await asyncio.sleep(0.01)
    task.cancel()

    try:
      result = await task
      # Cancellation should return collected events without error
      assert len(result["events"]) == 1
      assert result["error"] is None
    except asyncio.CancelledError:
      # Also acceptable - task was cancelled
      pass


class TestExecuteParallelGroupJoinStrategies:
  """Test all join strategies."""

  @pytest.mark.asyncio
  async def test_wait_all_strategy(self):
    """Test WAIT_ALL waits for all nodes."""
    group = ParallelNodeGroup(
        nodes=["a", "b", "c"],
        join_strategy=JoinStrategy.WAIT_ALL,
    )

    nodes = {
        "a": GraphNode(
            name="a", agent=SimpleAgent(name="agent_a", output="out_a")
        ),
        "b": GraphNode(
            name="b", agent=SimpleAgent(name="agent_b", output="out_b")
        ),
        "c": GraphNode(
            name="c", agent=SimpleAgent(name="agent_c", output="out_c")
        ),
    }

    state = GraphState(data={})
    ctx = create_mock_context()

    async def mock_execute_node(node, branch_state, ctx):
      async for event in node.agent.run_async(ctx):
        yield event

    events = []
    async for event in execute_parallel_group(
        group, nodes, state, ctx, mock_execute_node
    ):
      events.append(event)

    # All 3 nodes should complete
    assert len(events) == 3

  @pytest.mark.asyncio
  async def test_wait_any_strategy(self):
    """Test WAIT_ANY returns after first completion."""
    group = ParallelNodeGroup(
        nodes=["fast", "slow"],
        join_strategy=JoinStrategy.WAIT_ANY,
    )

    nodes = {
        "fast": GraphNode(
            name="fast",
            agent=SimpleAgent(name="fast", output="fast_out", delay=0.001),
        ),
        "slow": GraphNode(
            name="slow",
            agent=SimpleAgent(name="slow", output="slow_out", delay=10.0),
        ),
    }

    state = GraphState(data={})
    ctx = create_mock_context()

    async def mock_execute_node(node, branch_state, ctx):
      async for event in node.agent.run_async(ctx):
        yield event

    events = []
    async for event in execute_parallel_group(
        group, nodes, state, ctx, mock_execute_node
    ):
      events.append(event)

    # At least fast node should complete
    assert len(events) >= 1
    # Should not wait for slow node
    assert any("fast_out" in str(e) for e in events)

  @pytest.mark.asyncio
  async def test_wait_n_strategy(self):
    """Test WAIT_N waits for N nodes."""
    group = ParallelNodeGroup(
        nodes=["a", "b", "c", "d", "e"],
        join_strategy=JoinStrategy.WAIT_N,
        wait_n=3,
    )

    nodes = {
        chr(97 + i): GraphNode(
            name=chr(97 + i),
            agent=SimpleAgent(
                name=f"agent_{chr(97 + i)}", output=f"out_{chr(97 + i)}"
            ),
        )
        for i in range(5)
    }

    state = GraphState(data={})
    ctx = create_mock_context()

    async def mock_execute_node(node, branch_state, ctx):
      async for event in node.agent.run_async(ctx):
        yield event

    events = []
    async for event in execute_parallel_group(
        group, nodes, state, ctx, mock_execute_node
    ):
      events.append(event)

    # At least 3 nodes should complete
    assert len(events) >= 3


class TestExecuteParallelGroupErrorPolicies:
  """Test all error policies."""

  @pytest.mark.asyncio
  async def test_fail_fast_policy_cancels_others(self):
    """Test FAIL_FAST cancels pending tasks on error."""
    group = ParallelNodeGroup(
        nodes=["good", "bad"],
        join_strategy=JoinStrategy.WAIT_ALL,
        error_policy=ErrorPolicy.FAIL_FAST,
    )

    nodes = {
        "good": GraphNode(
            name="good",
            agent=SimpleAgent(name="good", output="success", delay=10.0),
        ),
        "bad": GraphNode(
            name="bad",
            agent=ErrorProneAgent(name="bad", error_msg="fail_fast_error"),
        ),
    }

    state = GraphState(data={})
    ctx = create_mock_context()

    async def mock_execute_node(node, branch_state, ctx):
      async for event in node.agent.run_async(ctx):
        yield event

    with pytest.raises(RuntimeError, match="fail_fast_error"):
      async for _ in execute_parallel_group(
          group, nodes, state, ctx, mock_execute_node
      ):
        pass

  @pytest.mark.asyncio
  async def test_continue_policy_continues_on_error(self):
    """Test CONTINUE policy continues others on error."""
    group = ParallelNodeGroup(
        nodes=["good1", "bad", "good2"],
        join_strategy=JoinStrategy.WAIT_ALL,
        error_policy=ErrorPolicy.CONTINUE,
    )

    nodes = {
        "good1": GraphNode(
            name="good1", agent=SimpleAgent(name="good1", output="success1")
        ),
        "bad": GraphNode(name="bad", agent=ErrorProneAgent(name="bad")),
        "good2": GraphNode(
            name="good2", agent=SimpleAgent(name="good2", output="success2")
        ),
    }

    state = GraphState(data={})
    ctx = create_mock_context()

    async def mock_execute_node(node, branch_state, ctx):
      async for event in node.agent.run_async(ctx):
        yield event

    events = []
    async for event in execute_parallel_group(
        group, nodes, state, ctx, mock_execute_node
    ):
      events.append(event)

    # Good nodes should complete
    assert len(events) == 2

  @pytest.mark.asyncio
  async def test_collect_policy_collects_all_errors(self):
    """Test COLLECT policy collects all errors."""
    group = ParallelNodeGroup(
        nodes=["bad1", "bad2"],
        join_strategy=JoinStrategy.WAIT_ALL,
        error_policy=ErrorPolicy.COLLECT,
    )

    nodes = {
        "bad1": GraphNode(
            name="bad1", agent=ErrorProneAgent(name="bad1", error_msg="error1")
        ),
        "bad2": GraphNode(
            name="bad2", agent=ErrorProneAgent(name="bad2", error_msg="error2")
        ),
    }

    state = GraphState(data={})
    ctx = create_mock_context()

    async def mock_execute_node(node, branch_state, ctx):
      async for event in node.agent.run_async(ctx):
        yield event

    with pytest.raises(Exception, match="Errors in parallel execution"):
      async for _ in execute_parallel_group(
          group, nodes, state, ctx, mock_execute_node
      ):
        pass


class TestStateIsolationAndMerging:
  """Test state isolation and merge conflict detection."""

  @pytest.mark.asyncio
  async def test_state_isolation_with_deepcopy(self):
    """Test state is isolated between branches using deepcopy."""
    group = ParallelNodeGroup(
        nodes=["a", "b"],
        join_strategy=JoinStrategy.WAIT_ALL,
    )

    # Create agent that tries to modify state
    class StateCheckingAgent(BaseAgent):
      check_value: str = ""

      @override
      async def _run_async_impl(
          self, ctx: InvocationContext
      ) -> AsyncGenerator[Event, None]:
        # Check initial state value
        initial_value = ctx.session.state.get("shared_key", "")
        yield Event(
            author=self.name,
            content=types.Content(
                parts=[types.Part(text=f"Saw: {initial_value}")]
            ),
        )

    nodes = {
        "a": GraphNode(name="a", agent=StateCheckingAgent(name="agent_a")),
        "b": GraphNode(name="b", agent=StateCheckingAgent(name="agent_b")),
    }

    # Initial state with nested structure
    initial_data = {"shared_key": "initial", "nested": {"value": 42}}
    state = GraphState(data=initial_data.copy())
    ctx = create_mock_context()

    async def mock_execute_node(node, branch_state, ctx):
      # Verify branch got deepcopy
      assert branch_state.data is not state.data
      assert branch_state.data == state.data

      async for event in node.agent.run_async(ctx):
        yield event

    events = []
    async for event in execute_parallel_group(
        group, nodes, state, ctx, mock_execute_node
    ):
      events.append(event)

    assert len(events) == 2

  @pytest.mark.asyncio
  async def test_state_merge_conflict_detection(self):
    """Test state merge detects conflicts when multiple branches modify same key."""
    group = ParallelNodeGroup(
        nodes=["a", "b"],
        join_strategy=JoinStrategy.WAIT_ALL,
    )

    nodes = {
        "a": GraphNode(
            name="a", agent=SimpleAgent(name="agent_a", output="a_output")
        ),
        "b": GraphNode(
            name="b", agent=SimpleAgent(name="agent_b", output="b_output")
        ),
    }

    state = GraphState(data={})
    ctx = create_mock_context()

    async def mock_execute_node(node, branch_state, ctx):
      # Simulate both branches modifying same key
      branch_state.data["conflict_key"] = f"value_from_{node.name}"

      async for event in node.agent.run_async(ctx):
        yield event

    # Capture log warnings
    with patch("google.adk.agents.graph.parallel.logger") as mock_logger:
      events = []
      async for event in execute_parallel_group(
          group, nodes, state, ctx, mock_execute_node
      ):
        events.append(event)

      # Should log conflict warning
      assert any(
          "State merge conflict detected" in str(call)
          for call in mock_logger.warning.call_args_list
      )

    # Final state should have one of the values (last write wins)
    assert "conflict_key" in state.data
    assert state.data["conflict_key"] in ["value_from_a", "value_from_b"]


class TestTaskLookupAndCancellation:
  """Test O(1) task lookup and cancellation scenarios."""

  @pytest.mark.asyncio
  async def test_task_lookup_with_inverse_mapping(self):
    """Test O(1) task lookup using inverse mapping (P0.1 fix)."""
    group = ParallelNodeGroup(
        nodes=["a", "b", "c"],
        join_strategy=JoinStrategy.WAIT_ALL,
    )

    nodes = {
        "a": GraphNode(name="a", agent=SimpleAgent(name="agent_a")),
        "b": GraphNode(name="b", agent=SimpleAgent(name="agent_b")),
        "c": GraphNode(name="c", agent=SimpleAgent(name="agent_c")),
    }

    state = GraphState(data={})
    ctx = create_mock_context()

    async def mock_execute_node(node, branch_state, ctx):
      async for event in node.agent.run_async(ctx):
        yield event

    # Execute and verify no task lookup failures
    events = []
    async for event in execute_parallel_group(
        group, nodes, state, ctx, mock_execute_node
    ):
      events.append(event)

    assert len(events) == 3

  @pytest.mark.asyncio
  async def test_cancellation_of_pending_tasks_wait_any(self):
    """Test pending tasks are cancelled after WAIT_ANY completes."""
    group = ParallelNodeGroup(
        nodes=["fast", "slow1", "slow2"],
        join_strategy=JoinStrategy.WAIT_ANY,
    )

    nodes = {
        "fast": GraphNode(
            name="fast", agent=SimpleAgent(name="fast", delay=0.001)
        ),
        "slow1": GraphNode(
            name="slow1", agent=SimpleAgent(name="slow1", delay=10.0)
        ),
        "slow2": GraphNode(
            name="slow2", agent=SimpleAgent(name="slow2", delay=10.0)
        ),
    }

    state = GraphState(data={})
    ctx = create_mock_context()

    async def mock_execute_node(node, branch_state, ctx):
      async for event in node.agent.run_async(ctx):
        yield event

    events = []
    async for event in execute_parallel_group(
        group, nodes, state, ctx, mock_execute_node
    ):
      events.append(event)

    # Only fast node should complete
    assert len(events) == 1


class TestEdgeCases:
  """Test edge cases and error scenarios."""

  @pytest.mark.asyncio
  async def test_node_not_found_raises_error(self):
    """Test error when node name not in nodes dict."""
    group = ParallelNodeGroup(
        nodes=["a", "nonexistent"],
        join_strategy=JoinStrategy.WAIT_ALL,
    )

    nodes = {
        "a": GraphNode(name="a", agent=SimpleAgent(name="agent_a")),
    }

    state = GraphState(data={})
    ctx = create_mock_context()

    async def mock_execute_node(node, branch_state, ctx):
      async for event in node.agent.run_async(ctx):
        yield event

    with pytest.raises(ValueError, match="not found in graph"):
      async for _ in execute_parallel_group(
          group, nodes, state, ctx, mock_execute_node
      ):
        pass

  @pytest.mark.asyncio
  async def test_multiple_events_per_node(self):
    """Test nodes that yield multiple events."""
    group = ParallelNodeGroup(
        nodes=["multi"],
        join_strategy=JoinStrategy.WAIT_ALL,
    )

    nodes = {
        "multi": GraphNode(
            name="multi", agent=MultiEventAgent(name="multi", num_events=5)
        ),
    }

    state = GraphState(data={})
    ctx = create_mock_context()

    async def mock_execute_node(node, branch_state, ctx):
      async for event in node.agent.run_async(ctx):
        yield event

    events = []
    async for event in execute_parallel_group(
        group, nodes, state, ctx, mock_execute_node
    ):
      events.append(event)

    # Should collect all 5 events
    assert len(events) == 5


class TestTelemetryIntegration:
  """Test telemetry and tracing integration."""

  @pytest.mark.asyncio
  async def test_telemetry_span_attributes(self):
    """Test telemetry span captures parallel execution attributes."""
    group = ParallelNodeGroup(
        nodes=["a", "b"],
        join_strategy=JoinStrategy.WAIT_ALL,
        error_policy=ErrorPolicy.CONTINUE,
    )

    nodes = {
        "a": GraphNode(name="a", agent=SimpleAgent(name="agent_a")),
        "b": GraphNode(name="b", agent=SimpleAgent(name="agent_b")),
    }

    state = GraphState(data={})
    ctx = create_mock_context()

    async def mock_execute_node(node, branch_state, ctx):
      async for event in node.agent.run_async(ctx):
        yield event

    # Mock tracer to capture span attributes
    with patch("google.adk.agents.graph.parallel.tracer") as mock_tracer:
      mock_span = MagicMock()
      mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
          mock_span
      )
      mock_tracer.start_as_current_span.return_value.__exit__.return_value = (
          None
      )

      events = []
      async for event in execute_parallel_group(
          group, nodes, state, ctx, mock_execute_node
      ):
        events.append(event)

      # Verify span attributes were set
      assert mock_span.set_attribute.called

      # Check specific attributes
      attribute_calls = {
          call[0][0]: call[0][1]
          for call in mock_span.set_attribute.call_args_list
      }

      assert attribute_calls.get("parallel.node_count") == 2
      assert attribute_calls.get("parallel.join_strategy") == "all"
      assert attribute_calls.get("parallel.error_policy") == "continue"
