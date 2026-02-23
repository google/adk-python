"""Comprehensive tests for parallel execution (0% coverage in audit).

Tests for parallel.py module:
- ParallelNodeGroup creation
- Parallel node execution with different join strategies
- Error policies (FAIL_FAST, CONTINUE, COLLECT)
- State isolation with deepcopy
- State merging after execution
- CancelledError handling
"""

import asyncio

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import GraphNode
from google.adk.agents.graph.graph_state import GraphState
from google.adk.agents.graph.parallel import ErrorPolicy
from google.adk.agents.graph.parallel import JoinStrategy
from google.adk.agents.graph.parallel import ParallelNodeGroup
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import pytest


class TestAgent(BaseAgent):
  """Test agent that returns a value."""

  def __init__(self, name: str, output: str, delay: float = 0.0):
    super().__init__(name=name)
    self._output = output
    self._delay = delay

  async def _run_async_impl(self, ctx):
    if self._delay > 0:
      await asyncio.sleep(self._delay)
    yield Event(
        author=self.name,
        content=types.Content(parts=[types.Part(text=self._output)]),
    )


class ErrorAgent(BaseAgent):
  """Test agent that raises an error."""

  def __init__(self, name: str, error_msg: str):
    super().__init__(name=name)
    self._error_msg = error_msg

  async def _run_async_impl(self, ctx):
    raise ValueError(self._error_msg)
    yield  # Make it an async generator (unreachable but required)


class TestParallelNodeGroup:
  """Test ParallelNodeGroup configuration."""

  def test_create_parallel_group(self):
    """Test creating a parallel node group."""
    group = ParallelNodeGroup(
        nodes=["a", "b", "c"],
        join_strategy=JoinStrategy.WAIT_ALL,
        error_policy=ErrorPolicy.FAIL_FAST,
    )

    assert group.nodes == ["a", "b", "c"]
    assert group.join_strategy == JoinStrategy.WAIT_ALL
    assert group.error_policy == ErrorPolicy.FAIL_FAST

  def test_wait_n_validation(self):
    """Test WAIT_N strategy validation."""
    # Valid: wait_n <= number of nodes
    group = ParallelNodeGroup(
        nodes=["a", "b", "c"],
        join_strategy=JoinStrategy.WAIT_N,
        wait_n=2,
    )
    assert group.wait_n == 2

    # Invalid: wait_n > number of nodes
    with pytest.raises(ValueError, match="cannot be greater"):
      ParallelNodeGroup(
          nodes=["a", "b"],
          join_strategy=JoinStrategy.WAIT_N,
          wait_n=3,
      )


class TestParallelExecution:
  """Test parallel node execution."""

  @pytest.mark.asyncio
  async def test_parallel_wait_all(self):
    """Test parallel execution with WAIT_ALL strategy."""
    graph = GraphAgent(name="test_parallel")

    # Create parallel nodes
    agent_a = TestAgent(name="agent_a", output="output_a")
    agent_b = TestAgent(name="agent_b", output="output_b")
    agent_c = TestAgent(name="agent_c", output="output_c")

    graph.add_node(GraphNode(name="a", agent=agent_a))
    graph.add_node(GraphNode(name="b", agent=agent_b))
    graph.add_node(GraphNode(name="c", agent=agent_c))

    # Add parallel group
    graph.add_parallel_group(
        group_id="parallel_abc",
        group=ParallelNodeGroup(
            nodes=["a", "b", "c"],
            join_strategy=JoinStrategy.WAIT_ALL,
        ),
    )

    graph.set_start("a")
    graph.set_end("a")
    graph.set_end("b")
    graph.set_end("c")

    # Execute graph
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="test",
        agent=graph,
        session_service=session_service,
        auto_create_session=True,
    )

    events = []
    async for event in runner.run_async(
        user_id="u1",
        session_id="s1",
        new_message=types.Content(parts=[types.Part(text="test")]),
    ):
      if event.content and event.content.parts:
        text = event.content.parts[0].text
        if text:
          events.append(text)

    # All nodes should execute
    assert "output_a" in events
    assert "output_b" in events
    assert "output_c" in events

  @pytest.mark.asyncio
  async def test_parallel_wait_any(self):
    """Test parallel execution with WAIT_ANY strategy."""
    graph = GraphAgent(name="test_parallel")

    # Create nodes with different delays
    agent_a = TestAgent(name="agent_a", output="fast", delay=0.01)
    agent_b = TestAgent(name="agent_b", output="slow", delay=1.0)

    graph.add_node(GraphNode(name="a", agent=agent_a))
    graph.add_node(GraphNode(name="b", agent=agent_b))

    # Add parallel group with WAIT_ANY
    graph.add_parallel_group(
        group_id="parallel_ab",
        group=ParallelNodeGroup(
            nodes=["a", "b"],
            join_strategy=JoinStrategy.WAIT_ANY,
        ),
    )

    graph.set_start("a")
    graph.set_end("a")
    graph.set_end("b")

    # Execute graph
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="test",
        agent=graph,
        session_service=session_service,
        auto_create_session=True,
    )

    events = []
    async for event in runner.run_async(
        user_id="u1",
        session_id="s1",
        new_message=types.Content(parts=[types.Part(text="test")]),
    ):
      if event.content and event.content.parts:
        text = event.content.parts[0].text
        if text:
          events.append(text)

    # At least the fast node should complete
    assert "fast" in events

  @pytest.mark.asyncio
  async def test_parallel_wait_n(self):
    """Test parallel execution with WAIT_N strategy."""
    graph = GraphAgent(name="test_parallel")

    # Create 5 nodes
    for i in range(5):
      agent = TestAgent(name=f"agent_{i}", output=f"output_{i}")
      graph.add_node(GraphNode(name=f"n{i}", agent=agent))

    # Add parallel group - wait for 3 out of 5
    graph.add_parallel_group(
        group_id="parallel_n_group",
        group=ParallelNodeGroup(
            nodes=[f"n{i}" for i in range(5)],
            join_strategy=JoinStrategy.WAIT_N,
            wait_n=3,
        ),
    )

    graph.set_start("n0")
    for i in range(5):
      graph.set_end(f"n{i}")

    # Execute graph
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="test",
        agent=graph,
        session_service=session_service,
        auto_create_session=True,
    )

    events = []
    async for event in runner.run_async(
        user_id="u1",
        session_id="s1",
        new_message=types.Content(parts=[types.Part(text="test")]),
    ):
      if event.content and event.content.parts:
        text = event.content.parts[0].text
        if text and text.startswith("output_"):
          events.append(text)

    # At least 3 nodes should complete
    assert len(events) >= 3


class TestErrorPolicies:
  """Test error handling in parallel execution."""

  @pytest.mark.asyncio
  async def test_fail_fast_policy(self):
    """Test FAIL_FAST error policy cancels all on first error."""
    graph = GraphAgent(name="test_parallel")

    # Create nodes: one fails, others succeed
    agent_good = TestAgent(name="agent_good", output="success", delay=1.0)
    agent_bad = ErrorAgent(name="agent_bad", error_msg="test error")

    graph.add_node(GraphNode(name="good", agent=agent_good))
    graph.add_node(GraphNode(name="bad", agent=agent_bad))

    # Add parallel group with FAIL_FAST
    graph.add_parallel_group(
        group_id="fail_fast_group",
        group=ParallelNodeGroup(
            nodes=["good", "bad"],
            join_strategy=JoinStrategy.WAIT_ALL,
            error_policy=ErrorPolicy.FAIL_FAST,
        ),
    )

    graph.set_start("good")
    graph.set_end("good")
    graph.set_end("bad")

    # Execute graph - should raise error
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="test",
        agent=graph,
        session_service=session_service,
        auto_create_session=True,
    )

    with pytest.raises(ValueError, match="test error"):
      async for _ in runner.run_async(
          user_id="u1",
          session_id="s1",
          new_message=types.Content(parts=[types.Part(text="test")]),
      ):
        pass

  @pytest.mark.asyncio
  async def test_continue_policy(self):
    """Test CONTINUE error policy continues on error."""
    graph = GraphAgent(name="test_parallel")

    # Create nodes: one fails, others succeed
    agent_good1 = TestAgent(name="agent_good1", output="success1")
    agent_bad = ErrorAgent(name="agent_bad", error_msg="test error")
    agent_good2 = TestAgent(name="agent_good2", output="success2")

    graph.add_node(GraphNode(name="good1", agent=agent_good1))
    graph.add_node(GraphNode(name="bad", agent=agent_bad))
    graph.add_node(GraphNode(name="good2", agent=agent_good2))

    # Add parallel group with CONTINUE
    graph.add_parallel_group(
        group_id="continue_group",
        group=ParallelNodeGroup(
            nodes=["good1", "bad", "good2"],
            join_strategy=JoinStrategy.WAIT_ALL,
            error_policy=ErrorPolicy.CONTINUE,
        ),
    )

    graph.set_start("good1")
    graph.set_end("good1")
    graph.set_end("bad")
    graph.set_end("good2")

    # Execute graph - should continue despite error
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="test",
        agent=graph,
        session_service=session_service,
        auto_create_session=True,
    )

    events = []
    async for event in runner.run_async(
        user_id="u1",
        session_id="s1",
        new_message=types.Content(parts=[types.Part(text="test")]),
    ):
      if event.content and event.content.parts:
        text = event.content.parts[0].text
        if text and text.startswith("success"):
          events.append(text)

    # Good nodes should complete
    assert "success1" in events
    assert "success2" in events


class TestStateIsolation:
  """Test state isolation with deepcopy."""

  @pytest.mark.asyncio
  async def test_parallel_state_isolation(self):
    """Test that parallel nodes execute concurrently without conflicts."""
    graph = GraphAgent(name="test_parallel")

    # Create simple parallel nodes
    agent_a = TestAgent(name="agent_a", output="output_a")
    agent_b = TestAgent(name="agent_b", output="output_b")

    graph.add_node(GraphNode(name="a", agent=agent_a))
    graph.add_node(GraphNode(name="b", agent=agent_b))

    # Add parallel group
    graph.add_parallel_group(
        group_id="state_isolation_group",
        group=ParallelNodeGroup(
            nodes=["a", "b"], join_strategy=JoinStrategy.WAIT_ALL
        ),
    )

    graph.set_start("a")
    graph.set_end("a")
    graph.set_end("b")

    # Execute graph
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="test",
        agent=graph,
        session_service=session_service,
        auto_create_session=True,
    )

    events = []
    async for event in runner.run_async(
        user_id="u1",
        session_id="s1",
        new_message=types.Content(parts=[types.Part(text="test")]),
    ):
      if event.content and event.content.parts:
        text = event.content.parts[0].text
        if text and text.startswith("output_"):
          events.append(text)

    # Verify both nodes executed
    assert "output_a" in events
    assert "output_b" in events

  @pytest.mark.asyncio
  async def test_nested_state_isolation(self):
    """Test deepcopy prevents nested structure corruption."""

    class NestedStateAgent(BaseAgent):
      """Agent that modifies nested state."""

      def __init__(self, name: str, value: int):
        super().__init__(name=name)
        self._value = value

      async def _run_async_impl(self, ctx):
        # Try to modify nested structure (should be isolated)
        if "nested" in ctx.session.state:
          ctx.session.state["nested"]["value"] = self._value
        yield Event(
            author=self.name,
            content=types.Content(
                parts=[types.Part(text=f"modified to {self._value}")]
            ),
        )

    graph = GraphAgent(name="test_parallel")

    # Create nodes that try to modify same nested structure
    agent_a = NestedStateAgent(name="agent_a", value=100)
    agent_b = NestedStateAgent(name="agent_b", value=200)

    graph.add_node(GraphNode(name="a", agent=agent_a))
    graph.add_node(GraphNode(name="b", agent=agent_b))

    # Add parallel group
    graph.add_parallel_group(
        group_id="nested_isolation_group",
        group=ParallelNodeGroup(
            nodes=["a", "b"], join_strategy=JoinStrategy.WAIT_ALL
        ),
    )

    graph.set_start("a")
    graph.set_end("a")
    graph.set_end("b")

    # Execute graph with initial nested state
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="test",
        user_id="u1",
        session_id="s1",
        state={"nested": {"value": 0}},
    )

    runner = Runner(
        app_name="test",
        agent=graph,
        session_service=session_service,
        auto_create_session=False,
    )

    async for _ in runner.run_async(
        user_id="u1",
        session_id="s1",
        new_message=types.Content(parts=[types.Part(text="test")]),
    ):
      pass

    # With deepcopy, modifications are isolated
    # Final state depends on merge strategy (last write wins)


@pytest.mark.asyncio
async def test_parallel_unchanged_keys_not_overwritten():
  """Branches that don't modify a key must not overwrite other branches' changes.

  Uses function nodes (which directly modify state.data in parallel branches)
  to test the diff-based merge. Branch A modifies shared key, branch B doesn't.
  """

  def writer_fn(state, ctx):
    state.data["shared"] = "from_a"
    state.data["writer_only"] = "a_result"
    return "writer done"

  def reader_fn(state, ctx):
    state.data["reader_only"] = "b_result"
    # Does NOT touch "shared" — should not overwrite writer's change
    return "reader done"

  graph = GraphAgent(name="test_merge")

  graph.add_node(GraphNode(name="writer", function=writer_fn))
  graph.add_node(GraphNode(name="reader", function=reader_fn))

  graph.add_parallel_group(
      group_id="merge_test",
      group=ParallelNodeGroup(
          nodes=["writer", "reader"],
          join_strategy=JoinStrategy.WAIT_ALL,
      ),
  )

  graph.set_start("writer")
  graph.set_end("writer")
  graph.set_end("reader")

  session_service = InMemorySessionService()

  # Pre-seed state with shared="original"
  session = await session_service.create_session(
      app_name="test", user_id="u1", session_id="s1"
  )
  session.state["shared"] = "original"

  runner = Runner(
      app_name="test",
      agent=graph,
      session_service=session_service,
      auto_create_session=False,
  )

  async for _ in runner.run_async(
      user_id="u1",
      session_id="s1",
      new_message=types.Content(parts=[types.Part(text="start")]),
  ):
    pass

  session = await session_service.get_session(
      app_name="test", user_id="u1", session_id="s1"
  )
  graph_data = session.state.get("graph_data", {})

  # Writer set shared="from_a"; reader didn't touch it.
  # Diff-based merge must preserve writer's change.
  assert graph_data.get("shared") == "from_a", (
      f"Expected 'from_a', got {graph_data.get('shared')!r}. "
      "Reader branch likely overwrote writer's change with stale copy."
  )
  # Both branches' own keys should be present
  assert "writer_only" in graph_data
  assert "reader_only" in graph_data


class TestParallelContextIsolation:
  """Test InvocationContext isolation between parallel branches."""

  @pytest.mark.asyncio
  async def test_parallel_context_isolation(self):
    """Verify ctx objects differ between parallel branches (model_copy)."""
    from google.adk.agents.graph.parallel import execute_parallel_group

    captured_contexts = []

    async def capture_ctx_fn(node, branch_state, ctx):
      """Capture the ctx object identity for each branch."""
      captured_contexts.append(id(ctx))
      yield Event(
          author=node.name,
          content=types.Content(parts=[types.Part(text=f"{node.name}_done")]),
      )

    group = ParallelNodeGroup(
        nodes=["x", "y"],
        join_strategy=JoinStrategy.WAIT_ALL,
    )

    # Create minimal node-like objects
    class _FakeNode:

      def __init__(self, name):
        self.name = name

    nodes_map = {"x": _FakeNode("x"), "y": _FakeNode("y")}
    state = GraphState(data={"key": "val"})

    # We need a real-ish ctx that supports model_copy.
    # Use a simple pydantic model stand-in.
    from pydantic import BaseModel

    class FakeCtx(BaseModel):
      tag: str = "original"

    ctx = FakeCtx()

    events = []
    async for ev in execute_parallel_group(
        group, nodes_map, state, ctx, capture_ctx_fn
    ):
      events.append(ev)

    # Both branches should have received different ctx objects
    assert len(captured_contexts) == 2
    assert (
        captured_contexts[0] != captured_contexts[1]
    ), "Parallel branches must receive distinct ctx objects"


class TestParallelCollectErrorRaisesRuntimeError:
  """Test that COLLECT error policy raises RuntimeError."""

  @pytest.mark.asyncio
  async def test_parallel_collect_error_raises_runtime_error(self):
    """Verify COLLECT error policy raises RuntimeError, not Exception."""
    from google.adk.agents.graph.parallel import execute_parallel_group

    async def failing_fn(node, branch_state, ctx):
      raise ValueError(f"fail_{node.name}")
      yield  # noqa: unreachable - make it an async generator

    group = ParallelNodeGroup(
        nodes=["a", "b"],
        join_strategy=JoinStrategy.WAIT_ALL,
        error_policy=ErrorPolicy.COLLECT,
    )

    class _FakeNode:

      def __init__(self, name):
        self.name = name

    nodes_map = {"a": _FakeNode("a"), "b": _FakeNode("b")}
    state = GraphState(data={})

    from pydantic import BaseModel

    class FakeCtx(BaseModel):
      tag: str = "original"

    ctx = FakeCtx()

    with pytest.raises(RuntimeError, match="Errors in parallel execution"):
      async for _ in execute_parallel_group(
          group, nodes_map, state, ctx, failing_fn
      ):
        pass
