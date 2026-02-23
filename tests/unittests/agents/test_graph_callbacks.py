"""Test suite for GraphAgent callback infrastructure.

Tests callback-based observability and extensibility:
- NodeCallback (before/after node execution)
- EdgeCallback (on edge condition evaluation)
- Custom observability patterns
- Nested graph hierarchy tracking
"""

from typing import AsyncGenerator
from typing import Optional

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import create_nested_observability_callback
from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import GraphNode
from google.adk.agents.graph import NodeCallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
import pytest
from typing_extensions import override


# Mock agent for testing
class MockAgent(BaseAgent):
  _response: str

  def __init__(self, name: str, response: str = "mock", **kwargs):
    super().__init__(name=name, **kwargs)
    self._response = response

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield Event(
        author=self.name,
        content=types.Content(parts=[types.Part(text=self._response)]),
    )


@pytest.mark.asyncio
async def test_before_node_callback_invoked():
  """Test that before_node_callback is invoked before node execution."""
  callback_invocations = []

  async def before_callback(ctx: NodeCallbackContext) -> Optional[Event]:
    callback_invocations.append(("before", ctx.node.name))
    return Event(
        author="test",
        content=types.Content(
            parts=[types.Part(text=f"Before: {ctx.node.name}")]
        ),
    )

  graph = GraphAgent(name="test_graph", before_node_callback=before_callback)
  node_a = GraphNode(name="node_a", agent=MockAgent("agent_a", "output_a"))
  node_b = GraphNode(name="node_b", agent=MockAgent("agent_b", "output_b"))

  graph.add_node(node_a).add_node(node_b)
  graph.add_edge("node_a", "node_b")
  graph.set_start("node_a").set_end("node_b")

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="test_app", agent=graph, session_service=session_service
  )

  # Create session first
  await session_service.create_session(
      app_name="test_app", user_id="test_user", session_id="test_session"
  )

  events = []
  async for event in runner.run_async(
      user_id="test_user",
      session_id="test_session",
      new_message=types.Content(
          role="user", parts=[types.Part(text="test input")]
      ),
  ):
    events.append(event)

  # Verify callback was invoked for both nodes
  assert len(callback_invocations) == 2
  assert callback_invocations[0] == ("before", "node_a")
  assert callback_invocations[1] == ("before", "node_b")

  # Verify callback events were emitted
  before_events = [
      e
      for e in events
      if e.content and "Before:" in (e.content.parts[0].text or "")
  ]
  assert len(before_events) == 2


@pytest.mark.asyncio
async def test_after_node_callback_invoked():
  """Test that after_node_callback is invoked after node execution."""
  callback_invocations = []

  async def after_callback(ctx: NodeCallbackContext) -> Optional[Event]:
    callback_invocations.append(
        ("after", ctx.node.name, ctx.metadata.get("output"))
    )
    return Event(
        author="test",
        content=types.Content(
            parts=[types.Part(text=f"After: {ctx.node.name}")]
        ),
    )

  graph = GraphAgent(name="test_graph", after_node_callback=after_callback)
  node_a = GraphNode(name="node_a", agent=MockAgent("agent_a", "output_a"))
  graph.add_node(node_a)
  graph.set_start("node_a").set_end("node_a")

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="test_app", agent=graph, session_service=session_service
  )

  # Create session first
  await session_service.create_session(
      app_name="test_app", user_id="test_user", session_id="test_session"
  )

  events = []
  async for event in runner.run_async(
      user_id="test_user",
      session_id="test_session",
      new_message=types.Content(
          role="user", parts=[types.Part(text="test input")]
      ),
  ):
    events.append(event)

  # Verify callback was invoked with output
  assert len(callback_invocations) == 1
  assert callback_invocations[0][0] == "after"
  assert callback_invocations[0][1] == "node_a"
  assert callback_invocations[0][2] == "output_a"


@pytest.mark.asyncio
async def test_callback_returning_none_skips_event():
  """Test that callback returning None skips event emission."""
  callback_invocations = []

  async def selective_callback(ctx: NodeCallbackContext) -> Optional[Event]:
    callback_invocations.append(ctx.node.name)
    # Only emit for node_a
    if ctx.node.name == "node_a":
      return Event(
          author="test",
          content=types.Content(parts=[types.Part(text="Event")]),
      )
    return None  # Skip for node_b

  graph = GraphAgent(name="test_graph", before_node_callback=selective_callback)
  node_a = GraphNode(name="node_a", agent=MockAgent("agent_a"))
  node_b = GraphNode(name="node_b", agent=MockAgent("agent_b"))

  graph.add_node(node_a).add_node(node_b)
  graph.add_edge("node_a", "node_b")
  graph.set_start("node_a").set_end("node_b")

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="test_app", agent=graph, session_service=session_service
  )

  # Create session first
  await session_service.create_session(
      app_name="test_app", user_id="test_user", session_id="test_session"
  )

  events = []
  async for event in runner.run_async(
      user_id="test_user",
      session_id="test_session",
      new_message=types.Content(
          role="user", parts=[types.Part(text="test input")]
      ),
  ):
    events.append(event)

  # Callback invoked for both
  assert len(callback_invocations) == 2

  # But only one event emitted
  test_events = [e for e in events if e.author == "test"]
  assert len(test_events) == 1


@pytest.mark.asyncio
async def test_callback_has_full_context():
  """Test that callback receives full context including state and iteration."""
  captured_contexts = []

  async def capture_callback(ctx: NodeCallbackContext) -> Optional[Event]:
    captured_contexts.append({
        "node_name": ctx.node.name,
        "iteration": ctx.iteration,
        "state_data_keys": list(ctx.state.data.keys()),
        "agent_path": list(ctx.metadata.get("agent_path", [])),
        "path": list(ctx.metadata.get("path", [])),
    })
    return None

  graph = GraphAgent(name="test_graph", before_node_callback=capture_callback)
  node_a = GraphNode(name="node_a", agent=MockAgent("agent_a", "output_a"))
  node_b = GraphNode(name="node_b", agent=MockAgent("agent_b", "output_b"))

  graph.add_node(node_a).add_node(node_b)
  graph.add_edge("node_a", "node_b")
  graph.set_start("node_a").set_end("node_b")

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="test_app", agent=graph, session_service=session_service
  )

  # Create session first
  await session_service.create_session(
      app_name="test_app", user_id="test_user", session_id="test_session"
  )

  async for _ in runner.run_async(
      user_id="test_user",
      session_id="test_session",
      new_message=types.Content(
          role="user", parts=[types.Part(text="test input")]
      ),
  ):
    pass

  # Verify contexts
  assert len(captured_contexts) == 2

  # First node
  assert captured_contexts[0]["node_name"] == "node_a"
  assert captured_contexts[0]["iteration"] == 1
  assert "input" in captured_contexts[0]["state_data_keys"]
  assert captured_contexts[0]["agent_path"] == ["test_graph"]
  assert captured_contexts[0]["path"] == ["node_a"]

  # Second node
  assert captured_contexts[1]["node_name"] == "node_b"
  assert captured_contexts[1]["iteration"] == 2
  assert captured_contexts[1]["agent_path"] == ["test_graph"]
  assert captured_contexts[1]["path"] == ["node_a", "node_b"]


@pytest.mark.asyncio
async def test_nested_observability_callback():
  """Test create_nested_observability_callback shows hierarchy."""
  graph = GraphAgent(
      name="outer_graph",
      before_node_callback=create_nested_observability_callback(),
  )
  node_a = GraphNode(name="node_a", agent=MockAgent("agent_a"))
  graph.add_node(node_a)
  graph.set_start("node_a").set_end("node_a")

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="test_app", agent=graph, session_service=session_service
  )

  # Create session first
  await session_service.create_session(
      app_name="test_app", user_id="test_user", session_id="test_session"
  )

  events = []
  async for event in runner.run_async(
      user_id="test_user",
      session_id="test_session",
      new_message=types.Content(
          role="user", parts=[types.Part(text="test input")]
      ),
  ):
    events.append(event)

  # Find observability event
  obs_events = [e for e in events if e.author == "observability"]
  assert len(obs_events) == 1

  # Check hierarchy is shown
  event_text = obs_events[0].content.parts[0].text
  assert "outer_graph" in event_text
  assert "node_a" in event_text


@pytest.mark.asyncio
async def test_both_callbacks_invoked_in_order():
  """Test that before and after callbacks are invoked in correct order."""
  invocation_order = []

  async def before_callback(ctx: NodeCallbackContext) -> Optional[Event]:
    invocation_order.append(f"before_{ctx.node.name}")
    return None

  async def after_callback(ctx: NodeCallbackContext) -> Optional[Event]:
    invocation_order.append(f"after_{ctx.node.name}")
    return None

  graph = GraphAgent(
      name="test_graph",
      before_node_callback=before_callback,
      after_node_callback=after_callback,
  )
  node_a = GraphNode(name="node_a", agent=MockAgent("agent_a"))
  node_b = GraphNode(name="node_b", agent=MockAgent("agent_b"))

  graph.add_node(node_a).add_node(node_b)
  graph.add_edge("node_a", "node_b")
  graph.set_start("node_a").set_end("node_b")

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="test_app", agent=graph, session_service=session_service
  )

  # Create session first
  await session_service.create_session(
      app_name="test_app", user_id="test_user", session_id="test_session"
  )

  async for _ in runner.run_async(
      user_id="test_user",
      session_id="test_session",
      new_message=types.Content(
          role="user", parts=[types.Part(text="test input")]
      ),
  ):
    pass

  # Verify order: before_a, after_a, before_b, after_b
  assert invocation_order == [
      "before_node_a",
      "after_node_a",
      "before_node_b",
      "after_node_b",
  ]


@pytest.mark.asyncio
async def test_before_callback_error_is_caught_and_graph_continues():
  """Test that errors in before_node_callback are caught and graph continues."""
  callback_invocations = []

  async def failing_before_callback(
      ctx: NodeCallbackContext,
  ) -> Optional[Event]:
    callback_invocations.append(ctx.node.name)
    raise ValueError(f"Callback error for {ctx.node.name}")

  graph = GraphAgent(
      name="test_graph", before_node_callback=failing_before_callback
  )
  node_a = GraphNode(name="node_a", agent=MockAgent("agent_a", "output_a"))
  node_b = GraphNode(name="node_b", agent=MockAgent("agent_b", "output_b"))

  graph.add_node(node_a).add_node(node_b)
  graph.add_edge("node_a", "node_b")
  graph.set_start("node_a").set_end("node_b")

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="test_app", agent=graph, session_service=session_service
  )

  await session_service.create_session(
      app_name="test_app", user_id="test_user", session_id="test_session"
  )

  events = []
  # Graph should complete despite callback errors
  async for event in runner.run_async(
      user_id="test_user",
      session_id="test_session",
      new_message=types.Content(
          role="user", parts=[types.Part(text="test input")]
      ),
  ):
    events.append(event)

  # Callbacks were attempted for both nodes
  assert len(callback_invocations) == 2
  assert callback_invocations == ["node_a", "node_b"]

  # Graph execution completed (node outputs present)
  agent_events = [e for e in events if e.author in ["agent_a", "agent_b"]]
  assert len(agent_events) == 2


@pytest.mark.asyncio
async def test_after_callback_error_is_caught_and_graph_continues():
  """Test that errors in after_node_callback are caught and graph continues."""
  callback_invocations = []

  async def failing_after_callback(ctx: NodeCallbackContext) -> Optional[Event]:
    callback_invocations.append(ctx.node.name)
    raise RuntimeError(f"After callback error for {ctx.node.name}")

  graph = GraphAgent(
      name="test_graph", after_node_callback=failing_after_callback
  )
  node_a = GraphNode(name="node_a", agent=MockAgent("agent_a", "output_a"))
  node_b = GraphNode(name="node_b", agent=MockAgent("agent_b", "output_b"))

  graph.add_node(node_a).add_node(node_b)
  graph.add_edge("node_a", "node_b")
  graph.set_start("node_a").set_end("node_b")

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="test_app", agent=graph, session_service=session_service
  )

  await session_service.create_session(
      app_name="test_app", user_id="test_user", session_id="test_session"
  )

  events = []
  # Graph should complete despite callback errors
  async for event in runner.run_async(
      user_id="test_user",
      session_id="test_session",
      new_message=types.Content(
          role="user", parts=[types.Part(text="test input")]
      ),
  ):
    events.append(event)

  # Callbacks were attempted for both nodes
  assert len(callback_invocations) == 2
  assert callback_invocations == ["node_a", "node_b"]

  # Graph execution completed
  agent_events = [e for e in events if e.author in ["agent_a", "agent_b"]]
  assert len(agent_events) == 2


@pytest.mark.asyncio
async def test_both_callbacks_error_graph_still_completes():
  """Test that graph completes even when both callbacks raise errors."""
  before_invocations = []
  after_invocations = []

  async def failing_before(ctx: NodeCallbackContext) -> Optional[Event]:
    before_invocations.append(ctx.node.name)
    raise ValueError("Before error")

  async def failing_after(ctx: NodeCallbackContext) -> Optional[Event]:
    after_invocations.append(ctx.node.name)
    raise ValueError("After error")

  graph = GraphAgent(
      name="test_graph",
      before_node_callback=failing_before,
      after_node_callback=failing_after,
  )
  node_a = GraphNode(name="node_a", agent=MockAgent("agent_a", "output_a"))
  graph.add_node(node_a)
  graph.set_start("node_a").set_end("node_a")

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="test_app", agent=graph, session_service=session_service
  )

  await session_service.create_session(
      app_name="test_app", user_id="test_user", session_id="test_session"
  )

  events = []
  async for event in runner.run_async(
      user_id="test_user",
      session_id="test_session",
      new_message=types.Content(
          role="user", parts=[types.Part(text="test input")]
      ),
  ):
    events.append(event)

  # Both callbacks were attempted
  assert len(before_invocations) == 1
  assert len(after_invocations) == 1

  # Graph execution still completed
  agent_events = [e for e in events if e.author == "agent_a"]
  assert len(agent_events) == 1


@pytest.mark.asyncio
async def test_callback_error_is_logged(caplog):
  """Test that callback errors are logged with details."""
  import logging

  async def failing_callback(ctx: NodeCallbackContext) -> Optional[Event]:
    raise ValueError("Intentional callback error")

  graph = GraphAgent(name="test_graph", before_node_callback=failing_callback)
  node_a = GraphNode(name="node_a", agent=MockAgent("agent_a"))
  graph.add_node(node_a)
  graph.set_start("node_a").set_end("node_a")

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="test_app", agent=graph, session_service=session_service
  )

  await session_service.create_session(
      app_name="test_app", user_id="test_user", session_id="test_session"
  )

  with caplog.at_level(logging.ERROR):
    async for _ in runner.run_async(
        user_id="test_user",
        session_id="test_session",
        new_message=types.Content(
            role="user", parts=[types.Part(text="test input")]
        ),
    ):
      pass

  # Verify error was logged
  assert len(caplog.records) > 0
  error_logs = [r for r in caplog.records if r.levelname == "ERROR"]
  assert len(error_logs) == 1
  assert "before_node_callback failed" in error_logs[0].message
  assert "node_a" in error_logs[0].message
  assert "Intentional callback error" in error_logs[0].message


@pytest.mark.asyncio
async def test_partial_callback_errors_do_not_affect_successful_callbacks():
  """Test that errors in one node's callback don't affect other nodes."""
  invocations = []

  async def selective_failing_callback(
      ctx: NodeCallbackContext,
  ) -> Optional[Event]:
    invocations.append(ctx.node.name)
    if ctx.node.name == "node_a":
      raise ValueError("Error for node_a")
    # node_b succeeds
    return Event(
        author="callback",
        content=types.Content(
            parts=[types.Part(text=f"Success: {ctx.node.name}")]
        ),
    )

  graph = GraphAgent(
      name="test_graph", before_node_callback=selective_failing_callback
  )
  node_a = GraphNode(name="node_a", agent=MockAgent("agent_a"))
  node_b = GraphNode(name="node_b", agent=MockAgent("agent_b"))

  graph.add_node(node_a).add_node(node_b)
  graph.add_edge("node_a", "node_b")
  graph.set_start("node_a").set_end("node_b")

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="test_app", agent=graph, session_service=session_service
  )

  await session_service.create_session(
      app_name="test_app", user_id="test_user", session_id="test_session"
  )

  events = []
  async for event in runner.run_async(
      user_id="test_user",
      session_id="test_session",
      new_message=types.Content(
          role="user", parts=[types.Part(text="test input")]
      ),
  ):
    events.append(event)

  # Both callbacks were attempted
  assert invocations == ["node_a", "node_b"]

  # Only node_b's callback event was emitted (node_a failed)
  callback_events = [e for e in events if e.author == "callback"]
  assert len(callback_events) == 1
  assert "Success: node_b" in callback_events[0].content.parts[0].text


@pytest.mark.asyncio
async def test_parallel_trigger_node_fires_both_callbacks():
  """Verify both before/after callbacks fire for parallel trigger nodes.

  The trigger node (the node that enters a parallel group) should NOT get
  before_node_callback (it's a parallel dispatch, not a regular execution)
  but SHOULD get after_node_callback after the parallel group completes.
  """
  from google.adk.agents.graph.parallel import JoinStrategy
  from google.adk.agents.graph.parallel import ParallelNodeGroup

  invocation_order = []

  async def before_callback(ctx: NodeCallbackContext) -> Optional[Event]:
    invocation_order.append(f"before_{ctx.node.name}")
    return None

  async def after_callback(ctx: NodeCallbackContext) -> Optional[Event]:
    invocation_order.append(f"after_{ctx.node.name}")
    return None

  graph = GraphAgent(
      name="test_graph",
      before_node_callback=before_callback,
      after_node_callback=after_callback,
  )

  # Two parallel nodes + a merge node
  node_a = GraphNode(name="node_a", agent=MockAgent("agent_a", "out_a"))
  node_b = GraphNode(name="node_b", agent=MockAgent("agent_b", "out_b"))
  node_merge = GraphNode(
      name="node_merge", agent=MockAgent("agent_merge", "merged")
  )

  graph.add_node(node_a).add_node(node_b).add_node(node_merge)
  graph.add_edge("node_a", "node_merge")
  graph.add_edge("node_b", "node_merge")
  graph.set_start("node_a").set_end("node_merge")

  graph.add_parallel_group(
      "pg",
      ParallelNodeGroup(
          nodes=["node_a", "node_b"],
          join_strategy=JoinStrategy.WAIT_ALL,
      ),
  )

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="test_app", agent=graph, session_service=session_service
  )
  await session_service.create_session(
      app_name="test_app", user_id="test_user", session_id="test_session"
  )

  async for _ in runner.run_async(
      user_id="test_user",
      session_id="test_session",
      new_message=types.Content(role="user", parts=[types.Part(text="test")]),
  ):
    pass

  # node_a is the trigger node for the parallel group.
  # It should get after_node_callback but NOT before_node_callback
  # (before_node_callback is now after the parallel group check).
  # node_b is in the already-executed group so it gets skipped entirely.
  # node_merge is a regular node and gets both callbacks.
  assert "after_node_a" in invocation_order, (
      "after_node_callback should fire for parallel trigger node. Got:"
      f" {invocation_order}"
  )
  assert "before_node_merge" in invocation_order
  assert "after_node_merge" in invocation_order


@pytest.mark.asyncio
async def test_before_callback_skipped_for_already_executed_parallel_nodes():
  """Verify callback does NOT fire for nodes in already-executed parallel groups.

  When node_b is visited after the parallel group already executed (via
  node_a as trigger), before_node_callback should NOT fire for node_b
  because the parallel group check `continue`s before reaching the callback.
  """
  from google.adk.agents.graph.parallel import JoinStrategy
  from google.adk.agents.graph.parallel import ParallelNodeGroup

  before_invocations = []

  async def before_callback(ctx: NodeCallbackContext) -> Optional[Event]:
    before_invocations.append(ctx.node.name)
    return None

  graph = GraphAgent(
      name="test_graph",
      before_node_callback=before_callback,
  )

  node_a = GraphNode(name="node_a", agent=MockAgent("agent_a", "out_a"))
  node_b = GraphNode(name="node_b", agent=MockAgent("agent_b", "out_b"))
  node_merge = GraphNode(
      name="node_merge", agent=MockAgent("agent_merge", "merged")
  )

  graph.add_node(node_a).add_node(node_b).add_node(node_merge)
  graph.add_edge("node_a", "node_merge")
  graph.add_edge("node_b", "node_merge")
  graph.set_start("node_a").set_end("node_merge")

  graph.add_parallel_group(
      "pg",
      ParallelNodeGroup(
          nodes=["node_a", "node_b"],
          join_strategy=JoinStrategy.WAIT_ALL,
      ),
  )

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="test_app", agent=graph, session_service=session_service
  )
  await session_service.create_session(
      app_name="test_app", user_id="test_user", session_id="test_session"
  )

  async for _ in runner.run_async(
      user_id="test_user",
      session_id="test_session",
      new_message=types.Content(role="user", parts=[types.Part(text="test")]),
  ):
    pass

  # before_node_callback should NOT have been called for node_a or node_b
  # (node_a triggers parallel group before callback; node_b is skipped
  # entirely via the already-executed check + continue).
  # Only node_merge should get before_node_callback.
  assert "node_a" not in before_invocations, (
      "before_node_callback should NOT fire for parallel trigger node. Got:"
      f" {before_invocations}"
  )
  assert "node_b" not in before_invocations, (
      "before_node_callback should NOT fire for already-executed parallel"
      f" node. Got: {before_invocations}"
  )
  assert "node_merge" in before_invocations, (
      "before_node_callback should fire for regular nodes. Got:"
      f" {before_invocations}"
  )
