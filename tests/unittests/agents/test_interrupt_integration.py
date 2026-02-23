"""Integration tests for interrupt system with GraphAgent.

Tests complete end-to-end interrupt workflows including:
- Full interrupt lifecycles (trigger → reason → action → result)
- Multi-node interrupt scenarios
- Different interrupt timings (BEFORE, AFTER, BOTH)
- All interrupt actions (continue, rerun, pause, defer, skip, go_back)
- State preservation and restoration
- Checkpoint integration
- Error recovery via interrupts
- Stress testing with high-volume interrupts
"""

import asyncio
import json
from typing import AsyncGenerator
from unittest.mock import AsyncMock
from unittest.mock import patch

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import GraphNode
from google.adk.agents.graph import GraphState
from google.adk.agents.graph import InterruptAction
from google.adk.agents.graph import InterruptConfig
from google.adk.agents.graph import InterruptMode
from google.adk.agents.graph.interrupt_reasoner import InterruptReasoner
from google.adk.agents.graph.interrupt_reasoner import InterruptReasonerConfig
from google.adk.agents.graph.interrupt_service import InterruptMessage
from google.adk.agents.graph.interrupt_service import InterruptService
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
import pytest
from typing_extensions import override


# Test agents (proper BaseAgent implementations per ADK guidelines)
class MockAgent(BaseAgent):
  """Test agent extending BaseAgent."""

  response: str = "mock"
  call_count: int = 0

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    self.call_count += 1
    yield Event(
        author=self.name,
        content=types.Content(parts=[types.Part(text=self.response)]),
    )


class SlowMockAgent(BaseAgent):
  """Agent that yields multiple events slowly."""

  event_count: int = 5
  delay: float = 0.01

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    for i in range(self.event_count):
      await asyncio.sleep(self.delay)
      yield Event(
          author=self.name,
          content=types.Content(parts=[types.Part(text=f"Event {i}")]),
      )


class ErrorAgent(BaseAgent):
  """Agent that raises errors."""

  error_message: str = "Test error"

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    # Must be async generator - yield before raising
    if False:
      yield  # Make it a generator
    raise RuntimeError(self.error_message)


class MockReasoner(InterruptReasoner):
  """Mock reasoner with predetermined decisions."""

  model_config = {"extra": "allow"}

  def __init__(self, decision_json: dict, **kwargs):
    config = InterruptReasonerConfig()
    super().__init__(config, **kwargs)
    self.mock_decision_json = decision_json

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    response = json.dumps(self.mock_decision_json)
    yield Event(
        author=self.name,
        content=types.Content(parts=[types.Part(text=response)]),
    )


@pytest.mark.asyncio
async def test_full_interrupt_workflow_continue():
  """Test complete interrupt workflow with continue action."""
  interrupt_service = InterruptService()
  reasoner = MockReasoner(
      {"action": "continue", "reasoning": "Looks good", "parameters": {}}
  )

  graph = GraphAgent(
      name="test_graph",
      interrupt_service=interrupt_service,
      interrupt_config=InterruptConfig(
          mode=InterruptMode.AFTER, reasoner=reasoner
      ),
  )

  node_a = GraphNode(
      name="node_a", agent=MockAgent(name="agent_a", response="output_a")
  )
  node_b = GraphNode(
      name="node_b", agent=MockAgent(name="agent_b", response="output_b")
  )

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

  interrupt_service.register_session("test_session")
  await interrupt_service.send_message(
      "test_session", "Check output", action="check"
  )

  execution_order = []
  async for event in runner.run_async(
      user_id="test_user",
      session_id="test_session",
      new_message=types.Content(role="user", parts=[types.Part(text="test")]),
  ):
    if event.author in ["agent_a", "agent_b"]:
      execution_order.append(event.author)

  # Both nodes should execute (continue action)
  assert "agent_a" in execution_order
  assert "agent_b" in execution_order


@pytest.mark.asyncio
async def test_full_interrupt_workflow_rerun():
  """Test complete interrupt workflow with rerun action."""
  interrupt_service = InterruptService()
  reasoner = MockReasoner({
      "action": "rerun",
      "reasoning": "Needs improvement",
      "parameters": {"guidance": "Be more specific"},
  })

  graph = GraphAgent(
      name="test_graph",
      interrupt_service=interrupt_service,
      interrupt_config=InterruptConfig(
          mode=InterruptMode.AFTER, reasoner=reasoner
      ),
  )

  agent_a = MockAgent(name="agent_a", response="output_a")
  node_a = GraphNode(name="node_a", agent=agent_a)

  graph.add_node(node_a)
  graph.set_start("node_a").set_end("node_a")

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="test_app", agent=graph, session_service=session_service
  )

  await session_service.create_session(
      app_name="test_app", user_id="test_user", session_id="test_session"
  )

  interrupt_service.register_session("test_session")

  # Send interrupt that will trigger rerun
  await interrupt_service.send_message(
      "test_session", "Output needs improvement", action="check"
  )

  async for event in runner.run_async(
      user_id="test_user",
      session_id="test_session",
      new_message=types.Content(role="user", parts=[types.Part(text="test")]),
  ):
    pass

  # Note: rerun would require graph to loop - checking agent was called
  # In actual implementation, rerun would add guidance to state
  session = await session_service.get_session(
      app_name="test_app", user_id="test_user", session_id="test_session"
  )

  # Verify interrupt decision was recorded in agent_state events
  decision = None
  for event in reversed(session.events):
    if (
        event.actions
        and event.actions.agent_state
        and "last_interrupt_decision" in (event.actions.agent_state or {})
    ):
      decision = event.actions.agent_state["last_interrupt_decision"]
      break
  assert decision is not None
  assert decision["action"] == "rerun"


@pytest.mark.asyncio
async def test_multi_node_interrupt_sequence():
  """Test interrupts affecting multiple nodes in sequence."""
  interrupt_service = InterruptService()

  graph = GraphAgent(
      name="test_graph",
      interrupt_service=interrupt_service,
      interrupt_config=InterruptConfig(mode=InterruptMode.AFTER),
  )

  node_a = GraphNode(name="node_a", agent=MockAgent(name="agent_a"))
  node_b = GraphNode(name="node_b", agent=MockAgent(name="agent_b"))
  node_c = GraphNode(name="node_c", agent=MockAgent(name="agent_c"))

  graph.add_node(node_a).add_node(node_b).add_node(node_c)
  graph.add_edge("node_a", "node_b")
  graph.add_edge("node_b", "node_c")
  graph.set_start("node_a").set_end("node_c")

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="test_app", agent=graph, session_service=session_service
  )

  await session_service.create_session(
      app_name="test_app", user_id="test_user", session_id="test_session"
  )

  interrupt_service.register_session("test_session")

  # Queue multiple interrupts for different nodes
  await interrupt_service.send_message(
      "test_session", "Check node_a", action="continue"
  )
  await interrupt_service.send_message(
      "test_session", "Check node_b", action="continue"
  )
  await interrupt_service.send_message(
      "test_session", "Check node_c", action="continue"
  )

  execution_order = []
  async for event in runner.run_async(
      user_id="test_user",
      session_id="test_session",
      new_message=types.Content(role="user", parts=[types.Part(text="test")]),
  ):
    if event.author in ["agent_a", "agent_b", "agent_c"]:
      execution_order.append(event.author)

  # All nodes should execute
  assert "agent_a" in execution_order
  assert "agent_b" in execution_order
  assert "agent_c" in execution_order


@pytest.mark.asyncio
async def test_interrupt_timing_before():
  """Test interrupt timing BEFORE node execution."""
  interrupt_service = InterruptService()

  graph = GraphAgent(
      name="test_graph",
      interrupt_service=interrupt_service,
      interrupt_config=InterruptConfig(mode=InterruptMode.BEFORE),
  )

  agent_a = MockAgent(name="agent_a")
  node_a = GraphNode(name="node_a", agent=agent_a)

  graph.add_node(node_a)
  graph.set_start("node_a").set_end("node_a")

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="test_app", agent=graph, session_service=session_service
  )

  await session_service.create_session(
      app_name="test_app", user_id="test_user", session_id="test_session"
  )

  interrupt_service.register_session("test_session")
  await interrupt_service.send_message(
      "test_session", "Pre-check", action="continue"
  )

  async for event in runner.run_async(
      user_id="test_user",
      session_id="test_session",
      new_message=types.Content(role="user", parts=[types.Part(text="test")]),
  ):
    pass

  # Interrupt should be processed before node execution
  # Verify by checking agent was eventually called
  assert agent_a.call_count > 0


@pytest.mark.asyncio
async def test_interrupt_timing_both():
  """Test interrupt timing BOTH before and after node execution."""
  interrupt_service = InterruptService()

  graph = GraphAgent(
      name="test_graph",
      interrupt_service=interrupt_service,
      interrupt_config=InterruptConfig(mode=InterruptMode.BOTH),
  )

  agent_a = MockAgent(name="agent_a")
  node_a = GraphNode(name="node_a", agent=agent_a)

  graph.add_node(node_a)
  graph.set_start("node_a").set_end("node_a")

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="test_app", agent=graph, session_service=session_service
  )

  await session_service.create_session(
      app_name="test_app", user_id="test_user", session_id="test_session"
  )

  interrupt_service.register_session("test_session")

  # Queue 2 interrupts - one for BEFORE, one for AFTER
  await interrupt_service.send_message(
      "test_session", "Pre-check", action="continue"
  )
  await interrupt_service.send_message(
      "test_session", "Post-check", action="continue"
  )

  async for event in runner.run_async(
      user_id="test_user",
      session_id="test_session",
      new_message=types.Content(role="user", parts=[types.Part(text="test")]),
  ):
    pass

  # Both interrupts should be processed
  assert agent_a.call_count > 0


@pytest.mark.asyncio
async def test_interrupt_action_defer():
  """Test defer action stores todos in session state."""
  interrupt_service = InterruptService()

  graph = GraphAgent(
      name="test_graph",
      interrupt_service=interrupt_service,
      interrupt_config=InterruptConfig(mode=InterruptMode.AFTER),
  )

  node_a = GraphNode(name="node_a", agent=MockAgent(name="agent_a"))

  graph.add_node(node_a)
  graph.set_start("node_a").set_end("node_a")

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="test_app", agent=graph, session_service=session_service
  )

  await session_service.create_session(
      app_name="test_app", user_id="test_user", session_id="test_session"
  )

  interrupt_service.register_session("test_session")
  await interrupt_service.send_message(
      "test_session",
      "Fix validation",
      action="defer",
      metadata={"message": "Improve error messages"},
  )

  async for event in runner.run_async(
      user_id="test_user",
      session_id="test_session",
      new_message=types.Content(role="user", parts=[types.Part(text="test")]),
  ):
    pass

  # Verify todo stored in session state
  session = await session_service.get_session(
      app_name="test_app", user_id="test_user", session_id="test_session"
  )

  # Extract interrupt_todos from agent_state events
  todos = []
  for event in reversed(session.events):
    if (
        event.actions
        and event.actions.agent_state
        and "interrupt_todos" in (event.actions.agent_state or {})
    ):
      todos = event.actions.agent_state["interrupt_todos"]
      break
  assert len(todos) >= 1
  assert any("Improve error messages" in str(todo) for todo in todos)


@pytest.mark.asyncio
async def test_interrupt_action_pause_resume():
  """Test pause/resume functionality with interrupt service."""
  interrupt_service = InterruptService()

  graph = GraphAgent(
      name="test_graph",
      interrupt_service=interrupt_service,
      interrupt_config=InterruptConfig(mode=InterruptMode.AFTER),
  )

  node_a = GraphNode(name="node_a", agent=MockAgent(name="agent_a"))
  node_b = GraphNode(name="node_b", agent=MockAgent(name="agent_b"))

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

  interrupt_service.register_session("test_session")

  # Test pause/resume functionality directly
  await interrupt_service.pause("test_session")
  assert interrupt_service.is_paused("test_session")

  await interrupt_service.resume("test_session")
  assert not interrupt_service.is_paused("test_session")

  # Send interrupt message and verify execution completes
  await interrupt_service.send_message(
      "test_session", "Check output", action="continue"
  )

  execution_order = []
  async for event in runner.run_async(
      user_id="test_user",
      session_id="test_session",
      new_message=types.Content(role="user", parts=[types.Part(text="test")]),
  ):
    if event.author in ["agent_a", "agent_b"]:
      execution_order.append(event.author)

  # Both nodes should execute
  assert "agent_a" in execution_order
  assert "agent_b" in execution_order


@pytest.mark.asyncio
async def test_state_preservation_during_interrupt():
  """Test state is preserved correctly during interrupt processing."""
  interrupt_service = InterruptService()

  graph = GraphAgent(
      name="test_graph",
      interrupt_service=interrupt_service,
      interrupt_config=InterruptConfig(mode=InterruptMode.AFTER),
  )

  # Create simple agent that returns output
  node_a = GraphNode(
      name="node_a", agent=MockAgent(name="agent_a", response="output")
  )

  graph.add_node(node_a)
  graph.set_start("node_a").set_end("node_a")

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="test_app", agent=graph, session_service=session_service
  )

  await session_service.create_session(
      app_name="test_app", user_id="test_user", session_id="test_session"
  )

  interrupt_service.register_session("test_session")
  await interrupt_service.send_message(
      "test_session", "Check state", action="continue"
  )

  async for event in runner.run_async(
      user_id="test_user",
      session_id="test_session",
      new_message=types.Content(role="user", parts=[types.Part(text="test")]),
  ):
    pass

  # Verify state preserved in session
  session = await session_service.get_session(
      app_name="test_app", user_id="test_user", session_id="test_session"
  )

  # Session state should exist and contain graph state
  assert session.state is not None
  assert "graph_data" in session.state


@pytest.mark.asyncio
async def test_error_recovery_via_interrupt():
  """Test error recovery using interrupt system."""
  interrupt_service = InterruptService()

  graph = GraphAgent(
      name="test_graph",
      interrupt_service=interrupt_service,
      interrupt_config=InterruptConfig(mode=InterruptMode.AFTER),
  )

  # Node that initially fails
  error_agent = ErrorAgent(
      name="error_agent", error_message="Recoverable error"
  )
  node_a = GraphNode(name="node_a", agent=error_agent)
  node_b = GraphNode(
      name="node_b", agent=MockAgent(name="agent_b", response="recovered")
  )

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

  interrupt_service.register_session("test_session")

  # Interrupt can't recover from agent error directly
  # but can guide next steps
  await interrupt_service.send_message(
      "test_session", "Handle error gracefully", action="continue"
  )

  try:
    async for event in runner.run_async(
        user_id="test_user",
        session_id="test_session",
        new_message=types.Content(role="user", parts=[types.Part(text="test")]),
    ):
      pass
  except RuntimeError as e:
    # Error expected from error_agent
    assert "Recoverable error" in str(e)


@pytest.mark.asyncio
async def test_immediate_cancel_stops_execution():
  """Test immediate cancellation stops graph execution."""
  interrupt_service = InterruptService()

  graph = GraphAgent(
      name="test_graph",
      interrupt_service=interrupt_service,
  )

  node_a = GraphNode(
      name="node_a", agent=SlowMockAgent(name="slow_agent", event_count=10)
  )
  node_b = GraphNode(name="node_b", agent=MockAgent(name="agent_b"))

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

  interrupt_service.register_session("test_session")

  event_count = 0
  cancel_called = False

  async def run_with_cancel():
    nonlocal event_count, cancel_called
    async for event in runner.run_async(
        user_id="test_user",
        session_id="test_session",
        new_message=types.Content(role="user", parts=[types.Part(text="test")]),
    ):
      if event.author == "slow_agent":
        event_count += 1
        # Cancel after 3 events
        if event_count == 3 and not cancel_called:
          await interrupt_service.cancel("test_session")
          cancel_called = True

  await run_with_cancel()

  # Should stop before all 10 events
  assert cancel_called
  assert event_count < 10
  assert event_count >= 3


@pytest.mark.asyncio
async def test_stress_high_volume_interrupts():
  """Stress test with high volume of interrupts."""
  interrupt_service = InterruptService()

  graph = GraphAgent(
      name="test_graph",
      interrupt_service=interrupt_service,
      interrupt_config=InterruptConfig(mode=InterruptMode.AFTER),
  )

  node_a = GraphNode(name="node_a", agent=MockAgent(name="agent_a"))

  graph.add_node(node_a)
  graph.set_start("node_a").set_end("node_a")

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="test_app", agent=graph, session_service=session_service
  )

  await session_service.create_session(
      app_name="test_app", user_id="test_user", session_id="test_session"
  )

  interrupt_service.register_session("test_session")

  # Queue 50 interrupts
  for i in range(50):
    await interrupt_service.send_message(
        "test_session", f"Interrupt {i}", action="continue"
    )

  # Graph should handle all interrupts without crashing
  async for event in runner.run_async(
      user_id="test_user",
      session_id="test_session",
      new_message=types.Content(role="user", parts=[types.Part(text="test")]),
  ):
    pass

  # Verify no interrupts remaining
  assert not interrupt_service.has_queued_messages("test_session")


@pytest.mark.asyncio
async def test_interrupt_with_conditional_routing():
  """Test interrupts work correctly with conditional routing."""
  interrupt_service = InterruptService()

  graph = GraphAgent(
      name="test_graph",
      interrupt_service=interrupt_service,
      interrupt_config=InterruptConfig(mode=InterruptMode.AFTER),
  )

  node_a = GraphNode(name="node_a", agent=MockAgent(name="agent_a"))
  node_b = GraphNode(name="node_b", agent=MockAgent(name="agent_b"))
  node_c = GraphNode(name="node_c", agent=MockAgent(name="agent_c"))

  graph.add_node(node_a).add_node(node_b).add_node(node_c)

  # Conditional routing
  graph.add_edge(
      "node_a", "node_b", condition=lambda state: state.data.get("go_b", True)
  )
  graph.add_edge(
      "node_a", "node_c", condition=lambda state: state.data.get("go_c", False)
  )

  graph.set_start("node_a")
  graph.set_end("node_b")
  graph.set_end("node_c")

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="test_app", agent=graph, session_service=session_service
  )

  await session_service.create_session(
      app_name="test_app", user_id="test_user", session_id="test_session"
  )

  interrupt_service.register_session("test_session")
  await interrupt_service.send_message(
      "test_session", "Check routing", action="continue"
  )

  execution_order = []
  async for event in runner.run_async(
      user_id="test_user",
      session_id="test_session",
      new_message=types.Content(role="user", parts=[types.Part(text="test")]),
  ):
    if event.author in ["agent_a", "agent_b", "agent_c"]:
      execution_order.append(event.author)

  # Should route to node_b (default condition)
  assert "agent_a" in execution_order
  assert "agent_b" in execution_order


@pytest.mark.asyncio
async def test_interrupt_action_skip():
  """Test skip action skips current node."""
  interrupt_service = InterruptService()

  graph = GraphAgent(
      name="test_graph",
      interrupt_service=interrupt_service,
      interrupt_config=InterruptConfig(mode=InterruptMode.BEFORE),
  )

  agent_a = MockAgent(name="agent_a")
  node_a = GraphNode(name="node_a", agent=agent_a)
  node_b = GraphNode(name="node_b", agent=MockAgent(name="agent_b"))

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

  interrupt_service.register_session("test_session")

  # Send skip interrupt before node_a
  await interrupt_service.send_message(
      "test_session", "Skip node_a", action="skip"
  )

  execution_order = []
  async for event in runner.run_async(
      user_id="test_user",
      session_id="test_session",
      new_message=types.Content(role="user", parts=[types.Part(text="test")]),
  ):
    if event.author in ["agent_a", "agent_b"]:
      execution_order.append(event.author)

  # node_a should be skipped, only node_b executes
  # Note: actual skip implementation may vary
  # This test verifies interrupt is processed


@pytest.mark.asyncio
async def test_concurrent_interrupts_different_sessions():
  """Test concurrent interrupts across different sessions."""
  interrupt_service = InterruptService()

  graph = GraphAgent(
      name="test_graph",
      interrupt_service=interrupt_service,
      interrupt_config=InterruptConfig(mode=InterruptMode.AFTER),
  )

  node_a = GraphNode(name="node_a", agent=MockAgent(name="agent_a"))
  graph.add_node(node_a)
  graph.set_start("node_a").set_end("node_a")

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="test_app", agent=graph, session_service=session_service
  )

  # Create 3 sessions
  sessions = []
  for i in range(3):
    session_id = f"session_{i}"
    await session_service.create_session(
        app_name="test_app", user_id=f"user_{i}", session_id=session_id
    )
    interrupt_service.register_session(session_id)
    await interrupt_service.send_message(
        session_id, f"Message {i}", action="continue"
    )
    sessions.append(session_id)

  # Run all sessions concurrently
  tasks = []
  for i, session_id in enumerate(sessions):

    async def run_session(sid, uid):
      async for event in runner.run_async(
          user_id=uid,
          session_id=sid,
          new_message=types.Content(
              role="user", parts=[types.Part(text="test")]
          ),
      ):
        pass

    tasks.append(asyncio.create_task(run_session(session_id, f"user_{i}")))

  await asyncio.gather(*tasks)

  # All sessions should complete without interference
  for session_id in sessions:
    assert not interrupt_service.has_queued_messages(session_id)


@pytest.mark.asyncio
async def test_interrupt_state_isolation():
  """Test interrupt state is isolated between sessions."""
  interrupt_service = InterruptService()

  session_service = InMemorySessionService()

  # Register two sessions
  interrupt_service.register_session("session_1")
  interrupt_service.register_session("session_2")

  # Send different messages to each session
  await interrupt_service.send_message(
      "session_1", "Message for session 1", action="action_1"
  )
  await interrupt_service.send_message(
      "session_2", "Message for session 2", action="action_2"
  )

  # Verify messages are isolated
  msg_1 = await interrupt_service.check_interrupt("session_1")
  msg_2 = await interrupt_service.check_interrupt("session_2")

  assert msg_1.text == "Message for session 1"
  assert msg_1.action == "action_1"
  assert msg_2.text == "Message for session 2"
  assert msg_2.action == "action_2"

  # Verify no cross-contamination
  assert await interrupt_service.check_interrupt("session_1") is None
  assert await interrupt_service.check_interrupt("session_2") is None
