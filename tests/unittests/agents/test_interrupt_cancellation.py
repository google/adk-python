"""Tests for GraphAgent immediate cancellation with state persistence.

Tests the cancellation functionality where interrupt_service.cancel() is called
to immediately abort graph execution and save state for resume.
"""

import asyncio

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import GraphNode
from google.adk.agents.graph.interrupt import InterruptConfig
from google.adk.agents.graph.interrupt import InterruptMode
from google.adk.agents.graph.interrupt_service import InterruptService
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
import pytest


class SlowTestAgent(BaseAgent):
  """Test agent that takes time to execute (simulates real work)."""

  model_config = {"extra": "allow", "arbitrary_types_allowed": True}

  def __init__(self, name: str, response: str, delay: float = 0.2):
    super().__init__(name=name)
    object.__setattr__(self, "_response", response)
    object.__setattr__(self, "_delay", delay)

  async def _run_async_impl(self, ctx):
    """Slow agent implementation."""
    delay = object.__getattribute__(self, "_delay")
    response = object.__getattribute__(self, "_response")

    # Simulate slow operation
    await asyncio.sleep(delay)

    yield Event(
        author=self.name,
        content=types.Content(parts=[types.Part(text=response)]),
    )


@pytest.mark.asyncio
async def test_immediate_cancellation_saves_state():
  """Test that cancellation saves state for resume."""
  session_service = InMemorySessionService()
  interrupt_service = InterruptService()

  graph = GraphAgent(
      name="test_graph",
      interrupt_service=interrupt_service,
      interrupt_config=InterruptConfig(mode=InterruptMode.BOTH),
  )

  # Use slower agents to have time to cancel
  agent1 = SlowTestAgent("agent1", "step1", delay=0.1)
  agent2 = SlowTestAgent("agent2", "step2", delay=0.5)  # Longer delay for node2
  agent3 = SlowTestAgent("agent3", "step3", delay=0.1)

  graph.add_node(GraphNode(name="node1", agent=agent1))
  graph.add_node(GraphNode(name="node2", agent=agent2))
  graph.add_node(GraphNode(name="node3", agent=agent3))
  graph.add_edge("node1", "node2")
  graph.add_edge("node2", "node3")
  graph.set_start("node1")
  graph.set_end("node3")

  # Create session
  app_name = "test"
  user_id = "u1"
  session = await session_service.create_session(
      app_name=app_name, user_id=user_id
  )
  interrupt_service.register_session(session.id)

  # Cancel during node2 execution (after node1 completes)
  async def cancel_during_node2():
    await asyncio.sleep(0.3)  # Wait for node1 to complete, node2 to start
    await interrupt_service.cancel(session.id)

  cancel_task = asyncio.create_task(cancel_during_node2())

  # Execute
  runner = Runner(
      app_name=app_name,
      agent=graph,
      session_service=session_service,
      auto_create_session=False,
  )

  events = []
  try:
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=types.Content(parts=[types.Part(text="start")]),
    ):
      if event.content and event.content.parts:
        events.append(event.content.parts[0].text)
  except asyncio.CancelledError:
    pass  # Cancellation is expected

  await cancel_task

  # Verify node1 executed
  event_text = " ".join(events)
  assert "step1" in event_text

  # Verify state was saved for resume
  session = await session_service.get_session(
      app_name=app_name, user_id=user_id, session_id=session.id
  )

  # Check cancellation state flags
  assert (
      session.state.get("graph_cancelled") == True
  ), "graph_cancelled flag not set"
  assert (
      "graph_cancelled_at_node" in session.state
  ), "graph_cancelled_at_node not saved"
  assert (
      session.state.get("graph_can_resume") == True
  ), "graph_can_resume flag not set"

  # Verify partial execution state was saved
  assert "graph_cancelled" in session.state
  assert "graph_data" in session.state

  # Verify node2 was interrupted (not completed)
  cancelled_at = session.state.get("graph_cancelled_at_node")
  assert (
      cancelled_at == "node2"
  ), f"Should be cancelled at node2, got {cancelled_at}"


@pytest.mark.asyncio
async def test_cancellation_unregisters_session():
  """Test that cancellation properly cleans up session registration."""
  session_service = InMemorySessionService()
  interrupt_service = InterruptService()

  graph = GraphAgent(
      name="test_graph",
      interrupt_service=interrupt_service,
      interrupt_config=InterruptConfig(mode=InterruptMode.BOTH),
  )

  agent = SlowTestAgent("agent", "response", delay=0.5)
  graph.add_node(GraphNode(name="node", agent=agent))
  graph.set_start("node")
  graph.set_end("node")

  # Create session
  app_name = "test"
  user_id = "u1"
  session = await session_service.create_session(
      app_name=app_name, user_id=user_id
  )
  interrupt_service.register_session(session.id)

  # Verify registered
  assert interrupt_service.is_active(session.id)

  # Cancel immediately
  async def cancel_immediately():
    await asyncio.sleep(0.1)
    await interrupt_service.cancel(session.id)

  cancel_task = asyncio.create_task(cancel_immediately())

  # Execute
  runner = Runner(
      app_name=app_name,
      agent=graph,
      session_service=session_service,
      auto_create_session=False,
  )

  try:
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=types.Content(parts=[types.Part(text="start")]),
    ):
      pass
  except asyncio.CancelledError:
    pass

  await cancel_task

  # Session should still be registered (cancellation doesn't unregister)
  # It just sets the cancellation event
  assert (
      interrupt_service.is_active(session.id) == False
  )  # Cancelled means not active


@pytest.mark.asyncio
async def test_cancellation_clears_message_queue():
  """Test that cancellation clears pending interrupt messages."""
  session_service = InMemorySessionService()
  interrupt_service = InterruptService()

  graph = GraphAgent(
      name="test_graph",
      interrupt_service=interrupt_service,
      interrupt_config=InterruptConfig(mode=InterruptMode.BOTH),
  )

  agent = SlowTestAgent("agent", "response", delay=1.0)
  graph.add_node(GraphNode(name="node", agent=agent))
  graph.set_start("node")
  graph.set_end("node")

  # Create session
  app_name = "test"
  user_id = "u1"
  session = await session_service.create_session(
      app_name=app_name, user_id=user_id
  )
  interrupt_service.register_session(session.id)

  # Send some messages then cancel
  await interrupt_service.send_message(session_id=session.id, text="msg1")
  await interrupt_service.send_message(session_id=session.id, text="msg2")

  # Verify messages are queued
  status = interrupt_service.get_queue_status(session.id)
  assert status.queue_depth == 2

  # Cancel should clear queue
  await interrupt_service.cancel(session.id)

  # Verify queue is cleared
  status = interrupt_service.get_queue_status(session.id)
  assert status.queue_depth == 0, "Cancellation should clear message queue"
