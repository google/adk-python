"""Tests for GraphAgent parallel node execution.

Tests parallel execution following ParallelAgent patterns with:
- Concurrent node execution
- Join strategies (WAIT_ALL, WAIT_ANY, WAIT_N)
- Error policies (FAIL_FAST, CONTINUE, COLLECT)
- State isolation and merging
"""

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import END
from google.adk.agents.graph import ErrorPolicy
from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import GraphNode
from google.adk.agents.graph import GraphState
from google.adk.agents.graph import JoinStrategy
from google.adk.agents.graph import ParallelNodeGroup
from google.adk.agents.graph import START
from google.adk.agents.graph import StateReducer
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import pytest

from google import genai


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


class StatefulAgent(BaseAgent):
  """Agent that modifies state."""

  def __init__(self, name: str, state_key: str, state_value: str):
    super().__init__(name=name)
    self._state_key = state_key
    self._state_value = state_value

  async def _run_async_impl(self, ctx):
    ctx.session.state[self._state_key] = self._state_value
    yield Event(
        author=self.name,
        content=types.Content(
            parts=[
                types.Part(text=f"Set {self._state_key}={self._state_value}")
            ]
        ),
    )


class SlowAgent(BaseAgent):
  """Agent that simulates slow operation."""

  def __init__(self, name: str, delay_ms: int):
    super().__init__(name=name)
    self._delay_ms = delay_ms

  async def _run_async_impl(self, ctx):
    import asyncio

    await asyncio.sleep(self._delay_ms / 1000.0)
    yield Event(
        author=self.name,
        content=types.Content(
            parts=[types.Part(text=f"Completed after {self._delay_ms}ms")]
        ),
    )


class ErrorAgent(BaseAgent):
  """Agent that raises an error."""

  def __init__(self, name: str, error_message: str):
    super().__init__(name=name)
    self._error_message = error_message

  async def _run_async_impl(self, ctx):
    raise ValueError(self._error_message)
    yield  # Make it an async generator (unreachable but needed for type)


@pytest.fixture
async def session_service():
  """Create InMemorySessionService for tests."""
  return InMemorySessionService()


@pytest.mark.asyncio
async def test_parallel_basic(session_service):
  """Test basic parallel execution with WAIT_ALL."""
  # Create graph with parallel nodes
  graph = GraphAgent(name="test_graph")

  agent1 = SimpleAgent(name="fetch_user", output="user_data")
  agent2 = SimpleAgent(name="fetch_products", output="product_data")
  agent3 = SimpleAgent(name="process", output="processed")

  graph.add_node(GraphNode(name="fetch_user", agent=agent1))
  graph.add_node(GraphNode(name="fetch_products", agent=agent2))
  graph.add_node(GraphNode(name="process", agent=agent3))

  # Add parallel group
  graph.add_parallel_group(
      "fetch_group",
      ParallelNodeGroup(
          nodes=["fetch_user", "fetch_products"],
          join_strategy=JoinStrategy.WAIT_ALL,
      ),
  )

  # Setup edges: both fetch nodes -> process
  graph.add_edge("fetch_user", "process")
  graph.add_edge("fetch_products", "process")

  # Set fetch_user as start (but parallel group will execute both)
  graph.set_start("fetch_user")
  graph.set_end("process")

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

  # Verify both fetch nodes executed
  event_texts = [
      e.content.parts[0].text for e in events if e.content and e.content.parts
  ]
  assert "user_data" in event_texts
  assert "product_data" in event_texts
  assert "processed" in event_texts


@pytest.mark.asyncio
async def test_parallel_wait_any(session_service):
  """Test parallel execution with WAIT_ANY strategy."""
  graph = GraphAgent(name="test_graph")

  # Create agents with different speeds
  fast_agent = SlowAgent(name="fast", delay_ms=10)
  slow_agent = SlowAgent(name="slow", delay_ms=100)
  process_agent = SimpleAgent(name="process", output="done")

  graph.add_node(GraphNode(name="fast", agent=fast_agent))
  graph.add_node(GraphNode(name="slow", agent=slow_agent))
  graph.add_node(GraphNode(name="process", agent=process_agent))

  # Add parallel group with WAIT_ANY
  graph.add_parallel_group(
      "race_group",
      ParallelNodeGroup(
          nodes=["fast", "slow"], join_strategy=JoinStrategy.WAIT_ANY
      ),
  )

  graph.add_edge("fast", "process")
  graph.add_edge("slow", "process")

  graph.set_start("fast")
  graph.set_end("process")

  # Execute
  runner = Runner(
      app_name="test_app",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  events = []
  new_message = types.Content(parts=[types.Part(text="Start")])
  async for event in runner.run_async(
      user_id="u1", session_id="s2", new_message=new_message
  ):
    events.append(event)

  # Verify at least fast agent completed
  event_texts = [
      e.content.parts[0].text for e in events if e.content and e.content.parts
  ]
  assert "Completed after 10ms" in event_texts
  assert "done" in event_texts


@pytest.mark.asyncio
async def test_parallel_wait_n(session_service):
  """Test parallel execution with WAIT_N strategy."""
  graph = GraphAgent(name="test_graph")

  agent1 = SimpleAgent(name="task1", output="result1")
  agent2 = SimpleAgent(name="task2", output="result2")
  agent3 = SimpleAgent(name="task3", output="result3")
  process_agent = SimpleAgent(name="process", output="done")

  graph.add_node(GraphNode(name="task1", agent=agent1))
  graph.add_node(GraphNode(name="task2", agent=agent2))
  graph.add_node(GraphNode(name="task3", agent=agent3))
  graph.add_node(GraphNode(name="process", agent=process_agent))

  # Add parallel group with WAIT_N (wait for 2 out of 3)
  graph.add_parallel_group(
      "task_group",
      ParallelNodeGroup(
          nodes=["task1", "task2", "task3"],
          join_strategy=JoinStrategy.WAIT_N,
          wait_n=2,
      ),
  )

  graph.add_edge("task1", "process")
  graph.add_edge("task2", "process")
  graph.add_edge("task3", "process")

  graph.set_start("task1")
  graph.set_end("process")

  # Execute
  runner = Runner(
      app_name="test_app",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  events = []
  new_message = types.Content(parts=[types.Part(text="Start")])
  async for event in runner.run_async(
      user_id="u1", session_id="s3", new_message=new_message
  ):
    events.append(event)

  # Verify at least 2 tasks completed
  event_texts = [
      e.content.parts[0].text for e in events if e.content and e.content.parts
  ]
  results_found = sum(
      1 for text in event_texts if text in ["result1", "result2", "result3"]
  )
  assert results_found >= 2
  assert "done" in event_texts


@pytest.mark.asyncio
async def test_parallel_error_fail_fast(session_service):
  """Test parallel execution with FAIL_FAST error policy."""
  graph = GraphAgent(name="test_graph")

  good_agent = SimpleAgent(name="good", output="success")
  bad_agent = ErrorAgent(name="bad", error_message="Test error")

  graph.add_node(GraphNode(name="good", agent=good_agent))
  graph.add_node(GraphNode(name="bad", agent=bad_agent))

  # Add parallel group with FAIL_FAST
  graph.add_parallel_group(
      "mixed_group",
      ParallelNodeGroup(
          nodes=["good", "bad"], error_policy=ErrorPolicy.FAIL_FAST
      ),
  )

  # Both nodes are in parallel group, no edges needed between them
  # Just set start and end
  graph.set_start("good")
  graph.set_end("good")

  # Execute - should raise error
  runner = Runner(
      app_name="test_app",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  with pytest.raises(ValueError, match="Test error"):
    new_message = types.Content(parts=[types.Part(text="Start")])
    async for event in runner.run_async(
        user_id="u1", session_id="s4", new_message=new_message
    ):
      pass


@pytest.mark.asyncio
async def test_parallel_error_continue(session_service):
  """Test parallel execution with CONTINUE error policy."""
  graph = GraphAgent(name="test_graph")

  good_agent = SimpleAgent(name="good", output="success")
  bad_agent = ErrorAgent(name="bad", error_message="Test error")
  process_agent = SimpleAgent(name="process", output="processed")

  graph.add_node(GraphNode(name="good", agent=good_agent))
  graph.add_node(GraphNode(name="bad", agent=bad_agent))
  graph.add_node(GraphNode(name="process", agent=process_agent))

  # Add parallel group with CONTINUE
  graph.add_parallel_group(
      "mixed_group",
      ParallelNodeGroup(
          nodes=["good", "bad"], error_policy=ErrorPolicy.CONTINUE
      ),
  )

  graph.add_edge("good", "process")
  graph.add_edge("bad", "process")

  graph.set_start("good")
  graph.set_end("process")

  # Execute - should continue despite error
  runner = Runner(
      app_name="test_app",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  events = []
  new_message = types.Content(parts=[types.Part(text="Start")])
  async for event in runner.run_async(
      user_id="u1", session_id="s5", new_message=new_message
  ):
    events.append(event)

  # Verify good agent completed
  event_texts = [
      e.content.parts[0].text for e in events if e.content and e.content.parts
  ]
  assert "success" in event_texts
  assert "processed" in event_texts


@pytest.mark.asyncio
async def test_parallel_group_validation(session_service):
  """Test parallel group validation."""
  graph = GraphAgent(name="test_graph")

  agent1 = SimpleAgent(name="agent1", output="result1")
  graph.add_node(GraphNode(name="agent1", agent=agent1))

  # Try to add parallel group with non-existent node
  with pytest.raises(ValueError, match="not found in graph"):
    graph.add_parallel_group(
        "invalid_group",
        ParallelNodeGroup(nodes=["agent1", "nonexistent"]),
    )


@pytest.mark.asyncio
async def test_parallel_group_already_executed(session_service):
  """Test that parallel group routes correctly to merge node (executes merge once)."""
  graph = GraphAgent(name="test_graph")

  agent1 = StatefulAgent(name="agent1", state_key="count1", state_value="1")
  agent2 = StatefulAgent(name="agent2", state_key="count2", state_value="2")
  merge_agent = SimpleAgent(name="merge", output="merged")

  graph.add_node(GraphNode(name="agent1", agent=agent1))
  graph.add_node(GraphNode(name="agent2", agent=agent2))
  graph.add_node(GraphNode(name="merge", agent=merge_agent))

  # Add parallel group
  graph.add_parallel_group(
      "parallel_group",
      ParallelNodeGroup(nodes=["agent1", "agent2"]),
  )

  # Both parallel nodes route to merge
  graph.add_edge("agent1", "merge")
  graph.add_edge("agent2", "merge")

  graph.set_start("agent1")
  graph.set_end("merge")

  # Execute
  runner = Runner(
      app_name="test_app",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  events = []
  new_message = types.Content(parts=[types.Part(text="Start")])
  async for event in runner.run_async(
      user_id="u1", session_id="s6", new_message=new_message
  ):
    events.append(event)

  # Verify both agents executed in parallel
  event_texts = [
      e.content.parts[0].text for e in events if e.content and e.content.parts
  ]
  assert "Set count1=1" in event_texts
  assert "Set count2=2" in event_texts

  # CRITICAL: Merge should execute only ONCE after parallel group
  assert (
      event_texts.count("merged") == 1
  ), f"Expected merge to execute once, got {event_texts.count('merged')} times"


@pytest.mark.asyncio
async def test_parallel_wait_n_validation(session_service):
  """Test WAIT_N validation."""
  # wait_n cannot be greater than number of nodes
  with pytest.raises(
      ValueError, match="cannot be greater than number of nodes"
  ):
    ParallelNodeGroup(
        nodes=["agent1", "agent2"],
        join_strategy=JoinStrategy.WAIT_N,
        wait_n=3,
    )


@pytest.mark.asyncio
async def test_parallel_integration_with_conditional_routing(session_service):
  """Test parallel execution with conditional routing."""
  graph = GraphAgent(name="test_graph")

  # Create agents
  validate_agent = SimpleAgent(name="validate", output="valid")
  process1_agent = SimpleAgent(name="process1", output="result1")
  process2_agent = SimpleAgent(name="process2", output="result2")
  merge_agent = SimpleAgent(name="merge", output="merged")

  graph.add_node(GraphNode(name="validate", agent=validate_agent))
  graph.add_node(GraphNode(name="process1", agent=process1_agent))
  graph.add_node(GraphNode(name="process2", agent=process2_agent))
  graph.add_node(GraphNode(name="merge", agent=merge_agent))

  # Add parallel group for processing
  graph.add_parallel_group(
      "process_group",
      ParallelNodeGroup(nodes=["process1", "process2"]),
  )

  # Setup edges with conditional routing
  graph.add_edge("validate", "process1")
  graph.add_edge("validate", "process2")
  graph.add_edge("process1", "merge")
  graph.add_edge("process2", "merge")

  graph.set_start("validate")
  graph.set_end("merge")

  # Execute
  runner = Runner(
      app_name="test_app",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  events = []
  new_message = types.Content(parts=[types.Part(text="Start")])
  async for event in runner.run_async(
      user_id="u1", session_id="s7", new_message=new_message
  ):
    events.append(event)

  # Verify execution order: validate -> parallel (process1, process2) -> merge
  event_texts = [
      e.content.parts[0].text for e in events if e.content and e.content.parts
  ]
  assert "valid" in event_texts
  assert "result1" in event_texts
  assert "result2" in event_texts
  assert "merged" in event_texts

  # Validate should come before processing
  valid_idx = event_texts.index("valid")
  result1_idx = event_texts.index("result1")
  result2_idx = event_texts.index("result2")
  merge_idx = event_texts.index("merged")

  assert valid_idx < result1_idx
  assert valid_idx < result2_idx
  assert result1_idx < merge_idx
  assert result2_idx < merge_idx
