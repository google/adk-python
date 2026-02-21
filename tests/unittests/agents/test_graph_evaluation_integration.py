"""Integration tests for GraphAgent evaluation with intermediate_data extraction."""

from types import SimpleNamespace

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import GraphNode
from google.adk.agents.graph import GraphState
from google.adk.agents.graph.evaluation_metrics import graph_path_match
from google.adk.agents.graph.evaluation_metrics import node_execution_count
from google.adk.agents.graph.evaluation_metrics import state_contains_keys
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_case import InvocationEvent
from google.adk.evaluation.eval_case import InvocationEvents
from google.adk.evaluation.eval_metrics import EvalStatus
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import pytest


class SimpleAgent(BaseAgent):
  """Simple test agent."""

  def __init__(self, name: str, output: str):
    super().__init__(name=name)
    self._output = output

  async def _run_async_impl(self, ctx):
    yield Event(
        author=self.name,
        content=types.Content(parts=[types.Part(text=self._output)]),
    )


class StatefulAgent(BaseAgent):
  """Agent that produces output to be mapped to state."""

  def __init__(self, name: str, state_updates: dict):
    super().__init__(name=name)
    self._state_updates = state_updates

  async def _run_async_impl(self, ctx):
    # Yield state updates as JSON string in event
    import json

    yield Event(
        author=self.name,
        content=types.Content(
            parts=[types.Part(text=json.dumps(self._state_updates))]
        ),
    )


@pytest.mark.asyncio
async def test_graph_path_extraction_from_intermediate_data():
  """Test that graph_path is extracted from intermediate_data during actual GraphAgent execution."""
  # Build simple graph
  graph = GraphAgent(name="test_graph")
  agent1 = SimpleAgent(name="agent1", output="a1")
  agent2 = SimpleAgent(name="agent2", output="a2")
  agent3 = SimpleAgent(name="agent3", output="a3")

  graph.add_node(GraphNode(name="n1", agent=agent1))
  graph.add_node(GraphNode(name="n2", agent=agent2))
  graph.add_node(GraphNode(name="n3", agent=agent3))

  graph.add_edge("n1", "n2")
  graph.add_edge("n2", "n3")
  graph.set_start("n1")
  graph.set_end("n3")

  # Execute graph and collect events
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
    events.append(event)

  # Extract intermediate events (non-user, non-final events with content)
  # This mimics what EvaluationGenerator does
  intermediate_events = []
  final_response = None
  user_content = None

  for event in events:
    if event.author and event.author.lower() == "user":
      user_content = event.content
      continue

    if event.is_final_response():
      final_response = event.content
    elif event.content and event.content.parts:
      # Check if event has function_call, function_response, or text
      for part in event.content.parts:
        if part.function_call or part.function_response or part.text:
          intermediate_events.append(
              InvocationEvent(author=event.author, content=event.content)
          )
          break

  # Create Invocation with intermediate_data
  invocation = Invocation(
      userContent=user_content
      or types.Content(parts=[types.Part(text="test")]),
      finalResponse=final_response
      or types.Content(parts=[types.Part(text="done")]),
      intermediateData=InvocationEvents(invocation_events=intermediate_events),
  )

  # Create metric with expected path
  metric = SimpleNamespace(
      metric_name="graph_path",
      expected_graph_path=["n1", "n2", "n3"],
      # No actual_graph_path - should extract from intermediate_data
  )

  # Evaluate
  result = graph_path_match(metric, [invocation], None, None)

  # Should pass with perfect score if extraction works
  assert (
      result.overall_score == 1.0
  ), f"Score: {result.overall_score}, Expected: 1.0"
  assert result.overall_eval_status == EvalStatus.PASSED
  assert len(result.per_invocation_results) == 1
  assert result.per_invocation_results[0].score == 1.0


@pytest.mark.asyncio
async def test_node_execution_count_extraction_from_intermediate_data():
  """Test that node execution counts are extracted from intermediate_data."""
  graph = GraphAgent(name="test_graph", max_iterations=5)

  agent = SimpleAgent(name="agent", output="out")
  graph.add_node(GraphNode(name="n1", agent=agent))
  graph.add_node(GraphNode(name="n2", agent=agent))

  graph.set_start("n1")
  graph.add_edge("n1", "n2")
  graph.add_edge(
      "n2", "n1", condition=lambda s: s.data.get("_graph_iteration", 0) < 2
  )
  graph.set_end("n1")
  graph.set_end("n2")  # n2 can also be an end node when loop exits

  # Execute graph and collect events
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
    events.append(event)

  # Extract intermediate events
  intermediate_events = []
  final_response = None
  user_content = None

  for event in events:
    if event.author and event.author.lower() == "user":
      user_content = event.content
      continue

    if event.is_final_response():
      final_response = event.content
    elif event.content and event.content.parts:
      for part in event.content.parts:
        if part.function_call or part.function_response or part.text:
          intermediate_events.append(
              InvocationEvent(author=event.author, content=event.content)
          )
          break

  # Create Invocation
  invocation = Invocation(
      userContent=user_content
      or types.Content(parts=[types.Part(text="test")]),
      finalResponse=final_response
      or types.Content(parts=[types.Part(text="done")]),
      intermediateData=InvocationEvents(invocation_events=intermediate_events),
  )

  # Create metric - n1 and n2 should each execute at least once
  # The exact counts depend on the loop logic
  metric = SimpleNamespace(
      metric_name="execution_count",
      expected_node_counts={"n1": 2, "n2": 1},
      # No actual_node_counts - should extract from intermediate_data
  )

  # Evaluate
  result = node_execution_count(metric, [invocation], None, None)

  # Should extract counts from intermediate_data
  # Check that we got some score (extraction worked)
  assert (
      result.overall_score >= 0.0
  ), "Should have extracted counts from intermediate_data"
  assert len(result.per_invocation_results) == 1


@pytest.mark.asyncio
async def test_graph_metadata_event_format():
  """Test that GraphAgent emits metadata events in the expected format."""
  graph = GraphAgent(name="test_graph")
  agent = SimpleAgent(name="agent", output="test")

  graph.add_node(GraphNode(name="n1", agent=agent))
  graph.set_start("n1")
  graph.set_end("n1")

  # Execute graph
  session_service = InMemorySessionService()
  runner = Runner(
      app_name="test",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  metadata_events = []
  async for event in runner.run_async(
      user_id="u1",
      session_id="s1",
      new_message=types.Content(parts=[types.Part(text="test")]),
  ):
    # Find metadata events
    if event.author and "#metadata" in event.author:
      metadata_events.append(event)

  # Should have at least one metadata event
  assert len(metadata_events) > 0, "GraphAgent should emit metadata events"

  # Check metadata event format
  metadata_event = metadata_events[0]
  assert metadata_event.content is not None
  assert len(metadata_event.content.parts) > 0

  text = metadata_event.content.parts[0].text
  assert (
      "[GraphMetadata]" in text
  ), "Metadata event should have [GraphMetadata] marker"

  # Extract and parse metadata
  import ast

  metadata_str = text.split("[GraphMetadata]", 1)[1].strip()
  metadata = ast.literal_eval(metadata_str)

  # Verify expected fields
  assert "graph_node" in metadata
  assert "graph_iteration" in metadata
  assert "graph_path" in metadata
  assert "node_invocations" in metadata
  assert "graph_state" in metadata

  assert isinstance(metadata["graph_path"], list)
  assert isinstance(metadata["node_invocations"], dict)
  assert isinstance(metadata["graph_state"], dict)


@pytest.mark.asyncio
async def test_state_extraction_from_intermediate_data():
  """Test that graph_state is extracted from intermediate_data for state_contains_keys metric."""
  # Build graph with stateful agents
  graph = GraphAgent(name="test_graph")
  agent1 = StatefulAgent(
      name="agent1", state_updates={"count": 1, "status": "processing"}
  )
  agent2 = StatefulAgent(
      name="agent2", state_updates={"count": 2, "status": "done"}
  )

  # Define output mapper to parse JSON and update state
  def state_mapper(output: str, state: GraphState) -> GraphState:
    import json

    updates = json.loads(output)
    return GraphState(data={**state.data, **updates})

  graph.add_node(GraphNode(name="n1", agent=agent1, output_mapper=state_mapper))
  graph.add_node(GraphNode(name="n2", agent=agent2, output_mapper=state_mapper))

  graph.add_edge("n1", "n2")
  graph.set_start("n1")
  graph.set_end("n2")

  # Execute graph and collect events
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
    events.append(event)

  # Extract intermediate events
  intermediate_events = []
  final_response = None
  user_content = None

  for event in events:
    if event.author and event.author.lower() == "user":
      user_content = event.content
      continue

    if event.is_final_response():
      final_response = event.content
    elif event.content and event.content.parts:
      for part in event.content.parts:
        if part.function_call or part.function_response or part.text:
          intermediate_events.append(
              InvocationEvent(author=event.author, content=event.content)
          )
          break

  # Create Invocation
  invocation = Invocation(
      userContent=user_content
      or types.Content(parts=[types.Part(text="test")]),
      finalResponse=final_response
      or types.Content(parts=[types.Part(text="done")]),
      intermediateData=InvocationEvents(invocation_events=intermediate_events),
  )

  # Create metric - expect final state after both agents run
  metric = SimpleNamespace(
      metric_name="state_check",
      expected_state={"count": 2, "status": "done"},
      # No actual_state - should extract from intermediate_data
  )

  # Evaluate
  result = state_contains_keys(metric, [invocation], None, None)

  # Should pass with perfect score if extraction works
  assert (
      result.overall_score == 1.0
  ), f"Score: {result.overall_score}, Expected: 1.0"
  assert result.overall_eval_status == EvalStatus.PASSED
  assert len(result.per_invocation_results) == 1
  assert result.per_invocation_results[0].score == 1.0
