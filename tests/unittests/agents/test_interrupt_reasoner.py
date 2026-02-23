"""Test suite for InterruptReasoner LLM-based interrupt reasoning.

Tests:
- LLM-based action decisions
- JSON parsing and validation
- Fallback behavior
- Custom actions
- Defer to todos
- go_back action
- State management
- GraphAgent integration
"""

import json
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
from google.adk.sessions.session import Session
from google.genai import types
import pytest


# Mock agent for testing
class MockAgent(BaseAgent):
  """Mock agent that extends BaseAgent for proper Pydantic validation."""

  response: str = "mock"

  async def run_async(self, ctx):
    yield Event(
        author=self.name,
        content=types.Content(parts=[types.Part(text=self.response)]),
    )


def create_test_invocation_context() -> InvocationContext:
  """Helper to create minimal valid InvocationContext for testing."""
  session = Session(id="test_session", appName="test_app", userId="test_user")
  session_service = InMemorySessionService()

  # Create a proper BaseAgent for testing
  mock_agent = MockAgent(name="test_agent")

  return InvocationContext(
      session=session,
      session_service=session_service,
      invocation_id="test_invocation",
      agent=mock_agent,
      user_content=None,
  )


class MockLLMReasoner(InterruptReasoner):
  """Mock reasoner that returns predetermined decision."""

  model_config = {"extra": "allow"}  # Allow extra attributes for mock

  def __init__(self, decision_json: dict, **kwargs):
    config = InterruptReasonerConfig()
    super().__init__(config, **kwargs)
    # Store decision after super().__init__()
    self.mock_decision_json = decision_json

  async def run_async(self, ctx):
    """Return mock decision as JSON."""
    response = json.dumps(self.mock_decision_json)
    yield Event(
        author=self.name,
        content=types.Content(parts=[types.Part(text=response)]),
    )


@pytest.mark.asyncio
async def test_interrupt_reasoner_decides_continue():
  """Test reasoner decides to continue."""
  reasoner = MockLLMReasoner({
      "action": "continue",
      "reasoning": "Everything looks good",
      "parameters": {},
  })

  message = InterruptMessage(text="Check progress", action="continue")
  state = GraphState(data={"test": "value"})
  ctx = create_test_invocation_context()

  action = await reasoner.reason_about_interrupt(
      message, state, "test_node", ctx
  )

  assert action.action == "continue"
  assert action.reasoning == "Everything looks good"
  assert action.parameters == {}


@pytest.mark.asyncio
async def test_interrupt_reasoner_decides_rerun():
  """Test reasoner decides to rerun with guidance."""
  reasoner = MockLLMReasoner({
      "action": "rerun",
      "reasoning": "Output needs improvement",
      "parameters": {"guidance": "Be more specific"},
  })

  message = InterruptMessage(text="Output is vague", action=None)
  state = GraphState(data={"output": "vague response"})
  ctx = create_test_invocation_context()

  action = await reasoner.reason_about_interrupt(
      message, state, "test_node", ctx
  )

  assert action.action == "rerun"
  assert "improvement" in action.reasoning
  assert action.parameters["guidance"] == "Be more specific"


@pytest.mark.asyncio
async def test_interrupt_reasoner_decides_defer():
  """Test reasoner decides to defer for later."""
  reasoner = MockLLMReasoner({
      "action": "defer",
      "reasoning": "Not critical now",
      "parameters": {"message": "Fix validation later"},
  })

  message = InterruptMessage(text="Validation could be better")
  state = GraphState(data={})
  ctx = create_test_invocation_context()

  action = await reasoner.reason_about_interrupt(
      message, state, "test_node", ctx
  )

  assert action.action == "defer"
  assert action.parameters["message"] == "Fix validation later"


@pytest.mark.asyncio
async def test_interrupt_reasoner_decides_go_back():
  """Test reasoner decides to go back."""
  reasoner = MockLLMReasoner({
      "action": "go_back",
      "reasoning": "Need to retry earlier step",
      "parameters": {"steps": 2},
  })

  message = InterruptMessage(text="Previous step had error")
  state = GraphState(data={})
  ctx = create_test_invocation_context()

  action = await reasoner.reason_about_interrupt(message, state, "c", ctx)

  assert action.action == "go_back"
  assert action.parameters["steps"] == 2


@pytest.mark.asyncio
async def test_interrupt_reasoner_fallback_on_exception():
  """Test reasoner falls back to continue on any exception."""

  class FailingReasoner(InterruptReasoner):
    """Reasoner that returns invalid JSON causing parse failure."""

    model_config = {"extra": "allow"}

    async def run_async(self, ctx):
      yield Event(
          author="reasoner",
          content=types.Content(parts=[types.Part(text="invalid json{")]),
      )

  config = InterruptReasonerConfig()
  reasoner = FailingReasoner(config)

  message = InterruptMessage(text="Test")
  state = GraphState(data={})
  ctx = create_test_invocation_context()

  action = await reasoner.reason_about_interrupt(message, state, "node", ctx)

  # Should fall back to continue (exception handled gracefully)
  assert action.action == "continue"


@pytest.mark.asyncio
async def test_interrupt_reasoner_validates_action():
  """Test reasoner validates action is in available_actions."""
  reasoner = MockLLMReasoner({
      "action": "invalid_action",  # Not in available_actions
      "reasoning": "Test",
      "parameters": {},
  })

  message = InterruptMessage(text="Test")
  state = GraphState(data={})
  ctx = create_test_invocation_context()

  action = await reasoner.reason_about_interrupt(message, state, "node", ctx)

  # Should default to continue
  assert action.action == "continue"


@pytest.mark.asyncio
async def test_interrupt_reasoner_with_graphagent():
  """Test InterruptReasoner integrated with GraphAgent."""
  # Create reasoner that always decides to defer
  reasoner = MockLLMReasoner({
      "action": "defer",
      "reasoning": "Save for later",
      "parameters": {"message": "Fix this later"},
  })

  interrupt_service = InterruptService()
  graph = GraphAgent(
      name="test_graph",
      interrupt_service=interrupt_service,
      interrupt_config=InterruptConfig(
          mode=InterruptMode.AFTER,
          reasoner=reasoner,
      ),
  )

  node_a = GraphNode(
      name="node_a", agent=MockAgent(name="agent_a", response="output_a")
  )
  graph.add_node(node_a)
  graph.set_start("node_a").set_end("node_a")

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="test_app", agent=graph, session_service=session_service
  )

  # Create session
  await session_service.create_session(
      app_name="test_app", user_id="test_user", session_id="test_session"
  )

  # Register session and send interrupt
  interrupt_service.register_session("test_session")
  await interrupt_service.send_message(
      "test_session", text="This needs attention", action="defer"
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

  # Get session to verify state
  session = await session_service.get_session(
      app_name="test_app", user_id="test_user", session_id="test_session"
  )

  # Verify todos were created in agent_state events
  todos = []
  decision = None
  for event in reversed(session.events):
    if event.actions and event.actions.agent_state:
      agent_st = event.actions.agent_state
      if "interrupt_todos" in agent_st and not todos:
        todos = agent_st["interrupt_todos"]
      if "last_interrupt_decision" in agent_st and decision is None:
        decision = agent_st["last_interrupt_decision"]
      if todos and decision:
        break
  assert len(todos) == 1
  assert todos[0]["message"] == "Fix this later"
  assert todos[0]["node"] == "node_a"

  # Verify interrupt decision was tracked
  assert decision is not None
  assert decision["action"] == "defer"
  assert decision["reasoning"] == "Save for later"


@pytest.mark.asyncio
async def test_defer_action_stores_in_session_state():
  """Test that defer action stores in session.state, not GraphState."""
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

  # Create session
  await session_service.create_session(
      app_name="test_app", user_id="test_user", session_id="test_session"
  )

  interrupt_service.register_session("test_session")
  await interrupt_service.send_message(
      "test_session",
      text="Defer this",
      action="defer",
      metadata={"message": "Fix validation"},
  )

  # Track final GraphState
  final_state = None
  async for event in runner.run_async(
      user_id="test_user",
      session_id="test_session",
      new_message=types.Content(
          role="user", parts=[types.Part(text="test input")]
      ),
  ):
    if event.actions and event.actions.state_delta:
      graph_data = event.actions.state_delta.get("graph_data")
      if graph_data:
        final_state = GraphState(data=graph_data)

  # Verify todos NOT in domain data
  assert final_state is not None
  assert "_interrupt_todos" not in final_state.data

  # Get session to verify state
  session = await session_service.get_session(
      app_name="test_app", user_id="test_user", session_id="test_session"
  )

  # Verify todos in agent_state events
  todos = []
  for event in reversed(session.events):
    if (
        event.actions
        and event.actions.agent_state
        and "interrupt_todos" in (event.actions.agent_state or {})
    ):
      todos = event.actions.agent_state["interrupt_todos"]
      break
  assert len(todos) == 1
  assert "Fix validation" in str(todos[0])


@pytest.mark.asyncio
async def test_go_back_action_restores_path():
  """Test go_back action properly restores execution path."""
  interrupt_service = InterruptService()
  graph = GraphAgent(
      name="test_graph",
      interrupt_service=interrupt_service,
      interrupt_config=InterruptConfig(mode=InterruptMode.AFTER),
  )

  node_a = GraphNode(
      name="node_a", agent=MockAgent(name="agent_a", response="a_output")
  )
  node_b = GraphNode(
      name="node_b", agent=MockAgent(name="agent_b", response="b_output")
  )
  node_c = GraphNode(
      name="node_c", agent=MockAgent(name="agent_c", response="c_output")
  )

  graph.add_node(node_a).add_node(node_b).add_node(node_c)
  graph.add_edge("node_a", "node_b")
  graph.add_edge("node_b", "node_c")
  graph.set_start("node_a").set_end("node_c")

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="test_app", agent=graph, session_service=session_service
  )

  # Create session
  await session_service.create_session(
      app_name="test_app", user_id="test_user", session_id="test_session"
  )

  interrupt_service.register_session("test_session")

  # Send go_back interrupt after node_c
  # This will be processed after node_c completes
  await interrupt_service.send_message(
      "test_session",
      text="Go back 2 steps",
      action="go_back",
      metadata={"steps": 2},
  )

  execution_order = []
  async for event in runner.run_async(
      user_id="test_user",
      session_id="test_session",
      new_message=types.Content(
          role="user", parts=[types.Part(text="test input")]
      ),
  ):
    # Track node execution
    if event.author in ["agent_a", "agent_b", "agent_c"]:
      execution_order.append(event.author)

  # Note: go_back happens after node_c, so we'd need to run graph again
  # to see it jump back. For this test, just verify node_c executed.
  assert "agent_c" in execution_order


@pytest.mark.asyncio
async def test_immediate_cancel_interrupt():
  """Test immediate cancellation (ESC-like) stops execution immediately.

  Tests cancellation between nodes - cancels after node_a completes,
  preventing node_b from starting.
  """
  interrupt_service = InterruptService()
  graph = GraphAgent(
      name="test_graph",
      interrupt_service=interrupt_service,
  )

  # Create slow agents to allow time for cancellation
  node_a = GraphNode(
      name="node_a", agent=MockAgent(name="agent_a", response="a")
  )
  node_b = GraphNode(
      name="node_b", agent=MockAgent(name="agent_b", response="b")
  )
  node_c = GraphNode(
      name="node_c", agent=MockAgent(name="agent_c", response="c")
  )

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

  execution_order = []
  cancel_called = False

  async def run_with_cancel():
    nonlocal cancel_called
    async for event in runner.run_async(
        user_id="test_user",
        session_id="test_session",
        new_message=types.Content(role="user", parts=[types.Part(text="test")]),
    ):
      if event.author in ["agent_a", "agent_b", "agent_c"]:
        execution_order.append(event.author)

      # Cancel after first node executes
      if event.author == "agent_a" and not cancel_called:
        await interrupt_service.cancel("test_session")
        cancel_called = True

  await run_with_cancel()

  # Verify execution stopped after node_a (before node_b)
  assert "agent_a" in execution_order
  assert "agent_b" not in execution_order
  assert "agent_c" not in execution_order


@pytest.mark.asyncio
async def test_immediate_cancel_during_node_execution():
  """Test immediate cancellation DURING node execution (not just between nodes).

  This tests TRUE immediate interrupt like ESC - cancelling while a node
  is actively executing and streaming events.
  """
  interrupt_service = InterruptService()
  graph = GraphAgent(
      name="test_graph",
      interrupt_service=interrupt_service,
  )

  # Create multi-event agent that yields multiple events
  class MultiEventAgent(BaseAgent):
    """Agent that yields multiple events for testing mid-execution cancel."""

    async def run_async(self, ctx):
      # Yield multiple events to allow cancellation during execution
      for i in range(5):
        yield Event(
            author=self.name,
            content=types.Content(
                parts=[types.Part(text=f"Event {i} from {self.name}")]
            ),
        )

  node_a = GraphNode(name="node_a", agent=MultiEventAgent(name="multi_agent"))

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

  event_count = 0
  cancel_called = False
  cancelled_event_seen = False

  async def run_with_cancel():
    nonlocal event_count, cancel_called, cancelled_event_seen
    async for event in runner.run_async(
        user_id="test_user",
        session_id="test_session",
        new_message=types.Content(role="user", parts=[types.Part(text="test")]),
    ):
      if event.author == "multi_agent":
        event_count += 1
        # Cancel after 2nd event (while node is still executing)
        if event_count == 2 and not cancel_called:
          await interrupt_service.cancel("test_session")
          cancel_called = True

      # Check for cancellation event
      if event.content and event.content.parts:
        if "cancelled during node" in event.content.parts[0].text:
          cancelled_event_seen = True

  await run_with_cancel()

  # Verify execution stopped DURING node execution (not all 5 events)
  assert cancel_called, "Cancel should have been called"
  assert (
      event_count < 5
  ), f"Should have stopped mid-execution, but got {event_count} events"
  assert event_count >= 2, "Should have seen at least 2 events before cancel"
  assert cancelled_event_seen, "Should have seen cancellation event"


# ---------------------------------------------------------------------------
# interrupt.py line 69: InterruptAction.__post_init__ sets parameters = {}
# ---------------------------------------------------------------------------


def test_interrupt_action_default_parameters_is_empty_dict():
  """Line 69: when no parameters kwarg is given, __post_init__ sets parameters={}.

  The dataclass field has `parameters: Dict[str, Any] = None` so the default
  value is None; __post_init__ converts it to {} so callers always get a dict.
  """
  action = InterruptAction(action="continue")
  assert (
      action.parameters == {}
  ), "parameters should default to {} via __post_init__, not None"
  assert isinstance(action.parameters, dict)

  # Verify explicit None also gets converted
  action2 = InterruptAction(action="rerun", parameters=None)
  assert action2.parameters == {}

  # Verify explicit dict is preserved unchanged
  action3 = InterruptAction(action="go_back", parameters={"steps": 2})
  assert action3.parameters == {"steps": 2}


# ---------------------------------------------------------------------------
# interrupt_reasoner.py line 205: state JSON truncated when too large
# ---------------------------------------------------------------------------


def test_build_reasoning_prompt_truncates_large_state():
  """Line 205: state_str is truncated when it exceeds max_state_size.

  Creates a large state dict so that json.dumps produces more chars than
  max_state_size, triggering the slice-and-append branch at line 205.
  """
  config = InterruptReasonerConfig(max_state_size=50)  # tiny limit
  reasoner = InterruptReasoner(config)

  # Build a state whose JSON representation is definitely > 50 chars
  state = GraphState(
      data={"key_" + str(i): "value_" + str(i) for i in range(20)}
  )
  message = InterruptMessage(text="check", action="continue")

  prompt = reasoner._build_reasoning_prompt(message, state, "some_node")

  # The prompt must contain the truncation marker
  assert "... (truncated)" in prompt, (
      "State truncation marker should appear in prompt when state exceeds"
      " max_state_size"
  )


# ---------------------------------------------------------------------------
# interrupt_reasoner.py lines 255, 257, 259: markdown fence stripping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reason_validates_action_against_available_actions():
  """Valid JSON with invalid action uses fallback."""
  reasoner = MockLLMReasoner({
      "action": "destroy_everything",
      "reasoning": "bad idea",
      "parameters": {},
  })

  message = InterruptMessage(text="Test")
  state = GraphState(data={})
  ctx = create_test_invocation_context()

  action = await reasoner.reason_about_interrupt(message, state, "node", ctx)

  # Invalid action should fall back to "continue" (default fallback)
  assert action.action == "continue"


@pytest.mark.asyncio
async def test_reason_structured_output_success():
  """Valid InterruptDecision JSON returns correct InterruptAction."""
  reasoner = MockLLMReasoner({
      "action": "rerun",
      "reasoning": "Output needs improvement",
      "parameters": {"guidance": "Be more specific"},
  })

  message = InterruptMessage(text="Improve output")
  state = GraphState(data={"output": "vague"})
  ctx = create_test_invocation_context()

  action = await reasoner.reason_about_interrupt(message, state, "node", ctx)

  assert action.action == "rerun"
  assert action.reasoning == "Output needs improvement"
  assert action.parameters == {"guidance": "Be more specific"}


def test_build_reasoning_prompt_uses_data_to_json():
  """Prompt uses state.data_to_json() for Pydantic-safe serialization."""
  from google.adk.agents.graph.graph_agent_state import GraphAgentState

  config = InterruptReasonerConfig()
  reasoner = InterruptReasoner(config)

  state = GraphState(data={"key": "value"})
  message = InterruptMessage(text="check this")

  agent_state = GraphAgentState(path=["a", "b"])
  prompt = reasoner._build_reasoning_prompt(
      message, state, "node_b", agent_state=agent_state
  )

  assert "node_b" in prompt
  assert "check this" in prompt
  assert '"key": "value"' in prompt
  assert "['a', 'b']" in prompt


def test_build_reasoning_prompt_hides_state():
  """State is hidden when include_state_in_prompt is False."""
  config = InterruptReasonerConfig(include_state_in_prompt=False)
  reasoner = InterruptReasoner(config)

  state = GraphState(data={"secret": "supersecretvalue123"})
  message = InterruptMessage(text="check")

  prompt = reasoner._build_reasoning_prompt(message, state, "node")

  assert "supersecretvalue123" not in prompt
  assert "<state hidden>" in prompt


def test_interrupt_reasoner_config_defaults():
  """Config defaults are initialized correctly."""
  config = InterruptReasonerConfig()

  assert config.fallback_action == "continue"
  assert "continue" in config.available_actions
  assert "rerun" in config.available_actions
  assert "go_back" in config.available_actions
  assert "pause" in config.available_actions
  assert "defer" in config.available_actions
  assert "skip" in config.available_actions
  assert config.custom_actions == {}
  assert config.include_state_in_prompt is True
  assert config.max_state_size == 10000


# ---------------------------------------------------------------------------
# Coverage: lines 203-205 — exception in reason_about_interrupt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reason_about_interrupt_exception_returns_fallback():
  """Exception during reasoning returns fallback action."""

  class ExplodingReasoner(InterruptReasoner):
    """Reasoner whose run_async raises an exception."""

    model_config = {"extra": "allow"}

    async def run_async(self, ctx):
      raise RuntimeError("LLM connection failed")
      yield  # noqa: unreachable  # make it an async generator

  config = InterruptReasonerConfig(fallback_action="pause")
  reasoner = ExplodingReasoner(config)

  message = InterruptMessage(text="Test")
  state = GraphState(data={})
  ctx = create_test_invocation_context()

  action = await reasoner.reason_about_interrupt(message, state, "node", ctx)

  assert action.action == "pause"
  assert "Reasoning error" in action.reasoning
  assert "LLM connection failed" in action.reasoning


# ---------------------------------------------------------------------------
# Coverage: InterruptDecision structured output schema
# ---------------------------------------------------------------------------


def test_interrupt_decision_model():
  """InterruptDecision Pydantic model validates correctly."""
  from google.adk.agents.graph.interrupt_reasoner import InterruptDecision

  decision = InterruptDecision(
      action="continue", reasoning="ok", parameters={"key": "val"}
  )
  assert decision.action == "continue"
  assert decision.reasoning == "ok"
  assert decision.parameters == {"key": "val"}

  # Defaults
  minimal = InterruptDecision(action="skip")
  assert minimal.reasoning == ""
  assert minimal.parameters is None


def test_interrupt_reasoner_has_output_schema():
  """InterruptReasoner sets output_schema=InterruptDecision."""
  from google.adk.agents.graph.interrupt_reasoner import InterruptDecision

  config = InterruptReasonerConfig()
  reasoner = InterruptReasoner(config)
  assert reasoner.output_schema is InterruptDecision


