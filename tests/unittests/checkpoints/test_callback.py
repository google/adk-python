"""Tests for CheckpointCallback (agent-level) and GraphCheckpointCallback (node-level)."""

from dataclasses import dataclass
from typing import Any
from typing import AsyncGenerator
from typing import Dict
from unittest.mock import MagicMock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.graph.callbacks import NodeCallbackContext
from google.adk.agents.graph.checkpoint_callback import GraphCheckpointCallback
from google.adk.agents.graph.graph_state import GraphState
from google.adk.agents.invocation_context import InvocationContext
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.checkpoints import CheckpointCallback
from google.adk.checkpoints import CheckpointService
from google.adk.events.event import Event
from google.adk.sessions.in_memory_session_service import InMemorySessionService
import pytest


@pytest.fixture
async def session_service():
  """Create InMemorySessionService."""
  return InMemorySessionService()


@pytest.fixture
async def artifact_service():
  """Create InMemoryArtifactService."""
  return InMemoryArtifactService()


@pytest.fixture
async def checkpoint_service(session_service, artifact_service):
  """Create CheckpointService."""
  return CheckpointService(
      session_service=session_service,
      artifact_service=artifact_service,
  )


@pytest.fixture
async def session(session_service):
  """Create test session."""
  return await session_service.create_session(
      app_name="test_app",
      user_id="test_user",
      session_id="test_session",
  )


@pytest.fixture
def simple_agent():
  """Create simple test agent."""

  class TestAgent(BaseAgent):

    async def _run_async_impl(self, ctx):
      yield Event(author=self.name)

  return TestAgent(name="test_agent", description="Test agent")


class TestCheckpointCallback:
  """Test CheckpointCallback functionality."""

  @pytest.mark.asyncio
  async def test_before_agent_creates_checkpoint(
      self, checkpoint_service, session, simple_agent
  ):
    """Test that before_agent callback creates a checkpoint."""
    callback = CheckpointCallback(checkpoint_service)

    # Create invocation context
    ctx = InvocationContext(
        session=session,
        agent=simple_agent,
        session_service=checkpoint_service.session_service,
        invocation_id="test-invocation-1",
    )

    # Create callback context
    callback_context = CallbackContext(invocation_context=ctx)

    # Call before_agent
    result = await callback.before_agent(callback_context)

    # Should return None (doesn't override execution)
    assert result is None

    # Checkpoint should be created
    checkpoint_id = f"{session.id}-{simple_agent.name}-before"
    checkpoint = await checkpoint_service.get_checkpoint(
        session=session,
        checkpoint_id=checkpoint_id,
    )

    assert checkpoint is not None
    assert checkpoint.checkpoint_id == checkpoint_id
    assert checkpoint.agent_name == simple_agent.name
    assert "Before test_agent execution" in checkpoint.description

  @pytest.mark.asyncio
  async def test_after_agent_creates_checkpoint(
      self, checkpoint_service, session, simple_agent
  ):
    """Test that after_agent callback creates a checkpoint."""
    callback = CheckpointCallback(checkpoint_service)

    ctx = InvocationContext(
        session=session,
        agent=simple_agent,
        session_service=checkpoint_service.session_service,
        invocation_id="test-invocation-2",
    )

    callback_context = CallbackContext(invocation_context=ctx)

    # Call after_agent
    result = await callback.after_agent(callback_context)

    # Should return None
    assert result is None

    # Checkpoint should be created
    checkpoint_id = f"{session.id}-{simple_agent.name}-after"
    checkpoint = await checkpoint_service.get_checkpoint(
        session=session,
        checkpoint_id=checkpoint_id,
    )

    assert checkpoint is not None
    assert checkpoint.checkpoint_id == checkpoint_id
    assert "After test_agent execution" in checkpoint.description

  @pytest.mark.asyncio
  async def test_checkpoint_before_only(
      self, checkpoint_service, session, simple_agent
  ):
    """Test callback with checkpoint_before=True, checkpoint_after=False."""
    callback = CheckpointCallback(
        checkpoint_service,
        checkpoint_before=True,
        checkpoint_after=False,
    )

    ctx = InvocationContext(
        session=session,
        agent=simple_agent,
        session_service=checkpoint_service.session_service,
        invocation_id="test-invocation",
    )
    callback_context = CallbackContext(invocation_context=ctx)

    # Before should create checkpoint
    await callback.before_agent(callback_context)
    before_id = f"{session.id}-{simple_agent.name}-before"
    before_cp = await checkpoint_service.get_checkpoint(session, before_id)
    assert before_cp is not None

    # After should NOT create checkpoint
    await callback.after_agent(callback_context)
    after_id = f"{session.id}-{simple_agent.name}-after"
    # Should raise CheckpointNotFoundError (P0.4 fix)
    from google.adk.checkpoints.models import CheckpointNotFoundError

    with pytest.raises(CheckpointNotFoundError):
      await checkpoint_service.get_checkpoint(session, after_id)

  @pytest.mark.asyncio
  async def test_checkpoint_after_only(
      self, checkpoint_service, session, simple_agent
  ):
    """Test callback with checkpoint_before=False, checkpoint_after=True."""
    callback = CheckpointCallback(
        checkpoint_service,
        checkpoint_before=False,
        checkpoint_after=True,
    )

    ctx = InvocationContext(
        session=session,
        agent=simple_agent,
        session_service=checkpoint_service.session_service,
        invocation_id="test-invocation",
    )
    callback_context = CallbackContext(invocation_context=ctx)

    # Before should NOT create checkpoint
    await callback.before_agent(callback_context)
    before_id = f"{session.id}-{simple_agent.name}-before"
    # Should raise CheckpointNotFoundError (P0.4 fix)
    from google.adk.checkpoints.models import CheckpointNotFoundError

    with pytest.raises(CheckpointNotFoundError):
      await checkpoint_service.get_checkpoint(session, before_id)

    # After should create checkpoint
    await callback.after_agent(callback_context)
    after_id = f"{session.id}-{simple_agent.name}-after"
    after_cp = await checkpoint_service.get_checkpoint(session, after_id)
    assert after_cp is not None

  @pytest.mark.asyncio
  async def test_multiple_agents_same_session(
      self, checkpoint_service, session
  ):
    """Test checkpointing multiple different agents in same session."""

    class Agent1(BaseAgent):

      async def _run_async_impl(self, ctx):
        yield Event(author=self.name)

    class Agent2(BaseAgent):

      async def _run_async_impl(self, ctx):
        yield Event(author=self.name)

    agent1 = Agent1(name="agent1", description="First agent")
    agent2 = Agent2(name="agent2", description="Second agent")

    callback = CheckpointCallback(checkpoint_service)

    # Checkpoint agent1
    ctx1 = InvocationContext(
        session=session,
        agent=agent1,
        session_service=checkpoint_service.session_service,
        invocation_id="test-invocation-agent1",
    )
    callback_ctx1 = CallbackContext(invocation_context=ctx1)
    await callback.before_agent(callback_ctx1)

    # Checkpoint agent2
    ctx2 = InvocationContext(
        session=session,
        agent=agent2,
        session_service=checkpoint_service.session_service,
        invocation_id="test-invocation-agent2",
    )
    callback_ctx2 = CallbackContext(invocation_context=ctx2)
    await callback.before_agent(callback_ctx2)

    # Both checkpoints should exist
    cp1 = await checkpoint_service.get_checkpoint(
        session, f"{session.id}-agent1-before"
    )
    cp2 = await checkpoint_service.get_checkpoint(
        session, f"{session.id}-agent2-before"
    )

    assert cp1 is not None
    assert cp2 is not None
    assert cp1.agent_name == "agent1"
    assert cp2.agent_name == "agent2"

  @pytest.mark.asyncio
  async def test_callback_with_agent_execution(
      self, checkpoint_service, session_service, artifact_service
  ):
    """Test callback integration with actual agent execution."""

    class CounterAgent(BaseAgent):

      async def _run_async_impl(self, ctx):
        # Simulate agent work
        ctx.session.state["counter"] = ctx.session.state.get("counter", 0) + 1
        yield Event(author=self.name)

    agent = CounterAgent(name="counter", description="Counter agent")

    # Create callback
    checkpoint_callback = CheckpointCallback(
        CheckpointService(session_service, artifact_service)
    )

    # Set callbacks on agent
    agent.before_agent_callback = checkpoint_callback.before_agent
    agent.after_agent_callback = checkpoint_callback.after_agent

    # Create session
    session = await session_service.create_session(
        app_name="test",
        user_id="user",
        session_id="callback_test",
    )

    # Run agent
    ctx = InvocationContext(
        session=session,
        agent=agent,
        session_service=checkpoint_service.session_service,
        invocation_id="test-invocation-callback",
    )
    events = []
    async for event in agent.run_async(ctx):
      events.append(event)

    # Checkpoints should exist
    service = CheckpointService(session_service, artifact_service)
    response = await service.list_checkpoints(session)

    # Should have before and after checkpoints
    assert len(response.checkpoints) >= 2

    checkpoint_ids = [cp.checkpoint_id for cp in response.checkpoints]
    assert any("before" in cp_id for cp_id in checkpoint_ids)
    assert any("after" in cp_id for cp_id in checkpoint_ids)


def _make_node_ctx(
    session, node_name: str, iteration: int = 0
) -> NodeCallbackContext:
  """Helper: create a NodeCallbackContext for testing."""
  node = MagicMock()
  node.name = node_name

  invocation_context = MagicMock()
  invocation_context.session = session

  return NodeCallbackContext(
      node=node,
      state=GraphState(),
      iteration=iteration,
      invocation_context=invocation_context,
      metadata={},
  )


class TestCheckpointCallbackNodeLevel:
  """Tests for node-level before_node/after_node callbacks."""

  @pytest.mark.asyncio
  async def test_after_node_creates_checkpoint(
      self, checkpoint_service, session
  ):
    """after_node creates checkpoint for the executed node."""
    callback = GraphCheckpointCallback(
        checkpoint_service,
        checkpoint_before=False,
        checkpoint_after=True,
    )

    ctx = _make_node_ctx(session, "my_node", iteration=0)
    result = await callback.after_node(ctx)

    assert result is None  # No event emitted

    expected_id = f"{session.id}-my_node-0-after"
    checkpoint = await checkpoint_service.get_checkpoint(session, expected_id)
    assert checkpoint.agent_name == "my_node"
    assert "my_node" in checkpoint.description
    assert "iteration 0" in checkpoint.description

  @pytest.mark.asyncio
  async def test_before_node_creates_checkpoint(
      self, checkpoint_service, session
  ):
    """before_node creates checkpoint before node execution."""
    callback = GraphCheckpointCallback(
        checkpoint_service,
        checkpoint_before=True,
        checkpoint_after=False,
    )

    ctx = _make_node_ctx(session, "critical_node", iteration=1)
    result = await callback.before_node(ctx)

    assert result is None

    expected_id = f"{session.id}-critical_node-1-before"
    checkpoint = await checkpoint_service.get_checkpoint(session, expected_id)
    assert checkpoint.agent_name == "critical_node"

  @pytest.mark.asyncio
  async def test_selective_checkpoint_nodes_included(
      self, checkpoint_service, session
  ):
    """checkpoint_nodes filters which nodes get checkpointed."""
    callback = GraphCheckpointCallback(
        checkpoint_service,
        checkpoint_after=True,
        checkpoint_nodes={"node_a", "node_c"},  # only these
    )

    # node_a: should be checkpointed
    ctx_a = _make_node_ctx(session, "node_a", iteration=0)
    await callback.after_node(ctx_a)

    expected_a = f"{session.id}-node_a-0-after"
    cp_a = await checkpoint_service.get_checkpoint(session, expected_a)
    assert cp_a is not None

  @pytest.mark.asyncio
  async def test_selective_checkpoint_nodes_excluded(
      self, checkpoint_service, session
  ):
    """Nodes not in checkpoint_nodes are not checkpointed."""
    from google.adk.checkpoints.models import CheckpointNotFoundError

    callback = GraphCheckpointCallback(
        checkpoint_service,
        checkpoint_after=True,
        checkpoint_nodes={"node_a", "node_c"},  # node_b NOT included
    )

    # node_b: should NOT be checkpointed
    ctx_b = _make_node_ctx(session, "node_b", iteration=0)
    await callback.after_node(ctx_b)

    expected_b = f"{session.id}-node_b-0-after"
    with pytest.raises(CheckpointNotFoundError):
      await checkpoint_service.get_checkpoint(session, expected_b)

  @pytest.mark.asyncio
  async def test_checkpoint_nodes_none_checkpoints_all(
      self, checkpoint_service, session
  ):
    """checkpoint_nodes=None (default) checkpoints ALL nodes."""
    callback = GraphCheckpointCallback(
        checkpoint_service,
        checkpoint_after=True,
        checkpoint_nodes=None,  # all nodes
    )

    for node_name in ("alpha", "beta", "gamma"):
      ctx = _make_node_ctx(session, node_name, iteration=0)
      await callback.after_node(ctx)

    for node_name in ("alpha", "beta", "gamma"):
      expected_id = f"{session.id}-{node_name}-0-after"
      cp = await checkpoint_service.get_checkpoint(session, expected_id)
      assert cp is not None, f"Expected checkpoint for {node_name}"

  @pytest.mark.asyncio
  async def test_after_node_disabled(self, checkpoint_service, session):
    """checkpoint_after=False means after_node does nothing."""
    from google.adk.checkpoints.models import CheckpointNotFoundError

    callback = GraphCheckpointCallback(
        checkpoint_service,
        checkpoint_after=False,
    )

    ctx = _make_node_ctx(session, "some_node", iteration=0)
    await callback.after_node(ctx)

    expected_id = f"{session.id}-some_node-0-after"
    with pytest.raises(CheckpointNotFoundError):
      await checkpoint_service.get_checkpoint(session, expected_id)

  @pytest.mark.asyncio
  async def test_before_node_disabled(self, checkpoint_service, session):
    """checkpoint_before=False means before_node does nothing."""
    from google.adk.checkpoints.models import CheckpointNotFoundError

    callback = GraphCheckpointCallback(
        checkpoint_service,
        checkpoint_before=False,
    )

    ctx = _make_node_ctx(session, "some_node", iteration=0)
    await callback.before_node(ctx)

    expected_id = f"{session.id}-some_node-0-before"
    with pytest.raises(CheckpointNotFoundError):
      await checkpoint_service.get_checkpoint(session, expected_id)

  @pytest.mark.asyncio
  async def test_iteration_included_in_checkpoint_id(
      self, checkpoint_service, session
  ):
    """Iteration number is included in checkpoint ID for uniqueness."""
    callback = GraphCheckpointCallback(
        checkpoint_service,
        checkpoint_after=True,
    )

    # Same node, different iterations
    for iteration in (0, 1, 2):
      ctx = _make_node_ctx(session, "loop_node", iteration=iteration)
      await callback.after_node(ctx)

    for iteration in (0, 1, 2):
      expected_id = f"{session.id}-loop_node-{iteration}-after"
      cp = await checkpoint_service.get_checkpoint(session, expected_id)
      assert cp is not None, f"Expected checkpoint for iteration {iteration}"

  @pytest.mark.asyncio
  async def test_checkpoint_request_key_true_creates_checkpoint(
      self, checkpoint_service, session
  ):
    """LLM flag=True triggers an agent-requested checkpoint."""
    from google.adk.checkpoints.models import CheckpointNotFoundError

    callback = GraphCheckpointCallback(
        checkpoint_service,
        checkpoint_before=False,
        checkpoint_after=False,  # no automatic checkpoints
        checkpoint_request_key="analyzer.checkpoint_requested",
    )

    ctx = _make_node_ctx(session, "analyzer", iteration=0)
    ctx.state.data["analyzer"] = {
        "checkpoint_requested": True,
        "risk_level": "high",
    }
    await callback.after_node(ctx)

    requested_id = f"{session.id}-analyzer-0-requested"
    cp = await checkpoint_service.get_checkpoint(session, requested_id)
    assert cp is not None, "Expected agent-requested checkpoint"

    # No automatic 'after' checkpoint should exist
    after_id = f"{session.id}-analyzer-0-after"
    with pytest.raises(CheckpointNotFoundError):
      await checkpoint_service.get_checkpoint(session, after_id)

  @pytest.mark.asyncio
  async def test_checkpoint_request_key_false_skips(
      self, checkpoint_service, session
  ):
    """LLM flag=False does not create an agent-requested checkpoint."""
    from google.adk.checkpoints.models import CheckpointNotFoundError

    callback = GraphCheckpointCallback(
        checkpoint_service,
        checkpoint_before=False,
        checkpoint_after=False,
        checkpoint_request_key="analyzer.checkpoint_requested",
    )

    ctx = _make_node_ctx(session, "analyzer", iteration=0)
    ctx.state.data["analyzer"] = {
        "checkpoint_requested": False,
        "risk_level": "low",
    }
    await callback.after_node(ctx)

    requested_id = f"{session.id}-analyzer-0-requested"
    with pytest.raises(CheckpointNotFoundError):
      await checkpoint_service.get_checkpoint(session, requested_id)

  @pytest.mark.asyncio
  async def test_checkpoint_request_key_parses_json_string(
      self, checkpoint_service, session
  ):
    """Handles JSON string format (real LlmAgent output_schema format)."""
    import json

    callback = GraphCheckpointCallback(
        checkpoint_service,
        checkpoint_before=False,
        checkpoint_after=False,
        checkpoint_request_key="analyzer.checkpoint_requested",
    )

    ctx = _make_node_ctx(session, "analyzer", iteration=0)
    # LlmAgent with output_schema stores output as a JSON string, not a dict
    ctx.state.data["analyzer"] = json.dumps(
        {"checkpoint_requested": True, "risk_level": "high", "finding": "test"}
    )
    await callback.after_node(ctx)

    requested_id = f"{session.id}-analyzer-0-requested"
    cp = await checkpoint_service.get_checkpoint(session, requested_id)
    assert cp is not None, "Expected checkpoint from JSON-string state value"

  @pytest.mark.asyncio
  async def test_checkpoint_request_key_invalid_json_skips(
      self, checkpoint_service, session
  ):
    """Invalid JSON string in state is treated as empty dict; no checkpoint."""
    from google.adk.checkpoints.models import CheckpointNotFoundError

    callback = GraphCheckpointCallback(
        checkpoint_service,
        checkpoint_before=False,
        checkpoint_after=False,
        checkpoint_request_key="analyzer.checkpoint_requested",
    )

    ctx = _make_node_ctx(session, "analyzer", iteration=0)
    ctx.state.data["analyzer"] = "not-valid-json"
    await callback.after_node(ctx)

    requested_id = f"{session.id}-analyzer-0-requested"
    with pytest.raises(CheckpointNotFoundError):
      await checkpoint_service.get_checkpoint(session, requested_id)

  @pytest.mark.asyncio
  async def test_checkpoint_request_key_missing_field_skips(
      self, checkpoint_service, session
  ):
    """Missing checkpoint_requested field in state dict does not crash."""
    from google.adk.checkpoints.models import CheckpointNotFoundError

    callback = GraphCheckpointCallback(
        checkpoint_service,
        checkpoint_before=False,
        checkpoint_after=False,
        checkpoint_request_key="analyzer.checkpoint_requested",
    )

    ctx = _make_node_ctx(session, "analyzer", iteration=0)
    ctx.state.data["analyzer"] = {
        "risk_level": "high"
    }  # no checkpoint_requested
    await callback.after_node(ctx)

    requested_id = f"{session.id}-analyzer-0-requested"
    with pytest.raises(CheckpointNotFoundError):
      await checkpoint_service.get_checkpoint(session, requested_id)

  @pytest.mark.asyncio
  async def test_checkpoint_request_key_node_mismatch_skips(
      self, checkpoint_service, session
  ):
    """checkpoint_request_key only fires for the matching node name."""
    from google.adk.checkpoints.models import CheckpointNotFoundError

    callback = GraphCheckpointCallback(
        checkpoint_service,
        checkpoint_before=False,
        checkpoint_after=False,
        checkpoint_request_key="analyzer.checkpoint_requested",
    )

    # Node is "validator", key watches "analyzer"
    ctx = _make_node_ctx(session, "validator", iteration=0)
    ctx.state.data["analyzer"] = {"checkpoint_requested": True}
    await callback.after_node(ctx)

    with pytest.raises(CheckpointNotFoundError):
      await checkpoint_service.get_checkpoint(
          session, f"{session.id}-validator-0-requested"
      )

  @pytest.mark.asyncio
  async def test_checkpoint_request_key_with_checkpoint_after_creates_both(
      self, checkpoint_service, session
  ):
    """checkpoint_after=True and flag=True both produce distinct checkpoints."""
    callback = GraphCheckpointCallback(
        checkpoint_service,
        checkpoint_before=False,
        checkpoint_after=True,
        checkpoint_request_key="analyzer.checkpoint_requested",
    )

    ctx = _make_node_ctx(session, "analyzer", iteration=0)
    ctx.state.data["analyzer"] = {
        "checkpoint_requested": True,
        "risk_level": "high",
    }
    await callback.after_node(ctx)

    after_id = f"{session.id}-analyzer-0-after"
    requested_id = f"{session.id}-analyzer-0-requested"

    cp_after = await checkpoint_service.get_checkpoint(session, after_id)
    cp_requested = await checkpoint_service.get_checkpoint(
        session, requested_id
    )
    assert cp_after is not None
    assert cp_requested is not None
    assert cp_after.checkpoint_id != cp_requested.checkpoint_id

  @pytest.mark.asyncio
  async def test_before_node_selective_nodes_excluded(
      self, checkpoint_service, session
  ):
    """before_node respects checkpoint_nodes filter (covers line 204)."""
    from google.adk.checkpoints.models import CheckpointNotFoundError

    callback = GraphCheckpointCallback(
        checkpoint_service,
        checkpoint_before=True,
        checkpoint_nodes={"node_a"},  # node_b excluded
    )

    ctx = _make_node_ctx(session, "node_b", iteration=0)
    await callback.before_node(ctx)

    with pytest.raises(CheckpointNotFoundError):
      await checkpoint_service.get_checkpoint(
          session, f"{session.id}-node_b-0-before"
      )
