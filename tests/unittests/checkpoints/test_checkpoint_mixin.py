"""Tests for CheckpointableMixin."""

from google.adk.agents.base_agent import BaseAgent
from google.adk.checkpoints import CheckpointableMixin
from google.adk.checkpoints import CheckpointService
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import pytest


class CheckpointableTestAgent(CheckpointableMixin, BaseAgent):
  """Test agent with checkpointing capabilities."""

  def __init__(
      self,
      name: str,
      checkpoint_service=None,
      auto_checkpoint=False,
      output="test output",
  ):
    # Initialize BaseAgent first
    BaseAgent.__init__(self, name=name)
    # Then initialize mixin
    CheckpointableMixin.__init__(
        self,
        checkpoint_service=checkpoint_service,
        auto_checkpoint=auto_checkpoint,
        checkpoint_before=True,
        checkpoint_after=True,
    )
    self._output = output

  async def _run_async_impl(self, ctx):
    # Create checkpoint before execution (if auto)
    await self._checkpoint_before_execution(ctx.session)

    # Do work
    ctx.session.state["agent_ran"] = True
    ctx.session.state["output"] = self._output

    yield Event(
        author=self.name,
        content=types.Content(parts=[types.Part(text=self._output)]),
    )

    # Create checkpoint after execution (if auto)
    await self._checkpoint_after_execution(ctx.session)


@pytest.mark.asyncio
async def test_mixin_manual_checkpoint():
  """Test manual checkpoint creation."""
  session_service = InMemorySessionService()
  checkpoint_service = CheckpointService(session_service=session_service)

  agent = CheckpointableTestAgent(
      name="test_agent",
      checkpoint_service=checkpoint_service,
      auto_checkpoint=False,  # Manual mode
  )

  assert agent.checkpointing_enabled
  assert agent.last_checkpoint_id is None

  # Create session
  session = await session_service.create_session(
      app_name="test", user_id="user1", session_id="session1"
  )

  # Set some state
  session.state["test_key"] = "test_value"

  # Create checkpoint manually
  checkpoint_id = await agent.create_checkpoint(
      session=session,
      description="Manual checkpoint",
      metadata={"manual": True},
  )

  assert checkpoint_id is not None
  assert agent.last_checkpoint_id == checkpoint_id

  # Verify checkpoint was created
  checkpoints = await agent.list_checkpoints(session, page=1, page_size=10)
  assert len(checkpoints) == 1
  assert checkpoints[0].checkpoint_id == checkpoint_id
  assert checkpoints[0].description == "Manual checkpoint"


@pytest.mark.asyncio
async def test_mixin_auto_checkpoint():
  """Test automatic checkpoint creation before/after execution."""
  session_service = InMemorySessionService()
  checkpoint_service = CheckpointService(session_service=session_service)

  agent = CheckpointableTestAgent(
      name="test_agent",
      checkpoint_service=checkpoint_service,
      auto_checkpoint=True,  # Auto mode
  )

  runner = Runner(
      app_name="test",
      agent=agent,
      session_service=session_service,
      auto_create_session=True,
  )

  # Run agent
  async for event in runner.run_async(
      user_id="user1",
      session_id="session1",
      new_message=types.Content(parts=[types.Part(text="test")]),
  ):
    pass

  # Get session
  session = await session_service.get_session(
      app_name="test", user_id="user1", session_id="session1"
  )

  # Should have created 2 checkpoints (before and after)
  checkpoints = await agent.list_checkpoints(session, page=1, page_size=10)
  assert len(checkpoints) == 2

  # Check descriptions
  descriptions = [cp.description for cp in checkpoints]
  assert any("Before" in desc for desc in descriptions)
  assert any("After" in desc for desc in descriptions)


@pytest.mark.asyncio
async def test_mixin_restore_checkpoint():
  """Test checkpoint restoration."""
  session_service = InMemorySessionService()
  checkpoint_service = CheckpointService(session_service=session_service)

  agent = CheckpointableTestAgent(
      name="test_agent",
      checkpoint_service=checkpoint_service,
  )

  # Create session
  session = await session_service.create_session(
      app_name="test", user_id="user1", session_id="session1"
  )

  # Set initial state
  session.state["value"] = 100

  # Create checkpoint 1
  cp1_id = await agent.create_checkpoint(session, description="Checkpoint 1")

  # Modify state
  session.state["value"] = 200

  # Create checkpoint 2
  cp2_id = await agent.create_checkpoint(session, description="Checkpoint 2")

  # Restore checkpoint 1
  restored_state = await agent.restore_checkpoint(session, cp1_id)

  # Should have original value
  assert restored_state.get("value") == 100

  # Session state should be updated
  updated_session = await session_service.get_session(
      app_name="test", user_id="user1", session_id="session1"
  )
  assert updated_session.state.get("value") == 100


@pytest.mark.asyncio
async def test_mixin_checkpoint_diff():
  """Test computing diff between checkpoints."""
  session_service = InMemorySessionService()
  checkpoint_service = CheckpointService(session_service=session_service)

  agent = CheckpointableTestAgent(
      name="test_agent",
      checkpoint_service=checkpoint_service,
  )

  # Create session
  session = await session_service.create_session(
      app_name="test", user_id="user1", session_id="session1"
  )

  # State 1 - use full snapshot (no delta)
  session.state.clear()
  session.state.update({"a": 1, "b": 2, "c": 3})
  cp1_id = await agent.create_checkpoint(
      session, description="State 1", use_delta=False
  )

  # State 2 - modify state, use full snapshot
  session.state.clear()
  session.state.update({"a": 1, "b": 99, "d": 4})
  cp2_id = await agent.create_checkpoint(
      session, description="State 2", use_delta=False
  )

  # Get updated session (checkpoints are stored in session state)
  session = await session_service.get_session(
      app_name="test", user_id="user1", session_id="session1"
  )

  # Compute diff
  diff = await agent.get_checkpoint_diff(session, cp1_id, cp2_id)

  # Verify diff with full snapshots
  assert "a" in diff["unchanged"]  # Same value in both
  assert "b" in diff["changed"]  # Changed from 2 to 99
  assert diff["changed"]["b"]["old"] == 2
  assert diff["changed"]["b"]["new"] == 99
  assert "c" in diff["removed"]  # Removed in state2
  assert "d" in diff["added"]  # Added in state2


@pytest.mark.asyncio
async def test_mixin_delta_compression():
  """Test that delta compression works through the mixin."""
  session_service = InMemorySessionService()
  checkpoint_service = CheckpointService(session_service=session_service)

  agent = CheckpointableTestAgent(
      name="test_agent",
      checkpoint_service=checkpoint_service,
  )

  # Create session
  session = await session_service.create_session(
      app_name="test", user_id="user1", session_id="session1"
  )

  # Initial large state
  session.state.clear()
  session.state.update({
      "large_data": "x" * 1000,
      "field1": "value1",
      "field2": "value2",
  })

  # First checkpoint - full snapshot
  cp1_id = await agent.create_checkpoint(
      session, description="Full snapshot", use_delta=False
  )

  # Get checkpoint metadata to verify it's not a delta
  session = await session_service.get_session(
      app_name="test", user_id="user1", session_id="session1"
  )
  metadata1 = await checkpoint_service.get_checkpoint(session, cp1_id)
  assert metadata1 is not None
  assert not metadata1.is_delta  # First checkpoint should be full snapshot

  # Make small change
  session.state["field1"] = "updated_value"

  # Second checkpoint with delta (default)
  cp2_id = await agent.create_checkpoint(
      session, description="Delta checkpoint"  # use_delta=True by default
  )

  # Get checkpoint metadata to verify delta compression
  session = await session_service.get_session(
      app_name="test", user_id="user1", session_id="session1"
  )
  metadata2 = await checkpoint_service.get_checkpoint(session, cp2_id)
  assert metadata2 is not None
  assert metadata2.is_delta  # Second checkpoint should be delta
  assert (
      metadata2.base_checkpoint_id == cp1_id
  )  # Should reference first checkpoint

  # Verify delta snapshot contains changed data
  assert "field1" in metadata2.state_snapshot  # Changed field
  assert metadata2.state_snapshot["field1"] == "updated_value"

  # Delta compression: unchanged fields are marked as None
  # This is the delta format: {changed_key: new_value, unchanged_key: None}
  assert "field2" in metadata2.state_snapshot
  assert metadata2.state_snapshot["field2"] is None  # Unchanged
  assert "large_data" in metadata2.state_snapshot
  assert metadata2.state_snapshot["large_data"] is None  # Unchanged

  # Verify base checkpoint is referenced
  assert metadata2.base_checkpoint_id == cp1_id

  # Note: Restoration from delta checkpoint returns the delta snapshot itself
  # The full reconstruction would require merging with base checkpoint
  # This is handled by the checkpoint service internally
  restored_state = await agent.restore_checkpoint(session, cp2_id)
  assert restored_state.get("field1") == "updated_value"
  # The restored state from a delta checkpoint is just the delta
  # Full restoration would be handled by the checkpoint service if needed


@pytest.mark.asyncio
async def test_mixin_disabled_checkpointing():
  """Test that checkpointing works when disabled (no checkpoint_service)."""
  agent = CheckpointableTestAgent(
      name="test_agent",
      checkpoint_service=None,  # No service
      auto_checkpoint=False,
  )

  assert not agent.checkpointing_enabled
  assert agent.last_checkpoint_id is None

  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name="test", user_id="user1", session_id="session1"
  )

  # create_checkpoint should return None
  result = await agent.create_checkpoint(session, description="Test")
  assert result is None

  # list_checkpoints should raise ValueError
  with pytest.raises(ValueError, match="Checkpointing is disabled"):
    await agent.list_checkpoints(session)

  # restore_checkpoint should raise ValueError
  with pytest.raises(ValueError, match="Checkpointing is disabled"):
    await agent.restore_checkpoint(session, "fake_id")


@pytest.mark.asyncio
async def test_mixin_auto_checkpoint_warning():
  """Test warning when auto_checkpoint=True but no service."""
  import warnings

  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    agent = CheckpointableTestAgent(
        name="test_agent",
        checkpoint_service=None,  # No service
        auto_checkpoint=True,  # But auto enabled - should warn
    )

    # Should have issued a warning
    assert len(w) == 1
    assert "auto_checkpoint=True but no checkpoint_service" in str(w[0].message)
    assert not agent.checkpointing_enabled


@pytest.mark.asyncio
async def test_mixin_restore_checkpoint_returns_none():
  """restore_checkpoint raises ValueError when service returns None."""
  from unittest.mock import AsyncMock

  session_service = InMemorySessionService()
  checkpoint_service = CheckpointService(session_service=session_service)

  agent = CheckpointableTestAgent(
      name="test_agent", checkpoint_service=checkpoint_service
  )

  session = await session_service.create_session(
      app_name="test", user_id="user1", session_id="session1"
  )

  # Make the underlying service return None (defensive code path)
  checkpoint_service.restore_checkpoint = AsyncMock(return_value=None)

  with pytest.raises(ValueError, match="not found"):
    await agent.restore_checkpoint(session, "nonexistent_id")


@pytest.mark.asyncio
async def test_mixin_diff_checkpoints_disabled():
  """diff_checkpoints raises ValueError when checkpoint_service is None."""
  agent = CheckpointableTestAgent(
      name="test_agent",
      checkpoint_service=None,
  )

  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name="test", user_id="user1", session_id="session1"
  )

  with pytest.raises(ValueError, match="Checkpointing is disabled"):
    await agent.get_checkpoint_diff(session, "cp1", "cp2")


@pytest.mark.asyncio
async def test_mixin_diff_checkpoints_one_not_found():
  """diff_checkpoints raises ValueError when a checkpoint doesn't exist."""
  from unittest.mock import AsyncMock

  session_service = InMemorySessionService()
  checkpoint_service = CheckpointService(session_service=session_service)

  agent = CheckpointableTestAgent(
      name="test_agent", checkpoint_service=checkpoint_service
  )

  session = await session_service.create_session(
      app_name="test", user_id="user1", session_id="session1"
  )

  # First call returns metadata, second returns None
  checkpoint_service.get_checkpoint = AsyncMock(
      side_effect=[object(), None]  # first found, second not found
  )

  with pytest.raises(ValueError, match="not found"):
    await agent.get_checkpoint_diff(session, "cp1_exists", "cp2_missing")
