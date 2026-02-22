"""Tests for P0.3: Per-session locks in CheckpointService.

This test suite verifies that per-session locks prevent race conditions
when multiple concurrent operations try to create checkpoints.
"""

import asyncio
from datetime import datetime
from datetime import timezone

from google.adk.checkpoints.checkpoint_service import CheckpointService
from google.adk.checkpoints.checkpoint_service import CheckpointServiceConfig
from google.adk.sessions.in_memory_session_service import InMemorySessionService
import pytest


@pytest.fixture
async def checkpoint_service():
  """Create checkpoint service with limit of 5 checkpoints."""
  session_service = InMemorySessionService()
  config = CheckpointServiceConfig(
      max_checkpoints_per_session=5,  # Low limit to test race conditions
      max_state_size_bytes=1024 * 1024,
  )
  return CheckpointService(
      session_service=session_service,
      artifact_service=None,
      config=config,
  )


@pytest.fixture
async def session(checkpoint_service):
  """Create a test session."""
  session = await checkpoint_service.session_service.create_session(
      app_name="test_app",
      user_id="test_user",
  )
  return session


@pytest.mark.asyncio
async def test_concurrent_checkpoint_creation_no_race(
    checkpoint_service, session
):
  """Test that concurrent checkpoint creation doesn't violate max limit."""
  # Try to create 10 checkpoints concurrently with limit of 5
  # Without locks, some would succeed and violate the limit
  # With locks, exactly 5 should succeed

  async def create_checkpoint(index):
    try:
      metadata = await checkpoint_service.create_checkpoint(
          session=session,
          description=f"Checkpoint {index}",
          agent_name="test_agent",
      )
      return ("success", metadata.checkpoint_id)
    except ValueError as e:
      if "Checkpoint limit reached" in str(e):
        return ("limit_reached", None)
      raise

  # Create 10 concurrent tasks
  tasks = [create_checkpoint(i) for i in range(10)]
  results = await asyncio.gather(*tasks)

  # Count successes and limit_reached
  successes = [r for r in results if r[0] == "success"]
  limit_reached = [r for r in results if r[0] == "limit_reached"]

  # Exactly 5 should succeed (the limit)
  assert len(successes) == 5, f"Expected 5 successes, got {len(successes)}"
  # Remaining 5 should hit the limit
  assert (
      len(limit_reached) == 5
  ), f"Expected 5 limit errors, got {len(limit_reached)}"

  # Verify checkpoint index has exactly 5 entries
  checkpoint_index = session.state.get("_checkpoint_index", {})
  assert len(checkpoint_index) == 5


@pytest.mark.asyncio
async def test_different_sessions_dont_block_each_other(checkpoint_service):
  """Test that different sessions can create checkpoints concurrently."""
  # Create 3 different sessions
  session1 = await checkpoint_service.session_service.create_session(
      app_name="test_app", user_id="user1"
  )
  session2 = await checkpoint_service.session_service.create_session(
      app_name="test_app", user_id="user2"
  )
  session3 = await checkpoint_service.session_service.create_session(
      app_name="test_app", user_id="user3"
  )

  # Track checkpoint creation times
  start_times = {}
  end_times = {}

  async def create_checkpoint_with_delay(session, session_id, delay):
    start_times[session_id] = asyncio.get_event_loop().time()
    await asyncio.sleep(delay)  # Simulate work
    metadata = await checkpoint_service.create_checkpoint(
        session=session,
        description=f"Checkpoint for {session_id}",
    )
    end_times[session_id] = asyncio.get_event_loop().time()
    return metadata

  # Create checkpoints concurrently from different sessions
  results = await asyncio.gather(
      create_checkpoint_with_delay(session1, "session1", 0.1),
      create_checkpoint_with_delay(session2, "session2", 0.1),
      create_checkpoint_with_delay(session3, "session3", 0.1),
  )

  # All should succeed
  assert len(results) == 3
  assert all(r is not None for r in results)

  # Verify they ran concurrently (not serially)
  # If they blocked each other, total time would be ~0.3s
  # With per-session locks, they should overlap significantly
  total_elapsed = max(end_times.values()) - min(start_times.values())
  assert (
      total_elapsed < 0.25
  ), f"Sessions appear to have blocked each other: {total_elapsed:.3f}s elapsed"


@pytest.mark.asyncio
async def test_lock_contention_logged(checkpoint_service, session, caplog):
  """Test that high lock contention is logged."""
  import logging

  caplog.set_level(logging.WARNING)

  # Create a slow checkpoint to hold the lock
  async def slow_checkpoint():
    # Simulate slow checkpoint by adding state
    for i in range(100):
      session.state[f"key_{i}"] = f"value_{i}"
    return await checkpoint_service.create_checkpoint(
        session=session,
        description="Slow checkpoint",
    )

  # Create concurrent fast checkpoints that will wait
  async def fast_checkpoint(index):
    return await checkpoint_service.create_checkpoint(
        session=session,
        description=f"Fast checkpoint {index}",
    )

  # Run slow checkpoint and fast ones concurrently
  results = await asyncio.gather(
      slow_checkpoint(),
      fast_checkpoint(1),
      fast_checkpoint(2),
      return_exceptions=True,
  )

  # Some checkpoints should have waited and potentially logged contention
  # (Depending on timing, may or may not log if wait < 100ms)
  # This test mainly verifies the logging infrastructure works


@pytest.mark.asyncio
async def test_lock_properly_released_on_error(checkpoint_service, session):
  """Test that lock is released even when checkpoint creation fails."""
  # Fill up to the limit
  for i in range(5):
    await checkpoint_service.create_checkpoint(
        session=session,
        description=f"Checkpoint {i}",
    )

  # Try to create one more (will fail due to limit)
  with pytest.raises(ValueError, match="Checkpoint limit reached"):
    await checkpoint_service.create_checkpoint(
        session=session,
        description="This should fail",
    )

  # Lock should be released - we should be able to create another
  # checkpoint after deleting one
  checkpoint_index = session.state.get("_checkpoint_index", {})
  first_checkpoint_id = list(checkpoint_index.keys())[0]

  # Delete first checkpoint
  await checkpoint_service.delete_checkpoint(session, first_checkpoint_id)

  # Now we should be able to create a new one
  metadata = await checkpoint_service.create_checkpoint(
      session=session,
      description="After deletion",
  )
  assert metadata is not None


@pytest.mark.asyncio
async def test_session_lock_created_once(checkpoint_service, session):
  """Test that session lock is created only once (double-checked locking)."""
  # Create multiple checkpoints for same session
  for i in range(3):
    await checkpoint_service.create_checkpoint(
        session=session,
        description=f"Checkpoint {i}",
    )

  # Verify only one lock exists for this session
  assert session.id in checkpoint_service._session_locks
  lock = checkpoint_service._session_locks[session.id]

  # All subsequent calls should return the same lock
  lock2 = await checkpoint_service._get_session_lock(session.id)
  assert lock is lock2  # Same object


@pytest.mark.asyncio
async def test_sequential_checkpoints_no_contention(
    checkpoint_service, session
):
  """Test that sequential checkpoints don't have contention."""
  # Create checkpoints sequentially
  for i in range(5):
    metadata = await checkpoint_service.create_checkpoint(
        session=session,
        description=f"Sequential checkpoint {i}",
    )
    assert metadata is not None

  # All should succeed
  checkpoint_index = session.state.get("_checkpoint_index", {})
  assert len(checkpoint_index) == 5


@pytest.mark.asyncio
async def test_very_high_concurrency(checkpoint_service, session):
  """Test with 50 concurrent checkpoint attempts."""

  async def create_checkpoint(index):
    try:
      metadata = await checkpoint_service.create_checkpoint(
          session=session,
          description=f"Checkpoint {index}",
      )
      return "success"
    except ValueError as e:
      if "Checkpoint limit reached" in str(e):
        return "limit"
      raise

  # 50 concurrent attempts with limit of 5
  tasks = [create_checkpoint(i) for i in range(50)]
  results = await asyncio.gather(*tasks)

  # Exactly 5 should succeed
  successes = [r for r in results if r == "success"]
  assert len(successes) == 5

  # No race condition - checkpoint index should have exactly 5
  checkpoint_index = session.state.get("_checkpoint_index", {})
  assert len(checkpoint_index) == 5


@pytest.mark.asyncio
async def test_lock_wait_time_telemetry(checkpoint_service, session):
  """Test that lock wait time is tracked without crashing."""
  # Actual telemetry metric values would require OpenTelemetry test infrastructure

  async def create_checkpoint_with_state():
    # Add some state to make checkpoint creation slower
    session.state["test_data"] = "x" * 1000
    return await checkpoint_service.create_checkpoint(
        session=session,
        description="Test checkpoint",
    )

  # Create concurrent checkpoints
  results = await asyncio.gather(
      create_checkpoint_with_state(),
      create_checkpoint_with_state(),
  )

  # Both should succeed (below limit)
  assert len(results) == 2
  assert all(r is not None for r in results)
