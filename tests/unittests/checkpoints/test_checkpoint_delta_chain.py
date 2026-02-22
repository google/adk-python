"""Tests for P0.4: Delta chain error handling with proper exceptions.

This test suite verifies that delta reconstruction properly distinguishes
between different error conditions:
- CheckpointNotFoundError: checkpoint doesn't exist
- DeltaChainBrokenError: base checkpoint missing
- CheckpointCorruptedError: invalid checkpoint data
"""

from google.adk.checkpoints.checkpoint_service import CheckpointService
from google.adk.checkpoints.checkpoint_service import CheckpointServiceConfig
from google.adk.checkpoints.models import CheckpointCorruptedError
from google.adk.checkpoints.models import CheckpointNotFoundError
from google.adk.checkpoints.models import DeltaChainBrokenError
from google.adk.sessions.in_memory_session_service import InMemorySessionService
import pytest


@pytest.fixture
async def checkpoint_service():
  """Create checkpoint service for testing."""
  session_service = InMemorySessionService()
  config = CheckpointServiceConfig(
      max_checkpoints_per_session=100,
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
async def test_checkpoint_not_found_error(checkpoint_service, session):
  """Test that CheckpointNotFoundError is raised when checkpoint doesn't exist."""
  # Try to get nonexistent checkpoint
  with pytest.raises(CheckpointNotFoundError, match="not found"):
    await checkpoint_service.get_checkpoint(session, "nonexistent")


@pytest.mark.asyncio
async def test_checkpoint_corrupted_error(checkpoint_service, session):
  """Test that CheckpointCorruptedError is raised for invalid data."""
  # Manually add corrupted checkpoint data
  session.state["_checkpoint_corrupted"] = {
      "invalid": "data",
      "missing_required_fields": True,
  }

  # Should raise CheckpointCorruptedError
  with pytest.raises(CheckpointCorruptedError, match="corrupted"):
    await checkpoint_service.get_checkpoint(session, "corrupted")


@pytest.mark.asyncio
async def test_delta_chain_broken_error(checkpoint_service, session):
  """Test that DeltaChainBrokenError is raised when base checkpoint is missing."""
  # Create a checkpoint
  session.state["data"] = "initial"
  checkpoint1 = await checkpoint_service.create_checkpoint(
      session=session,
      description="Checkpoint 1",
      use_delta=False,
  )

  # Create a delta checkpoint
  session.state["data"] = "modified"
  checkpoint2 = await checkpoint_service.create_checkpoint(
      session=session,
      description="Checkpoint 2 (delta)",
      use_delta=True,
  )

  # Delete the base checkpoint (checkpoint1) to break the chain
  await checkpoint_service.delete_checkpoint(session, checkpoint1.checkpoint_id)

  # Try to reconstruct delta - should raise DeltaChainBrokenError
  with pytest.raises(DeltaChainBrokenError, match="Delta chain broken"):
    await checkpoint_service.get_checkpoint(
        session, checkpoint2.checkpoint_id, reconstruct_delta=True
    )


@pytest.mark.asyncio
async def test_delta_chain_broken_with_intermediate_missing(
    checkpoint_service, session
):
  """Test delta chain with intermediate checkpoint missing."""
  # Create chain: checkpoint1 → checkpoint2 → checkpoint3
  session.state["data"] = "step1"
  checkpoint1 = await checkpoint_service.create_checkpoint(
      session=session,
      description="Step 1",
      use_delta=False,
  )

  session.state["data"] = "step2"
  checkpoint2 = await checkpoint_service.create_checkpoint(
      session=session,
      description="Step 2 (delta)",
      use_delta=True,
  )

  session.state["data"] = "step3"
  checkpoint3 = await checkpoint_service.create_checkpoint(
      session=session,
      description="Step 3 (delta)",
      use_delta=True,
  )

  # Delete intermediate checkpoint (checkpoint2)
  await checkpoint_service.delete_checkpoint(session, checkpoint2.checkpoint_id)

  # Try to reconstruct checkpoint3 - should raise DeltaChainBrokenError
  # because checkpoint2 (the base) is missing
  with pytest.raises(DeltaChainBrokenError, match="Delta chain broken"):
    await checkpoint_service.get_checkpoint(
        session, checkpoint3.checkpoint_id, reconstruct_delta=True
    )


@pytest.mark.asyncio
async def test_long_delta_chain_success(checkpoint_service, session):
  """Test successful reconstruction of long delta chain."""
  # Create chain: checkpoint1 → checkpoint2 → checkpoint3 → checkpoint4
  checkpoints = []

  # Base checkpoint
  session.state["data"] = "step1"
  session.state["step"] = 1
  checkpoint1 = await checkpoint_service.create_checkpoint(
      session=session,
      description="Step 1",
      use_delta=False,
  )
  checkpoints.append(checkpoint1)

  # Delta checkpoint 2
  session.state["data"] = "step2"
  session.state["step"] = 2
  checkpoint2 = await checkpoint_service.create_checkpoint(
      session=session,
      description="Step 2 (delta)",
      use_delta=True,
  )
  checkpoints.append(checkpoint2)

  # Delta checkpoint 3
  session.state["data"] = "step3"
  session.state["step"] = 3
  checkpoint3 = await checkpoint_service.create_checkpoint(
      session=session,
      description="Step 3 (delta)",
      use_delta=True,
  )
  checkpoints.append(checkpoint3)

  # Delta checkpoint 4
  session.state["data"] = "step4"
  session.state["step"] = 4
  checkpoint4 = await checkpoint_service.create_checkpoint(
      session=session,
      description="Step 4 (delta)",
      use_delta=True,
  )
  checkpoints.append(checkpoint4)

  # Reconstruct each checkpoint
  for i, checkpoint in enumerate(checkpoints, 1):
    metadata = await checkpoint_service.get_checkpoint(
        session, checkpoint.checkpoint_id, reconstruct_delta=True
    )
    assert metadata.state_snapshot["data"] == f"step{i}"
    assert metadata.state_snapshot["step"] == i


@pytest.mark.asyncio
async def test_delta_chain_with_deletions(checkpoint_service, session):
  """Test delta chain correctly handles key deletions."""
  # Base checkpoint with multiple keys
  session.state["key1"] = "value1"
  session.state["key2"] = "value2"
  session.state["key3"] = "value3"
  checkpoint1 = await checkpoint_service.create_checkpoint(
      session=session,
      description="Base",
      use_delta=False,
  )

  # Delete key2
  del session.state["key2"]
  checkpoint2 = await checkpoint_service.create_checkpoint(
      session=session,
      description="After deletion",
      use_delta=True,
  )

  # Reconstruct - key2 should be absent
  metadata = await checkpoint_service.get_checkpoint(
      session, checkpoint2.checkpoint_id, reconstruct_delta=True
  )
  assert "key1" in metadata.state_snapshot
  assert "key2" not in metadata.state_snapshot
  assert "key3" in metadata.state_snapshot


@pytest.mark.asyncio
async def test_restore_checkpoint_not_found_error(checkpoint_service, session):
  """Test that restore_checkpoint raises CheckpointNotFoundError."""
  # Try to restore nonexistent checkpoint
  with pytest.raises(CheckpointNotFoundError, match="not found"):
    await checkpoint_service.restore_checkpoint(session, "nonexistent")


@pytest.mark.asyncio
async def test_restore_checkpoint_delta_chain_broken(
    checkpoint_service, session
):
  """Test that restore_checkpoint raises DeltaChainBrokenError."""
  # Create base and delta checkpoint
  session.state["data"] = "base"
  checkpoint1 = await checkpoint_service.create_checkpoint(
      session=session,
      description="Base",
      use_delta=False,
  )

  session.state["data"] = "modified"
  checkpoint2 = await checkpoint_service.create_checkpoint(
      session=session,
      description="Delta",
      use_delta=True,
  )

  # Delete base checkpoint
  await checkpoint_service.delete_checkpoint(session, checkpoint1.checkpoint_id)

  # Try to restore delta checkpoint - should raise DeltaChainBrokenError
  with pytest.raises(DeltaChainBrokenError, match="Delta chain broken"):
    await checkpoint_service.restore_checkpoint(
        session, checkpoint2.checkpoint_id, restore_state=True
    )


@pytest.mark.asyncio
async def test_list_checkpoints_skips_corrupted(checkpoint_service, session):
  """Test that list_checkpoints skips corrupted checkpoints."""
  # Create valid checkpoint
  session.state["data"] = "valid"
  checkpoint1 = await checkpoint_service.create_checkpoint(
      session=session,
      description="Valid",
  )

  # Manually add corrupted checkpoint
  session.state["_checkpoint_corrupted"] = {"invalid": "data"}
  session.state["_checkpoint_index"]["corrupted"] = {
      "timestamp": "2024-01-01T00:00:00Z",
      "agent": "test",
  }

  # List should skip corrupted and return only valid
  response = await checkpoint_service.list_checkpoints(session)
  assert len(response.checkpoints) == 1
  assert response.checkpoints[0].checkpoint_id == checkpoint1.checkpoint_id


@pytest.mark.asyncio
async def test_create_checkpoint_with_corrupted_base(
    checkpoint_service, session
):
  """Test that create_checkpoint falls back to full snapshot if base is corrupted."""
  # Create valid base checkpoint
  session.state["data"] = "base"
  checkpoint1 = await checkpoint_service.create_checkpoint(
      session=session,
      description="Base",
      use_delta=False,
  )

  # Corrupt the base checkpoint
  session.state[f"_checkpoint_{checkpoint1.checkpoint_id}"] = {
      "invalid": "data"
  }

  # Create new checkpoint with delta (should fall back to full snapshot)
  session.state["data"] = "modified"
  checkpoint2 = await checkpoint_service.create_checkpoint(
      session=session,
      description="Should use full snapshot",
      use_delta=True,
  )

  # Checkpoint2 should not be a delta (fell back to full snapshot)
  metadata = await checkpoint_service.get_checkpoint(
      session, checkpoint2.checkpoint_id
  )
  assert not metadata.is_delta
  assert metadata.base_checkpoint_id is None


@pytest.mark.asyncio
async def test_error_telemetry_attributes(checkpoint_service, session):
  """Test that telemetry attributes are set for errors."""
  # This test verifies the telemetry code path doesn't crash
  # Actual telemetry verification would require OpenTelemetry test infrastructure

  # CheckpointNotFoundError
  with pytest.raises(CheckpointNotFoundError):
    await checkpoint_service.get_checkpoint(session, "nonexistent")

  # CheckpointCorruptedError
  session.state["_checkpoint_corrupted"] = {"invalid": "data"}
  with pytest.raises(CheckpointCorruptedError):
    await checkpoint_service.get_checkpoint(session, "corrupted")

  # DeltaChainBrokenError
  session.state["data"] = "base"
  checkpoint1 = await checkpoint_service.create_checkpoint(
      session=session,
      description="Base",
      use_delta=False,
  )
  session.state["data"] = "modified"
  checkpoint2 = await checkpoint_service.create_checkpoint(
      session=session,
      description="Delta",
      use_delta=True,
  )
  await checkpoint_service.delete_checkpoint(session, checkpoint1.checkpoint_id)

  with pytest.raises(DeltaChainBrokenError):
    await checkpoint_service.get_checkpoint(
        session, checkpoint2.checkpoint_id, reconstruct_delta=True
    )


@pytest.mark.asyncio
async def test_delta_reconstruction_without_reconstruct_flag(
    checkpoint_service, session
):
  """Test that delta checkpoints return raw delta without reconstruct flag."""
  # Create base and delta
  session.state["data"] = "base"
  session.state["extra"] = "value"
  checkpoint1 = await checkpoint_service.create_checkpoint(
      session=session,
      description="Base",
      use_delta=False,
  )

  session.state["data"] = "modified"
  # extra unchanged
  checkpoint2 = await checkpoint_service.create_checkpoint(
      session=session,
      description="Delta",
      use_delta=True,
  )

  # Get without reconstruct_delta - should return raw delta
  metadata = await checkpoint_service.get_checkpoint(
      session, checkpoint2.checkpoint_id, reconstruct_delta=False
  )

  # Delta should only contain changed key
  assert metadata.is_delta
  assert "data" in metadata.state_snapshot
  assert metadata.state_snapshot["data"] == "modified"
  # extra shouldn't be in delta (unchanged)

  # Get with reconstruct_delta - should return full state
  metadata_full = await checkpoint_service.get_checkpoint(
      session, checkpoint2.checkpoint_id, reconstruct_delta=True
  )

  # Full state should have both keys
  assert "data" in metadata_full.state_snapshot
  assert "extra" in metadata_full.state_snapshot
  assert metadata_full.state_snapshot["data"] == "modified"
  assert metadata_full.state_snapshot["extra"] == "value"
