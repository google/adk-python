"""Tests for CheckpointService.

Tests checkpoint creation, retrieval, restoration, and deletion using
InMemorySessionService and InMemoryArtifactService.
"""

from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.checkpoints import CheckpointMetadata
from google.adk.checkpoints import CheckpointService
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
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
  """Create CheckpointService with session and artifact services."""
  return CheckpointService(
      session_service=session_service,
      artifact_service=artifact_service,
  )


@pytest.fixture
async def session(session_service):
  """Create a test session."""
  return await session_service.create_session(
      app_name="test_app",
      user_id="test_user",
      session_id="test_session",
      state={"counter": 0, "data": "initial"},
  )


class TestCheckpointService:
  """Test CheckpointService functionality."""

  @pytest.mark.asyncio
  async def test_create_checkpoint_without_artifacts(
      self, checkpoint_service, session_service, session
  ):
    """Test creating a checkpoint without artifact tracking."""
    # Create checkpoint
    metadata = await checkpoint_service.create_checkpoint(
        session=session,
        checkpoint_id="cp1",
        description="Test checkpoint",
        agent_name="test_agent",
    )

    assert metadata is not None
    assert metadata.checkpoint_id == "cp1"
    assert metadata.description == "Test checkpoint"
    assert metadata.agent_name == "test_agent"
    assert metadata.state_snapshot == {"counter": 0, "data": "initial"}
    assert metadata.artifact_versions == {}

    # Verify checkpoint in session state
    checkpoint = await checkpoint_service.get_checkpoint(
        session=session,
        checkpoint_id="cp1",
    )

    assert checkpoint is not None
    assert checkpoint.checkpoint_id == "cp1"
    assert checkpoint.description == "Test checkpoint"
    assert checkpoint.agent_name == "test_agent"
    assert checkpoint.state_snapshot == {"counter": 0, "data": "initial"}
    assert checkpoint.artifact_versions == {}

  @pytest.mark.asyncio
  async def test_create_checkpoint_with_artifacts(
      self,
      checkpoint_service,
      session_service,
      artifact_service,
      session,
  ):
    """Test creating a checkpoint with artifact version tracking."""
    # Save some artifacts
    await artifact_service.save_artifact(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        filename="file1.txt",
        artifact=types.Part(text="version 0"),
    )

    await artifact_service.save_artifact(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        filename="file1.txt",
        artifact=types.Part(text="version 1"),
    )

    await artifact_service.save_artifact(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        filename="file2.txt",
        artifact=types.Part(text="data"),
    )

    # Create checkpoint
    event = await checkpoint_service.create_checkpoint(
        session=session,
        checkpoint_id="cp2",
        description="With artifacts",
    )

    # Verify artifact versions captured
    checkpoint = await checkpoint_service.get_checkpoint(
        session=session,
        checkpoint_id="cp2",
    )

    assert checkpoint is not None
    assert checkpoint.artifact_versions == {
        "file1.txt": 1,  # Latest version
        "file2.txt": 0,  # Only version
    }

  @pytest.mark.asyncio
  async def test_create_checkpoint_selective_artifacts(
      self,
      checkpoint_service,
      artifact_service,
      session,
  ):
    """Test creating checkpoint with selective artifact tracking."""
    # Save multiple artifacts
    await artifact_service.save_artifact(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        filename="important.txt",
        artifact=types.Part(text="important"),
    )

    await artifact_service.save_artifact(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        filename="temp.txt",
        artifact=types.Part(text="temporary"),
    )

    # Create checkpoint tracking only important.txt
    await checkpoint_service.create_checkpoint(
        session=session,
        checkpoint_id="cp3",
        artifact_filenames=["important.txt"],
    )

    checkpoint = await checkpoint_service.get_checkpoint(
        session=session,
        checkpoint_id="cp3",
    )

    assert checkpoint is not None
    assert "important.txt" in checkpoint.artifact_versions
    assert "temp.txt" not in checkpoint.artifact_versions

  @pytest.mark.asyncio
  async def test_get_nonexistent_checkpoint(self, checkpoint_service, session):
    """Test retrieving a checkpoint that doesn't exist raises CheckpointNotFoundError."""
    from google.adk.checkpoints.models import CheckpointNotFoundError

    # Should raise CheckpointNotFoundError (P0.4 fix)
    with pytest.raises(CheckpointNotFoundError):
      await checkpoint_service.get_checkpoint(
          session=session,
          checkpoint_id="nonexistent",
      )

  @pytest.mark.asyncio
  async def test_list_checkpoints(self, checkpoint_service, session):
    """Test listing all checkpoints in a session."""
    # Create multiple checkpoints
    await checkpoint_service.create_checkpoint(
        session=session,
        checkpoint_id="cp1",
        description="First",
    )

    await checkpoint_service.create_checkpoint(
        session=session,
        checkpoint_id="cp2",
        description="Second",
    )

    await checkpoint_service.create_checkpoint(
        session=session,
        checkpoint_id="cp3",
        description="Third",
    )

    # List checkpoints
    response = await checkpoint_service.list_checkpoints(session=session)

    assert len(response.checkpoints) == 3
    assert response.total_count == 3
    assert not response.has_next
    assert not response.has_previous
    checkpoints = response.checkpoints
    # Most recent first
    assert checkpoints[0].checkpoint_id == "cp3"
    assert checkpoints[1].checkpoint_id == "cp2"
    assert checkpoints[2].checkpoint_id == "cp1"

  @pytest.mark.asyncio
  async def test_list_checkpoints_empty(self, checkpoint_service, session):
    """Test listing checkpoints when none exist."""
    response = await checkpoint_service.list_checkpoints(session=session)
    assert response.checkpoints == []
    assert response.total_count == 0
    assert not response.has_next
    assert not response.has_previous

  @pytest.mark.asyncio
  async def test_restore_checkpoint_state(
      self, checkpoint_service, session_service, session
  ):
    """Test restoring session state from a checkpoint."""
    # Create checkpoint with initial state
    await checkpoint_service.create_checkpoint(
        session=session,
        checkpoint_id="restore_test",
    )

    # Modify session state
    session.state["counter"] = 42
    session.state["new_field"] = "added"

    # Restore checkpoint
    metadata = await checkpoint_service.restore_checkpoint(
        session=session,
        checkpoint_id="restore_test",
        restore_state=True,
        restore_artifacts=False,
    )

    assert metadata is not None

    # State should be restored to checkpoint snapshot
    # Note: The restore adds to state via state_delta, so new fields persist
    assert session.state["counter"] == 0  # Restored
    assert session.state["data"] == "initial"  # Restored

  @pytest.mark.asyncio
  async def test_restore_nonexistent_checkpoint(
      self, checkpoint_service, session
  ):
    """Test restoring a checkpoint that doesn't exist raises CheckpointNotFoundError."""
    from google.adk.checkpoints.models import CheckpointNotFoundError

    # Should raise CheckpointNotFoundError (P0.4 fix)
    with pytest.raises(CheckpointNotFoundError):
      await checkpoint_service.restore_checkpoint(
          session=session,
          checkpoint_id="nonexistent",
      )

  @pytest.mark.asyncio
  async def test_delete_checkpoint(self, checkpoint_service, session):
    """Test deleting a checkpoint."""
    # Create checkpoint
    await checkpoint_service.create_checkpoint(
        session=session,
        checkpoint_id="to_delete",
    )

    # Verify it exists
    checkpoint = await checkpoint_service.get_checkpoint(
        session=session,
        checkpoint_id="to_delete",
    )
    assert checkpoint is not None

    # Delete it
    deleted = await checkpoint_service.delete_checkpoint(
        session=session,
        checkpoint_id="to_delete",
    )
    assert deleted is True

    # Verify it's gone (should raise CheckpointNotFoundError - P0.4 fix)
    from google.adk.checkpoints.models import CheckpointNotFoundError

    with pytest.raises(CheckpointNotFoundError):
      await checkpoint_service.get_checkpoint(
          session=session,
          checkpoint_id="to_delete",
      )

  @pytest.mark.asyncio
  async def test_delete_nonexistent_checkpoint(
      self, checkpoint_service, session
  ):
    """Test deleting a checkpoint that doesn't exist."""
    deleted = await checkpoint_service.delete_checkpoint(
        session=session,
        checkpoint_id="nonexistent",
    )
    assert deleted is False

  @pytest.mark.asyncio
  async def test_checkpoint_with_custom_metadata(
      self, checkpoint_service, session
  ):
    """Test creating checkpoint with custom metadata."""
    await checkpoint_service.create_checkpoint(
        session=session,
        checkpoint_id="custom",
        custom_metadata={
            "iteration": 5,
            "accuracy": 0.95,
            "tags": ["important", "milestone"],
        },
    )

    checkpoint = await checkpoint_service.get_checkpoint(
        session=session,
        checkpoint_id="custom",
    )

    assert checkpoint is not None
    assert checkpoint.custom_metadata["iteration"] == 5
    assert checkpoint.custom_metadata["accuracy"] == 0.95
    assert checkpoint.custom_metadata["tags"] == ["important", "milestone"]

  @pytest.mark.asyncio
  async def test_checkpoint_without_artifact_service(
      self, session_service, session
  ):
    """Test CheckpointService works without ArtifactService."""
    # Create service without artifact service
    checkpoint_service = CheckpointService(
        session_service=session_service,
        artifact_service=None,
    )

    # Create checkpoint
    await checkpoint_service.create_checkpoint(
        session=session,
        checkpoint_id="no_artifacts",
    )

    checkpoint = await checkpoint_service.get_checkpoint(
        session=session,
        checkpoint_id="no_artifacts",
    )

    assert checkpoint is not None
    assert checkpoint.artifact_versions == {}

  @pytest.mark.asyncio
  async def test_multiple_checkpoints_same_session(
      self, checkpoint_service, session
  ):
    """Test creating multiple checkpoints in the same session."""
    # Checkpoint 1
    await checkpoint_service.create_checkpoint(
        session=session,
        checkpoint_id="step1",
        description="Initial state",
    )

    # Modify state
    session.state["counter"] = 10

    # Checkpoint 2
    await checkpoint_service.create_checkpoint(
        session=session,
        checkpoint_id="step2",
        description="After increment",
    )

    # Verify both checkpoints exist with different state snapshots
    cp1 = await checkpoint_service.get_checkpoint(session, "step1")
    cp2 = await checkpoint_service.get_checkpoint(session, "step2")

    assert cp1 is not None
    assert cp2 is not None
    assert cp1.state_snapshot["counter"] == 0
    assert cp2.state_snapshot["counter"] == 10

  @pytest.mark.asyncio
  async def test_checkpoint_event_integration(
      self, checkpoint_service, session
  ):
    """Test that checkpoints integrate with event timeline."""
    # Get initial event count
    initial_event_count = len(session.events)

    # Create checkpoint
    metadata = await checkpoint_service.create_checkpoint(
        session=session,
        checkpoint_id="event_test",
    )

    # Checkpoint metadata should be returned
    assert metadata is not None
    assert metadata.checkpoint_id == "event_test"

    # Event should be added to session timeline
    assert len(session.events) == initial_event_count + 1

    # Latest event should have correct structure
    latest_event = session.events[-1]
    assert latest_event.actions is not None
    assert latest_event.actions.state_delta is not None
    assert "_checkpoint_event_test" in latest_event.actions.state_delta

    # Checkpoint should be retrievable from state
    checkpoint = await checkpoint_service.get_checkpoint(
        session=session, checkpoint_id="event_test"
    )
    assert checkpoint is not None
    assert checkpoint.checkpoint_id == "event_test"

  @pytest.mark.asyncio
  async def test_auto_generated_checkpoint_id(
      self, checkpoint_service, session
  ):
    """Test that checkpoint_id is auto-generated when not provided."""
    # Create checkpoint without specifying ID
    metadata = await checkpoint_service.create_checkpoint(
        session=session,
        description="Auto-generated ID test",
        agent_name="test_agent",
    )

    # Should have auto-generated ID
    assert metadata is not None
    assert metadata.checkpoint_id is not None
    assert metadata.checkpoint_id.startswith("checkpoint-")

    # Should be retrievable
    checkpoint = await checkpoint_service.get_checkpoint(
        session=session,
        checkpoint_id=metadata.checkpoint_id,
    )
    assert checkpoint is not None
    assert checkpoint.checkpoint_id == metadata.checkpoint_id
    assert checkpoint.description == "Auto-generated ID test"

    # Create another auto-generated checkpoint - should have different ID
    metadata2 = await checkpoint_service.create_checkpoint(
        session=session,
        description="Second auto-generated",
    )
    assert metadata2.checkpoint_id != metadata.checkpoint_id

  @pytest.mark.asyncio
  async def test_list_checkpoints_skips_invalid_data(
      self, checkpoint_service, session
  ):
    """Test that list_checkpoints skips invalid checkpoint data."""
    # Create a valid checkpoint
    await checkpoint_service.create_checkpoint(
        session=session,
        checkpoint_id="valid",
    )

    # Manually add invalid checkpoint data to session state
    session.state["_checkpoint_invalid"] = {"bad": "data"}

    # list_checkpoints should not crash and should return only valid checkpoint
    response = await checkpoint_service.list_checkpoints(session)

    assert len(response.checkpoints) == 1
    assert response.checkpoints[0].checkpoint_id == "valid"


class TestCheckpointLimits:
  """Test CheckpointService resource limits."""

  @pytest.mark.asyncio
  async def test_checkpoint_count_limit(self, session_service):
    """Test that checkpoint count limit is enforced."""
    from google.adk.checkpoints import CheckpointServiceConfig

    # Create service with low limit
    config = CheckpointServiceConfig(max_checkpoints_per_session=3)
    checkpoint_service = CheckpointService(
        session_service=session_service,
        config=config,
    )

    session = await session_service.create_session(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        state={"data": "test"},
    )

    # Create 3 checkpoints (at limit)
    await checkpoint_service.create_checkpoint(session, checkpoint_id="cp1")
    await checkpoint_service.create_checkpoint(session, checkpoint_id="cp2")
    await checkpoint_service.create_checkpoint(session, checkpoint_id="cp3")

    # Fourth checkpoint should fail
    with pytest.raises(ValueError, match="Checkpoint limit reached"):
      await checkpoint_service.create_checkpoint(session, checkpoint_id="cp4")

    # Verify only 3 checkpoints exist
    response = await checkpoint_service.list_checkpoints(session)
    assert len(response.checkpoints) == 3

  @pytest.mark.asyncio
  async def test_checkpoint_count_limit_zero_unlimited(self, session_service):
    """Test that limit=0 means unlimited checkpoints."""
    from google.adk.checkpoints import CheckpointServiceConfig

    config = CheckpointServiceConfig(max_checkpoints_per_session=0)
    checkpoint_service = CheckpointService(
        session_service=session_service,
        config=config,
    )

    session = await session_service.create_session(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        state={"data": "test"},
    )

    # Should be able to create many checkpoints
    for i in range(150):
      await checkpoint_service.create_checkpoint(
          session, checkpoint_id=f"cp{i}"
      )

    response = await checkpoint_service.list_checkpoints(
        session, page_size=1000
    )
    assert len(response.checkpoints) == 150

  @pytest.mark.asyncio
  async def test_state_size_limit(self, session_service):
    """Test that state size limit is enforced."""
    from google.adk.checkpoints import CheckpointServiceConfig

    # Create service with small size limit (1KB)
    config = CheckpointServiceConfig(max_state_size_bytes=1024)
    checkpoint_service = CheckpointService(
        session_service=session_service,
        config=config,
    )

    # Create session with large state (>1KB)
    large_data = "x" * 2000  # 2KB of data
    session = await session_service.create_session(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        state={"large_field": large_data},
    )

    # Checkpoint should fail due to size limit
    with pytest.raises(ValueError, match="State size .* bytes exceeds limit"):
      await checkpoint_service.create_checkpoint(session)

  @pytest.mark.asyncio
  async def test_state_size_limit_zero_unlimited(self, session_service):
    """Test that size limit=0 means unlimited size."""
    from google.adk.checkpoints import CheckpointServiceConfig

    config = CheckpointServiceConfig(max_state_size_bytes=0)
    checkpoint_service = CheckpointService(
        session_service=session_service,
        config=config,
    )

    # Create session with very large state
    large_data = "x" * 1000000  # 1MB
    session = await session_service.create_session(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        state={"large_field": large_data},
    )

    # Should succeed with unlimited size
    metadata = await checkpoint_service.create_checkpoint(session)
    assert metadata is not None

  @pytest.mark.asyncio
  async def test_state_size_with_delta_compression(self, session_service):
    """Test that delta compression helps with size limits."""
    from google.adk.checkpoints import CheckpointServiceConfig

    # Small size limit
    config = CheckpointServiceConfig(max_state_size_bytes=500)
    checkpoint_service = CheckpointService(
        session_service=session_service,
        config=config,
    )

    # Create session with moderate state
    session = await session_service.create_session(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        state={"field1": "x" * 200, "field2": "y" * 200},
    )

    # First checkpoint (full snapshot) might fail
    try:
      await checkpoint_service.create_checkpoint(
          session, checkpoint_id="cp1", use_delta=False
      )
      first_succeeded = True
    except ValueError:
      first_succeeded = False

    # Make small change
    session.state["field3"] = "small change"

    # Second checkpoint with delta should succeed (only changed data)
    metadata = await checkpoint_service.create_checkpoint(
        session, checkpoint_id="cp2", use_delta=True
    )
    assert metadata is not None
    assert (
        metadata.is_delta or first_succeeded
    )  # Either delta or full was small enough

  @pytest.mark.asyncio
  async def test_delete_checkpoint_frees_limit(self, session_service):
    """Test that deleting checkpoint frees up slot for new checkpoints."""
    from google.adk.checkpoints import CheckpointServiceConfig

    config = CheckpointServiceConfig(max_checkpoints_per_session=2)
    checkpoint_service = CheckpointService(
        session_service=session_service,
        config=config,
    )

    session = await session_service.create_session(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        state={"data": "test"},
    )

    # Create 2 checkpoints (at limit)
    await checkpoint_service.create_checkpoint(session, checkpoint_id="cp1")
    await checkpoint_service.create_checkpoint(session, checkpoint_id="cp2")

    # Third should fail
    with pytest.raises(ValueError, match="Checkpoint limit reached"):
      await checkpoint_service.create_checkpoint(session, checkpoint_id="cp3")

    # Delete one checkpoint
    await checkpoint_service.delete_checkpoint(session, "cp1")

    # Now third checkpoint should succeed
    metadata = await checkpoint_service.create_checkpoint(
        session, checkpoint_id="cp3"
    )
    assert metadata.checkpoint_id == "cp3"

  def test_config_validation(self):
    """Test that config validates parameters."""
    from google.adk.checkpoints import CheckpointServiceConfig

    # Negative checkpoint limit should fail
    with pytest.raises(ValueError, match="max_checkpoints_per_session"):
      CheckpointServiceConfig(max_checkpoints_per_session=-1)

    # Negative size limit should fail
    with pytest.raises(ValueError, match="max_state_size_bytes"):
      CheckpointServiceConfig(max_state_size_bytes=-1)

    # Valid configs should work
    config = CheckpointServiceConfig(
        max_checkpoints_per_session=100,
        max_state_size_bytes=10485760,
    )
    assert config.max_checkpoints_per_session == 100
    assert config.max_state_size_bytes == 10485760


class TestCheckpointExceptions:
  """Test exception handling in CheckpointService."""

  @pytest.mark.asyncio
  async def test_restore_artifact_failure_continues(
      self, session_service, artifact_service
  ):
    """Test that artifact restoration continues despite individual failures."""
    checkpoint_service = CheckpointService(
        session_service=session_service,
        artifact_service=artifact_service,
    )

    session = await session_service.create_session(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        state={"data": "test"},
    )

    # Create artifacts
    await artifact_service.save_artifact(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        filename="file1.txt",
        artifact=types.Part(text="data1"),
    )
    await artifact_service.save_artifact(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        filename="file2.txt",
        artifact=types.Part(text="data2"),
    )

    # Create checkpoint
    metadata = await checkpoint_service.create_checkpoint(
        session, artifact_filenames=["file1.txt", "file2.txt"]
    )

    # Delete one artifact to simulate failure
    await artifact_service.delete_artifact(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        filename="file1.txt",
    )

    # Restore should succeed (logs warning for missing artifact)
    restored = await checkpoint_service.restore_checkpoint(
        session, metadata.checkpoint_id, restore_artifacts=True
    )
    assert restored is not None

  @pytest.mark.asyncio
  async def test_get_checkpoint_with_corrupted_data(self, session_service):
    """Test get_checkpoint raises CheckpointCorruptedError for corrupted data."""
    from google.adk.checkpoints.models import CheckpointCorruptedError

    checkpoint_service = CheckpointService(session_service=session_service)

    session = await session_service.create_session(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        state={"data": "test"},
    )

    # Manually add corrupted checkpoint data
    session.state["_checkpoint_corrupted"] = {"invalid": "data"}

    # get_checkpoint should raise CheckpointCorruptedError (P0.4 fix)
    with pytest.raises(CheckpointCorruptedError):
      await checkpoint_service.get_checkpoint(session, "corrupted")

  @pytest.mark.asyncio
  async def test_restore_nonexistent_checkpoint(self, session_service):
    """Test restoring nonexistent checkpoint raises CheckpointNotFoundError."""
    from google.adk.checkpoints.models import CheckpointNotFoundError

    checkpoint_service = CheckpointService(session_service=session_service)

    session = await session_service.create_session(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        state={"data": "test"},
    )

    # Restore nonexistent checkpoint (P0.4 fix - raises exception)
    with pytest.raises(CheckpointNotFoundError):
      await checkpoint_service.restore_checkpoint(session, "nonexistent")

  @pytest.mark.asyncio
  async def test_delete_nonexistent_checkpoint(self, session_service):
    """Test deleting nonexistent checkpoint succeeds (idempotent)."""
    checkpoint_service = CheckpointService(session_service=session_service)

    session = await session_service.create_session(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        state={"data": "test"},
    )

    # Delete nonexistent checkpoint (should be idempotent)
    await checkpoint_service.delete_checkpoint(session, "nonexistent")

    # Verify checkpoint index is unchanged (delete was a no-op)
    checkpoint_index = session.state.get("_checkpoint_index", {})
    assert "nonexistent" not in checkpoint_index

  @pytest.mark.asyncio
  async def test_create_checkpoint_with_empty_state(self, session_service):
    """Test creating checkpoint with empty state."""
    checkpoint_service = CheckpointService(session_service=session_service)

    session = await session_service.create_session(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        state={},  # Empty state
    )

    # Should succeed with empty state
    metadata = await checkpoint_service.create_checkpoint(session)
    assert metadata is not None
    assert metadata.state_snapshot == {}
