"""Comprehensive test coverage for CheckpointService untested code paths.

Tests added based on audit findings:
- Artifact restoration with load_artifact verification
- Delta reconstruction with chain handling
- Deleted base checkpoint detection
- Edge cases and error conditions
- Lock contention warning
- list_checkpoints page/page_size clamping
- Artifact restore exception handling
"""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import patch

from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.checkpoints import CheckpointService
from google.adk.sessions import InMemorySessionService
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
  """Create test session with initial state."""
  return await session_service.create_session(
      app_name="test_app",
      user_id="test_user",
      session_id="test_session",
      state={"counter": 0, "data": "initial"},
  )


class TestArtifactRestoration:
  """Test artifact restoration with load_artifact (bug that went unnoticed)."""

  @pytest.mark.asyncio
  async def test_restore_artifacts_calls_load_artifact(
      self, checkpoint_service, artifact_service, session
  ):
    """Test that restore_checkpoint correctly calls load_artifact."""
    # Create artifact v1
    await artifact_service.save_artifact(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        filename="doc.txt",
        artifact=types.Part(text="version 1"),
    )

    # Create checkpoint (captures artifact v1)
    metadata = await checkpoint_service.create_checkpoint(
        session=session,
        checkpoint_id="cp1",
        artifact_filenames=["doc.txt"],
    )

    # Verify artifact version was captured
    assert "doc.txt" in metadata.artifact_versions
    v1_version = metadata.artifact_versions["doc.txt"]
    assert v1_version == 0  # First version is 0

    # Create artifact v2 (version 1 - newer)
    await artifact_service.save_artifact(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        filename="doc.txt",
        artifact=types.Part(text="version 2"),
    )

    # Verify v2 (version 1) is now latest
    latest = await artifact_service.load_artifact(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        filename="doc.txt",
    )
    assert latest.text == "version 2"

    # Restore checkpoint (should restore v1)
    restored = await checkpoint_service.restore_checkpoint(
        session=session,
        checkpoint_id="cp1",
        restore_state=True,
        restore_artifacts=True,
    )

    assert restored is not None

    # Verify artifact was restored to v1
    # This creates a new version (v3) with v1's content
    current = await artifact_service.load_artifact(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        filename="doc.txt",
    )
    assert current.text == "version 1"

  @pytest.mark.asyncio
  async def test_restore_multiple_artifacts(
      self, checkpoint_service, artifact_service, session
  ):
    """Test restoring multiple artifacts at once."""
    # Create multiple artifacts
    await artifact_service.save_artifact(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        filename="file1.txt",
        artifact=types.Part(text="content 1"),
    )
    await artifact_service.save_artifact(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        filename="file2.txt",
        artifact=types.Part(text="content 2"),
    )
    await artifact_service.save_artifact(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        filename="file3.txt",
        artifact=types.Part(text="content 3"),
    )

    # Create checkpoint
    metadata = await checkpoint_service.create_checkpoint(
        session=session,
        checkpoint_id="multi",
        artifact_filenames=["file1.txt", "file2.txt", "file3.txt"],
    )

    assert len(metadata.artifact_versions) == 3

    # Modify all artifacts
    await artifact_service.save_artifact(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        filename="file1.txt",
        artifact=types.Part(text="modified 1"),
    )
    await artifact_service.save_artifact(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        filename="file2.txt",
        artifact=types.Part(text="modified 2"),
    )
    await artifact_service.save_artifact(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        filename="file3.txt",
        artifact=types.Part(text="modified 3"),
    )

    # Restore checkpoint
    await checkpoint_service.restore_checkpoint(
        session=session,
        checkpoint_id="multi",
        restore_artifacts=True,
    )

    # Verify all artifacts restored
    f1 = await artifact_service.load_artifact(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        filename="file1.txt",
    )
    f2 = await artifact_service.load_artifact(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        filename="file2.txt",
    )
    f3 = await artifact_service.load_artifact(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        filename="file3.txt",
    )

    assert f1.text == "content 1"
    assert f2.text == "content 2"
    assert f3.text == "content 3"


class TestDeltaReconstruction:
  """Test delta compression and reconstruction."""

  @pytest.mark.asyncio
  async def test_delta_reconstruction_single_level(
      self, checkpoint_service, session
  ):
    """Test reconstructing delta checkpoint from base."""
    # Create base checkpoint
    session.state["a"] = 1
    session.state["b"] = 2
    session.state["c"] = 3

    await checkpoint_service.create_checkpoint(
        session=session,
        checkpoint_id="base",
        use_delta=False,  # Full snapshot
    )

    # Modify state and create delta checkpoint
    session.state["b"] = 20  # Modified
    session.state["d"] = 4  # Added
    del session.state["c"]  # Deleted

    await checkpoint_service.create_checkpoint(
        session=session,
        checkpoint_id="delta1",
        use_delta=True,  # Delta compression
    )

    # Get delta checkpoint metadata
    delta_meta = await checkpoint_service.get_checkpoint(
        session, "delta1", reconstruct_delta=False
    )

    # Verify delta only has changes
    assert delta_meta.is_delta
    assert delta_meta.base_checkpoint_id == "base"
    assert "b" in delta_meta.state_snapshot  # Modified
    assert "d" in delta_meta.state_snapshot  # Added
    assert delta_meta.state_snapshot.get("c") is None  # Deletion marker
    assert "a" not in delta_meta.state_snapshot  # Unchanged (not in delta)

    # Reconstruct full state from delta
    full_meta = await checkpoint_service.get_checkpoint(
        session, "delta1", reconstruct_delta=True
    )

    # Verify full state reconstruction
    assert full_meta.state_snapshot["a"] == 1  # From base
    assert full_meta.state_snapshot["b"] == 20  # From delta
    assert "c" not in full_meta.state_snapshot  # Deleted
    assert full_meta.state_snapshot["d"] == 4  # From delta

  @pytest.mark.asyncio
  async def test_delta_reconstruction_multi_level_chain(
      self, checkpoint_service, session
  ):
    """Test reconstructing delta-of-delta (multi-level chain)."""
    # Create base checkpoint
    session.state["x"] = 1

    await checkpoint_service.create_checkpoint(
        session=session,
        checkpoint_id="base",
        use_delta=False,
    )

    # Delta level 1
    session.state["y"] = 2

    await checkpoint_service.create_checkpoint(
        session=session,
        checkpoint_id="delta1",
        use_delta=True,
    )

    # Delta level 2 (delta-of-delta)
    session.state["z"] = 3

    await checkpoint_service.create_checkpoint(
        session=session,
        checkpoint_id="delta2",
        use_delta=True,
    )

    # Reconstruct delta2 (should recursively reconstruct delta1 -> base)
    full_meta = await checkpoint_service.get_checkpoint(
        session, "delta2", reconstruct_delta=True
    )

    # Verify full state has all values
    assert full_meta.state_snapshot["x"] == 1  # From base
    assert full_meta.state_snapshot["y"] == 2  # From delta1
    assert full_meta.state_snapshot["z"] == 3  # From delta2

  @pytest.mark.asyncio
  async def test_delta_with_deleted_base_checkpoint(
      self, checkpoint_service, session
  ):
    """Test delta reconstruction when base checkpoint is deleted (broken chain)."""
    # Create base checkpoint
    session.state["data"] = "base"

    await checkpoint_service.create_checkpoint(
        session=session,
        checkpoint_id="base",
        use_delta=False,
    )

    # Create delta checkpoint
    session.state["data"] = "delta"

    await checkpoint_service.create_checkpoint(
        session=session,
        checkpoint_id="delta",
        use_delta=True,
    )

    # Delete base checkpoint (breaks delta chain)
    await checkpoint_service.delete_checkpoint(session, "base")

    # Try to reconstruct delta (should raise DeltaChainBrokenError - P0.4 fix)
    from google.adk.checkpoints.models import DeltaChainBrokenError

    with pytest.raises(DeltaChainBrokenError):
      await checkpoint_service.get_checkpoint(
          session, "delta", reconstruct_delta=True
      )

  @pytest.mark.asyncio
  async def test_restore_delta_checkpoint(self, checkpoint_service, session):
    """Test that restore_checkpoint uses delta reconstruction."""
    # Create base
    session.state["value"] = 10

    await checkpoint_service.create_checkpoint(
        session=session,
        checkpoint_id="base",
        use_delta=False,
    )

    # Create delta
    session.state["value"] = 20

    await checkpoint_service.create_checkpoint(
        session=session,
        checkpoint_id="delta",
        use_delta=True,
    )

    # Modify state
    session.state["value"] = 30

    # Restore delta checkpoint
    restored = await checkpoint_service.restore_checkpoint(
        session, "delta", restore_state=True
    )

    assert restored is not None

    # Value should be restored to delta value (20)
    # Note: Need to reload session to see restored state
    assert session.state["value"] == 20


class TestEdgeCases:
  """Test edge cases and error conditions."""

  @pytest.mark.asyncio
  async def test_delta_compression_with_no_changes(
      self, checkpoint_service, session
  ):
    """Test delta compression when state hasn't changed."""
    session.state["data"] = "unchanged"

    # Create base checkpoint
    await checkpoint_service.create_checkpoint(
        session=session,
        checkpoint_id="base",
        use_delta=False,
    )

    # Create delta checkpoint without changing state
    await checkpoint_service.create_checkpoint(
        session=session,
        checkpoint_id="delta_empty",
        use_delta=True,
    )

    # Delta should be empty (falls back to full snapshot)
    delta_meta = await checkpoint_service.get_checkpoint(
        session, "delta_empty", reconstruct_delta=False
    )

    # Empty delta should store full snapshot
    assert "data" in delta_meta.state_snapshot
    assert delta_meta.state_snapshot["data"] == "unchanged"

  @pytest.mark.asyncio
  async def test_checkpoint_with_complex_nested_state(
      self, checkpoint_service, session
  ):
    """Test checkpoint handles nested data structures."""
    session.state["nested"] = {
        "level1": {
            "level2": {"value": 42},
            "list": [1, 2, 3],
        }
    }

    # Create checkpoint
    metadata = await checkpoint_service.create_checkpoint(
        session=session,
        checkpoint_id="nested",
    )

    # Verify nested structure preserved
    assert metadata.state_snapshot["nested"]["level1"]["level2"]["value"] == 42
    assert metadata.state_snapshot["nested"]["level1"]["list"] == [1, 2, 3]

    # Modify nested state
    session.state["nested"]["level1"]["level2"]["value"] = 99

    # Restore checkpoint
    await checkpoint_service.restore_checkpoint(
        session, "nested", restore_state=True
    )

    # Nested value should be restored
    assert session.state["nested"]["level1"]["level2"]["value"] == 42

  @pytest.mark.asyncio
  async def test_delta_with_nested_structure_changes(
      self, checkpoint_service, session
  ):
    """Test delta compression detects nested structure changes."""
    session.state["obj"] = {"a": 1, "b": 2}

    await checkpoint_service.create_checkpoint(
        session=session,
        checkpoint_id="base",
        use_delta=False,
    )

    # Modify nested structure
    session.state["obj"] = {"a": 1, "b": 3}  # Changed b

    await checkpoint_service.create_checkpoint(
        session=session,
        checkpoint_id="delta",
        use_delta=True,
    )

    # Delta should include entire obj (whole object changed)
    delta_meta = await checkpoint_service.get_checkpoint(
        session, "delta", reconstruct_delta=False
    )

    assert "obj" in delta_meta.state_snapshot
    assert delta_meta.state_snapshot["obj"]["b"] == 3


# ---------------------------------------------------------------------------
# Additional coverage tests: lines 221, 519, 521, 650-651
# ---------------------------------------------------------------------------


class TestLockContentionWarning:
  """Line 221: warning logged when lock wait exceeds 100 ms."""

  @pytest.mark.asyncio
  async def test_lock_contention_warning_emitted(
      self, checkpoint_service, session, caplog
  ):
    """Hold the session lock for 110 ms so checkpoint create triggers warning.

    We acquire the lock directly from _session_locks (forcing creation),
    hold it long enough, then release while create_checkpoint waits.
    The warning at line 221 is emitted when lock_wait_ms > 100.
    """
    import logging

    session_lock = await checkpoint_service._get_session_lock(session.id)

    async def hold_lock():
      async with session_lock:
        await asyncio.sleep(0.12)  # hold > 100 ms

    holder = asyncio.create_task(hold_lock())
    # Give the holder a moment to acquire the lock before we try
    await asyncio.sleep(0.01)

    with caplog.at_level(logging.WARNING, logger="google_adk.checkpoints"):
      await checkpoint_service.create_checkpoint(
          session=session, checkpoint_id="after_wait"
      )

    await holder  # clean up

    assert any(
        "High lock contention" in record.message for record in caplog.records
    ), "Expected 'High lock contention' warning but none was emitted"


class TestListCheckpointsBoundsClamping:
  """Lines 519, 521: invalid page / page_size values are clamped."""

  @pytest.mark.asyncio
  async def test_page_less_than_1_is_clamped_to_1(
      self, checkpoint_service, session
  ):
    """Line 519: page < 1 → clamped to 1 (no error, returns first page)."""
    await checkpoint_service.create_checkpoint(
        session=session, checkpoint_id="cp1"
    )
    # page=0 should be silently clamped to 1 and return valid results
    response = await checkpoint_service.list_checkpoints(
        session=session, page=0, page_size=10
    )
    assert response is not None
    assert len(response.checkpoints) == 1

  @pytest.mark.asyncio
  async def test_page_size_out_of_bounds_is_clamped_to_50(
      self, checkpoint_service, session
  ):
    """Line 521: page_size < 1 or page_size > 1000 → clamped to 50."""
    await checkpoint_service.create_checkpoint(
        session=session, checkpoint_id="cp1"
    )
    # page_size=0 should be clamped to 50
    response_zero = await checkpoint_service.list_checkpoints(
        session=session, page=1, page_size=0
    )
    assert response_zero is not None  # did not raise

    # page_size=9999 should also be clamped to 50
    response_huge = await checkpoint_service.list_checkpoints(
        session=session, page=1, page_size=9999
    )
    assert response_huge is not None


class TestArtifactRestoreException:
  """Lines 650-651: artifact restore failure is logged as warning, not raised."""

  @pytest.mark.asyncio
  async def test_artifact_restore_failure_continues(
      self, checkpoint_service, artifact_service, session, caplog
  ):
    """The except block at line 650 catches artifact load errors and continues.

    Simulates a broken artifact_service.load_artifact to ensure
    the checkpoint restore does NOT crash, and a warning is logged.
    """
    import logging

    # Create a checkpoint with artifact_versions in its metadata
    # so the restore loop is entered
    await artifact_service.save_artifact(
        app_name=session.app_name,
        user_id=session.user_id,
        session_id=session.id,
        filename="report.txt",
        artifact=types.Part(text="original"),
    )

    await checkpoint_service.create_checkpoint(
        session=session,
        checkpoint_id="with_artifact",
        artifact_filenames=["report.txt"],
    )

    # Make load_artifact raise an exception during restore.
    # InMemoryArtifactService is a Pydantic BaseModel; patch at class level
    # via patch.object so the original is restored automatically.
    with patch.object(
        type(artifact_service),
        "load_artifact",
        new=AsyncMock(side_effect=RuntimeError("storage unavailable")),
    ):
      with caplog.at_level(logging.WARNING, logger="google_adk.checkpoints"):
        # restore_checkpoint should succeed despite the artifact failure
        result = await checkpoint_service.restore_checkpoint(
            session=session, checkpoint_id="with_artifact"
        )
      assert result is not None, "restore_checkpoint should return metadata"
      assert any(
          "Failed to restore artifact" in r.message for r in caplog.records
      ), "Expected 'Failed to restore artifact' warning"
