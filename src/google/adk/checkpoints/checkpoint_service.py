"""CheckpointService for managing agent execution checkpoints.

This service provides checkpoint/resume capabilities using existing ADK primitives:
- SessionService for state persistence via EventActions.state_delta
- ArtifactService for versioned artifact tracking

The service is stateless - all checkpoint data is stored in session state.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import logging
import time
from typing import Any
from typing import AsyncIterator
from typing import Dict
from typing import Optional
import uuid

logger = logging.getLogger(__name__)

from ..artifacts.base_artifact_service import BaseArtifactService
from ..events.event import Event
from ..events.event_actions import EventActions
from ..sessions.base_session_service import BaseSessionService
from ..sessions.session import Session
from ..telemetry.checkpoint_tracing import record_checkpoint_metrics
from ..telemetry.checkpoint_tracing import trace_checkpoint_create
from ..telemetry.checkpoint_tracing import trace_checkpoint_delete
from ..telemetry.checkpoint_tracing import trace_checkpoint_list
from ..telemetry.checkpoint_tracing import trace_checkpoint_restore
from ..telemetry.checkpoint_tracing import tracer
from .models import CheckpointCorruptedError
from .models import CheckpointMetadata
from .models import CheckpointNotFoundError
from .models import DeltaChainBrokenError
from .models import ListCheckpointsResponse


@dataclass
class CheckpointServiceConfig:
  """Configuration for CheckpointService resource limits.

  Attributes:
      max_checkpoints_per_session: Maximum number of checkpoints allowed per session.
          Default: 100. Set to 0 for unlimited (not recommended for production).
      max_state_size_bytes: Maximum size of state snapshot in bytes.
          Default: 10MB. Prevents memory exhaustion from very large states.
  """

  max_checkpoints_per_session: int = 100
  max_state_size_bytes: int = 10 * 1024 * 1024  # 10MB

  def __post_init__(self) -> None:
    """Validate configuration values."""
    if self.max_checkpoints_per_session < 0:
      raise ValueError("max_checkpoints_per_session must be >= 0")
    if self.max_state_size_bytes < 0:
      raise ValueError("max_state_size_bytes must be >= 0")


class CheckpointService:
  """Service for creating and managing checkpoints.

  Integrates with SessionService and ArtifactService to provide checkpoint
  capabilities for any ADK agent. Checkpoints are stored as events with
  state_delta, making them fully integrated with ADK's event system.

  Example:
      ```python
      checkpoint_service = CheckpointService(
          session_service=session_service,
          artifact_service=artifact_service,
      )

      # Create checkpoint (ID auto-generated)
      checkpoint_metadata = await checkpoint_service.create_checkpoint(
          session=session,
          description="Before critical operation",
          agent_name="my_agent",
      )
      print(f"Created checkpoint: {checkpoint_metadata.checkpoint_id}")

      # Or create with explicit ID
      checkpoint_metadata = await checkpoint_service.create_checkpoint(
          session=session,
          checkpoint_id="checkpoint-1",
          description="Before critical operation",
          agent_name="my_agent",
      )

      # Resume from checkpoint
      metadata = await checkpoint_service.get_checkpoint(
          session=session,
          checkpoint_id=checkpoint_metadata.checkpoint_id,
      )
      ```
  """

  def __init__(
      self,
      session_service: BaseSessionService,
      artifact_service: Optional[BaseArtifactService] = None,
      config: Optional[CheckpointServiceConfig] = None,
  ) -> None:
    """Initialize checkpoint service.

    Args:
        session_service: Session service for state persistence
        artifact_service: Optional artifact service for tracking artifact versions
        config: Optional configuration for resource limits. If None, uses defaults.
    """
    self.session_service = session_service
    self.artifact_service = artifact_service
    self.config = config or CheckpointServiceConfig()

    # Per-session locks prevent concurrent writes to the same session's checkpoints
    self._session_locks: Dict[str, asyncio.Lock] = {}
    self._locks_lock = asyncio.Lock()

  async def _get_session_lock(self, session_id: str) -> asyncio.Lock:
    """Get or create lock for a specific session.

    Uses double-checked locking pattern for thread-safe lock acquisition.
    Per-session locks ensure different sessions don't block each other.

    Args:
        session_id: Session ID to get lock for

    Returns:
        Lock for the session
    """
    # Fast path: lock exists
    if session_id in self._session_locks:
      return self._session_locks[session_id]

    # Slow path: create lock with synchronization
    async with self._locks_lock:
      # Double-check: another coroutine may have created it
      if session_id not in self._session_locks:
        self._session_locks[session_id] = asyncio.Lock()
        logger.debug(f"Created new checkpoint lock for session {session_id}")
      return self._session_locks[session_id]

  @asynccontextmanager
  async def _traced(
      self, operation: str, checkpoint_id: str, session_id: str
  ) -> AsyncIterator[Any]:
    """Internal helper for tracing checkpoint operations with metrics."""
    start_time = time.time()
    with tracer.start_as_current_span(
        f"checkpoint.{operation}",
        attributes={
            "checkpoint.operation": operation,
            "checkpoint.id": checkpoint_id,
            "checkpoint.session_id": session_id,
        },
    ) as span:
      try:
        yield span
        duration_ms = (time.time() - start_time) * 1000
        record_checkpoint_metrics(operation, duration_ms, "success")
      except Exception:
        duration_ms = (time.time() - start_time) * 1000
        record_checkpoint_metrics(operation, duration_ms, "error")
        raise

  async def create_checkpoint(
      self,
      session: Session,
      description: Optional[str] = None,
      agent_name: Optional[str] = None,
      checkpoint_id: Optional[str] = None,
      artifact_filenames: Optional[list[str]] = None,
      custom_metadata: Optional[dict[str, Any]] = None,
      use_delta: bool = True,
  ) -> CheckpointMetadata:
    """Create a checkpoint by storing current session state.

    This method creates a checkpoint event with state_delta containing:
    - Current session state snapshot
    - Artifact versions at checkpoint time (if artifact_service provided)
    - Metadata about the checkpoint

    The checkpoint is fully integrated with ADK's event system - it's just
    an event with state_delta that can be inspected, replayed, or exported.

    Args:
        session: Session to checkpoint
        description: Human-readable description
        agent_name: Name of agent creating checkpoint
        checkpoint_id: Optional unique identifier. Auto-generates UUID if not provided.
        artifact_filenames: Optional list of artifact filenames to track.
            If None and artifact_service is provided, tracks all artifacts.
        custom_metadata: Additional user-defined metadata
        use_delta: Whether to use delta compression (default: True).
            If True, stores only state changes from previous checkpoint.
            If False, stores full state snapshot. Delta mode saves memory.

    Returns:
        CheckpointMetadata with checkpoint details including generated checkpoint_id

    Note:
        This service is not thread-safe. Use separate instances for concurrent access.
    """
    # Auto-generate checkpoint ID if not provided
    if checkpoint_id is None:
      checkpoint_id = f"checkpoint-{uuid.uuid4()}"

    # Acquire per-session lock to prevent race conditions
    session_lock = await self._get_session_lock(session.id)
    lock_start_time = time.time()

    async with session_lock:
      lock_wait_ms = (time.time() - lock_start_time) * 1000

      # Log high contention for monitoring
      if lock_wait_ms > 100:
        logger.warning(
            f"High lock contention for session {session.id}: "
            f"waited {lock_wait_ms:.2f}ms to acquire checkpoint lock"
        )

      async with self._traced("create", checkpoint_id, session.id) as _:
        # Record lock wait time in telemetry
        with tracer.start_as_current_span("checkpoint.lock_wait") as span:
          span.set_attribute("checkpoint.lock_wait_ms", lock_wait_ms)
          span.set_attribute("checkpoint.session_id", session.id)

        # Validate checkpoint count limit (now synchronized)
        if self.config.max_checkpoints_per_session > 0:
          checkpoint_index = session.state.get("_checkpoint_index", {})
          current_count = len(checkpoint_index)
          if current_count >= self.config.max_checkpoints_per_session:
            raise ValueError(
                f"Checkpoint limit reached: {current_count} checkpoints exist"
                f" (max: {self.config.max_checkpoints_per_session}). Delete old"
                " checkpoints or increase max_checkpoints_per_session limit."
            )

        # Collect artifact versions if artifact service available
        artifact_versions = {}
        if self.artifact_service and session.app_name and session.user_id:
          if artifact_filenames is None:
            # Track all artifacts in session
            artifact_filenames = await self.artifact_service.list_artifact_keys(
                app_name=session.app_name,
                user_id=session.user_id,
                session_id=session.id,
            )

          # Get current version for each artifact
          for filename in artifact_filenames:
            versions = await self.artifact_service.list_versions(
                app_name=session.app_name,
                user_id=session.user_id,
                filename=filename,
                session_id=session.id,
            )
            if versions:
              artifact_versions[filename] = max(versions)

        # Compute state snapshot (delta or full)
        state_snapshot = {}
        base_checkpoint_id = None

        if use_delta:
          # Get previous checkpoint to compute delta
          checkpoint_index = session.state.get("_checkpoint_index", {})
          if checkpoint_index:
            # Get most recent checkpoint
            sorted_ids = sorted(
                checkpoint_index.keys(),
                key=lambda cid: checkpoint_index[cid]["timestamp"],
                reverse=True,
            )
            if sorted_ids:
              base_checkpoint_id = sorted_ids[0]
              try:
                base_metadata = await self.get_checkpoint(
                    session, base_checkpoint_id
                )
                prev_state = base_metadata.state_snapshot
                # Compute delta (only changed keys)
                for key, value in session.state.items():
                  if key.startswith("_checkpoint"):
                    continue  # Skip checkpoint keys
                  if key not in prev_state or prev_state[key] != value:
                    state_snapshot[key] = value
                # Track deleted keys
                for key in prev_state:
                  if key not in session.state and not key.startswith(
                      "_checkpoint"
                  ):
                    state_snapshot[key] = None  # Deletion marker
              except (
                  CheckpointNotFoundError,
                  CheckpointCorruptedError,
              ) as e:
                # Base checkpoint not found or corrupted - fall back to full snapshot
                logger.warning(
                    "Cannot compute delta: base checkpoint"
                    f" {base_checkpoint_id} unavailable ({type(e).__name__})."
                    " Using full snapshot instead."
                )
                base_checkpoint_id = None  # Reset to force full snapshot

        if not state_snapshot:  # Empty delta or full snapshot mode
          # Full snapshot (filter out checkpoint keys)
          state_snapshot = {
              k: v
              for k, v in session.state.items()
              if not k.startswith("_checkpoint")
          }

        # Calculate state size for telemetry and validation
        # Use PydanticJSONEncoder to handle any Pydantic models in state
        from .utils import PydanticJSONEncoder

        state_size = len(
            json.dumps(state_snapshot, cls=PydanticJSONEncoder).encode("utf-8")
        )

        # Validate state size limit
        if self.config.max_state_size_bytes > 0:
          if state_size > self.config.max_state_size_bytes:
            raise ValueError(
                f"State size {state_size} bytes exceeds limit"
                f" ({self.config.max_state_size_bytes} bytes). Consider using"
                " delta compression (use_delta=True) or reducing state size."
                f" Current state has {len(state_snapshot)} keys."
            )

        # Create checkpoint metadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            description=description,
            agent_name=agent_name,
            artifact_versions=artifact_versions,
            state_snapshot=state_snapshot,
            is_delta=use_delta and base_checkpoint_id is not None,
            base_checkpoint_id=base_checkpoint_id,
            custom_metadata=custom_metadata or {},
        )

        # Update checkpoint index (O(1) lookups)
        checkpoint_index = session.state.get("_checkpoint_index", {})
        checkpoint_index[checkpoint_id] = {
            "timestamp": metadata.timestamp,
            "agent": agent_name,
        }

        # Atomic operation: batch checkpoint + index in single event
        checkpoint_event = Event(
            author=agent_name or "checkpoint_service",
            actions=EventActions(
                state_delta={
                    f"_checkpoint_{checkpoint_id}": metadata.model_dump(),
                    "_checkpoint_index": checkpoint_index,
                }
            ),
        )

        # Append to session (updates session.state automatically)
        await self.session_service.append_event(session, checkpoint_event)

        # Trace call
        trace_checkpoint_create(
            checkpoint_id=checkpoint_id,
            session=session,
            agent_name=agent_name,
            description=description,
            artifact_count=len(artifact_versions),
            state_size_bytes=state_size,
        )

        return metadata

  async def get_checkpoint(
      self,
      session: Session,
      checkpoint_id: str,
      reconstruct_delta: bool = False,
  ) -> CheckpointMetadata:
    """Retrieve checkpoint metadata from session state.

    Args:
        session: Session containing the checkpoint
        checkpoint_id: Checkpoint identifier
        reconstruct_delta: If True and checkpoint is delta, reconstruct full state

    Returns:
        CheckpointMetadata with checkpoint data

    Raises:
        CheckpointNotFoundError: If checkpoint does not exist
        DeltaChainBrokenError: If delta chain is broken (base checkpoint missing)
        CheckpointCorruptedError: If checkpoint data is invalid
    """
    async with self._traced("get", checkpoint_id, session.id) as span:
      checkpoint_key = f"_checkpoint_{checkpoint_id}"
      checkpoint_data = session.state.get(checkpoint_key)

      if checkpoint_data is None:
        # Checkpoint doesn't exist - raise specific error for clear error handling
        span.set_attribute("checkpoint.error", "not_found")
        span.add_event(
            "checkpoint_not_found",
            {"checkpoint_id": checkpoint_id, "session_id": session.id},
        )
        logger.error(
            f"Checkpoint {checkpoint_id} not found in session {session.id}"
        )
        raise CheckpointNotFoundError(
            f"Checkpoint {checkpoint_id} not found in session {session.id}"
        )

      # Parse checkpoint metadata - catch corruption
      try:
        metadata = CheckpointMetadata(**checkpoint_data)
      except Exception as e:
        # Checkpoint data is corrupted
        span.set_attribute("checkpoint.error", "corrupted")
        span.add_event(
            "checkpoint_corrupted",
            {
                "checkpoint_id": checkpoint_id,
                "session_id": session.id,
                "error": str(e),
            },
        )
        logger.error(
            f"Checkpoint {checkpoint_id} data is corrupted: {e}",
            exc_info=True,
        )
        raise CheckpointCorruptedError(
            f"Checkpoint {checkpoint_id} data is corrupted: {e}"
        ) from e

      # Reconstruct full state from delta chain if requested
      if (
          reconstruct_delta
          and metadata.is_delta
          and metadata.base_checkpoint_id
      ):
        # Track delta reconstruction in telemetry
        with tracer.start_as_current_span(
            "checkpoint.delta_reconstruction"
        ) as delta_span:
          delta_span.set_attribute("checkpoint.checkpoint_id", checkpoint_id)
          delta_span.set_attribute(
              "checkpoint.base_checkpoint_id", metadata.base_checkpoint_id
          )

          # Load base checkpoint (recursively reconstruct if it's also a delta)
          try:
            base_metadata = await self.get_checkpoint(
                session, metadata.base_checkpoint_id, reconstruct_delta=True
            )
          except CheckpointNotFoundError as e:
            # Base checkpoint missing - delta chain is broken
            delta_span.set_attribute("checkpoint.error", "chain_broken")
            delta_span.add_event(
                "delta_chain_broken",
                {
                    "checkpoint_id": checkpoint_id,
                    "base_checkpoint_id": metadata.base_checkpoint_id,
                    "session_id": session.id,
                },
            )
            logger.error(
                "Delta chain broken: base checkpoint "
                f"{metadata.base_checkpoint_id} not found for delta "
                f"checkpoint {checkpoint_id}. Cannot reconstruct full state."
            )
            raise DeltaChainBrokenError(
                "Delta chain broken: base checkpoint "
                f"{metadata.base_checkpoint_id} not found for delta "
                f"checkpoint {checkpoint_id}"
            ) from e

          # Reconstruct full state by applying delta on top of base
          full_state = dict(base_metadata.state_snapshot)
          for key, value in metadata.state_snapshot.items():
            if value is None:  # Deletion marker
              full_state.pop(key, None)
            else:
              full_state[key] = value

          # Return metadata with full state
          metadata.state_snapshot = full_state

          delta_span.set_attribute(
              "checkpoint.reconstructed_keys", len(full_state)
          )

      return metadata

  async def list_checkpoints(
      self,
      session: Session,
      page: int = 1,
      page_size: int = 50,
  ) -> ListCheckpointsResponse:
    """List checkpoints in a session with pagination.

    Uses checkpoint index for O(1) performance instead of O(n) iteration.
    Follows ADK pagination patterns (similar to ConversationListResponse).

    Args:
        session: Session to list checkpoints from
        page: Page number (1-indexed, default: 1)
        page_size: Number of checkpoints per page (default: 50, max: 1000)

    Returns:
        ListCheckpointsResponse with paginated checkpoints

    Note:
        This service is not thread-safe. Use separate instances for concurrent access.
    """
    # Validate bounds (ADK pattern)
    if page < 1:
      page = 1
    if page_size < 1 or page_size > 1000:
      page_size = 50

    async with self._traced("list", "list", session.id) as _:
      # O(1) lookup from index
      checkpoint_index = session.state.get("_checkpoint_index", {})

      # Sort by timestamp (most recent first)
      sorted_ids = sorted(
          checkpoint_index.keys(),
          key=lambda cid: checkpoint_index[cid]["timestamp"],
          reverse=True,
      )

      total_count = len(sorted_ids)

      # Paginate
      offset = (page - 1) * page_size
      paginated_ids = sorted_ids[offset : offset + page_size]

      # Fetch metadata for current page
      checkpoints = []
      for cid in paginated_ids:
        try:
          metadata = await self.get_checkpoint(session, cid)
          checkpoints.append(metadata)
        except (CheckpointNotFoundError, CheckpointCorruptedError) as e:
          # Skip corrupted/missing checkpoints in listing
          logger.warning(
              f"Skipping checkpoint {cid} in list: {type(e).__name__}"
          )

      # Build response
      response = ListCheckpointsResponse(
          checkpoints=checkpoints,
          total_count=total_count,
          page=page,
          page_size=page_size,
          has_next=offset + len(paginated_ids) < total_count,
          has_previous=page > 1,
      )

      trace_checkpoint_list(
          session=session,
          checkpoint_count=len(checkpoints),
      )

      return response

  async def restore_checkpoint(
      self,
      session: Session,
      checkpoint_id: str,
      restore_state: bool = True,
      restore_artifacts: bool = True,
  ) -> Optional[CheckpointMetadata]:
    """Restore session state and artifacts from a checkpoint.

    This method:
    1. Retrieves checkpoint metadata
    2. Optionally restores session state to checkpoint snapshot
    3. Optionally restores artifacts to their checkpoint versions

    Note: State restoration creates a new event with state_delta to restore
    the state, maintaining the event timeline.

    Args:
        session: Session to restore
        checkpoint_id: Checkpoint identifier
        restore_state: Whether to restore session state
        restore_artifacts: Whether to restore artifact versions

    Returns:
        CheckpointMetadata with restored checkpoint data

    Raises:
        CheckpointNotFoundError: If checkpoint does not exist
        DeltaChainBrokenError: If delta chain is broken
        CheckpointCorruptedError: If checkpoint data is invalid
    """
    async with self._traced("restore", checkpoint_id, session.id) as _:
      # Get checkpoint metadata with delta reconstruction
      # Raises CheckpointNotFoundError / CheckpointCorruptedError on failure
      metadata = await self.get_checkpoint(
          session, checkpoint_id, reconstruct_delta=True
      )

      # Restore state via event (maintains event timeline)
      if restore_state:
        restore_event = Event(
            author="checkpoint_service",
            actions=EventActions(state_delta=metadata.state_snapshot),
        )
        await self.session_service.append_event(session, restore_event)

      # Restore artifacts to checkpoint versions
      if (
          restore_artifacts
          and self.artifact_service
          and session.app_name
          and session.user_id
      ):
        artifact_versions = metadata.artifact_versions

        if artifact_versions:
          # Restore each artifact to checkpoint version
          for filename, version in artifact_versions.items():
            try:
              # Get artifact at checkpoint version
              artifact_data = await self.artifact_service.load_artifact(
                  app_name=session.app_name,
                  user_id=session.user_id,
                  session_id=session.id,
                  filename=filename,
                  version=version,
              )

              # Save as latest version (creates new version)
              if artifact_data:
                await self.artifact_service.save_artifact(
                    app_name=session.app_name,
                    user_id=session.user_id,
                    session_id=session.id,
                    filename=filename,
                    artifact=artifact_data,
                )
            except Exception as e:
              logger.warning(
                  f"Failed to restore artifact {filename} "
                  f"version {version}: {e}"
              )
              # Continue with other artifacts

      trace_checkpoint_restore(
          checkpoint_id=checkpoint_id,
          session=session,
      )

      return metadata

  async def delete_checkpoint(
      self,
      session: Session,
      checkpoint_id: str,
  ) -> bool:
    """Delete a checkpoint from session state.

    Creates an event that removes the checkpoint from state.

    Args:
        session: Session containing the checkpoint
        checkpoint_id: Checkpoint identifier

    Returns:
        True if checkpoint was deleted, False if not found
    """
    async with self._traced("delete", checkpoint_id, session.id) as _:
      checkpoint_key = f"_checkpoint_{checkpoint_id}"

      if checkpoint_key not in session.state:
        return False

      # Update checkpoint index (remove this checkpoint)
      checkpoint_index = session.state.get("_checkpoint_index", {})
      if checkpoint_id in checkpoint_index:
        del checkpoint_index[checkpoint_id]

      # Create deletion event (atomic: remove checkpoint + update index)
      delete_event = Event(
          author="checkpoint_service",
          actions=EventActions(
              state_delta={
                  checkpoint_key: None,  # Delete checkpoint
                  "_checkpoint_index": checkpoint_index,  # Update index
              }
          ),
      )

      await self.session_service.append_event(session, delete_event)

      trace_checkpoint_delete(
          checkpoint_id=checkpoint_id,
          session=session,
      )

      return True
