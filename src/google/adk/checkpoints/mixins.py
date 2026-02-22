"""Checkpointing mixin for making any agent checkpointable.

This mixin allows any agent to easily add checkpointing capabilities
without modifying core agent logic.

Example:
    ```python
    from google.adk.checkpoints.mixins import CheckpointableMixin

    class MyCustomAgent(CheckpointableMixin, BaseAgent):
        def __init__(self, name: str, checkpoint_service: CheckpointService):
            BaseAgent.__init__(self, name=name)
            CheckpointableMixin.__init__(
                self,
                checkpoint_service=checkpoint_service,
                auto_checkpoint=True
            )

        async def _run_async_impl(self, ctx):
            # Your agent logic here
            yield Event(...)

            # Optionally create checkpoint at key points
            if should_checkpoint:
                await self.create_checkpoint(
                    ctx.session,
                    description="After critical operation"
                )
    ```
"""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import Optional
import warnings

from ..sessions import Session
from .checkpoint_service import CheckpointService


class CheckpointableMixin:
  """Mixin to add checkpointing capabilities to any agent.

  This mixin provides:
  - Automatic checkpoint creation before/after agent execution
  - Manual checkpoint creation at any point
  - Checkpoint restoration
  - State diff tracking between checkpoints

  Usage:
      Inherit from this mixin BEFORE BaseAgent in your class definition:

      ```python
      class MyAgent(CheckpointableMixin, BaseAgent):
          def __init__(self, name, checkpoint_service):
              BaseAgent.__init__(self, name=name)
              CheckpointableMixin.__init__(
                  self,
                  checkpoint_service=checkpoint_service
              )
      ```
  """

  def __init__(
      self,
      checkpoint_service: Optional[CheckpointService] = None,
      auto_checkpoint: bool = False,
      checkpoint_before: bool = False,
      checkpoint_after: bool = True,
  ):
    """Initialize checkpointing mixin.

    Args:
        checkpoint_service: CheckpointService instance for creating/managing checkpoints.
            If None, checkpointing is disabled.
        auto_checkpoint: Enable automatic checkpointing (default: False)
        checkpoint_before: Create checkpoint BEFORE agent execution (default: False)
        checkpoint_after: Create checkpoint AFTER agent execution (default: True)
    """
    self._checkpoint_service = checkpoint_service
    self._auto_checkpoint = auto_checkpoint
    self._checkpoint_before = checkpoint_before
    self._checkpoint_after = checkpoint_after
    self._last_checkpoint_id: Optional[str] = None

    if auto_checkpoint and not checkpoint_service:
      warnings.warn(
          "auto_checkpoint=True but no checkpoint_service provided. "
          "Checkpointing will be disabled."
      )

  async def create_checkpoint(
      self,
      session: Session,
      description: Optional[str] = None,
      metadata: Optional[Dict[str, Any]] = None,
      use_delta: bool = True,
  ) -> Optional[str]:
    """Create a checkpoint manually.

    Args:
        session: Current session
        description: Optional description for this checkpoint
        metadata: Optional metadata dict
        use_delta: Whether to use delta compression (default: True)

    Returns:
        Checkpoint ID if created, None if checkpointing disabled
    """
    if not self._checkpoint_service:
      return None

    checkpoint_metadata = await self._checkpoint_service.create_checkpoint(
        session=session,
        description=description or f"Checkpoint from {self.name}",  # type: ignore
        custom_metadata=metadata or {},
        use_delta=use_delta,
    )
    checkpoint_id: str = checkpoint_metadata.checkpoint_id
    self._last_checkpoint_id = checkpoint_id
    return checkpoint_id

  async def restore_checkpoint(
      self,
      session: Session,
      checkpoint_id: str,
  ) -> Dict[str, Any]:
    """Restore state from a checkpoint.

    Args:
        session: Current session
        checkpoint_id: ID of checkpoint to restore

    Returns:
        Restored state dict

    Raises:
        ValueError: If checkpointing is disabled or checkpoint not found
    """
    if not self._checkpoint_service:
      raise ValueError("Checkpointing is disabled (no checkpoint_service)")

    metadata = await self._checkpoint_service.restore_checkpoint(
        session=session,
        checkpoint_id=checkpoint_id,
    )

    if metadata is None:
      raise ValueError(f"Checkpoint {checkpoint_id} not found")

    # Return the state snapshot from metadata
    state_snapshot: Dict[str, Any] = metadata.state_snapshot
    return state_snapshot

  async def list_checkpoints(
      self,
      session: Session,
      page: int = 1,
      page_size: int = 50,
  ) -> list[Any]:
    """List checkpoints for this session.

    Args:
        session: Current session
        page: Page number (1-indexed, default: 1)
        page_size: Number of checkpoints per page (default: 50)

    Returns:
        List of checkpoint metadata

    Raises:
        ValueError: If checkpointing is disabled
    """
    if not self._checkpoint_service:
      raise ValueError("Checkpointing is disabled (no checkpoint_service)")

    response = await self._checkpoint_service.list_checkpoints(
        session=session,
        page=page,
        page_size=page_size,
    )
    checkpoints: list[Any] = response.checkpoints
    return checkpoints

  async def get_checkpoint_diff(
      self,
      session: Session,
      checkpoint_id_1: str,
      checkpoint_id_2: str,
  ) -> Dict[str, Any]:
    """Get diff between two checkpoints.

    Args:
        session: Current session
        checkpoint_id_1: First checkpoint ID
        checkpoint_id_2: Second checkpoint ID

    Returns:
        Dict with added, removed, and changed keys

    Raises:
        ValueError: If checkpointing is disabled
    """
    if not self._checkpoint_service:
      raise ValueError("Checkpointing is disabled (no checkpoint_service)")

    from .utils import compute_state_diff

    # Get checkpoint metadata WITHOUT restoring (to avoid modifying session state)
    metadata1 = await self._checkpoint_service.get_checkpoint(
        session, checkpoint_id_1
    )
    metadata2 = await self._checkpoint_service.get_checkpoint(
        session, checkpoint_id_2
    )

    if metadata1 is None or metadata2 is None:
      raise ValueError("One or both checkpoints not found")

    # Extract state snapshots from metadata
    state1 = metadata1.state_snapshot
    state2 = metadata2.state_snapshot

    diff_result: Dict[str, Any] = compute_state_diff(state1, state2)
    return diff_result

  @property
  def last_checkpoint_id(self) -> Optional[str]:
    """Get ID of most recently created checkpoint."""
    return self._last_checkpoint_id

  @property
  def checkpointing_enabled(self) -> bool:
    """Check if checkpointing is enabled."""
    return self._checkpoint_service is not None

  async def _checkpoint_before_execution(self, session: Session) -> None:
    """Internal: Create checkpoint before execution if auto_checkpoint enabled."""
    if self._auto_checkpoint and self._checkpoint_before:
      await self.create_checkpoint(
          session,
          description=f"Before {self.name} execution",  # type: ignore
          metadata={"timing": "before", "agent": self.name},  # type: ignore
      )

  async def _checkpoint_after_execution(self, session: Session) -> None:
    """Internal: Create checkpoint after execution if auto_checkpoint enabled."""
    if self._auto_checkpoint and self._checkpoint_after:
      await self.create_checkpoint(
          session,
          description=f"After {self.name} execution",  # type: ignore
          metadata={"timing": "after", "agent": self.name},  # type: ignore
      )
