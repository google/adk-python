"""Generic agent-level checkpoint callback for CheckpointService.

This module provides CheckpointCallback for integrating CheckpointService
with any BaseAgent subclass via the before_agent/after_agent callback system.

For GraphAgent node-level checkpointing (before_node/after_node), use
GraphCheckpointCallback from google.adk.agents.graph.checkpoint_callback.
"""

from __future__ import annotations

from typing import Optional

from google.genai import types

from ..agents.callback_context import CallbackContext
from .checkpoint_service import CheckpointService


class CheckpointCallback:
  """Callback wrapper for CheckpointService integration with any BaseAgent.

  Enables automatic checkpointing before and/or after agent execution
  for ANY BaseAgent subclass (GraphAgent, LlmAgent, SequentialAgent, etc.).

  Example:
      ```python
      from google.adk.checkpoints import CheckpointService, CheckpointCallback

      checkpoint_service = CheckpointService(
          session_service=session_service,
          artifact_service=artifact_service,
      )
      checkpoint_callback = CheckpointCallback(checkpoint_service)

      agent = LlmAgent(
          name="my_agent",
          before_agent_callback=checkpoint_callback.before_agent,
          after_agent_callback=checkpoint_callback.after_agent,
      )
      ```
  """

  def __init__(
      self,
      checkpoint_service: CheckpointService,
      checkpoint_before: bool = True,
      checkpoint_after: bool = True,
  ):
    """Initialize checkpoint callback.

    Args:
        checkpoint_service: CheckpointService instance to use
        checkpoint_before: Create checkpoint before agent execution
        checkpoint_after: Create checkpoint after agent execution
    """
    self.service = checkpoint_service
    self.checkpoint_before = checkpoint_before
    self.checkpoint_after = checkpoint_after

  async def before_agent(
      self,
      callback_context: CallbackContext,
  ) -> Optional[types.Content]:
    """Create checkpoint before agent execution.

    Args:
        callback_context: Callback context from BaseAgent

    Returns:
        None (doesn't override agent execution)
    """
    if not self.checkpoint_before:
      return None

    session = callback_context.session
    agent_name = callback_context.agent_name

    checkpoint_id = f"{session.id}-{agent_name}-before"

    await self.service.create_checkpoint(
        session=session,
        checkpoint_id=checkpoint_id,
        description=f"Before {agent_name} execution",
        agent_name=agent_name,
    )

    return None

  async def after_agent(
      self,
      callback_context: CallbackContext,
  ) -> Optional[types.Content]:
    """Create checkpoint after agent execution.

    Args:
        callback_context: Callback context from BaseAgent

    Returns:
        None (doesn't override agent execution)
    """
    if not self.checkpoint_after:
      return None

    session = callback_context.session
    agent_name = callback_context.agent_name

    checkpoint_id = f"{session.id}-{agent_name}-after"

    await self.service.create_checkpoint(
        session=session,
        checkpoint_id=checkpoint_id,
        description=f"After {agent_name} execution",
        agent_name=agent_name,
    )

    return None
