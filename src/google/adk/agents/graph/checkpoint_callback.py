"""Graph-specific checkpoint callback for node-level checkpointing.

Extends the generic CheckpointCallback with before_node/after_node methods
that integrate with GraphAgent's node callback system.
"""

from __future__ import annotations

import json
from typing import Optional
from typing import Set
from typing import TYPE_CHECKING

from ...checkpoints.callback import CheckpointCallback
from ...checkpoints.checkpoint_service import CheckpointService

if TYPE_CHECKING:
  from .callbacks import NodeCallbackContext


class GraphCheckpointCallback(CheckpointCallback):
  """Checkpoint callback with GraphAgent node-level support.

  Extends CheckpointCallback with before_node/after_node methods for
  per-node checkpointing in GraphAgent workflows.

  Inherits agent-level callbacks (before_agent/after_agent) from
  CheckpointCallback — works with any BaseAgent subclass.

  Node-level example (selective checkpointing per node):
      ```python
      from google.adk.agents.graph import GraphCheckpointCallback

      checkpoint_service = CheckpointService(session_service=session_service)
      # Checkpoint only after critical nodes, not every node
      checkpoint_callback = GraphCheckpointCallback(
          checkpoint_service,
          checkpoint_nodes={"analyze", "execute"},  # only these nodes
      )

      graph = GraphAgent(
          name="workflow",
          after_node_callback=checkpoint_callback.after_node,
      )
      ```

  Agent-proposed checkpoint example (LLM decides when to checkpoint):
      ```python
      from pydantic import BaseModel

      class AnalysisOutput(BaseModel):
          finding: str
          risk_level: str
          checkpoint_requested: bool = False

      analyzer = LlmAgent(
          name="analyzer",
          output_schema=AnalysisOutput,
          instruction="... Set checkpoint_requested=true if risk_level is 'high'.",
      )

      checkpoint_callback = GraphCheckpointCallback(
          checkpoint_service,
          checkpoint_after=False,  # no automatic checkpoints
          checkpoint_request_key="analyzer.checkpoint_requested",
      )
      graph = GraphAgent(
          name="workflow",
          after_node_callback=checkpoint_callback.after_node,
      )
      ```
  """

  def __init__(
      self,
      checkpoint_service: CheckpointService,
      checkpoint_before: bool = True,
      checkpoint_after: bool = True,
      checkpoint_nodes: Optional[Set[str]] = None,
      checkpoint_request_key: Optional[str] = None,
  ):
    """Initialize graph checkpoint callback.

    Args:
        checkpoint_service: CheckpointService instance to use
        checkpoint_before: Create checkpoint before agent/node execution
        checkpoint_after: Create checkpoint after agent/node execution
        checkpoint_nodes: For node-level callbacks, only checkpoint these nodes.
            None means checkpoint all nodes. Has no effect on agent-level callbacks.
        checkpoint_request_key: Dotted path "state_key.bool_field" that an LLM
            agent can set to propose a checkpoint (e.g. "analyzer.checkpoint_requested").
            When the named node finishes and the field is truthy, an additional
            checkpoint is created. Default None (disabled). Opt-in only.
    """
    super().__init__(
        checkpoint_service=checkpoint_service,
        checkpoint_before=checkpoint_before,
        checkpoint_after=checkpoint_after,
    )
    self.checkpoint_nodes = checkpoint_nodes
    # Pre-parse dotted key once to avoid repeated string splits at runtime
    if checkpoint_request_key:
      parts = checkpoint_request_key.split(".", 1)
      self._req_state_key: Optional[str] = parts[0]
      self._req_field: Optional[str] = (
          parts[1] if len(parts) > 1 else "checkpoint_requested"
      )
    else:
      self._req_state_key = None
      self._req_field = None

  def _should_checkpoint_node(self, node_name: str) -> bool:
    """Check if a specific node should be checkpointed."""
    if self.checkpoint_nodes is None:
      return True
    return node_name in self.checkpoint_nodes

  async def before_node(self, ctx: "NodeCallbackContext") -> None:
    """Create checkpoint before a GraphAgent node executes.

    Used with GraphAgent's before_node_callback. Supports selective
    checkpointing via checkpoint_nodes parameter.

    Args:
        ctx: Node callback context from GraphAgent

    Returns:
        None (checkpoint stored via session_service.append_event)
    """
    if not self.checkpoint_before:
      return None

    if not self._should_checkpoint_node(ctx.node.name):
      return None

    session = ctx.invocation_context.session
    checkpoint_id = f"{session.id}-{ctx.node.name}-{ctx.iteration}-before"

    await self.service.create_checkpoint(
        session=session,
        checkpoint_id=checkpoint_id,
        description=f"Before node {ctx.node.name} (iteration {ctx.iteration})",
        agent_name=ctx.node.name,
    )

    return None

  async def after_node(self, ctx: "NodeCallbackContext") -> None:
    """Create checkpoint after a GraphAgent node completes.

    Used with GraphAgent's after_node_callback. Supports selective
    checkpointing via checkpoint_nodes parameter, and agent-proposed
    checkpointing via checkpoint_request_key.

    Args:
        ctx: Node callback context from GraphAgent

    Returns:
        None (checkpoint stored via session_service.append_event)
    """
    session = ctx.invocation_context.session

    # Infrastructure-driven checkpoint (checkpoint_after + checkpoint_nodes filter)
    if self.checkpoint_after and self._should_checkpoint_node(ctx.node.name):
      checkpoint_id = f"{session.id}-{ctx.node.name}-{ctx.iteration}-after"
      await self.service.create_checkpoint(
          session=session,
          checkpoint_id=checkpoint_id,
          description=f"After node {ctx.node.name} (iteration {ctx.iteration})",
          agent_name=ctx.node.name,
      )

    # Agent-proposed checkpoint: LLM sets a bool flag in its output schema
    if self._req_state_key and ctx.node.name == self._req_state_key:
      raw = ctx.state.data.get(self._req_state_key, {})
      if isinstance(raw, str):
        try:
          raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
          raw = {}
      if isinstance(raw, dict) and raw.get(self._req_field, False):
        checkpoint_id = (
            f"{session.id}-{ctx.node.name}-{ctx.iteration}-requested"
        )
        await self.service.create_checkpoint(
            session=session,
            checkpoint_id=checkpoint_id,
            description=(
                f"Agent-requested checkpoint at {ctx.node.name}"
                f" (iteration {ctx.iteration})"
            ),
            agent_name=ctx.node.name,
        )

    return None
