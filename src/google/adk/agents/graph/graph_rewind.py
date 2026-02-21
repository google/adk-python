"""Graph rewind functionality.

Standalone function for rewinding graph execution to a specific node.
Integrates with ADK's Runner.rewind_async for temporal navigation.
"""

from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from .graph_agent import GraphAgent


async def rewind_to_node(
    graph: GraphAgent,
    session_service: Any,
    app_name: str,
    user_id: str,
    session_id: str,
    node_name: str,
    invocation_index: int = -1,
) -> None:
  """Rewind graph execution to before a specific node execution.

  Enables temporal navigation within graph workflows:
  - Retry failed nodes with different inputs
  - Explore alternative execution paths
  - Debug workflow issues
  - Select specific iteration in loops

  Args:
      graph: GraphAgent instance
      session_service: Session service instance
      app_name: Application name
      user_id: User ID
      session_id: Session ID
      node_name: Node to rewind to
      invocation_index: Which invocation (-1 for most recent)

  Raises:
      ValueError: If node has not been executed yet
      ValueError: If invocation_index is out of range
  """
  session = await session_service.get_session(
      app_name=app_name, user_id=user_id, session_id=session_id
  )
  if not session:
    raise ValueError(f"Session not found: {session_id}")

  # Extract node_invocations from latest agent_state event
  all_node_invocations: dict[str, list[str]] = {}
  for event in reversed(session.events):
    if (
        event.actions
        and event.actions.agent_state
        and "node_invocations" in (event.actions.agent_state or {})
    ):
      all_node_invocations = event.actions.agent_state["node_invocations"]
      break

  node_invocations = all_node_invocations.get(node_name, [])
  if not node_invocations:
    raise ValueError(
        f"Node '{node_name}' has not been executed yet."
        f" Available nodes: {list(all_node_invocations.keys())}"
    )

  if invocation_index < -len(node_invocations) or (
      invocation_index >= len(node_invocations)
  ):
    raise ValueError(
        f"Invocation index {invocation_index} out of range. "
        f"Node '{node_name}' has"
        f" {len(node_invocations)} invocations."
    )

  invocation_id = node_invocations[invocation_index]

  from ...runners import Runner

  runner = Runner(
      app_name=app_name,
      agent=graph,
      session_service=session_service,
  )
  await runner.rewind_async(
      user_id=user_id,
      session_id=session_id,
      rewind_before_invocation_id=invocation_id,
  )
