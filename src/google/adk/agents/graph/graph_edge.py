"""Conditional edges for graph routing."""

from __future__ import annotations

from typing import Callable
from typing import Optional

from .graph_state import GraphState


class EdgeCondition:
  """Conditional edge that routes based on state.

  Edges connect nodes in the graph and can have optional conditions,
  priorities, and weights for advanced routing strategies.

  Example:
      ```python
      # Unconditional edge (always taken)
      edge = EdgeCondition(target_node="next_node")

      # Conditional edge (taken if score > 0.8)
      edge = EdgeCondition(
          target_node="high_score_handler",
          condition=lambda state: state.data.get("score", 0) > 0.8
      )

      # Priority-based routing (higher priority evaluated first)
      edge1 = EdgeCondition(
          target_node="critical_path",
          condition=lambda state: state.data.get("is_critical", False),
          priority=10  # High priority
      )
      edge2 = EdgeCondition(
          target_node="normal_path",
          priority=5  # Lower priority
      )

      # Weighted random selection (among matching edges)
      edge1 = EdgeCondition(target_node="path_a", weight=0.7)  # 70% chance
      edge2 = EdgeCondition(target_node="path_b", weight=0.3)  # 30% chance

      # Fallback edge (priority=0 always matches if no other edge matched)
      edge_fallback = EdgeCondition(
          target_node="default_handler",
          priority=0  # Fallback priority
      )
      ```
  """

  def __init__(
      self,
      target_node: str,
      condition: Optional[Callable[[GraphState], bool]] = None,
      priority: int = 1,
      weight: float = 1.0,
  ):
    """Initialize edge condition.

    Args:
        target_node: Name of the target node
        condition: Function that returns True if this edge should be taken.
            If None, edge is always taken (unconditional).
        priority: Priority for edge evaluation (higher = evaluated first).
            Priority 0 is special: treated as fallback (always matches if reached).
            Default is 1 (normal priority).
        weight: Weight for weighted random selection among matching edges.
            Only used when multiple edges match. Higher weight = higher probability.
            Default is 1.0.
    """
    self.target_node = target_node
    self.has_condition = condition is not None
    self.condition = condition or (lambda _: True)
    self.priority = priority
    self.weight = weight

  def should_route(self, state: GraphState) -> bool:
    """Check if this edge should be taken given the current state.

    Args:
        state: Current graph state

    Returns:
        True if edge condition is satisfied, False otherwise
    """
    # Priority 0 is fallback - always matches
    if self.priority == 0:
      return True
    return self.condition(state)

  def __repr__(self) -> str:
    """String representation for debugging."""
    return (
        f"EdgeCondition(target={self.target_node}, "
        f"priority={self.priority}, weight={self.weight}, "
        f"has_condition={self.has_condition})"
    )
