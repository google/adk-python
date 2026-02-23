"""Human-in-the-loop interrupt modes and configuration for graph execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional


class InterruptMode(str, Enum):
  """When to interrupt execution for human-in-the-loop.

  Interrupts allow pausing graph execution at specific nodes
  to enable human review, approval, or intervention before continuing.

  Example:
      ```python
      from google.adk.agents.graph import GraphAgent, InterruptMode, InterruptConfig

      graph = GraphAgent(
          name="workflow",
          interrupt_config=InterruptConfig(
              mode=InterruptMode.AFTER,  # Check after node execution
              nodes=["critical_node"],   # Only these nodes
          )
      )
      ```
  """

  BEFORE = "before"  # Interrupt before node execution
  AFTER = "after"  # Interrupt after node execution
  BOTH = "both"  # Interrupt both before and after node execution


@dataclass
class InterruptConfig:
  """Configuration for interrupt behavior in GraphAgent.

  Attributes:
      mode: When to check for interrupts (BEFORE, AFTER, or BOTH)
      nodes: Specific node names to interrupt, None = all nodes
      reasoner: Optional LLM-based interrupt reasoner for intelligent decisions
  """

  mode: InterruptMode = InterruptMode.AFTER
  nodes: Optional[List[str]] = None
  reasoner: Optional[Any] = None  # InterruptReasoner (avoiding circular import)


@dataclass
class InterruptAction:
  """Action to take after processing an interrupt message.

  Returned by interrupt reasoner to indicate what action to take.

  Attributes:
      action: Action type ("continue", "rerun", "go_back", "pause", "defer")
      reasoning: LLM's reasoning for this decision (optional)
      parameters: Additional parameters for the action (e.g., target_node, guidance)
  """

  action: str
  reasoning: str = ""
  parameters: Dict[str, Any] = None  # type: ignore[assignment]

  def __post_init__(self) -> None:
    """Initialize parameters dict if None."""
    if self.parameters is None:
      self.parameters = {}
