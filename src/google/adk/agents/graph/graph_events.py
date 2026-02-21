"""Typed event streams for GraphAgent execution."""

from __future__ import annotations

from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import Field

from .graph_state import GraphState


class GraphEventType(str, Enum):
  """Types of graph execution events."""

  NODE_START = "node_start"  # Node execution starting
  NODE_END = "node_end"  # Node execution completed
  EDGE_TRAVERSAL = "edge_traversal"  # Edge being traversed
  CHECKPOINT = "checkpoint"  # Checkpoint created
  INTERRUPT = "interrupt"  # Human-in-the-loop interrupt
  ERROR = "error"  # Execution error
  COMPLETE = "complete"  # Graph execution complete


class GraphEvent(BaseModel):  # type: ignore[misc]
  """Typed event for graph execution streaming.

  GraphEvent provides structured events for monitoring and debugging
  graph execution. These events can be streamed to track execution progress,
  state changes, and control flow.

  Example:
      ```python
      async for event in graph.run_async_with_events(ctx):
          if event.event_type == GraphEventType.NODE_START:
              print(f"Starting node: {event.node_name}")
          elif event.event_type == GraphEventType.CHECKPOINT:
              print(f"Checkpoint at iteration {event.iteration}")
      ```
  """

  event_type: GraphEventType
  timestamp: str = Field(description="ISO timestamp of event")

  # Node information
  node_name: Optional[str] = None
  iteration: Optional[int] = None

  # State information
  graph_state: Optional[Dict[str, Any]] = None
  state_delta: Optional[Dict[str, Any]] = None

  # Edge information
  source_node: Optional[str] = None
  target_node: Optional[str] = None
  edge_condition_result: Optional[bool] = None

  # Interrupt information
  interrupt_mode: Optional[str] = None
  interrupt_message: Optional[str] = None

  # Error information
  error_message: Optional[str] = None
  error_type: Optional[str] = None

  # Checkpoint information
  checkpoint_id: Optional[str] = None

  # Additional metadata
  metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphStreamMode(str, Enum):
  """Stream modes for graph execution.

  Different stream modes provide different levels of detail:
  - VALUES: Only final node outputs
  - UPDATES: State updates after each node
  - MESSAGES: All agent messages and events
  - DEBUG: All events including edges, checkpoints, interrupts
  """

  VALUES = "values"  # Stream final values only
  UPDATES = "updates"  # Stream state updates
  MESSAGES = "messages"  # Stream all messages
  DEBUG = "debug"  # Stream all debug events
