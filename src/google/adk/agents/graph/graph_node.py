"""Graph node wrapper for agents and functions."""

from __future__ import annotations

from copy import deepcopy
import logging
import random
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import TYPE_CHECKING

from ..base_agent import BaseAgent
from .graph_edge import EdgeCondition
from .graph_state import GraphState
from .graph_state import StateReducer

if TYPE_CHECKING:
  from ..llm_agent import LlmAgent

logger = logging.getLogger(__name__)


class GraphNode:
  """A node in the graph that can wrap ANY ADK agent or custom function.

  Supports all BaseAgent types:
  - LLMAgent: Single agent with tools
  - SequentialAgent: Chain of agents executed in sequence
  - ParallelAgent: Multiple agents executed concurrently
  - LoopAgent: Iterative agent execution
  - GraphAgent: Recursive graph workflows (graphs within graphs!)
  - Custom agents: Any subclass of BaseAgent

  This enables powerful composition patterns like:
  - Nested workflows (GraphAgent containing other GraphAgents)
  - Validation pipelines (SequentialAgent as a node)
  - Parallel analysis (ParallelAgent as a node)
  """

  def __init__(
      self,
      name: str,
      agent: Optional[BaseAgent] = None,
      function: Optional[Callable[..., Any]] = None,
      input_mapper: Optional[Callable[[GraphState], str]] = None,
      output_mapper: Optional[Callable[[str, GraphState], GraphState]] = None,
      reducer: StateReducer = StateReducer.OVERWRITE,
      custom_reducer: Optional[Callable[..., Any]] = None,
  ):
    """Initialize graph node.

    Args:
        name: Node name
        agent: ANY ADK BaseAgent subclass (LLMAgent, SequentialAgent, ParallelAgent,
               LoopAgent, GraphAgent, or custom agents)
        function: Custom async function to execute (alternative to agent)
        input_mapper: Maps GraphState to agent input
        output_mapper: Maps agent output back to GraphState
        reducer: Strategy for merging output into state
        custom_reducer: Custom reduction function
    """
    if agent is None and function is None:
      raise ValueError("Either agent or function must be provided")

    self.name = name
    self.agent = agent
    self.function = function
    self.input_mapper = input_mapper or self._default_input_mapper
    self.output_mapper = output_mapper or self._default_output_mapper
    self.reducer = reducer
    self.custom_reducer = custom_reducer
    self.edges: List[EdgeCondition] = []
    self._sorted_edges_cache: Optional[List[tuple[int, EdgeCondition]]] = None

    # Auto-default output_key for LlmAgent with output_schema
    from ..llm_agent import LlmAgent

    if isinstance(agent, LlmAgent):
      if agent.output_schema and not agent.output_key:
        logger.info(
            f"Node '{name}': Auto-defaulting output_key to agent name"
            f" '{agent.name}' (LlmAgent has output_schema but no explicit"
            " output_key)"
        )
        # Use model_copy to avoid mutating the caller's agent instance
        self.agent = agent.model_copy(update={"output_key": agent.name})

  def add_edge(
      self, target_node: str, condition: Optional[Callable[..., Any]] = None
  ) -> None:
    """Add an edge to another node.

    Args:
        target_node: Name of the target node
        condition: Optional condition function for conditional routing
    """
    self.edges.append(EdgeCondition(target_node, condition))
    self._sorted_edges_cache = None  # Invalidate cache

  def _default_input_mapper(self, state: GraphState) -> str:
    """Default input mapper: extract 'input' or 'messages' from state.

    Args:
        state: Current graph state

    Returns:
        Input string for the node
    """
    return str(state.data.get("input", state.data.get("messages", "")))

  def _default_output_mapper(
      self, output: str, state: GraphState
  ) -> GraphState:
    """Default output mapper: store output in state with node name as key.

    Applies the configured reducer strategy to merge the output into state.

    Args:
        output: Node execution output
        state: Current graph state

    Returns:
        New graph state with output merged
    """
    new_state = GraphState(data=deepcopy(state.data))

    if self.reducer == StateReducer.OVERWRITE:
      new_state.data[self.name] = output
    elif self.reducer == StateReducer.APPEND:
      if self.name not in new_state.data:
        new_state.data[self.name] = []
      new_state.data[self.name].append(output)
    elif self.reducer == StateReducer.SUM:
      existing = new_state.data.get(self.name)
      if existing is None:
        # Infer zero-value from output type: "" for str, 0 for int/float, [] for list
        existing = type(output)()
      try:
        new_state.data[self.name] = existing + output
      except TypeError:
        raise TypeError(
            f"StateReducer.SUM: cannot add {type(existing).__name__} + "
            f"{type(output).__name__} for node '{self.name}'. "
            "Ensure consistent types or use a custom output_mapper."
        )
    elif self.reducer == StateReducer.CUSTOM and self.custom_reducer:
      new_state.data[self.name] = self.custom_reducer(
          new_state.data.get(self.name), output
      )

    return new_state

  def get_next_node(self, state: GraphState) -> Optional[str]:
    """Determine next node based on conditional edges with priority and weight.

    Enhanced routing logic:
    1. Sort edges by priority (highest first), preserving insertion order
    2. Evaluate conditions in priority order
    3. Within same priority, return first match (insertion order)
    4. If edges have different weights, use weighted random selection
    5. Priority 0 edges are fallbacks (always match if no higher priority matched)

    Args:
        state: Current graph state

    Returns:
        Name of next node, or None if no edge matches
    """
    if not self.edges:
      return None

    # Use cached sorted edges (sort once, invalidated by add_edge)
    if self._sorted_edges_cache is None:
      indexed_edges = [(i, e) for i, e in enumerate(self.edges)]
      self._sorted_edges_cache = sorted(
          indexed_edges, key=lambda x: (-x[1].priority, x[0])
      )
    sorted_edges = self._sorted_edges_cache

    # Group edges by priority
    current_priority = sorted_edges[0][1].priority
    matching_edges: list[tuple[int, EdgeCondition]] = []

    for idx, edge in sorted_edges:
      # If we've moved to a lower priority and already have matches, stop
      if edge.priority < current_priority and matching_edges:
        break

      # Update current priority
      current_priority = edge.priority

      # Check if edge matches
      if edge.should_route(state):
        matching_edges.append((idx, edge))

    # No matching edges
    if not matching_edges:
      return None

    # Single matching edge - return it
    if len(matching_edges) == 1:
      return matching_edges[0][1].target_node

    # Multiple matching edges at same priority
    # Check if weights are all the same (default behavior: first match)
    weights = [e.weight for _, e in matching_edges]
    all_same_weight = len(set(weights)) == 1

    if all_same_weight:
      # All weights equal - return first match in insertion order
      return matching_edges[0][1].target_node

    # Different weights - use weighted random selection
    total_weight = sum(weights)
    if total_weight == 0:
      # All weights are 0 — guard against ZeroDivisionError in weighted
      # random selection below. Fall back to first match in insertion order.
      return matching_edges[0][1].target_node

    # Weighted random choice
    rand_value = random.random() * total_weight
    cumulative = 0.0
    for idx, edge in matching_edges:
      cumulative += edge.weight
      if rand_value <= cumulative:
        return edge.target_node

    # Fallback (shouldn't reach here, but safety)
    return matching_edges[-1][1].target_node
