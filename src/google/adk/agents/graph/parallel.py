"""Parallel execution support for GraphAgent.

Enables concurrent execution of independent nodes following ParallelAgent patterns.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from enum import Enum
import logging
from typing import Any
from typing import AsyncGenerator
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

from ...events.event import Event
from ...telemetry.tracing import tracer
from ...utils.feature_decorator import experimental
from .graph_state import GraphState

logger = logging.getLogger("google_adk." + __name__)


class JoinStrategy(str, Enum):
  """Strategy for joining parallel executions.

  - WAIT_ALL: Wait for all nodes to complete
  - WAIT_ANY: Continue when first node completes
  - WAIT_N: Wait for N nodes to complete
  """

  WAIT_ALL = "all"
  WAIT_ANY = "any"
  WAIT_N = "n"


class ErrorPolicy(str, Enum):
  """Policy for handling errors in parallel execution.

  - FAIL_FAST: Cancel all on first error
  - CONTINUE: Continue others on error
  - COLLECT: Collect all errors
  """

  FAIL_FAST = "fail_fast"
  CONTINUE = "continue"
  COLLECT = "collect"


@experimental
class ParallelNodeGroup:
  """Defines nodes that execute concurrently.

  Example:
      >>> group = ParallelNodeGroup(
      ...     nodes=["fetch_user", "fetch_products"],
      ...     join_strategy=JoinStrategy.WAIT_ALL,
      ...     error_policy=ErrorPolicy.FAIL_FAST
      ... )
  """

  def __init__(
      self,
      nodes: List[str],
      join_strategy: JoinStrategy = JoinStrategy.WAIT_ALL,
      error_policy: ErrorPolicy = ErrorPolicy.FAIL_FAST,
      wait_n: int = 1,
  ):
    """Initialize parallel node group.

    Args:
        nodes: List of node names to execute in parallel
        join_strategy: Strategy for joining parallel executions
        error_policy: Policy for handling errors
        wait_n: Number of nodes to wait for (when join_strategy is WAIT_N)
    """
    self.nodes = nodes
    self.join_strategy = join_strategy
    self.error_policy = error_policy
    self.wait_n = wait_n

    if join_strategy == JoinStrategy.WAIT_N and wait_n > len(nodes):
      raise ValueError(
          f"wait_n ({wait_n}) cannot be greater than number of nodes"
          f" ({len(nodes)})"
      )


async def _collect_events(
    generator: AsyncGenerator[Event, None],
) -> Dict[str, Any]:
  """Collect all events from a generator.

  Args:
      generator: Event generator

  Returns:
      Dict with events and any error
  """
  events = []
  error = None

  try:
    async for event in generator:
      events.append(event)
  except asyncio.CancelledError:
    # Task was cancelled - not an error, just return collected events
    pass
  except Exception as e:
    error = e

  return {"events": events, "error": error}


async def execute_parallel_group(
    group: ParallelNodeGroup,
    nodes: Dict[str, Any],  # GraphNode instances
    state: GraphState,
    ctx: Any,  # InvocationContext
    execute_node_fn: Callable[..., AsyncGenerator[Event, None]],
) -> AsyncGenerator[Event, None]:
  """Execute parallel nodes following ParallelAgent pattern.

  Uses asyncio.wait with FIRST_COMPLETED to stream events as they arrive.

  Args:
      group: Parallel node group configuration
      nodes: Dict of node name to GraphNode instance
      state: Current graph state
      ctx: Invocation context
      execute_node_fn: Function to execute a single node

  Yields:
      Events from parallel node executions

  Raises:
      Exception: If error_policy is FAIL_FAST and a node fails
  """
  with tracer.start_as_current_span("parallel_group_execution") as span:
    span.set_attribute("parallel.node_count", len(group.nodes))
    span.set_attribute("parallel.join_strategy", group.join_strategy.value)
    span.set_attribute("parallel.error_policy", group.error_policy.value)
    span.set_attribute("parallel.nodes", ",".join(group.nodes))

    logger.info(
        f"Executing parallel group with {len(group.nodes)} nodes: {group.nodes}"
    )

    # Create isolated state copies for each branch
    branch_states = {}
    node_generators = {}

    for node_name in group.nodes:
      if node_name not in nodes:
        raise ValueError(f"Node '{node_name}' not found in graph")

      # Create isolated branch context with deep copy for proper isolation
      branch_state = GraphState(data=deepcopy(state.data))
      branch_states[node_name] = branch_state

      # Create generator for each node
      node = nodes[node_name]
      node_generators[node_name] = execute_node_fn(node, branch_state, ctx)

    # Start all executions (ParallelAgent pattern)
    tasks = {
        node_name: asyncio.create_task(_collect_events(gen))
        for node_name, gen in node_generators.items()
    }

    # Create inverse mapping for O(1) task lookup (fixes P0.1)
    task_to_node: Dict[asyncio.Task[Dict[str, Any]], str] = {
        task: node_name for node_name, task in tasks.items()
    }

    pending = set(tasks.values())
    results = {}
    errors = []

    # Wait for completions based on join strategy
    num_to_wait = {
        JoinStrategy.WAIT_ALL: len(group.nodes),
        JoinStrategy.WAIT_ANY: 1,
        JoinStrategy.WAIT_N: group.wait_n,
    }[group.join_strategy]

    completed_count = 0

    while pending and completed_count < num_to_wait:
      # Wait for next completion (FIRST_COMPLETED pattern)
      done, pending = await asyncio.wait(
          pending, return_when=asyncio.FIRST_COMPLETED
      )

      for task in done:
        # O(1) task lookup using inverse mapping (fixes P0.1)
        task_node_name = task_to_node.get(task)

        if task_node_name is None:
          # Task not in mapping - critical error, should never happen
          logger.error(
              f"Task identity tracking failure: task {task} not found in"
              " mapping. This indicates a critical bug."
          )
          span.set_attribute("parallel.task_lookup_failure", True)
          raise RuntimeError(
              f"Task {task} not found in task_to_node mapping. This should"
              " never happen and indicates a critical bug in parallel"
              " execution."
          )

        logger.debug(f"Task for node '{task_node_name}' completed")
        span.add_event(
            "task_completed",
            {
                "node_name": task_node_name,
                "completed_count": completed_count + 1,
            },
        )

        result = task.result()
        results[task_node_name] = {
            "state": branch_states[task_node_name],
            "events": result["events"],
            "error": result["error"],
        }

        # Handle errors based on policy
        if result["error"]:
          errors.append((task_node_name, result["error"]))
          span.add_event(
              "task_error",
              {
                  "node_name": task_node_name,
                  "error": str(result["error"]),
                  "error_policy": group.error_policy.value,
              },
          )

          if group.error_policy == ErrorPolicy.FAIL_FAST:
            # Cancel all pending tasks
            for p in pending:
              p.cancel()

            raise result["error"]

        # Yield events from completed node
        for event in result["events"]:
          yield event

        completed_count += 1

    # Cancel remaining tasks if we satisfied join strategy
    if pending:
      span.add_event(
          "cancelling_pending_tasks",
          {"pending_count": len(pending)},
      )
      for task in pending:
        task.cancel()

      # Wait for cancellations
      await asyncio.gather(*pending, return_exceptions=True)

    # Handle collected errors
    if errors and group.error_policy == ErrorPolicy.COLLECT:
      error_msg = f"Errors in parallel execution: {errors}"
      span.set_attribute("parallel.collected_errors", len(errors))
      raise Exception(error_msg)

    # Merge branch states back into main state with conflict detection.
    # Only merge keys that actually changed from the pre-branch snapshot
    # to avoid stale copies overwriting other branches' modifications.
    conflicts_detected = []
    keys_merged: set[str] = set()
    original_data = deepcopy(state.data)

    for node_name in group.nodes:
      if node_name not in results:
        continue
      branch_data = results[node_name]["state"].data

      for key, value in branch_data.items():
        # Skip keys unchanged from original — prevents stale overwrites
        if key in original_data and value == original_data[key]:
          continue

        # This key was added or changed by the branch
        if key in keys_merged and state.data.get(key) != value:
          conflicts_detected.append({
              "key": key,
              "node": node_name,
              "existing_value": state.data[key],
              "new_value": value,
          })
          logger.warning(
              "State merge conflict: key '%s' written by multiple parallel"
              " branches. Last write from node '%s' wins.",
              key,
              node_name,
          )

        state.data[key] = value
        keys_merged.add(key)

    span.set_attribute("parallel.completed_count", completed_count)
    span.set_attribute("parallel.branches_merged", len(results))
    span.set_attribute("parallel.conflicts_detected", len(conflicts_detected))

    if conflicts_detected:
      span.add_event(
          "state_merge_conflicts",
          {
              "conflict_count": len(conflicts_detected),
              "conflicting_keys": [c["key"] for c in conflicts_detected],
          },
      )

    logger.info(
        f"Parallel group completed. {completed_count}/{len(group.nodes)} nodes"
        f" finished. Merged state from {len(results)} branches."
    )
