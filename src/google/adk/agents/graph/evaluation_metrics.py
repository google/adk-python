"""Custom evaluation metrics for GraphAgent workflows.

These metrics enable evaluating graph execution paths, state transitions,
and workflow behavior in ADK's evaluation framework.

Example usage:
    ```python
    from google.adk.evaluation import EvalMetric
    from google.adk.agents.graph.evaluation_metrics import (
        graph_path_match,
        state_contains_keys,
        node_execution_count,
    )

    # In eval config:
    metrics = [
        EvalMetric(
            name="graph_path",
            custom_function_path="google.adk.agents.graph.evaluation_metrics.graph_path_match",
        ),
    ]
    ```
"""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from ...evaluation.eval_case import ConversationScenario
from ...evaluation.eval_case import Invocation
from ...evaluation.eval_metrics import EvalMetric
from ...evaluation.eval_metrics import EvalStatus
from ...evaluation.evaluator import EvaluationResult
from ...evaluation.evaluator import PerInvocationResult


def graph_path_match(
    eval_metric: EvalMetric,
    actual_invocations: List[Invocation],
    expected_invocations: Optional[List[Invocation]],
    conversation_scenario: Optional[ConversationScenario] = None,
) -> EvaluationResult:
  """Evaluate if graph execution path matches expected path.

  Checks if the sequence of nodes executed matches the expected path.
  Looks for 'graph_path' in session state.

  Args:
      eval_metric: Metric configuration
      actual_invocations: Actual agent invocations
      expected_invocations: Expected invocations (unused, path comes from scenario)
      conversation_scenario: Test scenario with expected path in metadata

  Returns:
      EvaluationResult with scores based on path matching

  Expected format in conversation_scenario.metadata:
      {
          "expected_graph_path": ["node1", "node2", "node3"]
      }
  """
  results = []
  overall_score = 0.0
  overall_status = EvalStatus.PASSED

  # Get expected path from eval_metric custom fields (for testing)
  # In production, would come from scenario or expected_invocations
  expected_path = getattr(eval_metric, "expected_graph_path", None)

  for actual_inv in actual_invocations:
    # Extract actual path from intermediate_data (production)
    # or from eval_metric custom fields (for testing)
    actual_path = getattr(eval_metric, "actual_graph_path", None)

    if actual_path is None and actual_inv.intermediate_data:
      # Extract from InvocationEvents
      from ...evaluation.eval_case import InvocationEvents

      if isinstance(actual_inv.intermediate_data, InvocationEvents):
        # Parse graph metadata from intermediate events
        # Get the LAST/latest metadata event (final graph state)
        for event in reversed(actual_inv.intermediate_data.invocation_events):
          if event.content and event.content.parts:
            for part in event.content.parts:
              if part.text and "[GraphMetadata]" in part.text:
                # Extract metadata from text
                import ast

                try:
                  # Parse the dict from text like "[GraphMetadata] {'graph_path': [...]}"
                  metadata_str = part.text.split("[GraphMetadata]", 1)[
                      1
                  ].strip()
                  metadata = ast.literal_eval(metadata_str)
                  actual_path = metadata.get("graph_path")
                  if actual_path:
                    break
                except Exception:
                  continue
          if actual_path:
            break

    # Compute score
    score = 0.0
    status = EvalStatus.NOT_EVALUATED

    if expected_path is None:
      # No expected path specified
      status = EvalStatus.NOT_EVALUATED
    elif actual_path is None:
      # No actual path found
      status = EvalStatus.FAILED
      score = 0.0
    elif actual_path == expected_path:
      # Exact match
      status = EvalStatus.PASSED
      score = 1.0
    else:
      # Partial match - score based on how many nodes match
      matched = sum(1 for a, e in zip(actual_path, expected_path) if a == e)
      max_len = max(len(actual_path), len(expected_path))
      score = matched / max_len if max_len > 0 else 0.0
      status = EvalStatus.FAILED if score < 0.5 else EvalStatus.PASSED

    results.append(
        PerInvocationResult(
            actual_invocation=actual_inv,
            score=score,
            eval_status=status,
        )
    )

    if status == EvalStatus.FAILED:
      overall_status = EvalStatus.FAILED

    overall_score += score

  # Average score across invocations
  if results:
    overall_score /= len(results)

  # If all results are NOT_EVALUATED, set overall status to NOT_EVALUATED
  if results and all(
      r.eval_status == EvalStatus.NOT_EVALUATED for r in results
  ):
    overall_status = EvalStatus.NOT_EVALUATED

  return EvaluationResult(
      overall_score=overall_score,
      overall_eval_status=overall_status,
      per_invocation_results=results,
  )


def state_contains_keys(
    eval_metric: EvalMetric,
    actual_invocations: List[Invocation],
    expected_invocations: Optional[List[Invocation]],
    conversation_scenario: Optional[ConversationScenario] = None,
) -> EvaluationResult:
  """Evaluate if final state contains expected keys.

  Checks if session state contains all expected keys with correct values.

  Args:
      eval_metric: Metric configuration
      actual_invocations: Actual agent invocations
      expected_invocations: Expected invocations (unused)
      conversation_scenario: Test scenario with expected state in metadata

  Returns:
      EvaluationResult with scores based on state matching

  Expected format in conversation_scenario.metadata:
      {
          "expected_state": {"key1": "value1", "key2": 42},
          "actual_state": {"key1": "value1", "key2": 42}  # For testing
      }
  """
  results = []
  overall_score = 0.0
  overall_status = EvalStatus.PASSED

  # Get expected state from eval_metric custom fields (for testing)
  expected_state = getattr(eval_metric, "expected_state", None)

  for actual_inv in actual_invocations:
    # Extract actual state from eval_metric custom fields (for testing)
    # or from intermediate_data (production)
    actual_state = getattr(eval_metric, "actual_state", None)

    if actual_state is None and actual_inv.intermediate_data:
      # Extract from InvocationEvents
      from ...evaluation.eval_case import InvocationEvents

      if isinstance(actual_inv.intermediate_data, InvocationEvents):
        # Parse graph_state from metadata events
        # Get the LAST/latest metadata event (final state)
        for event in reversed(actual_inv.intermediate_data.invocation_events):
          if event.content and event.content.parts:
            for part in event.content.parts:
              if part.text and "[GraphMetadata]" in part.text:
                import ast

                try:
                  metadata_str = part.text.split("[GraphMetadata]", 1)[
                      1
                  ].strip()
                  metadata = ast.literal_eval(metadata_str)
                  actual_state = metadata.get("graph_state")
                  if actual_state:
                    break
                except Exception:
                  continue
          if actual_state:
            break

    # Compute score
    score = 0.0
    status = EvalStatus.NOT_EVALUATED

    if expected_state is None:
      status = EvalStatus.NOT_EVALUATED
    elif actual_state is None:
      status = EvalStatus.FAILED
      score = 0.0
    else:
      # Check each expected key
      total_keys = len(expected_state)
      matched_keys = 0

      for key, expected_value in expected_state.items():
        if key in actual_state and actual_state[key] == expected_value:
          matched_keys += 1

      score = matched_keys / total_keys if total_keys > 0 else 0.0
      status = EvalStatus.PASSED if score >= 1.0 else EvalStatus.FAILED

    results.append(
        PerInvocationResult(
            actual_invocation=actual_inv,
            score=score,
            eval_status=status,
        )
    )

    if status == EvalStatus.FAILED:
      overall_status = EvalStatus.FAILED

    overall_score += score

  # Average score across invocations
  if results:
    overall_score /= len(results)

  # If all results are NOT_EVALUATED, set overall status to NOT_EVALUATED
  if results and all(
      r.eval_status == EvalStatus.NOT_EVALUATED for r in results
  ):
    overall_status = EvalStatus.NOT_EVALUATED

  return EvaluationResult(
      overall_score=overall_score,
      overall_eval_status=overall_status,
      per_invocation_results=results,
  )


def node_execution_count(
    eval_metric: EvalMetric,
    actual_invocations: List[Invocation],
    expected_invocations: Optional[List[Invocation]],
    conversation_scenario: Optional[ConversationScenario] = None,
) -> EvaluationResult:
  """Evaluate if nodes executed expected number of times.

  Checks node_invocations tracking in session state.

  Args:
      eval_metric: Metric configuration
      actual_invocations: Actual agent invocations
      expected_invocations: Expected invocations (unused)
      conversation_scenario: Test scenario with expected counts in metadata

  Returns:
      EvaluationResult with scores based on execution counts

  Expected format in conversation_scenario.metadata:
      {
          "expected_node_counts": {"node1": 1, "node2": 3},
          "actual_node_counts": {"node1": 1, "node2": 3}  # For testing
      }
  """
  results = []
  overall_score = 0.0
  overall_status = EvalStatus.PASSED

  # Get expected counts from eval_metric custom fields (for testing)
  expected_counts = getattr(eval_metric, "expected_node_counts", None)

  for actual_inv in actual_invocations:
    # Extract actual counts from eval_metric custom fields (for testing)
    # In production, would come from intermediate_data
    actual_counts = getattr(eval_metric, "actual_node_counts", {})

    if not actual_counts and actual_inv.intermediate_data:
      # Extract from InvocationEvents
      from ...evaluation.eval_case import InvocationEvents

      if isinstance(actual_inv.intermediate_data, InvocationEvents):
        # Parse node_invocations from graph metadata events
        for event in actual_inv.intermediate_data.invocation_events:
          if event.content and event.content.parts:
            for part in event.content.parts:
              if part.text and "[GraphMetadata]" in part.text:
                import ast

                try:
                  metadata_str = part.text.split("[GraphMetadata]", 1)[
                      1
                  ].strip()
                  metadata = ast.literal_eval(metadata_str)
                  node_invocs = metadata.get("node_invocations", {})
                  if node_invocs:
                    actual_counts = node_invocs
                    # Continue to get the latest counts
                except Exception:
                  continue

    # Compute score
    score = 0.0
    status = EvalStatus.NOT_EVALUATED

    if expected_counts is None:
      status = EvalStatus.NOT_EVALUATED
    elif not actual_counts:
      status = EvalStatus.FAILED
      score = 0.0
    else:
      # Check each expected node count
      total_nodes = len(expected_counts)
      matched_nodes = 0

      for node_name, expected_count in expected_counts.items():
        actual_count = actual_counts.get(node_name, 0)
        if actual_count == expected_count:
          matched_nodes += 1

      score = matched_nodes / total_nodes if total_nodes > 0 else 0.0
      status = EvalStatus.PASSED if score >= 1.0 else EvalStatus.FAILED

    results.append(
        PerInvocationResult(
            actual_invocation=actual_inv,
            score=score,
            eval_status=status,
        )
    )

    if status == EvalStatus.FAILED:
      overall_status = EvalStatus.FAILED

    overall_score += score

  # Average score across invocations
  if results:
    overall_score /= len(results)

  # If all results are NOT_EVALUATED, set overall status to NOT_EVALUATED
  if results and all(
      r.eval_status == EvalStatus.NOT_EVALUATED for r in results
  ):
    overall_status = EvalStatus.NOT_EVALUATED

  return EvaluationResult(
      overall_score=overall_score,
      overall_eval_status=overall_status,
      per_invocation_results=results,
  )
