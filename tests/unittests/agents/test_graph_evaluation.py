"""Tests for GraphAgent evaluation metrics."""

from types import SimpleNamespace

from google.adk.agents.graph.evaluation_metrics import graph_path_match
from google.adk.agents.graph.evaluation_metrics import node_execution_count
from google.adk.agents.graph.evaluation_metrics import state_contains_keys
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalStatus
from google.genai import types
import pytest


@pytest.mark.asyncio
async def test_graph_path_match_exact():
  """Test graph_path_match metric with exact path match."""
  invocation = Invocation(
      userContent=types.Content(parts=[types.Part(text="test")]),
      finalResponse=types.Content(parts=[types.Part(text="response")]),
  )

  # Create metric with custom attributes (SimpleNamespace for testing)
  # NOTE: In production, actual_graph_path would come from intermediate_data
  metric = SimpleNamespace(
      metric_name="graph_path",
      expected_graph_path=["n1", "n2", "n3"],
      actual_graph_path=["n1", "n2", "n3"],  # Exact match
  )

  # Evaluate
  result = graph_path_match(metric, [invocation], None, None)

  # Should pass with perfect score
  assert result.overall_score == 1.0
  assert result.overall_eval_status == EvalStatus.PASSED
  assert len(result.per_invocation_results) == 1
  assert result.per_invocation_results[0].score == 1.0


@pytest.mark.asyncio
async def test_graph_path_match_partial():
  """Test graph_path_match with partial path match."""
  invocation = Invocation(
      userContent=types.Content(parts=[types.Part(text="test")]),
      finalResponse=types.Content(parts=[types.Part(text="response")]),
  )

  metric = SimpleNamespace(
      metric_name="graph_path",
      expected_graph_path=["n1", "n3", "n4"],
      actual_graph_path=["n1", "n2"],  # Partial match
  )

  result = graph_path_match(metric, [invocation], None, None)

  # Should have partial score (1 match out of 3 expected)
  assert result.overall_score < 1.0
  assert result.overall_score > 0.0  # At least n1 matches


@pytest.mark.asyncio
async def test_state_contains_keys_exact():
  """Test state_contains_keys metric with exact match."""
  invocation = Invocation(
      userContent=types.Content(parts=[types.Part(text="test")]),
      finalResponse=types.Content(parts=[types.Part(text="done")]),
  )

  metric = SimpleNamespace(
      metric_name="state_check",
      expected_state={"key1": "value1", "key2": 42},
      actual_state={"key1": "value1", "key2": 42},  # Exact match
  )

  result = state_contains_keys(metric, [invocation], None, None)

  # Should pass with perfect score
  assert result.overall_score == 1.0
  assert result.overall_eval_status == EvalStatus.PASSED


@pytest.mark.asyncio
async def test_state_contains_keys_partial():
  """Test state_contains_keys with partial match."""
  invocation = Invocation(
      userContent=types.Content(parts=[types.Part(text="test")]),
      finalResponse=types.Content(parts=[types.Part(text="done")]),
  )

  metric = SimpleNamespace(
      metric_name="state_check",
      expected_state={"key1": "value1", "key2": 42},
      actual_state={"key1": "value1", "key2": 999},  # key2 wrong
  )

  result = state_contains_keys(metric, [invocation], None, None)

  # Should have partial score (1 out of 2 keys match)
  assert result.overall_score == 0.5
  assert result.overall_eval_status == EvalStatus.FAILED


@pytest.mark.asyncio
async def test_node_execution_count_exact():
  """Test node_execution_count with exact counts."""
  invocation = Invocation(
      userContent=types.Content(parts=[types.Part(text="test")]),
      finalResponse=types.Content(parts=[types.Part(text="done")]),
  )

  metric = SimpleNamespace(
      metric_name="execution_count",
      expected_node_counts={"loop_node": 3},
      actual_node_counts={"loop_node": 3},  # Exact match
  )

  result = node_execution_count(metric, [invocation], None, None)

  # Should pass if count matches
  assert result.overall_score == 1.0
  assert result.overall_eval_status == EvalStatus.PASSED


@pytest.mark.asyncio
async def test_metrics_with_no_expected_data():
  """Test metrics skip when no expected data provided."""
  invocation = Invocation(
      userContent=types.Content(parts=[types.Part(text="test")]),
      finalResponse=types.Content(parts=[types.Part(text="done")]),
  )

  metric = SimpleNamespace(metric_name="test")  # No custom fields

  # All metrics should return NOT_EVALUATED when no expected data
  result1 = graph_path_match(metric, [invocation], None, None)
  assert result1.overall_eval_status == EvalStatus.NOT_EVALUATED

  result2 = state_contains_keys(metric, [invocation], None, None)
  assert result2.overall_eval_status == EvalStatus.NOT_EVALUATED

  result3 = node_execution_count(metric, [invocation], None, None)
  assert result3.overall_eval_status == EvalStatus.NOT_EVALUATED


# ---------------------------------------------------------------------------
# InvocationEvents-based paths: exception handlers and None-result branches
# ---------------------------------------------------------------------------


def _make_invocation_with_event(text: str) -> "Invocation":
  """Helper: Invocation whose intermediate_data carries a single text event."""
  from google.adk.evaluation.eval_case import InvocationEvent
  from google.adk.evaluation.eval_case import InvocationEvents

  evt = InvocationEvent(
      author="graph",
      content=types.Content(parts=[types.Part(text=text)]),
  )
  return Invocation(
      userContent=types.Content(parts=[types.Part(text="q")]),
      finalResponse=types.Content(parts=[types.Part(text="a")]),
      intermediateData=InvocationEvents(invocationEvents=[evt]),
  )


def test_graph_path_match_malformed_metadata_exception_handled():
  """Lines 92-93: ast.literal_eval fails on malformed metadata → continue.

  The except block swallows the error and leaves actual_path as None,
  ultimately producing FAILED status (expected path set, no actual path found).
  """
  metric = SimpleNamespace(
      metric_name="path",
      expected_graph_path=["n1", "n2"],
      # No actual_graph_path shortcut → will try InvocationEvents path
  )
  inv = _make_invocation_with_event(
      "[GraphMetadata] {this is: not valid python}"
  )

  result = graph_path_match(metric, [inv], None, None)

  # Parsing failed → actual_path stays None → FAILED
  assert result.overall_eval_status == EvalStatus.FAILED
  assert result.overall_score == 0.0


def test_graph_path_match_actual_path_none_from_events():
  """Lines 106-107: expected_path set but actual_path is None → FAILED.

  Valid [GraphMetadata] event but the dict has no 'graph_path' key.
  """
  metric = SimpleNamespace(
      metric_name="path",
      expected_graph_path=["n1", "n2"],
  )
  # Valid Python dict in [GraphMetadata] but no 'graph_path' key
  inv = _make_invocation_with_event("[GraphMetadata] {'graph_state': {'x': 1}}")

  result = graph_path_match(metric, [inv], None, None)

  assert result.overall_eval_status == EvalStatus.FAILED
  assert result.overall_score == 0.0


def test_state_contains_keys_actual_state_none_from_events():
  """Lines 217-218: expected_state set but actual_state is None → FAILED.

  Valid [GraphMetadata] event whose dict has no 'graph_state' key.
  """
  metric = SimpleNamespace(
      metric_name="state",
      expected_state={"key1": "v1"},
  )
  inv = _make_invocation_with_event("[GraphMetadata] {'graph_path': ['n1']}")

  result = state_contains_keys(metric, [inv], None, None)

  assert result.overall_eval_status == EvalStatus.FAILED
  assert result.overall_score == 0.0


def test_state_contains_keys_malformed_metadata_exception_handled():
  """Lines 205-206: malformed [GraphMetadata] in state metric → continue."""
  metric = SimpleNamespace(
      metric_name="state",
      expected_state={"key1": "v1"},
  )
  inv = _make_invocation_with_event("[GraphMetadata] << invalid >>")

  result = state_contains_keys(metric, [inv], None, None)

  # Parsing error → actual_state stays None → FAILED
  assert result.overall_eval_status == EvalStatus.FAILED


def test_node_execution_count_empty_actual_counts():
  """Lines 327-328: expected_counts set but no actual counts found → FAILED."""
  metric = SimpleNamespace(
      metric_name="count",
      expected_node_counts={"loop_node": 3},
  )
  # [GraphMetadata] present but no 'node_invocations' key
  inv = _make_invocation_with_event("[GraphMetadata] {'graph_path': ['n1']}")

  result = node_execution_count(metric, [inv], None, None)

  assert result.overall_eval_status == EvalStatus.FAILED
  assert result.overall_score == 0.0


def test_node_execution_count_malformed_metadata_exception_handled():
  """Lines 317-318: malformed [GraphMetadata] in count metric → continue."""
  metric = SimpleNamespace(
      metric_name="count",
      expected_node_counts={"loop_node": 3},
  )
  inv = _make_invocation_with_event("[GraphMetadata] *** bad ***")

  result = node_execution_count(metric, [inv], None, None)

  # Exception swallowed → actual_counts empty → FAILED
  assert result.overall_eval_status == EvalStatus.FAILED
