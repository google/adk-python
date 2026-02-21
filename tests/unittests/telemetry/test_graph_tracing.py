"""Tests for GraphAgent telemetry instrumentation."""

from unittest import mock

from google.adk.telemetry import graph_tracing
import pytest


def test_telemetry_module_imports():
  """Test that all telemetry exports are available."""
  assert hasattr(graph_tracing, "tracer")
  assert hasattr(graph_tracing, "meter")
  assert hasattr(graph_tracing, "logger")
  assert hasattr(graph_tracing, "otel_logger")

  # Metrics
  assert hasattr(graph_tracing, "node_execution_counter")
  assert hasattr(graph_tracing, "node_execution_latency")
  assert hasattr(graph_tracing, "edge_evaluation_counter")
  assert hasattr(graph_tracing, "edge_evaluation_latency")
  assert hasattr(graph_tracing, "graph_iteration_counter")
  assert hasattr(graph_tracing, "parallel_group_counter")
  assert hasattr(graph_tracing, "parallel_group_latency")
  assert hasattr(graph_tracing, "callback_execution_counter")
  assert hasattr(graph_tracing, "callback_execution_latency")
  assert hasattr(graph_tracing, "interrupt_check_counter")

  # Semantic conventions
  assert hasattr(graph_tracing, "GRAPH_AGENT_NAME")
  assert hasattr(graph_tracing, "GRAPH_NODE_NAME")
  assert hasattr(graph_tracing, "GRAPH_NODE_TYPE")
  assert hasattr(graph_tracing, "GRAPH_EDGE_SOURCE")
  assert hasattr(graph_tracing, "GRAPH_EDGE_TARGET")

  # Recording functions
  assert hasattr(graph_tracing, "record_node_execution")
  assert hasattr(graph_tracing, "record_edge_evaluation")
  assert hasattr(graph_tracing, "record_graph_iteration")
  assert hasattr(graph_tracing, "record_parallel_group_execution")
  assert hasattr(graph_tracing, "record_callback_execution")
  assert hasattr(graph_tracing, "record_interrupt_check")


def test_record_node_execution_success():
  """Test recording successful node execution."""
  with (
      mock.patch.object(
          graph_tracing.node_execution_counter, "add"
      ) as mock_counter,
      mock.patch.object(
          graph_tracing.node_execution_latency, "record"
      ) as mock_latency,
  ):
    graph_tracing.record_node_execution(
        node_name="test_node",
        node_type="agent",
        agent_name="test_graph",
        latency_ms=100.5,
        success=True,
    )

    # Verify counter was called
    assert mock_counter.call_count == 1
    call_args = mock_counter.call_args
    assert (
        call_args.kwargs["attributes"][graph_tracing.GRAPH_NODE_NAME]
        == "test_node"
    )
    assert (
        call_args.kwargs["attributes"][graph_tracing.GRAPH_NODE_TYPE] == "agent"
    )
    assert (
        call_args.kwargs["attributes"][graph_tracing.GRAPH_AGENT_NAME]
        == "test_graph"
    )
    assert call_args.kwargs["attributes"]["success"] is True

    # Verify latency was recorded
    assert mock_latency.call_count == 1
    latency_call_args = mock_latency.call_args
    assert latency_call_args.args[0] == 100.5


def test_record_node_execution_failure():
  """Test recording failed node execution."""
  with (
      mock.patch.object(
          graph_tracing.node_execution_counter, "add"
      ) as mock_counter,
      mock.patch.object(
          graph_tracing.node_execution_latency, "record"
      ) as mock_latency,
  ):
    graph_tracing.record_node_execution(
        node_name="failing_node",
        node_type="function",
        agent_name="test_graph",
        latency_ms=50.0,
        success=False,
    )

    # Verify counter was called with success=False
    assert mock_counter.call_count == 1
    call_args = mock_counter.call_args
    assert call_args.kwargs["attributes"]["success"] is False
    assert (
        call_args.kwargs["attributes"][graph_tracing.GRAPH_NODE_TYPE]
        == "function"
    )

    # Verify latency was still recorded
    assert mock_latency.call_count == 1


def test_record_edge_evaluation():
  """Test recording edge condition evaluation."""
  with (
      mock.patch.object(
          graph_tracing.edge_evaluation_counter, "add"
      ) as mock_counter,
      mock.patch.object(
          graph_tracing.edge_evaluation_latency, "record"
      ) as mock_latency,
  ):
    graph_tracing.record_edge_evaluation(
        source_node="node_a",
        target_node="node_b",
        agent_name="test_graph",
        condition_result=True,
        latency_ms=5.2,
        priority=2,
    )

    # Verify counter was called
    assert mock_counter.call_count == 1
    call_args = mock_counter.call_args
    assert (
        call_args.kwargs["attributes"][graph_tracing.GRAPH_EDGE_SOURCE]
        == "node_a"
    )
    assert (
        call_args.kwargs["attributes"][graph_tracing.GRAPH_EDGE_TARGET]
        == "node_b"
    )
    assert (
        call_args.kwargs["attributes"][
            graph_tracing.GRAPH_EDGE_CONDITION_RESULT
        ]
        == "True"
    )
    assert (
        call_args.kwargs["attributes"][graph_tracing.GRAPH_EDGE_PRIORITY] == 2
    )

    # Verify latency was recorded
    assert mock_latency.call_count == 1
    latency_call_args = mock_latency.call_args
    assert latency_call_args.args[0] == 5.2


def test_record_edge_evaluation_false_condition():
  """Test recording edge evaluation with false condition result."""
  with mock.patch.object(
      graph_tracing.edge_evaluation_counter, "add"
  ) as mock_counter:
    graph_tracing.record_edge_evaluation(
        source_node="node_a",
        target_node="node_c",
        agent_name="test_graph",
        condition_result=False,
        latency_ms=3.1,
        priority=1,
    )

    call_args = mock_counter.call_args
    assert (
        call_args.kwargs["attributes"][
            graph_tracing.GRAPH_EDGE_CONDITION_RESULT
        ]
        == "False"
    )


def test_record_graph_iteration():
  """Test recording graph iteration metrics."""
  with mock.patch.object(
      graph_tracing.graph_iteration_counter, "add"
  ) as mock_counter:
    graph_tracing.record_graph_iteration(
        agent_name="test_graph", iteration=5, path_length=10
    )

    # Verify counter was called
    assert mock_counter.call_count == 1
    call_args = mock_counter.call_args
    assert (
        call_args.kwargs["attributes"][graph_tracing.GRAPH_AGENT_NAME]
        == "test_graph"
    )
    assert call_args.kwargs["attributes"][graph_tracing.GRAPH_ITERATION] == 5
    assert call_args.kwargs["attributes"]["path_length"] == 10


def test_record_parallel_group_execution():
  """Test recording parallel group execution metrics."""
  with (
      mock.patch.object(
          graph_tracing.parallel_group_counter, "add"
      ) as mock_counter,
      mock.patch.object(
          graph_tracing.parallel_group_latency, "record"
      ) as mock_latency,
  ):
    graph_tracing.record_parallel_group_execution(
        agent_name="test_graph",
        node_count=3,
        strategy="all",
        latency_ms=250.5,
        completed_count=3,
    )

    # Verify counter was called
    assert mock_counter.call_count == 1
    call_args = mock_counter.call_args
    assert (
        call_args.kwargs["attributes"][graph_tracing.GRAPH_PARALLEL_NODE_COUNT]
        == 3
    )
    assert (
        call_args.kwargs["attributes"][graph_tracing.GRAPH_PARALLEL_STRATEGY]
        == "all"
    )
    assert call_args.kwargs["attributes"]["completed_count"] == 3

    # Verify latency was recorded
    assert mock_latency.call_count == 1
    latency_call_args = mock_latency.call_args
    assert latency_call_args.args[0] == 250.5


def test_record_parallel_group_partial_completion():
  """Test recording parallel group with partial completion."""
  with mock.patch.object(
      graph_tracing.parallel_group_counter, "add"
  ) as mock_counter:
    graph_tracing.record_parallel_group_execution(
        agent_name="test_graph",
        node_count=5,
        strategy="any",
        latency_ms=100.0,
        completed_count=2,
    )

    call_args = mock_counter.call_args
    assert (
        call_args.kwargs["attributes"][graph_tracing.GRAPH_PARALLEL_NODE_COUNT]
        == 5
    )
    assert (
        call_args.kwargs["attributes"][graph_tracing.GRAPH_PARALLEL_STRATEGY]
        == "any"
    )
    assert call_args.kwargs["attributes"]["completed_count"] == 2


def test_record_callback_execution_before_node():
  """Test recording before_node callback execution."""
  with (
      mock.patch.object(
          graph_tracing.callback_execution_counter, "add"
      ) as mock_counter,
      mock.patch.object(
          graph_tracing.callback_execution_latency, "record"
      ) as mock_latency,
  ):
    graph_tracing.record_callback_execution(
        callback_type="before_node",
        agent_name="test_graph",
        latency_ms=10.5,
        success=True,
    )

    # Verify counter was called
    assert mock_counter.call_count == 1
    call_args = mock_counter.call_args
    assert (
        call_args.kwargs["attributes"][graph_tracing.GRAPH_CALLBACK_TYPE]
        == "before_node"
    )
    assert call_args.kwargs["attributes"]["success"] is True

    # Verify latency was recorded
    assert mock_latency.call_count == 1


def test_record_callback_execution_after_node():
  """Test recording after_node callback execution."""
  with mock.patch.object(
      graph_tracing.callback_execution_counter, "add"
  ) as mock_counter:
    graph_tracing.record_callback_execution(
        callback_type="after_node",
        agent_name="test_graph",
        latency_ms=15.2,
        success=True,
    )

    call_args = mock_counter.call_args
    assert (
        call_args.kwargs["attributes"][graph_tracing.GRAPH_CALLBACK_TYPE]
        == "after_node"
    )


def test_record_callback_execution_on_edge():
  """Test recording on_edge callback execution."""
  with mock.patch.object(
      graph_tracing.callback_execution_counter, "add"
  ) as mock_counter:
    graph_tracing.record_callback_execution(
        callback_type="on_edge",
        agent_name="test_graph",
        latency_ms=5.0,
        success=True,
    )

    call_args = mock_counter.call_args
    assert (
        call_args.kwargs["attributes"][graph_tracing.GRAPH_CALLBACK_TYPE]
        == "on_edge"
    )


def test_record_callback_execution_failure():
  """Test recording failed callback execution."""
  with mock.patch.object(
      graph_tracing.callback_execution_counter, "add"
  ) as mock_counter:
    graph_tracing.record_callback_execution(
        callback_type="before_node",
        agent_name="test_graph",
        latency_ms=8.0,
        success=False,
    )

    call_args = mock_counter.call_args
    assert call_args.kwargs["attributes"]["success"] is False


def test_record_interrupt_check():
  """Test recording interrupt check metrics."""
  with mock.patch.object(
      graph_tracing.interrupt_check_counter, "add"
  ) as mock_counter:
    graph_tracing.record_interrupt_check(
        mode="before", agent_name="test_graph", session_id="session_123"
    )

    # Verify counter was called
    assert mock_counter.call_count == 1
    call_args = mock_counter.call_args
    assert (
        call_args.kwargs["attributes"][graph_tracing.GRAPH_INTERRUPT_MODE]
        == "before"
    )
    assert (
        call_args.kwargs["attributes"][graph_tracing.GRAPH_AGENT_NAME]
        == "test_graph"
    )
    assert (
        call_args.kwargs["attributes"][graph_tracing.GRAPH_SESSION_ID]
        == "session_123"
    )


def test_record_interrupt_check_after_mode():
  """Test recording interrupt check in after mode."""
  with mock.patch.object(
      graph_tracing.interrupt_check_counter, "add"
  ) as mock_counter:
    graph_tracing.record_interrupt_check(
        mode="after", agent_name="test_graph", session_id="session_456"
    )

    call_args = mock_counter.call_args
    assert (
        call_args.kwargs["attributes"][graph_tracing.GRAPH_INTERRUPT_MODE]
        == "after"
    )
    assert (
        call_args.kwargs["attributes"][graph_tracing.GRAPH_SESSION_ID]
        == "session_456"
    )


def test_record_interrupt_check_both_mode():
  """Test recording interrupt check in both mode."""
  with mock.patch.object(
      graph_tracing.interrupt_check_counter, "add"
  ) as mock_counter:
    graph_tracing.record_interrupt_check(
        mode="both", agent_name="test_graph", session_id="session_789"
    )

    call_args = mock_counter.call_args
    assert (
        call_args.kwargs["attributes"][graph_tracing.GRAPH_INTERRUPT_MODE]
        == "both"
    )


def test_semantic_convention_values():
  """Test that semantic convention constants have correct values."""
  assert graph_tracing.GRAPH_AGENT_NAME == "graph.agent.name"
  assert graph_tracing.GRAPH_NODE_NAME == "graph.node.name"
  assert graph_tracing.GRAPH_NODE_TYPE == "graph.node.type"
  assert graph_tracing.GRAPH_NODE_ITERATION == "graph.node.iteration"
  assert graph_tracing.GRAPH_EDGE_SOURCE == "graph.edge.source"
  assert graph_tracing.GRAPH_EDGE_TARGET == "graph.edge.target"
  assert (
      graph_tracing.GRAPH_EDGE_CONDITION_RESULT == "graph.edge.condition.result"
  )
  assert graph_tracing.GRAPH_EDGE_PRIORITY == "graph.edge.priority"
  assert graph_tracing.GRAPH_ITERATION == "graph.iteration"
  assert graph_tracing.GRAPH_PATH == "graph.path"
  assert graph_tracing.GRAPH_PARALLEL_NODE_COUNT == "graph.parallel.node_count"
  assert graph_tracing.GRAPH_PARALLEL_STRATEGY == "graph.parallel.strategy"
  assert graph_tracing.GRAPH_PARALLEL_WAIT_N == "graph.parallel.wait_n"
  assert graph_tracing.GRAPH_CALLBACK_TYPE == "graph.callback.type"
  assert graph_tracing.GRAPH_INTERRUPT_MODE == "graph.interrupt.mode"
  assert graph_tracing.GRAPH_SESSION_ID == "graph.session.id"


def test_multiple_node_executions():
  """Test recording multiple node executions with different types."""
  with mock.patch.object(
      graph_tracing.node_execution_counter, "add"
  ) as mock_counter:
    # Record agent node
    graph_tracing.record_node_execution(
        node_name="agent_node",
        node_type="agent",
        agent_name="test_graph",
        latency_ms=100.0,
        success=True,
    )

    # Record function node
    graph_tracing.record_node_execution(
        node_name="function_node",
        node_type="function",
        agent_name="test_graph",
        latency_ms=50.0,
        success=True,
    )

    # Verify both were recorded
    assert mock_counter.call_count == 2

    # Verify first call
    first_call_args = mock_counter.call_args_list[0]
    assert (
        first_call_args.kwargs["attributes"][graph_tracing.GRAPH_NODE_NAME]
        == "agent_node"
    )
    assert (
        first_call_args.kwargs["attributes"][graph_tracing.GRAPH_NODE_TYPE]
        == "agent"
    )

    # Verify second call
    second_call_args = mock_counter.call_args_list[1]
    assert (
        second_call_args.kwargs["attributes"][graph_tracing.GRAPH_NODE_NAME]
        == "function_node"
    )
    assert (
        second_call_args.kwargs["attributes"][graph_tracing.GRAPH_NODE_TYPE]
        == "function"
    )


def test_edge_evaluations_with_different_priorities():
  """Test recording edge evaluations with different priorities."""
  with mock.patch.object(
      graph_tracing.edge_evaluation_counter, "add"
  ) as mock_counter:
    # High priority edge
    graph_tracing.record_edge_evaluation(
        source_node="node_a",
        target_node="node_b",
        agent_name="test_graph",
        condition_result=True,
        latency_ms=5.0,
        priority=10,
    )

    # Low priority edge
    graph_tracing.record_edge_evaluation(
        source_node="node_a",
        target_node="node_c",
        agent_name="test_graph",
        condition_result=False,
        latency_ms=3.0,
        priority=1,
    )

    assert mock_counter.call_count == 2

    # Verify priorities
    assert (
        mock_counter.call_args_list[0].kwargs["attributes"][
            graph_tracing.GRAPH_EDGE_PRIORITY
        ]
        == 10
    )
    assert (
        mock_counter.call_args_list[1].kwargs["attributes"][
            graph_tracing.GRAPH_EDGE_PRIORITY
        ]
        == 1
    )
