"""OpenTelemetry instrumentation for GraphAgent workflow execution.

This module provides tracing, logging, and metrics for graph orchestration
following OpenTelemetry semantic conventions.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from opentelemetry import _logs
from opentelemetry import metrics
from opentelemetry import trace
from opentelemetry.semconv.schemas import Schemas

from .. import version

if TYPE_CHECKING:
  from ..agents.graph.graph_state import GraphState
  from ..agents.invocation_context import InvocationContext

# OpenTelemetry tracer for graph execution
tracer = trace.get_tracer(
    instrumenting_module_name="gcp.vertex.agent.graph",
    instrumenting_library_version=version.__version__,
    schema_url=Schemas.V1_36_0.value,
)

# OpenTelemetry logger for structured logs
otel_logger = _logs.get_logger(
    instrumenting_module_name="gcp.vertex.agent.graph",
    instrumenting_library_version=version.__version__,
    schema_url=Schemas.V1_36_0.value,
)

# Python logger for standard logging
logger = logging.getLogger("google_adk." + __name__)

# OpenTelemetry meter for metrics
meter = metrics.get_meter(
    name="gcp.vertex.agent.graph",
    version=version.__version__,
    schema_url=Schemas.V1_36_0.value,
)

# Metrics - Node Execution
node_execution_counter = meter.create_counter(
    name="graph.node.executions",
    description="Number of node executions",
    unit="1",
)

node_execution_latency = meter.create_histogram(
    name="graph.node.latency",
    description="Node execution latency in milliseconds",
    unit="ms",
)

# Metrics - Edge Evaluation
edge_evaluation_counter = meter.create_counter(
    name="graph.edge.evaluations",
    description="Number of edge condition evaluations",
    unit="1",
)

edge_evaluation_latency = meter.create_histogram(
    name="graph.edge.latency",
    description="Edge condition evaluation latency in milliseconds",
    unit="ms",
)

# Metrics - Graph Iterations
graph_iteration_counter = meter.create_counter(
    name="graph.iterations",
    description="Graph execution iterations",
    unit="1",
)

# Metrics - Parallel Groups
parallel_group_counter = meter.create_counter(
    name="graph.parallel.executions",
    description="Parallel group executions",
    unit="1",
)

parallel_group_latency = meter.create_histogram(
    name="graph.parallel.latency",
    description="Parallel group execution latency in milliseconds",
    unit="ms",
)

# Metrics - Callbacks
callback_execution_counter = meter.create_counter(
    name="graph.callback.executions",
    description="Callback executions",
    unit="1",
)

callback_execution_latency = meter.create_histogram(
    name="graph.callback.latency",
    description="Callback execution latency in milliseconds",
    unit="ms",
)

# Metrics - Interrupts
interrupt_check_counter = meter.create_counter(
    name="graph.interrupt.checks",
    description="Interrupt check operations",
    unit="1",
)

# Metrics - State Reducers
state_reducer_counter = meter.create_counter(
    name="graph.state.reducer.applications",
    description="State reducer applications",
    unit="1",
)

state_reducer_latency = meter.create_histogram(
    name="graph.state.reducer.latency",
    description="State reducer application latency in milliseconds",
    unit="ms",
)

# Metrics - Mappers
mapper_counter = meter.create_counter(
    name="graph.mapper.applications",
    description="Mapper function applications",
    unit="1",
)

mapper_latency = meter.create_histogram(
    name="graph.mapper.latency",
    description="Mapper function execution latency in milliseconds",
    unit="ms",
)

# Semantic Conventions - Graph Attributes
GRAPH_AGENT_NAME = "graph.agent.name"
GRAPH_NODE_NAME = "graph.node.name"
GRAPH_NODE_TYPE = "graph.node.type"  # agent|function
GRAPH_NODE_ITERATION = "graph.node.iteration"
GRAPH_EDGE_SOURCE = "graph.edge.source"
GRAPH_EDGE_TARGET = "graph.edge.target"
GRAPH_EDGE_CONDITION_RESULT = "graph.edge.condition.result"
GRAPH_EDGE_PRIORITY = "graph.edge.priority"
GRAPH_ITERATION = "graph.iteration"
GRAPH_PATH = "graph.path"
GRAPH_PARALLEL_NODE_COUNT = "graph.parallel.node_count"
GRAPH_PARALLEL_STRATEGY = "graph.parallel.strategy"
GRAPH_PARALLEL_WAIT_N = "graph.parallel.wait_n"
GRAPH_CALLBACK_TYPE = "graph.callback.type"  # before_node|after_node|on_edge
GRAPH_INTERRUPT_MODE = "graph.interrupt.mode"  # before|after|both
GRAPH_SESSION_ID = "graph.session.id"
GRAPH_STATE_REDUCER_TYPE = (  # OVERWRITE|APPEND|SUM|CUSTOM
    "graph.state.reducer.type"
)
GRAPH_STATE_KEY = "graph.state.key"  # Key being modified in state
GRAPH_MAPPER_TYPE = "graph.mapper.type"  # input|output
GRAPH_MAPPER_IS_DEFAULT = (  # Whether using default mapper
    "graph.mapper.is_default"
)


def record_node_execution(
    node_name: str,
    node_type: str,
    agent_name: str,
    latency_ms: float,
    success: bool = True,
) -> None:
  """Record node execution metrics.

  Args:
      node_name: Name of the executed node
      node_type: Type of node (agent or function)
      agent_name: Name of the GraphAgent
      latency_ms: Execution latency in milliseconds
      success: Whether execution succeeded
  """
  attributes = {
      GRAPH_NODE_NAME: node_name,
      GRAPH_NODE_TYPE: node_type,
      GRAPH_AGENT_NAME: agent_name,
      "success": success,
  }

  node_execution_counter.add(1, attributes=attributes)
  node_execution_latency.record(latency_ms, attributes=attributes)


def record_edge_evaluation(
    source_node: str,
    target_node: str,
    agent_name: str,
    condition_result: bool,
    latency_ms: float,
    priority: int = 0,
) -> None:
  """Record edge condition evaluation metrics.

  Args:
      source_node: Source node name
      target_node: Target node name
      agent_name: Name of the GraphAgent
      condition_result: Result of condition evaluation
      latency_ms: Evaluation latency in milliseconds
      priority: Edge priority
  """
  attributes = {
      GRAPH_EDGE_SOURCE: source_node,
      GRAPH_EDGE_TARGET: target_node,
      GRAPH_AGENT_NAME: agent_name,
      GRAPH_EDGE_CONDITION_RESULT: str(condition_result),
      GRAPH_EDGE_PRIORITY: priority,
  }

  edge_evaluation_counter.add(1, attributes=attributes)
  edge_evaluation_latency.record(latency_ms, attributes=attributes)


def record_graph_iteration(
    agent_name: str,
    iteration: int,
    path_length: int,
) -> None:
  """Record graph iteration metrics.

  Args:
      agent_name: Name of the GraphAgent
      iteration: Current iteration number
      path_length: Length of execution path so far
  """
  attributes = {
      GRAPH_AGENT_NAME: agent_name,
      GRAPH_ITERATION: iteration,
      "path_length": path_length,
  }

  graph_iteration_counter.add(1, attributes=attributes)


def record_parallel_group_execution(
    agent_name: str,
    node_count: int,
    strategy: str,
    latency_ms: float,
    completed_count: int,
) -> None:
  """Record parallel group execution metrics.

  Args:
      agent_name: Name of the GraphAgent
      node_count: Number of nodes in parallel group
      strategy: Join strategy (all, any, n)
      latency_ms: Total execution latency in milliseconds
      completed_count: Number of nodes that completed successfully
  """
  attributes = {
      GRAPH_AGENT_NAME: agent_name,
      GRAPH_PARALLEL_NODE_COUNT: node_count,
      GRAPH_PARALLEL_STRATEGY: strategy,
      "completed_count": completed_count,
  }

  parallel_group_counter.add(1, attributes=attributes)
  parallel_group_latency.record(latency_ms, attributes=attributes)


def record_callback_execution(
    callback_type: str,
    agent_name: str,
    latency_ms: float,
    success: bool = True,
) -> None:
  """Record callback execution metrics.

  Args:
      callback_type: Type of callback (before_node, after_node, on_edge)
      agent_name: Name of the GraphAgent
      latency_ms: Execution latency in milliseconds
      success: Whether callback succeeded
  """
  attributes = {
      GRAPH_CALLBACK_TYPE: callback_type,
      GRAPH_AGENT_NAME: agent_name,
      "success": success,
  }

  callback_execution_counter.add(1, attributes=attributes)
  callback_execution_latency.record(latency_ms, attributes=attributes)


def record_interrupt_check(
    mode: str,
    agent_name: str,
    session_id: str,
) -> None:
  """Record interrupt check metrics.

  Args:
      mode: Interrupt mode (before, after, both)
      agent_name: Name of the GraphAgent
      session_id: Session identifier
  """
  attributes = {
      GRAPH_INTERRUPT_MODE: mode,
      GRAPH_AGENT_NAME: agent_name,
      GRAPH_SESSION_ID: session_id,
  }

  interrupt_check_counter.add(1, attributes=attributes)


def record_state_reducer(
    node_name: str,
    reducer_type: str,
    state_key: str,
    agent_name: str,
    latency_ms: float,
    had_previous_value: bool,
) -> None:
  """Record state reducer application metrics.

  Args:
      node_name: Name of the node applying the reducer
      reducer_type: Type of reducer (OVERWRITE, APPEND, SUM, CUSTOM)
      state_key: Key being modified in state.data
      agent_name: Name of the GraphAgent
      latency_ms: Reducer application latency in milliseconds
      had_previous_value: Whether the key existed in state before reduction
  """
  attributes = {
      GRAPH_NODE_NAME: node_name,
      GRAPH_STATE_REDUCER_TYPE: reducer_type,
      GRAPH_STATE_KEY: state_key,
      GRAPH_AGENT_NAME: agent_name,
      "had_previous_value": had_previous_value,
  }

  state_reducer_counter.add(1, attributes=attributes)
  state_reducer_latency.record(latency_ms, attributes=attributes)


def record_mapper(
    node_name: str,
    mapper_type: str,
    agent_name: str,
    latency_ms: float,
    is_default: bool,
) -> None:
  """Record mapper transformation metrics.

  Args:
      node_name: Name of the node using the mapper
      mapper_type: Type of mapper (input or output)
      agent_name: Name of the GraphAgent
      latency_ms: Mapper execution latency in milliseconds
      is_default: Whether using default mapper implementation
  """
  attributes = {
      GRAPH_NODE_NAME: node_name,
      GRAPH_MAPPER_TYPE: mapper_type,
      GRAPH_AGENT_NAME: agent_name,
      GRAPH_MAPPER_IS_DEFAULT: is_default,
  }

  mapper_counter.add(1, attributes=attributes)
  mapper_latency.record(latency_ms, attributes=attributes)
