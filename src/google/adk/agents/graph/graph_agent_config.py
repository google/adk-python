"""Config definition for GraphAgent."""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional

from pydantic import BaseModel  # type: ignore[attr-defined]
from pydantic import ConfigDict
from pydantic import Field

from ...utils.feature_decorator import experimental
from ..base_agent_config import BaseAgentConfig  # type: ignore[attr-defined]
from ..common_configs import AgentRefConfig
from ..common_configs import CodeConfig


@experimental
class GraphNodeConfig(BaseModel):  # type: ignore[misc]
  """Configuration for a single node in the graph.

  A node can contain either an agent reference or a function reference,
  plus optional mappers and reducers for state management.
  """

  model_config = ConfigDict(extra="forbid")

  name: str = Field(description="Node name")

  # Node can reference an agent (sub_agents) OR a function
  sub_agents: Optional[List[AgentRefConfig]] = Field(
      default=None,
      description="Sub-agents for this node",
  )

  function_ref: Optional[str] = Field(
      default=None,
      description="Reference to a function (e.g., 'module.function_name')",
  )

  input_mapper_ref: Optional[str] = Field(
      default=None,
      description="Reference to custom input mapper function",
  )

  output_mapper_ref: Optional[str] = Field(
      default=None,
      description="Reference to custom output mapper function",
  )

  reducer: str = Field(
      default="overwrite",
      description="State reducer strategy: overwrite|append|sum|custom",
  )

  custom_reducer_ref: Optional[str] = Field(
      default=None,
      description="Reference to custom reducer function (if reducer=custom)",
  )


@experimental
class GraphEdgeConfig(BaseModel):  # type: ignore[misc]
  """Configuration for an edge between nodes.

  Edges can have optional conditions for conditional routing.
  """

  model_config = ConfigDict(extra="forbid")

  source_node: str = Field(description="Source node name")

  target_node: str = Field(description="Target node name")

  condition: Optional[str] = Field(
      default=None,
      description=(
          "AST-safe condition expression evaluated against graph state."
          " Allowed names: state, data, metadata, True, False, None."
          " Allowed methods: .get(), .get_parsed(), .get_str(),"
          " .get_dict(). Supports comparisons, boolean ops, 'in'."
          " Example: \"data.get('approved') is True\""
      ),
  )

  priority: int = Field(
      default=1,
      description="Edge priority for routing (higher = evaluated first)",
  )

  weight: float = Field(
      default=1.0,
      description="Edge weight for weighted random routing",
  )


@experimental
class InterruptConfigYaml(BaseModel):  # type: ignore[misc]
  """Configuration for interrupt handling."""

  model_config = ConfigDict(extra="forbid")

  mode: Optional[Literal["before", "after", "both"]] = Field(
      default=None,
      description="Interrupt mode (None = disabled, before|after|both)",
  )

  interrupt_service: Optional[CodeConfig] = Field(
      default=None,
      description="Interrupt service configuration (CodeConfig)",
  )


@experimental
class ParallelGroupConfig(BaseModel):  # type: ignore[misc]
  """Configuration for parallel node execution."""

  model_config = ConfigDict(extra="forbid")

  nodes: List[str] = Field(
      description="List of node names to execute in parallel"
  )

  join_strategy: str = Field(
      default="all",
      description="Join strategy: all|any|n",
  )

  error_policy: str = Field(
      default="fail_fast",
      description="Error policy: fail_fast|continue|collect",
  )

  wait_n: int = Field(
      default=1,
      description="Number of nodes to wait for (when join_strategy=n)",
  )


@experimental
class TelemetryConfig(BaseModel):  # type: ignore[misc]
  """Configuration for GraphAgent telemetry.

  Controls OpenTelemetry instrumentation for graph workflow execution.
  """

  model_config = ConfigDict(extra="forbid")

  enabled: bool = Field(
      default=True,
      description="Enable/disable all telemetry collection",
  )

  trace_nodes: bool = Field(
      default=True,
      description="Enable tracing for node executions",
  )

  trace_edges: bool = Field(
      default=True,
      description="Enable tracing for edge condition evaluations",
  )

  trace_iterations: bool = Field(
      default=True,
      description="Enable metrics for graph iterations",
  )

  trace_parallel_groups: bool = Field(
      default=True,
      description="Enable tracing for parallel group executions",
  )

  trace_callbacks: bool = Field(
      default=True,
      description="Enable tracing for callback executions",
  )

  trace_interrupts: bool = Field(
      default=True,
      description="Enable tracing for interrupt checks",
  )

  sampling_rate: float = Field(
      default=1.0,
      ge=0.0,
      le=1.0,
      description="Sampling rate for telemetry (0.0-1.0, 1.0 = 100%)",
  )

  additional_attributes: Optional[Dict[str, Any]] = Field(
      default=None,
      description="Additional custom attributes to add to all telemetry",
  )


@experimental
class GraphAgentConfig(BaseAgentConfig):  # type: ignore[misc]
  """The config for the YAML schema of a GraphAgent.

  This config supports defining graph structure, nodes, edges, and
  advanced features like interrupts and parallel execution.

  Example YAML:
      ```yaml
      agent_class: GraphAgent
      name: my_graph
      description: My graph workflow
      start_node: start
      end_nodes:
        - end
      max_iterations: 10
      checkpointing: true
      nodes:
        - name: start
          sub_agents:
            - agent1
        - name: middle
          sub_agents:
            - agent2
        - name: end
          sub_agents:
            - agent3
      edges:
        - source_node: start
          target_node: middle
        - source_node: middle
          target_node: end
      ```
  """

  model_config = ConfigDict(extra="forbid")

  agent_class: str = Field(
      default="GraphAgent",
      description=(
          "The value is used to uniquely identify the GraphAgent class."
      ),
  )

  start_node: str = Field(description="Name of the starting node")

  end_nodes: List[str] = Field(
      default_factory=list,
      description="List of end node names",
  )

  max_iterations: int = Field(
      default=20,
      description="Maximum iterations for cyclic graphs",
  )

  checkpointing: bool = Field(
      default=False,
      description="Enable automatic checkpointing",
  )

  checkpoint_service: Optional[CodeConfig] = Field(
      default=None,
      description="Checkpoint service configuration (CodeConfig)",
  )

  # Graph structure
  nodes: List[GraphNodeConfig] = Field(
      default_factory=list,
      description="List of node configurations",
  )

  edges: List[GraphEdgeConfig] = Field(
      default_factory=list,
      description="List of edge configurations",
  )

  # Advanced features
  interrupt_config: Optional[InterruptConfigYaml] = Field(
      default=None,
      description="Interrupt configuration",
  )

  telemetry_config: Optional[TelemetryConfig] = Field(
      default=None,
      description="Telemetry configuration for OpenTelemetry instrumentation",
  )

  parallel_groups: List[ParallelGroupConfig] = Field(
      default_factory=list,
      description="List of parallel execution group configurations",
  )

  # Callbacks (following ADK CodeConfig pattern)
  before_node_callbacks: Optional[List[CodeConfig]] = Field(
      default=None,
      description="Callbacks executed before each node",
  )

  after_node_callbacks: Optional[List[CodeConfig]] = Field(
      default=None,
      description="Callbacks executed after each node",
  )

  on_edge_condition_callbacks: Optional[List[CodeConfig]] = Field(
      default=None,
      description="Callbacks executed when evaluating edge conditions",
  )
