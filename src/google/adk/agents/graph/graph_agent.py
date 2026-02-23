"""Graph-based workflow orchestration for ADK.

GraphAgent is ADK's fourth workflow agent type (alongside Sequential, Loop, Parallel),
enabling directed graph-based orchestration with conditional routing and complex branching.

GraphAgent enables workflow creation using directed graphs where:
- Nodes are agents or functions
- Edges define allowed transitions with optional conditions
- State flows through the graph with configurable reducers
- Full checkpointing support via CheckpointService integration

Key features:
- Directed graph workflows with conditional routing
- State management with custom reducers (OVERWRITE, APPEND, SUM, CUSTOM)
- Always-on observability: node lifecycle events for every execution
- Human-in-the-loop interrupts via InterruptService (retrospective feedback)
- CheckpointService integration for checkpoint/resume
- DatabaseSessionService support for persistence
- Cyclic execution with max_iterations
- Event-based state persistence (ADK-native)

Inspired by adk-graph (Rust) and LangGraph patterns.

Checkpointing Integration:
    For checkpoint/resume functionality, use CheckpointService with CheckpointCallback:

    ```python
    from google.adk.agents.graph import GraphAgent
    from google.adk.checkpoints import CheckpointService, CheckpointCallback
    from google.adk.sessions import InMemorySessionService

    # Create services
    session_service = InMemorySessionService()
    checkpoint_service = CheckpointService(session_service)

    # Create graph with checkpoint callback
    graph = GraphAgent(name="workflow", checkpointing=True)
    graph.add_node(...)
    graph.set_callbacks([
        CheckpointCallback(checkpoint_service, checkpoint_after=True)
    ])

    # Checkpoints are created automatically after each node
    # Use checkpoint_service to list/delete/export/import checkpoints
    ```
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import time
from typing import Any
from typing import AsyncGenerator
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING

from google.genai import types
from pydantic import ConfigDict
from pydantic import Field
from typing_extensions import override

from ...events.event import Event
from ...events.event_actions import EventActions
from ...telemetry import graph_tracing
from ...telemetry.tracing import tracer
from ...utils.feature_decorator import experimental
from ..base_agent import BaseAgent
from ..invocation_context import InvocationContext
from ..llm_agent import LlmAgent
from .callbacks import EdgeCallback
from .callbacks import NodeCallback
from .graph_agent_state import GraphAgentState
from .graph_edge import EdgeCondition
from .graph_interrupt_handler import GraphInterruptMixin
from .graph_node import GraphNode
from .graph_state import GraphState
from .graph_state import StateReducer
from .graph_telemetry import GraphTelemetryMixin
from .interrupt import InterruptAction
from .interrupt import InterruptConfig
from .interrupt import InterruptMode
from .interrupt_reasoner import InterruptReasoner
from .interrupt_service import InterruptMessage
from .interrupt_service import InterruptService
from .parallel import execute_parallel_group
from .parallel import ParallelNodeGroup

if TYPE_CHECKING:
  from .graph_agent_config import TelemetryConfig

logger = logging.getLogger("google_adk." + __name__)

# Keys stored in session.state by GraphAgent's own state_delta events.
# These must be excluded when syncing session.state → state.data to avoid
# circular references (state.data["graph_data"] → state.data) and to keep
# domain data clean of graph-internal bookkeeping.
_GRAPH_INTERNAL_KEYS = frozenset({
    "graph_data",
    "graph_checkpoint",
    "graph_cancelled",
    "graph_cancelled_at_node",
    "graph_task_cancelled",
    "graph_can_resume",
    "graph_iterations",
    "graph_path",
    "graph_partial_output",
    "graph_state",
})


_SAFE_NAMES = frozenset({
    "state",
    "data",
    "True",
    "False",
    "None",
})
_SAFE_METHODS = frozenset({
    "get",
    "get_parsed",
    "get_str",
    "get_dict",
})
_SAFE_BUILTINS = frozenset({
    "len",
    "min",
    "max",
    "abs",
    "bool",
    "int",
    "float",
    "str",
    "isinstance",
    "type",
})


def _validate_condition_ast(node: ast.AST) -> None:
  """Walk AST and reject any unsafe node types.

  Only allows: comparisons, boolean ops, unary not, attribute access,
  safe method calls (.get, .get_parsed, .get_str, .get_dict),
  constants, and whitelisted names.

  Raises:
      ValueError: If an unsafe AST node is encountered.
  """
  if isinstance(node, ast.Expression):
    _validate_condition_ast(node.body)
  elif isinstance(node, ast.BoolOp):
    for value in node.values:
      _validate_condition_ast(value)
  elif isinstance(node, ast.UnaryOp):
    if not isinstance(node.op, ast.Not):
      raise ValueError(f"Unsafe unary operator: {type(node.op).__name__}")
    _validate_condition_ast(node.operand)
  elif isinstance(node, ast.Compare):
    _validate_condition_ast(node.left)
    for comparator in node.comparators:
      _validate_condition_ast(comparator)
  elif isinstance(node, ast.Call):
    if isinstance(node.func, ast.Attribute):
      if node.func.attr not in _SAFE_METHODS:
        raise ValueError(f"Unsafe method call: .{node.func.attr}()")
      _validate_condition_ast(node.func.value)
    elif isinstance(node.func, ast.Name) and node.func.id in _SAFE_BUILTINS:
      pass  # Safe builtin call
    else:
      raise ValueError(f"Unsafe call: {ast.dump(node.func)}")
    for arg in node.args:
      _validate_condition_ast(arg)
    for kw in node.keywords:
      _validate_condition_ast(kw.value)
  elif isinstance(node, ast.Attribute):
    # Block dunder attribute access to prevent sandbox escape
    # (e.g., state.__class__.__init__.__globals__)
    if node.attr.startswith("_"):
      raise ValueError(f"Unsafe attribute access: '{node.attr}'")
    _validate_condition_ast(node.value)
  elif isinstance(node, ast.Subscript):
    _validate_condition_ast(node.value)
    _validate_condition_ast(node.slice)
  elif isinstance(node, ast.Name):
    if node.id not in _SAFE_NAMES and node.id not in _SAFE_BUILTINS:
      raise ValueError(f"Unsafe name: '{node.id}'")
  elif isinstance(node, ast.Constant):
    pass  # string, int, float, bool, None literals are safe
  elif isinstance(node, (ast.List, ast.Tuple)):
    for elt in node.elts:
      _validate_condition_ast(elt)
  else:
    raise ValueError(f"Unsafe expression node: {type(node).__name__}")


def _parse_condition_string(condition_str: str) -> Callable[[GraphState], bool]:
  """Parse YAML condition string to a safe callable function.

  Conditions are parsed via AST and validated against a whitelist of
  safe operations before compilation. This prevents arbitrary code
  execution while supporting common condition expressions.

  Allowed in conditions:
  - Names: state, data, metadata, True, False, None
  - Methods: .get(), .get_parsed(), .get_str(), .get_dict()
  - Builtins: len, min, max, abs, bool, int, float, str, isinstance, type
  - Operators: ==, !=, <, >, <=, >=, is, is not, in, not in
  - Boolean: and, or, not
  - Literals: strings, numbers, booleans, None

  Examples:
      "data.get('approved') is True"
      "data.get('count', 0) < 10"
      "'CONTINUE' in data.get('status', '')"

  Args:
      condition_str: Python expression string to evaluate safely

  Returns:
      Callable that takes GraphState and returns bool

  Raises:
      ValueError: If condition contains unsafe expressions
  """
  # Parse and validate at definition time (fail fast)
  tree = ast.parse(condition_str, mode="eval")
  _validate_condition_ast(tree.body)
  code = compile(tree, "<condition>", "eval")

  def condition_func(state: GraphState) -> bool:
    import builtins as _builtins_mod

    safe_builtins = {
        name: getattr(_builtins_mod, name) for name in _SAFE_BUILTINS
    }
    namespace = {
        "state": state,
        "data": state.data,
    }
    try:
      result = eval(code, {"__builtins__": safe_builtins}, namespace)  # noqa: S307
      return bool(result)
    except Exception as e:
      logger.error(
          f"Condition evaluation failed: '{condition_str}' - {e}",
          exc_info=True,
      )
      return False

  return condition_func


# Sentinel constants for graph boundaries
START = "__start__"
END = "__end__"


@experimental
class GraphAgent(GraphInterruptMixin, GraphTelemetryMixin, BaseAgent):  # type: ignore[misc]
  """Graph-based workflow agent for ADK.

  GraphAgent is the fourth workflow agent type in ADK (alongside SequentialAgent,
  LoopAgent, and ParallelAgent), enabling directed graph-based orchestration with
  conditional routing, state management, and full checkpointing support.

  Workflow agents control execution flow through deterministic logic rather than LLM
  reasoning, providing predictable, reliable, and structured agent orchestration.

  Features:
  - Directed graph workflow with nodes (agents/functions) and edges
  - Conditional routing based on state predicates
  - Cyclic execution support (loops, iterative refinement, ReAct pattern)
  - Always-on observability: node lifecycle events emitted for every execution
  - Human-in-the-loop interrupts via InterruptService (retrospective feedback)
  - CheckpointService integration for state persistence
  - DatabaseSessionService support for persistence
  - Full ADK event system integration

  Example:
      >>> from google.adk.agents.graph import GraphAgent, GraphNode
      >>> from google.adk.agents import LlmAgent
      >>> from google.adk.checkpoints import CheckpointService, CheckpointCallback
      >>> from google.adk.runners import Runner
      >>>
      >>> # Selective node-level checkpointing (only critical nodes)
      >>> checkpoint_service = CheckpointService(session_service)
      >>> checkpoint_cb = CheckpointCallback(
      ...     checkpoint_service,
      ...     checkpoint_before=False,
      ...     checkpoint_after=True,
      ...     checkpoint_nodes={"analyze", "process"},  # only these nodes
      ... )
      >>>
      >>> graph = GraphAgent(
      ...     name="workflow",
      ...     after_node_callback=checkpoint_cb.after_node,
      ... )
      >>> graph.add_node(GraphNode(name="analyze", agent=LlmAgent(...)))
      >>> graph.add_node(GraphNode(name="process", agent=LlmAgent(...)))
      >>> graph.add_edge("analyze", "process")
      >>> graph.set_start("analyze")
      >>> graph.set_end("process")
      >>>
      >>> # Run with automatic checkpointing at critical nodes
      >>> runner = Runner(app_name="app", agent=graph)
      >>> async for event in runner.run_async(...):
      ...     print(event)
      >>>
      >>> # Legacy: checkpoint after EVERY node (all-or-nothing)
      >>> graph_legacy = GraphAgent(name="workflow", checkpointing=True)
  """

  model_config = ConfigDict(arbitrary_types_allowed=True)

  nodes: Dict[str, GraphNode] = Field(default_factory=dict)
  start_node: Optional[str] = None
  end_nodes: List[str] = Field(default_factory=list)
  max_iterations: int = 50  # Prevent infinite loops
  checkpointing: bool = False
  parallel_groups: Dict[str, Any] = Field(
      default_factory=dict,
      description="Parallel node groups for concurrent execution",
  )
  interrupt_service: Optional[InterruptService] = Field(
      default=None,
      description="Optional InterruptService for dynamic runtime interrupts",
  )
  interrupt_config: Optional[InterruptConfig] = Field(
      default=None,
      description="Configuration for interrupt timing and behavior",
  )
  telemetry_config: Optional[Any] = Field(
      default=None,
      description="Configuration for OpenTelemetry instrumentation",
  )
  before_node_callback: Optional[NodeCallback] = Field(
      default=None,
      description="Callback invoked before each node execution",
  )
  after_node_callback: Optional[NodeCallback] = Field(
      default=None,
      description="Callback invoked after each node execution",
  )
  on_edge_condition_callback: Optional[EdgeCallback] = Field(
      default=None,
      description="Callback invoked when evaluating edge conditions",
  )

  def __init__(
      self,
      name: str,
      description: str = "",
      max_iterations: int = 50,
      checkpointing: bool = False,
      interrupt_service: Optional[InterruptService] = None,
      interrupt_config: Optional[InterruptConfig] = None,
      telemetry_config: Optional[Any] = None,
      before_node_callback: Optional[NodeCallback] = None,
      after_node_callback: Optional[NodeCallback] = None,
      on_edge_condition_callback: Optional[EdgeCallback] = None,
      **kwargs: Any,
  ) -> None:
    """Initialize GraphAgent.

    Args:
        name: Agent name
        description: Agent description
        max_iterations: Max iterations to prevent infinite loops
        checkpointing: Enable state checkpointing after each node
            Note: For full checkpoint/resume, use CheckpointCallback
        interrupt_service: Optional InterruptService for dynamic runtime interrupts
        interrupt_config: Configuration for interrupt timing and behavior
        telemetry_config: Configuration for OpenTelemetry instrumentation
        before_node_callback: Callback invoked before each node execution
        after_node_callback: Callback invoked after each node execution
        on_edge_condition_callback: Callback invoked when evaluating edge conditions
    """
    super().__init__(name=name, description=description, **kwargs)
    self.nodes = {}
    self.start_node = None
    self.end_nodes = []
    self.max_iterations = max_iterations
    self.interrupt_service = interrupt_service
    self.interrupt_config = interrupt_config
    self.telemetry_config = telemetry_config
    self.before_node_callback = before_node_callback
    self.after_node_callback = after_node_callback
    self.on_edge_condition_callback = on_edge_condition_callback
    self.checkpointing = checkpointing
    # parallel_groups initialized by Field default_factory

  def add_node(
      self,
      node: GraphNode | str,
      agent: Optional[BaseAgent] = None,
      function: Optional[Callable[..., Any]] = None,
      **kwargs: Any,
  ) -> "GraphAgent":
    """Add a node to the graph.

    Supports two usage patterns:
    1. Pass a GraphNode directly
    2. Pass node name and agent/function (convenience method)

    Args:
        node: GraphNode to add, or string name (convenience)
        agent: Optional agent for convenience pattern
        function: Optional function for convenience pattern
        **kwargs: Additional GraphNode parameters (output_mapper, state_reducer, etc.)

    Returns:
        Self for chaining

    Examples:
        >>> # Pattern 1: GraphNode (explicit)
        >>> graph.add_node(GraphNode(name="validate", agent=validator))

        >>> # Pattern 2: Convenience (name + agent)
        >>> graph.add_node("validate", agent=validator)

        >>> # Pattern 3: Convenience with kwargs
        >>> graph.add_node("validate", agent=validator, state_reducer=StateReducer.OVERWRITE)
    """
    if isinstance(node, GraphNode):
      # Pattern 1: Direct GraphNode
      if agent is not None or function is not None or kwargs:
        raise ValueError(
            "When passing a GraphNode, do not specify agent, function, or"
            " kwargs"
        )
      if node.name in self.nodes:
        raise ValueError(f"Node '{node.name}' already exists in graph")
      self._validate_node_configuration(node)
      self.nodes[node.name] = node
    elif isinstance(node, str):
      # Pattern 2: Convenience (name + agent/function)
      if agent is None and function is None:
        raise ValueError(
            "When passing node name as string, must specify agent or function"
        )
      if agent is not None and function is not None:
        raise ValueError("Cannot specify both agent and function")

      if node in self.nodes:
        raise ValueError(f"Node '{node}' already exists in graph")

      graph_node = GraphNode(
          name=node, agent=agent, function=function, **kwargs
      )
      self._validate_node_configuration(graph_node)
      self.nodes[node] = graph_node
    else:
      raise TypeError(
          f"node must be GraphNode or str, got {type(node).__name__}"
      )

    # Register statically-known agents from the node in sub_agents
    graph_node = self.nodes[node.name if isinstance(node, GraphNode) else node]
    self._register_node_agents(graph_node)

    return self

  def _get_node_agent(self, node: "GraphNode") -> Optional[BaseAgent]:
    """Extract the primary agent from a graph node, including pattern nodes.

    Args:
        node: GraphNode (or pattern subclass) to inspect

    Returns:
        The agent associated with this node, or None
    """
    if node.agent is not None:
      return node.agent
    from .patterns import DynamicNode
    from .patterns import NestedGraphNode

    if isinstance(node, NestedGraphNode):
      return node.graph_agent
    if isinstance(node, DynamicNode):
      return node.fallback_agent
    return None

  def _register_node_agents(self, node: "GraphNode") -> None:
    """Register statically-known agents from a node into sub_agents.

    Handles regular agent nodes and pattern nodes (NestedGraphNode,
    DynamicNode). Skips DynamicParallelGroup (runtime-only agents).

    Args:
        node: GraphNode to extract agents from

    Raises:
        ValueError: If agent name collides with graph name, duplicates
            an existing sub_agent name, or agent already has a parent.
    """
    agent = self._get_node_agent(node)
    if agent is None:
      return

    # Identity check: same instance already registered (e.g., shared across nodes)
    if any(agent is sa for sa in self.sub_agents):
      return

    # Reject agent name that shadows the graph itself — find_agent() would
    # return the graph instead of the sub-agent, causing silent bugs.
    if agent.name == self.name:
      raise ValueError(
          f"Node agent name '{agent.name}' collides with GraphAgent name."
          " Rename the agent to avoid find_agent() ambiguity."
      )

    # Validate unique name among existing sub_agents
    for sa in self.sub_agents:
      if sa.name == agent.name:
        raise ValueError(
            f"Duplicate sub_agent name '{agent.name}'. Another node already"
            " registered an agent with this name."
        )

    # Single-parent constraint (matches BaseAgent behavior)
    existing_parent = getattr(agent, "parent_agent", None)
    if existing_parent is not None:
      raise ValueError(
          f"Agent '{agent.name}' already has a parent agent"
          f" '{existing_parent.name}', cannot add to '{self.name}'"
      )

    agent.parent_agent = self
    self.sub_agents.append(agent)

  @override
  def find_sub_agent(self, name: str) -> Optional[BaseAgent]:
    """Find agent by name, searching sub_agents then graph nodes as fallback.

    Overrides BaseAgent.find_sub_agent to also search graph node agents
    that may not be in sub_agents (e.g., agents added to nodes before
    registration). NestedGraphNode agents are recursively searched.

    Note: DynamicNode runtime-selected agents (chosen by agent_selector
    at execution time) are NOT discoverable via find_sub_agent because
    they don't exist until the graph runs. Only DynamicNode.fallback_agent
    (if set) is registered and searchable.

    Args:
        name: The agent name to find

    Returns:
        The matching agent, or None
    """
    # Standard sub_agents search first
    for sub_agent in self.sub_agents:
      if result := sub_agent.find_agent(name):
        return result
    # Fallback: search graph nodes for agents not in sub_agents
    for node in self.nodes.values():
      agent = self._get_node_agent(node)
      if agent is not None:
        if agent.name == name:
          return agent
        if result := agent.find_agent(name):
          return result
    return None

  def _validate_node_configuration(self, node: "GraphNode") -> None:
    """Validate node configuration before adding to graph.

    Emits warnings for potential misconfiguration issues.

    Args:
        node: GraphNode to validate
    """
    # Warn if output_schema present but was auto-defaulted
    if isinstance(node.agent, LlmAgent):
      if node.agent.output_schema and node.agent.output_key:
        # Check if it looks like it was auto-defaulted (matches agent name)
        if node.agent.output_key == node.agent.name:
          logger.warning(
              f"Node '{node.name}': Using auto-defaulted"
              f" output_key='{node.agent.output_key}'. To silence this warning,"
              " explicitly set output_key on the LlmAgent."
          )

  def set_start(self, node_name: str) -> "GraphAgent":
    """Set the starting node.

    Args:
        node_name: Name of the start node

    Returns:
        Self for chaining

    Raises:
        ValueError: If node not found in graph
    """
    if node_name not in self.nodes:
      raise ValueError(f"Node {node_name} not found in graph")
    self.start_node = node_name
    return self

  def set_end(self, node_name: str) -> "GraphAgent":
    """Mark a node as an end node.

    Args:
        node_name: Name of the end node

    Returns:
        Self for chaining

    Raises:
        ValueError: If node not found in graph
    """
    if node_name not in self.nodes:
      raise ValueError(f"Node {node_name} not found in graph")
    if node_name not in self.end_nodes:
      self.end_nodes.append(node_name)
    return self

  def add_edge(
      self,
      source_node: str,
      target_node: str | EdgeCondition,
      condition: Optional[Callable[[GraphState], bool]] = None,
      priority: Optional[int] = None,
      weight: Optional[float] = None,
  ) -> "GraphAgent":
    """Add an edge from source node to target node.

    Supports two usage patterns:
    1. Pass EdgeCondition as target_node (explicit)
    2. Pass target node name with optional params (convenience)

    Advanced routing features:
    - Priority-based routing (higher priority evaluated first)
    - Weighted random selection (probabilistic routing)
    - Fallback edges (priority=0 always matches)

    Args:
        source_node: Source node name
        target_node: Target node name OR EdgeCondition object
        condition: Optional condition (ignored if target_node is EdgeCondition)
        priority: Optional priority (ignored if target_node is EdgeCondition)
        weight: Optional weight (ignored if target_node is EdgeCondition)

    Returns:
        Self for chaining

    Raises:
        ValueError: If nodes not found in graph
        TypeError: If target_node is not str or EdgeCondition

    Examples:
        >>> # Pattern 1: EdgeCondition (explicit)
        >>> graph.add_edge("validate", EdgeCondition(
        ...     target_node="process",
        ...     condition=lambda s: s.data.get("valid"),
        ...     priority=10
        ... ))

        >>> # Pattern 2: Convenience - simple edge
        >>> graph.add_edge("validate", "process")

        >>> # Pattern 2: Convenience - conditional edge
        >>> graph.add_edge("validate", "process", condition=lambda s: s.data.get("valid"))

        >>> # Pattern 2: Convenience - priority-based routing
        >>> graph.add_edge("check", "critical", condition=lambda s: s.data["score"] > 0.9, priority=10)
        >>> graph.add_edge("check", "normal", priority=0)  # Fallback

        >>> # Pattern 2: Convenience - weighted random routing
        >>> graph.add_edge("start", "server_a", condition=lambda s: True, priority=1, weight=0.5)
        >>> graph.add_edge("start", "server_b", condition=lambda s: True, priority=1, weight=0.3)
    """
    if source_node not in self.nodes:
      raise ValueError(f"Source node {source_node} not found")

    if isinstance(target_node, EdgeCondition):
      # Pattern 1: EdgeCondition
      if condition is not None or priority is not None or weight is not None:
        raise ValueError(
            "When passing EdgeCondition, do not specify condition, priority, or"
            " weight"
        )
      if target_node.target_node not in self.nodes:
        raise ValueError(f"Target node {target_node.target_node} not found")

      # Check for duplicate edge
      if (
          hasattr(self.nodes[source_node], "edges")
          and self.nodes[source_node].edges is not None
      ):
        for existing_edge in self.nodes[source_node].edges:
          if existing_edge.target_node == target_node.target_node:
            raise ValueError(
                f"Edge from '{source_node}' to '{target_node.target_node}'"
                " already exists. Cannot add duplicate edge."
            )

      self.nodes[source_node].edges.append(target_node)
      self.nodes[source_node]._sorted_edges_cache = None

    elif isinstance(target_node, str):
      # Pattern 2: Convenience
      if target_node not in self.nodes:
        raise ValueError(f"Target node {target_node} not found")

      # Check for duplicate edge
      if (
          hasattr(self.nodes[source_node], "edges")
          and self.nodes[source_node].edges is not None
      ):
        for existing_edge in self.nodes[source_node].edges:
          if existing_edge.target_node == target_node:
            raise ValueError(
                f"Edge from '{source_node}' to '{target_node}' already exists."
                " Cannot add duplicate edge."
            )

      # If priority or weight specified, create EdgeCondition
      if priority is not None or weight is not None:
        edge_condition = EdgeCondition(
            target_node=target_node,
            condition=condition,
            priority=priority if priority is not None else 1,
            weight=weight if weight is not None else 1.0,
        )
        self.nodes[source_node].edges.append(edge_condition)
        self.nodes[source_node]._sorted_edges_cache = None
      else:
        # Simple edge (no priority/weight)
        self.nodes[source_node].add_edge(target_node, condition)
    else:
      raise TypeError(
          "target_node must be str or EdgeCondition, got"
          f" {type(target_node).__name__}"
      )

    return self

  def add_parallel_group(
      self,
      group_id: str,
      group: "ParallelNodeGroup",
  ) -> "GraphAgent":
    """Add a parallel node group for concurrent execution.

    Args:
        group_id: Unique identifier for the group
        group: ParallelNodeGroup configuration

    Returns:
        Self for chaining

    Raises:
        ValueError: If nodes in group not found

    Example:
        >>> from google.adk.agents.graph import ParallelNodeGroup, JoinStrategy
        >>> graph.add_parallel_group(
        ...     "fetch_group",
        ...     ParallelNodeGroup(
        ...         nodes=["fetch_user", "fetch_products"],
        ...         join_strategy=JoinStrategy.WAIT_ALL
        ...     )
        ... )
    """
    # Validate all nodes exist
    for node_name in group.nodes:
      if node_name not in self.nodes:
        raise ValueError(f"Node {node_name} not found in graph")

    self.parallel_groups[group_id] = group
    return self

  def _find_parallel_group(self, node_name: str) -> Optional[Tuple[str, Any]]:
    """Find if a node is part of a parallel group.

    Args:
        node_name: Node name to check

    Returns:
        Tuple of (group_id, ParallelNodeGroup) if found, None otherwise
    """
    for group_id, group in self.parallel_groups.items():
      if node_name in group.nodes:
        return (group_id, group)
    return None

  # Export methods moved to graph_export.py
  # rewind_to_node moved to graph_rewind.py

  # Telemetry methods inherited from GraphTelemetryMixin

  async def _execute_node(
      self,
      node: GraphNode,
      state: GraphState,
      ctx: InvocationContext,
      effective_config: Optional[TelemetryConfig] = None,
      output_holder: Optional[Dict[str, Any]] = None,
      iteration: int = 0,
  ) -> AsyncGenerator[Event, None]:
    """Execute a single node and yield events.

    Output is stored in output_holder["output"] for the caller.

    Args:
        node: GraphNode to execute
        state: Current graph state
        ctx: Invocation context
        effective_config: Effective telemetry config (merged parent + own)
        output_holder: Mutable dict to store node output
        iteration: Current iteration number

    Yields:
        Events from node execution
    """
    # Determine node type
    node_type = "function" if node.function else "agent"
    start_time = time.time()

    # Create telemetry span for node execution
    with graph_tracing.tracer.start_as_current_span(
        f"graph_node {node.name}"
    ) as span:
      # Add attributes with additional_attributes support
      attrs = self._get_telemetry_attributes(
          {
              graph_tracing.GRAPH_NODE_NAME: node.name,
              graph_tracing.GRAPH_NODE_TYPE: node_type,
              graph_tracing.GRAPH_NODE_ITERATION: iteration,
              graph_tracing.GRAPH_AGENT_NAME: self.name,
          },
          effective_config=effective_config,
      )
      for key, value in attrs.items():
        span.set_attribute(key, value)

      try:
        # Map state to node input with telemetry
        mapper_start = time.time()
        node_input = node.input_mapper(state)
        mapper_latency_ms = (time.time() - mapper_start) * 1000

        # Record input mapper telemetry (check sampling)
        if self._should_sample(effective_config=effective_config):
          is_default_mapper = (
              node.input_mapper.__name__ == "_default_input_mapper"
          )
          graph_tracing.record_mapper(
              node_name=node.name,
              mapper_type="input",
              agent_name=self.name,
              latency_ms=mapper_latency_ms,
              is_default=is_default_mapper,
          )

        # Execute node (agent or function)
        output = ""
        if node.agent:
          # Create new context with updated user_content for this node
          node_content = types.Content(
              role="user", parts=[types.Part(text=node_input)]
          )
          node_ctx = ctx.model_copy(update={"user_content": node_content})

          # Execute ADK agent with updated context
          # (BaseAgent will create invoke_agent span automatically)
          async for event in node.agent.run_async(node_ctx):
            # Extract output from final response
            if event.content and event.content.parts:
              output = event.content.parts[0].text or ""
            yield event
            # ADK resumability: pause when long-running tool detected.
            # Function nodes don't set "pause" (they run synchronously,
            # no tool calls) so output_holder.get("pause") is always
            # falsy for them — safe to check unconditionally in caller.
            if ctx.should_pause_invocation(event):
              if output_holder is not None:
                output_holder["output"] = output
                output_holder["pause"] = True
              return
        elif node.function:
          # Execute custom function (CRITICAL: no automatic span)
          if asyncio.iscoroutinefunction(node.function):
            output = await node.function(state, ctx)
          else:
            output = node.function(state, ctx)
        else:  # pragma: no cover
          # Defensive: This should never happen due to GraphNode validation
          raise ValueError(f"Node {node.name} has no agent or function")

        # Store output for caller retrieval
        if output_holder is not None:
          output_holder["output"] = output

        # Record success metrics (check sampling)
        latency_ms = (time.time() - start_time) * 1000
        span.set_attribute("graph.node.success", True)
        if self._should_sample(effective_config=effective_config):
          graph_tracing.record_node_execution(
              node_name=node.name,
              node_type=node_type,
              agent_name=self.name,
              latency_ms=latency_ms,
              success=True,
          )

      except Exception as e:
        # Record failure metrics (check sampling)
        latency_ms = (time.time() - start_time) * 1000
        span.set_attribute("graph.node.success", False)
        span.set_attribute("graph.node.error", str(e))
        if self._should_sample(effective_config=effective_config):
          graph_tracing.record_node_execution(
              node_name=node.name,
              node_type=node_type,
              agent_name=self.name,
              latency_ms=latency_ms,
              success=False,
          )
        raise

  # _check_interrupt_with_telemetry inherited from GraphInterruptMixin

  def _get_next_node_with_telemetry(
      self,
      current_node: GraphNode,
      state: GraphState,
      effective_config: Optional[TelemetryConfig] = None,
  ) -> Optional[str]:
    """Get next node with edge evaluation telemetry.

    Args:
        current_node: Current graph node
        state: Current graph state
        effective_config: Effective telemetry config (merged parent + own)

    Returns:
        Name of next node, or None if no edge matches
    """
    # Track all condition results for detailed telemetry
    condition_results = []

    # Evaluate each edge with telemetry
    for edge in current_node.edges:
      start_time = time.time()

      # Create span for edge evaluation
      with graph_tracing.tracer.start_as_current_span(
          f"edge_condition {edge.target_node}"
      ) as span:
        # Add attributes with additional_attributes support
        attrs = self._get_telemetry_attributes(
            {
                graph_tracing.GRAPH_EDGE_SOURCE: current_node.name,
                graph_tracing.GRAPH_EDGE_TARGET: edge.target_node,
                graph_tracing.GRAPH_EDGE_PRIORITY: edge.priority,
                graph_tracing.GRAPH_AGENT_NAME: self.name,
            },
            effective_config=effective_config,
        )
        for key, value in attrs.items():
          span.set_attribute(key, value)

        try:
          # Evaluate condition
          result = edge.should_route(state)
          span.set_attribute(
              graph_tracing.GRAPH_EDGE_CONDITION_RESULT, str(result)
          )

          # Track condition result details for debugging
          condition_results.append({
              "target_node": edge.target_node,
              "condition_matched": result,
              "condition_name": getattr(edge.condition, "__name__", "<lambda>"),
              "priority": edge.priority,
          })

          # Record metrics (check sampling)
          if self._should_sample(effective_config=effective_config):
            latency_ms = (time.time() - start_time) * 1000
            graph_tracing.record_edge_evaluation(
                source_node=current_node.name,
                target_node=edge.target_node,
                agent_name=self.name,
                condition_result=result,
                latency_ms=latency_ms,
                priority=edge.priority,
            )

        except Exception as e:
          span.set_attribute("graph.edge.error", str(e))
          span.set_attribute(graph_tracing.GRAPH_EDGE_CONDITION_RESULT, "false")
          raise

    # Add detailed condition results to GraphState for debugging
    # This helps identify routing issues by showing ALL edge evaluations
    if condition_results:
      state.data["_debug_edge_evaluations"] = {
          "source_node": current_node.name,
          "evaluations": condition_results,
          "timestamp": time.time(),
      }

    # Use original get_next_node for routing logic
    selected_node = current_node.get_next_node(state)

    # Log final node selection decision with all context
    if selected_node:
      # Find which edge was selected (if any)
      selected_edge_info = next(
          (
              r
              for r in condition_results
              if r["target_node"] == selected_node and r["condition_matched"]
          ),
          None,
      )

      # Add node selection to debug info
      state.data.setdefault("_debug_node_selections", []).append({
          "from_node": current_node.name,
          "to_node": selected_node,
          "selected_edge": selected_edge_info,
          "num_edges_evaluated": len(condition_results),
          "timestamp": time.time(),
      })

      # Log structured selection event
      graph_tracing.logger.debug(
          f"Node selected: {current_node.name} -> {selected_node}",
          extra={
              "source_node": current_node.name,
              "selected_node": selected_node,
              "condition_name": (
                  selected_edge_info["condition_name"]
                  if selected_edge_info
                  else None
              ),
              "priority": (
                  selected_edge_info["priority"] if selected_edge_info else None
              ),
              "edges_evaluated": len(condition_results),
              "agent_name": self.name,
          },
      )

    return selected_node

  def _get_resume_state(
      self, agent_state: GraphAgentState
  ) -> Tuple[Optional[str], int, bool]:
    """Get resume point from loaded agent state.

    Mirrors SequentialAgent._get_start_index() pattern.

    Args:
        agent_state: Loaded agent state (may have current_node from prior run)

    Returns:
        Tuple of (start_node_name, start_iteration, is_resuming)
    """
    if agent_state.current_node and agent_state.current_node in self.nodes:
      return agent_state.current_node, agent_state.iteration, True
    if agent_state.current_node and agent_state.current_node not in self.nodes:
      logger.warning(
          "Saved node '%s' no longer exists in graph. Restarting from '%s'.",
          agent_state.current_node,
          self.start_node,
      )
    return self.start_node, 0, False

  async def _execute_callback(
      self,
      callback: Callable[..., Any],
      callback_type: str,
      current_node: GraphNode,
      current_node_name: str,
      state: GraphState,
      iteration: int,
      ctx: InvocationContext,
      agent_state: GraphAgentState,
      effective_config: Optional["TelemetryConfig"] = None,
      output: str = "",
  ) -> Optional[Event]:
    """Execute a node callback (before_node or after_node) with telemetry.

    Args:
        callback: The callback function to execute
        callback_type: "before_node" or "after_node"
        current_node: The current GraphNode
        current_node_name: Name of the current node
        state: Current graph state
        iteration: Current iteration number
        ctx: Invocation context
        agent_state: Execution tracking state
        effective_config: Effective telemetry config
        output: Node output (only for after_node callbacks)

    Returns:
        Event from callback, or None
    """
    from .callbacks import NodeCallbackContext

    metadata: Dict[str, Any] = {
        "agent_path": list(agent_state.agent_path),
        "path": list(agent_state.path),
    }
    if callback_type == "after_node":
      metadata["output"] = output

    callback_ctx = NodeCallbackContext(
        node=current_node,
        state=state,
        iteration=iteration,
        invocation_context=ctx,
        metadata=metadata,
    )

    callback_start_time = time.time()
    with graph_tracing.tracer.start_as_current_span(
        f"graph_callback {callback_type}"
    ) as cb_span:
      attrs = self._get_telemetry_attributes(
          {
              graph_tracing.GRAPH_CALLBACK_TYPE: callback_type,
              graph_tracing.GRAPH_AGENT_NAME: self.name,
              graph_tracing.GRAPH_NODE_NAME: current_node_name,
          },
          effective_config=effective_config,
      )
      for key, value in attrs.items():
        cb_span.set_attribute(key, value)

      try:
        event = await callback(callback_ctx)
        cb_span.set_attribute("graph.callback.success", True)
        if self._should_sample(effective_config=effective_config):
          callback_latency_ms = (time.time() - callback_start_time) * 1000
          graph_tracing.record_callback_execution(
              callback_type=callback_type,
              agent_name=self.name,
              latency_ms=callback_latency_ms,
              success=True,
          )
        return event

      except Exception as e:
        cb_span.set_attribute("graph.callback.success", False)
        cb_span.set_attribute("graph.callback.error", str(e))
        if self._should_sample(effective_config=effective_config):
          callback_latency_ms = (time.time() - callback_start_time) * 1000
          graph_tracing.record_callback_execution(
              callback_type=callback_type,
              agent_name=self.name,
              latency_ms=callback_latency_ms,
              success=False,
          )
        logger.error(
            "%s_callback failed for node '%s': %s",
            callback_type,
            current_node_name,
            e,
            exc_info=True,
        )
        return None

  def _sync_state_and_reduce(
      self,
      current_node: GraphNode,
      current_node_name: str,
      state: GraphState,
      ctx: InvocationContext,
      output: str,
      effective_config: Optional["TelemetryConfig"] = None,
      agent_state: Optional[GraphAgentState] = None,
  ) -> GraphState:
    """Sync session state into GraphState and apply output_mapper + reducer.

    Args:
        current_node: The current GraphNode
        current_node_name: Name of the current node
        state: Current graph state (mutated in-place for session sync)
        ctx: Invocation context
        output: Node output string
        effective_config: Effective telemetry config
        agent_state: Execution tracking state for output key tracking

    Returns:
        Updated GraphState after sync and reduction
    """
    # Sync session state into GraphState.data
    for _sk, _sv in ctx.session.state.items():
      if not _sk.startswith("_") and _sk not in _GRAPH_INTERNAL_KEYS:
        state.data[_sk] = _sv

    # Apply output mapper with reducer
    if output:
      had_previous_value = current_node.name in state.data
      reducer_start = time.time()

      # Snapshot keys before output_mapper to track what gets written
      keys_before = set(state.data.keys())

      prev_state = state
      state = current_node.output_mapper(output, state)
      if state is None:
        state = prev_state

      # Track which keys were written by this node's output_mapper
      if agent_state is not None:
        keys_after = set(state.data.keys())
        written_keys = list(keys_after - keys_before)
        # Also include the node name key if it was overwritten
        if (
            current_node_name in state.data
            and current_node_name not in written_keys
        ):
          written_keys.append(current_node_name)
        agent_state.output_keys[current_node_name] = written_keys

      reducer_latency_ms = (time.time() - reducer_start) * 1000
      if self._should_sample(effective_config=effective_config):
        graph_tracing.record_state_reducer(
            node_name=current_node.name,
            reducer_type=current_node.reducer.name,
            state_key=current_node.name,
            agent_name=self.name,
            latency_ms=reducer_latency_ms,
            had_previous_value=had_previous_value,
        )
        is_default_mapper = (
            current_node.output_mapper.__name__ == "_default_output_mapper"
        )
        graph_tracing.record_mapper(
            node_name=current_node.name,
            mapper_type="output",
            agent_name=self.name,
            latency_ms=reducer_latency_ms,
            is_default=is_default_mapper,
        )

    return state

  def _build_cancellation_events(
      self,
      ctx: InvocationContext,
      agent_state: GraphAgentState,
      current_node_name: str,
      state: GraphState,
      *,
      state_key: str = "graph_cancelled",
      message: str,
      iteration: Optional[int] = None,
      partial_output: Optional[str] = None,
      path: Optional[List[str]] = None,
  ) -> List[Event]:
    """Build agent-state + cancellation events for graph abort scenarios.

    Consolidates the repeated pattern of saving agent state then yielding
    a cancellation event with appropriate state_delta keys.

    Args:
        ctx: Invocation context
        agent_state: Execution tracking state (saved before cancel)
        current_node_name: Node where cancellation occurred
        state: Current graph state
        state_key: Key for the cancellation flag (e.g. "graph_cancelled",
            "graph_task_cancelled")
        message: Human-readable cancellation message
        iteration: Current iteration (included in state_delta when set)
        partial_output: Partial node output (included when set)
        path: Execution path (included in state_delta when set)

    Returns:
        List of two events: [agent_state_event, cancellation_event]
    """
    ctx.set_agent_state(self.name, agent_state=agent_state)
    state_event = self._create_agent_state_event(ctx)

    state_delta: Dict[str, Any] = {
        state_key: True,
        "graph_cancelled_at_node": current_node_name,
        "graph_data": state.data,
        "graph_can_resume": True,
    }
    if iteration is not None:
      state_delta["graph_iteration"] = iteration
    if partial_output is not None:
      state_delta["graph_partial_output"] = partial_output
    if path is not None:
      state_delta["graph_path"] = path

    cancel_event = Event(
        author=self.name,
        content=types.Content(
            parts=[types.Part(text=f"\u26a0\ufe0f {message}")]
        ),
        actions=EventActions(
            escalate=False,
            state_delta=state_delta,
        ),
    )
    return [state_event, cancel_event]

  async def _execute_parallel_phase(
      self,
      group_id: str,
      parallel_group: "ParallelNodeGroup",
      current_node: GraphNode,
      current_node_name: str,
      state: GraphState,
      ctx: InvocationContext,
      effective_config: Optional["TelemetryConfig"],
      agent_state: GraphAgentState,
      executed_parallel_groups: set[str],
      result: Dict[str, Any],
  ) -> AsyncGenerator[Event, None]:
    """Execute a parallel node group phase with telemetry.

    Handles already-executed check, telemetry instrumentation,
    parallel execution, group marking, and next-node routing.
    Sets result["next"] to the next node name (or None if at end node).

    Args:
        group_id: Parallel group identifier
        parallel_group: ParallelNodeGroup configuration
        current_node: Current GraphNode (for edge routing)
        current_node_name: Name of current node
        state: Current graph state
        ctx: Invocation context
        effective_config: Telemetry config
        agent_state: Execution tracking state
        executed_parallel_groups: Set of already-executed group IDs (mutated)
        result: Mutable dict; sets result["next"] to next node name or None

    Yields:
        Events from parallel execution

    Raises:
        ValueError: If parallel group has no outgoing edges and node is not
            an end node
    """
    # Check if this group has already been executed
    if group_id in executed_parallel_groups:
      logger.info(
          f"Skipping node '{current_node_name}' - already executed as"
          f" part of parallel group '{group_id}'"
      )
      next_node_name = self._get_next_node_with_telemetry(
          current_node, state, effective_config=effective_config
      )
      if next_node_name is None:
        if current_node_name in self.end_nodes:
          result["next"] = None
          return
        else:
          raise ValueError(
              f"Node {current_node_name} has no outgoing edges and is"
              " not an end node"
          )
      result["next"] = next_node_name
      return

    # Execute entire parallel group
    logger.info(
        f"Executing parallel group '{group_id}' with nodes:"
        f" {parallel_group.nodes}"
    )

    parallel_start_time = time.time()
    with graph_tracing.tracer.start_as_current_span(
        f"parallel_group {group_id}"
    ) as pg_span:
      attrs = self._get_telemetry_attributes(
          {
              graph_tracing.GRAPH_PARALLEL_NODE_COUNT: len(
                  parallel_group.nodes
              ),
              graph_tracing.GRAPH_PARALLEL_STRATEGY: (
                  parallel_group.join_strategy.value
              ),
              graph_tracing.GRAPH_PARALLEL_WAIT_N: parallel_group.wait_n,
              graph_tracing.GRAPH_AGENT_NAME: self.name,
          },
          effective_config=effective_config,
      )
      for key, value in attrs.items():
        pg_span.set_attribute(key, value)

      completed_count = 0
      async for event in execute_parallel_group(
          parallel_group,
          self.nodes,
          state,
          ctx,
          self._execute_node,
      ):
        yield event
        if event.author != self.name:
          completed_count = min(completed_count + 1, len(parallel_group.nodes))

      pg_span.set_attribute("graph.parallel.completed_count", completed_count)
      if self._should_sample(effective_config=effective_config):
        parallel_latency_ms = (time.time() - parallel_start_time) * 1000
        graph_tracing.record_parallel_group_execution(
            agent_name=self.name,
            node_count=len(parallel_group.nodes),
            strategy=parallel_group.join_strategy.value,
            latency_ms=parallel_latency_ms,
            completed_count=completed_count,
        )

    # Mark group as executed
    executed_parallel_groups.add(group_id)
    agent_state.executed_parallel_groups = list(executed_parallel_groups)

    # Route to next node
    next_node_name = self._get_next_node_with_telemetry(
        current_node, state, effective_config=effective_config
    )
    if next_node_name is None:
      if current_node_name in self.end_nodes:
        result["next"] = None
        return
      else:
        raise ValueError(
            f"Parallel group '{group_id}' has no outgoing edges and"
            f" node '{current_node_name}' is not an end node"
        )
    result["next"] = next_node_name

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    """Core graph execution logic.

    Executes nodes in graph order, following conditional edges,
    supporting loops and human-in-the-loop interrupts.

    Args:
        ctx: Invocation context

    Yields:
        Events from graph execution

    Raises:
        ValueError: If start node not set or graph structure invalid
    """
    if not self.start_node:
      raise ValueError("Start node not set. Call set_start() first.")

    # Register session with InterruptService if enabled
    if self.interrupt_service:
      self.interrupt_service.register_session(ctx.session.id)

    # Get effective telemetry config for nested graph inheritance
    effective_config = self._get_effective_telemetry_config(ctx)

    with tracer.start_as_current_span(
        f"graph_agent_execution {self.name}"
    ) as span:
      span.set_attribute("graph_agent.name", self.name)
      span.set_attribute("graph_agent.start_node", self.start_node)
      span.set_attribute("graph_agent.max_iterations", self.max_iterations)
      try:
        # Load execution tracking state (BaseAgentState pattern)
        agent_state = (
            self._load_agent_state(ctx, GraphAgentState) or GraphAgentState()
        )

        # Store telemetry config for nested graph inheritance
        if effective_config:
          agent_state.telemetry_config_dict = effective_config.model_dump()

        # Initialize domain data from session state or user input.
        # Exclude graph-internal keys to prevent circular references
        # (state.data["graph_data"] → state.data) and keep domain data clean.
        domain_data = {
            k: v
            for k, v in ctx.session.state.items()
            if not k.startswith("_") and k not in _GRAPH_INTERNAL_KEYS
        }
        if domain_data:
          state = GraphState(data=domain_data)
        else:
          # Extract text from Content object
          user_text = ""
          if (
              hasattr(ctx, "user_content")
              and ctx.user_content
              and ctx.user_content.parts
          ):
            user_text = (
                ctx.user_content.parts[0].text
                if ctx.user_content.parts[0].text
                else ""
            )
          state = GraphState(data={"input": user_text})

        # Track which parallel groups have been executed
        executed_parallel_groups = set(agent_state.executed_parallel_groups)

        # ADK resumability: resume from saved node or start fresh.
        #
        # Design note: SequentialAgent ONLY emits state events when
        # ctx.is_resumable is True, because its state events serve only
        # resumability. GraphAgent's state events serve multiple consumers
        # (rewind, interrupts, telemetry) that are orthogonal to
        # resumability. Therefore:
        #   - Per-iteration state events: always emitted (multi-consumer)
        #   - Resume skip: first iteration skipped when resuming (already
        #     persisted before pause, avoids duplicate)
        #   - end_of_agent: guarded by is_resumable (purely a resumability
        #     lifecycle signal, has no other consumers)
        #   - Interrupt/cancellation state saves: always emitted (they
        #     serve interrupt functionality, not just resumability)
        current_node_name, iteration, resuming = self._get_resume_state(
            agent_state
        )
        pause_invocation = False

        while current_node_name and iteration < self.max_iterations:
          iteration += 1
          current_node = self.nodes[current_node_name]

          # Check for immediate cancellation (ESC-like interrupt)
          # Allows user to abort execution at any time, not just at pause points
          if self.interrupt_service and not self.interrupt_service.is_active(
              ctx.session.id
          ):
            logger.info(
                "GraphAgent execution cancelled (immediate interrupt) for"
                f" session {ctx.session.id}"
            )
            for _ce in self._build_cancellation_events(
                ctx,
                agent_state,
                current_node_name,
                state,
                message="Execution cancelled by user",
                iteration=iteration,
                path=list(agent_state.path),
            ):
              yield _ce
            break  # Exit immediately but state is saved

          # Track execution path in agent_state
          agent_state.path.append(current_node_name)
          agent_state.iteration = iteration
          agent_state.current_node = current_node_name
          agent_state.node_invocations.setdefault(current_node_name, []).append(
              ctx.invocation_id
          )

          # ADK resumability: reset sub-agent states on cycle revisit
          # (mirrors LoopAgent pattern at loop_agent.py:114)
          # O(1) lookup via node_invocations instead of O(N) path.count()
          if len(agent_state.node_invocations.get(current_node_name, [])) > 1:
            if current_node.agent:
              ctx.reset_sub_agent_states(current_node.agent.name)
            # Reset parallel group tracking so groups can re-execute
            # in cyclic workflows
            executed_parallel_groups.clear()
            agent_state.executed_parallel_groups = []

          # Track agent path for nested graph support
          if self.name not in agent_state.agent_path:
            agent_state.agent_path.append(self.name)

          # Persist execution tracking via agent_state event.
          # These events are consumed by rewind, interrupts, and telemetry
          # (not just resumability), so they're always emitted.
          # Skip only on first iteration when resuming (already persisted).
          if not resuming:
            ctx.set_agent_state(self.name, agent_state=agent_state)
            yield self._create_agent_state_event(ctx)
          else:
            resuming = False  # Only skip first iteration after resume

          # Handle BEFORE-node interrupt (validation timing)
          if (
              self._should_interrupt_before(current_node_name)
              and self.interrupt_service
          ):
            _b_events, _b_ctrl = await self._handle_before_node_interrupt(
                current_node_name, current_node, state, ctx, agent_state
            )
            for _e in _b_events:
              yield _e
            # Persist agent_state after interrupt handler may have mutated it
            ctx.set_agent_state(self.name, agent_state=agent_state)
            yield self._create_agent_state_event(ctx)
            if _b_ctrl == "break":
              break
            elif _b_ctrl is not None:
              if isinstance(_b_ctrl, tuple):
                current_node_name = _b_ctrl[1]
              continue

          # Check if current node is part of a parallel group
          parallel_group_info = self._find_parallel_group(current_node_name)
          if parallel_group_info:
            group_id, parallel_group = parallel_group_info
            _pg_result: Dict[str, Any] = {}
            async for event in self._execute_parallel_phase(
                group_id,
                parallel_group,
                current_node,
                current_node_name,
                state,
                ctx,
                effective_config,
                agent_state,
                executed_parallel_groups,
                _pg_result,
            ):
              yield event
            # Fire after_node_callback for parallel trigger node
            if self.after_node_callback:
              event = await self._execute_callback(
                  self.after_node_callback,
                  "after_node",
                  current_node,
                  current_node_name,
                  state,
                  iteration,
                  ctx,
                  agent_state,
                  effective_config,
              )
              if event:
                yield event
            current_node_name = _pg_result.get("next")
            if current_node_name is None:
              break
            continue

          # Invoke before_node_callback (custom observability)
          if self.before_node_callback:
            event = await self._execute_callback(
                self.before_node_callback,
                "before_node",
                current_node,
                current_node_name,
                state,
                iteration,
                ctx,
                agent_state,
                effective_config,
            )
            if event:
              yield event

          # Execute node with immediate cancellation support
          # Check cancellation while streaming events from node execution
          output_holder: Dict[str, Any] = {"output": ""}
          try:
            async for event in self._execute_node(
                current_node,
                state,
                ctx,
                effective_config,
                output_holder=output_holder,
                iteration=iteration,
            ):
              # Check for immediate cancellation DURING node execution
              if (
                  self.interrupt_service
                  and not self.interrupt_service.is_active(ctx.session.id)
              ):
                logger.info(
                    "GraphAgent execution cancelled (immediate interrupt"
                    f" during node '{current_node_name}') for session"
                    f" {ctx.session.id}"
                )
                for _ce in self._build_cancellation_events(
                    ctx,
                    agent_state,
                    current_node_name,
                    state,
                    message=(
                        f"Execution cancelled during node '{current_node_name}'"
                    ),
                    partial_output=output_holder["output"],
                ):
                  yield _ce
                return
              yield event
          except asyncio.CancelledError:
            # Task cancelled externally (e.g., timeout, user abort)
            logger.info(
                f"GraphAgent task cancelled during node '{current_node_name}'"
                f" for session {ctx.session.id}"
            )
            for _ce in self._build_cancellation_events(
                ctx,
                agent_state,
                current_node_name,
                state,
                state_key="graph_task_cancelled",
                message=f"Task cancelled during node '{current_node_name}'",
                partial_output=output_holder["output"],
            ):
              yield _ce
            raise

          # ADK resumability: check if node execution was paused
          if output_holder.get("pause"):
            pause_invocation = True
            return

          # Sync session state + apply output_mapper/reducer
          output = output_holder["output"]
          state = self._sync_state_and_reduce(
              current_node,
              current_node_name,
              state,
              ctx,
              output,
              effective_config,
              agent_state=agent_state,
          )

          # Emit output_mapper changes as state_delta so domain data
          # flows through ADK's event pipeline to session.state.
          # This enables downstream LlmAgent nodes to read
          # output_mapper results via dynamic instructions.
          if output:
            delta = {}
            for _k, _v in state.data.items():
              if (
                  not _k.startswith("_")
                  and _k not in _GRAPH_INTERNAL_KEYS
                  and ctx.session.state.get(_k) != _v
              ):
                delta[_k] = _v
            if delta:
              yield Event(
                  author=self.name,
                  actions=EventActions(state_delta=delta),
              )

          # Invoke after_node_callback (custom observability)
          if self.after_node_callback:
            event = await self._execute_callback(
                self.after_node_callback,
                "after_node",
                current_node,
                current_node_name,
                state,
                iteration,
                ctx,
                agent_state,
                effective_config,
                output=output,
            )
            if event:
              yield event

          # Emit graph metadata event for evaluation framework
          # This will be captured in Invocation.intermediate_data by EvaluationGenerator
          # Set partial=True so is_final_response() returns False (making it an intermediate event)
          graph_metadata = {
              "graph_node": current_node_name,
              "graph_iteration": iteration,
              "graph_path": list(agent_state.path),
              "node_invocations": {
                  name: len(invocs)
                  for name, invocs in agent_state.node_invocations.items()
              },
              "graph_state": dict(state.data),
          }
          yield Event(
              author=f"{self.name}#metadata",
              content=types.Content(
                  parts=[types.Part(text=f"[GraphMetadata] {graph_metadata}")]
              ),
              partial=True,  # Mark as intermediate event
          )

          # Handle AFTER-node interrupt (retrospective feedback timing)
          # This enables retrospective feedback: observe past, steer future
          if (
              self._should_interrupt_after(current_node_name)
              and self.interrupt_service
          ):
            _a_events, _a_ctrl = await self._handle_after_node_interrupt(
                current_node_name, state, ctx, agent_state
            )
            for _e in _a_events:
              yield _e
            # Persist agent_state after interrupt handler may have mutated it
            ctx.set_agent_state(self.name, agent_state=agent_state)
            yield self._create_agent_state_event(ctx)
            if _a_ctrl == "break":
              break
            elif _a_ctrl is not None:
              if isinstance(_a_ctrl, tuple):
                current_node_name = _a_ctrl[1]
              continue

          # Checkpointing - yield event with state_delta to persist checkpoint
          # Note: For full checkpoint/resume functionality, use CheckpointCallback
          if self.checkpointing:
            ctx.set_agent_state(self.name, agent_state=agent_state)
            yield self._create_agent_state_event(ctx)
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text=f"Checkpoint: {current_node_name}")]
                ),
                actions=EventActions(
                    state_delta={
                        "graph_data": state.data,
                        "graph_checkpoint": {
                            "node": current_node_name,
                            "iteration": iteration,
                        },
                    }
                ),
            )

          # Inject transient execution data for edge conditions
          state.data["_graph_iteration"] = agent_state.iteration
          state.data["_graph_path"] = list(agent_state.path)
          state.data["_conditions"] = dict(agent_state.conditions)

          # Get next node via conditional routing
          next_node_name = self._get_next_node_with_telemetry(
              current_node, state, effective_config=effective_config
          )

          # Clean up transient keys
          for _tk in ("_graph_iteration", "_graph_path", "_conditions"):
            state.data.pop(_tk, None)
          if next_node_name is None:
            # No more edges - check if we're at an end node
            if current_node_name in self.end_nodes:
              break
            else:
              # Not at an end node and no edges - error
              raise ValueError(
                  f"Node {current_node_name} has no outgoing edges and is not"
                  " an end node"
              )

          current_node_name = next_node_name

          # Record iteration metrics (check sampling)
          if self._should_sample(effective_config=effective_config):
            graph_tracing.record_graph_iteration(
                agent_name=self.name,
                iteration=iteration,
                path_length=len(agent_state.path),
            )

        # ADK resumability: skip final response + end_of_agent when paused
        if not pause_invocation:
          # Final response - yield event with graph metadata
          # Include last node's output ONLY if:
          # 1. explicit final_output is set, OR
          # 2. last node was a function (doesn't yield events, so we need to show output)
          # Don't include output for agent nodes (they already yielded their output)
          final_output = state.data.get("final_output", "")

          # If no explicit final_output, check if last node was a function
          if not final_output and current_node_name:
            last_node = self.nodes.get(current_node_name)
            if last_node and last_node.function:
              # Function node - include its output
              final_output = state.data.get(current_node_name, "")

          response_text = f"{final_output}"

          yield Event(
              author=self.name,
              content=types.Content(parts=[types.Part(text=response_text)]),
              actions=EventActions(
                  state_delta={
                      "graph_data": state.data,
                      "graph_iterations": iteration,
                      "graph_path": list(agent_state.path),
                  }
              ),
          )
          # end_of_agent is guarded by is_resumable because it is purely a
          # resumability lifecycle signal (tells the runner "this agent is
          # done, don't re-run it on resume"). Unlike per-iteration state
          # events which serve rewind/interrupts/telemetry, end_of_agent
          # has no other consumers.
          if ctx.is_resumable:
            ctx.set_agent_state(self.name, end_of_agent=True)
            yield self._create_agent_state_event(ctx)

      finally:
        # Unregister session from InterruptService and finalize tracing
        if self.interrupt_service:
          self.interrupt_service.unregister_session(ctx.session.id)
        span.set_attribute("graph_agent.completed", True)

  # Interrupt methods inherited from GraphInterruptMixin

  @override
  @classmethod
  def _parse_config(
      cls,
      config: Any,  # GraphAgentConfig
      config_abs_path: str,
      kwargs: Dict[str, Any],
  ) -> Dict[str, Any]:
    """Parse GraphAgentConfig and return kwargs for GraphAgent constructor.

    Args:
        config: GraphAgentConfig instance
        config_abs_path: Absolute path to config file
        kwargs: Base kwargs from BaseAgent

    Returns:
        Updated kwargs with graph-specific configuration
    """
    from .graph_agent_config import GraphAgentConfig

    if not isinstance(config, GraphAgentConfig):
      return kwargs

    # Basic graph config
    if hasattr(config, "start_node") and config.start_node:
      kwargs["start_node"] = config.start_node

    if hasattr(config, "max_iterations") and config.max_iterations:
      kwargs["max_iterations"] = config.max_iterations

    if hasattr(config, "checkpointing"):
      kwargs["checkpointing"] = config.checkpointing

    # Interrupt configuration
    if hasattr(config, "interrupt_config") and config.interrupt_config:
      from .interrupt import InterruptConfig
      from .interrupt import InterruptMode

      interrupt_cfg = config.interrupt_config
      if interrupt_cfg.mode:  # None = disabled, only process if mode is set
        mode = InterruptMode(interrupt_cfg.mode)
        kwargs["interrupt_config"] = InterruptConfig(mode=mode)

    # Callbacks
    from ..config_agent_utils import resolve_code_reference

    if (
        hasattr(config, "before_node_callback_ref")
        and config.before_node_callback_ref
    ):
      kwargs["before_node_callback"] = resolve_code_reference(
          config.before_node_callback_ref
      )

    if (
        hasattr(config, "after_node_callback_ref")
        and config.after_node_callback_ref
    ):
      kwargs["after_node_callback"] = resolve_code_reference(
          config.after_node_callback_ref
      )

    if (
        hasattr(config, "on_edge_condition_callback_ref")
        and config.on_edge_condition_callback_ref
    ):
      kwargs["on_edge_condition_callback"] = resolve_code_reference(
          config.on_edge_condition_callback_ref
      )

    return kwargs

  @override
  @classmethod
  def from_config(
      cls,
      config: Any,  # GraphAgentConfig
      config_abs_path: str,
  ) -> "GraphAgent":
    """Creates a GraphAgent from a YAML config.

    This method performs post-construction setup to add nodes, edges,
    end nodes, and parallel groups from the config.

    Args:
        config: GraphAgentConfig instance
        config_abs_path: Absolute path to config file

    Returns:
        Configured GraphAgent instance
    """
    from ..config_agent_utils import resolve_agent_reference
    from ..config_agent_utils import resolve_code_reference
    from .graph_agent_config import GraphAgentConfig

    # Create base agent instance
    graph_instance = super().from_config(config, config_abs_path)

    # Type assertion: we know this is a GraphAgent because cls is GraphAgent
    assert isinstance(
        graph_instance, cls
    ), "Expected GraphAgent instance from super().from_config()"
    graph: GraphAgent = graph_instance  # type: ignore[assignment]

    if not isinstance(config, GraphAgentConfig):
      return graph

    # Add nodes
    if hasattr(config, "nodes") and config.nodes:
      for node_config in config.nodes:
        # Resolve sub-agents for this node
        sub_agents = []
        if node_config.sub_agents:
          for agent_ref in node_config.sub_agents:
            agent = resolve_agent_reference(agent_ref, config_abs_path)
            sub_agents.append(agent)

        # Resolve function ref
        function = None
        if node_config.function_ref:
          function = resolve_code_reference(node_config.function_ref)

        # Create GraphNode
        node = GraphNode(
            name=node_config.name,
            agent=sub_agents[0] if sub_agents else None,
            function=function,
        )
        graph.add_node(node)

    # Add edges
    if hasattr(config, "edges") and config.edges:
      from .graph_edge import EdgeCondition

      for edge_config in config.edges:
        condition = None
        if edge_config.condition:
          # Parse string condition to callable
          condition = _parse_condition_string(edge_config.condition)

        # Create EdgeCondition with priority and weight support
        edge = EdgeCondition(
            target_node=edge_config.target_node,
            condition=condition,
            priority=edge_config.priority,
            weight=edge_config.weight,
        )

        # Add edge directly to the node's edges list
        if edge_config.source_node in graph.nodes:
          graph.nodes[edge_config.source_node].edges.append(edge)
          graph.nodes[edge_config.source_node]._sorted_edges_cache = None
        else:
          raise ValueError(
              f"Source node {edge_config.source_node} not found in graph"
          )

    # Set start node
    if hasattr(config, "start_node") and config.start_node:
      graph.set_start(config.start_node)

    # Set end nodes
    if hasattr(config, "end_nodes") and config.end_nodes:
      for end_node in config.end_nodes:
        graph.set_end(end_node)

    # Add parallel groups
    if hasattr(config, "parallel_groups") and config.parallel_groups:
      from .parallel import ErrorPolicy
      from .parallel import JoinStrategy
      from .parallel import ParallelNodeGroup

      for pg_config in config.parallel_groups:
        join_strategy = JoinStrategy(pg_config.join_strategy)
        error_policy = ErrorPolicy(pg_config.error_policy)

        parallel_group = ParallelNodeGroup(
            nodes=pg_config.nodes,
            join_strategy=join_strategy,
            error_policy=error_policy,
            wait_n=pg_config.wait_n,
        )
        # Store parallel group (keyed by first node name for simplicity)
        graph.parallel_groups[pg_config.nodes[0]] = parallel_group

    return graph
