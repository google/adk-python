"""Advanced graph patterns for common agentic workflows.

This module provides first-class APIs for advanced patterns:
- DynamicNode: Runtime agent selection based on state
- NestedGraphNode: Hierarchical workflow composition (graph within graph)
- DynamicParallelGroup: Dynamic concurrent execution with variable agent count
"""

from __future__ import annotations

import asyncio
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
import uuid

from google.genai import types

from google import genai

from ...utils.feature_decorator import experimental
from ..base_agent import BaseAgent
from ..invocation_context import InvocationContext
from .graph_node import GraphNode
from .graph_state import GraphState


@experimental
class DynamicNode(GraphNode):
  """Node with runtime agent selection based on state.

  Enables dynamic dispatch pattern where the agent to execute is selected
  at runtime based on the current graph state. This is useful for:
  - Task routing (route to specialized agents based on task type)
  - Adaptive execution (select agent based on difficulty/complexity)
  - Multi-agent orchestration (dispatch to different agents per request)

  Example:
      ```python
      def select_agent(state: GraphState) -> BaseAgent:
          task_type = state.data.get("task_type", "simple")
          return complex_agent if task_type == "complex" else simple_agent

      node = DynamicNode(
          name="dispatcher",
          agent_selector=select_agent,
          fallback_agent=default_agent
      )
      ```
  """

  def __init__(
      self,
      name: str,
      agent_selector: Callable[[GraphState], Optional[BaseAgent]],
      fallback_agent: Optional[BaseAgent] = None,
      **kwargs: Any,
  ):
    """Initialize dynamic node.

    Args:
        name: Node name
        agent_selector: Function that selects agent based on state
        fallback_agent: Agent to use if selector returns None
        **kwargs: Additional arguments passed to GraphNode (input_mapper, output_mapper, etc.)
    """
    self.agent_selector = agent_selector
    self.fallback_agent = fallback_agent
    super().__init__(
        name=name, agent=None, function=self._execute_dynamic, **kwargs
    )

  async def _execute_dynamic(
      self, state: GraphState, ctx: InvocationContext
  ) -> str:
    """Execute selected agent based on state.

    Args:
        state: Current graph state
        ctx: Invocation context

    Returns:
        Agent output as string

    Raises:
        ValueError: If no agent selected and no fallback
    """
    selected = self.agent_selector(state) or self.fallback_agent
    if not selected:
      raise ValueError(
          f"No agent selected for {self.name} and no fallback provided"
      )

    # Observability: track selected agent for debugging
    state.data[f"_debug_{self.name}_selected_agent"] = selected.name

    node_input = self.input_mapper(state)
    node_ctx = ctx.model_copy(
        update={
            "user_content": types.Content(
                role="user", parts=[types.Part(text=node_input)]
            )
        }
    )

    output = ""
    async for event in selected.run_async(node_ctx):
      if event.content and event.content.parts:
        text = "".join(p.text for p in event.content.parts if p.text)
        if text:
          output += text
    return output


@experimental
class NestedGraphNode(GraphNode):
  """Node that executes a GraphAgent as sub-workflow.

  Enables hierarchical workflow composition where a GraphAgent is executed
  as a node within a parent graph. This is useful for:
  - Multi-step validation (validation graph within main workflow)
  - Conditional sub-workflows (execute different graphs based on conditions)
  - Workflow reuse (share common sub-workflows across graphs)

  The nested graph automatically inherits telemetry config from parent via
  context propagation, ensuring consistent observability.

  Example:
      ```python
      # Create validation sub-workflow
      validation_graph = GraphAgent(name="validation")
      validation_graph.add_node(GraphNode(name="check1", agent=checker1))
      validation_graph.add_node(GraphNode(name="check2", agent=checker2))

      # Embed in parent workflow
      nested = NestedGraphNode(
          name="validate",
          graph_agent=validation_graph,
          inherit_session=True
      )
      parent_graph.add_node(nested)
      ```
  """

  def __init__(
      self,
      name: str,
      graph_agent: "BaseAgent",  # Avoid circular import, will be GraphAgent at runtime
      inherit_session: bool = True,
      **kwargs: Any,
  ):
    """Initialize nested graph node.

    Args:
        name: Node name
        graph_agent: GraphAgent to execute as nested workflow
        inherit_session: If True, nested graph uses parent session (shares state/history)
                       If False, creates isolated session for nested execution
        **kwargs: Additional arguments passed to GraphNode
    """
    self.graph_agent = graph_agent
    self.inherit_session = inherit_session
    super().__init__(
        name=name, agent=None, function=self._execute_nested, **kwargs
    )

  async def _execute_nested(
      self, state: GraphState, ctx: InvocationContext
  ) -> str:
    """Execute nested graph workflow.

    Args:
        state: Current graph state
        ctx: Invocation context

    Returns:
        Final output from nested graph
    """
    if self.inherit_session:
      # Use parent session - nested graph sees parent state
      nested_ctx = ctx.model_copy(
          update={
              "user_content": types.Content(
                  role="user", parts=[types.Part(text=self.input_mapper(state))]
              )
          }
      )
    else:
      # Create isolated session - nested graph has clean state
      # Note: This requires access to session_service from parent context
      nested_session_id = f"nested_{uuid.uuid4().hex[:8]}"
      await ctx.session_service.create_session(
          app_name=self.graph_agent.name,
          user_id=ctx.session.user_id,
          session_id=nested_session_id,
      )
      nested_session = await ctx.session_service.get_session(
          app_name=self.graph_agent.name,
          user_id=ctx.session.user_id,
          session_id=nested_session_id,
      )
      nested_ctx = ctx.model_copy(
          update={
              "session": nested_session,
              "user_content": types.Content(
                  role="user",
                  parts=[types.Part(text=self.input_mapper(state))],
              ),
          }
      )

    # Execute nested graph
    # Filter empty-text and [GraphMetadata] sentinel events; keep the last
    # meaningful text output (agent nodes already emitted their content, but
    # the inner GraphAgent's final response event has empty text for agent
    # end-nodes, which would otherwise overwrite the real result).
    final_output = ""
    async for event in self.graph_agent.run_async(nested_ctx):
      if event.content and event.content.parts:
        text = "".join(p.text for p in event.content.parts if p.text)
        if text and not text.startswith("[GraphMetadata]"):
          final_output += text

    # Observability: track nested graph output (truncated)
    state.data[f"_debug_{self.name}_output"] = final_output[:500]

    return final_output


@experimental
class DynamicParallelGroup(GraphNode):
  """Node that executes multiple agents in parallel with dynamic concurrency.

  Enables dynamic parallel execution where the number of parallel agents
  is determined at runtime based on state. This is useful for:
  - Tree of Thoughts (generate N thoughts in parallel)
  - Parallel search (search multiple sources concurrently)
  - Batch processing (process variable-size batches)

  Example:
      ```python
      def gen_agents(state: GraphState) -> List[BaseAgent]:
          num_thoughts = state.data.get("num_thoughts", 3)
          return [thought_generator for _ in range(num_thoughts)]

      def aggregate(results: List[str], state: GraphState) -> str:
          return "\\n---\\n".join(f"Thought {i}: {r}" for i, r in enumerate(results))

      node = DynamicParallelGroup(
          name="generate_thoughts",
          agent_generator=gen_agents,
          aggregator=aggregate,
          max_parallelism=5  # Limit concurrent execution
      )
      ```
  """

  def __init__(
      self,
      name: str,
      agent_generator: Callable[[GraphState], List[BaseAgent]],
      aggregator: Callable[[List[str], GraphState], str],
      max_parallelism: int = 5,
      **kwargs: Any,
  ):
    """Initialize dynamic parallel group.

    Args:
        name: Node name
        agent_generator: Function that generates list of agents based on state
        aggregator: Function that aggregates all agent outputs into single result
        max_parallelism: Maximum number of agents to execute concurrently (default: 5)
        **kwargs: Additional arguments passed to GraphNode
    """
    self.agent_generator = agent_generator
    self.aggregator = aggregator
    self.max_parallelism = max_parallelism
    super().__init__(
        name=name, agent=None, function=self._execute_parallel, **kwargs
    )

  async def _execute_parallel(
      self, state: GraphState, ctx: InvocationContext
  ) -> str:
    """Execute agents in parallel with concurrency limit.

    Args:
        state: Current graph state
        ctx: Invocation context

    Returns:
        Aggregated output from all agents
    """
    agents = self.agent_generator(state)

    # Observability: track parallel execution count
    state.data[f"_debug_{self.name}_parallel_count"] = len(agents)

    if not agents:
      return self.aggregator([], state)

    semaphore = asyncio.Semaphore(self.max_parallelism)

    async def run_agent(agent: BaseAgent) -> str:
      """Execute single agent under the shared concurrency semaphore."""
      async with semaphore:
        node_input = self.input_mapper(state)
        node_ctx = ctx.model_copy(
            update={
                "user_content": types.Content(
                    role="user", parts=[types.Part(text=node_input)]
                )
            }
        )

        output = ""
        async for event in agent.run_async(node_ctx):
          if event.content and event.content.parts:
            text = "".join(p.text for p in event.content.parts if p.text)
            if text:
              output += text
        return output

    results = await asyncio.gather(*[run_agent(a) for a in agents])

    return self.aggregator(results, state)
