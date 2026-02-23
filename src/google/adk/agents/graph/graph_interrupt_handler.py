"""Interrupt handling mixin for GraphAgent.

Extracts interrupt-related methods to keep GraphAgent focused on
core graph execution. The mixin pattern allows potential reuse
by other agent types that need step-level interrupt semantics.
"""

from __future__ import annotations

import logging
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING

from google.genai import types

from ...events.event import Event
from ...events.event_actions import EventActions
from ...telemetry import graph_tracing
from .graph_state import GraphState
from .interrupt import InterruptAction
from .interrupt import InterruptMode

if TYPE_CHECKING:
  from ..invocation_context import InvocationContext
  from .graph_agent_config import TelemetryConfig
  from .graph_agent_state import GraphAgentState
  from .graph_node import GraphNode
  from .interrupt_service import InterruptMessage

logger = logging.getLogger("google_adk." + __name__)


class GraphInterruptMixin:
  """Mixin providing interrupt handling for graph-based agents.

  Expects the host class to have:
  - self.interrupt_service: Optional[InterruptService]
  - self.interrupt_config: Optional[InterruptConfig]
  - self.name: str
  - self._get_telemetry_attributes() (from AgentTelemetryMixin)
  - self._should_sample() (from AgentTelemetryMixin)
  - self._get_next_node_with_telemetry() (from GraphAgent)
  """

  # Type stubs for fields provided by the host class
  interrupt_service: Any
  interrupt_config: Any
  name: str

  async def _check_interrupt_with_telemetry(
      self,
      session_id: str,
      mode: str,
      effective_config: Optional[TelemetryConfig] = None,
  ) -> Optional[Any]:
    """Check interrupt with telemetry.

    Args:
        session_id: Session identifier
        mode: Interrupt mode (before, after, both)
        effective_config: Effective telemetry config (merged parent + own)

    Returns:
        Interrupt message if any, None otherwise
    """
    if not self.interrupt_service:
      return None

    with graph_tracing.tracer.start_as_current_span("interrupt_check") as span:
      attrs = self._get_telemetry_attributes(  # type: ignore[attr-defined]
          {
              graph_tracing.GRAPH_INTERRUPT_MODE: mode,
              graph_tracing.GRAPH_SESSION_ID: session_id,
              graph_tracing.GRAPH_AGENT_NAME: self.name,
          },
          effective_config=effective_config,
      )
      for key, value in attrs.items():
        span.set_attribute(key, value)

      interrupt_message = await self.interrupt_service.check_interrupt(
          session_id
      )

      if self._should_sample(effective_config=effective_config):  # type: ignore[attr-defined]
        graph_tracing.record_interrupt_check(
            mode=mode, agent_name=self.name, session_id=session_id
        )

      return interrupt_message

  async def _dispatch_interrupt_action(
      self,
      action_result: str | Tuple[str, str],
      ctx: InvocationContext,
      timing: str,
      current_node: Optional[GraphNode] = None,
      state: Optional[GraphState] = None,
  ) -> str | Tuple[str, str] | None:
    """Route an interrupt action result to the appropriate control flow.

    Shared logic for both before-node and after-node interrupt handlers.

    Args:
        action_result: Result from _process_interrupt_message
        ctx: Invocation context
        timing: "before" or "after"
        current_node: The GraphNode (needed for "skip" in before-node)
        state: Current graph state (needed for "skip" routing)

    Returns:
        Control flow signal: None, "rerun", "break", ("go_back", target),
        or ("skip", next_node).
    """
    if isinstance(action_result, tuple):
      action, target_node = action_result
      if action == "go_back":
        return ("go_back", target_node)
    elif action_result == "rerun":
      return "rerun"
    elif action_result == "skip" and timing == "before":
      next_node_name = self._get_next_node_with_telemetry(current_node, state)  # type: ignore[attr-defined]
      return ("skip", next_node_name) if next_node_name else "break"
    elif action_result == "pause":
      try:
        resumed = await self.interrupt_service.wait_if_paused(ctx.session.id)
        if not resumed:
          logger.info(
              "GraphAgent execution cancelled for session %s",
              ctx.session.id,
          )
          return "break"
      except TimeoutError as e:
        logger.warning("Interrupt wait timeout: %s", e)
        return "break"

    return None

  async def _handle_before_node_interrupt(
      self,
      current_node_name: str,
      current_node: GraphNode,
      state: GraphState,
      ctx: InvocationContext,
      agent_state: GraphAgentState,
  ) -> Tuple[List[Event], str | Tuple[str, str] | None]:
    """Handle a BEFORE-node interrupt and return events + routing control.

    Args:
        current_node_name: Name of the node about to execute.
        current_node: The GraphNode about to execute (needed for "skip").
        state: Current graph state.
        ctx: Invocation context.
        agent_state: Execution tracking state.

    Returns:
        Tuple of (events_to_yield, control) where control is:
        - None: proceed to normal node execution.
        - "rerun": re-run current node (continue the loop).
        - "break": exit the main loop immediately.
        - ("go_back", target_node): jump to target_node.
        - ("skip", next_node | None): skip node, route to next_node.
    """
    assert self.interrupt_service is not None
    interrupt_message = await self._check_interrupt_with_telemetry(
        ctx.session.id, "before"
    )
    if not interrupt_message:
      return [], None

    action_result = await self._process_interrupt_message(
        interrupt_message, state, current_node_name, ctx, agent_state
    )

    should_escalate = (
        action_result == "pause"
        if isinstance(action_result, str)
        else (isinstance(action_result, tuple) and action_result[0] == "pause")
    )

    event = Event(
        author=self.name,
        content=types.Content(
            parts=[
                types.Part(
                    text=(
                        "\U0001f6d1 INTERRUPT (BEFORE):"
                        f" {interrupt_message.text}"
                    )
                )
            ]
        ),
        actions=EventActions(
            escalate=should_escalate,
            state_delta={
                "interrupt_message": interrupt_message.text,
                "interrupt_timing": "before",
                "interrupt_node": current_node_name,
            },
        ),
    )

    control = await self._dispatch_interrupt_action(
        action_result,
        ctx,
        "before",
        current_node=current_node,
        state=state,
    )
    return [event], control

  async def _handle_after_node_interrupt(
      self,
      current_node_name: str,
      state: GraphState,
      ctx: InvocationContext,
      agent_state: GraphAgentState,
  ) -> Tuple[List[Event], str | Tuple[str, str] | None]:
    """Handle an AFTER-node interrupt and return events + routing control.

    Args:
        current_node_name: Name of the node that just executed.
        state: Current graph state (includes the node's output).
        ctx: Invocation context.
        agent_state: Execution tracking state.

    Returns:
        Tuple of (events_to_yield, control) where control is:
        - None: accept results and proceed to next node.
        - "rerun": re-run current node.
        - "break": exit the main loop.
        - ("go_back", target_node): jump to target_node.
    """
    assert self.interrupt_service is not None
    interrupt_message = await self._check_interrupt_with_telemetry(
        ctx.session.id, "after"
    )
    if not interrupt_message:
      return [], None

    action_result = await self._process_interrupt_message(
        interrupt_message, state, current_node_name, ctx, agent_state
    )

    should_escalate = (
        action_result == "pause"
        if isinstance(action_result, str)
        else (isinstance(action_result, tuple) and action_result[0] == "pause")
    )

    state_delta_dict: Dict[str, Any] = {
        "interrupt_message": interrupt_message.text,
        "interrupt_timing": "after",
        "interrupt_metadata": interrupt_message.metadata,
        "interrupt_action": interrupt_message.action,
        "interrupt_node": current_node_name,
    }

    event = Event(
        author=self.name,
        content=types.Content(
            parts=[
                types.Part(
                    text=(
                        "\U0001f6d1 INTERRUPT (AFTER):"
                        f" {interrupt_message.text}"
                    )
                )
            ]
        ),
        actions=EventActions(
            escalate=should_escalate, state_delta=state_delta_dict
        ),
    )

    control = await self._dispatch_interrupt_action(
        action_result,
        ctx,
        "after",
    )
    return [event], control

  async def _process_interrupt_message(
      self,
      message: InterruptMessage,
      state: GraphState,
      current_node_name: str,
      ctx: InvocationContext,
      agent_state: GraphAgentState,
  ) -> str | Tuple[str, str]:
    """Process interrupt message using LLM reasoner if configured.

    Args:
        message: InterruptMessage from human
        state: Current graph state
        current_node_name: Name of the current node
        ctx: Invocation context
        agent_state: Execution tracking state

    Returns:
        Action string, or tuple (action, target_node) for go_back
    """
    agent_state.interrupt_history.append({
        "text": message.text,
        "action": message.action,
        "metadata": message.metadata or {},
        "timestamp": time.time(),
        "node": agent_state.current_node,
        "iteration": agent_state.iteration,
    })

    if self.interrupt_config and self.interrupt_config.reasoner:
      logger.debug("Using InterruptReasoner to decide action")
      action_obj = await self.interrupt_config.reasoner.reason_about_interrupt(
          message, state, current_node_name, ctx, agent_state
      )
      agent_state.last_interrupt_decision = {
          "action": action_obj.action,
          "reasoning": action_obj.reasoning,
          "parameters": action_obj.parameters,
          "node": current_node_name,
          "timestamp": time.time(),
      }
      logger.info(
          "InterruptReasoner decided: %s - %s",
          action_obj.action,
          action_obj.reasoning,
      )
    else:
      action_obj = InterruptAction(
          action=message.action or "continue",
          reasoning="Direct action from interrupt message",
          parameters=message.metadata or {},
      )

    return await self._execute_interrupt_action(
        action_obj, state, ctx, agent_state
    )

  async def _execute_interrupt_action(
      self,
      action: InterruptAction,
      state: GraphState,
      ctx: InvocationContext,
      agent_state: GraphAgentState,
  ) -> str | Tuple[str, str]:
    """Execute interrupt action based on LLM reasoner decision.

    Args:
        action: InterruptAction from reasoner
        state: Current graph state
        ctx: Invocation context
        agent_state: Execution tracking state

    Returns:
        Action string, or tuple (action, target_node) for go_back
    """
    if action.action == "defer":
      agent_state.interrupt_todos.append({
          "message": action.parameters.get("message", ""),
          "metadata": action.parameters,
          "timestamp": time.time(),
          "node": agent_state.current_node,
          "iteration": agent_state.iteration,
      })
      logger.info(
          "Deferred interrupt to todos: %s",
          action.parameters.get("message", ""),
      )
      return "continue"

    elif action.action == "rerun":
      if action.parameters.get("guidance"):
        agent_state.rerun_guidance = action.parameters["guidance"]
        logger.info(
            "Rerunning with guidance: %s",
            action.parameters["guidance"],
        )
      return "rerun"

    elif action.action == "go_back":
      steps = action.parameters.get("steps", 1)
      current_path = list(agent_state.path)

      if len(current_path) >= steps + 1:
        target_node = current_path[-(steps + 1)]
        nodes_to_clear = current_path[-steps:]
        agent_state.path = current_path[:-steps]

        for node_name in nodes_to_clear:
          # Use tracked output keys if available, fall back to node name
          tracked_keys = agent_state.output_keys.get(node_name)
          if tracked_keys:
            for key in tracked_keys:
              state.data.pop(key, None)
          else:
            logger.warning(
                "go_back: no tracked output_keys for node '%s', "
                "falling back to clearing key '%s'",
                node_name,
                node_name,
            )
            state.data.pop(node_name, None)

        logger.info(
            "Going back %d steps to node '%s' (cleared: %s)",
            steps,
            target_node,
            nodes_to_clear,
        )
        return ("go_back", target_node)
      else:
        logger.warning(
            "Cannot go back %d steps, only %d nodes in path. Continuing.",
            steps,
            len(current_path),
        )
        return "continue"

    elif action.action == "pause":
      return "pause"

    elif action.action == "skip":
      logger.info("Skipping current node execution")
      return "skip"

    elif action.action == "update_state":
      if action.parameters:
        for key in action.parameters:
          if key.startswith("_") or key.startswith("graph_"):
            raise ValueError(
                f"Cannot update reserved key '{key}'. "
                "Reserved prefixes: '_', 'graph_'"
            )
        state.data.update(action.parameters)
        logger.info(
            "Interrupt updated state: %s",
            list(action.parameters.keys()),
        )
      return "continue"

    elif action.action == "change_condition":
      if action.parameters:
        agent_state.conditions.update(action.parameters)
        logger.info("Interrupt changed conditions: %s", action.parameters)
      return "continue"

    else:  # "continue" or unknown
      logger.info("Interrupt action: %s", action.action)
      return "continue"

  def _should_interrupt_before(self, node_name: str) -> bool:
    """Check if should interrupt before this node."""
    if not self.interrupt_config:
      return False
    mode = self.interrupt_config.mode
    nodes = self.interrupt_config.nodes
    return mode in (InterruptMode.BEFORE, InterruptMode.BOTH) and (
        nodes is None or node_name in nodes
    )

  def _should_interrupt_after(self, node_name: str) -> bool:
    """Check if should interrupt after this node."""
    if not self.interrupt_config:
      return False
    mode = self.interrupt_config.mode
    nodes = self.interrupt_config.nodes
    return mode in (InterruptMode.AFTER, InterruptMode.BOTH) and (
        nodes is None or node_name in nodes
    )
