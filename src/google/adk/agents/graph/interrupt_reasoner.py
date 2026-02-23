"""LLM-based interrupt reasoning for GraphAgent.

This module provides an LLM agent that intelligently reasons about
interrupt messages and decides what action to take based on context.

The InterruptReasoner is a2a compatible and can be used as a standard
LlmAgent in the ADK framework.

Example:
    ```python
    from google.adk.agents.graph import (
        GraphAgent,
        InterruptConfig,
        InterruptMode,
    )
    from google.adk.agents.graph.interrupt_reasoner import (
        InterruptReasoner,
        InterruptReasonerConfig,
    )

    # Create reasoner with custom config
    reasoner = InterruptReasoner(InterruptReasonerConfig(
        model="gemini-2.0-flash-exp",
        available_actions=["continue", "rerun", "go_back", "pause", "defer"],
    ))

    # Use in GraphAgent
    graph = GraphAgent(
        name="my_graph",
        interrupt_config=InterruptConfig(
            mode=InterruptMode.AFTER,
            reasoner=reasoner,
        ),
    )
    ```
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from ..llm_agent import LlmAgent as LlmAgentType
else:
  LlmAgentType = Any

from pydantic import BaseModel

from google import genai

from ..llm_agent import LlmAgent
from .graph_state import GraphState
from .interrupt import InterruptAction
from .interrupt_service import InterruptMessage

logger = logging.getLogger("google_adk." + __name__)


class InterruptDecision(BaseModel):  # type: ignore[misc]
  """Structured output schema for InterruptReasoner LLM responses.

  Used as output_schema on the LlmAgent to get API-level JSON enforcement.
  The LLM returns valid JSON matching this schema without markdown wrapping.
  """

  action: str
  reasoning: str = ""
  parameters: Optional[Dict[str, Any]] = None


@dataclass
class InterruptReasonerConfig:
  """Configuration for InterruptReasoner.

  Attributes:
      model: LLM model to use for reasoning (default: gemini-2.0-flash-exp)
      instruction: System instruction for the reasoner
      available_actions: List of available actions the reasoner can choose
      custom_actions: Dict of custom action handlers (extensible)
      include_state_in_prompt: Whether to include full state in prompt (default: True)
      max_state_size: Maximum state size to include in prompt (default: 10000)
  """

  model: str = "gemini-2.0-flash-exp"
  instruction: str = (
      "You are an interrupt reasoning agent for a graph-based workflow system."
      " Analyze interrupt messages from humans and decide what action to take"
      " based on the current execution context, node output, and state."
  )
  available_actions: List[str] = None  # type: ignore[assignment]
  custom_actions: Dict[str, Callable[..., Any]] = None  # type: ignore[assignment]
  include_state_in_prompt: bool = True
  max_state_size: int = 10000
  fallback_action: str = "continue"

  def __post_init__(self) -> None:
    """Initialize default values."""
    if self.available_actions is None:
      self.available_actions = [
          "continue",
          "rerun",
          "go_back",
          "pause",
          "defer",
          "skip",
      ]
    if self.custom_actions is None:
      self.custom_actions = {}


class InterruptReasoner(LlmAgent):  # type: ignore[misc]
  """LLM agent that reasons about interrupt messages and decides actions.

  This agent receives interrupt messages, analyzes the execution context,
  and uses an LLM to intelligently decide what action to take.

  The reasoner is a2a compatible and can be used as a standard ADK agent.

  Attributes:
      config: InterruptReasonerConfig for this reasoner
      available_actions: List of actions the reasoner can choose from
      custom_actions: Dictionary of custom action handlers
  """

  def __init__(
      self,
      config: InterruptReasonerConfig,
      name: str = "interrupt_reasoner",
      **kwargs: Any,
  ):
    """Initialize InterruptReasoner.

    Args:
        config: Configuration for the reasoner
        name: Agent name (default: "interrupt_reasoner")
        **kwargs: Additional arguments passed to LlmAgent
    """
    super().__init__(
        name=name,
        model=config.model,
        instruction=config.instruction,
        output_schema=InterruptDecision,
        output_key=name,
        **kwargs,
    )
    # Store in private attributes (Pydantic allows these)
    self._config = config
    self._available_actions = config.available_actions
    self._custom_actions = config.custom_actions

  async def reason_about_interrupt(
      self,
      message: InterruptMessage,
      state: GraphState,
      current_node: str,
      ctx: Any,  # InvocationContext
      agent_state: Any = None,  # GraphAgentState
  ) -> InterruptAction:
    """Use LLM to reason about interrupt message and decide action.

    Args:
        message: Interrupt message from human
        state: Current graph state
        current_node: Node that just executed (or is about to execute)
        ctx: Invocation context
        agent_state: Execution tracking state (GraphAgentState)

    Returns:
        InterruptAction with decision (action, reasoning, parameters)
    """
    # Build reasoning prompt
    prompt = self._build_reasoning_prompt(
        message, state, current_node, agent_state
    )

    logger.debug(
        f"InterruptReasoner: reasoning about interrupt at node '{current_node}'"
    )

    # Call LLM via self.run_async()
    try:
      content = genai.types.Content(
          role="user", parts=[genai.types.Part(text=prompt)]
      )
      node_ctx = ctx.model_copy(update={"user_content": content})

      response_text = ""
      async for event in self.run_async(node_ctx):
        if event.content and event.content.parts:
          response_text = event.content.parts[0].text or ""

      # output_schema=InterruptDecision guarantees valid JSON from the API
      decision = InterruptDecision.model_validate_json(response_text.strip())
      validated_action = (
          decision.action
          if decision.action in self._available_actions
          else self._config.fallback_action
      )
      return InterruptAction(
          action=validated_action,
          reasoning=decision.reasoning,
          parameters=decision.parameters or {},
      )

    except Exception as e:
      logger.error("InterruptReasoner: Error during reasoning: %s", e)
      return InterruptAction(
          action=self._config.fallback_action,
          reasoning=f"Reasoning error: {e}",
          parameters={},
      )

  def _build_reasoning_prompt(
      self,
      message: InterruptMessage,
      state: GraphState,
      current_node: str,
      agent_state: Any = None,
  ) -> str:
    """Build reasoning prompt for LLM.

    Args:
        message: Interrupt message
        state: Current graph state
        current_node: Current node name
        agent_state: Execution tracking state (GraphAgentState)

    Returns:
        Formatted prompt string
    """
    # Use type-safe serialization (handles Pydantic models)
    state_str = state.data_to_json()
    if len(state_str) > self._config.max_state_size:
      state_str = state_str[: self._config.max_state_size] + "\n... (truncated)"

    path = list(agent_state.path) if agent_state else []
    iteration = agent_state.iteration if agent_state else "unknown"

    # Build prompt
    prompt = f"""
Current Situation:
- Node: {current_node}
- State: {state_str if self._config.include_state_in_prompt else "<state hidden>"}
- Execution path: {path}
- Iteration: {iteration}

Human Interrupt Message:
{message.text}

Metadata: {json.dumps(message.metadata, indent=2) if message.metadata else "None"}

Available Actions: {', '.join(self._available_actions)}

Analyze the interrupt and decide what to do. Consider:
- What did the node just produce?
- What is the human asking for?
- Should we continue, rerun with guidance, go back, pause, or defer?

"""
    return prompt
