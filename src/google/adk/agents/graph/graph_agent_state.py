"""Execution tracking state for GraphAgent.

Follows ADK's BaseAgentState pattern: persisted via
ctx.agent_states / Event.actions.agent_state.

Domain data (node outputs) remains in GraphState via state_delta.
"""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import Field

from ...utils.feature_decorator import experimental
from ..base_agent import BaseAgentState


@experimental
class GraphAgentState(BaseAgentState):  # type: ignore[misc]
  """Execution tracking state for GraphAgent.

  Serialized via model_dump(mode='json'), restored via model_validate().
  """

  current_node: str = ""
  prev_node: str = ""
  iteration: int = 0
  execution_start: float = 0.0

  path: List[str] = Field(default_factory=list)
  node_invocations: Dict[str, List[str]] = Field(default_factory=dict)
  conditions: Dict[str, Any] = Field(default_factory=dict)
  rerun_guidance: str = ""

  interrupt_history: List[Dict[str, Any]] = Field(default_factory=list)
  interrupt_todos: List[Dict[str, Any]] = Field(default_factory=list)
  last_interrupt_decision: Optional[Dict[str, Any]] = None

  telemetry_config_dict: Optional[Dict[str, Any]] = None

  agent_path: List[str] = Field(default_factory=list)
  executed_parallel_groups: List[str] = Field(default_factory=list)
