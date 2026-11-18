# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Execution and delegation mode constants for ADK agents."""

from __future__ import annotations

import enum
from typing import Literal
from typing import TypeAlias


class AgentMode(str, enum.Enum):
  """Execution and delegation mode for an ADK agent.

  Subclasses ``str`` and returns ``self.value`` from ``__str__`` so members
  compare, format, and serialize identically to their plain string values
  (``'chat'``, ``'task'``, ``'single_turn'``).
  """

  CHAT = 'chat'
  """Conversational agent reachable via ``transfer_to_agent``."""

  TASK = 'task'
  """Multi-turn task agent that completes a task via ``finish_task``."""

  SINGLE_TURN = 'single_turn'
  """Single-turn agent that completes a task from its input without user replies."""

  def __str__(self) -> str:
    return self.value


LlmAgentMode: TypeAlias = Literal['chat', 'task', 'single_turn'] | AgentMode
"""Type alias for modes supported by ``LlmAgent``."""

SingleTurnAgentMode: TypeAlias = (
    Literal['single_turn'] | Literal[AgentMode.SINGLE_TURN]
)
"""Type alias for agents that only support ``single_turn`` delegation mode."""

TaskAgentMode: TypeAlias = Literal['task'] | Literal[AgentMode.TASK]
"""Type alias for agents that only support ``task`` delegation mode."""

DefaultLlmNodeMode: TypeAlias = (
    Literal['chat', 'single_turn']
    | Literal[AgentMode.CHAT, AgentMode.SINGLE_TURN]
)
"""Type alias for default LLM node modes in ``build_node``."""

DELEGATED_TASK_MODES: frozenset[str] = frozenset({
    AgentMode.TASK,
    AgentMode.SINGLE_TURN,
})
"""Agent modes that execute as tool-delegated tasks rather than LLM transfers."""
