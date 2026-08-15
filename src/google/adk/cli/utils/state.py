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

from __future__ import annotations

import re
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

from ...agents.llm_agent import LlmAgent

if TYPE_CHECKING:
  from ...agents.base_agent import BaseAgent
  from ...workflow import BaseNode


def _create_empty_state(
    agent: BaseNode, all_state: dict[str, Any], visited: set[int]
) -> None:
  agent_id = id(agent)
  if agent_id in visited:
    return
  visited.add(agent_id)

  for sub_agent in getattr(agent, 'sub_agents', []) or []:
    _create_empty_state(sub_agent, all_state, visited)

  graph = getattr(agent, 'graph', None)
  if graph is not None:
    for graph_node in graph.nodes:
      _create_empty_state(graph_node, all_state, visited)

  if (
      isinstance(agent, LlmAgent)
      and agent.instruction
      and isinstance(agent.instruction, str)
  ):
    for key in re.findall(r'{([\w]+)}', agent.instruction):
      all_state[key] = ''


# `agent` is typed `BaseAgent | BaseNode` rather than just `BaseNode` (which
# would suffice, since BaseAgent subclasses BaseNode) so the public-API
# breaking-change detector sees a backward-compatible widening of the previous
# `BaseAgent` annotation instead of an incompatible type change.
def create_empty_state(
    agent: BaseAgent | BaseNode,
    initialized_states: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
  """Creates empty str for non-initialized states."""
  non_initialized_states: dict[str, Any] = {}
  _create_empty_state(agent, non_initialized_states, set())
  for key in initialized_states or {}:
    if key in non_initialized_states:
      del non_initialized_states[key]
  return non_initialized_states
