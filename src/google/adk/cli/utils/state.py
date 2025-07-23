# Copyright 2025 Google LLC
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

from ...agents.base_agent import BaseAgent
from ...agents.llm_agent import LlmAgent


def _create_empty_state(agent: BaseAgent, all_state: dict[str, Any]):
  for sub_agent in agent.sub_agents:
    _create_empty_state(sub_agent, all_state)

  if (
      isinstance(agent, LlmAgent)
      and agent.instruction
      and isinstance(agent.instruction, str)
  ):
    for key in re.findall(r'{([\w]+)}', agent.instruction):
      all_state[key] = ''


def create_empty_state(
    agent: BaseAgent, initialized_states: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
  """Creates empty str for non-initialized states."""
  non_initialized_states = {}
  _create_empty_state(agent, non_initialized_states)
  for key in initialized_states or {}:
    if key in non_initialized_states:
      del non_initialized_states[key]
  return non_initialized_states
