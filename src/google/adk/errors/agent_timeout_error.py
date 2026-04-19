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

import enum


class TimeoutType(str, enum.Enum):
  SINGLE_TURN = 'single_turn'
  TOTAL = 'total'


class TimeoutTrigger(str, enum.Enum):
  LLM_CALL = 'llm_call'
  TOOL_CALL = 'tool_call'
  USER_INPUT = 'user_input'


class AgentTimeoutError(TimeoutError):
  def __init__(
      self,
      message: str,
      timeout_type: TimeoutType | str,
      elapsed_time: float,
      trigger: TimeoutTrigger | str,
      agent_name: str | None = None,
  ):
    self.timeout_type = (
        timeout_type.value
        if isinstance(timeout_type, TimeoutType)
        else timeout_type
    )
    self.elapsed_time = elapsed_time
    self.trigger = trigger.value if isinstance(trigger, TimeoutTrigger) else trigger
    self.agent_name = agent_name

    timeout_desc = (
        'Single-turn LLM call'
        if self.timeout_type == TimeoutType.SINGLE_TURN
        else 'Total agent execution'
    )
    agent_info = f' for agent "{agent_name}"' if agent_name else ''
    full_message = (
        f'{timeout_desc} timed out{agent_info}. '
        f'Elapsed time: {elapsed_time:.2f}s. '
        f'Triggered during: {self.trigger}.'
    )
    if message:
      full_message = f'{message} {full_message}'
    self.message = full_message
    super().__init__(self.message)
