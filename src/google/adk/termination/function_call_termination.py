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

"""Terminates when a specific function (tool) has been executed."""

from __future__ import annotations

from typing import Optional
from typing import Sequence

from ..events.event import Event
from .termination_condition import TerminationCondition
from .termination_condition import TerminationResult


class FunctionCallTermination(TerminationCondition):
  """Terminates when a tool with a specific name has been executed.

  The condition checks ``FunctionResponse`` parts in events.

  Example::

      # Stop when the "approve" tool is called
      condition = FunctionCallTermination('approve')
  """

  def __init__(self, function_name: str) -> None:
    self._function_name = function_name
    self._terminated = False

  @property
  def terminated(self) -> bool:
    return self._terminated

  async def check(
      self, events: Sequence[Event]
  ) -> Optional[TerminationResult]:
    if self._terminated:
      return None

    for event in events:
      for response in event.get_function_responses():
        if response.name == self._function_name:
          self._terminated = True
          return TerminationResult(
              reason=f"Function '{self._function_name}' was executed"
          )
    return None

  async def reset(self) -> None:
    self._terminated = False
