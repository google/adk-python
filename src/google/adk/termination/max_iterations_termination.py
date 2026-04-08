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

"""Terminates after a maximum number of events have been processed."""

from __future__ import annotations

from typing import Optional
from typing import Sequence

from ..events.event import Event
from .termination_condition import TerminationCondition
from .termination_condition import TerminationResult


class MaxIterationsTermination(TerminationCondition):
  """Terminates the conversation after a maximum number of events.

  Example::

      # Stop after 10 events
      condition = MaxIterationsTermination(10)
  """

  def __init__(self, max_iterations: int) -> None:
    if max_iterations <= 0:
      raise ValueError('max_iterations must be a positive integer.')
    self._max_iterations = max_iterations
    self._count = 0
    self._terminated = False

  @property
  def terminated(self) -> bool:
    return self._terminated

  async def check(
      self, events: Sequence[Event]
  ) -> Optional[TerminationResult]:
    if self._terminated:
      return None
    self._count += len(events)

    if self._count >= self._max_iterations:
      self._terminated = True
      return TerminationResult(
          reason=(
              f'Maximum iterations of {self._max_iterations} reached,'
              f' current count: {self._count}'
          )
      )
    return None

  async def reset(self) -> None:
    self._terminated = False
    self._count = 0
