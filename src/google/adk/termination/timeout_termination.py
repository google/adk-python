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

"""Terminates after a specified duration has elapsed."""

from __future__ import annotations

import time
from typing import Optional
from typing import Sequence

from ..events.event import Event
from .termination_condition import TerminationCondition
from .termination_condition import TerminationResult


class TimeoutTermination(TerminationCondition):
  """Terminates the conversation after a specified duration has elapsed.

  The timer starts on the first ``check()`` call.

  Example::

      # Stop after 30 seconds
      condition = TimeoutTermination(30)
  """

  def __init__(self, timeout_seconds: float) -> None:
    if timeout_seconds <= 0:
      raise ValueError('timeout_seconds must be a positive number.')
    self._timeout_seconds = timeout_seconds
    self._start_time: Optional[float] = None
    self._terminated = False

  @property
  def terminated(self) -> bool:
    return self._terminated

  async def check(self, events: Sequence[Event]) -> Optional[TerminationResult]:
    if self._terminated:
      return None

    if self._start_time is None:
      self._start_time = time.monotonic()

    elapsed = time.monotonic() - self._start_time
    if elapsed >= self._timeout_seconds:
      self._terminated = True
      return TerminationResult(
          reason=(
              f'Timeout of {self._timeout_seconds}s reached'
              f' (elapsed: {elapsed:.2f}s)'
          )
      )
    return None

  async def reset(self) -> None:
    self._terminated = False
    self._start_time = None
