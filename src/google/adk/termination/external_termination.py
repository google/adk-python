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

"""A termination condition controlled programmatically via ``set()``."""

from __future__ import annotations

from typing import Optional
from typing import Sequence

from ..events.event import Event
from .termination_condition import TerminationCondition
from .termination_condition import TerminationResult


class ExternalTermination(TerminationCondition):
  """A termination condition that is controlled externally by calling ``set()``.

  Useful for integrating external stop signals such as a UI "Stop" button
  or application-level logic.

  Example::

      stop_button = ExternalTermination()

      agent = LoopAgent(
          name='my_loop',
          sub_agents=[...],
          termination_condition=stop_button,
      )

      # Elsewhere (e.g. from a UI event handler):
      stop_button.set()
  """

  def __init__(self) -> None:
    self._terminated = False

  @property
  def terminated(self) -> bool:
    return self._terminated

  def set(self) -> None:
    """Signals that the conversation should terminate at the next check."""
    self._terminated = True

  async def check(
      self, events: Sequence[Event]
  ) -> Optional[TerminationResult]:
    if self._terminated:
      return TerminationResult(reason='Externally terminated')
    return None

  async def reset(self) -> None:
    self._terminated = False
