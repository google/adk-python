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

"""Chronological sequence barrier for deterministic replay ordering."""

from __future__ import annotations

import asyncio


class ReplaySequenceBarrier:
  """Unified chronological sequence barrier to ensure deterministic replay ordering."""

  def __init__(self, sequence: list[str], timeout_sec: float = 15.0) -> None:
    self.sequence = sequence
    self.timeout_sec = timeout_sec
    self.current_index = 0
    self.events = {key: asyncio.Event() for key in sequence}
    if sequence:
      self.events[sequence[0]].set()

  async def wait(self, key: str) -> None:
    """Wait for the barrier if the key is part of the expected chronological sequence.

    Only wait if the node had a terminal event (output, route, or interrupt).
    "Silent" nodes that only yielded state updates but didn't produce
    output are not in the sequence barrier, so they fast-forward immediately.
    """
    if key in self.events:
      try:
        await asyncio.wait_for(
            self.events[key].wait(), timeout=self.timeout_sec
        )
      except asyncio.TimeoutError:
        raise RuntimeError(
            "Replay divergence detected: Timed out waiting for sequence key"
            f" '{key}' to be unblocked."
        )

  def check_and_advance(self, key: str) -> None:
    """Advance the sequence if the key matches the current expected execution."""
    if self.current_index < len(self.sequence):
      expected_key = self.sequence[self.current_index]
      if key == expected_key:
        self.current_index += 1
        if self.current_index < len(self.sequence):
          next_key = self.sequence[self.current_index]
          self.events[next_key].set()
