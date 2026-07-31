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

import asyncio
from enum import Enum

REFLECT_AND_RETRY_RESPONSE_TYPE = "ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN"
GLOBAL_SCOPE_KEY = "__global_reflect_and_retry_scope__"


class TrackingScope(Enum):
  """Defines the lifecycle scope for tracking failure counts."""

  INVOCATION = "invocation"
  GLOBAL = "global"


# A mapping from an item's (tool or model) name to its consecutive failure count.
PerItemFailuresCounter = dict[str, int]


def resolve_scope_key(scope: TrackingScope, invocation_id: str | None) -> str:
  """Returns the scope key for failure tracking."""
  if scope == TrackingScope.INVOCATION:
    if not invocation_id:
      raise ValueError("invocation_id must be provided for INVOCATION scope")
    return invocation_id
  elif scope == TrackingScope.GLOBAL:
    return GLOBAL_SCOPE_KEY
  raise ValueError(f"Unknown scope: {scope}")


class ScopedFailureTracker:
  """Thread-safe failure counter scoped by invocation or global key."""

  def __init__(self) -> None:
    self._scoped_failure_counters: dict[str, PerItemFailuresCounter] = {}
    self._lock = asyncio.Lock()

  async def increment(self, scope_key: str, item_name: str) -> int:
    """Atomically increments and returns the failure count for an item."""
    async with self._lock:
      failure_counter = self._scoped_failure_counters.setdefault(scope_key, {})
      current = failure_counter.get(item_name, 0) + 1
      failure_counter[item_name] = current
      return current

  async def reset(self, scope_key: str, item_name: str) -> None:
    """Atomically resets the failure count for an item and cleans up state."""
    async with self._lock:
      if scope_key in self._scoped_failure_counters:
        counter = self._scoped_failure_counters[scope_key]
        counter.pop(item_name, None)
