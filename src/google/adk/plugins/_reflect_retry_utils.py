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
from typing import TYPE_CHECKING

from ..events._interrupts import index_open_interrupts

if TYPE_CHECKING:
  from ..agents.invocation_context import InvocationContext

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


def is_paused_invocation(invocation_context: InvocationContext) -> bool:
  """Returns whether a run ended paused on an interrupt nobody has answered.

  The answer to the interrupt continues the invocation under the same
  invocation id, resumable app or not, so its retry budget must carry over to
  that run. A failure response from this plugin settles the interrupt, since no
  answer will follow.
  """
  if invocation_context.is_aborted is True:
    return False
  invocation_events = [
      event
      for event in invocation_context.session.events
      if event.invocation_id == invocation_context.invocation_id
  ]
  failed_ids = set()
  for event in invocation_events:
    for fr in event.get_function_responses():
      if (
          fr.id
          and fr.response
          and fr.response.get("response_type")
          == REFLECT_AND_RETRY_RESPONSE_TYPE
      ):
        failed_ids.add(fr.id)
  return bool(set(index_open_interrupts(invocation_events)) - failed_ids)


class ScopedFailureTracker:
  """Failure counter scoped by invocation or global key.

  Updates are serialized by an asyncio lock, so concurrent coroutines on one
  event loop cannot lose an increment. The lock gives no mutual exclusion
  between threads, so an instance must not be shared across them.
  """

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

  async def clear_scope(self, scope_key: str) -> None:
    """Atomically drops every failure count recorded under a scope."""
    async with self._lock:
      self._scoped_failure_counters.pop(scope_key, None)
