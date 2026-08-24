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

"""In-memory stand-in for the part of redis.asyncio that ADK calls."""

from __future__ import annotations

from collections.abc import AsyncIterator


class FakeRedisAsync:
  """In-memory asynchronous Redis mock for testing."""

  def __init__(self) -> None:
    self._store: dict[str, str] = {}
    self._ex_store: dict[str, int | None] = {}
    self._created_at: dict[str, float] = {}
    self._current_time: float = 0.0

  def advance_time(self, seconds: float) -> None:
    self._current_time += seconds

  def _is_expired(self, key: str) -> bool:
    if key not in self._store:
      return True
    ttl = self._ex_store.get(key)
    if ttl is not None and ttl > 0:
      created = self._created_at.get(key, 0.0)
      if self._current_time - created >= ttl:
        self._store.pop(key, None)
        self._ex_store.pop(key, None)
        self._created_at.pop(key, None)
        return True
    return False

  async def get(self, key: str) -> str | None:
    if self._is_expired(key):
      return None
    return self._store.get(key)

  async def set(
      self,
      key: str,
      value: str,
      ex: int | None = None,
      nx: bool = False,
  ) -> bool | None:
    if nx and not self._is_expired(key):
      return None
    self._store[key] = value
    self._ex_store[key] = ex
    self._created_at[key] = self._current_time
    return True

  async def delete(self, key: str) -> int:
    self._ex_store.pop(key, None)
    self._created_at.pop(key, None)
    if key in self._store:
      del self._store[key]
      return 1
    return 0

  async def scan_iter(self, match: str) -> AsyncIterator[str]:
    prefix = match.rstrip("*")
    for k in list(self._store):
      if not self._is_expired(k) and k.startswith(prefix):
        yield k
