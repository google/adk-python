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

"""Shared utilities for environment tools."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import contextlib
import threading
from typing import TYPE_CHECKING
import weakref

from ._constants import MAX_OUTPUT_CHARS

if TYPE_CHECKING:
  from ...agents.context import Context
  from ...environment._base_environment import BaseEnvironment

# Locks serializing initialize and close, per environment. They are
# threading.Locks rather than asyncio.Locks because tool calls can run on
# several event loops at once: the live tool thread pool runs each call in
# its own loop.
_env_locks: weakref.WeakKeyDictionary[BaseEnvironment, threading.Lock] = (
    weakref.WeakKeyDictionary()
)
_env_locks_lock = threading.Lock()
_LOCK_POLL_INTERVAL_SECONDS = 0.01


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
  """Truncate text to *limit* characters with a notice."""
  if len(text) <= limit:
    return text
  return text[:limit] + f'\n... (truncated, {len(text)} total chars)'


@contextlib.asynccontextmanager
async def _env_lock(environment: BaseEnvironment) -> AsyncIterator[None]:
  """Holds *environment*'s lock, across event loops and threads."""
  with _env_locks_lock:
    lock = _env_locks.get(environment)
    if lock is None:
      lock = _env_locks[environment] = threading.Lock()
  # Poll rather than block, so this event loop keeps running while another
  # call holds the lock. A waiter that is cancelled holds nothing to release.
  while not lock.acquire(blocking=False):
    await asyncio.sleep(_LOCK_POLL_INTERVAL_SECONDS)
  try:
    yield
  finally:
    lock.release()


async def ensure_initialized(
    environment: BaseEnvironment, ctx: Context
) -> None:
  """Call ``initialize`` on *environment* with the calling tool's *ctx*.

  Called before every tool call, not only the first: an environment that
  keeps a sandbox per session finds it through ``ctx``, for example in
  ``ctx.state``, which ``is_initialized`` cannot see. ``initialize`` is
  idempotent, so an environment that is already set up returns early.

  Calls for one environment run one at a time, even from different event
  loops, so concurrent tool calls do not set it up twice. Marks the
  environment initialized afterwards if it did not do so itself.
  """
  # TODO: Once environments declare a scope, lock per session for
  # session-scoped environments, so one session's initialize does not block
  # tool calls in other sessions.
  async with _env_lock(environment):
    await environment.initialize(ctx=ctx)
    # Only assign when needed: some subclasses override `is_initialized`
    # with a read-only property that `initialize` keeps up to date.
    if not environment.is_initialized:
      environment.is_initialized = True


async def close_if_initialized(environment: BaseEnvironment) -> None:
  """Close *environment* if it is initialized, and mark it uninitialized.

  Holds the lock ``ensure_initialized`` uses, so concurrent calls close the
  environment once and an initialization does not run during a close.
  """
  async with _env_lock(environment):
    if environment.is_initialized:
      await environment.close()
      if environment.is_initialized:
        environment.is_initialized = False
