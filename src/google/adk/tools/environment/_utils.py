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

from ...environment._environment_scope import EnvironmentScope
from ._constants import MAX_OUTPUT_CHARS

if TYPE_CHECKING:
  from pathlib import Path

  from ...agents.context import Context
  from ...environment._base_environment import BaseEnvironment
  from ...sessions.session import Session

# Identifies one lock of an environment: empty for the environment as a whole,
# otherwise the user or session that `initialize` is running for.
_LockKey = tuple[str, ...]


class _LockEntry:
  """A lock and the number of calls holding or waiting for it."""

  def __init__(self) -> None:
    self.lock = threading.Lock()
    self.users = 0


class _CloseGate:
  """Keeps a close and the initialize calls of an environment apart."""

  def __init__(self) -> None:
    # The number of `initialize` calls that have passed the gate.
    self.initializing = 0
    # The number of closes waiting for or holding the environment.
    self.closing = 0


# Locks serializing initialize and close, per environment and key. They are
# threading.Locks rather than asyncio.Locks because tool calls can run on
# several event loops at once: the live tool thread pool runs each call in
# its own loop.
_env_locks: weakref.WeakKeyDictionary[
    BaseEnvironment, dict[_LockKey, _LockEntry]
] = weakref.WeakKeyDictionary()
_env_gates: weakref.WeakKeyDictionary[BaseEnvironment, _CloseGate] = (
    weakref.WeakKeyDictionary()
)
# Guards both dictionaries and the counters of their values.
_env_locks_lock = threading.Lock()
_LOCK_POLL_INTERVAL_SECONDS = 0.01


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
  """Truncate text to *limit* characters with a notice."""
  if len(text) <= limit:
    return text
  return text[:limit] + f'\n... (truncated, {len(text)} total chars)'


def working_dir_or_none(environment: BaseEnvironment) -> Path | None:
  """Returns *environment*'s working directory, or None when it has none yet.

  Environments written before ``working_dir`` could return ``None`` raise
  ``RuntimeError`` from it until ``initialize`` has run. That counts as
  ``None`` here, so reading it while building a model request does not fail.
  """
  try:
    return environment.working_dir
  except RuntimeError:
    return None


@contextlib.asynccontextmanager
async def _env_lock(
    environment: BaseEnvironment, key: _LockKey = ()
) -> AsyncIterator[None]:
  """Holds *environment*'s lock for *key*, across event loops and threads."""
  with _env_locks_lock:
    entries = _env_locks.setdefault(environment, {})
    entry = entries.get(key)
    if entry is None:
      entry = entries[key] = _LockEntry()
    entry.users += 1
  try:
    # Poll rather than block, so this event loop keeps running while another
    # call holds the lock. A waiter that is cancelled holds nothing to release.
    while not entry.lock.acquire(blocking=False):
      await asyncio.sleep(_LOCK_POLL_INTERVAL_SECONDS)
    try:
      yield
    finally:
      entry.lock.release()
  finally:
    with _env_locks_lock:
      entry.users -= 1
      # Drop a lock nobody uses, so an environment that serves many sessions
      # does not keep one lock for each of them.
      if not entry.users:
        del entries[key]


@contextlib.asynccontextmanager
async def _initializing(environment: BaseEnvironment) -> AsyncIterator[None]:
  """Runs an ``initialize`` call when no close is waiting or running."""
  with _env_locks_lock:
    gate = _env_gates.setdefault(environment, _CloseGate())
  while True:
    with _env_locks_lock:
      if not gate.closing:
        gate.initializing += 1
        break
    await asyncio.sleep(_LOCK_POLL_INTERVAL_SECONDS)
  try:
    yield
  finally:
    with _env_locks_lock:
      gate.initializing -= 1


@contextlib.asynccontextmanager
async def _closing(environment: BaseEnvironment) -> AsyncIterator[None]:
  """Runs a close once the ``initialize`` calls in flight have finished.

  New ``initialize`` calls wait from the moment the close starts waiting, so
  calls that keep arriving cannot hold the close off.
  """
  with _env_locks_lock:
    gate = _env_gates.setdefault(environment, _CloseGate())
    gate.closing += 1
  try:
    while True:
      with _env_locks_lock:
        if not gate.initializing:
          break
      await asyncio.sleep(_LOCK_POLL_INTERVAL_SECONDS)
    yield
  finally:
    with _env_locks_lock:
      gate.closing -= 1


def _scope_lock_key(environment: BaseEnvironment, ctx: Context) -> _LockKey:
  """Returns the key of the lock ``initialize`` holds for *ctx*.

  Only the ``USER`` and ``SESSION`` scopes have a lock of their own. Any other
  value locks the environment as a whole, as ``SHARED`` does, which is the
  safe reading of a scope this function does not know.

  Raises:
    ValueError: If the environment's scope is ``USER`` or ``SESSION`` and
      *ctx* has no session to take the user or session from.
  """
  scope = environment.scope
  if scope == EnvironmentScope.USER:
    session = _require_session(environment, EnvironmentScope.USER, ctx)
    return ('user', session.app_name, session.user_id)
  if scope == EnvironmentScope.SESSION:
    session = _require_session(environment, EnvironmentScope.SESSION, ctx)
    return ('session', session.app_name, session.user_id, session.id)
  return ()


def _require_session(
    environment: BaseEnvironment, scope: EnvironmentScope, ctx: Context
) -> Session:
  """Returns the session of *ctx*, which a scoped environment needs."""
  # Not `getattr(ctx, 'session', None)`: that would also hide an
  # AttributeError raised inside the `session` property of a context.
  session = None if ctx is None else ctx.session
  if session is None:
    raise ValueError(
        f'{type(environment).__name__} has the {scope.value} scope, so it can'
        ' only be initialized with a context that has a session.'
    )
  return session


async def ensure_initialized(
    environment: BaseEnvironment, ctx: Context
) -> None:
  """Call ``initialize`` on *environment* with the calling tool's *ctx*.

  Called before every tool call, not only the first: an environment that
  keeps a sandbox per session finds it through ``ctx``, for example in
  ``ctx.state``, which ``is_initialized`` cannot see. ``initialize`` is
  idempotent, so an environment that is already set up returns early.

  Calls within the environment's ``scope`` run one at a time, even from
  different event loops, so concurrent tool calls do not set a resource up
  twice: all calls for a ``SHARED`` environment, and the calls of one user or
  one session for a ``USER`` or ``SESSION`` environment. Calls for other users
  or sessions of such an environment run at the same time. No call runs
  while ``close_if_initialized`` closes the environment. Marks the environment
  initialized afterwards if it did not do so itself.
  """
  key = _scope_lock_key(environment, ctx)
  async with _initializing(environment), _env_lock(environment, key):
    await environment.initialize(ctx=ctx)
    # Only assign when needed: some subclasses override `is_initialized`
    # with a read-only property that `initialize` keeps up to date.
    if not environment.is_initialized:
      environment.is_initialized = True


async def close_if_initialized(environment: BaseEnvironment) -> None:
  """Close *environment* if it is initialized, and mark it uninitialized.

  Concurrent calls close the environment once. A close waits for the
  ``initialize`` calls that ``ensure_initialized`` has in flight, whatever
  user or session they are for, and calls that arrive meanwhile wait for the
  close. So a close neither runs during an initialization nor misses one
  that is about to mark the environment initialized.
  """
  async with _closing(environment), _env_lock(environment):
    if environment.is_initialized:
      await environment.close()
      if environment.is_initialized:
        environment.is_initialized = False
