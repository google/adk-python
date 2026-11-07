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

"""Caching for values that belong to the event loop that created them."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import cached_property
import threading
import time
from typing import Any
from typing import cast
from typing import Generic
from typing import TYPE_CHECKING
from typing import TypeVar
import weakref

_T = TypeVar('_T')


class _NoRunningLoop:
  """Stands in for "read from outside any event loop" as a cache key.

  Weak keys cannot be ``None``, and every read made with no loop running has to
  share one entry. This module holds the single instance for the life of the
  process, so that entry is never evicted.
  """


_NO_RUNNING_LOOP = _NoRunningLoop()
_MISSING = object()

# The clock entries are aged against. A module attribute so tests can replace it
# without touching the clock the event loop itself runs on.
_now = time.monotonic


class _PerLoopCache(Generic[_T]):
  """Holds one value per event loop for a single attribute of a single owner.

  An async client holds sockets, locks and tasks that belong to the event loop
  that was running when they were opened. Caching such a client once for the
  lifetime of its owner therefore breaks as soon as a second loop uses it: the
  second loop reaches into a loop that has already been closed. Owners outlive
  loops routinely -- the synchronous runner entry points run each call in a new
  loop, and servers that dispatch requests to a thread pool do the same.

  Values are keyed weakly by the running loop, so every loop builds its own and
  an entry vanishes once its loop is collected, whether or not that loop was
  ever closed. Reading a value for a new loop additionally discards the entries
  whose loop reports itself closed, which releases the value promptly for a
  closed loop that something else still holds. Reads with no loop running share
  a single entry, which is never discarded. Sharing that one entry is safe
  because a reader with no loop running goes on to use the synchronous surface,
  which has no loop affinity. A caller that instead read the value off-loop and
  drove it from a loop would keep using it after that loop closed, since this
  is the one key the sweep never reaches.

  Keying weakly on the loop object is safe against address reuse. A lookup hits
  only when the stored key both hashes equal to and compares equal to the key
  being looked up, and a ``weakref.ref`` whose referent has died compares equal
  to nothing but itself. A reference to a new loop that happens to land on a
  collected loop's address is therefore a miss even though the two addresses,
  and so the two hashes, agree. The reference's own callback has usually
  dropped the entry by then in any case.
  """

  def __init__(self) -> None:
    self._lock = threading.Lock()
    self._values: weakref.WeakKeyDictionary[Any, _T] = (
        weakref.WeakKeyDictionary()
    )
    # Bumped by `clear`, so a build under way across one does not cache a value
    # that began before the caches were dropped.
    self._generation = 0
    # When each value's build began, on the `_now` clock.
    self._built_at: weakref.WeakKeyDictionary[Any, float] = (
        weakref.WeakKeyDictionary()
    )
    self._build_locks: weakref.WeakKeyDictionary[Any, threading.Lock] = (
        weakref.WeakKeyDictionary()
    )

  def __reduce__(self) -> tuple[type[_PerLoopCache[Any]], tuple[()]]:
    """Copies and pickles as an empty cache.

    Every cached value belongs to an event loop in this process, so none can
    be carried into a copy. The owner's copy builds its own on first read,
    which keeps an owner copyable and picklable after it has been used.
    """
    return (_PerLoopCache, ())

  def clear(self) -> None:
    """Drops every cached value, for every loop, so the next read rebuilds.

    Like the sweep of closed loops, this runs no disposal hook on what it drops.
    A caller that is still using a dropped value keeps it alive, and it is
    released when that caller lets go of it. A build already under way when this
    runs still returns its value to its own caller, but does not cache it.
    """
    with self._lock:
      self._generation += 1
      self._values.clear()
      self._built_at.clear()

  def _lookup(self, loop: Any, max_age: float | None) -> tuple[bool, Any]:
    """Returns ``(True, value)`` when ``loop`` has a value young enough to reuse.

    One lookup of the value, so a concurrent ``clear`` cannot land between a
    membership test and the read that follows it.
    """
    value = self._values.get(loop, _MISSING)
    if value is _MISSING:
      return False, None
    if max_age is not None:
      built_at = self._built_at.get(loop)
      if built_at is None or _now() - built_at >= max_age:
        return False, None
    return True, value

  def get(self, build: Callable[[], _T], max_age: float | None = None) -> _T:
    """Returns the value for the running loop, calling ``build`` on a miss.

    Args:
      build: Called to produce the value when the running loop has none.
      max_age: When set, a value whose build began this many seconds ago or
        longer is replaced by a fresh one rather than reused. The old value is
        only dropped from the cache, never closed, so a caller still holding it
        keeps a working value.
    """
    try:
      loop: Any = asyncio.get_running_loop()
    except RuntimeError:
      loop = _NO_RUNNING_LOOP

    # A tuple rather than a default, so a cached value of None is a hit. The key
    # is held by `loop` for as long as this call runs, so the entry cannot be
    # collected between the two lookups.
    hit, value = self._lookup(loop, max_age)
    if hit:
      return cast(_T, value)

    # Only the miss path is guarded. A hit is an unlocked dictionary lookup,
    # and the entry it finds belongs to the running loop, which cannot be
    # closed. The shared lock covers the eviction sweep and the insert, which
    # would otherwise raise "dictionary changed size during iteration", and
    # hands out one build lock per key. Building is left to that per-key lock:
    # a build opens sockets and takes tens of milliseconds, and every fresh
    # loop misses cold, so building under the shared lock would make loops
    # that share nothing queue behind each other.
    with self._lock:
      hit, value = self._lookup(loop, max_age)
      if hit:
        return cast(_T, value)
      # Dropping an entry runs no disposal hook. An entry is only dropped once
      # its loop is closed or already collected, and neither state can run the
      # asynchronous teardown a client would need; the value releases its
      # sockets when it is itself collected.
      #
      # Iterate the mapping itself rather than its underlying table: that
      # defers the collection callbacks, which fire on whichever thread
      # happened to drop the last reference to a loop.
      for stale in [
          key
          for key in self._values
          if key is not _NO_RUNNING_LOOP and key.is_closed()
      ]:
        del self._values[stale]
        self._built_at.pop(stale, None)
      build_lock = self._build_locks.setdefault(loop, threading.Lock())

    # Two threads missing the same key at once would otherwise each build a
    # value and only one would be kept: the other is live, holds open sockets,
    # and nothing goes on to close it. Only reads made with no loop running
    # can collide that way, since they are the only ones that share a key.
    with build_lock:
      hit, value = self._lookup(loop, max_age)
      if hit:
        return cast(_T, value)
      generation = self._generation
      started_at = _now()
      built = build()
      with self._lock:
        if generation == self._generation:
          self._values[loop] = built
          self._built_at[loop] = started_at
      return built


def _cache_for(owner: Any, key: str) -> _PerLoopCache[Any]:
  """Returns ``owner``'s cache for ``key``, creating it on first use.

  The cache lives in ``owner.__dict__``, the way
  ``functools.cached_property`` stores its own value, so an owner that is
  copied or discarded takes its values with it.
  """
  cache = owner.__dict__.get(key)
  if cache is None:
    # setdefault so that two threads racing the first read share one cache.
    cache = owner.__dict__.setdefault(key, _PerLoopCache())
  return cast('_PerLoopCache[Any]', cache)


def per_loop_value(
    owner: Any,
    key: str,
    build: Callable[[], _T],
    max_age: float | None = None,
) -> _T:
  """Returns a value for ``owner``, built once per running event loop.

  Args:
    owner: The object the value belongs to.
    key: The ``owner.__dict__`` entry holding the cache. Must not collide with a
      field name.
    build: Called to produce the value when the running loop has none.
    max_age: When set, seconds after which a cached value is rebuilt on its
      next read. ``None`` keeps a value for as long as its loop lives.

  Returns:
    The value belonging to the running event loop.
  """
  cache: _PerLoopCache[_T] = _cache_for(owner, key)
  return cache.get(build, max_age)


def clear_per_loop_value(owner: Any, key: str) -> None:
  """Drops every loop's cached value for ``key`` on ``owner``, if any.

  The next read from any loop rebuilds. Nothing is closed: a caller still using
  a dropped value keeps a working one. Does nothing when ``key`` was never read.
  """
  cache = owner.__dict__.get(key)
  if cache is not None:
    cache.clear()


if TYPE_CHECKING:
  # A type checker reads this as the decorator it stands in for, rather than as
  # the subclass below. Subclasses out in the wild override a decorated
  # attribute with a plain ``functools.cached_property``, and a checker rejects
  # an override whose descriptor type differs from the one it overrides -- it
  # recognizes only ``functools.cached_property`` itself, not a subclass of it.
  # Declaring the alias keeps every such override valid.
  PerLoopCachedProperty = cached_property
else:

  class PerLoopCachedProperty(cached_property):
    """A ``cached_property`` that keeps one value per running event loop.

    Reads behave like ``cached_property`` except that the value is cached
    against the running loop rather than once for the owner. See
    ``_PerLoopCache`` for why a client has to be cached that way.

    Like ``cached_property`` this defines no ``__set__``, so assigning over the
    attribute still writes to the owner's ``__dict__`` and shadows the
    descriptor from then on. The cache lives under a separate key, leaving the
    attribute's own slot free for those assignments.

    Deletion is where the two part company. ``cached_property`` invalidates
    itself through ``del owner.attr``; here a read writes only the cache, so
    that raises ``AttributeError`` unless a value was assigned over the
    attribute first, and deleting an assigned value uncovers the descriptor
    rather than clearing what it has cached.

    An owner may define a ``_per_loop_max_age`` attribute holding a number of
    seconds (or ``None``); a cached value older than that is rebuilt on its next
    read. Owners without it keep their values for as long as their loop lives.

    A Pydantic owner must additionally list this class in
    ``model_config['ignored_types']``, because Pydantic exempts
    ``cached_property`` by module name rather than by type.
    """

    def __get__(self, instance, owner=None):
      if instance is None:
        return self
      return per_loop_value(
          instance,
          f'_per_loop_{self.attrname}',
          lambda: self.func(instance),
          getattr(instance, '_per_loop_max_age', None),
      )
