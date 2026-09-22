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

"""Tests for the per-event-loop value cache."""

import asyncio
import concurrent.futures
import gc
import threading
import time
import warnings
import weakref

from google.adk.utils._event_loop_cache import PerLoopCachedProperty
import pytest


class _Owner:
  """A plain owner, so the descriptor is exercised on its own."""

  def __init__(self, factory=None):
    self._factory = factory or (lambda: object())
    # list.append is atomic, so this counts builds from several threads.
    self.builds = []

  @PerLoopCachedProperty
  def value(self):
    built = self._factory()
    self.builds.append(built)
    return built


def _cache(owner):
  """The per-loop entries held for `owner`."""
  return owner.__dict__['_per_loop_value']._values


async def _read(owner):
  return owner.value


def _abandon_loops(owner, count):
  """Reads from `count` loops that are dropped without ever being closed."""
  observers = []
  for _ in range(count):
    loop = asyncio.new_event_loop()
    observers.append(weakref.ref(loop))
    loop.run_until_complete(_read(owner))
    del loop
  return observers


def test_value_is_built_once_and_reused_within_one_loop():
  owner = _Owner()

  async def read_twice():
    return owner.value, owner.value

  first, second = asyncio.run(read_twice())

  assert first is second
  assert len(owner.builds) == 1


def test_a_different_loop_gets_its_own_value():
  owner = _Owner()

  first = asyncio.run(_read(owner))
  second = asyncio.run(_read(owner))

  assert first is not second
  assert len(owner.builds) == 2


def test_reads_with_no_running_loop_share_one_value():
  owner = _Owner()

  outside = owner.value

  assert owner.value is outside
  assert asyncio.run(_read(owner)) is not outside
  assert owner.value is outside
  assert len(owner.builds) == 2


def test_two_owners_do_not_share_a_value():
  first_owner = _Owner()
  second_owner = _Owner()

  async def read_both():
    return first_owner.value, second_owner.value

  first, second = asyncio.run(read_both())

  assert first is not second


def test_concurrent_live_loops_each_get_their_own_value():
  owner = _Owner()
  thread_count = 8
  # Every loop reads while all the others are still open.
  barrier = threading.Barrier(thread_count, timeout=30)

  async def read_while_all_loops_are_open():
    await asyncio.get_running_loop().run_in_executor(None, barrier.wait)
    first = owner.value
    assert owner.value is first
    return first

  def run_in_its_own_loop():
    try:
      return asyncio.run(read_while_all_loops_are_open())
    except BaseException:  # pylint: disable=broad-except
      # Release the others instead of leaving them to wait out the barrier's
      # timeout; the failure still reaches the test through the future.
      barrier.abort()
      raise

  # One worker per party, or the barrier could never be satisfied.
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=thread_count
  ) as executor:
    futures = [
        executor.submit(run_in_its_own_loop) for _ in range(thread_count)
    ]
    # Every result is retrieved, so an exception in any loop fails the test.
    seen = [future.result() for future in futures]

  assert len(seen) == thread_count
  assert len({id(value) for value in seen}) == thread_count


def test_threads_missing_the_same_key_build_only_one_value():
  """A second value would hold open sockets that nothing goes on to close."""
  thread_count = 8
  # Every thread reads with no loop running, so they contend for one entry.
  ready = threading.Barrier(thread_count, timeout=30)

  def slow_build():
    # Widen the gap between the miss and the insert, so a build left outside
    # the lock would let the other threads through behind it.
    time.sleep(0.05)
    return object()

  owner = _Owner(factory=slow_build)

  def read():
    ready.wait()
    return owner.value

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=thread_count
  ) as executor:
    futures = [executor.submit(read) for _ in range(thread_count)]
    seen = [future.result() for future in futures]

  assert len(owner.builds) == 1
  assert len({id(value) for value in seen}) == 1


def test_unrelated_loops_build_at_the_same_time():
  """Every fresh loop misses cold, and a client takes tens of ms to open."""
  thread_count = 8
  # Satisfied only if all the builds overlap, so a build held behind one lock
  # for the whole cache breaks the barrier instead of waiting out its timeout.
  building = threading.Barrier(thread_count, timeout=10)

  def build_alongside_the_others():
    building.wait()
    return object()

  owner = _Owner(factory=build_alongside_the_others)

  def read_in_its_own_loop():
    try:
      return asyncio.run(_read(owner))
    except BaseException:  # pylint: disable=broad-except
      # Release the others rather than leaving them to wait out the timeout;
      # the failure still reaches the test through the future.
      building.abort()
      raise

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=thread_count
  ) as executor:
    futures = [
        executor.submit(read_in_its_own_loop) for _ in range(thread_count)
    ]
    seen = [future.result() for future in futures]

  assert len({id(value) for value in seen}) == thread_count


def test_a_cached_value_of_none_is_not_rebuilt():
  owner = _Owner(factory=lambda: None)

  async def read_twice():
    return owner.value, owner.value

  assert asyncio.run(read_twice()) == (None, None)
  assert len(owner.builds) == 1


def test_the_entry_for_a_closed_loop_is_discarded_on_the_next_miss():
  owner = _Owner()
  closed = asyncio.new_event_loop()
  closed.run_until_complete(_read(owner))
  closed.close()

  # Still held, so nothing has collected it; the sweep is what removes it.
  assert list(_cache(owner)) == [closed]

  survivor = asyncio.new_event_loop()
  try:
    survivor.run_until_complete(_read(owner))
    assert list(_cache(owner)) == [survivor]
  finally:
    survivor.close()


def test_loops_abandoned_without_being_closed_are_not_retained():
  owner = _Owner()

  observers = _abandon_loops(owner, count=50)
  with warnings.catch_warnings():
    # Collecting an unclosed loop is what this test is about.
    warnings.simplefilter('ignore', ResourceWarning)
    gc.collect()

  assert [observer for observer in observers if observer() is not None] == []
  assert len(_cache(owner)) == 0


def test_a_new_loop_never_reads_a_collected_loop_value():
  owner = _Owner()

  # Each cycle closes and drops its loop, so later loops are free to land on
  # an address a collected loop used to occupy.
  values = [asyncio.run(_read(owner)) for _ in range(50)]

  assert len({id(value) for value in values}) == 50
  assert len(owner.builds) == 50


def test_an_assigned_value_shadows_the_descriptor_everywhere():
  """Assigning has to keep working, the way cached_property allows it to."""
  owner = _Owner()
  assigned = object()

  owner.value = assigned

  assert owner.value is assigned
  assert asyncio.run(_read(owner)) is assigned
  assert owner.builds == []


def test_deleting_a_built_value_raises_the_way_a_plain_attribute_does():
  """cached_property's invalidation idiom; a read writes only the cache."""
  owner = _Owner()
  asyncio.run(_read(owner))

  with pytest.raises(AttributeError):
    del owner.value


def test_deleting_an_assigned_value_restores_building_per_loop():
  owner = _Owner()
  owner.value = object()

  del owner.value

  first = asyncio.run(_read(owner))
  assert asyncio.run(_read(owner)) is not first
  assert len(owner.builds) == 2


def test_class_level_access_returns_the_descriptor():
  assert isinstance(_Owner.value, PerLoopCachedProperty)
