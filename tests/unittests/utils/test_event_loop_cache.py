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
import copy
import gc
import pickle
import threading
import time
import warnings
import weakref

from google.adk.utils import _event_loop_cache
from google.adk.utils._event_loop_cache import clear_per_loop_value
from google.adk.utils._event_loop_cache import per_loop_value
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


class _PicklableOwner:
  """A module-level owner, so pickle can find its class."""

  @PerLoopCachedProperty
  def value(self):
    return object()


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


def test_a_used_owner_deep_copies_without_its_cached_values():
  """Deep-copying an owner drops cached loop values so the copy builds fresh."""
  owner = _Owner()
  original = asyncio.run(_read(owner))

  copied = copy.deepcopy(owner)

  assert not _cache(copied)
  assert asyncio.run(_read(copied)) is not original


def test_a_used_owner_pickles_without_its_cached_values():
  """Pickling an owner drops cached loop values so unpickling builds fresh."""
  owner = _PicklableOwner()
  original = owner.value

  restored = pickle.loads(pickle.dumps(owner))

  assert not _cache(restored)
  assert restored.value is not original


class _Clock:
  """A settable stand-in for the clock entries are aged against."""

  def __init__(self):
    self.now = 1000.0

  def __call__(self):
    return self.now


@pytest.fixture(name='clock')
def _clock(monkeypatch):
  clock = _Clock()
  monkeypatch.setattr(_event_loop_cache, '_now', clock)
  return clock


class _AgingOwner(_Owner):
  """An owner that asks for its values to be rebuilt after `max_age` seconds."""

  def __init__(self, max_age=None):
    super().__init__()
    self._per_loop_max_age = max_age


def test_without_a_max_age_a_value_never_expires(clock):
  owner = _AgingOwner(max_age=None)
  first = owner.value

  clock.now += 10**9

  assert owner.value is first
  assert len(owner.builds) == 1


def test_a_value_younger_than_max_age_is_reused(clock):
  owner = _AgingOwner(max_age=60)
  first = owner.value

  clock.now += 59.999

  assert owner.value is first
  assert len(owner.builds) == 1


def test_a_value_exactly_max_age_old_is_rebuilt(clock):
  owner = _AgingOwner(max_age=60)
  first = owner.value

  clock.now += 60

  second = owner.value
  assert second is not first
  assert owner.value is second
  assert len(owner.builds) == 2


def test_a_rebuilt_value_ages_from_its_own_build(clock):
  owner = _AgingOwner(max_age=60)
  _ = owner.value
  clock.now += 60
  second = owner.value

  clock.now += 59

  assert owner.value is second
  clock.now += 1
  assert owner.value is not second
  assert len(owner.builds) == 3


def test_age_is_counted_from_when_the_build_began(clock):
  def slow_build():
    clock.now += 50
    return object()

  owner = _AgingOwner(max_age=60)
  owner._factory = slow_build
  first = owner.value

  clock.now += 10

  assert owner.value is not first


def test_max_age_can_change_between_reads(clock):
  owner = _AgingOwner(max_age=None)
  first = owner.value
  clock.now += 100

  assert owner.value is first
  owner._per_loop_max_age = 50
  assert owner.value is not first


def test_expiry_in_one_loop_leaves_a_live_loop_untouched(clock):
  owner = _AgingOwner(max_age=60)
  results = {}

  async def scenario():
    loop_a_value = owner.value
    results['a_before'] = loop_a_value

    def read_in_other_loop():
      async def read():
        return owner.value

      return asyncio.run(read())

    other = await asyncio.get_running_loop().run_in_executor(
        None, read_in_other_loop
    )
    results['other'] = other
    assert owner.value is loop_a_value

  asyncio.run(scenario())
  assert results['other'] is not results['a_before']
  assert len(owner.builds) == 2


def test_an_expired_value_is_dropped_but_never_closed(clock):
  class Closable:
    closed = False

    def close(self):
      self.closed = True

  owner = _AgingOwner(max_age=60)
  owner._factory = Closable
  first = owner.value
  clock.now += 60

  assert owner.value is not first
  assert not first.closed


def test_clear_drops_values_for_every_loop():
  owner = _Owner()
  in_loop = asyncio.run(_read(owner))
  outside_loop = owner.value

  clear_per_loop_value(owner, '_per_loop_value')

  assert not _cache(owner)
  assert owner.value is not outside_loop
  assert asyncio.run(_read(owner)) is not in_loop


def test_clear_before_any_read_is_a_no_op():
  owner = _Owner()

  clear_per_loop_value(owner, '_per_loop_value')

  assert owner.builds == []
  assert '_per_loop_value' not in owner.__dict__


def test_clear_on_one_key_leaves_other_keys_alone():
  owner = _Owner()
  first = owner.value
  other = per_loop_value(owner, 'other_key', object)

  clear_per_loop_value(owner, '_per_loop_value')

  assert per_loop_value(owner, 'other_key', object) is other
  assert owner.value is not first


def test_a_build_under_way_during_clear_is_returned_but_not_cached():
  owner = _Owner()
  seen = []

  def build():
    clear_per_loop_value(owner, '_per_loop_value')
    value = object()
    seen.append(value)
    return value

  owner._factory = build
  owner.value  # Creates the cache, building across a clear of that cache.
  owner._factory = object

  assert not _cache(owner)
  assert owner.value is not seen[0]


def test_threads_reading_while_values_are_cleared_never_fail(clock):
  owner = _AgingOwner(max_age=0.001)
  stop = threading.Event()
  errors = []

  def read():
    while not stop.is_set():
      try:
        owner.value
      except BaseException as e:  # pylint: disable=broad-except
        errors.append(e)
        return

  def clear():
    while not stop.is_set():
      clear_per_loop_value(owner, '_per_loop_value')
      clock.now += 1

  threads = [threading.Thread(target=read) for _ in range(4)]
  threads.append(threading.Thread(target=clear))
  for thread in threads:
    thread.start()
  time.sleep(0.3)
  stop.set()
  for thread in threads:
    thread.join(timeout=10)

  assert errors == []
