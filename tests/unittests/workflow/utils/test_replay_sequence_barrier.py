# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ReplaySequenceBarrier."""

from __future__ import annotations

import asyncio

from google.adk.workflow.utils._replay_sequence_barrier import ReplaySequenceBarrier
import pytest


@pytest.mark.asyncio
async def test_barrier_initialization():
  """Verifies that barrier initializes sequence index and sets the first event."""
  # Given a chronological sequence of completions
  sequence = ['NodeA@1', 'NodeB@1']

  # When barrier is created
  barrier = ReplaySequenceBarrier(sequence)

  # Then state is correctly set
  assert barrier.sequence == sequence
  assert barrier.current_index == 0
  assert len(barrier.events) == 2
  assert barrier.events['NodeA@1'].is_set()
  assert not barrier.events['NodeB@1'].is_set()


@pytest.mark.asyncio
async def test_barrier_wait_blocks_and_unblocks():
  """Verifies that wait blocks on subsequent keys and is unblocked by advance."""
  sequence = ['NodeA@1', 'NodeB@1']
  barrier = ReplaySequenceBarrier(sequence)

  # When first key waits, it completes instantly
  await barrier.wait('NodeA@1')

  # When second key waits, it blocks
  b_completed = False

  async def wait_b():
    nonlocal b_completed
    await barrier.wait('NodeB@1')
    b_completed = True

  task = asyncio.create_task(wait_b())
  await asyncio.sleep(0.05)
  assert not b_completed  # Still blocked

  # When first key advances the sequence
  barrier.check_and_advance('NodeA@1')

  # Then index progresses and second event is released
  await task
  assert b_completed
  assert barrier.current_index == 1
  assert barrier.events['NodeB@1'].is_set()


def test_barrier_advance_out_of_order_ignored():
  """Verifies that out-of-order advance calls are ignored and do not progress index."""
  sequence = ['NodeA@1', 'NodeB@1']
  barrier = ReplaySequenceBarrier(sequence)

  # When second key tries to advance out of order
  barrier.check_and_advance('NodeB@1')

  # Then state remains unchanged
  assert barrier.current_index == 0
  assert not barrier.events['NodeB@1'].is_set()


@pytest.mark.asyncio
async def test_barrier_wait_non_existent_key():
  """Verifies that waiting on a key not in sequence does not block."""
  sequence = ['NodeA@1']
  barrier = ReplaySequenceBarrier(sequence)

  # When a key not in sequence waits, it passes instantly
  await barrier.wait('NonExistent@1')

  # No blocks, successfully completes!
  assert True
