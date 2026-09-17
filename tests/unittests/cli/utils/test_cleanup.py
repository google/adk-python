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

"""Tests for shutting down runners on server teardown."""

from __future__ import annotations

import asyncio

from google.adk.cli.utils.cleanup import close_runners
import pytest


class _FakeRunner:
  """Stands in for a Runner; only close() is exercised by the helper."""

  def __init__(self, delay: float = 0.0, error: Exception | None = None):
    self._delay = delay
    self._error = error
    self.closed = False

  async def close(self):
    if self._delay:
      await asyncio.sleep(self._delay)
    if self._error is not None:
      raise self._error
    self.closed = True


@pytest.mark.asyncio
async def test_close_runners_closes_every_runner():
  runners = [_FakeRunner(), _FakeRunner(), _FakeRunner()]

  await close_runners(runners)

  assert [r.closed for r in runners] == [True, True, True]


@pytest.mark.asyncio
async def test_close_runners_waits_for_the_slowest_runner():
  slow = _FakeRunner(delay=0.05)
  fast = _FakeRunner()

  await close_runners([fast, slow])

  # Returning as soon as the first runner finished would leave `slow` open.
  assert fast.closed
  assert slow.closed


@pytest.mark.asyncio
async def test_close_runners_does_not_let_one_failure_abort_the_rest():
  first = _FakeRunner()
  broken = _FakeRunner(error=RuntimeError('close failed'))
  last = _FakeRunner(delay=0.02)

  # Teardown is best-effort: a runner that blows up must not propagate or
  # strand the other runners.
  await close_runners([first, broken, last])

  assert first.closed
  assert last.closed


@pytest.mark.asyncio
async def test_close_runners_with_no_runners_is_a_noop():
  await close_runners([])
