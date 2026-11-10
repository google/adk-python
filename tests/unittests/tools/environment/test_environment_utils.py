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

"""Tests for the environment tool utilities."""

import asyncio
import concurrent.futures
from pathlib import Path
from typing import Any
from typing import Optional
import weakref

from google.adk.environment._base_environment import BaseEnvironment
from google.adk.environment._base_environment import ExecutionResult
from google.adk.tools.environment import _utils
import pytest


class _CountingEnvironment(BaseEnvironment):
  """Environment that records initialize calls and never sets the flag."""

  def __init__(self, *, fail_first_init: bool = False) -> None:
    self.init_contexts: list[Any] = []
    self.close_count = 0
    self.max_concurrent_inits = 0
    self._running_inits = 0
    self._fail_next_init = fail_first_init

  @property
  def working_dir(self) -> Path:
    return Path('/workspace')

  async def initialize(self, *, ctx: Any = None) -> None:
    self._running_inits += 1
    self.max_concurrent_inits = max(
        self.max_concurrent_inits, self._running_inits
    )
    try:
      await asyncio.sleep(0)
      self.init_contexts.append(ctx)
      if self._fail_next_init:
        self._fail_next_init = False
        raise RuntimeError('initialize failed')
    finally:
      self._running_inits -= 1

  async def close(self) -> None:
    await asyncio.sleep(0)
    self.close_count += 1

  async def execute(
      self,
      command: str,
      *,
      timeout: Optional[float] = None,
      ctx: Any = None,
  ) -> ExecutionResult:
    raise NotImplementedError

  async def read_file(self, path: Path, *, ctx: Any = None) -> bytes:
    raise NotImplementedError

  async def write_file(
      self, path: Path, content: str | bytes, *, ctx: Any = None
  ) -> None:
    raise NotImplementedError


@pytest.mark.asyncio
async def test_ensure_initialized_calls_initialize_with_every_ctx():
  """Every call reaches initialize with its ctx, even once initialized."""
  env = _CountingEnvironment()
  first_ctx = object()
  second_ctx = object()

  await _utils.ensure_initialized(env, first_ctx)
  assert env.is_initialized
  await _utils.ensure_initialized(env, second_ctx)

  assert env.init_contexts == [first_ctx, second_ctx]
  assert env.is_initialized


@pytest.mark.asyncio
async def test_ensure_initialized_concurrent_calls_run_one_at_a_time():
  """Concurrent calls for one environment do not run initialize together."""
  env = _CountingEnvironment()

  await asyncio.gather(
      *(_utils.ensure_initialized(env, object()) for _ in range(8))
  )

  assert len(env.init_contexts) == 8
  assert env.max_concurrent_inits == 1


@pytest.mark.asyncio
async def test_ensure_initialized_environments_are_independent():
  """Initializing one environment does not initialize another."""
  env1 = _CountingEnvironment()
  env2 = _CountingEnvironment()
  ctx1 = object()
  ctx1_again = object()
  ctx2 = object()

  await _utils.ensure_initialized(env1, ctx1)
  assert not env2.is_initialized

  await asyncio.gather(
      _utils.ensure_initialized(env1, ctx1_again),
      _utils.ensure_initialized(env2, ctx2),
  )

  assert env1.init_contexts == [ctx1, ctx1_again]
  assert env2.init_contexts == [ctx2]


@pytest.mark.asyncio
async def test_ensure_initialized_retries_after_failed_initialize():
  """A failed initialize leaves the environment uninitialized for a retry."""
  env = _CountingEnvironment(fail_first_init=True)
  retry_ctx = object()

  with pytest.raises(RuntimeError, match='initialize failed'):
    await _utils.ensure_initialized(env, object())
  assert not env.is_initialized

  await _utils.ensure_initialized(env, retry_ctx)

  assert env.init_contexts[-1] is retry_ctx
  assert len(env.init_contexts) == 2
  assert env.is_initialized


@pytest.mark.asyncio
async def test_close_if_initialized_skips_uninitialized_environment():
  """An environment that was never initialized is not closed."""
  env = _CountingEnvironment()

  await _utils.close_if_initialized(env)

  assert env.close_count == 0


@pytest.mark.asyncio
async def test_close_if_initialized_closes_once_and_allows_reinit():
  """Concurrent closes close once; the environment can be initialized again."""
  env = _CountingEnvironment()
  await _utils.ensure_initialized(env, object())

  await asyncio.gather(*(_utils.close_if_initialized(env) for _ in range(8)))

  assert env.close_count == 1
  assert not env.is_initialized
  await _utils.ensure_initialized(env, object())
  assert len(env.init_contexts) == 2


class _ReadOnlyFlagEnvironment(_CountingEnvironment):
  """Keeps its own flag behind a read-only `is_initialized` property."""

  def __init__(self) -> None:
    super().__init__()
    self._initialized = False

  @property
  def is_initialized(self) -> bool:
    return self._initialized

  async def initialize(self, *, ctx: Any = None) -> None:
    await super().initialize(ctx=ctx)
    self._initialized = True

  async def close(self) -> None:
    await super().close()
    self._initialized = False


@pytest.mark.asyncio
async def test_read_only_is_initialized_property_is_not_assigned():
  """An environment that tracks its own flag read-only works with both."""
  env = _ReadOnlyFlagEnvironment()
  ctx = object()
  second_ctx = object()

  await _utils.ensure_initialized(env, ctx)
  await _utils.ensure_initialized(env, second_ctx)
  await _utils.close_if_initialized(env)

  assert env.init_contexts == [ctx, second_ctx]
  assert env.close_count == 1
  assert not env.is_initialized


class _SandboxEnvironment(_CountingEnvironment):
  """Creates a sandbox unless it has one, as Daytona and E2B do."""

  def __init__(self) -> None:
    super().__init__()
    self.sandboxes_created = 0
    self._sandbox: Optional[object] = None

  async def initialize(self, *, ctx: Any = None) -> None:
    if self._sandbox is not None:
      return
    await asyncio.sleep(0.05)
    self._sandbox = object()
    self.sandboxes_created += 1


def test_ensure_initialized_serializes_calls_on_different_event_loops():
  """Calls on separate event loops, as in the live tool thread pool, wait."""
  env = _SandboxEnvironment()

  def _call_on_own_loop() -> None:
    asyncio.run(_utils.ensure_initialized(env, object()))

  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for future in [pool.submit(_call_on_own_loop) for _ in range(4)]:
      future.result()

  assert env.sandboxes_created == 1


@pytest.mark.asyncio
async def test_cancelled_waiter_leaves_the_lock_free():
  """A call cancelled while it waits for the lock does not keep it."""
  env = _SandboxEnvironment()
  holder = asyncio.create_task(_utils.ensure_initialized(env, object()))
  await asyncio.sleep(0)
  waiter = asyncio.create_task(_utils.ensure_initialized(env, object()))
  await asyncio.sleep(0)

  waiter.cancel()
  with pytest.raises(asyncio.CancelledError):
    await waiter
  await holder

  await asyncio.wait_for(_utils.close_if_initialized(env), timeout=1)
  assert env.close_count == 1


def test_lock_map_does_not_keep_environment_alive():
  """The lock map holds environments weakly."""
  env = _CountingEnvironment()
  asyncio.run(_utils.ensure_initialized(env, object()))
  assert env in _utils._env_locks
  env_ref = weakref.ref(env)

  del env

  assert env_ref() is None
