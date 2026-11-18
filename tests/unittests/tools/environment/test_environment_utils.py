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
import types
from typing import Any
from typing import Optional
import weakref

from google.adk.environment._base_environment import BaseEnvironment
from google.adk.environment._base_environment import ExecutionResult
from google.adk.environment._environment_scope import EnvironmentScope
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


def _ctx(
    *, user_id: str = 'user', session_id: str = 'session', app_name: str = 'app'
) -> Any:
  """A stand-in for a tool context of the given session."""
  return types.SimpleNamespace(
      session=types.SimpleNamespace(
          app_name=app_name, user_id=user_id, id=session_id
      )
  )


class _ScopedEnvironment(_CountingEnvironment):
  """Records which contexts are inside `initialize` at the same time."""

  def __init__(self, scope: EnvironmentScope) -> None:
    super().__init__()
    self._scope = scope
    self._inside: list[Any] = []
    self.overlaps: set[frozenset[int]] = set()

  @property
  def scope(self) -> EnvironmentScope:
    return self._scope

  async def initialize(self, *, ctx: Any = None) -> None:
    for other in self._inside:
      self.overlaps.add(frozenset((id(other), id(ctx))))
    self._inside.append(ctx)
    try:
      # Long enough for every concurrent call to reach its lock.
      await asyncio.sleep(0.05)
    finally:
      self._inside.remove(ctx)

  def overlapped(self, first: Any, second: Any) -> bool:
    return frozenset((id(first), id(second))) in self.overlaps


async def test_shared_scope_serializes_calls_of_different_sessions():
  """A shared environment initializes one call at a time, whatever the session."""
  env = _ScopedEnvironment(EnvironmentScope.SHARED)
  first = _ctx(user_id='alice', session_id='s1')
  second = _ctx(user_id='bob', session_id='s2')

  await asyncio.gather(
      _utils.ensure_initialized(env, first),
      _utils.ensure_initialized(env, second),
  )

  assert not env.overlaps


async def test_session_scope_serializes_calls_of_one_session():
  """Concurrent calls of one session do not initialize together."""
  env = _ScopedEnvironment(EnvironmentScope.SESSION)
  first = _ctx(session_id='s1')
  second = _ctx(session_id='s1')

  await asyncio.gather(
      _utils.ensure_initialized(env, first),
      _utils.ensure_initialized(env, second),
  )

  assert not env.overlaps


async def test_session_scope_initializes_different_sessions_concurrently():
  """One session's initialize does not make another session wait."""
  env = _ScopedEnvironment(EnvironmentScope.SESSION)
  first = _ctx(session_id='s1')
  second = _ctx(session_id='s2')

  await asyncio.gather(
      _utils.ensure_initialized(env, first),
      _utils.ensure_initialized(env, second),
  )

  assert env.overlapped(first, second)


async def test_session_scope_tells_apart_equal_session_ids_of_other_users():
  """Sessions with the same id under different users or apps are distinct."""
  env = _ScopedEnvironment(EnvironmentScope.SESSION)
  first = _ctx(user_id='alice', session_id='s1')
  other_user = _ctx(user_id='bob', session_id='s1')
  other_app = _ctx(user_id='alice', session_id='s1', app_name='other_app')

  await asyncio.gather(
      _utils.ensure_initialized(env, first),
      _utils.ensure_initialized(env, other_user),
      _utils.ensure_initialized(env, other_app),
  )

  assert env.overlapped(first, other_user)
  assert env.overlapped(first, other_app)


async def test_user_scope_serializes_sessions_of_one_user():
  """Two sessions of one user share a resource, so they do not overlap."""
  env = _ScopedEnvironment(EnvironmentScope.USER)
  first = _ctx(user_id='alice', session_id='s1')
  second = _ctx(user_id='alice', session_id='s2')

  await asyncio.gather(
      _utils.ensure_initialized(env, first),
      _utils.ensure_initialized(env, second),
  )

  assert not env.overlaps


async def test_user_scope_initializes_different_users_concurrently():
  """One user's initialize does not make another user wait."""
  env = _ScopedEnvironment(EnvironmentScope.USER)
  first = _ctx(user_id='alice')
  second = _ctx(user_id='bob')

  await asyncio.gather(
      _utils.ensure_initialized(env, first),
      _utils.ensure_initialized(env, second),
  )

  assert env.overlapped(first, second)


def test_session_scope_serializes_one_session_on_different_event_loops():
  """Calls of one session on separate event loops still run one at a time."""
  env = _ScopedEnvironment(EnvironmentScope.SESSION)
  contexts = [_ctx(session_id='s1') for _ in range(4)]

  def _call_on_own_loop(ctx: Any) -> None:
    asyncio.run(_utils.ensure_initialized(env, ctx))

  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for future in [pool.submit(_call_on_own_loop, ctx) for ctx in contexts]:
      future.result()

  assert not env.overlaps


@pytest.mark.parametrize(
    'scope', [EnvironmentScope.USER, EnvironmentScope.SESSION]
)
@pytest.mark.parametrize('ctx', [None, types.SimpleNamespace(session=None)])
async def test_scoped_environment_rejects_a_context_without_a_session(
    scope: EnvironmentScope, ctx: Any
):
  """A scoped environment needs a session to know whose resource to set up."""
  env = _ScopedEnvironment(scope)

  with pytest.raises(ValueError, match=f'has the {scope.value} scope'):
    await _utils.ensure_initialized(env, ctx)

  assert not env.is_initialized


async def test_shared_environment_accepts_a_context_without_a_session():
  """A shared environment does not read the session."""
  env = _ScopedEnvironment(EnvironmentScope.SHARED)

  await _utils.ensure_initialized(env, types.SimpleNamespace(session=None))

  assert env.is_initialized


async def test_scoped_locks_are_released_when_no_call_uses_them():
  """An environment keeps no lock for a session once its calls are done."""
  env = _ScopedEnvironment(EnvironmentScope.SESSION)

  await asyncio.gather(*(
      _utils.ensure_initialized(env, _ctx(session_id=f's{i}')) for i in range(5)
  ))

  assert not _utils._env_locks[env]


async def test_cancelled_waiter_of_a_session_leaves_no_lock_behind():
  """A call cancelled while it waits for its session's lock keeps nothing."""
  env = _ScopedEnvironment(EnvironmentScope.SESSION)
  holder = asyncio.create_task(
      _utils.ensure_initialized(env, _ctx(session_id='s1'))
  )
  await asyncio.sleep(0)
  waiter = asyncio.create_task(
      _utils.ensure_initialized(env, _ctx(session_id='s1'))
  )
  await asyncio.sleep(0)

  waiter.cancel()
  with pytest.raises(asyncio.CancelledError):
    await waiter
  await holder

  assert not _utils._env_locks[env]
  await asyncio.wait_for(
      _utils.ensure_initialized(env, _ctx(session_id='s1')), timeout=1
  )


async def test_scoped_lock_is_released_when_initialize_fails():
  """A failed initialize lets a call that waits for the same session run.

  Setup: an environment whose first initialize fails after a second call for
    the same session has started to wait for the lock.
  Act: let the first call fail.
  Assert: the first call raises, the waiting call completes, and no lock is
    left behind.
  """

  class _FailingOnce(_ScopedEnvironment):

    def __init__(self) -> None:
      super().__init__(EnvironmentScope.SESSION)
      self.calls = 0
      self.first_call_running = asyncio.Event()
      self.fail_first_call = asyncio.Event()

    async def initialize(self, *, ctx: Any = None) -> None:
      self.calls += 1
      if self.calls == 1:
        self.first_call_running.set()
        await self.fail_first_call.wait()
        raise RuntimeError('initialize failed')

  env = _FailingOnce()
  failing = asyncio.create_task(
      _utils.ensure_initialized(env, _ctx(session_id='s1'))
  )
  await env.first_call_running.wait()
  waiting = asyncio.create_task(
      _utils.ensure_initialized(env, _ctx(session_id='s1'))
  )
  # Let the second call reach the lock the first call holds.
  await asyncio.sleep(0.05)
  assert not waiting.done()

  env.fail_first_call.set()

  with pytest.raises(RuntimeError, match='initialize failed'):
    await failing
  await asyncio.wait_for(waiting, timeout=1)
  assert env.calls == 2
  assert not _utils._env_locks[env]


class _BlockingInitEnvironment(_ScopedEnvironment):
  """Blocks in `initialize` until released, and records the call order."""

  def __init__(self, scope: EnvironmentScope) -> None:
    super().__init__(scope)
    self.log: list[str] = []
    self.initialize_running = asyncio.Event()
    self.release_initialize = asyncio.Event()

  async def initialize(self, *, ctx: Any = None) -> None:
    self.log.append('initialize started')
    self.initialize_running.set()
    await self.release_initialize.wait()
    self.log.append('initialize finished')

  async def close(self) -> None:
    await super().close()
    self.log.append('close')


@pytest.mark.parametrize(
    'scope',
    [EnvironmentScope.SHARED, EnvironmentScope.USER, EnvironmentScope.SESSION],
)
async def test_close_waits_for_an_initialize_in_flight(
    scope: EnvironmentScope,
):
  """A close during a first initialize closes what that initialize set up.

  Setup: the first initialize of an environment is running, so the
    environment is not marked initialized yet.
  Act: close the environment, then let the initialize finish.
  Assert: the close does not return early; it runs after the initialize and
    leaves the environment uninitialized.
  """
  env = _BlockingInitEnvironment(scope)
  initializing = asyncio.create_task(_utils.ensure_initialized(env, _ctx()))
  await env.initialize_running.wait()

  closing = asyncio.create_task(_utils.close_if_initialized(env))
  await asyncio.sleep(0.05)
  assert not closing.done()
  env.release_initialize.set()
  await asyncio.gather(initializing, closing)

  assert env.log == ['initialize started', 'initialize finished', 'close']
  assert not env.is_initialized


async def test_initialize_of_another_session_waits_for_a_close():
  """A close that has started keeps a new session's initialize out.

  Setup: session s1 is initializing and a close waits for it.
  Act: session s2 calls initialize, then s1 is released.
  Assert: s2 initializes only after the close.
  """
  env = _BlockingInitEnvironment(EnvironmentScope.SESSION)
  first = asyncio.create_task(
      _utils.ensure_initialized(env, _ctx(session_id='s1'))
  )
  await env.initialize_running.wait()
  closing = asyncio.create_task(_utils.close_if_initialized(env))
  await asyncio.sleep(0.05)

  second = asyncio.create_task(
      _utils.ensure_initialized(env, _ctx(session_id='s2'))
  )
  await asyncio.sleep(0.05)
  started_before_close = env.log.count('initialize started')
  env.release_initialize.set()
  await asyncio.gather(first, closing, second)

  assert started_before_close == 1
  assert env.log == [
      'initialize started',
      'initialize finished',
      'close',
      'initialize started',
      'initialize finished',
  ]
  assert env.is_initialized


async def test_cancelled_close_lets_initialize_calls_run_again():
  """A close cancelled while it waits does not keep later calls out."""
  env = _BlockingInitEnvironment(EnvironmentScope.SESSION)
  first = asyncio.create_task(
      _utils.ensure_initialized(env, _ctx(session_id='s1'))
  )
  await env.initialize_running.wait()
  closing = asyncio.create_task(_utils.close_if_initialized(env))
  await asyncio.sleep(0.05)

  closing.cancel()
  with pytest.raises(asyncio.CancelledError):
    await closing
  env.release_initialize.set()
  await first

  await asyncio.wait_for(
      _utils.ensure_initialized(env, _ctx(session_id='s2')), timeout=1
  )
  assert env.close_count == 0
  assert env.is_initialized


async def test_close_runs_after_a_failed_initialize():
  """A failed initialize does not keep a later close waiting."""
  env = _CountingEnvironment(fail_first_init=True)
  with pytest.raises(RuntimeError, match='initialize failed'):
    await _utils.ensure_initialized(env, object())
  await _utils.ensure_initialized(env, object())

  await asyncio.wait_for(_utils.close_if_initialized(env), timeout=1)

  assert env.close_count == 1


async def test_close_runs_after_a_cancelled_initialize():
  """An initialize cancelled while it runs does not keep a close waiting."""
  env = _BlockingInitEnvironment(EnvironmentScope.SESSION)
  initializing = asyncio.create_task(_utils.ensure_initialized(env, _ctx()))
  await env.initialize_running.wait()
  initializing.cancel()
  with pytest.raises(asyncio.CancelledError):
    await initializing

  await asyncio.wait_for(_utils.close_if_initialized(env), timeout=1)

  assert env.log == ['initialize started']
  assert not env.is_initialized


@pytest.mark.parametrize(
    'scope', ['SESSION', None, object()], ids=['wrong_string', 'none', 'object']
)
async def test_unknown_scope_value_locks_the_environment_as_a_whole(
    scope: Any,
):
  """A scope that is not USER or SESSION is locked like SHARED."""
  env = _ScopedEnvironment(scope)

  await asyncio.gather(
      _utils.ensure_initialized(env, _ctx(session_id='s1')),
      _utils.ensure_initialized(env, _ctx(session_id='s2')),
  )

  assert not env.overlaps


async def test_scope_given_as_its_string_value_still_needs_a_session():
  """A scope returned as the plain string 'session' is the SESSION scope."""
  env = _ScopedEnvironment('session')

  with pytest.raises(ValueError, match='has the session scope'):
    await _utils.ensure_initialized(env, None)


async def test_unknown_scope_value_accepts_a_context_without_a_session():
  """A scope that is not USER or SESSION does not need a session."""
  env = _ScopedEnvironment(None)

  await _utils.ensure_initialized(env, object())

  assert env.is_initialized


async def test_error_reading_the_session_is_not_reported_as_a_missing_session():
  """An AttributeError from `ctx.session` reaches the caller unchanged."""

  class _BrokenContext:

    @property
    def session(self) -> Any:
      raise AttributeError('no invocation context')

  env = _ScopedEnvironment(EnvironmentScope.SESSION)

  with pytest.raises(AttributeError, match='no invocation context'):
    await _utils.ensure_initialized(env, _BrokenContext())
