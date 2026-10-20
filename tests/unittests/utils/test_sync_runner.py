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

"""Unit tests for _sync_runner module."""

from __future__ import annotations

import functools
from typing import Any

from google.adk.utils._sync_runner import _invoke_callable
from google.adk.utils._sync_runner import _SYNC_CALLABLE_RUNNER
from google.adk.utils._sync_runner import _use_sync_callable_runner
import pytest


@pytest.mark.asyncio
async def test_invoke_callable_sync_without_runner() -> None:
  """_invoke_callable runs sync functions inline when no runner is bound."""

  def add(a: int, b: int) -> int:
    return a + b

  assert _SYNC_CALLABLE_RUNNER.get() is None
  result = await _invoke_callable(add, {'a': 2, 'b': 3})
  assert result == 5


@pytest.mark.asyncio
async def test_invoke_callable_sync_uses_bound_runner_and_resets() -> None:
  """_invoke_callable uses the active runner and resets on exit."""
  calls: list[tuple[Any, dict[str, Any]]] = []

  async def recording_runner(
      target: Any, kwargs: dict[str, Any]
  ) -> dict[str, Any]:
    calls.append((target, dict(kwargs)))
    return {'ran_via_runner': target(**kwargs)}

  def multiply(x: int, y: int) -> int:
    return x * y

  with _use_sync_callable_runner(recording_runner):
    outer_result = await _invoke_callable(multiply, {'x': 4, 'y': 5})
    with _use_sync_callable_runner(None):
      cleared_result = await _invoke_callable(multiply, {'x': 2, 'y': 3})
    resumed_result = await _invoke_callable(multiply, {'x': 6, 'y': 7})

  assert outer_result == {'ran_via_runner': 20}
  assert cleared_result == 6
  assert resumed_result == {'ran_via_runner': 42}
  assert len(calls) == 2
  assert _SYNC_CALLABLE_RUNNER.get() is None


@pytest.mark.asyncio
async def test_invoke_callable_async_targets_bypass_sync_runner() -> None:
  """_invoke_callable awaits async callables without using the sync runner."""
  runner_called = False

  async def forbidden_runner(target: Any, kwargs: dict[str, Any]) -> Any:
    nonlocal runner_called
    runner_called = True
    return target(**kwargs)

  async def async_add(a: int, b: int) -> int:
    return a + b

  class AsyncCallable:

    async def __call__(self, base: int, value: int = 0) -> int:
      return (base + value) * 10

  partial_async_callable = functools.partial(AsyncCallable(), 10)

  with _use_sync_callable_runner(forbidden_runner):
    assert await _invoke_callable(async_add, {'a': 1, 'b': 2}) == 3
    assert (
        await _invoke_callable(
            partial_async_callable,
            {'value': 5},
            unwrapped=AsyncCallable.__call__,
        )
        == 150
    )
    assert await _invoke_callable(AsyncCallable(), {'base': 3}) == 30

  assert not runner_called


@pytest.mark.asyncio
async def test_invoke_callable_awaits_awaitable_from_sync_wrapper() -> None:
  """_invoke_callable awaits coroutines returned by sync wrapper functions."""

  async def inner_async(x: int) -> int:
    return x * 3

  def sync_wrapper(x: int):
    return inner_async(x)

  assert await _invoke_callable(sync_wrapper, {'x': 4}) == 12

  async def passthrough_runner(target: Any, kwargs: dict[str, Any]) -> Any:
    return target(**kwargs)

  with _use_sync_callable_runner(passthrough_runner):
    assert await _invoke_callable(sync_wrapper, {'x': 5}) == 15
