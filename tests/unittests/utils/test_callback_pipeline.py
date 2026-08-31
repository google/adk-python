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

from __future__ import annotations

from google.adk.utils._callback_pipeline import _normalize_callbacks
from google.adk.utils._callback_pipeline import _run_callbacks
from google.adk.utils._callback_pipeline import _stop_on_non_none
from google.adk.utils._callback_pipeline import _stop_on_truthy
import pytest


def test_normalize_none_returns_empty_list():
  """A missing callback normalizes to an empty list."""
  assert _normalize_callbacks(None) == []


def test_normalize_single_callback_returns_single_item_list():
  """A single callback normalizes to a one-item list."""

  def callback() -> None:
    return None

  assert _normalize_callbacks(callback) == [callback]


def test_normalize_callback_list_preserves_identity():
  """A callback list is returned without copying it."""

  def callback() -> None:
    return None

  callbacks = [callback]

  result = _normalize_callbacks(callbacks)

  assert result is callbacks


def test_normalize_empty_callback_list_preserves_identity():
  """An empty callback list is returned without copying it."""
  callbacks = []

  result = _normalize_callbacks(callbacks)

  assert result is callbacks


@pytest.mark.asyncio
async def test_empty_pipeline_returns_none():
  """An empty callback sequence returns None."""
  result = await _run_callbacks([], _stop_on_truthy)

  assert result is None


@pytest.mark.asyncio
async def test_pipeline_runs_sync_callback():
  """A synchronous callback result is returned."""

  def callback() -> str:
    return 'sync result'

  result = await _run_callbacks([callback], _stop_on_truthy)

  assert result == 'sync result'


@pytest.mark.asyncio
async def test_pipeline_runs_async_callback():
  """An asynchronous callback result is awaited and returned."""

  async def callback() -> str:
    return 'async result'

  result = await _run_callbacks([callback], _stop_on_truthy)

  assert result == 'async result'


@pytest.mark.asyncio
async def test_pipeline_runs_mixed_sync_and_async_callbacks():
  """Synchronous and asynchronous callbacks run in their declared order."""
  calls = []

  def sync_callback() -> None:
    calls.append('sync')

  async def async_callback() -> str:
    calls.append('async')
    return 'result'

  result = await _run_callbacks(
      [sync_callback, async_callback],
      _stop_on_truthy,
  )

  assert result == 'result'
  assert calls == ['sync', 'async']


@pytest.mark.asyncio
async def test_pipeline_passes_positional_and_keyword_arguments():
  """Callback arguments are forwarded without modification."""

  def callback(prefix: str, *, value: int) -> str:
    return f'{prefix}-{value}'

  result = await _run_callbacks(
      [callback],
      _stop_on_truthy,
      'item',
      value=3,
  )

  assert result == 'item-3'


@pytest.mark.asyncio
async def test_pipeline_propagates_callback_exception():
  """A callback exception reaches the caller unchanged."""

  def callback() -> None:
    raise ValueError('callback failed')

  with pytest.raises(ValueError, match='callback failed'):
    await _run_callbacks([callback], _stop_on_truthy)


@pytest.mark.asyncio
async def test_truthy_condition_continues_after_empty_dict():
  """A falsy result does not prevent a later truthy callback from running."""
  calls = []

  def empty_callback() -> dict[str, str]:
    calls.append('empty')
    return {}

  def result_callback() -> dict[str, str]:
    calls.append('result')
    return {'source': 'result'}

  result = await _run_callbacks(
      [empty_callback, result_callback],
      _stop_on_truthy,
  )

  assert result == {'source': 'result'}
  assert calls == ['empty', 'result']


@pytest.mark.asyncio
async def test_truthy_condition_returns_final_none_result():
  """The final None result replaces an earlier falsy result."""

  def empty_callback() -> dict[str, str]:
    return {}

  def none_callback() -> None:
    return None

  result = await _run_callbacks(
      [empty_callback, none_callback],
      _stop_on_truthy,
  )

  assert result is None


@pytest.mark.asyncio
async def test_truthy_condition_returns_final_empty_dict():
  """The final empty mapping is returned when no callback is truthy."""

  def none_callback() -> None:
    return None

  def empty_callback() -> dict[str, str]:
    return {}

  result = await _run_callbacks(
      [none_callback, empty_callback],
      _stop_on_truthy,
  )

  assert result == {}


@pytest.mark.asyncio
async def test_truthy_condition_skips_callbacks_after_truthy_result():
  """Callbacks after a truthy result are not invoked."""

  def result_callback() -> str:
    return 'result'

  def unexpected_callback() -> str:
    raise AssertionError('callback should not run')

  result = await _run_callbacks(
      [result_callback, unexpected_callback],
      _stop_on_truthy,
  )

  assert result == 'result'


@pytest.mark.asyncio
async def test_non_none_condition_stops_on_empty_dict():
  """An empty mapping stops callbacks that use non-None semantics."""
  calls = []

  def empty_callback() -> dict[str, str]:
    calls.append('empty')
    return {}

  def unexpected_callback() -> dict[str, str]:
    calls.append('unexpected')
    return {'source': 'unexpected'}

  result = await _run_callbacks(
      [empty_callback, unexpected_callback],
      _stop_on_non_none,
  )

  assert result == {}
  assert calls == ['empty']


@pytest.mark.asyncio
async def test_non_none_condition_continues_after_none():
  """A None result allows the next non-None callback to run."""
  calls = []

  def none_callback() -> None:
    calls.append('none')

  def empty_callback() -> dict[str, str]:
    calls.append('empty')
    return {}

  result = await _run_callbacks(
      [none_callback, empty_callback],
      _stop_on_non_none,
  )

  assert result == {}
  assert calls == ['none', 'empty']
