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


@pytest.mark.asyncio
async def test_callback_adapts_keyword_args_to_positional_params():
  """Callbacks with positional param names accept keyword arguments."""

  def callback(ctx: str, resp: str) -> str:
    return f'{ctx}:{resp}'

  result = await _run_callbacks(
      [callback],
      _stop_on_truthy,
      callback_context='context_val',
      llm_response='response_val',
  )

  assert result == 'context_val:response_val'


@pytest.mark.asyncio
async def test_async_callback_adapts_keyword_args_to_positional_params():
  """Async callbacks with positional param names accept keyword arguments."""

  async def callback(a: str, b: str) -> str:
    return f'{a}:{b}'

  result = await _run_callbacks(
      [callback],
      _stop_on_truthy,
      callback_context='arg_a',
      llm_response='arg_b',
  )

  assert result == 'arg_a:arg_b'


@pytest.mark.asyncio
async def test_lambda_callback_adapts_keyword_args():
  """Lambda callbacks with non-matching param names accept keyword arguments."""
  callback = lambda c, r: f'lambda_{c}_{r}'

  result = await _run_callbacks(
      [callback],
      _stop_on_truthy,
      callback_context='c1',
      llm_response='r1',
  )

  assert result == 'lambda_c1_r1'


@pytest.mark.asyncio
async def test_callback_with_body_type_error_is_not_masked():
  """A TypeError raised inside the callback body is preserved."""

  def callback(ctx: str, resp: str) -> None:
    raise TypeError('error inside callback body')

  with pytest.raises(TypeError, match='error inside callback body'):
    await _run_callbacks(
        [callback],
        _stop_on_truthy,
        callback_context='ctx',
        llm_response='resp',
    )


@pytest.mark.asyncio
async def test_callback_with_reordered_kwargs_maintains_canonical_order():
  """Reordering kwargs at call site still binds positional parameters in canonical order."""
  callback = lambda ctx, resp: f'{ctx}:{resp}'

  result = await _run_callbacks(
      [callback],
      _stop_on_truthy,
      llm_response='resp_val',
      callback_context='ctx_val',
  )

  assert result == 'ctx_val:resp_val'


@pytest.mark.asyncio
async def test_callback_with_callback_kwarg_does_not_collide():
  """A kwarg named 'callback' does not collide with _callback parameter."""
  callback = lambda c, cb: f'{c}:{cb}'

  result = await _run_callbacks(
      [callback],
      _stop_on_truthy,
      callback_context='ctx_val',
      callback='custom_callback',
  )

  assert result == 'ctx_val:custom_callback'


@pytest.mark.asyncio
async def test_callback_with_incompatible_signature_raises_type_error():
  """When both keyword and positional binding fail, original TypeError is raised."""

  def callback(a: str, b: str, c: str, d: str) -> None:
    pass

  with pytest.raises(TypeError):
    await _run_callbacks(
        [callback],
        _stop_on_truthy,
        callback_context='ctx',
        llm_response='resp',
    )


@pytest.mark.asyncio
async def test_before_tool_callback_with_reordered_kwargs_maintains_canonical_order():
  """Tool callbacks receive (tool, args, tool_context) in canonical order even if kwargs are reordered."""
  callback = lambda t, a, tc: f'{t}:{a}:{tc}'

  result = await _run_callbacks(
      [callback],
      _stop_on_non_none,
      tool_context='ctx_val',
      args='args_val',
      tool='tool_val',
  )

  assert result == 'tool_val:args_val:ctx_val'


@pytest.mark.asyncio
async def test_after_tool_callback_with_reordered_kwargs_maintains_canonical_order():
  """After-tool callbacks receive (tool, args, tool_context, response) in canonical order."""
  callback = lambda t, a, tc, r: f'{t}:{a}:{tc}:{r}'

  result = await _run_callbacks(
      [callback],
      _stop_on_non_none,
      response='resp_val',
      tool_context='ctx_val',
      args='args_val',
      tool='tool_val',
  )

  assert result == 'tool_val:args_val:ctx_val:resp_val'


@pytest.mark.asyncio
async def test_after_tool_callback_with_tool_response_kwarg_maintains_canonical_order():
  """After-tool callbacks with tool_response kwarg receive parameters in canonical order."""
  callback = lambda t, a, tc, tr: f'{t}:{a}:{tc}:{tr}'

  result = await _run_callbacks(
      [callback],
      _stop_on_non_none,
      tool_response='tr_val',
      tool_context='ctx_val',
      args='args_val',
      tool='tool_val',
  )

  assert result == 'tool_val:args_val:ctx_val:tr_val'


@pytest.mark.asyncio
async def test_on_tool_error_callback_with_reordered_kwargs_maintains_canonical_order():
  """Tool error callbacks receive (tool, args, tool_context, error) in canonical order."""
  callback = lambda t, a, tc, err: f'{t}:{a}:{tc}:{err}'

  result = await _run_callbacks(
      [callback],
      _stop_on_non_none,
      error='err_val',
      tool_context='ctx_val',
      args='args_val',
      tool='tool_val',
  )

  assert result == 'tool_val:args_val:ctx_val:err_val'


@pytest.mark.asyncio
async def test_on_model_error_callback_with_reordered_kwargs_maintains_canonical_order():
  """Model error callbacks receive (callback_context, llm_request, error) in canonical order."""
  callback = lambda ctx, req, err: f'{ctx}:{req}:{err}'

  result = await _run_callbacks(
      [callback],
      _stop_on_truthy,
      error='err_val',
      llm_request='req_val',
      callback_context='ctx_val',
  )

  assert result == 'ctx_val:req_val:err_val'
