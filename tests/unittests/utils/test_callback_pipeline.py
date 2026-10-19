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

import asyncio

from google.adk.agents.context import Context
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.run_config import RunConfig
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.telemetry import tracing
from google.adk.telemetry.context import TelemetryConfig
from google.adk.utils._callback_pipeline import _normalize_callbacks
from google.adk.utils._callback_pipeline import _run_callbacks
from google.adk.utils._callback_pipeline import _stop_on_non_none
from google.adk.utils._callback_pipeline import _stop_on_truthy
from google.genai import errors as genai_errors
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
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


@pytest.fixture
def spans(monkeypatch):
  exporter = InMemorySpanExporter()
  provider = TracerProvider()
  provider.add_span_processor(SimpleSpanProcessor(exporter))
  monkeypatch.setattr(tracing, 'tracer', provider.get_tracer('callbacks'))
  monkeypatch.setenv('ADK_EXPERIMENTAL_TELEMETRY', 'true')
  yield exporter
  provider.shutdown()


async def test_callback_span_covers_the_awaited_result(spans):
  """The callback's full execution belongs to a child of the caller span."""
  seen = []

  async def callback():
    seen.append(trace.get_current_span().get_span_context().span_id)
    await asyncio.sleep(0)
    seen.append(trace.get_current_span().get_span_context().span_id)
    return 'answer'

  with tracing.tracer.start_as_current_span('parent') as parent:
    result = await _run_callbacks([callback], _stop_on_truthy)
    assert trace.get_current_span() is parent

  child, parent_span = spans.get_finished_spans()
  assert result == 'answer'
  assert child.name == f'execute_callback {callback.__qualname__}'
  assert seen == [child.context.span_id, child.context.span_id]
  assert child.parent.span_id == parent_span.context.span_id
  assert not child.attributes


@pytest.mark.parametrize('telemetry_enabled', [False, True])
@pytest.mark.parametrize('is_async', [False, True], ids=['sync', 'async'])
@pytest.mark.parametrize(
    'error,expected_error_type',
    [
        (ValueError('callback failed'), 'ValueError'),
        (genai_errors.ClientError(429, {'message': 'quota exceeded'}), '429'),
    ],
)
async def test_callback_error_is_recorded_and_propagated(
    spans, monkeypatch, telemetry_enabled, is_async, error, expected_error_type
):
  """Tracing records a failure without replacing the callback's exception."""
  monkeypatch.setenv('ADK_EXPERIMENTAL_TELEMETRY', str(telemetry_enabled))

  def callback():
    raise error

  async def async_callback():
    raise error

  with pytest.raises(type(error)) as caught:
    await _run_callbacks(
        [async_callback if is_async else callback], _stop_on_truthy
    )

  assert caught.value is error
  finished_spans = spans.get_finished_spans()
  assert len(finished_spans) == int(telemetry_enabled)
  if telemetry_enabled:
    (span,) = finished_spans
    assert span.status.status_code == StatusCode.ERROR
    assert span.events[0].name == 'exception'
    assert span.attributes['error.type'] == expected_error_type


@pytest.mark.parametrize('stop_condition', [_stop_on_truthy, _stop_on_non_none])
async def test_only_executed_callbacks_have_spans(spans, stop_condition):
  """Returning an override prevents later callbacks from running or tracing."""

  def first():
    return 'override'

  def later():
    raise AssertionError('later callback ran')

  assert await _run_callbacks([first, later], stop_condition) == 'override'
  assert [s.name for s in spans.get_finished_spans()] == [
      f'execute_callback {first.__qualname__}'
  ]


@pytest.mark.parametrize(
    'env_enabled,request_enabled,expected_count',
    [(False, None, 0), (True, False, 0), (False, True, 1), (True, None, 1)],
)
async def test_request_telemetry_setting_controls_callback_spans(
    spans, monkeypatch, env_enabled, request_enabled, expected_count
):
  """The request override takes precedence over the experimental env setting."""
  monkeypatch.setenv('ADK_EXPERIMENTAL_TELEMETRY', str(env_enabled))
  service = InMemorySessionService()
  session = await service.create_session(app_name='app', user_id='user')
  ctx = Context(
      InvocationContext(
          invocation_id='invocation',
          session=session,
          session_service=service,
          run_config=RunConfig(
              telemetry=TelemetryConfig(
                  adk_experimental_telemetry_opt_in=request_enabled
              )
          ),
      )
  )

  def callback(callback_context):
    assert callback_context is ctx
    return 'answer'

  assert (
      await _run_callbacks([callback], _stop_on_truthy, callback_context=ctx)
      == 'answer'
  )
  assert len(spans.get_finished_spans()) == expected_count


async def test_cancelling_callback_ends_its_span(spans):
  """Cancellation reaches the callback and ends its span without an error."""
  started = asyncio.Event()
  cleaned_up = []

  async def callback():
    try:
      started.set()
      await asyncio.Event().wait()
    finally:
      cleaned_up.append(True)

  task = asyncio.create_task(_run_callbacks([callback], _stop_on_truthy))
  await started.wait()
  task.cancel()
  with pytest.raises(asyncio.CancelledError):
    await task

  assert cleaned_up == [True]
  (span,) = spans.get_finished_spans()
  assert span.end_time is not None
  assert span.status.status_code == StatusCode.UNSET
  assert 'error.type' not in span.attributes


async def test_callable_object_has_a_named_span(spans):
  """A callable instance does not need a function's qualname to be traced."""

  class Callback:

    def __call__(self):
      return 'answer'

  assert await _run_callbacks([Callback()], _stop_on_truthy) == 'answer'
  (span,) = spans.get_finished_spans()
  assert span.name == f'execute_callback {Callback.__qualname__}'


async def test_nested_callback_spans_preserve_the_parent(spans):
  """An inner pipeline is traced beneath the callback that invoked it."""

  def inner():
    return 'answer'

  async def outer():
    return await _run_callbacks([inner], _stop_on_truthy)

  with tracing.tracer.start_as_current_span('parent'):
    assert await _run_callbacks([outer], _stop_on_truthy) == 'answer'

  inner_span, outer_span, parent_span = spans.get_finished_spans()
  assert inner_span.parent.span_id == outer_span.context.span_id
  assert outer_span.parent.span_id == parent_span.context.span_id


async def test_concurrent_callbacks_keep_separate_spans(spans):
  """Awaiting another callback does not attach its span to this task."""
  first_started = asyncio.Event()
  second_started = asyncio.Event()
  seen = {}

  async def first():
    before = trace.get_current_span().get_span_context().span_id
    first_started.set()
    await second_started.wait()
    seen['first'] = (
        before,
        trace.get_current_span().get_span_context().span_id,
    )

  async def second():
    await first_started.wait()
    second_started.set()
    seen['second'] = trace.get_current_span().get_span_context().span_id

  with tracing.tracer.start_as_current_span('parent'):
    await asyncio.gather(
        _run_callbacks([first], _stop_on_truthy),
        _run_callbacks([second], _stop_on_truthy),
    )

  callback_spans = [s for s in spans.get_finished_spans() if s.name != 'parent']
  assert len(callback_spans) == 2
  assert callback_spans[0].parent.span_id == callback_spans[1].parent.span_id
  assert seen['first'][0] == seen['first'][1]
  assert seen['first'][0] != seen['second']
