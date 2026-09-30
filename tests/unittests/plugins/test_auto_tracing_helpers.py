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

"""Unit tests for the AutoTracingPlugin helper functions."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from typing import Any
from typing import Iterator

from google.adk.plugins import auto_tracing_helpers
from opentelemetry import trace as trace_api
import pytest

_CAPS = auto_tracing_helpers.Caps()


class _FakeSpan:
  """Minimal span recording the attributes written to it."""

  def __init__(self, recording: bool = True):
    self._recording = recording
    self.attributes: dict[str, Any] = {}

  def is_recording(self) -> bool:
    return self._recording

  def set_attribute(self, key: str, value: Any) -> None:
    self.attributes[key] = value


class _FakeTracer:
  """A recording tracer (deliberately not a NoOpTracer) handing out one span."""

  def __init__(self, span: _FakeSpan):
    self.span = span
    self.span_names: list[str] = []

  @contextlib.contextmanager
  def start_as_current_span(self, name: str) -> Iterator[_FakeSpan]:
    self.span_names.append(name)
    yield self.span


def _module_level_fn(x: int) -> int:
  return x


class _Holder:

  def method(self) -> None:
    return None


def _sync_shape(x: int) -> int:
  return x


async def _coroutine_shape(x: int) -> int:
  return x


def _generator_shape(x: int) -> Iterator[int]:
  yield x


async def _async_generator_shape(x: int):
  yield x


def _callable_shape(fn: Any) -> str:
  if inspect.isasyncgenfunction(fn):
    return 'asyncgen'
  if asyncio.iscoroutinefunction(fn):
    return 'coroutine'
  if inspect.isgeneratorfunction(fn):
    return 'generator'
  return 'sync'


def test_public_slot_names_string_shorthand_is_one_name():
  """``__slots__ = "child"`` declares one slot, not five one-letter slots."""
  cls = type('_Shorthand', (), {'__slots__': 'child'})

  assert auto_tracing_helpers.public_slot_names(cls) == {'child'}


def test_public_slot_names_unions_mro_and_drops_underscored():
  base = type('_Base', (), {'__slots__': ('shared', '_private')})
  sub = type('_Sub', (base,), {'__slots__': ('own',)})

  assert auto_tracing_helpers.public_slot_names(sub) == {'shared', 'own'}


def test_public_slot_names_without_slots_is_empty():
  cls = type('_Plain', (), {})

  assert auto_tracing_helpers.public_slot_names(cls) == set()


def test_positional_param_names_keeps_only_positional_kinds():
  def fn(pos_only, /, normal, *args, kw_only=None, **kwargs):
    del pos_only, normal, args, kw_only, kwargs

  assert auto_tracing_helpers.positional_param_names(fn) == (
      'pos_only',
      'normal',
  )


def test_positional_param_names_empty_when_not_introspectable():
  # A plain instance is not callable, so ``inspect.signature`` raises and the
  # helper must degrade to "no names" rather than propagate.
  assert auto_tracing_helpers.positional_param_names(object()) == ()


def test_name_value_pairs_skips_self_and_names_positionals():
  pairs = auto_tracing_helpers.name_value_pairs(
      ('self', 'x', 'y'), (object(), 1, 'a'), {}, _CAPS
  )

  assert pairs == [('x', '1'), ('y', "'a'")]


def test_name_value_pairs_falls_back_to_index_names_for_extra_args():
  pairs = auto_tracing_helpers.name_value_pairs(('x',), (1, 2, 3), {}, _CAPS)

  assert pairs == [('x', '1'), ('arg1', '2'), ('arg2', '3')]


def test_name_value_pairs_appends_kwargs_after_positionals():
  pairs = auto_tracing_helpers.name_value_pairs(
      ('x',), (1,), {'flag': True, 'note': 'hi'}, _CAPS
  )

  assert pairs == [('x', '1'), ('flag', 'True'), ('note', "'hi'")]


def test_name_value_pairs_caps_long_reprs():
  caps = auto_tracing_helpers.Caps(max_repr_len=5)

  pairs = auto_tracing_helpers.name_value_pairs(('x',), ('y' * 10,), {}, caps)

  # repr() of the value is "'yyyyyyyyyy'" -- 12 chars, so 7 are dropped.
  assert pairs == [('x', "'yyyy...[7 more chars]")]


def test_record_io_on_span_writes_args_and_return():
  span = _FakeSpan()

  auto_tracing_helpers.record_io_on_span(span, [('x', '1')], 'ok', None, _CAPS)

  assert span.attributes == {
      'adk.fn.arg.x': '1',
      'adk.fn.return': "'ok'",
  }


def test_record_io_on_span_records_exception_instead_of_return():
  span = _FakeSpan()

  auto_tracing_helpers.record_io_on_span(
      span, [('x', '1')], 'unused', ValueError('boom'), _CAPS
  )

  assert span.attributes['adk.fn.arg.x'] == '1'
  assert span.attributes['adk.fn.exc_type'] == 'ValueError'
  assert 'boom' in span.attributes['adk.fn.exc_repr']
  # A raising call has no return value to record.
  assert 'adk.fn.return' not in span.attributes


@pytest.mark.parametrize(
    'fn,expected',
    [
        (_module_level_fn, '_module_level_fn'),
        (_Holder.method, '_Holder.method'),
    ],
)
def test_display_name_for_keeps_owner_and_name(fn, expected):
  assert auto_tracing_helpers.display_name_for(fn) == expected


def test_stream_result_repr_for_empty_stream():
  result = auto_tracing_helpers.StreamResult([], _CAPS, 0)

  assert repr(result) == '<generator: 0 items yielded>'


def test_stream_result_repr_reports_total_beyond_sample():
  result = auto_tracing_helpers.StreamResult([1, 2], _CAPS, 5)

  assert repr(result) == (
      '<generator: 5 items yielded; first 2: [1, 2] ... + 3 more>'
  )


def test_stream_result_repr_has_no_more_suffix_when_fully_sampled():
  result = auto_tracing_helpers.StreamResult([1, 2], _CAPS, 2)

  assert repr(result) == '<generator: 2 items yielded; first 2: [1, 2]>'


def test_build_tracing_wrapper_returns_original_for_noop_tracer():
  wrapped = auto_tracing_helpers.build_tracing_wrapper(
      _sync_shape, trace_api.NoOpTracer(), _CAPS
  )

  assert wrapped is _sync_shape
  assert not hasattr(_sync_shape, auto_tracing_helpers.WRAPPED_ATTR)


@pytest.mark.parametrize(
    'fn,expected_shape',
    [
        (_sync_shape, 'sync'),
        (_coroutine_shape, 'coroutine'),
        (_generator_shape, 'generator'),
        (_async_generator_shape, 'asyncgen'),
    ],
)
def test_build_tracing_wrapper_preserves_callable_shape(fn, expected_shape):
  wrapped = auto_tracing_helpers.build_tracing_wrapper(
      fn, _FakeTracer(_FakeSpan()), _CAPS
  )

  assert _callable_shape(wrapped) == expected_shape
  assert getattr(wrapped, auto_tracing_helpers.WRAPPED_ATTR) is True
  assert wrapped.__name__ == fn.__name__


def test_build_tracing_wrapper_records_io_under_the_display_name():
  span = _FakeSpan()
  tracer = _FakeTracer(span)

  def add_one(x: int) -> int:
    return x + 1

  wrapped = auto_tracing_helpers.build_tracing_wrapper(add_one, tracer, _CAPS)

  assert wrapped(3) == 4
  assert tracer.span_names == [auto_tracing_helpers.display_name_for(add_one)]
  assert span.attributes == {'adk.fn.arg.x': '3', 'adk.fn.return': '4'}


async def test_build_tracing_wrapper_records_awaited_result():
  span = _FakeSpan()

  async def double(x: int) -> int:
    return x * 2

  wrapped = auto_tracing_helpers.build_tracing_wrapper(
      double, _FakeTracer(span), _CAPS
  )

  assert await wrapped(4) == 8
  assert span.attributes == {'adk.fn.arg.x': '4', 'adk.fn.return': '8'}


def test_build_tracing_wrapper_records_nothing_on_non_recording_span():
  span = _FakeSpan(recording=False)

  def add_one(x: int) -> int:
    return x + 1

  wrapped = auto_tracing_helpers.build_tracing_wrapper(
      add_one, _FakeTracer(span), _CAPS
  )

  assert wrapped(3) == 4
  assert span.attributes == {}
