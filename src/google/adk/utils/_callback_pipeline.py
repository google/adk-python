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

from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Sequence
from contextlib import nullcontext
import inspect
from typing import Any
from typing import TypeVar

from typing_extensions import ParamSpec

_P = ParamSpec('_P')
_TCallback = TypeVar('_TCallback', bound=Callable[..., object])
_TResult = TypeVar('_TResult')


_CANONICAL_PARAM_ORDER: tuple[str, ...] = (
    'tool',
    'args',
    'tool_context',
    'response',
    'tool_response',
    'callback_context',
    'llm_request',
    'llm_response',
    'error',
)


def _canonical_order_key(name: str) -> int:
  try:
    return _CANONICAL_PARAM_ORDER.index(name)
  except ValueError:
    return len(_CANONICAL_PARAM_ORDER)


def _invoke_callback(
    _callback: Callable[..., Any], /, *args: Any, **kwargs: Any
) -> Any:
  """Invokes callback, adapting keyword arguments to positional if needed."""
  if kwargs:
    use_pos = False
    pos_args: tuple[Any, ...] = ()
    try:
      sig = inspect.signature(_callback)
      try:
        sig.bind(*args, **kwargs)
      except TypeError:
        sorted_kwarg_values = [
            val
            for _, val in sorted(
                kwargs.items(), key=lambda item: _canonical_order_key(item[0])
            )
        ]
        pos_args = args + tuple(sorted_kwarg_values)
        sig.bind(*pos_args)
        use_pos = True
    except (ValueError, TypeError):
      pass

    if use_pos:
      return _callback(*pos_args)

  return _callback(*args, **kwargs)


async def _run_callbacks(
    callbacks: Sequence[
        Callable[
            _P,
            Awaitable[_TResult | None] | _TResult | None,
        ]
    ],
    _stop_condition: Callable[[_TResult | None], bool],
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> _TResult | None:
  """Runs callbacks in order while preserving their stop semantics.

  Experimental telemetry records a child span for each executed callback.
  """
  from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE

  from ..agents.context import Context
  from ..telemetry import tracing

  if not callbacks:
    return None
  callback_context = kwargs.get('callback_context') or kwargs.get(
      'tool_context'
  )
  invocation_context = (
      callback_context.get_invocation_context()
      if isinstance(callback_context, Context)
      else None
  )
  telemetry_config = tracing._telemetry_config_from_invocation_context(
      invocation_context
  )
  result: _TResult | None = None
  for callback in callbacks:
    callback_name = getattr(
        callback, '__qualname__', type(callback).__qualname__
    )
    scope = (
        tracing.tracer.start_as_current_span(
            f'execute_callback {callback_name}'
        )
        if telemetry_config.should_emit_experimental_telemetry
        else nullcontext()
    )
    with scope as span:
      try:
        callback_result = _invoke_callback(callback, *args, **kwargs)
        if inspect.isawaitable(callback_result):
          result = await callback_result
        else:
          result = callback_result
      except Exception as error:
        if span is not None:
          span.set_attribute(ERROR_TYPE, tracing.resolve_error_type(error))
        raise
    if _stop_condition(result):
      return result
  return result


def _normalize_callbacks(
    callback: _TCallback | list[_TCallback] | None,
) -> list[_TCallback]:
  """Normalizes an optional callback or callback list to a list."""
  if callback is None:
    return []
  if isinstance(callback, list):
    return callback
  return [callback]


def _stop_on_truthy(result: object | None) -> bool:
  """Returns whether a callback produced a truthy result."""
  return bool(result)


def _stop_on_non_none(result: object | None) -> bool:
  """Returns whether a callback produced a non-None result."""
  return result is not None
