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
from typing import cast
from typing import TypeVar

from ..agents._callback_metadata import _callback_scope
from ..agents._callback_metadata import _CallbackContext
from ..agents._callback_metadata import CallbackType

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


# Not ParamSpec: PEP 612 forbids the keyword-only callback_type after
# *args: _P.args, and callers pass keyword arguments to positional callback
# types that _invoke_callback adapts, which ParamSpec could not check anyway.
async def _run_callbacks(
    callbacks: Sequence[
        Callable[
            ...,
            Awaitable[_TResult | None] | _TResult | None,
        ]
    ],
    _stop_condition: Callable[[_TResult | None], bool],
    *args: object,
    callback_type: CallbackType | None = None,
    **kwargs: object,
) -> _TResult | None:
  """Runs callbacks in order while preserving their stop semantics."""
  callback_context = kwargs.get('callback_context')
  if callback_context is None:
    callback_context = kwargs.get('tool_context')
  callback_scope = (
      _callback_scope(
          callback_type,
          # callback_context and tool_context kwargs are Context instances.
          cast(_CallbackContext, callback_context),
      )
      if callback_context is not None and callback_type is not None
      else nullcontext()
  )
  with callback_scope:
    result: _TResult | None = None
    for callback in callbacks:
      callback_result = _invoke_callback(callback, *args, **kwargs)
      if inspect.isawaitable(callback_result):
        result = await callback_result
      else:
        result = callback_result
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
