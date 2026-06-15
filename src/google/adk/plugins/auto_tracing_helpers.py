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

"""AutoTracingPlugin helpers: arg capture, span attrs, tracing wrapper."""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import inspect
import logging
import re
from typing import Any
from typing import Callable
from typing import Sequence

from opentelemetry import trace as trace_api

logger = logging.getLogger("google_adk." + __name__)

DEFAULT_MAX_REPR_LEN = 4096
DEFAULT_MAX_RECORDED_YIELDS = 16

NamedArg = tuple[str, str]
WRAPPED_ATTR = "_adk_auto_tracing_wrapped"
_SELF_OR_CLS = frozenset({"self", "cls"})
_SCALAR_TYPES = frozenset({int, float, bool, str, bytes, type(None)})
_DEFAULT_REPR_RE = re.compile(r"^<.+ object at 0x[0-9a-fA-F]+>$")


@dataclasses.dataclass(frozen=True)
class Caps:
  """Bounds for captured repr strings and recorded generator yields."""

  max_repr_len: int = DEFAULT_MAX_REPR_LEN
  max_recorded_yields: int = DEFAULT_MAX_RECORDED_YIELDS


class StreamResult:
  """Capped sample (``items``) + true yield count (``total``) for a wrapped generator."""

  def __init__(self, items: Sequence[Any], caps: Caps, total: int):
    self._items = items
    self._caps = caps
    self._total = total

  def __repr__(self) -> str:
    if self._total == 0:
      return "<generator: 0 items yielded>"
    sample = [safe_repr(it, self._caps) for it in self._items]
    suffix = (
        f" ... + {self._total - len(sample)} more"
        if self._total > len(sample)
        else ""
    )
    return (
        f"<generator: {self._total} items yielded; first {len(sample)}:"
        f" [{', '.join(sample)}]{suffix}>"
    )


def safe_repr(value: Any, caps: Caps) -> str:
  """``repr(value)`` capped, resilient, with default-form objects summarized."""
  max_len = caps.max_repr_len
  # Fast path: scalars never hit the default-repr regex or summary.
  if type(value) in _SCALAR_TYPES:
    r = repr(value)
    return (
        r
        if len(r) <= max_len
        else r[:max_len] + f"...[{len(r) - max_len} more chars]"
    )
  try:
    r = repr(value)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    logger.warning(
        "AutoTracingPlugin: repr() failed for %s: %s",
        type(value).__name__,
        exc,
    )
    r = f"<unrepr-able {type(value).__name__}: {exc!r}>"
  if _DEFAULT_REPR_RE.match(r):
    r = _summarize_default(value)
  if len(r) > max_len:
    r = r[:max_len] + f"...[{len(r) - max_len} more chars]"
  return r


def public_slot_names(cls: type) -> set[str]:
  """Public attr names declared in ``__slots__`` across ``cls.__mro__``.

  Handles the ``__slots__ = "x"`` shorthand (must be treated as a single
  name, not iterated as characters).
  """
  names: set[str] = set()
  for klass in cls.__mro__:
    slots = getattr(klass, "__slots__", None)
    if slots is None:
      continue
    if isinstance(slots, str):
      slots = (slots,)
    for slot in slots:
      if slot and not slot.startswith("_"):
        names.add(slot)
  return names


def _summarize_default(value: Any) -> str:
  """Replaces ``<X object at 0x..>`` with a public-field summary (handles ``__slots__``)."""
  cls = type(value).__name__
  public: list[tuple[str, Any]] = []
  instance_dict = getattr(value, "__dict__", None)
  if isinstance(instance_dict, dict):
    public.extend(
        (k, v) for k, v in instance_dict.items() if not k.startswith("_")
    )
  for slot_name in public_slot_names(type(value)):
    try:
      public.append((slot_name, getattr(value, slot_name)))
    except AttributeError:
      continue
  if not public:
    return f"<{cls}>"
  fields = []
  for k, v in public:
    try:
      vr = repr(v)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      logger.warning(
          "AutoTracingPlugin: repr() failed for %s.%s (%s): %s",
          cls,
          k,
          type(v).__name__,
          exc,
      )
      vr = f"<unrepr-able {type(v).__name__}>"
    fields.append(f"{k}={vr}")
  return f"<{cls} fields={{{', '.join(fields)}}}>"


def positional_param_names(fn: Callable[..., Any]) -> tuple[str, ...]:
  """Returns ``fn``'s positional parameter names; ``()`` if introspection fails."""
  try:
    return tuple(
        n
        for n, p in inspect.signature(fn).parameters.items()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    )
  except (TypeError, ValueError):
    return ()


def name_value_pairs(
    param_names: Sequence[str],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    caps: Caps,
) -> list[NamedArg]:
  """Returns ``[(name, repr)]`` for args + kwargs (no self/cls)."""
  pairs: list[NamedArg] = []
  for i, v in enumerate(args):
    name = param_names[i] if i < len(param_names) else f"arg{i}"
    if name in _SELF_OR_CLS:
      continue
    pairs.append((name, safe_repr(v, caps)))
  for k, v in kwargs.items():
    pairs.append((k, safe_repr(v, caps)))
  return pairs


def record_io_on_span(
    span: trace_api.Span,
    pairs: Sequence[NamedArg],
    result: Any,
    exc: BaseException | None,
    caps: Caps,
) -> None:
  """Writes ``adk.fn.*`` attributes onto ``span`` for the call's IO."""
  s = span.set_attribute
  for k, v in pairs:
    s(f"adk.fn.arg.{k}", v)
  if exc is not None:
    s("adk.fn.exc_type", type(exc).__qualname__)
    s("adk.fn.exc_repr", safe_repr(exc, caps))
    return
  s("adk.fn.return", safe_repr(result, caps))


def display_name_for(fn: Callable[..., Any]) -> str:
  """Returns the short (Class.method or function) name for ``fn``."""
  qn = fn.__qualname__
  return ".".join(qn.split(".")[-2:]) if "." in qn else qn


def tracer_will_record(tracer: trace_api.Tracer) -> bool:
  """True iff ``tracer`` will record (not a NoOpTracer)."""
  return not isinstance(tracer, trace_api.NoOpTracer)


def build_tracing_wrapper(
    fn: Callable[..., Any],
    tracer: trace_api.Tracer,
    caps: Caps,
) -> Callable[..., Any]:
  """Returns a tracing wrapper for ``fn`` matching its sync/async/gen shape."""
  # A non-recording tracer never produces IO; don't pay span/context cost.
  if not tracer_will_record(tracer):
    return fn

  display_name = display_name_for(fn)
  # inspect.signature is expensive; resolve once at wrap time.
  param_names = positional_param_names(fn)
  yield_cap = caps.max_recorded_yields

  def _finish(span, args, kwargs, result, exc):
    if not span.is_recording():
      return
    pairs = name_value_pairs(param_names, args, kwargs, caps)
    record_io_on_span(span, pairs, result, exc, caps)

  @functools.wraps(fn)
  async def async_wrapper(*args, **kwargs):
    with tracer.start_as_current_span(display_name) as span:
      try:
        r = await fn(*args, **kwargs)
      except BaseException as exc:
        _finish(span, args, kwargs, None, exc)
        raise
      _finish(span, args, kwargs, r, None)
      return r

  @functools.wraps(fn)
  async def async_gen_wrapper(*args, **kwargs):
    with tracer.start_as_current_span(display_name) as span:
      items: list[Any] = []
      total = 0
      try:
        async for item in fn(*args, **kwargs):
          total += 1
          if len(items) < yield_cap:
            items.append(item)
          yield item
      except BaseException as exc:
        _finish(span, args, kwargs, StreamResult(items, caps, total), exc)
        raise
      _finish(span, args, kwargs, StreamResult(items, caps, total), None)

  @functools.wraps(fn)
  def gen_wrapper(*args, **kwargs):
    with tracer.start_as_current_span(display_name) as span:
      items: list[Any] = []
      total = 0
      try:
        for item in fn(*args, **kwargs):
          total += 1
          if len(items) < yield_cap:
            items.append(item)
          yield item
      except BaseException as exc:
        _finish(span, args, kwargs, StreamResult(items, caps, total), exc)
        raise
      _finish(span, args, kwargs, StreamResult(items, caps, total), None)

  @functools.wraps(fn)
  def sync_wrapper(*args, **kwargs):
    with tracer.start_as_current_span(display_name) as span:
      try:
        r = fn(*args, **kwargs)
      except BaseException as exc:
        _finish(span, args, kwargs, None, exc)
        raise
      _finish(span, args, kwargs, r, None)
      return r

  if inspect.isasyncgenfunction(fn):
    wrapper = async_gen_wrapper
  elif asyncio.iscoroutinefunction(fn):
    wrapper = async_wrapper
  elif inspect.isgeneratorfunction(fn):
    wrapper = gen_wrapper
  else:
    wrapper = sync_wrapper
  setattr(wrapper, WRAPPED_ATTR, True)
  return wrapper
