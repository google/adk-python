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
from collections.abc import Mapping
import dataclasses
import functools
import inspect
import logging
import re
from typing import Any
from typing import AsyncIterator
from typing import Callable
from typing import Iterator
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

# Types whose repr() renders live secrets (tokens, keys, passwords). Matched by
# name over the MRO so this module never imports ``google.adk.auth``.
_CREDENTIAL_TYPE_NAMES = frozenset({
    "AuthConfig",
    "AuthCredential",
    "AuthToolArguments",
    "Credentials",
    "HttpAuth",
    "HttpCredentials",
    "OAuth2Auth",
    "OAuth2Session",
    "ServiceAccount",
    "ServiceAccountCredential",
})
# Parameter names that conventionally carry secret material.
_CREDENTIAL_ARG_NAMES = frozenset({
    "api_key",
    "auth_config",
    "auth_credential",
    "authorization",
    "cookie",
    "cookies",
    "credential",
    "credentials",
    "password",
    "private_key",
    "secret",
    "token",
})
_CREDENTIAL_ARG_SUFFIXES = (
    "_api_key",
    "_auth_config",
    "_authorization",
    "_cookie",
    "_cookies",
    "_credential",
    "_credentials",
    "_password",
    "_private_key",
    "_secret",
    "_token",
)
# Bounds for the structural walk below. Both are deliberately generous: only
# containers and objects consume node budget, so a list of a million ints
# costs one node.
_MAX_REDACT_DEPTH = 10
_MAX_REDACT_NODES = 1024


def _mro_holds_credential(cls: type) -> bool:
  """True iff ``cls`` or one of its bases is a credential-bearing type."""
  return any(k.__name__ in _CREDENTIAL_TYPE_NAMES for k in cls.__mro__)


# Cached because the walk asks this of every non-scalar node it visits. The
# annotation is spelled out because lru_cache erases the wrapped signature to
# ``*args: Hashable``, which the ``type(value)`` the callers pass does not
# satisfy.
_is_credential_type: Callable[[type], bool] = functools.lru_cache(maxsize=512)(
    _mro_holds_credential
)


@functools.lru_cache(maxsize=1024)
def _is_credential_arg_name(name: str) -> bool:
  """True iff a parameter called ``name`` conventionally holds a secret."""
  lowered = name.lower()
  return lowered in _CREDENTIAL_ARG_NAMES or lowered.endswith(
      _CREDENTIAL_ARG_SUFFIXES
  )


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


def _plain_repr(value: Any) -> str:
  """``repr(value)`` that never raises."""
  try:
    return repr(value)
  except Exception:  # pylint: disable=broad-exception-caught
    return f"<unrepr-able {type(value).__name__}>"


def _redacted_repr(value: Any) -> str | None:
  """Renders ``value`` with nested credentials masked, or ``None`` if clean.

  ``None`` means "no secret material anywhere in here", and the caller keeps
  plain ``repr()``. Otherwise the walk rebuilds the rendering element by
  element -- through mappings, sequences, sets, NamedTuples, dataclasses,
  pydantic models and plain objects -- so a credential is masked wherever it
  sits rather than only at the top level. Clean subtrees are still rendered
  with ``repr()``, so the text of an ordinary value is unchanged.

  The walk is bounded three ways: nesting depth, the number of container
  nodes visited, and an id set that stops cycles. A subtree it refuses to
  walk is elided rather than repr'd, so hitting a bound can never uncover a
  secret. An object that hides state behind a leading underscore is
  *inspected* there but only ever *rendered* from its public attributes, so
  the redacted form never shows more than the original repr would have.
  """
  budget = [_MAX_REDACT_NODES]
  active: set[int] = set()

  def member(name: Any, v: Any, depth: int) -> str | None:
    """Like ``walk`` but masks by field/key name too."""
    if isinstance(name, str) and _is_credential_arg_name(name):
      return f"<{type(v).__name__}>"
    return walk(v, depth)

  def members(items: Any, depth: int) -> list[str] | None:
    """``["name=text", ...]``, or ``None`` when nothing needed masking.

    Clean children are only rendered once the node is known to be dirty, so a
    value with no secret in it costs a traversal and not a repr per node.
    """
    walked = [(name, v, member(name, v, depth)) for name, v in items]
    if all(text is None for _, _, text in walked):
      return None
    return [
        f"{name}={text if text is not None else _plain_repr(v)}"
        for name, v, text in walked
    ]

  def walk(v: Any, depth: int) -> str | None:
    if type(v) in _SCALAR_TYPES:
      return None
    cls = type(v)
    if _is_credential_type(cls):
      return f"<{cls.__name__}>"
    if isinstance(v, type) or inspect.ismodule(v):
      return None
    # StreamResult already renders each sampled item through safe_repr, so
    # walking it would only cost the yield count its own repr reports.
    if isinstance(v, StreamResult):
      return None
    budget[0] -= 1
    marker = id(v)
    if budget[0] < 0 or depth >= _MAX_REDACT_DEPTH or marker in active:
      return f"<{cls.__name__} ...>"
    active.add(marker)
    try:
      return descend(v, depth + 1)
    finally:
      active.discard(marker)

  def descend(v: Any, depth: int) -> str | None:
    cls = type(v)
    name = cls.__name__
    speaks_for_itself = getattr(cls, "__repr__", None) is not object.__repr__
    if speaks_for_itself and isinstance(v, tuple) and hasattr(cls, "_fields"):
      parts = members(zip(cls._fields, v), depth)
      return f"{name}({', '.join(parts)})" if parts else None
    if speaks_for_itself and isinstance(v, Mapping):
      walked = [
          (k, walk(k, depth), item, member(k, item, depth))
          for k, item in v.items()
      ]
      if all(kt is None and vt is None for _, kt, _, vt in walked):
        return None
      body = ", ".join(
          f"{kt if kt is not None else _plain_repr(k)}:"
          f" {vt if vt is not None else _plain_repr(item)}"
          for k, kt, item, vt in walked
      )
      return "{" + body + "}"
    if isinstance(v, (list, tuple, set, frozenset)):
      elements = [(item, walk(item, depth)) for item in v]
      if all(text is None for _, text in elements):
        return None
      parts = [
          text if text is not None else _plain_repr(item)
          for item, text in elements
      ]
      if isinstance(v, list):
        return f"[{', '.join(parts)}]"
      if isinstance(v, tuple):
        return f"({parts[0]},)" if len(parts) == 1 else f"({', '.join(parts)})"
      body = "{" + ", ".join(parts) + "}"
      return body if isinstance(v, set) else f"frozenset({body})"
    if (
        speaks_for_itself
        and dataclasses.is_dataclass(v)
        and not isinstance(v, type)
    ):
      parts = members(
          ((f.name, getattr(v, f.name, None)) for f in dataclasses.fields(v)),
          depth,
      )
      return f"{name}({', '.join(parts)})" if parts else None
    if speaks_for_itself and isinstance(
        getattr(cls, "model_fields", None), dict
    ):
      declared = getattr(v, "__dict__", None) or {}
      extra = getattr(v, "__pydantic_extra__", None) or {}
      parts = members(list(declared.items()) + list(extra.items()), depth)
      return f"{name}({', '.join(parts)})" if parts else None
    return summarize_object(v, depth)

  def summarize_object(v: Any, depth: int) -> str | None:
    """Public-attribute summary; private state is inspected but never shown."""
    held: list[tuple[str, Any]] = []
    instance_dict = getattr(v, "__dict__", None)
    if isinstance(instance_dict, dict):
      held.extend(instance_dict.items())
    for slot in sorted(public_slot_names(type(v))):
      try:
        held.append((slot, getattr(v, slot)))
      except AttributeError:
        continue
    walked = [(name, item, member(name, item, depth)) for name, item in held]
    if all(text is None for _, _, text in walked):
      return None
    parts = [
        f"{name}={text if text is not None else _plain_repr(item)}"
        for name, item, text in walked
        if not name.startswith("_")
    ]
    cls_name = type(v).__name__
    if not parts:
      return f"<{cls_name}>"
    return f"<{cls_name} fields={{{', '.join(parts)}}}>"

  return walk(value, 0)


def safe_repr(value: Any, caps: Caps) -> str:
  """``repr(value)`` capped, resilient, credential-masked, defaults summarized."""
  max_len = caps.max_repr_len
  # Fast path: scalars never hit the default-repr regex or summary.
  if type(value) in _SCALAR_TYPES:
    r = repr(value)
    return (
        r
        if len(r) <= max_len
        else r[:max_len] + f"...[{len(r) - max_len} more chars]"
    )
  if _is_credential_type(type(value)):
    return f"<{type(value).__name__}>"
  try:
    redacted = _redacted_repr(value)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    # Elided rather than repr'd: the walk stopped partway, so nothing here
    # says the value is free of secrets.
    logger.warning(
        "AutoTracingPlugin: redaction failed for %s: %s",
        type(value).__name__,
        exc,
    )
    return f"<{type(value).__name__} ...>"
  if redacted is not None:
    r = redacted
  else:
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
    if _is_credential_arg_name(k) or _is_credential_type(type(v)):
      fields.append(f"{k}=<{type(v).__name__}>")
      continue
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
  """Returns ``[(name, repr)]`` for args + kwargs (no self/cls).

  An argument whose name marks it as secret material is dropped outright
  rather than masked: at the top level the key alone already says the call
  took a token, and no rendering of the value is worth recording. Values
  *nested* inside a recorded argument are masked in place instead, because
  dropping them would misreport the shape of the value that is being traced.
  """
  pairs: list[NamedArg] = []
  for i, v in enumerate(args):
    name = param_names[i] if i < len(param_names) else f"arg{i}"
    if name in _SELF_OR_CLS or _is_credential_arg_name(name):
      continue
    pairs.append((name, safe_repr(v, caps)))
  for k, v in kwargs.items():
    if _is_credential_arg_name(k):
      continue
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
    # Repeats the filter in name_value_pairs on purpose: both functions are
    # public, so pairs may come from a caller that never ran that filter.
    if _is_credential_arg_name(k):
      continue
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

  def _finish(
      span: trace_api.Span,
      args: tuple[Any, ...],
      kwargs: dict[str, Any],
      result: Any,
      exc: BaseException | None,
  ) -> None:
    if not span.is_recording():
      return
    pairs = name_value_pairs(param_names, args, kwargs, caps)
    record_io_on_span(span, pairs, result, exc, caps)

  @functools.wraps(fn)
  async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
    with tracer.start_as_current_span(display_name) as span:
      try:
        r = await fn(*args, **kwargs)
      except BaseException as exc:
        _finish(span, args, kwargs, None, exc)
        raise
      _finish(span, args, kwargs, r, None)
      return r

  @functools.wraps(fn)
  async def async_gen_wrapper(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
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
  def gen_wrapper(*args: Any, **kwargs: Any) -> Iterator[Any]:
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
  def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
    with tracer.start_as_current_span(display_name) as span:
      try:
        r = fn(*args, **kwargs)
      except BaseException as exc:
        _finish(span, args, kwargs, None, exc)
        raise
      _finish(span, args, kwargs, r, None)
      return r

  wrapper: Callable[..., Any]
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
