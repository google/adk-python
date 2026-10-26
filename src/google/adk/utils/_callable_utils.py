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

"""Unified callable introspection and specification utilities for ADK.

This module provides common primitives for inspecting, unwrapping, and extracting
schemas from Python callables. It serves as the shared foundation for both
FunctionTool (LLM tool calling) and FunctionNode (Workflow execution).
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import logging
import sys
from types import UnionType
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import ForwardRef
from typing import get_args
from typing import get_origin
from typing import Literal
from typing import TypeVar
from typing import Union

from typing_extensions import Annotated

logger = logging.getLogger("google_adk." + __name__)

_F = TypeVar("_F", bound=Callable[..., Any])

_DEFAULT_CALL_DOC = inspect.getdoc(type.__call__)
_DEFAULT_PARTIAL_DOC = inspect.getdoc(functools.partial)

_METHOD_WRAPPER_TYPES = (
    type((1).__add__),
    type(object.__str__),
    type(str.lower),
)


def lru_cache_allowing_unhashable(maxsize: int) -> Callable[[_F], _F]:
  """Wraps functools.lru_cache to run uncached when arguments are unhashable."""

  def decorator(func: _F) -> _F:
    cached = functools.lru_cache(maxsize=maxsize)(func)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
      try:
        hash(args + tuple(kwargs.values()))
      except TypeError:
        return wrapper.__wrapped__(*args, **kwargs)
      return cached(*args, **kwargs)

    for attribute in ("cache_clear", "cache_info", "cache_parameters"):
      setattr(wrapper, attribute, getattr(cached, attribute))
    # wraps preserves the runtime signature; cast preserves _F for mypy.
    return cast(_F, wrapper)

  return decorator


def _is_routine(func: Any) -> bool:
  """Returns True if func is a routine or method-wrapper across Python versions."""
  return inspect.isroutine(func) or isinstance(func, _METHOD_WRAPPER_TYPES)


def _is_dataclass_generated_doc(func: Any, doc: str) -> bool:
  """Returns True if doc is the default @dataclass signature docstring."""
  if not dataclasses.is_dataclass(func) or isinstance(func, type):
    return False
  cls = type(func)
  try:
    text_sig = str(inspect.signature(cls)).replace(" -> None", "")
  except (TypeError, ValueError):
    text_sig = ""
  return doc == inspect.cleandoc(cls.__name__ + text_sig)


def unwrap_callable(
    func: Callable[..., Any],
    *,
    follow_wrapped: bool = False,
) -> Callable[..., Any]:
  """Unwraps partials, bound methods and callable objects to find the stable underlying function."""
  seen: set[int] = set()
  while id(func) not in seen:
    seen.add(id(func))
    if isinstance(func, functools.partial):
      func = func.func
    elif hasattr(func, "__func__"):  # bound method
      func = func.__func__
    elif (
        hasattr(func, "__call__")
        and not _is_routine(func)
        and not isinstance(func, type)
    ):
      if follow_wrapped:
        wrapped = getattr(func, "__wrapped__", None)
        if wrapped is not None and wrapped is not func:
          func = wrapped
          continue
      # callable object instance, unwrap to its __call__ method
      call_method = getattr(func, "__call__", None)
      if call_method is None or call_method is func:
        break
      func = call_method
    else:
      break
  return func


def get_callable_globals(func: Any) -> dict[str, Any]:
  """Returns the module globals dict for a callable or class, if available."""
  target = unwrap_callable(func, follow_wrapped=True)
  if hasattr(target, "__wrapped__"):
    target = unwrap_callable(inspect.unwrap(target), follow_wrapped=True)
  func_globals = getattr(target, "__globals__", None)
  if isinstance(func_globals, dict):
    return func_globals
  module_name = getattr(target, "__module__", None)
  if isinstance(module_name, str):
    module = sys.modules.get(module_name)
    if module is not None:
      return vars(module)
  return {}


def get_adk_context_localns(func: Any) -> dict[str, Any]:
  """Returns fallback ADK context types absent from `func`'s own globals."""
  from ..agents.callback_context import CallbackContext
  from ..agents.context import Context
  from ..agents.readonly_context import ReadonlyContext
  from ..tools.tool_context import ToolContext

  func_globals = get_callable_globals(func)
  return {
      name: cls
      for name, cls in {
          "CallbackContext": CallbackContext,
          "Context": Context,
          "ReadonlyContext": ReadonlyContext,
          "ToolContext": ToolContext,
      }.items()
      if name not in func_globals
  }


def is_unresolvable(ann: Any) -> bool:
  """Returns True if `ann` contains an unresolvable string or ForwardRef.

  Paired with `resolve_annotation`; both functions must cover the same type
  shapes (Annotated, containers/generics, ForwardRef, str).
  """
  if isinstance(ann, (str, ForwardRef)):
    return True
  origin = get_origin(ann)
  if origin is not None:
    if origin is Literal:
      return False
    args = get_args(ann)
    if origin is Annotated:
      return bool(args and is_unresolvable(args[0]))
    return any(is_unresolvable(arg) for arg in args)
  return False


def resolve_annotation(
    ann: Any,
    globalns: dict[str, Any],
    localns: dict[str, Any] | None = None,
    *,
    include_extras: bool = True,
) -> Any:
  """Resolves string, ForwardRef, and generic type annotations.

  Paired with `is_unresolvable`; both functions must cover the same type
  shapes (Annotated, containers/generics, ForwardRef, str).
  """
  if isinstance(ann, (str, ForwardRef)):
    if not globalns and not localns:
      return ann
    holder = lambda: None
    holder.__annotations__ = {"_ann": ann}
    try:
      resolved = typing.get_type_hints(
          holder,
          globalns=globalns,
          localns=localns,
          include_extras=include_extras,
      )["_ann"]
    except Exception:  # pylint: disable=broad-exception-caught
      return ann
    if resolved == ann:
      return ann
    return resolve_annotation(
        resolved, globalns, localns, include_extras=include_extras
    )

  origin = get_origin(ann)
  if origin is not None:
    if origin is Literal:
      return ann
    args = get_args(ann)
    if not args:
      return ann
    if origin is Annotated:
      resolved_inner = resolve_annotation(
          args[0], globalns, localns, include_extras=include_extras
      )
      if not include_extras:
        return resolved_inner
      if resolved_inner is not args[0]:
        return Annotated[(resolved_inner, *args[1:])]
      return ann
    resolved_args = tuple(
        resolve_annotation(
            arg, globalns, localns, include_extras=include_extras
        )
        for arg in args
    )
    if resolved_args != args:
      try:
        if origin is Union or origin is UnionType:
          return Union[resolved_args]
        if len(resolved_args) == 1:
          return origin[resolved_args[0]]
        return origin[resolved_args]
      except Exception:  # pylint: disable=broad-exception-caught
        return ann
  return ann


@functools.lru_cache(maxsize=1024)
def _get_type_hints_for_unwrapped(
    func: Callable[..., Any],
    include_extras: bool = False,
) -> dict[str, Any]:
  """Cached version of typing.get_type_hints for an unwrapped function.

  On NameError, retries once with fallback ADK context types in localns; any
  remaining NameError propagates uncaught so unresolved forward references are
  not cached by lru_cache, allowing recovery once defined.
  """
  try:
    try:
      return typing.get_type_hints(func, include_extras=include_extras)
    except NameError:
      return typing.get_type_hints(
          func,
          localns=get_adk_context_localns(func),
          include_extras=include_extras,
      )
  except (TypeError, AttributeError):
    return {}


def _resolve_best_effort_annotations(
    func: Callable[..., Any],
    include_extras: bool = False,
) -> dict[str, Any]:
  """Resolves each annotation on `func` individually, skipping failures."""
  target = unwrap_callable(func, follow_wrapped=True)
  try:
    raw_annotations = getattr(target, "__annotations__", None)
  except Exception:  # pylint: disable=broad-exception-caught
    raw_annotations = None
  if not isinstance(raw_annotations, dict) or not raw_annotations:
    if (
        hasattr(func, "__call__")
        and not _is_routine(func)
        and not isinstance(func, type)
    ):
      target = func.__call__
      try:
        raw_annotations = getattr(target, "__annotations__", None)
      except Exception:  # pylint: disable=broad-exception-caught
        raw_annotations = None
  if not isinstance(raw_annotations, dict) or not raw_annotations:
    return {}

  globalns = get_callable_globals(target)
  localns = get_adk_context_localns(target)

  holder = lambda: None
  resolved: dict[str, Any] = {}
  for name, ann in raw_annotations.items():
    holder.__annotations__ = {name: ann}
    try:
      hint = typing.get_type_hints(
          holder,
          globalns=globalns,
          localns=localns,
          include_extras=include_extras,
      )[name]
    except Exception:  # pylint: disable=broad-exception-caught
      continue
    if is_unresolvable(hint):
      hint = resolve_annotation(
          hint, globalns, localns, include_extras=include_extras
      )
      if is_unresolvable(hint):
        continue
    resolved[name] = hint
  return resolved


def get_best_effort_type_hints(
    func: Callable[..., Any],
    include_extras: bool = False,
) -> dict[str, Any]:
  """Resolves callable type hints per parameter, skipping unresolvable annotations.

  When `from __future__ import annotations` is active and a single parameter
  references a `TYPE_CHECKING`-only symbol, `typing.get_type_hints` fails with
  `NameError` for the entire function. This helper evaluates each annotation
  individually against the callable's globals and ADK context types so that
  resolvable sibling parameters still retain their type hints.
  """
  return get_type_hints_cached(
      func, include_extras=include_extras
  ) or _resolve_best_effort_annotations(func, include_extras=include_extras)


def get_type_hints_cached(
    func: Callable[..., Any],
    include_extras: bool = False,
) -> dict[str, Any]:
  """Cached version of typing.get_type_hints with robust callable unwrapping.

  Unwraps functools.partial and custom callable objects to ensure type hints
  are resolved even when typing.get_type_hints would otherwise raise TypeError
  on partial objects (such as under `from __future__ import annotations`).
  """
  unwrapped = unwrap_callable(func, follow_wrapped=True)
  try:
    hints = _get_type_hints_for_unwrapped(
        unwrapped, include_extras=include_extras
    )
  except NameError:
    hints = {}

  if (
      not hints
      and hasattr(func, "__call__")
      and not _is_routine(func)
      and not isinstance(func, type)
  ):
    try:
      hints = _get_type_hints_for_unwrapped(
          func.__call__, include_extras=include_extras
      )
    except NameError:
      hints = {}
  return hints


def _clean_doc(obj: Any) -> str:
  """Returns the cleaned docstring of obj, skipping MRO for classes."""
  if isinstance(obj, type):
    doc = getattr(obj, "__doc__", None)
    return inspect.cleandoc(doc) if isinstance(doc, str) else ""
  return inspect.getdoc(obj) or ""


def get_callable_doc(func: Callable[..., Any]) -> str:
  """Returns the cleaned docstring of a callable, or "" if it has none.

  A functools.partial without a docstring of its own resolves to the wrapped
  callable's docstring, and a callable object without one falls back to its
  `__call__` docstring. Note that with `JSON_SCHEMA_FOR_FUNC_DECL` off,
  `from_function_with_options` still reads `func.__doc__` directly for routines,
  so the two can differ there.
  """
  doc = _clean_doc(func)
  if isinstance(func, functools.partial) and doc == _DEFAULT_PARTIAL_DOC:
    return get_callable_doc(func.func)
  if _is_dataclass_generated_doc(func, doc):
    doc = ""
  if not doc and not _is_routine(func) and not isinstance(func, type):
    call_method = getattr(func, "__call__", None)
    if call_method is not None:
      call_doc = _clean_doc(call_method)
      if call_doc and call_doc != _DEFAULT_CALL_DOC:
        doc = call_doc
  if doc == _DEFAULT_CALL_DOC:
    doc = ""
  return doc


class CallableSpec:
  """Unified specification and introspection for a callable.

  Encapsulates function signature, clean docstrings, cached type hints,
  and context parameter detection. Used by both FunctionTool and FunctionNode
  to avoid duplicated reflection logic.
  """

  func: Callable[..., Any] | None
  unwrapped_func: Callable[..., Any] | None
  doc: str
  context_param_name: str | None
  _signature: inspect.Signature | None
  _has_signature: bool
  _type_hints: dict[str, Any] | None
  _best_effort_type_hints: dict[str, Any] | None

  def __init__(self, func: Callable[..., Any] | None) -> None:
    if func is not None and not callable(func):
      raise TypeError(f"Expected a callable object, got {type(func).__name__}")
    self.func = func
    self.unwrapped_func = unwrap_callable(func) if func is not None else None

    if func is not None:
      self.doc = get_callable_doc(func)

      # Function-level import avoids circular import with context_utils.
      from . import context_utils

      self.context_param_name = context_utils.find_context_parameter(func)

      # Resolve signature presence at initialization
      try:
        self._signature = inspect.signature(func)
        self._has_signature = True
      except (ValueError, TypeError):
        self._signature = None
        self._has_signature = False
    else:
      self.doc = ""
      self.context_param_name = None
      self._signature = None
      self._has_signature = False

    self._type_hints = None
    self._best_effort_type_hints = None

  @property
  def has_signature(self) -> bool:
    """Returns True if the callable has an introspectable signature."""
    return self._has_signature

  @property
  def signature(self) -> inspect.Signature:
    """Returns the inspect.Signature of the callable.

    Raises:
      ValueError: If no signature can be found for the callable (e.g. builtins).
    """
    if not self._has_signature:
      raise ValueError(
          f"no signature found for builtin or callable {self.func!r}"
      )
    assert self._signature is not None
    return self._signature

  @property
  def is_generator(self) -> bool:
    """Returns True if the callable is a sync or async generator function."""
    if self.unwrapped_func is None:
      return False
    return inspect.isgeneratorfunction(
        self.unwrapped_func
    ) or inspect.isasyncgenfunction(self.unwrapped_func)

  @property
  def type_hints(self) -> dict[str, Any]:
    """Returns resolved type hints, retrying if earlier attempts had unresolved forward refs."""
    if self.func is None:
      return {}
    if not self._type_hints:
      hints = get_type_hints_cached(self.func)
      if hints:
        self._type_hints = hints
      else:
        return hints
    return self._type_hints

  def get_best_effort_type_hints(self) -> dict[str, Any]:
    """Returns resolved type hints, memoizing per-parameter fallback evaluation.

    Unlike `type_hints`, once the per-parameter fallback runs, its partial
    dictionary is memoized on this `CallableSpec` instance to avoid repeating
    per-parameter evaluation on every call when a `TYPE_CHECKING`-only symbol
    remains permanently undefined at runtime.
    """
    if self.func is None:
      return {}
    if self._type_hints is not None:
      return self._type_hints
    if self._best_effort_type_hints is not None:
      return self._best_effort_type_hints
    hints = self.type_hints
    if hints:
      return hints
    self._best_effort_type_hints = _resolve_best_effort_annotations(self.func)
    return self._best_effort_type_hints
