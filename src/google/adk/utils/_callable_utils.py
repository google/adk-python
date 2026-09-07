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

import functools
import inspect
import logging
import typing
from typing import Any
from typing import Callable

from . import context_utils

logger = logging.getLogger("google_adk." + __name__)

_DEFAULT_CALL_DOC = inspect.getdoc(type.__call__)

_METHOD_WRAPPER_TYPES = (
    type((1).__add__),
    type(object.__str__),
    type(str.lower),
)


def _is_routine(func: Any) -> bool:
  """Returns True if func is a routine or method-wrapper across Python versions."""
  return inspect.isroutine(func) or isinstance(func, _METHOD_WRAPPER_TYPES)


def unwrap_callable(func: Callable[..., Any]) -> Callable[..., Any]:
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
      # callable object instance, unwrap to its __call__ method
      call_method = getattr(func, "__call__", None)
      if call_method is None or call_method is func:
        break
      func = call_method
    else:
      break
  return func


@functools.lru_cache(maxsize=1024)
def _get_type_hints_for_unwrapped(func: Callable[..., Any]) -> dict[str, Any]:
  """Cached version of typing.get_type_hints for an unwrapped function.

  Note: NameError is deliberately not caught here so that unresolved forward
  references are not cached by lru_cache, allowing recovery once defined.
  """
  try:
    return typing.get_type_hints(func)
  except (TypeError, AttributeError):
    return {}


def get_type_hints_cached(func: Callable[..., Any]) -> dict[str, Any]:
  """Cached version of typing.get_type_hints with robust callable unwrapping.

  Unwraps functools.partial and custom callable objects to ensure type hints
  are resolved even when typing.get_type_hints would otherwise raise TypeError
  on partial objects (such as under `from __future__ import annotations`).
  """
  unwrapped = unwrap_callable(func)
  try:
    hints = _get_type_hints_for_unwrapped(unwrapped)
  except NameError:
    hints = {}

  if (
      not hints
      and hasattr(func, "__call__")
      and not _is_routine(func)
      and not isinstance(func, type)
  ):
    try:
      hints = _get_type_hints_for_unwrapped(func.__call__)
    except NameError:
      hints = {}
  return hints


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

  def __init__(self, func: Callable[..., Any] | None) -> None:
    if func is not None and not callable(func):
      raise TypeError(f"Expected a callable object, got {type(func).__name__}")
    self.func = func
    self.unwrapped_func = unwrap_callable(func) if func is not None else None

    # Docstring resolution (prioritize direct func.__doc__, then func.__call__.__doc__)
    if func is not None:
      doc = inspect.getdoc(func) or ""
      if not doc and not _is_routine(func) and not isinstance(func, type):
        call_method = getattr(func, "__call__", None)
        if call_method is not None:
          call_doc = inspect.getdoc(call_method) or ""
          if call_doc and call_doc != _DEFAULT_CALL_DOC:
            doc = call_doc
      if doc == _DEFAULT_CALL_DOC:
        doc = ""
      self.doc = doc

      # Context parameter detection
      self.context_param_name = context_utils.find_context_parameter(func)

      # Resolve signature presence at initialization
      try:
        self._signature = inspect.signature(self.func)
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
