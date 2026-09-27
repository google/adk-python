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

"""Function tool declaration builder using Pydantic's JSON schema generation.

This module provides a streamlined approach to building FunctionDeclaration
objects by leveraging Pydantic's `create_model` and `model_json_schema()`
capabilities instead of manual type parsing.

The GenAI SDK supports `parameters_json_schema` which accepts raw JSON schema,
allowing us to delegate schema generation complexity to Pydantic.
"""

from __future__ import annotations

import collections.abc
import functools
import inspect
import logging
import sys
from types import UnionType
from typing import Any
from typing import Callable
from typing import cast
from typing import ForwardRef
from typing import get_args
from typing import get_origin
from typing import get_type_hints
from typing import Literal
from typing import Optional
from typing import Type
from typing import Union

from google.genai import types
import pydantic
from pydantic import create_model
from pydantic import fields as pydantic_fields
from typing_extensions import Annotated

from ..utils.variant_utils import get_google_llm_variant
from ..utils.variant_utils import GoogleLLMVariant

logger = logging.getLogger('google_adk.' + __name__)


def _is_union_type(origin: Any) -> bool:
  """Returns True if origin is Union or UnionType."""
  return origin is Union or origin is UnionType


def _is_optional_type(tp: Any) -> bool:
  """Returns True if tp is Optional or a Union containing NoneType."""
  origin = get_origin(tp)
  if _is_union_type(origin):
    return type(None) in get_args(tp)
  return tp is type(None)


def _collapse_redundant_outer_optional(tp: Any) -> Any:
  """Collapses redundant outer Optional added by Python 3.10 get_type_hints.

  On Python 3.10, get_type_hints(func, include_extras=True) wraps parameter
  annotations in Optional[...] when default is None, even if the inner
  Annotated type is already Optional (e.g. Annotated[Optional[T], Field(...)]).
  This causes Pydantic to generate doubly-nested anyOf schemas. We unwrap the
  outer Optional when the inner Annotated type is already optional.
  """
  origin = get_origin(tp)
  if _is_union_type(origin):
    args = get_args(tp)
    if type(None) in args:
      non_none = [a for a in args if a is not type(None)]
      if len(non_none) == 1 and get_origin(non_none[0]) is Annotated:
        inner_type = get_args(non_none[0])[0]
        if _is_optional_type(inner_type):
          return non_none[0]
  return tp


def _get_callable_globals(func: Callable[..., Any]) -> dict[str, Any]:
  """Extracts the globals dictionary from a callable."""
  while True:
    if isinstance(func, functools.partial):
      func = func.func
    elif hasattr(func, '__wrapped__'):
      func = inspect.unwrap(func)
      if isinstance(func, functools.partial):
        func = func.func
      else:
        break
    else:
      break
  if hasattr(func, '__globals__'):
    return func.__globals__
  if hasattr(func, '__call__') and hasattr(func.__call__, '__globals__'):
    return cast(dict[str, Any], func.__call__.__globals__)
  module_name = getattr(func, '__module__', None)
  if module_name and module_name in sys.modules:
    return cast(
        dict[str, Any], getattr(sys.modules[module_name], '__dict__', {})
    )
  return {}


def _resolve_annotation(ann: Any, globalns: dict[str, Any]) -> Any:
  """Resolves string, ForwardRef, and generic type annotations in globalns.

  Paired with `_is_unresolvable`; both functions must cover the same type
  shapes (Annotated, containers/generics, ForwardRef, str).
  """
  while isinstance(ann, str):
    if globalns:
      try:
        resolved = eval(ann, globalns)
        if resolved == ann:
          break
        ann = resolved
      except Exception:
        break
    else:
      break

  if isinstance(ann, ForwardRef):
    if globalns:
      try:
        resolved = eval(ann.__forward_arg__, globalns)
        return _resolve_annotation(resolved, globalns)
      except Exception:
        return ann
    return ann

  origin = get_origin(ann)
  if origin is not None:
    if origin is Literal:
      return ann
    args = get_args(ann)
    if not args:
      return ann
    if origin is Annotated:
      resolved_inner = _resolve_annotation(args[0], globalns)
      if resolved_inner is not args[0]:
        return Annotated[(resolved_inner, *args[1:])]
      return ann
    resolved_args = tuple(_resolve_annotation(arg, globalns) for arg in args)
    if resolved_args != args:
      try:
        if _is_union_type(origin):
          return Union[resolved_args]
        if len(resolved_args) == 1:
          return origin[resolved_args[0]]
        return origin[resolved_args]
      except Exception:
        return ann
  return ann


def _is_unresolvable(ann: Any) -> bool:
  """Returns True if ann contains an unresolvable string or ForwardRef.

  Paired with `_resolve_annotation`; both functions must cover the same type
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
      return bool(args and _is_unresolvable(args[0]))
    return any(_is_unresolvable(arg) for arg in args)
  return False


def _get_function_fields(
    func: Callable[..., Any],
    ignore_params: Optional[list[str]] = None,
) -> dict[str, tuple[type[Any], Any]]:
  """Extract function parameters as Pydantic field definitions.

  Args:
    func: The callable to extract parameters from.
    ignore_params: List of parameter names to exclude from the schema.

  Returns:
    A dictionary mapping parameter names to (type, default) tuples suitable
    for Pydantic's create_model.
  """
  if ignore_params is None:
    ignore_params = []

  sig = inspect.signature(func)
  fields: dict[str, tuple[type[Any], Any]] = {}

  # Get type hints with forward reference resolution
  try:
    type_hints = get_type_hints(func, include_extras=True)
  except (TypeError, NameError, AttributeError):
    # TypeError can happen with mock objects or complex annotations. NameError
    # / AttributeError happen when an annotation is an unresolvable forward
    # reference at runtime. Fall back to resolving individual annotations below.
    type_hints = {}

  func_globals = _get_callable_globals(func)

  for name, param in sig.parameters.items():
    if name in ignore_params:
      continue

    if param.kind not in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_ONLY,
    ):
      continue

    # Get annotation, preferring resolved type hints
    if name in type_hints:
      ann = type_hints[name]
    elif param.annotation is not inspect._empty:
      ann = param.annotation
    else:
      ann = Any

    if _is_unresolvable(ann):
      ann = _resolve_annotation(ann, func_globals)

    if _is_unresolvable(ann):
      logger.warning(
          'Parameter %r of %r has unresolvable type annotation %r; dropping'
          ' parameter schema.',
          name,
          get_callable_name(func),
          ann,
      )
      return {}

    ann = _collapse_redundant_outer_optional(ann)

    if param.default is inspect._empty:
      default = pydantic_fields.PydanticUndefined
    else:
      default = param.default

    if get_origin(ann) is Annotated:
      try:
        field_info = pydantic_fields.FieldInfo.from_annotation(ann)
        field_info.alias = None
        field_info.validation_alias = None
        field_info.serialization_alias = None
        field_info.default = default
        field_info.default_factory = None
        ann_type = (
            field_info.annotation if field_info.annotation is not None else Any
        )
        fields[name] = (ann_type, field_info)
      except Exception:
        logger.warning(
            'Failed to strip alias from Annotated parameter %r; falling back'
            ' to raw annotation',
            name,
            exc_info=True,
        )
        fields[name] = (ann, default)
    else:
      fields[name] = (ann, default)

  return fields


def get_callable_name(func: Callable[..., Any]) -> str:
  """Returns the name a callable is advertised and registered under.

  Callable objects carry no `__name__`, so they fall back to their class name.
  This is the single source of truth for both the declaration sent to the model
  and the key the tool is registered under: if the two disagree, the model is
  told about a tool it cannot invoke.

  Args:
    func: The callable backing a tool.

  Returns:
    The name to use for the callable.
  """
  return getattr(func, '__name__', None) or func.__class__.__name__


def _flatten_optional_any_of(schema: dict[str, Any]) -> dict[str, Any]:
  """Flattens `Optional[X]`-style `anyOf` schemas for Vertex AI.

  Pydantic serializes `Optional[X]` fields as
  `{"anyOf": [<X schema>, {"type": "null"}], ...}`. Vertex AI rejects such
  schemas because the wrapping schema itself has no top-level `type` field
  (see https://github.com/googleapis/python-genai/issues/1807), even though
  the Gemini Developer API (AI Studio) accepts them as-is. This merges the
  non-null branch into the parent schema and marks it `nullable` instead,
  which Vertex AI accepts.

  True unions with more than one non-null variant (e.g. `Union[int, str]`)
  can't be losslessly flattened this way and are left untouched.
  """
  any_of = schema.get('anyOf')
  if not isinstance(any_of, list):
    return schema

  # We need at least one null variant and at least one non-null variant
  # to flatten an optional/nullable schema.
  if len(any_of) < 2:
    return schema

  null_variants = [
      variant
      for variant in any_of
      if isinstance(variant, dict) and variant.get('type') == 'null'
  ]
  # We only flatten if there is exactly one null variant (i.e. it is optional).
  if len(null_variants) != 1:
    return schema

  non_null_variants = [v for v in any_of if v not in null_variants]
  if len(non_null_variants) == 1:
    # Optional[X] -> X + nullable: true
    flattened = dict(non_null_variants[0])
    for key, value in schema.items():
      if key != 'anyOf':
        flattened.setdefault(key, value)
    flattened['nullable'] = True
    return flattened
  else:
    # Optional[Union[A, B, ...]] -> Union[A, B, ...] + nullable: true
    # We keep the anyOf but remove the null variant.
    flattened = dict(schema)
    flattened['anyOf'] = non_null_variants
    flattened['nullable'] = True
    return flattened


def _sanitize_json_schema_for_vertex(schema: Any) -> Any:
  """Recursively rewrites `Optional[X]` `anyOf` schemas for Vertex AI.

  Vertex AI's schema validator requires every (sub)schema to declare a
  top-level `type`, which Pydantic's `anyOf`-based representation of
  `Optional`/`Union` fields does not provide. This is only applied for the
  Vertex AI backend since the Gemini Developer API (AI Studio) already
  accepts the unmodified Pydantic schema.
  """
  if isinstance(schema, list):
    return [_sanitize_json_schema_for_vertex(item) for item in schema]
  if not isinstance(schema, dict):
    return schema

  sanitized = {
      key: _sanitize_json_schema_for_vertex(value)
      for key, value in schema.items()
  }
  return _flatten_optional_any_of(sanitized)


def _build_parameters_json_schema(
    func: Callable[..., Any],
    ignore_params: Optional[list[str]] = None,
) -> Optional[dict[str, Any]]:
  """Build JSON schema for function parameters using Pydantic.

  Args:
    func: The callable to generate schema for.
    ignore_params: List of parameter names to exclude.

  Returns:
    A JSON schema dict, or None if the function has no parameters.
  """
  fields = _get_function_fields(func, ignore_params)
  if not fields:
    return None

  # Create a Pydantic model dynamically
  func_name = get_callable_name(func)
  model = create_model(
      f'{func_name}Params',
      **fields,  # type: ignore[arg-type]
  )

  return model.model_json_schema()


def _build_response_json_schema(
    func: Callable[..., Any],
) -> Optional[dict[str, Any]]:
  """Build JSON schema for function return type using Pydantic.

  Args:
    func: The callable to generate return schema for.

  Returns:
    A JSON schema dict for the return type, or None if no return annotation.
  """
  return_annotation = inspect.signature(func).return_annotation

  if return_annotation is inspect._empty:
    return None

  try:
    type_hints = get_type_hints(func, include_extras=True)
    return_annotation = type_hints.get('return', return_annotation)
  except (TypeError, NameError, AttributeError):
    func_globals = _get_callable_globals(func)
    return_annotation = _resolve_annotation(return_annotation, func_globals)

  if _is_unresolvable(return_annotation):
    func_globals = _get_callable_globals(func)
    return_annotation = _resolve_annotation(return_annotation, func_globals)

  if _is_unresolvable(return_annotation):
    return None

  # Handle AsyncGenerator and Generator return types (streaming tools)
  # AsyncGenerator[YieldType, SendType] -> use YieldType as response schema
  # Generator[YieldType, SendType, ReturnType] -> use YieldType as response schema
  origin = get_origin(return_annotation)
  if origin is not None and (
      origin is collections.abc.AsyncGenerator
      or origin is collections.abc.Generator
  ):
    type_args = get_args(return_annotation)
    if type_args:
      # First type argument is the yield type
      return_annotation = type_args[0]

  try:
    try:
      adapter = pydantic.TypeAdapter(
          return_annotation,
          config=pydantic.ConfigDict(arbitrary_types_allowed=True),
      )
    except pydantic.PydanticUserError as e:
      # If it failed, maybe it was because of the config argument (e.g. for dataclasses).
      # Retry without config.
      logging.debug(
          'Failed to build schema with config, retrying without config for'
          ' %s: %s',
          func.__name__,
          e,
      )
      adapter = pydantic.TypeAdapter(return_annotation)
    return adapter.json_schema()
  except Exception:
    logging.warning(
        'Failed to build response JSON schema for %s',
        func.__name__,
        exc_info=True,
    )
    # Fall back to untyped response
    return None


def build_function_declaration_with_json_schema(
    func: Callable[..., Any] | Type[pydantic.BaseModel],
    ignore_params: Optional[list[str]] = None,
) -> types.FunctionDeclaration:
  """Build a FunctionDeclaration using Pydantic's JSON schema generation.

  This function provides a simplified approach compared to manual type parsing.
  It uses Pydantic's `create_model` to dynamically create a model from function
  parameters, then uses `model_json_schema()` to generate the JSON schema.

  The generated schema is passed to `parameters_json_schema` which the GenAI
  SDK supports natively.

  Args:
    func: The callable or Pydantic model to generate declaration for.
    ignore_params: List of parameter names to exclude from the schema.

  Returns:
    A FunctionDeclaration with the function's schema.

  Example:
    >>> from enum import Enum
    >>> from typing import List, Optional
    >>>
    >>> class Color(Enum):
    ...     RED = "red"
    ...     GREEN = "green"
    ...
    >>> def paint_room(
    ...     color: Color,
    ...     rooms: List[str],
    ...     dry_time_hours: Optional[int] = None,
    ... ) -> str:
    ...     '''Paint rooms with the specified color.'''
    ...     return f"Painted {len(rooms)} rooms {color.value}"
    >>>
    >>> decl = build_function_declaration_with_json_schema(paint_room)
    >>> decl.name
    'paint_room'
  """
  is_vertex_ai = get_google_llm_variant() == GoogleLLMVariant.VERTEX_AI

  # Handle Pydantic BaseModel classes
  if isinstance(func, type) and issubclass(func, pydantic.BaseModel):
    schema = func.model_json_schema()
    if is_vertex_ai:
      schema = _sanitize_json_schema_for_vertex(schema)
    description = inspect.cleandoc(func.__doc__) if func.__doc__ else None
    return types.FunctionDeclaration(
        name=func.__name__,
        description=description,
        parameters_json_schema=schema,
    )

  # Handle Callable functions
  description = inspect.cleandoc(func.__doc__) if func.__doc__ else None
  func_name = get_callable_name(func)
  declaration = types.FunctionDeclaration(
      name=func_name,
      description=description,
  )

  parameters_schema = _build_parameters_json_schema(func, ignore_params)
  if parameters_schema:
    if is_vertex_ai:
      parameters_schema = _sanitize_json_schema_for_vertex(parameters_schema)
    declaration.parameters_json_schema = parameters_schema

  response_schema = _build_response_json_schema(func)
  if response_schema:
    if is_vertex_ai:
      response_schema = _sanitize_json_schema_for_vertex(response_schema)
    declaration.response_json_schema = response_schema

  return declaration
