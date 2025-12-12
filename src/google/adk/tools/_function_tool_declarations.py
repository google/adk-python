# Copyright 2025 Google LLC
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

import inspect
import logging
from typing import Any
from typing import Callable
from typing import get_type_hints
from typing import Optional
from typing import Type

from google.genai import types
import pydantic
from pydantic import create_model
from pydantic import fields as pydantic_fields


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
    type_hints = get_type_hints(func)
  except TypeError:
    # Can happen with mock objects or complex annotations
    type_hints = {}

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

    if param.default is inspect._empty:
      default = pydantic_fields.PydanticUndefined
    else:
      default = param.default

    fields[name] = (ann, default)

  return fields


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
  func_name = getattr(func, '__name__', 'Callable')
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

  # Handle string annotations (forward references)
  if isinstance(return_annotation, str):
    try:
      type_hints = get_type_hints(func)
      return_annotation = type_hints.get('return', return_annotation)
    except TypeError:
      pass

  try:
    adapter = pydantic.TypeAdapter(
        return_annotation,
        config=pydantic.ConfigDict(arbitrary_types_allowed=True),
    )
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
  # Handle Pydantic BaseModel classes
  if isinstance(func, type) and issubclass(func, pydantic.BaseModel):
    schema = func.model_json_schema()
    description = inspect.cleandoc(func.__doc__) if func.__doc__ else None
    return types.FunctionDeclaration(
        name=func.__name__,
        description=description,
        parameters_json_schema=schema,
    )

  # Handle Callable functions
  description = inspect.cleandoc(func.__doc__) if func.__doc__ else None
  func_name = getattr(func, '__name__', 'Callable')
  declaration = types.FunctionDeclaration(
      name=func_name,
      description=description,
  )

  parameters_schema = _build_parameters_json_schema(func, ignore_params)
  if parameters_schema:
    declaration.parameters_json_schema = parameters_schema

  response_schema = _build_response_json_schema(func)
  if response_schema:
    declaration.response_json_schema = response_schema

  return declaration
