# Copyright 2024 Google LLC
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
#

import inspect
import logging
import re # Ensure re is imported
import types as typing_types
from typing import _GenericAlias
from typing import Any
from typing import get_args
from typing import get_origin
from typing import Literal
from typing import Union

from google.genai import types
import pydantic

_py_builtin_type_to_schema_type = {
    str: types.Type.STRING,
    int: types.Type.INTEGER,
    float: types.Type.NUMBER,
    bool: types.Type.BOOLEAN,
    list: types.Type.ARRAY,
    dict: types.Type.OBJECT,
}

logger = logging.getLogger(__name__)


def _is_builtin_primitive_or_compound(
    annotation: inspect.Parameter.annotation,
) -> bool:
  return annotation in _py_builtin_type_to_schema_type.keys()


def _raise_for_any_of_if_mldev(schema: types.Schema):
  if schema.any_of:
    raise ValueError(
        'AnyOf is not supported in function declaration schema for Google AI.'
    )


def _update_for_default_if_mldev(schema: types.Schema):
  if schema.default is not None:
    # TODO(kech): Remove this walkaround once mldev supports default value.
    schema.default = None
    logger.warning(
        'Default value is not supported in function declaration schema for'
        ' Google AI.'
    )


def _raise_if_schema_unsupported(variant: str, schema: types.Schema):
  if not variant == 'VERTEX_AI':
    _raise_for_any_of_if_mldev(schema)
    _update_for_default_if_mldev(schema)


def _is_default_value_compatible(
    default_value: Any, annotation: inspect.Parameter.annotation
) -> bool:
  # None type is expected to be handled external to this function
  if _is_builtin_primitive_or_compound(annotation):
    return isinstance(default_value, annotation)

  if (
      isinstance(annotation, _GenericAlias)
      or isinstance(annotation, typing_types.GenericAlias)
      or isinstance(annotation, typing_types.UnionType)
  ):
    origin = get_origin(annotation)
    if origin in (Union, typing_types.UnionType):
      return any(
          _is_default_value_compatible(default_value, arg)
          for arg in get_args(annotation)
      )

    if origin is dict:
      return isinstance(default_value, dict)

    if origin is list:
      if not isinstance(default_value, list):
        return False
      # most tricky case, element in list is union type
      # need to apply any logic within all
      # see test case test_generic_alias_complex_array_with_default_value
      # a: typing.List[int | str | float | bool]
      # default_value: [1, 'a', 1.1, True]
      return all(
          any(
              _is_default_value_compatible(item, arg)
              for arg in get_args(annotation)
          )
          for item in default_value
      )

    if origin is Literal:
      return default_value in get_args(annotation)

  # return False for any other unrecognized annotation
  # let caller handle the raise
  return False


def _parse_schema_from_parameter(
    variant: str, param: inspect.Parameter, func_name: str
) -> types.Schema:
  """parse schema from parameter.

  from the simplest case to the most complex case.
  """
  schema = types.Schema()
  default_value_error_msg = (
      f'Default value {param.default} of parameter {param} of function'
      f' {func_name} is not compatible with the parameter annotation'
      f' {param.annotation}.'
  )
  if _is_builtin_primitive_or_compound(param.annotation):
    if param.default is not inspect.Parameter.empty:
      if not _is_default_value_compatible(param.default, param.annotation):
        raise ValueError(default_value_error_msg)
      schema.default = param.default
    schema.type = _py_builtin_type_to_schema_type[param.annotation]
    _raise_if_schema_unsupported(variant, schema)
    return schema
  # >>> ANFANG DES NEUEN PATCH-TEILS für Optional[str] etc. <<<
  elif isinstance(param.annotation, str):
      # Attempt to handle string annotations, common with `from __future__ import annotations`
      cleaned_annotation_str = param.annotation.strip("'\"") # Entfernt äußere Anführungszeichen
      is_optional = False
      base_type_str = None

      # 1. Check for Optional[T] pattern
      optional_match = re.match(r'^Optional\[(.+)\]$', cleaned_annotation_str, re.IGNORECASE)
      if optional_match:
          is_optional = True
          base_type_str = optional_match.group(1).strip()

      # 2. Check for Union[T, None] or Union[None, T] pattern
      union_match = re.match(r'^Union\[(.+)\]$', cleaned_annotation_str, re.IGNORECASE)
      if union_match:
          parts = [p.strip() for p in union_match.group(1).split(',')]
          if 'None' in parts or 'NoneType' in parts: # auch NoneType prüfen
              is_optional = True
              # Find the non-None type string
              non_none_parts = [p for p in parts if p not in ('None', 'NoneType')]
              if len(non_none_parts) == 1:
                 base_type_str = non_none_parts[0]
              # else: more complex Union, let it fall through for now

      # 3. Check if the extracted base_type_str is a known primitive/builtin
      actual_base_type = None
      if base_type_str:
          # Try direct match first (e.g., 'str', 'int')
          for type_obj in _py_builtin_type_to_schema_type.keys():
             if type_obj.__name__ == base_type_str:
                 actual_base_type = type_obj
                 break
          # TODO: Add checks for list, dict if needed, e.g., if base_type_str == 'list'

      # 4. If we successfully identified Optional[KnownPrimitiveType]
      if actual_base_type and is_optional:
          # We found Optional[PrimitiveType] like Optional[str]
          schema.type = _py_builtin_type_to_schema_type[actual_base_type]
          schema.nullable = True # Mark as nullable

          if param.default is not inspect.Parameter.empty:
              # Default value check needs careful handling for Optional
              if param.default is not None and not _is_default_value_compatible(param.default, actual_base_type):
                   raise ValueError(default_value_error_msg + f" (for base type {actual_base_type.__name__})")
              # If default is None, it's compatible with Optional
              schema.default = param.default

          _raise_if_schema_unsupported(variant, schema)
          # print(f"DEBUG: Handled Optional string annotation '{param.annotation}' -> {schema}") # Debug-Ausgabe
          return schema
      # If it's not a simple Optional[Primitive] pattern recognised here,
      # let it fall through to the next elif (simple string type) or the final error.

      # 5. Fallback: Check if the string is just a simple type name (previous patch logic)
      elif cleaned_annotation_str in [t.__name__ for t in _py_builtin_type_to_schema_type.keys()]:
          actual_type = None
          for type_obj in _py_builtin_type_to_schema_type.keys():
              if type_obj.__name__ == cleaned_annotation_str:
                  actual_type = type_obj
                  break

          if actual_type:
              if param.default is not inspect.Parameter.empty:
                  if not _is_default_value_compatible(param.default, actual_type):
                      raise ValueError(default_value_error_msg)
                  schema.default = param.default
              schema.type = _py_builtin_type_to_schema_type[actual_type]
              _raise_if_schema_unsupported(variant, schema)
              # print(f"DEBUG: Handled simple string annotation '{param.annotation}' -> {schema}") # Debug-Ausgabe
              return schema
      # else: Fall through to other checks or the final error

  # >>> ENDE DES NEUEN PATCH-TEILS <<<

  elif ( # Bestehender Code für Union-Typen (nicht Strings)
      get_origin(param.annotation) is Union
      # only parse simple UnionType, example int | str | float | bool
      # complex types.UnionType will be invoked in raise branch
      and all(
          (_is_builtin_primitive_or_compound(arg) or arg is type(None))
          for arg in get_args(param.annotation)
      )
  ):
    schema.type = types.Type.OBJECT
    schema.any_of = []
    unique_types = set()
    for arg in get_args(param.annotation):
      if arg.__name__ == 'NoneType':  # Optional type
        schema.nullable = True
        continue
      schema_in_any_of = _parse_schema_from_parameter(
          variant,
          inspect.Parameter(
              'item', inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=arg
          ),
          func_name,
      )
      if (
          schema_in_any_of.model_dump_json(exclude_none=True)
          not in unique_types
      ):
        schema.any_of.append(schema_in_any_of)
        unique_types.add(schema_in_any_of.model_dump_json(exclude_none=True))
    if len(schema.any_of) == 1:  # param: list | None -> Array
      schema.type = schema.any_of[0].type
      schema.any_of = None
    if (
        param.default is not inspect.Parameter.empty
        and param.default is not None
    ):
      if not _is_default_value_compatible(param.default, param.annotation):
        raise ValueError(default_value_error_msg)
      schema.default = param.default
    _raise_if_schema_unsupported(variant, schema)
    return schema
  if isinstance(param.annotation, _GenericAlias) or isinstance(
      param.annotation, typing_types.GenericAlias
  ):
    origin = get_origin(param.annotation)
    args = get_args(param.annotation)
    if origin is dict:
      schema.type = types.Type.OBJECT
      if param.default is not inspect.Parameter.empty:
        if not _is_default_value_compatible(param.default, param.annotation):
          raise ValueError(default_value_error_msg)
        schema.default = param.default
      _raise_if_schema_unsupported(variant, schema)
      return schema
    if origin is Literal:
      if not all(isinstance(arg, str) for arg in args):
        raise ValueError(
            f'Literal type {param.annotation} must be a list of strings.'
        )
      schema.type = types.Type.STRING
      schema.enum = list(args)
      if param.default is not inspect.Parameter.empty:
        if not _is_default_value_compatible(param.default, param.annotation):
          raise ValueError(default_value_error_msg)
        schema.default = param.default
      _raise_if_schema_unsupported(variant, schema)
      return schema
    if origin is list:
      schema.type = types.Type.ARRAY
      schema.items = _parse_schema_from_parameter(
          variant,
          inspect.Parameter(
              'item',
              inspect.Parameter.POSITIONAL_OR_KEYWORD,
              annotation=args[0],
          ),
          func_name,
      )
      if param.default is not inspect.Parameter.empty:
        if not _is_default_value_compatible(param.default, param.annotation):
          raise ValueError(default_value_error_msg)
        schema.default = param.default
      _raise_if_schema_unsupported(variant, schema)
      return schema
    if origin is Union:
      schema.any_of = []
      schema.type = types.Type.OBJECT
      unique_types = set()
      for arg in args:
        if arg.__name__ == 'NoneType':  # Optional type
          schema.nullable = True
          continue
        schema_in_any_of = _parse_schema_from_parameter(
            variant,
            inspect.Parameter(
                'item',
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=arg,
            ),
            func_name,
        )
        if (
            len(param.annotation.__args__) == 2
            and type(None) in param.annotation.__args__
        ):  # Optional type
          for optional_arg in param.annotation.__args__:
            if (
                hasattr(optional_arg, '__origin__')
                and optional_arg.__origin__ is list
            ):
              # Optional type with list, for example Optional[list[str]]
              schema.items = schema_in_any_of.items
        if (
            schema_in_any_of.model_dump_json(exclude_none=True)
            not in unique_types
        ):
          schema.any_of.append(schema_in_any_of)
          unique_types.add(schema_in_any_of.model_dump_json(exclude_none=True))
      if len(schema.any_of) == 1:  # param: Union[List, None] -> Array
        schema.type = schema.any_of[0].type
        schema.any_of = None
      if (
          param.default is not None
          and param.default is not inspect.Parameter.empty
      ):
        if not _is_default_value_compatible(param.default, param.annotation):
          raise ValueError(default_value_error_msg)
        schema.default = param.default
      _raise_if_schema_unsupported(variant, schema)
      return schema
      # all other generic alias will be invoked in raise branch
  if (
      inspect.isclass(param.annotation)
      # for user defined class, we only support pydantic model
      and issubclass(param.annotation, pydantic.BaseModel)
  ):
    if (
        param.default is not inspect.Parameter.empty
        and param.default is not None
    ):
      schema.default = param.default
    schema.type = types.Type.OBJECT
    schema.properties = {}
    for field_name, field_info in param.annotation.model_fields.items():
      schema.properties[field_name] = _parse_schema_from_parameter(
          variant,
          inspect.Parameter(
              field_name,
              inspect.Parameter.POSITIONAL_OR_KEYWORD,
              annotation=field_info.annotation,
          ),
          func_name,
      )
    _raise_if_schema_unsupported(variant, schema)
    return schema

  # Letzter Ausweg: Fehler (Originaler Fehlerfall, jetzt mit detaillierterer Meldung)
  raise ValueError(
      f'Failed to parse the parameter {param.name}: {param.annotation!r} = {param.default!r} (Annotation Type: {type(param.annotation).__name__}) of function {func_name} for' # Detailliertere Fehlermeldung
      ' automatic function calling. Automatic function calling works best with'
      ' simpler function signature schema, consider manually parse your'
      f' function declaration for function {func_name}.'
  )


def _get_required_fields(schema: types.Schema) -> list[str]:
  if not schema.properties:
    return
  return [
      field_name
      for field_name, field_schema in schema.properties.items()
      if not field_schema.nullable and field_schema.default is None
  ]
