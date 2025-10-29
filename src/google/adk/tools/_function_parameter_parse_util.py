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

from __future__ import annotations

from enum import Enum
import inspect
import logging
import types as typing_types
from typing import _GenericAlias
from typing import Any
from typing import get_args
from typing import get_origin
from typing import Literal
from typing import Union

from google.genai import types
import pydantic

from ..utils.variant_utils import GoogleLLMVariant

_py_builtin_type_to_schema_type = {
    str: types.Type.STRING,
    int: types.Type.INTEGER,
    float: types.Type.NUMBER,
    bool: types.Type.BOOLEAN,
    list: types.Type.ARRAY,
    dict: types.Type.OBJECT,
    None: types.Type.NULL,
    # TODO requested google GenAI SDK to add a Type.ANY and do the mapping on
    # their side, once new enum is added, replace the below one with
    # Any: types.Type.ANY
    Any: None,
}

logger = logging.getLogger('google_adk.' + __name__)


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
    # TODO(kech): Remove this workaround once mldev supports default value.
    schema.default = None
    logger.warning(
        'Default value is not supported in function declaration schema for'
        ' Google AI.'
    )


def _raise_if_schema_unsupported(
    variant: GoogleLLMVariant, schema: types.Schema
):
  if variant == GoogleLLMVariant.GEMINI_API:
    _raise_for_any_of_if_mldev(schema)
    # _update_for_default_if_mldev(schema) # No need of this since GEMINI now supports default value


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
    variant: GoogleLLMVariant, param: inspect.Parameter, func_name: str
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
  if isinstance(param.annotation, type) and issubclass(param.annotation, Enum):
    schema.type = types.Type.STRING
    schema.enum = [e.value for e in param.annotation]
    if param.default is not inspect.Parameter.empty:
      default_value = (
          param.default.value
          if isinstance(param.default, Enum)
          else param.default
      )
      if default_value not in schema.enum:
        raise ValueError(default_value_error_msg)
      schema.default = default_value
    _raise_if_schema_unsupported(variant, schema)
    return schema
  if (
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
  if param.annotation is None:
    # https://swagger.io/docs/specification/v3_0/data-models/data-types/#null
    # null is not a valid type in schema, use object instead.
    schema.type = types.Type.OBJECT
    schema.nullable = True
    _raise_if_schema_unsupported(variant, schema)
    return schema
  raise ValueError(
      f'Failed to parse the parameter {param} of function {func_name} for'
      ' automatic function calling. Automatic function calling works best with'
      ' simpler function signature schema, consider manually parsing your'
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
