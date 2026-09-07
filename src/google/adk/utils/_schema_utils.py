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

"""General schema utilities.

This module is for ADK internal use only.
Please do not rely on the implementation details.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from types import UnionType
from typing import Any
from typing import get_args
from typing import get_origin
from typing import Optional
from typing import Union

from google.genai import types
from pydantic import BaseModel
from pydantic import TypeAdapter

from . import _json_utils

logger = logging.getLogger("google_adk." + __name__)

# Use SchemaUnion from google.genai.types to support all schema types
# that the underlying API supports.
SchemaType = types.SchemaUnion
"""Type for schema fields (e.g., output_schema, input_schema).

Supports all schema types that the underlying Google GenAI API supports:
  - type[BaseModel]: A pydantic model class (e.g., MySchema)
  - GenericAlias: Generic types like list[str], list[MySchema], dict[str, int]
  - dict: Raw dict schemas
  - Schema: Google's Schema type
"""


def is_basemodel_schema(schema: SchemaType) -> bool:
  """Check if the schema is a BaseModel type (not a generic alias).

  Args:
    schema: The schema to check.

  Returns:
    True if schema is a BaseModel class, False otherwise.
  """
  return isinstance(schema, type) and issubclass(schema, BaseModel)


def is_list_of_basemodel(schema: SchemaType) -> bool:
  """Check if the schema is a list of BaseModel type.

  Args:
    schema: The schema to check.

  Returns:
    True if schema is list[SomeBaseModel], False otherwise.
  """
  origin = get_origin(schema)
  if origin is not list:
    return False

  args = get_args(schema)
  if not args:
    return False

  inner_type = args[0]
  return isinstance(inner_type, type) and issubclass(inner_type, BaseModel)


def get_list_inner_type(schema: SchemaType) -> Optional[type[BaseModel]]:
  """Get the inner BaseModel type from a list[BaseModel] schema.

  Args:
    schema: The schema (expected to be list[SomeBaseModel]).

  Returns:
    The inner BaseModel type, or None if not a list of BaseModel.
  """
  if not is_list_of_basemodel(schema):
    return None

  args = get_args(schema)
  return args[0]


def schema_to_json_schema(schema: SchemaType) -> dict[str, Any]:
  """Converts a SchemaType to a JSON Schema dict.

  Args:
    schema: The schema to convert.

  Returns:
    A JSON Schema dict representation of the schema.
  """
  if isinstance(schema, dict):
    return schema
  return TypeAdapter(schema).json_schema()


def lowercase_schema_types(value: object) -> None:
  """Lowercases the JSON Schema ``type`` strings in a schema, in place.

  ``types.Schema`` serializes its type as the uppercase enum name (``STRING``),
  while JSON Schema and the model providers that consume it expect ``string``.
  A type may also be a list of names, as in ``["STRING", "NULL"]``. Nested
  subschemas are reached through the schema keywords only, so a ``type`` key
  inside a ``default`` or ``example`` value is left untouched.

  Args:
    value: A JSON Schema dict, or a list of them. Mutated in place.
  """
  if isinstance(value, list):
    for item in value:
      lowercase_schema_types(item)
    return

  if not isinstance(value, dict):
    return

  schema_type = value.get("type")
  if isinstance(schema_type, str):
    value["type"] = schema_type.lower()
  elif isinstance(schema_type, list):
    value["type"] = [
        item.lower() if isinstance(item, str) else item for item in schema_type
    ]

  for dict_key in (
      "$defs",
      "definitions",
      "defs",
      "dependentSchemas",
      "patternProperties",
      "properties",
  ):
    child_dict = value.get(dict_key)
    if isinstance(child_dict, dict):
      for child_value in child_dict.values():
        lowercase_schema_types(child_value)

  for single_key in (
      "additionalProperties",
      "additional_properties",
      "contains",
      "else",
      "if",
      "items",
      "not",
      "propertyNames",
      "then",
      "unevaluatedProperties",
  ):
    child_value = value.get(single_key)
    if isinstance(child_value, (dict, list)):
      lowercase_schema_types(child_value)

  for list_key in (
      "allOf",
      "all_of",
      "anyOf",
      "any_of",
      "oneOf",
      "one_of",
      "prefixItems",
  ):
    child_list = value.get(list_key)
    if isinstance(child_list, list):
      lowercase_schema_types(child_list)


def _strip_json_code_fence(json_text: str) -> str:
  """Removes a markdown code fence wrapping the entire JSON payload, if present.

  A model asked for structured output occasionally wraps it in a
  ```json ... ``` fence, most often when tools are configured alongside an
  output schema and the schema constraint becomes best-effort. Well-formed JSON
  never starts with a fence, so this is a no-op on valid input.
  """
  stripped = json_text.strip()
  match = re.fullmatch(r"```\w*\s*(.*?)\s*```", stripped, re.DOTALL)
  return match.group(1).strip() if match else json_text


def validate_schema(schema: SchemaType, json_text: str) -> Any:
  """Validate JSON text against a schema and return the result.

  Args:
    schema: The schema to validate against.
    json_text: The JSON text to validate.

  Returns:
    The validated result. Type depends on the schema:
      - dict for BaseModel
      - list of dicts for list[BaseModel]
      - raw value for other schema types (list[str], dict, etc.)
  """
  json_text = _strip_json_code_fence(json_text)

  if is_basemodel_schema(schema):
    # For regular BaseModel, use model_validate_json
    return schema.model_validate_json(json_text).model_dump(exclude_none=True)
  elif is_list_of_basemodel(schema):
    # For list[BaseModel], use TypeAdapter to validate
    type_adapter = TypeAdapter(schema)
    validated: list[Any] = type_adapter.validate_json(json_text)
    return [item.model_dump(exclude_none=True) for item in validated]
  else:
    # For other schema types (list[str], dict, Schema, etc.),
    return _json_utils.safe_json_loads(json_text, context="schema value")


def validate_node_data(
    schema: Optional[SchemaType],
    data: Any,
    *,
    preserve_content: bool = False,
) -> Any:
  """Validates and sanitizes node input or output data against a schema."""
  if data is None or schema is None:
    return data

  if isinstance(schema, (dict, types.Schema)):
    return data

  def _to_serializable(val: Any) -> Any:
    if isinstance(val, BaseModel):
      return val.model_dump(exclude_none=True)
    if isinstance(val, list):
      return [_to_serializable(item) for item in val]
    if isinstance(val, dict):
      return {k: _to_serializable(v) for k, v in val.items()}
    return val

  def _validate_python_object(val: Any) -> Any:
    validated: Any = TypeAdapter(schema).validate_python(val)
    return _to_serializable(validated)

  # If schema expects Content, do not unwrap
  if isinstance(schema, type) and issubclass(schema, types.Content):
    return _validate_python_object(data)
  if schema is types.Content:
    return _validate_python_object(data)

  if isinstance(data, types.Content):
    # Extract text part
    text_parts = [p.text for p in data.parts if p.text] if data.parts else []
    text_str = "".join(text_parts)

    # Validate the text
    if schema is str:
      validated_payload = text_str
    else:
      # Try to parse text as JSON first
      try:
        # Kept as json.loads: the surrounding except json.JSONDecodeError
        # distinguishes decode failure from schema validation failure.
        parsed_json = json.loads(text_str)
        validated_payload = _validate_python_object(parsed_json)
      except json.JSONDecodeError:
        # Fallback to validate raw string
        validated_payload = _validate_python_object(text_str)

    if not preserve_content:
      return validated_payload

    # Re-wrap in Content
    new_parts = [p for p in data.parts if not p.text] if data.parts else []
    new_parts.append(
        types.Part(
            text=json.dumps(validated_payload)
            if not isinstance(validated_payload, str)
            else validated_payload
        )
    )
    return types.Content(role=data.role, parts=new_parts)

  # If data is a string (but not wrapped in Content)
  if isinstance(data, str):
    if schema is str:
      return data
    return _validate_python_object(data)

  # For any other Python object (dict, BaseModel instance, etc.)
  return _validate_python_object(data)


def preprocess_args(
    args: dict[str, Any],
    signature: inspect.Signature | None,
    type_hints: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Preprocess and convert incoming argument dictionary before invocation.

  Converts dictionary values to Pydantic model instances where the function
  signature expects a BaseModel subclass or Optional[BaseModel].

  Args:
    args: The incoming argument dictionary to preprocess.
    signature: The inspected Signature of the target callable, or None if
      unintrospectable.
    type_hints: Optional cached type hints dictionary for the callable.

  Returns:
    A copy of args with applicable values converted to Pydantic model instances.
  """
  converted_args = args.copy()
  if signature is None:
    return converted_args

  if type_hints is None:
    type_hints = {}

  for param_name, param in signature.parameters.items():
    if param_name in args:
      target_type = type_hints.get(param_name, param.annotation)
      if target_type != inspect.Parameter.empty:
        origin = get_origin(target_type)
        if origin is Union or origin is UnionType:
          union_args = get_args(target_type)
          non_none_types = [arg for arg in union_args if arg is not type(None)]
          if len(non_none_types) == 1:
            target_type = non_none_types[0]
            origin = get_origin(target_type)
          elif len(non_none_types) > 1 and all(
              inspect.isclass(t) and issubclass(t, BaseModel)
              for t in non_none_types
          ):
            if args[param_name] is None or isinstance(
                args[param_name], tuple(non_none_types)
            ):
              continue
            try:
              converted_args[param_name] = TypeAdapter(
                  target_type
              ).validate_python(args[param_name])
            except Exception as e:
              logger.warning(
                  "Failed to convert argument '%s' to %s: %s",
                  param_name,
                  target_type,
                  e,
              )
            continue

        # Some session stores persist call args as a proto `Struct`, whose
        # only number type is double, so an int comes back as a float.
        if target_type is int and type(args[param_name]) is float:
          if args[param_name].is_integer():
            converted_args[param_name] = int(args[param_name])
          else:
            logger.warning(
                "Argument '%s' is typed int but got non-integral %r; passing it"
                " through unchanged.",
                param_name,
                args[param_name],
            )
          continue

        if inspect.isclass(target_type) and issubclass(target_type, BaseModel):
          if args[param_name] is None:
            continue
          if not isinstance(args[param_name], target_type):
            try:
              converted_args[param_name] = target_type.model_validate(
                  args[param_name]
              )
            except Exception as e:
              logger.warning(
                  "Failed to convert argument '%s' to Pydantic model %s: %s",
                  param_name,
                  target_type,
                  e,
              )
            continue

        # Handle list of BaseModel subclasses
        if is_list_of_basemodel(target_type) and isinstance(
            args[param_name], list
        ):
          item_type = get_list_inner_type(target_type)
          if item_type is not None:
            converted_list = []
            for item in args[param_name]:
              if isinstance(item, dict):
                try:
                  converted_list.append(item_type.model_validate(item))
                except Exception as e:
                  logger.warning(
                      "Failed to convert item in '%s' to %s: %s",
                      param_name,
                      item_type,
                      e,
                  )
                  converted_list.append(item)
              else:
                converted_list.append(item)
            converted_args[param_name] = converted_list

  return converted_args
