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

from __future__ import annotations

import re
from typing import Any
from typing import Optional

from google.genai.types import JSONSchema
from google.genai.types import Schema
from pydantic import Field

from ..utils.variant_utils import get_google_llm_variant


class _ExtendedJSONSchema(JSONSchema):
  property_ordering: Optional[list[str]] = Field(
      default=None,
      description="""Optional. The order of the properties. Not a standard field in open api spec. Only used to support the order of the properties.""",
  )


def _to_snake_case(text: str) -> str:
  """Converts a string into snake_case.

  Handles lowerCamelCase, UpperCamelCase, or space-separated case, acronyms
  (e.g., "REST API") and consecutive uppercase letters correctly.  Also handles
  mixed cases with and without spaces.

  Examples:
  ```
  to_snake_case('camelCase') -> 'camel_case'
  to_snake_case('UpperCamelCase') -> 'upper_camel_case'
  to_snake_case('space separated') -> 'space_separated'
  ```

  Args:
      text: The input string.

  Returns:
      The snake_case version of the string.
  """

  # Handle spaces and non-alphanumeric characters (replace with underscores)
  text = re.sub(r"[^a-zA-Z0-9]+", "_", text)

  # Insert underscores before uppercase letters (handling both CamelCases)
  text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)  # lowerCamelCase
  text = re.sub(
      r"([A-Z]+)([A-Z][a-z])", r"\1_\2", text
  )  # UpperCamelCase and acronyms

  # Convert to lowercase
  text = text.lower()

  # Remove consecutive underscores (clean up extra underscores)
  text = re.sub(r"_+", "_", text)

  # Remove leading and trailing underscores
  text = text.strip("_")

  return text


def _sanitize_schema_type(schema: dict[str, Any]) -> dict[str, Any]:
  if ("type" not in schema or not schema["type"]) and schema.keys().isdisjoint(
      schema
  ):
    schema["type"] = "object"
  if isinstance(schema.get("type"), list):
    nullable = False
    non_null_type = None
    for t in schema["type"]:
      if t == "null":
        nullable = True
      elif not non_null_type:
        non_null_type = t
    if not non_null_type:
      non_null_type = "object"
    if nullable:
      schema["type"] = [non_null_type, "null"]
    else:
      schema["type"] = non_null_type
  elif schema.get("type") == "null":
    schema["type"] = ["object", "null"]

  return schema


def _sanitize_schema_formats_for_gemini(
    schema: dict[str, Any],
) -> dict[str, Any]:
  """Filters the schema to only include fields that are supported by JSONSchema."""
  supported_fields: set[str] = set(_ExtendedJSONSchema.model_fields.keys())
  schema_field_names: set[str] = {"items"}  # 'additional_properties' to come
  list_schema_field_names: set[str] = {
      "any_of",  # 'one_of', 'all_of', 'not' to come
  }
  snake_case_schema = {}
  dict_schema_field_names: tuple[str] = ("properties",)  # 'defs' to come
  for field_name, field_value in schema.items():
    field_name = _to_snake_case(field_name)
    if field_name in schema_field_names:
      snake_case_schema[field_name] = _sanitize_schema_formats_for_gemini(
          field_value
      )
    elif field_name in list_schema_field_names:
      snake_case_schema[field_name] = [
          _sanitize_schema_formats_for_gemini(value) for value in field_value
      ]
    elif field_name in dict_schema_field_names:
      snake_case_schema[field_name] = {
          key: _sanitize_schema_formats_for_gemini(value)
          for key, value in field_value.items()
      }
    # special handle of format field
    elif field_name == "format" and field_value:
      current_type = schema.get("type")
      if (
          # only "int32" and "int64" are supported for integer or number type
          (current_type == "integer" or current_type == "number")
          and field_value in ("int32", "int64")
          or
          # only 'enum' and 'date-time' are supported for STRING type"
          (current_type == "string" and field_value in ("date-time", "enum"))
      ):
        snake_case_schema[field_name] = field_value
    elif field_name in supported_fields and field_value is not None:
      snake_case_schema[field_name] = field_value

  return _sanitize_schema_type(snake_case_schema)


def _to_gemini_schema(openapi_schema: dict[str, Any]) -> Schema:
  """Converts an OpenAPI schema dictionary to a Gemini Schema object."""
  if openapi_schema is None:
    return None

  if not isinstance(openapi_schema, dict):
    raise TypeError("openapi_schema must be a dictionary")

  openapi_schema = _sanitize_schema_formats_for_gemini(openapi_schema)
  return Schema.from_json_schema(
      json_schema=_ExtendedJSONSchema.model_validate(openapi_schema),
      api_option=get_google_llm_variant(),
  )
