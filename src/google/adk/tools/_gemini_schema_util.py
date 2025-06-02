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


def _sanitize_schema_formats_for_gemini(schema_node: Any) -> Any:
  """Helper function to sanitize schema formats for Gemini compatibility"""
  if isinstance(schema_node, dict):
    new_node = {}
    current_type = schema_node.get("type")

    for key, value in schema_node.items():
      key = _to_snake_case(key)

      # special handle of format field
      if key == "format":
        current_format = value
        format_to_keep = None
        if current_format:
          if current_type == "integer" or current_type == "number":
            if current_format in ("int32", "int64"):
              format_to_keep = current_format
          elif current_type == "string":
            # only 'enum' and 'date-time' are supported for STRING type"
            if current_format in ("date-time", "enum"):
              format_to_keep = current_format
          # For any other type or unhandled format
          # the 'format' key will be effectively removed for that node.
          if format_to_keep:
            new_node[key] = format_to_keep
        continue
      # don't change property name
      if key == "properties":
        new_node[key] = {
            k: _sanitize_schema_formats_for_gemini(v) for k, v in value.items()
        }
        continue
      # Recursively sanitize other parts of the schema
      new_node[key] = _sanitize_schema_formats_for_gemini(value)
    return new_node
  elif isinstance(schema_node, list):
    return [_sanitize_schema_formats_for_gemini(item) for item in schema_node]
  else:
    return schema_node


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
