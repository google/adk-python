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

import json
from typing import Any
from typing import get_args
from typing import get_origin
from typing import Optional

from google.genai import types
from pydantic import BaseModel
from pydantic import TypeAdapter

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
  if is_basemodel_schema(schema):
    # For regular BaseModel, use model_validate_json
    return schema.model_validate_json(json_text).model_dump(exclude_none=True)
  elif is_list_of_basemodel(schema):
    # For list[BaseModel], use TypeAdapter to validate
    type_adapter = TypeAdapter(schema)
    validated = type_adapter.validate_json(json_text)
    return [item.model_dump(exclude_none=True) for item in validated]
  else:
    # For other schema types (list[str], dict, Schema, etc.),
    # just parse JSON without pydantic validation
    return json.loads(json_text)
