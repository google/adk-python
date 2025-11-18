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

from typing import Any
from typing import Dict

from google.genai.types import Schema
from google.genai.types import Type
import mcp.types as mcp_types

from ..base_tool import BaseTool


def adk_to_mcp_tool_type(tool: BaseTool) -> mcp_types.Tool:
  """Convert a Tool in ADK into MCP tool type.

  This function transforms an ADK tool definition into its equivalent
  representation in the MCP (Model Context Protocol) system.

  Args:
      tool: The ADK tool to convert. It should be an instance of a class derived
        from `BaseTool`.

  Returns:
      An object of MCP Tool type, representing the converted tool.

  Examples:
      # Assuming 'my_tool' is an instance of a BaseTool derived class
      mcp_tool = adk_to_mcp_tool_type(my_tool)
      print(mcp_tool)
  """
  tool_declaration = tool._get_declaration()
  if not tool_declaration:
    input_schema = {}
  elif tool_declaration.parameters_json_schema:
    # Use JSON schema directly if available
    input_schema = tool_declaration.parameters_json_schema
  elif tool_declaration.parameters:
    # Convert from Schema object
    input_schema = gemini_to_json_schema(tool_declaration.parameters)
  else:
    input_schema = {}
  return mcp_types.Tool(
      name=tool.name,
      description=tool.description,
      inputSchema=input_schema,
  )


def gemini_to_json_schema(gemini_schema: Schema) -> Dict[str, Any]:
  """Converts a Gemini Schema object into a JSON Schema dictionary.

  Args:
      gemini_schema: An instance of the Gemini Schema class.

  Returns:
      A dictionary representing the equivalent JSON Schema.

  Raises:
      TypeError: If the input is not an instance of the expected Schema class.
      ValueError: If an invalid Gemini Type enum value is encountered.
  """
  if not isinstance(gemini_schema, Schema):
    raise TypeError(
        f"Input must be an instance of Schema, got {type(gemini_schema)}"
    )

  json_schema_dict: Dict[str, Any] = {}

  # Map Type
  gemini_type = getattr(gemini_schema, "type", None)
  if gemini_type and gemini_type != Type.TYPE_UNSPECIFIED:
    json_schema_dict["type"] = gemini_type.lower()
  else:
    json_schema_dict["type"] = "null"

  # Map Nullable
  if getattr(gemini_schema, "nullable", None) == True:
    json_schema_dict["nullable"] = True

  # --- Map direct fields ---
  direct_mappings = {
      "title": "title",
      "description": "description",
      "default": "default",
      "enum": "enum",
      "format": "format",
      "example": "example",
  }
  for gemini_key, json_key in direct_mappings.items():
    value = getattr(gemini_schema, gemini_key, None)
    if value is not None:
      json_schema_dict[json_key] = value

  # String validation
  if gemini_type == Type.STRING:
    str_mappings = {
        "pattern": "pattern",
        "min_length": "minLength",
        "max_length": "maxLength",
    }
    for gemini_key, json_key in str_mappings.items():
      value = getattr(gemini_schema, gemini_key, None)
      if value is not None:
        json_schema_dict[json_key] = value

  # Number/Integer validation
  if gemini_type in (Type.NUMBER, Type.INTEGER):
    num_mappings = {
        "minimum": "minimum",
        "maximum": "maximum",
    }
    for gemini_key, json_key in num_mappings.items():
      value = getattr(gemini_schema, gemini_key, None)
      if value is not None:
        json_schema_dict[json_key] = value

  # Array validation (Recursive call for items)
  if gemini_type == Type.ARRAY:
    items_schema = getattr(gemini_schema, "items", None)
    if items_schema is not None:
      json_schema_dict["items"] = gemini_to_json_schema(items_schema)

    arr_mappings = {
        "min_items": "minItems",
        "max_items": "maxItems",
    }
    for gemini_key, json_key in arr_mappings.items():
      value = getattr(gemini_schema, gemini_key, None)
      if value is not None:
        json_schema_dict[json_key] = value

  # Object validation (Recursive call for properties)
  if gemini_type == Type.OBJECT:
    properties_dict = getattr(gemini_schema, "properties", None)
    if properties_dict is not None:
      json_schema_dict["properties"] = {
          prop_name: gemini_to_json_schema(prop_schema)
          for prop_name, prop_schema in properties_dict.items()
      }

    obj_mappings = {
        "required": "required",
        "min_properties": "minProperties",
        "max_properties": "maxProperties",
        # Note: Ignoring 'property_ordering' as it's not standard JSON Schema
    }
    for gemini_key, json_key in obj_mappings.items():
      value = getattr(gemini_schema, gemini_key, None)
      if value is not None:
        json_schema_dict[json_key] = value

  # Map anyOf (Recursive call for subschemas)
  any_of_list = getattr(gemini_schema, "any_of", None)
  if any_of_list is not None:
    json_schema_dict["anyOf"] = [
        gemini_to_json_schema(sub_schema) for sub_schema in any_of_list
    ]

  return json_schema_dict
