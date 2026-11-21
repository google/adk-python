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

"""Tests for MCP tool conversion utilities."""

from __future__ import annotations

from unittest import mock

from google.adk.features._feature_registry import FeatureName
from google.adk.features._feature_registry import temporary_feature_override
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.mcp_tool.conversion_utils import adk_to_mcp_tool_type
from google.adk.tools.mcp_tool.conversion_utils import gemini_to_json_schema
from google.genai import types
from jsonschema import Draft202012Validator
import mcp.types as mcp_types
import pytest

from ._sdk_compat import field


class TestAdkToMcpToolType:
  """Tests for adk_to_mcp_tool_type function."""

  @pytest.mark.parametrize("json_schema_enabled", [False, True])
  @pytest.mark.parametrize("record_id", ["record-7", 7])
  def test_function_tool_union_arguments_remain_valid(
      self, env_variables, json_schema_enabled, record_id
  ):
    """Exported tools accept union arguments, including nested array items."""

    def lookup_records(
        record_id: str | int, related_ids: list[str | int]
    ) -> str:
      """Look up records by numeric ID or name."""
      return str(record_id)

    with temporary_feature_override(
        FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, json_schema_enabled
    ):
      result = adk_to_mcp_tool_type(FunctionTool(lookup_records))
    validator = Draft202012Validator(field(result, "inputSchema"))

    validator.validate({"record_id": record_id, "related_ids": ["record-8", 8]})
    assert not validator.is_valid({"record_id": None, "related_ids": []})
    assert not validator.is_valid(
        {"record_id": record_id, "related_ids": [None]}
    )
    assert not validator.is_valid({"related_ids": []})

  def test_tool_with_no_declaration(self):
    """Test conversion when tool has no declaration."""
    mock_tool = mock.Mock(spec=BaseTool)
    mock_tool.name = "test_tool"
    mock_tool.description = "Test tool"
    mock_tool._get_declaration.return_value = None

    result = adk_to_mcp_tool_type(mock_tool)

    assert isinstance(result, mcp_types.Tool)
    assert result.name == "test_tool"
    assert result.description == "Test tool"
    assert field(result, "inputSchema") == {}

  def test_tool_with_parameters_schema(self):
    """Test conversion when tool has parameters Schema object."""
    mock_tool = mock.Mock(spec=BaseTool)
    mock_tool.name = "get_weather"
    mock_tool.description = "Gets weather information"

    declaration = types.FunctionDeclaration(
        name="get_weather",
        description="Gets weather information",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "location": types.Schema(
                    type=types.Type.STRING,
                    description="The location to get weather for",
                ),
                "units": types.Schema(
                    type=types.Type.STRING,
                    description="Temperature units",
                ),
            },
            required=["location"],
        ),
    )
    mock_tool._get_declaration.return_value = declaration

    result = adk_to_mcp_tool_type(mock_tool)

    assert isinstance(result, mcp_types.Tool)
    assert result.name == "get_weather"
    assert result.description == "Gets weather information"
    assert "type" in field(result, "inputSchema")
    assert field(result, "inputSchema")["type"] == "object"
    assert "properties" in field(result, "inputSchema")
    assert "location" in field(result, "inputSchema")["properties"]
    assert "units" in field(result, "inputSchema")["properties"]
    assert (
        field(result, "inputSchema")["properties"]["location"]["type"]
        == "string"
    )
    assert "required" in field(result, "inputSchema")
    assert "location" in field(result, "inputSchema")["required"]

  def test_tool_with_parameters_json_schema(self):
    """Test conversion when tool has parameters_json_schema."""
    mock_tool = mock.Mock(spec=BaseTool)
    mock_tool.name = "search_database"
    mock_tool.description = "Searches a database"

    json_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results",
            },
        },
        "required": ["query"],
    }

    declaration = types.FunctionDeclaration(
        name="search_database",
        description="Searches a database",
        parameters_json_schema=json_schema,
    )
    mock_tool._get_declaration.return_value = declaration

    result = adk_to_mcp_tool_type(mock_tool)

    assert isinstance(result, mcp_types.Tool)
    assert result.name == "search_database"
    assert result.description == "Searches a database"
    # Should use the JSON schema directly
    assert field(result, "inputSchema") == json_schema

  def test_tool_with_no_parameters(self):
    """Test conversion when tool has declaration but no parameters."""
    mock_tool = mock.Mock(spec=BaseTool)
    mock_tool.name = "get_current_time"
    mock_tool.description = "Gets the current time"

    declaration = types.FunctionDeclaration(
        name="get_current_time",
        description="Gets the current time",
    )
    mock_tool._get_declaration.return_value = declaration

    result = adk_to_mcp_tool_type(mock_tool)

    assert isinstance(result, mcp_types.Tool)
    assert result.name == "get_current_time"
    assert result.description == "Gets the current time"
    assert not field(result, "inputSchema")

  def test_tool_prefers_json_schema_over_parameters(self):
    """Test that parameters_json_schema is preferred over parameters."""
    mock_tool = mock.Mock(spec=BaseTool)
    mock_tool.name = "test_tool"
    mock_tool.description = "Test tool"

    json_schema = {
        "type": "object",
        "properties": {
            "json_param": {"type": "string"},
        },
    }

    # Create a declaration with BOTH parameters and parameters_json_schema
    declaration = types.FunctionDeclaration(
        name="test_tool",
        description="Test tool",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "schema_param": types.Schema(type=types.Type.STRING),
            },
        ),
        parameters_json_schema=json_schema,
    )
    mock_tool._get_declaration.return_value = declaration

    result = adk_to_mcp_tool_type(mock_tool)

    # Should use parameters_json_schema, not parameters
    assert field(result, "inputSchema") == json_schema
    assert "json_param" in field(result, "inputSchema")["properties"]
    assert "schema_param" not in field(result, "inputSchema")["properties"]

  def test_tool_with_complex_nested_schema(self):
    """Test conversion with complex nested parameters_json_schema."""
    mock_tool = mock.Mock(spec=BaseTool)
    mock_tool.name = "create_user"
    mock_tool.description = "Creates a new user"

    json_schema = {
        "type": "object",
        "properties": {
            "username": {"type": "string"},
            "profile": {
                "type": "object",
                "properties": {
                    "email": {"type": "string"},
                    "age": {"type": "integer"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["email"],
            },
        },
        "required": ["username", "profile"],
    }

    declaration = types.FunctionDeclaration(
        name="create_user",
        description="Creates a new user",
        parameters_json_schema=json_schema,
    )
    mock_tool._get_declaration.return_value = declaration

    result = adk_to_mcp_tool_type(mock_tool)

    assert isinstance(result, mcp_types.Tool)
    assert field(result, "inputSchema") == json_schema


class TestGeminiToJsonSchema:
  """Tests for gemini_to_json_schema function."""

  def test_non_schema_input_raises_type_error(self):
    """A plain dict is not a Schema and must be rejected, not coerced."""
    with pytest.raises(TypeError, match="Input must be an instance of Schema"):
      gemini_to_json_schema({"type": "STRING"})

  @pytest.mark.parametrize("schema_type", [None, types.Type.TYPE_UNSPECIFIED])
  def test_unspecified_type_does_not_restrict_values(self, schema_type):
    """Omitting a JSON Schema type imposes no type constraint."""
    result = gemini_to_json_schema(types.Schema(type=schema_type))
    validator = Draft202012Validator(result)

    assert "type" not in result
    for value in ["record-7", 7, True, None, [], {}]:
      validator.validate(value)

  def test_explicit_null_type_only_accepts_null(self):
    """An explicit NULL type still restricts values to null."""
    result = gemini_to_json_schema(types.Schema(type=types.Type.NULL))
    validator = Draft202012Validator(result)

    validator.validate(None)
    assert not validator.is_valid("record-7")
    assert not validator.is_valid(7)

  def test_type_is_lower_cased(self):
    """Gemini spells types upper case; JSON Schema requires lower case."""
    assert gemini_to_json_schema(types.Schema(type=types.Type.STRING)) == {
        "type": "string"
    }

  def test_direct_fields_are_copied_under_the_same_name(self):
    """title/description/default/enum/format/example carry over unchanged."""
    schema = types.Schema(
        type=types.Type.STRING,
        title="City",
        description="A city name",
        default="Paris",
        enum=["Paris", "Rome"],
        format="enum",
        example="Rome",
    )

    assert gemini_to_json_schema(schema) == {
        "type": "string",
        "title": "City",
        "description": "A city name",
        "default": "Paris",
        "enum": ["Paris", "Rome"],
        "format": "enum",
        "example": "Rome",
    }

  def test_nullable_true_is_emitted(self):
    schema = types.Schema(type=types.Type.STRING, nullable=True)

    assert gemini_to_json_schema(schema) == {
        "type": "string",
        "nullable": True,
    }

  def test_nullable_false_is_omitted(self):
    """Only an explicit True is meaningful; False is the default already."""
    schema = types.Schema(type=types.Type.STRING, nullable=False)

    assert "nullable" not in gemini_to_json_schema(schema)

  def test_string_constraints_are_renamed_to_camel_case(self):
    schema = types.Schema(
        type=types.Type.STRING,
        pattern="^a.*",
        min_length=2,
        max_length=8,
    )

    assert gemini_to_json_schema(schema) == {
        "type": "string",
        "pattern": "^a.*",
        "minLength": 2,
        "maxLength": 8,
    }

  def test_string_constraints_are_dropped_for_non_string_type(self):
    """minLength on an integer is not valid JSON Schema, so it must not leak."""
    schema = types.Schema(
        type=types.Type.INTEGER, min_length=2, max_length=8, minimum=1
    )

    assert gemini_to_json_schema(schema) == {"type": "integer", "minimum": 1}

  def test_numeric_constraints_are_dropped_for_string_type(self):
    """minimum/maximum are numeric keywords and do not apply to strings."""
    schema = types.Schema(
        type=types.Type.STRING, minimum=1, maximum=5, pattern="x"
    )

    assert gemini_to_json_schema(schema) == {"type": "string", "pattern": "x"}

  def test_numeric_constraints_are_kept_for_number_type(self):
    schema = types.Schema(type=types.Type.NUMBER, minimum=0.5, maximum=9.5)

    assert gemini_to_json_schema(schema) == {
        "type": "number",
        "minimum": 0.5,
        "maximum": 9.5,
    }

  def test_array_items_are_converted_recursively(self):
    """The item schema is itself a Gemini Schema and needs the same mapping."""
    schema = types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(type=types.Type.STRING, max_length=4),
        min_items=1,
        max_items=3,
    )

    assert gemini_to_json_schema(schema) == {
        "type": "array",
        "items": {"type": "string", "maxLength": 4},
        "minItems": 1,
        "maxItems": 3,
    }

  def test_array_without_items_omits_items_key(self):
    schema = types.Schema(type=types.Type.ARRAY)

    assert gemini_to_json_schema(schema) == {"type": "array"}

  def test_object_properties_are_converted_recursively(self):
    schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "name": types.Schema(type=types.Type.STRING, max_length=10),
            "tags": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING),
            ),
        },
        required=["name"],
        min_properties=1,
        max_properties=2,
    )

    assert gemini_to_json_schema(schema) == {
        "type": "object",
        "properties": {
            "name": {"type": "string", "maxLength": 10},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name"],
        "minProperties": 1,
        "maxProperties": 2,
    }

  def test_property_ordering_is_not_emitted(self):
    """property_ordering is a Gemini hint with no JSON Schema equivalent."""
    schema = types.Schema(
        type=types.Type.OBJECT,
        properties={"b": types.Schema(type=types.Type.STRING)},
        property_ordering=["b"],
    )

    result = gemini_to_json_schema(schema)

    assert result == {"type": "object", "properties": {"b": {"type": "string"}}}

  def test_any_of_subschemas_are_converted_recursively(self):
    schema = types.Schema(
        any_of=[
            types.Schema(type=types.Type.STRING),
            types.Schema(type=types.Type.INTEGER, minimum=0),
        ]
    )

    result = gemini_to_json_schema(schema)

    assert result["anyOf"] == [
        {"type": "string"},
        {"type": "integer", "minimum": 0},
    ]
    validator = Draft202012Validator(result)
    validator.validate("record-7")
    validator.validate(7)
    for invalid in [None, -1, 1.5, True, [], {}]:
      assert not validator.is_valid(invalid)

  def test_explicit_type_still_constrains_any_of(self):
    """An explicit type is intersected with anyOf, not silently discarded."""
    schema = types.Schema(
        type=types.Type.STRING,
        any_of=[
            types.Schema(type=types.Type.STRING),
            types.Schema(type=types.Type.INTEGER),
        ],
    )

    result = gemini_to_json_schema(schema)
    validator = Draft202012Validator(result)

    validator.validate("record-7")
    assert not validator.is_valid(7)

  def test_object_any_of_of_objects_keeps_object_type(self):
    """OBJECT is kept when every anyOf branch is an object."""
    schema = types.Schema(
        type=types.Type.OBJECT,
        any_of=[
            types.Schema(type=types.Type.OBJECT, required=["a"]),
            types.Schema(type=types.Type.OBJECT, required=["b"]),
        ],
    )

    result = gemini_to_json_schema(schema)

    assert result["type"] == "object"
