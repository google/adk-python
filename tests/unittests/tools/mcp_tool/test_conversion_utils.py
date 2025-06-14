from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from google.genai.types import Schema, Type

from src.google.adk.tools.mcp_tool.conversion_utils import (
    adk_to_mcp_tool_type,
    gemini_to_json_schema,
)
from src.google.adk.tools.base_tool import BaseTool
import mcp.types as mcp_types


def test_adk_to_mcp_tool_type():
    """Test adk_to_mcp_tool_type with a mock BaseTool."""
    mock_tool = MagicMock(spec=BaseTool)
    mock_tool.name = "test_tool"
    mock_tool.description = "test_description"
    mock_tool._get_declaration.return_value = None

    mcp_tool = adk_to_mcp_tool_type(mock_tool)

    assert isinstance(mcp_tool, mcp_types.Tool)
    assert mcp_tool.name == "test_tool"
    assert mcp_tool.description == "test_description"
    assert mcp_tool.inputSchema == {}

    mock_declaration = MagicMock()
    mock_declaration.parameters = Schema(type=Type.STRING)
    mock_tool._get_declaration.return_value = mock_declaration
    mcp_tool = adk_to_mcp_tool_type(mock_tool)
    assert mcp_tool.inputSchema == {"type": "string"}


def test_gemini_to_json_schema_string():
    """Test gemini_to_json_schema with a STRING schema."""
    gemini_schema = Schema(type=Type.STRING, title="test_string", description="test string description", default="test", enum=["a", "b"], format="test_format", example="example", pattern="test_pattern", min_length=1, max_length=10)
    json_schema = gemini_to_json_schema(gemini_schema)
    assert json_schema == {
        "type": "string",
        "title": "test_string",
        "description": "test string description",
        "default": "test",
        "enum": ["a", "b"],
        "format": "test_format",
        "example": "example",
        "pattern": "test_pattern",
        "minLength": 1,
        "maxLength": 10,
    }


def test_gemini_to_json_schema_number():
    """Test gemini_to_json_schema with a NUMBER schema."""
    gemini_schema = Schema(type=Type.NUMBER, minimum=1.0, maximum=10.0)
    json_schema = gemini_to_json_schema(gemini_schema)
    assert json_schema == {"type": "number", "minimum": 1.0, "maximum": 10.0}


def test_gemini_to_json_schema_integer():
    """Test gemini_to_json_schema with an INTEGER schema."""
    gemini_schema = Schema(type=Type.INTEGER, minimum=1, maximum=10)
    json_schema = gemini_to_json_schema(gemini_schema)
    assert json_schema == {"type": "integer", "minimum": 1, "maximum": 10}


def test_gemini_to_json_schema_array():
    """Test gemini_to_json_schema with an ARRAY schema."""
    gemini_schema = Schema(type=Type.ARRAY, items=Schema(type=Type.STRING), min_items=1, max_items=10)
    json_schema = gemini_to_json_schema(gemini_schema)
    assert json_schema == {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 10}


def test_gemini_to_json_schema_object():
    """Test gemini_to_json_schema with an OBJECT schema."""
    gemini_schema = Schema(type=Type.OBJECT, properties={"prop1": Schema(type=Type.STRING)}, required=["prop1"], min_properties=1, max_properties=10)
    json_schema = gemini_to_json_schema(gemini_schema)
    assert json_schema == {"type": "object", "properties": {"prop1": {"type": "string"}}, "required": ["prop1"], "minProperties": 1, "maxProperties": 10}


def test_gemini_to_json_schema_nullable():
    """Test gemini_to_json_schema with a nullable schema."""
    gemini_schema = Schema(type=Type.STRING, nullable=True)
    json_schema = gemini_to_json_schema(gemini_schema)
    assert json_schema == {"type": "string", "nullable": True}


def test_gemini_to_json_schema_any_of():
    """Test gemini_to_json_schema with an anyOf schema."""
    gemini_schema = Schema(type=Type.OBJECT, any_of=[Schema(type=Type.STRING), Schema(type=Type.INTEGER)])
    json_schema = gemini_to_json_schema(gemini_schema)
    assert json_schema == {"type": "object", "anyOf": [{"type": "string"}, {"type": "integer"}]}


def test_gemini_to_json_schema_type_error():
    """Test gemini_to_json_schema raises TypeError when input is not a Schema."""
    with pytest.raises(TypeError):
        gemini_to_json_schema("not a schema") 


def test_gemini_to_json_schema_unspecified_type():
    """Test gemini_to_json_schema with an unspecified type."""
    gemini_schema = Schema()
    json_schema = gemini_to_json_schema(gemini_schema)
    assert json_schema == {"type": "null"}