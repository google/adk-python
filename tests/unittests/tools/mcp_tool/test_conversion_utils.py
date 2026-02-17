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

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.mcp_tool.conversion_utils import adk_to_mcp_tool_type
from google.genai import types
import mcp.types as mcp_types


class TestAdkToMcpToolType:
  """Tests for adk_to_mcp_tool_type function."""

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
    assert result.inputSchema == {}

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
    assert "type" in result.inputSchema
    assert result.inputSchema["type"] == "object"
    assert "properties" in result.inputSchema
    assert "location" in result.inputSchema["properties"]
    assert "units" in result.inputSchema["properties"]
    assert result.inputSchema["properties"]["location"]["type"] == "string"
    assert "required" in result.inputSchema
    assert "location" in result.inputSchema["required"]

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
    assert result.inputSchema == json_schema

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
    assert not result.inputSchema

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
    assert result.inputSchema == json_schema
    assert "json_param" in result.inputSchema["properties"]
    assert "schema_param" not in result.inputSchema["properties"]

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
    assert result.inputSchema == json_schema
