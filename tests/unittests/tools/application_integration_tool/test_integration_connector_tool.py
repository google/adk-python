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

from unittest import mock

from google.adk.tools.application_integration_tool.integration_connector_tool import IntegrationConnectorTool
from google.adk.tools.openapi_tool.openapi_spec_parser.rest_api_tool import RestApiTool
from google.genai.types import FunctionDeclaration
from google.genai.types import Schema
from google.genai.types import Tool
from google.genai.types import Type
import pytest


@pytest.fixture
def mock_rest_api_tool():
  """Fixture for a mocked RestApiTool."""
  mock_tool = mock.MagicMock(spec=RestApiTool)
  mock_tool.name = "mock_rest_tool"
  mock_tool.description = "Mock REST tool description."
  # Mock the internal parser needed for _get_declaration
  mock_parser = mock.MagicMock()
  mock_parser.get_json_schema.return_value = {
      "type": "object",
      "properties": {
          "user_id": {"type": "string", "description": "User ID"},
          "connection_name": {"type": "string"},
          "host": {"type": "string"},
          "service_name": {"type": "string"},
          "entity": {"type": "string"},
          "operation": {"type": "string"},
          "action": {"type": "string"},
          "page_size": {"type": "integer"},
          "filter": {"type": "string"},
      },
      "required": ["user_id", "page_size", "filter", "connection_name"],
  }
  mock_tool._operation_parser = mock_parser
  mock_tool.call.return_value = {"status": "success", "data": "mock_data"}
  return mock_tool


@pytest.fixture
def integration_tool(mock_rest_api_tool):
  """Fixture for an IntegrationConnectorTool instance."""
  return IntegrationConnectorTool(
      name="test_integration_tool",
      description="Test integration tool description.",
      connection_name="test-conn",
      connection_host="test.example.com",
      connection_service_name="test-service",
      entity="TestEntity",
      operation="LIST",
      action="TestAction",
      rest_api_tool=mock_rest_api_tool,
  )


def test_get_declaration(integration_tool):
  """Tests the generation of the function declaration."""
  declaration = integration_tool._get_declaration()

  assert isinstance(declaration, FunctionDeclaration)
  assert declaration.name == "test_integration_tool"
  assert declaration.description == "Test integration tool description."

  # Check parameters schema
  params = declaration.parameters
  assert isinstance(params, Schema)
  print(f"params: {params}")
  assert params.type == Type.OBJECT

  # Check properties (excluded fields should not be present)
  assert "user_id" in params.properties
  assert "connection_name" not in params.properties
  assert "host" not in params.properties
  assert "service_name" not in params.properties
  assert "entity" not in params.properties
  assert "operation" not in params.properties
  assert "action" not in params.properties
  assert "page_size" in params.properties
  assert "filter" in params.properties

  # Check required fields (optional and excluded fields should not be required)
  assert "user_id" in params.required
  assert "page_size" not in params.required
  assert "filter" not in params.required
  assert "connection_name" not in params.required


@pytest.mark.asyncio
async def test_run_async(integration_tool, mock_rest_api_tool):
  """Tests the async execution delegates correctly to the RestApiTool."""
  input_args = {"user_id": "user123", "page_size": 10}
  expected_call_args = {
      "user_id": "user123",
      "page_size": 10,
      "connection_name": "test-conn",
      "host": "test.example.com",
      "service_name": "test-service",
      "entity": "TestEntity",
      "operation": "LIST",
      "action": "TestAction",
  }

  result = await integration_tool.run_async(args=input_args, tool_context=None)

  # Assert the underlying rest_api_tool.call was called correctly
  mock_rest_api_tool.call.assert_called_once_with(
      args=expected_call_args, tool_context=None
  )

  # Assert the result is what the mocked call returned
  assert result == {"status": "success", "data": "mock_data"}
