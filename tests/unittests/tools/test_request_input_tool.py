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

from unittest.mock import MagicMock

from google.adk.tools._request_input_tool import request_input
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext
import pytest


class TestRequestInputTool:
  """Test cases for RequestInputTool integration and properties."""

  def test_init(self):
    """request_input initializes with correct name and properties."""
    # Given & When & Then
    assert request_input.name == 'adk_request_input'
    assert request_input.is_long_running is True
    assert isinstance(request_input, LongRunningFunctionTool)
    assert (
        'Ask the user a question and wait for their response'
        in request_input.description
    )

  def test_get_declaration(self):
    """request_input returns a function declaration with correct parameters."""
    # Given & When
    declaration = request_input._get_declaration()

    # Then
    assert declaration is not None
    assert declaration.name == 'adk_request_input'
    assert 'Ask the user a question' in declaration.description

    # Verify the parameter schema matches the declaration
    parameters = declaration.parameters
    parameters_schema = declaration.parameters_json_schema
    assert (parameters is not None) or (parameters_schema is not None)

    if parameters_schema is not None:
      # Verify camelCase / snake_case properties in JSON schema format
      assert 'message' in parameters_schema['properties']
      assert (
          'response_schema' in parameters_schema['properties']
          or 'responseSchema' in parameters_schema['properties']
      )
      assert 'message' in parameters_schema['required']

      # Check parameter type specifications
      assert parameters_schema['properties']['message']['type'] == 'string'
    else:
      # Verify types.Schema format
      assert 'message' in parameters.properties
      assert 'response_schema' in parameters.properties
      assert 'message' in parameters.required

      # Check parameter type specifications
      from google.genai import types

      assert parameters.properties['message'].type == types.Type.STRING

  @pytest.mark.asyncio
  async def test_run_async_returns_none(self):
    """request_input execution returns None to trigger LRO suspension."""
    # Given
    args = {'message': 'What is your name?'}
    tool_context = MagicMock(spec=ToolContext)

    # When
    result = await request_input.run_async(args=args, tool_context=tool_context)

    # Then
    assert result is None

  @pytest.mark.asyncio
  async def test_run_async_with_schema_argument_returns_none(self):
    """request_input handles both simple text and structured schema arguments correctly."""
    # Given
    args = {
        'message': 'Enter your username:',
        'response_schema': {'type': 'string'},
    }
    tool_context = MagicMock(spec=ToolContext)

    # When
    result = await request_input.run_async(args=args, tool_context=tool_context)

    # Then
    assert result is None

  @pytest.mark.asyncio
  async def test_run_async_missing_mandatory_message_returns_error(self):
    """request_input returns an error dict if the mandatory 'message' argument is missing."""
    # Given
    args = {'response_schema': {'type': 'string'}}
    tool_context = MagicMock(spec=ToolContext)

    # When
    result = await request_input.run_async(args=args, tool_context=tool_context)

    # Then
    assert isinstance(result, dict)
    assert 'error' in result
    assert 'mandatory input parameters are not present' in result['error']
