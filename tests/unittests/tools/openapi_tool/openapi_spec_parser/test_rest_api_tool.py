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


import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi.openapi.models import MediaType
from fastapi.openapi.models import Operation
from fastapi.openapi.models import Parameter as OpenAPIParameter
from fastapi.openapi.models import RequestBody
from fastapi.openapi.models import Schema as OpenAPISchema
from google.adk.sessions.state import State
from google.adk.tools.openapi_tool.auth.auth_helpers import token_to_scheme_credential
from google.adk.tools.openapi_tool.common.common import ApiParameter
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_spec_parser import OperationEndpoint
from google.adk.tools.openapi_tool.openapi_spec_parser.operation_parser import OperationParser
from google.adk.tools.openapi_tool.openapi_spec_parser.rest_api_tool import RestApiTool
from google.adk.tools.openapi_tool.openapi_spec_parser.rest_api_tool import snake_to_lower_camel
from google.adk.tools.tool_context import ToolContext
from google.genai.types import FunctionDeclaration
from google.genai.types import Schema
import pytest


class TestRestApiTool:

  @pytest.fixture
  def mock_tool_context(self):
    """Fixture for a mock OperationParser."""
    mock_context = MagicMock(spec=ToolContext)
    mock_context.state = State({}, {})
    mock_context.get_auth_response.return_value = {}
    mock_context.request_credential.return_value = {}
    return mock_context

  @pytest.fixture
  def mock_operation_parser(self):
    """Fixture for a mock OperationParser."""
    mock_parser = MagicMock(spec=OperationParser)
    mock_parser.get_function_name.return_value = "mock_function_name"
    mock_parser.get_json_schema.return_value = {}
    mock_parser.get_parameters.return_value = []
    mock_parser.get_return_type_hint.return_value = "str"
    mock_parser.get_pydoc_string.return_value = "Mock docstring"
    mock_parser.get_signature_parameters.return_value = []
    mock_parser.get_return_type_value.return_value = str
    mock_parser.get_annotations.return_value = {}
    return mock_parser

  @pytest.fixture
  def sample_endpoint(self):
    return OperationEndpoint(
        base_url="https://example.com", path="/test", method="GET"
    )

  @pytest.fixture
  def sample_operation(self):
    return Operation(
        operationId="testOperation",
        description="Test operation",
        parameters=[],
        requestBody=RequestBody(
            content={
                "application/json": MediaType(
                    schema=OpenAPISchema(
                        type="object",
                        properties={
                            "testBodyParam": OpenAPISchema(type="string")
                        },
                    )
                )
            }
        ),
    )

  @pytest.fixture
  def sample_api_parameters(self):
    return [
        ApiParameter(
            original_name="test_param",
            py_name="test_param",
            param_location="query",
            param_schema=OpenAPISchema(type="string"),
            is_required=True,
        ),
        ApiParameter(
            original_name="",
            py_name="test_body_param",
            param_location="body",
            param_schema=OpenAPISchema(type="string"),
            is_required=True,
        ),
    ]

  @pytest.fixture
  def sample_return_parameter(self):
    return ApiParameter(
        original_name="test_param",
        py_name="test_param",
        param_location="query",
        param_schema=OpenAPISchema(type="string"),
        is_required=True,
    )

  @pytest.fixture
  def sample_auth_scheme(self):
    scheme, _ = token_to_scheme_credential(
        "apikey", "header", "", "sample_auth_credential_internal_test"
    )
    return scheme

  @pytest.fixture
  def sample_auth_credential(self):
    _, credential = token_to_scheme_credential(
        "apikey", "header", "", "sample_auth_credential_internal_test"
    )
    return credential

  def test_init(
      self,
      sample_endpoint,
      sample_operation,
      sample_auth_scheme,
      sample_auth_credential,
  ):
    tool = RestApiTool(
        name="test_tool",
        description="Test Tool",
        endpoint=sample_endpoint,
        operation=sample_operation,
        auth_scheme=sample_auth_scheme,
        auth_credential=sample_auth_credential,
    )
    assert tool.name == "test_tool"
    assert tool.description == "Test Tool"
    assert tool.endpoint == sample_endpoint
    assert tool.operation == sample_operation
    assert tool.auth_credential == sample_auth_credential
    assert tool.auth_scheme == sample_auth_scheme
    assert tool.credential_exchanger is not None

  def test_from_parsed_operation_str(
      self,
      sample_endpoint,
      sample_api_parameters,
      sample_return_parameter,
      sample_operation,
  ):
    parsed_operation_str = json.dumps({
        "name": "test_operation",
        "description": "Test Description",
        "endpoint": sample_endpoint.model_dump(),
        "operation": sample_operation.model_dump(),
        "auth_scheme": None,
        "auth_credential": None,
        "parameters": [p.model_dump() for p in sample_api_parameters],
        "return_value": sample_return_parameter.model_dump(),
    })

    tool = RestApiTool.from_parsed_operation_str(parsed_operation_str)
    assert tool.name == "test_operation"

  def test_get_declaration(
      self, sample_endpoint, sample_operation, mock_operation_parser
  ):
    tool = RestApiTool(
        name="test_tool",
        description="Test description",
        endpoint=sample_endpoint,
        operation=sample_operation,
        should_parse_operation=False,
    )
    tool._operation_parser = mock_operation_parser

    declaration = tool._get_declaration()
    assert isinstance(declaration, FunctionDeclaration)
    assert declaration.name == "test_tool"
    assert declaration.description == "Test description"
    assert isinstance(declaration.parameters, Schema)

  @patch(
      "google.adk.tools.openapi_tool.openapi_spec_parser.rest_api_tool.requests.request"
  )
  @pytest.mark.asyncio
  async def test_call_success(
      self,
      mock_request,
      mock_tool_context,
      sample_endpoint,
      sample_operation,
      sample_auth_scheme,
      sample_auth_credential,
  ):
    mock_response = MagicMock()
    mock_response.json.return_value = {"result": "success"}
    mock_request.return_value = mock_response

    tool = RestApiTool(
        name="test_tool",
        description="Test Tool",
        endpoint=sample_endpoint,
        operation=sample_operation,
        auth_scheme=sample_auth_scheme,
        auth_credential=sample_auth_credential,
    )

    # Call the method
    result = await tool.call(args={}, tool_context=mock_tool_context)

    # Check the result
    assert result == {"result": "success"}

  @patch(
      "google.adk.tools.openapi_tool.openapi_spec_parser.rest_api_tool.requests.request"
  )
  @pytest.mark.asyncio
  async def test_call_auth_pending(
      self,
      mock_request,
      sample_endpoint,
      sample_operation,
      sample_auth_scheme,
      sample_auth_credential,
  ):

    tool = RestApiTool(
        name="test_tool",
        description="Test Tool",
        endpoint=sample_endpoint,
        operation=sample_operation,
        auth_scheme=sample_auth_scheme,
        auth_credential=sample_auth_credential,
    )
    with patch(
        "google.adk.tools.openapi_tool.openapi_spec_parser.rest_api_tool.ToolAuthHandler.from_tool_context"
    ) as mock_from_tool_context:
      mock_tool_auth_handler_instance = MagicMock()
      mock_prepare_result = MagicMock()
      mock_prepare_result.state = "pending"
      mock_tool_auth_handler_instance.prepare_auth_credentials = AsyncMock(
          return_value=mock_prepare_result
      )
      mock_from_tool_context.return_value = mock_tool_auth_handler_instance

      response = await tool.call(args={}, tool_context=None)
      assert response == {
          "pending": True,
          "message": "Needs your authorization to access your data.",
      }

  def test_prepare_request_params_query_body(
      self, sample_endpoint, sample_auth_credential, sample_auth_scheme
  ):
    # Create a mock Operation object
    mock_operation = Operation(
        operationId="test_op",
        parameters=[
            OpenAPIParameter(**{
                "name": "testQueryParam",
                "in": "query",
                "schema": OpenAPISchema(type="string"),
            })
        ],
        requestBody=RequestBody(
            content={
                "application/json": MediaType(
                    schema=OpenAPISchema(
                        type="object",
                        properties={
                            "param1": OpenAPISchema(type="string"),
                            "param2": OpenAPISchema(type="integer"),
                        },
                    )
                )
            }
        ),
    )

    tool = RestApiTool(
        name="test_tool",
        description="test",
        endpoint=sample_endpoint,
        operation=mock_operation,
        auth_credential=sample_auth_credential,
        auth_scheme=sample_auth_scheme,
    )

    params = [
        ApiParameter(
            original_name="param1",
            py_name="param1",
            param_location="body",
            param_schema=OpenAPISchema(type="string"),
        ),
        ApiParameter(
            original_name="param2",
            py_name="param2",
            param_location="body",
            param_schema=OpenAPISchema(type="integer"),
        ),
        ApiParameter(
            original_name="testQueryParam",
            py_name="test_query_param",
            param_location="query",
            param_schema=OpenAPISchema(type="string"),
        ),
    ]
    kwargs = {
        "param1": "value1",
        "param2": 123,
        "test_query_param": "query_value",
    }

    request_params = tool._prepare_request_params(params, kwargs)
    assert request_params["method"] == "get"
    assert request_params["url"] == "https://example.com/test"
    assert request_params["json"] == {"param1": "value1", "param2": 123}
    assert request_params["params"] == {"testQueryParam": "query_value"}

  def test_prepare_request_params_array(
      self, sample_endpoint, sample_auth_scheme, sample_auth_credential
  ):
    mock_operation = Operation(
        operationId="test_op",
        requestBody=RequestBody(
            content={
                "application/json": MediaType(
                    schema=OpenAPISchema(
                        type="array", items=OpenAPISchema(type="string")
                    )
                )
            }
        ),
    )

    tool = RestApiTool(
        name="test_tool",
        description="test",
        endpoint=sample_endpoint,
        operation=mock_operation,
        auth_credential=sample_auth_credential,
        auth_scheme=sample_auth_scheme,
    )
    params = [
        ApiParameter(
            original_name="array",  # Match the parameter name
            py_name="array",
            param_location="body",
            param_schema=OpenAPISchema(
                type="array", items=OpenAPISchema(type="string")
            ),
        )
    ]
    kwargs = {"array": ["item1", "item2"]}

    request_params = tool._prepare_request_params(params, kwargs)

    assert request_params["json"] == ["item1", "item2"]

  def test_prepare_request_params_string(
      self, sample_endpoint, sample_auth_credential, sample_auth_scheme
  ):
    mock_operation = Operation(
        operationId="test_op",
        requestBody=RequestBody(
            content={
                "text/plain": MediaType(schema=OpenAPISchema(type="string"))
            }
        ),
    )
    tool = RestApiTool(
        name="test_tool",
        description="Test Tool",
        endpoint=sample_endpoint,
        operation=mock_operation,
        auth_credential=sample_auth_credential,
        auth_scheme=sample_auth_scheme,
    )
    params = [
        ApiParameter(
            original_name="",
            py_name="input_string",
            param_location="body",
            param_schema=OpenAPISchema(type="string"),
        )
    ]
    kwargs = {"input_string": "test_value"}

    request_params = tool._prepare_request_params(params, kwargs)

    assert request_params["data"] == "test_value"
    assert request_params["headers"]["Content-Type"] == "text/plain"

  def test_prepare_request_params_form_data(
      self, sample_endpoint, sample_auth_scheme, sample_auth_credential
  ):
    mock_operation = Operation(
        operationId="test_op",
        requestBody=RequestBody(
            content={
                "application/x-www-form-urlencoded": MediaType(
                    schema=OpenAPISchema(
                        type="object",
                        properties={"key1": OpenAPISchema(type="string")},
                    )
                )
            }
        ),
    )
    tool = RestApiTool(
        name="test_tool",
        description="test",
        endpoint=sample_endpoint,
        operation=mock_operation,
        auth_credential=sample_auth_credential,
        auth_scheme=sample_auth_scheme,
    )
    params = [
        ApiParameter(
            original_name="key1",
            py_name="key1",
            param_location="body",
            param_schema=OpenAPISchema(type="string"),
        )
    ]
    kwargs = {"key1": "value1"}

    request_params = tool._prepare_request_params(params, kwargs)

    assert request_params["data"] == {"key1": "value1"}
    assert (
        request_params["headers"]["Content-Type"]
        == "application/x-www-form-urlencoded"
    )

  def test_prepare_request_params_multipart(
      self, sample_endpoint, sample_auth_credential, sample_auth_scheme
  ):
    mock_operation = Operation(
        operationId="test_op",
        requestBody=RequestBody(
            content={
                "multipart/form-data": MediaType(
                    schema=OpenAPISchema(
                        type="object",
                        properties={
                            "file1": OpenAPISchema(
                                type="string", format="binary"
                            )
                        },
                    )
                )
            }
        ),
    )
    tool = RestApiTool(
        name="test_tool",
        description="test",
        endpoint=sample_endpoint,
        operation=mock_operation,
        auth_credential=sample_auth_credential,
        auth_scheme=sample_auth_scheme,
    )
    params = [
        ApiParameter(
            original_name="file1",
            py_name="file1",
            param_location="body",
            param_schema=OpenAPISchema(type="string", format="binary"),
        )
    ]
    kwargs = {"file1": b"file_content"}

    request_params = tool._prepare_request_params(params, kwargs)

    assert request_params["files"] == {"file1": b"file_content"}
    assert request_params["headers"]["Content-Type"] == "multipart/form-data"

  def test_prepare_request_params_octet_stream(
      self, sample_endpoint, sample_auth_scheme, sample_auth_credential
  ):
    mock_operation = Operation(
        operationId="test_op",
        requestBody=RequestBody(
            content={
                "application/octet-stream": MediaType(
                    schema=OpenAPISchema(type="string", format="binary")
                )
            }
        ),
    )
    tool = RestApiTool(
        name="test_tool",
        description="test",
        endpoint=sample_endpoint,
        operation=mock_operation,
        auth_credential=sample_auth_credential,
        auth_scheme=sample_auth_scheme,
    )
    params = [
        ApiParameter(
            original_name="",
            py_name="data",
            param_location="body",
            param_schema=OpenAPISchema(type="string", format="binary"),
        )
    ]
    kwargs = {"data": b"binary_data"}

    request_params = tool._prepare_request_params(params, kwargs)

    assert request_params["data"] == b"binary_data"
    assert (
        request_params["headers"]["Content-Type"] == "application/octet-stream"
    )

  def test_prepare_request_params_path_param(
      self, sample_endpoint, sample_auth_credential, sample_auth_scheme
  ):
    mock_operation = Operation(operationId="test_op")
    tool = RestApiTool(
        name="test_tool",
        description="Test Tool",
        endpoint=sample_endpoint,
        operation=mock_operation,
        auth_credential=sample_auth_credential,
        auth_scheme=sample_auth_scheme,
    )
    params = [
        ApiParameter(
            original_name="user_id",
            py_name="user_id",
            param_location="path",
            param_schema=OpenAPISchema(type="string"),
        )
    ]
    kwargs = {"user_id": "123"}
    endpoint_with_path = OperationEndpoint(
        base_url="https://example.com", path="/test/{user_id}", method="get"
    )
    tool.endpoint = endpoint_with_path

    request_params = tool._prepare_request_params(params, kwargs)

    assert (
        request_params["url"] == "https://example.com/test/123"
    )  # Path param replaced

  def test_prepare_request_params_header_param(
      self,
      sample_endpoint,
      sample_auth_credential,
      sample_auth_scheme,
      sample_operation,
  ):
    tool = RestApiTool(
        name="test_tool",
        description="Test Tool",
        endpoint=sample_endpoint,
        operation=sample_operation,
        auth_credential=sample_auth_credential,
        auth_scheme=sample_auth_scheme,
    )
    params = [
        ApiParameter(
            original_name="X-Custom-Header",
            py_name="x_custom_header",
            param_location="header",
            param_schema=OpenAPISchema(type="string"),
        )
    ]
    kwargs = {"x_custom_header": "header_value"}

    request_params = tool._prepare_request_params(params, kwargs)

    assert request_params["headers"]["X-Custom-Header"] == "header_value"

  def test_prepare_request_params_cookie_param(
      self,
      sample_endpoint,
      sample_auth_credential,
      sample_auth_scheme,
      sample_operation,
  ):
    tool = RestApiTool(
        name="test_tool",
        description="Test Tool",
        endpoint=sample_endpoint,
        operation=sample_operation,
        auth_credential=sample_auth_credential,
        auth_scheme=sample_auth_scheme,
    )
    params = [
        ApiParameter(
            original_name="session_id",
            py_name="session_id",
            param_location="cookie",
            param_schema=OpenAPISchema(type="string"),
        )
    ]
    kwargs = {"session_id": "cookie_value"}

    request_params = tool._prepare_request_params(params, kwargs)

    assert request_params["cookies"]["session_id"] == "cookie_value"

  def test_prepare_request_params_multiple_mime_types(
      self, sample_endpoint, sample_auth_credential, sample_auth_scheme
  ):
    # Test what happens when multiple mime types are specified. It should take
    # the first one.
    mock_operation = Operation(
        operationId="test_op",
        requestBody=RequestBody(
            content={
                "application/json": MediaType(
                    schema=OpenAPISchema(type="string")
                ),
                "text/plain": MediaType(schema=OpenAPISchema(type="string")),
            }
        ),
    )
    tool = RestApiTool(
        name="test_tool",
        description="Test Tool",
        endpoint=sample_endpoint,
        operation=mock_operation,
        auth_credential=sample_auth_credential,
        auth_scheme=sample_auth_scheme,
    )
    params = [
        ApiParameter(
            original_name="",
            py_name="input",
            param_location="body",
            param_schema=OpenAPISchema(type="string"),
        )
    ]
    kwargs = {"input": "some_value"}

    request_params = tool._prepare_request_params(params, kwargs)

    assert request_params["headers"]["Content-Type"] == "application/json"

  def test_prepare_request_params_unknown_parameter(
      self,
      sample_endpoint,
      sample_auth_credential,
      sample_auth_scheme,
      sample_operation,
  ):
    tool = RestApiTool(
        name="test_tool",
        description="Test Tool",
        endpoint=sample_endpoint,
        operation=sample_operation,
        auth_credential=sample_auth_credential,
        auth_scheme=sample_auth_scheme,
    )
    params = [
        ApiParameter(
            original_name="known_param",
            py_name="known_param",
            param_location="query",
            param_schema=OpenAPISchema(type="string"),
        )
    ]
    kwargs = {"known_param": "value", "unknown_param": "unknown"}

    request_params = tool._prepare_request_params(params, kwargs)

    # Make sure unknown parameters are ignored and do not raise errors.
    assert "unknown_param" not in request_params["params"]

  def test_prepare_request_params_base_url_handling(
      self, sample_auth_credential, sample_auth_scheme, sample_operation
  ):
    # No base_url provided, should use path as is
    tool_no_base = RestApiTool(
        name="test_tool_no_base",
        description="Test Tool",
        endpoint=OperationEndpoint(base_url="", path="/no_base", method="get"),
        operation=sample_operation,
        auth_credential=sample_auth_credential,
        auth_scheme=sample_auth_scheme,
    )
    params = []
    kwargs = {}

    request_params_no_base = tool_no_base._prepare_request_params(
        params, kwargs
    )
    assert request_params_no_base["url"] == "/no_base"

    tool_trailing_slash = RestApiTool(
        name="test_tool",
        description="Test Tool",
        endpoint=OperationEndpoint(
            base_url="https://example.com/", path="/trailing", method="get"
        ),
        operation=sample_operation,
        auth_credential=sample_auth_credential,
        auth_scheme=sample_auth_scheme,
    )

    request_params_trailing = tool_trailing_slash._prepare_request_params(
        params, kwargs
    )
    assert request_params_trailing["url"] == "https://example.com/trailing"

  def test_prepare_request_params_no_unrecognized_query_parameter(
      self,
      sample_endpoint,
      sample_auth_credential,
      sample_auth_scheme,
      sample_operation,
  ):
    tool = RestApiTool(
        name="test_tool",
        description="Test Tool",
        endpoint=sample_endpoint,
        operation=sample_operation,
        auth_credential=sample_auth_credential,
        auth_scheme=sample_auth_scheme,
    )
    params = [
        ApiParameter(
            original_name="unrecognized_param",
            py_name="unrecognized_param",
            param_location="query",
            param_schema=OpenAPISchema(type="string"),
        )
    ]
    kwargs = {"unrecognized_param": None}  # Explicitly passing None
    request_params = tool._prepare_request_params(params, kwargs)

    # Query param not in sample_operation. It should be ignored.
    assert "unrecognized_param" not in request_params["params"]

  def test_prepare_request_params_no_credential(
      self,
      sample_endpoint,
      sample_operation,
  ):
    tool = RestApiTool(
        name="test_tool",
        description="Test Tool",
        endpoint=sample_endpoint,
        operation=sample_operation,
        auth_credential=None,
        auth_scheme=None,
    )
    params = [
        ApiParameter(
            original_name="param_name",
            py_name="param_name",
            param_location="query",
            param_schema=OpenAPISchema(type="string"),
        )
    ]
    kwargs = {"param_name": "aaa", "empty_param": ""}

    request_params = tool._prepare_request_params(params, kwargs)

    assert "param_name" in request_params["params"]
    assert "empty_param" not in request_params["params"]


def test_snake_to_lower_camel():
  assert snake_to_lower_camel("single") == "single"
  assert snake_to_lower_camel("two_words") == "twoWords"
  assert snake_to_lower_camel("three_word_example") == "threeWordExample"
  assert not snake_to_lower_camel("")
  assert snake_to_lower_camel("alreadyCamelCase") == "alreadyCamelCase"
