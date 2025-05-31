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

from fastapi.openapi.models import MediaType
from fastapi.openapi.models import Operation
from fastapi.openapi.models import Parameter
from fastapi.openapi.models import RequestBody
from fastapi.openapi.models import Response
from fastapi.openapi.models import Schema
from google.adk.tools.openapi_tool.common.common import ApiParameter
from google.adk.tools.openapi_tool.openapi_spec_parser.operation_parser import OperationParser
import pytest


@pytest.fixture
def sample_operation() -> Operation:
  """Fixture to provide a sample OpenAPI Operation object."""
  return Operation(
      operationId='test_operation',
      summary='Test Summary',
      description='Test Description',
      parameters=[
          Parameter(**{
              'name': 'param1',
              'in': 'query',
              'schema': Schema(type='string'),
              'description': 'Parameter 1',
          }),
          Parameter(**{
              'name': 'param2',
              'in': 'header',
              'schema': Schema(type='string'),
              'description': 'Parameter 2',
          }),
      ],
      requestBody=RequestBody(
          content={
              'application/json': MediaType(
                  schema=Schema(
                      type='object',
                      properties={
                          'prop1': Schema(
                              type='string', description='Property 1'
                          ),
                          'prop2': Schema(
                              type='integer', description='Property 2'
                          ),
                      },
                  )
              )
          },
          description='Request body description',
      ),
      responses={
          '200': Response(
              description='Success',
              content={
                  'application/json': MediaType(schema=Schema(type='string'))
              },
          ),
          '400': Response(description='Client Error'),
      },
      security=[{'oauth2': ['resource: read', 'resource: write']}],
  )


def test_operation_parser_initialization(sample_operation):
  """Test initialization of OperationParser."""
  parser = OperationParser(sample_operation)
  assert parser._operation == sample_operation
  assert len(parser._params) == 4  # 2 params + 2 request body props
  assert parser._return_value is not None


def test_process_operation_parameters(sample_operation):
  """Test _process_operation_parameters method."""
  parser = OperationParser(sample_operation, should_parse=False)
  parser._process_operation_parameters()
  assert len(parser._params) == 2
  assert parser._params[0].original_name == 'param1'
  assert parser._params[0].param_location == 'query'
  assert parser._params[1].original_name == 'param2'
  assert parser._params[1].param_location == 'header'


def test_process_request_body(sample_operation):
  """Test _process_request_body method."""
  parser = OperationParser(sample_operation, should_parse=False)
  parser._process_request_body()
  assert len(parser._params) == 2  # 2 properties in request body
  assert parser._params[0].original_name == 'prop1'
  assert parser._params[0].param_location == 'body'
  assert parser._params[1].original_name == 'prop2'
  assert parser._params[1].param_location == 'body'


def test_process_request_body_array():
  """Test _process_request_body method with array schema."""
  operation = Operation(
      requestBody=RequestBody(
          content={
              'application/json': MediaType(
                  schema=Schema(
                      type='array',
                      items=Schema(
                          type='object',
                          properties={
                              'item_prop1': Schema(
                                  type='string', description='Item Property 1'
                              ),
                              'item_prop2': Schema(
                                  type='integer', description='Item Property 2'
                              ),
                          },
                      ),
                  )
              )
          }
      )
  )

  parser = OperationParser(operation, should_parse=False)
  parser._process_request_body()
  assert len(parser._params) == 1
  assert parser._params[0].original_name == 'array'
  assert parser._params[0].param_location == 'body'
  # Check that schema is correctly propagated and is a dictionary
  assert parser._params[0].param_schema.type == 'array'
  assert parser._params[0].param_schema.items.type == 'object'
  assert 'item_prop1' in parser._params[0].param_schema.items.properties
  assert 'item_prop2' in parser._params[0].param_schema.items.properties
  assert (
      parser._params[0].param_schema.items.properties['item_prop1'].description
      == 'Item Property 1'
  )
  assert (
      parser._params[0].param_schema.items.properties['item_prop2'].description
      == 'Item Property 2'
  )


def test_process_request_body_no_name():
  """Test _process_request_body with a schema that has no properties (unnamed)"""
  operation = Operation(
      requestBody=RequestBody(
          content={'application/json': MediaType(schema=Schema(type='string'))}
      )
  )
  parser = OperationParser(operation, should_parse=False)
  parser._process_request_body()
  assert len(parser._params) == 1
  assert parser._params[0].original_name == ''  # No name
  assert parser._params[0].param_location == 'body'


def test_process_request_body_empty_object():
  """Test _process_request_body with a schema that is of type object but with no properties."""
  operation = Operation(
      requestBody=RequestBody(
          content={'application/json': MediaType(schema=Schema(type='object'))}
      )
  )
  parser = OperationParser(operation, should_parse=False)
  parser._process_request_body()
  assert len(parser._params) == 0


def test_dedupe_param_names(sample_operation):
  """Test _dedupe_param_names method."""
  parser = OperationParser(sample_operation, should_parse=False)
  # Add duplicate named parameters.
  parser._params = [
      ApiParameter(original_name='test', param_location='', param_schema={}),
      ApiParameter(original_name='test', param_location='', param_schema={}),
      ApiParameter(original_name='test', param_location='', param_schema={}),
  ]
  parser._dedupe_param_names()
  assert parser._params[0].py_name == 'test'
  assert parser._params[1].py_name == 'test_0'
  assert parser._params[2].py_name == 'test_1'


def test_process_return_value(sample_operation):
  """Test _process_return_value method."""
  parser = OperationParser(sample_operation, should_parse=False)
  parser._process_return_value()
  assert parser._return_value is not None
  assert parser._return_value.type_hint == 'str'


def test_process_return_value_no_2xx(sample_operation):
  """Tests _process_return_value when no 2xx response exists."""
  operation_no_2xx = Operation(
      responses={'400': Response(description='Client Error')}
  )
  parser = OperationParser(operation_no_2xx, should_parse=False)
  parser._process_return_value()
  assert parser._return_value is not None
  assert parser._return_value.type_hint == 'Any'


def test_process_return_value_multiple_2xx(sample_operation):
  """Tests _process_return_value when multiple 2xx responses exist."""
  operation_multi_2xx = Operation(
      responses={
          '201': Response(
              description='Success',
              content={
                  'application/json': MediaType(schema=Schema(type='integer'))
              },
          ),
          '202': Response(
              description='Success',
              content={'text/plain': MediaType(schema=Schema(type='string'))},
          ),
          '200': Response(
              description='Success',
              content={
                  'application/pdf': MediaType(schema=Schema(type='boolean'))
              },
          ),
          '400': Response(
              description='Failure',
              content={
                  'application/xml': MediaType(schema=Schema(type='object'))
              },
          ),
      }
  )

  parser = OperationParser(operation_multi_2xx, should_parse=False)
  parser._process_return_value()

  assert parser._return_value is not None
  # Take the content type of the 200 response since it's the smallest response
  # code
  assert parser._return_value.param_schema.type == 'boolean'


def test_process_return_value_no_content(sample_operation):
  """Test when 2xx response has no content"""
  operation_no_content = Operation(
      responses={'200': Response(description='Success', content={})}
  )
  parser = OperationParser(operation_no_content, should_parse=False)
  parser._process_return_value()
  assert parser._return_value.type_hint == 'Any'


def test_process_return_value_no_schema(sample_operation):
  """Tests when the 2xx response's content has no schema."""
  operation_no_schema = Operation(
      responses={
          '200': Response(
              description='Success',
              content={'application/json': MediaType(schema=None)},
          )
      }
  )
  parser = OperationParser(operation_no_schema, should_parse=False)
  parser._process_return_value()
  assert parser._return_value.type_hint == 'Any'


def test_get_function_name(sample_operation):
  """Test get_function_name method."""
  parser = OperationParser(sample_operation)
  assert parser.get_function_name() == 'test_operation'


def test_get_function_name_missing_id():
  """Tests get_function_name when operationId is missing"""
  operation = Operation()  # No ID
  parser = OperationParser(operation)
  with pytest.raises(ValueError, match='Operation ID is missing'):
    parser.get_function_name()


def test_get_return_type_hint(sample_operation):
  """Test get_return_type_hint method."""
  parser = OperationParser(sample_operation)
  assert parser.get_return_type_hint() == 'str'


def test_get_return_type_value(sample_operation):
  """Test get_return_type_value method."""
  parser = OperationParser(sample_operation)
  assert parser.get_return_type_value() == str


def test_get_parameters(sample_operation):
  """Test get_parameters method."""
  parser = OperationParser(sample_operation)
  params = parser.get_parameters()
  assert len(params) == 4  # Correct count after processing
  assert all(isinstance(p, ApiParameter) for p in params)


def test_get_return_value(sample_operation):
  """Test get_return_value method."""
  parser = OperationParser(sample_operation)
  return_value = parser.get_return_value()
  assert isinstance(return_value, ApiParameter)


def test_get_auth_scheme_name(sample_operation):
  """Test get_auth_scheme_name method."""
  parser = OperationParser(sample_operation)
  assert parser.get_auth_scheme_name() == 'oauth2'


def test_get_auth_scheme_name_no_security():
  """Test get_auth_scheme_name when no security is present."""
  operation = Operation(responses={})
  parser = OperationParser(operation)
  assert parser.get_auth_scheme_name() == ''


def test_get_pydoc_string(sample_operation):
  """Test get_pydoc_string method."""
  parser = OperationParser(sample_operation)
  pydoc_string = parser.get_pydoc_string()
  assert 'Test Summary' in pydoc_string
  assert 'Args:' in pydoc_string
  assert 'param1 (str): Parameter 1' in pydoc_string
  assert 'prop1 (str): Property 1' in pydoc_string
  assert 'Returns (str):' in pydoc_string
  assert 'Success' in pydoc_string


def test_get_json_schema(sample_operation):
  """Test get_json_schema method."""
  parser = OperationParser(sample_operation)
  json_schema = parser.get_json_schema()
  assert json_schema['title'] == 'test_operation_Arguments'
  assert json_schema['type'] == 'object'
  assert 'param1' in json_schema['properties']
  assert 'prop1' in json_schema['properties']
  # By default nothing is required unless explicitly stated
  assert 'required' not in json_schema or json_schema['required'] == []


def test_get_signature_parameters(sample_operation):
  """Test get_signature_parameters method."""
  parser = OperationParser(sample_operation)
  signature_params = parser.get_signature_parameters()
  assert len(signature_params) == 4
  assert signature_params[0].name == 'param1'
  assert signature_params[0].annotation == str
  assert signature_params[2].name == 'prop1'
  assert signature_params[2].annotation == str


def test_get_annotations(sample_operation):
  """Test get_annotations method."""
  parser = OperationParser(sample_operation)
  annotations = parser.get_annotations()
  assert len(annotations) == 5  # 4 parameters + return
  assert annotations['param1'] == str
  assert annotations['prop1'] == str
  assert annotations['return'] == str


def test_load():
  """Test the load classmethod."""
  operation = Operation(operationId='my_op')  # Minimal operation
  params = [
      ApiParameter(
          original_name='p1',
          param_location='',
          param_schema={'type': 'integer'},
      )
  ]
  return_value = ApiParameter(
      original_name='', param_location='', param_schema={'type': 'string'}
  )

  parser = OperationParser.load(operation, params, return_value)

  assert isinstance(parser, OperationParser)
  assert parser._operation == operation
  assert parser._params == params
  assert parser._return_value == return_value
  assert (
      parser.get_function_name() == 'my_op'
  )  # Check that the operation is loaded


def test_operation_parser_with_dict():
  """Test initialization of OperationParser with a dictionary."""
  operation_dict = {
      'operationId': 'test_dict_operation',
      'parameters': [
          {'name': 'dict_param', 'in': 'query', 'schema': {'type': 'string'}}
      ],
      'responses': {
          '200': {
              'description': 'Dict Success',
              'content': {'application/json': {'schema': {'type': 'string'}}},
          }
      },
  }
  parser = OperationParser(operation_dict)
  assert parser._operation.operationId == 'test_dict_operation'
  assert len(parser._params) == 1
  assert parser._params[0].original_name == 'dict_param'
  assert parser._return_value.type_hint == 'str'
