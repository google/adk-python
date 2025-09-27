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

from typing import Any
from typing import Dict
from typing import List

from fastapi.openapi.models import Response
from fastapi.openapi.models import Schema
from google.adk.tools.openapi_tool.common.common import ApiParameter
from google.adk.tools.openapi_tool.common.common import PydocHelper
from google.adk.tools.openapi_tool.common.common import rename_python_keywords
from google.adk.tools.openapi_tool.common.common import TypeHintHelper
import pytest


def dict_to_responses(input: Dict[str, Any]) -> Dict[str, Response]:
  return {k: Response.model_validate(input[k]) for k in input}


class TestRenamePythonKeywords:

  @pytest.mark.parametrize(
      'input_str, expected_output',
      [
          ('in', 'param_in'),
          ('for', 'param_for'),
          ('class', 'param_class'),
          ('normal', 'normal'),
          ('param_if', 'param_if'),
          ('', ''),
      ],
  )
  def test_rename_python_keywords(self, input_str, expected_output):
    assert rename_python_keywords(input_str) == expected_output


class TestApiParameter:

  def test_api_parameter_initialization(self):
    schema = Schema(type='string', description='A string parameter')
    param = ApiParameter(
        original_name='testParam',
        description='A string description',
        param_location='query',
        param_schema=schema,
    )
    assert param.original_name == 'testParam'
    assert param.param_location == 'query'
    assert param.param_schema.type == 'string'
    assert param.param_schema.description == 'A string parameter'
    assert param.py_name == 'test_param'
    assert param.type_hint == 'str'
    assert param.type_value == str
    assert param.description == 'A string description'

  def test_api_parameter_keyword_rename(self):
    schema = Schema(type='string')
    param = ApiParameter(
        original_name='in',
        param_location='query',
        param_schema=schema,
    )
    assert param.py_name == 'param_in'

  def test_api_parameter_custom_py_name(self):
    schema = Schema(type='integer')
    param = ApiParameter(
        original_name='testParam',
        param_location='query',
        param_schema=schema,
        py_name='custom_name',
    )
    assert param.py_name == 'custom_name'

  def test_api_parameter_str_representation(self):
    schema = Schema(type='number')
    param = ApiParameter(
        original_name='testParam',
        param_location='query',
        param_schema=schema,
    )
    assert str(param) == 'test_param: float'

  def test_api_parameter_to_arg_string(self):
    schema = Schema(type='boolean')
    param = ApiParameter(
        original_name='testParam',
        param_location='query',
        param_schema=schema,
    )
    assert param.to_arg_string() == 'test_param=test_param'

  def test_api_parameter_to_dict_property(self):
    schema = Schema(type='string')
    param = ApiParameter(
        original_name='testParam',
        param_location='path',
        param_schema=schema,
    )
    assert param.to_dict_property() == '"test_param": test_param'

  def test_api_parameter_model_serializer(self):
    schema = Schema(type='string', description='test description')
    param = ApiParameter(
        original_name='TestParam',
        param_location='path',
        param_schema=schema,
        py_name='test_param_custom',
        description='test description',
    )

    serialized_param = param.model_dump(mode='json', exclude_none=True)

    assert serialized_param == {
        'original_name': 'TestParam',
        'param_location': 'path',
        'param_schema': {'type': 'string', 'description': 'test description'},
        'description': 'test description',
        'py_name': 'test_param_custom',
    }

  @pytest.mark.parametrize(
      'schema, expected_type_value, expected_type_hint',
      [
          ({'type': 'integer'}, int, 'int'),
          ({'type': 'number'}, float, 'float'),
          ({'type': 'boolean'}, bool, 'bool'),
          ({'type': 'string'}, str, 'str'),
          (
              {'type': 'string', 'format': 'date'},
              str,
              'str',
          ),
          (
              {'type': 'string', 'format': 'date-time'},
              str,
              'str',
          ),
          (
              {'type': 'array', 'items': {'type': 'integer'}},
              List[int],
              'List[int]',
          ),
          (
              {'type': 'array', 'items': {'type': 'string'}},
              List[str],
              'List[str]',
          ),
          (
              {
                  'type': 'array',
                  'items': {'type': 'object'},
              },
              List[Dict[str, Any]],
              'List[Dict[str, Any]]',
          ),
          ({'type': 'object'}, Dict[str, Any], 'Dict[str, Any]'),
          ({}, Any, 'Any'),
      ],
  )
  def test_api_parameter_type_hint_helper(
      self, schema, expected_type_value, expected_type_hint
  ):
    param = ApiParameter(
        original_name='test', param_location='query', param_schema=schema
    )
    assert param.type_value == expected_type_value
    assert param.type_hint == expected_type_hint
    assert (
        TypeHintHelper.get_type_hint(param.param_schema) == expected_type_hint
    )
    assert (
        TypeHintHelper.get_type_value(param.param_schema) == expected_type_value
    )

  def test_api_parameter_description(self):
    schema = Schema(type='string')
    param = ApiParameter(
        original_name='param1',
        param_location='query',
        param_schema=schema,
        description='The description',
    )
    assert param.description == 'The description'

  def test_api_parameter_description_use_schema_fallback(self):
    schema = Schema(type='string', description='The description')
    param = ApiParameter(
        original_name='param1',
        param_location='query',
        param_schema=schema,
    )
    assert param.description == 'The description'


class TestTypeHintHelper:

  @pytest.mark.parametrize(
      'schema, expected_type_value, expected_type_hint',
      [
          ({'type': 'integer'}, int, 'int'),
          ({'type': 'number'}, float, 'float'),
          ({'type': 'string'}, str, 'str'),
          (
              {
                  'type': 'array',
                  'items': {'type': 'string'},
              },
              List[str],
              'List[str]',
          ),
      ],
  )
  def test_get_type_value_and_hint(
      self, schema, expected_type_value, expected_type_hint
  ):

    param = ApiParameter(
        original_name='test_param',
        param_location='query',
        param_schema=schema,
        description='Test parameter',
    )
    assert (
        TypeHintHelper.get_type_value(param.param_schema) == expected_type_value
    )
    assert (
        TypeHintHelper.get_type_hint(param.param_schema) == expected_type_hint
    )


class TestPydocHelper:

  def test_generate_param_doc_simple(self):
    schema = Schema(type='string')
    param = ApiParameter(
        original_name='test_param',
        param_location='query',
        param_schema=schema,
        description='Test description',
    )

    expected_doc = 'test_param (str): Test description'
    assert PydocHelper.generate_param_doc(param) == expected_doc

  def test_generate_param_doc_no_description(self):
    schema = Schema(type='integer')
    param = ApiParameter(
        original_name='test_param',
        param_location='query',
        param_schema=schema,
    )
    expected_doc = 'test_param (int): '
    assert PydocHelper.generate_param_doc(param) == expected_doc

  def test_generate_param_doc_object(self):
    schema = Schema(
        type='object',
        properties={
            'prop1': {'type': 'string', 'description': 'Prop1 desc'},
            'prop2': {'type': 'integer'},
        },
    )
    param = ApiParameter(
        original_name='test_param',
        param_location='query',
        param_schema=schema,
        description='Test object parameter',
    )
    expected_doc = (
        'test_param (Dict[str, Any]): Test object parameter Object'
        ' properties:\n       prop1 (str): Prop1 desc\n       prop2'
        ' (int): \n'
    )
    assert PydocHelper.generate_param_doc(param) == expected_doc

  def test_generate_param_doc_object_no_properties(self):
    schema = Schema(type='object', description='A test schema')
    param = ApiParameter(
        original_name='test_param',
        param_location='query',
        param_schema=schema,
        description='The description.',
    )
    expected_doc = 'test_param (Dict[str, Any]): The description.'
    assert PydocHelper.generate_param_doc(param) == expected_doc

  def test_generate_return_doc_simple(self):
    responses = {
        '200': {
            'description': 'Successful response',
            'content': {'application/json': {'schema': {'type': 'string'}}},
        }
    }
    expected_doc = 'Returns (str): Successful response'
    assert (
        PydocHelper.generate_return_doc(dict_to_responses(responses))
        == expected_doc
    )

  def test_generate_return_doc_no_content(self):
    responses = {'204': {'description': 'No content'}}
    assert not PydocHelper.generate_return_doc(dict_to_responses(responses))

  def test_generate_return_doc_object(self):
    responses = {
        '200': {
            'description': 'Successful object response',
            'content': {
                'application/json': {
                    'schema': {
                        'type': 'object',
                        'properties': {
                            'prop1': {
                                'type': 'string',
                                'description': 'Prop1 desc',
                            },
                            'prop2': {'type': 'integer'},
                        },
                    }
                }
            },
        }
    }

    return_doc = PydocHelper.generate_return_doc(dict_to_responses(responses))

    assert 'Returns (Dict[str, Any]): Successful object response' in return_doc
    assert 'prop1 (str): Prop1 desc' in return_doc
    assert 'prop2 (int):' in return_doc

  def test_generate_return_doc_multiple_success(self):
    responses = {
        '200': {
            'description': 'Successful response',
            'content': {'application/json': {'schema': {'type': 'string'}}},
        },
        '400': {'description': 'Bad request'},
    }
    expected_doc = 'Returns (str): Successful response'
    assert (
        PydocHelper.generate_return_doc(dict_to_responses(responses))
        == expected_doc
    )

  def test_generate_return_doc_2xx_smallest_status_code_response(self):
    responses = {
        '201': {
            'description': '201 response',
            'content': {'application/json': {'schema': {'type': 'integer'}}},
        },
        '200': {
            'description': '200 response',
            'content': {'application/json': {'schema': {'type': 'string'}}},
        },
        '400': {'description': 'Bad request'},
    }

    expected_doc = 'Returns (str): 200 response'
    assert (
        PydocHelper.generate_return_doc(dict_to_responses(responses))
        == expected_doc
    )

  def test_generate_return_doc_contentful_response(self):
    responses = {
        '200': {'description': 'No content response'},
        '201': {
            'description': '201 response',
            'content': {'application/json': {'schema': {'type': 'string'}}},
        },
        '400': {'description': 'Bad request'},
    }
    expected_doc = 'Returns (str): 201 response'
    assert (
        PydocHelper.generate_return_doc(dict_to_responses(responses))
        == expected_doc
    )


if __name__ == '__main__':
  pytest.main([__file__])
