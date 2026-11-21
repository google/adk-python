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

"""Tests for SetModelResponseTool."""

import copy
import inspect
from typing import Optional

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.features._feature_registry import FeatureName
from google.adk.features._feature_registry import temporary_feature_override
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.set_model_response_tool import SetModelResponseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import jsonschema
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic.alias_generators import to_camel
import pytest


class PersonSchema(BaseModel):
  """Test schema for structured output."""

  name: str = Field(description="A person's name")
  age: int = Field(description="A person's age")
  city: str = Field(description='The city they live in')


class ComplexSchema(BaseModel):
  """More complex test schema."""

  id: int
  title: str
  tags: list[str] = Field(default_factory=list)
  metadata: dict[str, str] = Field(default_factory=dict)
  is_active: bool = True


async def _create_invocation_context(agent: LlmAgent) -> InvocationContext:
  """Helper to create InvocationContext for testing."""
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )
  return InvocationContext(
      invocation_id='test-id',
      agent=agent,
      session=session,
      session_service=session_service,
      run_config=RunConfig(),
  )


def test_tool_initialization_simple_schema():
  """Test tool initialization with a simple schema."""
  tool = SetModelResponseTool(PersonSchema)

  assert tool.output_schema == PersonSchema
  assert tool.name == 'set_model_response'
  assert tool.name == SetModelResponseTool.NAME
  assert tool.func.__name__ == SetModelResponseTool.NAME
  assert 'Set your final response' in tool.description
  assert tool.func is not None


def test_tool_initialization_complex_schema():
  """Test tool initialization with a complex schema."""
  tool = SetModelResponseTool(ComplexSchema)

  assert tool.output_schema == ComplexSchema
  assert tool.name == 'set_model_response'
  assert tool.func is not None


def test_function_signature_generation():
  """Test that function signature is correctly generated from schema."""
  tool = SetModelResponseTool(PersonSchema)

  sig = inspect.signature(tool.func)

  # Check that parameters match schema fields
  assert 'name' in sig.parameters
  assert 'age' in sig.parameters
  assert 'city' in sig.parameters

  # All parameters should be keyword-only
  for param in sig.parameters.values():
    assert param.kind == inspect.Parameter.KEYWORD_ONLY


def test_get_declaration():
  """Test that tool declaration is properly generated."""
  tool = SetModelResponseTool(PersonSchema)

  declaration = tool._get_declaration()

  assert declaration is not None
  assert declaration.name == 'set_model_response'
  assert declaration.description is not None


def test_get_declaration_marks_only_schema_required_fields_required():
  """Fields carrying a default must not be advertised as required."""
  tool = SetModelResponseTool(ComplexSchema)

  declaration = tool._get_declaration()

  assert declaration is not None
  schema = declaration.model_dump(exclude_none=True)['parameters_json_schema']
  assert schema['required'] == ComplexSchema.model_json_schema()['required']
  assert sorted(schema['required']) == ['id', 'title']


def test_get_declaration_preserves_field_defaults():
  """Defaults declared on the output schema reach the generated declaration."""
  tool = SetModelResponseTool(ComplexSchema)

  declaration = tool._get_declaration()

  assert declaration is not None
  properties = declaration.model_dump(exclude_none=True)[
      'parameters_json_schema'
  ]['properties']
  assert properties['tags']['default'] == []
  assert properties['metadata']['default'] == {}
  assert properties['is_active']['default'] is True


def test_get_declaration_preserves_field_descriptions():
  """Field(description=...) from output_schema must reach the tool schema."""
  tool = SetModelResponseTool(PersonSchema)

  declaration = tool._get_declaration()

  assert declaration is not None
  properties = declaration.model_dump(exclude_none=True)[
      'parameters_json_schema'
  ]['properties']
  assert properties['name']['description'] == "A person's name"
  assert properties['age']['description'] == "A person's age"
  assert properties['city']['description'] == 'The city they live in'


def test_get_declaration_preserves_nested_basemodel_field_descriptions():
  """Nested BaseModel Field descriptions reach the schema in fallback and default mode."""

  class Address(BaseModel):
    street: str = Field(description='Street address')
    city: str = Field(description='City name')

  class User(BaseModel):
    name: str = Field(description="User's name")
    address: Address = Field(description="User's address")
    previous_addresses: list[Address] = Field(
        default_factory=list, description='Past addresses'
    )

  # Default JSON_SCHEMA_FOR_FUNC_DECL mode
  tool = SetModelResponseTool(User)
  declaration = tool._get_declaration()
  assert declaration is not None
  schema = declaration.model_dump(exclude_none=True)['parameters_json_schema']
  assert schema['properties']['name']['description'] == "User's name"
  assert schema['properties']['address']['description'] == "User's address"
  assert (
      schema['properties']['previous_addresses']['description']
      == 'Past addresses'
  )
  assert (
      schema['$defs']['Address']['properties']['street']['description']
      == 'Street address'
  )
  assert (
      schema['$defs']['Address']['properties']['city']['description']
      == 'City name'
  )

  # Fallback mode (types.Schema)
  with temporary_feature_override(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, False):
    fallback_tool = SetModelResponseTool(User)
    fallback_decl = fallback_tool._get_declaration()

  assert fallback_decl is not None
  assert fallback_decl.parameters is not None
  props = fallback_decl.parameters.properties
  assert props['name'].description == "User's name"
  assert props['address'].description == "User's address"
  assert props['address'].properties['street'].description == 'Street address'
  assert props['address'].properties['city'].description == 'City name'
  assert props['previous_addresses'].description == 'Past addresses'
  assert (
      props['previous_addresses'].items.properties['street'].description
      == 'Street address'
  )
  assert (
      props['previous_addresses'].items.properties['city'].description
      == 'City name'
  )


def test_get_declaration_preserves_list_item_field_descriptions():
  """list[BaseModel] item Field descriptions reach the schema in fallback mode."""

  class Item(BaseModel):
    id: int = Field(description='Item ID')
    name: str = Field(description='Item name')

  with temporary_feature_override(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, False):
    tool = SetModelResponseTool(list[Item])
    declaration = tool._get_declaration()

  assert declaration is not None
  assert declaration.parameters is not None
  items_schema = declaration.parameters.properties['items']
  assert items_schema.items is not None
  item_props = items_schema.items.properties
  assert item_props['id'].description == 'Item ID'
  assert item_props['name'].description == 'Item name'


def test_get_declaration_preserves_list_item_field_descriptions_default_mode():
  """list[BaseModel] item Field descriptions reach the schema in default mode."""

  class Item(BaseModel):
    id: int = Field(description='Item ID')
    name: str = Field(description='Item name')

  tool = SetModelResponseTool(list[Item])
  declaration = tool._get_declaration()

  assert declaration is not None
  schema = declaration.model_dump(exclude_none=True)['parameters_json_schema']
  assert schema['$defs']['Item']['properties']['id']['description'] == 'Item ID'
  assert (
      schema['$defs']['Item']['properties']['name']['description']
      == 'Item name'
  )


@pytest.mark.asyncio
async def test_run_async_valid_data():
  """Test tool execution with valid data."""
  tool = SetModelResponseTool(PersonSchema)

  agent = LlmAgent(name='test_agent', model='gemini-2.5-flash')
  invocation_context = await _create_invocation_context(agent)
  tool_context = ToolContext(invocation_context)

  # Execute with valid data
  result = await tool.run_async(
      args={'name': 'Alice', 'age': 25, 'city': 'Seattle'},
      tool_context=tool_context,
  )

  # Verify the tool now returns dict directly
  assert result is not None
  assert result['name'] == 'Alice'
  assert result['age'] == 25
  assert result['city'] == 'Seattle'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'args',
    [
        {'first_name': 'Ada', 'age_years': 36},
        {'firstName': 'Ada', 'ageYears': 36},
    ],
    ids=['declared_field_names', 'aliases'],
)
async def test_run_async_aliased_schema_accepts_field_names_and_aliases(args):
  """An aliased schema accepts the field names it declares and its aliases."""

  class AliasedSchema(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel)
    first_name: str
    age_years: int

  tool = SetModelResponseTool(AliasedSchema)
  declaration = tool._get_declaration()
  assert declaration is not None
  properties = declaration.model_dump(exclude_none=True)[
      'parameters_json_schema'
  ]['properties']
  assert set(properties) == {'first_name', 'age_years'}

  agent = LlmAgent(name='test_agent', model='gemini-2.5-flash')
  invocation_context = await _create_invocation_context(agent)
  tool_context = ToolContext(invocation_context)

  result = await tool.run_async(args=args, tool_context=tool_context)

  assert result == {'first_name': 'Ada', 'age_years': 36}
  assert tool_context.actions.set_model_response == result


@pytest.mark.asyncio
async def test_run_async_complex_schema():
  """Test tool execution with complex schema."""
  tool = SetModelResponseTool(ComplexSchema)

  agent = LlmAgent(name='test_agent', model='gemini-2.5-flash')
  invocation_context = await _create_invocation_context(agent)
  tool_context = ToolContext(invocation_context)

  # Execute with complex data
  result = await tool.run_async(
      args={
          'id': 123,
          'title': 'Test Item',
          'tags': ['tag1', 'tag2'],
          'metadata': {'key': 'value'},
          'is_active': False,
      },
      tool_context=tool_context,
  )

  # Verify the tool now returns dict directly
  assert result is not None
  assert result['id'] == 123
  assert result['title'] == 'Test Item'
  assert result['tags'] == ['tag1', 'tag2']
  assert result['metadata'] == {'key': 'value'}
  assert result['is_active'] is False
  assert tool_context.actions.set_model_response == result


@pytest.mark.asyncio
async def test_run_async_validation_error():
  """Test tool execution with invalid data returns validation feedback."""
  tool = SetModelResponseTool(PersonSchema)

  agent = LlmAgent(name='test_agent', model='gemini-2.5-flash')
  invocation_context = await _create_invocation_context(agent)
  tool_context = ToolContext(invocation_context)

  # Execute with invalid data (wrong type for age)
  result = await tool.run_async(
      args={'name': 'Bob', 'age': 'not_a_number', 'city': 'Portland'},
      tool_context=tool_context,
  )

  assert result is not None
  assert 'error' in result
  assert 'Validation Error found' in result['error']
  assert 'age' in result['error']
  assert 'int_parsing' in result['error']
  assert tool_context.actions.set_model_response is None


@pytest.mark.asyncio
async def test_run_async_missing_required_field():
  """Test tool execution with missing required field returns feedback."""
  tool = SetModelResponseTool(PersonSchema)

  agent = LlmAgent(name='test_agent', model='gemini-2.5-flash')
  invocation_context = await _create_invocation_context(agent)
  tool_context = ToolContext(invocation_context)

  # Execute with missing required field
  result = await tool.run_async(
      args={'name': 'Charlie', 'city': 'Denver'},  # Missing age
      tool_context=tool_context,
  )

  assert result is not None
  assert 'error' in result
  assert 'Validation Error found' in result['error']
  assert 'age' in result['error']
  assert 'Field required' in result['error']
  assert tool_context.actions.set_model_response is None


@pytest.mark.asyncio
async def test_session_state_storage_key():
  """Test that response is no longer stored in session state."""
  tool = SetModelResponseTool(PersonSchema)

  agent = LlmAgent(name='test_agent', model='gemini-2.5-flash')
  invocation_context = await _create_invocation_context(agent)
  tool_context = ToolContext(invocation_context)

  result = await tool.run_async(
      args={'name': 'Diana', 'age': 35, 'city': 'Miami'},
      tool_context=tool_context,
  )

  # Verify response is returned directly
  assert result is not None
  assert result['name'] == 'Diana'
  assert result['age'] == 35
  assert result['city'] == 'Miami'
  assert tool_context.actions.set_model_response == result


@pytest.mark.asyncio
async def test_multiple_executions_return_latest():
  """Test that multiple executions return latest response independently."""
  tool = SetModelResponseTool(PersonSchema)

  agent = LlmAgent(name='test_agent', model='gemini-2.5-flash')
  invocation_context = await _create_invocation_context(agent)
  tool_context = ToolContext(invocation_context)

  # First execution
  result1 = await tool.run_async(
      args={'name': 'First', 'age': 20, 'city': 'City1'},
      tool_context=tool_context,
  )

  # Second execution should return its own response
  result2 = await tool.run_async(
      args={'name': 'Second', 'age': 30, 'city': 'City2'},
      tool_context=tool_context,
  )

  # Verify each execution returns its own dict
  assert result1['name'] == 'First'
  assert result1['age'] == 20
  assert result1['city'] == 'City1'

  assert result2['name'] == 'Second'
  assert result2['age'] == 30
  assert result2['city'] == 'City2'


def test_function_return_value_consistency():
  """Test that function return value matches run_async return value."""
  tool = SetModelResponseTool(PersonSchema)

  # Direct function call
  direct_result = tool.func()

  # Both should return the same value
  assert direct_result == 'Response set successfully.'


# Tests for list[BaseModel] schema support


class ItemSchema(BaseModel):
  """Simple item schema for list testing."""

  id: int = Field(description='Item ID')
  name: str = Field(description='Item name')


def test_tool_initialization_list_schema():
  """Test tool initialization with a list schema."""
  tool = SetModelResponseTool(list[ItemSchema])

  assert tool.output_schema == list[ItemSchema]
  assert tool._is_list_of_basemodel
  assert tool.name == 'set_model_response'
  assert 'Set your final response' in tool.description
  assert tool.func is not None


def test_function_signature_generation_list_schema():
  """Test that function signature is correctly generated for list schema."""
  tool = SetModelResponseTool(list[ItemSchema])

  sig = inspect.signature(tool.func)

  # Should have a single 'items' parameter
  assert 'items' in sig.parameters
  assert len(sig.parameters) == 1

  # Parameter should be keyword-only with correct annotation
  assert sig.parameters['items'].kind == inspect.Parameter.KEYWORD_ONLY
  assert sig.parameters['items'].annotation == list[ItemSchema]


def test_get_declaration_list_schema():
  """Test that tool declaration is properly generated for list schema."""
  tool = SetModelResponseTool(list[ItemSchema])

  declaration = tool._get_declaration()

  assert declaration is not None
  assert declaration.name == 'set_model_response'
  assert declaration.description is not None


@pytest.mark.asyncio
async def test_run_async_list_schema_valid_data():
  """Test tool execution with valid list data."""
  tool = SetModelResponseTool(list[ItemSchema])

  agent = LlmAgent(name='test_agent', model='gemini-2.5-flash')
  invocation_context = await _create_invocation_context(agent)
  tool_context = ToolContext(invocation_context)

  # Execute with valid list data
  result = await tool.run_async(
      args={
          'items': [
              {'id': 1, 'name': 'Item 1'},
              {'id': 2, 'name': 'Item 2'},
              {'id': 3, 'name': 'Item 3'},
          ]
      },
      tool_context=tool_context,
  )

  # Verify the tool returns list of dicts
  assert result is not None
  assert isinstance(result, list)
  assert len(result) == 3
  assert result[0]['id'] == 1
  assert result[0]['name'] == 'Item 1'
  assert result[1]['id'] == 2
  assert result[2]['id'] == 3


@pytest.mark.asyncio
async def test_run_async_list_schema_empty_list():
  """Test tool execution with empty list."""
  tool = SetModelResponseTool(list[ItemSchema])

  agent = LlmAgent(name='test_agent', model='gemini-2.5-flash')
  invocation_context = await _create_invocation_context(agent)
  tool_context = ToolContext(invocation_context)

  # Execute with empty list
  result = await tool.run_async(
      args={'items': []},
      tool_context=tool_context,
  )

  # Verify the tool returns empty list
  assert result is not None
  assert isinstance(result, list)
  assert len(result) == 0
  assert tool_context.actions.set_model_response == result


@pytest.mark.asyncio
async def test_run_async_list_schema_validation_error():
  """Test tool execution with invalid list data returns validation feedback."""
  tool = SetModelResponseTool(list[ItemSchema])

  agent = LlmAgent(name='test_agent', model='gemini-2.5-flash')
  invocation_context = await _create_invocation_context(agent)
  tool_context = ToolContext(invocation_context)

  # Execute with invalid data (wrong type for id)
  result = await tool.run_async(
      args={
          'items': [
              {'id': 'not_a_number', 'name': 'Item 1'},
          ]
      },
      tool_context=tool_context,
  )

  assert result is not None
  assert 'error' in result
  assert 'Validation Error found' in result['error']
  assert '0.id' in result['error']
  assert 'int_parsing' in result['error']
  assert tool_context.actions.set_model_response is None


# Tests for other schema types (list[str], dict, etc.)


def test_tool_initialization_list_str_schema():
  """Test tool initialization with list[str] schema."""
  tool = SetModelResponseTool(list[str])

  assert tool.output_schema == list[str]
  assert not tool._is_basemodel
  assert not tool._is_list_of_basemodel
  assert tool.name == 'set_model_response'
  assert tool.func is not None


def test_function_signature_generation_list_str_schema():
  """Test that function signature is correctly generated for list[str] schema."""
  tool = SetModelResponseTool(list[str])

  sig = inspect.signature(tool.func)

  # Should have a single 'response' parameter with list[str] annotation
  assert 'response' in sig.parameters
  assert len(sig.parameters) == 1
  assert sig.parameters['response'].kind == inspect.Parameter.KEYWORD_ONLY
  assert sig.parameters['response'].annotation == list[str]


@pytest.mark.asyncio
async def test_run_async_list_str_schema():
  """Test tool execution with list[str] data."""
  tool = SetModelResponseTool(list[str])

  agent = LlmAgent(name='test_agent', model='gemini-2.5-flash')
  invocation_context = await _create_invocation_context(agent)
  tool_context = ToolContext(invocation_context)

  # Execute with list of strings
  result = await tool.run_async(
      args={'response': ['apple', 'banana', 'cherry']},
      tool_context=tool_context,
  )

  # Verify the tool returns the list directly
  assert result is not None
  assert isinstance(result, list)
  assert result == ['apple', 'banana', 'cherry']


def test_tool_initialization_dict_schema():
  """Test tool initialization with dict schema."""
  tool = SetModelResponseTool(dict[str, int])

  assert tool.output_schema == dict[str, int]
  assert not tool._is_basemodel
  assert not tool._is_list_of_basemodel
  assert tool.name == 'set_model_response'
  assert tool.func is not None


def test_function_signature_generation_dict_schema():
  """Test that function signature is correctly generated for dict schema."""
  tool = SetModelResponseTool(dict[str, int])

  sig = inspect.signature(tool.func)

  # Should have a single 'response' parameter with dict[str, int] annotation
  assert 'response' in sig.parameters
  assert len(sig.parameters) == 1
  assert sig.parameters['response'].kind == inspect.Parameter.KEYWORD_ONLY
  assert sig.parameters['response'].annotation == dict[str, int]


@pytest.mark.asyncio
async def test_run_async_dict_schema():
  """Test tool execution with dict data."""
  tool = SetModelResponseTool(dict[str, int])

  agent = LlmAgent(name='test_agent', model='gemini-2.5-flash')
  invocation_context = await _create_invocation_context(agent)
  tool_context = ToolContext(invocation_context)

  # Execute with dict data
  result = await tool.run_async(
      args={'response': {'a': 1, 'b': 2, 'c': 3}},
      tool_context=tool_context,
  )

  # Verify the tool returns the dict directly
  assert result is not None
  assert isinstance(result, dict)
  assert result == {'a': 1, 'b': 2, 'c': 3}


def test_tool_initialization_raw_dict_schema():
  """Raw dict output_schema must not crash and must be stored as-is."""
  raw_schema = {
      'type': 'object',
      'properties': {'result': {'type': 'string'}},
  }

  tool = SetModelResponseTool(raw_schema)

  assert tool.output_schema == raw_schema
  assert not tool._is_basemodel
  assert not tool._is_list_of_basemodel
  assert tool.name == 'set_model_response'
  assert tool.func is not None


def test_function_signature_generation_raw_dict_schema():
  """Raw dict schemas should produce a single `response: dict` parameter.

  The annotation must be the `dict` type (hashable), not the dict instance,
  so downstream `_is_builtin_primitive_or_compound` does not raise
  `TypeError: unhashable type: 'dict'`.
  """
  raw_schema = {
      'type': 'object',
      'properties': {'result': {'type': 'string'}},
  }

  tool = SetModelResponseTool(raw_schema)

  sig = inspect.signature(tool.func)

  assert 'response' in sig.parameters
  assert len(sig.parameters) == 1
  assert sig.parameters['response'].kind == inspect.Parameter.KEYWORD_ONLY
  # The annotation is the hashable `dict` type, not the dict instance.
  assert sig.parameters['response'].annotation is dict


def test_get_declaration_raw_dict_schema():
  """`_get_declaration` must not raise when given a raw dict schema."""
  raw_schema = {
      'type': 'object',
      'properties': {'result': {'type': 'string'}},
  }

  tool = SetModelResponseTool(raw_schema)

  declaration = tool._get_declaration()

  assert declaration is not None
  assert declaration.name == 'set_model_response'
  assert declaration.description is not None


@pytest.mark.asyncio
async def test_run_async_raw_dict_schema():
  """Tool execution with a raw dict schema returns the response unchanged."""
  raw_schema = {
      'type': 'object',
      'properties': {'result': {'type': 'string'}},
  }
  tool = SetModelResponseTool(raw_schema)

  agent = LlmAgent(name='test_agent', model='gemini-1.5-flash')
  invocation_context = await _create_invocation_context(agent)
  tool_context = ToolContext(invocation_context)

  result = await tool.run_async(
      args={'response': {'result': 'hello'}},
      tool_context=tool_context,
  )

  assert result == {'result': 'hello'}


def test_tool_initialization_schema_instance():
  """types.Schema instance output_schema must be converted to dict and not crash."""
  schema_instance = types.Schema(
      type=types.Type.OBJECT,
      properties={'result': types.Schema(type=types.Type.STRING)},
  )

  tool = SetModelResponseTool(schema_instance)

  # Check that it converted it to a dictionary
  assert isinstance(tool.output_schema, dict)
  assert 'result' in tool.output_schema['properties']

  sig = inspect.signature(tool.func)
  assert 'response' in sig.parameters
  assert sig.parameters['response'].annotation is dict

  # Check that get_declaration works and doesn't crash with TypeError
  declaration = tool._get_declaration()
  assert declaration is not None
  assert declaration.name == 'set_model_response'


@pytest.mark.parametrize(
    'raw_schema',
    [
        {
            'type': 'object',
            'properties': {
                'city': {'type': 'string'},
                'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']},
            },
            'required': ['city'],
        },
        {'type': 'array', 'items': {'type': 'string'}},
        {'type': 'string', 'enum': ['yes', 'no']},
    ],
    ids=['object', 'array', 'string_enum'],
)
def test_get_declaration_declares_raw_dict_schema_as_response(raw_schema):
  """The `response` parameter is declared with the raw dict schema itself."""
  tool = SetModelResponseTool(raw_schema)

  declaration = tool._get_declaration()

  parameters = declaration.parameters_json_schema
  assert parameters['properties']['response'] == raw_schema
  assert parameters['required'] == ['response']


@pytest.mark.parametrize(
    ('schema_instance', 'expected_response_schema'),
    [
        (
            types.Schema(
                type=types.Type.OBJECT,
                properties={'city': types.Schema(type=types.Type.STRING)},
                required=['city'],
            ),
            {
                'type': 'object',
                'properties': {'city': {'type': 'string'}},
                'required': ['city'],
            },
        ),
        (
            types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING),
            ),
            {'type': 'array', 'items': {'type': 'string'}},
        ),
    ],
    ids=['object', 'array'],
)
def test_get_declaration_declares_schema_instance_as_response(
    schema_instance, expected_response_schema
):
  """A types.Schema output_schema is declared as the matching JSON Schema."""
  tool = SetModelResponseTool(schema_instance)

  declaration = tool._get_declaration()

  assert (
      declaration.parameters_json_schema['properties']['response']
      == expected_response_schema
  )


def test_get_declaration_resolves_raw_dict_schema_refs():
  """$ref pointers in a raw dict schema still resolve once it is nested."""

  class Address(BaseModel):
    zip_code: str

  class Person(BaseModel):
    name: str
    home: Address

  tool = SetModelResponseTool(Person.model_json_schema())

  declaration = tool._get_declaration()

  validator = jsonschema.Draft202012Validator(
      declaration.parameters_json_schema
  )
  assert validator.is_valid(
      {'response': {'name': 'Ada', 'home': {'zip_code': '94043'}}}
  )
  assert not validator.is_valid({'response': {'name': 'Ada', 'home': {}}})


def test_get_declaration_resolves_raw_dict_schema_draft_07_refs():
  """Draft-07 `definitions` also move to the root, so their refs resolve."""
  tool = SetModelResponseTool({
      'type': 'object',
      'properties': {'home': {'$ref': '#/definitions/Address'}},
      'required': ['home'],
      'definitions': {
          'Address': {
              'type': 'object',
              'properties': {'zip_code': {'type': 'string'}},
              'required': ['zip_code'],
          }
      },
  })

  declaration = tool._get_declaration()

  validator = jsonschema.Draft7Validator(declaration.parameters_json_schema)
  assert validator.is_valid({'response': {'home': {'zip_code': '94043'}}})
  assert not validator.is_valid({'response': {'home': {}}})


def test_get_declaration_converts_schema_instance_defs():
  """The defs of a types.Schema are converted and kept at the root."""
  schema_instance = types.Schema(
      type=types.Type.OBJECT,
      properties={'home': types.Schema(ref='#/$defs/Address')},
      defs={
          'Address': types.Schema(
              type=types.Type.OBJECT,
              properties={
                  'zip_code': types.Schema(
                      type=types.Type.STRING, nullable=True
                  ),
              },
              required=['zip_code'],
          )
      },
  )

  declaration = SetModelResponseTool(schema_instance)._get_declaration()

  parameters = declaration.parameters_json_schema
  assert parameters['$defs'] == {
      'Address': {
          'type': 'object',
          'properties': {'zip_code': {'type': ['null', 'string']}},
          'required': ['zip_code'],
      }
  }
  assert parameters['properties']['response'] == {
      'type': 'object',
      'properties': {'home': {'$ref': '#/$defs/Address'}},
  }


def test_get_declaration_resolves_schema_instance_refs_into_defs():
  """A types.Schema ref spelled `#/defs/...` still resolves under `$defs`."""
  address_ref = types.Schema(ref='#/defs/Address')
  schema_instance = types.Schema(
      type=types.Type.OBJECT,
      properties={
          'home': address_ref,
          'past': types.Schema(type=types.Type.ARRAY, items=address_ref),
          'work': types.Schema(
              any_of=[address_ref, types.Schema(type=types.Type.STRING)]
          ),
      },
      defs={
          'Address': types.Schema(
              type=types.Type.OBJECT,
              properties={'zip_code': types.Schema(type=types.Type.STRING)},
              required=['zip_code'],
          )
      },
  )

  declaration = SetModelResponseTool(schema_instance)._get_declaration()

  validator = jsonschema.Draft202012Validator(
      declaration.parameters_json_schema
  )
  assert validator.is_valid({
      'response': {
          'home': {'zip_code': '94043'},
          'past': [{'zip_code': '10001'}],
          'work': 'remote',
      }
  })
  assert not validator.is_valid({'response': {'home': {}}})
  assert not validator.is_valid({'response': {'past': [{}]}})
  assert not validator.is_valid({'response': {'work': {}}})


def test_get_declaration_lowercases_schema_instance_additional_properties():
  """A schema in additional_properties is declared with lowercase types."""
  schema_instance = types.Schema(
      type=types.Type.OBJECT,
      additional_properties=types.Schema(type=types.Type.STRING),
  )

  declaration = SetModelResponseTool(schema_instance)._get_declaration()

  assert declaration.parameters_json_schema['properties']['response'] == {
      'type': 'object',
      'additionalProperties': {'type': 'string'},
  }


def test_get_declaration_drops_document_keywords_from_raw_dict_schema():
  """$schema and $id, which belong at a document root, are not nested."""
  tool = SetModelResponseTool({
      '$schema': 'https://json-schema.org/draft/2020-12/schema',
      '$id': 'https://example.com/answer.json',
      'type': 'object',
      'properties': {'answer': {'type': 'string'}},
  })

  declaration = tool._get_declaration()

  assert declaration.parameters_json_schema['properties']['response'] == {
      'type': 'object',
      'properties': {'answer': {'type': 'string'}},
  }


def test_get_declaration_lowercases_raw_dict_schema_types():
  """A dict that uses Gemini's uppercase type names gets JSON Schema types."""
  tool = SetModelResponseTool({
      'type': 'OBJECT',
      'properties': {'tags': {'type': 'ARRAY', 'items': {'type': 'STRING'}}},
  })

  declaration = tool._get_declaration()

  assert declaration.parameters_json_schema['properties']['response'] == {
      'type': 'object',
      'properties': {'tags': {'type': 'array', 'items': {'type': 'string'}}},
  }


@pytest.mark.parametrize('json_schema_for_func_decl', [True, False])
@pytest.mark.parametrize(
    'output_schema',
    [
        {
            '$schema': 'https://json-schema.org/draft/2020-12/schema',
            'type': 'OBJECT',
            'properties': {'home': {'$ref': '#/$defs/Address'}},
            '$defs': {'Address': {'type': 'STRING'}},
        },
        types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING),
        ),
    ],
    ids=['raw_dict', 'schema_instance'],
)
def test_get_declaration_leaves_output_schema_unchanged(
    output_schema, json_schema_for_func_decl
):
  """Building the declaration does not modify the caller's output_schema."""
  original = copy.deepcopy(output_schema)

  with temporary_feature_override(
      FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, json_schema_for_func_decl
  ):
    SetModelResponseTool(output_schema)._get_declaration()

  assert output_schema == original


@pytest.mark.parametrize(
    ('raw_schema', 'expected_response_schema'),
    [
        (
            {
                'type': 'object',
                'properties': {'city': {'type': 'string'}},
                'required': ['city'],
            },
            types.Schema(
                type=types.Type.OBJECT,
                properties={'city': types.Schema(type=types.Type.STRING)},
                required=['city'],
            ),
        ),
        (
            {'type': 'array', 'items': {'type': 'string'}},
            types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING),
            ),
        ),
        (
            {'type': 'string', 'enum': ['yes', 'no']},
            types.Schema(type=types.Type.STRING, enum=['yes', 'no']),
        ),
        (
            {'type': 'OBJECT', 'properties': {'a': {'type': 'STRING'}}},
            types.Schema(
                type=types.Type.OBJECT,
                properties={'a': types.Schema(type=types.Type.STRING)},
            ),
        ),
        (
            {
                'type': 'object',
                'properties': {'home': {'$ref': '#/$defs/Address'}},
                '$defs': {'Address': {'type': 'string'}},
            },
            types.Schema(
                type=types.Type.OBJECT,
                properties={'home': types.Schema(type=types.Type.STRING)},
            ),
        ),
    ],
    ids=['object', 'array', 'string_enum', 'uppercase_types', 'refs'],
)
def test_get_declaration_declares_raw_dict_schema_without_json_schema(
    raw_schema, expected_response_schema
):
  """With JSON_SCHEMA_FOR_FUNC_DECL off, a raw dict becomes a types.Schema."""
  with temporary_feature_override(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, False):
    declaration = SetModelResponseTool(raw_schema)._get_declaration()

  assert declaration.parameters_json_schema is None
  assert (
      declaration.parameters.properties['response'] == expected_response_schema
  )


def test_get_declaration_declares_schema_instance_without_json_schema():
  """With JSON_SCHEMA_FOR_FUNC_DECL off, a types.Schema is declared as is."""
  schema_instance = types.Schema(
      type=types.Type.ARRAY,
      items=types.Schema(type=types.Type.STRING),
  )

  with temporary_feature_override(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, False):
    declaration = SetModelResponseTool(schema_instance)._get_declaration()

  assert declaration.parameters.properties['response'] == schema_instance


def test_get_declaration_moves_schema_instance_defs_without_json_schema():
  """With JSON_SCHEMA_FOR_FUNC_DECL off, defs move to the parameters root."""
  address = types.Schema(
      type=types.Type.OBJECT,
      properties={'zip_code': types.Schema(type=types.Type.STRING)},
  )
  schema_instance = types.Schema(
      type=types.Type.OBJECT,
      properties={'home': types.Schema(ref='#/defs/Address')},
      defs={'Address': address},
  )

  with temporary_feature_override(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, False):
    declaration = SetModelResponseTool(schema_instance)._get_declaration()

  assert declaration.parameters.defs == {'Address': address}
  assert declaration.parameters.properties['response'] == types.Schema(
      type=types.Type.OBJECT,
      properties={'home': types.Schema(ref='#/defs/Address')},
  )
  assert schema_instance.defs == {'Address': address}


def test_get_declaration_keeps_any_object_for_unconvertible_raw_dict_schema(
    caplog,
):
  """A dict that types.Schema cannot express keeps `response` as any object."""
  # Draft-07 tuple validation: `items` is a list, which types.Schema lacks.
  raw_schema = {
      'type': 'array',
      'items': [{'type': 'integer'}, {'type': 'string'}],
  }

  with (
      temporary_feature_override(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, False),
      caplog.at_level('WARNING', logger='google_adk'),
  ):
    declaration = SetModelResponseTool(raw_schema)._get_declaration()

  assert declaration.parameters.properties['response'] == types.Schema(
      type=types.Type.OBJECT
  )
  assert 'Could not convert the output schema' in caplog.text


class SubSchema(BaseModel):

  field1: str = Field(description='Field 1')
  field2: int = Field(description='Field 2')


class ConsolidatedOptionalSchema(BaseModel):

  nested: Optional[SubSchema] = Field(default=None, description='Nested model')
  nested_list: Optional[list[SubSchema]] = Field(
      default=None, description='Nested list of models'
  )
  pep604_nested: SubSchema | None = Field(
      default=None, description='PEP 604 optional nested model'
  )
  pep604_raw_list: list | None = Field(default=None, description='Raw list')


def test_get_declaration_optional_fields():
  """Test that tool declaration preserves properties for various optional fields."""
  with temporary_feature_override(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, False):
    tool = SetModelResponseTool(ConsolidatedOptionalSchema)

    declaration = tool._get_declaration()

  assert declaration is not None
  assert declaration.name == 'set_model_response'
  params_schema = declaration.parameters
  assert params_schema is not None
  assert params_schema.type == 'OBJECT'

  # 1. Optional[SubSchema]
  assert 'nested' in params_schema.properties
  nested_schema = params_schema.properties['nested']
  assert nested_schema.type == 'OBJECT'
  assert nested_schema.properties is not None
  assert nested_schema.properties['field1'].type == 'STRING'
  assert nested_schema.properties['field2'].type == 'INTEGER'

  # 2. Optional[list[SubSchema]]
  assert 'nested_list' in params_schema.properties
  nested_list_schema = params_schema.properties['nested_list']
  assert nested_list_schema.type == 'ARRAY'
  assert nested_list_schema.items is not None
  items_schema = nested_list_schema.items
  assert items_schema.type == 'OBJECT'
  assert items_schema.properties is not None
  assert items_schema.properties['field1'].type == 'STRING'
  assert items_schema.properties['field2'].type == 'INTEGER'

  # 3. SubSchema | None (PEP 604)
  assert 'pep604_nested' in params_schema.properties
  pep604_nested_schema = params_schema.properties['pep604_nested']
  assert pep604_nested_schema.type == 'OBJECT'
  assert pep604_nested_schema.properties is not None
  assert pep604_nested_schema.properties['field1'].type == 'STRING'
  assert pep604_nested_schema.properties['field2'].type == 'INTEGER'

  # 4. list | None (PEP 604)
  assert 'pep604_raw_list' in params_schema.properties
  pep604_raw_list_schema = params_schema.properties['pep604_raw_list']
  assert pep604_raw_list_schema.type == 'ARRAY'
