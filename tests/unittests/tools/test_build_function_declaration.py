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

from enum import Enum
from typing import Any

from google.adk.features import FeatureName
from google.adk.features._feature_registry import temporary_feature_override
from google.adk.tools import _automatic_function_calling_util
from google.adk.tools import _function_parameter_parse_util
from google.adk.tools.tool_context import ToolContext
from google.adk.utils.variant_utils import GoogleLLMVariant
from google.genai import types
# TODO: crewai requires python 3.10 as minimum
# from crewai_tools import FileReadTool
from pydantic import BaseModel
import pytest


class TestBuildFunctionDeclarationLegacy:

  @pytest.fixture(autouse=True)
  def disable_feature_flag(self):
    """Disable the JSON_SCHEMA_FOR_FUNC_DECL feature flag for legacy tests."""
    with temporary_feature_override(
        FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, False
    ):
      yield

  def test_string_input(self):
    def simple_function(input_str: str) -> str:
      return {'result': input_str}

    function_decl = _automatic_function_calling_util.build_function_declaration(
        func=simple_function
    )

    assert function_decl.name == 'simple_function'
    assert function_decl.parameters.type == 'OBJECT'
    assert function_decl.parameters.properties['input_str'].type == 'STRING'

  def test_int_input(self):
    def simple_function(input_str: int) -> str:
      return {'result': input_str}

    function_decl = _automatic_function_calling_util.build_function_declaration(
        func=simple_function
    )

    assert function_decl.name == 'simple_function'
    assert function_decl.parameters.type == 'OBJECT'
    assert function_decl.parameters.properties['input_str'].type == 'INTEGER'

  def test_float_input(self):
    def simple_function(input_str: float) -> str:
      return {'result': input_str}

    function_decl = _automatic_function_calling_util.build_function_declaration(
        func=simple_function
    )

    assert function_decl.name == 'simple_function'
    assert function_decl.parameters.type == 'OBJECT'
    assert function_decl.parameters.properties['input_str'].type == 'NUMBER'

  def test_bool_input(self):
    def simple_function(input_str: bool) -> str:
      return {'result': input_str}

    function_decl = _automatic_function_calling_util.build_function_declaration(
        func=simple_function
    )

    assert function_decl.name == 'simple_function'
    assert function_decl.parameters.type == 'OBJECT'
    assert function_decl.parameters.properties['input_str'].type == 'BOOLEAN'

  def test_array_input(self):
    def simple_function(input_str: list[str]) -> str:
      return {'result': input_str}

    function_decl = _automatic_function_calling_util.build_function_declaration(
        func=simple_function
    )

    assert function_decl.name == 'simple_function'
    assert function_decl.parameters.type == 'OBJECT'
    assert function_decl.parameters.properties['input_str'].type == 'ARRAY'

  def test_dict_input(self):
    def simple_function(input_str: dict[str, str]) -> str:
      return {'result': input_str}

    function_decl = _automatic_function_calling_util.build_function_declaration(
        func=simple_function
    )

    assert function_decl.name == 'simple_function'
    assert function_decl.parameters.type == 'OBJECT'
    assert function_decl.parameters.properties['input_str'].type == 'OBJECT'
    assert (
        function_decl.parameters.properties[
            'input_str'
        ].additional_properties.type
        == 'STRING'
    )

  def test_dict_input_with_int_values(self):
    def simple_function(input_str: dict[str, int]) -> str:
      return {'result': input_str}

    function_decl = _automatic_function_calling_util.build_function_declaration(
        func=simple_function
    )

    assert function_decl.name == 'simple_function'
    assert function_decl.parameters.type == 'OBJECT'
    assert function_decl.parameters.properties['input_str'].type == 'OBJECT'
    assert (
        function_decl.parameters.properties[
            'input_str'
        ].additional_properties.type
        == 'INTEGER'
    )

  def test_dict_input_with_any_values(self):
    def simple_function(input_str: dict[str, Any]) -> str:
      return {'result': input_str}

    function_decl = _automatic_function_calling_util.build_function_declaration(
        func=simple_function
    )

    assert function_decl.name == 'simple_function'
    assert function_decl.parameters.type == 'OBJECT'
    assert function_decl.parameters.properties['input_str'].type == 'OBJECT'
    assert (
        function_decl.parameters.properties[
            'input_str'
        ].additional_properties.type
        is None
    )

  def test_untyped_dict_input(self):
    def simple_function(input_str: dict) -> str:
      return {'result': input_str}

    function_decl = _automatic_function_calling_util.build_function_declaration(
        func=simple_function
    )

    assert function_decl.name == 'simple_function'
    assert function_decl.parameters.type == 'OBJECT'
    assert function_decl.parameters.properties['input_str'].type == 'OBJECT'
    assert (
        function_decl.parameters.properties['input_str'].additional_properties
        is None
    )

  def test_list_of_dict_input(self):
    """Test list[dict[str, str]] emits proper schema with additional_properties."""

    def simple_function(fruits: list[dict[str, str]]) -> str:
      return str(fruits)

    function_decl = _automatic_function_calling_util.build_function_declaration(
        func=simple_function
    )

    assert function_decl.name == 'simple_function'
    assert function_decl.parameters.type == 'OBJECT'
    assert function_decl.parameters.properties['fruits'].type == 'ARRAY'
    assert function_decl.parameters.properties['fruits'].items.type == 'OBJECT'
    assert (
        function_decl.parameters.properties[
            'fruits'
        ].items.additional_properties.type
        == 'STRING'
    )

  def test_basemodel_input(self):
    class CustomInput(BaseModel):
      input_str: str

    def simple_function(input: CustomInput) -> str:
      return {'result': input}

    function_decl = _automatic_function_calling_util.build_function_declaration(
        func=simple_function
    )

    assert function_decl.name == 'simple_function'
    assert function_decl.parameters.type == 'OBJECT'
    assert function_decl.parameters.properties['input'].type == 'OBJECT'
    assert (
        function_decl.parameters.properties['input']
        .properties['input_str']
        .type
        == 'STRING'
    )

  def test_basemodel_required_fields(self):
    class SearchRequest(BaseModel):
      query: str
      max_results: int
      filter: str = ''

    def search(request: SearchRequest) -> list:
      return []

    function_decl = _automatic_function_calling_util.build_function_declaration(
        func=search
    )

    inner = function_decl.parameters.properties['request']
    assert set(inner.required) == {'query', 'max_results'}
    assert 'filter' not in (inner.required or [])

  def test_basemodel_all_optional_fields_no_required(self):
    class Config(BaseModel):
      timeout: int = 30
      retries: int = 3

    def run(config: Config) -> str:
      return ''

    function_decl = _automatic_function_calling_util.build_function_declaration(
        func=run
    )

    inner = function_decl.parameters.properties['config']
    assert not inner.required

  def test_nested_basemodel_required_fields(self):
    class Inner(BaseModel):
      x: int
      y: int = 0

    class Outer(BaseModel):
      inner: Inner
      label: str = ''

    def process(data: Outer) -> str:
      return ''

    function_decl = _automatic_function_calling_util.build_function_declaration(
        func=process
    )

    outer = function_decl.parameters.properties['data']
    assert set(outer.required) == {'inner'}
    assert 'label' not in (outer.required or [])

    inner = outer.properties['inner']
    assert set(inner.required) == {'x'}
    assert 'y' not in (inner.required or [])

  def test_toolcontext_ignored(self):
    def simple_function(input_str: str, tool_context: ToolContext) -> str:
      return {'result': input_str}

    function_decl = _automatic_function_calling_util.build_function_declaration(
        func=simple_function, ignore_params=['tool_context']
    )

    assert function_decl.name == 'simple_function'
    assert function_decl.parameters.type == 'OBJECT'
    assert function_decl.parameters.properties['input_str'].type == 'STRING'
    assert 'tool_context' not in function_decl.parameters.properties

  def test_get_required_fields_no_properties_returns_empty_list(self):
    # A schema without properties must yield [] rather than None for any
    # caller that relies on the declared list[str] return.
    assert (
        _function_parameter_parse_util._get_required_fields(
            types.Schema(type='OBJECT')
        )
        == []
    )
    assert (
        _function_parameter_parse_util._get_required_fields(
            types.Schema(type='OBJECT', properties={})
        )
        == []
    )

  def test_build_declaration_includes_required_parameters(self):
    def simple_function(input_str: str) -> str:
      return {'result': input_str}

    function_decl = _automatic_function_calling_util.build_function_declaration(
        func=simple_function
    )

    assert function_decl.parameters.required == ['input_str']

  def test_all_parameters_ignored_results_in_no_parameters_schema(self):
    def only_context(tool_context: ToolContext) -> str:
      return ''

    function_decl = _automatic_function_calling_util.build_function_declaration(
        func=only_context, ignore_params=['tool_context']
    )

    # No parameters remain after ignoring tool_context, so the declaration
    # carries no parameter schema instead of one with required=None.
    assert function_decl.parameters is None

  def test_basemodel(self):
    class SimpleFunction(BaseModel):
      input_str: str
      custom_input: int

    function_decl = _automatic_function_calling_util.build_function_declaration(
        func=SimpleFunction, ignore_params=['custom_input']
    )

    assert function_decl.name == 'SimpleFunction'
    assert function_decl.parameters.type == 'OBJECT'
    assert function_decl.parameters.properties['input_str'].type == 'STRING'
    assert 'custom_input' not in function_decl.parameters.properties

  def test_nested_basemodel_input(self):
    class ChildInput(BaseModel):
      input_str: str

    class CustomInput(BaseModel):
      child: ChildInput

    def simple_function(input: CustomInput) -> str:
      return {'result': input}

    function_decl = _automatic_function_calling_util.build_function_declaration(
        func=simple_function
    )

    assert function_decl.name == 'simple_function'
    assert function_decl.parameters.type == 'OBJECT'
    assert function_decl.parameters.properties['input'].type == 'OBJECT'
    assert (
        function_decl.parameters.properties['input'].properties['child'].type
        == 'OBJECT'
    )
    assert (
        function_decl.parameters.properties['input']
        .properties['child']
        .properties['input_str']
        .type
        == 'STRING'
    )

  def test_basemodel_with_nested_basemodel(self):
    class ChildInput(BaseModel):
      input_str: str

    class CustomInput(BaseModel):
      child: ChildInput

    function_decl = _automatic_function_calling_util.build_function_declaration(
        func=CustomInput, ignore_params=['custom_input']
    )

    assert function_decl.name == 'CustomInput'
    assert function_decl.parameters.type == 'OBJECT'
    assert function_decl.parameters.properties['child'].type == 'OBJECT'
    assert (
        function_decl.parameters.properties['child']
        .properties['input_str']
        .type
        == 'STRING'
    )
    assert 'custom_input' not in function_decl.parameters.properties

  def test_list(self):
    def simple_function(
        input_str: list[str], input_dir: list[dict[str, str]]
    ) -> str:
      return {'result': input_str}

    function_decl = _automatic_function_calling_util.build_function_declaration(
        func=simple_function
    )

    assert function_decl.name == 'simple_function'
    assert function_decl.parameters.type == 'OBJECT'
    assert function_decl.parameters.properties['input_str'].type == 'ARRAY'
    assert (
        function_decl.parameters.properties['input_str'].items.type == 'STRING'
    )
    assert function_decl.parameters.properties['input_dir'].type == 'ARRAY'
    assert (
        function_decl.parameters.properties['input_dir'].items.type == 'OBJECT'
    )
    assert (
        function_decl.parameters.properties[
            'input_dir'
        ].items.additional_properties.type
        == 'STRING'
    )

  def test_enums(self):

    class InputEnum(Enum):
      AGENT = 'agent'
      TOOL = 'tool'

    def simple_function(input: InputEnum = InputEnum.AGENT):
      return input.value

    function_decl = _automatic_function_calling_util.build_function_declaration(
        func=simple_function
    )

    assert function_decl.name == 'simple_function'
    assert function_decl.parameters.type == 'OBJECT'
    assert function_decl.parameters.properties['input'].type == 'STRING'
    assert function_decl.parameters.properties['input'].default == 'agent'
    assert function_decl.parameters.properties['input'].enum == [
        'agent',
        'tool',
    ]

    def simple_function_with_wrong_enum(input: InputEnum = 'WRONG_ENUM'):
      return input.value

    with pytest.raises(ValueError):
      _automatic_function_calling_util.build_function_declaration(
          func=simple_function_with_wrong_enum
      )

  def test_basemodel_list(self):
    class ChildInput(BaseModel):
      input_str: str

    class CustomInput(BaseModel):
      child: ChildInput

    def simple_function(input_str: list[CustomInput]) -> str:
      return {'result': input_str}

    function_decl = _automatic_function_calling_util.build_function_declaration(
        func=simple_function
    )

    assert function_decl.name == 'simple_function'
    assert function_decl.parameters.type == 'OBJECT'
    assert function_decl.parameters.properties['input_str'].type == 'ARRAY'
    assert (
        function_decl.parameters.properties['input_str'].items.type == 'OBJECT'
    )
    assert (
        function_decl.parameters.properties['input_str']
        .items.properties['child']
        .type
        == 'OBJECT'
    )
    assert (
        function_decl.parameters.properties['input_str']
        .items.properties['child']
        .properties['input_str']
        .type
        == 'STRING'
    )

  # TODO: comment out this test for now as crewai requires python 3.10 as minimum
  # def test_crewai_tool():
  #   docs_tool = CrewaiTool(
  #       name='directory_read_tool',
  #       description='use this to find files for you.',
  #       tool=FileReadTool(),
  #   )
  #   function_decl = docs_tool.get_declaration()
  #   assert function_decl.name == 'directory_read_tool'
  #   assert function_decl.parameters.type == 'OBJECT'
  #   assert function_decl.parameters.properties['file_path'].type == 'STRING'

  def test_function_no_return_annotation_gemini_api(self):
    """Test function with no return annotation using GEMINI_API variant."""

    def function_no_return(param: str):
      """A function with no return annotation."""
      return None

    function_decl = _automatic_function_calling_util.build_function_declaration(
        func=function_no_return, variant=GoogleLLMVariant.GEMINI_API
    )

    assert function_decl.name == 'function_no_return'
    assert function_decl.parameters.type == 'OBJECT'
    assert function_decl.parameters.properties['param'].type == 'STRING'
    # GEMINI_API should not have response schema
    assert function_decl.response is None

  def test_function_no_return_annotation_vertex_ai(self):
    """Test function with no return annotation using VERTEX_AI variant."""

    def function_no_return(param: str):
      """A function with no return annotation."""
      return None

    function_decl = _automatic_function_calling_util.build_function_declaration(
        func=function_no_return, variant=GoogleLLMVariant.VERTEX_AI
    )

    assert function_decl.name == 'function_no_return'
    assert function_decl.parameters.type == 'OBJECT'
    assert function_decl.parameters.properties['param'].type == 'STRING'
    # VERTEX_AI should have response schema for functions with no return annotation
    # Changed: Now uses Any type instead of NULL for no return annotation
    assert function_decl.response is not None
    assert (
        function_decl.response.type is None
    )  # Any type maps to None in schema

  def test_function_explicit_none_return_vertex_ai(self):
    """Test function with explicit None return annotation using VERTEX_AI variant."""

    def function_none_return(param: str) -> None:
      """A function that explicitly returns None."""
      pass

    function_decl = _automatic_function_calling_util.build_function_declaration(
        func=function_none_return, variant=GoogleLLMVariant.VERTEX_AI
    )

    assert function_decl.name == 'function_none_return'
    assert function_decl.parameters.type == 'OBJECT'
    assert function_decl.parameters.properties['param'].type == 'STRING'
    # VERTEX_AI should have response schema for explicit None return
    assert function_decl.response is not None
    assert function_decl.response.type == types.Type.NULL

  def test_function_explicit_none_return_gemini_api(self):
    """Test function with explicit None return annotation using GEMINI_API variant."""

    def function_none_return(param: str) -> None:
      """A function that explicitly returns None."""
      pass

    function_decl = _automatic_function_calling_util.build_function_declaration(
        func=function_none_return, variant=GoogleLLMVariant.GEMINI_API
    )

    assert function_decl.name == 'function_none_return'
    assert function_decl.parameters.type == 'OBJECT'
    assert function_decl.parameters.properties['param'].type == 'STRING'
    # GEMINI_API should not have response schema
    assert function_decl.response is None

  def test_function_regular_return_type_vertex_ai(self):
    """Test function with regular return type using VERTEX_AI variant."""

    def function_string_return(param: str) -> str:
      """A function that returns a string."""
      return param

    function_decl = _automatic_function_calling_util.build_function_declaration(
        func=function_string_return, variant=GoogleLLMVariant.VERTEX_AI
    )

    assert function_decl.name == 'function_string_return'
    assert function_decl.parameters.type == 'OBJECT'
    assert function_decl.parameters.properties['param'].type == 'STRING'
    # VERTEX_AI should have response schema for string return
    assert function_decl.response is not None
    assert function_decl.response.type == types.Type.STRING

  def test_function_with_no_response_annotations(self):
    """Test a function that has no response annotations."""

    def transfer_to_agent(agent_name: str, tool_context: ToolContext):
      """Transfer the question to another agent."""
      tool_context.actions.transfer_to_agent = agent_name

    function_decl = _automatic_function_calling_util.build_function_declaration(
        func=transfer_to_agent,
        ignore_params=['tool_context'],
        variant=GoogleLLMVariant.VERTEX_AI,
    )

    assert function_decl.name == 'transfer_to_agent'
    assert function_decl.parameters.type == 'OBJECT'
    assert function_decl.parameters.properties['agent_name'].type == 'STRING'
    assert 'tool_context' not in function_decl.parameters.properties
    # This function has no return annotation, so it gets Any type instead of NULL
    # Changed: Now uses Any type instead of NULL for no return annotation
    assert function_decl.response is not None
    assert (
        function_decl.response.type is None
    )  # Any type maps to None in schema

  def test_transfer_to_agent_tool_with_enum_constraint(self):
    """Test TransferToAgentTool adds enum constraint to agent_name."""
    from google.adk.tools.transfer_to_agent_tool import TransferToAgentTool

    agent_names = ['agent_a', 'agent_b', 'agent_c']
    tool = TransferToAgentTool(agent_names=agent_names)

    function_decl = tool._get_declaration()

    assert function_decl.name == 'transfer_to_agent'
    assert function_decl.parameters.type == 'OBJECT'
    assert function_decl.parameters.properties['agent_name'].type == 'STRING'
    assert function_decl.parameters.properties['agent_name'].enum == agent_names
    assert 'tool_context' not in function_decl.parameters.properties

  def test_callable_object_is_declared_under_its_class_name(self):
    """A callable object has no __name__ and falls back to its class name."""

    class Calc:
      """Adds two numbers."""

      def __call__(self, a: int, b: int) -> int:
        return a + b

    function_decl = _automatic_function_calling_util.build_function_declaration(
        func=Calc()
    )

    assert function_decl.name == 'Calc'
    assert function_decl.parameters.properties['a'].type == 'INTEGER'

  def test_callable_object_reaches_the_response_schema_branch(self):
    """Only non-GEMINI_API variants build a response schema, which also names
    the callable."""

    class Calc:
      """Adds two numbers."""

      def __call__(self, a: int, b: int):
        return a + b

    function_decl = _automatic_function_calling_util.build_function_declaration(
        func=Calc(), variant=GoogleLLMVariant.VERTEX_AI
    )

    assert function_decl.name == 'Calc'
    assert function_decl.response is not None


class TestBuildFunctionDeclarationWithJsonSchema:
  """Tests for build_function_declaration when JSON_SCHEMA_FOR_FUNC_DECL is enabled."""

  @pytest.fixture(autouse=True)
  def enable_feature_flag(self):
    """Enable the JSON_SCHEMA_FOR_FUNC_DECL feature flag for all tests."""
    with temporary_feature_override(
        FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, True
    ):
      yield

  def test_basic_string_parameter(self):
    """Test basic string parameter with feature flag enabled."""

    def greet(name: str) -> str:
      """Greet someone."""
      return f'Hello, {name}!'

    decl = _automatic_function_calling_util.build_function_declaration(greet)

    assert decl.name == 'greet'
    assert decl.description == 'Greet someone.'
    assert decl.parameters_json_schema == {
        'properties': {'name': {'title': 'Name', 'type': 'string'}},
        'required': ['name'],
        'title': 'greetParams',
        'type': 'object',
    }

  def test_multiple_parameter_types(self):
    """Test multiple parameter types with feature flag enabled."""

    def create_user(name: str, age: int, active: bool) -> str:
      """Create a new user."""
      return f'Created {name}'

    decl = _automatic_function_calling_util.build_function_declaration(
        create_user
    )

    schema = decl.parameters_json_schema
    assert schema['properties'] == {
        'name': {'title': 'Name', 'type': 'string'},
        'age': {'title': 'Age', 'type': 'integer'},
        'active': {'title': 'Active', 'type': 'boolean'},
    }
    assert set(schema['required']) == {'name', 'age', 'active'}

  def test_list_parameter(self):
    """Test list parameter with feature flag enabled."""

    def sum_numbers(numbers: list[int]) -> int:
      """Sum a list of numbers."""
      return sum(numbers)

    decl = _automatic_function_calling_util.build_function_declaration(
        sum_numbers
    )

    schema = decl.parameters_json_schema
    assert schema['properties']['numbers'] == {
        'items': {'type': 'integer'},
        'title': 'Numbers',
        'type': 'array',
    }

  def test_dict_parameter(self):
    """Test dict parameter with feature flag enabled."""

    def process_data(data: dict[str, str]) -> str:
      """Process a dictionary."""
      return str(data)

    decl = _automatic_function_calling_util.build_function_declaration(
        process_data
    )

    schema = decl.parameters_json_schema
    assert schema['properties']['data'] == {
        'additionalProperties': {'type': 'string'},
        'title': 'Data',
        'type': 'object',
    }

  def test_dict_parameter_with_any(self):
    """Test dict[str, Any] parameter with feature flag enabled."""

    def process_data(data: dict[str, Any]) -> str:
      """Process a dictionary."""
      return str(data)

    decl = _automatic_function_calling_util.build_function_declaration(
        process_data
    )

    schema = decl.parameters_json_schema
    assert schema['properties']['data'] == {
        'additionalProperties': True,
        'title': 'Data',
        'type': 'object',
    }

  def test_untyped_dict_parameter(self):
    """Test untyped dict parameter with feature flag enabled."""

    def process_data(data: dict) -> str:
      """Process a dictionary."""
      return str(data)

    decl = _automatic_function_calling_util.build_function_declaration(
        process_data
    )

    schema = decl.parameters_json_schema
    assert schema['properties']['data'] == {
        'additionalProperties': True,
        'title': 'Data',
        'type': 'object',
    }

  def test_optional_parameter(self):
    """Test optional parameter with feature flag enabled."""

    def search(query: str, limit: int | None = None) -> str:
      """Search for something."""
      return query

    decl = _automatic_function_calling_util.build_function_declaration(search)

    schema = decl.parameters_json_schema
    assert schema['required'] == ['query']
    assert 'query' in schema['properties']
    assert 'limit' in schema['properties']

  def test_enum_parameter(self):
    """Test enum parameter with feature flag enabled."""

    class Color(Enum):
      RED = 'red'
      GREEN = 'green'
      BLUE = 'blue'

    def set_color(color: Color) -> str:
      """Set the color."""
      return color.value

    decl = _automatic_function_calling_util.build_function_declaration(
        set_color
    )

    schema = decl.parameters_json_schema
    assert schema['properties']['color'] == {
        '$ref': '#/$defs/Color',
    }
    assert schema['$defs']['Color'] == {
        'enum': ['red', 'green', 'blue'],
        'title': 'Color',
        'type': 'string',
    }

  def test_tool_context_ignored(self):
    """Test that tool_context is ignored."""

    def my_tool(query: str, tool_context: ToolContext) -> str:
      """A tool that uses context."""
      return query

    decl = _automatic_function_calling_util.build_function_declaration(
        my_tool, ignore_params=['tool_context']
    )

    schema = decl.parameters_json_schema
    assert set(schema['properties'].keys()) == {'query'}
    assert 'tool_context' not in schema['properties']

  def test_gemini_api_no_response_schema(self):
    """Test that GEMINI_API variant does not include response schema."""

    def get_data() -> dict[str, int]:
      """Get some data."""
      return {'count': 42}

    decl = _automatic_function_calling_util.build_function_declaration(
        get_data, variant=GoogleLLMVariant.GEMINI_API
    )

    # GEMINI_API should not have response_json_schema: the API rejects it.
    assert decl.response_json_schema is None

  @pytest.mark.parametrize(
      'variant, expect_response_schema',
      [
          (GoogleLLMVariant.GEMINI_API, False),
          (GoogleLLMVariant.VERTEX_AI, True),
      ],
  )
  def test_response_schema_by_variant(self, variant, expect_response_schema):
    """Test response schema generation based on the LLM variant."""

    def get_data() -> dict[str, int]:
      """Get some data."""
      return {'count': 42}

    decl = _automatic_function_calling_util.build_function_declaration(
        get_data, variant=variant
    )

    assert (decl.response_json_schema is not None) == expect_response_schema

  def test_pydantic_model_parameter(self):
    """Test Pydantic model parameter with feature flag enabled."""

    class Address(BaseModel):
      street: str
      city: str

    def save_address(address: Address) -> str:
      """Save an address."""
      return f'Saved address in {address.city}'

    decl = _automatic_function_calling_util.build_function_declaration(
        save_address
    )

    assert decl.parameters_json_schema is not None
    assert 'address' in decl.parameters_json_schema['properties']

  def test_no_parameters(self):
    """Test function with no parameters."""

    def get_time() -> str:
      """Get current time."""
      return '12:00'

    decl = _automatic_function_calling_util.build_function_declaration(get_time)

    assert decl.name == 'get_time'
    assert decl.parameters_json_schema is None

  def test_docstring_preserved(self):
    """Test that docstring is preserved as description."""

    def well_documented(x: int) -> int:
      """This is a well-documented function.

      It does something useful.
      """
      return x

    decl = _automatic_function_calling_util.build_function_declaration(
        well_documented
    )

    assert 'well-documented function' in decl.description
    assert 'something useful' in decl.description

  def test_default_values(self):
    """Test parameters with default values."""

    def greet(name: str = 'World') -> str:
      """Greet someone."""
      return f'Hello, {name}!'

    decl = _automatic_function_calling_util.build_function_declaration(greet)

    schema = decl.parameters_json_schema
    assert schema['properties']['name']['default'] == 'World'
    assert 'name' not in schema.get('required', [])


class TestBuildFunctionDeclarationFromSchemaDict:
  """Tests for the declaration builders that take a JSON schema dict.

  These are the entry points used by tool wrappers that already own a schema
  for their arguments instead of a Python signature to introspect.
  """

  def test_util_maps_schema_type_names_to_gemini_types(self):
    def tool_func(city: str) -> str:
      return city

    schema = {
        'properties': {
            'city': {'type': 'str'},
            'scores': {'type': 'tuple', 'items': {'type': 'float'}},
            'meta': {'type': 'Dict'},
            'anything': {'type': 'Any'},
        }
    }

    decl = _automatic_function_calling_util.build_function_declaration_util(
        False, 'lookup', 'Look a city up.', tool_func, schema
    )

    assert decl.name == 'lookup'
    assert decl.description == 'Look a city up.'
    assert decl.parameters.type == 'OBJECT'
    properties = decl.parameters.properties
    assert properties['city'].type == 'STRING'
    # Array element types are mapped too, not just the container.
    assert properties['scores'].type == 'ARRAY'
    assert properties['scores'].items.type == 'NUMBER'
    assert properties['meta'].type == 'OBJECT'
    assert properties['anything'].type == 'TYPE_UNSPECIFIED'

  def test_util_maps_unrecognized_type_name_to_type_unspecified(self):
    def tool_func(value: str) -> str:
      return value

    decl = _automatic_function_calling_util.build_function_declaration_util(
        False,
        'lookup',
        'Look something up.',
        tool_func,
        {'properties': {'value': {'type': 'complex128'}}},
    )

    assert decl.parameters.properties['value'].type == 'TYPE_UNSPECIFIED'

  def test_util_omits_parameters_when_schema_has_no_properties(self):
    def tool_func() -> str:
      return 'pong'

    decl = _automatic_function_calling_util.build_function_declaration_util(
        False, 'ping', 'Ping the service.', tool_func, {'properties': {}}
    )

    # A parameterless tool must not advertise an empty OBJECT schema.
    assert decl.parameters is None
    assert decl.name == 'ping'
    assert decl.description == 'Ping the service.'

  def test_util_sets_response_schema_from_return_annotation_for_vertexai(self):
    def tool_func(count: int) -> str:
      return str(count)

    decl = _automatic_function_calling_util.build_function_declaration_util(
        True,
        'stringify',
        'Stringify a count.',
        tool_func,
        {'properties': {'count': {'type': 'integer'}}},
    )

    assert decl.response.type == 'STRING'

  def test_util_omits_response_schema_when_not_vertexai(self):
    def tool_func(count: int) -> str:
      return str(count)

    decl = _automatic_function_calling_util.build_function_declaration_util(
        False,
        'stringify',
        'Stringify a count.',
        tool_func,
        {'properties': {'count': {'type': 'integer'}}},
    )

    # The Gemini API surface does not accept a response schema.
    assert decl.response is None

  def test_for_langchain_normalizes_properties_for_the_gemini_api(self):
    def tool_func(name: str) -> str:
      return name

    # Langchain hands over the `properties` block of its argument model's JSON
    # schema, which still carries pydantic's titles, defaults and unions.
    args = {
        'name': {'title': 'Name', 'type': 'string'},
        'nickname': {
            'anyOf': [{'type': 'string'}, {'type': 'null'}],
            'default': None,
            'title': 'Nickname',
        },
        'count': {'default': 3, 'title': 'Count', 'type': 'integer'},
    }

    decl = _automatic_function_calling_util.build_function_declaration_for_langchain(
        False, 'greet', 'Greet someone.', tool_func, args
    )

    properties = decl.parameters.properties
    assert set(properties) == {'name', 'nickname', 'count'}
    assert properties['name'].type == 'STRING'
    assert properties['count'].type == 'INTEGER'
    # An optional parameter collapses to its single non-null member type.
    assert properties['nickname'].type == 'STRING'
    # Only the parameter without a default is required of the model.
    assert decl.parameters.required == ['name']
    # None of the keywords the Gemini API surface rejects may survive.
    for property_schema in properties.values():
      assert property_schema.any_of is None
      assert property_schema.title is None
      assert property_schema.default is None
      assert property_schema.nullable is None

  def test_for_crewai_reads_properties_out_of_a_full_model_schema(self):
    class GreetArgs(BaseModel):
      name: str
      nickname: str | None = None
      count: int = 3

    def tool_func(name: str) -> str:
      return name

    # CrewAI hands over the whole `model_json_schema()`, not just its
    # `properties` block, so the schema's own top-level keys must not be
    # mistaken for parameters.
    decl = _automatic_function_calling_util.build_function_declaration_for_params_for_crewai(
        False,
        'greet',
        'Greet someone.',
        tool_func,
        GreetArgs.model_json_schema(),
    )

    properties = decl.parameters.properties
    assert set(properties) == {'name', 'nickname', 'count'}
    assert properties['name'].type == 'STRING'
    assert properties['nickname'].type == 'STRING'
    assert properties['count'].type == 'INTEGER'

  def test_for_crewai_marks_parameters_without_a_default_as_required(self):
    class GreetArgs(BaseModel):
      name: str
      count: int = 3

    def tool_func(name: str) -> str:
      return name

    decl = _automatic_function_calling_util.build_function_declaration_for_params_for_crewai(
        False,
        'greet',
        'Greet someone.',
        tool_func,
        GreetArgs.model_json_schema(),
    )

    assert decl.parameters.required == ['name']
