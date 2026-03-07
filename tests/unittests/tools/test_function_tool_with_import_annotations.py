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

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import Optional

from google.adk.tools import _automatic_function_calling_util
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.adk.utils.context_utils import find_context_parameter
from google.adk.utils.variant_utils import GoogleLLMVariant
from google.genai import types


def test_string_annotation_none_return_vertex():
  """Test function with string annotation 'None' return for VERTEX_AI."""

  def test_function(_param: str) -> None:
    """A test function that returns None with string annotation."""
    pass

  declaration = _automatic_function_calling_util.from_function_with_options(
      test_function, GoogleLLMVariant.VERTEX_AI
  )

  assert declaration.name == 'test_function'
  assert declaration.parameters.type == 'OBJECT'
  assert declaration.parameters.properties['_param'].type == 'STRING'
  # VERTEX_AI should have response schema for None return (stored as string)
  assert declaration.response is not None
  assert declaration.response.type == types.Type.NULL


def test_string_annotation_none_return_gemini():
  """Test function with string annotation 'None' return for GEMINI_API."""

  def test_function(_param: str) -> None:
    """A test function that returns None with string annotation."""
    pass

  declaration = _automatic_function_calling_util.from_function_with_options(
      test_function, GoogleLLMVariant.GEMINI_API
  )

  assert declaration.name == 'test_function'
  assert declaration.parameters.type == 'OBJECT'
  assert declaration.parameters.properties['_param'].type == 'STRING'
  # GEMINI_API should not have response schema
  assert declaration.response is None


def test_string_annotation_str_return_vertex():
  """Test function with string annotation 'str' return for VERTEX_AI."""

  def test_function(_param: str) -> str:
    """A test function that returns a string with string annotation."""
    return _param

  declaration = _automatic_function_calling_util.from_function_with_options(
      test_function, GoogleLLMVariant.VERTEX_AI
  )

  assert declaration.name == 'test_function'
  assert declaration.parameters.type == 'OBJECT'
  assert declaration.parameters.properties['_param'].type == 'STRING'
  # VERTEX_AI should have response schema for string return (stored as string)
  assert declaration.response is not None
  assert declaration.response.type == types.Type.STRING


def test_string_annotation_int_return_vertex():
  """Test function with string annotation 'int' return for VERTEX_AI."""

  def test_function(_param: str) -> int:
    """A test function that returns an int with string annotation."""
    return 42

  declaration = _automatic_function_calling_util.from_function_with_options(
      test_function, GoogleLLMVariant.VERTEX_AI
  )

  assert declaration.name == 'test_function'
  assert declaration.parameters.type == 'OBJECT'
  assert declaration.parameters.properties['_param'].type == 'STRING'
  # VERTEX_AI should have response schema for int return (stored as string)
  assert declaration.response is not None
  assert declaration.response.type == types.Type.INTEGER


def test_string_annotation_dict_return_vertex():
  """Test function with string annotation Dict return for VERTEX_AI."""

  def test_function(_param: str) -> Dict[str, str]:
    """A test function that returns a dict with string annotation."""
    return {'result': _param}

  declaration = _automatic_function_calling_util.from_function_with_options(
      test_function, GoogleLLMVariant.VERTEX_AI
  )

  assert declaration.name == 'test_function'
  assert declaration.parameters.type == 'OBJECT'
  assert declaration.parameters.properties['_param'].type == 'STRING'
  # VERTEX_AI should have response schema for dict return (stored as string)
  assert declaration.response is not None
  assert declaration.response.type == types.Type.OBJECT


def test_string_annotation_any_return_vertex():
  """Test function with string annotation 'Any' return for VERTEX_AI."""

  def test_function(_param: Any) -> Any:
    """A test function that uses Any type with string annotations."""
    return _param

  declaration = _automatic_function_calling_util.from_function_with_options(
      test_function, GoogleLLMVariant.VERTEX_AI
  )

  assert declaration.name == 'test_function'
  assert declaration.parameters.type == 'OBJECT'
  # Any type should map to None in schema (TYPE_UNSPECIFIED behavior)
  assert declaration.parameters.properties['_param'].type is None
  # VERTEX_AI should have response schema for Any return (stored as string)
  assert declaration.response is not None
  assert declaration.response.type is None  # Any type maps to None in schema


def test_string_annotation_mixed_parameters_vertex():
  """Test function with mixed string annotations for parameters."""

  def test_function(str_param: str, int_param: int, any_param: Any) -> str:
    """A test function with mixed parameter types as string annotations."""
    return f'{str_param}-{int_param}-{any_param}'

  declaration = _automatic_function_calling_util.from_function_with_options(
      test_function, GoogleLLMVariant.VERTEX_AI
  )

  assert declaration.name == 'test_function'
  assert declaration.parameters.type == 'OBJECT'
  assert declaration.parameters.properties['str_param'].type == 'STRING'
  assert declaration.parameters.properties['int_param'].type == 'INTEGER'
  assert declaration.parameters.properties['any_param'].type is None  # Any type
  # VERTEX_AI should have response schema for string return (stored as string)
  assert declaration.response is not None
  assert declaration.response.type == types.Type.STRING


def test_string_annotation_no_params_vertex():
  """Test function with no parameters but string annotation return."""

  def test_function() -> str:
    """A test function with no parameters that returns string (string annotation)."""
    return 'hello'

  declaration = _automatic_function_calling_util.from_function_with_options(
      test_function, GoogleLLMVariant.VERTEX_AI
  )

  assert declaration.name == 'test_function'
  # No parameters should result in no parameters field or empty parameters
  assert (
      declaration.parameters is None
      or len(declaration.parameters.properties) == 0
  )
  # VERTEX_AI should have response schema for string return (stored as string)
  assert declaration.response is not None
  assert declaration.response.type == types.Type.STRING


def test_find_context_parameter_detects_custom_name_with_future_annotations():
  """Test custom param name detection with deferred annotations."""

  # pylint: disable-next=unused-argument
  def my_tool(ctx: ToolContext) -> str:
    """A tool with a custom-named context parameter."""
    return ''

  assert find_context_parameter(my_tool) == 'ctx'


def test_find_context_parameter_optional_ctx_with_future_annotations():
  """Test Optional[ToolContext] detection with deferred annotations."""

  # pylint: disable-next=unused-argument
  def my_tool(ctx: Optional[ToolContext] = None) -> str:
    """A tool with an optional context parameter."""
    return ''

  assert find_context_parameter(my_tool) == 'ctx'


def test_function_tool_excludes_custom_context_param_with_future_annotations():
  """Test schema omits context param with deferred annotations."""

  # pylint: disable-next=unused-argument
  def my_tool(query: str, ctx: ToolContext) -> str:
    """A tool with a custom-named context parameter."""
    return query

  tool = FunctionTool(my_tool)
  declaration = tool._get_declaration()  # pylint: disable=protected-access

  assert declaration.name == 'my_tool'
  assert set(declaration.parameters.properties.keys()) == {'query'}
  assert 'ctx' not in declaration.parameters.properties
