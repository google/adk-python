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

from collections.abc import Sequence
from typing import Any
from typing import AsyncGenerator
from typing import Dict
from typing import Generator

from google.adk.tools import _automatic_function_calling_util
from google.adk.utils.variant_utils import GoogleLLMVariant
from google.genai import types
import pydantic
import pytest


def test_from_function_with_options_no_return_annotation_gemini():
  """Test from_function_with_options with no return annotation for GEMINI_API."""

  def test_function(param: str):
    """A test function with no return annotation."""
    return None

  declaration = _automatic_function_calling_util.from_function_with_options(
      test_function, GoogleLLMVariant.GEMINI_API
  )

  assert declaration.name == 'test_function'
  assert declaration.parameters.type == 'OBJECT'
  assert declaration.parameters.properties['param'].type == 'STRING'
  # GEMINI_API should not have response schema
  assert declaration.response is None


def test_from_function_with_options_no_return_annotation_vertex():
  """Test from_function_with_options with no return annotation for VERTEX_AI."""

  def test_function(param: str):
    """A test function with no return annotation."""
    return None

  declaration = _automatic_function_calling_util.from_function_with_options(
      test_function, GoogleLLMVariant.VERTEX_AI
  )

  assert declaration.name == 'test_function'
  assert declaration.parameters.type == 'OBJECT'
  assert declaration.parameters.properties['param'].type == 'STRING'
  # VERTEX_AI should have response schema for functions with no return annotation
  # Changed: Now uses Any type instead of NULL for no return annotation
  assert declaration.response is not None
  assert declaration.response.type is None  # Any type maps to None in schema


def test_from_function_with_options_explicit_none_return_vertex():
  """Test from_function_with_options with explicit None return for VERTEX_AI."""

  def test_function(param: str) -> None:
    """A test function that explicitly returns None."""
    pass

  declaration = _automatic_function_calling_util.from_function_with_options(
      test_function, GoogleLLMVariant.VERTEX_AI
  )

  assert declaration.name == 'test_function'
  assert declaration.parameters.type == 'OBJECT'
  assert declaration.parameters.properties['param'].type == 'STRING'
  # VERTEX_AI should have response schema for explicit None return
  assert declaration.response is not None
  assert declaration.response.type == types.Type.NULL


def test_from_function_with_options_explicit_none_return_gemini():
  """Test from_function_with_options with explicit None return for GEMINI_API."""

  def test_function(param: str) -> None:
    """A test function that explicitly returns None."""
    pass

  declaration = _automatic_function_calling_util.from_function_with_options(
      test_function, GoogleLLMVariant.GEMINI_API
  )

  assert declaration.name == 'test_function'
  assert declaration.parameters.type == 'OBJECT'
  assert declaration.parameters.properties['param'].type == 'STRING'
  # GEMINI_API should not have response schema
  assert declaration.response is None


def test_from_function_with_options_string_return_vertex():
  """Test from_function_with_options with string return for VERTEX_AI."""

  def test_function(param: str) -> str:
    """A test function that returns a string."""
    return param

  declaration = _automatic_function_calling_util.from_function_with_options(
      test_function, GoogleLLMVariant.VERTEX_AI
  )

  assert declaration.name == 'test_function'
  assert declaration.parameters.type == 'OBJECT'
  assert declaration.parameters.properties['param'].type == 'STRING'
  # VERTEX_AI should have response schema for string return
  assert declaration.response is not None
  assert declaration.response.type == types.Type.STRING


def test_from_function_with_options_dict_return_vertex():
  """Test from_function_with_options with dict return for VERTEX_AI."""

  def test_function(param: str) -> Dict[str, str]:
    """A test function that returns a dict."""
    return {'result': param}

  declaration = _automatic_function_calling_util.from_function_with_options(
      test_function, GoogleLLMVariant.VERTEX_AI
  )

  assert declaration.name == 'test_function'
  assert declaration.parameters.type == 'OBJECT'
  assert declaration.parameters.properties['param'].type == 'STRING'
  # VERTEX_AI should have response schema for dict return
  assert declaration.response is not None
  assert declaration.response.type == types.Type.OBJECT


def test_from_function_with_options_int_return_vertex():
  """Test from_function_with_options with int return for VERTEX_AI."""

  def test_function(param: str) -> int:
    """A test function that returns an int."""
    return 42

  declaration = _automatic_function_calling_util.from_function_with_options(
      test_function, GoogleLLMVariant.VERTEX_AI
  )

  assert declaration.name == 'test_function'
  assert declaration.parameters.type == 'OBJECT'
  assert declaration.parameters.properties['param'].type == 'STRING'
  # VERTEX_AI should have response schema for int return
  assert declaration.response is not None
  assert declaration.response.type == types.Type.INTEGER


def test_from_function_with_options_any_annotation_vertex():
  """Test from_function_with_options with Any type annotation for VERTEX_AI."""

  def test_function(param: Any) -> Any:
    """A test function that uses Any type annotations."""
    return param

  declaration = _automatic_function_calling_util.from_function_with_options(
      test_function, GoogleLLMVariant.VERTEX_AI
  )

  assert declaration.name == 'test_function'
  assert declaration.parameters.type == 'OBJECT'
  # Any type should map to None in schema (TYPE_UNSPECIFIED behavior)
  assert declaration.parameters.properties['param'].type is None
  # VERTEX_AI should have response schema for Any return
  assert declaration.response is not None
  assert declaration.response.type is None  # Any type maps to None in schema


def test_from_function_with_options_no_params():
  """Test from_function_with_options with no parameters."""

  def test_function() -> None:
    """A test function with no parameters that returns None."""
    pass

  declaration = _automatic_function_calling_util.from_function_with_options(
      test_function, GoogleLLMVariant.VERTEX_AI
  )

  assert declaration.name == 'test_function'
  # No parameters should result in no parameters field or empty parameters
  assert (
      declaration.parameters is None
      or len(declaration.parameters.properties) == 0
  )
  # VERTEX_AI should have response schema for None return
  assert declaration.response is not None
  assert declaration.response.type == types.Type.NULL


def test_from_function_with_collections_type_parameter():
  """Test from_function_with_options with collections type parameter."""

  def test_function(
      artifact_key: str,
      input_edit_ids: Sequence[str],
  ) -> str:
    """Saves a sequence of edit IDs."""
    return f'Saved {len(input_edit_ids)} edit IDs for artifact {artifact_key}'

  declaration = _automatic_function_calling_util.from_function_with_options(
      test_function, GoogleLLMVariant.VERTEX_AI
  )

  assert declaration.name == 'test_function'
  assert declaration.parameters.type == types.Type.OBJECT
  assert (
      declaration.parameters.properties['artifact_key'].type
      == types.Type.STRING
  )
  assert (
      declaration.parameters.properties['input_edit_ids'].type
      == types.Type.ARRAY
  )
  assert (
      declaration.parameters.properties['input_edit_ids'].items.type
      == types.Type.STRING
  )
  assert declaration.response.type == types.Type.STRING


def test_from_function_with_tuple_type_parameter():
  """Test from_function_with_options with fixed-size homogeneous tuple."""

  def test_function(
      coordinate: tuple[float, float],
  ) -> str:
    """Formats a coordinate pair."""
    return f'{coordinate[0]}, {coordinate[1]}'

  declaration = _automatic_function_calling_util.from_function_with_options(
      test_function, GoogleLLMVariant.VERTEX_AI
  )

  assert declaration.name == 'test_function'
  assert declaration.parameters.type == types.Type.OBJECT
  coordinate_schema = declaration.parameters.properties['coordinate']
  assert coordinate_schema.type == types.Type.ARRAY
  assert coordinate_schema.items.type == types.Type.NUMBER
  # Fixed-size tuples pin the array length so the model emits exactly the
  # expected number of items.
  assert coordinate_schema.min_items == 2
  assert coordinate_schema.max_items == 2
  assert declaration.response.type == types.Type.STRING


def test_from_function_with_variadic_tuple_type_parameter():
  """Test from_function_with_options with variable-length homogeneous tuple."""

  def test_function(
      tags: tuple[str, ...],
  ) -> str:
    """Joins tags."""
    return ', '.join(tags)

  declaration = _automatic_function_calling_util.from_function_with_options(
      test_function, GoogleLLMVariant.VERTEX_AI
  )

  tags_schema = declaration.parameters.properties['tags']
  assert tags_schema.type == types.Type.ARRAY
  assert tags_schema.items.type == types.Type.STRING
  # Variadic tuples are unbounded, so no size constraints are set.
  assert tags_schema.min_items is None
  assert tags_schema.max_items is None


def test_from_function_with_collections_return_type():
  """Test from_function_with_options with collections return type."""

  def test_function(
      names: list[str],
  ) -> Sequence[str]:
    """Returns a sequence of names."""
    return names

  declaration = _automatic_function_calling_util.from_function_with_options(
      test_function, GoogleLLMVariant.VERTEX_AI
  )

  assert declaration.name == 'test_function'
  assert declaration.response.type == types.Type.ARRAY
  assert declaration.response.items.type == types.Type.STRING


def test_from_function_with_async_generator_return_vertex():
  """Test from_function_with_options with AsyncGenerator return for VERTEX_AI."""

  async def test_function(param: str) -> AsyncGenerator[str, None]:
    """A streaming function that yields strings."""
    yield param

  declaration = _automatic_function_calling_util.from_function_with_options(
      test_function, GoogleLLMVariant.VERTEX_AI
  )

  assert declaration.name == 'test_function'
  assert declaration.parameters.type == 'OBJECT'
  assert declaration.parameters.properties['param'].type == 'STRING'
  # VERTEX_AI should extract yield type (str) from AsyncGenerator[str, None]
  assert declaration.response is not None
  assert declaration.response.type == types.Type.STRING


def test_from_function_with_async_generator_return_gemini():
  """Test from_function_with_options with AsyncGenerator return for GEMINI_API."""

  async def test_function(param: str) -> AsyncGenerator[str, None]:
    """A streaming function that yields strings."""
    yield param

  declaration = _automatic_function_calling_util.from_function_with_options(
      test_function, GoogleLLMVariant.GEMINI_API
  )

  assert declaration.name == 'test_function'
  assert declaration.parameters.type == 'OBJECT'
  assert declaration.parameters.properties['param'].type == 'STRING'
  # GEMINI_API should not have response schema
  assert declaration.response is None


def test_from_function_with_generator_return_vertex():
  """Test from_function_with_options with Generator return for VERTEX_AI."""

  def test_function(param: str) -> Generator[int, None, None]:
    """A streaming function that yields integers."""
    yield 42

  declaration = _automatic_function_calling_util.from_function_with_options(
      test_function, GoogleLLMVariant.VERTEX_AI
  )

  assert declaration.name == 'test_function'
  assert declaration.parameters.type == 'OBJECT'
  assert declaration.parameters.properties['param'].type == 'STRING'
  # VERTEX_AI should extract yield type (int) from Generator[int, None, None]
  assert declaration.response is not None
  assert declaration.response.type == types.Type.INTEGER


def test_from_function_with_async_generator_complex_yield_type_vertex():
  """Test from_function_with_options with AsyncGenerator yielding dict."""

  async def test_function(param: str) -> AsyncGenerator[Dict[str, str], None]:
    """A streaming function that yields dicts."""
    yield {'result': param}

  declaration = _automatic_function_calling_util.from_function_with_options(
      test_function, GoogleLLMVariant.VERTEX_AI
  )

  assert declaration.name == 'test_function'
  assert declaration.parameters.type == 'OBJECT'
  assert declaration.parameters.properties['param'].type == 'STRING'
  # VERTEX_AI should extract yield type (Dict[str, str]) from AsyncGenerator
  assert declaration.response is not None
  assert declaration.response.type == types.Type.OBJECT


def test_required_fields_set_with_optional_tuple_parameter():
  """Test that required fields are populated with optional tuple parameters."""

  def complex_tool(
      query: str,
      mode: str = 'default',
      tags: tuple[str, ...] | None = None,
  ) -> str:
    """A tool where one param has a complex union type."""
    return query

  declaration = _automatic_function_calling_util.from_function_with_options(
      complex_tool, GoogleLLMVariant.GEMINI_API
  )

  assert declaration.name == 'complex_tool'
  assert declaration.parameters == types.Schema(
      type=types.Type.OBJECT,
      required=['query'],
      properties={
          'query': types.Schema(type=types.Type.STRING),
          'mode': types.Schema(type=types.Type.STRING, default='default'),
          'tags': types.Schema(
              items=types.Schema(type=types.Type.STRING),
              nullable=True,
              type=types.Type.ARRAY,
          ),
      },
  )


def test_required_fields_set_in_json_schema_fallback():
  """Required fields are populated when the json_schema fallback path is used.

  A parameter whose type `_parse_schema_from_parameter` cannot handle (here
  `Sequence[str]`) forces from_function_with_options onto the pydantic
  json_schema fallback branch. This verifies that branch still derives required
  fields correctly: parameters without defaults are required, parameters with
  defaults are not.
  """

  def complex_tool(
      query: str,
      items: Sequence[str],
      mode: str = 'default',
  ) -> str:
    return query

  declaration = _automatic_function_calling_util.from_function_with_options(
      complex_tool, GoogleLLMVariant.VERTEX_AI
  )

  assert declaration.name == 'complex_tool'
  assert declaration.parameters.type == types.Type.OBJECT
  # query and items have no defaults -> required; mode has a default -> not.
  assert set(declaration.parameters.required) == {'query', 'items'}
  assert declaration.parameters.properties['items'].type == types.Type.ARRAY
  assert declaration.parameters.properties['mode'].default == 'default'


def test_schema_sanitization_for_complex_union_type():
  """Test schema is sanitized for complex union type."""

  def complex_tool(
      query: str,
      mode: str = 'default',
      tags: dict[str, str] | None = None,
  ) -> str:
    return query

  declaration = _automatic_function_calling_util.from_function_with_options(
      complex_tool, GoogleLLMVariant.GEMINI_API
  )

  assert declaration.parameters.properties['tags'] == types.Schema(
      type=types.Type.OBJECT,
      nullable=True,
  )


def test_format_preservation_for_vertex_fallback():
  """Test that format is preserved for VERTEX_AI variant in fallback path."""

  class ComplexModel(pydantic.BaseModel):
    # Field with format that would be stripped by Gemini sanitization
    email: str = pydantic.Field(json_schema_extra={'format': 'email'})
    # Complex field to trigger fallback (Sequence is not handled by
    # _parse_schema_from_parameter)
    complex_field: Sequence[str]

  def my_tool(param: ComplexModel) -> str:
    return f'ok {param}'

  # Run with VERTEX_AI, should preserve format
  declaration_vertex = (
      _automatic_function_calling_util.from_function_with_options(
          my_tool, GoogleLLMVariant.VERTEX_AI
      )
  )

  # Check that format is preserved
  param_schema_vertex = declaration_vertex.parameters.properties['param']
  assert param_schema_vertex.properties['email'].format == 'email'

  # Run with GEMINI_API, should strip format (current behavior)
  declaration_gemini = (
      _automatic_function_calling_util.from_function_with_options(
          my_tool, GoogleLLMVariant.GEMINI_API
      )
  )
  param_schema_gemini = declaration_gemini.parameters.properties['param']
  assert param_schema_gemini.properties['email'].format is None


def test_tuple_types_work_in_json_schema_fallback() -> None:
  """Test that tuple schemas work in json schema fallback."""

  def generate_image(
      prompt: str,
      input_bytes: list[tuple[bytes, str]] | None = None,
  ) -> dict[str, str]:
    """Generate an image from a prompt."""
    del input_bytes
    return {'status': prompt}

  declaration = _automatic_function_calling_util.from_function_with_options(
      generate_image, GoogleLLMVariant.GEMINI_API
  )

  assert declaration.parameters is not None
  assert declaration.parameters.required == ['prompt']
  input_bytes_schema = declaration.parameters.properties['input_bytes']
  assert input_bytes_schema.nullable
  assert input_bytes_schema.any_of is not None

  array_schema = next(
      schema
      for schema in input_bytes_schema.any_of
      if schema.type == types.Type.ARRAY
  )
  assert array_schema.items is not None
  assert array_schema.items.type == types.Type.ARRAY
  assert array_schema.items.max_items == 2
  assert array_schema.items.min_items == 2
  assert array_schema.items.items is not None
  assert array_schema.items.items.any_of is not None
  assert len(array_schema.items.items.any_of) == 2
  assert array_schema.items.items.any_of[0].type == types.Type.STRING
  assert array_schema.items.items.any_of[0].format is None
  assert array_schema.items.items.any_of[1].type == types.Type.STRING
