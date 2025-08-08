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

from google.adk.tools import _automatic_function_calling_util
from google.adk.utils.variant_utils import GoogleLLMVariant
from google.genai import types


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
