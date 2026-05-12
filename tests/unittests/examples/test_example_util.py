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

"""Tests for example_util."""

from google.adk.examples import base_example_provider
from google.adk.examples import example
from google.adk.examples import example_util
from google.genai import types
import pytest

BASIC_INPUT = types.Content(role="user", parts=[types.Part(text="test_input")])

BASIC_OUTPUT = [
    types.Content(role="model", parts=[types.Part(text="test_output")])
]

BASIC_EXAMPLE = example.Example(input=BASIC_INPUT, output=BASIC_OUTPUT)


class MockExampleProvider(base_example_provider.BaseExampleProvider):
  """Mocks an ExampleProvider object.

  This class provides mock implementation of the get_examples() function,
  allowing the user to test functions that rely on an ExampleProvider
  without creating a real ExampleProvider class and check that the correct
  inputs are being passed to it.
  """

  def __init__(
      self, test_examples: list[example.Example], test_query: str
  ) -> None:
    """Initializes a MockExampleProvider.

    Args:
        test_examples: The list of examples to be returned on a successful query.
        test_query: The query necessary to return a correct output.
    """
    self.test_examples = test_examples
    self.test_query = test_query

  def get_examples(self, query: str) -> list[example.Example]:
    """Mocks querying the ExampleProvider for examples.
    Verifies the query is correct, and returns an empty list if not.

    Args:
        query: The query to check examples for.
    """
    if query == self.test_query:
      return self.test_examples
    else:
      return []


@pytest.mark.parametrize(
    "model",
    ["gemini-2.5-flash", "llama3_vertex_agent", None],
)
def test_text_only_example_conversion(model):
  """Tests converting a text-only Example object to a string for use in a system instruction."""
  expected_output = (
      f"{example_util._EXAMPLES_INTRO}"
      f"{example_util._EXAMPLE_START.format(1)}"
      f"{example_util._USER_PREFIX}test_input\n"
      f"{example_util._MODEL_PREFIX}test_output\n"
      f"{example_util._EXAMPLE_END}"
      f"{example_util._EXAMPLES_END}"
  )

  assert (
      example_util.convert_examples_to_text(
          examples=[BASIC_EXAMPLE], model=model
      )
      == expected_output
  )


@pytest.mark.parametrize(
    "model",
    ["gemini-2.5-flash", "llama3_vertex_agent", None],
)
def test_multi_part_text_example_conversion(model):
  """Tests converting an Example object with multiple text Parts to a string for use in a system instruction."""
  output_content = [
      types.Content(
          role="model",
          parts=[
              types.Part(text="test_output_1"),
              types.Part(text="test_output_2"),
              types.Part(text="test_output_3"),
          ],
      )
  ]
  test_example = example.Example(input=BASIC_INPUT, output=output_content)

  expected_output = (
      f"{example_util._EXAMPLES_INTRO}"
      f"{example_util._EXAMPLE_START.format(1)}"
      f"{example_util._USER_PREFIX}test_input\n"
      f"{example_util._MODEL_PREFIX}test_output_1\ntest_output_2\ntest_output_3\n"
      f"{example_util._EXAMPLE_END}"
      f"{example_util._EXAMPLES_END}"
  )

  assert (
      example_util.convert_examples_to_text(
          examples=[test_example], model=model
      )
      == expected_output
  )


@pytest.mark.parametrize(
    "model",
    ["gemini-2.5-flash", "llama3_vertex_agent", None],
)
def test_example_conversion_prefix_insertion(model):
  """Tests if user and model prefixes are properly alternated when converting an Example object to text for use in a system instruction."""
  output_content = [
      types.Content(role="model", parts=[types.Part(text="test_output_1")]),
      types.Content(role="user", parts=[types.Part(text="test_output_2")]),
      types.Content(role="model", parts=[types.Part(text="test_output_3")]),
  ]
  test_example = example.Example(input=BASIC_INPUT, output=output_content)

  expected_output = (
      f"{example_util._EXAMPLES_INTRO}"
      f"{example_util._EXAMPLE_START.format(1)}"
      f"{example_util._USER_PREFIX}test_input\n"
      f"{example_util._MODEL_PREFIX}test_output_1\n"
      f"{example_util._USER_PREFIX}test_output_2\n"
      f"{example_util._MODEL_PREFIX}test_output_3\n"
      f"{example_util._EXAMPLE_END}"
      f"{example_util._EXAMPLES_END}"
  )

  assert (
      example_util.convert_examples_to_text(
          examples=[test_example], model=model
      )
      == expected_output
  )


@pytest.mark.parametrize(
    "model",
    ["gemini-2.5-flash", "llama3_vertex_agent", None],
)
def test_example_conversion_output_clumping(model):
  """Tests whether user and model inputs are properly clumped when converting an Example object to text for use in a system instruction."""
  output_content = [
      types.Content(role="model", parts=[types.Part(text="test_output_1")]),
      types.Content(role="model", parts=[types.Part(text="test_output_2")]),
      types.Content(role="user", parts=[types.Part(text="test_output_3")]),
      types.Content(role="user", parts=[types.Part(text="test_output_4")]),
  ]
  test_example = example.Example(input=BASIC_INPUT, output=output_content)

  expected_output = (
      f"{example_util._EXAMPLES_INTRO}"
      f"{example_util._EXAMPLE_START.format(1)}"
      f"{example_util._USER_PREFIX}test_input\n"
      f"{example_util._MODEL_PREFIX}test_output_1\ntest_output_2\n"
      f"{example_util._USER_PREFIX}test_output_3\ntest_output_4\n"
      f"{example_util._EXAMPLE_END}"
      f"{example_util._EXAMPLES_END}"
  )

  assert (
      example_util.convert_examples_to_text(
          examples=[test_example], model=model
      )
      == expected_output
  )


@pytest.mark.parametrize(
    "model",
    ["gemini-2.5-flash", "llama3_vertex_agent", None],
)
def test_empty_examples_list_conversion(model):
  """Tests Example conversion to text if the examples list is empty."""
  expected_output = (
      f"{example_util._EXAMPLES_INTRO}{example_util._EXAMPLES_END}"
  )

  assert (
      example_util.convert_examples_to_text(examples=[], model=model)
      == expected_output
  )


@pytest.mark.parametrize(
    "model",
    ["gemini-2.5-flash", "llama3_vertex_agent", None],
)
def test_example_conversion_with_function_call(model):
  """Tests converting an Example object containing a function call to a string for use in a system instruction."""
  test_function_call = types.FunctionCall(
      name="test_function",
      args={"test_string_argument": "test_value", "test_int_argument": 1},
  )
  output_content = [
      types.Content(
          role="model", parts=[types.Part(function_call=test_function_call)]
      )
  ]
  test_example = example.Example(input=BASIC_INPUT, output=output_content)

  gemini2 = model is None or "gemini-2" in model
  prefix = (
      example_util._FUNCTION_PREFIX
      if gemini2
      else example_util._FUNCTION_CALL_PREFIX
  )

  expected_output = (
      f"{example_util._EXAMPLES_INTRO}"
      f"{example_util._EXAMPLE_START.format(1)}"
      f"{example_util._USER_PREFIX}test_input\n"
      f"{example_util._MODEL_PREFIX}{prefix}"
      "test_function(test_string_argument='test_value', test_int_argument=1)"
      f"{example_util._FUNCTION_CALL_SUFFIX}"
      f"{example_util._EXAMPLE_END}"
      f"{example_util._EXAMPLES_END}"
  )

  assert (
      example_util.convert_examples_to_text(
          examples=[test_example], model=model
      )
      == expected_output
  )


@pytest.mark.parametrize(
    "model",
    ["gemini-2.5-flash", "llama3_vertex_agent", None],
)
def test_example_conversion_with_function_response(model):
  """Tests converting an Example object containing a function response to a string for use in a system instruction."""
  test_function_response = types.FunctionResponse(
      name="test_function",
      response={"test_string_argument": "test_value", "test_int_argument": 1},
  )
  output_content = [
      types.Content(
          role="model",
          parts=[types.Part(function_response=test_function_response)],
      )
  ]
  test_example = example.Example(input=BASIC_INPUT, output=output_content)

  gemini2 = model is None or "gemini-2" in model
  prefix = (
      example_util._FUNCTION_PREFIX
      if gemini2
      else example_util._FUNCTION_RESPONSE_PREFIX
  )

  expected_output = (
      f"{example_util._EXAMPLES_INTRO}"
      f"{example_util._EXAMPLE_START.format(1)}"
      f"{example_util._USER_PREFIX}test_input\n"
      f"{example_util._MODEL_PREFIX}{prefix}"
      f"{test_function_response.__dict__}"
      f"{example_util._FUNCTION_RESPONSE_SUFFIX}"
      f"{example_util._EXAMPLE_END}"
      f"{example_util._EXAMPLES_END}"
  )

  assert (
      example_util.convert_examples_to_text(
          examples=[test_example], model=model
      )
      == expected_output
  )


@pytest.mark.parametrize(
    "model",
    ["gemini-2.5-flash", "llama3_vertex_agent", None],
)
def test_example_conversion_with_function_call_response(model):
  """Tests converting an Example object containing a function call and response to a string for use in a system instruction."""
  test_function_call = types.FunctionCall(
      name="test_function",
      args={"test_string_argument": "test_value", "test_int_argument": 1},
  )
  test_function_response = types.FunctionResponse(
      name="test_function",
      response={"test_string_argument": "test_value", "test_int_argument": 1},
  )
  output_content = [
      types.Content(
          role="model",
          parts=[
              types.Part(function_call=test_function_call),
              types.Part(function_response=test_function_response),
          ],
      )
  ]
  test_example = example.Example(input=BASIC_INPUT, output=output_content)

  gemini2 = model is None or "gemini-2" in model
  response_prefix = (
      example_util._FUNCTION_PREFIX
      if gemini2
      else example_util._FUNCTION_RESPONSE_PREFIX
  )
  call_prefix = (
      example_util._FUNCTION_PREFIX
      if gemini2
      else example_util._FUNCTION_CALL_PREFIX
  )

  expected_output = (
      f"{example_util._EXAMPLES_INTRO}"
      f"{example_util._EXAMPLE_START.format(1)}"
      f"{example_util._USER_PREFIX}test_input\n"
      f"{example_util._MODEL_PREFIX}{call_prefix}"
      "test_function(test_string_argument='test_value', test_int_argument=1)"
      f"{example_util._FUNCTION_CALL_SUFFIX}"
      f"{response_prefix}"
      f"{test_function_response.__dict__}"
      f"{example_util._FUNCTION_RESPONSE_SUFFIX}"
      f"{example_util._EXAMPLE_END}"
      f"{example_util._EXAMPLES_END}"
  )

  assert (
      example_util.convert_examples_to_text(
          examples=[test_example], model=model
      )
      == expected_output
  )


@pytest.mark.parametrize(
    "model",
    ["gemini-2.5-flash", "llama3_vertex_agent", None],
)
def test_example_conversion_with_text_and_function_call_response(model):
  """Tests converting an Example object containing text, a function call, and a function response to a string for use in a system instruction."""
  test_function_call = types.FunctionCall(
      name="test_function",
      args={"test_string_argument": "test_value", "test_int_argument": 1},
  )
  test_function_response = types.FunctionResponse(
      name="test_function",
      response={"test_string_argument": "test_value", "test_int_argument": 1},
  )
  output_content = [
      types.Content(
          role="model",
          parts=[
              types.Part(text="test_output"),
              types.Part(function_call=test_function_call),
              types.Part(function_response=test_function_response),
          ],
      )
  ]
  test_example = example.Example(input=BASIC_INPUT, output=output_content)

  gemini2 = model is None or "gemini-2" in model
  response_prefix = (
      example_util._FUNCTION_PREFIX
      if gemini2
      else example_util._FUNCTION_RESPONSE_PREFIX
  )
  call_prefix = (
      example_util._FUNCTION_PREFIX
      if gemini2
      else example_util._FUNCTION_CALL_PREFIX
  )

  expected_output = (
      f"{example_util._EXAMPLES_INTRO}"
      f"{example_util._EXAMPLE_START.format(1)}"
      f"{example_util._USER_PREFIX}test_input\n"
      f"{example_util._MODEL_PREFIX}test_output\n"
      f"{call_prefix}"
      "test_function(test_string_argument='test_value', test_int_argument=1)"
      f"{example_util._FUNCTION_CALL_SUFFIX}"
      f"{response_prefix}"
      f"{test_function_response.__dict__}"
      f"{example_util._FUNCTION_RESPONSE_SUFFIX}"
      f"{example_util._EXAMPLE_END}"
      f"{example_util._EXAMPLES_END}"
  )

  assert (
      example_util.convert_examples_to_text(
          examples=[test_example], model=model
      )
      == expected_output
  )


@pytest.mark.parametrize(
    "model",
    ["gemini-2.5-flash", "llama3_vertex_agent", None],
)
def test_building_si_from_list(model):
  """Tests building System Information from a list of examples."""
  expected_output = (
      f"{example_util._EXAMPLES_INTRO}"
      f"{example_util._EXAMPLE_START.format(1)}"
      f"{example_util._USER_PREFIX}test_input\n"
      f"{example_util._MODEL_PREFIX}test_output\n"
      f"{example_util._EXAMPLE_END}"
      f"{example_util._EXAMPLES_END}"
  )

  assert (
      example_util.build_example_si(
          examples=[BASIC_EXAMPLE], query="", model=model
      )
      == expected_output
  )


@pytest.mark.parametrize(
    "model",
    ["gemini-2.5-flash", "llama3_vertex_agent", None],
)
def test_building_si_from_base_example_provider(model):
  """Tests building System Information from an example provider."""
  expected_output = (
      f"{example_util._EXAMPLES_INTRO}"
      f"{example_util._EXAMPLE_START.format(1)}"
      f"{example_util._USER_PREFIX}test_input\n"
      f"{example_util._MODEL_PREFIX}test_output\n"
      f"{example_util._EXAMPLE_END}"
      f"{example_util._EXAMPLES_END}"
  )

  example_provider = MockExampleProvider(
      test_examples=[BASIC_EXAMPLE], test_query="test_query"
  )

  assert (
      example_util.build_example_si(
          examples=example_provider, query="test_query", model=model
      )
      == expected_output
  )
