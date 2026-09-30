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

"""Tests for vertex_ai_example_store."""

from types import SimpleNamespace
from unittest import mock

from google.adk.examples.vertex_ai_example_store import VertexAiExampleStore
import pytest

_STORE_NAME = "projects/p/locations/l/exampleStores/s"


def _part(*, text=None, function_call=None, function_response=None):
  return SimpleNamespace(
      text=text,
      function_call=function_call,
      function_response=function_response,
  )


def _expected_content(*, role, parts):
  return SimpleNamespace(content=SimpleNamespace(role=role, parts=parts))


def _result(*, search_key="search key", expected_contents=(), score=1.0):
  return SimpleNamespace(
      similarity_score=score,
      example=SimpleNamespace(
          stored_contents_example=SimpleNamespace(
              search_key=search_key,
              contents_example=SimpleNamespace(
                  expected_contents=list(expected_contents)
              ),
          )
      ),
  )


@pytest.fixture
def mock_example_stores():
  with mock.patch(
      "google.adk.dependencies.vertexai.example_stores"
  ) as example_stores:
    yield example_stores


@pytest.fixture
def search_examples(mock_example_stores):
  return (
      mock_example_stores.ExampleStore.return_value.api_client.search_examples
  )


def test_get_examples_searches_the_configured_store(
    mock_example_stores, search_examples
):
  search_examples.return_value = SimpleNamespace(results=[])

  VertexAiExampleStore(_STORE_NAME).get_examples("what is the weather?")

  mock_example_stores.ExampleStore.assert_called_once_with(_STORE_NAME)
  search_examples.assert_called_once_with({
      "stored_contents_example_parameters": {
          "content_search_key": {
              "contents": [{
                  "role": "user",
                  "parts": [{"text": "what is the weather?"}],
              }],
              "search_key_generation_method": {"last_entry": {}},
          }
      },
      "top_k": 10,
      "example_store": _STORE_NAME,
  })


def test_get_examples_returns_empty_list_without_results(search_examples):
  search_examples.return_value = SimpleNamespace(results=[])

  assert VertexAiExampleStore(_STORE_NAME).get_examples("query") == []


def test_get_examples_converts_text_part(search_examples):
  search_examples.return_value = SimpleNamespace(
      results=[
          _result(
              search_key="what is the weather?",
              expected_contents=[
                  _expected_content(
                      role="model", parts=[_part(text="it is sunny")]
                  )
              ],
          )
      ]
  )

  examples = VertexAiExampleStore(_STORE_NAME).get_examples("query")

  assert len(examples) == 1
  assert examples[0].input.role == "user"
  assert [part.text for part in examples[0].input.parts] == [
      "what is the weather?"
  ]
  assert len(examples[0].output) == 1
  assert examples[0].output[0].role == "model"
  assert [part.text for part in examples[0].output[0].parts] == ["it is sunny"]


def test_get_examples_filters_results_below_similarity_threshold(
    search_examples,
):
  search_examples.return_value = SimpleNamespace(
      results=[
          _result(search_key="too dissimilar", score=0.49),
          _result(search_key="similar enough", score=0.5),
      ]
  )

  examples = VertexAiExampleStore(_STORE_NAME).get_examples("query")

  assert [example.input.parts[0].text for example in examples] == [
      "similar enough"
  ]


def test_get_examples_converts_function_call_part(search_examples):
  search_examples.return_value = SimpleNamespace(
      results=[
          _result(
              expected_contents=[
                  _expected_content(
                      role="model",
                      parts=[
                          _part(
                              function_call=SimpleNamespace(
                                  name="get_weather", args={"city": "London"}
                              )
                          )
                      ],
                  )
              ],
          )
      ]
  )

  examples = VertexAiExampleStore(_STORE_NAME).get_examples("query")

  function_call = examples[0].output[0].parts[0].function_call
  assert function_call.name == "get_weather"
  assert function_call.args == {"city": "London"}


def test_get_examples_converts_function_response_part(search_examples):
  search_examples.return_value = SimpleNamespace(
      results=[
          _result(
              expected_contents=[
                  _expected_content(
                      role="user",
                      parts=[
                          _part(
                              function_response=SimpleNamespace(
                                  name="get_weather",
                                  response={"temperature": 12},
                              )
                          )
                      ],
                  )
              ],
          )
      ]
  )

  examples = VertexAiExampleStore(_STORE_NAME).get_examples("query")

  function_response = examples[0].output[0].parts[0].function_response
  assert function_response.name == "get_weather"
  assert function_response.response == {"temperature": 12}
