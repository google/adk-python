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

from google.adk.examples.vertex_ai_example_store import VertexAiExampleStore
from google.genai import types
import pytest
from pytest_mock import MockerFixture

_STORE_NAME = "projects/p/locations/l/exampleStores/s"


def _expected_content(*, role, parts):
  return SimpleNamespace(content=types.Content(role=role, parts=parts))


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
def search_examples(mocker: MockerFixture):
  """Patches agentplatform.Client and returns its search_examples mock."""
  client = mocker.Mock()
  mocker.patch("agentplatform.Client", return_value=client)
  return client.example_stores.search_examples


def test_get_examples_searches_the_configured_store(search_examples):
  search_examples.return_value = SimpleNamespace(results=[])

  VertexAiExampleStore(_STORE_NAME).get_examples("what is the weather?")

  search_examples.assert_called_once_with(
      name=_STORE_NAME,
      stored_contents_example_parameters={
          "content_search_key": {
              "contents": [{
                  "role": "user",
                  "parts": [{"text": "what is the weather?"}],
              }],
              "search_key_generation_method": {"last_entry": {}},
          }
      },
      config={"top_k": 10},
  )


def test_get_examples_derives_project_and_location_from_the_store_name(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
):
  # The suite's conftest exports both, so the fallback is only reachable once
  # they are unset.
  monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
  monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
  client_factory = mocker.patch("agentplatform.Client")
  client_factory.return_value.example_stores.search_examples.return_value = (
      SimpleNamespace(results=[])
  )

  VertexAiExampleStore(_STORE_NAME).get_examples("query")

  client_factory.assert_called_once_with(project="p", location="l")


def test_get_examples_prefers_the_environment_over_the_store_name(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
):
  monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")
  monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "env-location")
  client_factory = mocker.patch("agentplatform.Client")
  client_factory.return_value.example_stores.search_examples.return_value = (
      SimpleNamespace(results=[])
  )

  VertexAiExampleStore(_STORE_NAME).get_examples("query")

  client_factory.assert_called_once_with(
      project="env-project", location="env-location"
  )


def test_get_examples_prefers_explicit_project_and_location(
    mocker: MockerFixture,
):
  client_factory = mocker.patch("agentplatform.Client")
  client_factory.return_value.example_stores.search_examples.return_value = (
      SimpleNamespace(results=[])
  )

  VertexAiExampleStore(
      _STORE_NAME, project="other-project", location="other-location"
  ).get_examples("query")

  client_factory.assert_called_once_with(
      project="other-project", location="other-location"
  )


def test_get_examples_returns_empty_list_without_results(search_examples):
  search_examples.return_value = SimpleNamespace(results=[])

  assert VertexAiExampleStore(_STORE_NAME).get_examples("query") == []


def test_get_examples_tolerates_unset_results(search_examples):
  # The response field is optional, so an empty search omits it entirely.
  search_examples.return_value = SimpleNamespace(results=None)

  assert VertexAiExampleStore(_STORE_NAME).get_examples("query") == []


def test_get_examples_converts_text_part(search_examples):
  search_examples.return_value = SimpleNamespace(
      results=[
          _result(
              search_key="what is the weather?",
              expected_contents=[
                  _expected_content(
                      role="model",
                      parts=[types.Part.from_text(text="it is sunny")],
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
                          types.Part.from_function_call(
                              name="get_weather", args={"city": "London"}
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
                          types.Part.from_function_response(
                              name="get_weather",
                              response={"temperature": 12},
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


def test_get_examples_preserves_multi_step_expected_output(search_examples):
  # expected_contents is repeated to represent iterative reasoning steps; all
  # of them belong in the example's output, in order.
  search_examples.return_value = SimpleNamespace(
      results=[
          _result(
              expected_contents=[
                  _expected_content(
                      role="model", parts=[types.Part.from_text(text="step 1")]
                  ),
                  _expected_content(
                      role="model", parts=[types.Part.from_text(text="step 2")]
                  ),
              ],
          )
      ]
  )

  examples = VertexAiExampleStore(_STORE_NAME).get_examples("query")

  assert [content.parts[0].text for content in examples[0].output] == [
      "step 1",
      "step 2",
  ]


def test_get_examples_skips_expected_contents_without_content(search_examples):
  search_examples.return_value = SimpleNamespace(
      results=[
          _result(
              expected_contents=[
                  SimpleNamespace(content=None),
                  _expected_content(
                      role="model", parts=[types.Part.from_text(text="kept")]
                  ),
              ],
          )
      ]
  )

  examples = VertexAiExampleStore(_STORE_NAME).get_examples("query")

  assert [content.parts[0].text for content in examples[0].output] == ["kept"]
