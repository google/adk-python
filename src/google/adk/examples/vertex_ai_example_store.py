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

from __future__ import annotations

from google.genai import types
from typing_extensions import override

from .base_example_provider import BaseExampleProvider
from .example import Example


class VertexAiExampleStore(BaseExampleProvider):
  """Provides examples from Vertex example store."""

  def __init__(self, examples_store_name: str):
    """Initializes the VertexAiExampleStore.

    Args:
        examples_store_name: The resource name of the vertex example store, in
          the format of
          ``projects/{project}/locations/{location}/exampleStores/{example_store}``.
    """
    self.examples_store_name = examples_store_name

  @override
  def get_examples(self, query: str) -> list[Example]:
    from ..dependencies.vertexai import example_stores

    example_store = example_stores.ExampleStore(self.examples_store_name)
    # Retrieve relevant examples.
    request = {
        "stored_contents_example_parameters": {
            "content_search_key": {
                "contents": [{"role": "user", "parts": [{"text": query}]}],
                "search_key_generation_method": {"last_entry": {}},
            }
        },
        "top_k": 10,
        "example_store": self.examples_store_name,
    }
    response = example_store.api_client.search_examples(request)

    returned_examples = []
    # Convert results to genai formats
    for result in response.results:
      if result.similarity_score < 0.5:
        continue
      expected_contents = [
          content.content
          for content in result.example.stored_contents_example.contents_example.expected_contents
      ]
      expected_output = []
      for content in expected_contents:
        expected_parts = []
        for part in content.parts:
          if part.text:
            expected_parts.append(types.Part.from_text(text=part.text))
          elif part.function_call:
            expected_parts.append(
                types.Part.from_function_call(
                    name=part.function_call.name,
                    args={
                        key: value
                        for key, value in part.function_call.args.items()
                    },
                )
            )
          elif part.function_response:
            expected_parts.append(
                types.Part.from_function_response(
                    name=part.function_response.name,
                    response={
                        key: value
                        for key, value in part.function_response.response.items()
                    },
                )
            )
        expected_output.append(
            types.Content(role=content.role, parts=expected_parts)
        )

      returned_examples.append(
          Example(
              input=types.Content(
                  role="user",
                  parts=[
                      types.Part.from_text(
                          text=result.example.stored_contents_example.search_key
                      )
                  ],
              ),
              output=expected_output,
          )
      )
    return returned_examples
