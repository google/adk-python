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

import os

from google.genai import types
from typing_extensions import override

from .base_example_provider import BaseExampleProvider
from .example import Example

_TOP_K = 10

# Below this an example is more likely to mislead the model than help it.
_SIMILARITY_THRESHOLD = 0.5


class VertexAiExampleStore(BaseExampleProvider):
  """Provides examples from Vertex example store."""

  def __init__(self, examples_store_name: str):
    """Initializes the VertexAiExampleStore.

    Args:
        examples_store_name: The resource name of the vertex example store, in
          the format of
          ``projects/{project}/locations/{location}/exampleStores/{example_store}``.
    """
    try:
      from ..dependencies._agentplatform import agentplatform  # noqa: F401
    except ImportError as e:
      from ..utils._dependency import missing_extra

      raise missing_extra("google-cloud-agentplatform", "gcp") from e

    self.examples_store_name = examples_store_name

    store_project = None
    store_location = None
    if examples_store_name.startswith("projects/"):
      parts = examples_store_name.split("/")
      if len(parts) >= 4 and parts[0] == "projects" and parts[2] == "locations":
        store_project = parts[1]
        store_location = parts[3]

    # A fully-qualified name names exactly one endpoint, so it wins; the
    # environment only fills in what a bare store ID leaves unspecified.
    self._project = store_project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    self._location = store_location or os.environ.get("GOOGLE_CLOUD_LOCATION")

  @override
  def get_examples(self, query: str) -> list[Example]:
    from ..dependencies._agentplatform import agentplatform

    client = agentplatform.Client(
        project=self._project, location=self._location
    )
    response = client.example_stores.search_examples(
        name=self.examples_store_name,
        stored_contents_example_parameters={
            "content_search_key": {
                "contents": [{"role": "user", "parts": [{"text": query}]}],
                "search_key_generation_method": {"last_entry": {}},
            }
        },
        config={"top_k": _TOP_K},
    )

    returned_examples = []
    for result in response.results or []:
      if (result.similarity_score or 0.0) < _SIMILARITY_THRESHOLD:
        continue
      stored_contents_example = (
          result.example.stored_contents_example if result.example else None
      )
      if not stored_contents_example:
        continue
      contents_example = stored_contents_example.contents_example
      expected_contents = (
          contents_example.expected_contents if contents_example else None
      )

      # The module hands back google.genai Content already, so the expected
      # output needs no part-by-part rebuilding.
      expected_output = [
          expected.content
          for expected in expected_contents or []
          if expected.content
      ]

      returned_examples.append(
          Example(
              input=types.Content(
                  role="user",
                  parts=[
                      types.Part.from_text(
                          text=stored_contents_example.search_key or ""
                      )
                  ],
              ),
              output=expected_output,
          )
      )
    return returned_examples
