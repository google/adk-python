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
from typing import Optional

from google.genai import types
from typing_extensions import override

from .base_example_provider import BaseExampleProvider
from .example import Example

_TOP_K = 10

# Below this an example is more likely to mislead the model than help it.
_SIMILARITY_THRESHOLD = 0.5


class VertexAiExampleStore(BaseExampleProvider):
  """Provides examples from Vertex example store."""

  def __init__(
      self,
      examples_store_name: str,
      *,
      project: Optional[str] = None,
      location: Optional[str] = None,
  ):
    """Initializes the VertexAiExampleStore.

    Args:
        examples_store_name: The resource name of the vertex example store, in
          the format of
          ``projects/{project}/locations/{location}/exampleStores/{example_store}``.
        project: The project to use for the Agent Platform client. If not set,
          the GOOGLE_CLOUD_PROJECT environment variable is used, falling back to
          the project in ``examples_store_name``.
        location: The location to use for the Agent Platform client. If not set,
          the GOOGLE_CLOUD_LOCATION environment variable is used, falling back
          to the location in ``examples_store_name``.
    """
    try:
      import agentplatform  # noqa: F401
    except ImportError as e:
      from ..utils._dependency import missing_extra

      raise missing_extra("google-cloud-aiplatform", "gcp") from e

    self.examples_store_name = examples_store_name
    self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    self._location = location or os.environ.get("GOOGLE_CLOUD_LOCATION")

    # Fallback: a fully-qualified store name already carries both, so a caller
    # that passed one should not also have to set the environment.
    if (not self._project or not self._location) and (
        examples_store_name.startswith("projects/")
    ):
      parts = examples_store_name.split("/")
      if len(parts) >= 4 and parts[0] == "projects" and parts[2] == "locations":
        self._project = self._project or parts[1]
        self._location = self._location or parts[3]

  @override
  def get_examples(self, query: str) -> list[Example]:
    import agentplatform

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
      stored_contents_example = result.example.stored_contents_example
      contents_example = stored_contents_example.contents_example

      # The module hands back google.genai Content already, so the expected
      # output needs no part-by-part rebuilding.
      expected_output = [
          expected.content
          for expected in contents_example.expected_contents or []
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
