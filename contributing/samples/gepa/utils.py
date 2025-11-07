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

"""Defines utility for GEPA experiments."""

import logging
from typing import Callable

from google.genai import types
from retry import retry

from google import genai


class FilterInferenceWarnings(logging.Filter):
  """Filters out Vertex inference warning about non-text parts in response."""

  def filter(self, record: logging.LogRecord) -> bool:
    """Filters out Vertex inference warning about non-text parts in response."""
    if record.levelname != 'WARNING':
      return True
    message_identifier = record.getMessage()
    return not message_identifier.startswith(
        'Warning: there are non-text parts in the response:'
    )


def reflection_inference_fn(model: str) -> Callable[[str], str]:
  """Returns an inference function on VertexAI based on provided model."""
  client = genai.Client()

  @retry(tries=3, delay=10, backoff=2)
  def _fn(prompt):
    return client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            candidate_count=1,
            thinking_config=types.ThinkingConfig(
                include_thoughts=True, thinking_budget=-1
            ),
        ),
    ).text

  return _fn
