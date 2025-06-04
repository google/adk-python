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

"""Utilities for Google LLM variants.

This module is for ADK internal use only.
Please do not rely on the implementation details.
"""

from __future__ import annotations

from enum import Enum
import os

_GOOGLE_LLM_VARIANT_VERTEX_AI = 'VERTEX_AI'
_GOOGLE_LLM_VARIANT_GEMINI_API = 'GEMINI_API'


class GoogleLLMVariant(Enum):
  """
  The Google LLM variant to use.
  see https://google.github.io/adk-docs/get-started/quickstart/#set-up-the-model
  """

  VERTEX_AI = _GOOGLE_LLM_VARIANT_VERTEX_AI
  """For using credentials from Google Vertex AI"""
  GEMINI_API = _GOOGLE_LLM_VARIANT_GEMINI_API
  """For using API Key from Google AI Studio"""


def get_google_llm_variant() -> str:
  return (
      GoogleLLMVariant.VERTEX_AI
      if os.environ.get('GOOGLE_GENAI_USE_VERTEXAI', '0').lower()
      in [
          'true',
          '1',
      ]
      else GoogleLLMVariant.GEMINI_API
  )
