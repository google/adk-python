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

from google.adk.models.gemma_llm import Gemma
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from google.genai.types import Content
from google.genai.types import Part
import pytest

DEFAULT_GEMMA_MODEL = "gemma-3-1b-it"


@pytest.fixture
def gemma_llm():
  return Gemma(model=DEFAULT_GEMMA_MODEL)


@pytest.fixture
def gemma_request():
  return LlmRequest(
      model=DEFAULT_GEMMA_MODEL,
      contents=[
          Content(
              role="user",
              parts=[
                  Part.from_text(text="You are a helpful assistant."),
                  Part.from_text(text="Hello!"),
              ],
          )
      ],
      config=types.GenerateContentConfig(
          temperature=0.1,
          response_modalities=[types.Modality.TEXT],
          system_instruction="Talk like a pirate.",
      ),
  )


@pytest.mark.asyncio
@pytest.mark.parametrize("llm_backend", ["GOOGLE_AI"])
async def test_generate_content_async(gemma_llm, gemma_request):
  async for response in gemma_llm.generate_content_async(gemma_request):
    assert isinstance(response, LlmResponse)
    assert response.content.parts[0].text
