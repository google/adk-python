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

from google.adk.models.google_llm import Gemini
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from google.genai.types import Content
from google.genai.types import Part
import pytest


@pytest.fixture
def gemini_llm():
  return Gemini(model="gemini-1.5-flash")


@pytest.fixture
def llm_request():
  return LlmRequest(
      model="gemini-1.5-flash",
      contents=[Content(role="user", parts=[Part.from_text(text="Hello")])],
      config=types.GenerateContentConfig(
          temperature=0.1,
          response_modalities=[types.Modality.TEXT],
          system_instruction="You are a helpful assistant",
      ),
  )


@pytest.mark.asyncio
async def test_generate_content_async(gemini_llm, llm_request):
  async for response in gemini_llm.generate_content_async(llm_request):
    assert isinstance(response, LlmResponse)
    assert response.content.parts[0].text


@pytest.mark.asyncio
async def test_generate_content_async_stream(gemini_llm, llm_request):
  responses = [
      resp
      async for resp in gemini_llm.generate_content_async(
          llm_request, stream=True
      )
  ]
  text = ""
  for i in range(len(responses) - 1):
    assert responses[i].partial is True
    assert responses[i].content.parts[0].text
    text += responses[i].content.parts[0].text

  # Last message should be accumulated text
  assert responses[-1].content.parts[0].text == text
  assert not responses[-1].partial
