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

from google.adk.models import LlmRequest
from google.adk.models import LlmResponse
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from google.genai.types import Content
from google.genai.types import Part
import pytest

_TEST_MODEL_NAME = "vertex_ai/meta/llama-4-maverick-17b-128e-instruct-maas"


_SYSTEM_PROMPT = """You are a helpful assistant."""


@pytest.fixture
def oss_llm():
  return LiteLlm(model=_TEST_MODEL_NAME)


@pytest.fixture
def llm_request():
  return LlmRequest(
      model=_TEST_MODEL_NAME,
      contents=[Content(role="user", parts=[Part.from_text(text="hello")])],
      config=types.GenerateContentConfig(
          temperature=0.1,
          response_modalities=[types.Modality.TEXT],
          system_instruction=_SYSTEM_PROMPT,
      ),
  )


@pytest.mark.asyncio
async def test_generate_content_async(oss_llm, llm_request):
  async for response in oss_llm.generate_content_async(llm_request):
    assert isinstance(response, LlmResponse)
    assert response.content.parts[0].text


# Note that, this test disabled streaming because streaming is not supported
# properly in the current test model for now.
@pytest.mark.asyncio
async def test_generate_content_async_stream(oss_llm, llm_request):
  responses = [
      resp
      async for resp in oss_llm.generate_content_async(
          llm_request, stream=False
      )
  ]
  part = responses[0].content.parts[0]
  assert len(part.text) > 0
