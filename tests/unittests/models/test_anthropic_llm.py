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

import os
import sys
from unittest import mock

from anthropic import types as anthropic_types
from google.adk import version as adk_version
from google.adk.models import anthropic_llm
from google.adk.models.anthropic_llm import Claude
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from google.genai import version as genai_version
from google.genai.types import Content
from google.genai.types import Part
import pytest


@pytest.fixture
def generate_content_response():
  return anthropic_types.Message(
      id="msg_vrtx_testid",
      content=[
          anthropic_types.TextBlock(
              citations=None, text="Hi! How can I help you today?", type="text"
          )
      ],
      model="claude-3-5-sonnet-v2-20241022",
      role="assistant",
      stop_reason="end_turn",
      stop_sequence=None,
      type="message",
      usage=anthropic_types.Usage(
          cache_creation_input_tokens=0,
          cache_read_input_tokens=0,
          input_tokens=13,
          output_tokens=12,
          server_tool_use=None,
          service_tier=None,
      ),
  )


@pytest.fixture
def generate_llm_response():
  return LlmResponse.create(
      types.GenerateContentResponse(
          candidates=[
              types.Candidate(
                  content=Content(
                      role="model",
                      parts=[Part.from_text(text="Hello, how can I help you?")],
                  ),
                  finish_reason=types.FinishReason.STOP,
              )
          ]
      )
  )


@pytest.fixture
def claude_llm():
  return Claude(model="claude-3-5-sonnet-v2@20241022")


@pytest.fixture
def llm_request():
  return LlmRequest(
      model="claude-3-5-sonnet-v2@20241022",
      contents=[Content(role="user", parts=[Part.from_text(text="Hello")])],
      config=types.GenerateContentConfig(
          temperature=0.1,
          response_modalities=[types.Modality.TEXT],
          system_instruction="You are a helpful assistant",
      ),
  )


def test_supported_models():
  models = Claude.supported_models()
  assert len(models) == 2
  assert models[0] == r"claude-3-.*"
  assert models[1] == r"claude-.*-4.*"


@pytest.mark.asyncio
async def test_generate_content_async(
    claude_llm, llm_request, generate_content_response, generate_llm_response
):
  with mock.patch.object(claude_llm, "_anthropic_client") as mock_client:
    with mock.patch.object(
        anthropic_llm,
        "message_to_generate_content_response",
        return_value=generate_llm_response,
    ):
      # Create a mock coroutine that returns the generate_content_response.
      async def mock_coro():
        return generate_content_response

      # Assign the coroutine to the mocked method
      mock_client.messages.create.return_value = mock_coro()

      responses = [
          resp
          async for resp in claude_llm.generate_content_async(
              llm_request, stream=False
          )
      ]
      assert len(responses) == 1
      assert isinstance(responses[0], LlmResponse)
      assert responses[0].content.parts[0].text == "Hello, how can I help you?"


@pytest.mark.asyncio
async def test_generate_content_async_with_max_tokens(
    llm_request, generate_content_response, generate_llm_response
):
  claude_llm = Claude(model="claude-3-5-sonnet-v2@20241022", max_tokens=4096)
  with mock.patch.object(claude_llm, "_anthropic_client") as mock_client:
    with mock.patch.object(
        anthropic_llm,
        "message_to_generate_content_response",
        return_value=generate_llm_response,
    ):
      # Create a mock coroutine that returns the generate_content_response.
      async def mock_coro():
        return generate_content_response

      # Assign the coroutine to the mocked method
      mock_client.messages.create.return_value = mock_coro()

      _ = [
          resp
          async for resp in claude_llm.generate_content_async(
              llm_request, stream=False
          )
      ]
      mock_client.messages.create.assert_called_once()
      _, kwargs = mock_client.messages.create.call_args
      assert kwargs["max_tokens"] == 4096
