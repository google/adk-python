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

import sys
from unittest import mock

from google.adk import version as adk_version
from google.adk.models.gemini_llm_connection import GeminiLlmConnection
from google.adk.models.google_llm import Gemini
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from google.genai import version as genai_version
from google.genai.types import Content
from google.genai.types import Part
import pytest


@pytest.fixture
def generate_content_response():
  return types.GenerateContentResponse(
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


def test_supported_models():
  models = Gemini.supported_models()
  assert len(models) == 3
  assert models[0] == r"gemini-.*"
  assert models[1] == r"projects\/.+\/locations\/.+\/endpoints\/.+"
  assert (
      models[2]
      == r"projects\/.+\/locations\/.+\/publishers\/google\/models\/gemini.+"
  )


def test_client_version_header():
  model = Gemini(model="gemini-1.5-flash")
  client = model.api_client
  adk_header = (
      f"google-adk/{adk_version.__version__} gl-python/{sys.version.split()[0]}"
  )
  genai_header = (
      f"google-genai-sdk/{genai_version.__version__} gl-python/{sys.version.split()[0]} "
  )
  expected_header = genai_header + adk_header

  assert (
      expected_header
      in client._api_client._http_options.headers["x-goog-api-client"]
  )
  assert (
      expected_header in client._api_client._http_options.headers["user-agent"]
  )


def test_maybe_append_user_content(gemini_llm, llm_request):
  # Test with user content already present
  gemini_llm._maybe_append_user_content(llm_request)
  assert len(llm_request.contents) == 1

  # Test with model content as the last message
  llm_request.contents.append(
      Content(role="model", parts=[Part.from_text(text="Response")])
  )
  gemini_llm._maybe_append_user_content(llm_request)
  assert len(llm_request.contents) == 3
  assert llm_request.contents[-1].role == "user"
  assert "Continue processing" in llm_request.contents[-1].parts[0].text


@pytest.mark.asyncio
async def test_generate_content_async(
    gemini_llm, llm_request, generate_content_response
):
  with mock.patch.object(gemini_llm, "api_client") as mock_client:
    # Create a mock coroutine that returns the generate_content_response
    async def mock_coro():
      return generate_content_response

    # Assign the coroutine to the mocked method
    mock_client.aio.models.generate_content.return_value = mock_coro()

    responses = [
        resp
        async for resp in gemini_llm.generate_content_async(
            llm_request, stream=False
        )
    ]

    assert len(responses) == 1
    assert isinstance(responses[0], LlmResponse)
    assert responses[0].content.parts[0].text == "Hello, how can I help you?"
    mock_client.aio.models.generate_content.assert_called_once()


@pytest.mark.asyncio
async def test_generate_content_async_stream(gemini_llm, llm_request):
  with mock.patch.object(gemini_llm, "api_client") as mock_client:
    # Create mock stream responses
    class MockAsyncIterator:

      def __init__(self, seq):
        self.iter = iter(seq)

      def __aiter__(self):
        return self

      async def __anext__(self):
        try:
          return next(self.iter)
        except StopIteration:
          raise StopAsyncIteration

    mock_responses = [
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=Content(
                        role="model", parts=[Part.from_text(text="Hello")]
                    ),
                    finish_reason=None,
                )
            ]
        ),
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=Content(
                        role="model", parts=[Part.from_text(text=", how")]
                    ),
                    finish_reason=None,
                )
            ]
        ),
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=Content(
                        role="model",
                        parts=[Part.from_text(text=" can I help you?")],
                    ),
                    finish_reason=types.FinishReason.STOP,
                )
            ]
        ),
    ]

    # Create a mock coroutine that returns the MockAsyncIterator
    async def mock_coro():
      return MockAsyncIterator(mock_responses)

    # Set the mock to return the coroutine
    mock_client.aio.models.generate_content_stream.return_value = mock_coro()

    responses = [
        resp
        async for resp in gemini_llm.generate_content_async(
            llm_request, stream=True
        )
    ]

    # Assertions remain the same
    assert len(responses) == 4
    assert responses[0].partial is True
    assert responses[1].partial is True
    assert responses[2].partial is True
    assert responses[3].content.parts[0].text == "Hello, how can I help you?"
    mock_client.aio.models.generate_content_stream.assert_called_once()


@pytest.mark.asyncio
async def test_generate_content_async_stream_preserves_thinking_and_text_parts(
    gemini_llm, llm_request
):
  with mock.patch.object(gemini_llm, "api_client") as mock_client:

    class MockAsyncIterator:

      def __init__(self, seq):
        self._iter = iter(seq)

      def __aiter__(self):
        return self

      async def __anext__(self):
        try:
          return next(self._iter)
        except StopIteration:
          raise StopAsyncIteration

    response1 = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=Content(
                    role="model",
                    parts=[Part(text="Think1", thought=True)],
                ),
                finish_reason=None,
            )
        ]
    )
    response2 = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=Content(
                    role="model",
                    parts=[Part(text="Think2", thought=True)],
                ),
                finish_reason=None,
            )
        ]
    )
    response3 = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=Content(
                    role="model",
                    parts=[Part.from_text(text="Answer.")],
                ),
                finish_reason=types.FinishReason.STOP,
            )
        ]
    )

    async def mock_coro():
      return MockAsyncIterator([response1, response2, response3])

    mock_client.aio.models.generate_content_stream.return_value = mock_coro()

    responses = [
        resp
        async for resp in gemini_llm.generate_content_async(
            llm_request, stream=True
        )
    ]

    assert len(responses) == 4
    assert responses[0].partial is True
    assert responses[1].partial is True
    assert responses[2].partial is True
    assert responses[3].content.parts[0].text == "Think1Think2"
    assert responses[3].content.parts[0].thought is True
    assert responses[3].content.parts[1].text == "Answer."
    mock_client.aio.models.generate_content_stream.assert_called_once()


@pytest.mark.asyncio
async def test_connect(gemini_llm, llm_request):
  # Create a mock connection
  mock_connection = mock.MagicMock(spec=GeminiLlmConnection)

  # Create a mock context manager
  class MockContextManager:

    async def __aenter__(self):
      return mock_connection

    async def __aexit__(self, *args):
      pass

  # Mock the connect method at the class level
  with mock.patch(
      "google.adk.models.google_llm.Gemini.connect",
      return_value=MockContextManager(),
  ):
    async with gemini_llm.connect(llm_request) as connection:
      assert connection is mock_connection
