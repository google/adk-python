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

"""Tests for custom client injection in ADK models."""

# pylint: disable=protected-access

from unittest import mock

from anthropic import AsyncAnthropic
from anthropic import AsyncAnthropicVertex
from anthropic import types as anthropic_types
from google.adk.models.anthropic_llm import AnthropicLlm
from google.adk.models.anthropic_llm import Claude
from google.adk.models.apigee_llm import ApigeeLlm
from google.adk.models.google_llm import Gemini
from google.adk.models.llm_request import LlmRequest
from google.genai import Client
from google.genai import types
from google.genai.types import Content
from google.genai.types import Part
import pytest


def test_gemini_custom_client():
  """Verify that Gemini uses the provided custom client."""
  mock_client = mock.MagicMock(spec=Client)
  gemini = Gemini(model="gemini-1.5-flash", client=mock_client)

  assert gemini.api_client is mock_client
  # Verify it persists (cached_property)
  assert gemini.api_client is mock_client
  assert gemini._live_api_client is mock_client


@pytest.mark.asyncio
async def test_gemini_uses_custom_client_in_connect():
  """Verify that Gemini connect uses the provided custom client."""
  mock_client = mock.MagicMock(spec=Client)
  mock_live_session = mock.AsyncMock()

  class MockLiveConnect:

    async def __aenter__(self):
      return mock_live_session

    async def __aexit__(self, *args):
      pass

  mock_client.aio.live.connect.return_value = MockLiveConnect()

  gemini = Gemini(model="gemini-1.5-flash", client=mock_client)
  request = LlmRequest(
      model="gemini-1.5-flash",
  )

  async with gemini.connect(request) as connection:
    mock_client.aio.live.connect.assert_called_once()
    assert connection._gemini_session is mock_live_session


def test_anthropic_custom_client():
  """Verify that AnthropicLlm uses the provided custom client."""
  mock_client = mock.MagicMock(spec=AsyncAnthropic)
  anthropic_llm = AnthropicLlm(
      model="claude-3-5-sonnet-20241022", client=mock_client
  )

  assert anthropic_llm._anthropic_client is mock_client


@pytest.mark.asyncio
async def test_gemini_uses_custom_client_in_call():
  """Verify that Gemini calls use the provided custom client's methods."""
  mock_client = mock.MagicMock(spec=Client)
  # Mock the nested aio.models.generate_content
  mock_aio_models = mock_client.aio.models

  gemini = Gemini(model="gemini-1.5-flash", client=mock_client)

  request = LlmRequest(
      model="gemini-1.5-flash",
      contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
  )

  # Mock the response
  mock_response = types.GenerateContentResponse(
      candidates=[
          types.Candidate(
              content=Content(
                  role="model", parts=[Part.from_text(text="Hello")]
              ),
              finish_reason=types.FinishReason.STOP,
          )
      ]
  )

  async def mock_coro(*_, **__):
    return mock_response

  mock_aio_models.generate_content.return_value = mock_coro()

  # We use stream=False to simplify the mock
  responses = [
      r async for r in gemini.generate_content_async(request, stream=False)
  ]

  assert len(responses) == 1
  assert responses[0].content.parts[0].text == "Hello"
  mock_aio_models.generate_content.assert_called()


@pytest.mark.asyncio
async def test_anthropic_uses_custom_client_in_call():
  """Verify that AnthropicLlm calls use the provided custom client's methods."""
  mock_client = mock.MagicMock(spec=AsyncAnthropic)
  mock_messages = mock_client.messages

  anthropic_llm = AnthropicLlm(
      model="claude-3-5-sonnet-20241022", client=mock_client
  )

  request = LlmRequest(
      model="claude-3-5-sonnet-20241022",
      contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
  )

  mock_response = anthropic_types.Message(
      id="msg_test",
      content=[anthropic_types.TextBlock(text="Hello", type="text")],
      model="claude-3-5-sonnet-20241022",
      role="assistant",
      stop_reason="end_turn",
      type="message",
      usage=anthropic_types.Usage(input_tokens=1, output_tokens=1),
  )

  async def mock_coro(*_, **__):
    return mock_response

  mock_messages.create.return_value = mock_coro()

  responses = [
      r
      async for r in anthropic_llm.generate_content_async(request, stream=False)
  ]

  assert len(responses) == 1
  assert responses[0].content.parts[0].text == "Hello"
  mock_messages.create.assert_called()


def test_apigee_custom_client():
  """Verify that ApigeeLlm uses the provided custom client."""
  mock_client = mock.MagicMock(spec=Client)
  apigee_llm = ApigeeLlm(
      model="apigee/gemini/gemini-1.5-flash", client=mock_client
  )

  assert apigee_llm.api_client is mock_client
  # Verify it persists (cached_property)
  assert apigee_llm.api_client is mock_client


@pytest.mark.asyncio
async def test_apigee_uses_custom_client_in_call():
  """Verify that ApigeeLlm calls use the provided custom client's methods."""
  mock_client = mock.MagicMock(spec=Client)
  mock_aio_models = mock_client.aio.models

  apigee_llm = ApigeeLlm(
      model="apigee/gemini/gemini-1.5-flash", client=mock_client
  )

  request = LlmRequest(
      model="apigee/gemini/gemini-1.5-flash",
      contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
  )

  mock_response = types.GenerateContentResponse(
      candidates=[
          types.Candidate(
              content=Content(
                  role="model", parts=[Part.from_text(text="Hello")]
              ),
              finish_reason=types.FinishReason.STOP,
          )
      ]
  )

  async def mock_coro(*_, **__):
    return mock_response

  mock_aio_models.generate_content.return_value = mock_coro()

  responses = [
      r async for r in apigee_llm.generate_content_async(request, stream=False)
  ]

  assert len(responses) == 1
  assert responses[0].content.parts[0].text == "Hello"
  mock_aio_models.generate_content.assert_called()


def test_claude_custom_client():
  """Verify that Claude uses the provided custom client."""
  mock_client = mock.MagicMock(spec=AsyncAnthropicVertex)
  claude = Claude(
      model="projects/p/locations/l/publishers/google/models/claude-3-5-sonnet-v2@20241022",
      client=mock_client,
  )

  assert claude._anthropic_client is mock_client


def test_claude_rejects_non_vertex_client():
  """Verify that Claude rejects an AsyncAnthropic client."""
  mock_client = mock.MagicMock(spec=AsyncAnthropic)
  claude = Claude(
      model="projects/p/locations/l/publishers/google/models/claude-3-5-sonnet-v2@20241022",
      client=mock_client,
  )

  with pytest.raises(
      ValueError, match="Claude requires an AsyncAnthropicVertex client."
  ):
    _ = claude._anthropic_client


def test_apigee_custom_client_warnings():
  """Verify warnings when custom client is used with conflicting options in ApigeeLlm."""
  mock_client = mock.MagicMock(spec=Client)

  # Warning when proxy_url is also provided
  with pytest.warns(
      UserWarning, match="Both client and proxy_url/custom_headers"
  ):
    ApigeeLlm(
        model="apigee/gemini/gemini-1.5-flash",
        client=mock_client,
        proxy_url="http://example.com",
    )

  # Warning when custom_headers are also provided
  with pytest.warns(
      UserWarning, match="Both client and proxy_url/custom_headers"
  ):
    ApigeeLlm(
        model="apigee/gemini/gemini-1.5-flash",
        client=mock_client,
        custom_headers={"X-Test": "test"},
    )

  # Warning when api_type is CHAT_COMPLETIONS
  with pytest.warns(
      UserWarning, match="injected client will be ignored for CHAT_COMPLETIONS"
  ):
    ApigeeLlm(
        model="apigee/openai/gpt-4",
        client=mock_client,
        api_type=ApigeeLlm.ApiType.CHAT_COMPLETIONS,
    )

  # Warning when proxy_url is set via env var
  with mock.patch.dict(
      "os.environ", {"APIGEE_PROXY_URL": "http://example.com"}
  ):
    with pytest.warns(
        UserWarning, match="Both client and proxy_url/custom_headers"
    ):
      ApigeeLlm(
          model="apigee/gemini/gemini-1.5-flash",
          client=mock_client,
      )
