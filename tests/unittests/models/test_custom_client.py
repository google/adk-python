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

import pytest
from unittest import mock
from google.genai import Client
from anthropic import AsyncAnthropic
from google.adk.models.google_llm import Gemini
from google.adk.models.anthropic_llm import AnthropicLlm
from google.adk.models.llm_request import LlmRequest
from google.genai import types
from google.genai.types import Content, Part

def test_gemini_custom_client():
    """Verify that Gemini uses the provided custom client."""
    mock_client = mock.MagicMock(spec=Client)
    gemini = Gemini(model="gemini-1.5-flash", client=mock_client)
    
    assert gemini.api_client is mock_client
    # Verify it persists (cached_property)
    assert gemini.api_client is mock_client

def test_anthropic_custom_client():
    """Verify that AnthropicLlm uses the provided custom client."""
    mock_client = mock.MagicMock(spec=AsyncAnthropic)
    anthropic_llm = AnthropicLlm(model="claude-3-5-sonnet-20241022", client=mock_client)
    
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
        contents=[Content(role="user", parts=[Part.from_text(text="Hi")])]
    )
    
    # Mock the response
    mock_response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=Content(role="model", parts=[Part.from_text(text="Hello")]),
                finish_reason=types.FinishReason.STOP
            )
        ]
    )
    
    async def mock_coro(*args, **kwargs):
        return mock_response
    
    mock_aio_models.generate_content.return_value = mock_coro()
    
    # We use stream=False to simplify the mock
    responses = [r async for r in gemini.generate_content_async(request, stream=False)]
    
    assert len(responses) == 1
    assert responses[0].content.parts[0].text == "Hello"
    mock_aio_models.generate_content.assert_called()

@pytest.mark.asyncio
async def test_anthropic_uses_custom_client_in_call():
    """Verify that AnthropicLlm calls use the provided custom client's methods."""
    mock_client = mock.MagicMock(spec=AsyncAnthropic)
    mock_messages = mock_client.messages
    
    anthropic_llm = AnthropicLlm(model="claude-3-5-sonnet-20241022", client=mock_client)
    
    request = LlmRequest(
        model="claude-3-5-sonnet-20241022",
        contents=[Content(role="user", parts=[Part.from_text(text="Hi")])]
    )
    
    from anthropic import types as anthropic_types
    mock_response = anthropic_types.Message(
        id="msg_test",
        content=[anthropic_types.TextBlock(text="Hello", type="text")],
        model="claude-3-5-sonnet-20241022",
        role="assistant",
        stop_reason="end_turn",
        type="message",
        usage=anthropic_types.Usage(input_tokens=1, output_tokens=1)
    )
    
    async def mock_coro(*args, **kwargs):
        return mock_response
    
    mock_messages.create.return_value = mock_coro()
    
    responses = [r async for r in anthropic_llm.generate_content_async(request, stream=False)]
    
    assert len(responses) == 1
    assert responses[0].content.parts[0].text == "Hello"
    mock_messages.create.assert_called()
