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

from unittest import mock

import httpx
import pytest
from google.genai import types
from google.genai.types import Content, Part

from google.adk.models.ollama_llm import OllamaLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse


@pytest.fixture
def ollama_response():
    return {
        "model": "llama2",
        "message": {
            "role": "assistant",
            "content": "Hello, how can I help you?"
        },
        "done": True
    }


@pytest.fixture
def ollama_stream_responses():
    return [
        {
            "model": "llama2",
            "message": {
                "role": "assistant",
                "content": "Hello"
            },
            "done": False
        },
        {
            "model": "llama2",
            "message": {
                "role": "assistant",
                "content": ", how"
            },
            "done": False
        },
        {
            "model": "llama2",
            "message": {
                "role": "assistant",
                "content": " can I help you?"
            },
            "done": True
        }
    ]


@pytest.fixture
def ollama_model():
    return OllamaLlm(model="llama2")


@pytest.fixture
def llm_request(ollama_model):
    return LlmRequest(
        model=ollama_model,
        contents=[Content(role="user", parts=[Part.from_text(text="Hello")])],
        config=types.GenerateContentConfig(
            temperature=0.7,
            system_instruction="You are a helpful assistant",
        ),
    )


def test_supported_models():
    models = OllamaLlm.supported_models()
    assert len(models) == 1
    assert models[0] == ".*"  # Ollama supports any locally available model


@pytest.mark.asyncio
async def test_generate_content_async(ollama_model, llm_request, ollama_response):
    with mock.patch.object(httpx, "AsyncClient") as mock_client:
        mock_response = mock.Mock()
        mock_response.raise_for_status = mock.Mock()
        mock_response.json.return_value = ollama_response
        
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

        responses = [
            resp
            async for resp in ollama_model.generate_content_async(
                llm_request, stream=False
            )
        ]

        assert len(responses) == 1
        assert isinstance(responses[0], LlmResponse)
        assert responses[0].candidates[0].parts[0].text == "Hello, how can I help you?"
        
        # Verify API call
        mock_client.return_value.__aenter__.return_value.post.assert_called_once_with(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama2",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
                "options": {"temperature": 0.7},
                "system": "You are a helpful assistant"
            }
        )


@pytest.mark.asyncio
async def test_generate_content_async_stream(ollama_model, llm_request, ollama_stream_responses):
    with mock.patch.object(httpx, "AsyncClient") as mock_client:
        # Mock streaming response
        mock_stream = mock.AsyncMock()
        mock_stream.raise_for_status = mock.Mock()
        
        # Create mock lines iterator
        async def mock_lines():
            for response in ollama_stream_responses:
                yield str(response)
        
        mock_stream.aiter_lines.return_value = mock_lines()
        mock_client.return_value.__aenter__.return_value.stream.return_value.__aenter__.return_value = mock_stream

        responses = [
            resp
            async for resp in ollama_model.generate_content_async(
                llm_request, stream=True
            )
        ]

        assert len(responses) == 3
        assert responses[0].candidates[0].parts[0].text == "Hello"
        assert responses[1].candidates[0].parts[0].text == ", how"
        assert responses[2].candidates[0].parts[0].text == " can I help you?"

        # Verify API call
        mock_client.return_value.__aenter__.return_value.stream.assert_called_once_with(
            'POST',
            "http://localhost:11434/api/chat",
            json={
                "model": "llama2",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
                "options": {"temperature": 0.7},
                "system": "You are a helpful assistant"
            }
        )


@pytest.mark.asyncio
async def test_generate_content_async_error_handling(ollama_model, llm_request):
    with mock.patch.object(httpx, "AsyncClient") as mock_client:
        # Mock error response
        error_response = {
            "error": "Model not found: llama2"
        }
        mock_response = mock.Mock()
        mock_response.raise_for_status = mock.Mock()
        mock_response.json.return_value = error_response
        
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

        with pytest.raises(RuntimeError) as exc_info:
            async for _ in ollama_model.generate_content_async(llm_request, stream=False):
                pass
        
        assert "Ollama API error: Model not found: llama2" in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_content_async_http_error(ollama_model, llm_request):
    with mock.patch.object(httpx, "AsyncClient") as mock_client:
        # Mock HTTP error
        mock_response = mock.Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPError("HTTP Error")
        
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

        with pytest.raises(httpx.HTTPError) as exc_info:
            async for _ in ollama_model.generate_content_async(llm_request, stream=False):
                pass
        
        assert "HTTP Error" in str(exc_info.value)


def test_custom_api_base():
    custom_base = "http://custom-ollama:11434"
    model = OllamaLlm(model="llama2", api_base=custom_base)
    assert model.api_base == custom_base


@pytest.mark.asyncio
async def test_empty_response(ollama_model, llm_request):
    with mock.patch.object(httpx, "AsyncClient") as mock_client:
        # Mock empty response
        mock_response = mock.Mock()
        mock_response.raise_for_status = mock.Mock()
        mock_response.json.return_value = {}
        
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

        with pytest.raises(RuntimeError) as exc_info:
            async for _ in ollama_model.generate_content_async(llm_request, stream=False):
                pass
        
        assert "Invalid response format from Ollama API" in str(exc_info.value)