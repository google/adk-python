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

import json
from unittest import mock
import asyncio
import httpx
import pytest
from google.genai import types
from google.genai.types import Content, Part

from google.adk.models.ollama_llm import OllamaLlm, _build_ollama_request_payload
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse


# --- Fixtures ---

@pytest.fixture
def ollama_non_stream_response():
    """Mock response for a non-streaming Ollama request."""
    return {
        "model": "test-model",
        "created_at": "2023-08-04T08:52:19.385406455Z",
        "message": {
            "role": "assistant",
            "content": "The capital of France is Paris."
        },
        "done": True,
        "total_duration": 5063913208,
        "load_duration": 2116083,
        "prompt_eval_count": 26,
        "prompt_eval_duration": 126918000,
        "eval_count": 6,
        "eval_duration": 39883000
    }

@pytest.fixture
def ollama_stream_response_chunks():
    """Mock response chunks for a streaming Ollama request."""
    return [
        {
            "model": "test-model",
            "created_at": "2023-08-04T19:22:45.499127Z",
            "message": {"role": "assistant", "content": "The"},
            "done": False
        },
        {
            "model": "test-model",
            "created_at": "2023-08-04T19:22:45.554171Z",
            "message": {"role": "assistant", "content": " capital"},
            "done": False
        },
        {
            "model": "test-model",
            "created_at": "2023-08-04T19:22:45.60783Z",
            "message": {"role": "assistant", "content": " of"},
            "done": False
        },
        {
            "model": "test-model",
            "created_at": "2023-08-04T19:22:45.669918Z",
            "message": {"role": "assistant", "content": " France"},
            "done": False
        },
        {
            "model": "test-model",
            "created_at": "2023-08-04T19:22:45.728998Z",
            "message": {"role": "assistant", "content": " is"},
            "done": False
        },
        {
            "model": "test-model",
            "created_at": "2023-08-04T19:22:45.786904Z",
            "message": {"role": "assistant", "content": " Paris."},
            "done": False # Note: Even last content chunk has done=False
        },
        { # Final metadata message
            "model": "test-model",
            "created_at": "2023-08-04T19:22:45.846583Z",
            "message": {"role": "assistant", "content": ""}, # Empty content
            "done": True,
            "total_duration": 5063913208,
            "load_duration": 2116083,
            "prompt_eval_count": 26,
            "prompt_eval_duration": 126918000,
            "eval_count": 7, # Different from non-stream
            "eval_duration": 51883000
        }
    ]

@pytest.fixture
def ollama_error_response():
    """Mock error response from Ollama."""
    return {"error": "Model 'unknown-model' not found"}


@pytest.fixture
def mock_httpx_client():
    """Fixture to mock httpx.AsyncClient."""
    with mock.patch("httpx.AsyncClient", autospec=True) as mock_client_class:
        mock_instance = mock_client_class.return_value
        # Mock async context manager methods
        mock_instance.__aenter__.return_value = mock_instance
        mock_instance.__aexit__.return_value = None
        yield mock_instance

@pytest.fixture
def ollama_llm():
    """Fixture for a default OllamaLlm instance."""
    return OllamaLlm(model="test-model")

@pytest.fixture
def basic_llm_request(ollama_llm):
    """Fixture for a basic LlmRequest."""
    # Correctly pass model name as string
    return LlmRequest(
        model="test-model", # Model name as string
        contents=[Content(role="user", parts=[Part.from_text("What is the capital of France?")])],
    )

@pytest.fixture
def llm_request_with_config(ollama_llm):
    """Fixture for LlmRequest with config."""
    # Correctly pass model name as string
    return LlmRequest(
        model="test-model", # Model name as string
        contents=[Content(role="user", parts=[Part.from_text("Hello")])],
        config=types.GenerateContentConfig(
            temperature=0.5,
            system_instruction="You are a test assistant.",
        ),
    )

# --- Helper Functions ---

# Helper to simulate httpx.AsyncClient.stream response
async def mock_aiter_lines(lines_data):
    """Helper to simulate streaming response lines."""
    for line in lines_data:
        yield json.dumps(line) # Simulate reading JSON lines from stream
        await asyncio.sleep(0) # Allow context switching without blocking tests

# Properly define the mock setup for streaming tests
@pytest.fixture
def mock_streaming_response(ollama_stream_response_chunks):
    """Setup proper mocking for streaming responses."""
    async def _setup_mock(mock_client):
        # Configure stream response mock
        mock_stream_response = mock.AsyncMock(spec=httpx.Response)
        mock_stream_response.raise_for_status = mock.Mock()
        
        # Prepare lines iterable
        async def _aiter_lines():
            for chunk in ollama_stream_response_chunks:
                yield json.dumps(chunk)
        
        # Set up the aiter_lines method
        mock_stream_response.aiter_lines.return_value = _aiter_lines()
        
        # Set up the context manager
        mock_stream_ctx = mock.AsyncMock()
        mock_stream_ctx.__aenter__.return_value = mock_stream_response
        mock_stream_ctx.__aexit__.return_value = None
        
        # Set the return value for the stream method
        mock_client.stream.return_value = mock_stream_ctx
    
    return _setup_mock

# --- Test Cases ---

def test_supported_models():
    """Test that OllamaLlm supports any model pattern."""
    models = OllamaLlm.supported_models()
    assert models == [".*"]

def test_ollama_client_creation(ollama_llm):
    """Test the cached property for the httpx client."""
    client1 = ollama_llm._client
    client2 = ollama_llm._client
    assert isinstance(client1, httpx.AsyncClient)
    assert client1 is client2 # Check if caching works
    assert client1.base_url == httpx.URL("http://localhost:11434")
    assert client1.timeout.read == 600.0

def test_ollama_client_creation_custom_config():
    """Test client creation with custom api_base and timeout."""
    custom_llm = OllamaLlm(model="custom", api_base="http://custom:1234", request_timeout=30.0)
    client = custom_llm._client
    assert isinstance(client, httpx.AsyncClient)
    assert client.base_url == httpx.URL("http://custom:1234")
    assert client.timeout.read == 30.0

def test_build_ollama_request_payload_basic(basic_llm_request):
    """Test payload generation for a basic request."""
    payload = _build_ollama_request_payload(basic_llm_request, "test-model", False)
    expected_payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "stream": False,
    }
    assert payload == expected_payload

def test_build_ollama_request_payload_with_config(llm_request_with_config):
    """Test payload generation with temperature and system instruction."""
    payload = _build_ollama_request_payload(llm_request_with_config, "test-model", True)
    expected_payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
        "options": {"temperature": 0.5},
        "system": "You are a test assistant.",
    }
    assert payload == expected_payload

def test_build_ollama_request_payload_with_history(ollama_llm):
    """Test payload generation with conversation history."""
    history_request = LlmRequest(
        model="test-model",
        contents=[
            Content(role="user", parts=[Part.from_text("Hi there!")]),
            Content(role="model", parts=[Part.from_text("Hello! How can I help?")]), # Note: role='model'
            Content(role="user", parts=[Part.from_text("Tell me a joke.")])
        ]
    )
    payload = _build_ollama_request_payload(history_request, "test-model", False)
    expected_payload = {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Hi there!"},
            {"role": "assistant", "content": "Hello! How can I help?"}, # Role mapped correctly
            {"role": "user", "content": "Tell me a joke."}
        ],
        "stream": False,
    }
    assert payload == expected_payload


@pytest.mark.asyncio
async def test_generate_content_async_non_stream(
    ollama_llm, basic_llm_request, ollama_non_stream_response, mock_httpx_client
):
    """Test non-streaming generation."""
    # Configure the mock client's post method
    mock_response = mock.AsyncMock(spec=httpx.Response)
    mock_response.raise_for_status = mock.Mock()
    mock_response.json.return_value = ollama_non_stream_response
    mock_httpx_client.post.return_value = mock_response

    # Call the method under test
    responses = [
        resp async for resp in ollama_llm.generate_content_async(basic_llm_request, stream=False)
    ]

    # Assertions
    assert len(responses) == 1
    response = responses[0]
    assert isinstance(response, LlmResponse)
    assert response.content.role == "model"
    assert response.content.parts[0].text == "The capital of France is Paris."
    assert not response.partial

    # Verify API call
    expected_payload = _build_ollama_request_payload(basic_llm_request, "test-model", False)
    mock_httpx_client.post.assert_called_once_with("/api/chat", json=expected_payload)

@pytest.mark.asyncio
async def test_generate_content_async_stream(
    ollama_llm, basic_llm_request, ollama_stream_response_chunks, mock_httpx_client, mock_streaming_response
):
    """Test streaming generation."""
    # Setup proper streaming mock
    await mock_streaming_response(mock_httpx_client)
    
    # Call the method under test
    responses = [
        resp async for resp in ollama_llm.generate_content_async(basic_llm_request, stream=True)
    ]

    # Assertions - only check non-empty content responses
    expected_texts = ["The", " capital", " of", " France", " is", " Paris."]
    content_responses = [r for r in responses if r.content and r.content.parts and r.content.parts[0].text]
    
    assert len(content_responses) == len(expected_texts)

    for i, response in enumerate(content_responses):
        assert isinstance(response, LlmResponse)
        assert response.content.role == "model"
        assert response.content.parts[0].text == expected_texts[i]
        assert response.partial  # All streamed chunks should be partial

    # Verify API call
    expected_payload = _build_ollama_request_payload(basic_llm_request, "test-model", True)
    mock_httpx_client.stream.assert_called_once_with('POST', "/api/chat", json=expected_payload)


@pytest.mark.asyncio
async def test_generate_content_async_ollama_error_non_stream(
    ollama_llm, basic_llm_request, ollama_error_response, mock_httpx_client
):
    """Test handling of Ollama API error during non-streaming."""
    mock_response = mock.AsyncMock(spec=httpx.Response)
    mock_response.raise_for_status = mock.Mock()
    mock_response.json.return_value = ollama_error_response
    mock_httpx_client.post.return_value = mock_response

    with pytest.raises(RuntimeError, match="Ollama API error: Model 'unknown-model' not found"):
        async for _ in ollama_llm.generate_content_async(basic_llm_request, stream=False):
            pass

@pytest.mark.asyncio
async def test_generate_content_async_ollama_error_stream(
    ollama_llm, basic_llm_request, ollama_error_response, mock_httpx_client
):
    """Test handling of Ollama API error during streaming."""
    # Setup simple error stream
    mock_stream_response = mock.AsyncMock(spec=httpx.Response)
    mock_stream_response.raise_for_status = mock.Mock()
    
    # Create async iterable for error response
    async def _error_iter():
        yield json.dumps(ollama_error_response)
    
    mock_stream_response.aiter_lines.return_value = _error_iter()
    
    # Setup context manager
    mock_stream_ctx = mock.AsyncMock()
    mock_stream_ctx.__aenter__.return_value = mock_stream_response
    mock_stream_ctx.__aexit__.return_value = None
    mock_httpx_client.stream.return_value = mock_stream_ctx

    with pytest.raises(RuntimeError, match="Ollama API error: Model 'unknown-model' not found"):
        async for _ in ollama_llm.generate_content_async(basic_llm_request, stream=True):
            pass


@pytest.mark.asyncio
async def test_generate_content_async_http_error_non_stream(
    ollama_llm, basic_llm_request, mock_httpx_client
):
    """Test handling of HTTP status errors during non-streaming."""
    mock_response = mock.AsyncMock(spec=httpx.Response)
    # Simulate a 404 Not Found error
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Not Found", request=mock.Mock(), response=mock.Mock(status_code=404)
    )
    mock_httpx_client.post.return_value = mock_response

    with pytest.raises(httpx.HTTPStatusError):
        async for _ in ollama_llm.generate_content_async(basic_llm_request, stream=False):
            pass


@pytest.mark.asyncio
async def test_generate_content_async_http_error_stream(
    ollama_llm, basic_llm_request, mock_httpx_client
):
    """Test handling of HTTP status errors during streaming."""
    mock_stream_response = mock.AsyncMock(spec=httpx.Response)
    # Simulate a 404 Not Found error when entering the stream context
    mock_stream_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Not Found", request=mock.Mock(), response=mock.Mock(status_code=404)
    )
    mock_stream_context = mock.AsyncMock()
    mock_stream_context.__aenter__.return_value = mock_stream_response
    mock_stream_context.__aexit__.return_value = None
    mock_httpx_client.stream.return_value = mock_stream_context

    with pytest.raises(httpx.HTTPStatusError):
        async for _ in ollama_llm.generate_content_async(basic_llm_request, stream=True):
            pass


@pytest.mark.asyncio
async def test_generate_content_async_request_error(
    ollama_llm, basic_llm_request, mock_httpx_client
):
    """Test handling of httpx request errors (e.g., connection error)."""
    mock_httpx_client.post.side_effect = httpx.RequestError("Connection failed", request=mock.Mock())

    with pytest.raises(RuntimeError, match="Failed to connect to Ollama API"):
        async for _ in ollama_llm.generate_content_async(basic_llm_request, stream=False):
            pass