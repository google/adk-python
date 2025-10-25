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

"""Tests for LiteLlm model with include_contents='none'."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.google.adk.models.lite_llm import LiteLlm
from src.google.adk.models.llm_request import LlmRequest
from google.genai import types


@pytest.mark.asyncio
async def test_include_contents_none_with_fallback():
    """Test that LiteLlm handles include_contents='none' without empty content error."""
    
    # Create a minimal LlmRequest with no contents
    config = types.GenerateContentConfig(
        system_instruction="Continue the phrase of the last agent with a short sentence"
    )
    
    llm_request = LlmRequest(
        contents=[],  # Empty contents simulating include_contents='none'
        config=config
    )
    
    # Mock the LiteLLM client to avoid actual API calls
    mock_response = MagicMock()
    mock_response.get.return_value = [{
        "message": {
            "content": "This is a test response."
        },
        "finish_reason": "stop"
    }]
    mock_response.__getitem__ = lambda self, key: {
        "choices": [{
            "message": {
                "content": "This is a test response."
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }[key]
    
    # Initialize LiteLlm model
    model = LiteLlm(model="gemini/gemini-2.0-flash")
    
    # Mock the acompletion method
    with patch.object(model.llm_client, 'acompletion', new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_response
        
        # This should not raise an error about empty content
        # Instead, it should add fallback content
        response_generator = model.generate_content_async(llm_request, stream=False)
        
        # Verify we can get a response without error
        response = None
        async for resp in response_generator:
            response = resp
            break
        
        # Assert response is not None and has expected structure
        assert response is not None
        assert response.content is not None
        assert response.content.role == "model"
        
        # Verify that acompletion was called with non-empty messages
        call_args = mock_acompletion.call_args
        messages = call_args.kwargs.get('messages', [])
        assert len(messages) > 0, "Messages should not be empty"
        
        # Verify the fallback message is present
        user_messages = [m for m in messages if m.get('role') == 'user']
        assert len(user_messages) > 0, "Should have at least one user message"
        assert "Handle the requests as specified in the System Instruction" in str(user_messages[0].get('content', ''))


@pytest.mark.asyncio
async def test_include_contents_none_with_tools():
    """Test that LiteLlm handles include_contents='none' with tools."""
    
    # Create a function declaration
    function_decl = types.FunctionDeclaration(
        name="get_weather",
        description="Get weather for a city",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "city": types.Schema(type=types.Type.STRING, description="City name")
            },
            required=["city"]
        )
    )
    
    config = types.GenerateContentConfig(
        system_instruction="You are a helpful assistant",
        tools=[types.Tool(function_declarations=[function_decl])]
    )
    
    llm_request = LlmRequest(
        contents=[],  # Empty contents
        config=config
    )
    
    # Mock response with tool call - use proper ChatCompletionMessageToolCall objects
    from litellm import ChatCompletionMessageToolCall, Function
    
    mock_tool_call = ChatCompletionMessageToolCall(
        type="function",
        id="call_123",
        function=Function(
            name="get_weather",
            arguments='{"city": "New York"}'
        )
    )
    
    mock_response = MagicMock()
    mock_response.__getitem__ = lambda self, key: {
        "choices": [{
            "message": {
                "content": None,
                "tool_calls": [mock_tool_call]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 10,
            "total_tokens": 25
        }
    }[key]
    
    model = LiteLlm(model="gemini/gemini-2.0-flash")
    
    # Mock the acompletion method
    with patch.object(model.llm_client, 'acompletion', new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_response
        
        # Should handle empty contents gracefully
        response_generator = model.generate_content_async(llm_request, stream=False)
        
        response = None
        async for resp in response_generator:
            response = resp
            break
        
        assert response is not None
        assert response.content is not None
        
        # Verify that acompletion was called with non-empty messages
        call_args = mock_acompletion.call_args
        messages = call_args.kwargs.get('messages', [])
        assert len(messages) > 0, "Messages should not be empty with tools"
        
        # Verify tools were passed
        tools = call_args.kwargs.get('tools', None)
        assert tools is not None, "Tools should be passed to acompletion"
        assert len(tools) > 0, "Should have at least one tool"


@pytest.mark.asyncio
async def test_include_contents_with_existing_content():
    """Test that LiteLlm works normally when contents are provided."""
    
    config = types.GenerateContentConfig(
        system_instruction="You are a helpful assistant"
    )
    
    # Provide actual content
    llm_request = LlmRequest(
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text="What is the weather in Paris?")]
            )
        ],
        config=config
    )
    
    mock_response = MagicMock()
    mock_response.__getitem__ = lambda self, key: {
        "choices": [{
            "message": {
                "content": "The weather in Paris is sunny."
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 8,
            "total_tokens": 28
        }
    }[key]
    
    model = LiteLlm(model="gemini/gemini-2.0-flash")
    
    with patch.object(model.llm_client, 'acompletion', new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_response
        
        response_generator = model.generate_content_async(llm_request, stream=False)
        
        response = None
        async for resp in response_generator:
            response = resp
            break
        
        assert response is not None
        assert response.content is not None
        assert response.content.role == "model"
        
        # Verify that user's actual content was used
        call_args = mock_acompletion.call_args
        messages = call_args.kwargs.get('messages', [])
        user_messages = [m for m in messages if m.get('role') == 'user']
        assert any("Paris" in str(m.get('content', '')) for m in user_messages)
