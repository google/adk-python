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

import os
import sys
from unittest import mock

from anthropic import types as anthropic_types
from google.adk import version as adk_version
from google.adk.models import anthropic_llm
from google.adk.models.anthropic_llm import AnthropicLlm
from google.adk.models.anthropic_llm import Claude
from google.adk.models.anthropic_llm import content_to_message_param
from google.adk.models.anthropic_llm import function_declaration_to_tool_param
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


function_declaration_test_cases = [
    (
        "function_with_no_parameters",
        types.FunctionDeclaration(
            name="get_current_time",
            description="Gets the current time.",
        ),
        anthropic_types.ToolParam(
            name="get_current_time",
            description="Gets the current time.",
            input_schema={"type": "object", "properties": {}},
        ),
    ),
    (
        "function_with_one_optional_parameter",
        types.FunctionDeclaration(
            name="get_weather",
            description="Gets weather information for a given location.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "location": types.Schema(
                        type=types.Type.STRING,
                        description="City and state, e.g., San Francisco, CA",
                    )
                },
            ),
        ),
        anthropic_types.ToolParam(
            name="get_weather",
            description="Gets weather information for a given location.",
            input_schema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": (
                            "City and state, e.g., San Francisco, CA"
                        ),
                    }
                },
            },
        ),
    ),
    (
        "function_with_one_required_parameter",
        types.FunctionDeclaration(
            name="get_stock_price",
            description="Gets the current price for a stock ticker.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "ticker": types.Schema(
                        type=types.Type.STRING,
                        description="The stock ticker, e.g., AAPL",
                    )
                },
                required=["ticker"],
            ),
        ),
        anthropic_types.ToolParam(
            name="get_stock_price",
            description="Gets the current price for a stock ticker.",
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker, e.g., AAPL",
                    }
                },
                "required": ["ticker"],
            },
        ),
    ),
    (
        "function_with_multiple_mixed_parameters",
        types.FunctionDeclaration(
            name="submit_order",
            description="Submits a product order.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "product_id": types.Schema(
                        type=types.Type.STRING, description="The product ID"
                    ),
                    "quantity": types.Schema(
                        type=types.Type.INTEGER,
                        description="The order quantity",
                    ),
                    "notes": types.Schema(
                        type=types.Type.STRING,
                        description="Optional order notes",
                    ),
                },
                required=["product_id", "quantity"],
            ),
        ),
        anthropic_types.ToolParam(
            name="submit_order",
            description="Submits a product order.",
            input_schema={
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The product ID",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "The order quantity",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Optional order notes",
                    },
                },
                "required": ["product_id", "quantity"],
            },
        ),
    ),
    (
        "function_with_complex_nested_parameter",
        types.FunctionDeclaration(
            name="create_playlist",
            description="Creates a playlist from a list of songs.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "playlist_name": types.Schema(
                        type=types.Type.STRING,
                        description="The name for the new playlist",
                    ),
                    "songs": types.Schema(
                        type=types.Type.ARRAY,
                        description="A list of songs to add to the playlist",
                        items=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "title": types.Schema(type=types.Type.STRING),
                                "artist": types.Schema(type=types.Type.STRING),
                            },
                            required=["title", "artist"],
                        ),
                    ),
                },
                required=["playlist_name", "songs"],
            ),
        ),
        anthropic_types.ToolParam(
            name="create_playlist",
            description="Creates a playlist from a list of songs.",
            input_schema={
                "type": "object",
                "properties": {
                    "playlist_name": {
                        "type": "string",
                        "description": "The name for the new playlist",
                    },
                    "songs": {
                        "type": "array",
                        "description": "A list of songs to add to the playlist",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "artist": {"type": "string"},
                            },
                            "required": ["title", "artist"],
                        },
                    },
                },
                "required": ["playlist_name", "songs"],
            },
        ),
    ),
    (
        "function_with_parameters_json_schema",
        types.FunctionDeclaration(
            name="search_database",
            description="Searches a database with given criteria.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                    },
                },
                "required": ["query"],
            },
        ),
        anthropic_types.ToolParam(
            name="search_database",
            description="Searches a database with given criteria.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                    },
                },
                "required": ["query"],
            },
        ),
    ),
]


@pytest.mark.parametrize(
    "_, function_declaration, expected_tool_param",
    function_declaration_test_cases,
    ids=[case[0] for case in function_declaration_test_cases],
)
async def test_function_declaration_to_tool_param(
    _, function_declaration, expected_tool_param
):
  """Test function_declaration_to_tool_param."""
  assert (
      function_declaration_to_tool_param(function_declaration)
      == expected_tool_param
  )


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
async def test_anthropic_llm_generate_content_async(
    llm_request, generate_content_response, generate_llm_response
):
  anthropic_llm_instance = AnthropicLlm(model="claude-sonnet-4-20250514")
  with mock.patch.object(
      anthropic_llm_instance, "_anthropic_client"
  ) as mock_client:
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
          async for resp in anthropic_llm_instance.generate_content_async(
              llm_request, stream=False
          )
      ]
      assert len(responses) == 1
      assert isinstance(responses[0], LlmResponse)
      assert responses[0].content.parts[0].text == "Hello, how can I help you?"


def test_claude_vertex_client_uses_tracking_headers():
  """Tests that Claude vertex client is called with tracking headers."""
  with mock.patch.object(
      anthropic_llm, "AsyncAnthropicVertex", autospec=True
  ) as mock_anthropic_vertex:
    with mock.patch.dict(
        os.environ,
        {
            "GOOGLE_CLOUD_PROJECT": "test-project",
            "GOOGLE_CLOUD_LOCATION": "us-central1",
        },
    ):
      instance = Claude(model="claude-3-5-sonnet-v2@20241022")
      _ = instance._anthropic_client
      mock_anthropic_vertex.assert_called_once()
      _, kwargs = mock_anthropic_vertex.call_args
      assert "default_headers" in kwargs
      assert "x-goog-api-client" in kwargs["default_headers"]
      assert "user-agent" in kwargs["default_headers"]
      assert (
          f"google-adk/{adk_version.__version__}"
          in kwargs["default_headers"]["user-agent"]
      )


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


def test_part_to_message_block_with_content():
  """Test that part_to_message_block handles content format."""
  from google.adk.models.anthropic_llm import part_to_message_block

  # Create a function response part with content array.
  mcp_response_part = types.Part.from_function_response(
      name="generate_sample_filesystem",
      response={
          "content": [{
              "type": "text",
              "text": '{"name":"root","node_type":"folder","children":[]}',
          }]
      },
  )
  mcp_response_part.function_response.id = "test_id_123"

  result = part_to_message_block(mcp_response_part)

  # ToolResultBlockParam is a TypedDict.
  assert isinstance(result, dict)
  assert result["tool_use_id"] == "test_id_123"
  assert result["type"] == "tool_result"
  assert not result["is_error"]
  # Verify the content was extracted from the content format.
  assert (
      '{"name":"root","node_type":"folder","children":[]}' in result["content"]
  )


def test_part_to_message_block_with_traditional_result():
  """Test that part_to_message_block handles traditional result format."""
  from google.adk.models.anthropic_llm import part_to_message_block

  # Create a function response part with traditional result format
  traditional_response_part = types.Part.from_function_response(
      name="some_tool",
      response={
          "result": "This is the result from the tool",
      },
  )
  traditional_response_part.function_response.id = "test_id_456"

  result = part_to_message_block(traditional_response_part)

  # ToolResultBlockParam is a TypedDict.
  assert isinstance(result, dict)
  assert result["tool_use_id"] == "test_id_456"
  assert result["type"] == "tool_result"
  assert not result["is_error"]
  # Verify the content was extracted from the traditional format
  assert "This is the result from the tool" in result["content"]


def test_part_to_message_block_with_multiple_content_items():
  """Test content with multiple items."""
  from google.adk.models.anthropic_llm import part_to_message_block

  # Create a function response with multiple content items
  multi_content_part = types.Part.from_function_response(
      name="multi_response_tool",
      response={
          "content": [
              {"type": "text", "text": "First part"},
              {"type": "text", "text": "Second part"},
          ]
      },
  )
  multi_content_part.function_response.id = "test_id_789"

  result = part_to_message_block(multi_content_part)

  # ToolResultBlockParam is a TypedDict.
  assert isinstance(result, dict)
  # Multiple text items should be joined with newlines
  assert result["content"] == "First part\nSecond part"


content_to_message_param_test_cases = [
    (
        "user_role_with_text_and_image",
        Content(
            role="user",
            parts=[
                Part.from_text(text="What's in this image?"),
                Part(
                    inline_data=types.Blob(
                        mime_type="image/jpeg", data=b"fake_image_data"
                    )
                ),
            ],
        ),
        "user",
        2,  # Expected content length
        False,  # Should not log warning
    ),
    (
        "model_role_with_text_and_image",
        Content(
            role="model",
            parts=[
                Part.from_text(text="I see a cat."),
                Part(
                    inline_data=types.Blob(
                        mime_type="image/png", data=b"fake_image_data"
                    )
                ),
            ],
        ),
        "assistant",
        1,  # Image filtered out, only text remains
        True,  # Should log warning
    ),
    (
        "assistant_role_with_text_and_image",
        Content(
            role="assistant",
            parts=[
                Part.from_text(text="Here's what I found."),
                Part(
                    inline_data=types.Blob(
                        mime_type="image/webp", data=b"fake_image_data"
                    )
                ),
            ],
        ),
        "assistant",
        1,  # Image filtered out, only text remains
        True,  # Should log warning
    ),
]


@pytest.mark.parametrize(
    "_, content, expected_role, expected_content_length, should_log_warning",
    content_to_message_param_test_cases,
    ids=[case[0] for case in content_to_message_param_test_cases],
)
def test_content_to_message_param_with_images(
    _, content, expected_role, expected_content_length, should_log_warning
):
  """Test content_to_message_param handles images correctly based on role."""
  with mock.patch("google.adk.models.anthropic_llm.logger") as mock_logger:
    result = content_to_message_param(content)

    assert result["role"] == expected_role
    assert len(result["content"]) == expected_content_length

    if should_log_warning:
      mock_logger.warning.assert_called_once_with(
          "Image data is not supported in Claude for assistant turns."
      )
    else:
      mock_logger.warning.assert_not_called()
