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

import base64
import json
import os
import sys
from unittest import mock
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from anthropic import types as anthropic_types
from google.adk import version as adk_version
from google.adk.models import anthropic_llm
from google.adk.models.anthropic_llm import AnthropicLlm
from google.adk.models.anthropic_llm import Claude
from google.adk.models.anthropic_llm import content_to_message_param
from google.adk.models.anthropic_llm import function_declaration_to_tool_param
from google.adk.models.anthropic_llm import part_to_message_block
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


def test_claude_anthropic_client_creation():
  # Test with environment variables
  with mock.patch.dict(
      os.environ,
      {
          "GOOGLE_CLOUD_PROJECT": "env-project",
          "GOOGLE_CLOUD_LOCATION": "env-location",
      },
  ):
    model = Claude(model="claude-3-5-sonnet-v2@20241022")
    with mock.patch(
        "google.adk.models.anthropic_llm.AsyncAnthropicVertex", autospec=True
    ) as mock_client_class:
      _ = model._anthropic_client
      mock_client_class.assert_called_once()
      _, kwargs = mock_client_class.call_args
      assert kwargs["project_id"] == "env-project"
      assert kwargs["region"] == "env-location"


def test_claude_anthropic_client_creation_with_full_resource_name():
  # Test with full resource name in model string
  model = Claude(
      model="projects/test-project/locations/test-location/publishers/anthropic/models/claude-3-5-sonnet-v2@20241022"
  )
  with mock.patch(
      "google.adk.models.anthropic_llm.AsyncAnthropicVertex", autospec=True
  ) as mock_client_class:
    _ = model._anthropic_client
    mock_client_class.assert_called_once()
    _, kwargs = mock_client_class.call_args
    assert kwargs["project_id"] == "test-project"
    assert kwargs["region"] == "test-location"


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
        "function_with_nested_object_parameter",
        types.FunctionDeclaration(
            name="update_profile",
            description="Updates a user profile.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "profile": types.Schema(
                        type=types.Type.OBJECT,
                        description="The profile data",
                        properties={
                            "name": types.Schema(
                                type=types.Type.STRING,
                                description="Full name",
                            ),
                            "address": types.Schema(
                                type=types.Type.OBJECT,
                                description="Mailing address",
                                properties={
                                    "city": types.Schema(
                                        type=types.Type.STRING,
                                    ),
                                    "state": types.Schema(
                                        type=types.Type.STRING,
                                    ),
                                },
                            ),
                        },
                    ),
                },
                required=["profile"],
            ),
        ),
        anthropic_types.ToolParam(
            name="update_profile",
            description="Updates a user profile.",
            input_schema={
                "type": "object",
                "properties": {
                    "profile": {
                        "type": "object",
                        "description": "The profile data",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Full name",
                            },
                            "address": {
                                "type": "object",
                                "description": "Mailing address",
                                "properties": {
                                    "city": {"type": "string"},
                                    "state": {"type": "string"},
                                },
                            },
                        },
                    },
                },
                "required": ["profile"],
            },
        ),
    ),
    (
        "function_with_any_of_parameter",
        types.FunctionDeclaration(
            name="set_value",
            description="Sets a value that can be a string or integer.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "value": types.Schema(
                        description="A string or integer value",
                        any_of=[
                            types.Schema(type=types.Type.STRING),
                            types.Schema(type=types.Type.INTEGER),
                        ],
                    ),
                },
                required=["value"],
            ),
        ),
        anthropic_types.ToolParam(
            name="set_value",
            description="Sets a value that can be a string or integer.",
            input_schema={
                "type": "object",
                "properties": {
                    "value": {
                        "description": "A string or integer value",
                        "anyOf": [
                            {"type": "string"},
                            {"type": "integer"},
                        ],
                    },
                },
                "required": ["value"],
            },
        ),
    ),
    (
        "function_with_additional_properties_parameter",
        types.FunctionDeclaration(
            name="store_metadata",
            description="Stores arbitrary key-value metadata.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "metadata": types.Schema(
                        type=types.Type.OBJECT,
                        description="Arbitrary metadata",
                        additional_properties=types.Schema(
                            type=types.Type.STRING,
                        ),
                    ),
                },
                required=["metadata"],
            ),
        ),
        anthropic_types.ToolParam(
            name="store_metadata",
            description="Stores arbitrary key-value metadata.",
            input_schema={
                "type": "object",
                "properties": {
                    "metadata": {
                        "type": "object",
                        "description": "Arbitrary metadata",
                        "additionalProperties": {"type": "string"},
                    },
                },
                "required": ["metadata"],
            },
        ),
    ),
    (
        "function_with_parameters_json_schema_combinators",
        types.FunctionDeclaration(
            name="validate_payload",
            description="Validates a payload with schema combinators.",
            parameters_json_schema={
                "type": "OBJECT",
                "properties": {
                    "choice": {
                        "oneOf": [
                            {"type": "STRING"},
                            {"type": "INTEGER"},
                        ],
                    },
                    "config": {
                        "allOf": [
                            {
                                "type": "OBJECT",
                                "properties": {
                                    "enabled": {"type": "BOOLEAN"},
                                },
                            },
                        ],
                    },
                    "blocked": {
                        "not": {
                            "type": "NULL",
                        },
                    },
                    "tuple_value": {
                        "type": "ARRAY",
                        "items": [
                            {"type": "STRING"},
                            {"type": "INTEGER"},
                        ],
                    },
                },
                "required": ["choice"],
            },
        ),
        anthropic_types.ToolParam(
            name="validate_payload",
            description="Validates a payload with schema combinators.",
            input_schema={
                "type": "object",
                "properties": {
                    "choice": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "integer"},
                        ],
                    },
                    "config": {
                        "allOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "enabled": {"type": "boolean"},
                                },
                            },
                        ],
                    },
                    "blocked": {
                        "not": {
                            "type": "null",
                        },
                    },
                    "tuple_value": {
                        "type": "array",
                        "items": [
                            {"type": "string"},
                            {"type": "integer"},
                        ],
                    },
                },
                "required": ["choice"],
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


def test_part_to_message_block_with_pdf_document():
  """Test that part_to_message_block handles PDF document parts."""
  pdf_data = b"%PDF-1.4 fake pdf content"
  part = Part(
      inline_data=types.Blob(mime_type="application/pdf", data=pdf_data)
  )

  result = part_to_message_block(part)

  assert isinstance(result, dict)
  assert result["type"] == "document"
  assert result["source"]["type"] == "base64"
  assert result["source"]["media_type"] == "application/pdf"
  assert result["source"]["data"] == base64.b64encode(pdf_data).decode()


def test_part_to_message_block_with_pdf_mime_type_parameters():
  """Test that PDF parts with MIME type parameters are handled correctly."""
  pdf_data = b"%PDF-1.4 fake pdf content"
  part = Part(
      inline_data=types.Blob(
          mime_type="application/pdf; name=doc.pdf", data=pdf_data
      )
  )

  result = part_to_message_block(part)

  assert isinstance(result, dict)
  assert result["type"] == "document"
  assert result["source"]["type"] == "base64"
  assert result["source"]["media_type"] == "application/pdf; name=doc.pdf"
  assert result["source"]["data"] == base64.b64encode(pdf_data).decode()


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
        None,  # No warning expected
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
        "Image data is not supported in Claude for assistant turns.",
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
        "Image data is not supported in Claude for assistant turns.",
    ),
    (
        "user_role_with_text_and_document",
        Content(
            role="user",
            parts=[
                Part.from_text(text="Summarize this document."),
                Part(
                    inline_data=types.Blob(
                        mime_type="application/pdf", data=b"fake_pdf_data"
                    )
                ),
            ],
        ),
        "user",
        2,  # Both text and document included
        None,  # No warning expected
    ),
    (
        "model_role_with_text_and_document",
        Content(
            role="model",
            parts=[
                Part.from_text(text="Here is the summary."),
                Part(
                    inline_data=types.Blob(
                        mime_type="application/pdf", data=b"fake_pdf_data"
                    )
                ),
            ],
        ),
        "assistant",
        1,  # Document filtered out, only text remains
        "PDF data is not supported in Claude for assistant turns.",
    ),
]


@pytest.mark.parametrize(
    "_, content, expected_role, expected_content_length, expected_warning",
    content_to_message_param_test_cases,
    ids=[case[0] for case in content_to_message_param_test_cases],
)
def test_content_to_message_param(
    _, content, expected_role, expected_content_length, expected_warning
):
  """Test content_to_message_param handles images and documents based on role."""
  with mock.patch("google.adk.models.anthropic_llm.logger") as mock_logger:
    result = content_to_message_param(content)

    assert result["role"] == expected_role
    assert len(result["content"]) == expected_content_length

    if expected_warning:
      mock_logger.warning.assert_called_once_with(expected_warning)
    else:
      mock_logger.warning.assert_not_called()


# --- Tests for Bug #2: json.dumps for dict/list function results ---


def test_part_to_message_block_dict_result_serialized_as_json():
  """Dict results should be serialized with json.dumps, not str()."""
  response_part = types.Part.from_function_response(
      name="get_topic",
      response={"result": {"topic": "travel", "active": True, "count": None}},
  )
  response_part.function_response.id = "test_id"

  result = part_to_message_block(response_part)
  content = result["content"]

  # Must be valid JSON (json.dumps produces "true"/"null", not "True"/"None")
  parsed = json.loads(content)
  assert parsed["topic"] == "travel"
  assert parsed["active"] is True
  assert parsed["count"] is None


def test_part_to_message_block_list_result_serialized_as_json():
  """List results should be serialized with json.dumps."""
  response_part = types.Part.from_function_response(
      name="get_items",
      response={"result": ["item1", "item2", "item3"]},
  )
  response_part.function_response.id = "test_id"

  result = part_to_message_block(response_part)
  content = result["content"]

  parsed = json.loads(content)
  assert parsed == ["item1", "item2", "item3"]


def test_part_to_message_block_empty_dict_result_not_dropped():
  """Empty dict results should produce '{}', not empty string."""
  response_part = types.Part.from_function_response(
      name="some_tool",
      response={"result": {}},
  )
  response_part.function_response.id = "test_id"

  result = part_to_message_block(response_part)
  assert result["content"] == "{}"


def test_part_to_message_block_empty_list_result_not_dropped():
  """Empty list results should produce '[]', not empty string."""
  response_part = types.Part.from_function_response(
      name="some_tool",
      response={"result": []},
  )
  response_part.function_response.id = "test_id"

  result = part_to_message_block(response_part)
  assert result["content"] == "[]"


def test_part_to_message_block_string_result_unchanged():
  """String results should still work as before (backward compat)."""
  response_part = types.Part.from_function_response(
      name="simple_tool",
      response={"result": "plain text result"},
  )
  response_part.function_response.id = "test_id"

  result = part_to_message_block(response_part)
  assert result["content"] == "plain text result"


def test_part_to_message_block_nested_dict_result():
  """Nested dict with arrays should produce valid JSON."""
  response_part = types.Part.from_function_response(
      name="search",
      response={
          "result": {
              "results": [
                  {"id": 1, "tags": ["a", "b"]},
                  {"id": 2, "meta": {"key": "val"}},
              ],
              "has_more": False,
          }
      },
  )
  response_part.function_response.id = "test_id"

  result = part_to_message_block(response_part)
  parsed = json.loads(result["content"])
  assert parsed["has_more"] is False
  assert parsed["results"][0]["tags"] == ["a", "b"]


# --- Tests for arbitrary dict fallback (e.g. SkillToolset load_skill) ---


def test_part_to_message_block_arbitrary_dict_serialized_as_json():
  """Dicts with keys other than 'content'/'result' should be JSON-serialized.

  This covers tools like load_skill that return arbitrary key structures
  such as {"skill_name": ..., "instructions": ..., "frontmatter": ...}.
  """
  response_part = types.Part.from_function_response(
      name="load_skill",
      response={
          "skill_name": "my_skill",
          "instructions": "Step 1: do this. Step 2: do that.",
          "frontmatter": {"version": "1.0", "tags": ["a", "b"]},
      },
  )
  response_part.function_response.id = "test_id"

  result = part_to_message_block(response_part)

  assert result["type"] == "tool_result"
  assert result["tool_use_id"] == "test_id"
  assert not result["is_error"]
  parsed = json.loads(result["content"])
  assert parsed["skill_name"] == "my_skill"
  assert parsed["instructions"] == "Step 1: do this. Step 2: do that."
  assert parsed["frontmatter"]["version"] == "1.0"


def test_part_to_message_block_run_skill_script_response():
  """run_skill_script response keys (stdout/stderr/status) should not be dropped."""
  response_part = types.Part.from_function_response(
      name="run_skill_script",
      response={
          "skill_name": "my_skill",
          "file_path": "scripts/setup.py",
          "stdout": "Done.",
          "stderr": "",
          "status": "success",
      },
  )
  response_part.function_response.id = "test_id_2"

  result = part_to_message_block(response_part)

  parsed = json.loads(result["content"])
  assert parsed["status"] == "success"
  assert parsed["stdout"] == "Done."


def test_part_to_message_block_error_response_not_dropped():
  """Error dicts like {"error": ..., "error_code": ...} should be serialized."""
  response_part = types.Part.from_function_response(
      name="load_skill",
      response={
          "error": "Skill 'missing' not found.",
          "error_code": "SKILL_NOT_FOUND",
      },
  )
  response_part.function_response.id = "test_id_3"

  result = part_to_message_block(response_part)

  parsed = json.loads(result["content"])
  assert parsed["error_code"] == "SKILL_NOT_FOUND"


def test_part_to_message_block_empty_response_stays_empty():
  """An empty response dict should still produce an empty content string."""
  response_part = types.Part.from_function_response(
      name="some_tool",
      response={},
  )
  response_part.function_response.id = "test_id_4"

  result = part_to_message_block(response_part)

  assert result["content"] == ""


# --- Tests for Bug #1: Streaming support ---


def _make_mock_stream_events(events):
  """Helper to create an async iterable from a list of events."""

  async def _stream():
    for event in events:
      yield event

  return _stream()


@pytest.mark.asyncio
async def test_streaming_text_yields_partial_and_final():
  """Streaming text should yield partial chunks then a final response."""
  llm = AnthropicLlm(model="claude-sonnet-4-20250514")

  events = [
      MagicMock(
          type="message_start",
          message=MagicMock(usage=MagicMock(input_tokens=10, output_tokens=0)),
      ),
      MagicMock(
          type="content_block_start",
          index=0,
          content_block=anthropic_types.TextBlock(text="", type="text"),
      ),
      MagicMock(
          type="content_block_delta",
          index=0,
          delta=anthropic_types.TextDelta(text="Hello ", type="text_delta"),
      ),
      MagicMock(
          type="content_block_delta",
          index=0,
          delta=anthropic_types.TextDelta(text="world!", type="text_delta"),
      ),
      MagicMock(type="content_block_stop", index=0),
      MagicMock(
          type="message_delta",
          delta=MagicMock(stop_reason="end_turn"),
          usage=MagicMock(output_tokens=5),
      ),
      MagicMock(type="message_stop"),
  ]

  mock_client = MagicMock()
  mock_client.messages.create = AsyncMock(
      return_value=_make_mock_stream_events(events)
  )

  llm_request = LlmRequest(
      model="claude-sonnet-4-20250514",
      contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
      config=types.GenerateContentConfig(
          system_instruction="You are helpful",
      ),
  )

  with mock.patch.object(llm, "_anthropic_client", mock_client):
    responses = [
        r async for r in llm.generate_content_async(llm_request, stream=True)
    ]

  # 2 partial text chunks + 1 final aggregated
  assert len(responses) == 3
  assert responses[0].partial is True
  assert responses[0].content.parts[0].text == "Hello "
  assert responses[1].partial is True
  assert responses[1].content.parts[0].text == "world!"
  assert responses[2].partial is False
  assert responses[2].content.parts[0].text == "Hello world!"
  assert responses[2].usage_metadata.prompt_token_count == 10
  assert responses[2].usage_metadata.candidates_token_count == 5


@pytest.mark.asyncio
async def test_streaming_tool_use_yields_function_call():
  """Streaming tool_use should accumulate args and yield in final."""
  llm = AnthropicLlm(model="claude-sonnet-4-20250514")

  events = [
      MagicMock(
          type="message_start",
          message=MagicMock(usage=MagicMock(input_tokens=20, output_tokens=0)),
      ),
      MagicMock(
          type="content_block_start",
          index=0,
          content_block=anthropic_types.TextBlock(text="", type="text"),
      ),
      MagicMock(
          type="content_block_delta",
          index=0,
          delta=anthropic_types.TextDelta(text="Checking.", type="text_delta"),
      ),
      MagicMock(type="content_block_stop", index=0),
      MagicMock(
          type="content_block_start",
          index=1,
          content_block=anthropic_types.ToolUseBlock(
              id="toolu_abc",
              name="get_weather",
              input={},
              type="tool_use",
          ),
      ),
      MagicMock(
          type="content_block_delta",
          index=1,
          delta=anthropic_types.InputJSONDelta(
              partial_json='{"city": "Paris"}',
              type="input_json_delta",
          ),
      ),
      MagicMock(type="content_block_stop", index=1),
      MagicMock(
          type="message_delta",
          delta=MagicMock(stop_reason="tool_use"),
          usage=MagicMock(output_tokens=12),
      ),
      MagicMock(type="message_stop"),
  ]

  mock_client = MagicMock()
  mock_client.messages.create = AsyncMock(
      return_value=_make_mock_stream_events(events)
  )

  llm_request = LlmRequest(
      model="claude-sonnet-4-20250514",
      contents=[
          Content(
              role="user",
              parts=[Part.from_text(text="Weather?")],
          )
      ],
      config=types.GenerateContentConfig(
          system_instruction="You are helpful",
      ),
  )

  with mock.patch.object(llm, "_anthropic_client", mock_client):
    responses = [
        r async for r in llm.generate_content_async(llm_request, stream=True)
    ]

  # 1 text partial + 1 final
  assert len(responses) == 2

  final = responses[-1]
  assert final.partial is False
  assert len(final.content.parts) == 2
  assert final.content.parts[0].text == "Checking."
  assert final.content.parts[1].function_call.name == "get_weather"
  assert final.content.parts[1].function_call.args == {"city": "Paris"}
  assert final.content.parts[1].function_call.id == "toolu_abc"


@pytest.mark.asyncio
async def test_streaming_passes_stream_true_to_create():
  """When stream=True, messages.create should be called with stream=True."""
  llm = AnthropicLlm(model="claude-sonnet-4-20250514")

  events = [
      MagicMock(
          type="message_start",
          message=MagicMock(usage=MagicMock(input_tokens=5, output_tokens=0)),
      ),
      MagicMock(
          type="content_block_start",
          index=0,
          content_block=anthropic_types.TextBlock(text="", type="text"),
      ),
      MagicMock(
          type="content_block_delta",
          index=0,
          delta=anthropic_types.TextDelta(text="Hi", type="text_delta"),
      ),
      MagicMock(type="content_block_stop", index=0),
      MagicMock(
          type="message_delta",
          delta=MagicMock(stop_reason="end_turn"),
          usage=MagicMock(output_tokens=1),
      ),
      MagicMock(type="message_stop"),
  ]

  mock_client = MagicMock()
  mock_client.messages.create = AsyncMock(
      return_value=_make_mock_stream_events(events)
  )

  llm_request = LlmRequest(
      model="claude-sonnet-4-20250514",
      contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
      config=types.GenerateContentConfig(
          system_instruction="Test",
      ),
  )

  with mock.patch.object(llm, "_anthropic_client", mock_client):
    _ = [r async for r in llm.generate_content_async(llm_request, stream=True)]

  mock_client.messages.create.assert_called_once()
  _, kwargs = mock_client.messages.create.call_args
  assert kwargs["stream"] is True


@pytest.mark.asyncio
async def test_non_streaming_does_not_pass_stream_param():
  """When stream=False, messages.create should NOT get stream param."""
  llm = AnthropicLlm(model="claude-sonnet-4-20250514")

  mock_message = anthropic_types.Message(
      id="msg_test",
      content=[
          anthropic_types.TextBlock(text="Hello!", type="text", citations=None)
      ],
      model="claude-sonnet-4-20250514",
      role="assistant",
      stop_reason="end_turn",
      stop_sequence=None,
      type="message",
      usage=anthropic_types.Usage(
          input_tokens=5,
          output_tokens=2,
          cache_creation_input_tokens=0,
          cache_read_input_tokens=0,
          server_tool_use=None,
          service_tier=None,
      ),
  )

  mock_client = MagicMock()
  mock_client.messages.create = AsyncMock(return_value=mock_message)

  llm_request = LlmRequest(
      model="claude-sonnet-4-20250514",
      contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
      config=types.GenerateContentConfig(
          system_instruction="Test",
      ),
  )

  with mock.patch.object(llm, "_anthropic_client", mock_client):
    responses = [
        r async for r in llm.generate_content_async(llm_request, stream=False)
    ]

  assert len(responses) == 1
  mock_client.messages.create.assert_called_once()
  _, kwargs = mock_client.messages.create.call_args
  assert "stream" not in kwargs
