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

import json
import os
from unittest import mock

from google.adk.labs.openai._openai_llm import _function_declaration_to_openai_tool
from google.adk.labs.openai._openai_llm import _part_to_openai_content
from google.adk.labs.openai._openai_llm import _update_type_string
from google.adk.labs.openai._openai_llm import OpenAILlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from google.genai.types import Content
from google.genai.types import Part
import pytest


def test_supported_models():
  models = OpenAILlm.supported_models()
  assert len(models) == 3
  assert models[0] == r"gpt-.*"
  assert models[1] == r"o1-.*"
  assert models[2] == r"o3-.*"


def test_update_type_string():
  schema = {
      "type": "OBJECT",
      "properties": {
          "name": {"type": "STRING"},
          "age": {"type": "INTEGER"},
          "tags": {"type": "ARRAY", "items": {"type": "STRING"}},
      },
  }
  _update_type_string(schema)
  assert schema["type"] == "object"
  assert schema["properties"]["name"]["type"] == "string"
  assert schema["properties"]["age"]["type"] == "integer"
  assert schema["properties"]["tags"]["type"] == "array"
  assert schema["properties"]["tags"]["items"]["type"] == "string"


def test_function_declaration_to_openai_tool():
  fd = types.FunctionDeclaration(
      name="get_weather",
      description="Get weather",
      parameters=types.Schema(
          type=types.Type.OBJECT,
          properties={"location": types.Schema(type=types.Type.STRING)},
          required=["location"],
      ),
  )
  tool = _function_declaration_to_openai_tool(fd)
  assert tool["type"] == "function"
  assert tool["function"]["name"] == "get_weather"
  assert tool["function"]["parameters"]["type"] == "object"
  assert (
      tool["function"]["parameters"]["properties"]["location"]["type"]
      == "string"
  )
  assert tool["function"]["parameters"]["required"] == ["location"]


def test_part_to_openai_content():
  # Test text part
  part = types.Part.from_text(text="Hello")
  content = _part_to_openai_content(part)
  assert content == "Hello"

  # Test thought part
  part = types.Part.from_text(text="I am thinking")
  part.thought = True
  content = _part_to_openai_content(part)
  assert content == "Thought: I am thinking"

  # Test image part (inline data)
  part = types.Part(
      inline_data=types.Blob(data=b"fake_data", mime_type="image/png")
  )
  content = _part_to_openai_content(part)
  assert isinstance(content, dict)
  assert content["type"] == "image_url"
  assert content["image_url"]["url"].startswith("data:image/png;base64,")


def test_content_to_openai_messages_with_empty_response():
  from google.adk.labs.openai._openai_llm import _content_to_openai_messages

  # Test with empty dict response
  content = types.Content(
      role="tool",
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  id="call_123",
                  name="get_weather",
                  response={},
              )
          )
      ],
  )
  messages = _content_to_openai_messages(content)
  assert len(messages) == 1
  assert messages[0]["role"] == "tool"
  assert messages[0]["tool_call_id"] == "call_123"
  assert messages[0]["content"] == "{}"

  # Test with None response
  content = types.Content(
      role="tool",
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  id="call_123",
                  name="get_weather",
                  response=None,
              )
          )
      ],
  )
  messages = _content_to_openai_messages(content)
  assert len(messages) == 1
  assert messages[0]["content"] == ""


@pytest.mark.asyncio
async def test_generate_content_async():
  with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
    openai_llm = OpenAILlm(model="gpt-4o")
    llm_request = LlmRequest(
        model="gpt-4o",
        contents=[Content(role="user", parts=[Part.from_text(text="Hello")])],
    )

    mock_response = mock.MagicMock()
    mock_choice = mock.MagicMock()
    mock_message = mock.MagicMock()
    mock_message.content = "Hello there!"
    mock_message.tool_calls = None
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15

    async def mock_create(*args, **kwargs):
      return mock_response

    with mock.patch(
        "google.adk.labs.openai._openai_llm.AsyncOpenAI"
    ) as mock_client_class:
      mock_client = mock.MagicMock()
      mock_client_class.return_value = mock_client
      mock_client.chat.completions.create = mock_create

      responses = [
          resp
          async for resp in openai_llm.generate_content_async(
              llm_request, stream=False
          )
      ]

      assert len(responses) == 1
      assert isinstance(responses[0], LlmResponse)
      assert responses[0].content.parts[0].text == "Hello there!"
      assert responses[0].usage_metadata.total_token_count == 15


@pytest.mark.asyncio
async def test_generate_content_async_with_config():
  with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
    openai_llm = OpenAILlm(model="gpt-4o")
    llm_request = LlmRequest(
        model="gpt-4o",
        contents=[Content(role="user", parts=[Part.from_text(text="Hello")])],
        config=types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.9,
            stop_sequences=["STOP"],
            max_output_tokens=100,
        ),
    )

    mock_response = mock.MagicMock()
    mock_choice = mock.MagicMock()
    mock_message = mock.MagicMock()
    mock_message.content = "Hello there!"
    mock_message.tool_calls = None
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_call = mock.MagicMock(return_value=mock_response)
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15

    create_kwargs = {}

    async def mock_create(*args, **kwargs):
      nonlocal create_kwargs
      create_kwargs = kwargs
      return mock_response

    with mock.patch(
        "google.adk.labs.openai._openai_llm.AsyncOpenAI"
    ) as mock_client_class:
      mock_client = mock.MagicMock()
      mock_client_class.return_value = mock_client
      mock_client.chat.completions.create = mock_create

      responses = [
          resp
          async for resp in openai_llm.generate_content_async(
              llm_request, stream=False
          )
      ]

      assert len(responses) == 1
      assert create_kwargs["temperature"] == 0.7
      assert create_kwargs["top_p"] == 0.9
      assert create_kwargs["stop"] == ["STOP"]
      assert create_kwargs["max_tokens"] == 100


@pytest.mark.asyncio
async def test_generate_content_async_with_system_instruction():
  with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
    openai_llm = OpenAILlm(model="gpt-4o")
    llm_request = LlmRequest(
        model="gpt-4o",
        contents=[Content(role="user", parts=[Part.from_text(text="Hello")])],
        config=types.GenerateContentConfig(
            system_instruction="You are a helpful assistant.",
        ),
    )

    mock_response = mock.MagicMock()
    mock_choice = mock.MagicMock()
    mock_message = mock.MagicMock()
    mock_message.content = "Hello there!"
    mock_message.tool_calls = None
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15

    create_kwargs = {}

    async def mock_create(*args, **kwargs):
      nonlocal create_kwargs
      create_kwargs = kwargs
      return mock_response

    with mock.patch(
        "google.adk.labs.openai._openai_llm.AsyncOpenAI"
    ) as mock_client_class:
      mock_client = mock.MagicMock()
      mock_client_class.return_value = mock_client
      mock_client.chat.completions.create = mock_create

      responses = [
          resp
          async for resp in openai_llm.generate_content_async(
              llm_request, stream=False
          )
      ]

      assert len(responses) == 1
      messages = create_kwargs["messages"]
      assert len(messages) == 2
      assert messages[0]["role"] == "system"
      assert messages[0]["content"] == "You are a helpful assistant."
      assert messages[1]["role"] == "user"
      assert messages[1]["content"] == "Hello"


@pytest.mark.asyncio
async def test_generate_content_async_with_image():
  with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
    openai_llm = OpenAILlm(model="gpt-4o")

    image_part = Part(
        inline_data=types.Blob(data=b"fake_image_data", mime_type="image/png")
    )

    llm_request = LlmRequest(
        model="gpt-4o",
        contents=[
            Content(
                role="user",
                parts=[Part.from_text(text="Analyze this"), image_part],
            )
        ],
    )

    mock_response = mock.MagicMock()
    mock_choice = mock.MagicMock()
    mock_message = mock.MagicMock()
    mock_message.content = "It's an image."
    mock_message.tool_calls = None
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15

    create_kwargs = {}

    async def mock_create(*args, **kwargs):
      nonlocal create_kwargs
      create_kwargs = kwargs
      return mock_response

    with mock.patch(
        "google.adk.labs.openai._openai_llm.AsyncOpenAI"
    ) as mock_client_class:
      mock_client = mock.MagicMock()
      mock_client_class.return_value = mock_client
      mock_client.chat.completions.create = mock_create

      responses = [
          resp
          async for resp in openai_llm.generate_content_async(
              llm_request, stream=False
          )
      ]

      assert len(responses) == 1
      messages = create_kwargs["messages"]
      assert len(messages) == 1
      assert messages[0]["role"] == "user"
      content = messages[0]["content"]
      assert isinstance(content, list)
      assert len(content) == 2
      assert content[0]["type"] == "text"
      assert content[0]["text"] == "Analyze this"
      assert content[1]["type"] == "image_url"
      assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
