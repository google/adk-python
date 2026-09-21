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
from google.adk.labs.openai._openai_llm import _map_finish_reason
from google.adk.labs.openai._openai_llm import _part_to_openai_content
from google.adk.labs.openai._openai_llm import _response_to_llm_response
from google.adk.labs.openai._openai_llm import OpenAILlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from google.genai.types import Content
from google.genai.types import Part
from openai import AsyncOpenAI
import pytest


def test_supported_models():
  models = OpenAILlm.supported_models()
  assert len(models) == 2
  assert models[0] == r"gpt-.*"
  assert models[1] == r"o\d+-.*"


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


def _completion_with_cached_tokens(cached_tokens):
  """Builds a mock ChatCompletion whose usage carries prompt_tokens_details."""
  mock_response = mock.MagicMock()
  mock_choice = mock.MagicMock()
  mock_message = mock.MagicMock()
  mock_message.content = "Hello there!"
  mock_message.tool_calls = None
  mock_choice.message = mock_message
  mock_response.choices = [mock_choice]
  mock_response.usage.prompt_tokens = 100
  mock_response.usage.completion_tokens = 5
  mock_response.usage.total_tokens = 105
  if cached_tokens is None:
    mock_response.usage.prompt_tokens_details = None
  else:
    mock_response.usage.prompt_tokens_details.cached_tokens = cached_tokens
  return mock_response


@pytest.mark.asyncio
async def test_generate_content_async_reports_cached_tokens():
  """prompt_tokens_details.cached_tokens populates cached_content_token_count."""
  with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
    openai_llm = OpenAILlm(model="gpt-4o")
    llm_request = LlmRequest(
        model="gpt-4o",
        contents=[Content(role="user", parts=[Part.from_text(text="Hello")])],
    )

    mock_response = _completion_with_cached_tokens(64)

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
      assert responses[0].usage_metadata.cached_content_token_count == 64
      assert responses[0].usage_metadata.prompt_token_count == 100


@pytest.mark.asyncio
async def test_generate_content_async_zero_cached_tokens():
  """No cache hit (cached_tokens=0) reports 0, not a regression."""
  with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
    openai_llm = OpenAILlm(model="gpt-4o")
    llm_request = LlmRequest(
        model="gpt-4o",
        contents=[Content(role="user", parts=[Part.from_text(text="Hello")])],
    )

    mock_response = _completion_with_cached_tokens(0)

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

      assert responses[0].usage_metadata.cached_content_token_count == 0


@pytest.mark.asyncio
async def test_generate_content_async_absent_prompt_tokens_details():
  """Missing prompt_tokens_details maps to None (no cached count reported)."""
  with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
    openai_llm = OpenAILlm(model="gpt-4o")
    llm_request = LlmRequest(
        model="gpt-4o",
        contents=[Content(role="user", parts=[Part.from_text(text="Hello")])],
    )

    mock_response = _completion_with_cached_tokens(None)

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

      assert responses[0].usage_metadata.cached_content_token_count is None


@pytest.mark.asyncio
async def test_generate_content_async_routes_through_provided_client():
  """Requests reach the pre-configured client, not a default one."""
  client = AsyncOpenAI(base_url="https://compatible.example/v1", api_key="k")
  openai_llm = OpenAILlm(model="my-model", client=client)
  llm_request = LlmRequest(
      model="my-model",
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
  mock_response.usage.prompt_tokens_details = None

  async def mock_create(*args, **kwargs):
    return mock_response

  with mock.patch.object(
      client.chat.completions, "create", side_effect=mock_create
  ) as mock_client_create:
    with mock.patch(
        "google.adk.labs.openai._openai_llm.AsyncOpenAI"
    ) as mock_client_class:
      responses = [
          resp
          async for resp in openai_llm.generate_content_async(
              llm_request, stream=False
          )
      ]

  mock_client_class.assert_not_called()
  mock_client_create.assert_called_once()
  assert responses[0].content.parts[0].text == "Hello there!"


@pytest.mark.asyncio
async def test_generate_content_async_streaming_tool_call():
  openai_llm = OpenAILlm(model="gpt-4o", api_key="k")
  llm_request = LlmRequest(
      model="gpt-4o",
      contents=[Content(role="user", parts=[Part.from_text(text="Weather?")])],
      config=types.GenerateContentConfig(
          tools=[
              types.Tool(
                  function_declarations=[
                      types.FunctionDeclaration(
                          name="get_weather",
                          description="Get weather",
                          parameters=types.Schema(
                              type=types.Type.OBJECT,
                              properties={
                                  "location": types.Schema(
                                      type=types.Type.STRING
                                  )
                              },
                          ),
                      )
                  ]
              )
          ]
      ),
  )

  chunk_1 = mock.MagicMock()
  choice_1 = mock.MagicMock()
  delta_1 = mock.MagicMock()
  delta_1.content = None
  tc_1 = mock.MagicMock()
  tc_1.index = 0
  tc_1.id = "call_123"
  tc_1.function = mock.MagicMock()
  tc_1.function.name = "get_weather"
  tc_1.function.arguments = ""
  delta_1.tool_calls = [tc_1]
  choice_1.delta = delta_1
  chunk_1.choices = [choice_1]

  chunk_2 = mock.MagicMock()
  choice_2 = mock.MagicMock()
  delta_2 = mock.MagicMock()
  delta_2.content = None
  tc_2 = mock.MagicMock()
  tc_2.index = 0
  tc_2.id = None
  tc_2.function = mock.MagicMock()
  tc_2.function.name = None
  tc_2.function.arguments = '{"location":'
  delta_2.tool_calls = [tc_2]
  choice_2.delta = delta_2
  chunk_2.choices = [choice_2]

  chunk_3 = mock.MagicMock()
  choice_3 = mock.MagicMock()
  delta_3 = mock.MagicMock()
  delta_3.content = None
  tc_3 = mock.MagicMock()
  tc_3.index = 0
  tc_3.id = None
  tc_3.function = mock.MagicMock()
  tc_3.function.name = None
  tc_3.function.arguments = ' "Paris"}'
  delta_3.tool_calls = [tc_3]
  choice_3.delta = delta_3
  chunk_3.choices = [choice_3]

  chunks = [chunk_1, chunk_2, chunk_3]

  async def mock_stream():
    for c in chunks:
      yield c

  with mock.patch(
      "google.adk.labs.openai._openai_llm.AsyncOpenAI"
  ) as mock_client_class:
    mock_client = mock.MagicMock()
    mock_client_class.return_value = mock_client
    mock_client.chat.completions.create = mock.AsyncMock(
        return_value=mock_stream()
    )

    responses = [
        resp
        async for resp in openai_llm.generate_content_async(
            llm_request, stream=True
        )
    ]

  assert len(responses) == 4

  assert responses[0].partial is True
  assert responses[0].content.parts[0].function_call.id == "call_123"
  assert responses[0].content.parts[0].function_call.name == "get_weather"
  assert responses[0].content.parts[0].function_call.will_continue is True
  assert responses[0].content.parts[0].function_call.partial_args is None

  assert responses[1].partial is True
  assert responses[1].content.parts[0].function_call.id == "call_123"
  assert responses[1].content.parts[0].function_call.partial_args is None
  assert responses[1].content.parts[0].function_call.will_continue is True

  assert responses[2].partial is True
  assert responses[2].content.parts[0].function_call.id == "call_123"
  assert (
      responses[2].content.parts[0].function_call.partial_args[0].json_path
      == "$.location"
  )
  assert (
      responses[2].content.parts[0].function_call.partial_args[0].string_value
      == "Paris"
  )
  assert responses[2].content.parts[0].function_call.will_continue is True

  assert responses[3].partial is False
  assert responses[3].content.parts[0].function_call.id == "call_123"
  assert responses[3].content.parts[0].function_call.name == "get_weather"
  assert responses[3].content.parts[0].function_call.args == {
      "location": "Paris"
  }


def _text_completion(content="Hi", finish_reason="stop"):
  """Builds a minimal mock ChatCompletion with a single text choice."""
  response = mock.MagicMock()
  choice = mock.MagicMock()
  message = mock.MagicMock()
  message.content = content
  message.tool_calls = None
  choice.message = message
  choice.finish_reason = finish_reason
  response.choices = [choice]
  response.usage.prompt_tokens = 10
  response.usage.completion_tokens = 5
  response.usage.total_tokens = 15
  response.usage.prompt_tokens_details = None
  return response


@pytest.mark.asyncio
async def test_response_maps_finish_reason():
  """OpenAI finish_reason maps onto LlmResponse.finish_reason."""
  with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
    openai_llm = OpenAILlm(model="gpt-4o")
    llm_request = LlmRequest(
        model="gpt-4o",
        contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
    )

    async def mock_create(*args, **kwargs):
      return _text_completion(finish_reason="length")

    with mock.patch(
        "google.adk.labs.openai._openai_llm.AsyncOpenAI"
    ) as mock_client_class:
      mock_client = mock.MagicMock()
      mock_client_class.return_value = mock_client
      mock_client.chat.completions.create = mock_create

      responses = [
          resp async for resp in openai_llm.generate_content_async(llm_request)
      ]

  assert responses[0].finish_reason == types.FinishReason.MAX_TOKENS


@pytest.mark.asyncio
async def test_response_without_usage_does_not_crash():
  """A response missing usage yields no usage metadata instead of raising."""
  response = _text_completion()
  response.usage = None

  with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
    openai_llm = OpenAILlm(model="gpt-4o")
    llm_request = LlmRequest(
        model="gpt-4o",
        contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
    )

    async def mock_create(*args, **kwargs):
      return response

    with mock.patch(
        "google.adk.labs.openai._openai_llm.AsyncOpenAI"
    ) as mock_client_class:
      mock_client = mock.MagicMock()
      mock_client_class.return_value = mock_client
      mock_client.chat.completions.create = mock_create

      responses = [
          resp async for resp in openai_llm.generate_content_async(llm_request)
      ]

  assert responses[0].usage_metadata is None
  assert responses[0].content.parts[0].text == "Hi"


def test_response_with_no_choices_returns_error():
  """A response with no choices maps to an OTHER error, not an IndexError."""
  response = mock.MagicMock()
  response.choices = []
  response.usage = None

  llm_response = _response_to_llm_response(response)

  assert llm_response.finish_reason == types.FinishReason.OTHER
  assert llm_response.error_code == types.FinishReason.OTHER


def test_response_no_content_non_stop_finish_returns_error():
  """No content plus an abnormal finish reason surfaces as an error."""
  response = _text_completion(content="", finish_reason="content_filter")

  llm_response = _response_to_llm_response(response)

  assert llm_response.content is None
  assert llm_response.finish_reason == types.FinishReason.SAFETY
  assert llm_response.error_code == types.FinishReason.SAFETY
  assert llm_response.error_message


def test_response_with_content_non_stop_finish_is_not_error():
  """A truncated-but-usable response (content + non-STOP) stays a success."""
  response = _text_completion(content="partial", finish_reason="length")

  llm_response = _response_to_llm_response(response)

  assert llm_response.content is not None
  assert llm_response.finish_reason == types.FinishReason.MAX_TOKENS
  assert llm_response.error_code is None


def test_response_no_content_stop_finish_is_not_promoted_here():
  """Empty content with a normal STOP finish stays a plain empty response.

  The parser leaves content=None, finish_reason=STOP and error_code=None; it is
  base_llm_flow (not this wrapper) that promotes a non-streaming empty STOP
  response to a MODEL_RETURNED_NO_CONTENT error downstream.
  """
  response = _text_completion(content="", finish_reason="stop")

  llm_response = _response_to_llm_response(response)

  assert llm_response.content is None
  assert llm_response.finish_reason == types.FinishReason.STOP
  assert llm_response.error_code is None


def test_response_no_content_no_finish_reason_yields_empty_response():
  """A response with neither content nor a finish reason stays non-error."""
  response = _text_completion(content="", finish_reason=None)

  llm_response = _response_to_llm_response(response)

  assert llm_response.content is None
  assert llm_response.finish_reason is None
  assert llm_response.error_code is None


def test_map_finish_reason_recognized_values():
  """Recognized OpenAI finish reasons map to specific ADK codes."""
  assert _map_finish_reason("stop") == types.FinishReason.STOP
  assert _map_finish_reason("tool_calls") == types.FinishReason.STOP
  assert _map_finish_reason("function_call") == types.FinishReason.STOP
  assert _map_finish_reason("length") == types.FinishReason.MAX_TOKENS
  assert _map_finish_reason("content_filter") == types.FinishReason.SAFETY
  assert _map_finish_reason(None) is None


def test_map_finish_reason_unknown_is_unspecified():
  """An unrecognized finish reason maps to UNSPECIFIED, not OTHER.

  Matches the convention in models/anthropic_llm.py and models/apigee_llm.py;
  OTHER is reserved for recognized abnormal terminations (e.g. no choices).
  """
  assert (
      _map_finish_reason("some_new_reason")
      == types.FinishReason.FINISH_REASON_UNSPECIFIED
  )
