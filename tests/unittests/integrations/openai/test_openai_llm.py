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

import contextlib
import json
import logging
import os
from unittest import mock

from google.adk.integrations.openai import OpenAIGenerateContentConfig
from google.adk.integrations.openai._openai_llm import _function_declaration_to_openai_tool
from google.adk.integrations.openai._openai_llm import _map_finish_reason
from google.adk.integrations.openai._openai_llm import _part_to_openai_content
from google.adk.integrations.openai._openai_llm import _response_to_llm_response
from google.adk.integrations.openai._openai_llm import _serialize_system_instruction
from google.adk.integrations.openai._openai_llm import _usage_metadata
from google.adk.integrations.openai._openai_llm import OpenAILlm
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
  from google.adk.integrations.openai._openai_llm import _content_to_openai_messages

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
        "google.adk.integrations.openai._openai_llm.AsyncOpenAI"
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
@pytest.mark.parametrize("model", ["gpt-5.6-sol", "gpt-6-astra", "o3-mini"])
async def test_reasoning_model_uses_max_completion_tokens_and_drops_temp(model):
  """Reasoning models send max_completion_tokens and drop temperature/top_p."""
  with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
    openai_llm = OpenAILlm(model=model, max_tokens=256)
    llm_request = LlmRequest(
        model=model,
        contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
        config=types.GenerateContentConfig(temperature=0.2, top_p=0.9),
    )

    mock_response = mock.MagicMock()
    mock_choice = mock.MagicMock()
    mock_choice.message.content = "hi"
    mock_choice.message.tool_calls = None
    mock_choice.finish_reason = "stop"
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 1
    mock_response.usage.completion_tokens = 1
    mock_response.usage.total_tokens = 2

    captured = {}

    async def mock_create(*args, **kwargs):
      captured.update(kwargs)
      return mock_response

    with mock.patch(
        "google.adk.integrations.openai._openai_llm.AsyncOpenAI"
    ) as mock_client_class:
      mock_client = mock.MagicMock()
      mock_client_class.return_value = mock_client
      mock_client.chat.completions.create = mock_create

      _ = [
          resp
          async for resp in openai_llm.generate_content_async(
              llm_request, stream=False
          )
      ]

    assert captured["max_completion_tokens"] == 256
    assert "max_tokens" not in captured
    assert "temperature" not in captured
    assert "top_p" not in captured


async def _capture_chat_kwargs(openai_llm, llm_request) -> dict:
  """Runs a mocked non-streaming request and returns the create() kwargs."""
  mock_response = mock.MagicMock()
  mock_choice = mock.MagicMock()
  mock_choice.message.content = "hi"
  mock_choice.message.tool_calls = None
  mock_choice.finish_reason = "stop"
  mock_response.choices = [mock_choice]
  mock_response.usage.prompt_tokens = 1
  mock_response.usage.completion_tokens = 1
  mock_response.usage.total_tokens = 2

  captured = {}

  async def mock_create(*args, **kwargs):
    captured.update(kwargs)
    return mock_response

  with mock.patch(
      "google.adk.integrations.openai._openai_llm.AsyncOpenAI"
  ) as mock_client_class:
    mock_client = mock.MagicMock()
    mock_client_class.return_value = mock_client
    mock_client.chat.completions.create = mock_create
    _ = [
        resp
        async for resp in openai_llm.generate_content_async(
            llm_request, stream=False
        )
    ]
  return captured


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,effort",
    [
        # Chat Completions: advanced models accept up to xhigh (not max).
        ("gpt-6-astra", "xhigh"),
        ("gpt-5.6-sol", "high"),
        ("gpt-5", "minimal"),
        ("o3", "medium"),
    ],
)
async def test_reasoning_effort_sent(model, effort):
  """OpenAIGenerateContentConfig.effort maps to the reasoning_effort kwarg."""
  with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
    openai_llm = OpenAILlm(model=model, max_tokens=256)
    llm_request = LlmRequest(
        model=model,
        contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
        config=OpenAIGenerateContentConfig(effort=effort),
    )
    captured = await _capture_chat_kwargs(openai_llm, llm_request)
    assert captured["reasoning_effort"] == effort


@pytest.mark.asyncio
async def test_reasoning_effort_absent_without_config():
  """No reasoning_effort is sent when effort is not configured."""
  with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
    openai_llm = OpenAILlm(model="gpt-5.6-sol", max_tokens=256)
    llm_request = LlmRequest(
        model="gpt-5.6-sol",
        contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
    )
    captured = await _capture_chat_kwargs(openai_llm, llm_request)
    assert "reasoning_effort" not in captured


@pytest.mark.asyncio
async def test_reasoning_effort_unsupported_tier_raises():
  """An effort tier the model does not accept raises before the request."""
  with mock.patch.dict(
      os.environ, {"OPENAI_API_KEY": "test_key", "OPENAI_BASE_URL": ""}
  ):
    openai_llm = OpenAILlm(model="o3", max_tokens=256)
    llm_request = LlmRequest(
        model="o3",
        contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
        config=OpenAIGenerateContentConfig(effort="minimal"),
    )
    with pytest.raises(ValueError, match="not supported by model 'o3'"):
      _ = [
          resp
          async for resp in openai_llm.generate_content_async(
              llm_request, stream=False
          )
      ]


@pytest.mark.asyncio
async def test_reasoning_effort_on_non_reasoning_model_raises():
  """Setting effort on a non-reasoning model raises before the request."""
  with mock.patch.dict(
      os.environ, {"OPENAI_API_KEY": "test_key", "OPENAI_BASE_URL": ""}
  ):
    openai_llm = OpenAILlm(model="gpt-4o")
    llm_request = LlmRequest(
        model="gpt-4o",
        contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
        config=OpenAIGenerateContentConfig(effort="high"),
    )
    with pytest.raises(ValueError, match="does not accept a reasoning effort"):
      _ = [
          resp
          async for resp in openai_llm.generate_content_async(
              llm_request, stream=False
          )
      ]


@pytest.mark.asyncio
async def test_reasoning_effort_max_rejected_on_chat():
  """``max`` is a Responses-only tier; Chat Completions rejects it."""
  with mock.patch.dict(
      os.environ, {"OPENAI_API_KEY": "test_key", "OPENAI_BASE_URL": ""}
  ):
    openai_llm = OpenAILlm(model="gpt-6-astra", max_tokens=256)
    llm_request = LlmRequest(
        model="gpt-6-astra",
        contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
        config=OpenAIGenerateContentConfig(effort="max"),
    )
    with pytest.raises(ValueError, match="on the chat API"):
      _ = [
          resp
          async for resp in openai_llm.generate_content_async(
              llm_request, stream=False
          )
      ]


@pytest.mark.asyncio
async def test_reasoning_effort_passthrough_on_base_url_backend():
  """With a custom base_url, effort is passed through without model gating.

  An OpenAI-compatible backend (e.g. Grok on Vertex AI) accepts
  reasoning_effort but its model id is not an OpenAI id, so the tier must not
  be validated against the OpenAI per-model tables; the backend rejects an
  unsupported one.
  """
  openai_llm = OpenAILlm(
      model="xai/grok-4.6",
      api_key="k",
      base_url="https://host.example/v1",
      max_tokens=256,
  )
  llm_request = LlmRequest(
      model="xai/grok-4.6",
      contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
      config=OpenAIGenerateContentConfig(effort="high"),
  )

  captured = {}

  async def mock_create(*args, **kwargs):
    nonlocal captured
    captured = kwargs
    return _text_completion()

  with mock.patch(
      "google.adk.integrations.openai._openai_llm.AsyncOpenAI"
  ) as mock_client_class:
    mock_client = mock.MagicMock()
    mock_client_class.return_value = mock_client
    mock_client.chat.completions.create = mock_create

    _ = [
        resp
        async for resp in openai_llm.generate_content_async(
            llm_request, stream=False
        )
    ]

  assert captured["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_reasoning_effort_passthrough_with_injected_client():
  """An injected client may point at a compatible backend; skip model gating.

  base_url is None, but the host can come from client=AsyncOpenAI(base_url=...),
  so the tier must be passed through rather than validated against the OpenAI
  per-model tables (grok-4.6 is not an OpenAI reasoning model).
  """
  captured = {}

  async def mock_create(*args, **kwargs):
    nonlocal captured
    captured = kwargs
    return _text_completion()

  client = AsyncOpenAI(api_key="k", base_url="https://host.example/v1")
  openai_llm = OpenAILlm(model="xai/grok-4.6", client=client, max_tokens=256)
  llm_request = LlmRequest(
      model="xai/grok-4.6",
      contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
      config=OpenAIGenerateContentConfig(effort="high"),
  )

  with mock.patch.object(client.chat.completions, "create", mock_create):
    _ = [
        resp
        async for resp in openai_llm.generate_content_async(
            llm_request, stream=False
        )
    ]

  assert captured["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_reasoning_effort_passthrough_with_openai_base_url_env():
  """OPENAI_BASE_URL points the default client at a compatible backend.

  base_url is None, but the SDK reads OPENAI_BASE_URL, so the tier must be
  passed through rather than validated against the OpenAI per-model tables.
  """
  captured = {}

  async def mock_create(*args, **kwargs):
    nonlocal captured
    captured = kwargs
    return _text_completion()

  with mock.patch.dict(
      os.environ,
      {"OPENAI_API_KEY": "k", "OPENAI_BASE_URL": "https://host.example/v1"},
  ):
    openai_llm = OpenAILlm(model="xai/grok-4.6", max_tokens=256)
    llm_request = LlmRequest(
        model="xai/grok-4.6",
        contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
        config=OpenAIGenerateContentConfig(effort="high"),
    )

    with mock.patch(
        "google.adk.integrations.openai._openai_llm.AsyncOpenAI"
    ) as mock_client_class:
      mock_client = mock.MagicMock()
      mock_client_class.return_value = mock_client
      mock_client.chat.completions.create = mock_create

      _ = [
          resp
          async for resp in openai_llm.generate_content_async(
              llm_request, stream=False
          )
      ]

  assert captured["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_reasoning_model_drops_sampling_params_with_warning(caplog):
  """A reasoning model drops temperature/top_p and logs a warning for each."""
  with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
    openai_llm = OpenAILlm(model="o3-mini")
    llm_request = LlmRequest(
        model="o3-mini",
        contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
        config=types.GenerateContentConfig(temperature=0.5, top_p=0.9),
    )

    with caplog.at_level(logging.WARNING):
      create_kwargs = await _capture_create_kwargs(openai_llm, llm_request)

  assert "temperature" not in create_kwargs
  assert "top_p" not in create_kwargs
  assert "Ignoring temperature" in caplog.text
  assert "Ignoring top_p" in caplog.text


@pytest.mark.asyncio
async def test_reasoning_model_keeps_default_temperature_without_warning(
    caplog,
):
  """A reasoning model accepts the default temperature/top_p (1); keep it."""
  with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
    openai_llm = OpenAILlm(model="o3-mini")
    llm_request = LlmRequest(
        model="o3-mini",
        contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
        config=types.GenerateContentConfig(temperature=1, top_p=1),
    )

    with caplog.at_level(logging.WARNING):
      create_kwargs = await _capture_create_kwargs(openai_llm, llm_request)

  assert create_kwargs["temperature"] == 1
  assert create_kwargs["top_p"] == 1
  assert "Ignoring" not in caplog.text


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
        "google.adk.integrations.openai._openai_llm.AsyncOpenAI"
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
        "google.adk.integrations.openai._openai_llm.AsyncOpenAI"
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
        "google.adk.integrations.openai._openai_llm.AsyncOpenAI"
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
        "google.adk.integrations.openai._openai_llm.AsyncOpenAI"
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
        "google.adk.integrations.openai._openai_llm.AsyncOpenAI"
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
        "google.adk.integrations.openai._openai_llm.AsyncOpenAI"
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
        "google.adk.integrations.openai._openai_llm.AsyncOpenAI"
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
  chunk_1.usage = None
  choice_1 = mock.MagicMock()
  choice_1.finish_reason = None
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
  chunk_2.usage = None
  choice_2 = mock.MagicMock()
  choice_2.finish_reason = None
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
  chunk_3.usage = None
  choice_3 = mock.MagicMock()
  choice_3.finish_reason = "tool_calls"
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

  # Trailing usage-only chunk (from stream_options include_usage); no choices.
  chunks = [chunk_1, chunk_2, chunk_3, _usage_only_chunk()]

  with _stream_client(chunks):
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
  # The trailing usage-only chunk and the final finish_reason are surfaced on
  # the final streamed response.
  assert responses[3].finish_reason == types.FinishReason.STOP
  assert responses[3].usage_metadata.prompt_token_count == 12
  assert responses[3].usage_metadata.candidates_token_count == 8
  assert responses[3].usage_metadata.total_token_count == 20


def _text_stream_chunk(content=None, finish_reason=None):
  """Builds a streaming chunk carrying a text delta (no tool calls)."""
  chunk = mock.MagicMock()
  chunk.usage = None
  choice = mock.MagicMock()
  choice.finish_reason = finish_reason
  delta = mock.MagicMock()
  delta.content = content
  delta.tool_calls = None
  choice.delta = delta
  chunk.choices = [choice]
  return chunk


def _usage_only_chunk(prompt=12, completion=8, total=20):
  """Builds the trailing usage-only chunk (no choices)."""
  chunk = mock.MagicMock()
  chunk.choices = []
  chunk.usage.prompt_tokens = prompt
  chunk.usage.completion_tokens = completion
  chunk.usage.total_tokens = total
  chunk.usage.prompt_tokens_details = None
  return chunk


@contextlib.contextmanager
def _stream_client(chunks):
  """Patches AsyncOpenAI so create() yields the given chunks for the block."""

  async def mock_stream():
    for c in chunks:
      yield c

  with mock.patch(
      "google.adk.integrations.openai._openai_llm.AsyncOpenAI"
  ) as mock_client_class:
    mock_client = mock.MagicMock()
    mock_client_class.return_value = mock_client
    mock_client.chat.completions.create = mock.AsyncMock(
        return_value=mock_stream()
    )
    yield mock_client


@pytest.mark.asyncio
async def test_generate_content_async_streaming_text_accumulates():
  """Text deltas stream as partials and merge into a final response."""
  openai_llm = OpenAILlm(model="gpt-4o", api_key="k")
  llm_request = LlmRequest(
      model="gpt-4o",
      contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
  )
  chunks = [
      _text_stream_chunk(content="Hello, "),
      _text_stream_chunk(content="world!", finish_reason="stop"),
      _usage_only_chunk(),
  ]

  with _stream_client(chunks):
    responses = [
        resp
        async for resp in openai_llm.generate_content_async(
            llm_request, stream=True
        )
    ]

  assert [r.partial for r in responses] == [True, True, False]
  assert responses[0].content.parts[0].text == "Hello, "
  assert responses[1].content.parts[0].text == "world!"
  assert responses[2].content.parts[0].text == "Hello, world!"
  assert responses[2].finish_reason == types.FinishReason.STOP
  assert responses[2].usage_metadata.total_token_count == 20


@pytest.mark.asyncio
async def test_generate_content_async_streaming_without_usage_chunk():
  """The stream completes normally when the backend omits the usage chunk."""
  openai_llm = OpenAILlm(model="gpt-4o", api_key="k")
  llm_request = LlmRequest(
      model="gpt-4o",
      contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
  )
  # Backends that ignore stream_options never send the trailing usage chunk.
  chunks = [
      _text_stream_chunk(content="Hello, "),
      _text_stream_chunk(content="world!", finish_reason="stop"),
  ]

  with _stream_client(chunks):
    responses = [
        resp
        async for resp in openai_llm.generate_content_async(
            llm_request, stream=True
        )
    ]

  assert [r.partial for r in responses] == [True, True, False]
  assert responses[-1].content.parts[0].text == "Hello, world!"
  assert responses[-1].finish_reason == types.FinishReason.STOP
  # No usage chunk arrived, so usage metadata stays absent rather than failing.
  assert responses[-1].usage_metadata is None


@pytest.mark.asyncio
async def test_generate_content_async_streaming_length_finish_maps_max_tokens():
  """A non-empty stream ending on length maps to MAX_TOKENS, not an error."""
  openai_llm = OpenAILlm(model="gpt-4o", api_key="k")
  llm_request = LlmRequest(
      model="gpt-4o",
      contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
  )
  chunks = [
      _text_stream_chunk(content="Hello, "),
      _text_stream_chunk(content="world", finish_reason="length"),
      _usage_only_chunk(),
  ]

  with _stream_client(chunks):
    responses = [
        resp
        async for resp in openai_llm.generate_content_async(
            llm_request, stream=True
        )
    ]

  final = responses[-1]
  assert final.partial is False
  assert final.content.parts[0].text == "Hello, world"
  assert final.finish_reason == types.FinishReason.MAX_TOKENS
  # Hitting the token limit with content present is not an error.
  assert final.error_code is None


@pytest.mark.asyncio
async def test_streaming_request_sends_stream_options_include_usage():
  """The streaming request asks for the trailing usage-only chunk."""
  openai_llm = OpenAILlm(model="gpt-4o", api_key="k")
  llm_request = LlmRequest(
      model="gpt-4o",
      contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
  )

  chunks = [
      _text_stream_chunk(content="Hi", finish_reason="stop"),
      _usage_only_chunk(),
  ]

  with _stream_client(chunks) as mock_client:
    _ = [
        resp
        async for resp in openai_llm.generate_content_async(
            llm_request, stream=True
        )
    ]

  create_kwargs = mock_client.chat.completions.create.call_args.kwargs
  assert create_kwargs["stream"] is True
  assert create_kwargs["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_generate_content_async_streaming_empty_abnormal_finish_is_error():
  """A stream ending with no content and an abnormal finish is an error."""
  openai_llm = OpenAILlm(model="gpt-4o", api_key="k")
  llm_request = LlmRequest(
      model="gpt-4o",
      contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
  )
  chunks = [_text_stream_chunk(content=None, finish_reason="content_filter")]

  with _stream_client(chunks):
    responses = [
        resp
        async for resp in openai_llm.generate_content_async(
            llm_request, stream=True
        )
    ]

  final = responses[-1]
  assert final.content is None
  assert final.finish_reason == types.FinishReason.SAFETY
  assert final.error_code == types.FinishReason.SAFETY
  assert final.error_message


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
async def test_api_key_string_is_passed_to_client():
  """A string api_key is forwarded to the default AsyncOpenAI client."""
  with mock.patch(
      "google.adk.integrations.openai._openai_llm.AsyncOpenAI"
  ) as client_cls:
    _ = OpenAILlm(model="gpt-4o", api_key="secret")._openai_client
  client_cls.assert_called_once_with(api_key="secret")


@pytest.mark.asyncio
async def test_base_url_is_passed_to_client():
  """base_url is forwarded to the default AsyncOpenAI client."""
  with mock.patch(
      "google.adk.integrations.openai._openai_llm.AsyncOpenAI"
  ) as client_cls:
    _ = OpenAILlm(
        model="gpt-4o", api_key="secret", base_url="https://host.example/v1"
    )._openai_client
  client_cls.assert_called_once_with(
      api_key="secret", base_url="https://host.example/v1"
  )


@pytest.mark.asyncio
async def test_callable_api_key_wrapped_as_async_provider():
  """A callable api_key becomes the async provider AsyncOpenAI refreshes.

  A Vertex OAuth bearer token expires ~1h, so ``AsyncOpenAI`` awaits its api_key
  provider on every request rather than freezing the key at construction. A sync
  callable is adapted into that async provider; the callable is not consumed at
  construction time and is re-invoked on each await.
  """
  calls = {"n": 0}

  def key_provider() -> str:
    calls["n"] += 1
    return f"token-{calls['n']}"

  with mock.patch(
      "google.adk.integrations.openai._openai_llm.AsyncOpenAI"
  ) as client_cls:
    _ = OpenAILlm(
        model="xai/grok-4.6",
        api_key=key_provider,
        base_url="https://host.example/v1",
    )._openai_client

  client_cls.assert_called_once()
  ctor_kwargs = client_cls.call_args.kwargs
  assert ctor_kwargs["base_url"] == "https://host.example/v1"
  provider = ctor_kwargs["api_key"]
  # Not resolved eagerly at construction...
  assert calls["n"] == 0
  # ...and re-invoked (awaited) on each request, yielding a fresh token.
  assert await provider() == "token-1"
  assert await provider() == "token-2"
  assert calls["n"] == 2


@pytest.mark.asyncio
async def test_async_api_key_callable_supported():
  """An async api_key provider is passed through for AsyncOpenAI to await."""

  async def _key() -> str:
    return "k"

  with mock.patch(
      "google.adk.integrations.openai._openai_llm.AsyncOpenAI"
  ) as client_cls:
    _ = OpenAILlm(model="gpt-4o", api_key=_key)._openai_client

  provider = client_cls.call_args.kwargs["api_key"]
  assert await provider() == "k"


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
        "google.adk.integrations.openai._openai_llm.AsyncOpenAI"
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
        "google.adk.integrations.openai._openai_llm.AsyncOpenAI"
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode, expected",
    [
        (types.FunctionCallingConfigMode.ANY, "required"),
        (types.FunctionCallingConfigMode.NONE, "none"),
        (types.FunctionCallingConfigMode.AUTO, "auto"),
    ],
)
async def test_tool_choice_follows_function_calling_mode(mode, expected):
  """function_calling_config.mode drives tool_choice for every mode."""
  with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
    openai_llm = OpenAILlm(model="gpt-4o")
    llm_request = LlmRequest(
        model="gpt-4o",
        contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
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
            ],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode=mode)
            ),
        ),
    )

    create_kwargs = await _capture_create_kwargs(openai_llm, llm_request)

  assert create_kwargs["tool_choice"] == expected


@pytest.mark.asyncio
async def test_multiple_tool_entries_are_all_declared():
  """Function declarations across multiple Tool entries are all sent."""
  with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
    openai_llm = OpenAILlm(model="gpt-4o")
    tool_a = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(name="a", description="A")
        ]
    )
    tool_b = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(name="b", description="B")
        ]
    )
    llm_request = LlmRequest(
        model="gpt-4o",
        contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
        config=types.GenerateContentConfig(tools=[tool_a, tool_b]),
    )

    create_kwargs = await _capture_create_kwargs(openai_llm, llm_request)

  names = [tool["function"]["name"] for tool in create_kwargs["tools"]]
  assert names == ["a", "b"]


@pytest.mark.asyncio
async def test_tool_without_function_declarations_is_skipped_with_warning(
    caplog,
):
  """A tool with no function declarations is skipped and logged, not sent."""
  with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
    openai_llm = OpenAILlm(model="gpt-4o")
    llm_request = LlmRequest(
        model="gpt-4o",
        contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(function_declarations=None),
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(name="a", description="A")
                    ]
                ),
            ]
        ),
    )

    with caplog.at_level(logging.WARNING):
      create_kwargs = await _capture_create_kwargs(openai_llm, llm_request)

  assert len(create_kwargs["tools"]) == 1
  assert "no function declarations" in caplog.text


@pytest.mark.asyncio
async def test_system_instruction_content_is_serialized():
  """A non-string system_instruction is flattened to system message text."""
  with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
    openai_llm = OpenAILlm(model="gpt-4o")
    llm_request = LlmRequest(
        model="gpt-4o",
        contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
        config=types.GenerateContentConfig(
            system_instruction=types.Content(
                parts=[
                    Part.from_text(text="Be "),
                    Part.from_text(text="concise."),
                ]
            )
        ),
    )

    create_kwargs = {}

    async def mock_create(*args, **kwargs):
      nonlocal create_kwargs
      create_kwargs = kwargs
      return _text_completion()

    with mock.patch(
        "google.adk.integrations.openai._openai_llm.AsyncOpenAI"
    ) as mock_client_class:
      mock_client = mock.MagicMock()
      mock_client_class.return_value = mock_client
      mock_client.chat.completions.create = mock_create

      _ = [
          resp async for resp in openai_llm.generate_content_async(llm_request)
      ]

  assert create_kwargs["messages"][0] == {
      "role": "system",
      "content": "Be concise.",
  }


def test_serialize_system_instruction_part_shaped_mapping():
  """A Part-shaped mapping serializes to its text."""
  assert _serialize_system_instruction({"text": "Be concise."}) == "Be concise."


def test_serialize_system_instruction_content_shaped_mapping():
  """A Content-shaped mapping is serialized instead of raising ValidationError.

  Previously the Mapping branch did types.Part(**mapping), which raised an
  uncaught pydantic ValidationError on a {'role': ..., 'parts': [...]} dict.
  """
  mapping = {
      "role": "system",
      "parts": [{"text": "Be "}, {"text": "concise."}],
  }
  assert _serialize_system_instruction(mapping) == "Be concise."


def test_serialize_system_instruction_unparseable_mapping_returns_none():
  """A mapping that fits neither Part nor Content is dropped, not raised."""
  assert _serialize_system_instruction({"not_a_field": 123}) is None


def test_serialize_system_instruction_list_joins_items_with_newline():
  """A list of instructions is flattened and joined with newlines."""
  instructions = [
      "Be concise.",
      types.Part.from_text(text="Cite sources."),
      {"text": "Avoid jargon."},
  ]
  assert (
      _serialize_system_instruction(instructions)
      == "Be concise.\nCite sources.\nAvoid jargon."
  )


def test_serialize_system_instruction_unsupported_type_warns(caplog):
  """An unsupported instruction type is dropped and logged."""
  with caplog.at_level(logging.WARNING):
    assert _serialize_system_instruction(types.File(name="f")) is None
  assert "unsupported type" in caplog.text


def test_serialize_system_instruction_non_string_keys_returns_none():
  """A mapping with non-string keys is dropped, not raised."""
  assert _serialize_system_instruction({1: "x"}) is None


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


async def _capture_create_kwargs(openai_llm, llm_request):
  """Runs one non-streaming request and returns the create() kwargs sent."""
  create_kwargs = {}

  async def mock_create(*args, **kwargs):
    nonlocal create_kwargs
    create_kwargs = kwargs
    return _text_completion()

  with mock.patch(
      "google.adk.integrations.openai._openai_llm.AsyncOpenAI"
  ) as mock_client_class:
    mock_client = mock.MagicMock()
    mock_client_class.return_value = mock_client
    mock_client.chat.completions.create = mock_create

    _ = [resp async for resp in openai_llm.generate_content_async(llm_request)]

  return create_kwargs


@pytest.mark.asyncio
async def test_response_schema_dict_becomes_json_schema_format():
  """A dict response_schema is sent as a strict json_schema response_format."""
  with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
    openai_llm = OpenAILlm(model="gpt-4o")
    llm_request = LlmRequest(
        model="gpt-4o",
        contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
        config=types.GenerateContentConfig(
            response_schema={
                "title": "Person",
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
        ),
    )

    create_kwargs = await _capture_create_kwargs(openai_llm, llm_request)

  response_format = create_kwargs["response_format"]
  assert response_format["type"] == "json_schema"
  assert response_format["json_schema"]["name"] == "Person"
  assert response_format["json_schema"]["strict"] is True
  assert "name" in response_format["json_schema"]["schema"]["properties"]


@pytest.mark.asyncio
async def test_response_mime_type_json_becomes_json_object_format():
  """response_mime_type application/json maps to a json_object format."""
  with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
    openai_llm = OpenAILlm(model="gpt-4o")
    llm_request = LlmRequest(
        model="gpt-4o",
        contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    create_kwargs = await _capture_create_kwargs(openai_llm, llm_request)

  assert create_kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_model_function_call_becomes_assistant_tool_calls():
  """A model-turn function_call part serializes to assistant tool_calls."""
  with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
    openai_llm = OpenAILlm(model="gpt-4o")
    llm_request = LlmRequest(
        model="gpt-4o",
        contents=[
            Content(role="user", parts=[Part.from_text(text="Weather?")]),
            Content(
                role="model",
                parts=[
                    Part.from_function_call(
                        name="get_weather", args={"location": "NYC"}
                    )
                ],
            ),
        ],
    )

    create_kwargs = await _capture_create_kwargs(openai_llm, llm_request)

  assistant_msgs = [
      m for m in create_kwargs["messages"] if m.get("role") == "assistant"
  ]
  assert len(assistant_msgs) == 1
  tool_calls = assistant_msgs[0]["tool_calls"]
  assert len(tool_calls) == 1
  assert tool_calls[0]["type"] == "function"
  assert tool_calls[0]["function"]["name"] == "get_weather"
  assert json.loads(tool_calls[0]["function"]["arguments"]) == {
      "location": "NYC"
  }


@pytest.mark.parametrize(
    "details, expected",
    [
        (mock.MagicMock(reasoning_tokens=42), 42),
        (mock.MagicMock(reasoning_tokens=0), 0),
        (None, None),
    ],
)
def test_usage_metadata_maps_reasoning_tokens(details, expected):
  """completion_tokens_details.reasoning_tokens maps to thoughts_token_count."""
  usage = mock.MagicMock(
      prompt_tokens=100,
      completion_tokens=50,
      total_tokens=150,
      prompt_tokens_details=None,
      completion_tokens_details=details,
  )
  metadata = _usage_metadata(usage)
  assert metadata.thoughts_token_count == expected
  assert metadata.candidates_token_count == 50
