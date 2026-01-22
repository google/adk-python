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
# limitations under the License

import contextlib
import json
import logging
import os
import sys
import tempfile
import unittest
from unittest.mock import ANY
from unittest.mock import AsyncMock
from unittest.mock import Mock
import warnings

from google.adk.models.lite_llm import _content_to_message_param
from google.adk.models.lite_llm import _FILE_ID_REQUIRED_PROVIDERS
from google.adk.models.lite_llm import _FINISH_REASON_MAPPING
from google.adk.models.lite_llm import _function_declaration_to_tool_param
from google.adk.models.lite_llm import _get_completion_inputs
from google.adk.models.lite_llm import _get_content
from google.adk.models.lite_llm import _get_provider_from_model
from google.adk.models.lite_llm import _message_to_generate_content_response
from google.adk.models.lite_llm import _MISSING_TOOL_RESULT_MESSAGE
from google.adk.models.lite_llm import _model_response_to_chunk
from google.adk.models.lite_llm import _model_response_to_generate_content_response
from google.adk.models.lite_llm import _parse_tool_calls_from_text
from google.adk.models.lite_llm import _redirect_litellm_loggers_to_stdout
from google.adk.models.lite_llm import _schema_to_dict
from google.adk.models.lite_llm import _split_message_content_and_tool_calls
from google.adk.models.lite_llm import _to_litellm_response_format
from google.adk.models.lite_llm import _to_litellm_role
from google.adk.models.lite_llm import FunctionChunk
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.lite_llm import LiteLLMClient
from google.adk.models.lite_llm import TextChunk
from google.adk.models.lite_llm import UsageMetadataChunk
from google.adk.models.llm_request import LlmRequest
from google.genai import types
import litellm
from litellm import ChatCompletionAssistantMessage
from litellm import ChatCompletionMessageToolCall
from litellm import Function
from litellm.types.utils import ChatCompletionDeltaToolCall
from litellm.types.utils import Choices
from litellm.types.utils import Delta
from litellm.types.utils import ModelResponse
from litellm.types.utils import StreamingChoices
from pydantic import BaseModel
from pydantic import Field
import pytest

LLM_REQUEST_WITH_FUNCTION_DECLARATION = LlmRequest(
    contents=[
        types.Content(
            role="user", parts=[types.Part.from_text(text="Test prompt")]
        )
    ],
    config=types.GenerateContentConfig(
        tools=[
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="test_function",
                        description="Test function description",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "test_arg": types.Schema(
                                    type=types.Type.STRING
                                ),
                                "array_arg": types.Schema(
                                    type=types.Type.ARRAY,
                                    items={
                                        "type": types.Type.STRING,
                                    },
                                ),
                                "nested_arg": types.Schema(
                                    type=types.Type.OBJECT,
                                    properties={
                                        "nested_key1": types.Schema(
                                            type=types.Type.STRING
                                        ),
                                        "nested_key2": types.Schema(
                                            type=types.Type.STRING
                                        ),
                                    },
                                ),
                            },
                        ),
                    )
                ]
            )
        ],
    ),
)

FILE_URI_TEST_CASES = [
    pytest.param("gs://bucket/document.pdf", "application/pdf", id="pdf"),
    pytest.param("gs://bucket/data.json", "application/json", id="json"),
    pytest.param("gs://bucket/data.txt", "text/plain", id="txt"),
]

FILE_BYTES_TEST_CASES = [
    pytest.param(
        b"test_pdf_data",
        "application/pdf",
        "data:application/pdf;base64,dGVzdF9wZGZfZGF0YQ==",
        id="pdf",
    ),
    pytest.param(
        b'{"hello":"world"}',
        "application/json",
        "data:application/json;base64,eyJoZWxsbyI6IndvcmxkIn0=",
        id="json",
    ),
]

STREAMING_MODEL_RESPONSE = [
    ModelResponse(
        model="test_model",
        choices=[
            StreamingChoices(
                finish_reason=None,
                delta=Delta(
                    role="assistant",
                    content="zero, ",
                ),
            )
        ],
    ),
    ModelResponse(
        model="test_model",
        choices=[
            StreamingChoices(
                finish_reason=None,
                delta=Delta(
                    role="assistant",
                    content="one, ",
                ),
            )
        ],
    ),
    ModelResponse(
        model="test_model",
        choices=[
            StreamingChoices(
                finish_reason=None,
                delta=Delta(
                    role="assistant",
                    content="two:",
                ),
            )
        ],
    ),
    ModelResponse(
        model="test_model",
        choices=[
            StreamingChoices(
                finish_reason=None,
                delta=Delta(
                    role="assistant",
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            type="function",
                            id="test_tool_call_id",
                            function=Function(
                                name="test_function",
                                arguments='{"test_arg": "test_',
                            ),
                            index=0,
                        )
                    ],
                ),
            )
        ],
    ),
    ModelResponse(
        model="test_model",
        choices=[
            StreamingChoices(
                finish_reason=None,
                delta=Delta(
                    role="assistant",
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            type="function",
                            id=None,
                            function=Function(
                                name=None,
                                arguments='value"}',
                            ),
                            index=0,
                        )
                    ],
                ),
            )
        ],
    ),
    ModelResponse(
        model="test_model",
        choices=[
            StreamingChoices(
                finish_reason="tool_use",
            )
        ],
    ),
]


class _StructuredOutput(BaseModel):
  value: int = Field(description="Value to emit")


class _ModelDumpOnly:
  """Test helper that mimics objects exposing only model_dump."""

  def __init__(self):
    self._schema = {
        "type": "object",
        "properties": {"foo": {"type": "string"}},
    }

  def model_dump(self, *, exclude_none=True, mode="json"):
    # The method signature matches pydantic BaseModel.model_dump to simulate
    # google.genai schema-like objects.
    del exclude_none
    del mode
    return self._schema


async def test_get_completion_inputs_formats_pydantic_schema_for_litellm():
  llm_request = LlmRequest(
      config=types.GenerateContentConfig(response_schema=_StructuredOutput)
  )

  _, _, response_format, _ = await _get_completion_inputs(
      llm_request, model="gemini/gemini-2.0-flash"
  )

  assert response_format == {
      "type": "json_object",
      "response_schema": _StructuredOutput.model_json_schema(),
  }


def test_to_litellm_response_format_passes_preformatted_dict():
  response_format = {
      "type": "json_object",
      "response_schema": {
          "type": "object",
          "properties": {"foo": {"type": "string"}},
      },
  }

  assert (
      _to_litellm_response_format(
          response_format, model="gemini/gemini-2.0-flash"
      )
      == response_format
  )


def test_to_litellm_response_format_wraps_json_schema_dict():
  schema = {
      "type": "object",
      "properties": {"foo": {"type": "string"}},
  }

  formatted = _to_litellm_response_format(
      schema, model="gemini/gemini-2.0-flash"
  )
  assert formatted["type"] == "json_object"
  assert formatted["response_schema"] == schema


def test_to_litellm_response_format_handles_model_dump_object():
  schema_obj = _ModelDumpOnly()

  formatted = _to_litellm_response_format(
      schema_obj, model="gemini/gemini-2.0-flash"
  )

  assert formatted["type"] == "json_object"
  assert formatted["response_schema"] == schema_obj.model_dump()


def test_to_litellm_response_format_handles_genai_schema_instance():
  schema_instance = types.Schema(
      type=types.Type.OBJECT,
      properties={"foo": types.Schema(type=types.Type.STRING)},
      required=["foo"],
  )

  formatted = _to_litellm_response_format(
      schema_instance, model="gemini/gemini-2.0-flash"
  )
  assert formatted["type"] == "json_object"
  assert formatted["response_schema"] == schema_instance.model_dump(
      exclude_none=True, mode="json"
  )


def test_to_litellm_response_format_uses_json_schema_for_openai_model():
  """Test that OpenAI models use json_schema format instead of response_schema."""
  formatted = _to_litellm_response_format(
      _StructuredOutput, model="gpt-4o-mini"
  )

  assert formatted["type"] == "json_schema"
  assert "json_schema" in formatted
  assert formatted["json_schema"]["name"] == "_StructuredOutput"
  assert formatted["json_schema"]["strict"] is True
  assert formatted["json_schema"]["schema"]["additionalProperties"] is False
  assert "additionalProperties" in formatted["json_schema"]["schema"]


def test_to_litellm_response_format_uses_response_schema_for_gemini_model():
  """Test that Gemini models continue to use response_schema format."""
  formatted = _to_litellm_response_format(
      _StructuredOutput, model="gemini/gemini-2.0-flash"
  )

  assert formatted["type"] == "json_object"
  assert "response_schema" in formatted
  assert formatted["response_schema"] == _StructuredOutput.model_json_schema()


def test_to_litellm_response_format_uses_response_schema_for_vertex_gemini():
  """Test that Vertex AI Gemini models use response_schema format."""
  formatted = _to_litellm_response_format(
      _StructuredOutput, model="vertex_ai/gemini-2.0-flash"
  )

  assert formatted["type"] == "json_object"
  assert "response_schema" in formatted
  assert formatted["response_schema"] == _StructuredOutput.model_json_schema()


def test_to_litellm_response_format_uses_json_schema_for_azure_openai():
  """Test that Azure OpenAI models use json_schema format."""
  formatted = _to_litellm_response_format(
      _StructuredOutput, model="azure/gpt-4o"
  )

  assert formatted["type"] == "json_schema"
  assert "json_schema" in formatted
  assert formatted["json_schema"]["name"] == "_StructuredOutput"
  assert formatted["json_schema"]["strict"] is True
  assert formatted["json_schema"]["schema"]["additionalProperties"] is False
  assert "additionalProperties" in formatted["json_schema"]["schema"]


def test_to_litellm_response_format_uses_json_schema_for_anthropic():
  """Test that Anthropic models use json_schema format."""
  formatted = _to_litellm_response_format(
      _StructuredOutput, model="anthropic/claude-3-5-sonnet"
  )

  assert formatted["type"] == "json_schema"
  assert "json_schema" in formatted
  assert formatted["json_schema"]["name"] == "_StructuredOutput"
  assert formatted["json_schema"]["strict"] is True
  assert formatted["json_schema"]["schema"]["additionalProperties"] is False
  assert "additionalProperties" in formatted["json_schema"]["schema"]


def test_to_litellm_response_format_with_dict_schema_for_openai():
  """Test dict schema with OpenAI model uses json_schema format."""
  schema = {
      "type": "object",
      "properties": {"foo": {"type": "string"}},
  }

  formatted = _to_litellm_response_format(schema, model="gpt-4o")

  assert formatted["type"] == "json_schema"
  assert formatted["json_schema"]["name"] == "response"
  assert formatted["json_schema"]["strict"] is True
  assert formatted["json_schema"]["schema"]["additionalProperties"] is False


async def test_get_completion_inputs_uses_openai_format_for_openai_model():
  """Test that _get_completion_inputs produces OpenAI-compatible format."""
  llm_request = LlmRequest(
      model="gpt-4o-mini",
      config=types.GenerateContentConfig(response_schema=_StructuredOutput),
  )

  _, _, response_format, _ = await _get_completion_inputs(
      llm_request, model="gpt-4o-mini"
  )

  assert response_format["type"] == "json_schema"
  assert "json_schema" in response_format
  assert response_format["json_schema"]["name"] == "_StructuredOutput"
  assert response_format["json_schema"]["strict"] is True
  assert (
      response_format["json_schema"]["schema"]["additionalProperties"] is False
  )


async def test_get_completion_inputs_uses_gemini_format_for_gemini_model():
  """Test that _get_completion_inputs produces Gemini-compatible format."""
  llm_request = LlmRequest(
      model="gemini/gemini-2.0-flash",
      config=types.GenerateContentConfig(response_schema=_StructuredOutput),
  )

  _, _, response_format, _ = await _get_completion_inputs(
      llm_request, model="gemini/gemini-2.0-flash"
  )

  assert response_format["type"] == "json_object"
  assert "response_schema" in response_format


async def test_get_completion_inputs_uses_passed_model_for_response_format():
  """Test that _get_completion_inputs uses the passed model parameter for response format.

  This verifies that when llm_request.model is None, the explicit model parameter
  is used to determine the correct response format (Gemini vs OpenAI).
  """
  llm_request = LlmRequest(
      model=None,  # No model in request
      config=types.GenerateContentConfig(response_schema=_StructuredOutput),
  )

  # Pass OpenAI model explicitly - should use json_schema format
  _, _, response_format, _ = await _get_completion_inputs(
      llm_request, model="gpt-4o-mini"
  )

  assert response_format["type"] == "json_schema"
  assert "json_schema" in response_format
  assert response_format["json_schema"]["name"] == "_StructuredOutput"
  assert response_format["json_schema"]["strict"] is True
  assert (
      response_format["json_schema"]["schema"]["additionalProperties"] is False
  )


async def test_get_completion_inputs_uses_passed_model_for_gemini_format():
  """Test that _get_completion_inputs uses passed model for Gemini response format.

  This verifies that when self.model is a Gemini model and passed explicitly,
  the response format uses the Gemini-specific format.
  """
  llm_request = LlmRequest(
      model=None,  # No model in request
      config=types.GenerateContentConfig(response_schema=_StructuredOutput),
  )

  # Pass Gemini model explicitly - should use response_schema format
  _, _, response_format, _ = await _get_completion_inputs(
      llm_request, model="gemini/gemini-2.0-flash"
  )

  assert response_format["type"] == "json_object"
  assert "response_schema" in response_format


@pytest.mark.asyncio
async def test_get_completion_inputs_inserts_missing_tool_results():
  user_content = types.Content(
      role="user", parts=[types.Part.from_text(text="Hi")]
  )
  assistant_content = types.Content(
      role="assistant",
      parts=[
          types.Part.from_text(text="Calling tool."),
          types.Part.from_function_call(
              name="get_weather", args={"location": "Seoul"}
          ),
      ],
  )
  assistant_content.parts[1].function_call.id = "tool_call_1"
  followup_user = types.Content(
      role="user", parts=[types.Part.from_text(text="Next question.")]
  )

  llm_request = LlmRequest(
      contents=[user_content, assistant_content, followup_user]
  )
  messages, _, _, _ = await _get_completion_inputs(
      llm_request, model="openai/gpt-4o"
  )

  assert [message["role"] for message in messages] == [
      "user",
      "assistant",
      "tool",
      "user",
  ]
  tool_message = messages[2]
  assert tool_message["tool_call_id"] == "tool_call_1"
  assert tool_message["content"] == _MISSING_TOOL_RESULT_MESSAGE


def test_schema_to_dict_filters_none_enum_values():
  # Use model_construct to bypass strict enum validation.
  top_level_schema = types.Schema.model_construct(
      type=types.Type.STRING,
      enum=["ACTIVE", None, "INACTIVE"],
  )
  nested_schema = types.Schema.model_construct(
      type=types.Type.OBJECT,
      properties={
          "status": types.Schema.model_construct(
              type=types.Type.STRING, enum=["READY", None, "DONE"]
          ),
      },
  )

  assert _schema_to_dict(top_level_schema)["enum"] == ["ACTIVE", "INACTIVE"]
  assert _schema_to_dict(nested_schema)["properties"]["status"]["enum"] == [
      "READY",
      "DONE",
  ]


MULTIPLE_FUNCTION_CALLS_STREAM = [
    ModelResponse(
        choices=[
            StreamingChoices(
                finish_reason=None,
                delta=Delta(
                    role="assistant",
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            type="function",
                            id="call_1",
                            function=Function(
                                name="function_1",
                                arguments='{"arg": "val',
                            ),
                            index=0,
                        )
                    ],
                ),
            )
        ]
    ),
    ModelResponse(
        choices=[
            StreamingChoices(
                finish_reason=None,
                delta=Delta(
                    role="assistant",
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            type="function",
                            id=None,
                            function=Function(
                                name=None,
                                arguments='ue1"}',
                            ),
                            index=0,
                        )
                    ],
                ),
            )
        ]
    ),
    ModelResponse(
        choices=[
            StreamingChoices(
                finish_reason=None,
                delta=Delta(
                    role="assistant",
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            type="function",
                            id="call_2",
                            function=Function(
                                name="function_2",
                                arguments='{"arg": "val',
                            ),
                            index=1,
                        )
                    ],
                ),
            )
        ]
    ),
    ModelResponse(
        choices=[
            StreamingChoices(
                finish_reason=None,
                delta=Delta(
                    role="assistant",
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            type="function",
                            id=None,
                            function=Function(
                                name=None,
                                arguments='ue2"}',
                            ),
                            index=1,
                        )
                    ],
                ),
            )
        ]
    ),
    ModelResponse(
        choices=[
            StreamingChoices(
                finish_reason="tool_calls",
            )
        ]
    ),
]


STREAM_WITH_EMPTY_CHUNK = [
    ModelResponse(
        choices=[
            StreamingChoices(
                finish_reason=None,
                delta=Delta(
                    role="assistant",
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            type="function",
                            id="call_abc",
                            function=Function(
                                name="test_function",
                                arguments='{"test_arg":',
                            ),
                            index=0,
                        )
                    ],
                ),
            )
        ]
    ),
    ModelResponse(
        choices=[
            StreamingChoices(
                finish_reason=None,
                delta=Delta(
                    role="assistant",
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            type="function",
                            id=None,
                            function=Function(
                                name=None,
                                arguments=' "value"}',
                            ),
                            index=0,
                        )
                    ],
                ),
            )
        ]
    ),
    # This is the problematic empty chunk that should be ignored.
    ModelResponse(
        choices=[
            StreamingChoices(
                finish_reason=None,
                delta=Delta(
                    role="assistant",
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            type="function",
                            id=None,
                            function=Function(
                                name=None,
                                arguments="",
                            ),
                            index=0,
                        )
                    ],
                ),
            )
        ]
    ),
    ModelResponse(
        choices=[StreamingChoices(finish_reason="tool_calls", delta=Delta())]
    ),
]


@pytest.fixture
def mock_response():
  return ModelResponse(
      model="test_model",
      choices=[
          Choices(
              message=ChatCompletionAssistantMessage(
                  role="assistant",
                  content="Test response",
                  tool_calls=[
                      ChatCompletionMessageToolCall(
                          type="function",
                          id="test_tool_call_id",
                          function=Function(
                              name="test_function",
                              arguments='{"test_arg": "test_value"}',
                          ),
                      )
                  ],
              )
          )
      ],
  )


# Test case reflecting litellm v1.71.2, ollama v0.9.0 streaming response
# no tool call ids
# indices all 0
# finish_reason stop instead of tool_calls
NON_COMPLIANT_MULTIPLE_FUNCTION_CALLS_STREAM = [
    ModelResponse(
        choices=[
            StreamingChoices(
                finish_reason=None,
                delta=Delta(
                    role="assistant",
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            type="function",
                            id=None,
                            function=Function(
                                name="function_1",
                                arguments='{"arg": "val',
                            ),
                            index=0,
                        )
                    ],
                ),
            )
        ]
    ),
    ModelResponse(
        choices=[
            StreamingChoices(
                finish_reason=None,
                delta=Delta(
                    role="assistant",
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            type="function",
                            id=None,
                            function=Function(
                                name=None,
                                arguments='ue1"}',
                            ),
                            index=0,
                        )
                    ],
                ),
            )
        ]
    ),
    ModelResponse(
        choices=[
            StreamingChoices(
                finish_reason=None,
                delta=Delta(
                    role="assistant",
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            type="function",
                            id=None,
                            function=Function(
                                name="function_2",
                                arguments='{"arg": "val',
                            ),
                            index=0,
                        )
                    ],
                ),
            )
        ]
    ),
    ModelResponse(
        choices=[
            StreamingChoices(
                finish_reason=None,
                delta=Delta(
                    role="assistant",
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            type="function",
                            id=None,
                            function=Function(
                                name=None,
                                arguments='ue2"}',
                            ),
                            index=0,
                        )
                    ],
                ),
            )
        ]
    ),
    ModelResponse(
        choices=[
            StreamingChoices(
                finish_reason="stop",
            )
        ]
    ),
]


@pytest.fixture
def mock_acompletion(mock_response):
  return AsyncMock(return_value=mock_response)


@pytest.fixture
def mock_completion(mock_response):
  return Mock(return_value=mock_response)


@pytest.fixture
def mock_client(mock_acompletion, mock_completion):
  return MockLLMClient(mock_acompletion, mock_completion)


@pytest.fixture
def lite_llm_instance(mock_client):
  return LiteLlm(model="test_model", llm_client=mock_client)


class MockLLMClient(LiteLLMClient):

  def __init__(self, acompletion_mock, completion_mock):
    self.acompletion_mock = acompletion_mock
    self.completion_mock = completion_mock

  async def acompletion(self, model, messages, tools, **kwargs):
    if kwargs.get("stream", False):
      kwargs_copy = dict(kwargs)
      kwargs_copy.pop("stream", None)

      async def stream_generator():
        stream_data = self.completion_mock(
            model=model,
            messages=messages,
            tools=tools,
            stream=True,
            **kwargs_copy,
        )
        for item in stream_data:
          yield item

      return stream_generator()
    else:
      return await self.acompletion_mock(
          model=model, messages=messages, tools=tools, **kwargs
      )

  def completion(self, model, messages, tools, stream, **kwargs):
    return self.completion_mock(
        model=model, messages=messages, tools=tools, stream=stream, **kwargs
    )


@pytest.mark.asyncio
async def test_generate_content_async(mock_acompletion, lite_llm_instance):

  async for response in lite_llm_instance.generate_content_async(
      LLM_REQUEST_WITH_FUNCTION_DECLARATION
  ):
    assert response.content.role == "model"
    assert response.content.parts[0].text == "Test response"
    assert response.content.parts[1].function_call.name == "test_function"
    assert response.content.parts[1].function_call.args == {
        "test_arg": "test_value"
    }
    assert response.content.parts[1].function_call.id == "test_tool_call_id"
    assert response.model_version == "test_model"

  mock_acompletion.assert_called_once()

  _, kwargs = mock_acompletion.call_args
  assert kwargs["model"] == "test_model"
  assert kwargs["messages"][0]["role"] == "user"
  assert kwargs["messages"][0]["content"] == "Test prompt"
  assert kwargs["tools"][0]["function"]["name"] == "test_function"
  assert (
      kwargs["tools"][0]["function"]["description"]
      == "Test function description"
  )
  assert (
      kwargs["tools"][0]["function"]["parameters"]["properties"]["test_arg"][
          "type"
      ]
      == "string"
  )


@pytest.mark.asyncio
async def test_generate_content_async_with_model_override(
    mock_acompletion, lite_llm_instance
):
  llm_request = LlmRequest(
      model="overridden_model",
      contents=[
          types.Content(
              role="user", parts=[types.Part.from_text(text="Test prompt")]
          )
      ],
  )

  async for response in lite_llm_instance.generate_content_async(llm_request):
    assert response.content.role == "model"
    assert response.content.parts[0].text == "Test response"

  mock_acompletion.assert_called_once()

  _, kwargs = mock_acompletion.call_args
  assert kwargs["model"] == "overridden_model"
  assert kwargs["messages"][0]["role"] == "user"
  assert kwargs["messages"][0]["content"] == "Test prompt"


@pytest.mark.asyncio
async def test_generate_content_async_without_model_override(
    mock_acompletion, lite_llm_instance
):
  llm_request = LlmRequest(
      model=None,
      contents=[
          types.Content(
              role="user", parts=[types.Part.from_text(text="Test prompt")]
          )
      ],
  )

  async for response in lite_llm_instance.generate_content_async(llm_request):
    assert response.content.role == "model"

  mock_acompletion.assert_called_once()

  _, kwargs = mock_acompletion.call_args
  assert kwargs["model"] == "test_model"


@pytest.mark.asyncio
async def test_generate_content_async_adds_fallback_user_message(
    mock_acompletion, lite_llm_instance
):
  llm_request = LlmRequest(
      contents=[
          types.Content(
              role="user",
              parts=[],
          )
      ]
  )

  async for _ in lite_llm_instance.generate_content_async(llm_request):
    pass

  mock_acompletion.assert_called_once()

  _, kwargs = mock_acompletion.call_args
  user_messages = [
      message for message in kwargs["messages"] if message["role"] == "user"
  ]
  assert any(
      message.get("content")
      == "Handle the requests as specified in the System Instruction."
      for message in user_messages
  )
  assert (
      sum(1 for content in llm_request.contents if content.role == "user") == 1
  )
  assert llm_request.contents[-1].parts[0].text == (
      "Handle the requests as specified in the System Instruction."
  )


litellm_append_user_content_test_cases = [
    pytest.param(
        LlmRequest(
            contents=[
                types.Content(
                    role="developer",
                    parts=[types.Part.from_text(text="Test prompt")],
                )
            ]
        ),
        2,
        id="litellm request without user content",
    ),
    pytest.param(
        LlmRequest(
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="user prompt")],
                )
            ]
        ),
        1,
        id="litellm request with user content",
    ),
    pytest.param(
        LlmRequest(
            contents=[
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="model prompt")],
                ),
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="user prompt")],
                ),
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="model prompt")],
                ),
            ]
        ),
        4,
        id="user content is not the last message scenario",
    ),
]


@pytest.mark.parametrize(
    "llm_request, expected_output", litellm_append_user_content_test_cases
)
def test_maybe_append_user_content(
    lite_llm_instance, llm_request, expected_output
):

  lite_llm_instance._maybe_append_user_content(llm_request)

  assert len(llm_request.contents) == expected_output


function_declaration_test_cases = [
    (
        "simple_function",
        types.FunctionDeclaration(
            name="test_function",
            description="Test function description",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "test_arg": types.Schema(type=types.Type.STRING),
                    "array_arg": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(
                            type=types.Type.STRING,
                        ),
                    ),
                    "nested_arg": types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "nested_key1": types.Schema(type=types.Type.STRING),
                            "nested_key2": types.Schema(type=types.Type.STRING),
                        },
                        required=["nested_key1"],
                    ),
                },
                required=["nested_arg"],
            ),
        ),
        {
            "type": "function",
            "function": {
                "name": "test_function",
                "description": "Test function description",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "test_arg": {"type": "string"},
                        "array_arg": {
                            "items": {"type": "string"},
                            "type": "array",
                        },
                        "nested_arg": {
                            "properties": {
                                "nested_key1": {"type": "string"},
                                "nested_key2": {"type": "string"},
                            },
                            "type": "object",
                            "required": ["nested_key1"],
                        },
                    },
                    "required": ["nested_arg"],
                },
            },
        },
    ),
    (
        "no_description",
        types.FunctionDeclaration(
            name="test_function_no_description",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "test_arg": types.Schema(type=types.Type.STRING),
                },
            ),
        ),
        {
            "type": "function",
            "function": {
                "name": "test_function_no_description",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "test_arg": {"type": "string"},
                    },
                },
            },
        },
    ),
    (
        "empty_parameters",
        types.FunctionDeclaration(
            name="test_function_empty_params",
            parameters=types.Schema(type=types.Type.OBJECT, properties={}),
        ),
        {
            "type": "function",
            "function": {
                "name": "test_function_empty_params",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
    ),
    (
        "nested_array",
        types.FunctionDeclaration(
            name="test_function_nested_array",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "array_arg": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "nested_key": types.Schema(
                                    type=types.Type.STRING
                                )
                            },
                        ),
                    ),
                },
            ),
        ),
        {
            "type": "function",
            "function": {
                "name": "test_function_nested_array",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "array_arg": {
                            "items": {
                                "properties": {
                                    "nested_key": {"type": "string"}
                                },
                                "type": "object",
                            },
                            "type": "array",
                        },
                    },
                },
            },
        },
    ),
    (
        "nested_properties",
        types.FunctionDeclaration(
            name="test_function_nested_properties",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "array_arg": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "nested_key": types.Schema(
                                    type=types.Type.OBJECT,
                                    properties={
                                        "inner_key": types.Schema(
                                            type=types.Type.STRING,
                                        )
                                    },
                                )
                            },
                        ),
                    ),
                },
            ),
        ),
        {
            "type": "function",
            "function": {
                "name": "test_function_nested_properties",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "array_arg": {
                            "items": {
                                "type": "object",
                                "properties": {
                                    "nested_key": {
                                        "type": "object",
                                        "properties": {
                                            "inner_key": {"type": "string"},
                                        },
                                    },
                                },
                            },
                            "type": "array",
                        },
                    },
                },
            },
        },
    ),
    (
        "no_parameters",
        types.FunctionDeclaration(
            name="test_function_no_params",
            description="Test function with no parameters",
        ),
        {
            "type": "function",
            "function": {
                "name": "test_function_no_params",
                "description": "Test function with no parameters",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
    ),
    (
        "parameters_without_required",
        types.FunctionDeclaration(
            name="test_function_no_required",
            description="Test function with parameters but no required field",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "optional_arg": types.Schema(type=types.Type.STRING),
                },
            ),
        ),
        {
            "type": "function",
            "function": {
                "name": "test_function_no_required",
                "description": (
                    "Test function with parameters but no required field"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "optional_arg": {"type": "string"},
                    },
                },
            },
        },
    ),
]


@pytest.mark.parametrize(
    "_, function_declaration, expected_output",
    function_declaration_test_cases,
    ids=[case[0] for case in function_declaration_test_cases],
)
def test_function_declaration_to_tool_param(
    _, function_declaration, expected_output
):
  assert (
      _function_declaration_to_tool_param(function_declaration)
      == expected_output
  )


def test_function_declaration_to_tool_param_without_required_attribute():
  """Ensure tools without a required field attribute don't raise errors."""

  class SchemaWithoutRequired:
    """Mimics a Schema object that lacks the required attribute."""

    def __init__(self):
      self.properties = {
          "optional_arg": types.Schema(type=types.Type.STRING),
      }

  func_decl = types.FunctionDeclaration(
      name="function_without_required_attr",
      description="Function missing required attribute",
  )
  func_decl.parameters = SchemaWithoutRequired()

  expected = {
      "type": "function",
      "function": {
          "name": "function_without_required_attr",
          "description": "Function missing required attribute",
          "parameters": {
              "type": "object",
              "properties": {
                  "optional_arg": {"type": "string"},
              },
          },
      },
  }

  assert _function_declaration_to_tool_param(func_decl) == expected


def test_function_declaration_to_tool_param_with_parameters_json_schema():
  """Ensure function declarations using parameters_json_schema are handled.

  This verifies that when a FunctionDeclaration includes a raw
  `parameters_json_schema` dict, it is used directly as the function
  parameters in the resulting tool param.
  """

  func_decl = types.FunctionDeclaration(
      name="fn_with_json",
      description="desc",
      parameters_json_schema={
          "type": "object",
          "properties": {
              "a": {"type": "string"},
              "b": {"type": "array", "items": {"type": "string"}},
          },
          "required": ["a"],
      },
  )

  expected = {
      "type": "function",
      "function": {
          "name": "fn_with_json",
          "description": "desc",
          "parameters": {
              "type": "object",
              "properties": {
                  "a": {"type": "string"},
                  "b": {"type": "array", "items": {"type": "string"}},
              },
              "required": ["a"],
          },
      },
  }

  assert _function_declaration_to_tool_param(func_decl) == expected


@pytest.mark.asyncio
async def test_generate_content_async_with_system_instruction(
    lite_llm_instance, mock_acompletion
):
  mock_response_with_system_instruction = ModelResponse(
      choices=[
          Choices(
              message=ChatCompletionAssistantMessage(
                  role="assistant",
                  content="Test response",
              )
          )
      ]
  )
  mock_acompletion.return_value = mock_response_with_system_instruction

  llm_request = LlmRequest(
      contents=[
          types.Content(
              role="user", parts=[types.Part.from_text(text="Test prompt")]
          )
      ],
      config=types.GenerateContentConfig(
          system_instruction="Test system instruction"
      ),
  )

  async for response in lite_llm_instance.generate_content_async(llm_request):
    assert response.content.role == "model"
    assert response.content.parts[0].text == "Test response"

  mock_acompletion.assert_called_once()

  _, kwargs = mock_acompletion.call_args
  assert kwargs["model"] == "test_model"
  assert kwargs["messages"][0]["role"] == "system"
  assert kwargs["messages"][0]["content"] == "Test system instruction"
  assert kwargs["messages"][1]["role"] == "user"
  assert kwargs["messages"][1]["content"] == "Test prompt"


@pytest.mark.asyncio
async def test_generate_content_async_with_tool_response(
    lite_llm_instance, mock_acompletion
):
  mock_response_with_tool_response = ModelResponse(
      choices=[
          Choices(
              message=ChatCompletionAssistantMessage(
                  role="tool",
                  content='{"result": "test_result"}',
                  tool_call_id="test_tool_call_id",
              )
          )
      ]
  )
  mock_acompletion.return_value = mock_response_with_tool_response

  llm_request = LlmRequest(
      contents=[
          types.Content(
              role="user", parts=[types.Part.from_text(text="Test prompt")]
          ),
          types.Content(
              role="tool",
              parts=[
                  types.Part.from_function_response(
                      name="test_function",
                      response={"result": "test_result"},
                  )
              ],
          ),
      ],
      config=types.GenerateContentConfig(
          system_instruction="test instruction",
      ),
  )
  async for response in lite_llm_instance.generate_content_async(llm_request):
    assert response.content.role == "model"
    assert response.content.parts[0].text == '{"result": "test_result"}'

  mock_acompletion.assert_called_once()

  _, kwargs = mock_acompletion.call_args
  assert kwargs["model"] == "test_model"

  assert kwargs["messages"][2]["role"] == "tool"
  assert kwargs["messages"][2]["content"] == '{"result": "test_result"}'


@pytest.mark.asyncio
async def test_generate_content_async_with_usage_metadata(
    lite_llm_instance, mock_acompletion
):
  mock_response_with_usage_metadata = ModelResponse(
      choices=[
          Choices(
              message=ChatCompletionAssistantMessage(
                  role="assistant",
                  content="Test response",
              )
          )
      ],
      usage={
          "prompt_tokens": 10,
          "completion_tokens": 5,
          "total_tokens": 15,
          "cached_tokens": 8,
      },
  )
  mock_acompletion.return_value = mock_response_with_usage_metadata

  llm_request = LlmRequest(
      contents=[
          types.Content(
              role="user", parts=[types.Part.from_text(text="Test prompt")]
          ),
      ],
      config=types.GenerateContentConfig(
          system_instruction="test instruction",
      ),
  )
  async for response in lite_llm_instance.generate_content_async(llm_request):
    assert response.content.role == "model"
    assert response.content.parts[0].text == "Test response"
    assert response.usage_metadata.prompt_token_count == 10
    assert response.usage_metadata.candidates_token_count == 5
    assert response.usage_metadata.total_token_count == 15
    assert response.usage_metadata.cached_content_token_count == 8

  mock_acompletion.assert_called_once()


@pytest.mark.asyncio
async def test_generate_content_async_ollama_chat_flattens_content(
    mock_acompletion, mock_completion
):
  llm_client = MockLLMClient(mock_acompletion, mock_completion)
  lite_llm_instance = LiteLlm(
      model="ollama_chat/qwen2.5:7b", llm_client=llm_client
  )
  llm_request = LlmRequest(
      contents=[
          types.Content(
              role="user",
              parts=[
                  types.Part.from_text(text="Describe this image."),
                  types.Part.from_bytes(
                      data=b"test_image", mime_type="image/png"
                  ),
              ],
          )
      ]
  )

  async for _ in lite_llm_instance.generate_content_async(llm_request):
    pass

  mock_acompletion.assert_called_once_with(
      model="ollama_chat/qwen2.5:7b",
      messages=ANY,
      tools=ANY,
      response_format=ANY,
  )
  _, kwargs = mock_acompletion.call_args
  message_content = kwargs["messages"][0]["content"]
  assert isinstance(message_content, str)
  assert "Describe this image." in message_content


@pytest.mark.asyncio
async def test_generate_content_async_custom_provider_flattens_content(
    mock_acompletion, mock_completion
):
  llm_client = MockLLMClient(mock_acompletion, mock_completion)
  lite_llm_instance = LiteLlm(
      model="qwen2.5:7b",
      llm_client=llm_client,
      custom_llm_provider="ollama_chat",
  )
  llm_request = LlmRequest(
      contents=[
          types.Content(
              role="user",
              parts=[
                  types.Part.from_text(text="Describe this image."),
                  types.Part.from_bytes(
                      data=b"test_image", mime_type="image/png"
                  ),
              ],
          )
      ]
  )

  async for _ in lite_llm_instance.generate_content_async(llm_request):
    pass

  mock_acompletion.assert_called_once()
  _, kwargs = mock_acompletion.call_args
  assert kwargs["custom_llm_provider"] == "ollama_chat"
  assert kwargs["model"] == "qwen2.5:7b"
  message_content = kwargs["messages"][0]["content"]
  assert isinstance(message_content, str)
  assert "Describe this image." in message_content


def test_flatten_ollama_content_accepts_tuple_blocks():
  from google.adk.models.lite_llm import _flatten_ollama_content

  content = (
      {"type": "text", "text": "first"},
      {"type": "text", "text": "second"},
  )
  flattened = _flatten_ollama_content(content)
  assert flattened == "first\nsecond"


@pytest.mark.parametrize(
    "content, expected",
    [
        (None, None),
        ("hello", "hello"),
        (
            [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ],
            "first\nsecond",
        ),
        (
            [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": "http://example.com"},
                },
            ],
            "Describe this image.",
        ),
    ],
)
def test_flatten_ollama_content_returns_str_or_none(content, expected):
  from google.adk.models.lite_llm import _flatten_ollama_content

  flattened = _flatten_ollama_content(content)
  assert flattened == expected
  assert flattened is None or isinstance(flattened, str)


def test_flatten_ollama_content_serializes_non_text_blocks_to_json():
  from google.adk.models.lite_llm import _flatten_ollama_content

  blocks = [
      {"type": "image_url", "image_url": {"url": "http://example.com"}},
  ]
  flattened = _flatten_ollama_content(blocks)
  assert isinstance(flattened, str)
  assert json.loads(flattened) == blocks


def test_flatten_ollama_content_serializes_dict_to_json():
  from google.adk.models.lite_llm import _flatten_ollama_content

  content = {"type": "image_url", "image_url": {"url": "http://example.com"}}
  flattened = _flatten_ollama_content(content)
  assert isinstance(flattened, str)
  assert json.loads(flattened) == content


@pytest.mark.asyncio
async def test_content_to_message_param_user_message():
  content = types.Content(
      role="user", parts=[types.Part.from_text(text="Test prompt")]
  )
  message = await _content_to_message_param(content)
  assert message["role"] == "user"
  assert message["content"] == "Test prompt"


@pytest.mark.asyncio
@pytest.mark.parametrize("file_uri,mime_type", FILE_URI_TEST_CASES)
async def test_content_to_message_param_user_message_with_file_uri(
    file_uri, mime_type
):
  file_part = types.Part.from_uri(file_uri=file_uri, mime_type=mime_type)
  content = types.Content(
      role="user",
      parts=[
          types.Part.from_text(text="Summarize this file."),
          file_part,
      ],
  )

  message = await _content_to_message_param(content)
  assert message == {
      "role": "user",
      "content": [
          {"type": "text", "text": "Summarize this file."},
          {"type": "file", "file": {"file_id": file_uri, "format": mime_type}},
      ],
  }


@pytest.mark.asyncio
@pytest.mark.parametrize("file_uri,mime_type", FILE_URI_TEST_CASES)
async def test_content_to_message_param_user_message_file_uri_only(
    file_uri, mime_type
):
  file_part = types.Part.from_uri(file_uri=file_uri, mime_type=mime_type)
  content = types.Content(
      role="user",
      parts=[
          file_part,
      ],
  )

  message = await _content_to_message_param(content)
  assert message == {
      "role": "user",
      "content": [
          {"type": "file", "file": {"file_id": file_uri, "format": mime_type}},
      ],
  }


@pytest.mark.asyncio
async def test_content_to_message_param_user_message_file_uri_without_mime_type():
  """Test handling of file_data without mime_type (GcsArtifactService scenario).

  When using GcsArtifactService, artifacts may have file_uri (gs://...) but
  without mime_type set. LiteLLM's Vertex AI backend requires the format
  field to be present, so we infer MIME type from the URI extension or use
  a default fallback to ensure compatibility.

  See: https://github.com/google/adk-python/issues/3787
  """
  file_part = types.Part(
      file_data=types.FileData(
          file_uri="gs://agent-artifact-bucket/app/user/session/artifact/0"
      )
  )
  content = types.Content(
      role="user",
      parts=[
          types.Part.from_text(text="Analyze this file."),
          file_part,
      ],
  )

  message = await _content_to_message_param(content)
  assert message == {
      "role": "user",
      "content": [
          {"type": "text", "text": "Analyze this file."},
          {
              "type": "file",
              "file": {
                  "file_id": (
                      "gs://agent-artifact-bucket/app/user/session/artifact/0"
                  ),
                  "format": "application/octet-stream",
              },
          },
      ],
  }


@pytest.mark.asyncio
async def test_content_to_message_param_user_message_file_uri_infer_mime_type():
  """Test MIME type inference from file_uri extension.

  When file_data has a file_uri with a recognizable extension but no explicit
  mime_type, the MIME type should be inferred from the extension.

  See: https://github.com/google/adk-python/issues/3787
  """
  file_part = types.Part(
      file_data=types.FileData(
          file_uri="gs://bucket/path/to/document.pdf",
      )
  )
  content = types.Content(
      role="user",
      parts=[file_part],
  )

  message = await _content_to_message_param(content)
  assert message == {
      "role": "user",
      "content": [
          {
              "type": "file",
              "file": {
                  "file_id": "gs://bucket/path/to/document.pdf",
                  "format": "application/pdf",
              },
          },
      ],
  }


@pytest.mark.asyncio
async def test_content_to_message_param_multi_part_function_response():
  part1 = types.Part.from_function_response(
      name="function_one",
      response={"result": "result_one"},
  )
  part1.function_response.id = "tool_call_1"

  part2 = types.Part.from_function_response(
      name="function_two",
      response={"value": 123},
  )
  part2.function_response.id = "tool_call_2"

  content = types.Content(
      role="tool",
      parts=[part1, part2],
  )
  messages = await _content_to_message_param(content)
  assert isinstance(messages, list)
  assert len(messages) == 2

  assert messages[0]["role"] == "tool"
  assert messages[0]["tool_call_id"] == "tool_call_1"
  assert messages[0]["content"] == '{"result": "result_one"}'

  assert messages[1]["role"] == "tool"
  assert messages[1]["tool_call_id"] == "tool_call_2"
  assert messages[1]["content"] == '{"value": 123}'


@pytest.mark.asyncio
async def test_content_to_message_param_function_response_with_extra_parts():
  tool_part = types.Part.from_function_response(
      name="load_image",
      response={"status": "success"},
  )
  tool_part.function_response.id = "tool_call_1"

  text_part = types.Part.from_text(text="[Image: img_123.png]")
  image_bytes = b"test_image_data"
  image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")

  content = types.Content(
      role="user",
      parts=[tool_part, text_part, image_part],
  )

  messages = await _content_to_message_param(content)
  assert isinstance(messages, list)
  assert messages == [
      {
          "role": "tool",
          "tool_call_id": "tool_call_1",
          "content": '{"status": "success"}',
      },
      {
          "role": "user",
          "content": [
              {"type": "text", "text": "[Image: img_123.png]"},
              {
                  "type": "image_url",
                  "image_url": {
                      "url": "data:image/png;base64,dGVzdF9pbWFnZV9kYXRh"
                  },
              },
          ],
      },
  ]


@pytest.mark.asyncio
async def test_content_to_message_param_function_response_preserves_string():
  """Tests that string responses are used directly without double-serialization.

  The google.genai FunctionResponse.response field is typed as dict, but
  _content_to_message_param defensively handles string responses to avoid
  double-serialization. This test verifies that behavior by mocking a
  function_response with a string response attribute.
  """
  response_payload = '{"type": "files", "count": 2}'

  # Create a Part with a dict response, then mock the response to be a string
  # to simulate edge cases where response might be set directly as a string
  part = types.Part.from_function_response(
      name="list_files",
      response={"placeholder": "will be mocked"},
  )

  # Mock the response attribute to return a string
  # Using Mock without spec_set to allow setting response to a string,
  # which simulates the edge case we're testing
  mock_function_response = Mock(spec=types.FunctionResponse)
  mock_function_response.response = response_payload
  mock_function_response.id = "tool_call_1"
  part.function_response = mock_function_response

  content = types.Content(
      role="tool",
      parts=[part],
  )
  message = await _content_to_message_param(content)

  assert message["role"] == "tool"
  assert message["tool_call_id"] == "tool_call_1"
  assert message["content"] == response_payload


@pytest.mark.asyncio
async def test_content_to_message_param_assistant_message():
  content = types.Content(
      role="assistant", parts=[types.Part.from_text(text="Test response")]
  )
  message = await _content_to_message_param(content)
  assert message["role"] == "assistant"
  assert message["content"] == "Test response"


@pytest.mark.asyncio
async def test_content_to_message_param_user_filters_thought_parts():
  thought_part = types.Part.from_text(text="internal reasoning")
  thought_part.thought = True
  content_part = types.Part.from_text(text="visible content")
  content = types.Content(role="user", parts=[thought_part, content_part])

  message = await _content_to_message_param(content)

  assert message["role"] == "user"
  assert message["content"] == "visible content"


@pytest.mark.asyncio
async def test_content_to_message_param_assistant_thought_message():
  part = types.Part.from_text(text="internal reasoning")
  part.thought = True
  content = types.Content(role="assistant", parts=[part])

  message = await _content_to_message_param(content)

  assert message["role"] == "assistant"
  assert message["content"] is None
  assert message["reasoning_content"] == "internal reasoning"


@pytest.mark.asyncio
async def test_content_to_message_param_model_thought_message():
  part = types.Part.from_text(text="internal reasoning")
  part.thought = True
  content = types.Content(role="model", parts=[part])

  message = await _content_to_message_param(content)

  assert message["role"] == "assistant"
  assert message["content"] is None
  assert message["reasoning_content"] == "internal reasoning"


@pytest.mark.asyncio
async def test_content_to_message_param_assistant_thought_and_content_message():
  thought_part = types.Part.from_text(text="internal reasoning")
  thought_part.thought = True
  content_part = types.Part.from_text(text="visible content")
  content = types.Content(role="assistant", parts=[thought_part, content_part])

  message = await _content_to_message_param(content)

  assert message["role"] == "assistant"
  assert message["content"] == "visible content"
  assert message["reasoning_content"] == "internal reasoning"


@pytest.mark.asyncio
async def test_content_to_message_param_function_call():
  content = types.Content(
      role="assistant",
      parts=[
          types.Part.from_text(text="test response"),
          types.Part.from_function_call(
              name="test_function", args={"test_arg": "test_value"}
          ),
      ],
  )
  content.parts[1].function_call.id = "test_tool_call_id"
  message = await _content_to_message_param(content)
  assert message["role"] == "assistant"
  assert message["content"] == "test response"

  tool_call = message["tool_calls"][0]
  assert tool_call["type"] == "function"
  assert tool_call["id"] == "test_tool_call_id"
  assert tool_call["function"]["name"] == "test_function"
  assert tool_call["function"]["arguments"] == '{"test_arg": "test_value"}'


@pytest.mark.asyncio
async def test_content_to_message_param_multipart_content():
  """Test handling of multipart content where final_content is a list with text objects."""
  content = types.Content(
      role="assistant",
      parts=[
          types.Part.from_text(text="text part"),
          types.Part.from_bytes(data=b"test_image_data", mime_type="image/png"),
      ],
  )
  message = await _content_to_message_param(content)
  assert message["role"] == "assistant"
  # When content is a list and the first element is a text object with type "text",
  # it should extract the text (for providers like ollama_chat that don't handle lists well)
  # This is the behavior implemented in the fix
  assert message["content"] == "text part"
  assert message["tool_calls"] is None


@pytest.mark.asyncio
async def test_content_to_message_param_single_text_object_in_list(mocker):
  """Test extraction of text from single text object in list (for ollama_chat compatibility)."""
  from google.adk.models import lite_llm

  # Mock _get_content to return a list with single text object
  async def mock_get_content(*args, **kwargs):
    return [{"type": "text", "text": "single text"}]

  mocker.patch.object(lite_llm, "_get_content", side_effect=mock_get_content)

  content = types.Content(
      role="assistant",
      parts=[types.Part.from_text(text="single text")],
  )
  message = await _content_to_message_param(content)
  assert message["role"] == "assistant"
  # Should extract the text from the single text object
  assert message["content"] == "single text"
  assert message["tool_calls"] is None


def test_message_to_generate_content_response_text():
  message = ChatCompletionAssistantMessage(
      role="assistant",
      content="Test response",
  )
  response = _message_to_generate_content_response(message)
  assert response.content.role == "model"
  assert response.content.parts[0].text == "Test response"


def test_message_to_generate_content_response_tool_call():
  message = ChatCompletionAssistantMessage(
      role="assistant",
      content=None,
      tool_calls=[
          ChatCompletionMessageToolCall(
              type="function",
              id="test_tool_call_id",
              function=Function(
                  name="test_function",
                  arguments='{"test_arg": "test_value"}',
              ),
          )
      ],
  )

  response = _message_to_generate_content_response(message)
  assert response.content.role == "model"
  assert response.content.parts[0].function_call.name == "test_function"
  assert response.content.parts[0].function_call.args == {
      "test_arg": "test_value"
  }
  assert response.content.parts[0].function_call.id == "test_tool_call_id"


def test_message_to_generate_content_response_inline_tool_call_text():
  message = ChatCompletionAssistantMessage(
      role="assistant",
      content=(
          '{"id":"inline_call","name":"get_current_time",'
          '"arguments":{"timezone_str":"Asia/Taipei"}} <|im_end|>system'
      ),
  )

  response = _message_to_generate_content_response(message)
  assert len(response.content.parts) == 2
  text_part = response.content.parts[0]
  tool_part = response.content.parts[1]
  assert text_part.text == "<|im_end|>system"
  assert tool_part.function_call.name == "get_current_time"
  assert tool_part.function_call.args == {"timezone_str": "Asia/Taipei"}
  assert tool_part.function_call.id == "inline_call"


def test_message_to_generate_content_response_with_model():
  message = ChatCompletionAssistantMessage(
      role="assistant",
      content="Test response",
  )
  response = _message_to_generate_content_response(
      message, model_version="gemini-2.5-pro"
  )
  assert response.content.role == "model"
  assert response.content.parts[0].text == "Test response"
  assert response.model_version == "gemini-2.5-pro"


def test_message_to_generate_content_response_reasoning_content():
  message = {
      "role": "assistant",
      "content": "Visible text",
      "reasoning_content": "Hidden chain",
  }
  response = _message_to_generate_content_response(message)

  assert len(response.content.parts) == 2
  thought_part = response.content.parts[0]
  text_part = response.content.parts[1]
  assert thought_part.text == "Hidden chain"
  assert thought_part.thought is True
  assert text_part.text == "Visible text"


def test_model_response_to_generate_content_response_reasoning_content():
  model_response = ModelResponse(
      model="thinking-model",
      choices=[{
          "message": {
              "role": "assistant",
              "content": "Answer",
              "reasoning_content": "Step-by-step",
          },
          "finish_reason": "stop",
      }],
  )

  response = _model_response_to_generate_content_response(model_response)

  assert response.content.parts[0].text == "Step-by-step"
  assert response.content.parts[0].thought is True
  assert response.content.parts[1].text == "Answer"


def test_parse_tool_calls_from_text_multiple_calls():
  text = (
      '{"name":"alpha","arguments":{"value":1}}\n'
      "Some filler text "
      '{"id":"custom","name":"beta","arguments":{"timezone":"Asia/Taipei"}} '
      "ignored suffix"
  )
  tool_calls, remainder = _parse_tool_calls_from_text(text)
  assert len(tool_calls) == 2
  assert tool_calls[0].function.name == "alpha"
  assert json.loads(tool_calls[0].function.arguments) == {"value": 1}
  assert tool_calls[1].id == "custom"
  assert tool_calls[1].function.name == "beta"
  assert json.loads(tool_calls[1].function.arguments) == {
      "timezone": "Asia/Taipei"
  }
  assert remainder == "Some filler text  ignored suffix"


def test_parse_tool_calls_from_text_invalid_json_returns_remainder():
  text = 'Leading {"unused": "payload"} trailing text'
  tool_calls, remainder = _parse_tool_calls_from_text(text)
  assert tool_calls == []
  assert remainder == 'Leading {"unused": "payload"} trailing text'


def test_split_message_content_and_tool_calls_inline_text():
  message = {
      "role": "assistant",
      "content": (
          'Intro {"name":"alpha","arguments":{"value":1}} trailing content'
      ),
  }
  content, tool_calls = _split_message_content_and_tool_calls(message)
  assert content == "Intro  trailing content"
  assert len(tool_calls) == 1
  assert tool_calls[0].function.name == "alpha"
  assert json.loads(tool_calls[0].function.arguments) == {"value": 1}


def test_split_message_content_prefers_existing_structured_calls():
  tool_call = ChatCompletionMessageToolCall(
      type="function",
      id="existing",
      function=Function(
          name="existing_call",
          arguments='{"arg": "value"}',
      ),
  )
  message = {
      "role": "assistant",
      "content": "ignored",
      "tool_calls": [tool_call],
  }
  content, tool_calls = _split_message_content_and_tool_calls(message)
  assert content == "ignored"
  assert tool_calls == [tool_call]


@pytest.mark.asyncio
async def test_get_content_does_not_filter_thought_parts():
  """Test that _get_content does not drop thought parts.

  Thought filtering is handled by the caller (e.g., _content_to_message_param)
  to avoid duplicating logic across helpers.
  """
  thought_part = types.Part(text="Internal reasoning...", thought=True)
  regular_part = types.Part.from_text(text="Visible response")

  content = await _get_content([thought_part, regular_part])

  assert content == [
      {"type": "text", "text": "Internal reasoning..."},
      {"type": "text", "text": "Visible response"},
  ]


@pytest.mark.asyncio
async def test_get_content_all_thought_parts():
  """Test that thought parts convert like regular text parts."""
  thought_part1 = types.Part(text="First reasoning...", thought=True)
  thought_part2 = types.Part(text="Second reasoning...", thought=True)

  content = await _get_content([thought_part1, thought_part2])

  assert content == [
      {"type": "text", "text": "First reasoning..."},
      {"type": "text", "text": "Second reasoning..."},
  ]


@pytest.mark.asyncio
async def test_get_content_text():
  parts = [types.Part.from_text(text="Test text")]
  content = await _get_content(parts)
  assert content == "Test text"


@pytest.mark.asyncio
async def test_get_content_text_inline_data_single_part():
  parts = [
      types.Part.from_bytes(
          data="Inline text".encode("utf-8"), mime_type="text/plain"
      )
  ]
  content = await _get_content(parts)
  assert content == "Inline text"


@pytest.mark.asyncio
async def test_get_content_text_inline_data_multiple_parts():
  parts = [
      types.Part.from_bytes(
          data="First part".encode("utf-8"), mime_type="text/plain"
      ),
      types.Part.from_text(text="Second part"),
  ]
  content = await _get_content(parts)
  assert content[0]["type"] == "text"
  assert content[0]["text"] == "First part"
  assert content[1]["type"] == "text"
  assert content[1]["text"] == "Second part"


@pytest.mark.asyncio
async def test_get_content_text_inline_data_fallback_decoding():
  parts = [
      types.Part.from_bytes(data=b"\xff", mime_type="text/plain"),
  ]
  content = await _get_content(parts)
  assert content == "ÿ"


@pytest.mark.asyncio
async def test_get_content_image():
  parts = [
      types.Part.from_bytes(data=b"test_image_data", mime_type="image/png")
  ]
  content = await _get_content(parts)
  assert content[0]["type"] == "image_url"
  assert (
      content[0]["image_url"]["url"]
      == "data:image/png;base64,dGVzdF9pbWFnZV9kYXRh"
  )
  assert "format" not in content[0]["image_url"]


@pytest.mark.asyncio
async def test_get_content_video():
  parts = [
      types.Part.from_bytes(data=b"test_video_data", mime_type="video/mp4")
  ]
  content = await _get_content(parts)
  assert content[0]["type"] == "video_url"
  assert (
      content[0]["video_url"]["url"]
      == "data:video/mp4;base64,dGVzdF92aWRlb19kYXRh"
  )
  assert "format" not in content[0]["video_url"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "file_data,mime_type,expected_base64", FILE_BYTES_TEST_CASES
)
async def test_get_content_file_bytes(file_data, mime_type, expected_base64):
  parts = [types.Part.from_bytes(data=file_data, mime_type=mime_type)]
  content = await _get_content(parts)
  assert content[0]["type"] == "file"
  assert content[0]["file"]["file_data"] == expected_base64
  assert "format" not in content[0]["file"]


@pytest.mark.asyncio
@pytest.mark.parametrize("file_uri,mime_type", FILE_URI_TEST_CASES)
async def test_get_content_file_uri(file_uri, mime_type):
  parts = [types.Part.from_uri(file_uri=file_uri, mime_type=mime_type)]
  content = await _get_content(parts)
  assert content[0] == {
      "type": "file",
      "file": {"file_id": file_uri, "format": mime_type},
  }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,model",
    [
        ("openai", "openai/gpt-4o"),
        ("azure", "azure/gpt-4"),
    ],
)
async def test_get_content_file_uri_file_id_required_falls_back_to_text(
    provider, model
):
  parts = [
      types.Part(
          file_data=types.FileData(
              file_uri="gs://bucket/path/to/document.pdf",
              mime_type="application/pdf",
              display_name="document.pdf",
          )
      )
  ]
  content = await _get_content(parts, provider=provider, model=model)
  assert content == [
      {"type": "text", "text": '[File reference: "document.pdf"]'}
  ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,model",
    [
        ("openai", "openai/gpt-4o"),
        ("azure", "azure/gpt-4"),
    ],
)
async def test_get_content_file_uri_file_id_required_preserves_file_id(
    provider, model
):
  parts = [
      types.Part(
          file_data=types.FileData(
              file_uri="file-abc123",
              mime_type="application/pdf",
          )
      )
  ]
  content = await _get_content(parts, provider=provider, model=model)
  assert content == [{"type": "file", "file": {"file_id": "file-abc123"}}]


@pytest.mark.asyncio
async def test_get_content_file_uri_anthropic_falls_back_to_text():
  parts = [
      types.Part(
          file_data=types.FileData(
              file_uri="gs://bucket/path/to/document.pdf",
              mime_type="application/pdf",
              display_name="document.pdf",
          )
      )
  ]
  content = await _get_content(
      parts, provider="anthropic", model="anthropic/claude-3-5"
  )
  assert content == [
      {"type": "text", "text": '[File reference: "document.pdf"]'}
  ]


@pytest.mark.asyncio
async def test_get_content_file_uri_anthropic_openai_file_id_falls_back_to_text():
  parts = [types.Part(file_data=types.FileData(file_uri="file-abc123"))]
  content = await _get_content(
      parts, provider="anthropic", model="anthropic/claude-3-5"
  )
  assert content == [
      {"type": "text", "text": '[File reference: "file-abc123"]'}
  ]


@pytest.mark.asyncio
async def test_get_content_file_uri_vertex_ai_non_gemini_falls_back_to_text():
  parts = [
      types.Part(
          file_data=types.FileData(
              file_uri="gs://bucket/path/to/document.pdf",
              mime_type="application/pdf",
              display_name="document.pdf",
          )
      )
  ]
  content = await _get_content(
      parts, provider="vertex_ai", model="vertex_ai/claude-3-5"
  )
  assert content == [
      {"type": "text", "text": '[File reference: "document.pdf"]'}
  ]


@pytest.mark.asyncio
async def test_get_content_file_uri_vertex_ai_gemini_keeps_file_block():
  parts = [
      types.Part(
          file_data=types.FileData(
              file_uri="gs://bucket/path/to/document.pdf",
              mime_type="application/pdf",
          )
      )
  ]
  content = await _get_content(
      parts, provider="vertex_ai", model="vertex_ai/gemini-2.5-flash"
  )
  assert content == [{
      "type": "file",
      "file": {
          "file_id": "gs://bucket/path/to/document.pdf",
          "format": "application/pdf",
      },
  }]


@pytest.mark.asyncio
async def test_get_content_file_uri_infer_mime_type():
  """Test MIME type inference from file_uri extension.

  When file_data has a file_uri with a recognizable extension but no explicit
  mime_type, the MIME type should be inferred from the extension.

  See: https://github.com/google/adk-python/issues/3787
  """
  # Use Part constructor directly to test MIME type inference in _get_content
  # (types.Part.from_uri does its own inference, so we bypass it)
  parts = [
      types.Part(
          file_data=types.FileData(file_uri="gs://bucket/path/to/document.pdf")
      )
  ]
  content = await _get_content(parts)
  assert content[0] == {
      "type": "file",
      "file": {
          "file_id": "gs://bucket/path/to/document.pdf",
          "format": "application/pdf",
      },
  }


@pytest.mark.asyncio
async def test_get_content_file_uri_versioned_infer_mime_type():
  """Test MIME type inference from versioned artifact URIs."""
  parts = [
      types.Part(
          file_data=types.FileData(
              file_uri="gs://bucket/path/to/document.pdf/0"
          )
      )
  ]
  content = await _get_content(parts)
  assert content[0]["file"]["format"] == "application/pdf"


@pytest.mark.asyncio
async def test_get_content_file_uri_infers_from_display_name():
  """Test MIME type inference from display_name when URI lacks extension."""
  parts = [
      types.Part(
          file_data=types.FileData(
              file_uri="gs://bucket/artifact/0",
              display_name="document.pdf",
          )
      )
  ]
  content = await _get_content(parts)
  assert content[0]["file"]["format"] == "application/pdf"


@pytest.mark.asyncio
async def test_get_content_file_uri_default_mime_type():
  """Test that file_uri without extension uses default MIME type.

  When file_data has a file_uri without a recognizable extension and no explicit
  mime_type, a default MIME type should be used to ensure compatibility with
  LiteLLM backends.

  See: https://github.com/google/adk-python/issues/3787
  """
  # Use Part constructor directly to create file_data without mime_type
  # (types.Part.from_uri requires a valid mime_type when it can't infer)
  parts = [
      types.Part(file_data=types.FileData(file_uri="gs://bucket/artifact/0"))
  ]
  content = await _get_content(parts)
  assert content[0] == {
      "type": "file",
      "file": {
          "file_id": "gs://bucket/artifact/0",
          "format": "application/octet-stream",
      },
  }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "uri,expected_mime_type",
    [
        ("gs://bucket/file.pdf", "application/pdf"),
        ("gs://bucket/path/to/document.json", "application/json"),
        ("gs://bucket/image.png", "image/png"),
        ("gs://bucket/image.jpg", "image/jpeg"),
        ("gs://bucket/audio.mp3", "audio/mpeg"),
        ("gs://bucket/video.mp4", "video/mp4"),
    ],
)
async def test_get_content_file_uri_mime_type_inference(
    uri, expected_mime_type
):
  """Test MIME type inference from various file extensions."""
  # Use Part constructor directly to test MIME type inference in _get_content
  parts = [types.Part(file_data=types.FileData(file_uri=uri))]
  content = await _get_content(parts)
  assert content[0]["file"]["format"] == expected_mime_type


@pytest.mark.asyncio
async def test_get_content_audio():
  parts = [
      types.Part.from_bytes(data=b"test_audio_data", mime_type="audio/mpeg")
  ]
  content = await _get_content(parts)
  assert content[0]["type"] == "audio_url"
  assert (
      content[0]["audio_url"]["url"]
      == "data:audio/mpeg;base64,dGVzdF9hdWRpb19kYXRh"
  )
  assert "format" not in content[0]["audio_url"]


def test_to_litellm_role():
  assert _to_litellm_role("model") == "assistant"
  assert _to_litellm_role("assistant") == "assistant"
  assert _to_litellm_role("user") == "user"
  assert _to_litellm_role(None) == "user"


@pytest.mark.parametrize(
    "response, expected_chunks, expected_usage_chunk, expected_finished",
    [
        (
            ModelResponse(
                choices=[
                    {
                        "message": {
                            "content": "this is a test",
                        }
                    }
                ]
            ),
            [TextChunk(text="this is a test")],
            UsageMetadataChunk(
                prompt_tokens=0, completion_tokens=0, total_tokens=0
            ),
            "stop",
        ),
        (
            ModelResponse(
                choices=[
                    {
                        "message": {
                            "content": "this is a test",
                        }
                    }
                ],
                usage={
                    "prompt_tokens": 3,
                    "completion_tokens": 5,
                    "total_tokens": 8,
                },
            ),
            [TextChunk(text="this is a test")],
            UsageMetadataChunk(
                prompt_tokens=3, completion_tokens=5, total_tokens=8
            ),
            "stop",
        ),
        (
            ModelResponse(
                choices=[
                    StreamingChoices(
                        finish_reason=None,
                        delta=Delta(
                            role="assistant",
                            tool_calls=[
                                ChatCompletionDeltaToolCall(
                                    type="function",
                                    id="1",
                                    function=Function(
                                        name="test_function",
                                        arguments='{"key": "va',
                                    ),
                                    index=0,
                                )
                            ],
                        ),
                    )
                ]
            ),
            [FunctionChunk(id="1", name="test_function", args='{"key": "va')],
            UsageMetadataChunk(
                prompt_tokens=0, completion_tokens=0, total_tokens=0
            ),
            None,
        ),
        (
            ModelResponse(choices=[{"finish_reason": "tool_calls"}]),
            [None],
            UsageMetadataChunk(
                prompt_tokens=0, completion_tokens=0, total_tokens=0
            ),
            "tool_calls",
        ),
        (
            ModelResponse(choices=[{}]),
            [None],
            UsageMetadataChunk(
                prompt_tokens=0, completion_tokens=0, total_tokens=0
            ),
            "stop",
        ),
        (
            ModelResponse(
                choices=[{
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": (
                            '{"id":"call_1","name":"get_current_time",'
                            '"arguments":{"timezone_str":"Asia/Taipei"}}'
                        ),
                    },
                }],
                usage={
                    "prompt_tokens": 7,
                    "completion_tokens": 9,
                    "total_tokens": 16,
                },
            ),
            [
                FunctionChunk(
                    id="call_1",
                    name="get_current_time",
                    args='{"timezone_str": "Asia/Taipei"}',
                    index=0,
                ),
            ],
            UsageMetadataChunk(
                prompt_tokens=7, completion_tokens=9, total_tokens=16
            ),
            "tool_calls",
        ),
        (
            ModelResponse(
                choices=[{
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": (
                            'Intro {"id":"call_2","name":"alpha",'
                            '"arguments":{"foo":"bar"}} wrap'
                        ),
                    },
                }],
                usage={
                    "prompt_tokens": 11,
                    "completion_tokens": 13,
                    "total_tokens": 24,
                },
            ),
            [
                TextChunk(text="Intro  wrap"),
                FunctionChunk(
                    id="call_2",
                    name="alpha",
                    args='{"foo": "bar"}',
                    index=0,
                ),
            ],
            UsageMetadataChunk(
                prompt_tokens=11, completion_tokens=13, total_tokens=24
            ),
            "tool_calls",
        ),
    ],
)
def test_model_response_to_chunk(
    response, expected_chunks, expected_usage_chunk, expected_finished
):
  result = list(_model_response_to_chunk(response))
  observed_chunks = []
  usage_chunk = None
  for chunk, finished in result:
    if isinstance(chunk, UsageMetadataChunk):
      usage_chunk = chunk
      continue
    observed_chunks.append((chunk, finished))

  assert len(observed_chunks) == len(expected_chunks)
  for (chunk, finished), expected_chunk in zip(
      observed_chunks, expected_chunks
  ):
    if expected_chunk is None:
      assert chunk is None
    else:
      assert isinstance(chunk, type(expected_chunk))
      assert chunk == expected_chunk
    assert finished == expected_finished

  if expected_usage_chunk is None:
    assert usage_chunk is None
  else:
    assert usage_chunk is not None
    assert usage_chunk == expected_usage_chunk


@pytest.mark.asyncio
async def test_acompletion_additional_args(mock_acompletion, mock_client):
  lite_llm_instance = LiteLlm(
      # valid args
      model="test_model",
      llm_client=mock_client,
      api_key="test_key",
      api_base="some://url",
      api_version="2024-09-12",
      # invalid args (ignored)
      stream=True,
      messages=[{"role": "invalid", "content": "invalid"}],
      tools=[{
          "type": "function",
          "function": {
              "name": "invalid",
          },
      }],
  )

  async for response in lite_llm_instance.generate_content_async(
      LLM_REQUEST_WITH_FUNCTION_DECLARATION
  ):
    assert response.content.role == "model"
    assert response.content.parts[0].text == "Test response"
    assert response.content.parts[1].function_call.name == "test_function"
    assert response.content.parts[1].function_call.args == {
        "test_arg": "test_value"
    }
    assert response.content.parts[1].function_call.id == "test_tool_call_id"

  mock_acompletion.assert_called_once()

  _, kwargs = mock_acompletion.call_args

  assert kwargs["model"] == "test_model"
  assert kwargs["messages"][0]["role"] == "user"
  assert kwargs["messages"][0]["content"] == "Test prompt"
  assert kwargs["tools"][0]["function"]["name"] == "test_function"
  assert "stream" not in kwargs
  assert "llm_client" not in kwargs
  assert kwargs["api_base"] == "some://url"


@pytest.mark.asyncio
async def test_acompletion_with_drop_params(mock_acompletion, mock_client):
  lite_llm_instance = LiteLlm(
      model="test_model", llm_client=mock_client, drop_params=True
  )

  async for _ in lite_llm_instance.generate_content_async(
      LLM_REQUEST_WITH_FUNCTION_DECLARATION
  ):
    pass

  mock_acompletion.assert_called_once()

  _, kwargs = mock_acompletion.call_args
  assert kwargs["drop_params"] is True


@pytest.mark.asyncio
async def test_completion_additional_args(mock_completion, mock_client):
  lite_llm_instance = LiteLlm(
      # valid args
      model="test_model",
      llm_client=mock_client,
      api_key="test_key",
      api_base="some://url",
      api_version="2024-09-12",
      # invalid args (ignored)
      stream=False,
      messages=[{"role": "invalid", "content": "invalid"}],
      tools=[{
          "type": "function",
          "function": {
              "name": "invalid",
          },
      }],
  )

  mock_completion.return_value = iter(STREAMING_MODEL_RESPONSE)

  responses = [
      response
      async for response in lite_llm_instance.generate_content_async(
          LLM_REQUEST_WITH_FUNCTION_DECLARATION, stream=True
      )
  ]
  assert len(responses) == 4
  mock_completion.assert_called_once()

  _, kwargs = mock_completion.call_args

  assert kwargs["model"] == "test_model"
  assert kwargs["messages"][0]["role"] == "user"
  assert kwargs["messages"][0]["content"] == "Test prompt"
  assert kwargs["tools"][0]["function"]["name"] == "test_function"
  assert kwargs["stream"]
  assert "llm_client" not in kwargs
  assert kwargs["api_base"] == "some://url"


@pytest.mark.asyncio
async def test_completion_with_drop_params(mock_completion, mock_client):
  lite_llm_instance = LiteLlm(
      model="test_model", llm_client=mock_client, drop_params=True
  )

  mock_completion.return_value = iter(STREAMING_MODEL_RESPONSE)

  responses = [
      response
      async for response in lite_llm_instance.generate_content_async(
          LLM_REQUEST_WITH_FUNCTION_DECLARATION, stream=True
      )
  ]
  assert len(responses) == 4

  mock_completion.assert_called_once()

  _, kwargs = mock_completion.call_args
  assert kwargs["drop_params"] is True


@pytest.mark.asyncio
async def test_generate_content_async_stream(
    mock_completion, lite_llm_instance
):

  mock_completion.return_value = iter(STREAMING_MODEL_RESPONSE)

  responses = [
      response
      async for response in lite_llm_instance.generate_content_async(
          LLM_REQUEST_WITH_FUNCTION_DECLARATION, stream=True
      )
  ]
  assert len(responses) == 4
  assert responses[0].content.role == "model"
  assert responses[0].content.parts[0].text == "zero, "
  assert responses[0].model_version == "test_model"
  assert responses[1].content.role == "model"
  assert responses[1].content.parts[0].text == "one, "
  assert responses[1].model_version == "test_model"
  assert responses[2].content.role == "model"
  assert responses[2].content.parts[0].text == "two:"
  assert responses[2].model_version == "test_model"
  assert responses[3].content.role == "model"
  assert responses[3].content.parts[-1].function_call.name == "test_function"
  assert responses[3].content.parts[-1].function_call.args == {
      "test_arg": "test_value"
  }
  assert responses[3].content.parts[-1].function_call.id == "test_tool_call_id"
  assert responses[3].finish_reason == types.FinishReason.STOP
  assert responses[3].model_version == "test_model"
  mock_completion.assert_called_once()

  _, kwargs = mock_completion.call_args
  assert kwargs["model"] == "test_model"
  assert kwargs["messages"][0]["role"] == "user"
  assert kwargs["messages"][0]["content"] == "Test prompt"
  assert kwargs["tools"][0]["function"]["name"] == "test_function"
  assert (
      kwargs["tools"][0]["function"]["description"]
      == "Test function description"
  )
  assert (
      kwargs["tools"][0]["function"]["parameters"]["properties"]["test_arg"][
          "type"
      ]
      == "string"
  )


@pytest.mark.asyncio
async def test_generate_content_async_stream_sets_finish_reason(
    mock_completion, lite_llm_instance
):
  mock_completion.return_value = iter([
      ModelResponse(
          model="test_model",
          choices=[
              StreamingChoices(
                  finish_reason=None,
                  delta=Delta(role="assistant", content="Hello "),
              )
          ],
      ),
      ModelResponse(
          model="test_model",
          choices=[
              StreamingChoices(
                  finish_reason=None,
                  delta=Delta(role="assistant", content="world"),
              )
          ],
      ),
      ModelResponse(
          model="test_model",
          choices=[StreamingChoices(finish_reason="stop", delta=Delta())],
      ),
  ])

  llm_request = LlmRequest(
      contents=[
          types.Content(
              role="user", parts=[types.Part.from_text(text="Test prompt")]
          )
      ],
  )

  responses = [
      response
      async for response in lite_llm_instance.generate_content_async(
          llm_request, stream=True
      )
  ]

  assert responses[-1].partial is False
  assert responses[-1].finish_reason == types.FinishReason.STOP
  assert responses[-1].content.parts[0].text == "Hello world"


@pytest.mark.asyncio
async def test_generate_content_async_stream_with_usage_metadata(
    mock_completion, lite_llm_instance
):

  streaming_model_response_with_usage_metadata = [
      *STREAMING_MODEL_RESPONSE,
      ModelResponse(
          usage={
              "prompt_tokens": 10,
              "completion_tokens": 5,
              "total_tokens": 15,
          },
          choices=[
              StreamingChoices(
                  finish_reason=None,
              )
          ],
      ),
  ]

  mock_completion.return_value = iter(
      streaming_model_response_with_usage_metadata
  )

  responses = [
      response
      async for response in lite_llm_instance.generate_content_async(
          LLM_REQUEST_WITH_FUNCTION_DECLARATION, stream=True
      )
  ]
  assert len(responses) == 4
  assert responses[0].content.role == "model"
  assert responses[0].content.parts[0].text == "zero, "
  assert responses[1].content.role == "model"
  assert responses[1].content.parts[0].text == "one, "
  assert responses[2].content.role == "model"
  assert responses[2].content.parts[0].text == "two:"
  assert responses[3].content.role == "model"
  assert responses[3].content.parts[-1].function_call.name == "test_function"
  assert responses[3].content.parts[-1].function_call.args == {
      "test_arg": "test_value"
  }
  assert responses[3].content.parts[-1].function_call.id == "test_tool_call_id"
  assert responses[3].finish_reason == types.FinishReason.STOP

  assert responses[3].usage_metadata.prompt_token_count == 10
  assert responses[3].usage_metadata.candidates_token_count == 5
  assert responses[3].usage_metadata.total_token_count == 15

  mock_completion.assert_called_once()

  _, kwargs = mock_completion.call_args
  assert kwargs["model"] == "test_model"
  assert kwargs["messages"][0]["role"] == "user"
  assert kwargs["messages"][0]["content"] == "Test prompt"
  assert kwargs["tools"][0]["function"]["name"] == "test_function"
  assert (
      kwargs["tools"][0]["function"]["description"]
      == "Test function description"
  )
  assert (
      kwargs["tools"][0]["function"]["parameters"]["properties"]["test_arg"][
          "type"
      ]
      == "string"
  )


@pytest.mark.asyncio
async def test_generate_content_async_stream_with_usage_metadata(
    mock_completion, lite_llm_instance
):
  """Tests that cached prompt tokens are propagated in streaming mode."""
  streaming_model_response_with_usage_metadata = [
      *STREAMING_MODEL_RESPONSE,
      ModelResponse(
          usage={
              "prompt_tokens": 10,
              "completion_tokens": 5,
              "total_tokens": 15,
              "cached_tokens": 8,
          },
          choices=[
              StreamingChoices(
                  finish_reason=None,
              )
          ],
      ),
  ]

  mock_completion.return_value = iter(
      streaming_model_response_with_usage_metadata
  )

  responses = [
      response
      async for response in lite_llm_instance.generate_content_async(
          LLM_REQUEST_WITH_FUNCTION_DECLARATION, stream=True
      )
  ]
  assert len(responses) == 4
  assert responses[3].usage_metadata.prompt_token_count == 10
  assert responses[3].usage_metadata.candidates_token_count == 5
  assert responses[3].usage_metadata.total_token_count == 15
  assert responses[3].usage_metadata.cached_content_token_count == 8


@pytest.mark.asyncio
async def test_generate_content_async_multiple_function_calls(
    mock_completion, lite_llm_instance
):
  """Test handling of multiple function calls with different indices in streaming mode.

  This test verifies that:
  1. Multiple function calls with different indices are handled correctly
  2. Arguments and names are properly accumulated for each function call
  3. The final response contains all function calls with correct indices
  """
  mock_completion.return_value = MULTIPLE_FUNCTION_CALLS_STREAM

  llm_request = LlmRequest(
      contents=[
          types.Content(
              role="user",
              parts=[types.Part.from_text(text="Test multiple function calls")],
          )
      ],
      config=types.GenerateContentConfig(
          tools=[
              types.Tool(
                  function_declarations=[
                      types.FunctionDeclaration(
                          name="function_1",
                          description="First test function",
                          parameters=types.Schema(
                              type=types.Type.OBJECT,
                              properties={
                                  "arg": types.Schema(type=types.Type.STRING),
                              },
                          ),
                      ),
                      types.FunctionDeclaration(
                          name="function_2",
                          description="Second test function",
                          parameters=types.Schema(
                              type=types.Type.OBJECT,
                              properties={
                                  "arg": types.Schema(type=types.Type.STRING),
                              },
                          ),
                      ),
                  ]
              )
          ],
      ),
  )

  responses = []
  async for response in lite_llm_instance.generate_content_async(
      llm_request, stream=True
  ):
    responses.append(response)

  # Verify we got the final response with both function calls
  assert len(responses) > 0
  final_response = responses[-1]
  assert final_response.content.role == "model"
  assert len(final_response.content.parts) == 2

  # Verify first function call
  assert final_response.content.parts[0].function_call.name == "function_1"
  assert final_response.content.parts[0].function_call.id == "call_1"
  assert final_response.content.parts[0].function_call.args == {"arg": "value1"}

  # Verify second function call
  assert final_response.content.parts[1].function_call.name == "function_2"
  assert final_response.content.parts[1].function_call.id == "call_2"
  assert final_response.content.parts[1].function_call.args == {"arg": "value2"}


@pytest.mark.asyncio
async def test_generate_content_async_non_compliant_multiple_function_calls(
    mock_completion, lite_llm_instance
):
  """Test handling of multiple function calls with same 0 indices in streaming mode.

  This test verifies that:
  1. Multiple function calls with same indices (0) are handled correctly
  2. Arguments and names are properly accumulated for each function call
  3. The final response contains all function calls with correct incremented
  indices
  """
  mock_completion.return_value = NON_COMPLIANT_MULTIPLE_FUNCTION_CALLS_STREAM

  llm_request = LlmRequest(
      contents=[
          types.Content(
              role="user",
              parts=[types.Part.from_text(text="Test multiple function calls")],
          )
      ],
      config=types.GenerateContentConfig(
          tools=[
              types.Tool(
                  function_declarations=[
                      types.FunctionDeclaration(
                          name="function_1",
                          description="First test function",
                          parameters=types.Schema(
                              type=types.Type.OBJECT,
                              properties={
                                  "arg": types.Schema(type=types.Type.STRING),
                              },
                          ),
                      ),
                      types.FunctionDeclaration(
                          name="function_2",
                          description="Second test function",
                          parameters=types.Schema(
                              type=types.Type.OBJECT,
                              properties={
                                  "arg": types.Schema(type=types.Type.STRING),
                              },
                          ),
                      ),
                  ]
              )
          ],
      ),
  )

  responses = []
  async for response in lite_llm_instance.generate_content_async(
      llm_request, stream=True
  ):
    responses.append(response)

  # Verify we got the final response with both function calls
  assert len(responses) > 0
  final_response = responses[-1]
  assert final_response.content.role == "model"
  assert len(final_response.content.parts) == 2

  # Verify first function call
  assert final_response.content.parts[0].function_call.name == "function_1"
  assert final_response.content.parts[0].function_call.id == "0"
  assert final_response.content.parts[0].function_call.args == {"arg": "value1"}

  # Verify second function call
  assert final_response.content.parts[1].function_call.name == "function_2"
  assert final_response.content.parts[1].function_call.id == "1"
  assert final_response.content.parts[1].function_call.args == {"arg": "value2"}


@pytest.mark.asyncio
async def test_generate_content_async_stream_with_empty_chunk(
    mock_completion, lite_llm_instance
):
  """Tests that empty tool call chunks in a stream are ignored."""
  mock_completion.return_value = iter(STREAM_WITH_EMPTY_CHUNK)

  responses = [
      response
      async for response in lite_llm_instance.generate_content_async(
          LLM_REQUEST_WITH_FUNCTION_DECLARATION, stream=True
      )
  ]

  assert len(responses) == 1
  final_response = responses[0]
  assert final_response.content.role == "model"

  # Crucially, assert that only ONE tool call was generated,
  # proving the empty chunk was ignored.
  assert len(final_response.content.parts) == 1

  function_call = final_response.content.parts[0].function_call
  assert function_call.name == "test_function"
  assert function_call.id == "call_abc"
  assert function_call.args == {"test_arg": "value"}


@pytest.mark.asyncio
async def test_get_completion_inputs_generation_params():
  # Test that generation_params are extracted and mapped correctly
  req = LlmRequest(
      contents=[
          types.Content(role="user", parts=[types.Part.from_text(text="hi")]),
      ],
      config=types.GenerateContentConfig(
          temperature=0.33,
          max_output_tokens=123,
          top_p=0.88,
          top_k=7,
          stop_sequences=["foo", "bar"],
          presence_penalty=0.1,
          frequency_penalty=0.2,
      ),
  )

  _, _, _, generation_params = await _get_completion_inputs(
      req, model="gpt-4o-mini"
  )
  assert generation_params["temperature"] == 0.33
  assert generation_params["max_completion_tokens"] == 123
  assert generation_params["top_p"] == 0.88
  assert generation_params["top_k"] == 7
  assert generation_params["stop"] == ["foo", "bar"]
  assert generation_params["presence_penalty"] == 0.1
  assert generation_params["frequency_penalty"] == 0.2
  # Should not include max_output_tokens
  assert "max_output_tokens" not in generation_params
  assert "stop_sequences" not in generation_params


@pytest.mark.asyncio
async def test_get_completion_inputs_empty_generation_params():
  # Test that generation_params is None when no generation parameters are set
  req = LlmRequest(
      contents=[
          types.Content(role="user", parts=[types.Part.from_text(text="hi")]),
      ],
      config=types.GenerateContentConfig(),
  )

  _, _, _, generation_params = await _get_completion_inputs(
      req, model="gpt-4o-mini"
  )
  assert generation_params is None


@pytest.mark.asyncio
async def test_get_completion_inputs_minimal_config():
  # Test that generation_params is None when config has no generation parameters
  req = LlmRequest(
      contents=[
          types.Content(role="user", parts=[types.Part.from_text(text="hi")]),
      ],
      config=types.GenerateContentConfig(
          system_instruction="test instruction"  # Non-generation parameter
      ),
  )

  _, _, _, generation_params = await _get_completion_inputs(
      req, model="gpt-4o-mini"
  )
  assert generation_params is None


@pytest.mark.asyncio
async def test_get_completion_inputs_partial_generation_params():
  # Test that generation_params is correctly built even with only some parameters
  req = LlmRequest(
      contents=[
          types.Content(role="user", parts=[types.Part.from_text(text="hi")]),
      ],
      config=types.GenerateContentConfig(
          temperature=0.7,
          # Only temperature is set, others are None/default
      ),
  )

  _, _, _, generation_params = await _get_completion_inputs(
      req, model="gpt-4o-mini"
  )
  assert generation_params is not None
  assert generation_params["temperature"] == 0.7
  # Should only contain the temperature parameter
  assert len(generation_params) == 1


def test_function_declaration_to_tool_param_edge_cases():
  """Test edge cases for function declaration conversion that caused the original bug."""
  from google.adk.models.lite_llm import _function_declaration_to_tool_param

  # Test function with None parameters (the original bug scenario)
  func_decl = types.FunctionDeclaration(
      name="test_function_none_params",
      description="Function with None parameters",
      parameters=None,
  )
  result = _function_declaration_to_tool_param(func_decl)
  expected = {
      "type": "function",
      "function": {
          "name": "test_function_none_params",
          "description": "Function with None parameters",
          "parameters": {
              "type": "object",
              "properties": {},
          },
      },
  }
  assert result == expected

  # Verify no 'required' field is added when parameters is None
  assert "required" not in result["function"]["parameters"]


@pytest.mark.parametrize(
    "usage, expected_tokens",
    [
        ({"prompt_tokens_details": {"cached_tokens": 123}}, 123),
        (
            {
                "prompt_tokens_details": [
                    {"cached_tokens": 50},
                    {"cached_tokens": 25},
                ]
            },
            75,
        ),
        ({"cached_prompt_tokens": 45}, 45),
        ({"cached_tokens": 67}, 67),
        ({"prompt_tokens": 100}, 0),
        ({}, 0),
        ("not a dict", 0),
        (None, 0),
        ({"prompt_tokens_details": {"cached_tokens": "not a number"}}, 0),
        (json.dumps({"cached_tokens": 89}), 89),
        (json.dumps({"some_key": "some_value"}), 0),
    ],
)
def test_extract_cached_prompt_tokens(usage, expected_tokens):
  from google.adk.models.lite_llm import _extract_cached_prompt_tokens

  assert _extract_cached_prompt_tokens(usage) == expected_tokens


def test_gemini_via_litellm_warning(monkeypatch):
  """Test that Gemini via LiteLLM shows warning."""
  # Ensure environment variable is not set
  monkeypatch.delenv("ADK_SUPPRESS_GEMINI_LITELLM_WARNINGS", raising=False)
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    # Test with Google AI Studio Gemini via LiteLLM
    LiteLlm(model="gemini/gemini-2.5-pro-exp-03-25")
    assert len(w) == 1
    assert issubclass(w[0].category, UserWarning)
    assert "[GEMINI_VIA_LITELLM]" in str(w[0].message)
    assert "better performance" in str(w[0].message)
    assert "gemini-2.5-pro-exp-03-25" in str(w[0].message)
    assert "ADK_SUPPRESS_GEMINI_LITELLM_WARNINGS" in str(w[0].message)


def test_gemini_via_litellm_warning_vertex_ai(monkeypatch):
  """Test that Vertex AI Gemini via LiteLLM shows warning."""
  # Ensure environment variable is not set
  monkeypatch.delenv("ADK_SUPPRESS_GEMINI_LITELLM_WARNINGS", raising=False)
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    # Test with Vertex AI Gemini via LiteLLM
    LiteLlm(model="vertex_ai/gemini-1.5-flash")
    assert len(w) == 1
    assert issubclass(w[0].category, UserWarning)
    assert "[GEMINI_VIA_LITELLM]" in str(w[0].message)
    assert "vertex_ai/gemini-1.5-flash" in str(w[0].message)


def test_gemini_via_litellm_warning_suppressed(monkeypatch):
  """Test that Gemini via LiteLLM warning can be suppressed."""
  monkeypatch.setenv("ADK_SUPPRESS_GEMINI_LITELLM_WARNINGS", "true")
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    LiteLlm(model="gemini/gemini-2.5-pro-exp-03-25")
    assert len(w) == 0


def test_non_gemini_litellm_no_warning():
  """Test that non-Gemini models via LiteLLM don't show warning."""
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    # Test with non-Gemini model
    LiteLlm(model="openai/gpt-4o")
    assert len(w) == 0


@pytest.mark.parametrize(
    "finish_reason,response_content,expected_content,has_tool_calls",
    [
        ("length", "Test response", "Test response", False),
        ("stop", "Complete response", "Complete response", False),
        (
            "tool_calls",
            "",
            "",
            True,
        ),
        ("content_filter", "", "", False),
    ],
    ids=["length", "stop", "tool_calls", "content_filter"],
)
@pytest.mark.asyncio
async def test_finish_reason_propagation(
    mock_acompletion,
    lite_llm_instance,
    finish_reason,
    response_content,
    expected_content,
    has_tool_calls,
):
  """Test that finish_reason is properly propagated from LiteLLM response."""
  tool_calls = None
  if has_tool_calls:
    tool_calls = [
        ChatCompletionMessageToolCall(
            type="function",
            id="test_id",
            function=Function(
                name="test_function",
                arguments='{"arg": "value"}',
            ),
        )
    ]

  mock_response = ModelResponse(
      choices=[
          Choices(
              message=ChatCompletionAssistantMessage(
                  role="assistant",
                  content=response_content,
                  tool_calls=tool_calls,
              ),
              finish_reason=finish_reason,
          )
      ]
  )
  mock_acompletion.return_value = mock_response

  llm_request = LlmRequest(
      contents=[
          types.Content(
              role="user", parts=[types.Part.from_text(text="Test prompt")]
          )
      ],
  )

  async for response in lite_llm_instance.generate_content_async(llm_request):
    assert response.content.role == "model"
    # Verify finish_reason is mapped to FinishReason enum
    assert isinstance(response.finish_reason, types.FinishReason)
    # Verify correct enum mapping using the actual mapping from lite_llm
    assert response.finish_reason == _FINISH_REASON_MAPPING[finish_reason]
    if expected_content:
      assert response.content.parts[0].text == expected_content
    if has_tool_calls:
      assert len(response.content.parts) > 0
      assert response.content.parts[-1].function_call.name == "test_function"

  mock_acompletion.assert_called_once()


@pytest.mark.asyncio
async def test_finish_reason_unknown_maps_to_other(
    mock_acompletion, lite_llm_instance
):
  """Test that unknown finish_reason values map to FinishReason.OTHER."""
  mock_response = ModelResponse(
      choices=[
          Choices(
              message=ChatCompletionAssistantMessage(
                  role="assistant",
                  content="Test response",
              ),
              finish_reason="unknown_reason_type",
          )
      ]
  )
  mock_acompletion.return_value = mock_response

  llm_request = LlmRequest(
      contents=[
          types.Content(
              role="user", parts=[types.Part.from_text(text="Test prompt")]
          )
      ],
  )

  async for response in lite_llm_instance.generate_content_async(llm_request):
    assert response.content.role == "model"
    # Unknown finish_reason should map to OTHER
    assert isinstance(response.finish_reason, types.FinishReason)
    assert response.finish_reason == types.FinishReason.OTHER

  mock_acompletion.assert_called_once()


# Tests for provider detection and file_id support


@pytest.mark.parametrize(
    "model_string, expected_provider",
    [
        # Standard provider/model format
        ("openai/gpt-4o", "openai"),
        ("azure/gpt-4", "azure"),
        ("groq/llama3-70b", "groq"),
        ("anthropic/claude-3", "anthropic"),
        ("vertex_ai/gemini-pro", "vertex_ai"),
        # Fallback heuristics
        ("gpt-4o", "openai"),
        ("o1-preview", "openai"),
        ("azure-gpt-4", "azure"),
        # Unknown models
        ("custom-model", ""),
        ("", ""),
        (None, ""),
    ],
)
def test_get_provider_from_model(model_string, expected_provider):
  """Test provider extraction from model strings."""
  assert _get_provider_from_model(model_string) == expected_provider


@pytest.mark.parametrize(
    "provider, expected_in_list",
    [
        ("openai", True),
        ("azure", True),
        ("anthropic", False),
        ("vertex_ai", False),
    ],
)
def test_file_id_required_providers(provider, expected_in_list):
  """Test that the correct providers require file_id."""
  assert (provider in _FILE_ID_REQUIRED_PROVIDERS) == expected_in_list


@pytest.mark.asyncio
async def test_get_content_pdf_openai_uses_file_id(mocker):
  """Test that PDF files use file_id for OpenAI provider."""
  mock_file_response = mocker.create_autospec(litellm.FileObject)
  mock_file_response.id = "file-abc123"
  mock_acreate_file = AsyncMock(return_value=mock_file_response)
  mocker.patch.object(litellm, "acreate_file", new=mock_acreate_file)

  parts = [
      types.Part.from_bytes(data=b"test_pdf_data", mime_type="application/pdf")
  ]
  content = await _get_content(parts, provider="openai")

  assert content[0]["type"] == "file"
  assert content[0]["file"]["file_id"] == "file-abc123"
  assert "file_data" not in content[0]["file"]

  mock_acreate_file.assert_called_once_with(
      file=b"test_pdf_data",
      purpose="assistants",
      custom_llm_provider="openai",
  )


@pytest.mark.asyncio
async def test_get_content_pdf_non_openai_uses_file_data():
  """Test that PDF files use file_data for non-OpenAI providers."""
  parts = [
      types.Part.from_bytes(data=b"test_pdf_data", mime_type="application/pdf")
  ]
  content = await _get_content(parts, provider="anthropic")

  assert content[0]["type"] == "file"
  assert "file_data" in content[0]["file"]
  assert content[0]["file"]["file_data"].startswith(
      "data:application/pdf;base64,"
  )
  assert "file_id" not in content[0]["file"]


@pytest.mark.asyncio
async def test_get_content_pdf_azure_uses_file_id(mocker):
  """Test that PDF files use file_id for Azure provider."""
  mock_file_response = mocker.create_autospec(litellm.FileObject)
  mock_file_response.id = "file-xyz789"
  mock_acreate_file = AsyncMock(return_value=mock_file_response)
  mocker.patch.object(litellm, "acreate_file", new=mock_acreate_file)

  parts = [
      types.Part.from_bytes(data=b"test_pdf_data", mime_type="application/pdf")
  ]
  content = await _get_content(parts, provider="azure")

  assert content[0]["type"] == "file"
  assert content[0]["file"]["file_id"] == "file-xyz789"

  mock_acreate_file.assert_called_once_with(
      file=b"test_pdf_data",
      purpose="assistants",
      custom_llm_provider="azure",
  )


@pytest.mark.asyncio
async def test_get_completion_inputs_openai_file_upload(mocker):
  """Test that _get_completion_inputs uploads files for OpenAI models."""
  mock_file_response = mocker.create_autospec(litellm.FileObject)
  mock_file_response.id = "file-uploaded123"
  mock_acreate_file = AsyncMock(return_value=mock_file_response)
  mocker.patch.object(litellm, "acreate_file", new=mock_acreate_file)

  pdf_part = types.Part.from_bytes(
      data=b"test_pdf_content", mime_type="application/pdf"
  )
  llm_request = LlmRequest(
      model="openai/gpt-4o",
      contents=[
          types.Content(
              role="user",
              parts=[
                  types.Part.from_text(text="Analyze this PDF"),
                  pdf_part,
              ],
          )
      ],
      config=types.GenerateContentConfig(tools=[]),
  )

  messages, tools, response_format, generation_params = (
      await _get_completion_inputs(llm_request, model="openai/gpt-4o")
  )

  assert len(messages) == 1
  assert messages[0]["role"] == "user"
  content = messages[0]["content"]
  assert len(content) == 2
  assert content[0]["type"] == "text"
  assert content[0]["text"] == "Analyze this PDF"
  assert content[1]["type"] == "file"
  assert content[1]["file"]["file_id"] == "file-uploaded123"

  mock_acreate_file.assert_called_once()


@pytest.mark.asyncio
async def test_get_completion_inputs_non_openai_no_file_upload(mocker):
  """Test that _get_completion_inputs does not upload files for non-OpenAI models."""
  mock_acreate_file = AsyncMock()
  mocker.patch.object(litellm, "acreate_file", new=mock_acreate_file)

  pdf_part = types.Part.from_bytes(
      data=b"test_pdf_content", mime_type="application/pdf"
  )
  llm_request = LlmRequest(
      model="anthropic/claude-3-opus",
      contents=[
          types.Content(
              role="user",
              parts=[
                  types.Part.from_text(text="Analyze this PDF"),
                  pdf_part,
              ],
          )
      ],
      config=types.GenerateContentConfig(tools=[]),
  )

  messages, tools, response_format, generation_params = (
      await _get_completion_inputs(llm_request, model="anthropic/claude-3-opus")
  )

  assert len(messages) == 1
  content = messages[0]["content"]
  assert content[1]["type"] == "file"
  assert "file_data" in content[1]["file"]
  assert "file_id" not in content[1]["file"]

  mock_acreate_file.assert_not_called()


class TestRedirectLitellmLoggersToStdout(unittest.TestCase):
  """Tests for _redirect_litellm_loggers_to_stdout function."""

  def test_redirects_stderr_handler_to_stdout(self):
    """Test that handlers pointing to stderr are redirected to stdout."""
    test_logger = logging.getLogger("LiteLLM")
    # Create a handler pointing to stderr
    handler = logging.StreamHandler(sys.stderr)
    test_logger.addHandler(handler)

    try:
      self.assertIs(handler.stream, sys.stderr)

      _redirect_litellm_loggers_to_stdout()

      self.assertIs(handler.stream, sys.stdout)
    finally:
      # Clean up
      test_logger.removeHandler(handler)

  def test_preserves_stdout_handler(self):
    """Test that handlers already pointing to stdout are not modified."""
    test_logger = logging.getLogger("LiteLLM Proxy")
    # Create a handler already pointing to stdout
    handler = logging.StreamHandler(sys.stdout)
    test_logger.addHandler(handler)

    try:
      _redirect_litellm_loggers_to_stdout()

      self.assertIs(handler.stream, sys.stdout)
    finally:
      # Clean up
      test_logger.removeHandler(handler)

  def test_does_not_affect_non_stream_handlers(self):
    """Test that non-StreamHandler handlers are not affected."""
    test_logger = logging.getLogger("LiteLLM Router")
    # Create a FileHandler (not a StreamHandler)
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
      temp_file_name = temp_file.name
    with contextlib.closing(
        logging.FileHandler(temp_file_name)
    ) as file_handler:
      test_logger.addHandler(file_handler)

      try:
        _redirect_litellm_loggers_to_stdout()
        # FileHandler should not be modified (it doesn't point to stderr or stdout)
        self.assertEqual(file_handler.baseFilename, temp_file_name)
      finally:
        # Clean up
        test_logger.removeHandler(file_handler)
    os.unlink(temp_file_name)


@pytest.mark.parametrize(
    "logger_name",
    ["LiteLLM", "LiteLLM Proxy", "LiteLLM Router"],
    ids=["LiteLLM", "LiteLLM Proxy", "LiteLLM Router"],
)
def test_handles_litellm_logger_names(logger_name):
  """Test that LiteLLM logger names are processed."""
  test_logger = logging.getLogger(logger_name)
  handler = logging.StreamHandler(sys.stderr)
  test_logger.addHandler(handler)

  try:
    _redirect_litellm_loggers_to_stdout()

    assert handler.stream is sys.stdout
  finally:
    # Clean up
    test_logger.removeHandler(handler)
