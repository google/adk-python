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
from unittest.mock import AsyncMock
from unittest.mock import Mock

from google.adk.models.lite_llm import _content_to_message_param
from google.adk.models.lite_llm import _function_declaration_to_tool_param
from google.adk.models.lite_llm import _get_content
from google.adk.models.lite_llm import _message_to_generate_content_response
from google.adk.models.lite_llm import _model_response_to_chunk
from google.adk.models.lite_llm import _to_litellm_role
from google.adk.models.lite_llm import FunctionChunk
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.lite_llm import LiteLLMClient
from google.adk.models.lite_llm import TextChunk
from google.adk.models.lite_llm import UsageMetadataChunk
from google.adk.models.llm_request import LlmRequest
from google.genai import types
from litellm import ChatCompletionAssistantMessage
from litellm import ChatCompletionMessageToolCall
from litellm import Function
from litellm.types.utils import ChatCompletionDeltaToolCall
from litellm.types.utils import Choices
from litellm.types.utils import Delta
from litellm.types.utils import ModelResponse
from litellm.types.utils import StreamingChoices
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


STREAMING_MODEL_RESPONSE = [
    ModelResponse(
        choices=[
            StreamingChoices(
                finish_reason=None,
                delta=Delta(
                    role="assistant",
                    content="zero, ",
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
                    content="one, ",
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
                    content="two:",
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
                                arguments='value"}',
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
                finish_reason="tool_use",
            )
        ]
    ),
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


@pytest.fixture
def mock_response():
  return ModelResponse(
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
      ]
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
                    ),
                },
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
                        },
                    },
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
  assert kwargs["messages"][0]["role"] == "developer"
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

  mock_acompletion.assert_called_once()


def test_content_to_message_param_user_message():
  content = types.Content(
      role="user", parts=[types.Part.from_text(text="Test prompt")]
  )
  message = _content_to_message_param(content)
  assert message["role"] == "user"
  assert message["content"] == "Test prompt"


def test_content_to_message_param_multi_part_function_response():
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
  messages = _content_to_message_param(content)
  assert isinstance(messages, list)
  assert len(messages) == 2

  assert messages[0]["role"] == "tool"
  assert messages[0]["tool_call_id"] == "tool_call_1"
  assert messages[0]["content"] == '{"result": "result_one"}'

  assert messages[1]["role"] == "tool"
  assert messages[1]["tool_call_id"] == "tool_call_2"
  assert messages[1]["content"] == '{"value": 123}'


def test_content_to_message_param_assistant_message():
  content = types.Content(
      role="assistant", parts=[types.Part.from_text(text="Test response")]
  )
  message = _content_to_message_param(content)
  assert message["role"] == "assistant"
  assert message["content"] == "Test response"


def test_content_to_message_param_function_call():
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
  message = _content_to_message_param(content)
  assert message["role"] == "assistant"
  assert message["content"] == "test response"

  tool_call = message["tool_calls"][0]
  assert tool_call["type"] == "function"
  assert tool_call["id"] == "test_tool_call_id"
  assert tool_call["function"]["name"] == "test_function"
  assert tool_call["function"]["arguments"] == '{"test_arg": "test_value"}'


def test_content_to_message_param_multipart_content():
  """Test handling of multipart content where final_content is a list with text objects."""
  content = types.Content(
      role="assistant",
      parts=[
          types.Part.from_text(text="text part"),
          types.Part.from_bytes(data=b"test_image_data", mime_type="image/png"),
      ],
  )
  message = _content_to_message_param(content)
  assert message["role"] == "assistant"
  # When content is a list and the first element is a text object with type "text",
  # it should extract the text (for providers like ollama_chat that don't handle lists well)
  # This is the behavior implemented in the fix
  assert message["content"] == "text part"
  assert message["tool_calls"] is None


def test_content_to_message_param_single_text_object_in_list():
  """Test extraction of text from single text object in list (for ollama_chat compatibility)."""
  from unittest.mock import patch

  # Mock _get_content to return a list with single text object
  with patch("google.adk.models.lite_llm._get_content") as mock_get_content:
    mock_get_content.return_value = [{"type": "text", "text": "single text"}]

    content = types.Content(
        role="assistant",
        parts=[types.Part.from_text(text="single text")],
    )
    message = _content_to_message_param(content)
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


def test_get_content_text():
  parts = [types.Part.from_text(text="Test text")]
  content = _get_content(parts)
  assert content == "Test text"


def test_get_content_image():
  parts = [
      types.Part.from_bytes(data=b"test_image_data", mime_type="image/png")
  ]
  content = _get_content(parts)
  assert content[0]["type"] == "image_url"
  assert (
      content[0]["image_url"]["url"]
      == "data:image/png;base64,dGVzdF9pbWFnZV9kYXRh"
  )
  assert content[0]["image_url"]["format"] == "png"


def test_get_content_video():
  parts = [
      types.Part.from_bytes(data=b"test_video_data", mime_type="video/mp4")
  ]
  content = _get_content(parts)
  assert content[0]["type"] == "video_url"
  assert (
      content[0]["video_url"]["url"]
      == "data:video/mp4;base64,dGVzdF92aWRlb19kYXRh"
  )
  assert content[0]["video_url"]["format"] == "mp4"


def test_to_litellm_role():
  assert _to_litellm_role("model") == "assistant"
  assert _to_litellm_role("assistant") == "assistant"
  assert _to_litellm_role("user") == "user"
  assert _to_litellm_role(None) == "user"


@pytest.mark.parametrize(
    "response, expected_chunks, expected_finished",
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
            [
                TextChunk(text="this is a test"),
                UsageMetadataChunk(
                    prompt_tokens=0, completion_tokens=0, total_tokens=0
                ),
            ],
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
            [
                TextChunk(text="this is a test"),
                UsageMetadataChunk(
                    prompt_tokens=3, completion_tokens=5, total_tokens=8
                ),
            ],
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
            [
                FunctionChunk(id="1", name="test_function", args='{"key": "va'),
                UsageMetadataChunk(
                    prompt_tokens=0, completion_tokens=0, total_tokens=0
                ),
            ],
            None,
        ),
        (
            ModelResponse(choices=[{"finish_reason": "tool_calls"}]),
            [
                None,
                UsageMetadataChunk(
                    prompt_tokens=0, completion_tokens=0, total_tokens=0
                ),
            ],
            "tool_calls",
        ),
        (
            ModelResponse(choices=[{}]),
            [
                None,
                UsageMetadataChunk(
                    prompt_tokens=0, completion_tokens=0, total_tokens=0
                ),
            ],
            "stop",
        ),
    ],
)
def test_model_response_to_chunk(response, expected_chunks, expected_finished):
  result = list(_model_response_to_chunk(response))
  assert len(result) == 2
  chunk, finished = result[0]
  if expected_chunks:
    assert isinstance(chunk, type(expected_chunks[0]))
    assert chunk == expected_chunks[0]
  else:
    assert chunk is None
  assert finished == expected_finished

  usage_chunk, _ = result[1]
  assert usage_chunk is not None
  assert usage_chunk.prompt_tokens == expected_chunks[1].prompt_tokens
  assert usage_chunk.completion_tokens == expected_chunks[1].completion_tokens
  assert usage_chunk.total_tokens == expected_chunks[1].total_tokens


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
  3. The final response contains all function calls with correct incremented indices
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
def test_get_completion_inputs_generation_params():
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
  from google.adk.models.lite_llm import _get_completion_inputs

  _, _, _, generation_params = _get_completion_inputs(req)
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
