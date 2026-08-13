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

"""Unit tests for the OCI Generative AI LLM integration."""

import asyncio
import json
import os
from typing import Any
from unittest import mock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

# The tests patch oci.generative_ai_inference; skip if that submodule is absent.
pytest.importorskip(
    "oci.generative_ai_inference", reason="Requires oci (google-adk[oci])"
)

from google.adk.integrations.oci._oci_genai_llm import _content_to_oci_message
from google.adk.integrations.oci._oci_genai_llm import _function_declaration_to_oci_tool
from google.adk.integrations.oci._oci_genai_llm import _oci_response_to_llm_response
from google.adk.integrations.oci._oci_genai_llm import OCIGenAILlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from google.genai.types import Content
from google.genai.types import Part

# ---------------------------------------------------------------------------
# Helpers: build fake OCI SDK response objects without importing oci
# ---------------------------------------------------------------------------


def _make_oci_response(
    text: str = "Hello from OCI.",
    tool_calls: list = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> MagicMock:
  """Build a minimal MagicMock that mirrors the OCI GenAI chat response."""
  usage = MagicMock()
  usage.prompt_tokens = prompt_tokens
  usage.completion_tokens = completion_tokens

  content_block = MagicMock()
  content_block.text = text

  message = MagicMock()
  message.content = [content_block]
  message.tool_calls = tool_calls or []

  choice = MagicMock()
  choice.message = message

  chat_response = MagicMock()
  chat_response.choices = [choice]
  chat_response.usage = usage

  response = MagicMock()
  response.data.chat_response = chat_response
  return response


def _make_tool_call_response(name: str, args: dict) -> MagicMock:
  """Build a fake OCI tool-call response using FunctionCall (OCI SDK subtype)."""
  import oci.generative_ai_inference.models as oci_models

  fc = oci_models.FunctionCall(
      id="call_abc123",
      type=oci_models.FunctionCall.TYPE_FUNCTION,
      name=name,
      arguments=json.dumps(args),
  )

  usage = MagicMock()
  usage.prompt_tokens = 20
  usage.completion_tokens = 15

  message = MagicMock()
  message.content = []
  message.tool_calls = [fc]

  choice = MagicMock()
  choice.message = message

  chat_response = MagicMock()
  chat_response.choices = [choice]
  chat_response.usage = usage

  response = MagicMock()
  response.data.chat_response = chat_response
  return response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def oci_llm():
  return OCIGenAILlm(
      model="google.gemini-2.5-flash",
      compartment_id="ocid1.compartment.oc1..example",
      service_endpoint=(
          "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
      ),
  )


@pytest.fixture
def llm_request():
  return LlmRequest(
      model="google.gemini-2.5-flash",
      contents=[Content(role="user", parts=[Part.from_text(text="Hello")])],
      config=types.GenerateContentConfig(
          system_instruction="You are a helpful assistant.",
      ),
  )


# ---------------------------------------------------------------------------
# supported_models
# ---------------------------------------------------------------------------


def test_supported_models_gemini():
  assert any("gemini" in p for p in OCIGenAILlm.supported_models())


def test_supported_models_llama():
  assert any("llama" in p for p in OCIGenAILlm.supported_models())


def test_supported_models_gemma():
  assert any("gemma" in p for p in OCIGenAILlm.supported_models())


def test_supported_models_registry():
  from google.adk.models.registry import LLMRegistry

  assert LLMRegistry.resolve("google.gemini-2.0-flash-001") is OCIGenAILlm
  assert LLMRegistry.resolve("meta.llama-3.1-8b-instruct") is OCIGenAILlm
  assert LLMRegistry.resolve("google.gemma-3-27b-it") is OCIGenAILlm


# ---------------------------------------------------------------------------
# _content_to_oci_message
# ---------------------------------------------------------------------------


def test_content_to_oci_message_user_text():
  import oci.generative_ai_inference.models as oci_models

  content = Content(role="user", parts=[Part.from_text(text="Hi there")])
  msgs = _content_to_oci_message(content)
  assert isinstance(msgs, list)
  assert len(msgs) == 1
  msg = msgs[0]
  assert isinstance(msg, oci_models.UserMessage)
  assert msg.role == oci_models.UserMessage.ROLE_USER
  assert msg.content[0].text == "Hi there"


def test_content_to_oci_message_assistant_text():
  import oci.generative_ai_inference.models as oci_models

  content = Content(role="model", parts=[Part.from_text(text="I can help.")])
  msgs = _content_to_oci_message(content)
  assert isinstance(msgs, list)
  assert len(msgs) == 1
  msg = msgs[0]
  assert isinstance(msg, oci_models.AssistantMessage)
  assert msg.role == oci_models.AssistantMessage.ROLE_ASSISTANT
  assert msg.content[0].text == "I can help."


def test_content_to_oci_message_multi_part_text():
  import oci.generative_ai_inference.models as oci_models

  content = Content(
      role="user",
      parts=[
          Part.from_text(text="First"),
          Part.from_text(text="Second"),
      ],
  )
  msgs = _content_to_oci_message(content)
  assert isinstance(msgs, list)
  assert len(msgs) == 1
  msg = msgs[0]
  assert isinstance(msg, oci_models.UserMessage)
  assert "First" in msg.content[0].text
  assert "Second" in msg.content[0].text


def test_content_to_oci_message_function_call():
  import oci.generative_ai_inference.models as oci_models

  part = Part.from_function_call(name="get_weather", args={"city": "Toronto"})
  content = Content(role="model", parts=[part])
  msgs = _content_to_oci_message(content)
  assert isinstance(msgs, list)
  assert len(msgs) == 1
  msg = msgs[0]
  assert isinstance(msg, oci_models.AssistantMessage)
  assert msg.tool_calls is not None
  assert len(msg.tool_calls) == 1
  fc = msg.tool_calls[0]
  assert isinstance(fc, oci_models.FunctionCall)
  assert fc.name == "get_weather"
  assert json.loads(fc.arguments) == {"city": "Toronto"}


def test_content_to_oci_message_function_response():
  import oci.generative_ai_inference.models as oci_models

  part = Part.from_function_response(
      name="get_weather", response={"result": "Sunny, 22°C"}
  )
  part.function_response.id = "call_xyz"
  content = Content(role="user", parts=[part])
  msgs = _content_to_oci_message(content)
  assert isinstance(msgs, list)
  assert len(msgs) == 1
  msg = msgs[0]
  assert isinstance(msg, oci_models.ToolMessage)
  assert msg.tool_call_id == "call_xyz"
  assert msg.content[0].text


def test_content_to_oci_message_multiple_function_responses():
  import oci.generative_ai_inference.models as oci_models

  part1 = Part.from_function_response(
      name="get_weather", response={"result": "Sunny, 22°C"}
  )
  part1.function_response.id = "call_A"

  part2 = Part.from_function_response(
      name="get_price", response={"result": "$150"}
  )
  part2.function_response.id = "call_B"

  content = Content(role="user", parts=[part1, part2])
  msgs = _content_to_oci_message(content)

  assert isinstance(msgs, list)
  assert len(msgs) == 2

  assert isinstance(msgs[0], oci_models.ToolMessage)
  assert msgs[0].tool_call_id == "call_A"

  assert isinstance(msgs[1], oci_models.ToolMessage)
  assert msgs[1].tool_call_id == "call_B"


def test_content_to_oci_message_multiple_function_responses_no_id():
  import oci.generative_ai_inference.models as oci_models

  part1 = Part.from_function_response(
      name="get_weather", response={"result": "Sunny, 22°C"}
  )
  part2 = Part.from_function_response(
      name="get_price", response={"result": "$150"}
  )

  content = Content(role="user", parts=[part1, part2])
  msgs = _content_to_oci_message(content)

  assert isinstance(msgs, list)
  assert len(msgs) == 2

  assert isinstance(msgs[0], oci_models.ToolMessage)
  assert msgs[0].tool_call_id == ""
  assert len(msgs[0].content) == 1
  assert "Sunny" in msgs[0].content[0].text

  assert isinstance(msgs[1], oci_models.ToolMessage)
  assert msgs[1].tool_call_id == ""
  assert len(msgs[1].content) == 1
  assert "$150" in msgs[1].content[0].text


def test_content_to_oci_message_mixed_tool_and_text():
  import oci.generative_ai_inference.models as oci_models

  part1 = Part.from_function_response(
      name="get_weather", response={"result": "Sunny, 22°C"}
  )
  part1.function_response.id = "call_A"
  part2 = Part.from_text(text="Here is the weather and some extra text.")

  content = Content(role="user", parts=[part1, part2])
  msgs = _content_to_oci_message(content)

  assert isinstance(msgs, list)
  assert len(msgs) == 2

  assert isinstance(msgs[0], oci_models.ToolMessage)
  assert msgs[0].tool_call_id == "call_A"

  assert isinstance(msgs[1], oci_models.UserMessage)
  assert msgs[1].content[0].text == "Here is the weather and some extra text."


def test_build_chat_details_flattens_multiple_tool_messages(oci_llm):
  import oci.generative_ai_inference.models as oci_models

  part1 = Part.from_function_response(
      name="get_weather", response={"result": "Sunny, 22°C"}
  )
  part1.function_response.id = "call_A"

  part2 = Part.from_function_response(
      name="get_price", response={"result": "$150"}
  )
  part2.function_response.id = "call_B"

  request = LlmRequest(
      model="google.gemini-2.5-flash",
      contents=[Content(role="user", parts=[part1, part2])],
  )

  chat_details = oci_llm._build_chat_details(request)
  messages = chat_details.chat_request.messages

  assert len(messages) == 2
  assert isinstance(messages[0], oci_models.ToolMessage)
  assert messages[0].tool_call_id == "call_A"
  assert isinstance(messages[1], oci_models.ToolMessage)
  assert messages[1].tool_call_id == "call_B"


# ---------------------------------------------------------------------------
# _oci_response_to_llm_response
# ---------------------------------------------------------------------------


def test_oci_response_to_llm_response_text():
  response = _make_oci_response(
      text="Here is your answer.", prompt_tokens=8, completion_tokens=4
  )
  llm_resp = _oci_response_to_llm_response(response)

  assert isinstance(llm_resp, LlmResponse)
  assert llm_resp.content.role == "model"
  assert llm_resp.content.parts[0].text == "Here is your answer."
  assert llm_resp.usage_metadata.prompt_token_count == 8
  assert llm_resp.usage_metadata.candidates_token_count == 4
  assert llm_resp.usage_metadata.total_token_count == 12


def test_oci_response_to_llm_response_tool_call():
  response = _make_tool_call_response(
      name="get_weather", args={"city": "Chicago"}
  )
  llm_resp = _oci_response_to_llm_response(response)

  assert llm_resp.content.role == "model"
  fc = llm_resp.content.parts[0].function_call
  assert fc.name == "get_weather"
  assert fc.args == {"city": "Chicago"}
  assert fc.id == "call_abc123"


def test_oci_response_to_llm_response_empty_text():
  response = _make_oci_response(text="")
  response.data.chat_response.choices[0].message.content = []
  llm_resp = _oci_response_to_llm_response(response)
  assert llm_resp.content.parts == []


# ---------------------------------------------------------------------------
# _function_declaration_to_oci_tool
# ---------------------------------------------------------------------------


def test_function_declaration_to_oci_tool_no_parameters():
  import oci.generative_ai_inference.models as oci_models

  fn = types.FunctionDeclaration(
      name="ping",
      description="Check if the service is alive.",
  )
  tool = _function_declaration_to_oci_tool(fn)
  assert isinstance(tool, oci_models.FunctionDefinition)
  assert tool.name == "ping"
  assert tool.description == "Check if the service is alive."
  assert tool.parameters["type"] == "object"
  assert tool.parameters["properties"] == {}


def test_function_declaration_to_oci_tool_with_parameters():
  import oci.generative_ai_inference.models as oci_models

  fn = types.FunctionDeclaration(
      name="get_weather",
      description="Get weather for a city.",
      parameters=types.Schema(
          type=types.Type.OBJECT,
          properties={
              "city": types.Schema(
                  type=types.Type.STRING,
                  description="City name",
              )
          },
          required=["city"],
      ),
  )
  tool = _function_declaration_to_oci_tool(fn)
  assert isinstance(tool, oci_models.FunctionDefinition)
  assert tool.name == "get_weather"
  assert "city" in tool.parameters["properties"]
  assert tool.parameters["required"] == ["city"]


def test_function_declaration_to_oci_tool_json_schema():
  import oci.generative_ai_inference.models as oci_models

  fn = types.FunctionDeclaration(
      name="validate",
      description="Validates a payload.",
      parameters_json_schema={
          "type": "object",
          "properties": {"value": {"type": "string"}},
          "required": ["value"],
      },
  )
  tool = _function_declaration_to_oci_tool(fn)
  assert isinstance(tool, oci_models.FunctionDefinition)
  assert tool.parameters["required"] == ["value"]


# ---------------------------------------------------------------------------
# OCIGenAILlm.generate_content_async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_content_async_text(oci_llm, llm_request):
  fake_response = _make_oci_response(text="Hi! I am Gemini on OCI.")

  with patch.object(oci_llm, "_call_oci", return_value=fake_response):
    responses = [r async for r in oci_llm.generate_content_async(llm_request)]

  assert len(responses) == 1
  assert responses[0].content.parts[0].text == "Hi! I am Gemini on OCI."


@pytest.mark.asyncio
async def test_generate_content_async_yields_llm_response(oci_llm, llm_request):
  with patch.object(oci_llm, "_call_oci", return_value=_make_oci_response()):
    responses = [r async for r in oci_llm.generate_content_async(llm_request)]
  assert all(isinstance(r, LlmResponse) for r in responses)


@pytest.mark.asyncio
async def test_generate_content_async_with_tools(oci_llm):
  request = LlmRequest(
      model="google.gemini-2.0-flash-001",
      contents=[
          Content(
              role="user",
              parts=[Part.from_text(text="What is the weather in Chicago?")],
          )
      ],
      config=types.GenerateContentConfig(
          tools=[
              types.Tool(
                  function_declarations=[
                      types.FunctionDeclaration(
                          name="get_weather",
                          description="Get weather for a city.",
                          parameters=types.Schema(
                              type=types.Type.OBJECT,
                              properties={
                                  "city": types.Schema(type=types.Type.STRING)
                              },
                              required=["city"],
                          ),
                      )
                  ]
              )
          ]
      ),
  )
  tool_response = _make_tool_call_response("get_weather", {"city": "Chicago"})

  with patch.object(oci_llm, "_call_oci", return_value=tool_response):
    responses = [r async for r in oci_llm.generate_content_async(request)]

  fc = responses[0].content.parts[0].function_call
  assert fc.name == "get_weather"
  assert fc.args["city"] == "Chicago"


# ---------------------------------------------------------------------------
# OCIGenAILlm — streaming (stream=True)
# ---------------------------------------------------------------------------


def _make_sse_chunks(
    text_tokens: list[str],
    tool_calls: list[dict] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> list[dict[str, Any]]:
  """Build SSE chunks matching the real OCI GenAI /20231130/ streaming schema.

  Schema (verified against live OCI Gemini stream):
    text:  {"index": 0, "message": {"role": "ASSISTANT",
            "content": [{"type": "TEXT", "text": "..."}]}}
    tools: {"index": 0, "message": {"role": "ASSISTANT",
            "toolCalls": [{"type": "FUNCTION", "name": "...",
                          "arguments": "{...}"}]}}
    finish: {"finishReason": "stop"}
    usage:  {"usage": {"promptTokens": N, "completionTokens": N,
                       "totalTokens": N}}  # camelCase!
  """
  chunks = []

  for token in text_tokens:
    chunks.append({
        "index": 0,
        "message": {
            "role": "ASSISTANT",
            "content": [{"type": "TEXT", "text": token}],
        },
    })

  for tc_idx, tc in enumerate(tool_calls or []):
    chunks.append({
        "index": 0,
        "message": {
            "role": "ASSISTANT",
            "toolCalls": [{
                "type": "FUNCTION",
                "id": tc["id"],
                "name": tc["name"],
                "arguments": json.dumps(tc["args"]),
            }],
        },
    })

  chunks.append({"finishReason": "stop"})
  chunks.append({
      "usage": {
          "promptTokens": prompt_tokens,
          "completionTokens": completion_tokens,
          "totalTokens": prompt_tokens + completion_tokens,
      },
  })
  return chunks


@pytest.mark.asyncio
async def test_streaming_yields_partial_then_final(oci_llm, llm_request):
  """stream=True yields partial=True chunks then a final partial=False response."""
  chunks = _make_sse_chunks(["Hello", " world", "!"])

  with patch.object(oci_llm, "_call_oci_stream", return_value=chunks):
    responses = [
        r
        async for r in oci_llm.generate_content_async(llm_request, stream=True)
    ]

  partial = [r for r in responses if r.partial]
  final = [r for r in responses if not r.partial]

  assert len(partial) == 3  # one per text token
  assert len(final) == 1
  assert partial[0].content.parts[0].text == "Hello"
  assert partial[1].content.parts[0].text == " world"
  assert partial[2].content.parts[0].text == "!"
  # Final aggregates all text
  assert final[0].content.parts[0].text == "Hello world!"


@pytest.mark.asyncio
async def test_streaming_final_has_usage_metadata(oci_llm, llm_request):
  """Final streaming response includes token usage."""
  chunks = _make_sse_chunks(["Hi"], prompt_tokens=8, completion_tokens=3)

  with patch.object(oci_llm, "_call_oci_stream", return_value=chunks):
    responses = [
        r
        async for r in oci_llm.generate_content_async(llm_request, stream=True)
    ]

  final = responses[-1]
  assert not final.partial
  assert final.usage_metadata.prompt_token_count == 8
  assert final.usage_metadata.candidates_token_count == 3
  assert final.usage_metadata.total_token_count == 11


@pytest.mark.asyncio
async def test_streaming_tool_call(oci_llm):
  """Streaming assembles tool call arguments from delta chunks."""
  request = LlmRequest(
      model="google.gemini-2.5-flash",
      contents=[
          Content(
              role="user", parts=[Part.from_text(text="Weather in Chicago?")]
          )
      ],
  )
  chunks = _make_sse_chunks(
      text_tokens=[],
      tool_calls=[{
          "id": "call_stream_1",
          "name": "get_weather",
          "args": {"city": "Chicago"},
      }],
  )

  with patch.object(oci_llm, "_call_oci_stream", return_value=chunks):
    responses = [
        r async for r in oci_llm.generate_content_async(request, stream=True)
    ]

  final = responses[-1]
  assert not final.partial
  fc = final.content.parts[0].function_call
  assert fc.name == "get_weather"
  assert fc.args == {"city": "Chicago"}
  assert fc.id == "call_stream_1"


@pytest.mark.asyncio
async def test_streaming_empty_chunks(oci_llm, llm_request):
  """Empty SSE chunk list yields a single empty final response."""
  with patch.object(oci_llm, "_call_oci_stream", return_value=[]):
    responses = [
        r
        async for r in oci_llm.generate_content_async(llm_request, stream=True)
    ]

  assert len(responses) == 1
  assert not responses[0].partial


@pytest.mark.asyncio
async def test_nonstreaming_uses_call_oci_not_call_oci_stream(
    oci_llm, llm_request
):
  """stream=False path calls _call_oci, not _call_oci_stream."""
  with (
      patch.object(
          oci_llm, "_call_oci", return_value=_make_oci_response()
      ) as mock_call,
      patch.object(oci_llm, "_call_oci_stream") as mock_stream,
  ):
    responses = [
        r
        async for r in oci_llm.generate_content_async(llm_request, stream=False)
    ]

  mock_call.assert_called_once()
  mock_stream.assert_not_called()
  assert len(responses) == 1


@pytest.mark.asyncio
async def test_streaming_uses_call_oci_stream_not_call_oci(
    oci_llm, llm_request
):
  """stream=True path calls _call_oci_stream, not _call_oci."""
  chunks = _make_sse_chunks(["hi"])

  with (
      patch.object(
          oci_llm, "_call_oci_stream", return_value=chunks
      ) as mock_stream,
      patch.object(oci_llm, "_call_oci") as mock_call,
  ):
    responses = [
        r
        async for r in oci_llm.generate_content_async(llm_request, stream=True)
    ]

  mock_stream.assert_called_once()
  mock_call.assert_not_called()


@patch("oci.config.from_file", return_value={})
@patch("oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_call_oci_stream_iterates_sse_via_events_method(
    mock_client_cls, _mock_cfg
):
  """_call_oci_stream must use response.data.events(), not iterate response.data.

  Regression guard: OCI's SDK returns an SSEClient that exposes events() and
  close() but is not directly iterable. Iterating response.data raises
  TypeError at runtime against real OCI.
  """

  class FakeSSEEvent:

    def __init__(self, data: str):
      self.data = data

  class FakeSSEClient:
    """Mimics OCI's SSEClient: exposes events() + close(), not __iter__."""

    def __init__(self, events: list):
      self._events = events
      self.closed = False

    def events(self):
      return iter(self._events)

    def close(self):
      self.closed = True

    def __iter__(self):  # pragma: no cover — must NOT be reached
      raise TypeError("'SSEClient' object is not iterable")

  sse_payload = [
      FakeSSEEvent(
          json.dumps({
              "index": 0,
              "message": {
                  "role": "ASSISTANT",
                  "content": [{"type": "TEXT", "text": "Hi"}],
              },
          })
      ),
      FakeSSEEvent(json.dumps({"finishReason": "stop"})),
      FakeSSEEvent(
          json.dumps({
              "usage": {
                  "promptTokens": 4,
                  "completionTokens": 1,
                  "totalTokens": 5,
              },
          })
      ),
      FakeSSEEvent("[DONE]"),
  ]
  fake_sse = FakeSSEClient(sse_payload)

  mock_client_instance = MagicMock()
  mock_client_cls.return_value = mock_client_instance
  fake_response = MagicMock()
  fake_response.data = fake_sse
  mock_client_instance.chat.return_value = fake_response

  llm = OCIGenAILlm(
      model="google.gemini-2.5-flash",
      compartment_id="ocid1.compartment.oc1..example",
      service_endpoint=(
          "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
      ),
  )
  request = LlmRequest(
      model="google.gemini-2.5-flash",
      contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
  )
  chunks = llm._call_oci_stream(request)

  assert (
      len(chunks) == 3
  )  # text + finish + usage; [DONE] sentinel breaks the loop
  assert chunks[0]["message"]["content"][0]["text"] == "Hi"
  assert chunks[1]["finishReason"] == "stop"
  assert chunks[2]["usage"]["totalTokens"] == 5
  assert fake_sse.closed, "SSEClient.close() must be called after iteration"


# ---------------------------------------------------------------------------
# OCIGenAILlm — concurrent async calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_async_calls(oci_llm):
  """Multiple concurrent generate_content_async calls complete independently."""
  responses_by_call = {}

  async def run_call(call_id: int):
    request = LlmRequest(
        model="google.gemini-2.5-flash",
        contents=[
            Content(role="user", parts=[Part.from_text(text=f"Call {call_id}")])
        ],
    )
    with patch.object(
        oci_llm,
        "_call_oci",
        return_value=_make_oci_response(text=f"Response {call_id}"),
    ):
      results = [r async for r in oci_llm.generate_content_async(request)]
    responses_by_call[call_id] = results

  await asyncio.gather(*[run_call(i) for i in range(5)])

  assert len(responses_by_call) == 5
  for call_id, results in responses_by_call.items():
    assert results[0].content.parts[0].text == f"Response {call_id}"


@pytest.mark.asyncio
async def test_concurrent_streaming_calls(oci_llm):
  """Multiple concurrent streaming calls complete independently."""

  async def run_streaming(call_id: int):
    request = LlmRequest(
        model="google.gemini-2.5-flash",
        contents=[
            Content(
                role="user", parts=[Part.from_text(text=f"Stream {call_id}")]
            )
        ],
    )
    chunks = _make_sse_chunks([f"Stream{call_id}"])
    with patch.object(oci_llm, "_call_oci_stream", return_value=chunks):
      return [
          r async for r in oci_llm.generate_content_async(request, stream=True)
      ]

  all_results = await asyncio.gather(*[run_streaming(i) for i in range(3)])

  for call_id, results in enumerate(all_results):
    final = results[-1]
    assert not final.partial
    assert f"Stream{call_id}" in final.content.parts[0].text


# ---------------------------------------------------------------------------
# OCIGenAILlm — configuration & auth
# ---------------------------------------------------------------------------


def test_missing_compartment_id_raises(llm_request):
  llm = OCIGenAILlm(model="google.gemini-2.5-flash")
  with patch.dict(
      os.environ,
      {k: v for k, v in os.environ.items() if k != "OCI_COMPARTMENT_ID"},
  ):
    os.environ.pop("OCI_COMPARTMENT_ID", None)
    with pytest.raises(ValueError, match="compartment_id"):
      llm._resolve_compartment_id()


def test_compartment_id_from_env(llm_request):
  llm = OCIGenAILlm(model="google.gemini-2.0-flash-001")
  with patch.dict(
      os.environ, {"OCI_COMPARTMENT_ID": "ocid1.compartment.example"}
  ):
    assert llm._resolve_compartment_id() == "ocid1.compartment.example"


def test_service_endpoint_default():
  llm = OCIGenAILlm(model="google.gemini-2.0-flash-001")
  endpoint = llm._resolve_service_endpoint()
  assert "us-chicago-1" in endpoint


def test_service_endpoint_from_env():
  llm = OCIGenAILlm(model="google.gemini-2.0-flash-001")
  custom = "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com"
  with patch.dict(os.environ, {"OCI_SERVICE_ENDPOINT": custom}):
    assert llm._resolve_service_endpoint() == custom


def test_service_endpoint_explicit_overrides_env():
  llm = OCIGenAILlm(
      model="google.gemini-2.0-flash-001",
      service_endpoint="https://custom.endpoint.example.com",
  )
  with patch.dict(
      os.environ, {"OCI_SERVICE_ENDPOINT": "https://ignored.example.com"}
  ):
    assert (
        llm._resolve_service_endpoint() == "https://custom.endpoint.example.com"
    )


@patch("oci.config.from_file", return_value={"region": "us-chicago-1"})
@patch("oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_build_client_api_key(mock_client_cls, mock_from_file):
  llm = OCIGenAILlm(
      model="google.gemini-2.0-flash-001",
      auth_type="API_KEY",
      auth_profile="DEFAULT",
      auth_file_location="~/.oci/config",
  )
  llm._build_client(
      "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
  )
  mock_from_file.assert_called_once_with(
      file_location="~/.oci/config", profile_name="DEFAULT"
  )
  mock_client_cls.assert_called_once()


@patch("oci.auth.signers.InstancePrincipalsSecurityTokenSigner")
@patch("oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_build_client_instance_principal(mock_client_cls, mock_signer_cls):
  llm = OCIGenAILlm(
      model="google.gemini-2.0-flash-001",
      auth_type="INSTANCE_PRINCIPAL",
  )
  llm._build_client(
      "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
  )
  mock_signer_cls.assert_called_once()
  mock_client_cls.assert_called_once()
  _, kwargs = mock_client_cls.call_args
  assert kwargs["config"] == {}


@patch("oci.auth.signers.get_resource_principals_signer")
@patch("oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_build_client_resource_principal(mock_client_cls, mock_signer_fn):
  llm = OCIGenAILlm(
      model="google.gemini-2.0-flash-001",
      auth_type="RESOURCE_PRINCIPAL",
  )
  llm._build_client(
      "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
  )
  mock_signer_fn.assert_called_once()
  mock_client_cls.assert_called_once()


# ---------------------------------------------------------------------------
# OCIGenAILlm._call_oci — verify OCI SDK is called with correct parameters
# ---------------------------------------------------------------------------


@patch("oci.config.from_file", return_value={})
@patch("oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_call_oci_passes_model_and_compartment(mock_client_cls, _mock_cfg):
  mock_client_instance = MagicMock()
  mock_client_cls.return_value = mock_client_instance
  mock_client_instance.chat.return_value = _make_oci_response()

  import oci.generative_ai_inference.models as oci_models  # noqa: F401

  llm = OCIGenAILlm(
      model="google.gemini-2.0-flash-001",
      compartment_id="ocid1.compartment.oc1..example",
      service_endpoint=(
          "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
      ),
  )
  request = LlmRequest(
      model="google.gemini-2.0-flash-001",
      contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
  )
  llm._call_oci(request)

  mock_client_instance.chat.assert_called_once()
  chat_details = mock_client_instance.chat.call_args[0][0]
  assert chat_details.compartment_id == "ocid1.compartment.oc1..example"
  assert chat_details.serving_mode.model_id == "google.gemini-2.0-flash-001"


@patch("oci.config.from_file", return_value={})
@patch("oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_call_oci_passes_system_instruction(mock_client_cls, _mock_cfg):
  import oci.generative_ai_inference.models as oci_models

  mock_client_instance = MagicMock()
  mock_client_cls.return_value = mock_client_instance
  mock_client_instance.chat.return_value = _make_oci_response()

  llm = OCIGenAILlm(
      model="google.gemini-2.0-flash-001",
      compartment_id="ocid1.compartment.oc1..example",
      service_endpoint=(
          "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
      ),
  )
  request = LlmRequest(
      model="google.gemini-2.0-flash-001",
      contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
      config=types.GenerateContentConfig(
          system_instruction="Be concise.",
      ),
  )
  llm._call_oci(request)

  chat_details = mock_client_instance.chat.call_args[0][0]
  messages = chat_details.chat_request.messages
  # System instruction is prepended as a SystemMessage
  assert isinstance(messages[0], oci_models.SystemMessage)
  assert messages[0].content[0].text == "Be concise."


@patch("oci.config.from_file", return_value={})
@patch("oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_call_oci_passes_tools(mock_client_cls, _mock_cfg):
  mock_client_instance = MagicMock()
  mock_client_cls.return_value = mock_client_instance
  mock_client_instance.chat.return_value = _make_oci_response()

  llm = OCIGenAILlm(
      model="google.gemini-2.0-flash-001",
      compartment_id="ocid1.compartment.oc1..example",
      service_endpoint=(
          "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
      ),
  )
  request = LlmRequest(
      model="google.gemini-2.0-flash-001",
      contents=[Content(role="user", parts=[Part.from_text(text="Weather?")])],
      config=types.GenerateContentConfig(
          tools=[
              types.Tool(
                  function_declarations=[
                      types.FunctionDeclaration(
                          name="get_weather",
                          description="Get weather.",
                          parameters=types.Schema(
                              type=types.Type.OBJECT,
                              properties={
                                  "city": types.Schema(type=types.Type.STRING)
                              },
                          ),
                      )
                  ]
              )
          ]
      ),
  )
  llm._call_oci(request)

  chat_details = mock_client_instance.chat.call_args[0][0]
  assert chat_details.chat_request.tools is not None
  assert len(chat_details.chat_request.tools) == 1
  assert chat_details.chat_request.tools[0].name == "get_weather"


# ---------------------------------------------------------------------------
# Serving mode: on-demand (default) vs dedicated (endpoint_id)
# ---------------------------------------------------------------------------


@patch("oci.config.from_file", return_value={})
@patch("oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_call_oci_uses_on_demand_serving_mode_by_default(
    mock_client_cls, _mock_cfg
):
  import oci.generative_ai_inference.models as oci_models

  mock_client_instance = MagicMock()
  mock_client_cls.return_value = mock_client_instance
  mock_client_instance.chat.return_value = _make_oci_response()

  llm = OCIGenAILlm(
      model="google.gemini-2.5-flash",
      compartment_id="ocid1.compartment.oc1..example",
      service_endpoint=(
          "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
      ),
  )
  llm._call_oci(
      LlmRequest(
          model="google.gemini-2.5-flash",
          contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
      )
  )

  chat_details = mock_client_instance.chat.call_args[0][0]
  assert isinstance(chat_details.serving_mode, oci_models.OnDemandServingMode)
  assert chat_details.serving_mode.model_id == "google.gemini-2.5-flash"


@patch("oci.config.from_file", return_value={})
@patch("oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_call_oci_uses_dedicated_serving_mode_when_endpoint_id_set(
    mock_client_cls, _mock_cfg
):
  import oci.generative_ai_inference.models as oci_models

  mock_client_instance = MagicMock()
  mock_client_cls.return_value = mock_client_instance
  mock_client_instance.chat.return_value = _make_oci_response()

  endpoint_ocid = "ocid1.generativeaiendpoint.oc1.us-chicago-1.example"
  llm = OCIGenAILlm(
      model="meta.llama-3.1-70b-instruct",
      endpoint_id=endpoint_ocid,
      compartment_id="ocid1.compartment.oc1..example",
      service_endpoint=(
          "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
      ),
  )
  llm._call_oci(
      LlmRequest(
          model="meta.llama-3.1-70b-instruct",
          contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
      )
  )

  chat_details = mock_client_instance.chat.call_args[0][0]
  assert isinstance(chat_details.serving_mode, oci_models.DedicatedServingMode)
  assert chat_details.serving_mode.endpoint_id == endpoint_ocid


@patch.dict(
    os.environ, {"OCI_ENDPOINT_ID": "ocid1.generativeaiendpoint.oc1..env"}
)
@patch("oci.config.from_file", return_value={})
@patch("oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_call_oci_uses_dedicated_serving_mode_from_env_var(
    mock_client_cls, _mock_cfg
):
  import oci.generative_ai_inference.models as oci_models

  mock_client_instance = MagicMock()
  mock_client_cls.return_value = mock_client_instance
  mock_client_instance.chat.return_value = _make_oci_response()

  llm = OCIGenAILlm(
      model="meta.llama-3.1-70b-instruct",
      compartment_id="ocid1.compartment.oc1..example",
      service_endpoint=(
          "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
      ),
  )
  llm._call_oci(
      LlmRequest(
          model="meta.llama-3.1-70b-instruct",
          contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
      )
  )

  chat_details = mock_client_instance.chat.call_args[0][0]
  assert isinstance(chat_details.serving_mode, oci_models.DedicatedServingMode)
  assert (
      chat_details.serving_mode.endpoint_id
      == "ocid1.generativeaiendpoint.oc1..env"
  )


@patch("oci.config.from_file", return_value={})
@patch("oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_explicit_endpoint_id_overrides_env_var(mock_client_cls, _mock_cfg):
  import oci.generative_ai_inference.models as oci_models

  mock_client_instance = MagicMock()
  mock_client_cls.return_value = mock_client_instance
  mock_client_instance.chat.return_value = _make_oci_response()

  with patch.dict(
      os.environ, {"OCI_ENDPOINT_ID": "ocid1.generativeaiendpoint.oc1..env"}
  ):
    llm = OCIGenAILlm(
        model="meta.llama-3.1-70b-instruct",
        endpoint_id="ocid1.generativeaiendpoint.oc1..explicit",
        compartment_id="ocid1.compartment.oc1..example",
        service_endpoint=(
            "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
        ),
    )
    llm._call_oci(
        LlmRequest(
            model="meta.llama-3.1-70b-instruct",
            contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
        )
    )

  chat_details = mock_client_instance.chat.call_args[0][0]
  assert (
      chat_details.serving_mode.endpoint_id
      == "ocid1.generativeaiendpoint.oc1..explicit"
  )


# ---------------------------------------------------------------------------
# Sampling parameters and max_output_tokens passthrough
# ---------------------------------------------------------------------------


@patch("oci.config.from_file", return_value={})
@patch("oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_call_oci_passes_sampling_params(mock_client_cls, _mock_cfg):
  mock_client_instance = MagicMock()
  mock_client_cls.return_value = mock_client_instance
  mock_client_instance.chat.return_value = _make_oci_response()

  llm = OCIGenAILlm(
      model="google.gemini-2.5-flash",
      compartment_id="ocid1.compartment.oc1..example",
      service_endpoint=(
          "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
      ),
  )
  request = LlmRequest(
      model="google.gemini-2.5-flash",
      contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
      config=types.GenerateContentConfig(
          max_output_tokens=128,
          temperature=0.7,
          top_p=0.9,
          top_k=40,
          frequency_penalty=0.1,
          presence_penalty=0.2,
          seed=42,
          stop_sequences=["END", "STOP"],
      ),
  )
  llm._call_oci(request)

  cr = mock_client_instance.chat.call_args[0][0].chat_request
  assert cr.max_tokens == 128
  assert cr.temperature == 0.7
  assert cr.top_p == 0.9
  assert cr.top_k == 40
  assert cr.frequency_penalty == 0.1
  assert cr.presence_penalty == 0.2
  assert cr.seed == 42
  assert cr.stop == ["END", "STOP"]


@patch("oci.config.from_file", return_value={})
@patch("oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_call_oci_omits_unset_sampling_params(mock_client_cls, _mock_cfg):
  mock_client_instance = MagicMock()
  mock_client_cls.return_value = mock_client_instance
  mock_client_instance.chat.return_value = _make_oci_response()

  llm = OCIGenAILlm(
      model="google.gemini-2.5-flash",
      compartment_id="ocid1.compartment.oc1..example",
      service_endpoint=(
          "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
      ),
  )
  llm._call_oci(
      LlmRequest(
          model="google.gemini-2.5-flash",
          contents=[Content(role="user", parts=[Part.from_text(text="Hi")])],
      )
  )
  cr = mock_client_instance.chat.call_args[0][0].chat_request
  assert cr.temperature is None
  assert cr.top_p is None
  assert cr.top_k is None
  assert cr.stop is None


# ---------------------------------------------------------------------------
# Multimodal content
# ---------------------------------------------------------------------------


@patch("oci.config.from_file", return_value={})
@patch("oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_inline_image_becomes_image_content_with_data_url(
    mock_client_cls, _mock_cfg
):
  import oci.generative_ai_inference.models as oci_models

  mock_client_instance = MagicMock()
  mock_client_cls.return_value = mock_client_instance
  mock_client_instance.chat.return_value = _make_oci_response()

  llm = OCIGenAILlm(
      model="google.gemini-2.5-flash",
      compartment_id="ocid1.compartment.oc1..example",
      service_endpoint=(
          "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
      ),
  )
  png_bytes = b"\x89PNG\r\n\x1a\n_fake"
  request = LlmRequest(
      model="google.gemini-2.5-flash",
      contents=[
          Content(
              role="user",
              parts=[
                  Part.from_text(text="What is this?"),
                  Part(
                      inline_data=types.Blob(
                          mime_type="image/png", data=png_bytes
                      )
                  ),
              ],
          )
      ],
  )
  llm._call_oci(request)

  msg = mock_client_instance.chat.call_args[0][0].chat_request.messages[0]
  assert isinstance(msg, oci_models.UserMessage)
  blocks = msg.content
  assert len(blocks) == 2
  assert isinstance(blocks[0], oci_models.TextContent)
  assert blocks[0].text == "What is this?"
  assert isinstance(blocks[1], oci_models.ImageContent)
  assert blocks[1].image_url.url.startswith("data:image/png;base64,")
  import base64 as _b64

  encoded = blocks[1].image_url.url.split(",", 1)[1]
  assert _b64.b64decode(encoded) == png_bytes


@patch("oci.config.from_file", return_value={})
@patch("oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_file_data_audio_becomes_audio_content(mock_client_cls, _mock_cfg):
  import oci.generative_ai_inference.models as oci_models

  mock_client_instance = MagicMock()
  mock_client_cls.return_value = mock_client_instance
  mock_client_instance.chat.return_value = _make_oci_response()

  llm = OCIGenAILlm(
      model="google.gemini-2.5-flash",
      compartment_id="ocid1.compartment.oc1..example",
      service_endpoint=(
          "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
      ),
  )
  request = LlmRequest(
      model="google.gemini-2.5-flash",
      contents=[
          Content(
              role="user",
              parts=[
                  Part(
                      file_data=types.FileData(
                          file_uri="https://example.com/clip.mp3",
                          mime_type="audio/mpeg",
                      )
                  ),
              ],
          )
      ],
  )
  llm._call_oci(request)

  msg = mock_client_instance.chat.call_args[0][0].chat_request.messages[0]
  blocks = [b for b in msg.content if isinstance(b, oci_models.AudioContent)]
  assert len(blocks) == 1
  assert blocks[0].audio_url.url == "https://example.com/clip.mp3"


@patch("oci.config.from_file", return_value={})
@patch("oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_inline_pdf_becomes_document_content(mock_client_cls, _mock_cfg):
  import oci.generative_ai_inference.models as oci_models

  mock_client_instance = MagicMock()
  mock_client_cls.return_value = mock_client_instance
  mock_client_instance.chat.return_value = _make_oci_response()

  llm = OCIGenAILlm(
      model="google.gemini-2.5-flash",
      compartment_id="ocid1.compartment.oc1..example",
      service_endpoint=(
          "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
      ),
  )
  request = LlmRequest(
      model="google.gemini-2.5-flash",
      contents=[
          Content(
              role="user",
              parts=[
                  Part(
                      inline_data=types.Blob(
                          mime_type="application/pdf", data=b"%PDF-1.4"
                      )
                  ),
              ],
          )
      ],
  )
  llm._call_oci(request)

  msg = mock_client_instance.chat.call_args[0][0].chat_request.messages[0]
  blocks = [b for b in msg.content if isinstance(b, oci_models.DocumentContent)]
  assert len(blocks) == 1
  assert blocks[0].document_url.url.startswith("data:application/pdf;base64,")


# ---------------------------------------------------------------------------
# Response format / structured output
# ---------------------------------------------------------------------------


@patch("oci.config.from_file", return_value={})
@patch("oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_response_schema_emits_json_schema_response_format(
    mock_client_cls, _mock_cfg
):
  import oci.generative_ai_inference.models as oci_models

  mock_client_instance = MagicMock()
  mock_client_cls.return_value = mock_client_instance
  mock_client_instance.chat.return_value = _make_oci_response()

  schema = {
      "title": "Weather",
      "type": "object",
      "properties": {"city": {"type": "string"}, "temp_c": {"type": "number"}},
      "required": ["city", "temp_c"],
  }
  llm = OCIGenAILlm(
      model="google.gemini-2.5-flash",
      compartment_id="ocid1.compartment.oc1..example",
      service_endpoint=(
          "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
      ),
  )
  llm._call_oci(
      LlmRequest(
          model="google.gemini-2.5-flash",
          contents=[
              Content(
                  role="user", parts=[Part.from_text(text="Chicago weather?")]
              )
          ],
          config=types.GenerateContentConfig(
              response_mime_type="application/json",
              response_schema=schema,
          ),
      )
  )

  rf = mock_client_instance.chat.call_args[0][0].chat_request.response_format
  assert isinstance(rf, oci_models.JsonSchemaResponseFormat)
  assert rf.json_schema.name == "Weather"
  assert rf.json_schema.schema == schema
  assert rf.json_schema.is_strict is True


@patch("oci.config.from_file", return_value={})
@patch("oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_response_mime_type_only_emits_json_object_format(
    mock_client_cls, _mock_cfg
):
  import oci.generative_ai_inference.models as oci_models

  mock_client_instance = MagicMock()
  mock_client_cls.return_value = mock_client_instance
  mock_client_instance.chat.return_value = _make_oci_response()

  llm = OCIGenAILlm(
      model="google.gemini-2.5-flash",
      compartment_id="ocid1.compartment.oc1..example",
      service_endpoint=(
          "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
      ),
  )
  llm._call_oci(
      LlmRequest(
          model="google.gemini-2.5-flash",
          contents=[
              Content(role="user", parts=[Part.from_text(text="JSON please")])
          ],
          config=types.GenerateContentConfig(
              response_mime_type="application/json"
          ),
      )
  )

  rf = mock_client_instance.chat.call_args[0][0].chat_request.response_format
  assert isinstance(rf, oci_models.JsonObjectResponseFormat)


# ---------------------------------------------------------------------------
# Reasoning-token surfacing
# ---------------------------------------------------------------------------


@patch("oci.config.from_file", return_value={})
@patch("oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_nonstreaming_surfaces_reasoning_tokens(mock_client_cls, _mock_cfg):
  mock_client_instance = MagicMock()
  mock_client_cls.return_value = mock_client_instance
  resp = _make_oci_response(prompt_tokens=10, completion_tokens=5)
  resp.data.chat_response.usage.completion_tokens_details = MagicMock(
      reasoning_tokens=42
  )
  mock_client_instance.chat.return_value = resp

  llm = OCIGenAILlm(
      model="google.gemini-2.5-flash",
      compartment_id="ocid1.compartment.oc1..example",
      service_endpoint=(
          "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
      ),
  )
  out = _oci_response_to_llm_response(resp)
  assert out.usage_metadata.thoughts_token_count == 42


@pytest.mark.asyncio
async def test_streaming_surfaces_reasoning_tokens(oci_llm, llm_request):
  chunks = _make_sse_chunks(["Hi"], prompt_tokens=8, completion_tokens=3)
  # Inject reasoning tokens into the usage chunk
  chunks[-1]["usage"]["completionTokensDetails"] = {"reasoningTokens": 17}

  with patch.object(oci_llm, "_call_oci_stream", return_value=chunks):
    responses = [
        r
        async for r in oci_llm.generate_content_async(llm_request, stream=True)
    ]
  final = responses[-1]
  assert not final.partial
  assert final.usage_metadata.thoughts_token_count == 17
