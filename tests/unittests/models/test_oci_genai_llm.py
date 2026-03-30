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

from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.models.oci_genai_llm import _content_to_oci_message
from google.adk.models.oci_genai_llm import _function_declaration_to_oci_tool
from google.adk.models.oci_genai_llm import _oci_response_to_llm_response
from google.adk.models.oci_genai_llm import OCIGenAILlm
from google.genai import types
from google.genai.types import Content
from google.genai.types import Part
import pytest


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
      service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
  )


@pytest.fixture
def llm_request():
  return LlmRequest(
      model="google.gemini-2.5-flash",
      contents=[
          Content(role="user", parts=[Part.from_text(text="Hello")])
      ],
      config=types.GenerateContentConfig(
          system_instruction="You are a helpful assistant.",
      ),
  )


# ---------------------------------------------------------------------------
# supported_models
# ---------------------------------------------------------------------------


def test_supported_models_gemini():
  assert any(
      "gemini" in p for p in OCIGenAILlm.supported_models()
  )


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
  msg = _content_to_oci_message(content)
  assert isinstance(msg, oci_models.UserMessage)
  assert msg.role == oci_models.UserMessage.ROLE_USER
  assert msg.content[0].text == "Hi there"


def test_content_to_oci_message_assistant_text():
  import oci.generative_ai_inference.models as oci_models

  content = Content(role="model", parts=[Part.from_text(text="I can help.")])
  msg = _content_to_oci_message(content)
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
  msg = _content_to_oci_message(content)
  assert isinstance(msg, oci_models.UserMessage)
  assert "First" in msg.content[0].text
  assert "Second" in msg.content[0].text


def test_content_to_oci_message_function_call():
  import oci.generative_ai_inference.models as oci_models

  part = Part.from_function_call(name="get_weather", args={"city": "Toronto"})
  content = Content(role="model", parts=[part])
  msg = _content_to_oci_message(content)
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
  msg = _content_to_oci_message(content)
  assert isinstance(msg, oci_models.ToolMessage)
  assert msg.tool_call_id == "call_xyz"
  assert msg.content[0].text


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
    responses = [
        r async for r in oci_llm.generate_content_async(llm_request)
    ]

  assert len(responses) == 1
  assert responses[0].content.parts[0].text == "Hi! I am Gemini on OCI."


@pytest.mark.asyncio
async def test_generate_content_async_yields_llm_response(oci_llm, llm_request):
  with patch.object(oci_llm, "_call_oci", return_value=_make_oci_response()):
    responses = [
        r async for r in oci_llm.generate_content_async(llm_request)
    ]
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
  """Build a list of OpenAI-compatible SSE chunk dicts (OCI streaming format)."""
  chunks = []

  # Text delta chunks
  for token in text_tokens:
    chunks.append({
        "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}]
    })

  # Tool call chunks
  for tc in tool_calls or []:
    # First chunk: id + name
    chunks.append({
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": tc["id"],
                    "function": {"name": tc["name"], "arguments": ""},
                }]
            },
            "finish_reason": None,
        }]
    })
    # Second chunk: arguments
    chunks.append({
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": {"arguments": json.dumps(tc["args"])},
                }]
            },
            "finish_reason": None,
        }]
    })

  # Final chunk with usage
  chunks.append({
      "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
      "usage": {
          "prompt_tokens": prompt_tokens,
          "completion_tokens": completion_tokens,
          "total_tokens": prompt_tokens + completion_tokens,
      },
  })
  return chunks


@pytest.mark.asyncio
async def test_streaming_yields_partial_then_final(oci_llm, llm_request):
  """stream=True yields partial=True chunks then a final partial=False response."""
  chunks = _make_sse_chunks(["Hello", " world", "!"])

  with patch.object(oci_llm, "_call_oci_stream", return_value=chunks):
    responses = [
        r async for r in oci_llm.generate_content_async(llm_request, stream=True)
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
        r async for r in oci_llm.generate_content_async(llm_request, stream=True)
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
          Content(role="user", parts=[Part.from_text(text="Weather in Chicago?")])
      ],
  )
  chunks = _make_sse_chunks(
      text_tokens=[],
      tool_calls=[{"id": "call_stream_1", "name": "get_weather", "args": {"city": "Chicago"}}],
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
        r async for r in oci_llm.generate_content_async(llm_request, stream=True)
    ]

  assert len(responses) == 1
  assert not responses[0].partial


@pytest.mark.asyncio
async def test_nonstreaming_uses_call_oci_not_call_oci_stream(oci_llm, llm_request):
  """stream=False path calls _call_oci, not _call_oci_stream."""
  with patch.object(oci_llm, "_call_oci", return_value=_make_oci_response()) as mock_call, \
       patch.object(oci_llm, "_call_oci_stream") as mock_stream:
    responses = [r async for r in oci_llm.generate_content_async(llm_request, stream=False)]

  mock_call.assert_called_once()
  mock_stream.assert_not_called()
  assert len(responses) == 1


@pytest.mark.asyncio
async def test_streaming_uses_call_oci_stream_not_call_oci(oci_llm, llm_request):
  """stream=True path calls _call_oci_stream, not _call_oci."""
  chunks = _make_sse_chunks(["hi"])

  with patch.object(oci_llm, "_call_oci_stream", return_value=chunks) as mock_stream, \
       patch.object(oci_llm, "_call_oci") as mock_call:
    responses = [r async for r in oci_llm.generate_content_async(llm_request, stream=True)]

  mock_stream.assert_called_once()
  mock_call.assert_not_called()


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
        contents=[Content(role="user", parts=[Part.from_text(text=f"Call {call_id}")])],
    )
    with patch.object(
        oci_llm, "_call_oci",
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
        contents=[Content(role="user", parts=[Part.from_text(text=f"Stream {call_id}")])],
    )
    chunks = _make_sse_chunks([f"Stream{call_id}"])
    with patch.object(oci_llm, "_call_oci_stream", return_value=chunks):
      return [r async for r in oci_llm.generate_content_async(request, stream=True)]

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
  with patch.dict(os.environ, {k: v for k, v in os.environ.items() if k != "OCI_COMPARTMENT_ID"}):
    os.environ.pop("OCI_COMPARTMENT_ID", None)
    with pytest.raises(ValueError, match="compartment_id"):
      llm._resolve_compartment_id()


def test_compartment_id_from_env(llm_request):
  llm = OCIGenAILlm(model="google.gemini-2.0-flash-001")
  with patch.dict(os.environ, {"OCI_COMPARTMENT_ID": "ocid1.compartment.example"}):
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
  with patch.dict(os.environ, {"OCI_SERVICE_ENDPOINT": "https://ignored.example.com"}):
    assert llm._resolve_service_endpoint() == "https://custom.endpoint.example.com"


@patch("oci.config.from_file", return_value={"region": "us-chicago-1"})
@patch("oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_build_client_api_key(mock_client_cls, mock_from_file):
  llm = OCIGenAILlm(
      model="google.gemini-2.0-flash-001",
      auth_type="API_KEY",
      auth_profile="DEFAULT",
      auth_file_location="~/.oci/config",
  )
  llm._build_client("https://inference.generativeai.us-chicago-1.oci.oraclecloud.com")
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
  llm._build_client("https://inference.generativeai.us-chicago-1.oci.oraclecloud.com")
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
  llm._build_client("https://inference.generativeai.us-chicago-1.oci.oraclecloud.com")
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
      service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
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
      service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
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
      service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
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
