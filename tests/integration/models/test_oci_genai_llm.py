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

"""Integration tests for OCIGenAILlm against live OCI Generative AI service.

Required environment variables:
  OCI_COMPARTMENT_ID   — OCI compartment OCID
  OCI_REGION           — OCI region (default: us-chicago-1)

Optional:
  OCI_AUTH_TYPE        — API_KEY | INSTANCE_PRINCIPAL | RESOURCE_PRINCIPAL
                         (default: API_KEY)
  OCI_AUTH_PROFILE     — OCI config profile (default: DEFAULT)
  OCI_AUTH_FILE        — path to OCI config file (default: ~/.oci/config)
"""

import os

from google.adk.models.llm_request import LlmRequest
from google.adk.models.oci_genai_llm import OCIGenAILlm
from google.genai import types
from google.genai.types import Content
from google.genai.types import Part
import pytest


# ---------------------------------------------------------------------------
# Skip the entire module when required env vars are absent
# ---------------------------------------------------------------------------

# OCI tests do not use any Google backend (GOOGLE_AI / Vertex AI).
# Override the autouse llm_backend fixture from the integration conftest so
# these tests are not duplicated across backends.
@pytest.fixture(autouse=True)
def llm_backend():
  yield


pytestmark = pytest.mark.skipif(
    not os.environ.get("OCI_COMPARTMENT_ID"),
    reason=(
        "OCI integration tests require OCI_COMPARTMENT_ID to be set. "
        "Set OCI_COMPARTMENT_ID (and optionally OCI_REGION) to run."
    ),
)

_COMPARTMENT_ID = os.environ.get("OCI_COMPARTMENT_ID", "")
_REGION = os.environ.get("OCI_REGION", "us-chicago-1")
_SERVICE_ENDPOINT = (
    f"https://inference.generativeai.{_REGION}.oci.oraclecloud.com"
)
_AUTH_TYPE = os.environ.get("OCI_AUTH_TYPE", "API_KEY")
_AUTH_PROFILE = os.environ.get("OCI_AUTH_PROFILE", "DEFAULT")
_AUTH_FILE = os.environ.get("OCI_AUTH_FILE", "~/.oci/config")

_GEMINI_MODEL = "google.gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gemini_llm() -> OCIGenAILlm:
  return OCIGenAILlm(
      model=_GEMINI_MODEL,
      compartment_id=_COMPARTMENT_ID,
      service_endpoint=_SERVICE_ENDPOINT,
      auth_type=_AUTH_TYPE,
      auth_profile=_AUTH_PROFILE,
      auth_file_location=_AUTH_FILE,
      max_tokens=512,
  )



def _simple_request(model: str, text: str = "Reply with one word: hello.") -> LlmRequest:
  return LlmRequest(
      model=model,
      contents=[Content(role="user", parts=[Part.from_text(text=text)])],
  )


def _request_with_system(model: str) -> LlmRequest:
  return LlmRequest(
      model=model,
      contents=[
          Content(
              role="user",
              parts=[Part.from_text(text="What is your name?")],
          )
      ],
      config=types.GenerateContentConfig(
          system_instruction="Your name is Oracle. Always introduce yourself as Oracle.",
      ),
  )


def _request_with_tool(model: str) -> LlmRequest:
  return LlmRequest(
      model=model,
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
                          description="Get the current weather for a city.",
                          parameters=types.Schema(
                              type=types.Type.OBJECT,
                              properties={
                                  "city": types.Schema(
                                      type=types.Type.STRING,
                                      description="The city name.",
                                  )
                              },
                              required=["city"],
                          ),
                      )
                  ]
              )
          ]
      ),
  )


# ---------------------------------------------------------------------------
# Gemini (google.gemini-2.0-flash-001) tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gemini_generate_content_text(gemini_llm):
  """Gemini on OCI returns a non-empty text response."""
  responses = [
      r
      async for r in gemini_llm.generate_content_async(
          _simple_request(_GEMINI_MODEL), stream=False
      )
  ]
  assert len(responses) == 1
  assert responses[0].content.role == "model"
  assert responses[0].content.parts
  assert responses[0].content.parts[0].text.strip()


@pytest.mark.asyncio
async def test_gemini_generate_content_usage_metadata(gemini_llm):
  """Response includes token usage metadata."""
  responses = [
      r
      async for r in gemini_llm.generate_content_async(
          _simple_request(_GEMINI_MODEL), stream=False
      )
  ]
  usage = responses[0].usage_metadata
  assert usage.prompt_token_count > 0
  assert usage.candidates_token_count > 0
  assert usage.total_token_count == (
      usage.prompt_token_count + usage.candidates_token_count
  )


@pytest.mark.asyncio
async def test_gemini_generate_content_with_system_instruction(gemini_llm):
  """System instruction is respected."""
  responses = [
      r
      async for r in gemini_llm.generate_content_async(
          _request_with_system(_GEMINI_MODEL), stream=False
      )
  ]
  text = responses[0].content.parts[0].text.lower()
  assert "oracle" in text


@pytest.mark.asyncio
async def test_gemini_generate_content_tool_call(gemini_llm):
  """Gemini returns a function call when a tool is provided."""
  responses = [
      r
      async for r in gemini_llm.generate_content_async(
          _request_with_tool(_GEMINI_MODEL), stream=False
      )
  ]
  parts = responses[0].content.parts
  function_calls = [p for p in parts if p.function_call]
  assert function_calls, "Expected at least one function call in the response"
  fc = function_calls[0].function_call
  assert fc.name == "get_weather"
  assert "city" in fc.args


@pytest.mark.asyncio
async def test_gemini_generate_content_streaming_text(gemini_llm):
  """Streaming returns partial chunks followed by a final non-partial response."""
  responses = [
      r
      async for r in gemini_llm.generate_content_async(
          _simple_request(_GEMINI_MODEL), stream=True
      )
  ]
  assert responses, "Expected at least one response chunk"
  partial_responses = [r for r in responses if r.partial]
  final_responses = [r for r in responses if not r.partial]
  assert partial_responses, "Expected at least one partial (streaming) chunk"
  assert len(final_responses) == 1, "Expected exactly one final (non-partial) response"
  full_text = "".join(
      p.text
      for r in partial_responses
      for p in (r.content.parts or [])
      if p.text
  )
  assert full_text.strip(), "Streamed text should be non-empty"


@pytest.mark.asyncio
async def test_gemini_generate_content_streaming_usage_metadata(gemini_llm):
  """Final streaming response includes token usage metadata."""
  responses = [
      r
      async for r in gemini_llm.generate_content_async(
          _simple_request(_GEMINI_MODEL), stream=True
      )
  ]
  final = next(r for r in responses if not r.partial)
  usage = final.usage_metadata
  assert usage is not None
  assert usage.prompt_token_count > 0
  assert usage.candidates_token_count > 0
  assert usage.total_token_count == (
      usage.prompt_token_count + usage.candidates_token_count
  )


@pytest.mark.asyncio
async def test_gemini_generate_content_streaming_tool_call(gemini_llm):
  """Streaming returns a function call when a tool is provided."""
  responses = [
      r
      async for r in gemini_llm.generate_content_async(
          _request_with_tool(_GEMINI_MODEL), stream=True
      )
  ]
  final = next(r for r in responses if not r.partial)
  parts = final.content.parts or []
  function_calls = [p for p in parts if p.function_call]
  assert function_calls, "Expected at least one function call in the streaming response"
  fc = function_calls[0].function_call
  assert fc.name == "get_weather"
  assert "city" in fc.args


@pytest.mark.asyncio
async def test_gemini_generate_content_concurrent(gemini_llm):
  """Multiple concurrent non-streaming requests complete independently."""
  import asyncio

  async def single_call(text: str) -> str:
    responses = [
        r
        async for r in gemini_llm.generate_content_async(
            _simple_request(_GEMINI_MODEL, text=text), stream=False
        )
    ]
    return responses[0].content.parts[0].text

  results = await asyncio.gather(*[
      single_call(f"Reply with the number {i} only.")
      for i in range(3)
  ])
  assert len(results) == 3
  for result in results:
    assert result.strip(), "Each concurrent response should be non-empty"


@pytest.mark.asyncio
async def test_gemini_multi_turn(gemini_llm):
  """Multi-turn conversation passes history correctly."""
  history = [
      Content(role="user", parts=[Part.from_text(text="My favourite colour is blue.")]),
      Content(role="model", parts=[Part.from_text(text="Got it, blue is a great colour!")]),
  ]
  follow_up = Content(
      role="user",
      parts=[Part.from_text(text="What is my favourite colour?")],
  )
  request = LlmRequest(
      model=_GEMINI_MODEL,
      contents=history + [follow_up],
  )
  responses = [r async for r in gemini_llm.generate_content_async(request)]
  text = responses[0].content.parts[0].text.lower()
  assert "blue" in text
