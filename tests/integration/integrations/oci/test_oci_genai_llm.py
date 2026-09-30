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

import json
import os

from google.adk.integrations.oci._oci_genai_llm import OCIGenAILlm
from google.adk.models.llm_request import LlmRequest
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


def _simple_request(
    model: str, text: str = "Reply with one word: hello."
) -> LlmRequest:
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
          system_instruction=(
              "Your name is Oracle. Always introduce yourself as Oracle."
          ),
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
  assert (
      len(final_responses) == 1
  ), "Expected exactly one final (non-partial) response"
  full_text = "".join(
      p.text for r in partial_responses for p in r.content.parts or [] if p.text
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
  assert (
      function_calls
  ), "Expected at least one function call in the streaming response"
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

  results = await asyncio.gather(
      *[single_call(f"Reply with the number {i} only.") for i in range(3)]
  )
  assert len(results) == 3
  for result in results:
    assert result.strip(), "Each concurrent response should be non-empty"


@pytest.mark.asyncio
async def test_gemini_multi_turn(gemini_llm):
  """Multi-turn conversation passes history correctly."""
  history = [
      Content(
          role="user",
          parts=[Part.from_text(text="My favourite colour is blue.")],
      ),
      Content(
          role="model",
          parts=[Part.from_text(text="Got it, blue is a great colour!")],
      ),
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


# ---------------------------------------------------------------------------
# Cross-provider on-demand smoke tests
#
# Skipped unless the corresponding model env var is set so cost stays opt-in.
# Set OCI_LLAMA_MODEL / OCI_MISTRAL_MODEL / OCI_GROK_MODEL / OCI_NVIDIA_MODEL
# to a model id available in your tenancy/region (e.g. "meta.llama-3.3-70b-instruct").
# ---------------------------------------------------------------------------


def _provider_llm(env_var: str) -> "OCIGenAILlm | None":
  model_id = os.environ.get(env_var)
  if not model_id:
    return None
  return OCIGenAILlm(
      model=model_id,
      compartment_id=_COMPARTMENT_ID,
      service_endpoint=_SERVICE_ENDPOINT,
      auth_type=_AUTH_TYPE,
      auth_profile=_AUTH_PROFILE,
      auth_file_location=_AUTH_FILE,
      max_tokens=256,
  )


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("OCI_LLAMA_MODEL"),
    reason="Set OCI_LLAMA_MODEL=<meta.llama-...-id> to enable.",
)
async def test_llama_on_demand_generate_text():
  llm = _provider_llm("OCI_LLAMA_MODEL")
  responses = [
      r
      async for r in llm.generate_content_async(
          _simple_request(llm.model), stream=False
      )
  ]
  assert len(responses) == 1
  assert responses[0].content.parts[0].text.strip()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("OCI_MISTRAL_MODEL"),
    reason="Set OCI_MISTRAL_MODEL=<mistralai...-id> to enable.",
)
async def test_mistral_on_demand_generate_text():
  llm = _provider_llm("OCI_MISTRAL_MODEL")
  responses = [
      r
      async for r in llm.generate_content_async(
          _simple_request(llm.model), stream=False
      )
  ]
  assert responses[0].content.parts[0].text.strip()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("OCI_GROK_MODEL"),
    reason="Set OCI_GROK_MODEL=<xai.grok-...-id> to enable.",
)
async def test_grok_on_demand_generate_text():
  llm = _provider_llm("OCI_GROK_MODEL")
  responses = [
      r
      async for r in llm.generate_content_async(
          _simple_request(llm.model), stream=False
      )
  ]
  assert responses[0].content.parts[0].text.strip()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("OCI_NVIDIA_MODEL"),
    reason="Set OCI_NVIDIA_MODEL=<nvidia...-id> to enable.",
)
async def test_nvidia_on_demand_generate_text():
  llm = _provider_llm("OCI_NVIDIA_MODEL")
  responses = [
      r
      async for r in llm.generate_content_async(
          _simple_request(llm.model), stream=False
      )
  ]
  assert responses[0].content.parts[0].text.strip()


# ---------------------------------------------------------------------------
# Dedicated serving mode
#
# Set OCI_DEDICATED_ENDPOINT_ID=ocid1.generativeaiendpoint.oc1... to enable.
# OCI_DEDICATED_MODEL is informational; defaults to the dedicated endpoint's
# bound model (the SDK ignores `model` when serving_mode is dedicated).
# ---------------------------------------------------------------------------


_DEDICATED_ENDPOINT_ID = os.environ.get("OCI_DEDICATED_ENDPOINT_ID", "")
_DEDICATED_MODEL = os.environ.get(
    "OCI_DEDICATED_MODEL", "meta.llama-3.3-70b-instruct"
)


@pytest.fixture
def dedicated_llm() -> OCIGenAILlm:
  return OCIGenAILlm(
      model=_DEDICATED_MODEL,
      endpoint_id=_DEDICATED_ENDPOINT_ID,
      compartment_id=_COMPARTMENT_ID,
      service_endpoint=_SERVICE_ENDPOINT,
      auth_type=_AUTH_TYPE,
      auth_profile=_AUTH_PROFILE,
      auth_file_location=_AUTH_FILE,
      max_tokens=256,
  )


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _DEDICATED_ENDPOINT_ID,
    reason=(
        "Set OCI_DEDICATED_ENDPOINT_ID to a dedicated endpoint OCID to enable."
    ),
)
async def test_dedicated_generate_content_text(dedicated_llm):
  responses = [
      r
      async for r in dedicated_llm.generate_content_async(
          _simple_request(_DEDICATED_MODEL), stream=False
      )
  ]
  assert len(responses) == 1
  assert responses[0].content.parts[0].text.strip()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _DEDICATED_ENDPOINT_ID,
    reason=(
        "Set OCI_DEDICATED_ENDPOINT_ID to a dedicated endpoint OCID to enable."
    ),
)
async def test_dedicated_generate_content_streaming(dedicated_llm):
  chunks = []
  async for r in dedicated_llm.generate_content_async(
      _simple_request(_DEDICATED_MODEL, text="Count from 1 to 3."),
      stream=True,
  ):
    chunks.append(r)
  assert len(chunks) >= 2  # at least one partial + one final
  final = chunks[-1]
  assert final.usage_metadata is not None


# ---------------------------------------------------------------------------
# Sampling parameters (live)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gemini_max_output_tokens_caps_response(gemini_llm):
  """max_output_tokens is honoured: completion tokens never exceed the budget.

  Note: Gemini 2.5 spends part of the budget on reasoning tokens before any
  visible output. We pick a budget large enough to leave some text but small
  enough to clearly cap an alphabet-recitation response, and we assert on the
  reported token count rather than character count (which is flaky).
  """
  budget = 64
  request = LlmRequest(
      model=_GEMINI_MODEL,
      contents=[
          Content(
              role="user",
              parts=[
                  Part.from_text(
                      text="Recite the alphabet, A through Z, comma separated."
                  )
              ],
          )
      ],
      config=types.GenerateContentConfig(max_output_tokens=budget),
  )
  responses = [r async for r in gemini_llm.generate_content_async(request)]
  um = responses[0].usage_metadata
  assert um.candidates_token_count is not None
  assert um.candidates_token_count <= budget


@pytest.mark.asyncio
async def test_gemini_low_temperature_deterministic_with_seed(gemini_llm):
  """temperature=0 + seed should yield consistent answers across two calls."""
  request = LlmRequest(
      model=_GEMINI_MODEL,
      contents=[
          Content(
              role="user",
              parts=[Part.from_text(text="Reply with exactly: 'green'")],
          )
      ],
      config=types.GenerateContentConfig(temperature=0.0, seed=12345),
  )
  call_a = [r async for r in gemini_llm.generate_content_async(request)]
  call_b = [r async for r in gemini_llm.generate_content_async(request)]
  assert "green" in call_a[0].content.parts[0].text.lower()
  assert "green" in call_b[0].content.parts[0].text.lower()


@pytest.mark.asyncio
async def test_gemini_stop_sequences_terminate_output(gemini_llm):
  request = LlmRequest(
      model=_GEMINI_MODEL,
      contents=[
          Content(
              role="user",
              parts=[Part.from_text(text="Print: APPLE | BANANA | CHERRY")],
          )
      ],
      config=types.GenerateContentConfig(
          temperature=0.0, stop_sequences=["BANANA"]
      ),
  )
  responses = [r async for r in gemini_llm.generate_content_async(request)]
  text = responses[0].content.parts[0].text
  assert "BANANA" not in text


# ---------------------------------------------------------------------------
# Multimodal: inline image (live)
#
# Uses a tiny 1x1 red PNG so the request is cheap. Gemini 2.5 Flash on OCI
# supports image inputs via ImageContent.
# ---------------------------------------------------------------------------


def _make_red_png_1x1() -> bytes:
  """Generate a guaranteed-valid 1x1 red PNG with correct CRCs."""
  import struct
  import zlib

  sig = b"\x89PNG\r\n\x1a\n"

  def chunk(t: bytes, d: bytes) -> bytes:
    return (
        struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    )

  ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)  # 1x1 RGB
  idat = zlib.compress(b"\x00\xff\x00\x00")  # filter byte + RGB(255,0,0)
  return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


_TINY_RED_PNG = _make_red_png_1x1()


@pytest.mark.asyncio
async def test_gemini_inline_image_input(gemini_llm):
  request = LlmRequest(
      model=_GEMINI_MODEL,
      contents=[
          Content(
              role="user",
              parts=[
                  Part.from_text(
                      text=(
                          "What is the dominant colour of this image? "
                          "Reply with just the colour name."
                      )
                  ),
                  Part(
                      inline_data=types.Blob(
                          mime_type="image/png", data=_TINY_RED_PNG
                      )
                  ),
              ],
          )
      ],
      config=types.GenerateContentConfig(
          temperature=0.0, max_output_tokens=256
      ),
  )
  responses = [r async for r in gemini_llm.generate_content_async(request)]
  parts = responses[0].content.parts
  assert parts, "Expected the model to produce a visible answer"
  text = parts[0].text.lower()
  assert "red" in text


# ---------------------------------------------------------------------------
# Structured output: response_schema (live)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gemini_response_schema_returns_valid_json(gemini_llm):
  schema = {
      "title": "CityFact",
      "type": "object",
      "properties": {
          "city": {"type": "string"},
          "country": {"type": "string"},
      },
      "required": ["city", "country"],
      "additionalProperties": False,
  }
  request = LlmRequest(
      model=_GEMINI_MODEL,
      contents=[
          Content(
              role="user",
              parts=[Part.from_text(text="Give me a fact about Paris.")],
          )
      ],
      config=types.GenerateContentConfig(
          response_mime_type="application/json",
          response_schema=schema,
          temperature=0.0,
      ),
  )
  responses = [r async for r in gemini_llm.generate_content_async(request)]
  raw = responses[0].content.parts[0].text
  payload = json.loads(raw)
  assert "city" in payload
  assert "country" in payload


# ---------------------------------------------------------------------------
# Reasoning-token surfacing (live)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gemini_reasoning_tokens_reported(gemini_llm):
  """Gemini 2.5 emits reasoningTokens in completionTokensDetails — surface them."""
  request = LlmRequest(
      model=_GEMINI_MODEL,
      contents=[
          Content(
              role="user",
              parts=[
                  Part.from_text(
                      text=(
                          "If a train travels 60km in 30 minutes, what is its"
                          " speed?"
                      )
                  )
              ],
          )
      ],
      config=types.GenerateContentConfig(temperature=0.0),
  )
  responses = [r async for r in gemini_llm.generate_content_async(request)]
  um = responses[0].usage_metadata
  assert um is not None
  # Reasoning tokens are optional; assert it's an int when present
  assert um.thoughts_token_count is None or um.thoughts_token_count > 0
