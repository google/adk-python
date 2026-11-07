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

"""Tests for the LlmRequest -> generateContent body conversion."""

import base64
import json

from google.adk.models._generate_content_body import to_generate_content_body
from google.adk.models.llm_request import LlmRequest
from google.genai import _common as _genai_common
from google.genai import models as _genai_models
from google.genai import types
import pytest

from google import genai

_CONFIG_FIELDS = types.GenerateContentConfig.model_fields

# Fields newer than google-genai 2.30.0, the oldest release the pin allows, so
# they are expected only where the installed release has them.
# generation_output_config arrived in 2.32 (Vertex AI only, sent as
# outputConfig) and store in 2.33 (Gemini Developer API only).
_HAS_GENERATION_OUTPUT_CONFIG = "generation_output_config" in _CONFIG_FIELDS
_HAS_STORE = "store" in _CONFIG_FIELDS

# Fields that sit at the top level of a Vertex generateContent request. This
# list exists to be asserted against, not to be used: the helper derives the
# split rather than reading it. If a release moves one of these fields,
# test_full_field_set_splits_top_level_generation_and_client_only fails and
# names it.
TOP_LEVEL = {
    "cachedContent",
    "continuationToken",
    "labels",
    "modelArmorConfig",
    "safetySettings",
    "serviceTier",
    "systemInstruction",
    "toolConfig",
    "tools",
} | ({"outputConfig"} if _HAS_GENERATION_OUTPUT_CONFIG else set())

# Config fields the client library consumes. The endpoint rejects them.
CLIENT_ONLY = {
    "automaticContinuation",
    "automaticFunctionCalling",
    "httpOptions",
    "shouldReturnHttpResponse",
}


def _tool() -> types.Tool:
  return types.Tool(
      function_declarations=[
          types.FunctionDeclaration(
              name="a_tool",
              description="Look up the weather in a city.",
              parameters=types.Schema(
                  type=types.Type.OBJECT,
                  properties={"city": types.Schema(type=types.Type.STRING)},
              ),
          )
      ]
  )


def _request(**config_kwargs) -> LlmRequest:
  return LlmRequest(
      model="gemini-2.5-flash",
      contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
      config=types.GenerateContentConfig(**config_kwargs),
  )


def _full_config_kwargs() -> dict:
  """Every field that can be set at once on a Vertex-bound request.

  Left out on purpose: response_json_schema, which is mutually exclusive with
  response_schema, and the Developer API only fields, which have tests of
  their own: enable_enhanced_civic_answers in
  test_developer_api_only_field_is_named_in_the_error and store in
  test_store_is_top_level_on_the_developer_api. generation_output_config is
  set only where the installed google-genai has it.
  """
  kwargs = dict(
      system_instruction="You are a research agent.",
      tools=[_tool()],
      tool_config=types.ToolConfig(
          function_calling_config=types.FunctionCallingConfig(mode="AUTO")
      ),
      safety_settings=[
          types.SafetySetting(
              category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH"
          )
      ],
      cached_content="projects/p/locations/us-central1/cachedContents/123",
      labels={"team": "research"},
      http_options=types.HttpOptions(timeout=60_000),
      automatic_function_calling=types.AutomaticFunctionCallingConfig(
          disable=True
      ),
      should_return_http_response=True,
      temperature=0.2,
      top_p=0.9,
      top_k=40,
      candidate_count=1,
      max_output_tokens=8192,
      stop_sequences=["STOP"],
      response_logprobs=True,
      logprobs=3,
      presence_penalty=0.1,
      frequency_penalty=0.1,
      seed=42,
      response_mime_type="application/json",
      response_schema=types.Schema(
          type=types.Type.OBJECT,
          properties={"answer": types.Schema(type=types.Type.STRING)},
      ),
      routing_config=types.GenerationConfigRoutingConfig(
          auto_mode=types.GenerationConfigRoutingConfigAutoRoutingMode(
              model_routing_preference="BALANCED"
          )
      ),
      model_selection_config=types.ModelSelectionConfig(
          feature_selection_preference="BALANCED"
      ),
      response_modalities=["TEXT"],
      media_resolution="MEDIA_RESOLUTION_MEDIUM",
      speech_config=types.SpeechConfig(
          voice_config=types.VoiceConfig(
              prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")
          )
      ),
      audio_timestamp=True,
      thinking_config=types.ThinkingConfig(
          include_thoughts=True, thinking_budget=4096
      ),
      image_config=types.ImageConfig(aspect_ratio="1:1"),
      model_armor_config=types.ModelArmorConfig(
          prompt_template_name="projects/p/locations/us-central1/templates/t"
      ),
      service_tier="standard",
      audio_transcription_config=types.AudioTranscriptionConfig(),
      continuation_token=b"token",
      automatic_continuation=True,
  )
  if _HAS_GENERATION_OUTPUT_CONFIG:
    kwargs["generation_output_config"] = types.GenerationOutputConfig(
        output_uri_prefix="gs://bucket/prefix/"
    )
  return kwargs


def test_tools_are_top_level_not_buried_in_generation_config():
  """Tools sit at the top level of the request body rather than generationConfig."""
  body = to_generate_content_body(_request(temperature=0.2, tools=[_tool()]))

  assert "tools" in body
  assert "tools" not in body["generationConfig"]


def test_temperature_stays_in_generation_config():
  """Sampling parameters remain nested inside generationConfig."""
  body = to_generate_content_body(_request(temperature=0.2, tools=[_tool()]))

  assert body["generationConfig"]["temperature"] == pytest.approx(0.2)
  assert "temperature" not in body


def test_client_only_fields_never_reach_the_wire():
  """Client-only GenerateContentConfig fields are omitted from the wire body."""
  body = to_generate_content_body(
      _request(
          temperature=0.2,
          http_options=types.HttpOptions(timeout=60_000),
          automatic_function_calling=types.AutomaticFunctionCallingConfig(
              disable=True
          ),
          should_return_http_response=True,
          automatic_continuation=True,
      )
  )

  emitted = set(body) | set(body.get("generationConfig", {}))
  assert not (CLIENT_ONLY & emitted)


def test_url_routing_key_is_stripped():
  """The internal _url routing key produced by the converter is removed."""
  body = to_generate_content_body(_request(temperature=0.2))

  assert "_url" not in body


def test_body_survives_a_json_round_trip():
  """Nested config objects and inline bytes come back as plain JSON types."""
  request = LlmRequest(
      model="gemini-2.5-flash",
      contents=[
          types.Content(
              role="user",
              parts=[
                  types.Part(text="What is in this image?"),
                  types.Part.from_bytes(data=b"\x89PNG", mime_type="image/png"),
              ],
          )
      ],
      config=types.GenerateContentConfig(
          tools=[_tool()],
          safety_settings=[
              types.SafetySetting(
                  category="HARM_CATEGORY_HATE_SPEECH",
                  threshold="BLOCK_ONLY_HIGH",
              )
          ],
          thinking_config=types.ThinkingConfig(thinking_budget=128),
      ),
  )

  body = to_generate_content_body(request)

  assert json.loads(json.dumps(body)) == body


def test_inline_bytes_are_base64_encoded():
  """Inline data is sent as base64 text that decodes to the original bytes."""
  request = LlmRequest(
      model="gemini-2.5-flash",
      contents=[
          types.Content(
              role="user",
              parts=[
                  types.Part.from_bytes(data=b"\x89PNG", mime_type="image/png")
              ],
          )
      ],
  )

  body = to_generate_content_body(request)

  data = body["contents"][0]["parts"][0]["inlineData"]["data"]
  assert base64.b64decode(data) == b"\x89PNG"


def test_full_field_set_splits_top_level_generation_and_client_only():
  """All Vertex-compatible fields land in their expected wire locations."""
  body = to_generate_content_body(
      _request(**_full_config_kwargs()), vertexai=True
  )

  top = set(body) - {"contents", "generationConfig"}
  generation = set(body["generationConfig"])

  assert top == TOP_LEVEL
  assert not (TOP_LEVEL & generation), "a top-level field was buried"
  assert not (CLIENT_ONLY & (top | generation))
  # 35 set (36 with generation_output_config), 9 up top (or 10), 4 dropped, so
  # 22 sampling settings and nothing lost.
  assert len(generation) == 22


def test_model_selection_config_is_renamed_to_model_config():
  """model_selection_config is emitted as modelConfig inside generationConfig."""
  body = to_generate_content_body(
      _request(
          model_selection_config=types.ModelSelectionConfig(
              feature_selection_preference="BALANCED"
          )
      ),
      vertexai=True,
  )

  assert "modelConfig" in body["generationConfig"]
  assert "modelSelectionConfig" not in body["generationConfig"]


def test_developer_api_only_field_is_named_in_the_error():
  """Setting a Developer-API-only field on Vertex AI raises an annotated error."""
  request = _request(enable_enhanced_civic_answers=True)

  with pytest.raises(ValueError) as excinfo:
    to_generate_content_body(request, vertexai=True)

  message = str(excinfo.value)
  assert "enable_enhanced_civic_answers" in message
  assert "LlmRequest.config.enable_enhanced_civic_answers" in message
  assert "Vertex AI" in message


def test_vertex_only_field_is_named_in_the_error_on_developer_api():
  """Setting a Vertex-only field on the Developer API raises an annotated error."""
  request = _request(
      model_armor_config=types.ModelArmorConfig(
          prompt_template_name="projects/p/locations/us-central1/templates/t"
      )
  )

  with pytest.raises(ValueError) as excinfo:
    to_generate_content_body(request, vertexai=False)

  message = str(excinfo.value)
  assert "model_armor_config" in message
  assert "LlmRequest.config.model_armor_config" in message
  assert "the Gemini Developer API" in message


def test_developer_api_accepts_what_vertex_rejects():
  """enable_enhanced_civic_answers succeeds when targeting the Developer API."""
  body = to_generate_content_body(
      _request(enable_enhanced_civic_answers=True), vertexai=False
  )

  assert body["generationConfig"]["enableEnhancedCivicAnswers"] is True


@pytest.mark.skipif(not _HAS_STORE, reason="store arrived in google-genai 2.33")
def test_store_is_top_level_on_the_developer_api():
  """store sits at the top level of a Developer API request body."""
  body = to_generate_content_body(_request(store=True), vertexai=False)

  assert body["store"] is True
  assert "store" not in body.get("generationConfig", {})


def test_vertexai_defaults_to_the_developer_api_like_the_rest_of_adk(
    monkeypatch,
):
  """Without vertexai, the body is for ADK's default, the Developer API."""
  monkeypatch.delenv("GOOGLE_GENAI_USE_ENTERPRISE", raising=False)
  monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
  request = _request(enable_enhanced_civic_answers=True)

  body = to_generate_content_body(request)

  assert body["generationConfig"]["enableEnhancedCivicAnswers"] is True
  assert request.to_generate_content_body() == body


def test_vertexai_defaults_to_vertex_ai_in_enterprise_mode(monkeypatch):
  """GOOGLE_GENAI_USE_ENTERPRISE moves the default to Vertex AI, as in ADK."""
  monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "1")
  template = "projects/p/locations/us-central1/templates/t"
  request = _request(
      model_armor_config=types.ModelArmorConfig(prompt_template_name=template)
  )

  body = to_generate_content_body(request)

  assert body["modelArmorConfig"] == {"prompt_template_name": template}
  assert request.to_generate_content_body() == body


@pytest.mark.parametrize(
    "unwrap", [False, True], ids=["Client", "BaseApiClient"]
)
def test_api_client_resolves_vertex_cached_content_name(unwrap):
  """A genai Client, or its BaseApiClient, expands short cached_content names."""
  client = genai.Client(
      vertexai=True, project="my-proj", location="us-central1"
  )
  request = _request(cached_content="cachedContents/123")

  body = request.to_generate_content_body(
      api_client=client._api_client if unwrap else client
  )

  assert (
      body["cachedContent"]
      == "projects/my-proj/locations/us-central1/cachedContents/123"
  )


@pytest.mark.parametrize("cached_content", ["cachedContents/123", "123"])
def test_short_cached_content_on_vertex_without_api_client_names_the_field(
    cached_content,
):
  """A short cached_content name on Vertex AI needs a client, and says so."""
  request = _request(cached_content=cached_content)

  with pytest.raises(ValueError) as excinfo:
    to_generate_content_body(request, vertexai=True)

  message = str(excinfo.value)
  assert "LlmRequest.config.cached_content" in message
  assert "api_client" in message


def test_short_cached_content_on_developer_api_needs_no_client():
  """The Developer API expands a short cached_content name without a client."""
  body = to_generate_content_body(
      _request(cached_content="123"), vertexai=False
  )

  assert body["cachedContent"] == "cachedContents/123"


def test_string_system_instruction_becomes_a_content_block():
  """A string system_instruction is converted into a Content dictionary."""
  body = to_generate_content_body(_request(system_instruction="Be terse."))

  assert body["systemInstruction"]["parts"][0]["text"] == "Be terse."


def test_no_config_still_produces_a_body():
  """A request with default config still produces a valid request body."""
  request = LlmRequest(
      model="gemini-2.5-flash",
      contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
  )

  body = to_generate_content_body(request)

  assert body["contents"][0]["parts"][0]["text"] == "hi"


@pytest.mark.parametrize(
    "module, symbol",
    [
        (types, "_GenerateContentParameters"),
        (_genai_models, "_GenerateContentParameters_to_vertex"),
        (_genai_models, "_GenerateContentParameters_to_mldev"),
        (_genai_common, "convert_to_dict"),
        (_genai_common, "encode_unserializable_types"),
    ],
)
def test_private_genai_symbols_still_exist(module, symbol):
  """Fail in CI when a google-genai release renames a private helper we call.

  These are underscore-prefixed, so they sit outside genai's semver contract
  while our pin allows any 2.x. Without this, a rename would surface as an
  AttributeError inside someone's BaseLlm at request time.
  """
  assert hasattr(module, symbol), (
      f"google-genai no longer exposes {module.__name__}.{symbol}, which"
      " to_generate_content_body depends on."
  )


def test_method_matches_function():
  """LlmRequest.to_generate_content_body delegates to to_generate_content_body."""
  request = _request(**_full_config_kwargs())

  assert request.to_generate_content_body(
      vertexai=True
  ) == to_generate_content_body(request, vertexai=True)


def test_method_passes_the_api_mode_through():
  """LlmRequest.to_generate_content_body forwards the vertexai argument."""
  request = _request(enable_enhanced_civic_answers=True)

  assert request.to_generate_content_body(vertexai=False) is not None
  with pytest.raises(ValueError):
    request.to_generate_content_body(vertexai=True)
