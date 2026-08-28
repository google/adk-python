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

import pytest
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from google.adk.models._generate_content_body import to_generate_content_body

# Fields that sit at the top level of a Vertex generateContent request, as of
# google-genai 2.18.1. This list exists to be asserted against, not to be used:
# the helper derives the split rather than reading it. If a release moves a
# field, test_every_config_field_is_classified fails first and names it.
TOP_LEVEL = {
    "cachedContent",
    "labels",
    "modelArmorConfig",
    "safetySettings",
    "serviceTier",
    "systemInstruction",
    "toolConfig",
    "tools",
}

# Config fields the client library consumes. The endpoint rejects them.
CLIENT_ONLY = {
    "httpOptions",
    "automaticFunctionCalling",
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
      model="gemini-3.5-flash",
      contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
      config=types.GenerateContentConfig(**config_kwargs),
  )


def _full_config_kwargs() -> dict:
  """Every field that can be set at once on a Vertex-bound request.

  Two of the 35 are left out on purpose. response_json_schema is mutually
  exclusive with response_schema, and enable_enhanced_civic_answers is Developer
  API only, which test_developer_api_only_field_is_named covers separately.
  """
  return dict(
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
  )


# The bug this whole module exists to prevent.
def test_tools_are_top_level_not_buried_in_generation_config():
  body = to_generate_content_body(_request(temperature=0.2, tools=[_tool()]))

  assert "tools" in body
  assert "tools" not in body["generationConfig"]


def test_temperature_stays_in_generation_config():
  body = to_generate_content_body(_request(temperature=0.2, tools=[_tool()]))

  assert body["generationConfig"]["temperature"] == pytest.approx(0.2)
  assert "temperature" not in body


def test_client_only_fields_never_reach_the_wire():
  body = to_generate_content_body(
      _request(
          temperature=0.2,
          http_options=types.HttpOptions(timeout=60_000),
          automatic_function_calling=types.AutomaticFunctionCallingConfig(
              disable=True
          ),
          should_return_http_response=True,
      )
  )

  emitted = set(body) | set(body.get("generationConfig", {}))
  assert not (CLIENT_ONLY & emitted)


def test_url_routing_key_is_stripped():
  # The converter returns the model under "_url" for the genai client to build
  # a path with. It is not body content and Vertex 400s on it.
  body = to_generate_content_body(_request(temperature=0.2))

  assert "_url" not in body


def test_full_field_set_splits_eight_twentytwo_three():
  body = to_generate_content_body(_request(**_full_config_kwargs()))

  top = set(body) - {"contents", "generationConfig"}
  generation = set(body["generationConfig"])

  assert top == TOP_LEVEL
  assert not (TOP_LEVEL & generation), "a top-level field was buried"
  assert not (CLIENT_ONLY & (top | generation))
  # 33 set, 8 up top, 3 dropped, so 22 sampling settings and nothing lost.
  assert len(generation) == 22


def test_model_selection_config_is_renamed_to_model_config():
  # The only field that changes name in transit. Anyone diffing their own config
  # keys against the emitted body gets a false miss on this one.
  body = to_generate_content_body(
      _request(
          model_selection_config=types.ModelSelectionConfig(
              feature_selection_preference="BALANCED"
          )
      )
  )

  assert "modelConfig" in body["generationConfig"]
  assert "modelSelectionConfig" not in body["generationConfig"]


def test_developer_api_only_field_is_named_in_the_error():
  request = _request(enable_enhanced_civic_answers=True)

  with pytest.raises(ValueError) as excinfo:
    to_generate_content_body(request, vertexai=True)

  message = str(excinfo.value)
  assert "enable_enhanced_civic_answers" in message
  # The point of the annotation: say where it was set, not just that it is bad.
  assert "LlmRequest.config" in message
  assert "Vertex AI" in message


def test_developer_api_accepts_what_vertex_rejects():
  body = to_generate_content_body(
      _request(enable_enhanced_civic_answers=True), vertexai=False
  )

  assert body is not None


def test_string_system_instruction_becomes_a_content_block():
  body = to_generate_content_body(_request(system_instruction="Be terse."))

  assert body["systemInstruction"]["parts"][0]["text"] == "Be terse."


def test_no_config_still_produces_a_body():
  request = LlmRequest(
      model="gemini-3.5-flash",
      contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
  )

  body = to_generate_content_body(request)

  assert body["contents"][0]["parts"][0]["text"] == "hi"


# The test that keeps the other tests honest.
def test_every_config_field_is_classified():
  """Fail when GenerateContentConfig gains a field nobody has classified.

  This is the whole argument for the helper. A hand written mapping goes stale
  silently: the new field lands in generationConfig, the endpoint ignores it,
  and no one finds out until a feature quietly stops working. This turns that
  into a red test naming the field.
  """
  known = (
      TOP_LEVEL
      | CLIENT_ONLY
      | {"enableEnhancedCivicAnswers", "responseJsonSchema"}
  )
  body = to_generate_content_body(_request(**_full_config_kwargs()))
  classified = (
      (set(body) - {"contents", "generationConfig"})
      | set(body["generationConfig"])
      | known
  )

  # Aliases as the wire spells them, which is how the body is keyed.
  declared = {
      field.alias or name
      for name, field in types.GenerateContentConfig.model_fields.items()
  }
  # modelSelectionConfig is emitted as modelConfig; see the rename test.
  declared.discard("modelSelectionConfig")
  classified.add("modelSelectionConfig")

  unclassified = declared - classified
  assert not unclassified, (
      "GenerateContentConfig gained field(s) with no known destination:"
      f" {sorted(unclassified)}. Add them to TOP_LEVEL or CLIENT_ONLY and"
      " confirm where the converter puts them."
  )


def test_method_matches_function():
  """LlmRequest.to_generate_content_body is a thin delegate, not a fork."""
  request = _request(**_full_config_kwargs())

  assert request.to_generate_content_body() == to_generate_content_body(request)


def test_method_passes_the_api_mode_through():
  request = _request(enable_enhanced_civic_answers=True)

  assert request.to_generate_content_body(vertexai=False) is not None
  with pytest.raises(ValueError):
    request.to_generate_content_body(vertexai=True)
