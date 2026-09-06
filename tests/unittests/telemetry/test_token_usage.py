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

from google.adk.models.llm_response import LlmResponse
from google.adk.telemetry import _token_usage
from google.genai import types
import pytest


@pytest.fixture(name="usage_metadata")
def fixture_usage_metadata() -> types.GenerateContentResponseUsageMetadata:
  """Provides a baseline GenerateContentResponseUsageMetadata fixture with all token counts initialized to None."""
  m = types.GenerateContentResponseUsageMetadata()
  m.prompt_token_count = None
  m.tool_use_prompt_token_count = None
  m.candidates_token_count = None
  m.thoughts_token_count = None
  m.cached_content_token_count = None
  return m


def test_from_llm_responses_keeps_usage_a_trailing_response_omits():
  """Tests from_llm_responses when the last response reports no usage."""
  reported = LlmResponse(
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          prompt_token_count=10, candidates_token_count=4
      )
  )
  trailing = LlmResponse(partial=True)

  token_usage = _token_usage.TokenUsage.from_llm_responses([reported, trailing])

  assert token_usage is not None
  assert token_usage.input_tokens == 10
  assert token_usage.output_tokens == 4


def test_from_llm_responses_takes_the_newest_report():
  """Tests from_llm_responses when several responses report usage."""
  first = LlmResponse(
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          prompt_token_count=10, candidates_token_count=1
      )
  )
  newest = LlmResponse(
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          prompt_token_count=10, candidates_token_count=4
      )
  )

  token_usage = _token_usage.TokenUsage.from_llm_responses([first, newest])

  assert token_usage is not None
  assert token_usage.output_tokens == 4


def test_from_llm_responses_without_any_report():
  """Tests from_llm_responses when no response reports usage."""
  assert _token_usage.TokenUsage.from_llm_responses([]) is None
  assert _token_usage.TokenUsage.from_llm_responses([LlmResponse()]) is None


def test_from_llm_responses_report_counting_no_tokens():
  """Tests from_llm_responses when the report counts neither direction."""
  countless = LlmResponse(
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          total_token_count=7
      )
  )

  assert _token_usage.TokenUsage.from_llm_responses([countless]) is None


def test_input_tokens_all_present(
    usage_metadata: types.GenerateContentResponseUsageMetadata,
):
  """Tests input_tokens when all components are present."""
  usage_metadata.prompt_token_count = 10
  usage_metadata.tool_use_prompt_token_count = 5
  token_usage = _token_usage.TokenUsage.from_usage_metadata(usage_metadata)
  assert token_usage.input_tokens == 15


def test_input_tokens_only_prompt(
    usage_metadata: types.GenerateContentResponseUsageMetadata,
):
  """Tests input_tokens when only prompt_token_count is present."""
  usage_metadata.prompt_token_count = 10
  usage_metadata.tool_use_prompt_token_count = None
  token_usage = _token_usage.TokenUsage.from_usage_metadata(usage_metadata)
  assert token_usage.input_tokens == 10


def test_input_tokens_only_tool(
    usage_metadata: types.GenerateContentResponseUsageMetadata,
):
  """Tests input_tokens when only tool_use_prompt_token_count is present."""
  usage_metadata.prompt_token_count = None
  usage_metadata.tool_use_prompt_token_count = 5
  token_usage = _token_usage.TokenUsage.from_usage_metadata(usage_metadata)
  assert token_usage.input_tokens == 5


def test_input_tokens_none(
    usage_metadata: types.GenerateContentResponseUsageMetadata,
):
  """Tests input_tokens when all components are None."""
  usage_metadata.prompt_token_count = None
  usage_metadata.tool_use_prompt_token_count = None
  token_usage = _token_usage.TokenUsage.from_usage_metadata(usage_metadata)
  assert token_usage.input_tokens is None


def test_input_tokens_zero(
    usage_metadata: types.GenerateContentResponseUsageMetadata,
):
  """Tests input_tokens when all components are zero."""
  usage_metadata.prompt_token_count = 0
  usage_metadata.tool_use_prompt_token_count = 0
  token_usage = _token_usage.TokenUsage.from_usage_metadata(usage_metadata)
  assert token_usage.input_tokens == 0


def test_input_tokens_metadata_none():
  """Tests input_tokens when usage_metadata is None."""
  token_usage = _token_usage.TokenUsage.from_usage_metadata(None)
  assert token_usage.input_tokens is None


def test_input_tokens_missing_tool_use_attr():
  """Tests input_tokens when tool_use_prompt_token_count is missing."""
  token_usage = _token_usage.TokenUsage.from_usage_metadata(
      types.GenerateContentResponseUsageMetadata(prompt_token_count=10)
  )
  assert token_usage.input_tokens == 10


def test_output_tokens_all_present(
    usage_metadata: types.GenerateContentResponseUsageMetadata,
):
  """Tests output_tokens when all components are present."""
  usage_metadata.candidates_token_count = 20
  usage_metadata.thoughts_token_count = 8
  token_usage = _token_usage.TokenUsage.from_usage_metadata(usage_metadata)
  assert token_usage.output_tokens == 28


def test_output_tokens_only_candidates(
    usage_metadata: types.GenerateContentResponseUsageMetadata,
):
  """Tests output_tokens when only candidates_token_count is present."""
  usage_metadata.candidates_token_count = 20
  usage_metadata.thoughts_token_count = None
  token_usage = _token_usage.TokenUsage.from_usage_metadata(usage_metadata)
  assert token_usage.output_tokens == 20


def test_output_tokens_only_thoughts(
    usage_metadata: types.GenerateContentResponseUsageMetadata,
):
  """Tests output_tokens when only thoughts_token_count is present."""
  usage_metadata.candidates_token_count = None
  usage_metadata.thoughts_token_count = 8
  token_usage = _token_usage.TokenUsage.from_usage_metadata(usage_metadata)
  assert token_usage.output_tokens == 8


def test_output_tokens_none(
    usage_metadata: types.GenerateContentResponseUsageMetadata,
):
  """Tests output_tokens when all components are None."""
  usage_metadata.candidates_token_count = None
  usage_metadata.thoughts_token_count = None
  token_usage = _token_usage.TokenUsage.from_usage_metadata(usage_metadata)
  assert token_usage.output_tokens is None


def test_output_tokens_zero(
    usage_metadata: types.GenerateContentResponseUsageMetadata,
):
  """Tests output_tokens when all components are zero."""
  usage_metadata.candidates_token_count = 0
  usage_metadata.thoughts_token_count = 0
  token_usage = _token_usage.TokenUsage.from_usage_metadata(usage_metadata)
  assert token_usage.output_tokens == 0


def test_output_tokens_metadata_none():
  """Tests output_tokens when usage_metadata is None."""
  token_usage = _token_usage.TokenUsage.from_usage_metadata(None)
  assert token_usage.output_tokens is None


def test_to_attributes_full(
    usage_metadata: types.GenerateContentResponseUsageMetadata,
):
  """Tests to_attributes with all attributes present."""
  usage_metadata.prompt_token_count = 10
  usage_metadata.tool_use_prompt_token_count = 5
  usage_metadata.candidates_token_count = 20
  usage_metadata.thoughts_token_count = 8
  usage_metadata.cached_content_token_count = 100

  token_usage = _token_usage.TokenUsage.from_usage_metadata(usage_metadata)
  attrs = token_usage.to_attributes()
  assert attrs[_token_usage.GEN_AI_USAGE_INPUT_TOKENS] == 15
  assert attrs[_token_usage.GEN_AI_USAGE_OUTPUT_TOKENS] == 28
  assert attrs[_token_usage.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 100
  assert attrs[_token_usage.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS] == 8


def test_to_attributes_partial(
    usage_metadata: types.GenerateContentResponseUsageMetadata,
):
  """Tests to_attributes with only some attributes present."""
  usage_metadata.prompt_token_count = 10
  usage_metadata.tool_use_prompt_token_count = None
  usage_metadata.candidates_token_count = None
  usage_metadata.thoughts_token_count = None
  usage_metadata.cached_content_token_count = None

  token_usage = _token_usage.TokenUsage.from_usage_metadata(usage_metadata)
  attrs = token_usage.to_attributes()
  assert attrs[_token_usage.GEN_AI_USAGE_INPUT_TOKENS] == 10
  assert _token_usage.GEN_AI_USAGE_OUTPUT_TOKENS not in attrs
  assert _token_usage.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS not in attrs
  assert _token_usage.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS not in attrs


def test_to_attributes_metadata_none():
  """Tests to_attributes when usage_metadata is None."""
  token_usage = _token_usage.TokenUsage.from_usage_metadata(None)
  assert token_usage.to_attributes() == {}


def test_to_attributes_with_zeros(
    usage_metadata: types.GenerateContentResponseUsageMetadata,
):
  """Tests to_attributes when all attributes are zero."""
  usage_metadata.prompt_token_count = 0
  usage_metadata.tool_use_prompt_token_count = 0
  usage_metadata.candidates_token_count = 0
  usage_metadata.thoughts_token_count = 0
  usage_metadata.cached_content_token_count = 0

  token_usage = _token_usage.TokenUsage.from_usage_metadata(usage_metadata)
  attrs = token_usage.to_attributes()
  assert attrs[_token_usage.GEN_AI_USAGE_INPUT_TOKENS] == 0
  assert attrs[_token_usage.GEN_AI_USAGE_OUTPUT_TOKENS] == 0
  assert attrs[_token_usage.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 0
  assert attrs[_token_usage.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS] == 0


def test_to_attributes_missing_optional_attrs():
  """Tests to_attributes when optional attributes are missing from metadata object."""
  token_usage = _token_usage.TokenUsage.from_usage_metadata(
      types.GenerateContentResponseUsageMetadata(
          prompt_token_count=10, candidates_token_count=20
      )
  )
  attrs = token_usage.to_attributes()
  assert attrs[_token_usage.GEN_AI_USAGE_INPUT_TOKENS] == 10
  assert attrs[_token_usage.GEN_AI_USAGE_OUTPUT_TOKENS] == 20


def test_to_attributes_cache_creation(
    usage_metadata: types.GenerateContentResponseUsageMetadata,
):
  """Tests to_attributes when cache_creation_input_tokens is present."""
  usage_metadata.prompt_token_count = 10
  object.__setattr__(usage_metadata, "cache_creation_input_tokens", 50)

  token_usage = _token_usage.TokenUsage.from_usage_metadata(usage_metadata)
  attrs = token_usage.to_attributes()
  assert attrs[_token_usage.GEN_AI_USAGE_INPUT_TOKENS] == 10
  assert attrs[_token_usage.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS] == 50


def test_subset_bucket_accessors(
    usage_metadata: types.GenerateContentResponseUsageMetadata,
):
  """The cache_read / tool / reasoning buckets read their Gemini fields."""
  usage_metadata.prompt_token_count = 100
  usage_metadata.tool_use_prompt_token_count = 20
  usage_metadata.candidates_token_count = 30
  usage_metadata.thoughts_token_count = 15
  usage_metadata.cached_content_token_count = 60

  token_usage = _token_usage.TokenUsage.from_usage_metadata(usage_metadata)
  assert token_usage.cache_read_input_tokens == 60
  assert token_usage.tool_input_tokens == 20
  assert token_usage.reasoning_output_tokens == 15

  empty = _token_usage.TokenUsage.from_usage_metadata(None)
  assert empty.cache_read_input_tokens is None
  assert empty.tool_input_tokens is None
  assert empty.reasoning_output_tokens is None


def test_invocation_totals_sum_every_bucket_across_calls():
  """Totals accumulate every bucket, and derive the total they report."""
  calls = 3
  prompt_tokens = 100
  tool_use_prompt_tokens = 20
  candidates_tokens = 30
  thoughts_tokens = 15
  cached_content_tokens = 60

  totals = _token_usage.TokenUsage()
  for _ in range(calls):
    totals.add(
        _token_usage.TokenUsage.from_usage_metadata(
            types.GenerateContentResponseUsageMetadata(
                prompt_token_count=prompt_tokens,
                tool_use_prompt_token_count=tool_use_prompt_tokens,
                candidates_token_count=candidates_tokens,
                thoughts_token_count=thoughts_tokens,
                cached_content_token_count=cached_content_tokens,
                # Deliberately inconsistent; the derived total must ignore it.
                total_token_count=9999,
            )
        )
    )

  want_input = calls * (prompt_tokens + tool_use_prompt_tokens)
  want_output = calls * (candidates_tokens + thoughts_tokens)
  assert totals.input_tokens == want_input
  assert totals.output_tokens == want_output
  assert totals.cache_read_input_tokens == calls * cached_content_tokens
  assert totals.reasoning_output_tokens == calls * thoughts_tokens
  assert totals.tool_input_tokens == calls * tool_use_prompt_tokens
  assert totals.total_tokens == want_input + want_output


def test_add_keeps_a_bucket_none_until_something_reports_it():
  """A bucket no call reported stays None; a reported zero does not."""
  totals = _token_usage.TokenUsage()
  totals.add(
      _token_usage.TokenUsage.from_usage_metadata(
          types.GenerateContentResponseUsageMetadata(
              prompt_token_count=10,
              candidates_token_count=4,
              cached_content_token_count=0,
          )
      )
  )

  assert totals.cache_read_input_tokens == 0
  assert totals.reasoning_output_tokens is None
  assert totals.to_attributes() == {
      _token_usage.GEN_AI_USAGE_INPUT_TOKENS: 10,
      _token_usage.GEN_AI_USAGE_OUTPUT_TOKENS: 4,
      _token_usage.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS: 0,
  }


def test_add_starts_a_bucket_the_first_report_brings():
  """A later call reporting a bucket earlier ones omitted starts it there."""
  totals = _token_usage.TokenUsage(input_tokens=10)
  totals.add(_token_usage.TokenUsage(input_tokens=5, tool_input_tokens=5))

  assert totals.input_tokens == 15
  assert totals.tool_input_tokens == 5


def test_add_sums_the_span_only_buckets():
  """cache_creation and system_instruction accumulate like the rest."""
  totals = _token_usage.TokenUsage(
      cache_creation_input_tokens=30, system_instruction_tokens=5
  )
  totals.add(
      _token_usage.TokenUsage(
          cache_creation_input_tokens=20, system_instruction_tokens=5
      )
  )

  assert totals.cache_creation_input_tokens == 50
  assert totals.system_instruction_tokens == 10
