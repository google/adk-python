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


def test_input_token_count_all_present(
    usage_metadata: types.GenerateContentResponseUsageMetadata,
):
  """Tests input_token_count when all components are present."""
  usage_metadata.prompt_token_count = 10
  usage_metadata.tool_use_prompt_token_count = 5
  token_usage = _token_usage.TokenUsage(usage_metadata)
  assert token_usage.input_token_count == 15


def test_input_token_count_only_prompt(
    usage_metadata: types.GenerateContentResponseUsageMetadata,
):
  """Tests input_token_count when only prompt_token_count is present."""
  usage_metadata.prompt_token_count = 10
  usage_metadata.tool_use_prompt_token_count = None
  token_usage = _token_usage.TokenUsage(usage_metadata)
  assert token_usage.input_token_count == 10


def test_input_token_count_only_tool(
    usage_metadata: types.GenerateContentResponseUsageMetadata,
):
  """Tests input_token_count when only tool_use_prompt_token_count is present."""
  usage_metadata.prompt_token_count = None
  usage_metadata.tool_use_prompt_token_count = 5
  token_usage = _token_usage.TokenUsage(usage_metadata)
  assert token_usage.input_token_count == 5


def test_input_token_count_none(
    usage_metadata: types.GenerateContentResponseUsageMetadata,
):
  """Tests input_token_count when all components are None."""
  usage_metadata.prompt_token_count = None
  usage_metadata.tool_use_prompt_token_count = None
  token_usage = _token_usage.TokenUsage(usage_metadata)
  assert token_usage.input_token_count is None


def test_input_token_count_zero(
    usage_metadata: types.GenerateContentResponseUsageMetadata,
):
  """Tests input_token_count when all components are zero."""
  usage_metadata.prompt_token_count = 0
  usage_metadata.tool_use_prompt_token_count = 0
  token_usage = _token_usage.TokenUsage(usage_metadata)
  assert token_usage.input_token_count == 0


def test_input_token_count_metadata_none():
  """Tests input_token_count when usage_metadata is None."""
  token_usage = _token_usage.TokenUsage(None)
  assert token_usage.input_token_count is None


def test_input_token_count_missing_tool_use_attr():
  """Tests input_token_count when tool_use_prompt_token_count is missing."""
  token_usage = _token_usage.TokenUsage(
      types.GenerateContentResponseUsageMetadata(prompt_token_count=10)
  )
  assert token_usage.input_token_count == 10


def test_output_token_count_all_present(
    usage_metadata: types.GenerateContentResponseUsageMetadata,
):
  """Tests output_token_count when all components are present."""
  usage_metadata.candidates_token_count = 20
  usage_metadata.thoughts_token_count = 8
  token_usage = _token_usage.TokenUsage(usage_metadata)
  assert token_usage.output_token_count == 28


def test_output_token_count_only_candidates(
    usage_metadata: types.GenerateContentResponseUsageMetadata,
):
  """Tests output_token_count when only candidates_token_count is present."""
  usage_metadata.candidates_token_count = 20
  usage_metadata.thoughts_token_count = None
  token_usage = _token_usage.TokenUsage(usage_metadata)
  assert token_usage.output_token_count == 20


def test_output_token_count_only_thoughts(
    usage_metadata: types.GenerateContentResponseUsageMetadata,
):
  """Tests output_token_count when only thoughts_token_count is present."""
  usage_metadata.candidates_token_count = None
  usage_metadata.thoughts_token_count = 8
  token_usage = _token_usage.TokenUsage(usage_metadata)
  assert token_usage.output_token_count == 8


def test_output_token_count_none(
    usage_metadata: types.GenerateContentResponseUsageMetadata,
):
  """Tests output_token_count when all components are None."""
  usage_metadata.candidates_token_count = None
  usage_metadata.thoughts_token_count = None
  token_usage = _token_usage.TokenUsage(usage_metadata)
  assert token_usage.output_token_count is None


def test_output_token_count_zero(
    usage_metadata: types.GenerateContentResponseUsageMetadata,
):
  """Tests output_token_count when all components are zero."""
  usage_metadata.candidates_token_count = 0
  usage_metadata.thoughts_token_count = 0
  token_usage = _token_usage.TokenUsage(usage_metadata)
  assert token_usage.output_token_count == 0


def test_output_token_count_metadata_none():
  """Tests output_token_count when usage_metadata is None."""
  token_usage = _token_usage.TokenUsage(None)
  assert token_usage.output_token_count is None


def test_to_attributes_full(
    usage_metadata: types.GenerateContentResponseUsageMetadata,
):
  """Tests to_attributes with all attributes present."""
  usage_metadata.prompt_token_count = 10
  usage_metadata.tool_use_prompt_token_count = 5
  usage_metadata.candidates_token_count = 20
  usage_metadata.thoughts_token_count = 8
  usage_metadata.cached_content_token_count = 100

  token_usage = _token_usage.TokenUsage(usage_metadata)
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

  token_usage = _token_usage.TokenUsage(usage_metadata)
  attrs = token_usage.to_attributes()
  assert attrs[_token_usage.GEN_AI_USAGE_INPUT_TOKENS] == 10
  assert _token_usage.GEN_AI_USAGE_OUTPUT_TOKENS not in attrs
  assert _token_usage.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS not in attrs
  assert _token_usage.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS not in attrs


def test_to_attributes_metadata_none():
  """Tests to_attributes when usage_metadata is None."""
  token_usage = _token_usage.TokenUsage(None)
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

  token_usage = _token_usage.TokenUsage(usage_metadata)
  attrs = token_usage.to_attributes()
  assert attrs[_token_usage.GEN_AI_USAGE_INPUT_TOKENS] == 0
  assert attrs[_token_usage.GEN_AI_USAGE_OUTPUT_TOKENS] == 0
  assert attrs[_token_usage.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 0
  assert attrs[_token_usage.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS] == 0


def test_to_attributes_missing_optional_attrs():
  """Tests to_attributes when optional attributes are missing from metadata object."""
  token_usage = _token_usage.TokenUsage(
      types.GenerateContentResponseUsageMetadata(
          prompt_token_count=10, candidates_token_count=20
      )
  )
  attrs = token_usage.to_attributes()
  assert attrs[_token_usage.GEN_AI_USAGE_INPUT_TOKENS] == 10
  assert attrs[_token_usage.GEN_AI_USAGE_OUTPUT_TOKENS] == 20
