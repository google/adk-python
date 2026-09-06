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

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from google.genai import types
  from opentelemetry.util.types import AttributeValue

  from ..models.llm_response import LlmResponse

# Centralized OpenTelemetry Semantic Conventions
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_USAGE_INPUT_TOKENS
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_USAGE_OUTPUT_TOKENS

# Use the import symbol once the minimum OpenTelemetry SDK version is updated to 1.40.0
# from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS
GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = 'gen_ai.usage.cache_read.input_tokens'

# Use the import symbol once the minimum OpenTelemetry SDK version is updated to 1.41.0
# from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS
GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS = (
    'gen_ai.usage.cache_creation.input_tokens'
)

# Use the import symbol once the minimum OpenTelemetry SDK version is updated to 1.42.0
# from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_USAGE_REASONING_OUTPUT_TOKENS
GEN_AI_USAGE_REASONING_OUTPUT_TOKENS = 'gen_ai.usage.reasoning.output_tokens'

# What each token count means. This module owns the definition: the properties
# below implement it and `_metrics` interpolates it into the published metric
# descriptions, so the two cannot drift apart.
INPUT_TOKENS_MEANING = (
    'Input (prompt) tokens, summing the prompt itself and the results of'
    ' server-side tool calls. Prompt tokens served from a cache are part of the'
    ' prompt count, so they are included here too.'
)
OUTPUT_TOKENS_MEANING = (
    'Output (completion) tokens, summing candidate and reasoning tokens. The'
    ' tokens a model spends emitting a tool call are candidate tokens, so they'
    ' count here.'
)
TOTAL_TOKENS_MEANING = (
    'Input plus output tokens, derived from those two rather than taken from'
    ' the model-reported total.'
)
CACHE_READ_INPUT_TOKENS_MEANING = (
    'Input tokens served from a provider-managed cache.'
)
REASONING_OUTPUT_TOKENS_MEANING = (
    'Output tokens spent on reasoning (chain-of-thought / extended thinking).'
)
TOOL_INPUT_TOKENS_MEANING = (
    'Input tokens from server-side tool results the model fed back to itself'
    ' within one request, such as code execution or search grounding. Zero for'
    ' client-side function tools, whose responses arrive as ordinary content on'
    ' the next request and bill as plain prompt tokens.'
)


def _merge(current: int | None, incoming: int | None) -> int | None:
  """Sums two counts, staying None for a bucket neither side reported."""
  if current is None:
    return incoming
  if incoming is None:
    return current
  return current + incoming


@dataclasses.dataclass
class TokenUsage:
  """GenAI token usage, either one model call's or a sum over several.

  A bucket is None when nothing reported it, which is not the same as a
  reported zero: `to_attributes` omits the former and emits the latter. `add`
  preserves that, so a sum reports only the buckets something reported.

  `cache_read_input_tokens` and `tool_input_tokens` are subsets of
  `input_tokens`; `reasoning_output_tokens` is a subset of `output_tokens`.
  `cache_creation_input_tokens` and `system_instruction_tokens` reach spans
  only; no metric reads them.
  """

  input_tokens: int | None = None
  output_tokens: int | None = None
  cache_read_input_tokens: int | None = None
  reasoning_output_tokens: int | None = None
  tool_input_tokens: int | None = None
  cache_creation_input_tokens: int | None = None
  system_instruction_tokens: int | None = None

  @classmethod
  def from_usage_metadata(
      cls, usage_metadata: types.GenerateContentResponseUsageMetadata | None
  ) -> TokenUsage:
    """Reads one model call's buckets out of its reported usage metadata."""
    if usage_metadata is None:
      return cls()

    # OTel semconv for `gen_ai.client.token.usage` states that token counts
    # should be categorized under `gen_ai.token.type` as either "input" or
    # "output". We aggregate prompt and tool use tokens for "input", and
    # candidates and thoughts tokens for "output":
    # https://github.com/open-telemetry/semantic-conventions/blob/v1.41.0/docs/registry/attributes/gen-ai.md
    prompt_tokens = usage_metadata.prompt_token_count
    tool_tokens = usage_metadata.tool_use_prompt_token_count
    candidates_tokens = usage_metadata.candidates_token_count
    thoughts_tokens = usage_metadata.thoughts_token_count
    return cls(
        input_tokens=(
            None
            if prompt_tokens is None and tool_tokens is None
            else (prompt_tokens or 0) + (tool_tokens or 0)
        ),
        output_tokens=(
            None
            if candidates_tokens is None and thoughts_tokens is None
            else (candidates_tokens or 0) + (thoughts_tokens or 0)
        ),
        cache_read_input_tokens=usage_metadata.cached_content_token_count,
        reasoning_output_tokens=thoughts_tokens,
        tool_input_tokens=tool_tokens,
        # Absent from Gemini's type; the Anthropic paths set them by hand.
        cache_creation_input_tokens=getattr(
            usage_metadata, 'cache_creation_input_tokens', None
        ),
        system_instruction_tokens=getattr(
            usage_metadata, 'system_instruction_tokens', None
        ),
    )

  @classmethod
  def from_llm_responses(
      cls, responses: list[LlmResponse]
  ) -> TokenUsage | None:
    """Returns what one model call spent, or None if it reported nothing.

    Each report is cumulative for the call so far, which makes the newest one
    the whole figure and summing them a double count. Taking the newest report
    rather than the last response keeps a cut-short stream's usage, which a
    trailing chunk that carries none would otherwise drop.

    Args:
      responses: The responses produced by a single model call, in order.
    """
    reported = [
        response.usage_metadata
        for response in responses
        if response.usage_metadata
    ]
    if not reported:
      return None
    usage = cls.from_usage_metadata(reported[-1])
    if usage.input_tokens is None and usage.output_tokens is None:
      return None
    return usage

  @property
  def total_tokens(self) -> int:
    """Derived, so it always agrees with the two directions it sums.

    The model's own reported total is deliberately not used.
    """
    return (self.input_tokens or 0) + (self.output_tokens or 0)

  def add(self, usage: TokenUsage) -> None:
    """Folds another call's usage into these totals, bucket by bucket."""
    self.input_tokens = _merge(self.input_tokens, usage.input_tokens)
    self.output_tokens = _merge(self.output_tokens, usage.output_tokens)
    self.cache_read_input_tokens = _merge(
        self.cache_read_input_tokens, usage.cache_read_input_tokens
    )
    self.reasoning_output_tokens = _merge(
        self.reasoning_output_tokens, usage.reasoning_output_tokens
    )
    self.tool_input_tokens = _merge(
        self.tool_input_tokens, usage.tool_input_tokens
    )
    self.cache_creation_input_tokens = _merge(
        self.cache_creation_input_tokens, usage.cache_creation_input_tokens
    )
    self.system_instruction_tokens = _merge(
        self.system_instruction_tokens, usage.system_instruction_tokens
    )

  def to_attributes(self) -> dict[str, AttributeValue]:
    """Returns a dictionary of OpenTelemetry token usage attributes."""
    attrs: dict[str, AttributeValue] = {}
    if self.input_tokens is not None:
      attrs[GEN_AI_USAGE_INPUT_TOKENS] = self.input_tokens
    if self.output_tokens is not None:
      attrs[GEN_AI_USAGE_OUTPUT_TOKENS] = self.output_tokens
    if self.cache_read_input_tokens is not None:
      attrs[GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] = self.cache_read_input_tokens
    if self.cache_creation_input_tokens is not None:
      attrs[GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS] = (
          self.cache_creation_input_tokens
      )
    if self.reasoning_output_tokens is not None:
      attrs[GEN_AI_USAGE_REASONING_OUTPUT_TOKENS] = self.reasoning_output_tokens
    if self.system_instruction_tokens is not None:
      attrs['gen_ai.usage.experimental.system_instruction_tokens'] = (
          self.system_instruction_tokens
      )
    return attrs
