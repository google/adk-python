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

# Centralized OpenTelemetry Semantic Conventions
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_USAGE_INPUT_TOKENS
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_USAGE_OUTPUT_TOKENS

# Use the import symbol once the minimum OpenTelemetry SDK version is updated to 1.40.0
# from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS
GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = 'gen_ai.usage.cache_read.input_tokens'

# Use the import symbol once the minimum OpenTelemetry SDK version is updated to 1.42.0
# from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_USAGE_REASONING_OUTPUT_TOKENS
GEN_AI_USAGE_REASONING_OUTPUT_TOKENS = 'gen_ai.usage.reasoning.output_tokens'


@dataclasses.dataclass
class TokenUsage:
  """Centralized representation and processing of GenAI token usage metadata."""

  usage_metadata: types.GenerateContentResponseUsageMetadata | None

  @property
  def input_token_count(self) -> int | None:
    if self.usage_metadata is None:
      return None
    # OTel semconv for `gen_ai.client.token.usage` states that token counts should
    # be categorized under `gen_ai.token.type` as either "input" or "output".
    # We aggregate prompt and tool use tokens for "input".
    prompt_tokens = self.usage_metadata.prompt_token_count
    tool_tokens = self.usage_metadata.tool_use_prompt_token_count
    if prompt_tokens is None and tool_tokens is None:
      return None
    return (prompt_tokens or 0) + (tool_tokens or 0)

  @property
  def output_token_count(self) -> int | None:
    if self.usage_metadata is None:
      return None
    # According to OpenTelemetry Semantic Conventions:
    # https://github.com/open-telemetry/semantic-conventions/blob/v1.41.0/docs/registry/attributes/gen-ai.md
    # gen_ai.usage.reasoning.output_tokens (thoughts_token_count) SHOULD be included in gen_ai.usage.output_tokens.
    candidates_tokens = self.usage_metadata.candidates_token_count
    thoughts_tokens = self.usage_metadata.thoughts_token_count
    if candidates_tokens is None and thoughts_tokens is None:
      return None
    return (candidates_tokens or 0) + (thoughts_tokens or 0)

  def to_attributes(self) -> dict[str, AttributeValue]:
    """Returns a dictionary of OpenTelemetry token usage attributes."""
    attrs: dict[str, AttributeValue] = {}
    if self.input_token_count is not None:
      attrs[GEN_AI_USAGE_INPUT_TOKENS] = self.input_token_count
    if self.output_token_count is not None:
      attrs[GEN_AI_USAGE_OUTPUT_TOKENS] = self.output_token_count

    if self.usage_metadata is not None:
      cached_tokens = self.usage_metadata.cached_content_token_count
      if cached_tokens is not None:
        attrs[GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] = cached_tokens

      thoughts_tokens = self.usage_metadata.thoughts_token_count
      if thoughts_tokens is not None:
        attrs[GEN_AI_USAGE_REASONING_OUTPUT_TOKENS] = thoughts_tokens

      system_instruction_tokens = getattr(
          self.usage_metadata, 'system_instruction_tokens', None
      )
      if system_instruction_tokens is not None:
        attrs['gen_ai.usage.experimental.system_instruction_tokens'] = (
            system_instruction_tokens
        )

    return attrs
