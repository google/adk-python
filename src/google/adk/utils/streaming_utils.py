# Copyright 2025 Google LLC
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

from typing import AsyncGenerator
from typing import Optional

from google.genai import types

from ..features import FeatureName
from ..features import is_feature_enabled
from ..models.llm_response import LlmResponse


class StreamingResponseAggregator:
  """Aggregates partial streaming responses.

  It aggregates content from partial responses, and generates LlmResponses for
  individual (partial) model responses, as well as for aggregated content.
  """

  def __init__(self):
    self._text = ''
    self._thought_text = ''
    self._usage_metadata = None
    self._response = None

    # For progressive SSE streaming mode: accumulate parts in order
    self._parts_sequence: list[types.Part] = []
    self._current_text_buffer: str = ''
    self._current_text_is_thought: Optional[bool] = None
    self._finish_reason: Optional[types.FinishReason] = None

  def _flush_text_buffer_to_sequence(self):
    """Flush current text buffer to parts sequence.

    This helper is used in progressive SSE mode to maintain part ordering.
    It only merges consecutive text parts of the same type (thought or regular).
    """
    if self._current_text_buffer:
      if self._current_text_is_thought:
        self._parts_sequence.append(
            types.Part(text=self._current_text_buffer, thought=True)
        )
      else:
        self._parts_sequence.append(
            types.Part.from_text(text=self._current_text_buffer)
        )
      self._current_text_buffer = ''
      self._current_text_is_thought = None

  async def process_response(
      self, response: types.GenerateContentResponse
  ) -> AsyncGenerator[LlmResponse, None]:
    """Processes a single model response.

    Args:
      response: The response to process.

    Yields:
      The generated LlmResponse(s), for the partial response, and the aggregated
      response if needed.
    """
    # results = []
    self._response = response
    llm_response = LlmResponse.create(response)
    self._usage_metadata = llm_response.usage_metadata

    # ========== Progressive SSE Streaming (new feature) ==========
    # Save finish_reason for final aggregation
    if llm_response.finish_reason:
      self._finish_reason = llm_response.finish_reason

    if is_feature_enabled(FeatureName.PROGRESSIVE_SSE_STREAMING):
      # Accumulate parts while preserving their order
      # Only merge consecutive text parts of the same type (thought or regular)
      if llm_response.content and llm_response.content.parts:
        for part in llm_response.content.parts:
          if part.text:
            # Check if we need to flush the current buffer first
            # (when text type changes from thought to regular or vice versa)
            if (
                self._current_text_buffer
                and part.thought != self._current_text_is_thought
            ):
              self._flush_text_buffer_to_sequence()

            # Accumulate text to buffer
            if not self._current_text_buffer:
              self._current_text_is_thought = part.thought
            self._current_text_buffer += part.text
          else:
            # Non-text part (function_call, bytes, etc.)
            # Flush any buffered text first, then add the non-text part
            self._flush_text_buffer_to_sequence()
            self._parts_sequence.append(part)

      # Mark ALL intermediate chunks as partial
      llm_response.partial = True
      yield llm_response
      return

    # ========== Non-Progressive SSE Streaming (old behavior) ==========
    if (
        llm_response.content
        and llm_response.content.parts
        and llm_response.content.parts[0].text
    ):
      part0 = llm_response.content.parts[0]
      if part0.thought:
        self._thought_text += part0.text
      else:
        self._text += part0.text
      llm_response.partial = True
    elif (self._thought_text or self._text) and (
        not llm_response.content
        or not llm_response.content.parts
        # don't yield the merged text event when receiving audio data
        or not llm_response.content.parts[0].inline_data
    ):
      parts = []
      if self._thought_text:
        parts.append(types.Part(text=self._thought_text, thought=True))
      if self._text:
        parts.append(types.Part.from_text(text=self._text))
      yield LlmResponse(
          content=types.ModelContent(parts=parts),
          usage_metadata=llm_response.usage_metadata,
      )
      self._thought_text = ''
      self._text = ''
    yield llm_response

  def close(self) -> Optional[LlmResponse]:
    """Generate an aggregated response at the end, if needed.

    This should be called after all the model responses are processed.

    Returns:
      The aggregated LlmResponse.
    """
    # ========== Progressive SSE Streaming (new feature) ==========
    if is_feature_enabled(FeatureName.PROGRESSIVE_SSE_STREAMING):
      # Always generate final aggregated response in progressive mode
      if self._response and self._response.candidates:
        # Flush any remaining text buffer to complete the sequence
        self._flush_text_buffer_to_sequence()

        # Use the parts sequence which preserves original ordering
        final_parts = self._parts_sequence

        if final_parts:
          candidate = self._response.candidates[0]
          finish_reason = self._finish_reason or candidate.finish_reason

          return LlmResponse(
              content=types.ModelContent(parts=final_parts),
              error_code=None
              if finish_reason == types.FinishReason.STOP
              else finish_reason,
              error_message=None
              if finish_reason == types.FinishReason.STOP
              else candidate.finish_message,
              usage_metadata=self._usage_metadata,
              finish_reason=finish_reason,
              partial=False,
          )

        return None

    # ========== Non-Progressive SSE Streaming (old behavior) ==========
    if (
        (self._text or self._thought_text)
        and self._response
        and self._response.candidates
    ):
      parts = []
      if self._thought_text:
        parts.append(types.Part(text=self._thought_text, thought=True))
      if self._text:
        parts.append(types.Part.from_text(text=self._text))
      candidate = self._response.candidates[0]
      return LlmResponse(
          content=types.ModelContent(parts=parts),
          error_code=None
          if candidate.finish_reason == types.FinishReason.STOP
          else candidate.finish_reason,
          error_message=None
          if candidate.finish_reason == types.FinishReason.STOP
          else candidate.finish_message,
          usage_metadata=self._usage_metadata,
      )
