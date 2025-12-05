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

from typing import Any
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

    # For streaming function call arguments
    self._current_fc_name: Optional[str] = None
    self._current_fc_args: dict[str, Any] = {}
    self._current_fc_id: Optional[str] = None
    self._current_thought_signature: Optional[str] = None

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

  def _get_value_from_partial_arg(
      self, partial_arg: types.PartialArg, json_path: str
  ):
    """Extract value from a partial argument.

    Args:
      partial_arg: The partial argument object
      json_path: JSONPath for this argument

    Returns:
      Tuple of (value, has_value) where has_value indicates if a value exists
    """
    value = None
    has_value = False

    if partial_arg.string_value is not None:
      # For streaming strings, append chunks to existing value
      string_chunk = partial_arg.string_value
      has_value = True

      # Get current value for this path (if any)
      path_without_prefix = (
          json_path[2:] if json_path.startswith('$.') else json_path
      )
      path_parts = path_without_prefix.split('.')

      # Try to get existing value
      existing_value = self._current_fc_args
      for part in path_parts:
        if isinstance(existing_value, dict) and part in existing_value:
          existing_value = existing_value[part]
        else:
          existing_value = None
          break

      # Append to existing string or set new value
      if isinstance(existing_value, str):
        value = existing_value + string_chunk
      else:
        value = string_chunk

    elif partial_arg.number_value is not None:
      value = partial_arg.number_value
      has_value = True
    elif partial_arg.bool_value is not None:
      value = partial_arg.bool_value
      has_value = True
    elif partial_arg.null_value is not None:
      value = None
      has_value = True

    return value, has_value

  def _set_value_by_json_path(self, json_path: str, value: Any):
    """Set a value in _current_fc_args using JSONPath notation.

    Args:
      json_path: JSONPath string like "$.location" or "$.location.latitude"
      value: The value to set
    """
    # Remove leading "$." from jsonPath
    if json_path.startswith('$.'):
      path = json_path[2:]
    else:
      path = json_path

    # Split path into components
    path_parts = path.split('.')

    # Navigate to the correct location and set the value
    current = self._current_fc_args
    for part in path_parts[:-1]:
      if part not in current:
        current[part] = {}
      current = current[part]

    # Set the final value
    current[path_parts[-1]] = value

  def _flush_function_call_to_sequence(self):
    """Flush current function call to parts sequence.

    This creates a complete FunctionCall part from accumulated partial args.
    """
    if self._current_fc_name:
      # Create function call part with accumulated args
      fc_part = types.Part.from_function_call(
          name=self._current_fc_name,
          args=self._current_fc_args.copy(),
      )

      # Set the ID if provided (directly on the function_call object)
      if self._current_fc_id and fc_part.function_call:
        fc_part.function_call.id = self._current_fc_id

      # Set thought_signature if provided (on the Part, not FunctionCall)
      if self._current_thought_signature:
        fc_part.thought_signature = self._current_thought_signature

      self._parts_sequence.append(fc_part)

      # Reset FC state
      self._current_fc_name = None
      self._current_fc_args = {}
      self._current_fc_id = None
      self._current_thought_signature = None

  def _process_streaming_function_call(self, fc: types.FunctionCall):
    """Process a streaming function call with partialArgs.

    Args:
      fc: The function call object with partial_args
    """
    # Save function name if present (first chunk)
    if fc.name:
      self._current_fc_name = fc.name
    if fc.id:
      self._current_fc_id = fc.id

    # Process each partial argument
    for partial_arg in getattr(fc, 'partial_args', []):
      json_path = partial_arg.json_path
      if not json_path:
        continue

      # Extract value from partial arg
      value, has_value = self._get_value_from_partial_arg(
          partial_arg, json_path
      )

      # Set the value using JSONPath (only if a value was provided)
      if has_value:
        self._set_value_by_json_path(json_path, value)

    # Check if function call is complete
    fc_will_continue = getattr(fc, 'will_continue', False)
    if not fc_will_continue:
      # Function call complete, flush it
      self._flush_text_buffer_to_sequence()
      self._flush_function_call_to_sequence()

  def _process_function_call_part(self, part: types.Part):
    """Process a function call part (streaming or non-streaming).

    Args:
      part: The part containing a function call
    """
    fc = part.function_call

    # Check if this is a streaming FC (has partialArgs)
    if hasattr(fc, 'partial_args') and fc.partial_args:
      # Streaming function call arguments

      # Save thought_signature from the part (first chunk should have it)
      if part.thought_signature and not self._current_thought_signature:
        self._current_thought_signature = part.thought_signature
      self._process_streaming_function_call(fc)
    else:
      # Non-streaming function call (standard format with args)
      # Skip empty function calls (used as streaming end markers)
      if fc.name:
        # Flush any buffered text first, then add the FC part
        self._flush_text_buffer_to_sequence()
        self._parts_sequence.append(part)

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
          elif part.function_call:
            # Process function call (handles both streaming Args and
            # non-streaming Args)
            self._process_function_call_part(part)
          else:
            # Other non-text parts (bytes, etc.)
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
        # Flush any remaining buffers to complete the sequence
        self._flush_text_buffer_to_sequence()
        self._flush_function_call_to_sequence()

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
