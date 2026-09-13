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

import json
import re
from typing import Any
from typing import AsyncGenerator
from typing import Optional

from google.genai import types

from ..features import FeatureName
from ..features import is_feature_enabled
from ..models.llm_response import LlmResponse

_MAX_ARRAY_INDEX = 10_000

_JSON_PATH_TOKEN_RE = re.compile(
    r"""
    \[\s*(\d+)\s*\]
    | \['((?:[^'\\]|\\.)*)'\]
    | \["((?:[^"\\]|\\.)*)"\]
    | (?:\.?((?:[^.\[\]\\]|\\.)+))
    """,
    re.VERBOSE,
)


def _unescape_json_path_string(s: str) -> str:
  def replace(match: re.Match[str]) -> str:
    if match.group(1):
      return chr(int(match.group(1), 16))
    ch = match.group(2)
    escapes = {
        'n': '\n',
        'r': '\r',
        't': '\t',
        'b': '\b',
        'f': '\f',
    }
    return escapes.get(ch, ch)

  return re.sub(r'\\u([0-9a-fA-F]{4})|\\(.)', replace, s)


def _parse_json_path(json_path: str) -> list[str | int]:
  if json_path.startswith('$.'):
    path = json_path[2:]
  elif json_path.startswith('$'):
    path = json_path[1:]
  else:
    path = json_path

  result: list[str | int] = []
  for match in _JSON_PATH_TOKEN_RE.finditer(path):
    if match.group(1) is not None:
      result.append(int(match.group(1)))
    elif match.group(2) is not None:
      result.append(_unescape_json_path_string(match.group(2)))
    elif match.group(3) is not None:
      result.append(_unescape_json_path_string(match.group(3)))
    elif match.group(4) is not None:
      result.append(_unescape_json_path_string(match.group(4)))
  return result


def _append_json_path_key(path: str, key: str) -> str:
  if re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_]*', key):
    return f'{path}.{key}'
  escaped = (
      key.replace('\\', '\\\\')
      .replace("'", "\\'")
      .replace('\n', '\\n')
      .replace('\r', '\\r')
      .replace('\t', '\\t')
      .replace('\b', '\\b')
      .replace('\f', '\\f')
  )
  return f"{path}['{escaped}']"


def _get_value_by_json_path(
    target: Any, parsed_path: list[str | int]
) -> tuple[Any, bool]:
  current = target
  for part in parsed_path:
    if isinstance(part, str):
      if isinstance(current, dict) and part in current:
        current = current[part]
      else:
        return None, False
    elif isinstance(part, int):
      if isinstance(current, list) and 0 <= part < len(current):
        current = current[part]
      else:
        return None, False
  return current, True


def _set_value_by_json_path(
    target: dict[str, Any] | list[Any],
    parsed_path: list[str | int],
    value: Any,
) -> None:
  if not parsed_path:
    return

  current = target
  for i, part in enumerate(parsed_path[:-1]):
    next_part = parsed_path[i + 1]

    if isinstance(part, str):
      if not isinstance(current, dict):
        return
      if part not in current:
        current[part] = [] if isinstance(next_part, int) else {}
      current = current[part]
    elif isinstance(part, int):
      if not isinstance(current, list) or part < 0 or part > _MAX_ARRAY_INDEX:
        return
      while len(current) <= part:
        current.append(None)
      if current[part] is None:
        current[part] = [] if isinstance(next_part, int) else {}
      current = current[part]

  last_part = parsed_path[-1]
  if isinstance(last_part, str):
    if not isinstance(current, dict):
      return
    current[last_part] = value
  elif isinstance(last_part, int):
    if (
        not isinstance(current, list)
        or last_part < 0
        or last_part > _MAX_ARRAY_INDEX
    ):
      return
    while len(current) <= last_part:
      current.append(None)
    current[last_part] = value


class StreamingResponseAggregator:
  """Aggregates partial streaming responses.

  It aggregates content from partial responses, and generates LlmResponses for
  individual (partial) model responses, as well as for aggregated content.
  """

  def __init__(self) -> None:
    self._text: list[str] = []
    self._thought_text: list[str] = []
    self._usage_metadata: Optional[
        types.GenerateContentResponseUsageMetadata
    ] = None
    self._grounding_metadata: Optional[types.GroundingMetadata] = None
    self._citation_metadata: Optional[types.CitationMetadata] = None
    self._response = None

    # For progressive SSE streaming mode: accumulate parts in order
    self._parts_sequence: list[types.Part] = []
    self._current_text_buffer: list[str] = []
    self._current_text_is_thought: Optional[bool] = None
    self._current_text_thought_signature: Optional[bytes] = None
    self._finish_reason: Optional[types.FinishReason] = None

    # For streaming function call arguments
    self._current_fc_name: Optional[str] = None
    self._current_fc_args: dict[str, Any] = {}
    self._current_fc_arg_chunks: dict[str, list[str]] = {}
    self._current_fc_id: Optional[str] = None
    self._current_thought_signature: Optional[bytes] = None

  def _flush_text_buffer_to_sequence(self) -> None:
    """Flush current text buffer to parts sequence.

    This helper is used in progressive SSE mode to maintain part ordering.
    It only merges consecutive text parts of the same type (thought or regular).

    The merged part is built from scratch, so any thought signature seen on the
    chunks that fed the buffer has to be carried over explicitly. The model
    expects that signature back verbatim on the next request, and dropping it
    makes it redo the reasoning the signature stood for.
    """
    if self._current_text_buffer:
      buffered_text = ''.join(self._current_text_buffer)
      if self._current_text_is_thought:
        merged_part = types.Part(text=buffered_text, thought=True)
      else:
        merged_part = types.Part.from_text(text=buffered_text)
      if self._current_text_thought_signature:
        merged_part.thought_signature = self._current_text_thought_signature
      self._parts_sequence.append(merged_part)
      self._current_text_buffer = []
      self._current_text_is_thought = None
      self._current_text_thought_signature = None

  def _buffer_string_chunk(self, json_path: str, string_chunk: str) -> None:
    """Buffer a streamed string chunk for a JSONPath.

    Chunks are kept in a list and joined once, when the function call is
    flushed, so accumulating a large argument stays linear in its length.

    Args:
      json_path: JSONPath for this argument
      string_chunk: The chunk to append
    """
    chunks = self._current_fc_arg_chunks.get(json_path)
    if chunks is None:
      parsed_path = _parse_json_path(json_path)
      existing_value, found = _get_value_by_json_path(
          self._current_fc_args, parsed_path
      )
      chunks = (
          [existing_value]
          if (found and isinstance(existing_value, str))
          else []
      )
      self._current_fc_arg_chunks[json_path] = chunks
      # Reserve the key so the flushed args keep their arrival order.
      self._set_value_by_json_path(json_path, '')

    chunks.append(string_chunk)

  def _get_value_from_partial_arg(
      self, partial_arg: types.PartialArg
  ) -> tuple[Any, bool]:
    """Extract a non-string value from a partial argument.

    Partial arguments without a populated value field (such as empty containers)
    return has_value=False and are dropped from aggregation.

    Args:
      partial_arg: The partial argument object

    Returns:
      Tuple of (value, has_value) where has_value indicates if a value exists
    """
    value: Any = None
    has_value = False

    if partial_arg.number_value is not None:
      value = partial_arg.number_value
      has_value = True
    elif partial_arg.bool_value is not None:
      value = partial_arg.bool_value
      has_value = True
    elif partial_arg.null_value is not None:
      value = None
      has_value = True

    return value, has_value

  def _set_value_by_json_path(self, json_path: str, value: Any) -> None:
    """Set a value in _current_fc_args using JSONPath notation.

    Args:
      json_path: JSONPath string like "$.location" or "$.location.latitude"
      value: The value to set
    """
    parsed_path = _parse_json_path(json_path)
    _set_value_by_json_path(self._current_fc_args, parsed_path, value)

  def _flush_function_call_to_sequence(self) -> None:
    """Flush current function call to parts sequence.

    This creates a complete FunctionCall part from accumulated partial args.
    """
    if self._current_fc_name:
      # Join the buffered string chunks into their final values
      for json_path, chunks in self._current_fc_arg_chunks.items():
        self._set_value_by_json_path(json_path, ''.join(chunks))

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
      self._current_fc_arg_chunks = {}
      self._current_fc_id = None
      self._current_thought_signature = None

  def _process_streaming_function_call(self, fc: types.FunctionCall) -> None:
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
    for partial_arg in fc.partial_args or []:
      json_path = partial_arg.json_path
      if not json_path:
        continue

      if partial_arg.string_value is not None:
        self._buffer_string_chunk(json_path, partial_arg.string_value)
        continue

      # Extract value from partial arg
      value, has_value = self._get_value_from_partial_arg(partial_arg)

      # Set the value using JSONPath (only if a value was provided)
      if has_value:
        # A scalar replaces anything buffered for this path.
        self._current_fc_arg_chunks.pop(json_path, None)
        self._set_value_by_json_path(json_path, value)

    # Check if function call is complete
    if not fc.will_continue:
      # Function call complete, flush it
      self._flush_text_buffer_to_sequence()
      self._flush_function_call_to_sequence()

  def _process_function_call_part(self, part: types.Part) -> None:
    """Process a function call part (streaming or non-streaming).

    Args:
      part: The part containing a function call
    """
    fc = part.function_call
    if fc is None:
      return

    # Check if this is a streaming FC (has partialArgs or will_continue=True)
    # The first chunk of a streaming function call may have will_continue=True
    # but no partial_args yet, so we need to check both conditions.
    if fc.partial_args or fc.will_continue:
      # Streaming function call arguments

      # Generate ID on first chunk if not provided by LLM
      if not fc.id and not self._current_fc_id:
        # Lazy import to avoid circular dependency
        from ..flows.llm_flows.functions import generate_client_function_call_id

        fc.id = generate_client_function_call_id()

      # Save thought_signature from the part (first chunk should have it)
      if part.thought_signature and not self._current_thought_signature:
        self._current_thought_signature = part.thought_signature
      self._process_streaming_function_call(fc)
    else:
      # Non-streaming function call (standard format with args)
      # Skip empty function calls (used as streaming end markers)
      if fc.name:
        # Generate ID if not provided by LLM
        if not fc.id:
          # Lazy import to avoid circular dependency
          from ..flows.llm_flows.functions import generate_client_function_call_id

          fc.id = generate_client_function_call_id()
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
    # Usage is typically reported on a single chunk; keep the last reported
    # value rather than letting a usage-less trailing chunk erase it.
    if llm_response.usage_metadata:
      self._usage_metadata = llm_response.usage_metadata
    if llm_response.grounding_metadata:
      self._grounding_metadata = llm_response.grounding_metadata
    if llm_response.citation_metadata:
      self._citation_metadata = llm_response.citation_metadata

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
            self._current_text_buffer.append(part.text)
            # Carry the signature over to whatever part this buffer becomes.
            # It can land on any chunk of the run, so keep the first one seen.
            if (
                part.thought_signature
                and not self._current_text_thought_signature
            ):
              self._current_text_thought_signature = part.thought_signature
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
      part_text = part0.text or ''
      if part0.thought:
        self._thought_text.append(part_text)
      else:
        self._text.append(part_text)
      llm_response.partial = True
    elif (self._thought_text or self._text) and (
        not llm_response.content
        or not llm_response.content.parts
        # don't yield the merged text event when receiving audio data
        or not llm_response.content.parts[0].inline_data
    ):
      parts = []
      if self._thought_text:
        parts.append(types.Part(text=''.join(self._thought_text), thought=True))
      if self._text:
        parts.append(types.Part.from_text(text=''.join(self._text)))
      yield LlmResponse(
          content=types.ModelContent(parts=parts),
          usage_metadata=llm_response.usage_metadata,
          grounding_metadata=llm_response.grounding_metadata,
          citation_metadata=llm_response.citation_metadata,
          finish_reason=llm_response.finish_reason,
          model_version=llm_response.model_version,
      )
      self._thought_text = []
      self._text = []
    yield llm_response

  def close(self) -> Optional[LlmResponse]:
    """Generate an aggregated response at the end, if needed.

    This should be called after all the model responses are processed.

    Returns:
      The aggregated LlmResponse.
    """
    if not self._response:
      return None

    candidate = (
        self._response.candidates[0] if self._response.candidates else None
    )

    finish_reason = self._finish_reason
    if not finish_reason and candidate:
      finish_reason = candidate.finish_reason

    error_code = None
    error_message = None
    if finish_reason and finish_reason != types.FinishReason.STOP:
      error_code = finish_reason
      error_message = candidate.finish_message if candidate else None
    elif not candidate and self._response.prompt_feedback:
      error_code = self._response.prompt_feedback.block_reason
      error_message = self._response.prompt_feedback.block_reason_message

    # ========== Progressive SSE Streaming (new feature) ==========
    if is_feature_enabled(FeatureName.PROGRESSIVE_SSE_STREAMING):
      self._flush_text_buffer_to_sequence()
      self._flush_function_call_to_sequence()

      final_parts = self._parts_sequence
      content = types.ModelContent(parts=final_parts) if final_parts else None

      return LlmResponse(
          content=content,
          grounding_metadata=self._grounding_metadata,
          citation_metadata=self._citation_metadata,
          error_code=error_code,
          error_message=error_message,
          usage_metadata=self._usage_metadata,
          finish_reason=finish_reason,
          partial=False,
          model_version=self._response.model_version,
      )

    # ========== Non-Progressive SSE Streaming (old behavior) ==========
    parts = []
    if self._thought_text:
      parts.append(types.Part(text=''.join(self._thought_text), thought=True))
    if self._text:
      parts.append(types.Part.from_text(text=''.join(self._text)))
    content = types.ModelContent(parts=parts) if parts else None

    return LlmResponse(
        content=content,
        grounding_metadata=self._grounding_metadata,
        citation_metadata=self._citation_metadata,
        error_code=error_code,
        error_message=error_message,
        usage_metadata=self._usage_metadata,
        finish_reason=finish_reason,
        partial=False,
        model_version=self._response.model_version,
    )


class _JsonPathTracker:
  """Tracks JSON paths and values from a streaming JSON string."""

  def __init__(self) -> None:
    self.accumulated_parts: list[str] = []
    self.previous_dict: dict[str, Any] = {}
    self.previous_completed = ''
    self.need_reset = False
    self._stack: list[str] = []
    self._in_string = False
    self._escaped = False
    self._seen_open = False
    self._current_path: str | None = None
    self._parsed_current_path: list[str | int] | None = None
    self._open_string_path: str | None = None

  def handle_chunk(self, chunk: str) -> list[types.PartialArg]:
    """Handles a new chunk of JSON and returns the detected PartialArgs."""
    if not chunk:
      return []

    if self.need_reset:
      self.accumulated_parts = []
      self.previous_completed = ''
      self.previous_dict = {}
      self.need_reset = False
      self._stack = []
      self._in_string = False
      self._escaped = False
      self._seen_open = False
      self._current_path = None
      self._parsed_current_path = None
      self._open_string_path = None

    # Fast-path for incremental string streaming: avoids O(N^2) re-parsing
    # and re-joining when appending plain characters to an open string.
    if (
        self._in_string
        and not self._escaped
        and self._current_path is not None
        and self._parsed_current_path is not None
        and '"' not in chunk
        and '\\' not in chunk
    ):
      self.accumulated_parts.append(chunk)
      existing_val, found = _get_value_by_json_path(
          self.previous_dict, self._parsed_current_path
      )
      new_val = (
          existing_val if (found and isinstance(existing_val, str)) else ''
      ) + chunk
      _set_value_by_json_path(
          self.previous_dict, self._parsed_current_path, new_val
      )
      return [
          types.PartialArg(
              json_path=self._current_path,
              string_value=chunk,
              will_continue=True,
          )
      ]

    self.accumulated_parts.append(chunk)
    self._current_path = None
    self._parsed_current_path = None

    for char in chunk:
      if self._in_string:
        if self._escaped:
          self._escaped = False
        elif char == '\\':
          self._escaped = True
        elif char == '"':
          self._in_string = False
      else:
        if char == '"':
          self._in_string = True
        elif char in '{[':
          self._stack.append(char)
          self._seen_open = True
        elif char in '}]':
          if self._stack and (
              (char == '}' and self._stack[-1] == '{')
              or (char == ']' and self._stack[-1] == '[')
          ):
            self._stack.pop()

    if self._seen_open and not self._stack and not self._in_string:
      self.need_reset = True

    completed = self._complete_json()
    if not completed:
      return []

    if completed == self.previous_completed:
      if not self._in_string and self._open_string_path is not None:
        closed_path = self._open_string_path
        self._open_string_path = None
        return [
            types.PartialArg(
                json_path=closed_path,
                string_value='',
                will_continue=False,
            )
        ]
      return []
    self.previous_completed = completed

    try:
      current_dict = json.loads(completed)
    except (json.JSONDecodeError, RecursionError):
      return []

    if not isinstance(current_dict, dict):
      return []

    diffs = self._get_diff(self.previous_dict, current_dict)
    self.previous_dict = current_dict

    if self._open_string_path is not None and not any(
        d.json_path == self._open_string_path for d in diffs
    ):
      diffs.insert(
          0,
          types.PartialArg(
              json_path=self._open_string_path,
              string_value='',
              will_continue=False,
          ),
      )
      self._open_string_path = None

    if self._in_string:
      if diffs:
        for d in reversed(diffs):
          if d.string_value is not None and d.json_path is not None:
            self._current_path = d.json_path
            self._parsed_current_path = _parse_json_path(d.json_path)
            self._open_string_path = d.json_path
            d.will_continue = True
            break
    else:
      self._current_path = None
      self._parsed_current_path = None
      self._open_string_path = None

    return diffs

  def _complete_json(self) -> str:
    if not self._seen_open:
      return ''

    if self._in_string:
      if self._escaped:
        return ''
      suffix = '"' + ''.join(
          '}' if op == '{' else ']' for op in reversed(self._stack)
      )
      return ''.join(self.accumulated_parts) + suffix

    last_non_ws = ''
    last_part_idx = -1
    last_char_idx = -1
    for i in range(len(self.accumulated_parts) - 1, -1, -1):
      part = self.accumulated_parts[i]
      stripped = part.rstrip()
      if stripped:
        last_non_ws = stripped[-1]
        last_part_idx = i
        last_char_idx = len(stripped) - 1
        break

    if not last_non_ws:
      return ''

    # A dangling colon or a trailing numeric literal not yet terminated by
    # whitespace/delimiter must not be prematurely completed.
    if last_non_ws == ':' or last_non_ws.isdigit():
      return ''
    elif (last_non_ws == '{' and len(self._stack) > 1) or last_non_ws == '[':
      return ''
    elif last_non_ws == ',':
      prefix = ''.join(self.accumulated_parts[:last_part_idx]) + (
          self.accumulated_parts[last_part_idx][:last_char_idx]
      )
      suffix = ''.join(
          '}' if op == '{' else ']' for op in reversed(self._stack)
      )
      return prefix + suffix
    else:
      suffix = ''.join(
          '}' if op == '{' else ']' for op in reversed(self._stack)
      )
      return ''.join(self.accumulated_parts) + suffix

  def _get_diff(
      self, prev: Any, curr: Any, path: str = '$'
  ) -> list[types.PartialArg]:
    diffs = []
    if isinstance(curr, dict) and isinstance(prev, dict):
      for k, v in curr.items():
        curr_path = _append_json_path_key(path, k)
        if k not in prev:
          diffs.extend(self._get_diff_new_value(curr_path, v))
        else:
          diffs.extend(self._get_diff(prev[k], v, curr_path))
    elif isinstance(curr, list) and isinstance(prev, list):
      for i, v in enumerate(curr):
        curr_path = f'{path}[{i}]'
        if i >= len(prev):
          diffs.extend(self._get_diff_new_value(curr_path, v))
        else:
          diffs.extend(self._get_diff(prev[i], v, curr_path))
    else:
      if prev != curr:
        if isinstance(curr, str) and isinstance(prev, str):
          if curr.startswith(prev):
            delta = curr[len(prev) :]
            if delta:
              diffs.append(
                  types.PartialArg(
                      json_path=path,
                      string_value=delta,
                      will_continue=False,
                  )
              )
          else:
            diffs.append(
                types.PartialArg(
                    json_path=path,
                    string_value=curr,
                    will_continue=False,
                )
            )
        else:
          diffs.extend(self._get_diff_new_value(path, curr))
    return diffs

  def _get_diff_new_value(self, path: str, val: Any) -> list[types.PartialArg]:
    diffs = []
    if isinstance(val, dict) and val:
      for k, v in val.items():
        diffs.extend(
            self._get_diff_new_value(_append_json_path_key(path, k), v)
        )
    elif isinstance(val, list) and val:
      for i, v in enumerate(val):
        diffs.extend(self._get_diff_new_value(f'{path}[{i}]', v))
    else:
      diffs.append(self._create_partial_arg(path, val))
    return diffs

  def _create_partial_arg(self, path: str, val: Any) -> types.PartialArg:
    """Creates a PartialArg for a leaf value.

    PartialArg only supports scalar values (string, number, bool, null) and
    cannot represent empty containers ({}, []). Empty containers fall through
    to a PartialArg with only json_path set, which downstream aggregators drop
    because no value field is populated and a valueless leaf cannot distinguish
    an empty dict from an empty list.

    Args:
      path: JSONPath for this argument.
      val: The leaf value.

    Returns:
      A PartialArg with the corresponding value field set, or a valueless
      PartialArg for unsupported non-scalar leaf values like empty containers.
    """
    if isinstance(val, str):
      return types.PartialArg(
          json_path=path, string_value=val, will_continue=False
      )
    elif isinstance(val, (int, float)) and not isinstance(val, bool):
      return types.PartialArg(
          json_path=path, number_value=val, will_continue=False
      )
    elif isinstance(val, bool):
      return types.PartialArg(
          json_path=path, bool_value=val, will_continue=False
      )
    elif val is None:
      return types.PartialArg(
          json_path=path, null_value='NULL_VALUE', will_continue=False
      )
    return types.PartialArg(json_path=path, will_continue=False)
