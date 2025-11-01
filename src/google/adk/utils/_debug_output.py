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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from ..events.event import Event

# Constants for debug output truncation
_ARGS_MAX_LEN = 50  # Keep arg previews short for readability
_RESPONSE_MAX_LEN = 100  # Show more of response for context
_CODE_OUTPUT_MAX_LEN = 100  # Code execution output preview length


def _truncate(text: str, max_len: int) -> str:
  """Truncate text to max length, appending '...' if truncated.

  Args:
      text: The text to truncate.
      max_len: Maximum length before truncation.

  Returns:
      The truncated text with '...' appended if it exceeds max_len.
  """
  return text[:max_len] + '...' if len(text) > max_len else text


def print_event(event: Event, *, verbose: bool = False) -> None:
  """Print an event to stdout in a user-friendly format.

  Args:
      event: The event to print.
      verbose: If True, shows detailed tool calls and responses. If False,
          shows only text responses for cleaner output.
  """
  if not event.content or not event.content.parts:
    return

  # Collect consecutive text parts to avoid repeating author prefix
  text_buffer: list[str] = []

  def flush_text() -> None:
    """Flush accumulated text parts as a single output."""
    if text_buffer:
      combined_text = ''.join(text_buffer)
      print(f'{event.author} > {combined_text}')
      text_buffer.clear()

  for part in event.content.parts:
    # Text parts are always shown regardless of verbose setting
    # because they contain the actual agent responses users expect
    if part.text:
      text_buffer.append(part.text)
    else:
      # Flush any accumulated text before handling non-text parts
      flush_text()

      # Non-text parts (tool calls, code, etc.) are hidden by default
      # to reduce clutter and show only what matters: the final results
      if verbose:
        # Tool invocations show the behind-the-scenes processing
        if part.function_call:
          print(
              f'{event.author} > [Calling tool:'
              f' {part.function_call.name}('
              f'{_truncate(str(part.function_call.args), _ARGS_MAX_LEN)})]'
          )
        # Handle function response parts (tool results)
        elif part.function_response:
          print(
              f'{event.author} > [Tool result:'
              f' {_truncate(str(part.function_response.response), _RESPONSE_MAX_LEN)}]'
          )
        # Handle executable code parts
        elif part.executable_code:
          lang = part.executable_code.language or 'code'
          print(f'{event.author} > [Executing {lang} code...]')
        # Handle code execution result parts
        elif part.code_execution_result:
          output = part.code_execution_result.output or 'result'
          print(
              f'{event.author} > [Code output:'
              f' {_truncate(str(output), _CODE_OUTPUT_MAX_LEN)}]'
          )
        # Handle inline data (images, files)
        elif part.inline_data:
          mime_type = part.inline_data.mime_type or 'data'
          print(f'{event.author} > [Inline data: {mime_type}]')
        # Handle file data
        elif part.file_data:
          uri = part.file_data.file_uri or 'file'
          print(f'{event.author} > [File: {uri}]')

  # Flush any remaining text at the end
  flush_text()
