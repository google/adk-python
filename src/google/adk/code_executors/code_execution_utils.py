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

"""Utility functions for code execution."""

from __future__ import annotations

import base64
import binascii
import copy
import dataclasses
import re
from typing import List
from typing import Optional

from google.genai import types


@dataclasses.dataclass(frozen=True)
class File:
  """A structure that contains a file name and its content."""

  name: str
  """
  The name of the file with file extension (e.g., "file.csv").
  """

  content: str | bytes
  """
  The base64-encoded bytes of the file content or the original bytes of the file content.
  """

  mime_type: str = 'text/plain'
  """
  The mime type of the file (e.g., "image/png").
  """


@dataclasses.dataclass
class CodeExecutionInput:
  """A structure that contains the input of code execution."""

  code: str
  """
  The code to execute.
  """

  input_files: list[File] = dataclasses.field(default_factory=list)
  """
  The input files available to the code.
  """

  execution_id: Optional[str] = None
  """
  The execution ID for the stateful code execution.
  """


@dataclasses.dataclass
class CodeExecutionResult:
  """A structure that contains the result of code execution."""

  stdout: str = ''
  """
  The standard output of the code execution.
  """

  stderr: str = ''
  """
  The standard error of the code execution.
  """

  output_files: list[File] = dataclasses.field(default_factory=list)
  """
  The output files from the code execution.
  """


class CodeExecutionUtils:
  """Utility functions for code execution."""

  @staticmethod
  def get_encoded_file_content(data: bytes) -> bytes:
    """Gets the file content as a base64-encoded bytes.

    Args:
      data: The file content bytes.

    Returns:
      The file content as a base64-encoded bytes.
    """

    def _is_base64_encoded(data: bytes) -> bool:
      try:
        return base64.b64encode(base64.b64decode(data)) == data
      except binascii.Error:
        return False

    return data if _is_base64_encoded(data) else base64.b64encode(data)

  @staticmethod
  def extract_code_and_truncate_content(
      content: types.Content,
      code_block_delimiters: List[tuple[str, str]],
  ) -> Optional[str]:
    """Extracts the first code block from the content and truncate everything after it.

    Args:
      content: The mutable content to extract the code from.
      code_block_delimiters: The list of the enclosing delimiters to identify
        the code blocks.

    Returns:
      The first code block if found; otherwise, None.
    """
    if not content or not content.parts:
      return

    # Extract the code from the executable code parts if there are no associated
    # code execution result parts.
    for idx, part in enumerate(content.parts):
      if part.executable_code and (
          idx == len(content.parts) - 1
          or not content.parts[idx + 1].code_execution_result
      ):
        content.parts = content.parts[: idx + 1]
        return part.executable_code.code

    # Extract the code from the text parts.
    text_parts = [p for p in content.parts if p.text]
    if not text_parts:
      return

    first_text_part = copy.deepcopy(text_parts[0])
    response_text = '\n'.join([p.text for p in text_parts])

    # Find the first code block.
    leading_delimiter_pattern = '|'.join(d[0] for d in code_block_delimiters)
    trailing_delimiter_pattern = '|'.join(d[1] for d in code_block_delimiters)
    pattern = re.compile(
        (
            rf'(?P<prefix>.*?)({leading_delimiter_pattern})(?P<code>.*?)({trailing_delimiter_pattern})(?P<suffix>.*?)$'
        ).encode(),
        re.DOTALL,
    )
    pattern_match = pattern.search(response_text.encode())
    if pattern_match is None:
      return

    code_str = pattern_match.group('code').decode()
    if not code_str:
      return

    content.parts = []
    if pattern_match.group('prefix'):
      first_text_part.text = pattern_match.group('prefix').decode()
      content.parts.append(first_text_part)
    content.parts.append(
        CodeExecutionUtils.build_executable_code_part(code_str)
    )
    return pattern_match.group('code').decode()

  @staticmethod
  def build_executable_code_part(code: str) -> types.Part:
    """Builds an executable code part with code string.

    Args:
      code: The code string.

    Returns:
      The constructed executable code part.
    """
    return types.Part.from_executable_code(
        code=code,
        language='PYTHON',
    )

  @staticmethod
  def build_code_execution_result_part(
      code_execution_result: CodeExecutionResult,
  ) -> types.Part:
    """Builds the code execution result part from the code execution result.

    Args:
      code_execution_result: The code execution result.

    Returns:
      The constructed code execution result part.
    """
    if code_execution_result.stderr:
      return types.Part.from_code_execution_result(
          outcome='OUTCOME_FAILED',
          output=code_execution_result.stderr,
      )
    final_result = []
    if code_execution_result.stdout or not code_execution_result.output_files:
      final_result.append(
          'Code execution result:\n' + '%s\n' % code_execution_result.stdout
      )
    if code_execution_result.output_files:
      final_result.append(
          'Saved artifacts:\n'
          + ','.join(
              ['`%s`' % f.name for f in code_execution_result.output_files]
          )
      )
    return types.Part.from_code_execution_result(
        outcome='OUTCOME_OK',
        output='\n\n'.join(final_result),
    )

  @staticmethod
  def convert_code_execution_parts(
      content: types.Content,
      code_block_delimiter: tuple[str, str],
      execution_result_delimiters: tuple[str, str],
  ):
    """Converts the code execution parts to text parts in a Content.

    Args:
      content: The mutable content to convert the code execution parts to text
        parts.
      code_block_delimiter: The delimiter to format the code block.
      execution_result_delimiters: The delimiter to format the code execution
        result.
    """
    if not content.parts:
      return

    # Handle the conversion of trailing executable code parts.
    if content.parts[-1].executable_code:
      content.parts[-1] = types.Part(
          text=(
              code_block_delimiter[0]
              + content.parts[-1].executable_code.code
              + code_block_delimiter[1]
          )
      )
    # Handle the conversion of trailing code execution result parts.
    # Skip if the Content has multiple parts, which means the Content is
    # likely generated by the model.
    elif len(content.parts) == 1 and content.parts[-1].code_execution_result:
      content.parts[-1] = types.Part(
          text=execution_result_delimiters[0]
          + content.parts[-1].code_execution_result.output
          + execution_result_delimiters[1]
      )
      content.role = 'user'
