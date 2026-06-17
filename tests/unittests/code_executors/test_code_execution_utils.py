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

import signal

from google.adk.code_executors import code_execution_utils
from google.genai import types


def test_extract_code_and_truncate_content_basic():
  """Tests basic code extraction and content truncation."""
  content = types.Content(
      role="model",
      parts=[
          types.Part(
              text=(
                  "Here is some code:\n```python\nx = 1\n```\nAnd some text"
                  " after."
              )
          )
      ],
  )
  delimiters = [("```python\n", "\n```")]
  code = (
      code_execution_utils.CodeExecutionUtils.extract_code_and_truncate_content(
          content, delimiters
      )
  )
  assert code == "x = 1"
  assert len(content.parts) == 2
  assert content.parts[0].text == "Here is some code:\n"
  assert content.parts[1].executable_code.code == "x = 1"


def test_extract_code_and_truncate_content_multiple_blocks():
  """Tests that the first code block is extracted when multiple exist."""
  content = types.Content(
      role="model",
      parts=[
          types.Part(
              text=(
                  "First:\n"
                  "```python\n"
                  "x = 1\n"
                  "```\n"
                  "Second:\n"
                  "```python\n"
                  "y = 2\n"
                  "```"
              )
          )
      ],
  )
  delimiters = [("```python\n", "\n```")]
  code = (
      code_execution_utils.CodeExecutionUtils.extract_code_and_truncate_content(
          content, delimiters
      )
  )
  assert code == "x = 1"
  assert len(content.parts) == 2
  assert content.parts[0].text == "First:\n"
  assert content.parts[1].executable_code.code == "x = 1"


def test_extract_code_and_truncate_content_no_delimiter():
  """Tests when no delimiters are found in the content."""
  content = types.Content(
      role="model",
      parts=[types.Part(text="Just plain text without code.")],
  )
  delimiters = [("```python\n", "\n```")]
  code = (
      code_execution_utils.CodeExecutionUtils.extract_code_and_truncate_content(
          content, delimiters
      )
  )
  assert code is None
  # Content should be unmodified.
  assert len(content.parts) == 1
  assert content.parts[0].text == "Just plain text without code."


def test_extract_code_and_truncate_content_redos_vulnerability():
  """Tests that a string that would cause ReDoS behaves reasonably."""
  # Construct a long string that contains repeating patterns without matching delimiters.
  # The old regex pattern would backtrack exponentially.
  ticks = "`" * 3
  long_invalid_payload = ticks + "python\n" + "x = 1\n" * 5000 + "not_matching"
  content = types.Content(
      role="model",
      parts=[types.Part(text=long_invalid_payload)],
  )
  delimiters = [(ticks + "python\n", "\n" + ticks)]

  def handler(_signum, _frame):
    raise TimeoutError("Test timed out (possible ReDoS regression)")

  signal.signal(signal.SIGALRM, handler)
  signal.alarm(2)
  try:
    # If ReDoS vulnerability exists, this call will hang or take a very long time.
    code = code_execution_utils.CodeExecutionUtils.extract_code_and_truncate_content(
        content, delimiters
    )
  finally:
    signal.alarm(0)
  assert code is None


def test_extract_code_and_truncate_content_multiple_delimiter_pairs():
  """Tests code extraction when multiple different delimiter pairs are provided."""
  ticks = "`" * 3
  # Case 1: First delimiter pair matches first
  content = types.Content(
      role="model",
      parts=[
          types.Part(
              text="Here is tool code:\n"
              + ticks
              + "tool_code\nx = 1\n"
              + ticks
              + "\nAnd python code:\n"
              + ticks
              + "python\ny = 2\n"
              + ticks
          )
      ],
  )
  delimiters = [
      (ticks + "tool_code\n", "\n" + ticks),
      (ticks + "python\n", "\n" + ticks),
  ]
  code = (
      code_execution_utils.CodeExecutionUtils.extract_code_and_truncate_content(
          content, delimiters
      )
  )
  assert code == "x = 1"
  assert len(content.parts) == 2
  assert content.parts[0].text == "Here is tool code:\n"
  assert content.parts[1].executable_code.code == "x = 1"

  # Case 2: Second delimiter pair matches first
  content = types.Content(
      role="model",
      parts=[
          types.Part(
              text="Here is python code:\n"
              + ticks
              + "python\ny = 2\n"
              + ticks
              + "\nAnd tool code:\n"
              + ticks
              + "tool_code\nx = 1\n"
              + ticks
          )
      ],
  )
  code = (
      code_execution_utils.CodeExecutionUtils.extract_code_and_truncate_content(
          content, delimiters
      )
  )
  assert code == "y = 2"
  assert len(content.parts) == 2
  assert content.parts[0].text == "Here is python code:\n"
  assert content.parts[1].executable_code.code == "y = 2"
