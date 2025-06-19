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

"""Unit tests for CodeExecutionUtils."""

import base64

import pytest

from google.adk.code_executors.code_execution_utils import CodeExecutionInput
from google.adk.code_executors.code_execution_utils import CodeExecutionResult
from google.adk.code_executors.code_execution_utils import CodeExecutionUtils
from google.adk.code_executors.code_execution_utils import File
from google.genai import types


def test_file_dataclass():
  f = File(name="test.txt", content="YQ==", mime_type="text/plain")
  assert f.name == "test.txt"
  assert f.content == "YQ=="
  assert f.mime_type == "text/plain"


def test_code_execution_input_dataclass():
  cei = CodeExecutionInput(
      code="print('hello')",
      input_files=[File(name="f.txt", content="Yg==")],
      execution_id="exec1",
  )
  assert cei.code == "print('hello')"
  assert cei.input_files[0].name == "f.txt"
  assert cei.execution_id == "exec1"


def test_code_execution_result_dataclass():
  cer = CodeExecutionResult(
      stdout="hello",
      stderr="",
      output_files=[File(name="out.txt", content="Yw==")],
  )
  assert cer.stdout == "hello"
  assert cer.stderr == ""
  assert cer.output_files[0].name == "out.txt"


class TestCodeExecutionUtils:

  def test_get_encoded_file_content_already_encoded(self):
    encoded_data = base64.b64encode(b"test data")
    assert CodeExecutionUtils.get_encoded_file_content(encoded_data) == encoded_data

  def test_get_encoded_file_content_not_encoded(self):
    data = b"test data"
    expected_encoded_data = base64.b64encode(data)
    assert CodeExecutionUtils.get_encoded_file_content(data) == expected_encoded_data

  def test_extract_code_and_truncate_content_no_content(self):
    content = types.Content()
    assert CodeExecutionUtils.extract_code_and_truncate_content(content, []) is None

  def test_extract_code_and_truncate_content_no_parts(self):
    content = types.Content(parts=[])
    assert CodeExecutionUtils.extract_code_and_truncate_content(content, []) is None

  def test_extract_code_and_truncate_content_no_code_block(self):
    content = types.Content(parts=[types.Part(text="This is just text.")])
    delimiters = [("```python\n", "\n```")]
    original_content_copy = content.model_copy(deep=True)
    code = CodeExecutionUtils.extract_code_and_truncate_content(content, delimiters)
    assert code is None
    assert content == original_content_copy  # Content should not be modified

  def test_extract_code_and_truncate_content_executable_code_part(self):
    code_str = "print('hello from executable')"
    content = types.Content(
        parts=[
            types.Part(text="Some text before"),
            types.Part(executable_code=types.ExecutableCode(code=code_str, language="PYTHON")),
            types.Part(text="Some text after"),
        ]
    )
    original_parts_len = len(content.parts)
    code = CodeExecutionUtils.extract_code_and_truncate_content(content, [])
    assert code == code_str
    assert len(content.parts) == 2 # Truncated to include executable_code and before
    assert content.parts[1].executable_code.code == code_str

  @pytest.mark.filterwarnings("ignore:OK is not a valid outcome")
  def test_extract_code_and_truncate_content_executable_code_with_result(self):
    code_str = "print('this should not be extracted')"
    content = types.Content(
        parts=[
            types.Part(executable_code=types.ExecutableCode(code=code_str, language="PYTHON")),
            types.Part(code_execution_result=types.CodeExecutionResult(outcome="OK", output="out"))
        ]
    )
    original_content_copy = content.model_copy(deep=True)
    code = CodeExecutionUtils.extract_code_and_truncate_content(content, [])
    assert code is None # Should not extract if there's a result part after it
    assert content == original_content_copy


  def test_extract_code_and_truncate_content_text_code_block(self):
    code_str = "print('hello from text')"
    delimiters = [("```python\n", "\n```")]
    content = types.Content(
        parts=[
            types.Part(text=f"Prefix\n{delimiters[0][0]}{code_str}{delimiters[0][1]}\nSuffix")
        ]
    )
    code = CodeExecutionUtils.extract_code_and_truncate_content(content, delimiters)
    assert code == code_str
    assert len(content.parts) == 2 # Prefix text part + executable code part
    assert content.parts[0].text == "Prefix\n"
    assert content.parts[1].executable_code.code == code_str

  def test_extract_code_and_truncate_content_multiple_delimiters(self):
    code_str = "print('tool code')"
    delimiters = [("```tool_code\n", "\n```"), ("```python\n", "\n```")]
    content = types.Content(
        parts=[
            types.Part(text=f"{delimiters[0][0]}{code_str}{delimiters[0][1]}")
        ]
    )
    code = CodeExecutionUtils.extract_code_and_truncate_content(content, delimiters)
    assert code == code_str
    assert content.parts[0].executable_code.code == code_str

  def test_build_executable_code_part(self):
    code_str = "import os"
    part = CodeExecutionUtils.build_executable_code_part(code_str)
    assert part.executable_code.code == code_str
    assert part.executable_code.language == "PYTHON"

  def test_build_code_execution_result_part_success(self):
    result = CodeExecutionResult(stdout="output", output_files=[File(name="f.txt", content="YQ==")])
    part = CodeExecutionUtils.build_code_execution_result_part(result)
    assert part.code_execution_result.outcome == "OUTCOME_OK"
    assert "Code execution result:\noutput" in part.code_execution_result.output
    assert "Saved artifacts:\n`f.txt`" in part.code_execution_result.output

  def test_build_code_execution_result_part_failure(self):
    result = CodeExecutionResult(stderr="error occurred")
    part = CodeExecutionUtils.build_code_execution_result_part(result)
    assert part.code_execution_result.outcome == "OUTCOME_FAILED"
    assert part.code_execution_result.output == "error occurred"

  def test_convert_code_execution_parts_no_parts(self):
    content = types.Content(parts=[])
    original_content_copy = content.model_copy(deep=True)
    CodeExecutionUtils.convert_code_execution_parts(content, ("<", ">"), ("<<", ">>"))
    assert content == original_content_copy

  def test_convert_code_execution_parts_executable_code(self):
    code_str = "print(1)"
    content = types.Content(parts=[types.Part(executable_code=types.ExecutableCode(code=code_str, language="PYTHON"))])
    CodeExecutionUtils.convert_code_execution_parts(content, ("`", "`"), ("``", "``"))
    assert content.parts[0].text == f"`{code_str}`"

  @pytest.mark.filterwarnings("ignore:OK is not a valid outcome")
  def test_convert_code_execution_parts_execution_result(self):
    result_output = "output"
    content = types.Content(parts=[types.Part(code_execution_result=types.CodeExecutionResult(outcome="OK", output=result_output))])
    CodeExecutionUtils.convert_code_execution_parts(content, ("`", "`"), ("``", "``"))
    assert content.parts[0].text == f"``{result_output}``"
    assert content.role == "user" # Role should change to user for result part

  @pytest.mark.filterwarnings("ignore:OK is not a valid outcome")
  def test_convert_code_execution_parts_multiple_parts_no_conversion(self):
    # Conversion should only happen for trailing executable_code or single-part code_execution_result
    content = types.Content(parts=[
        types.Part(text="text before"),
        types.Part(code_execution_result=types.CodeExecutionResult(outcome="OK", output="output"))
    ])
    original_content_copy = content.model_copy(deep=True)
    CodeExecutionUtils.convert_code_execution_parts(content, ("`", "`"), ("``", "``"))
    assert content == original_content_copy