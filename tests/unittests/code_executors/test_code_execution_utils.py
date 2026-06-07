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

from unittest.mock import patch

from google.adk.code_executors import code_execution_utils as utils_module
from google.adk.code_executors.code_execution_utils import CodeExecutionUtils
from google.genai import types


CODE_BLOCK_DELIMITERS = [('```python\n', '\n```')]


def test_extract_code_skips_regex_when_opening_delimiter_is_missing():
  content = types.Content(parts=[types.Part(text='x' * 100_000)])

  with patch.object(
      utils_module.re,
      'compile',
      side_effect=AssertionError('the full regex should be skipped'),
  ):
    code = CodeExecutionUtils.extract_code_and_truncate_content(
        content, CODE_BLOCK_DELIMITERS
    )

  assert code is None
  assert content.parts[0].text == 'x' * 100_000


def test_extract_code_from_text_part_still_truncates_after_first_block():
  content = types.Content(
      parts=[
          types.Part(text='before ```python\nprint("ok")\n``` after'),
      ]
  )

  code = CodeExecutionUtils.extract_code_and_truncate_content(
      content, CODE_BLOCK_DELIMITERS
  )

  assert code == 'print("ok")'
  assert content.parts[0].text == 'before '
  assert content.parts[1].executable_code.code == 'print("ok")'
