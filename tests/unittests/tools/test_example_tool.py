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

"""Tests for ExampleTool."""

from unittest.mock import MagicMock

from google.adk.models.llm_request import LlmRequest
from google.adk.tools.example_tool import ExampleTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest

_EXAMPLE = {
    'input': {'role': 'user', 'parts': [{'text': 'hello'}]},
    'output': [{'role': 'model', 'parts': [{'text': 'hi'}]}],
}


def _tool_context(user_content):
  tool_context = MagicMock(spec=ToolContext)
  tool_context.user_content = user_content
  return tool_context


@pytest.mark.asyncio
async def test_process_llm_request_without_user_content():
  """A live run opens with no user content, which must not raise."""
  tool = ExampleTool(examples=[_EXAMPLE])
  llm_request = LlmRequest()

  await tool.process_llm_request(
      tool_context=_tool_context(None), llm_request=llm_request
  )

  assert not llm_request.config.system_instruction


@pytest.mark.asyncio
async def test_process_llm_request_appends_examples():
  """A user message still selects examples and appends the instruction."""
  tool = ExampleTool(examples=[_EXAMPLE])
  llm_request = LlmRequest()
  user_content = types.Content(
      role='user', parts=[types.Part.from_text(text='hello')]
  )

  await tool.process_llm_request(
      tool_context=_tool_context(user_content), llm_request=llm_request
  )

  assert llm_request.config.system_instruction
