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

from google.adk.examples.base_example_provider import BaseExampleProvider
from google.adk.examples.example import Example
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.example_tool import ExampleTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest

_EXAMPLE = {
    'input': {'role': 'user', 'parts': [{'text': 'hello'}]},
    'output': [{'role': 'model', 'parts': [{'text': 'hi'}]}],
}


class _RecordingExampleProvider(BaseExampleProvider):

  def __init__(self, examples: list[Example]):
    self._examples = examples
    self.queries: list[str] = []

  def get_examples(self, query: str) -> list[Example]:
    self.queries.append(query)
    return self._examples


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


@pytest.mark.asyncio
async def test_process_llm_request_leading_non_text_part_with_static_list():
  """A leading non-text part must not drop static list[Example] instructions."""
  tool = ExampleTool(examples=[_EXAMPLE])
  llm_request = LlmRequest()
  user_content = types.Content(
      role='user',
      parts=[
          types.Part.from_bytes(data=b'fake_pdf', mime_type='application/pdf'),
          types.Part.from_text(text='Summarize this document'),
      ],
  )

  await tool.process_llm_request(
      tool_context=_tool_context(user_content), llm_request=llm_request
  )

  assert llm_request.config.system_instruction


@pytest.mark.asyncio
async def test_process_llm_request_leading_non_text_part_appends_examples():
  """A leading non-text part must not drop dynamic provider examples."""
  provider = _RecordingExampleProvider([Example.model_validate(_EXAMPLE)])
  tool = ExampleTool(examples=provider)
  llm_request = LlmRequest()
  user_content = types.Content(
      role='user',
      parts=[
          types.Part.from_bytes(data=b'fake_pdf', mime_type='application/pdf'),
          types.Part.from_text(text='Summarize this document'),
      ],
  )

  await tool.process_llm_request(
      tool_context=_tool_context(user_content), llm_request=llm_request
  )

  assert llm_request.config.system_instruction
  assert provider.queries == ['Summarize this document']


@pytest.mark.asyncio
async def test_process_llm_request_concatenates_multiple_text_parts():
  """Multiple text parts in user_content are joined with spaces."""
  provider = _RecordingExampleProvider([Example.model_validate(_EXAMPLE)])
  tool = ExampleTool(examples=provider)
  llm_request = LlmRequest()
  user_content = types.Content(
      role='user',
      parts=[
          types.Part.from_text(text='[Attachment]'),
          types.Part.from_bytes(data=b'fake_image', mime_type='image/png'),
          types.Part.from_text(text='What error is shown?'),
      ],
  )

  await tool.process_llm_request(
      tool_context=_tool_context(user_content), llm_request=llm_request
  )

  assert llm_request.config.system_instruction
  assert provider.queries == ['[Attachment] What error is shown?']


@pytest.mark.asyncio
async def test_process_llm_request_excludes_thought_parts_from_query():
  """Thought parts in user_content are excluded from the provider query."""
  provider = _RecordingExampleProvider([Example.model_validate(_EXAMPLE)])
  tool = ExampleTool(examples=provider)
  llm_request = LlmRequest()
  user_content = types.Content(
      role='user',
      parts=[
          types.Part(text='internal reasoning', thought=True),
          types.Part.from_text(text='What error is shown?'),
      ],
  )

  await tool.process_llm_request(
      tool_context=_tool_context(user_content), llm_request=llm_request
  )

  assert llm_request.config.system_instruction
  assert provider.queries == ['What error is shown?']


@pytest.mark.asyncio
async def test_process_llm_request_non_text_turn_with_static_list():
  """A non-text user turn still appends static list[Example] instructions."""
  tool = ExampleTool(examples=[_EXAMPLE])
  llm_request = LlmRequest()
  user_content = types.Content(
      role='user',
      parts=[types.Part.from_bytes(data=b'fake_image', mime_type='image/png')],
  )

  await tool.process_llm_request(
      tool_context=_tool_context(user_content), llm_request=llm_request
  )

  assert llm_request.config.system_instruction


@pytest.mark.asyncio
async def test_process_llm_request_only_empty_or_non_text_parts_is_noop():
  """When user_content has no non-empty text, provider does not append."""
  provider = _RecordingExampleProvider([Example.model_validate(_EXAMPLE)])
  tool = ExampleTool(examples=provider)
  llm_request = LlmRequest()
  user_content = types.Content(
      role='user',
      parts=[
          types.Part.from_text(text=''),
          types.Part.from_bytes(data=b'fake_image', mime_type='image/png'),
      ],
  )

  await tool.process_llm_request(
      tool_context=_tool_context(user_content), llm_request=llm_request
  )

  assert not llm_request.config.system_instruction
  assert provider.queries == []
