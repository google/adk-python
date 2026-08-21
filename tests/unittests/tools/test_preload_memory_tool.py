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

from unittest import mock

from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.models.gemini_context_cache_manager import GeminiContextCacheManager
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.genai import types
import pytest


def _tool_context(*memories: MemoryEntry):
  tool_context = mock.Mock()
  tool_context.user_content = types.UserContent('current query')
  tool_context.search_memory = mock.AsyncMock(
      return_value=SearchMemoryResponse(memories=list(memories))
  )
  return tool_context


def _memory(text: str) -> MemoryEntry:
  return MemoryEntry(
      content=types.UserContent(text),
      author='user',
      timestamp='2026-07-13T12:00:00Z',
  )


@pytest.mark.asyncio
async def test_preload_memory_keeps_system_prefix_stable():
  """Recalled memory goes into contents, never into the system instruction."""
  request = LlmRequest(
      contents=[
          types.UserContent('historical question'),
          types.ModelContent('historical answer'),
          types.UserContent('current query'),
      ]
  )
  request.config.system_instruction = 'stable instruction'

  await PreloadMemoryTool().process_llm_request(
      tool_context=_tool_context(_memory('likes tea')),
      llm_request=request,
  )

  assert request.config.system_instruction == 'stable instruction'
  assert [content.role for content in request.contents] == [
      'user',
      'model',
      'user',
      'user',
  ]
  assert 'likes tea' in request.contents[-2].parts[0].text
  assert request.contents[-1] == types.UserContent('current query')


@pytest.mark.asyncio
async def test_preload_memory_stays_after_function_response_boundary():
  """Recalled memory lands after a trailing function response."""
  function_response = types.Content(
      role='user',
      parts=[
          types.Part.from_function_response(
              name='lookup', response={'result': 'done'}
          )
      ],
  )
  request = LlmRequest(
      contents=[
          types.UserContent('current query'),
          types.ModelContent(
              types.Part.from_function_call(name='lookup', args={})
          ),
          function_response,
      ]
  )

  await PreloadMemoryTool().process_llm_request(
      tool_context=_tool_context(_memory('likes tea')),
      llm_request=request,
  )

  assert request.contents[-2] is function_response
  assert 'likes tea' in request.contents[-1].parts[0].text


@pytest.mark.asyncio
async def test_preload_memory_does_not_change_cacheable_prefix_fingerprint():
  """Different recalled memories keep the same prefix fingerprint."""
  requests = []
  for memory_text in ('likes tea', 'likes coffee'):
    request = LlmRequest(
        model='gemini-2.5-flash',
        contents=[
            types.UserContent('historical question'),
            types.ModelContent('historical answer'),
            types.UserContent('current query'),
        ],
    )
    request.config.system_instruction = 'stable instruction'
    await PreloadMemoryTool().process_llm_request(
        tool_context=_tool_context(_memory(memory_text)),
        llm_request=request,
    )
    requests.append(request)

  client = mock.Mock(vertexai=False)
  client._api_client = None
  manager = GeminiContextCacheManager(client)
  prefix_counts = [
      manager._find_count_of_contents_to_cache(request.contents)
      for request in requests
  ]
  fingerprints = [
      manager._generate_cache_fingerprint(request, prefix_count)
      for request, prefix_count in zip(requests, prefix_counts)
  ]

  assert prefix_counts == [2, 2]
  assert fingerprints[0] == fingerprints[1]


@pytest.mark.asyncio
async def test_preload_memory_search_failure_is_noop():
  """A failing memory search leaves the request completely untouched."""
  request = LlmRequest(contents=[types.UserContent('current query')])
  request.config.system_instruction = 'stable instruction'
  original = request.model_copy(deep=True)
  tool_context = _tool_context()
  tool_context.search_memory.side_effect = RuntimeError('unavailable')

  await PreloadMemoryTool().process_llm_request(
      tool_context=tool_context,
      llm_request=request,
  )

  assert request == original
