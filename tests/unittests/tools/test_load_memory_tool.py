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

from contextlib import closing
import sqlite3
from typing import Literal
from unittest import mock
from unittest.mock import AsyncMock

from google.adk.agents.llm_agent import Agent
from google.adk.events.event import Event
from google.adk.features import FeatureName
from google.adk.features._feature_registry import temporary_feature_override
from google.adk.memory import SqliteMemoryService
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.load_memory_tool import load_memory_tool
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest

from .. import testing_utils


@pytest.mark.parametrize('fts_mode', ['off', 'on'])
async def test_runner_loads_only_visible_sqlite_memory(
    fts_mode: Literal['off', 'on'],
) -> None:
  """The next model request contains visible memory without thought parts."""
  if fts_mode == 'on':
    with closing(sqlite3.connect(':memory:')) as conn:
      try:
        conn.execute('CREATE VIRTUAL TABLE test_fts USING fts5(col)')
      except sqlite3.OperationalError:
        pytest.skip('FTS5 not available in this SQLite build.')
  memory_service = SqliteMemoryService(':memory:', fts=fts_mode)
  model = testing_utils.MockModel.create(
      responses=[
          types.Part.from_function_call(
              name='load_memory', args={'query': 'saffron'}
          ),
          types.Part(text='Your preferred tea is saffron.'),
      ]
  )
  runner = testing_utils.InMemoryRunner(
      Agent(name='assistant', model=model, tools=[load_memory_tool])
  )
  runner.runner.memory_service = memory_service
  try:
    await memory_service.add_events_to_memory(
        app_name=runner.app_name,
        user_id='test_user',
        session_id='history',
        events=[
            Event(
                author='assistant',
                content=types.Content(
                    role='model',
                    parts=[
                        types.Part(
                            text='Synthetic obsidiansecret', thought=True
                        ),
                        types.Part(text='My preferred tea is saffron.'),
                    ],
                ),
            )
        ],
    )

    await runner.run_async('What is my preferred tea?')

    assert len(model.requests) == 2
    function_responses = [
        part.function_response
        for content in model.requests[1].contents
        for part in content.parts or []
        if part.function_response
        and part.function_response.name == 'load_memory'
    ]
    assert len(function_responses) == 1
    payload = function_responses[0].model_dump(mode='json')
    memories = payload['response']['result']['memories']
    assert len(memories) == 1
    assert types.Content.model_validate(
        memories[0]['content']
    ) == types.Content(
        role='model', parts=[types.Part(text='My preferred tea is saffron.')]
    )
    assert 'obsidiansecret' not in model.requests[1].model_dump_json()
  finally:
    await memory_service.close()
    await runner.runner.close()


def test_get_declaration_with_json_schema_feature_disabled():
  """Test that _get_declaration uses parameters when feature is disabled."""
  with temporary_feature_override(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, False):
    declaration = load_memory_tool._get_declaration()

  assert declaration.name == 'load_memory'
  assert declaration.parameters_json_schema is None
  assert isinstance(declaration.parameters, types.Schema)
  assert declaration.parameters.type == types.Type.OBJECT
  assert 'query' in declaration.parameters.properties


def test_get_declaration_with_json_schema_feature_enabled():
  """Test that _get_declaration uses parameters_json_schema when feature is enabled."""
  with temporary_feature_override(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, True):
    declaration = load_memory_tool._get_declaration()

  assert declaration.name == 'load_memory'
  assert declaration.parameters is None
  assert declaration.parameters_json_schema == {
      'type': 'object',
      'properties': {
          'query': {'type': 'string'},
      },
      'required': ['query'],
  }


@pytest.mark.asyncio
async def test_process_llm_request_registers_the_tool():
  """The base class contribution: the model can actually call load_memory."""
  tool_context = mock.Mock(spec=ToolContext)
  llm_request = LlmRequest()

  await load_memory_tool.process_llm_request(
      tool_context=tool_context, llm_request=llm_request
  )

  assert llm_request.tools_dict['load_memory'] is load_memory_tool


@pytest.mark.asyncio
async def test_process_llm_request_tells_the_model_it_has_memory():
  """Without the instruction the model never knows to call the tool."""
  tool_context = mock.Mock(spec=ToolContext)
  llm_request = LlmRequest()

  await load_memory_tool.process_llm_request(
      tool_context=tool_context, llm_request=llm_request
  )

  assert 'You have memory.' in llm_request.config.system_instruction
  assert (
      'call load_memory function with a query'
      in llm_request.config.system_instruction
  )


@pytest.mark.asyncio
async def test_process_llm_request_appends_to_existing_system_instruction():
  """The memory instruction must not clobber instructions already there."""
  tool_context = mock.Mock(spec=ToolContext)
  llm_request = LlmRequest()
  llm_request.config.system_instruction = 'be terse'

  await load_memory_tool.process_llm_request(
      tool_context=tool_context, llm_request=llm_request
  )

  assert llm_request.config.system_instruction.startswith('be terse')
  assert 'You have memory.' in llm_request.config.system_instruction


@pytest.mark.asyncio
async def test_preload_memory_registers_transient_user_content():
  """Test that PreloadMemoryTool registers memory as transient user content."""
  tool = PreloadMemoryTool()
  tool_context = mock.Mock(spec=ToolContext)
  tool_context.user_content = types.Content(
      role='user', parts=[types.Part.from_text(text='my query')]
  )
  mock_memory = mock.Mock()
  mock_memory.timestamp = None
  mock_memory.author = 'user'
  mock_memory.content = types.Content(
      parts=[
          types.Part.from_text(text='User prefers Python programming language.')
      ]
  )

  tool_context.search_memory = AsyncMock(
      return_value=mock.Mock(memories=[mock_memory])
  )

  llm_request = LlmRequest()

  await tool.process_llm_request(
      tool_context=tool_context, llm_request=llm_request
  )

  assert len(llm_request._dynamic_instructions) == 0
  assert llm_request.config.system_instruction is None
  assert len(llm_request.contents) == 1
  assert llm_request.contents[0].role == 'user'
  assert '<PAST_CONVERSATIONS>' in llm_request.contents[0].parts[0].text
