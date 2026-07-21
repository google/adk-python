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
from unittest.mock import AsyncMock

from google.adk.features import FeatureName
from google.adk.features._feature_registry import temporary_feature_override
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.load_memory_tool import load_memory_tool
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest


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
async def test_preload_memory_registers_dynamic_instructions():
  """Test that PreloadMemoryTool registers memory into _dynamic_instructions."""
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

  assert len(llm_request._dynamic_instructions) == 1
  assert '<PAST_CONVERSATIONS>' in llm_request._dynamic_instructions[0]
  assert llm_request.config.system_instruction is None
  assert len(llm_request.contents) == 0
