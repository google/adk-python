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

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.plugins.multimodal_tool_results_plugin import MultimodalToolResultsPlugin
from google.adk.plugins.multimodal_tool_results_plugin import PARTS_RETURNED_BY_TOOLS_ID
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest

from .. import testing_utils


@pytest.fixture
def plugin() -> MultimodalToolResultsPlugin:
  """Create a default plugin instance for testing."""
  return MultimodalToolResultsPlugin()


@pytest.fixture
def mock_tool() -> MockTool:
  """Create a mock tool for testing."""
  return Mock(spec=BaseTool)


@pytest.fixture
async def tool_context() -> ToolContext:
  """Create a mock tool context."""
  return ToolContext(
      invocation_context=await testing_utils.create_invocation_context(
          agent=Mock(spec=BaseAgent)
      )
  )


@pytest.mark.asyncio
async def test_tool_returning_parts_are_added_to_llm_request(
    plugin: MultimodalToolResultsPlugin,
    mock_tool: MockTool,
    tool_context: ToolContext,
):
  """Test that parts returned by a tool are present in the llm_request later."""
  parts = [types.Part(text="part1"), types.Part(text="part2")]

  result = await plugin.after_tool_callback(
      tool=mock_tool,
      tool_args={},
      tool_context=tool_context,
      result=parts,
  )

  assert result == None
  assert PARTS_RETURNED_BY_TOOLS_ID in tool_context.state
  assert tool_context.state[PARTS_RETURNED_BY_TOOLS_ID] == parts

  callback_context = Mock(spec=CallbackContext)
  callback_context.state = tool_context.state
  llm_request = LlmRequest(contents=[types.Content(parts=[])])

  await plugin.before_model_callback(
      callback_context=callback_context, llm_request=llm_request
  )

  assert llm_request.contents[-1].parts == parts


@pytest.mark.asyncio
async def test_tool_returning_non_list_of_parts_is_unchanged(
    plugin: MultimodalToolResultsPlugin,
    mock_tool: MockTool,
    tool_context: ToolContext,
):
  """Test where tool returning non list of parts, has this result unchanged."""
  original_result = {"some": "data"}

  result = await plugin.after_tool_callback(
      tool=mock_tool,
      tool_args={},
      tool_context=tool_context,
      result=original_result,
  )

  assert result == original_result
  assert PARTS_RETURNED_BY_TOOLS_ID not in tool_context.state

  callback_context = Mock(spec=CallbackContext)
  callback_context.state = tool_context.state
  llm_request = LlmRequest(
      contents=[types.Content(parts=[types.Part(text="original")])]
  )
  original_parts = list(llm_request.contents[-1].parts)

  await plugin.before_model_callback(
      callback_context=callback_context, llm_request=llm_request
  )

  assert llm_request.contents[-1].parts == original_parts


@pytest.mark.asyncio
async def test_multiple_tools_returning_parts_are_accumulated(
    plugin: ToolReturningGenAiPartsPlugin,
    mock_tool: MockTool,
    tool_context: ToolContext,
):
  """Test that parts from multiple tool calls are accumulated."""
  parts1 = [types.Part(text="part1")]
  parts2 = [types.Part(text="part2")]

  await plugin.after_tool_callback(
      tool=mock_tool,
      tool_args={},
      tool_context=tool_context,
      result=parts1,
  )

  await plugin.after_tool_callback(
      tool=mock_tool,
      tool_args={},
      tool_context=tool_context,
      result=parts2,
  )

  assert PARTS_RETURNED_BY_TOOLS_ID in tool_context.state
  assert tool_context.state[PARTS_RETURNED_BY_TOOLS_ID] == parts1 + parts2

  callback_context = Mock(spec=CallbackContext)
  callback_context.state = tool_context.state
  llm_request = LlmRequest(contents=[types.Content(parts=[])])

  await plugin.before_model_callback(
      callback_context=callback_context, llm_request=llm_request
  )

  assert llm_request.contents[-1].parts == parts1 + parts2
