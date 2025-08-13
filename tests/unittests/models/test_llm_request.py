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

"""Tests for LlmRequest functionality."""

import asyncio
from typing import Optional

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.models.llm_request import LlmRequest
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest


def dummy_tool(query: str) -> str:
  """A dummy tool for testing."""
  return f'Searched for: {query}'


def test_append_tools_with_none_config_tools():
  """Test that append_tools initializes config.tools when it's None."""
  request = LlmRequest()

  # Initially config.tools should be None
  assert request.config.tools is None

  # Create a tool to append
  tool = FunctionTool(func=dummy_tool)

  # This should not raise an AttributeError
  request.append_tools([tool])

  # Now config.tools should be initialized and contain the tool
  assert request.config.tools is not None
  assert len(request.config.tools) == 1
  assert len(request.config.tools[0].function_declarations) == 1
  assert request.config.tools[0].function_declarations[0].name == 'dummy_tool'

  # Tool should also be in tools_dict
  assert 'dummy_tool' in request.tools_dict
  assert request.tools_dict['dummy_tool'] == tool


def test_append_tools_with_existing_tools():
  """Test that append_tools works correctly when config.tools already exists."""
  request = LlmRequest()

  # Pre-initialize config.tools with an existing tool
  existing_declaration = types.FunctionDeclaration(
      name='existing_tool', description='An existing tool', parameters={}
  )
  request.config.tools = [
      types.Tool(function_declarations=[existing_declaration])
  ]

  # Create a new tool to append
  tool = FunctionTool(func=dummy_tool)

  # Append the new tool
  request.append_tools([tool])

  # Should still have 1 tool but now with 2 function declarations
  assert len(request.config.tools) == 1
  assert len(request.config.tools[0].function_declarations) == 2

  # Verify both declarations are present
  decl_names = {
      decl.name for decl in request.config.tools[0].function_declarations
  }
  assert decl_names == {'existing_tool', 'dummy_tool'}


def test_append_tools_empty_list():
  """Test that append_tools handles empty list correctly."""
  request = LlmRequest()

  # This should not modify anything
  request.append_tools([])

  # config.tools should still be None
  assert request.config.tools is None
  assert len(request.tools_dict) == 0


def test_append_tools_tool_with_no_declaration():
  """Test append_tools with a BaseTool that returns None from _get_declaration."""
  from google.adk.tools.base_tool import BaseTool

  request = LlmRequest()

  # Create a mock tool that inherits from BaseTool and returns None for declaration
  class NoDeclarationTool(BaseTool):

    def __init__(self):
      super().__init__(
          name='no_decl_tool', description='A tool with no declaration'
      )

    def _get_declaration(self):
      return None

  tool = NoDeclarationTool()

  # This should not add anything to config.tools but should handle gracefully
  request.append_tools([tool])

  # config.tools should still be None since no declarations were added
  assert request.config.tools is None
  # tools_dict should be empty since no valid declaration
  assert len(request.tools_dict) == 0


def test_append_tools_consolidates_declarations_in_single_tool():
  """Test that append_tools puts all function declarations in a single Tool."""
  request = LlmRequest()

  # Create multiple tools
  tool1 = FunctionTool(func=dummy_tool)

  def another_tool(param: str) -> str:
    return f'Another: {param}'

  def third_tool(value: int) -> int:
    return value * 2

  tool2 = FunctionTool(func=another_tool)
  tool3 = FunctionTool(func=third_tool)

  # Append all tools at once
  request.append_tools([tool1, tool2, tool3])

  # Should have exactly 1 Tool with 3 function declarations
  assert len(request.config.tools) == 1
  assert len(request.config.tools[0].function_declarations) == 3

  # Verify all tools are in tools_dict
  assert len(request.tools_dict) == 3
  assert 'dummy_tool' in request.tools_dict
  assert 'another_tool' in request.tools_dict
  assert 'third_tool' in request.tools_dict


async def _create_tool_context() -> ToolContext:
  """Helper to create a ToolContext for testing."""
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )
  agent = SequentialAgent(name='test_agent')
  invocation_context = InvocationContext(
      invocation_id='invocation_id',
      agent=agent,
      session=session,
      session_service=session_service,
  )
  return ToolContext(invocation_context)


class _MockTool(BaseTool):
  """Mock tool for testing process_llm_request behavior."""

  def __init__(self, name: str):
    super().__init__(name=name, description=f'Mock tool {name}')

  def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters=types.Schema(type=types.Type.STRING, title='param'),
    )


@pytest.mark.asyncio
async def test_process_llm_request_consolidates_declarations_in_single_tool():
  """Test that multiple process_llm_request calls consolidate in single Tool."""
  request = LlmRequest()
  tool_context = await _create_tool_context()

  # Create multiple tools
  tool1 = _MockTool('tool1')
  tool2 = _MockTool('tool2')
  tool3 = _MockTool('tool3')

  # Process each tool individually (simulating what happens in real usage)
  await tool1.process_llm_request(
      tool_context=tool_context, llm_request=request
  )
  await tool2.process_llm_request(
      tool_context=tool_context, llm_request=request
  )
  await tool3.process_llm_request(
      tool_context=tool_context, llm_request=request
  )

  # Should have exactly 1 Tool with 3 function declarations
  assert len(request.config.tools) == 1
  assert len(request.config.tools[0].function_declarations) == 3

  # Verify all function declaration names
  decl_names = [
      decl.name for decl in request.config.tools[0].function_declarations
  ]
  assert 'tool1' in decl_names
  assert 'tool2' in decl_names
  assert 'tool3' in decl_names

  # Verify all tools are in tools_dict
  assert len(request.tools_dict) == 3
  assert 'tool1' in request.tools_dict
  assert 'tool2' in request.tools_dict
  assert 'tool3' in request.tools_dict


@pytest.mark.asyncio
async def test_append_tools_and_process_llm_request_consistent_behavior():
  """Test that append_tools and process_llm_request produce same structure."""
  tool_context = await _create_tool_context()

  # Test 1: Using append_tools
  request1 = LlmRequest()
  tool1 = _MockTool('tool1')
  tool2 = _MockTool('tool2')
  tool3 = _MockTool('tool3')
  request1.append_tools([tool1, tool2, tool3])

  # Test 2: Using process_llm_request
  request2 = LlmRequest()
  tool4 = _MockTool('tool1')  # Same names for comparison
  tool5 = _MockTool('tool2')
  tool6 = _MockTool('tool3')
  await tool4.process_llm_request(
      tool_context=tool_context, llm_request=request2
  )
  await tool5.process_llm_request(
      tool_context=tool_context, llm_request=request2
  )
  await tool6.process_llm_request(
      tool_context=tool_context, llm_request=request2
  )

  # Both approaches should produce identical structure
  assert len(request1.config.tools) == len(request2.config.tools) == 1
  assert len(request1.config.tools[0].function_declarations) == 3
  assert len(request2.config.tools[0].function_declarations) == 3

  # Function declaration names should match
  decl_names1 = {
      decl.name for decl in request1.config.tools[0].function_declarations
  }
  decl_names2 = {
      decl.name for decl in request2.config.tools[0].function_declarations
  }
  assert decl_names1 == decl_names2 == {'tool1', 'tool2', 'tool3'}


def test_multiple_append_tools_calls_consolidate():
  """Test that multiple append_tools calls add to the same Tool."""
  request = LlmRequest()

  # First call to append_tools
  tool1 = FunctionTool(func=dummy_tool)
  request.append_tools([tool1])

  # Should have 1 tool with 1 declaration
  assert len(request.config.tools) == 1
  assert len(request.config.tools[0].function_declarations) == 1
  assert request.config.tools[0].function_declarations[0].name == 'dummy_tool'

  # Second call to append_tools with different tools
  def another_tool(param: str) -> str:
    return f'Another: {param}'

  def third_tool(value: int) -> int:
    return value * 2

  tool2 = FunctionTool(func=another_tool)
  tool3 = FunctionTool(func=third_tool)
  request.append_tools([tool2, tool3])

  # Should still have 1 tool but now with 3 declarations
  assert len(request.config.tools) == 1
  assert len(request.config.tools[0].function_declarations) == 3

  # Verify all declaration names are present
  decl_names = {
      decl.name for decl in request.config.tools[0].function_declarations
  }
  assert decl_names == {'dummy_tool', 'another_tool', 'third_tool'}

  # Verify all tools are in tools_dict
  assert len(request.tools_dict) == 3
  assert 'dummy_tool' in request.tools_dict
  assert 'another_tool' in request.tools_dict
  assert 'third_tool' in request.tools_dict
