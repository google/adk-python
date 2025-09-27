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


def test_append_instructions_with_string_list():
  """Test that append_instructions works with list of strings (existing behavior)."""
  request = LlmRequest()

  # Initially system_instruction should be None
  assert request.config.system_instruction is None

  # Append first set of instructions
  request.append_instructions(['First instruction', 'Second instruction'])

  # Should be joined with double newlines
  expected = 'First instruction\n\nSecond instruction'
  assert request.config.system_instruction == expected
  assert len(request.contents) == 0


def test_append_instructions_with_string_list_multiple_calls():
  """Test multiple calls to append_instructions with string lists."""
  request = LlmRequest()

  # First call
  request.append_instructions(['First instruction'])
  assert request.config.system_instruction == 'First instruction'

  # Second call should append with double newlines
  request.append_instructions(['Second instruction', 'Third instruction'])
  expected = 'First instruction\n\nSecond instruction\n\nThird instruction'
  assert request.config.system_instruction == expected


def test_append_instructions_with_content():
  """Test that append_instructions works with types.Content (new behavior)."""
  request = LlmRequest()

  # Create a Content object
  content = types.Content(
      role='user', parts=[types.Part(text='This is content-based instruction')]
  )

  # Append content
  request.append_instructions(content)

  # Should be set as system_instruction
  assert len(request.contents) == 0
  assert request.config.system_instruction == content


def test_append_instructions_with_content_multiple_calls():
  """Test multiple calls to append_instructions with Content objects."""
  request = LlmRequest()

  # Add some existing content first
  existing_content = types.Content(
      role='user', parts=[types.Part(text='Existing content')]
  )
  request.contents.append(existing_content)

  # First Content instruction
  content1 = types.Content(
      role='user', parts=[types.Part(text='First instruction')]
  )
  request.append_instructions(content1)

  # Should be set as system_instruction, existing content unchanged
  assert len(request.contents) == 1
  assert request.contents[0] == existing_content
  assert request.config.system_instruction == content1

  # Second Content instruction
  content2 = types.Content(
      role='user', parts=[types.Part(text='Second instruction')]
  )
  request.append_instructions(content2)

  # Second Content should be merged with first in system_instruction
  assert len(request.contents) == 1
  assert request.contents[0] == existing_content
  assert isinstance(request.config.system_instruction, types.Content)
  assert len(request.config.system_instruction.parts) == 2
  assert request.config.system_instruction.parts[0].text == 'First instruction'
  assert request.config.system_instruction.parts[1].text == 'Second instruction'


def test_append_instructions_with_content_multipart():
  """Test append_instructions with Content containing multiple parts."""
  request = LlmRequest()

  # Create Content with multiple parts (text and potentially files)
  content = types.Content(
      role='user',
      parts=[
          types.Part(text='Text instruction'),
          types.Part(text='Additional text part'),
      ],
  )

  request.append_instructions(content)

  assert len(request.contents) == 0
  assert request.config.system_instruction == content
  assert len(request.config.system_instruction.parts) == 2
  assert request.config.system_instruction.parts[0].text == 'Text instruction'
  assert (
      request.config.system_instruction.parts[1].text == 'Additional text part'
  )


def test_append_instructions_mixed_string_and_content():
  """Test mixing string list and Content instructions."""
  request = LlmRequest()

  # First add string instructions
  request.append_instructions(['String instruction'])
  assert request.config.system_instruction == 'String instruction'

  # Then add Content instruction
  content = types.Content(
      role='user', parts=[types.Part(text='Content instruction')]
  )
  request.append_instructions(content)

  # String and Content should be merged in system_instruction
  assert len(request.contents) == 0
  assert isinstance(request.config.system_instruction, types.Content)
  assert len(request.config.system_instruction.parts) == 2
  assert request.config.system_instruction.parts[0].text == 'String instruction'
  assert (
      request.config.system_instruction.parts[1].text == 'Content instruction'
  )


def test_append_instructions_empty_string_list():
  """Test append_instructions with empty list of strings."""
  request = LlmRequest()

  # Empty list should not modify anything
  request.append_instructions([])

  assert request.config.system_instruction is None
  assert len(request.contents) == 0


def test_append_instructions_invalid_input():
  """Test append_instructions with invalid input types."""
  request = LlmRequest()

  # Test with invalid types
  with pytest.raises(
      TypeError, match='instructions must be list\\[str\\] or types.Content'
  ):
    request.append_instructions('single string')  # Should be list[str]

  with pytest.raises(
      TypeError, match='instructions must be list\\[str\\] or types.Content'
  ):
    request.append_instructions(123)  # Invalid type

  with pytest.raises(
      TypeError, match='instructions must be list\\[str\\] or types.Content'
  ):
    request.append_instructions(
        ['valid string', 123]
    )  # Mixed valid/invalid in list


def test_append_instructions_content_preserves_role_and_parts():
  """Test that Content objects have text extracted regardless of role or parts."""
  request = LlmRequest()

  # Create Content with specific role and parts
  content = types.Content(
      role='system',  # Different role
      parts=[
          types.Part(text='System instruction'),
          types.Part(text='Additional system part'),
      ],
  )

  request.append_instructions(content)

  # Text should be extracted and concatenated to system_instruction string
  assert len(request.contents) == 0
  assert (
      request.config.system_instruction
      == 'System instruction\n\nAdditional system part'
  )


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


# Updated tests for simplified string-only append_instructions behavior


def test_append_instructions_with_content():
  """Test that append_instructions extracts text from types.Content."""
  request = LlmRequest()

  # Create a Content object
  content = types.Content(
      role='user', parts=[types.Part(text='This is content-based instruction')]
  )

  # Append content
  request.append_instructions(content)

  # Should extract text and set as system_instruction string
  assert len(request.contents) == 0
  assert (
      request.config.system_instruction == 'This is content-based instruction'
  )


def test_append_instructions_with_content_multiple_calls():
  """Test multiple calls to append_instructions with Content objects."""
  request = LlmRequest()

  # Add some existing content first
  existing_content = types.Content(
      role='user', parts=[types.Part(text='Existing content')]
  )
  request.contents.append(existing_content)

  # First Content instruction
  content1 = types.Content(
      role='user', parts=[types.Part(text='First instruction')]
  )
  request.append_instructions(content1)

  # Should extract text and set as system_instruction, existing content unchanged
  assert len(request.contents) == 1
  assert request.contents[0] == existing_content
  assert request.config.system_instruction == 'First instruction'

  # Second Content instruction
  content2 = types.Content(
      role='user', parts=[types.Part(text='Second instruction')]
  )
  request.append_instructions(content2)

  # Second Content text should be appended to existing string
  assert len(request.contents) == 1
  assert request.contents[0] == existing_content
  assert (
      request.config.system_instruction
      == 'First instruction\n\nSecond instruction'
  )


def test_append_instructions_with_content_multipart():
  """Test append_instructions with Content containing multiple text parts."""
  request = LlmRequest()

  # Create Content with multiple text parts
  content = types.Content(
      role='user',
      parts=[
          types.Part(text='Text instruction'),
          types.Part(text='Additional text part'),
      ],
  )

  request.append_instructions(content)

  # Should extract and join all text parts
  assert len(request.contents) == 0
  assert (
      request.config.system_instruction
      == 'Text instruction\n\nAdditional text part'
  )


def test_append_instructions_mixed_string_and_content():
  """Test mixing string list and Content instructions."""
  request = LlmRequest()

  # First add string instructions
  request.append_instructions(['String instruction'])
  assert request.config.system_instruction == 'String instruction'

  # Then add Content instruction
  content = types.Content(
      role='user', parts=[types.Part(text='Content instruction')]
  )
  request.append_instructions(content)

  # Content text should be appended to existing string
  assert len(request.contents) == 0
  assert (
      request.config.system_instruction
      == 'String instruction\n\nContent instruction'
  )


def test_append_instructions_content_extracts_text_only():
  """Test that Content objects have text extracted regardless of role."""
  request = LlmRequest()

  # Create Content with specific role and parts
  content = types.Content(
      role='system',  # Different role
      parts=[
          types.Part(text='System instruction'),
          types.Part(text='Additional system part'),
      ],
  )

  request.append_instructions(content)

  # Only text should be extracted and concatenated
  assert len(request.contents) == 0
  assert (
      request.config.system_instruction
      == 'System instruction\n\nAdditional system part'
  )


def test_append_instructions_content_with_non_text_parts():
  """Test that non-text parts in Content are processed with references."""
  request = LlmRequest()

  # Create Content with text and non-text parts
  content = types.Content(
      role='user',
      parts=[
          types.Part(text='Text instruction'),
          types.Part(
              inline_data=types.Blob(data=b'file_data', mime_type='text/plain')
          ),
          types.Part(text='More text'),
      ],
  )

  user_contents = request.append_instructions(content)

  # Text parts should be extracted with references to non-text parts
  expected_system = (
      'Text instruction\n\n'
      '[Reference to inline binary data: inline_data_0 (type: text/plain)]\n\n'
      'More text'
  )
  assert request.config.system_instruction == expected_system

  # Should return user content for the non-text part
  assert len(user_contents) == 1
  assert user_contents[0].role == 'user'
  assert len(user_contents[0].parts) == 2
  assert (
      user_contents[0].parts[0].text == 'Referenced inline data: inline_data_0'
  )
  assert user_contents[0].parts[1].inline_data.data == b'file_data'


def test_append_instructions_content_no_text_parts():
  """Test that Content with no text parts processes non-text parts with references."""
  request = LlmRequest()

  # Set initial system instruction
  request.config.system_instruction = 'Initial'

  # Create Content with only non-text parts
  content = types.Content(
      role='user',
      parts=[
          types.Part(
              inline_data=types.Blob(data=b'file_data', mime_type='text/plain')
          ),
      ],
  )

  user_contents = request.append_instructions(content)

  # Should add reference to non-text part to system instruction
  expected_system = (
      'Initial\n\n[Reference to inline binary data: inline_data_0 (type:'
      ' text/plain)]'
  )
  assert request.config.system_instruction == expected_system

  # Should return user content for the non-text part
  assert len(user_contents) == 1
  assert user_contents[0].role == 'user'
  assert len(user_contents[0].parts) == 2
  assert (
      user_contents[0].parts[0].text == 'Referenced inline data: inline_data_0'
  )
  assert user_contents[0].parts[1].inline_data.data == b'file_data'


def test_append_instructions_content_empty_text_parts():
  """Test that Content with empty text parts are skipped."""
  request = LlmRequest()

  # Create Content with empty and non-empty text parts
  content = types.Content(
      role='user',
      parts=[
          types.Part(text='Valid text'),
          types.Part(text=''),  # Empty text
          types.Part(text=None),  # None text
          types.Part(text='More valid text'),
      ],
  )

  request.append_instructions(content)

  # Only non-empty text should be extracted
  assert request.config.system_instruction == 'Valid text\n\nMore valid text'


def test_append_instructions_warning_unsupported_system_instruction_type(
    caplog,
):
  """Test that warnings are logged for unsupported system_instruction types."""
  import logging

  request = LlmRequest()

  # Set unsupported type as system_instruction
  request.config.system_instruction = {'unsupported': 'dict'}

  with caplog.at_level(logging.WARNING):
    # Try appending Content - should log warning and skip
    content = types.Content(role='user', parts=[types.Part(text='Test')])
    request.append_instructions(content)

    # Should remain unchanged
    assert request.config.system_instruction == {'unsupported': 'dict'}

    # Try appending strings - should also log warning and skip
    request.append_instructions(['Test string'])

    # Should remain unchanged
    assert request.config.system_instruction == {'unsupported': 'dict'}

  # Check that warnings were logged
  assert (
      len(
          [record for record in caplog.records if record.levelname == 'WARNING']
      )
      >= 1
  )
  assert (
      'Cannot append to system_instruction of unsupported type' in caplog.text
  )


@pytest.mark.parametrize('llm_backend', ['GOOGLE_AI', 'VERTEX'])
def test_append_instructions_with_mixed_content(llm_backend):
  """Test append_instructions with mixed text and non-text content."""
  request = LlmRequest()

  # Create static instruction with mixed content
  static_content = types.Content(
      role='user',
      parts=[
          types.Part(text='Analyze this:'),
          types.Part(
              inline_data=types.Blob(
                  data=b'test_data',
                  mime_type='image/png',
                  display_name='test.png',
              )
          ),
          types.Part(text='Focus on details.'),
          types.Part(
              file_data=types.FileData(
                  file_uri='files/doc123',
                  mime_type='text/plain',
                  display_name='document.txt',
              )
          ),
      ],
  )

  user_contents = request.append_instructions(static_content)

  # System instruction should contain text with references
  expected_system = (
      'Analyze this:\n\n[Reference to inline binary data: inline_data_0'
      " ('test.png', type: image/png)]\n\nFocus on details.\n\n[Reference to"
      " file data: file_data_1 ('document.txt', URI: files/doc123, type:"
      ' text/plain)]'
  )
  assert request.config.system_instruction == expected_system

  # Should return user contents for non-text parts
  assert len(user_contents) == 2

  # Check inline_data content
  assert user_contents[0].role == 'user'
  assert len(user_contents[0].parts) == 2
  assert (
      user_contents[0].parts[0].text == 'Referenced inline data: inline_data_0'
  )
  assert user_contents[0].parts[1].inline_data.data == b'test_data'
  assert user_contents[0].parts[1].inline_data.display_name == 'test.png'

  # Check file_data content
  assert user_contents[1].role == 'user'
  assert len(user_contents[1].parts) == 2
  assert user_contents[1].parts[0].text == 'Referenced file data: file_data_1'
  assert user_contents[1].parts[1].file_data.file_uri == 'files/doc123'
  assert user_contents[1].parts[1].file_data.display_name == 'document.txt'


@pytest.mark.parametrize('llm_backend', ['GOOGLE_AI', 'VERTEX'])
def test_append_instructions_with_only_text_parts(llm_backend):
  """Test append_instructions with only text parts."""
  request = LlmRequest()

  static_content = types.Content(
      role='user',
      parts=[
          types.Part(text='First instruction'),
          types.Part(text='Second instruction'),
      ],
  )

  user_contents = request.append_instructions(static_content)

  # Should only have text in system instruction
  assert (
      request.config.system_instruction
      == 'First instruction\n\nSecond instruction'
  )

  # Should return empty list since no non-text parts
  assert user_contents == []
