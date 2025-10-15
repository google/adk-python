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

"""Unit tests for BaseToolset."""

from typing import Optional

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.models.llm_request import LlmRequest
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
import pytest


class _TestingTool(BaseTool):
  """A test implementation of BaseTool."""

  async def run_async(self, *, args, tool_context):
    return 'test result'


class _TestingToolset(BaseToolset):
  """A test implementation of BaseToolset."""

  def __init__(self, *args, tools: Optional[list[BaseTool]] = None, **kwargs):
    super().__init__(*args, **kwargs)
    self._tools = tools or []

  async def get_tools(
      self, readonly_context: Optional[ReadonlyContext] = None
  ) -> list[BaseTool]:
    return self._tools

  async def close(self) -> None:
    pass


@pytest.mark.asyncio
async def test_process_llm_request_default_implementation():
  """Test that the default process_llm_request implementation does nothing."""
  toolset = _TestingToolset()

  # Create test objects
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )
  agent = SequentialAgent(name='test_agent')
  invocation_context = InvocationContext(
      invocation_id='test_id',
      agent=agent,
      session=session,
      session_service=session_service,
  )
  tool_context = ToolContext(invocation_context)
  llm_request = LlmRequest()

  # The default implementation should not modify the request
  original_request = LlmRequest.model_validate(llm_request.model_dump())

  await toolset.process_llm_request(
      tool_context=tool_context, llm_request=llm_request
  )

  # Verify the request was not modified
  assert llm_request.model_dump() == original_request.model_dump()


@pytest.mark.asyncio
async def test_process_llm_request_can_be_overridden():
  """Test that process_llm_request can be overridden by subclasses."""

  class _CustomToolset(_TestingToolset):

    async def process_llm_request(
        self, *, tool_context: ToolContext, llm_request: LlmRequest
    ) -> None:
      # Add some custom processing
      if not llm_request.contents:
        llm_request.contents = []
      llm_request.contents.append('Custom processing applied')

  toolset = _CustomToolset()

  # Create test objects
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )
  agent = SequentialAgent(name='test_agent')
  invocation_context = InvocationContext(
      invocation_id='test_id',
      agent=agent,
      session=session,
      session_service=session_service,
  )
  tool_context = ToolContext(invocation_context)
  llm_request = LlmRequest()

  await toolset.process_llm_request(
      tool_context=tool_context, llm_request=llm_request
  )

  # Verify the custom processing was applied
  assert llm_request.contents == ['Custom processing applied']


@pytest.mark.asyncio
async def test_prefix_functionality_disabled_by_default():
  """Test that prefix functionality is disabled by default."""
  tool1 = _TestingTool(name='tool1', description='Test tool 1')
  tool2 = _TestingTool(name='tool2', description='Test tool 2')
  toolset = _TestingToolset(tools=[tool1, tool2])

  # When tool_name_prefix is None (default), get_tools_with_prefix should return original tools
  prefixed_tools = await toolset.get_tools_with_prefix()

  assert len(prefixed_tools) == 2
  assert prefixed_tools[0].name == 'tool1'
  assert prefixed_tools[1].name == 'tool2'
  assert toolset.tool_name_prefix is None


@pytest.mark.asyncio
async def test_prefix_functionality_with_custom_prefix():
  """Test prefix functionality with custom prefix."""
  tool1 = _TestingTool(name='tool1', description='Test tool 1')
  tool2 = _TestingTool(name='tool2', description='Test tool 2')
  toolset = _TestingToolset(tools=[tool1, tool2], tool_name_prefix='custom')

  # Should use the provided prefix
  prefixed_tools = await toolset.get_tools_with_prefix()

  assert len(prefixed_tools) == 2
  assert prefixed_tools[0].name == 'custom_tool1'
  assert prefixed_tools[1].name == 'custom_tool2'
  assert toolset.tool_name_prefix == 'custom'


@pytest.mark.asyncio
async def test_prefix_with_none_has_no_effect():
  """Test that when prefix is None, tools are returned unchanged."""
  tool1 = _TestingTool(name='tool1', description='Test tool 1')
  tool2 = _TestingTool(name='tool2', description='Test tool 2')
  toolset = _TestingToolset(tools=[tool1, tool2], tool_name_prefix=None)

  prefixed_tools = await toolset.get_tools_with_prefix()

  assert len(prefixed_tools) == 2
  assert prefixed_tools[0].name == 'tool1'
  assert prefixed_tools[1].name == 'tool2'
  assert toolset.tool_name_prefix is None


@pytest.mark.asyncio
async def test_prefix_with_empty_string():
  """Test prefix functionality with empty string prefix."""
  tool1 = _TestingTool(name='tool1', description='Test tool 1')
  toolset = _TestingToolset(tools=[tool1], tool_name_prefix='')

  prefixed_tools = await toolset.get_tools_with_prefix()

  # Empty prefix should be treated as no prefix
  assert len(prefixed_tools) == 1
  assert prefixed_tools[0].name == 'tool1'
  assert toolset.tool_name_prefix == ''


@pytest.mark.asyncio
async def test_prefix_assignment():
  """Test that prefix is properly assigned."""
  toolset = _TestingToolset(tool_name_prefix='explicit')
  assert toolset.tool_name_prefix == 'explicit'

  # Test None assignment
  toolset_none = _TestingToolset(tool_name_prefix=None)
  assert toolset_none.tool_name_prefix is None


@pytest.mark.asyncio
async def test_prefix_creates_tool_copies():
  """Test that prefixing creates copies and preserves original tools."""
  original_tool = _TestingTool(
      name='original', description='Original description'
  )
  original_tool.is_long_running = True
  original_tool.custom_attribute = 'custom_value'

  toolset = _TestingToolset(tools=[original_tool], tool_name_prefix='test')
  prefixed_tools = await toolset.get_tools_with_prefix()

  prefixed_tool = prefixed_tools[0]

  # Name should be prefixed in the copy
  assert prefixed_tool.name == 'test_original'

  # Other attributes should be preserved
  assert prefixed_tool.description == 'Original description'
  assert prefixed_tool.is_long_running == True
  assert prefixed_tool.custom_attribute == 'custom_value'

  # Original tool should remain unchanged
  assert original_tool.name == 'original'
  assert original_tool is not prefixed_tool


@pytest.mark.asyncio
async def test_get_tools_vs_get_tools_with_prefix():
  """Test that get_tools returns tools without prefixing."""
  tool1 = _TestingTool(name='test_tool1', description='Test tool 1')
  tool2 = _TestingTool(name='test_tool2', description='Test tool 2')
  toolset = _TestingToolset(tools=[tool1, tool2], tool_name_prefix='prefix')

  # get_tools should return original tools (unmodified)
  original_tools = await toolset.get_tools()
  assert len(original_tools) == 2
  assert original_tools[0].name == 'test_tool1'
  assert original_tools[1].name == 'test_tool2'

  # Now calling get_tools_with_prefix should return prefixed copies
  prefixed_tools = await toolset.get_tools_with_prefix()
  assert len(prefixed_tools) == 2
  assert prefixed_tools[0].name == 'prefix_test_tool1'
  assert prefixed_tools[1].name == 'prefix_test_tool2'

  # Original tools should remain unchanged
  assert original_tools[0].name == 'test_tool1'
  assert original_tools[1].name == 'test_tool2'

  # The prefixed tools should be different instances
  assert prefixed_tools[0] is not original_tools[0]
  assert prefixed_tools[1] is not original_tools[1]


@pytest.mark.asyncio
async def test_empty_toolset_with_prefix():
  """Test prefix functionality with empty toolset."""
  toolset = _TestingToolset(tools=[], tool_name_prefix='test')

  prefixed_tools = await toolset.get_tools_with_prefix()
  assert len(prefixed_tools) == 0


@pytest.mark.asyncio
async def test_function_declarations_are_prefixed():
  """Test that function declarations have prefixed names."""

  def test_function(param1: str, param2: int) -> str:
    """A test function for checking prefixes."""
    return f'{param1}_{param2}'

  function_tool = FunctionTool(test_function)
  toolset = _TestingToolset(
      tools=[function_tool],
      tool_name_prefix='prefix',
  )

  prefixed_tools = await toolset.get_tools_with_prefix()
  prefixed_tool = prefixed_tools[0]

  # Tool name should be prefixed
  assert prefixed_tool.name == 'prefix_test_function'

  # Function declaration should also have prefixed name
  declaration = prefixed_tool._get_declaration()
  assert declaration is not None
  assert declaration.name == 'prefix_test_function'

  # Description should remain unchanged
  assert 'A test function for checking prefixes.' in declaration.description


@pytest.mark.asyncio
async def test_prefixed_tools_in_llm_request():
  """Test that prefixed tools are properly added to LLM request."""

  def test_function(param: str) -> str:
    """A test function."""
    return f'result: {param}'

  function_tool = FunctionTool(test_function)
  toolset = _TestingToolset(tools=[function_tool], tool_name_prefix='test')

  prefixed_tools = await toolset.get_tools_with_prefix()
  prefixed_tool = prefixed_tools[0]

  # Create LLM request and tool context
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )
  agent = SequentialAgent(name='test_agent')
  invocation_context = InvocationContext(
      invocation_id='test_id',
      agent=agent,
      session=session,
      session_service=session_service,
  )
  tool_context = ToolContext(invocation_context)
  llm_request = LlmRequest()

  # Process the LLM request with the prefixed tool
  await prefixed_tool.process_llm_request(
      tool_context=tool_context, llm_request=llm_request
  )

  # Verify the tool is registered with prefixed name in tools_dict
  assert 'test_test_function' in llm_request.tools_dict
  assert llm_request.tools_dict['test_test_function'] == prefixed_tool

  # Verify the function declaration has prefixed name
  assert llm_request.config is not None
  assert llm_request.config.tools is not None
  assert len(llm_request.config.tools) == 1
  tool_config = llm_request.config.tools[0]
  assert len(tool_config.function_declarations) == 1
  func_decl = tool_config.function_declarations[0]
  assert func_decl.name == 'test_test_function'


@pytest.mark.asyncio
async def test_multiple_tools_have_correct_declarations():
  """Test that each tool maintains its own function declaration after prefixing."""

  def tool_one(param: str) -> str:
    """Function one."""
    return f'one: {param}'

  def tool_two(param: int) -> str:
    """Function two."""
    return f'two: {param}'

  tool1 = FunctionTool(tool_one)
  tool2 = FunctionTool(tool_two)
  toolset = _TestingToolset(tools=[tool1, tool2], tool_name_prefix='test')

  prefixed_tools = await toolset.get_tools_with_prefix()

  # Verify each tool has its own correct declaration
  decl1 = prefixed_tools[0]._get_declaration()
  decl2 = prefixed_tools[1]._get_declaration()

  assert decl1.name == 'test_tool_one'
  assert decl2.name == 'test_tool_two'

  assert 'Function one.' in decl1.description
  assert 'Function two.' in decl2.description


@pytest.mark.asyncio
async def test_no_duplicate_prefixing():
  """Test that multiple calls to get_tools_with_prefix don't cause duplicate prefixing."""
  original_tool = _TestingTool(name='original', description='Original tool')
  toolset = _TestingToolset(tools=[original_tool], tool_name_prefix='test')

  # First call
  prefixed_tools_1 = await toolset.get_tools_with_prefix()
  assert len(prefixed_tools_1) == 1
  assert prefixed_tools_1[0].name == 'test_original'

  # Second call - should not double-prefix
  prefixed_tools_2 = await toolset.get_tools_with_prefix()
  assert len(prefixed_tools_2) == 1
  assert prefixed_tools_2[0].name == 'test_original'  # Not 'test_test_original'

  # Original tool should remain unchanged
  original_tools = await toolset.get_tools()
  assert original_tools[0].name == 'original'

  # The prefixed tools should be different instances
  assert prefixed_tools_1[0] is not prefixed_tools_2[0]
  assert prefixed_tools_1[0] is not original_tools[0]
