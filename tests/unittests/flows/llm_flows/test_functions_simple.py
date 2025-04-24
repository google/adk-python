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

from typing import Any
from typing import AsyncGenerator
from typing import Callable

from google.adk.agents import Agent
from google.adk.tools import ToolContext
from google.adk.tools.function_tool import FunctionTool
from google.genai import types
import pytest

from ... import utils


def test_simple_function():
  function_call_1 = types.Part.from_function_call(
      name='increase_by_one', args={'x': 1}
  )
  function_respones_2 = types.Part.from_function_response(
      name='increase_by_one', response={'result': 2}
  )
  responses: list[types.Content] = [
      function_call_1,
      'response1',
      'response2',
      'response3',
      'response4',
  ]
  function_called = 0
  mock_model = utils.MockModel.create(responses=responses)

  def increase_by_one(x: int) -> int:
    nonlocal function_called
    function_called += 1
    return x + 1

  agent = Agent(name='root_agent', model=mock_model, tools=[increase_by_one])
  runner = utils.InMemoryRunner(agent)
  assert utils.simplify_events(runner.run('test')) == [
      ('root_agent', function_call_1),
      ('root_agent', function_respones_2),
      ('root_agent', 'response1'),
  ]

  # Asserts the requests.
  assert utils.simplify_contents(mock_model.requests[0].contents) == [
      ('user', 'test')
  ]
  assert utils.simplify_contents(mock_model.requests[1].contents) == [
      ('user', 'test'),
      ('model', function_call_1),
      ('user', function_respones_2),
  ]

  # Asserts the function calls.
  assert function_called == 1


@pytest.mark.asyncio
async def test_async_function():
  function_calls = [
      types.Part.from_function_call(name='increase_by_one', args={'x': 1}),
      types.Part.from_function_call(name='multiple_by_two', args={'x': 2}),
      types.Part.from_function_call(name='multiple_by_two_sync', args={'x': 3}),
  ]
  function_responses = [
      types.Part.from_function_response(
          name='increase_by_one', response={'result': 2}
      ),
      types.Part.from_function_response(
          name='multiple_by_two', response={'result': 4}
      ),
      types.Part.from_function_response(
          name='multiple_by_two_sync', response={'result': 6}
      ),
  ]

  responses: list[types.Content] = [
      function_calls,
      'response1',
      'response2',
      'response3',
      'response4',
  ]
  function_called = 0
  mock_model = utils.MockModel.create(responses=responses)

  async def increase_by_one(x: int) -> int:
    nonlocal function_called
    function_called += 1
    return x + 1

  async def multiple_by_two(x: int) -> int:
    nonlocal function_called
    function_called += 1
    return x * 2

  def multiple_by_two_sync(x: int) -> int:
    nonlocal function_called
    function_called += 1
    return x * 2

  agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[increase_by_one, multiple_by_two, multiple_by_two_sync],
  )
  runner = utils.TestInMemoryRunner(agent)
  events = await runner.run_async_with_new_session('test')
  assert utils.simplify_events(events) == [
      ('root_agent', function_calls),
      ('root_agent', function_responses),
      ('root_agent', 'response1'),
  ]

  # Asserts the requests.
  assert utils.simplify_contents(mock_model.requests[0].contents) == [
      ('user', 'test')
  ]
  assert utils.simplify_contents(mock_model.requests[1].contents) == [
      ('user', 'test'),
      ('model', function_calls),
      ('user', function_responses),
  ]

  # Asserts the function calls.
  assert function_called == 3


@pytest.mark.asyncio
async def test_function_tool():
  function_calls = [
      types.Part.from_function_call(name='increase_by_one', args={'x': 1}),
      types.Part.from_function_call(name='multiple_by_two', args={'x': 2}),
      types.Part.from_function_call(name='multiple_by_two_sync', args={'x': 3}),
  ]
  function_responses = [
      types.Part.from_function_response(
          name='increase_by_one', response={'result': 2}
      ),
      types.Part.from_function_response(
          name='multiple_by_two', response={'result': 4}
      ),
      types.Part.from_function_response(
          name='multiple_by_two_sync', response={'result': 6}
      ),
  ]

  responses: list[types.Content] = [
      function_calls,
      'response1',
      'response2',
      'response3',
      'response4',
  ]
  function_called = 0
  mock_model = utils.MockModel.create(responses=responses)

  async def increase_by_one(x: int) -> int:
    nonlocal function_called
    function_called += 1
    return x + 1

  async def multiple_by_two(x: int) -> int:
    nonlocal function_called
    function_called += 1
    return x * 2

  def multiple_by_two_sync(x: int) -> int:
    nonlocal function_called
    function_called += 1
    return x * 2

  class TestTool(FunctionTool):

    def __init__(self, func: Callable[..., Any]):
      super().__init__(func=func)

  wrapped_increase_by_one = TestTool(func=increase_by_one)
  agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[wrapped_increase_by_one, multiple_by_two, multiple_by_two_sync],
  )
  runner = utils.TestInMemoryRunner(agent)
  events = await runner.run_async_with_new_session('test')
  assert utils.simplify_events(events) == [
      ('root_agent', function_calls),
      ('root_agent', function_responses),
      ('root_agent', 'response1'),
  ]

  # Asserts the requests.
  assert utils.simplify_contents(mock_model.requests[0].contents) == [
      ('user', 'test')
  ]
  assert utils.simplify_contents(mock_model.requests[1].contents) == [
      ('user', 'test'),
      ('model', function_calls),
      ('user', function_responses),
  ]

  # Asserts the function calls.
  assert function_called == 3


def test_update_state():
  mock_model = utils.MockModel.create(
      responses=[
          types.Part.from_function_call(name='update_state', args={}),
          'response1',
      ]
  )

  def update_state(tool_context: ToolContext):
    tool_context.state['x'] = 1

  agent = Agent(name='root_agent', model=mock_model, tools=[update_state])
  runner = utils.InMemoryRunner(agent)
  runner.run('test')
  assert runner.session.state['x'] == 1


@pytest.mark.asyncio
async def test_mcp_tool_in_live_mode_raises_error():
    """Tests that using MCPTool in live mode raises NotImplementedError."""
    # Import the MCPTool and McpBaseTool classes.
    from google.adk.tools.mcp_tool import MCPTool
    from mcp.types import Tool as McpBaseTool
    from google.adk.flows.llm_flows.functions import _process_function_live_helper
    from google.adk.tools import ToolContext
    
    # Create a minimal MCPTool instance.
    mock_mcp_base_tool = McpBaseTool(name="mock_mcp_tool", description="A mock MCP tool", inputSchema={})
    
    class MockMCPSession:
        async def call_tool(self, name, arguments):
            return {"result": "mock_result"}

    class MockMCPSessionManager:
        pass

    mcp_tool = MCPTool(
        mcp_tool=mock_mcp_base_tool,
        mcp_session=MockMCPSession(),
        mcp_session_manager=MockMCPSessionManager()
    )

    # Create a Mock Agent for InvocationContext.
    mock_agent = Agent(name="mock_agent", model="mock_model")
    
    # Use utils.create_invocation_context to create InvocationContext.
    invocation_context = utils.create_invocation_context(agent=mock_agent)
    
    # Create ToolContext.
    tool_context = ToolContext(invocation_context)
    function_call = {"name": "mock_mcp_tool", "arguments": {}}
    
    # Assert that the expected exception is raised.
    with pytest.raises(
        NotImplementedError,
        match="MCPTool is not yet supported in live/streaming mode."
    ):
        await _process_function_live_helper(
            tool=mcp_tool,
            tool_context=tool_context,
            function_call=function_call,
            function_args={},
            invocation_context=invocation_context
        )


def test_function_call_id():
  responses = [
      types.Part.from_function_call(name='increase_by_one', args={'x': 1}),
      'response1',
  ]
  mock_model = utils.MockModel.create(responses=responses)

  def increase_by_one(x: int) -> int:
    return x + 1

  agent = Agent(name='root_agent', model=mock_model, tools=[increase_by_one])
  runner = utils.InMemoryRunner(agent)
  events = runner.run('test')
  for reqeust in mock_model.requests:
    for content in reqeust.contents:
      for part in content.parts:
        if part.function_call:
          assert part.function_call.id is None
        if part.function_response:
          assert part.function_response.id is None
  assert events[0].content.parts[0].function_call.id.startswith('adk-')
  assert events[1].content.parts[0].function_response.id.startswith('adk-')
