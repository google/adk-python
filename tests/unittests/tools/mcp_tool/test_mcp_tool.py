from google.adk import Agent
from google.adk.tools import FunctionTool
from google.adk.tools.mcp_tool import MCPTool
from tests.unittests import testing_utils
from google.genai.types import Part
from mcp.types import Tool as McpBaseTool


class TestMCPSession(object):

  def __init__(self, function_tool: FunctionTool):
    self._function_tool = function_tool

  async def call_tool(self, name, arguments):
    return self._function_tool.func(**arguments)


class TestMCPSessionManager(object):

  def __init__(self, function_tool: FunctionTool):
    self._function_tool = function_tool

  async def create_session(self):
    return TestMCPSession(self._function_tool)

  async def close(self):
    pass


def mcp_tool(function_tool: FunctionTool, prefix=''):
  return MCPTool(
      mcp_tool=McpBaseTool(
          name=function_tool.name,
          description=function_tool.description,
          inputSchema=function_tool._get_declaration().parameters.json_schema.model_dump(
              exclude_none=True
          ),
      ),
      mcp_session_manager=TestMCPSessionManager(function_tool),
      tool_name_prefix=prefix,
  )


def test_mcp_tool():
  @FunctionTool
  def add(a: int, b: int):
    """Add a and b and  retuirn the result"""
    return a + b

  mcp_add = mcp_tool(add, 'mcp_')

  add_call = Part.from_function_call(name='add', args={'a': 1, 'b': 2})
  add_response = Part.from_function_response(name='add', response={'result': 3})

  mcp_add_call = Part.from_function_call(name='mcp_add', args={'a': 5, 'b': 10})
  mcp_add_response = Part.from_function_response(
      name='mcp_add', response={'result': 15}
  )

  mock_model = testing_utils.MockModel.create(
      responses=[
          add_call,
          mcp_add_call,
          'response1',
      ]
  )

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[add, mcp_add],
  )

  runner = testing_utils.InMemoryRunner(root_agent)

  assert testing_utils.simplify_events(runner.run('test1')) == [
      ('root_agent', add_call),
      ('root_agent', add_response),
      ('root_agent', mcp_add_call),
      ('root_agent', mcp_add_response),
      ('root_agent', 'response1'),
  ]
