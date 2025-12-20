from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import Agent
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.google_search_agent_tool import GoogleSearchAgentTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from google.genai.types import Part
from pytest import mark

from .. import testing_utils

function_call_no_schema = Part.from_function_call(
    name='tool_agent', args={'request': 'test1'}
)


@mark.asyncio
async def test_run_async_handles_none_parts_in_response():
  """Verify run_async handles None parts in the response without crashing."""

  # Mock model for the tool_agent that returns content with parts=None
  # This simulates the condition causing the TypeError
  tool_agent_model = testing_utils.MockModel.create(
      responses=[
          LlmResponse(
              content=types.Content(parts=None),
          )
      ]
  )

  tool_agent = Agent(
      name='tool_agent',
      model=tool_agent_model,
  )

  agent_tool = GoogleSearchAgentTool(agent=tool_agent)

  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )

  invocation_context = InvocationContext(
      invocation_id='invocation_id',
      agent=tool_agent,
      session=session,
      session_service=session_service,
  )
  tool_context = ToolContext(invocation_context=invocation_context)

  # This should not raise TypeError: 'NoneType' object is not iterable
  # It should ideally return an empty string or handle it gracefully
  tool_result = await agent_tool.run_async(
      args=function_call_no_schema.function_call.args, tool_context=tool_context
  )

  assert tool_result == ''
