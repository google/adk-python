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

grounding_metadata = types.GroundingMetadata(web_search_queries=['test query'])


# TODO(b/448114567): Remove test_grounding_metadata_ tests once the workaround
# is no longer needed.


@mark.asyncio
async def test_grounding_metadata_is_stored_in_state_during_invocation():
  """Verify grounding_metadata is stored in the state during invocation."""

  # Mock model for the tool_agent that returns grounding_metadata
  tool_agent_model = testing_utils.MockModel.create(
      responses=[
          LlmResponse(
              content=types.Content(
                  parts=[Part.from_text(text='response from tool')]
              ),
              grounding_metadata=grounding_metadata,
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
  tool_result = await agent_tool.run_async(
      args=function_call_no_schema.function_call.args, tool_context=tool_context
  )

  # Verify the tool result
  assert tool_result == 'response from tool'

  # Verify grounding_metadata is stored in the state
  assert tool_context.state['temp:_adk_grounding_metadata'] == (
      grounding_metadata
  )


@mark.asyncio
async def test_grounding_metadata_is_not_stored_in_state_after_invocation():
  """Verify grounding_metadata is not stored in the state after invocation."""

  # Mock model for the tool_agent that returns grounding_metadata
  tool_agent_model = testing_utils.MockModel.create(
      responses=[
          LlmResponse(
              content=types.Content(
                  parts=[Part.from_text(text='response from tool')]
              ),
              grounding_metadata=grounding_metadata,
          )
      ]
  )

  tool_agent = Agent(
      name='tool_agent',
      model=tool_agent_model,
  )

  # Mock model for the root_agent
  root_agent_model = testing_utils.MockModel.create(
      responses=[
          function_call_no_schema,  # Call the tool_agent
          'Final response from root',
      ]
  )

  root_agent = Agent(
      name='root_agent',
      model=root_agent_model,
      tools=[GoogleSearchAgentTool(agent=tool_agent)],
  )

  runner = testing_utils.InMemoryRunner(root_agent)
  events = runner.run('test input')

  # Find the function response event
  function_response_event = None
  for event in events:
    if event.get_function_responses():
      function_response_event = event
      break

  # Verify the function response
  assert function_response_event is not None
  function_responses = function_response_event.get_function_responses()
  assert len(function_responses) == 1
  tool_output = function_responses[0].response
  assert tool_output == {'result': 'response from tool'}

  # Verify grounding_metadata is not stored in the root_agent's state
  assert 'temp:_adk_grounding_metadata' not in runner.session.state
