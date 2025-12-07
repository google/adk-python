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

import pytest
from google.genai import types
from google.adk.events.event import Event
from google.adk.tools.function_tool import FunctionTool
from google.adk.agents.llm_agent import Agent
from google.adk.flows.llm_flows.functions import handle_function_calls_async
from ... import testing_utils

@pytest.mark.asyncio
async def test_function_call_with_skip_synthesis():
  """Test that skip_synthesis is propagated to the response event."""
  
  def simple_fn(**kwargs) -> dict:
    return {'result': 'test'}

  # Create tool with skip_synthesis=True
  tool = FunctionTool(simple_fn, skip_synthesis=True)
  
  model = testing_utils.MockModel.create(responses=[])
  agent = Agent(
      name='test_agent',
      model=model,
      tools=[tool],
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content=''
  )

  function_call = types.FunctionCall(name=tool.name, args={})
  content = types.Content(parts=[types.Part(function_call=function_call)])
  event = Event(
      invocation_id=invocation_context.invocation_id,
      author=agent.name,
      content=content,
  )
  tools_dict = {tool.name: tool}

  # Execute the function call
  result_event = await handle_function_calls_async(
      invocation_context,
      event,
      tools_dict,
  )

  # Verify that the resulting event has SKIP_SYNTHESIS
  assert result_event is not None
  assert result_event.actions is not None
  assert result_event.actions.skip_synthesis is True
