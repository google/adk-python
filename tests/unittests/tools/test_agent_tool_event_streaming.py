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

"""Unit tests for AgentTool event streaming."""

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.events.event import Event
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.plugins.plugin_manager import PluginManager
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from pytest import mark

from .. import testing_utils


@mark.asyncio
async def test_agent_tool_run_async_with_events_yields_sub_agent_events():
  """Test that run_async_with_events yields events from sub-agent."""
  mock_model = testing_utils.MockModel.create(
      responses=[
          'Step 1: Starting',
          'Step 2: Processing',
          'Step 3: Complete',
      ]
  )

  sub_agent = Agent(
      name='sub_agent',
      model=mock_model,
  )

  agent_tool = AgentTool(agent=sub_agent)

  # Create a minimal tool context
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )

  from google.adk.agents.invocation_context import InvocationContext
  from google.adk.agents.run_config import RunConfig
  from google.adk.plugins.plugin_manager import PluginManager

  invocation_context = InvocationContext(
      artifact_service=InMemoryArtifactService(),
      session_service=session_service,
      memory_service=InMemoryMemoryService(),
      plugin_manager=PluginManager(),
      invocation_id='test_invocation',
      agent=Agent(name='root_agent', model=mock_model),
      session=session,
      run_config=RunConfig(),
  )

  tool_context = ToolContext(invocation_context)

  # Collect events from run_async_with_events
  events = []
  async for event in agent_tool.run_async_with_events(
      args={'request': 'test request'}, tool_context=tool_context
  ):
    events.append(event)

  # Verify events were yielded
  assert len(events) > 0

  # Verify events have content from sub-agent
  text_events = [
      event
      for event in events
      if event.content and any(part.text for part in event.content.parts)
  ]
  assert len(text_events) > 0

  # Verify the events contain expected text
  all_text = ' '.join(
      part.text
      for event in text_events
      for part in (event.content.parts or [])
      if part.text
  )
  assert 'Step' in all_text or 'Complete' in all_text


@mark.asyncio
async def test_agent_tool_run_async_with_events_forwards_state_delta():
  """Test that run_async_with_events forwards state deltas to parent."""

  mock_model = testing_utils.MockModel.create(responses=['Response'])

  def sub_agent_callback(callback_context: CallbackContext):
    """A callback to modify the sub-agent's state."""
    callback_context.state['sub_agent_key'] = 'sub_agent_value'

  sub_agent = Agent(
      name='sub_agent',
      model=mock_model,
      before_agent_callback=sub_agent_callback,
  )

  agent_tool = AgentTool(agent=sub_agent)

  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )

  invocation_context = InvocationContext(
      artifact_service=InMemoryArtifactService(),
      session_service=session_service,
      memory_service=InMemoryMemoryService(),
      plugin_manager=PluginManager(),
      invocation_id='test_invocation',
      agent=Agent(name='root_agent', model=mock_model),
      session=session,
      run_config=RunConfig(),
  )

  tool_context = ToolContext(invocation_context)

  # Set initial state
  tool_context.state['initial_key'] = 'initial_value'

  # Run and collect events, allowing run_async_with_events to update state
  async for _ in agent_tool.run_async_with_events(
      args={'request': 'test'}, tool_context=tool_context
  ):
    pass

  # Verify state was updated by run_async_with_events
  assert 'initial_key' in tool_context.state
  assert tool_context.state['initial_key'] == 'initial_value'
  assert 'sub_agent_key' in tool_context.state
  assert tool_context.state['sub_agent_key'] == 'sub_agent_value'


def test_agent_tool_event_streaming_in_runner():
  """Test that AgentTool event streaming works in a runner context."""
  mock_model = testing_utils.MockModel.create(
      responses=[
          types.Part.from_function_call(
              name='sub_agent', args={'request': 'test'}
          ),
          'Sub-agent step 1',
          'Sub-agent step 2',
          'Sub-agent final response',
          'Root agent final response',
      ]
  )

  sub_agent = Agent(
      name='sub_agent',
      model=mock_model,
  )

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[AgentTool(agent=sub_agent)],
  )

  runner = testing_utils.InMemoryRunner(root_agent)

  # Run and collect all events
  events = list(runner.run('test request'))

  # Verify we got events from both root and sub-agent
  event_authors = [event.author for event in events if event.author]
  assert 'root_agent' in event_authors
  assert 'sub_agent' in event_authors

  # Verify sub-agent events are present (not just the final response)
  sub_agent_events = [
      event
      for event in events
      if event.author == 'sub_agent'
      and event.content
      and any(part.text for part in event.content.parts)
  ]
  # Should have multiple events from sub-agent (not just the function response)
  assert len(sub_agent_events) > 0
