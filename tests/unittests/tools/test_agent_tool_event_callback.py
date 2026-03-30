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

"""Tests for AgentTool event_callback functionality."""

from google.adk.agents.llm_agent import Agent
from google.adk.events.event import Event
from google.adk.tools.agent_tool import AgentTool
from google.genai.types import Part
from pytest import mark

from .. import testing_utils


@mark.asyncio
async def test_event_callback_sync_invocation():
  """Test that synchronous event callbacks are invoked correctly."""
  captured_events = []

  def sync_callback(event: Event) -> None:
    captured_events.append(event)

  function_call = Part.from_function_call(
      name='tool_agent', args={'request': 'test1'}
  )

  mock_model = testing_utils.MockModel.create(
      responses=[
          function_call,
          'response1',
          'response2',
      ]
  )

  tool_agent = Agent(
      name='tool_agent',
      model=mock_model,
  )

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[AgentTool(agent=tool_agent, event_callback=sync_callback)],
  )

  runner = testing_utils.InMemoryRunner(root_agent)
  runner.run('test1')

  # Verify that events were captured
  assert len(captured_events) > 0
  # All captured items should be Event instances
  assert all(isinstance(e, Event) for e in captured_events)
  # Should capture the tool agent's response
  assert any(
      e.content and any(p.text == 'response1' for p in e.content.parts)
      for e in captured_events
  )


@mark.asyncio
async def test_event_callback_async_invocation():
  """Test that asynchronous event callbacks are invoked correctly."""
  captured_events = []

  async def async_callback(event: Event) -> None:
    captured_events.append(event)

  function_call = Part.from_function_call(
      name='tool_agent', args={'request': 'test1'}
  )

  mock_model = testing_utils.MockModel.create(
      responses=[
          function_call,
          'response1',
          'response2',
      ]
  )

  tool_agent = Agent(
      name='tool_agent',
      model=mock_model,
  )

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[AgentTool(agent=tool_agent, event_callback=async_callback)],
  )

  runner = testing_utils.InMemoryRunner(root_agent)
  runner.run('test1')

  # Verify that events were captured
  assert len(captured_events) > 0
  # All captured items should be Event instances
  assert all(isinstance(e, Event) for e in captured_events)
  # Should capture the tool agent's response
  assert any(
      e.content and any(p.text == 'response1' for p in e.content.parts)
      for e in captured_events
  )


@mark.asyncio
async def test_event_callback_receives_all_events():
  """Test that callbacks receive all child agent events."""
  captured_events = []

  def capture_callback(event: Event) -> None:
    captured_events.append(event)

  function_call = Part.from_function_call(
      name='tool_agent', args={'request': 'test1'}
  )

  mock_model = testing_utils.MockModel.create(
      responses=[
          function_call,
          'response1',
          'response2',
      ]
  )

  tool_agent = Agent(
      name='tool_agent',
      model=mock_model,
  )

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[AgentTool(agent=tool_agent, event_callback=capture_callback)],
  )

  runner = testing_utils.InMemoryRunner(root_agent)
  runner.run('test1')

  # Verify multiple events were captured (should include at least response)
  assert len(captured_events) >= 1

  # Check that events have expected structure
  for event in captured_events:
    assert isinstance(event, Event)
    assert hasattr(event, 'author')
    assert hasattr(event, 'content')
    assert hasattr(event, 'actions')


@mark.asyncio
async def test_event_callback_backward_compatibility():
  """Test AgentTool works without event_callback (backward compatibility)."""
  function_call = Part.from_function_call(
      name='tool_agent', args={'request': 'test1'}
  )

  function_response = Part.from_function_response(
      name='tool_agent', response={'result': 'response1'}
  )

  mock_model = testing_utils.MockModel.create(
      responses=[
          function_call,
          'response1',
          'response2',
      ]
  )

  tool_agent = Agent(
      name='tool_agent',
      model=mock_model,
  )

  # Create AgentTool without event_callback parameter
  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[AgentTool(agent=tool_agent)],
  )

  runner = testing_utils.InMemoryRunner(root_agent)

  # Should work without errors
  result = testing_utils.simplify_events(runner.run('test1'))

  assert result == [
      ('root_agent', function_call),
      ('root_agent', function_response),
      ('root_agent', 'response2'),
  ]


@mark.asyncio
async def test_event_callback_can_access_event_metadata():
  """Test that callbacks can access event metadata like grounding_metadata."""
  captured_metadata = []

  def metadata_callback(event: Event) -> None:
    if event.grounding_metadata:
      captured_metadata.append(event.grounding_metadata)

  function_call = Part.from_function_call(
      name='tool_agent', args={'request': 'test1'}
  )

  mock_model = testing_utils.MockModel.create(
      responses=[
          function_call,
          'response1',
          'response2',
      ]
  )

  tool_agent = Agent(
      name='tool_agent',
      model=mock_model,
  )

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[AgentTool(agent=tool_agent, event_callback=metadata_callback)],
  )

  runner = testing_utils.InMemoryRunner(root_agent)
  runner.run('test1')

  # Test passes if no errors occur (grounding_metadata access works)
  # Note: captured_metadata may be empty if mock doesn't provide metadata


@mark.asyncio
async def test_event_callback_with_multiple_tool_calls():
  """Test that callbacks work correctly with multiple tool invocations."""
  captured_events = []

  def capture_callback(event: Event) -> None:
    captured_events.append(event)

  function_call_1 = Part.from_function_call(
      name='tool_agent', args={'request': 'call1'}
  )
  function_call_2 = Part.from_function_call(
      name='tool_agent', args={'request': 'call2'}
  )

  mock_model = testing_utils.MockModel.create(
      responses=[
          function_call_1,
          'response1',
          function_call_2,
          'response2',
          'final',
      ]
  )

  tool_agent = Agent(
      name='tool_agent',
      model=mock_model,
  )

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[AgentTool(agent=tool_agent, event_callback=capture_callback)],
  )

  runner = testing_utils.InMemoryRunner(root_agent)
  runner.run('test1')

  # Should capture events from both tool invocations
  assert len(captured_events) >= 2

  # Verify we got responses from both calls
  response_texts = []
  for event in captured_events:
    if event.content:
      for part in event.content.parts:
        if part.text:
          response_texts.append(part.text)

  assert 'response1' in response_texts
  assert 'response2' in response_texts
