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

"""Behavioral tests for other agent message processing in contents module."""

from google.adk.agents.llm_agent import Agent
from google.adk.events.event import Event
from google.adk.flows.llm_flows.contents import request_processor
from google.adk.models.llm_request import LlmRequest
from google.genai import types
import pytest

from ... import testing_utils


@pytest.mark.asyncio
async def test_other_agent_message_appears_as_user_context():
  """Test that messages from other agents appear as user context."""
  agent = Agent(model="gemini-2.5-flash", name="current_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  # Add event from another agent
  other_agent_event = Event(
      invocation_id="test_inv",
      author="other_agent",
      content=types.ModelContent("Hello from other agent"),
  )
  invocation_context.session.events = [other_agent_event]

  # Process the request
  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  # Verify the other agent's message is presented as user context
  assert llm_request.contents[0].role == "user"
  assert llm_request.contents[0].parts == [
      types.Part(text="For context:"),
      types.Part(text="[other_agent] said: Hello from other agent"),
  ]


@pytest.mark.asyncio
async def test_other_agent_thoughts_are_excluded():
  """Test that thoughts from other agents are excluded from context."""
  agent = Agent(model="gemini-2.5-flash", name="current_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  # Add event from other agent with both regular text and thoughts
  other_agent_event = Event(
      invocation_id="test_inv",
      author="other_agent",
      content=types.ModelContent([
          types.Part(text="Public message", thought=False),
          types.Part(text="Private thought", thought=True),
          types.Part(text="Another public message"),
      ]),
  )
  invocation_context.session.events = [other_agent_event]

  # Process the request
  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  # Verify only non-thought parts are included (thoughts excluded)
  assert llm_request.contents[0].role == "user"
  assert llm_request.contents[0].parts == [
      types.Part(text="For context:"),
      types.Part(text="[other_agent] said: Public message"),
      types.Part(text="[other_agent] said: Another public message"),
  ]


@pytest.mark.asyncio
async def test_other_agent_function_calls():
  """Test that function calls from other agents are preserved in context."""
  agent = Agent(model="gemini-2.5-flash", name="current_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  # Add event from other agent with function call
  function_call = types.FunctionCall(
      id="func_123", name="search_tool", args={"query": "test query"}
  )
  other_agent_event = Event(
      invocation_id="test_inv",
      author="other_agent",
      content=types.ModelContent([types.Part(function_call=function_call)]),
  )
  invocation_context.session.events = [other_agent_event]

  # Process the request
  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  # Verify function call is presented as context
  assert llm_request.contents[0].role == "user"
  assert llm_request.contents[0].parts == [
      types.Part(text="For context:"),
      types.Part(
          text="""\
[other_agent] called tool `search_tool` with parameters: {'query': 'test query'}"""
      ),
  ]


@pytest.mark.asyncio
async def test_other_agent_function_responses():
  """Test that function responses from other agents are properly formatted."""
  agent = Agent(model="gemini-2.5-flash", name="current_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  # Add event from other agent with function response
  function_response = types.FunctionResponse(
      id="func_123",
      name="search_tool",
      response={"results": ["item1", "item2"]},
  )
  other_agent_event = Event(
      invocation_id="test_inv",
      author="other_agent",
      content=types.Content(
          role="user", parts=[types.Part(function_response=function_response)]
      ),
  )
  invocation_context.session.events = [other_agent_event]

  # Process the request
  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  # Verify function response is presented as context
  assert llm_request.contents[0].role == "user"
  assert llm_request.contents[0].parts == [
      types.Part(text="For context:"),
      types.Part(
          text=(
              "[other_agent] `search_tool` tool returned result: {'results':"
              " ['item1', 'item2']}"
          )
      ),
  ]


@pytest.mark.asyncio
async def test_other_agent_function_call_response():
  """Test function call and response sequence from other agents."""
  agent = Agent(model="gemini-2.5-flash", name="current_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  # Add function call event from other agent
  function_call = types.FunctionCall(
      id="func_123", name="calc_tool", args={"query": "6x7"}
  )
  call_event = Event(
      invocation_id="test_inv1",
      author="other_agent",
      content=types.ModelContent([
          types.Part(text="Let me calculate this"),
          types.Part(function_call=function_call),
      ]),
  )
  # Add function response event
  function_response = types.FunctionResponse(
      id="func_123", name="calc_tool", response={"result": 42}
  )
  response_event = Event(
      invocation_id="test_inv2",
      author="other_agent",
      content=types.UserContent(
          parts=[types.Part(function_response=function_response)]
      ),
  )
  invocation_context.session.events = [call_event, response_event]

  # Process the request
  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  # Verify function call and response are properly formatted
  assert len(llm_request.contents) == 2

  # Function call from other agent
  assert llm_request.contents[0].role == "user"
  assert llm_request.contents[0].parts == [
      types.Part(text="For context:"),
      types.Part(text="[other_agent] said: Let me calculate this"),
      types.Part(
          text=(
              "[other_agent] called tool `calc_tool` with parameters: {'query':"
              " '6x7'}"
          )
      ),
  ]
  # Function response from other agent
  assert llm_request.contents[1].role == "user"
  assert llm_request.contents[1].parts == [
      types.Part(text="For context:"),
      types.Part(
          text="[other_agent] `calc_tool` tool returned result: {'result': 42}"
      ),
  ]


@pytest.mark.asyncio
async def test_other_agent_empty_content():
  """Test that other agent messages with only thoughts or empty content are filtered out."""
  agent = Agent(model="gemini-2.5-flash", name="current_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  # Add events: user message, other agents with empty content, user message
  events = [
      Event(
          invocation_id="inv1",
          author="user",
          content=types.UserContent("Hello"),
      ),
      # Other agent with only thoughts
      Event(
          invocation_id="inv2",
          author="other_agent1",
          content=types.ModelContent([
              types.Part(text="This is a private thought", thought=True),
              types.Part(text="Another private thought", thought=True),
          ]),
      ),
      # Other agent with empty text and thoughts
      Event(
          invocation_id="inv3",
          author="other_agent2",
          content=types.ModelContent([
              types.Part(text="", thought=False),
              types.Part(text="Secret thought", thought=True),
          ]),
      ),
      Event(
          invocation_id="inv4",
          author="user",
          content=types.UserContent("World"),
      ),
  ]
  invocation_context.session.events = events

  # Process the request
  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  # Verify empty content events are completely filtered out
  assert llm_request.contents == [
      types.UserContent("Hello"),
      types.UserContent("World"),
  ]


@pytest.mark.asyncio
async def test_multiple_agents_in_conversation():
  """Test handling multiple agents in a conversation flow."""
  agent = Agent(model="gemini-2.5-flash", name="current_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  # Create a multi-agent conversation
  events = [
      Event(
          invocation_id="inv1",
          author="user",
          content=types.UserContent("Hello everyone"),
      ),
      Event(
          invocation_id="inv2",
          author="agent1",
          content=types.ModelContent("Hi from agent1"),
      ),
      Event(
          invocation_id="inv3",
          author="agent2",
          content=types.ModelContent("Hi from agent2"),
      ),
  ]
  invocation_context.session.events = events

  # Process the request
  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  # Verify all messages are properly processed
  assert len(llm_request.contents) == 3

  # User message should remain as user
  assert llm_request.contents[0] == types.UserContent("Hello everyone")
  # Other agents' messages should be converted to user context
  assert llm_request.contents[1].role == "user"
  assert llm_request.contents[1].parts == [
      types.Part(text="For context:"),
      types.Part(text="[agent1] said: Hi from agent1"),
  ]
  assert llm_request.contents[2].role == "user"
  assert llm_request.contents[2].parts == [
      types.Part(text="For context:"),
      types.Part(text="[agent2] said: Hi from agent2"),
  ]


@pytest.mark.asyncio
async def test_current_agent_messages_not_converted():
  """Test that the current agent's own messages are not converted."""
  agent = Agent(model="gemini-2.5-flash", name="current_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  # Add events from both current agent and other agent
  events = [
      Event(
          invocation_id="inv1",
          author="current_agent",
          content=types.ModelContent("My own message"),
      ),
      Event(
          invocation_id="inv2",
          author="other_agent",
          content=types.ModelContent("Other agent message"),
      ),
  ]
  invocation_context.session.events = events

  # Process the request
  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  # Verify current agent's message stays as model role
  # and other agent's message is converted to user context
  assert len(llm_request.contents) == 2
  assert llm_request.contents[0] == types.ModelContent("My own message")
  assert llm_request.contents[1].role == "user"
  assert llm_request.contents[1].parts == [
      types.Part(text="For context:"),
      types.Part(text="[other_agent] said: Other agent message"),
  ]


@pytest.mark.asyncio
async def test_user_messages_preserved():
  """Test that user messages are preserved as-is."""
  agent = Agent(model="gemini-2.5-flash", name="current_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  # Add user message
  user_event = Event(
      invocation_id="inv1",
      author="user",
      content=types.UserContent("User message"),
  )
  invocation_context.session.events = [user_event]

  # Process the request
  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  # Verify user message is preserved exactly
  assert len(llm_request.contents) == 1
  assert llm_request.contents[0] == types.UserContent("User message")
