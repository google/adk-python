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

from google.adk.agents.llm_agent import Agent
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.flows.llm_flows import contents
from google.adk.flows.llm_flows.functions import REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
from google.adk.flows.llm_flows.functions import REQUEST_EUC_FUNCTION_CALL_NAME
from google.adk.models.llm_request import LlmRequest
from google.genai import types
import pytest

from ... import testing_utils


@pytest.mark.asyncio
async def test_include_contents_default_full_history():
  """Test that include_contents='default' includes full conversation history."""
  agent = Agent(
      model="gemini-2.5-flash", name="test_agent", include_contents="default"
  )
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  # Create a multi-turn conversation
  events = [
      Event(
          invocation_id="inv1",
          author="user",
          content=types.UserContent("First message"),
      ),
      Event(
          invocation_id="inv2",
          author="test_agent",
          content=types.ModelContent("First response"),
      ),
      Event(
          invocation_id="inv3",
          author="user",
          content=types.UserContent("Second message"),
      ),
      Event(
          invocation_id="inv4",
          author="test_agent",
          content=types.ModelContent("Second response"),
      ),
      Event(
          invocation_id="inv5",
          author="user",
          content=types.UserContent("Third message"),
      ),
  ]
  invocation_context.session.events = events

  # Process the request
  async for _ in contents.request_processor.run_async(
      invocation_context, llm_request
  ):
    pass

  # Verify full conversation history is included
  assert llm_request.contents == [
      types.UserContent("First message"),
      types.ModelContent("First response"),
      types.UserContent("Second message"),
      types.ModelContent("Second response"),
      types.UserContent("Third message"),
  ]


@pytest.mark.asyncio
async def test_include_contents_none_current_turn_only():
  """Test that include_contents='none' includes only current turn context."""
  agent = Agent(
      model="gemini-2.5-flash", name="test_agent", include_contents="none"
  )
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  # Create a multi-turn conversation
  events = [
      Event(
          invocation_id="inv1",
          author="user",
          content=types.UserContent("First message"),
      ),
      Event(
          invocation_id="inv2",
          author="test_agent",
          content=types.ModelContent("First response"),
      ),
      Event(
          invocation_id="inv3",
          author="user",
          content=types.UserContent("Second message"),
      ),
      Event(
          invocation_id="inv4",
          author="test_agent",
          content=types.ModelContent("Second response"),
      ),
      Event(
          invocation_id="inv5",
          author="user",
          content=types.UserContent("Current turn message"),
      ),
  ]
  invocation_context.session.events = events

  # Process the request
  async for _ in contents.request_processor.run_async(
      invocation_context, llm_request
  ):
    pass

  # Verify only current turn is included (from last user message)
  assert llm_request.contents == [
      types.UserContent("Current turn message"),
  ]


@pytest.mark.asyncio
async def test_include_contents_none_multi_agent_current_turn():
  """Test current turn detection in multi-agent scenarios with include_contents='none'."""
  agent = Agent(
      model="gemini-2.5-flash", name="current_agent", include_contents="none"
  )
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  # Create multi-agent conversation where current turn starts from user
  events = [
      Event(
          invocation_id="inv1",
          author="user",
          content=types.UserContent("First user message"),
      ),
      Event(
          invocation_id="inv2",
          author="other_agent",
          content=types.ModelContent("Other agent response"),
      ),
      Event(
          invocation_id="inv3",
          author="current_agent",
          content=types.ModelContent("Current agent first response"),
      ),
      Event(
          invocation_id="inv4",
          author="user",
          content=types.UserContent("Current turn request"),
      ),
      Event(
          invocation_id="inv5",
          author="another_agent",
          content=types.ModelContent("Another agent responds"),
      ),
      Event(
          invocation_id="inv6",
          author="current_agent",
          content=types.ModelContent("Current agent in turn"),
      ),
  ]
  invocation_context.session.events = events

  # Process the request
  async for _ in contents.request_processor.run_async(
      invocation_context, llm_request
  ):
    pass

  # Verify current turn starts from the most recent other agent message (inv5)
  assert len(llm_request.contents) == 2
  assert llm_request.contents[0].role == "user"
  assert llm_request.contents[0].parts == [
      types.Part(text="For context:"),
      types.Part(text="[another_agent] said: Another agent responds"),
  ]
  assert llm_request.contents[1] == types.ModelContent("Current agent in turn")


@pytest.mark.asyncio
async def test_include_contents_none_multi_branch_current_turn():
  """Test current turn detection in multi-branch scenarios with include_contents='none'."""
  agent = Agent(
      model="gemini-2.5-flash", name="current_agent", include_contents="none"
  )
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  invocation_context.branch = "root.parent_agent"

  # Create multi-branch conversation where current turn starts from user
  # This can arise from having a Parallel Agent with two or more Sequential
  # Agents as sub agents, each with two Llm Agents as sub agents
  events = [
      Event(
          invocation_id="inv1",
          branch="root",
          author="user",
          content=types.UserContent("First user message"),
      ),
      Event(
          invocation_id="inv1",
          branch="root.parent_agent",
          author="sibling_agent",
          content=types.ModelContent("Sibling agent response"),
      ),
      Event(
          invocation_id="inv1",
          branch="root.uncle_agent",
          author="cousin_agent",
          content=types.ModelContent("Cousin agent response"),
      ),
  ]
  invocation_context.session.events = events

  # Process the request
  async for _ in contents.request_processor.run_async(
      invocation_context, llm_request
  ):
    pass

  # Verify current turn starts from the most recent other agent message of the current branch
  assert len(llm_request.contents) == 1
  assert llm_request.contents[0].role == "user"
  assert llm_request.contents[0].parts == [
      types.Part(text="For context:"),
      types.Part(text="[sibling_agent] said: Sibling agent response"),
  ]


@pytest.mark.asyncio
async def test_authentication_events_are_filtered():
  """Test that authentication function calls and responses are filtered out."""
  agent = Agent(model="gemini-2.5-flash", name="test_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  # Create authentication function call and response
  auth_function_call = types.FunctionCall(
      id="auth_123",
      name=REQUEST_EUC_FUNCTION_CALL_NAME,
      args={"credential_type": "oauth"},
  )
  auth_response = types.FunctionResponse(
      id="auth_123",
      name=REQUEST_EUC_FUNCTION_CALL_NAME,
      response={
          "auth_config": {"exchanged_auth_credential": {"token": "secret"}}
      },
  )

  events = [
      Event(
          invocation_id="inv1",
          author="user",
          content=types.UserContent("Please authenticate"),
      ),
      Event(
          invocation_id="inv2",
          author="test_agent",
          content=types.ModelContent(
              [types.Part(function_call=auth_function_call)]
          ),
      ),
      Event(
          invocation_id="inv3",
          author="user",
          content=types.Content(
              parts=[types.Part(function_response=auth_response)], role="user"
          ),
      ),
      Event(
          invocation_id="inv4",
          author="user",
          content=types.UserContent("Continue after auth"),
      ),
  ]
  invocation_context.session.events = events

  # Process the request
  async for _ in contents.request_processor.run_async(
      invocation_context, llm_request
  ):
    pass

  # Verify both authentication call and response are filtered out
  assert llm_request.contents == [
      types.UserContent("Please authenticate"),
      types.UserContent("Continue after auth"),
  ]


@pytest.mark.asyncio
async def test_confirmation_events_are_filtered():
  """Test that confirmation function calls and responses are filtered out."""
  agent = Agent(model="gemini-2.5-flash", name="test_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  # Create confirmation function call and response
  confirmation_function_call = types.FunctionCall(
      id="confirm_123",
      name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
      args={"action": "delete_file", "confirmation": True},
  )
  confirmation_response = types.FunctionResponse(
      id="confirm_123",
      name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
      response={"response": '{"confirmed": true}'},
  )

  events = [
      Event(
          invocation_id="inv1",
          author="user",
          content=types.UserContent("Delete the file"),
      ),
      Event(
          invocation_id="inv2",
          author="test_agent",
          content=types.ModelContent(
              [types.Part(function_call=confirmation_function_call)]
          ),
      ),
      Event(
          invocation_id="inv3",
          author="user",
          content=types.Content(
              parts=[types.Part(function_response=confirmation_response)],
              role="user",
          ),
      ),
      Event(
          invocation_id="inv4",
          author="user",
          content=types.UserContent("File deleted successfully"),
      ),
  ]
  invocation_context.session.events = events

  # Process the request
  async for _ in contents.request_processor.run_async(
      invocation_context, llm_request
  ):
    pass

  # Verify both confirmation call and response are filtered out
  assert llm_request.contents == [
      types.UserContent("Delete the file"),
      types.UserContent("File deleted successfully"),
  ]


@pytest.mark.asyncio
async def test_rewind_events_are_filtered_out():
  """Test that events are filtered based on rewind action."""
  agent = Agent(model="gemini-2.5-flash", name="test_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  events = [
      Event(
          invocation_id="inv1",
          author="user",
          content=types.UserContent("First message"),
      ),
      Event(
          invocation_id="inv1",
          author="test_agent",
          content=types.ModelContent("First response"),
      ),
      Event(
          invocation_id="inv2",
          author="user",
          content=types.UserContent("Second message"),
      ),
      Event(
          invocation_id="inv2",
          author="test_agent",
          content=types.ModelContent("Second response"),
      ),
      Event(
          invocation_id="rewind_inv",
          author="test_agent",
          actions=EventActions(rewind_before_invocation_id="inv2"),
      ),
      Event(
          invocation_id="inv3",
          author="user",
          content=types.UserContent("Third message"),
      ),
  ]
  invocation_context.session.events = events

  # Process the request
  async for _ in contents.request_processor.run_async(
      invocation_context, llm_request
  ):
    pass

  # Verify rewind correctly filters conversation history
  assert llm_request.contents == [
      types.UserContent("First message"),
      types.ModelContent("First response"),
      types.UserContent("Third message"),
  ]


@pytest.mark.asyncio
async def test_events_with_empty_content_are_skipped():
  """Test that events with empty content (state-only changes) are skipped."""
  agent = Agent(model="gemini-2.5-flash", name="test_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  events = [
      Event(
          invocation_id="inv1",
          author="user",
          content=types.UserContent("Hello"),
      ),
      # Event with no content (state-only change)
      Event(
          invocation_id="inv2",
          author="test_agent",
          actions=EventActions(state_delta={"key": "val"}),
      ),
      # Event with content that has no meaningful parts
      Event(
          invocation_id="inv4",
          author="test_agent",
          content=types.Content(parts=[], role="model"),
      ),
      Event(
          invocation_id="inv5",
          author="user",
          content=types.UserContent("How are you?"),
      ),
  ]
  invocation_context.session.events = events

  # Process the request
  async for _ in contents.request_processor.run_async(
      invocation_context, llm_request
  ):
    pass

  # Verify only events with meaningful content are included
  assert llm_request.contents == [
      types.UserContent("Hello"),
      types.UserContent("How are you?"),
  ]
