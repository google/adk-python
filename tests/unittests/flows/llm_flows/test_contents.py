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
from google.adk.flows.llm_flows import contents
from google.adk.flows.llm_flows.contents import _convert_foreign_event
from google.adk.flows.llm_flows.contents import _get_contents
from google.adk.flows.llm_flows.contents import _merge_function_response_events
from google.adk.flows.llm_flows.contents import _rearrange_events_for_async_function_responses_in_history
from google.adk.flows.llm_flows.contents import _rearrange_events_for_latest_function_response
from google.adk.models.llm_request import LlmRequest
from google.genai import types
import pytest

from ... import testing_utils


@pytest.mark.asyncio
async def test_content_processor_no_contents():
  """Test ContentLlmRequestProcessor when include_contents is 'none'."""
  agent = Agent(model="gemini-1.5-flash", name="agent", include_contents="none")
  llm_request = LlmRequest(model="gemini-1.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  # Collect events from async generator
  events = []
  async for event in contents.request_processor.run_async(
      invocation_context, llm_request
  ):
    events.append(event)

  # Should not yield any events
  assert len(events) == 0
  # Contents should not be set when include_contents is 'none'
  assert llm_request.contents == []


@pytest.mark.asyncio
async def test_content_processor_with_contents():
  """Test ContentLlmRequestProcessor when include_contents is not 'none'."""
  agent = Agent(model="gemini-1.5-flash", name="agent")
  llm_request = LlmRequest(model="gemini-1.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  # Add some test events to the session
  test_event = Event(
      invocation_id="test_inv",
      author="user",
      content=types.Content(
          role="user", parts=[types.Part.from_text(text="Hello")]
      ),
  )
  invocation_context.session.events = [test_event]

  # Collect events from async generator
  events = []
  async for event in contents.request_processor.run_async(
      invocation_context, llm_request
  ):
    events.append(event)

  # Should not yield any events (processor doesn't emit events, just modifies request)
  assert len(events) == 0
  # Contents should be set
  assert llm_request.contents is not None
  assert len(llm_request.contents) == 1
  assert llm_request.contents[0].role == "user"
  assert llm_request.contents[0].parts[0].text == "Hello"


@pytest.mark.asyncio
async def test_content_processor_non_llm_agent():
  """Test ContentLlmRequestProcessor with non-LLM agent."""
  from google.adk.agents.base_agent import BaseAgent

  # Create a base agent (not LLM agent)
  agent = BaseAgent(name="base_agent")
  llm_request = LlmRequest(model="gemini-1.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  # Collect events from async generator
  events = []
  async for event in contents.request_processor.run_async(
      invocation_context, llm_request
  ):
    events.append(event)

  # Should not yield any events and not modify request
  assert len(events) == 0
  assert llm_request.contents == []


def test_get_contents_empty_events():
  """Test _get_contents with empty events list."""
  contents_result = _get_contents(None, [], "test_agent")
  assert contents_result == []


def test_get_contents_with_events():
  """Test _get_contents with valid events."""
  test_event = Event(
      invocation_id="test_inv",
      author="user",
      content=types.Content(
          role="user", parts=[types.Part.from_text(text="Hello")]
      ),
  )

  contents_result = _get_contents(None, [test_event], "test_agent")
  assert len(contents_result) == 1
  assert contents_result[0].role == "user"
  assert contents_result[0].parts[0].text == "Hello"


def test_get_contents_filters_empty_events():
  """Test _get_contents filters out events with empty content."""
  # Event with empty text
  empty_event = Event(
      invocation_id="test_inv",
      author="user",
      content=types.Content(role="user", parts=[types.Part.from_text(text="")]),
  )

  # Event without content
  no_content_event = Event(
      invocation_id="test_inv",
      author="user",
  )

  # Valid event
  valid_event = Event(
      invocation_id="test_inv",
      author="user",
      content=types.Content(
          role="user", parts=[types.Part.from_text(text="Hello")]
      ),
  )

  contents_result = _get_contents(
      None, [empty_event, no_content_event, valid_event], "test_agent"
  )
  assert len(contents_result) == 1
  assert contents_result[0].role == "user"
  assert contents_result[0].parts[0].text == "Hello"


def test_convert_foreign_event():
  """Test _convert_foreign_event function."""
  agent_event = Event(
      invocation_id="test_inv",
      author="agent1",
      content=types.Content(
          role="model", parts=[types.Part.from_text(text="Agent response")]
      ),
  )

  converted_event = _convert_foreign_event(agent_event)

  assert converted_event.author == "user"
  assert converted_event.content.role == "user"
  assert len(converted_event.content.parts) == 2
  assert converted_event.content.parts[0].text == "For context:"
  assert (
      "[agent1] said: Agent response" in converted_event.content.parts[1].text
  )


def test_convert_event_with_function_call():
  """Test _convert_foreign_event with function call."""
  function_call = types.FunctionCall(
      id="func_123", name="test_function", args={"param": "value"}
  )

  agent_event = Event(
      invocation_id="test_inv",
      author="agent1",
      content=types.Content(
          role="model", parts=[types.Part(function_call=function_call)]
      ),
  )

  converted_event = _convert_foreign_event(agent_event)

  assert converted_event.author == "user"
  assert converted_event.content.role == "user"
  assert len(converted_event.content.parts) == 2
  assert converted_event.content.parts[0].text == "For context:"
  assert (
      "[agent1] called tool `test_function`"
      in converted_event.content.parts[1].text
  )
  assert "{'param': 'value'}" in converted_event.content.parts[1].text


def test_convert_event_with_function_response():
  """Test _convert_foreign_event with function response."""
  function_response = types.FunctionResponse(
      id="func_123", name="test_function", response={"result": "success"}
  )

  agent_event = Event(
      invocation_id="test_inv",
      author="agent1",
      content=types.Content(
          role="user", parts=[types.Part(function_response=function_response)]
      ),
  )

  converted_event = _convert_foreign_event(agent_event)

  assert converted_event.author == "user"
  assert converted_event.content.role == "user"
  assert len(converted_event.content.parts) == 2
  assert converted_event.content.parts[0].text == "For context:"
  assert (
      "[agent1] `test_function` tool returned result:"
      in converted_event.content.parts[1].text
  )
  assert "{'result': 'success'}" in converted_event.content.parts[1].text


def test_merge_function_response_events():
  """Test _merge_function_response_events function."""
  # Create initial function response event
  function_response1 = types.FunctionResponse(
      id="func_123", name="test_function", response={"status": "pending"}
  )

  initial_event = Event(
      invocation_id="test_inv",
      author="user",
      content=types.Content(
          role="user", parts=[types.Part(function_response=function_response1)]
      ),
  )

  # Create final function response event
  function_response2 = types.FunctionResponse(
      id="func_123", name="test_function", response={"result": "success"}
  )

  final_event = Event(
      invocation_id="test_inv2",
      author="user",
      content=types.Content(
          role="user", parts=[types.Part(function_response=function_response2)]
      ),
  )

  merged_event = _merge_function_response_events([initial_event, final_event])

  assert (
      merged_event.invocation_id == "test_inv"
  )  # Should keep initial event ID
  assert len(merged_event.content.parts) == 1
  # The first part should be replaced with the final response
  assert merged_event.content.parts[0].function_response.response == {
      "result": "success"
  }


def test_rearrange_events_for_async_function_responses():
  """Test _rearrange_events_for_async_function_responses_in_history function."""
  # Create function call event
  function_call = types.FunctionCall(
      id="func_123", name="test_function", args={"param": "value"}
  )

  call_event = Event(
      invocation_id="test_inv1",
      author="agent",
      content=types.Content(
          role="model", parts=[types.Part(function_call=function_call)]
      ),
  )

  # Create function response event
  function_response = types.FunctionResponse(
      id="func_123", name="test_function", response={"result": "success"}
  )

  response_event = Event(
      invocation_id="test_inv2",
      author="user",
      content=types.Content(
          role="user", parts=[types.Part(function_response=function_response)]
      ),
  )

  # Test rearrangement
  events = [call_event, response_event]
  rearranged = _rearrange_events_for_async_function_responses_in_history(events)

  # Should have both events in correct order
  assert len(rearranged) == 2
  assert rearranged[0] == call_event
  assert rearranged[1] == response_event


def test_rearrange_events_for_latest_function_response():
  """Test _rearrange_events_for_latest_function_response function."""
  # Create function call event
  function_call = types.FunctionCall(
      id="func_123", name="test_function", args={"param": "value"}
  )

  call_event = Event(
      invocation_id="test_inv1",
      author="agent",
      content=types.Content(
          role="model", parts=[types.Part(function_call=function_call)]
      ),
  )

  # Create intermediate event
  intermediate_event = Event(
      invocation_id="test_inv2",
      author="agent",
      content=types.Content(
          role="model", parts=[types.Part.from_text(text="Processing...")]
      ),
  )

  # Create function response event
  function_response = types.FunctionResponse(
      id="func_123", name="test_function", response={"result": "success"}
  )

  response_event = Event(
      invocation_id="test_inv3",
      author="user",
      content=types.Content(
          role="user", parts=[types.Part(function_response=function_response)]
      ),
  )

  # Test with matching function call and response
  events = [call_event, intermediate_event, response_event]
  rearranged = _rearrange_events_for_latest_function_response(events)

  # Should remove intermediate events and merge responses
  assert len(rearranged) == 2
  assert rearranged[0] == call_event
  assert rearranged[1] == response_event


def test_rearrange_events_for_latest_function_response_multiple_calls():
  """Test _rearrange_events_for_latest_function_response with multiple function calls."""
  # Create function call event with multiple calls
  function_call1 = types.FunctionCall(
      id="func_123", name="test_function", args={"param": "value1"}
  )
  function_call2 = types.FunctionCall(
      id="func_456", name="test_function2", args={"param": "value2"}
  )

  call_event = Event(
      invocation_id="test_inv1",
      author="agent",
      content=types.Content(
          role="model",
          parts=[
              types.Part(function_call=function_call1),
              types.Part(function_call=function_call2),
          ],
      ),
  )

  # Create intermediate event
  intermediate_event = Event(
      invocation_id="test_inv2",
      author="agent",
      content=types.Content(
          role="model", parts=[types.Part.from_text(text="Processing...")]
      ),
  )

  # Create function response event with only one response
  function_response = types.FunctionResponse(
      id="func_123", name="test_function", response={"result": "success"}
  )

  response_event = Event(
      invocation_id="test_inv3",
      author="user",
      content=types.Content(
          role="user", parts=[types.Part(function_response=function_response)]
      ),
  )

  # Test with matching function call and response
  events = [call_event, intermediate_event, response_event]
  rearranged = _rearrange_events_for_latest_function_response(events)

  # Should remove intermediate events and merge responses
  assert len(rearranged) == 2
  assert rearranged[0] == call_event
  assert rearranged[1] == response_event


def test_rearrange_events_for_latest_function_response_validation_error():
  """Test _rearrange_events_for_latest_function_response with validation error."""
  # Create function call event with one function call
  function_call = types.FunctionCall(
      id="func_123", name="test_function", args={"param": "value"}
  )

  call_event = Event(
      invocation_id="test_inv1",
      author="agent",
      content=types.Content(
          role="model", parts=[types.Part(function_call=function_call)]
      ),
  )

  # Create intermediate event
  intermediate_event = Event(
      invocation_id="test_inv2",
      author="agent",
      content=types.Content(
          role="model", parts=[types.Part.from_text(text="Processing...")]
      ),
  )

  # Create function response event with the matching function call AND an extra one
  function_response1 = types.FunctionResponse(
      id="func_123", name="test_function", response={"result": "success"}
  )
  function_response2 = types.FunctionResponse(
      id="func_456", name="other_function", response={"result": "other"}
  )

  response_event = Event(
      invocation_id="test_inv3",
      author="user",
      content=types.Content(
          role="user",
          parts=[
              types.Part(function_response=function_response1),
              types.Part(function_response=function_response2),
          ],
      ),
  )

  # Test with mismatched function call and response
  events = [call_event, intermediate_event, response_event]

  with pytest.raises(
      ValueError,
      match=(
          "Last response event should only contain the responses for the"
          " function calls in the same function call event"
      ),
  ):
    _rearrange_events_for_latest_function_response(events)


def test_rearrange_events_for_latest_function_response_mixed_responses():
  """Test _rearrange_events_for_latest_function_response with mixed function responses."""
  # Create function call event with two calls
  function_call1 = types.FunctionCall(
      id="func_123", name="test_function", args={"param": "value1"}
  )
  function_call2 = types.FunctionCall(
      id="func_456", name="test_function2", args={"param": "value2"}
  )

  call_event = Event(
      invocation_id="test_inv1",
      author="agent",
      content=types.Content(
          role="model",
          parts=[
              types.Part(function_call=function_call1),
              types.Part(function_call=function_call2),
          ],
      ),
  )

  # Create intermediate event
  intermediate_event = Event(
      invocation_id="test_inv2",
      author="agent",
      content=types.Content(
          role="model", parts=[types.Part.from_text(text="Processing...")]
      ),
  )

  # Create function response event with one matching and one non-matching response
  function_response1 = types.FunctionResponse(
      id="func_123", name="test_function", response={"result": "success"}
  )
  function_response2 = types.FunctionResponse(
      id="func_789", name="test_function3", response={"result": "other"}
  )

  response_event = Event(
      invocation_id="test_inv3",
      author="user",
      content=types.Content(
          role="user",
          parts=[
              types.Part(function_response=function_response1),
              types.Part(function_response=function_response2),
          ],
      ),
  )

  # Test with mixed function responses
  events = [call_event, intermediate_event, response_event]

  with pytest.raises(
      ValueError,
      match=(
          "Last response event should only contain the responses for the"
          " function calls in the same function call event"
      ),
  ):
    _rearrange_events_for_latest_function_response(events)
