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

"""Tests for function call/response rearrangement in contents module."""

from google.adk.agents.llm_agent import Agent
from google.adk.events.event import Event
from google.adk.flows.llm_flows import contents
from google.adk.models.llm_request import LlmRequest
from google.genai import types
import pytest

from ... import testing_utils


@pytest.mark.asyncio
async def test_basic_function_call_response_processing():
  """Test basic function call/response processing without rearrangement."""
  agent = Agent(model="gemini-2.5-flash", name="test_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  function_call = types.FunctionCall(
      id="call_123", name="search_tool", args={"query": "test"}
  )
  function_response = types.FunctionResponse(
      id="call_123",
      name="search_tool",
      response={"results": ["item1", "item2"]},
  )

  events = [
      Event(
          invocation_id="inv1",
          author="user",
          content=types.UserContent("Search for test"),
      ),
      Event(
          invocation_id="inv2",
          author="test_agent",
          content=types.ModelContent([types.Part(function_call=function_call)]),
      ),
      Event(
          invocation_id="inv3",
          author="user",
          content=types.UserContent(
              [types.Part(function_response=function_response)]
          ),
      ),
  ]
  invocation_context.session.events = events

  # Process the request
  async for _ in contents.request_processor.run_async(
      invocation_context, llm_request
  ):
    pass

  # Verify no rearrangement occurred
  assert llm_request.contents == [
      types.UserContent("Search for test"),
      types.ModelContent([types.Part(function_call=function_call)]),
      types.UserContent([types.Part(function_response=function_response)]),
  ]


@pytest.mark.asyncio
async def test_rearrangement_with_intermediate_function_response():
  """Test rearrangement when intermediate function response appears after call."""
  agent = Agent(model="gemini-2.5-flash", name="test_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  function_call = types.FunctionCall(
      id="long_call_123", name="long_running_tool", args={"task": "process"}
  )
  # First intermediate response
  intermediate_response = types.FunctionResponse(
      id="long_call_123",
      name="long_running_tool",
      response={"status": "processing", "progress": 50},
  )
  # Final response
  final_response = types.FunctionResponse(
      id="long_call_123",
      name="long_running_tool",
      response={"status": "completed", "result": "done"},
  )

  events = [
      Event(
          invocation_id="inv1",
          author="user",
          content=types.UserContent("Run long process"),
      ),
      # Function call
      Event(
          invocation_id="inv2",
          author="test_agent",
          content=types.ModelContent([types.Part(function_call=function_call)]),
      ),
      # Intermediate function response appears right after call
      Event(
          invocation_id="inv3",
          author="user",
          content=types.UserContent(
              [types.Part(function_response=intermediate_response)]
          ),
      ),
      # Some conversation happens
      Event(
          invocation_id="inv4",
          author="test_agent",
          content=types.ModelContent("Still processing..."),
      ),
      # Final function response (this triggers rearrangement)
      Event(
          invocation_id="inv5",
          author="user",
          content=types.UserContent(
              [types.Part(function_response=final_response)]
          ),
      ),
  ]
  invocation_context.session.events = events

  # Process the request
  async for _ in contents.request_processor.run_async(
      invocation_context, llm_request
  ):
    pass

  # Verify rearrangement: intermediate events removed, final response replaces intermediate
  assert llm_request.contents == [
      types.UserContent("Run long process"),
      types.ModelContent([types.Part(function_call=function_call)]),
      types.UserContent([types.Part(function_response=final_response)]),
  ]


@pytest.mark.asyncio
async def test_mixed_long_running_and_normal_function_calls():
  """Test rearrangement with mixed long-running and normal function calls in same event."""
  agent = Agent(model="gemini-2.5-flash", name="test_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  # Two function calls: one long-running, one normal
  long_running_call = types.FunctionCall(
      id="lro_call_456", name="long_running_tool", args={"task": "analyze"}
  )
  normal_call = types.FunctionCall(
      id="normal_call_789", name="search_tool", args={"query": "test"}
  )

  # Intermediate response for long-running tool
  lro_intermediate_response = types.FunctionResponse(
      id="lro_call_456",
      name="long_running_tool",
      response={"status": "processing", "progress": 25},
  )
  # Response for normal tool (complete)
  normal_response = types.FunctionResponse(
      id="normal_call_789",
      name="search_tool",
      response={"results": ["item1", "item2"]},
  )
  # Final response for long-running tool
  lro_final_response = types.FunctionResponse(
      id="lro_call_456",
      name="long_running_tool",
      response={"status": "completed", "analysis": "done"},
  )

  events = [
      Event(
          invocation_id="inv1",
          author="user",
          content=types.UserContent("Analyze data and search for info"),
      ),
      # Both function calls in same event
      Event(
          invocation_id="inv2",
          author="test_agent",
          content=types.ModelContent([
              types.Part(function_call=long_running_call),
              types.Part(function_call=normal_call),
          ]),
      ),
      # Intermediate responses for both tools
      Event(
          invocation_id="inv3",
          author="user",
          content=types.UserContent([
              types.Part(function_response=lro_intermediate_response),
              types.Part(function_response=normal_response),
          ]),
      ),
      # Some conversation
      Event(
          invocation_id="inv4",
          author="test_agent",
          content=types.ModelContent("Analysis in progress, search completed"),
      ),
      # Final response for long-running tool (triggers rearrangement)
      Event(
          invocation_id="inv5",
          author="user",
          content=types.UserContent(
              [types.Part(function_response=lro_final_response)]
          ),
      ),
  ]
  invocation_context.session.events = events

  # Process the request
  async for _ in contents.request_processor.run_async(
      invocation_context, llm_request
  ):
    pass

  # Verify rearrangement: LRO intermediate replaced by final, normal tool preserved
  assert llm_request.contents == [
      types.UserContent("Analyze data and search for info"),
      types.ModelContent([
          types.Part(function_call=long_running_call),
          types.Part(function_call=normal_call),
      ]),
      types.UserContent([
          types.Part(function_response=lro_final_response),
          types.Part(function_response=normal_response),
      ]),
  ]


@pytest.mark.asyncio
async def test_completed_long_running_function_in_history():
  """Test that completed long-running function calls in history.

  Function call/response are properly rearranged and don't affect subsequent
  conversation.
  """
  agent = Agent(model="gemini-2.5-flash", name="test_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  function_call = types.FunctionCall(
      id="history_call_123", name="long_running_tool", args={"task": "process"}
  )
  intermediate_response = types.FunctionResponse(
      id="history_call_123",
      name="long_running_tool",
      response={"status": "processing", "progress": 50},
  )
  final_response = types.FunctionResponse(
      id="history_call_123",
      name="long_running_tool",
      response={"status": "completed", "result": "done"},
  )

  events = [
      Event(
          invocation_id="inv1",
          author="user",
          content=types.UserContent("Start long process"),
      ),
      # Function call in history
      Event(
          invocation_id="inv2",
          author="test_agent",
          content=types.ModelContent([types.Part(function_call=function_call)]),
      ),
      # Intermediate response in history
      Event(
          invocation_id="inv3",
          author="user",
          content=types.UserContent(
              [types.Part(function_response=intermediate_response)]
          ),
      ),
      # Some conversation happens
      Event(
          invocation_id="inv4",
          author="test_agent",
          content=types.ModelContent("Still processing..."),
      ),
      # Final response completes the long-running function in history
      Event(
          invocation_id="inv5",
          author="user",
          content=types.UserContent(
              [types.Part(function_response=final_response)]
          ),
      ),
      # Agent acknowledges completion
      Event(
          invocation_id="inv6",
          author="test_agent",
          content=types.ModelContent("Process completed successfully!"),
      ),
      # Latest event is regular user message, not function response
      Event(
          invocation_id="inv7",
          author="user",
          content=types.UserContent("Great! What's next?"),
      ),
  ]
  invocation_context.session.events = events

  # Process the request
  async for _ in contents.request_processor.run_async(
      invocation_context, llm_request
  ):
    pass

  # Verify the long-running function in history was rearranged correctly:
  # - Intermediate response was replaced by final response
  # - Non-function events (like "Still processing...") are preserved
  # - No further rearrangement occurs since latest event is not function response
  assert llm_request.contents == [
      types.UserContent("Start long process"),
      types.ModelContent([types.Part(function_call=function_call)]),
      types.UserContent([types.Part(function_response=final_response)]),
      types.ModelContent("Still processing..."),
      types.ModelContent("Process completed successfully!"),
      types.UserContent("Great! What's next?"),
  ]


@pytest.mark.asyncio
async def test_completed_mixed_function_calls_in_history():
  """Test completed mixed long-running and normal function calls in history don't affect subsequent conversation."""
  agent = Agent(model="gemini-2.5-flash", name="test_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  # Two function calls: one long-running, one normal
  long_running_call = types.FunctionCall(
      id="history_lro_123", name="long_running_tool", args={"task": "analyze"}
  )
  normal_call = types.FunctionCall(
      id="history_normal_456", name="search_tool", args={"query": "data"}
  )

  # Intermediate response for long-running tool
  lro_intermediate_response = types.FunctionResponse(
      id="history_lro_123",
      name="long_running_tool",
      response={"status": "processing", "progress": 30},
  )
  # Complete response for normal tool
  normal_response = types.FunctionResponse(
      id="history_normal_456",
      name="search_tool",
      response={"results": ["result1", "result2"]},
  )
  # Final response for long-running tool
  lro_final_response = types.FunctionResponse(
      id="history_lro_123",
      name="long_running_tool",
      response={"status": "completed", "analysis": "finished"},
  )

  events = [
      Event(
          invocation_id="inv1",
          author="user",
          content=types.UserContent("Analyze and search simultaneously"),
      ),
      # Both function calls in history
      Event(
          invocation_id="inv2",
          author="test_agent",
          content=types.ModelContent([
              types.Part(function_call=long_running_call),
              types.Part(function_call=normal_call),
          ]),
      ),
      # Intermediate responses for both tools in history
      Event(
          invocation_id="inv3",
          author="user",
          content=types.UserContent([
              types.Part(function_response=lro_intermediate_response),
              types.Part(function_response=normal_response),
          ]),
      ),
      # Some conversation in history
      Event(
          invocation_id="inv4",
          author="test_agent",
          content=types.ModelContent("Analysis continuing, search done"),
      ),
      # Final response completes the long-running function in history
      Event(
          invocation_id="inv5",
          author="user",
          content=types.UserContent(
              [types.Part(function_response=lro_final_response)]
          ),
      ),
      # Agent acknowledges completion
      Event(
          invocation_id="inv6",
          author="test_agent",
          content=types.ModelContent("Both tasks completed successfully!"),
      ),
      # Latest event is regular user message, not function response
      Event(
          invocation_id="inv7",
          author="user",
          content=types.UserContent("Perfect! What should we do next?"),
      ),
  ]
  invocation_context.session.events = events

  # Process the request
  async for _ in contents.request_processor.run_async(
      invocation_context, llm_request
  ):
    pass

  # Verify mixed functions in history were rearranged correctly:
  # - LRO intermediate was replaced by final response
  # - Normal tool response was preserved
  # - Non-function events preserved, no further rearrangement
  assert llm_request.contents == [
      types.UserContent("Analyze and search simultaneously"),
      types.ModelContent([
          types.Part(function_call=long_running_call),
          types.Part(function_call=normal_call),
      ]),
      types.UserContent([
          types.Part(function_response=lro_final_response),
          types.Part(function_response=normal_response),
      ]),
      types.ModelContent("Analysis continuing, search done"),
      types.ModelContent("Both tasks completed successfully!"),
      types.UserContent("Perfect! What should we do next?"),
  ]


@pytest.mark.asyncio
async def test_function_rearrangement_preserves_other_content():
  """Test that non-function content is preserved during rearrangement."""
  agent = Agent(model="gemini-2.5-flash", name="test_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  function_call = types.FunctionCall(
      id="preserve_test", name="long_running_tool", args={"test": "value"}
  )
  intermediate_response = types.FunctionResponse(
      id="preserve_test",
      name="long_running_tool",
      response={"status": "processing"},
  )
  final_response = types.FunctionResponse(
      id="preserve_test",
      name="long_running_tool",
      response={"output": "preserved"},
  )

  events = [
      Event(
          invocation_id="inv1",
          author="user",
          content=types.UserContent("Before function call"),
      ),
      Event(
          invocation_id="inv2",
          author="test_agent",
          content=types.ModelContent([
              types.Part(text="I'll process this for you"),
              types.Part(function_call=function_call),
          ]),
      ),
      # Intermediate response with mixed content
      Event(
          invocation_id="inv3",
          author="user",
          content=types.UserContent([
              types.Part(text="Intermediate prefix"),
              types.Part(function_response=intermediate_response),
              types.Part(text="Processing..."),
          ]),
      ),
      # This should be removed during rearrangement
      Event(
          invocation_id="inv4",
          author="test_agent",
          content=types.ModelContent("Still working on it..."),
      ),
      # Final response with mixed content (triggers rearrangement)
      Event(
          invocation_id="inv5",
          author="user",
          content=types.UserContent([
              types.Part(text="Final prefix"),
              types.Part(function_response=final_response),
              types.Part(text="Final suffix"),
          ]),
      ),
  ]
  invocation_context.session.events = events

  # Process the request
  async for _ in contents.request_processor.run_async(
      invocation_context, llm_request
  ):
    pass

  # Verify non-function content is preserved during rearrangement
  # Intermediate response replaced by final, but ALL text content preserved
  assert llm_request.contents == [
      types.UserContent("Before function call"),
      types.ModelContent([
          types.Part(text="I'll process this for you"),
          types.Part(function_call=function_call),
      ]),
      types.UserContent([
          types.Part(text="Intermediate prefix"),
          types.Part(function_response=final_response),
          types.Part(text="Processing..."),
          types.Part(text="Final prefix"),
          types.Part(text="Final suffix"),
      ]),
  ]


@pytest.mark.asyncio
async def test_error_when_function_response_without_matching_call():
  """Test error when function response has no matching function call."""
  agent = Agent(model="gemini-2.5-flash", name="test_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  # Function response without matching call
  orphaned_response = types.FunctionResponse(
      id="no_matching_call",
      name="orphaned_tool",
      response={"error": "no matching call"},
  )

  events = [
      Event(
          invocation_id="inv1",
          author="user",
          content=types.UserContent("Regular message"),
      ),
      # Response without any prior matching function call
      Event(
          invocation_id="inv2",
          author="user",
          content=types.UserContent(
              [types.Part(function_response=orphaned_response)]
          ),
      ),
  ]
  invocation_context.session.events = events

  # This should raise a ValueError during processing
  with pytest.raises(ValueError, match="No function call event found"):
    async for _ in contents.request_processor.run_async(
        invocation_context, llm_request
    ):
      pass
