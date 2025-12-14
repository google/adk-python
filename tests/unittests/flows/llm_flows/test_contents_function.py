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


@pytest.mark.asyncio
async def test_interleaved_function_calls_are_merged():
  """Test that interleaved function call/response patterns are merged.

  This tests the fix for GitHub issue #3705 where Gemini 3 models with
  thinking enabled fail with "missing thought_signature" error when
  function calls and responses are interleaved.

  The pattern:
    [model(fc1), user(fr1), model(fc2), user(fr2)]
  should be merged to:
    [model([fc1, fc2]), user([fr1, fr2])]
  """
  agent = Agent(model="gemini-2.5-flash", name="test_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  # Create interleaved function calls and responses
  function_call_1 = types.FunctionCall(
      id="call_1", name="search_tool", args={"query": "topic 1"}
  )
  function_response_1 = types.FunctionResponse(
      id="call_1",
      name="search_tool",
      response={"results": ["result 1"]},
  )
  function_call_2 = types.FunctionCall(
      id="call_2", name="search_tool", args={"query": "topic 2"}
  )
  function_response_2 = types.FunctionResponse(
      id="call_2",
      name="search_tool",
      response={"results": ["result 2"]},
  )

  events = [
      Event(
          invocation_id="inv1",
          author="user",
          content=types.UserContent("Research two topics"),
      ),
      # First function call
      Event(
          invocation_id="inv2",
          author="test_agent",
          content=types.ModelContent(
              [types.Part(function_call=function_call_1)]
          ),
      ),
      # First function response
      Event(
          invocation_id="inv3",
          author="user",
          content=types.UserContent(
              [types.Part(function_response=function_response_1)]
          ),
      ),
      # Second function call (interleaved)
      Event(
          invocation_id="inv4",
          author="test_agent",
          content=types.ModelContent(
              [types.Part(function_call=function_call_2)]
          ),
      ),
      # Second function response
      Event(
          invocation_id="inv5",
          author="user",
          content=types.UserContent(
              [types.Part(function_response=function_response_2)]
          ),
      ),
  ]
  invocation_context.session.events = events

  # Process the request
  async for _ in contents.request_processor.run_async(
      invocation_context, llm_request
  ):
    pass

  # Verify interleaved pattern was merged:
  # [model(fc1), user(fr1), model(fc2), user(fr2)]
  # becomes:
  # [user(query), model([fc1, fc2]), user([fr1, fr2])]
  assert len(llm_request.contents) == 3
  assert llm_request.contents[0] == types.UserContent("Research two topics")

  # Check merged model content contains both function calls
  merged_model = llm_request.contents[1]
  assert merged_model.role == "model"
  assert len(merged_model.parts) == 2
  assert merged_model.parts[0].function_call == function_call_1
  assert merged_model.parts[1].function_call == function_call_2

  # Check merged user content contains both function responses
  merged_user = llm_request.contents[2]
  assert merged_user.role == "user"
  assert len(merged_user.parts) == 2
  assert merged_user.parts[0].function_response == function_response_1
  assert merged_user.parts[1].function_response == function_response_2


@pytest.mark.asyncio
async def test_three_interleaved_function_calls_are_merged():
  """Test that three or more interleaved function calls are properly merged."""
  agent = Agent(model="gemini-2.5-flash", name="test_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  # Create three interleaved function calls
  fc1 = types.FunctionCall(id="call_1", name="tool", args={"q": "1"})
  fr1 = types.FunctionResponse(id="call_1", name="tool", response={"r": "1"})
  fc2 = types.FunctionCall(id="call_2", name="tool", args={"q": "2"})
  fr2 = types.FunctionResponse(id="call_2", name="tool", response={"r": "2"})
  fc3 = types.FunctionCall(id="call_3", name="tool", args={"q": "3"})
  fr3 = types.FunctionResponse(id="call_3", name="tool", response={"r": "3"})

  events = [
      Event(
          invocation_id="inv1",
          author="user",
          content=types.UserContent("Query"),
      ),
      Event(
          invocation_id="inv2",
          author="test_agent",
          content=types.ModelContent([types.Part(function_call=fc1)]),
      ),
      Event(
          invocation_id="inv3",
          author="user",
          content=types.UserContent([types.Part(function_response=fr1)]),
      ),
      Event(
          invocation_id="inv4",
          author="test_agent",
          content=types.ModelContent([types.Part(function_call=fc2)]),
      ),
      Event(
          invocation_id="inv5",
          author="user",
          content=types.UserContent([types.Part(function_response=fr2)]),
      ),
      Event(
          invocation_id="inv6",
          author="test_agent",
          content=types.ModelContent([types.Part(function_call=fc3)]),
      ),
      Event(
          invocation_id="inv7",
          author="user",
          content=types.UserContent([types.Part(function_response=fr3)]),
      ),
  ]
  invocation_context.session.events = events

  async for _ in contents.request_processor.run_async(
      invocation_context, llm_request
  ):
    pass

  # Verify all three calls/responses are merged
  assert len(llm_request.contents) == 3

  merged_model = llm_request.contents[1]
  assert merged_model.role == "model"
  assert len(merged_model.parts) == 3
  assert merged_model.parts[0].function_call == fc1
  assert merged_model.parts[1].function_call == fc2
  assert merged_model.parts[2].function_call == fc3

  merged_user = llm_request.contents[2]
  assert merged_user.role == "user"
  assert len(merged_user.parts) == 3
  assert merged_user.parts[0].function_response == fr1
  assert merged_user.parts[1].function_response == fr2
  assert merged_user.parts[2].function_response == fr3


@pytest.mark.asyncio
async def test_interleaved_merge_with_text_after():
  """Test that interleaved merge works when followed by text content."""
  agent = Agent(model="gemini-2.5-flash", name="test_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  fc1 = types.FunctionCall(id="call_1", name="tool", args={"q": "1"})
  fr1 = types.FunctionResponse(id="call_1", name="tool", response={"r": "1"})
  fc2 = types.FunctionCall(id="call_2", name="tool", args={"q": "2"})
  fr2 = types.FunctionResponse(id="call_2", name="tool", response={"r": "2"})

  events = [
      Event(
          invocation_id="inv1",
          author="user",
          content=types.UserContent("Query"),
      ),
      Event(
          invocation_id="inv2",
          author="test_agent",
          content=types.ModelContent([types.Part(function_call=fc1)]),
      ),
      Event(
          invocation_id="inv3",
          author="user",
          content=types.UserContent([types.Part(function_response=fr1)]),
      ),
      Event(
          invocation_id="inv4",
          author="test_agent",
          content=types.ModelContent([types.Part(function_call=fc2)]),
      ),
      Event(
          invocation_id="inv5",
          author="user",
          content=types.UserContent([types.Part(function_response=fr2)]),
      ),
      # Text content after interleaved calls
      Event(
          invocation_id="inv6",
          author="test_agent",
          content=types.ModelContent("Here are the results"),
      ),
      Event(
          invocation_id="inv7",
          author="user",
          content=types.UserContent("Thanks!"),
      ),
  ]
  invocation_context.session.events = events

  async for _ in contents.request_processor.run_async(
      invocation_context, llm_request
  ):
    pass

  # Verify merge happened and text content is preserved
  assert len(llm_request.contents) == 5
  assert llm_request.contents[0] == types.UserContent("Query")

  # Merged function calls
  assert llm_request.contents[1].role == "model"
  assert len(llm_request.contents[1].parts) == 2

  # Merged function responses
  assert llm_request.contents[2].role == "user"
  assert len(llm_request.contents[2].parts) == 2

  # Text content preserved
  assert llm_request.contents[3] == types.ModelContent("Here are the results")
  assert llm_request.contents[4] == types.UserContent("Thanks!")
