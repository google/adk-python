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

from google.adk.agents.llm_agent import Agent
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.flows.llm_flows import contents
from google.adk.flows.llm_flows.contents import request_processor
from google.adk.flows.llm_flows.functions import REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
from google.adk.flows.llm_flows.functions import REQUEST_EUC_FUNCTION_CALL_NAME
from google.adk.models.google_llm import Gemini
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
      # Event with content that has only empty text part
      Event(
          invocation_id="inv6",
          author="user",
          content=types.Content(parts=[types.Part(text="")], role="model"),
      ),
      # Event with content that has multiple empty text parts
      Event(
          invocation_id="inv6_2",
          author="user",
          content=types.Content(
              parts=[types.Part(text=""), types.Part(text="")], role="model"
          ),
      ),
      # Event with content that has only inline data part
      Event(
          invocation_id="inv7",
          author="user",
          content=types.Content(
              parts=[
                  types.Part(
                      inline_data=types.Blob(
                          data=b"test", mime_type="image/png"
                      )
                  )
              ],
              role="user",
          ),
      ),
      # Event with content that has only file data part
      Event(
          invocation_id="inv8",
          author="user",
          content=types.Content(
              parts=[
                  types.Part(
                      file_data=types.FileData(
                          file_uri="gs://test_bucket/test_file.png",
                          mime_type="image/png",
                      )
                  )
              ],
              role="user",
          ),
      ),
      # Event with mixed empty and non-empty text parts
      Event(
          invocation_id="inv9",
          author="user",
          content=types.Content(
              parts=[types.Part(text=""), types.Part(text="Mixed content")],
              role="user",
          ),
      ),
      # Event with content that has executable code part
      Event(
          invocation_id="inv10",
          author="test_agent",
          content=types.Content(
              parts=[
                  types.Part(
                      executable_code=types.ExecutableCode(
                          code="print('hello')",
                          language="PYTHON",
                      )
                  )
              ],
              role="model",
          ),
      ),
      # Event with content that has code execution result part
      Event(
          invocation_id="inv11",
          author="test_agent",
          content=types.Content(
              parts=[
                  types.Part(
                      code_execution_result=types.CodeExecutionResult(
                          outcome="OUTCOME_OK",
                          output="hello",
                      )
                  )
              ],
              role="model",
          ),
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
      types.Content(
          parts=[
              types.Part(
                  inline_data=types.Blob(data=b"test", mime_type="image/png")
              )
          ],
          role="user",
      ),
      types.Content(
          parts=[
              types.Part(
                  file_data=types.FileData(
                      file_uri="gs://test_bucket/test_file.png",
                      mime_type="image/png",
                  )
              )
          ],
          role="user",
      ),
      types.Content(
          parts=[types.Part(text=""), types.Part(text="Mixed content")],
          role="user",
      ),
      types.Content(
          parts=[
              types.Part(
                  executable_code=types.ExecutableCode(
                      code="print('hello')",
                      language="PYTHON",
                  )
              )
          ],
          role="model",
      ),
      types.Content(
          parts=[
              types.Part(
                  code_execution_result=types.CodeExecutionResult(
                      outcome="OUTCOME_OK",
                      output="hello",
                  )
              )
          ],
          role="model",
      ),
  ]


@pytest.mark.asyncio
async def test_code_execution_result_events_are_not_skipped():
  """Test that events with code execution result are not skipped.

  This is a regression test for the endless loop bug where code executor
  outputs were not passed to the LLM because the events were incorrectly
  filtered as empty.
  """
  agent = Agent(model="gemini-2.5-flash", name="test_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  events = [
      Event(
          invocation_id="inv1",
          author="user",
          content=types.UserContent("Write code to calculate factorial"),
      ),
      # Model generates code
      Event(
          invocation_id="inv2",
          author="test_agent",
          content=types.Content(
              parts=[
                  types.Part(text="Here's the code:"),
                  types.Part(
                      executable_code=types.ExecutableCode(
                          code=(
                              "def factorial(n):\n  return 1 if n <= 1 else n *"
                              " factorial(n-1)\nprint(factorial(5))"
                          ),
                          language="PYTHON",
                      )
                  ),
              ],
              role="model",
          ),
      ),
      # Code execution result
      Event(
          invocation_id="inv3",
          author="test_agent",
          content=types.Content(
              parts=[
                  types.Part(
                      code_execution_result=types.CodeExecutionResult(
                          outcome="OUTCOME_OK",
                          output="120",
                      )
                  )
              ],
              role="model",
          ),
      ),
  ]
  invocation_context.session.events = events

  # Process the request
  async for _ in contents.request_processor.run_async(
      invocation_context, llm_request
  ):
    pass

  # Verify all three events are included, especially the code execution result
  assert len(llm_request.contents) == 3
  assert llm_request.contents[0] == types.UserContent(
      "Write code to calculate factorial"
  )
  # Second event has executable code
  assert llm_request.contents[1].parts[1].executable_code is not None
  # Third event has code execution result - this was the bug!
  assert llm_request.contents[2].parts[0].code_execution_result is not None
  assert llm_request.contents[2].parts[0].code_execution_result.output == "120"


@pytest.mark.asyncio
async def test_code_execution_result_not_in_first_part_is_not_skipped():
  """Test that code execution results aren't skipped.

  This covers results that appear in a non-first part.
  """
  agent = Agent(model="gemini-2.5-flash", name="test_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  events = [
      Event(
          invocation_id="inv1",
          author="user",
          content=types.UserContent("Run some code."),
      ),
      Event(
          invocation_id="inv2",
          author="test_agent",
          content=types.Content(
              parts=[
                  types.Part(text=""),
                  types.Part(
                      code_execution_result=types.CodeExecutionResult(
                          outcome="OUTCOME_OK",
                          output="42",
                      )
                  ),
              ],
              role="model",
          ),
      ),
  ]
  invocation_context.session.events = events

  async for _ in contents.request_processor.run_async(
      invocation_context, llm_request
  ):
    pass

  assert len(llm_request.contents) == 2
  assert any(
      part.code_execution_result is not None
      and part.code_execution_result.output == "42"
      for part in llm_request.contents[1].parts
  )


@pytest.mark.asyncio
async def test_function_call_with_thought_not_filtered():
  """Test that function calls marked as thought are not filtered out.

  Some models (e.g., Gemini 3 Flash Preview) may mark function calls as
  thought=True. These should still be included in the context because they
  represent actions that need to be executed.
  """
  agent = Agent(model="gemini-2.5-flash", name="test_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  # Create event with function call marked as thought (as Gemini 3 Flash does)
  function_call = types.FunctionCall(
      id="fc_123",
      name="test_tool",
      args={"query": "test"},
  )
  fc_part = types.Part(function_call=function_call)
  # Simulate model marking function call as thought
  fc_part.thought = True

  events = [
      Event(
          invocation_id="inv1",
          author="user",
          content=types.UserContent("Call the tool"),
      ),
      Event(
          invocation_id="inv2",
          author="test_agent",
          content=types.Content(
              role="model",
              parts=[
                  types.Part(text="Let me think about this", thought=True),
                  fc_part,  # Function call with thought=True
                  types.Part(text="Planning next steps", thought=True),
              ],
          ),
      ),
      Event(
          invocation_id="inv3",
          author="test_agent",
          content=types.Content(
              role="user",
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          id="fc_123",
                          name="test_tool",
                          response={"result": "success"},
                      )
                  )
              ],
          ),
      ),
  ]
  invocation_context.session.events = events

  async for _ in contents.request_processor.run_async(
      invocation_context, llm_request
  ):
    pass

  # Verify all 3 contents are present (user, model with FC, function response)
  assert len(llm_request.contents) == 3

  # Verify the function call is included (not filtered out)
  model_content = llm_request.contents[1]
  assert model_content.role == "model"
  fc_parts = [p for p in model_content.parts if p.function_call]
  assert len(fc_parts) == 1
  assert fc_parts[0].function_call.name == "test_tool"
  assert fc_parts[0].function_call.id == "fc_123"


@pytest.mark.asyncio
async def test_function_response_with_thought_not_filtered():
  """Test that function responses marked as thought are not filtered out."""
  agent = Agent(model="gemini-2.5-flash", name="test_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  function_call = types.FunctionCall(
      id="fc_456",
      name="calc_tool",
      args={"x": 1, "y": 2},
  )
  function_response = types.FunctionResponse(
      id="fc_456",
      name="calc_tool",
      response={"result": 3},
  )
  fr_part = types.Part(function_response=function_response)
  # Simulate marking function response as thought
  fr_part.thought = True

  events = [
      Event(
          invocation_id="inv1",
          author="user",
          content=types.UserContent("Calculate 1+2"),
      ),
      Event(
          invocation_id="inv2",
          author="test_agent",
          content=types.Content(
              role="model",
              parts=[types.Part(function_call=function_call)],
          ),
      ),
      Event(
          invocation_id="inv3",
          author="test_agent",
          content=types.Content(
              role="user",
              parts=[fr_part],  # Function response with thought=True
          ),
      ),
  ]
  invocation_context.session.events = events

  async for _ in contents.request_processor.run_async(
      invocation_context, llm_request
  ):
    pass

  # Verify all 3 contents are present
  assert len(llm_request.contents) == 3

  # Verify the function response is included (not filtered out)
  fr_content = llm_request.contents[2]
  fr_parts = [p for p in fr_content.parts if p.function_response]
  assert len(fr_parts) == 1
  assert fr_parts[0].function_response.name == "calc_tool"


@pytest.mark.asyncio
async def test_adk_function_call_ids_are_stripped_for_non_interactions_model():
  """Test ADK generated ids are removed for non-interactions requests."""
  agent = Agent(model="gemini-2.5-flash", name="test_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  function_call_id = "adk-test-call-id"
  events = [
      Event(
          invocation_id="inv1",
          author="user",
          content=types.UserContent("Call the tool"),
      ),
      Event(
          invocation_id="inv2",
          author="test_agent",
          content=types.Content(
              role="model",
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          id=function_call_id,
                          name="test_tool",
                          args={"x": 1},
                      )
                  )
              ],
          ),
      ),
      Event(
          invocation_id="inv3",
          author="test_agent",
          content=types.Content(
              role="user",
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          id=function_call_id,
                          name="test_tool",
                          response={"result": 2},
                      )
                  )
              ],
          ),
      ),
  ]
  invocation_context.session.events = events

  async for _ in contents.request_processor.run_async(
      invocation_context, llm_request
  ):
    pass

  model_fc_part = llm_request.contents[1].parts[0]
  assert model_fc_part.function_call is not None
  assert model_fc_part.function_call.id is None

  user_fr_part = llm_request.contents[2].parts[0]
  assert user_fr_part.function_response is not None
  assert user_fr_part.function_response.id is None


@pytest.mark.asyncio
async def test_adk_function_call_ids_preserved_for_interactions_model():
  """Test ADK generated ids are preserved for interactions requests."""
  agent = Agent(
      model=Gemini(
          model="gemini-2.5-flash",
          use_interactions_api=True,
      ),
      name="test_agent",
  )
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  function_call_id = "adk-test-call-id"
  events = [
      Event(
          invocation_id="inv1",
          author="user",
          content=types.UserContent("Call the tool"),
      ),
      Event(
          invocation_id="inv2",
          author="test_agent",
          content=types.Content(
              role="model",
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          id=function_call_id,
                          name="test_tool",
                          args={"x": 1},
                      )
                  )
              ],
          ),
      ),
      Event(
          invocation_id="inv3",
          author="test_agent",
          content=types.Content(
              role="user",
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          id=function_call_id,
                          name="test_tool",
                          response={"result": 2},
                      )
                  )
              ],
          ),
      ),
  ]
  invocation_context.session.events = events

  async for _ in contents.request_processor.run_async(
      invocation_context, llm_request
  ):
    pass

  model_fc_part = llm_request.contents[1].parts[0]
  assert model_fc_part.function_call is not None
  assert model_fc_part.function_call.id == function_call_id

  user_fr_part = llm_request.contents[2].parts[0]
  assert user_fr_part.function_response is not None
  assert user_fr_part.function_response.id == function_call_id
