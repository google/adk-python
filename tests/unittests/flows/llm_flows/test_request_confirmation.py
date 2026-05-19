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

import json
from unittest.mock import patch

from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from google.adk.flows.llm_flows import functions
from google.adk.flows.llm_flows.request_confirmation import request_processor
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.genai import types
import pytest

from ... import testing_utils

MOCK_TOOL_NAME = "mock_tool"
MOCK_FUNCTION_CALL_ID = "mock_function_call_id"
MOCK_CONFIRMATION_FUNCTION_CALL_ID = "mock_confirmation_function_call_id"


def mock_tool(param1: str):
  """Mock tool function."""
  return f"Mock tool result with {param1}"


@pytest.mark.asyncio
async def test_request_confirmation_processor_no_events():
  """Test that the processor returns None when there are no events."""
  agent = LlmAgent(name="test_agent", tools=[mock_tool])
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  llm_request = LlmRequest()

  events = []
  async for event in request_processor.run_async(
      invocation_context, llm_request
  ):
    events.append(event)

  assert not events


@pytest.mark.asyncio
async def test_request_confirmation_processor_no_function_responses():
  """Test that the processor returns None when the user event has no function responses."""
  agent = LlmAgent(name="test_agent", tools=[mock_tool])
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  llm_request = LlmRequest()

  invocation_context.session.events.append(
      Event(author="user", content=types.Content())
  )

  events = []
  async for event in request_processor.run_async(
      invocation_context, llm_request
  ):
    events.append(event)

  assert not events


@pytest.mark.asyncio
async def test_request_confirmation_processor_no_confirmation_function_response():
  """Test that the processor returns None when no confirmation function response is present."""
  agent = LlmAgent(name="test_agent", tools=[mock_tool])
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  llm_request = LlmRequest()

  invocation_context.session.events.append(
      Event(
          author="user",
          content=types.Content(
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          name="other_function", response={}
                      )
                  )
              ]
          ),
      )
  )

  events = []
  async for event in request_processor.run_async(
      invocation_context, llm_request
  ):
    events.append(event)

  assert not events


@pytest.mark.asyncio
async def test_request_confirmation_processor_success():
  """Test the successful processing of a tool confirmation."""
  agent = LlmAgent(name="test_agent", tools=[mock_tool])
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  llm_request = LlmRequest()

  original_function_call = types.FunctionCall(
      name=MOCK_TOOL_NAME, args={"param1": "test"}, id=MOCK_FUNCTION_CALL_ID
  )

  tool_confirmation = ToolConfirmation(confirmed=False, hint="test hint")
  tool_confirmation_args = {
      "originalFunctionCall": original_function_call.model_dump(
          exclude_none=True, by_alias=True
      ),
      "toolConfirmation": tool_confirmation.model_dump(
          by_alias=True, exclude_none=True
      ),
  }

  # Event with the request for confirmation
  invocation_context.session.events.append(
      Event(
          author="agent",
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name=functions.REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                          args=tool_confirmation_args,
                          id=MOCK_CONFIRMATION_FUNCTION_CALL_ID,
                      )
                  )
              ]
          ),
      )
  )

  # Event with the user's confirmation
  user_confirmation = ToolConfirmation(confirmed=True)
  invocation_context.session.events.append(
      Event(
          author="user",
          content=types.Content(
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          name=functions.REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                          id=MOCK_CONFIRMATION_FUNCTION_CALL_ID,
                          response={
                              "response": user_confirmation.model_dump_json()
                          },
                      )
                  )
              ]
          ),
      )
  )

  expected_event = Event(
      author="agent",
      content=types.Content(
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      name=MOCK_TOOL_NAME,
                      id=MOCK_FUNCTION_CALL_ID,
                      response={"result": "Mock tool result with test"},
                  )
              )
          ]
      ),
  )

  with patch(
      "google.adk.flows.llm_flows.functions.handle_function_call_list_async"
  ) as mock_handle_function_call_list_async:
    mock_handle_function_call_list_async.return_value = expected_event

    events = []
    async for event in request_processor.run_async(
        invocation_context, llm_request
    ):
      events.append(event)

    assert len(events) == 1
    assert events[0] == expected_event

    mock_handle_function_call_list_async.assert_called_once()
    args, _ = mock_handle_function_call_list_async.call_args

    assert list(args[1]) == [original_function_call]  # function_calls
    assert args[3] == {MOCK_FUNCTION_CALL_ID}  # tools_to_confirm
    assert (
        args[4][MOCK_FUNCTION_CALL_ID] == user_confirmation
    )  # tool_confirmation_dict


@pytest.mark.asyncio
async def test_request_confirmation_processor_tool_not_confirmed():
  """Test when the tool execution is not confirmed by the user."""
  agent = LlmAgent(name="test_agent", tools=[mock_tool])
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  llm_request = LlmRequest()

  original_function_call = types.FunctionCall(
      name=MOCK_TOOL_NAME, args={"param1": "test"}, id=MOCK_FUNCTION_CALL_ID
  )

  tool_confirmation = ToolConfirmation(confirmed=False, hint="test hint")
  tool_confirmation_args = {
      "originalFunctionCall": original_function_call.model_dump(
          exclude_none=True, by_alias=True
      ),
      "toolConfirmation": tool_confirmation.model_dump(
          by_alias=True, exclude_none=True
      ),
  }

  invocation_context.session.events.append(
      Event(
          author="agent",
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name=functions.REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                          args=tool_confirmation_args,
                          id=MOCK_CONFIRMATION_FUNCTION_CALL_ID,
                      )
                  )
              ]
          ),
      )
  )

  user_confirmation = ToolConfirmation(confirmed=False)
  invocation_context.session.events.append(
      Event(
          author="user",
          content=types.Content(
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          name=functions.REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                          id=MOCK_CONFIRMATION_FUNCTION_CALL_ID,
                          response={
                              "response": user_confirmation.model_dump_json()
                          },
                      )
                  )
              ]
          ),
      )
  )

  with patch(
      "google.adk.flows.llm_flows.functions.handle_function_call_list_async"
  ) as mock_handle_function_call_list_async:
    mock_handle_function_call_list_async.return_value = Event(
        author="agent",
        content=types.Content(
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name=MOCK_TOOL_NAME,
                        id=MOCK_FUNCTION_CALL_ID,
                        response={"error": "Tool execution not confirmed"},
                    )
                )
            ]
        ),
    )

    events = []
    async for event in request_processor.run_async(
        invocation_context, llm_request
    ):
      events.append(event)

    assert len(events) == 1
    mock_handle_function_call_list_async.assert_called_once()
    args, _ = mock_handle_function_call_list_async.call_args
    assert (
        args[4][MOCK_FUNCTION_CALL_ID] == user_confirmation
    )  # tool_confirmation_dict
