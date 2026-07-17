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

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from google.adk.flows.llm_flows import functions
from google.adk.flows.llm_flows.request_confirmation import request_processor
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.genai import types
import pytest

from ... import testing_utils

MOCK_TOOL_NAME = "mock_tool"
MOCK_FUNCTION_CALL_ID = "mock_function_call_id"
MOCK_CONFIRMATION_FUNCTION_CALL_ID = "mock_confirmation_function_call_id"


def append_tool_call_event(
    invocation_context: InvocationContext,
    function_call: types.FunctionCall,
    author: str | None = None,
    branch: str | None = None,
) -> Event:
  """Helper to append a tool call event to the session history."""
  if author is None:
    author = (
        invocation_context.agent.name
        if invocation_context.agent
        else "test_agent"
    )
  event = Event(
      author=author,
      branch=branch,
      content=types.Content(parts=[types.Part(function_call=function_call)]),
  )
  invocation_context.session.events.append(event)
  return event


def append_tool_response_event(
    invocation_context: InvocationContext,
    function_response: types.FunctionResponse,
    author: str = "user",
    branch: str | None = None,
) -> Event:
  """Helper to append a tool response event to the session history."""
  event = Event(
      author=author,
      branch=branch,
      content=types.Content(
          parts=[types.Part(function_response=function_response)]
      ),
  )
  invocation_context.session.events.append(event)
  return event


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
@pytest.mark.asyncio
async def test_request_confirmation_processor_success():
  """Test the successful processing of a tool confirmation."""
  agent = LlmAgent(
      name="test_agent",
      tools=[FunctionTool(mock_tool, require_confirmation=True)],
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  llm_request = LlmRequest()

  original_function_call = types.FunctionCall(
      name=MOCK_TOOL_NAME, args={"param1": "test"}, id=MOCK_FUNCTION_CALL_ID
  )

  # Event with the original tool call
  append_tool_call_event(invocation_context, original_function_call)

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
  append_tool_call_event(
      invocation_context,
      types.FunctionCall(
          name=functions.REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
          args=tool_confirmation_args,
          id=MOCK_CONFIRMATION_FUNCTION_CALL_ID,
      ),
  )

  # Event with the user's confirmation
  user_confirmation = ToolConfirmation(confirmed=True)
  append_tool_response_event(
      invocation_context,
      types.FunctionResponse(
          name=functions.REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
          id=MOCK_CONFIRMATION_FUNCTION_CALL_ID,
          response={"response": user_confirmation.model_dump_json()},
      ),
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
  agent = LlmAgent(
      name="test_agent",
      tools=[FunctionTool(mock_tool, require_confirmation=True)],
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  llm_request = LlmRequest()

  original_function_call = types.FunctionCall(
      name=MOCK_TOOL_NAME, args={"param1": "test"}, id=MOCK_FUNCTION_CALL_ID
  )

  # Event with the original tool call
  append_tool_call_event(invocation_context, original_function_call)

  tool_confirmation = ToolConfirmation(confirmed=False, hint="test hint")
  tool_confirmation_args = {
      "originalFunctionCall": original_function_call.model_dump(
          exclude_none=True, by_alias=True
      ),
      "toolConfirmation": tool_confirmation.model_dump(
          by_alias=True, exclude_none=True
      ),
  }

  append_tool_call_event(
      invocation_context,
      types.FunctionCall(
          name=functions.REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
          args=tool_confirmation_args,
          id=MOCK_CONFIRMATION_FUNCTION_CALL_ID,
      ),
  )

  user_confirmation = ToolConfirmation(confirmed=False)
  append_tool_response_event(
      invocation_context,
      types.FunctionResponse(
          name=functions.REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
          id=MOCK_CONFIRMATION_FUNCTION_CALL_ID,
          response={"response": user_confirmation.model_dump_json()},
      ),
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


@pytest.mark.asyncio
async def test_request_confirmation_processor_finds_user_confirmation_in_default_branch():
  """Processor finds user confirmation in default branch when agent is in child branch.

  Setup:
    - Agent in 'child_branch'.
    - RequestConfirmation event in 'child_branch'.
    - User response event in default branch (None).
  Act: Run request_processor.
  Assert: Processor finds the response and triggers tool execution.
  """
  # Arrange
  agent = LlmAgent(
      name="test_agent",
      tools=[FunctionTool(mock_tool, require_confirmation=True)],
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  # Set branch for the agent context
  invocation_context.branch = "child_branch"
  llm_request = LlmRequest()

  original_function_call = types.FunctionCall(
      name=MOCK_TOOL_NAME, args={"param1": "test"}, id=MOCK_FUNCTION_CALL_ID
  )

  # Event with the original tool call
  append_tool_call_event(
      invocation_context, original_function_call, branch="child_branch"
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

  # Event with the request for confirmation (in child branch)
  append_tool_call_event(
      invocation_context,
      types.FunctionCall(
          name=functions.REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
          args=tool_confirmation_args,
          id=MOCK_CONFIRMATION_FUNCTION_CALL_ID,
      ),
      branch="child_branch",
  )

  # Event with the user's confirmation (in default branch, branch=None)
  user_confirmation = ToolConfirmation(confirmed=True)
  append_tool_response_event(
      invocation_context,
      types.FunctionResponse(
          name=functions.REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
          id=MOCK_CONFIRMATION_FUNCTION_CALL_ID,
          response={"response": user_confirmation.model_dump_json()},
      ),
      branch=None,
  )

  expected_event = Event(
      author="agent",
      branch="child_branch",
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

  # Act & Assert
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tools, original_args, confirmation_args, expected_exception_match",
    [
        (
            [],
            {"param1": "test"},
            {"param1": "test"},
            "is not registered",
        ),
        (
            [FunctionTool(mock_tool, require_confirmation=False)],
            {"param1": "test"},
            {"param1": "test"},
            "does not require confirmation",
        ),
        (
            [FunctionTool(mock_tool, require_confirmation=True)],
            {"param1": "test"},
            {"param1": "tampered"},
            "arguments mismatch",
        ),
    ],
)
async def test_request_confirmation_processor_rejections(
    tools, original_args, confirmation_args, expected_exception_match
):
  """Test various validation rejections in request confirmation processor."""
  agent = LlmAgent(name="test_agent", tools=tools)
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  llm_request = LlmRequest()

  original_function_call = types.FunctionCall(
      name=MOCK_TOOL_NAME, args=original_args, id=MOCK_FUNCTION_CALL_ID
  )

  # Event with original function call
  append_tool_call_event(invocation_context, original_function_call)

  # Confirmation request
  confirmation_function_call = types.FunctionCall(
      name=MOCK_TOOL_NAME, args=confirmation_args, id=MOCK_FUNCTION_CALL_ID
  )
  tool_confirmation = ToolConfirmation(confirmed=False, hint="test hint")
  tool_confirmation_args = {
      "originalFunctionCall": confirmation_function_call.model_dump(
          exclude_none=True, by_alias=True
      ),
      "toolConfirmation": tool_confirmation.model_dump(
          by_alias=True, exclude_none=True
      ),
  }

  append_tool_call_event(
      invocation_context,
      types.FunctionCall(
          name=functions.REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
          args=tool_confirmation_args,
          id=MOCK_CONFIRMATION_FUNCTION_CALL_ID,
      ),
  )

  user_confirmation = ToolConfirmation(confirmed=True)
  append_tool_response_event(
      invocation_context,
      types.FunctionResponse(
          name=functions.REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
          id=MOCK_CONFIRMATION_FUNCTION_CALL_ID,
          response={"response": user_confirmation.model_dump_json()},
      ),
  )

  with pytest.raises(ValueError, match=expected_exception_match):
    async for _ in request_processor.run_async(invocation_context, llm_request):
      pass
