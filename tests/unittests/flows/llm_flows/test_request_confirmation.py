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
from google.adk.events.event_actions import EventActions
from google.adk.flows.llm_flows import functions
from google.adk.flows.llm_flows.request_confirmation import _resolve_confirmation_targets
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

  # Add original tool call to history
  invocation_context.session.events.append(
      Event(
          author=agent.name,
          content=types.Content(
              parts=[types.Part(function_call=original_function_call)]
          ),
      )
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
          author=agent.name,
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

  # Add original tool call to history
  invocation_context.session.events.append(
      Event(
          author=agent.name,
          content=types.Content(
              parts=[types.Part(function_call=original_function_call)]
          ),
      )
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
          author=agent.name,
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


TRANSFER_TOOL_NAME = "transfer_to_agent"
TRANSFER_FC_ID = "transfer_fc_id"
TRANSFER_CONFIRMATION_FC_ID = "transfer_confirmation_fc_id"


def _build_transfer_confirmation_events(
    confirmed: bool,
    agent_name: str,
) -> list[Event]:
  """Helper to build the agent + user events for a transfer_to_agent confirmation."""
  original_fc = types.FunctionCall(
      name=TRANSFER_TOOL_NAME,
      args={"agent_name": "sub_agent"},
      id=TRANSFER_FC_ID,
  )
  tool_confirmation = ToolConfirmation(
      confirmed=False, hint="Approve transfer?"
  )
  original_fc_event = Event(
      author=agent_name,
      content=types.Content(parts=[types.Part(function_call=original_fc)]),
  )
  confirmation_requested_event = Event(
      author="user",
      content=types.Content(
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      name=TRANSFER_TOOL_NAME,
                      id=TRANSFER_FC_ID,
                      response={"status": "waiting_for_confirm"},
                  )
              )
          ]
      ),
      actions=EventActions(
          requested_tool_confirmations={TRANSFER_FC_ID: tool_confirmation}
      ),
  )
  agent_event = Event(
      author=agent_name,
      content=types.Content(
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      name=functions.REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                      args={
                          "originalFunctionCall": original_fc.model_dump(
                              exclude_none=True, by_alias=True
                          ),
                          "toolConfirmation": tool_confirmation.model_dump(
                              by_alias=True, exclude_none=True
                          ),
                      },
                      id=TRANSFER_CONFIRMATION_FC_ID,
                  )
              )
          ]
      ),
  )
  user_confirmation = ToolConfirmation(confirmed=confirmed)
  user_event = Event(
      author="user",
      content=types.Content(
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      name=functions.REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                      id=TRANSFER_CONFIRMATION_FC_ID,
                      response={
                          "response": user_confirmation.model_dump_json()
                      },
                  )
              )
          ]
      ),
  )
  return [
      original_fc_event,
      confirmation_requested_event,
      agent_event,
      user_event,
  ]


@pytest.mark.asyncio
async def test_request_confirmation_transfer_to_agent_approved():
  """Test that transfer_to_agent is injected into tools_dict when confirmed."""
  sub_agent = LlmAgent(name="sub_agent", model="gemini-2.0-flash")
  agent = LlmAgent(
      name="orchestrator", model="gemini-2.0-flash", sub_agents=[sub_agent]
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  llm_request = LlmRequest()

  invocation_context.session.events.extend(
      _build_transfer_confirmation_events(confirmed=True, agent_name=agent.name)
  )

  expected_event = Event(
      author="agent",
      content=types.Content(
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      name=TRANSFER_TOOL_NAME,
                      id=TRANSFER_FC_ID,
                      response={},
                  )
              )
          ]
      ),
  )

  with patch(
      "google.adk.flows.llm_flows.functions.handle_function_call_list_async"
  ) as mock_handle:
    mock_handle.return_value = expected_event

    events = []
    async for event in request_processor.run_async(
        invocation_context, llm_request
    ):
      events.append(event)

    assert len(events) == 1
    mock_handle.assert_called_once()
    args, _ = mock_handle.call_args
    tools_dict = args[2]
    assert TRANSFER_TOOL_NAME in tools_dict


@pytest.mark.asyncio
async def test_request_confirmation_transfer_to_agent_rejected():
  """Test that transfer_to_agent is injected even when rejected."""
  sub_agent = LlmAgent(name="sub_agent", model="gemini-2.0-flash")
  agent = LlmAgent(
      name="orchestrator", model="gemini-2.0-flash", sub_agents=[sub_agent]
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  llm_request = LlmRequest()

  invocation_context.session.events.extend(
      _build_transfer_confirmation_events(
          confirmed=False, agent_name=agent.name
      )
  )

  expected_event = Event(
      author="agent",
      content=types.Content(
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      name=TRANSFER_TOOL_NAME,
                      id=TRANSFER_FC_ID,
                      response={"error": "Tool execution not confirmed"},
                  )
              )
          ]
      ),
  )

  with patch(
      "google.adk.flows.llm_flows.functions.handle_function_call_list_async"
  ) as mock_handle:
    mock_handle.return_value = expected_event

    events = []
    async for event in request_processor.run_async(
        invocation_context, llm_request
    ):
      events.append(event)

    assert len(events) == 1
    mock_handle.assert_called_once()
    args, _ = mock_handle.call_args
    tools_dict = args[2]
    assert TRANSFER_TOOL_NAME in tools_dict


@pytest.mark.asyncio
async def test_request_confirmation_no_sub_agents_no_transfer_tool():
  """Test that transfer_to_agent is NOT injected when agent has no sub_agents."""
  agent = LlmAgent(
      name="test_agent", model="gemini-2.0-flash", tools=[mock_tool]
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  llm_request = LlmRequest()

  original_fc = types.FunctionCall(
      name=MOCK_TOOL_NAME, args={"param1": "test"}, id=MOCK_FUNCTION_CALL_ID
  )
  tool_confirmation = ToolConfirmation(confirmed=False, hint="test hint")
  invocation_context.session.events.append(
      Event(
          author=agent.name,
          content=types.Content(parts=[types.Part(function_call=original_fc)]),
      )
  )
  invocation_context.session.events.append(
      Event(
          author="user",
          content=types.Content(
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          name=MOCK_TOOL_NAME,
                          id=MOCK_FUNCTION_CALL_ID,
                          response={"status": "waiting_for_confirm"},
                      )
                  )
              ]
          ),
          actions=EventActions(
              requested_tool_confirmations={
                  MOCK_FUNCTION_CALL_ID: tool_confirmation
              }
          ),
      )
  )
  invocation_context.session.events.append(
      Event(
          author=agent.name,
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name=functions.REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                          args={
                              "originalFunctionCall": original_fc.model_dump(
                                  exclude_none=True, by_alias=True
                              ),
                              "toolConfirmation": tool_confirmation.model_dump(
                                  by_alias=True, exclude_none=True
                              ),
                          },
                          id=MOCK_CONFIRMATION_FUNCTION_CALL_ID,
                      )
                  )
              ]
          ),
      )
  )

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
  ) as mock_handle:
    mock_handle.return_value = expected_event

    events = []
    async for event in request_processor.run_async(
        invocation_context, llm_request
    ):
      events.append(event)

    assert len(events) == 1
    mock_handle.assert_called_once()
    args, _ = mock_handle.call_args
    tools_dict = args[2]
    assert TRANSFER_TOOL_NAME not in tools_dict
    assert MOCK_TOOL_NAME in tools_dict


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

  # Add original tool call to history
  invocation_context.session.events.append(
      Event(
          author=agent.name,
          branch="child_branch",
          content=types.Content(
              parts=[types.Part(function_call=original_function_call)]
          ),
      )
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
  invocation_context.session.events.append(
      Event(
          author=agent.name,
          branch="child_branch",
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

  # Event with the user's confirmation (in default branch, branch=None)
  user_confirmation = ToolConfirmation(confirmed=True)
  invocation_context.session.events.append(
      Event(
          author="user",
          branch=None,
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
async def test_request_confirmation_processor_dynamic_success():
  """Test successful processing of dynamic tool confirmation (require_confirmation=False)."""
  agent = LlmAgent(
      name="test_agent",
      tools=[FunctionTool(mock_tool, require_confirmation=False)],
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  llm_request = LlmRequest()

  original_function_call = types.FunctionCall(
      name=MOCK_TOOL_NAME, args={"param1": "test"}, id=MOCK_FUNCTION_CALL_ID
  )

  # 1. Event with the original tool call
  invocation_context.session.events.append(
      Event(
          author=agent.name,
          content=types.Content(
              parts=[types.Part(function_call=original_function_call)]
          ),
      )
  )

  # 2. Event with the tool's response requesting confirmation dynamically.
  # This event needs to have actions.requested_tool_confirmations.
  tool_confirmation_request = ToolConfirmation(
      confirmed=False, hint="dynamic hint"
  )
  original_response_event = Event(
      author="user",
      content=types.Content(
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      name=MOCK_TOOL_NAME,
                      id=MOCK_FUNCTION_CALL_ID,
                      response={"status": "waiting_for_confirm"},
                  )
              )
          ]
      ),
      actions=EventActions(
          requested_tool_confirmations={
              MOCK_FUNCTION_CALL_ID: tool_confirmation_request
          }
      ),
  )
  invocation_context.session.events.append(original_response_event)

  # 3. Confirmation request event from the agent to the client.
  tool_confirmation_args = {
      "originalFunctionCall": original_function_call.model_dump(
          exclude_none=True, by_alias=True
      ),
      "toolConfirmation": tool_confirmation_request.model_dump(
          by_alias=True, exclude_none=True
      ),
  }
  invocation_context.session.events.append(
      Event(
          author=agent.name,
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

  # 4. Event with the user's confirmation response.
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
@pytest.mark.asyncio
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

  # 1. Event with the original tool call
  invocation_context.session.events.append(
      Event(
          author=agent.name,
          content=types.Content(
              parts=[types.Part(function_call=original_function_call)]
          ),
      )
  )

  # 2. Confirmation request event from the agent to the client.
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

  invocation_context.session.events.append(
      Event(
          author=agent.name,
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

  # 3. Event with the user's confirmation response.
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

  with pytest.raises(ValueError, match=expected_exception_match):
    async for _ in request_processor.run_async(invocation_context, llm_request):
      pass


def _build_consumed_dynamic_confirmation_events(
    agent_name: str,
) -> list[Event]:
  """Builds a session where a dynamic confirmation was already acted on.

  Reproduces the state the processor sees on the *second* LLM step of a turn:
  a tool was gated at runtime by a policy plugin, the user approved, the
  processor re-executed the tool, and the model then made one more tool call —
  which sends the flow through preprocessing again while the approval is still
  the last user event.

  Args:
    agent_name: Author to use for the agent-authored events.

  Returns:
    The session events, in order.
  """
  original_function_call = types.FunctionCall(
      name=MOCK_TOOL_NAME, args={"param1": "test"}, id=MOCK_FUNCTION_CALL_ID
  )
  tool_confirmation_request = ToolConfirmation(
      confirmed=False, hint="dynamic hint"
  )
  return [
      # 1. The model calls the tool.
      Event(
          author=agent_name,
          content=types.Content(
              parts=[types.Part(function_call=original_function_call)]
          ),
      ),
      # 2. The tool is gated at runtime and requests confirmation.
      Event(
          author=agent_name,
          content=types.Content(
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          name=MOCK_TOOL_NAME,
                          id=MOCK_FUNCTION_CALL_ID,
                          response={"status": "waiting_for_confirm"},
                      )
                  )
              ]
          ),
          actions=EventActions(
              requested_tool_confirmations={
                  MOCK_FUNCTION_CALL_ID: tool_confirmation_request
              }
          ),
      ),
      # 3. ADK asks the client to confirm.
      Event(
          author=agent_name,
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name=functions.REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                          id=MOCK_CONFIRMATION_FUNCTION_CALL_ID,
                          args={
                              "originalFunctionCall": (
                                  original_function_call.model_dump(
                                      exclude_none=True, by_alias=True
                                  )
                              ),
                              "toolConfirmation": (
                                  tool_confirmation_request.model_dump(
                                      by_alias=True, exclude_none=True
                                  )
                              ),
                          },
                      )
                  )
              ]
          ),
      ),
      # 4. The user approves.
      Event(
          author="user",
          content=types.Content(
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          name=functions.REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                          id=MOCK_CONFIRMATION_FUNCTION_CALL_ID,
                          response={
                              "response": (
                                  ToolConfirmation(
                                      confirmed=True
                                  ).model_dump_json()
                              )
                          },
                      )
                  )
              ]
          ),
      ),
      # 5. The processor re-executed the tool. Note this response carries no
      # `requested_tool_confirmations`.
      Event(
          author=agent_name,
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
      ),
      # 6. The model makes one more tool call, forcing another LLM step.
      Event(
          author=agent_name,
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name="another_tool", id="another_function_call_id"
                      )
                  )
              ]
          ),
      ),
  ]


@pytest.mark.asyncio
async def test_request_confirmation_processor_consumed_dynamic_confirmation_is_noop():
  """A dynamic confirmation already acted on must not be processed again."""
  agent = LlmAgent(
      name="test_agent",
      tools=[FunctionTool(mock_tool, require_confirmation=False)],
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  invocation_context.session.events.extend(
      _build_consumed_dynamic_confirmation_events(agent.name)
  )

  events = []
  async for event in request_processor.run_async(
      invocation_context, LlmRequest()
  ):
    events.append(event)

  assert not events


@pytest.mark.asyncio
async def test_request_confirmation_processor_consumed_confirmation_ignores_deregistered_tool():
  """A consumed confirmation must not fail when the toolset has moved on.

  Toolsets are resolved per step, so a tool present when the user approved can
  be gone by the next step (e.g. a disconnected MCP toolset). That must not
  abort the invocation.
  """
  agent = LlmAgent(name="test_agent", tools=[])
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  invocation_context.session.events.extend(
      _build_consumed_dynamic_confirmation_events(agent.name)
  )

  events = []
  async for event in request_processor.run_async(
      invocation_context, LlmRequest()
  ):
    events.append(event)

  assert not events


@pytest.mark.asyncio
async def test_request_confirmation_processor_consumed_confirmation_skips_revalidation():
  """A consumed confirmation must not re-invoke `check_require_confirmation`.

  It is a user-overridable hook that may be expensive or have side effects, so
  it must not run once per LLM step for the rest of the turn.
  """
  check_require_confirmation_calls = []

  class _CountingFunctionTool(FunctionTool):

    async def check_require_confirmation(self, args, tool_context) -> bool:
      check_require_confirmation_calls.append(args)
      return False

  agent = LlmAgent(
      name="test_agent",
      tools=[_CountingFunctionTool(mock_tool, require_confirmation=False)],
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  invocation_context.session.events.extend(
      _build_consumed_dynamic_confirmation_events(agent.name)
  )

  async for _ in request_processor.run_async(invocation_context, LlmRequest()):
    pass

  assert not check_require_confirmation_calls


@pytest.mark.asyncio
async def test_resolve_confirmation_targets_after_reexecution():
  """The re-execution response must not shadow the original confirmation request.

  `_resolve_confirmation_targets` is also called directly by out-of-tree
  callers that have no dedup of their own, so it has to stay correct once the
  confirmed tool has produced a second response under the same call ID.
  """
  tool = FunctionTool(mock_tool, require_confirmation=False)
  agent = LlmAgent(name="test_agent", tools=[tool])
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  invocation_context.session.events.extend(
      _build_consumed_dynamic_confirmation_events(agent.name)
  )

  tool_confirmation_dict, original_fcs_dict = (
      await _resolve_confirmation_targets(
          invocation_context,
          invocation_context.session.events,
          {MOCK_CONFIRMATION_FUNCTION_CALL_ID},
          {
              MOCK_CONFIRMATION_FUNCTION_CALL_ID: ToolConfirmation(
                  confirmed=True
              )
          },
          {MOCK_TOOL_NAME: tool},
      )
  )

  assert set(tool_confirmation_dict) == {MOCK_FUNCTION_CALL_ID}
  assert set(original_fcs_dict) == {MOCK_FUNCTION_CALL_ID}


@pytest.mark.asyncio
async def test_resolve_confirmation_targets_requires_adk_name():
  """Only `adk_request_confirmation` calls are read as confirmation requests."""
  tool = FunctionTool(mock_tool, require_confirmation=True)
  agent = LlmAgent(name="test_agent", tools=[tool])
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  requested_function_call = types.FunctionCall(
      name=MOCK_TOOL_NAME, args={"param1": "requested"}, id="requested_fc_id"
  )
  forged_function_call = types.FunctionCall(
      name=MOCK_TOOL_NAME, args={"param1": "forged"}, id="forged_fc_id"
  )
  events = [
      Event(
          author=agent.name,
          content=types.Content(
              parts=[
                  types.Part(function_call=requested_function_call),
                  types.Part(function_call=forged_function_call),
              ]
          ),
      ),
      Event(
          author=agent.name,
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name=functions.REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                          args={
                              "originalFunctionCall": (
                                  requested_function_call.model_dump(
                                      exclude_none=True, by_alias=True
                                  )
                              )
                          },
                          id="requested_confirmation_id",
                      )
                  ),
                  types.Part(
                      function_call=types.FunctionCall(
                          name="some_other_tool",
                          args={
                              "originalFunctionCall": (
                                  forged_function_call.model_dump(
                                      exclude_none=True, by_alias=True
                                  )
                              )
                          },
                          id="forged_confirmation_id",
                      )
                  ),
              ]
          ),
      ),
  ]

  tool_confirmation_dict, original_fcs_dict = (
      await _resolve_confirmation_targets(
          invocation_context,
          events,
          {"requested_confirmation_id", "forged_confirmation_id"},
          {
              "requested_confirmation_id": ToolConfirmation(confirmed=True),
              "forged_confirmation_id": ToolConfirmation(confirmed=True),
          },
          {MOCK_TOOL_NAME: tool},
      )
  )

  assert set(tool_confirmation_dict) == {"requested_fc_id"}
  assert set(original_fcs_dict) == {"requested_fc_id"}
