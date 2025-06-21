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

import pytest

from google.genai import types

from google.adk import Agent
from google.adk.approval.approval_grant import ApprovalAction, ApprovalActor, ApprovalGrant
from google.adk.approval.approval_policy import TOOL_NAMESPACE, tool_policy
from google.adk.approval.approval_request import ApprovalResponse, FunctionCallStatus
from google.adk.approval.approval_request_processor import REQUEST_APPROVAL_FUNCTION_CALL_NAME
from google.adk.approval.approval_request_processor import request_processor
from google.adk.models.llm_request import LlmRequest

from .. import utils
from ..flows.llm_flows.test_functions_sequential import function_call
from ..testing_utils import InMemoryRunner, MockModel


@pytest.mark.asyncio
async def test_request_processor__no_approvals():
  @tool_policy(
      actions=[ApprovalAction("mutate_numbers")],
      resources=lambda args: [f"{TOOL_NAMESPACE}:integers"],
  )
  def increase_by_one(x: int) -> int:
    return x + 1

  agent = Agent(
      name="root_agent",
      model=MockModel.create(responses=[]),
      tools=[increase_by_one],
  )
  runner = InMemoryRunner(agent)
  session = runner.session

  function_call_to_approve = types.FunctionCall(
      id="adk-tool-id-123",
      name="increase_by_one",
      args={"x": 1},
  )
  grant = ApprovalGrant(
      effect="allow",
      actions=[ApprovalAction("mutate_numbers")],
      resources=[f"{TOOL_NAMESPACE}:integers"],
      grantee=ApprovalActor(
          type="tool",
          id=f"tool:increase_by_one:{function_call_to_approve.id}",
          on_behalf_of=ApprovalActor(
              type="agent",
              id=runner.session.id,
              on_behalf_of=ApprovalActor(type="user", id="test_user"),
          ),
      ),
      grantor=ApprovalActor(type="user", id="test_user"),
  )
  new_message = types.Content(
      role="user",
      parts=[
          types.Part.from_function_response(
              name=REQUEST_APPROVAL_FUNCTION_CALL_NAME,
              response=ApprovalResponse(
                  grants=[grant],
              ).model_dump(),
          )
      ],
  )
  invocation_context = runner.runner._new_invocation_context(
      session, new_message=new_message
  )
  await runner.runner._append_new_message_to_session(
      session,
      new_message,
      invocation_context,
  )
  assert invocation_context.session.state.get("approvals__grants", []) == []
  invocation_context.session.state["approvals__grants"] = []
  invocation_context.session.state["approvals__suspended_function_calls"] = [
      FunctionCallStatus(
          function_call=function_call_to_approve,
          status="suspended",
          sequence=0,
      )
  ]
  events = [
      event
      async for event in request_processor.run_async(
          invocation_context, llm_request=LlmRequest()
      )
  ]
  state = {**invocation_context.session.state}
  for event in events:
    if event.actions.state_delta:
      state = {**state, **event.actions.state_delta}

  assert state["approvals__grants"] == [grant.model_dump(mode="json")]

  assert len(events) == 2
  grants_state_delta_event, response_event = events

  assert not grants_state_delta_event.content
  assert grants_state_delta_event.actions.state_delta == {
      "approvals__grants": [grant.model_dump(mode="json")],
  }

  assert len(response_event.content.parts) == 1
  assert (
      response_event.content.parts[0].function_response.id
      == function_call_to_approve.id
  )
  assert (
      response_event.content.parts[0].function_response.name
      == "increase_by_one"
  )
  assert response_event.content.parts[0].function_response.response == {
      "result": 2
  }
  assert state["approvals__suspended_function_calls"] == [{
      "function_call": {
          "args": {
              "x": 1,
          },
          "id": "adk-tool-id-123",
          "name": "increase_by_one",
      },
      "sequence": 1,
      "status": "resumed",
  }]
