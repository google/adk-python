import pytest
from google.genai import types

from google.adk import Agent
from google.adk.approval.approval_grant import ApprovalAction, ApprovalActor, ApprovalGrant
from google.adk.approval.approval_policy import TOOL_NAMESPACE, resource_parameter_map, resource_parameters, tool_policy
from google.adk.approval.approval_request import ApprovalResponse
from google.adk.approval.approval_request_processor import REQUEST_APPROVAL_FUNCTION_CALL_NAME
from ... import utils
from ...testing_utils import InMemoryRunner, MockModel


def test_call_approval_requested():
  responses = [
      types.Part.from_function_call(name="increase_by_one", args={"x": 1}),
      "response1",
      "response2",
  ]
  mock_model = MockModel.create(responses=responses)

  @tool_policy(
      actions=[ApprovalAction("mutate_numbers")],
      resources=lambda args: [f"{TOOL_NAMESPACE}:integers"],
  )
  def increase_by_one(x: int) -> int:
    return x + 1

  agent = Agent(name="root_agent", model=mock_model, tools=[increase_by_one])
  runner = InMemoryRunner(agent)
  preapproval_events = runner.run("test")
  assert len(preapproval_events) == 4

  state = runner.session.state
  for event in preapproval_events:
    if event.actions and event.actions.state_delta:
      state.update(event.actions.state_delta or {})
  (
      function_call_event,
      approval_request_event,
      function_response_event,
      model_summary_event,
  ) = preapproval_events

  assert len(function_call_event.content.parts) == 1
  assert (
      function_call_event.content.parts[0].function_call.name
      == "increase_by_one"
  )

  assert len(approval_request_event.content.parts) == 1
  assert (
      approval_request_event.content.parts[0].function_call.name
      == REQUEST_APPROVAL_FUNCTION_CALL_NAME
  )

  assert (
      function_call_event.content.parts[0].function_call.id
      != approval_request_event.content.parts[0].function_call.id
  )
  assert (
      function_call_event.content.parts[0].function_call.id
      == approval_request_event.content.parts[0].function_call.args[
          "function_call"
      ]["id"]
  )

  assert len(function_response_event.content.parts) == 1
  assert (
      function_response_event.content.parts[0].function_response.response[
          "status"
      ]
      == "approval_requested"
  )
  assert (
      function_call_event.content.parts[0].function_call.id
      == function_response_event.content.parts[0].function_response.id
  )

  assert len(model_summary_event.content.parts) == 1
  assert model_summary_event.content.parts[0].text == "response1"

  assert (
      function_call_event.content.parts[0].function_call.id
      in runner.session.state["approvals__suspended_function_calls"][0][
          "function_call"
      ]["id"]
  )


@pytest.mark.asyncio
async def test_call_function_resumed():
  responses = [
      types.Part.from_function_call(name="increase_by_one", args={"x": 1}),
      "response1",
      "response2",
      "response3",
  ]
  mock_model = MockModel.create(responses=responses)

  @tool_policy(
      actions=[ApprovalAction("mutate_numbers")],
      resources=lambda args: [f"{TOOL_NAMESPACE}:integers"],
  )
  def increase_by_one(x: int) -> int:
    return x + 1

  agent = Agent(name="root_agent", model=mock_model, tools=[increase_by_one])
  runner = InMemoryRunner(agent)
  preapproval_events = runner.run("test")

  (
      function_call_event,
      approval_request_event,
      function_response_event,
      model_summary_event,
  ) = preapproval_events

  assert (
      function_call_event.content.parts[0].function_call.name
      == "increase_by_one"
  )
  assert (
      approval_request_event.content.parts[0].function_call.name
      == REQUEST_APPROVAL_FUNCTION_CALL_NAME
  )
  assert (
      function_response_event.content.parts[0].function_response.response[
          "status"
      ]
      == "approval_requested"
  )
  assert model_summary_event.content.parts[0].text == "response1"

  assert (
      function_call_event.content.parts[0].function_call.id
      in runner.session.state["approvals__suspended_function_calls"][0][
          "function_call"
      ]["id"]
  )

  function_call_to_approve = approval_request_event.content.parts[
      0
  ].function_call.args["function_call"]
  grant = ApprovalGrant(
      effect="allow",
      actions=[ApprovalAction("mutate_numbers")],
      resources=[f"{TOOL_NAMESPACE}:integers"],
      grantee=ApprovalActor(
          type="tool",
          id=f"tool:increase_by_one:{function_call_to_approve['id']}",
          on_behalf_of=ApprovalActor(
              type="agent",
              id=runner.session_id,
              on_behalf_of=ApprovalActor(type="user", id="test_user"),
          ),
      ),
      grantor=ApprovalActor(type="user", id="test_user"),
  )
  postapproval_events = await runner.run_async(
      types.Content(
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
  )
  assert len(postapproval_events) == 3
  state_delta_event, function_response_event, summary_event = (
      postapproval_events
  )
  assert len(function_response_event.content.parts) == 1
  assert (
      function_response_event.content.parts[0].function_response.id
      == function_call_to_approve["id"]
  )
  assert (
      function_response_event.content.parts[0].function_response.name
      == "increase_by_one"
  )
  assert function_response_event.content.parts[
      0
  ].function_response.response == {"result": 2}

  assert len(summary_event.content.parts) == 1
  assert summary_event.content.parts[0].text == "response2"

  assert runner.session.state["approvals__suspended_function_calls"] == [{
      "function_call": {
          "args": {"x": 1},
          "id": function_call_event.content.parts[0].function_call.id,
          "name": "increase_by_one",
      },
      "sequence": 1,
      "status": "resumed",
  }]

  reapproval_events = await runner.run_async(
      types.Content(
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
  )
  assert reapproval_events == []


@pytest.mark.asyncio
async def test_partially_approving_challenges():
  responses = [
      types.Part.from_function_call(name="increase_by_one", args={"x": 1}),
      "response1",
      "response2",
      "response3",
  ]
  mock_model = MockModel.create(responses=responses)

  @tool_policy(
      actions=[ApprovalAction("mutate_numbers")],
      resources=lambda args: [f"{TOOL_NAMESPACE}:integers"],
  )
  @tool_policy(
      actions=[ApprovalAction("read_numbers")],
      resources=lambda args: [f"{TOOL_NAMESPACE}:integers"],
  )
  def increase_by_one(x: int) -> int:
    return x + 1

  agent = Agent(name="root_agent", model=mock_model, tools=[increase_by_one])
  runner = InMemoryRunner(agent)
  preapproval_events = await runner.run_async(
      types.Content(role="user", parts=[types.Part.from_text(text="test")])
  )

  (
      pre_function_call_event,
      pre_approval_request_event,
      pre_function_response_event,
      pre_model_summary_event,
  ) = preapproval_events

  assert (
      pre_function_call_event.content.parts[0].function_call.name
      == "increase_by_one"
  )
  assert (
      pre_approval_request_event.content.parts[0].function_call.name
      == REQUEST_APPROVAL_FUNCTION_CALL_NAME
  )
  assert (
      pre_function_response_event.content.parts[0].function_response.response[
          "status"
      ]
      == "approval_requested"
  )
  assert pre_model_summary_event.content.parts[0].text == "response1"

  assert len(runner.session.state["approvals__suspended_function_calls"]) == 1
  assert (
      pre_function_call_event.content.parts[0].function_call.id
      in runner.session.state["approvals__suspended_function_calls"][0][
          "function_call"
      ][
          "id"
      ]  # TODO filter to suspended
  )

  function_call_to_approve = pre_approval_request_event.content.parts[
      0
  ].function_call.args["function_call"]
  read_grant = ApprovalGrant(
      effect="allow",
      actions=[ApprovalAction("read_numbers")],
      resources=[f"{TOOL_NAMESPACE}:integers"],
      grantee=ApprovalActor(
          type="tool",
          id=f"tool:increase_by_one:{function_call_to_approve['id']}",
          on_behalf_of=ApprovalActor(
              type="agent",
              id=runner.session_id,
              on_behalf_of=ApprovalActor(type="user", id="test_user"),
          ),
      ),
      grantor=ApprovalActor(type="user", id="test_user"),
  )
  partial_approval_events = await runner.run_async(
      types.Content(
          role="user",
          parts=[
              types.Part.from_function_response(
                  name=REQUEST_APPROVAL_FUNCTION_CALL_NAME,
                  response=ApprovalResponse(
                      grants=[read_grant],
                  ).model_dump(),
              )
          ],
      )
  )
  (
      partial_grants_state_delta_event,
      partial_function_response_event,
      partial_summary_event,
  ) = partial_approval_events
  assert len(partial_function_response_event.content.parts) == 1
  assert (
      partial_function_response_event.content.parts[0].function_response.id
      == function_call_to_approve["id"]
  )
  assert (
      partial_function_response_event.content.parts[0].function_response.name
      == "increase_by_one"
  )
  assert (
      partial_function_response_event.content.parts[
          0
      ].function_response.response["status"]
      == "approval_requested"
  )

  assert len(partial_summary_event.content.parts) == 1
  assert partial_summary_event.content.parts[0].text == "response2"

  assert runner.session.state["approvals__suspended_function_calls"] == [{
      "function_call": function_call_to_approve,
      "sequence": 1,
      "status": "suspended",
  }]

  mutate_grant = ApprovalGrant(
      effect="allow",
      actions=[ApprovalAction("mutate_numbers")],
      resources=[f"{TOOL_NAMESPACE}:integers"],
      grantee=ApprovalActor(
          type="tool",
          id=f"tool:increase_by_one:{function_call_to_approve['id']}",
          on_behalf_of=ApprovalActor(
              type="agent",
              id=runner.session_id,
              on_behalf_of=ApprovalActor(type="user", id="test_user"),
          ),
      ),
      grantor=ApprovalActor(type="user", id="test_user"),
  )
  postapproval_events = await runner.run_async(
      types.Content(
          role="user",
          parts=[
              types.Part.from_function_response(
                  name=REQUEST_APPROVAL_FUNCTION_CALL_NAME,
                  response=ApprovalResponse(
                      grants=[mutate_grant],
                  ).model_dump(),
              )
          ],
      )
  )
  (
      post_grants_state_delta_event,
      post_function_response_event,
      post_summary_event,
  ) = postapproval_events
  assert len(post_function_response_event.content.parts) == 1
  assert (
      post_function_response_event.content.parts[0].function_response.id
      == function_call_to_approve["id"]
  )
  assert (
      post_function_response_event.content.parts[0].function_response.name
      == "increase_by_one"
  )
  assert post_function_response_event.content.parts[
      0
  ].function_response.response == {"result": 2}

  assert len(post_summary_event.content.parts) == 1
  assert post_summary_event.content.parts[0].text == "response3"

  assert runner.session.state["approvals__suspended_function_calls"] == [{
      "function_call": function_call_to_approve,
      "sequence": 1,
      "status": "resumed",
  }]

  reapproval_events = runner.run(
      types.Content(
          role="user",
          parts=[
              types.Part.from_function_response(
                  name=REQUEST_APPROVAL_FUNCTION_CALL_NAME,
                  response=ApprovalResponse(
                      grants=[mutate_grant],
                  ).model_dump(),
              )
          ],
      )
  )
  assert reapproval_events == []


async def test_parallel_call_function_resumed():
  responses = [
      [
          types.Part.from_function_call(name="increase_by_one", args={"x": 1}),
          types.Part.from_function_call(name="increase_by_one", args={"x": 5}),
      ],
      "response1",
      "response2",
      "response3",
  ]
  mock_model = MockModel.create(responses=responses)

  @tool_policy(
      actions=[ApprovalAction("mutate_numbers")],
      resources=lambda args: [f"{TOOL_NAMESPACE}:integers"],
  )
  def increase_by_one(x: int) -> int:
    return x + 1

  agent = Agent(name="root_agent", model=mock_model, tools=[increase_by_one])
  runner = InMemoryRunner(agent)
  preapproval_events = await runner.run_async(
      types.Content(role="user", parts=[types.Part.from_text(text="test")])
  )

  (
      function_call_event,
      approval_request_event,
      function_response_event0,
      model_summary_event,
  ) = preapproval_events

  assert (
      function_call_event.content.parts[0].function_call.name
      == "increase_by_one"
  )
  assert (
      approval_request_event.content.parts[0].function_call.name
      == REQUEST_APPROVAL_FUNCTION_CALL_NAME
  )
  assert (
      function_response_event0.content.parts[0].function_response.response[
          "status"
      ]
      == "approval_requested"
  )
  assert model_summary_event.content.parts[0].text == "response1"

  assert (
      function_call_event.content.parts[0].function_call.id
      in runner.session.state["approvals__suspended_function_calls"][0][
          "function_call"
      ]["id"]
  )

  function_call_to_approve0, function_call_to_approve1 = [
      part.function_call.args["function_call"] for part in approval_request_event.content.parts
  ]

  grant0, grant1 = [
      ApprovalGrant(
          effect="allow",
          actions=[ApprovalAction("mutate_numbers")],
          resources=[f"{TOOL_NAMESPACE}:integers"],
          grantee=ApprovalActor(
              type="tool",
              id=f"tool:increase_by_one:{function_call_part.function_call.args['function_call']['id']}",
              on_behalf_of=ApprovalActor(
                  type="agent",
                  id=runner.session_id,
                  on_behalf_of=ApprovalActor(type="user", id="test_user"),
              ),
          ),
          grantor=ApprovalActor(type="user", id="test_user"),
      )
      for function_call_part in approval_request_event.content.parts
  ]
  approval_events0 = await runner.run_async(
      types.Content(
          role="user",
          parts=[
              types.Part.from_function_response(
                  name=REQUEST_APPROVAL_FUNCTION_CALL_NAME,
                  response=ApprovalResponse(
                      grants=[grant0],
                  ).model_dump(),
              )
          ],
      )
  )
  assert len(approval_events0) == 3
  state_delta_event0, function_response_event0, summary_event0 = (
      approval_events0
  )
  for event in approval_events0:
    if event.actions and event.actions.state_delta:
      runner.session.state.update(event.actions.state_delta or {})
  assert len(function_response_event0.content.parts) == 2
  assert (
      function_response_event0.content.parts[0].function_response.id
      == function_call_to_approve0["id"]
  )
  assert (
      function_response_event0.content.parts[0].function_response.name
      == "increase_by_one"
  )
  assert function_response_event0.content.parts[
      0
  ].function_response.response == {"result": 2}

  assert len(summary_event0.content.parts) == 1
  assert summary_event0.content.parts[0].text == "response2"

  assert state_delta_event0.actions.state_delta["approvals__grants"] == [
      grant0.model_dump()
  ]

  assert runner.session.state["approvals__suspended_function_calls"] == [
      {
          "function_call": function_call_event.content.parts[
              0
          ].function_call.model_dump(),
          "sequence": 1,
          "status": "resumed",
      },
      {
          "function_call": function_call_event.content.parts[
              1
          ].function_call.model_dump(),
          "sequence": 2,
          "status": "suspended",
      },
  ]
  assert runner.session.state["approvals__grants"] == [grant0.model_dump()]

  approval_events1 = await runner.run_async(
      types.Content(
          role="user",
          parts=[
              types.Part.from_function_response(
                  name=REQUEST_APPROVAL_FUNCTION_CALL_NAME,
                  response=ApprovalResponse(
                      grants=[grant1],
                  ).model_dump(),
              )
          ],
      )
  )
  state_delta_event1, function_response_event1, summary_event1 = (
      approval_events1
  )
  for event in approval_events1:
    if event.actions and event.actions.state_delta:
      runner.session.state.update(event.actions.state_delta or {})

  assert runner.session.state["approvals__grants"] == [
      grant0.model_dump(),
      grant1.model_dump(),
  ]
  assert runner.session.state["approvals__suspended_function_calls"] == [
      {
          "function_call": function_call_event.content.parts[
              1
          ].function_call.model_dump(),
          "sequence": 1,
          "status": "resumed",
      },
  ]

  assert len(function_response_event1.content.parts) == 1
  assert (
      function_response_event1.content.parts[0].function_response.id
      == function_call_to_approve1["id"]
  )
  assert (
      function_response_event1.content.parts[0].function_response.name
      == "increase_by_one"
  )
  assert function_response_event1.content.parts[
      0
  ].function_response.response == {"result": 6}

  assert len(summary_event1.content.parts) == 1
  assert summary_event1.content.parts[0].text == "response3"


async def test_sequential_call_function():
  responses = [
      types.Part.from_function_call(name="increase_by_one", args={"x": 1}),
      "approval is needed",
      "tool was called",
      types.Part.from_function_call(name="increase_by_one", args={"x": 5}),
      "approval is needed again",
      "tool was called again",
  ]
  mock_model = MockModel.create(responses=responses)

  @tool_policy(
      actions=[ApprovalAction("mutate_numbers")],
      resources=lambda args: [f"{TOOL_NAMESPACE}:integers"],
  )
  def increase_by_one(x: int) -> int:
    return x + 1

  agent = Agent(name="root_agent", model=mock_model, tools=[increase_by_one])
  runner = InMemoryRunner(agent)
  preapproval_events = await runner.run_async(
      types.Content(role="user", parts=[types.Part.from_text(text="call the function with x as 1")])
  )

  (
      function_call_event0,
      approval_request_event0,
      function_response_event0,
      model_summary_event0,
  ) = preapproval_events

  assert (
      function_call_event0.content.parts[0].function_call.name
      == "increase_by_one"
  )
  assert (
      approval_request_event0.content.parts[0].function_call.name
      == REQUEST_APPROVAL_FUNCTION_CALL_NAME
  )
  assert (
      function_response_event0.content.parts[0].function_response.response[
          "status"
      ]
      == "approval_requested"
  )
  assert model_summary_event0.content.parts[0].text == "approval is needed"

  assert (
      function_call_event0.content.parts[0].function_call.id
      in runner.session.state["approvals__suspended_function_calls"][0][
          "function_call"
      ]["id"]
  )

  function_call_to_approve0 = approval_request_event0.content.parts[
      0
  ].function_call.args["function_call"]
  grant0 = ApprovalGrant(
      effect="allow",
      actions=[ApprovalAction("mutate_numbers")],
      resources=[f"{TOOL_NAMESPACE}:integers"],
      grantee=ApprovalActor(
          type="tool",
          id=f"tool:increase_by_one:{approval_request_event0.content.parts[0].function_call.args['function_call']['id']}",
          on_behalf_of=ApprovalActor(
              type="agent",
              id=runner.session_id,
              on_behalf_of=ApprovalActor(type="user", id="test_user"),
          ),
      ),
      grantor=ApprovalActor(type="user", id="test_user"),
  )
  approval_events0 = await runner.run_async(
      types.Content(
          role="user",
          parts=[
              types.Part.from_function_response(
                  name=REQUEST_APPROVAL_FUNCTION_CALL_NAME,
                  response=ApprovalResponse(
                      grants=[grant0],
                  ).model_dump(),
              )
          ],
      )
  )
  assert len(approval_events0) == 3
  state_delta_event0, function_response_event0, summary_event0 = (
      approval_events0
  )
  for event in approval_events0:
    if event.actions and event.actions.state_delta:
      runner.session.state.update(event.actions.state_delta or {})
  assert len(function_response_event0.content.parts) == 1
  assert (
      function_response_event0.content.parts[0].function_response.id
      == function_call_to_approve0["id"]
  )
  assert (
      function_response_event0.content.parts[0].function_response.name
      == "increase_by_one"
  )
  assert function_response_event0.content.parts[
      0
  ].function_response.response == {"result": 2}

  assert len(summary_event0.content.parts) == 1
  assert summary_event0.content.parts[0].text == "tool was called"

  assert state_delta_event0.actions.state_delta["approvals__grants"] == [
      grant0.model_dump()
  ]

  assert runner.session.state["approvals__suspended_function_calls"] == [
      {
          "function_call": function_call_event0.content.parts[
              0
          ].function_call.model_dump(),
          "sequence": 1,
          "status": "resumed",
      },
  ]
  assert runner.session.state["approvals__grants"] == [grant0.model_dump()]

  second_preapproval_events = await runner.run_async(
      types.Content(role="user", parts=[types.Part.from_text(text="call it again with 5")])
  )
  (
      function_call_event1,
      approval_request_event1,
      function_response_event1,
      model_summary_event1,
  ) = second_preapproval_events

  function_call_to_approve1 = approval_request_event1.content.parts[
      0
  ].function_call.args["function_call"]

  grant1 = ApprovalGrant(
      effect="allow",
      actions=[ApprovalAction("mutate_numbers")],
      resources=[f"{TOOL_NAMESPACE}:integers"],
      grantee=ApprovalActor(
          type="tool",
          id=f"tool:increase_by_one:{approval_request_event1.content.parts[0].function_call.args['function_call']['id']}",
          on_behalf_of=ApprovalActor(
              type="agent",
              id=runner.session_id,
              on_behalf_of=ApprovalActor(type="user", id="test_user"),
          ),
      ),
      grantor=ApprovalActor(type="user", id="test_user"),
  )
  approval_events1 = await runner.run_async(
      types.Content(
          role="user",
          parts=[
              types.Part.from_function_response(
                  name=REQUEST_APPROVAL_FUNCTION_CALL_NAME,
                  response=ApprovalResponse(
                      grants=[grant1],
                  ).model_dump(),
              )
          ],
      )
  )
  state_delta_event1, function_response_event1, summary_event1 = (
      approval_events1
  )
  for event in approval_events1:
    if event.actions and event.actions.state_delta:
      runner.session.state.update(event.actions.state_delta or {})

  assert runner.session.state["approvals__grants"] == [
      grant0.model_dump(),
      grant1.model_dump(),
  ]
  assert runner.session.state["approvals__suspended_function_calls"] == [
      {
          "function_call": function_call_event1.content.parts[
              0
          ].function_call.model_dump(),
          "sequence": 1,
          "status": "resumed",
      },
  ]

  assert len(function_response_event1.content.parts) == 1
  assert (
      function_response_event1.content.parts[0].function_response.id
      == function_call_to_approve1["id"]
  )
  assert (
      function_response_event1.content.parts[0].function_response.name
      == "increase_by_one"
  )
  assert function_response_event1.content.parts[
      0
  ].function_response.response == {"result": 6}

  assert len(summary_event1.content.parts) == 1
  assert summary_event1.content.parts[0].text == "tool was called again"

# TODO regression test for past tool calls being called again on second approval
# TODO validate in the tests that the message history is valid (no duplicate tool calls)
