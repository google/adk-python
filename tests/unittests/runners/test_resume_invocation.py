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
"""Tests for edge cases of resuming invocations."""

import asyncio
import copy
from typing import AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.genai import types
from google.genai.types import FunctionResponse
from google.genai.types import Part
import pytest

from .. import testing_utils


def transfer_call_part(agent_name: str) -> Part:
  return Part.from_function_call(
      name="transfer_to_agent", args={"agent_name": agent_name}
  )


TRANSFER_RESPONSE_PART = Part.from_function_response(
    name="transfer_to_agent", response={"result": None}
)


def test_tool():
  """A test tool; returns None to simulate a pending long-running operation."""
  return None


test_tool.__test__ = False


@pytest.mark.xfail(
    reason=(
        "Tests implementation details that are different in V2 and will be"
        " deprecated."
    )
)
@pytest.mark.asyncio
async def test_resume_invocation_from_sub_agent():
  """A test case for an edge case, where an invocation-to-resume starts from a sub-agent.

  For example:
    invocation1: root_agent -> sub_agent (sub_agent completes normally)
    invocation2: sub_agent calls long_running_tool -> pauses
    resume invocation2: sub_agent gets function response -> responds
  """
  # Step 1: Setup
  long_running_test_tool = LongRunningFunctionTool(func=test_tool)
  sub_agent = LlmAgent(
      name="sub_agent",
      model=testing_utils.MockModel.create(
          responses=[
              "first response from sub_agent",
              Part.from_function_call(name="test_tool", args={}),
              "response from sub_agent after resume",
          ]
      ),
      tools=[long_running_test_tool],
  )
  root_agent = LlmAgent(
      name="root_agent",
      model=testing_utils.MockModel.create(
          responses=[transfer_call_part(sub_agent.name)]
      ),
      sub_agents=[sub_agent],
  )
  runner = testing_utils.InMemoryRunner(
      app=App(
          name="test_app",
          root_agent=root_agent,
          resumability_config=ResumabilityConfig(is_resumable=True),
      )
  )

  # Step 2: Run the first invocation
  # root_agent transfers to sub_agent, sub_agent responds normally.
  invocation_1_events = await runner.run_async("test user query")
  inv1_behavioral = [
      e
      for e in testing_utils.simplify_resumable_app_events(
          copy.deepcopy(invocation_1_events)
      )
      if not isinstance(e[1], dict)
  ]
  assert inv1_behavioral == [
      (
          root_agent.name,
          transfer_call_part(sub_agent.name),
      ),
      (
          root_agent.name,
          TRANSFER_RESPONSE_PART,
      ),
      (
          root_agent.name,
          testing_utils.END_OF_AGENT,
      ),
      (
          sub_agent.name,
          "first response from sub_agent",
      ),
      (
          sub_agent.name,
          testing_utils.END_OF_AGENT,
      ),
  ]

  # Step 3: Run the second invocation
  # sub_agent is now active. It calls long_running_tool, which pauses.
  invocation_2_events = await runner.run_async("test user query 2")
  inv2_behavioral = [
      e
      for e in testing_utils.simplify_resumable_app_events(
          copy.deepcopy(invocation_2_events)
      )
      if not isinstance(e[1], dict)
  ]
  assert inv2_behavioral == [
      # execute_tools yields the interrupt event with long_running_tool_ids.
      (
          sub_agent.name,
          Part.from_function_call(name="test_tool", args={}),
      ),
  ]

  # Find the function_call_id for resume.
  invocation_2_function_call_id = None
  for ev in invocation_2_events:
    if (
        ev.content
        and ev.content.parts
        and ev.content.parts[0].function_call
        and ev.content.parts[0].function_call.name == "test_tool"
    ):
      invocation_2_function_call_id = ev.content.parts[0].function_call.id
      break
  assert invocation_2_function_call_id is not None

  # Step 4: Resume the second invocation with function response.
  resumed_invocation_2_events = await runner.run_async(
      invocation_id=invocation_2_events[0].invocation_id,
      new_message=testing_utils.UserContent(
          Part(
              function_response=FunctionResponse(
                  id=invocation_2_function_call_id,
                  name="test_tool",
                  response={"result": "test tool update"},
              )
          ),
      ),
  )
  resumed_inv2_behavioral = [
      e
      for e in testing_utils.simplify_resumable_app_events(
          copy.deepcopy(resumed_invocation_2_events)
      )
      if not isinstance(e[1], dict)
  ]
  assert resumed_inv2_behavioral == [
      # execute_tools yields the function response from resume.
      (
          sub_agent.name,
          Part.from_function_response(
              name="test_tool",
              response={"result": "test tool update"},
          ),
      ),
      (
          sub_agent.name,
          "response from sub_agent after resume",
      ),
      (sub_agent.name, testing_utils.END_OF_AGENT),
  ]


@pytest.mark.skip(
    reason=(
        "Cross-invocation resume (resuming a non-latest invocation) is not"
        " supported by the Mesh-based LlmAgent. The Mesh's output aggregation"
        " in node_output_utils.py collects events from multiple invocations,"
        " causing CallLlmResult to be wrapped in a list."
    )
)
@pytest.mark.asyncio
async def test_resume_any_invocation():
  """A test case for resuming a previous invocation instead of the last one."""
  # Step 1: Setup
  long_running_test_tool = LongRunningFunctionTool(
      func=test_tool,
  )
  root_agent = LlmAgent(
      name="root_agent",
      model=testing_utils.MockModel.create(
          responses=[
              Part.from_function_call(name="test_tool", args={}),
              "llm response in invocation 2",
              Part.from_function_call(name="test_tool", args={}),
              "llm response after resuming invocation 1",
          ]
      ),
      tools=[long_running_test_tool],
  )
  runner = testing_utils.InMemoryRunner(
      app=App(
          name="test_app",
          root_agent=root_agent,
          resumability_config=ResumabilityConfig(is_resumable=True),
      )
  )

  # Step 2: Run the first invocation, which pauses on the long running function.
  invocation_1_events = await runner.run_async("test user query")
  inv1_behavioral = [
      e
      for e in testing_utils.simplify_resumable_app_events(
          copy.deepcopy(invocation_1_events)
      )
      if not isinstance(e[1], dict)
  ]
  assert inv1_behavioral == [
      (
          root_agent.name,
          Part.from_function_call(name="test_tool", args={}),
      ),
  ]

  # Find the function_call_id for resume.
  invocation_1_function_call_id = None
  for ev in invocation_1_events:
    if (
        ev.content
        and ev.content.parts
        and ev.content.parts[0].function_call
        and ev.content.parts[0].function_call.name == "test_tool"
    ):
      invocation_1_function_call_id = ev.content.parts[0].function_call.id
      break
  assert invocation_1_function_call_id is not None

  # Step 3: Run the second invocation, expect it to finish normally.
  invocation_2_events = await runner.run_async(
      "test user query 2",
  )
  inv2_behavioral = [
      e
      for e in testing_utils.simplify_resumable_app_events(
          copy.deepcopy(invocation_2_events)
      )
      if not isinstance(e[1], dict)
  ]
  assert inv2_behavioral == [
      (
          root_agent.name,
          "llm response in invocation 2",
      ),
      (root_agent.name, testing_utils.END_OF_AGENT),
  ]

  # Step 4: Run the third invocation, which also pauses on the long running
  # function.
  invocation_3_events = await runner.run_async(
      "test user query 3",
  )
  inv3_behavioral = [
      e
      for e in testing_utils.simplify_resumable_app_events(
          copy.deepcopy(invocation_3_events)
      )
      if not isinstance(e[1], dict)
  ]
  assert inv3_behavioral == [
      (
          root_agent.name,
          Part.from_function_call(name="test_tool", args={}),
      ),
  ]

  # Step 5: Resume the first invocation with long running function response.
  resumed_invocation_1_events = await runner.run_async(
      invocation_id=invocation_1_events[0].invocation_id,
      new_message=testing_utils.UserContent(
          Part(
              function_response=FunctionResponse(
                  id=invocation_1_function_call_id,
                  name="test_tool",
                  response={"result": "test tool update"},
              )
          ),
      ),
  )
  resumed_inv1_behavioral = [
      e
      for e in testing_utils.simplify_resumable_app_events(
          copy.deepcopy(resumed_invocation_1_events)
      )
      if not isinstance(e[1], dict)
  ]
  assert resumed_inv1_behavioral == [
      (
          root_agent.name,
          "llm response after resuming invocation 1",
      ),
      (root_agent.name, testing_utils.END_OF_AGENT),
  ]


@pytest.mark.asyncio
async def test_resumable_parallel_agent_escalation_short_circuits_persisted_run():
  """Runner persists fast+escalating events and marks the parent run complete."""

  class _ParallelEscalationTestingAgent(BaseAgent):
    """A testing agent that emits a single event after a delay."""

    delay: float = 0
    response_text: str = ""
    escalate: bool = False
    emit_follow_up_after_first_event: bool = False

    def _create_event(
        self,
        ctx: InvocationContext,
        text: str,
        *,
        escalate: bool = False,
    ) -> Event:
      return Event(
          author=self.name,
          branch=ctx.branch,
          invocation_id=ctx.invocation_id,
          content=types.Content(role="model", parts=[types.Part(text=text)]),
          actions=EventActions(escalate=True) if escalate else EventActions(),
      )

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
      await asyncio.sleep(self.delay)
      yield self._create_event(ctx, self.response_text, escalate=self.escalate)
      if self.emit_follow_up_after_first_event:
        yield self._create_event(ctx, "This event should not be emitted.")

  fast_agent = _ParallelEscalationTestingAgent(
      name="fast_agent",
      delay=0.05,
      response_text="fast response",
  )
  escalating_agent = _ParallelEscalationTestingAgent(
      name="escalating_agent",
      delay=0.1,
      response_text="escalating response",
      escalate=True,
      emit_follow_up_after_first_event=True,
  )
  slow_agent = _ParallelEscalationTestingAgent(
      name="slow_agent",
      delay=0.5,
      response_text="slow response",
  )
  runner = testing_utils.InMemoryRunner(
      app=App(
          name="test_app",
          root_agent=ParallelAgent(
              name="root_agent",
              sub_agents=[fast_agent, escalating_agent, slow_agent],
          ),
          resumability_config=ResumabilityConfig(is_resumable=True),
      )
  )

  invocation_events = await runner.run_async("test user query")
  simplified_events = testing_utils.simplify_resumable_app_events(
      copy.deepcopy(invocation_events)
  )

  assert simplified_events == [
      ("root_agent", {}),
      ("fast_agent", "fast response"),
      ("escalating_agent", "escalating response"),
      ("root_agent", testing_utils.END_OF_AGENT),
  ]

  session = await runner.runner.session_service.get_session(
      app_name=runner.app_name,
      user_id="test_user",
      session_id=runner.session_id,
  )
  persisted_events = [
      event
      for event in session.events
      if event.invocation_id == invocation_events[0].invocation_id
      and event.author != "user"
  ]
  assert (
      testing_utils.simplify_resumable_app_events(
          copy.deepcopy(persisted_events)
      )
      == simplified_events
  )
  assert all(event.author != "slow_agent" for event in persisted_events)

  # A completed resumable invocation should not restart cancelled siblings.
  assert not await runner.run_async(
      invocation_id=invocation_events[0].invocation_id
  )


@pytest.mark.parametrize("resumable", [False, True])
@pytest.mark.parametrize("auth_stage", ["tool", "toolset"])
@pytest.mark.parametrize("nested", [False, True])
async def test_auth_response_resumes_restricted_sub_agent(
    resumable, auth_stage, nested
):
  """Authentication resumes its owner even when transfer to its parent is disabled.

  Setup: transfer to a restricted child that requests an OIDC credential.
  Act: return the credential using the emitted authentication call ID.
  Assert: the protected tool succeeds once, without another auth request;
    a subsequent ordinary user message still returns to the root agent.
  """
  from google.adk.auth.auth_credential import AuthCredential
  from google.adk.auth.auth_credential import OAuth2Auth
  from google.adk.auth.auth_schemes import OpenIdConnectWithConfig
  from google.adk.auth.auth_tool import AuthConfig
  from google.adk.runners import Runner
  from google.adk.sessions.in_memory_session_service import InMemorySessionService
  from google.adk.tools.base_toolset import BaseToolset
  from google.adk.tools.function_tool import FunctionTool
  from google.adk.tools.tool_context import ToolContext

  auth_config = AuthConfig(
      auth_scheme=OpenIdConnectWithConfig(
          authorization_endpoint="https://issuer.example/authorize",
          token_endpoint="https://issuer.example/token",
          scopes=["openid"],
      ),
      raw_auth_credential=AuthCredential(
          auth_type="oauth2",
          oauth2=OAuth2Auth(client_id="client", client_secret="secret"),
      ),
      credential_key="test-credential",
  )
  successful_calls = []

  def protected_tool(tool_context: ToolContext) -> dict:
    if auth_stage == "tool":
      credential = tool_context.get_auth_response(auth_config)
    else:
      credential = tool_context.get_invocation_context().credential_by_key.get(
          auth_config.credential_key
      )
    if not credential or not credential.oauth2.access_token:
      tool_context.request_credential(auth_config)
      return {"status": "authorization_required"}
    successful_calls.append(credential.oauth2.access_token)
    return {"status": "ok"}

  class AuthToolset(BaseToolset):

    def get_auth_config(self):
      return auth_config

    async def get_tools(self, readonly_context=None):
      return [FunctionTool(protected_tool)]

    async def close(self):
      pass

  child = LlmAgent(
      name="worker",
      disallow_transfer_to_parent=True,
      disallow_transfer_to_peers=True,
      model=testing_utils.MockModel.create([
          Part.from_function_call(name="protected_tool", args={}),
          "child completed",
      ]),
      tools=[protected_tool] if auth_stage == "tool" else [AuthToolset()],
  )
  delegate = child
  if nested:
    delegate = LlmAgent(
        name="middle",
        disallow_transfer_to_parent=True,
        model=testing_utils.MockModel.create([transfer_call_part(child.name)]),
        sub_agents=[child],
    )
  root = LlmAgent(
      name="root",
      model=testing_utils.MockModel.create([
          transfer_call_part(delegate.name),
          "root handles next message",
      ]),
      sub_agents=[delegate],
  )
  app = App(
      name="test_app",
      root_agent=root,
      resumability_config=ResumabilityConfig(is_resumable=resumable),
  )
  runner = Runner(app=app, session_service=InMemorySessionService())
  session = await runner.session_service.create_session(
      app_name=app.name, user_id="user"
  )

  async def collect(message):
    return [
        event
        async for event in runner.run_async(
            user_id=session.user_id,
            session_id=session.id,
            new_message=message,
        )
    ]

  try:
    initial = await collect(testing_utils.UserContent("run protected tool"))
    auth_events = [
        (event, call)
        for event in initial
        for call in event.get_function_calls()
        if call.name == "adk_request_credential"
    ]
    assert len(auth_events) == 1
    request_event, call = auth_events[0]
    assert call.id in request_event.long_running_tool_ids
    assert not successful_calls
    config = AuthConfig.model_validate(call.args["authConfig"])
    config.exchanged_auth_credential = AuthCredential(
        auth_type="oauth2", oauth2=OAuth2Auth(access_token="test-token")
    )
    resumed = await collect(
        testing_utils.UserContent(
            Part(
                function_response=FunctionResponse(
                    id=call.id,
                    name=call.name,
                    response=config.model_dump(mode="json", by_alias=True),
                )
            )
        )
    )

    assert successful_calls == ["test-token"]
    assert not any(
        call.name == "adk_request_credential"
        for event in resumed
        for call in event.get_function_calls()
    )
    assert any(
        event.author == child.name
        and event.content
        and any(part.text == "child completed" for part in event.content.parts)
        for event in resumed
    )
    # The resumed child keeps the original branch, including nested transfers.
    assert all(
        event.branch == request_event.branch
        for event in resumed
        if event.author == child.name and event.content
    )
    following = await collect(testing_utils.UserContent("a new request"))
    assert any(
        event.author == root.name
        and event.content
        and any(
            part.text == "root handles next message"
            for part in event.content.parts
        )
        for event in following
    )
    assert successful_calls == ["test-token"]
  finally:
    await runner.close()
