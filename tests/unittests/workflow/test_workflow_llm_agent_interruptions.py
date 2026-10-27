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

from __future__ import annotations

import asyncio
import copy
from typing import Any
from typing import AsyncGenerator

from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.run_config import RunConfig
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext
from google.adk.workflow import Edge
from google.adk.workflow import START
from google.adk.workflow._workflow import Workflow
from google.genai import types
import pytest

from tests.unittests import testing_utils
from tests.unittests.workflow import workflow_testing_utils


def long_running_tool_func():
  """A test tool that simulates a long-running operation."""
  return None


@pytest.mark.asyncio
async def test_workflow_pause_and_resume_simple(
    request: pytest.FixtureRequest,
):
  """Tests that a workflow can pause and resume with a single LlmAgent node."""

  mock_model = testing_utils.MockModel.create(
      responses=[
          types.Part.from_function_call(
              name='long_running_tool_func',
              args={},
          ),
          types.Part.from_text(text='LLM response after tool'),
      ]
  )

  # 1. Create agent with LRO tool
  node_a = LlmAgent(
      name='my_agent',
      model=mock_model,
      tools=[LongRunningFunctionTool(func=long_running_tool_func)],
  )

  # 2. Create workflow with single node
  wf = Workflow(
      name='test_workflow_hitl',
      edges=[
          (START, node_a),
      ],
  )

  app = App(
      name=request.function.__name__,
      root_agent=wf,
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # 3. First run: should pause on the long-running function call.
  user_event = testing_utils.get_user_content('start workflow')
  events1 = await runner.run_async(user_event)

  # Verify it paused on LRO
  assert any(e.long_running_tool_ids for e in events1)

  invocation_id = events1[0].invocation_id
  fc_event = workflow_testing_utils.find_function_call_event(
      events1, 'long_running_tool_func'
  )
  assert fc_event is not None
  function_call_id = fc_event.content.parts[0].function_call.id

  # 4. Prepare resume message
  tool_response = testing_utils.UserContent(
      types.Part(
          function_response=types.FunctionResponse(
              id=function_call_id,
              name='long_running_tool_func',
              response={'result': 'Final tool output'},
          )
      )
  )

  # 5. Resume with tool output.
  events2 = await runner.run_async(
      new_message=tool_response,
      invocation_id=invocation_id,
  )

  # 6. Verify completion
  content_texts = [
      p.text
      for e in events2
      if e.content and e.content.parts
      for p in e.content.parts
      if p.text
  ]

  assert any('LLM response after tool' in t for t in content_texts)


@pytest.mark.asyncio
async def test_workflow_pause_and_resume_task_mode(
    request: pytest.FixtureRequest,
):
  """Tests that a workflow can pause and resume with a single LlmAgent node in task mode."""

  mock_model = testing_utils.MockModel.create(
      responses=[
          types.Part.from_function_call(
              name='long_running_tool_func',
              args={},
          ),
          types.Part.from_text(text='LLM response after tool in task mode'),
      ]
  )

  # 1. Create agent with LRO tool and mode='task'
  node_a = LlmAgent(
      name='my_task_agent',
      model=mock_model,
      tools=[LongRunningFunctionTool(func=long_running_tool_func)],
      mode='task',
  )

  # 2. Create workflow with single node
  wf = Workflow(
      name='test_workflow_task_hitl',
      edges=[
          (START, node_a),
      ],
  )

  app = App(
      name=request.function.__name__,
      root_agent=wf,
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # 3. First run: should pause on the long-running function call.
  user_event = testing_utils.get_user_content('start workflow')
  events1 = await runner.run_async(user_event)

  # Verify it paused on LRO
  assert any(e.long_running_tool_ids for e in events1)

  invocation_id = events1[0].invocation_id
  fc_event = workflow_testing_utils.find_function_call_event(
      events1, 'long_running_tool_func'
  )
  assert fc_event is not None
  function_call_id = fc_event.content.parts[0].function_call.id

  # 4. Prepare resume message
  tool_response = testing_utils.UserContent(
      types.Part(
          function_response=types.FunctionResponse(
              id=function_call_id,
              name='long_running_tool_func',
              response={'result': 'Final tool output'},
          )
      )
  )

  # 5. Resume with tool output.
  events2 = await runner.run_async(
      new_message=tool_response,
      invocation_id=invocation_id,
  )

  # 6. Verify completion
  content_texts = [
      p.text
      for e in events2
      if e.content and e.content.parts
      for p in e.content.parts
      if p.text
  ]

  assert any('LLM response after tool in task mode' in t for t in content_texts)


@pytest.mark.asyncio
async def test_workflow_pause_and_resume_tool_confirmation(
    request: pytest.FixtureRequest,
):
  """Tests that a workflow can pause and resume with a tool requiring confirmation.

  Setup: Workflow with a single LlmAgent node having a tool requiring confirmation.
  Act:
    - Run 1: Start workflow, tool requests confirmation.
    - Run 2: Send confirmation response.
  Assert:
    - Run 1: Workflow pauses and yields confirmation request.
    - Run 2: Workflow resumes and completes with LLM response.
  """
  from google.adk.flows.llm_flows.functions import REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
  from google.adk.tools.function_tool import FunctionTool

  # Given a tool that requires confirmation and a mock model
  def _simple_tool_func():
    return {'result': 'tool executed'}

  mock_model = testing_utils.MockModel.create(
      responses=[
          types.Part.from_function_call(
              name='_simple_tool_func',
              args={},
          ),
          types.Part.from_text(text='LLM response after confirmation'),
      ]
  )

  node_a = LlmAgent(
      name='my_agent',
      model=mock_model,
      tools=[FunctionTool(func=_simple_tool_func, require_confirmation=True)],
  )

  wf = Workflow(
      name='test_workflow_confirmation',
      edges=[(START, node_a)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=wf,
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # When the workflow is started
  user_event = testing_utils.get_user_content('start workflow')
  events1 = await runner.run_async(user_event)

  # Then it should request confirmation
  fc_event = None
  for e in events1:
    if e.content and e.content.parts:
      for p in e.content.parts:
        if (
            p.function_call
            and p.function_call.name == REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
        ):
          fc_event = e
          break

  assert fc_event is not None, 'Did not find confirmation request event'

  ask_for_confirmation_function_call_id = fc_event.content.parts[
      0
  ].function_call.id
  invocation_id = events1[0].invocation_id

  # When the user confirms the tool call
  user_confirmation = testing_utils.UserContent(
      types.Part(
          function_response=types.FunctionResponse(
              id=ask_for_confirmation_function_call_id,
              name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
              response={'confirmed': True},
          )
      )
  )

  events2 = await runner.run_async(
      new_message=user_confirmation,
      invocation_id=invocation_id,
  )

  # Then the workflow completes with the LLM response
  content_texts = [
      p.text
      for e in events2
      if e.content and e.content.parts
      for p in e.content.parts
      if p.text
  ]

  assert any('LLM response after confirmation' in t for t in content_texts)


@pytest.mark.asyncio
async def test_workflow_pause_and_resume_auth_node(
    request: pytest.FixtureRequest,
):
  """Workflow pauses on missing credentials and resumes when provided.

  Setup: Workflow with a single node requiring auth.
  Act:
    - Run 1: Start workflow without credentials.
    - Run 2: Provide credentials via FunctionResponse.
  Assert:
    - Run 1: Workflow returns adk_request_credential request.
    - Run 2: Workflow completes and yields event with credential.
  """
  from fastapi.openapi.models import APIKey
  from fastapi.openapi.models import APIKeyIn
  from google.adk.auth.auth_credential import AuthCredential
  from google.adk.auth.auth_credential import AuthCredentialTypes
  from google.adk.auth.auth_tool import AuthConfig
  from google.adk.workflow import FunctionNode

  # Given a workflow with a node requiring auth
  auth_config = AuthConfig(
      auth_scheme=APIKey(**{'in': APIKeyIn.header, 'name': 'X-Api-Key'}),
      credential_key='my_key',
  )

  def fetch_weather(ctx):
    cred = ctx.get_auth_response(auth_config)
    api_key = cred.api_key if cred else 'unknown'
    from google.adk import Event

    yield Event(message=f'authed with {api_key}')

  node_a = FunctionNode(
      func=fetch_weather, auth_config=auth_config, rerun_on_resume=True
  )

  wf = Workflow(
      name='test_workflow_auth_node',
      edges=[(START, node_a)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=wf,
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # When the workflow is started without credentials
  events1 = await runner.run_async(testing_utils.get_user_content('start'))

  # Then it should pause and request credentials
  auth_fc_events = [
      e
      for e in events1
      if e.content
      and e.content.parts
      and e.content.parts[0].function_call
      and e.content.parts[0].function_call.name == 'adk_request_credential'
  ]
  assert len(auth_fc_events) == 1
  auth_fc_id = auth_fc_events[0].content.parts[0].function_call.id
  invocation_id = events1[0].invocation_id

  # When the user provides the credentials
  auth_response = AuthConfig(
      auth_scheme=auth_config.auth_scheme,
      credential_key=auth_config.credential_key,
      exchanged_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.API_KEY,
          api_key='secret_key',
      ),
  )

  user_credential_response = testing_utils.UserContent(
      types.Part(
          function_response=types.FunctionResponse(
              id=auth_fc_id,
              name='adk_request_credential',
              response=auth_response.model_dump(
                  exclude_none=True, by_alias=True
              ),
          ),
      ),
  )

  # When the workflow is resumed
  events2 = await runner.run_async(
      new_message=user_credential_response,
      invocation_id=invocation_id,
  )

  # Then the workflow should resume and complete
  content_texts = [
      p.text
      for e in events2
      if e.content and e.content.parts
      for p in e.content.parts
      if p.text
  ]
  assert any('authed with secret_key' in t for t in content_texts)


@pytest.mark.asyncio
async def test_workflow_pause_and_resume_parent_interruption(
    request: pytest.FixtureRequest,
):
  """Tests multi-agent workflow where parent produces an interruption."""

  # Child agent (does nothing special)
  child_agent = LlmAgent(
      name='child_agent',
      model=testing_utils.MockModel.create(
          responses=[
              types.Part.from_function_call(
                  name='finish_task',
                  args={'result': 'Child done'},
              )
          ]
      ),
      mode='task',
  )

  # Parent agent calls LRO tool first, then delegates to child
  fc = types.Part.from_function_call(name='long_running_tool_func', args={})
  call_child = types.Part.from_function_call(
      name='child_agent',
      args={'request': 'Start child task'},
  )

  parent_model = testing_utils.MockModel.create(
      responses=[
          fc,  # First call LRO
          call_child,  # Then call child
          'Parent all done',  # Finally finish
      ]
  )

  parent_agent = LlmAgent(
      name='parent_agent',
      model=parent_model,
      tools=[
          LongRunningFunctionTool(func=long_running_tool_func),
      ],
      sub_agents=[child_agent],
      mode='task',
  )

  wf = Workflow(
      name='test_workflow_parent_hitl',
      edges=[
          (START, parent_agent),
      ],
  )

  app = App(
      name=request.function.__name__,
      root_agent=wf,
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Run 1: Should pause on LRO
  events1 = await runner.run_async(testing_utils.get_user_content('start'))
  assert any(e.long_running_tool_ids for e in events1)

  invocation_id = events1[0].invocation_id
  fc_event = workflow_testing_utils.find_function_call_event(
      events1, 'long_running_tool_func'
  )
  assert fc_event is not None
  function_call_id = fc_event.content.parts[0].function_call.id

  # Resume with tool output
  tool_response = testing_utils.UserContent(
      types.Part(
          function_response=types.FunctionResponse(
              id=function_call_id,
              name='long_running_tool_func',
              response={'result': 'LRO done'},
          )
      )
  )

  events2 = await runner.run_async(
      new_message=tool_response,
      invocation_id=invocation_id,
  )

  # Verify completion
  content_texts = [
      p.text
      for e in events2
      if e.content and e.content.parts
      for p in e.content.parts
      if p.text
  ]
  assert any('Parent all done' in t for t in content_texts)


@pytest.mark.asyncio
async def test_workflow_pause_and_resume_child_interruption(
    request: pytest.FixtureRequest,
):
  """Tests multi-agent workflow where child produces an interruption."""

  # Child agent calls LRO tool
  fc = types.Part.from_function_call(name='long_running_tool_func', args={})
  child_model = testing_utils.MockModel.create(
      responses=[
          fc,
          types.Part.from_function_call(
              name='finish_task',
              args={'result': 'Child done after tool'},
          ),
      ]
  )

  child_agent = LlmAgent(
      name='child_agent',
      model=child_model,
      tools=[LongRunningFunctionTool(func=long_running_tool_func)],
      mode='task',
  )

  # Parent agent delegates to child first
  call_child = types.Part.from_function_call(
      name='child_agent',
      args={'request': 'Start child task'},
  )
  parent_model = testing_utils.MockModel.create(
      responses=[
          call_child,
          'Parent all done',
      ]
  )

  parent_agent = LlmAgent(
      name='parent_agent',
      model=parent_model,
      sub_agents=[child_agent],
      mode='task',
  )

  wf = Workflow(
      name='test_workflow_child_hitl',
      edges=[
          (START, parent_agent),
      ],
  )

  app = App(
      name=request.function.__name__,
      root_agent=wf,
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Run 1: Should enter parent, then child, then child pauses on LRO!
  events1 = await runner.run_async(testing_utils.get_user_content('start'))
  assert any(
      p.function_call and p.function_call.name == 'child_agent'
      for e in events1
      if e.content
      for p in e.content.parts
  )

  invocation_id = events1[0].invocation_id
  fc_event = workflow_testing_utils.find_function_call_event(
      events1, 'long_running_tool_func'
  )
  assert fc_event is not None
  function_call_id = fc_event.content.parts[0].function_call.id

  # Resume with tool output
  tool_response = testing_utils.UserContent(
      types.Part(
          function_response=types.FunctionResponse(
              id=function_call_id,
              name='long_running_tool_func',
              response={'result': 'LRO done'},
          )
      )
  )

  events2 = await runner.run_async(
      new_message=tool_response,
      invocation_id=invocation_id,
  )

  # Verify completion
  content_texts = [
      p.text
      for e in events2
      if e.content and e.content.parts
      for p in e.content.parts
      if p.text
  ]
  assert any('Parent all done' in t for t in content_texts)


def _append_function_response(
    session, invocation_id, branch, fc_id, func_name, response
):
  """Helper to append a FunctionResponse event to a session."""
  session.events.append(
      Event(
          invocation_id=invocation_id,
          author='user',
          branch=branch,
          content=types.Content(
              role='user',
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          id=fc_id,
                          name=func_name,
                          response=response,
                      )
                  )
              ],
          ),
      )
  )


@pytest.mark.asyncio
async def test_workflow_resume_inputs_fallback_branch(monkeypatch):
  """Resume inputs find function name in a different branch.

  Setup: Session contains a FunctionCall in branch_A.
  Act: Run the wrapper with resume_inputs for that FunctionCall in branch_B.
  Assert: The wrapper successfully finds the function name and completes.
  """

  # Arrange
  mock_model = testing_utils.MockModel.create(
      responses=[types.Part.from_text(text='I am done')]
  )
  agent = LlmAgent(name='test_agent', model=mock_model)

  # Create a dummy context and session

  session = Session(id='test_session', appName='test_app', userId='test_user')
  # Add an event with function call in branch 'branch_A'
  session.events.append(
      Event(
          invocation_id='test_inv',
          branch='branch_A',
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          id='fc_123',
                          name='my_target_func',
                          args={},
                      )
                  )
              ]
          ),
      )
  )

  # Create invocation context with branch 'branch_B'

  session_service = InMemorySessionService()

  ic = InvocationContext(
      session=session,
      branch='branch_B',
      session_service=session_service,
      invocation_id='test_inv',
      run_config=RunConfig(),
      agent=agent,
  )

  # Create context with resume_inputs
  ctx = Context(
      ic,
      node_path='test_agent',
      run_id='1',
      resume_inputs={'fc_123': {'result': 'ok'}},
  )

  # Mock prepare functions
  class DummyAgentCtx:

    def __init__(self, ic):
      self._ic = ic

    def get_invocation_context(self):
      return self._ic

  from google.adk.workflow import _llm_agent_wrapper

  monkeypatch.setattr(
      _llm_agent_wrapper,
      'prepare_llm_agent_context',
      lambda a, c: DummyAgentCtx(ic),
  )
  monkeypatch.setattr(
      _llm_agent_wrapper, 'prepare_llm_agent_input', lambda a, c, i: None
  )

  # Simulate Runner adding the event to the correct branch!
  _append_function_response(
      session,
      invocation_id='test_inv',
      branch='branch_A',
      fc_id='fc_123',
      func_name='my_target_func',
      response={'result': 'ok'},
  )

  # Act - Run the function directly
  from google.adk.workflow._llm_agent_wrapper import run_llm_agent_as_node

  gen = run_llm_agent_as_node(agent, ctx=ctx, node_input='start')
  try:
    await gen.__anext__()
  except StopAsyncIteration:
    pass

  # Assert - Verify that the event is there with correct branch
  # Initial events had 1 item + 1 simulated by Runner = 2!
  assert len(session.events) == 2
  event = session.events[-1]  # The newly injected event!
  assert (
      event.branch == 'branch_A'
  )  # Injected in the branch where call was made!
  assert event.content.parts[0].function_response.name == 'my_target_func'


@pytest.mark.asyncio
async def test_workflow_resume_inputs_multiple_branches(monkeypatch):
  """Resume inputs handle multiple items targeting different branches.

  Setup: Session contains FunctionCalls in branch_A and branch_B.
  Act: Run the wrapper with resume_inputs for both in branch_C.
  Assert: The wrapper successfully finds function names for both.
  """

  # Arrange
  mock_model = testing_utils.MockModel.create(
      responses=[types.Part.from_text(text='I am done')]
  )
  agent = LlmAgent(name='test_agent', model=mock_model)

  session = Session(id='test_session', appName='test_app', userId='test_user')

  # Add event 1 in branch_A
  session.events.append(
      Event(
          invocation_id='test_inv',
          branch='branch_A',
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          id='fc_A',
                          name='func_A',
                          args={},
                      )
                  )
              ]
          ),
      )
  )

  # Add event 2 in branch_B
  session.events.append(
      Event(
          invocation_id='test_inv',
          branch='branch_B',
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          id='fc_B',
                          name='func_B',
                          args={},
                      )
                  )
              ]
          ),
      )
  )

  session_service = InMemorySessionService()

  # Current branch is branch_C
  ic = InvocationContext(
      session=session,
      branch='branch_C',
      session_service=session_service,
      invocation_id='test_inv',
      run_config=RunConfig(),
      agent=agent,
  )

  ctx = Context(
      ic,
      node_path='test_agent',
      run_id='1',
      resume_inputs={'fc_A': {'result': 'ok_A'}, 'fc_B': {'result': 'ok_B'}},
  )

  class DummyAgentCtx:

    def __init__(self, ic):
      self._ic = ic

    def get_invocation_context(self):
      return self._ic

  from google.adk.workflow import _llm_agent_wrapper

  monkeypatch.setattr(
      _llm_agent_wrapper,
      'prepare_llm_agent_context',
      lambda a, c: DummyAgentCtx(ic),
  )
  monkeypatch.setattr(
      _llm_agent_wrapper, 'prepare_llm_agent_input', lambda a, c, i: None
  )

  # Simulate Runner adding the events to the correct branches!
  _append_function_response(
      session,
      invocation_id='test_inv',
      branch='branch_A',
      fc_id='fc_A',
      func_name='func_A',
      response={'result': 'ok_A'},
  )
  _append_function_response(
      session,
      invocation_id='test_inv',
      branch='branch_B',
      fc_id='fc_B',
      func_name='func_B',
      response={'result': 'ok_B'},
  )

  # Act - Run the function directly
  from google.adk.workflow._llm_agent_wrapper import run_llm_agent_as_node

  gen = run_llm_agent_as_node(agent, ctx=ctx, node_input='start')
  try:
    await gen.__anext__()
  except StopAsyncIteration:
    pass

  # Assert - Verify that the events are there with correct branches
  # Initial events had 2 items + 2 simulated by Runner = 4!
  assert len(session.events) == 4

  # Check both exist in the injected events
  injected_events = session.events[2:]
  branches = [e.branch for e in injected_events]
  names = [e.content.parts[0].function_response.name for e in injected_events]

  assert 'branch_A' in branches
  assert 'branch_B' in branches
  assert 'func_A' in names
  assert 'func_B' in names


@pytest.mark.asyncio
async def test_workflow_task_mode_plain_text_resume_auto_routing(
    request: pytest.FixtureRequest,
):
  """Tests that task mode agent can pause for text and resume with text without explicit invocation_id."""

  # LLM behavior:
  # Round 1: Outputs text asking for info.
  # Round 2: After user replies with text, calls finish_task with result.
  mock_model = testing_utils.MockModel.create(
      responses=[
          types.Part.from_text(text='Please provide the secret code:'),
          types.Part.from_function_call(
              name='finish_task',
              args={'result': 'Success with code'},
          ),
      ]
  )

  node_a = LlmAgent(
      name='my_task_agent',
      model=mock_model,
      mode='task',
  )

  wf = Workflow(
      name='test_workflow_plain_text_resume',
      edges=[
          (START, node_a),
      ],
  )

  app = App(
      name=request.function.__name__,
      root_agent=wf,
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # 1. First run: should yield the prompt and pause.
  events1 = await runner.run_async('start workflow')

  # Verify it yielded the prompt
  texts1 = [
      p.text
      for e in events1
      if e.content and e.content.parts
      for p in e.content.parts
      if p.text
  ]
  assert 'Please provide the secret code:' in texts1

  # 2. Second run: resume with plain text, NOT passing invocation_id!
  # It should automatically resolve invocation_id and isolation_scope.
  events2 = await runner.run_async('my_secret_code_123')

  # Verify completion
  # The last event should have output set from finish_task args
  assert any(e.output == {'result': 'Success with code'} for e in events2)


@pytest.mark.asyncio
async def test_workflow_mixed_turn_lro_pause(
    request: pytest.FixtureRequest,
):
  """Tests that in a mixed turn, if an LRO tool pauses, task delegation is executed and the node pauses."""

  # 1. Create a child agent (delegated task)
  child_agent = LlmAgent(
      name='child_agent',
      model=testing_utils.MockModel.create(
          responses=[
              types.Part.from_function_call(
                  name='finish_task',
                  args={'result': 'Child done'},
              )
          ]
      ),
      mode='task',
  )

  # 2. Parent agent calls both LRO and delegates to child in the same turn
  fc_lro = types.Part.from_function_call(name='long_running_tool_func', args={})
  fc_child = types.Part.from_function_call(
      name='child_agent',
      args={'request': 'Start child task'},
  )

  parent_model = testing_utils.MockModel.create(
      responses=[
          [fc_lro, fc_child],  # Mixed turn
          'Parent all done',  # After resume
      ]
  )

  parent_agent = LlmAgent(
      name='parent_agent',
      model=parent_model,
      tools=[
          LongRunningFunctionTool(func=long_running_tool_func),
      ],
      sub_agents=[child_agent],
      mode='chat',
  )

  wf = Workflow(
      name='test_workflow_mixed_turn_pause',
      edges=[
          (START, parent_agent),
      ],
  )

  app = App(
      name=request.function.__name__,
      root_agent=wf,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Run 1: Should pause on LRO, but child_agent should have been executed.
  events1 = await runner.run_async(testing_utils.get_user_content('start'))

  # Verify it paused on LRO (it has long_running_tool_ids)
  assert any(e.long_running_tool_ids for e in events1)

  # Verify that child_agent WAS executed.
  session_events = runner.session.events
  child_fr_events = [
      e
      for e in session_events
      if e.content
      and any(
          p.function_response and p.function_response.name == 'child_agent'
          for p in e.content.parts
      )
  ]
  assert child_fr_events, 'Child agent task was not dispatched!'

  # Verify parent did not finish yet (no "Parent all done")
  parent_finished_events = [
      e
      for e in events1
      if e.content
      and any(p.text and 'Parent all done' in p.text for p in e.content.parts)
  ]
  assert not parent_finished_events, 'Parent finished prematurely!'

  # Get the LRO FC ID and invocation ID to resume
  lro_fc = None
  invocation_id = None
  for event in events1:
    for fc in event.get_function_calls():
      if fc.name == 'long_running_tool_func':
        lro_fc = fc
        invocation_id = event.invocation_id
        break
    if lro_fc:
      break
  assert lro_fc is not None
  assert invocation_id is not None

  # Resume with LRO response
  tool_response = testing_utils.UserContent(
      types.Part(
          function_response=types.FunctionResponse(
              id=lro_fc.id,
              name='long_running_tool_func',
              response={'result': 'LRO done'},
          )
      )
  )

  events2 = await runner.run_async(
      new_message=tool_response,
      invocation_id=invocation_id,
  )

  # Verify completion in Run 2
  parent_finished_events2 = [
      e
      for e in events2
      if e.content
      and any(p.text and 'Parent all done' in p.text for p in e.content.parts)
  ]
  assert parent_finished_events2, 'Parent did not finish after resume!'


@pytest.mark.asyncio
async def test_static_node_fast_forward_resumes_transfer_target(
    request: pytest.FixtureRequest,
):
  """Resuming a fast-forwarded static node still resumes its transfer target."""
  beta_model = testing_utils.MockModel.create(
      responses=[
          types.Part.from_function_call(name='long_running_tool_func', args={}),
          types.Part.from_text(text='BETA_RESUMED_OUTPUT'),
      ]
  )
  alpha_model = testing_utils.MockModel.create(
      responses=[
          types.Part.from_function_call(
              name='transfer_to_agent', args={'agent_name': 'beta'}
          ),
      ]
  )

  beta = LlmAgent(
      name='beta',
      model=beta_model,
      tools=[LongRunningFunctionTool(func=long_running_tool_func)],
  )
  alpha = LlmAgent(name='alpha', model=alpha_model, sub_agents=[beta])

  wf = Workflow(name='wf', edges=[(START, alpha)])
  app = App(
      name=request.function.__name__,
      root_agent=wf,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Turn 1: alpha transfers to beta, beta pauses on the long-running tool.
  events1 = await runner.run_async(testing_utils.get_user_content('go'))
  assert any(e.long_running_tool_ids for e in events1)

  invocation_id = events1[0].invocation_id
  fc_event = workflow_testing_utils.find_function_call_event(
      events1, 'long_running_tool_func'
  )
  assert fc_event is not None
  function_call_id = fc_event.content.parts[0].function_call.id

  tool_response = testing_utils.UserContent(
      types.Part(
          function_response=types.FunctionResponse(
              id=function_call_id,
              name='long_running_tool_func',
              response={'result': 'tool done'},
          )
      )
  )

  # Turn 2: resume. alpha is fast-forwarded but must still reach beta.
  events2 = await runner.run_async(
      new_message=tool_response,
      invocation_id=invocation_id,
  )

  resumed_texts = [
      p.text
      for e in events2
      if e.content and e.content.parts
      for p in e.content.parts
      if p.text
  ]
  assert any('BETA_RESUMED_OUTPUT' in t for t in resumed_texts), (
      'Transfer target was not resumed after the static node was'
      f' fast-forwarded; got {resumed_texts!r}'
  )


def _make_two_confirmation_workflow(
    request: pytest.FixtureRequest, *, resumable: bool
) -> tuple[testing_utils.InMemoryRunner, list[Any], list[list[Any]]]:
  """Builds START -> agent -> report, where agent calls a confirmed tool twice.

  Returns the runner, the inputs `report` ran with, and the tool results each
  model call saw.
  """
  from google.adk.tools.function_tool import FunctionTool
  from google.adk.workflow import FunctionNode

  report_inputs: list[Any] = []
  model_views: list[list[Any]] = []

  class _ScriptedModel(testing_utils.MockModel):
    """Asks for act(1) and act(2) first; afterwards summarises the results."""

    async def generate_content_async(self, llm_request, stream=False):
      results = [
          part.function_response.response
          for content in llm_request.contents
          for part in content.parts or []
          if part.function_response
      ]
      model_views.append(results)
      if results:
        parts = [types.Part.from_text(text=f'done: {results}')]
      else:
        parts = [
            types.Part(
                function_call=types.FunctionCall(
                    id=f'call-{n}', name='act', args={'n': n}
                )
            )
            for n in (1, 2)
        ]
      yield LlmResponse(content=types.Content(role='model', parts=parts))

  def act(n: int) -> str:
    return f'acted {n}'

  def report(node_input):
    report_inputs.append(node_input)
    return node_input

  agent = LlmAgent(
      name='agent',
      model=_ScriptedModel(responses=[]),
      tools=[FunctionTool(act, require_confirmation=True)],
  )
  wf = Workflow(
      name='root',
      edges=[(START, agent), (agent, FunctionNode(func=report, name='report'))],
  )
  app = App(
      name=request.function.__name__,
      root_agent=wf,
      resumability_config=ResumabilityConfig(is_resumable=resumable),
  )
  return testing_utils.InMemoryRunner(app=app), report_inputs, model_views


def _confirmation_requests(events: list[Event]) -> list[types.FunctionCall]:
  from google.adk.flows.llm_flows.functions import REQUEST_CONFIRMATION_FUNCTION_CALL_NAME

  return [
      fc
      for e in events
      for fc in e.get_function_calls()
      if fc.name == REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
      and fc.id in (e.long_running_tool_ids or ())
  ]


def _confirm(
    calls: list[types.FunctionCall], *, confirmed: bool = True
) -> types.Content:
  return types.Content(
      role='user',
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  id=fc.id, name=fc.name, response={'confirmed': confirmed}
              )
          )
          for fc in calls
      ],
  )


@pytest.mark.parametrize('resumable', [False, True])
@pytest.mark.parametrize('first', [0, 1], ids=['first_call', 'second_call'])
async def test_llm_agent_node_waits_for_remaining_confirmations(
    request: pytest.FixtureRequest, resumable: bool, first: int
):
  """Answering one of two confirmations keeps the node, and the next node, waiting.

  Setup: START -> agent -> report. The model calls a confirmed tool twice in
    one turn, so the run pauses with two confirmation requests.
  Act:
    - Run 2: confirm only one of them (parametrized over which).
    - Run 3: confirm the other.
  Assert:
    - Run 2: the model is not consulted again and `report` does not run.
    - Run 3: the model sees both tool results once, and `report` runs exactly
      once, with that answer.
  """
  runner, report_inputs, model_views = _make_two_confirmation_workflow(
      request, resumable=resumable
  )

  events1 = await runner.run_async(testing_utils.get_user_content('go'))
  requests = _confirmation_requests(events1)
  assert len(requests) == 2
  invocation_id = events1[0].invocation_id
  resume_kwargs = {'invocation_id': invocation_id} if resumable else {}

  await runner.run_async(
      new_message=_confirm([requests[first]]), **resume_kwargs
  )

  assert len(model_views) == 1, 'model consulted with an answer still owed'
  assert report_inputs == [], 'report ran before all confirmations'

  await runner.run_async(
      new_message=_confirm([requests[1 - first]]), **resume_kwargs
  )

  assert model_views[1:] == [[{'result': 'acted 1'}, {'result': 'acted 2'}]]
  assert report_inputs == [
      "done: [{'result': 'acted 1'}, {'result': 'acted 2'}]"
  ]


@pytest.mark.parametrize('resumable', [False, True])
async def test_llm_agent_node_runs_tool_in_the_turn_it_is_confirmed(
    request: pytest.FixtureRequest, resumable: bool
):
  """A confirmed tool runs in the turn its confirmation arrives, not later.

  Setup: as above, two confirmation requests pending.
  Act: confirm the first call only.
  Assert: that turn yields the first tool's result event.
  """
  runner, _, _ = _make_two_confirmation_workflow(request, resumable=resumable)

  events1 = await runner.run_async(testing_utils.get_user_content('go'))
  requests = _confirmation_requests(events1)
  resume_kwargs = (
      {'invocation_id': events1[0].invocation_id} if resumable else {}
  )

  events2 = await runner.run_async(
      new_message=_confirm([requests[0]]), **resume_kwargs
  )

  tool_results = [
      fr.response for e in events2 for fr in e.get_function_responses()
  ]
  assert tool_results == [{'result': 'acted 1'}]


@pytest.mark.parametrize('resumable', [False, True])
async def test_llm_agent_node_rejecting_the_last_confirmation_completes(
    request: pytest.FixtureRequest, resumable: bool
):
  """Rejecting the remaining confirmation lets the node and the next node finish.

  Setup: as above, two confirmation requests pending.
  Act: confirm the first call, then reject the second.
  Assert: `report` runs once, on an answer that carries the rejection.
  """
  runner, report_inputs, _ = _make_two_confirmation_workflow(
      request, resumable=resumable
  )

  events1 = await runner.run_async(testing_utils.get_user_content('go'))
  requests = _confirmation_requests(events1)
  resume_kwargs = (
      {'invocation_id': events1[0].invocation_id} if resumable else {}
  )

  await runner.run_async(new_message=_confirm([requests[0]]), **resume_kwargs)
  await runner.run_async(
      new_message=_confirm([requests[1]], confirmed=False), **resume_kwargs
  )

  assert report_inputs == [
      "done: [{'result': 'acted 1'}, {'error': 'This tool call is rejected.'}]"
  ]


@pytest.mark.parametrize('resumable', [False, True])
async def test_llm_agent_node_confirming_all_at_once_completes(
    request: pytest.FixtureRequest, resumable: bool
):
  """Answering every pending confirmation in one message completes the node.

  Setup: as above, two confirmation requests pending.
  Act: confirm both in a single message.
  Assert: the model sees both results once and `report` runs exactly once.
  """
  runner, report_inputs, model_views = _make_two_confirmation_workflow(
      request, resumable=resumable
  )

  events1 = await runner.run_async(testing_utils.get_user_content('go'))
  requests = _confirmation_requests(events1)
  resume_kwargs = (
      {'invocation_id': events1[0].invocation_id} if resumable else {}
  )

  await runner.run_async(new_message=_confirm(requests), **resume_kwargs)

  assert model_views[1:] == [[{'result': 'acted 1'}, {'result': 'acted 2'}]]
  assert report_inputs == [
      "done: [{'result': 'acted 1'}, {'result': 'acted 2'}]"
  ]
