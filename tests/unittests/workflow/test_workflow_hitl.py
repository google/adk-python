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

"""Testings for the Workflow HITL scenarios."""

import asyncio
import copy
from typing import Any
from typing import AsyncGenerator
from unittest import mock

from google.adk.agents.context import Context
from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.workflow import BaseNode
from google.adk.workflow import Edge
from google.adk.workflow import node
from google.adk.workflow import START
from google.adk.workflow._node_status import NodeStatus
from google.adk.workflow._workflow import Workflow
from google.adk.workflow.utils._rehydration_utils import _wrap_response
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
from google.adk.workflow.utils._workflow_hitl_utils import get_request_input_interrupt_ids
from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_CREDENTIAL_FUNCTION_CALL_NAME
from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_INPUT_FUNCTION_CALL_NAME
from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
import pytest
from typing_extensions import override

from . import workflow_testing_utils
from .. import testing_utils
from .workflow_testing_utils import InputCapturingNode
from .workflow_testing_utils import RequestInputNode

ANY = mock.ANY


class _TestingNode(BaseNode):
  """A node that produces a simple message."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  name: str = Field(default='')
  message: str = Field(default='')
  delay: float = Field(default=0)

  @override
  def get_name(self) -> str:
    return self.name

  @override
  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    if self.delay > 0:
      await asyncio.sleep(self.delay)
    yield Event(output=self.message)


def long_running_tool_func():
  """A test tool that simulates a long-running operation."""
  return None


@pytest.mark.parametrize(
    'resumable',
    [
        pytest.param(
            False, marks=pytest.mark.xfail(reason='Fails in non-resumable mode')
        ),
        pytest.param(
            True, marks=pytest.mark.xfail(reason='Resumability broken in V2')
        ),
    ],
)
@pytest.mark.asyncio
async def test_workflow_pause_and_resume(
    request: pytest.FixtureRequest,
    resumable: bool,
):
  """Tests that a workflow can pause and resume.

  This test uses LlmAgent with LongRunningFunctionTool.
  """
  node_a = _TestingNode(name='NodeA', message='Executing A')

  node_b = LlmAgent(
      name='NodeB_agent',
      model=testing_utils.MockModel.create(
          responses=[
              types.Part.from_function_call(
                  name='long_running_tool_func',
                  args={},
              ),
              types.Part.from_text(text='LLM response after tool'),
          ]
      ),
      tools=[LongRunningFunctionTool(func=long_running_tool_func)],
  )
  node_c = _TestingNode(name='NodeC', message='Executing C')
  agent = Workflow(
      name='test_workflow_agent_hitl',
      edges=[
          (START, node_a),
          (node_a, node_b),
          (node_b, node_c),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # First run: should pause on the long-running function call.
  user_event = testing_utils.get_user_content('start workflow')
  events1 = await runner.run_async(user_event)

  invocation_id = events1[0].invocation_id
  fc_event = workflow_testing_utils.find_function_call_event(
      events1, 'long_running_tool_func'
  )
  function_call_id = fc_event.content.parts[0].function_call.id

  simplified_events1 = (
      workflow_testing_utils.simplify_events_with_node_and_agent_state(
          copy.deepcopy(events1),
      )
  )

  # Filter to outer workflow state checkpoint events only (LlmAgent as Mesh
  # emits internal state events that are implementation details).
  outer_state_events1 = [
      e
      for e in simplified_events1
      if e[0] == 'test_workflow_agent_hitl'
      and isinstance(e[1], dict)
      and 'nodes' in e[1]
  ]

  # Verify the outer workflow saw: NodeB_agent (interrupted).
  if resumable:
    assert outer_state_events1[-1] == (
        'test_workflow_agent_hitl',
        {
            'nodes': {
                'NodeA': {'status': NodeStatus.COMPLETED.value},
                'NodeB_agent': {
                    'status': NodeStatus.WAITING.value,
                    'interrupts': [function_call_id],
                },
            },
        },
    )

  tool_response = testing_utils.UserContent(
      types.Part(
          function_response=types.FunctionResponse(
              id=function_call_id,
              name='long_running_tool_func',
              response={'result': 'Final tool output'},
          )
      )
  )

  # Resume with tool output.
  # In resumable mode, reuse the invocation_id so agent state is loaded.
  # In non-resumable mode, use a new invocation so state is reconstructed
  # from session events.
  events2 = await runner.run_async(
      new_message=tool_response,
      invocation_id=invocation_id,
  )

  simplified_events2 = (
      workflow_testing_utils.simplify_events_with_node_and_agent_state(
          copy.deepcopy(events2),
          include_resume_inputs=True,
      )
  )

  # Filter to outer workflow state checkpoint events only.
  outer_state_events2 = [
      e
      for e in simplified_events2
      if e[0] == 'test_workflow_agent_hitl'
      and isinstance(e[1], dict)
      and 'nodes' in e[1]
  ]

  # Verify NodeB_agent resumed, completed, and NodeC ran.
  if resumable:
    assert outer_state_events2[-1] == (
        'test_workflow_agent_hitl',
        {
            'nodes': {
                'NodeA': {'status': NodeStatus.COMPLETED.value},
                'NodeB_agent': {'status': NodeStatus.COMPLETED.value},
                'NodeC': {'status': NodeStatus.COMPLETED.value},
            }
        },
    )
  # Verify end_of_agent was emitted.
  end_events = [
      e
      for e in simplified_events2
      if e[0] == 'test_workflow_agent_hitl'
      and e[1] == testing_utils.END_OF_AGENT
  ]
  assert len(end_events) == 1


@pytest.mark.xfail(reason='Resumability broken in V2')
@pytest.mark.asyncio
async def test_workflow_interrupt_allows_parallel_execution(
    request: pytest.FixtureRequest,
):
  """Tests that if one node is interrupted, parallel nodes can execute.

  This test uses LlmAgent with LongRunningFunctionTool, which requires
  resumability to preserve the LLM's conversation state across interrupts.
  """
  node_a = LlmAgent(
      name='NodeA',
      model=testing_utils.MockModel.create(
          responses=[
              types.Part.from_function_call(
                  name='long_running_tool_func',
                  args={},
              ),
          ]
      ),
      tools=[LongRunningFunctionTool(func=long_running_tool_func)],
  )
  node_b = _TestingNode(name='NodeB', message='Executing B', delay=0.5)
  agent = Workflow(
      name='test_workflow_agent_parallel_interrupt',
      edges=[
          (START, node_a),
          (START, node_b),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  user_event = testing_utils.get_user_content('start workflow')
  events = await runner.run_async(user_event)
  fc_event = workflow_testing_utils.find_function_call_event(
      events, 'long_running_tool_func'
  )
  function_call_id = fc_event.content.parts[0].function_call.id

  simplified = workflow_testing_utils.simplify_events_with_node_and_agent_state(
      copy.deepcopy(events)
  )
  # Filter to outer workflow state checkpoint events only (LlmAgent as Mesh
  # emits internal state events that are implementation details).
  outer_state = [
      e
      for e in simplified
      if e[0] == 'test_workflow_agent_parallel_interrupt'
      and isinstance(e[1], dict)
      and 'nodes' in e[1]
  ]

  # Verify final state: NodeA interrupted, NodeB completed.
  assert outer_state[-1] == (
      'test_workflow_agent_parallel_interrupt',
      {
          'nodes': {
              'NodeA': {
                  'status': NodeStatus.WAITING.value,
                  'interrupts': [function_call_id],
              },
              'NodeB': {'status': NodeStatus.COMPLETED.value},
          },
      },
  )


@pytest.mark.parametrize(
    'resumable',
    [
        False,
        pytest.param(
            True, marks=pytest.mark.xfail(reason='Resumability broken in V2')
        ),
    ],
)
@pytest.mark.asyncio
async def test_workflow_request_input_resume(
    request: pytest.FixtureRequest, resumable: bool
):
  """Tests resume with RequestInputEvent."""

  class UserDetails(BaseModel):
    name: str
    age: int

  node_a = RequestInputNode(
      name='NodeA_input',
      message='Please provide user details.',
      response_schema=UserDetails.model_json_schema(),
  )
  node_b = _TestingNode(name='NodeB', message='Received user details')
  agent = Workflow(
      name='test_workflow_agent_input_schema',
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=node_b),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Run and expect RequestInputEvent
  user_event = testing_utils.get_user_content('start workflow')
  events1 = await runner.run_async(user_event)

  request_input_event = workflow_testing_utils.find_function_call_event(
      events1, REQUEST_INPUT_FUNCTION_CALL_NAME
  )
  assert request_input_event is not None
  args = request_input_event.content.parts[0].function_call.args
  assert args['message'] == 'Please provide user details.'
  assert args['response_schema'] == {
      'properties': {
          'name': {'title': 'Name', 'type': 'string'},
          'age': {'title': 'Age', 'type': 'integer'},
      },
      'required': ['name', 'age'],
      'title': 'UserDetails',
      'type': 'object',
  }
  interrupt_id = get_request_input_interrupt_ids(request_input_event)[0]
  invocation_id = request_input_event.invocation_id

  simplified_events1 = (
      workflow_testing_utils.simplify_events_with_node_and_agent_state(
          copy.deepcopy(events1)
      )
  )
  expected_events1 = [
      (
          'test_workflow_agent_input_schema',
          {
              'nodes': {
                  'NodeA_input': {'status': NodeStatus.RUNNING.value},
              }
          },
      ),
      (
          'test_workflow_agent_input_schema@1/NodeA_input@1',
          types.Part(
              function_call=types.FunctionCall(
                  name=REQUEST_INPUT_FUNCTION_CALL_NAME,
                  args={
                      'interruptId': interrupt_id,
                      'message': 'Please provide user details.',
                      'payload': None,
                      'response_schema': {
                          'properties': {
                              'name': {'title': 'Name', 'type': 'string'},
                              'age': {'title': 'Age', 'type': 'integer'},
                          },
                          'required': ['name', 'age'],
                          'title': 'UserDetails',
                          'type': 'object',
                      },
                  },
              )
          ),
      ),
      (
          'test_workflow_agent_input_schema',
          {
              'nodes': {
                  'NodeA_input': {
                      'status': NodeStatus.WAITING.value,
                      'interrupts': [interrupt_id],
                  },
              },
          },
      ),
  ]
  if resumable:
    assert simplified_events1 == expected_events1
  else:
    assert simplified_events1 == (
        workflow_testing_utils.strip_checkpoint_events(expected_events1)
    )

  # Resume with user input
  user_input = create_request_input_response(
      interrupt_id, {'name': 'John', 'age': 30}
  )
  events2 = await runner.run_async(
      new_message=testing_utils.UserContent(user_input),
      invocation_id=invocation_id,
  )
  simplified_events2 = (
      workflow_testing_utils.simplify_events_with_node_and_agent_state(
          copy.deepcopy(events2)
      )
  )
  expected_events2 = [
      (
          'test_workflow_agent_input_schema@1/NodeA_input@1',
          {'output': {'age': 30, 'name': 'John'}},
      ),
      (
          'test_workflow_agent_input_schema',
          {
              'nodes': {
                  'NodeA_input': {'status': NodeStatus.COMPLETED.value},
                  'NodeB': {
                      'status': NodeStatus.RUNNING.value,
                  },
              }
          },
      ),
      (
          'test_workflow_agent_input_schema@1/NodeB@1',
          {
              'output': 'Received user details',
          },
      ),
      (
          'test_workflow_agent_input_schema',
          {
              'nodes': {
                  'NodeA_input': {'status': NodeStatus.COMPLETED.value},
                  'NodeB': {'status': NodeStatus.COMPLETED.value},
              }
          },
      ),
      ('test_workflow_agent_input_schema', testing_utils.END_OF_AGENT),
  ]
  if resumable:
    assert simplified_events2 == expected_events2
  else:
    # In V2 non-resumable mode, NodeA_input is skipped and does not yield output again.
    # So we filter out its output event.
    expected_non_resumable = [
        e
        for e in expected_events2
        if not (e[0].split('/')[-1].split('@')[0] == 'NodeA_input')
    ]
    expected_non_resumable = workflow_testing_utils.strip_checkpoint_events(
        expected_non_resumable
    )
    assert simplified_events2 == expected_non_resumable


@pytest.mark.asyncio
async def test_workflow_allows_mixing_output_and_request_input(
    request: pytest.FixtureRequest,
):
  """Tests that yielding both output and RequestInput is allowed in V2."""

  class _YieldOutputAndRequestInputNode(BaseNode):
    """A node that yields output and requests input."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = Field(default='')

    def __init__(self, *, name: str):
      super().__init__()
      object.__setattr__(self, 'name', name)

    @override
    def get_name(self) -> str:
      return self.name

    @override
    async def _run_impl(
        self,
        *,
        ctx: Context,
        node_input: Any,
    ) -> AsyncGenerator[Any, None]:
      yield Event(output='output 1')
      yield RequestInput(interrupt_id='req1')

  node_a = _YieldOutputAndRequestInputNode(name='NodeA')
  node_b = InputCapturingNode(name='NodeB')
  agent = Workflow(
      name='test_agent',
      edges=[
          (START, node_a),
          (node_a, node_b),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)

  events = await runner.run_async(testing_utils.get_user_content('start'))
  simplified = workflow_testing_utils.simplify_events_with_node_and_agent_state(
      events
  )

  # In V2, mixing output and interrupts is ALLOWED.
  # The node yields the output event and then the RequestInput event.
  assert len(simplified) == 2
  assert simplified[0] == (
      'test_agent@1/NodeA@1',
      {'output': 'output 1'},
  )
  assert simplified[1][0] == 'test_agent@1/NodeA@1'
  assert simplified[1][1].function_call.name == 'adk_request_input'
  assert simplified[1][1].function_call.args['interruptId'] == 'req1'


@pytest.mark.parametrize(
    'resumable', [False, pytest.param(True, marks=pytest.mark.xfail)]
)
@pytest.mark.asyncio
async def test_workflow_rerun_on_resume(
    request: pytest.FixtureRequest, resumable: bool
):
  """Tests node requests input and reruns itself upon resume."""

  class _RerunNode(BaseNode):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    rerun_on_resume: bool = Field(default=True)
    name: str = Field(default='')

    def __init__(self, *, name: str):
      super().__init__()
      object.__setattr__(self, 'name', name)

    @override
    def get_name(self) -> str:
      return self.name

    @override
    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      if 'count' not in ctx.session.state:
        ctx.session.state['count'] = 0

      approval = None
      if ctx.session.state['count'] == 0:
        if resume_input := ctx.resume_inputs.get('ask_approval'):
          ctx.session.state['count'] = 1
          approval = resume_input['approved']
        else:
          yield RequestInput(
              message='Needs approval', interrupt_id='ask_approval'
          )
          return
      yield Event(output={'approval': approval})

  node_a = _RerunNode(name='NodeA')
  agent = Workflow(
      name='test_agent',
      edges=[Edge(from_node=START, to_node=node_a)],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Run 1: node requests input
  events1 = await runner.run_async(testing_utils.get_user_content('start'))
  simplified_events1 = (
      workflow_testing_utils.simplify_events_with_node_and_agent_state(
          copy.deepcopy(events1),
      )
  )
  req_events = workflow_testing_utils.get_request_input_events(events1)
  assert len(req_events) == 1
  interrupt_id1 = get_request_input_interrupt_ids(req_events[0])[0]
  invocation_id = events1[0].invocation_id

  if resumable:
    assert simplified_events1[-1] == (
        'test_agent',
        {
            'nodes': {
                'NodeA': {
                    'status': NodeStatus.WAITING.value,
                    'interrupts': [interrupt_id1],
                },
            },
        },
    )

  # Run 2: provide input, node reruns and completes
  events2 = await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(interrupt_id1, {'approved': True})
      ),
      invocation_id=invocation_id,
  )
  simplified_events2 = (
      workflow_testing_utils.simplify_events_with_node_and_agent_state(
          copy.deepcopy(events2),
          include_resume_inputs=True,
      )
  )

  expected_events2 = [
      (
          'test_agent',
          {
              'nodes': {
                  'NodeA': {
                      'status': NodeStatus.RUNNING.value,
                      'resume_inputs': {interrupt_id1: {'approved': True}},
                  },
              }
          },
      ),
      (
          'test_agent@1/NodeA@1',
          {
              'output': {'approval': True},
          },
      ),
      (
          'test_agent',
          {
              'nodes': {
                  'NodeA': {'status': NodeStatus.COMPLETED.value},
              }
          },
      ),
      ('test_agent', testing_utils.END_OF_AGENT),
  ]
  if resumable:
    assert simplified_events2 == expected_events2
  else:
    assert simplified_events2 == (
        workflow_testing_utils.strip_checkpoint_events(expected_events2)
    )


@pytest.mark.parametrize(
    'resumable', [False, pytest.param(True, marks=pytest.mark.xfail)]
)
@pytest.mark.asyncio
async def test_workflow_rerun_with_multiple_inputs(
    request: pytest.FixtureRequest,
    resumable: bool,
):
  """Tests node with rerun_on_resume=True requests multiple inputs and resumed one by one."""

  class _RerunNodeWithTwoInputs(BaseNode):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    rerun_on_resume: bool = Field(default=True)
    name: str = Field(default='')

    def __init__(self, *, name: str):
      super().__init__()
      object.__setattr__(self, 'name', name)

    @override
    def get_name(self) -> str:
      return self.name

    @override
    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      if resume_input := ctx.resume_inputs.get('req1'):
        yield Event(state={'input1': resume_input['text']})
      if resume_input := ctx.resume_inputs.get('req2'):
        yield Event(state={'input2': resume_input['text']})

      if 'input1' not in ctx.state and 'req1' not in ctx.resume_inputs:
        yield RequestInput(message='input 1', interrupt_id='req1')
        return

      if 'input2' not in ctx.state and 'req2' not in ctx.resume_inputs:
        yield RequestInput(message='input 2', interrupt_id='req2')
        return

      input1 = ctx.resume_inputs['req1']['text']
      input2 = ctx.resume_inputs['req2']['text']
      yield Event(
          output={
              'input1': input1,
              'input2': input2,
          },
      )

  node_a = _RerunNodeWithTwoInputs(name='NodeA')
  agent = Workflow(
      name='test_agent',
      edges=[Edge(from_node=START, to_node=node_a)],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Run 1: node requests 1st input
  events1 = await runner.run_async(testing_utils.get_user_content('start'))
  simplified_events1 = (
      workflow_testing_utils.simplify_events_with_node_and_agent_state(
          copy.deepcopy(events1),
      )
  )
  req_events1 = workflow_testing_utils.get_request_input_events(events1)
  assert len(req_events1) == 1
  interrupt_id1 = get_request_input_interrupt_ids(req_events1[0])[0]
  assert interrupt_id1 == 'req1'
  invocation_id = events1[0].invocation_id
  if resumable:
    assert simplified_events1[-1] == (
        'test_agent',
        {
            'nodes': {
                'NodeA': {
                    'status': NodeStatus.WAITING.value,
                    'interrupts': [interrupt_id1],
                },
            },
        },
    )

  # Run 2: provide 1st input, node reruns and requests 2nd input
  events2 = await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(interrupt_id1, {'text': 'response 1'})
      ),
      invocation_id=invocation_id,
  )
  assert all(
      e.invocation_id == invocation_id for e in events2 if e.invocation_id
  )
  simplified_events2 = (
      workflow_testing_utils.simplify_events_with_node_and_agent_state(
          copy.deepcopy(events2),
          include_resume_inputs=True,
      )
  )
  req_events2 = workflow_testing_utils.get_request_input_events(events2)
  assert len(req_events2) == 1
  interrupt_id2 = get_request_input_interrupt_ids(req_events2[0])[0]
  assert interrupt_id2 == 'req2'

  expected_events2 = [
      (
          'test_agent',
          {
              'nodes': {
                  'NodeA': {
                      'status': NodeStatus.RUNNING.value,
                      'resume_inputs': {interrupt_id1: {'text': 'response 1'}},
                  },
              }
          },
      ),
      (
          'test_agent@1/NodeA@1',
          types.Part(
              function_call=types.FunctionCall(
                  name=REQUEST_INPUT_FUNCTION_CALL_NAME,
                  args={
                      'interruptId': 'req2',
                      'message': 'input 2',
                      'payload': None,
                      'response_schema': None,
                  },
              )
          ),
      ),
      (
          'test_agent',
          {
              'nodes': {
                  'NodeA': {
                      'status': NodeStatus.WAITING.value,
                      'interrupts': [interrupt_id2],
                      'resume_inputs': {interrupt_id1: {'text': 'response 1'}},
                  },
              },
          },
      ),
  ]
  if resumable:
    assert simplified_events2 == expected_events2
  else:
    assert simplified_events2 == (
        workflow_testing_utils.strip_checkpoint_events(expected_events2)
    )

  # Run 3: provide 2nd input, node reruns and completes
  events3 = await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(interrupt_id2, {'text': 'response 2'})
      ),
      invocation_id=invocation_id,
  )
  assert all(
      e.invocation_id == invocation_id for e in events3 if e.invocation_id
  )
  simplified_events3 = (
      workflow_testing_utils.simplify_events_with_node_and_agent_state(
          copy.deepcopy(events3),
          include_resume_inputs=True,
      )
  )

  expected_events3 = [
      (
          'test_agent',
          {
              'nodes': {
                  'NodeA': {
                      'status': NodeStatus.RUNNING.value,
                      'resume_inputs': {
                          interrupt_id1: {'text': 'response 1'},
                          interrupt_id2: {'text': 'response 2'},
                      },
                  },
              }
          },
      ),
      (
          'test_agent@1/NodeA@1',
          {
              'output': {'input1': 'response 1', 'input2': 'response 2'},
          },
      ),
      (
          'test_agent',
          {
              'nodes': {
                  'NodeA': {'status': NodeStatus.COMPLETED.value},
              }
          },
      ),
      ('test_agent', testing_utils.END_OF_AGENT),
  ]
  if resumable:
    assert simplified_events3 == expected_events3
  else:
    assert simplified_events3 == (
        workflow_testing_utils.strip_checkpoint_events(expected_events3)
    )


class _MultiHitlRerunNode(BaseNode):
  model_config = ConfigDict(arbitrary_types_allowed=True)

  rerun_on_resume: bool = Field(default=True)
  name: str = Field(default='')

  def __init__(self, *, name: str):
    super().__init__()
    object.__setattr__(self, 'name', name)

  @override
  def get_name(self) -> str:
    return self.name

  @override
  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    if not ctx.resume_inputs.get('req1'):
      yield RequestInput(interrupt_id='req1', message='request 1')
      return
    if not ctx.resume_inputs.get('req2'):
      yield RequestInput(interrupt_id='req2', message='request 2')
      return
    yield Event(output='final_output')


@pytest.mark.parametrize(
    'resumable', [False, pytest.param(True, marks=pytest.mark.xfail)]
)
@pytest.mark.asyncio
async def test_rerun_with_multiple_hitl_and_outputs(
    request: pytest.FixtureRequest,
    resumable: bool,
):
  """Tests that a re-runnable node with multiple HITL accumulates outputs."""
  node_a = _MultiHitlRerunNode(name='NodeA')
  node_b = InputCapturingNode(name='NodeB')
  agent = Workflow(
      name='test_agent_multi_hitl',
      edges=[
          (START, node_a),
          (node_a, node_b),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )
  session_service = InMemorySessionService()
  artifact_service = InMemoryArtifactService()
  memory_service = InMemoryMemoryService()
  runner1 = Runner(
      app=app,
      session_service=session_service,
      artifact_service=artifact_service,
      memory_service=memory_service,
  )
  runner2 = Runner(
      app=app,
      session_service=session_service,
      artifact_service=artifact_service,
      memory_service=memory_service,
  )
  runner3 = Runner(
      app=app,
      session_service=session_service,
      artifact_service=artifact_service,
      memory_service=memory_service,
  )
  session = await session_service.create_session(
      app_name=app.name, user_id='test_user'
  )

  async def collect_events(agen):
    events = []
    async for e in agen:
      events.append(e)
    return events

  # Run 1: node requests input1
  events1 = await collect_events(
      runner1.run_async(
          user_id=session.user_id,
          session_id=session.id,
          new_message=testing_utils.get_user_content('start'),
      )
  )
  req_events1 = workflow_testing_utils.get_request_input_events(events1)
  assert len(req_events1) == 1
  assert get_request_input_interrupt_ids(req_events1[0])[0] == 'req1'
  invocation_id = events1[0].invocation_id

  # Run 2: provide input1, node requests input2.
  events2 = await collect_events(
      runner2.run_async(
          user_id=session.user_id,
          session_id=session.id,
          new_message=testing_utils.UserContent(
              create_request_input_response('req1', {'text': 'response 1'})
          ),
          invocation_id=invocation_id if resumable else None,
      )
  )
  req_events2 = workflow_testing_utils.get_request_input_events(events2)
  assert len(req_events2) == 1
  assert get_request_input_interrupt_ids(req_events2[0])[0] == 'req2'

  # Run 3: provide input2, node yields final output and completes.
  await collect_events(
      runner3.run_async(
          user_id=session.user_id,
          session_id=session.id,
          new_message=testing_utils.UserContent(
              create_request_input_response('req2', {'text': 'response 2'})
          ),
          invocation_id=invocation_id if resumable else None,
      )
  )

  assert node_b.received_inputs == ['final_output']


@pytest.mark.parametrize(
    'resumable',
    [
        False,
        pytest.param(
            True, marks=pytest.mark.xfail(reason='Resumability broken in V2')
        ),
    ],
)
@pytest.mark.asyncio
async def test_rerun_on_resume_waits_for_all_interrupts(
    request: pytest.FixtureRequest,
    resumable: bool,
):
  """Tests that a rerun_on_resume node is not rerun until all pending interrupts are resolved."""

  class _SimultaneousInputsNode(BaseNode):
    """A node that requests multiple inputs simultaneously."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    rerun_on_resume: bool = Field(default=True)
    name: str = Field(default='')

    def __init__(self, *, name: str):
      super().__init__()
      object.__setattr__(self, 'name', name)

    @override
    def get_name(self) -> str:
      return self.name

    @override
    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      if resume_input := ctx.resume_inputs.get('req1'):
        yield Event(state={'input1': resume_input['text']})
      if resume_input := ctx.resume_inputs.get('req2'):
        yield Event(state={'input2': resume_input['text']})

      have_req1 = 'input1' in ctx.state or 'req1' in ctx.resume_inputs
      have_req2 = 'input2' in ctx.state or 'req2' in ctx.resume_inputs

      if not have_req1 or not have_req2:
        if not have_req1:
          yield RequestInput(interrupt_id='req1', message='input 1')
        if not have_req2:
          yield RequestInput(interrupt_id='req2', message='input 2')
        return

      val1 = ctx.state.get('input1') or ctx.resume_inputs['req1']['text']
      val2 = ctx.state.get('input2') or ctx.resume_inputs['req2']['text']

      yield Event(
          output={
              'input1': val1,
              'input2': val2,
          },
      )

  node_a = _SimultaneousInputsNode(name='NodeA')
  agent = Workflow(
      name='test_agent',
      edges=[Edge(from_node=START, to_node=node_a)],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Run 1: node requests both inputs simultaneously.
  events1 = await runner.run_async(testing_utils.get_user_content('start'))
  simplified1 = (
      workflow_testing_utils.simplify_events_with_node_and_agent_state(
          copy.deepcopy(events1),
          include_resume_inputs=True,
      )
  )
  req_events1 = workflow_testing_utils.get_request_input_events(events1)
  assert len(req_events1) == 2
  interrupt_ids = []
  for e in req_events1:
    interrupt_ids.extend(get_request_input_interrupt_ids(e))
  assert set(interrupt_ids) == {'req1', 'req2'}
  invocation_id = events1[0].invocation_id

  # Final checkpoint should show WAITING with both interrupt_ids.
  if resumable:
    final_state1 = simplified1[-1][1]
    assert final_state1['nodes']['NodeA']['status'] == (
        NodeStatus.WAITING.value
    )
    assert set(final_state1['nodes']['NodeA']['interrupts']) == {
        'req1',
        'req2',
    }

  # Run 2: provide only req1 — node should stay WAITING, NOT rerun.
  events2 = await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response('req1', {'text': 'response 1'})
      ),
      invocation_id=invocation_id,
  )
  simplified2 = (
      workflow_testing_utils.simplify_events_with_node_and_agent_state(
          copy.deepcopy(events2),
          include_resume_inputs=True,
      )
  )

  # Node should remain WAITING with req2 still pending.
  # resume_inputs should accumulate req1's response.
  if resumable:
    final_state2 = simplified2[-1][1]
    assert final_state2['nodes']['NodeA']['status'] == (
        NodeStatus.WAITING.value
    )
    assert final_state2['nodes']['NodeA']['interrupts'] == ['req2']
    assert final_state2['nodes']['NodeA']['resume_inputs'] == {
        'req1': {'text': 'response 1'},
    }

  # The node should NOT have produced any RequestInput or data output in resumable mode.
  # In non-resumable mode, it re-yields the pending interrupt 'req2'.
  req_events2 = workflow_testing_utils.get_request_input_events(events2)
  if resumable:
    assert len(req_events2) == 0
  else:
    assert len(req_events2) == 1
    assert get_request_input_interrupt_ids(req_events2[0]) == ['req2']

  # Run 3: provide req2 — now all interrupts resolved, node should rerun.
  events3 = await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response('req2', {'text': 'response 2'})
      ),
      invocation_id=invocation_id,
  )
  simplified3 = (
      workflow_testing_utils.simplify_events_with_node_and_agent_state(
          copy.deepcopy(events3),
          include_resume_inputs=True,
      )
  )

  # Node should have rerun and completed with both responses.
  # Last event is END_OF_AGENT, second-to-last is the final agent state.
  if resumable:
    final_state3 = simplified3[-2][1]
    assert final_state3['nodes']['NodeA']['status'] == (
        NodeStatus.COMPLETED.value
    )

  # Check the node produced the expected output (exclude workflow output).
  data_events = [
      e
      for e in events3
      if hasattr(e, 'node_info')
      and e.output is not None
      and isinstance(e.output, dict)
      and e.node_info.path.startswith(agent.name)
  ]
  assert len(data_events) == 1
  assert data_events[0].output == {
      'input1': 'response 1',
      'input2': 'response 2',
  }


# ---------------------------------------------------------------------------
# unwrap_response tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'resumable', [False, pytest.param(True, marks=pytest.mark.xfail)]
)
@pytest.mark.asyncio
async def test_wrapped_response_unwrapped_for_node(
    request: pytest.FixtureRequest, resumable: bool
):
  """Wrapped {"result": value} is unwrapped so the node receives the value."""
  from google.adk.workflow import FunctionNode

  def my_node():
    return RequestInput(interrupt_id='ask1', message='Give me data')

  node_a = FunctionNode(func=my_node)
  node_b = InputCapturingNode(name='NodeB')
  app = App(
      name=request.function.__name__,
      root_agent=Workflow(
          name='test_agent',
          edges=[(START, node_a), (node_a, node_b)],
      ),
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  events1 = await runner.run_async(testing_utils.get_user_content('go'))
  req_events = workflow_testing_utils.get_request_input_events(events1)
  assert len(req_events) == 1
  interrupt_id = get_request_input_interrupt_ids(req_events[0])[0]
  invocation_id = events1[0].invocation_id

  # Resume with a wrapped response (simulates adk web after rewrapping).
  await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(
              interrupt_id,
              _wrap_response('hello world'),
          )
      ),
      invocation_id=invocation_id,
  )

  # NodeB should receive the plain string, not {"result": "hello world"}.
  assert node_b.received_inputs == ['hello world']


@pytest.mark.parametrize(
    'resumable', [False, pytest.param(True, marks=pytest.mark.xfail)]
)
@pytest.mark.asyncio
async def test_dict_response_not_unwrapped(
    request: pytest.FixtureRequest, resumable: bool
):
  """A dict response without single "result" key passes through unchanged."""
  from google.adk.workflow import FunctionNode

  def my_node():
    return RequestInput(interrupt_id='ask1', message='Give me data')

  node_a = FunctionNode(func=my_node)
  node_b = InputCapturingNode(name='NodeB')
  app = App(
      name=request.function.__name__,
      root_agent=Workflow(
          name='test_agent',
          edges=[(START, node_a), (node_a, node_b)],
      ),
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  events1 = await runner.run_async(testing_utils.get_user_content('go'))
  req_events = workflow_testing_utils.get_request_input_events(events1)
  assert len(req_events) == 1
  interrupt_id = get_request_input_interrupt_ids(req_events[0])[0]
  invocation_id = events1[0].invocation_id

  # Resume with a raw dict (programmatic API or adk web with JSON dict input).
  raw_dict = {'name': 'John', 'age': 30}
  await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(interrupt_id, raw_dict)
      ),
      invocation_id=invocation_id,
  )

  # NodeB should receive the dict as-is.
  assert node_b.received_inputs == [{'name': 'John', 'age': 30}]


@pytest.mark.parametrize('resumable', [False, True])
@pytest.mark.asyncio
async def test_request_input_rerun_with_same_interrupt_id(
    request: pytest.FixtureRequest, resumable: bool
):
  """Reusing the same interrupt_id across loop iterations works.

  Regression test: state reconstruction matched FCs and FRs by set
  membership, so a previous FR with the same ID made the current
  interrupt appear "already resolved", causing the workflow to
  restart from scratch instead of resuming.
  """

  @node(rerun_on_resume=True)
  def review(ctx: Context):
    resume = ctx.resume_inputs.get('review')
    if not resume:
      yield RequestInput(
          interrupt_id='review',
          message='Approve or revise?',
      )
      return
    if resume == 'approve':
      yield Event(output='approved', route='approved')
    else:
      yield Event(route='revise')

  def process():
    return 'draft'

  capture = InputCapturingNode(name='capture')
  agent = Workflow(
      name='test_rerun_same_id',
      edges=[
          (START, process, review),
          (review, {'revise': process, 'approved': capture}),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Turn 1: start → process → review → interrupt
  events1 = await runner.run_async(testing_utils.get_user_content('go'))
  req1 = workflow_testing_utils.get_request_input_events(events1)
  assert len(req1) == 1
  assert 'review@1' in req1[0].node_info.path
  inv_id = events1[0].invocation_id

  # Turn 2: revise → process reruns → review reruns → interrupt again
  events2 = await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response('review', {'result': 'revise'})
      ),
      invocation_id=inv_id,
  )
  req2 = workflow_testing_utils.get_request_input_events(events2)
  assert len(req2) == 1, 'Expected second interrupt after revise'
  assert 'review@2' in req2[0].node_info.path
  inv_id = events2[0].invocation_id

  # Turn 3: approve → should complete, not loop
  events3 = await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response('review', {'result': 'approve'})
      ),
      invocation_id=inv_id,
  )
  req3 = workflow_testing_utils.get_request_input_events(events3)
  assert len(req3) == 0, 'Should not interrupt again after approve'
  assert capture.received_inputs == ['approved']


# ---------------------------------------------------------------------------
# auth_config tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'resumable', [False, pytest.param(True, marks=pytest.mark.xfail)]
)
@pytest.mark.asyncio
async def test_function_node_auth_config(
    request: pytest.FixtureRequest, resumable: bool
):
  """FunctionNode with auth_config pauses for auth, then runs after creds."""
  from fastapi.openapi.models import APIKey
  from fastapi.openapi.models import APIKeyIn
  from google.adk.auth.auth_credential import AuthCredential
  from google.adk.auth.auth_credential import AuthCredentialTypes
  from google.adk.auth.auth_tool import AuthConfig
  from google.adk.workflow import FunctionNode

  auth_config = AuthConfig(
      auth_scheme=APIKey(**{'in': APIKeyIn.header, 'name': 'X-Api-Key'}),
      raw_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.API_KEY,
          api_key='placeholder',
      ),
      credential_key='test_api_key',
  )

  call_count = 0
  received_cred = None

  def do_work(ctx: Context):
    nonlocal call_count, received_cred
    call_count += 1
    received_cred = ctx.get_auth_response(auth_config)
    return {'result': 'authed'}

  node_a = FunctionNode(
      func=do_work, auth_config=auth_config, rerun_on_resume=True
  )
  node_b = InputCapturingNode(name='NodeB')
  app = App(
      name=request.function.__name__,
      root_agent=Workflow(
          name='test_agent',
          edges=[(START, node_a), (node_a, node_b)],
      ),
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Run 1: should pause for auth.
  events1 = await runner.run_async(testing_utils.get_user_content('go'))

  auth_fc_events = workflow_testing_utils.get_auth_request_events(events1)
  assert len(auth_fc_events) == 1
  fc = auth_fc_events[0].content.parts[0].function_call
  auth_fc_id = fc.id
  invocation_id = events1[0].invocation_id
  assert call_count == 0

  # Run 2: provide auth credential — node should execute.
  auth_response = AuthConfig(
      auth_scheme=auth_config.auth_scheme,
      raw_auth_credential=auth_config.raw_auth_credential,
      exchanged_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.API_KEY,
          api_key='real_api_key_123',
      ),
      credential_key='test_api_key',
  )

  resume_part = types.Part(
      function_response=types.FunctionResponse(
          id=auth_fc_id,
          name=REQUEST_CREDENTIAL_FUNCTION_CALL_NAME,
          response=auth_response.model_dump(exclude_none=True, by_alias=True),
      )
  )
  await runner.run_async(
      new_message=testing_utils.UserContent(resume_part),
      invocation_id=invocation_id,
  )

  assert call_count == 1
  assert received_cred is not None
  assert received_cred.api_key == 'real_api_key_123'
  assert node_b.received_inputs == [{'result': 'authed'}]


@pytest.mark.parametrize(
    'resumable', [False, pytest.param(True, marks=pytest.mark.xfail)]
)
@pytest.mark.asyncio
async def test_second_auth_node_skips_auth_when_credential_exists(
    request: pytest.FixtureRequest, resumable: bool
):
  """Second FunctionNode with same credential_key skips auth if cred already stored."""
  from fastapi.openapi.models import APIKey
  from fastapi.openapi.models import APIKeyIn
  from google.adk.auth.auth_credential import AuthCredential
  from google.adk.auth.auth_credential import AuthCredentialTypes
  from google.adk.auth.auth_tool import AuthConfig

  auth_config = AuthConfig(
      auth_scheme=APIKey(**{'in': APIKeyIn.header, 'name': 'X-Api-Key'}),
      raw_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.API_KEY,
          api_key='placeholder',
      ),
      credential_key='shared_key',
  )

  call_log = []

  @node(auth_config=auth_config, rerun_on_resume=True)
  def first_task():
    call_log.append('first')
    return {'status': 'done'}

  @node(auth_config=auth_config, rerun_on_resume=True)
  def second_task():
    call_log.append('second')
    return {'status': 'done'}

  node_a = first_task
  node_b = second_task
  sink = InputCapturingNode(name='sink')

  app = App(
      name=request.function.__name__,
      root_agent=Workflow(
          name='test_agent',
          edges=[(START, node_a), (node_a, node_b), (node_b, sink)],
      ),
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Run 1: node_a pauses for auth.
  events1 = await runner.run_async(testing_utils.get_user_content('go'))
  auth_fc_events = workflow_testing_utils.get_auth_request_events(events1)
  assert len(auth_fc_events) == 1
  fc = auth_fc_events[0].content.parts[0].function_call
  auth_fc_id = fc.id
  invocation_id = events1[0].invocation_id
  assert not call_log

  # Run 2: provide credential — node_a runs, node_b should skip auth and run too.
  auth_response = AuthConfig(
      auth_scheme=auth_config.auth_scheme,
      raw_auth_credential=auth_config.raw_auth_credential,
      exchanged_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.API_KEY,
          api_key='the_real_key',
      ),
      credential_key='shared_key',
  )
  resume_part = types.Part(
      function_response=types.FunctionResponse(
          id=auth_fc_id,
          name=REQUEST_CREDENTIAL_FUNCTION_CALL_NAME,
          response=auth_response.model_dump(exclude_none=True, by_alias=True),
      )
  )
  await runner.run_async(
      new_message=testing_utils.UserContent(resume_part),
      invocation_id=invocation_id,
  )

  # Both nodes ran — node_b did NOT pause for a second auth request.
  assert call_log == ['first', 'second']
  assert sink.received_inputs == [{'status': 'done'}]


# --- Tests for input/triggered_by restoration on resume ---


class _InputCapturingRerunNode(BaseNode):
  """A rerun_on_resume node that captures node_input and triggered_by."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  rerun_on_resume: bool = Field(default=True)
  captured_inputs: list[Any] = Field(default_factory=list)

  @override
  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    self.captured_inputs.append(node_input)

    if resume_input := ctx.resume_inputs.get('approval'):
      yield Event(output={'approved': resume_input})
    else:
      yield RequestInput(message='Need approval', interrupt_id='approval')


@pytest.mark.parametrize(
    'resumable',
    [
        pytest.param(
            False,
            marks=pytest.mark.xfail(reason='Fails in non-resumable mode'),
        ),
        True,
    ],
)
@pytest.mark.asyncio
async def test_resume_preserves_node_input(
    request: pytest.FixtureRequest, resumable: bool
):
  """After resume, a rerun node receives the predecessor's output as input."""
  node_a = _TestingNode(name='NodeA', message='output_from_a')
  node_b = _InputCapturingRerunNode(name='NodeB')
  node_c = InputCapturingNode(name='NodeC')

  agent = Workflow(
      name='test_agent',
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=node_b),
          Edge(from_node=node_b, to_node=node_c),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Run 1: NodeA completes, NodeB interrupts.
  events1 = await runner.run_async(testing_utils.get_user_content('go'))
  invocation_id = events1[0].invocation_id
  req_events = workflow_testing_utils.get_request_input_events(events1)
  assert len(req_events) == 1
  interrupt_id = get_request_input_interrupt_ids(req_events[0])[0]

  # First run should have received NodeA's output.
  assert len(node_b.captured_inputs) == 1
  assert node_b.captured_inputs[0] == 'output_from_a'

  # Run 2: resume with approval.
  events2 = await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(interrupt_id, {'yes': True})
      ),
      invocation_id=invocation_id,
  )

  # Second run (rerun on resume) should also receive NodeA's output.
  assert len(node_b.captured_inputs) == 2
  assert node_b.captured_inputs[1] == 'output_from_a'

  # NodeC should have received NodeB's output.
  assert len(node_c.received_inputs) == 1
  assert node_c.received_inputs[0] == {'approved': {'yes': True}}


@pytest.mark.parametrize(
    'resumable',
    [
        pytest.param(
            False,
            marks=pytest.mark.xfail(reason='Fails in non-resumable mode'),
        ),
        True,
    ],
)
@pytest.mark.asyncio
async def test_resume_preserves_input_from_start(
    request: pytest.FixtureRequest, resumable: bool
):
  """After resume, a node directly after START receives workflow input."""
  node_a = _InputCapturingRerunNode(name='NodeA')

  agent = Workflow(
      name='test_agent',
      edges=[Edge(from_node=START, to_node=node_a)],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Run 1: NodeA interrupts.
  events1 = await runner.run_async(testing_utils.get_user_content('hello'))
  invocation_id = events1[0].invocation_id
  req_events = workflow_testing_utils.get_request_input_events(events1)
  assert len(req_events) == 1
  interrupt_id = get_request_input_interrupt_ids(req_events[0])[0]

  # First run: triggered_by should be START.
  assert len(node_a.captured_inputs) == 1

  # Run 2: resume.
  await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(interrupt_id, {'ok': True})
      ),
      invocation_id=invocation_id,
  )

  # Second run: triggered_by should still be START.
  assert len(node_a.captured_inputs) == 2


@pytest.mark.parametrize(
    'resumable',
    [
        pytest.param(
            False,
            marks=pytest.mark.xfail(reason='Fails in non-resumable mode'),
        ),
        True,
    ],
)
@pytest.mark.asyncio
async def test_resume_fan_in_both_predecessors_completed(
    request: pytest.FixtureRequest, resumable: bool
):
  """Fan-in: node C has two predecessors (A, B) that both completed.

  After resume, _find_predecessor_input should pick one of the available
  predecessor outputs for C.
  """
  node_a = _TestingNode(name='NodeA', message='output_from_a')
  node_b = _TestingNode(name='NodeB', message='output_from_b')
  node_c = _InputCapturingRerunNode(name='NodeC')

  agent = Workflow(
      name='test_agent',
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=START, to_node=node_b),
          Edge(from_node=node_a, to_node=node_c),
          Edge(from_node=node_b, to_node=node_c),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Run 1: A and B complete, C interrupts.
  events1 = await runner.run_async(testing_utils.get_user_content('go'))
  invocation_id = events1[0].invocation_id

  # C should have been triggered twice (once per predecessor).
  req_events = workflow_testing_utils.get_request_input_events(events1)
  assert len(req_events) >= 1
  interrupt_id = get_request_input_interrupt_ids(req_events[0])[0]

  # First run: C's input should come from one of its predecessors.
  assert len(node_c.captured_inputs) >= 1
  assert node_c.captured_inputs[0] in ('output_from_a', 'output_from_b')

  # Run 2: resume with approval.
  events2 = await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(interrupt_id, {'yes': True})
      ),
      invocation_id=invocation_id,
  )

  # After resume, C should again receive a valid predecessor output.
  assert len(node_c.captured_inputs) >= 2
  assert node_c.captured_inputs[-1] in ('output_from_a', 'output_from_b')


@pytest.mark.parametrize(
    'resumable',
    [
        pytest.param(
            False,
            marks=pytest.mark.xfail(reason='Fails in non-resumable mode'),
        ),
        True,
    ],
)
@pytest.mark.asyncio
async def test_resume_loop_receives_latest_input(
    request: pytest.FixtureRequest, resumable: bool
):
  """Loop: START -> A -> B --(loop)--> A.

  On the first iteration A receives START input and interrupts.
  After resume, A completes and triggers B. B routes back to A.
  On the loop-back iteration, A should receive B's output (not START
  input) and run_id should increment.

  Captures:
    [0] = first run (START input, interrupts)
    [1] = rerun on resume (START input, completes with approval)
    [2] = loop-back from B (B's output, interrupts again)
  """
  from google.adk.workflow._graph import Edge as GraphEdge

  class _RoutingNode(BaseNode):
    """A node that produces output with a route."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    route: str = Field(default='')
    message: str = Field(default='')

    @override
    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield Event(output=self.message, route=self.route)

  node_a = _InputCapturingRerunNode(name='NodeA')
  node_b = _RoutingNode(name='NodeB', message='output_from_b', route='loop')

  agent = Workflow(
      name='test_agent',
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=node_b),
          GraphEdge(from_node=node_b, to_node=node_a, route='loop'),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Run 1: A interrupts on first iteration.
  events1 = await runner.run_async(testing_utils.get_user_content('hello'))
  invocation_id = events1[0].invocation_id
  req_events = workflow_testing_utils.get_request_input_events(events1)
  assert len(req_events) == 1
  interrupt_id = get_request_input_interrupt_ids(req_events[0])[0]

  # First iteration: triggered by START.
  assert len(node_a.captured_inputs) == 1

  # Run 2: resume A -> A completes -> B fires -> B routes 'loop' -> A runs again.
  events2 = await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(interrupt_id, {'ok': True})
      ),
      invocation_id=invocation_id,
  )

  # Three captures: first run, rerun on resume, loop-back from B.
  assert len(node_a.captured_inputs) == 3

  # Capture[1] = rerun on resume, still triggered by START.

  # Capture[2] = loop-back from B with B's output.
  assert node_a.captured_inputs[2] == 'output_from_b'

  # run_id should have incremented (visible in event paths).
  node_a_paths = [
      e.node_info.path
      for e in events1 + events2
      if e.node_info and e.node_info.path and 'NodeA@' in e.node_info.path
  ]
  run_ids = sorted({p.split('NodeA@')[1].split('/')[0] for p in node_a_paths})
  assert len(run_ids) >= 2, f'Expected multiple run_ids, got {run_ids}'


@pytest.mark.asyncio
async def test_multiple_invocations_isolation(request: pytest.FixtureRequest):
  """Verify that a new invocation ignores events from a previous invocation."""

  class CounterNode(BaseNode):
    name: str = Field(default='counter_node')
    run_count: int = Field(default=0)

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      self.run_count += 1
      yield f'Run {self.run_count}'

  node_a = CounterNode()
  wf = Workflow(name='wf', edges=[(START, node_a)])

  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Invocation 1
  msg1 = types.Content(parts=[types.Part(text='go 1')], role='user')
  events1 = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg1
  ):
    events1.append(event)

  assert node_a.run_count == 1

  # Invocation 2 (New invocation in SAME session)
  msg2 = types.Content(parts=[types.Part(text='go 2')], role='user')
  events2 = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg2
  ):
    events2.append(event)

  # If isolation works, CounterNode should run AGAIN!
  assert node_a.run_count == 2


@pytest.mark.asyncio
async def test_multiple_pending_interrupts_isolation(
    request: pytest.FixtureRequest,
):
  """Verify that responding to one interrupt resumes its specific invocation and ignores others."""

  class InterruptNode(BaseNode):
    name: str = Field(default='interrupt_node')
    rerun_on_resume: bool = Field(default=True)

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      text = node_input.parts[0].text if node_input else ''
      if ctx.resume_inputs and 'req1' in ctx.resume_inputs:
        yield Event(output=f"Resumed 1: {ctx.resume_inputs['req1']['ans']}")
        return
      if ctx.resume_inputs and 'req2' in ctx.resume_inputs:
        yield Event(output=f"Resumed 2: {ctx.resume_inputs['req2']['ans']}")
        return

      fc_id = 'req1' if '1' in text else 'req2'
      yield Event(
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name=REQUEST_INPUT_FUNCTION_CALL_NAME,
                          args={
                              'interruptId': fc_id,
                              'message': f'input {fc_id}',
                          },
                          id=fc_id,
                      )
                  )
              ]
          ),
          long_running_tool_ids={fc_id},
      )
      return

  node = InterruptNode()
  wf = Workflow(name='wf', edges=[(START, node)])

  runner = testing_utils.InMemoryRunner(node=wf)

  # Invocation 1: yields req1
  events1 = await runner.run_async('go 1')

  # Find the function call ID generated for req1
  fc_event = workflow_testing_utils.find_function_call_event(
      events1, REQUEST_INPUT_FUNCTION_CALL_NAME
  )
  function_call_id = fc_event.content.parts[0].function_call.id

  # Invocation 2: yields req2 (New run, same session)
  events2 = await runner.run_async('go 2')

  # Invocation 3: respond to req1
  msg3 = types.Content(
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name=REQUEST_INPUT_FUNCTION_CALL_NAME,
                  id=function_call_id,
                  response={'ans': 'val1'},
              )
          )
      ],
      role='user',
  )
  events3 = await runner.run_async(msg3)

  # Verify that Invocation 1 resumed and produced output
  outputs3 = [e.output for e in events3 if e.output is not None]
  assert 'Resumed 1: val1' in outputs3


@pytest.mark.parametrize('resumable', [False, True])
@pytest.mark.asyncio
async def test_parallel_nodes_trigger_same_hitl_node(
    request: pytest.FixtureRequest, resumable: bool
):
  """Tests that two nodes running in parallel can trigger the same node with HITL."""

  @node(rerun_on_resume=True)
  def node_c(ctx: Context, node_input: Any):
    interrupt_id = f'req_c_{node_input}'
    if interrupt_id not in ctx.resume_inputs:
      yield RequestInput(interrupt_id=interrupt_id, message='input for c')
      return
    yield Event(
        output=f"c_{node_input}_{ctx.resume_inputs[interrupt_id]['text']}"
    )

  def node_a():
    return 'from_a'

  def node_b():
    return 'from_b'

  agent = Workflow(
      name='test_parallel_hitl',
      edges=[
          (START, (node_a, node_b), node_c),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )

  runner = testing_utils.InMemoryRunner(app=app)

  # Run 1: should pause on first branch that triggers NodeC
  events1 = await runner.run_async(testing_utils.get_user_content('start'))
  req_events1 = workflow_testing_utils.get_request_input_events(events1)
  assert len(req_events1) == 1

  from google.adk.workflow.utils._workflow_hitl_utils import get_request_input_interrupt_ids

  interrupt_id_1 = get_request_input_interrupt_ids(req_events1[0])[0]
  # The first trigger for node_c would be from node_a
  assert interrupt_id_1 == 'req_c_from_a'

  invocation_id = events1[0].invocation_id

  # Run 2: resume first interrupt. This should complete node_c for 'from_a',
  # and then allow the buffered trigger from 'node_b' to be processed,
  # yielding the second interrupt.
  events2 = await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(interrupt_id_1, {'text': 'response a'})
      ),
      invocation_id=invocation_id,
  )

  req_events2 = workflow_testing_utils.get_request_input_events(events2)
  assert len(req_events2) == 1
  interrupt_id_2 = get_request_input_interrupt_ids(req_events2[0])[0]
  assert interrupt_id_2 == 'req_c_from_b'

  outputs2 = workflow_testing_utils.get_outputs(events2)
  assert 'c_from_a_response a' in outputs2

  # Run 3: resume second interrupt. This should complete node_c for 'from_b'.
  events3 = await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(interrupt_id_2, {'text': 'response b'})
      ),
      invocation_id=invocation_id,
  )
  outputs3 = workflow_testing_utils.get_outputs(events3)
  assert 'c_from_b_response b' in outputs3


@pytest.mark.asyncio
async def test_trigger_buffer_insertion_order_deterministic(
    request: pytest.FixtureRequest,
):
  """Tests that trigger buffer processes triggers in arrival order."""
  from google.adk.workflow.utils._workflow_hitl_utils import get_request_input_interrupt_ids
  from google.genai import types

  @node(rerun_on_resume=True)
  def node_d(ctx: Context, node_input: Any):
    interrupt_id = f'req_d_{node_input}'
    if interrupt_id not in ctx.resume_inputs:
      yield RequestInput(interrupt_id=interrupt_id, message='input for d')
      return
    yield Event(output=f'd_done_{node_input}')

  @node(rerun_on_resume=True)
  def node_e(ctx: Context, node_input: Any):
    interrupt_id = f'req_e_{node_input}'
    if interrupt_id not in ctx.resume_inputs:
      yield RequestInput(interrupt_id=interrupt_id, message='input for e')
      return
    yield Event(output=f'e_done_{node_input}')

  def source_a():
    return 'from_a'

  def source_b():
    return 'from_b'

  agent = Workflow(
      name='test_trigger_order',
      max_concurrency=1,
      edges=[
          # Start node_d and node_e to make them busy (WAITING)
          (START, node_d),
          (START, node_e),
          # Then source_a and source_b run and trigger them again
          (START, source_a, node_d),
          (START, source_b, node_e),
      ],
  )

  app_instance = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )

  runner = testing_utils.InMemoryRunner(app=app_instance)

  # Run 1: all start, node_d and node_e become WAITING.
  # source_a and source_b complete and buffer triggers.
  events1 = await runner.run_async(testing_utils.get_user_content('start'))
  req_events1 = workflow_testing_utils.get_request_input_events(events1)

  # There should be 2 interrupts (from the initial START edges)
  assert len(req_events1) == 2

  interrupt_ids = []
  for e in req_events1:
    interrupt_ids.extend(get_request_input_interrupt_ids(e))

  id_d = next(id for id in interrupt_ids if 'req_d' in id)
  id_e = next(id for id in interrupt_ids if 'req_e' in id)

  invocation_id = events1[0].invocation_id

  # Run 2: We resolve BOTH initial interrupts.
  # This frees up node_d and node_e.
  # _schedule_ready_nodes will process the buffered triggers.
  # Since trigger for node_d was buffered first, and max_concurrency=1,
  # only node_d will be picked up!
  msg = types.Content(
      parts=[
          create_request_input_response(id_d, {'text': 'resolved d'}),
          create_request_input_response(id_e, {'text': 'resolved e'}),
      ],
      role='user',
  )

  events2 = await runner.run_async(
      new_message=msg,
      invocation_id=invocation_id,
  )

  req_events2 = workflow_testing_utils.get_request_input_events(events2)
  # Both interrupts are returned sequentially because yielding an interrupt frees the concurrency slot.
  assert len(req_events2) == 2
  interrupt_id_2_first = get_request_input_interrupt_ids(req_events2[0])[0]
  interrupt_id_2_second = get_request_input_interrupt_ids(req_events2[1])[0]
  # We assert that the FIRST trigger (for from_a) was processed FIRST!
  assert interrupt_id_2_first == 'req_d_from_a'
  assert interrupt_id_2_second == 'req_e_from_b'
