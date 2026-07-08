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

import copy
from typing import Any

from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.tools._node_tool import NodeTool
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.workflow import JoinNode
from google.adk.workflow import node
from google.adk.workflow import START
from google.adk.workflow._node_status import NodeStatus
from google.adk.workflow._workflow import Workflow
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
from google.adk.workflow.utils._workflow_hitl_utils import get_request_input_interrupt_ids
from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_INPUT_FUNCTION_CALL_NAME
from google.genai import types
from pydantic import BaseModel
from pydantic import Field
import pytest

from . import workflow_testing_utils
from .. import testing_utils
from .workflow_testing_utils import RequestInputNode


class UserInfo(BaseModel):
  name: str
  age: int


class DummyRequest(BaseModel):
  request: str = ''


def test_node_tool_requires_input_schema():
  """NodeTool raises ValueError if wrapped node has no input_schema."""
  wf = Workflow(name='no_schema_wf', edges=[])
  with pytest.raises(ValueError, match='does not have an input_schema defined'):
    NodeTool(node=wf)


@pytest.mark.skip(reason='Requires CL 2 subagent branch refactor')
@pytest.mark.asyncio
async def test_workflow_as_tool_hitl_resume(request: pytest.FixtureRequest):
  """Workflow-as-a-tool suspends on RequestInput and resumes successfully.

  Setup:
    - LlmAgent 'parent_agent' uses WorkflowTool 'collect_user_info_tool'.
    - The tool wraps 'sub_workflow' which has a RequestInputNode and a
    format_response node.
  Act:
    - Turn 1: Run with 'Start task'. The model calls the tool, which suspends.
    - Turn 2: Resume with the user input response to the interrupt.
  Assert:
    - Turn 1: Event history contains the RequestInput function call.
    - Turn 2: The workflow tool resumes and finishes, and parent agent produces
    final text response.
  """
  # 1. Define the sub-workflow that has an input interrupt
  input_node = RequestInputNode(
      name='input_node',
      message='What is your name and age?',
      response_schema=UserInfo.model_json_schema(),
  )

  def format_response(node_input: dict[str, Any]):
    yield Event(
        output=f"User {node_input['name']} is {node_input['age']} years old."
    )

  sub_workflow = Workflow(
      name='sub_workflow',
      edges=[
          (START, input_node),
          (input_node, format_response),
      ],
  )
  sub_workflow.input_schema = DummyRequest

  # 2. Wrap the sub-workflow as a WorkflowTool
  wf_tool = NodeTool(
      node=sub_workflow,
      name='collect_user_info_tool',
      description='Call this tool to collect customer name and age.',
  )

  # 3. Define the parent agent that calls this tool
  # In the first turn, the model decides to call the tool.
  # In the second turn, after the tool resumes and returns output, the model replies to the user.
  parent_agent = LlmAgent(
      name='parent_agent',
      model=testing_utils.MockModel.create(
          responses=[
              types.Part.from_function_call(
                  name='collect_user_info_tool',
                  args={},
              ),
              types.Part.from_text(
                  text='Thank you! I received the user details.'
              ),
          ]
      ),
      tools=[wf_tool],
  )

  # 4. Wrap the parent agent in an App with resumability enabled
  app = App(
      name=request.function.__name__,
      root_agent=parent_agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Turn 1: Run the agent, triggering the tool call.
  # The sub-workflow starts, hits the RequestInputNode, and suspends.
  user_event = testing_utils.get_user_content('Start task')
  events1 = await runner.run_async(user_event)

  simplified_events1 = (
      workflow_testing_utils.simplify_events_with_node_and_agent_state(
          copy.deepcopy(events1),
      )
  )

  # Verify that we got a RequestInput event
  request_input_event = workflow_testing_utils.find_function_call_event(
      events1, REQUEST_INPUT_FUNCTION_CALL_NAME
  )
  assert request_input_event is not None
  args = request_input_event.content.parts[0].function_call.args
  assert args['message'] == 'What is your name and age?'

  interrupt_id = get_request_input_interrupt_ids(request_input_event)[0]
  invocation_id = request_input_event.invocation_id

  # Turn 2: Resume with the user input resolving the interrupt.
  user_input = create_request_input_response(
      interrupt_id, {'name': 'Alice', 'age': 25}
  )
  events2 = await runner.run_async(
      new_message=testing_utils.UserContent(user_input),
      invocation_id=invocation_id,
  )

  simplified_events2 = (
      workflow_testing_utils.simplify_events_with_node_and_agent_state(
          copy.deepcopy(events2),
      )
  )

  # Verify the tool workflow finished executing, returned the output,
  # and the parent agent LLM produced its final response.
  text_responses = [
      event.content.parts[0].text
      for event in events2
      if event.content and event.content.parts and event.content.parts[0].text
  ]
  assert 'Thank you! I received the user details.' in text_responses


@pytest.mark.skip(reason='Requires CL 2 subagent branch refactor')
@pytest.mark.asyncio
async def test_workflow_as_tool_hitl_resume_non_resumable_app(
    request: pytest.FixtureRequest,
):
  """Workflow-as-a-tool suspends and resumes successfully even when the App has resumability disabled."""
  # 1. Define the sub-workflow that has an input interrupt
  input_node = RequestInputNode(
      name='input_node',
      message='What is your name and age?',
      response_schema=UserInfo.model_json_schema(),
  )

  def format_response(node_input: dict[str, Any]):
    yield Event(
        output=f"User {node_input['name']} is {node_input['age']} years old."
    )

  sub_workflow = Workflow(
      name='sub_workflow',
      edges=[
          (START, input_node),
          (input_node, format_response),
      ],
  )
  sub_workflow.input_schema = DummyRequest

  # 2. Wrap the sub-workflow as a WorkflowTool
  wf_tool = NodeTool(
      node=sub_workflow,
      name='collect_user_info_tool',
      description='Call this tool to collect customer name and age.',
  )

  # 3. Define the parent agent that calls this tool
  parent_agent = LlmAgent(
      name='parent_agent',
      model=testing_utils.MockModel.create(
          responses=[
              types.Part.from_function_call(
                  name='collect_user_info_tool',
                  args={},
              ),
              types.Part.from_text(
                  text='Thank you! I received the user details.'
              ),
          ]
      ),
      tools=[wf_tool],
  )

  # 4. Wrap the parent agent in an App with resumability disabled
  app = App(
      name=request.function.__name__,
      root_agent=parent_agent,
      resumability_config=None,
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Turn 1: Run the agent, triggering the tool call.
  user_event = testing_utils.get_user_content('Start task')
  events1 = await runner.run_async(user_event)

  # Verify that we got a RequestInput event
  request_input_event = workflow_testing_utils.find_function_call_event(
      events1, REQUEST_INPUT_FUNCTION_CALL_NAME
  )
  assert request_input_event is not None
  args = request_input_event.content.parts[0].function_call.args
  assert args['message'] == 'What is your name and age?'

  interrupt_id = get_request_input_interrupt_ids(request_input_event)[0]
  invocation_id = request_input_event.invocation_id

  # Turn 2: Resume with the user input resolving the interrupt.
  user_input = create_request_input_response(
      interrupt_id, {'name': 'Alice', 'age': 25}
  )
  events2 = await runner.run_async(
      new_message=testing_utils.UserContent(user_input),
      invocation_id=invocation_id,
  )

  # Verify the tool workflow finished executing, returned the output,
  # and the parent agent LLM produced its final response.
  text_responses = [
      event.content.parts[0].text
      for event in events2
      if event.content and event.content.parts and event.content.parts[0].text
  ]
  assert 'Thank you! I received the user details.' in text_responses


def test_node_tool_rejects_agent():
  """NodeTool raises ValueError if initialized with any BaseAgent."""
  agent = LlmAgent(
      name='my_agent',
      instruction='Answer questions',
  )
  with pytest.raises(ValueError, match='cannot be wrapped as a NodeTool'):
    NodeTool(node=agent)


class GreetRequest(BaseModel):
  request: str


@pytest.mark.asyncio
async def test_function_node_wrapped_as_tool_returns_output(
    request: pytest.FixtureRequest,
):
  """NodeTool wraps a function node and returns expected output."""

  @node
  def greet_node(request: str) -> str:
    return f'Hello, {request}!'

  greet_node.input_schema = GreetRequest
  greet_tool = NodeTool(node=greet_node, name='greet_tool')

  parent_agent = LlmAgent(
      name='parent_agent',
      model=testing_utils.MockModel.create(
          responses=[
              types.Part.from_function_call(
                  name='greet_tool',
                  args={'request': 'world'},
              ),
              types.Part.from_text(text='Processed greet.'),
          ]
      ),
      tools=[greet_tool],
  )

  app = App(
      name=request.function.__name__,
      root_agent=parent_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('Greet world'))

  func_response_events = [
      e
      for e in events
      if e.content and e.content.parts and e.content.parts[0].function_response
  ]
  assert len(func_response_events) == 1
  assert func_response_events[0].content.parts[
      0
  ].function_response.response == {'result': 'Hello, world!'}


@pytest.mark.asyncio
async def test_workflow_tool_with_join_node(request: pytest.FixtureRequest):
  """WorkflowTool containing a JoinNode works correctly when wrapped as a tool."""
  node_a = workflow_testing_utils.TestingNode(name='NodeA', output={'a': 1})
  node_b = workflow_testing_utils.TestingNode(name='NodeB', output={'b': 2})
  node_join = JoinNode(name='NodeJoin')

  def format_response(node_input: dict[str, Any]):
    yield Event(
        output=(
            f"A is {node_input['NodeA']['a']} and B is"
            f" {node_input['NodeB']['b']}."
        )
    )

  sub_workflow = Workflow(
      name='sub_workflow',
      edges=[
          (START, node_a),
          (START, node_b),
          (node_a, node_join),
          (node_b, node_join),
          (node_join, format_response),
      ],
  )
  sub_workflow.input_schema = DummyRequest

  wf_tool = NodeTool(
      node=sub_workflow,
      name='my_join_tool',
      description='Collect parallel items.',
  )

  parent_agent = LlmAgent(
      name='parent_agent',
      model=testing_utils.MockModel.create(
          responses=[
              types.Part.from_function_call(
                  name='my_join_tool',
                  args={},
              ),
              types.Part.from_text(text='Done.'),
          ]
      ),
      tools=[wf_tool],
  )

  app = App(
      name=request.function.__name__,
      root_agent=parent_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('Run join'))

  func_response_events = [
      e
      for e in events
      if e.content and e.content.parts and e.content.parts[0].function_response
  ]
  assert len(func_response_events) == 1
  assert func_response_events[0].content.parts[
      0
  ].function_response.response == {'result': 'A is 1 and B is 2.'}


@pytest.mark.asyncio
async def test_workflow_tool_with_dynamic_node(request: pytest.FixtureRequest):
  """WorkflowTool containing a dynamic node schedules and executes it correctly."""

  @node
  async def child(*, ctx, node_input):
    yield f'child got: {node_input}'

  @node(rerun_on_resume=True)
  async def parent_node(*, ctx, node_input):
    result = await ctx.run_node(child, node_input='hello')
    yield f'parent got: {result}'

  sub_workflow = Workflow(
      name='sub_workflow',
      edges=[
          (START, parent_node),
      ],
  )
  sub_workflow.input_schema = DummyRequest

  wf_tool = NodeTool(
      node=sub_workflow,
      name='my_dynamic_tool',
      description='Call dynamic node.',
  )

  parent_agent = LlmAgent(
      name='parent_agent',
      model=testing_utils.MockModel.create(
          responses=[
              types.Part.from_function_call(
                  name='my_dynamic_tool',
                  args={},
              ),
              types.Part.from_text(text='Done.'),
          ]
      ),
      tools=[wf_tool],
  )

  app = App(
      name=request.function.__name__,
      root_agent=parent_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('Run dynamic'))

  func_response_events = [
      e
      for e in events
      if e.content and e.content.parts and e.content.parts[0].function_response
  ]
  assert len(func_response_events) == 1
  assert func_response_events[0].content.parts[
      0
  ].function_response.response == {'result': 'parent got: child got: hello'}


@pytest.mark.asyncio
async def test_workflow_tool_with_nested_workflows(
    request: pytest.FixtureRequest,
):
  """WorkflowTool wrapping a nested workflow executes successfully."""
  inner_node = workflow_testing_utils.TestingNode(
      name='inner_node', output='inner_output'
  )
  inner_wf = Workflow(
      name='inner_wf',
      edges=[
          (START, inner_node),
      ],
  )
  inner_wf.input_schema = None

  outer_node = workflow_testing_utils.TestingNode(
      name='outer_node', output='outer_output'
  )
  outer_wf = Workflow(
      name='outer_wf',
      edges=[
          (START, outer_node, inner_wf),
      ],
  )
  outer_wf.input_schema = DummyRequest

  wf_tool = NodeTool(
      node=outer_wf,
      name='nested_wf_tool',
      description='Call nested workflow.',
  )

  parent_agent = LlmAgent(
      name='parent_agent',
      model=testing_utils.MockModel.create(
          responses=[
              types.Part.from_function_call(
                  name='nested_wf_tool',
                  args={},
              ),
              types.Part.from_text(text='Done.'),
          ]
      ),
      tools=[wf_tool],
  )

  app = App(
      name=request.function.__name__,
      root_agent=parent_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('Run nested'))

  func_response_events = [
      e
      for e in events
      if e.content and e.content.parts and e.content.parts[0].function_response
  ]
  assert len(func_response_events) == 1
  assert func_response_events[0].content.parts[
      0
  ].function_response.response == {'result': 'inner_output'}


@pytest.mark.skip(reason='Requires CL 2 subagent branch refactor')
@pytest.mark.asyncio
async def test_workflow_tool_with_dynamic_node_hitl_resume(
    request: pytest.FixtureRequest,
):
  """WorkflowTool with a dynamic node containing HITL suspends and resumes successfully."""
  # 1. Define dynamic node calling a child RequestInputNode
  input_node = RequestInputNode(
      name='input_node',
      message='Enter value:',
      response_schema=UserInfo.model_json_schema(),
  )

  @node(rerun_on_resume=True)
  async def parent_node(*, ctx, node_input):
    result = await ctx.run_node(input_node)
    yield f'parent got: {result["name"]}'

  sub_workflow = Workflow(
      name='sub_workflow',
      edges=[
          (START, parent_node),
      ],
  )
  sub_workflow.input_schema = DummyRequest

  # 2. Wrap as WorkflowTool
  wf_tool = NodeTool(
      node=sub_workflow,
      name='my_dynamic_hitl_tool',
      description='Call dynamic HITL node.',
  )

  # 3. Define parent agent
  parent_agent = LlmAgent(
      name='parent_agent',
      model=testing_utils.MockModel.create(
          responses=[
              types.Part.from_function_call(
                  name='my_dynamic_hitl_tool',
                  args={},
              ),
              types.Part.from_text(text='Task completed.'),
          ]
      ),
      tools=[wf_tool],
  )

  # 4. App with resumability enabled
  app = App(
      name=request.function.__name__,
      root_agent=parent_agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Turn 1: Run the agent, triggering the tool call and dynamic node suspend.
  user_event = testing_utils.get_user_content('Start')
  events1 = await runner.run_async(user_event)

  request_input_event = workflow_testing_utils.find_function_call_event(
      events1, REQUEST_INPUT_FUNCTION_CALL_NAME
  )
  assert request_input_event is not None
  interrupt_id = get_request_input_interrupt_ids(request_input_event)[0]
  invocation_id = request_input_event.invocation_id

  # Turn 2: Resume with the user input response.
  user_input = create_request_input_response(
      interrupt_id, {'name': 'Bob', 'age': 30}
  )
  events2 = await runner.run_async(
      new_message=testing_utils.UserContent(user_input),
      invocation_id=invocation_id,
  )

  # Verify the tool workflow finished executing, returned output,
  # and parent agent replied.
  text_responses = [
      event.content.parts[0].text
      for event in events2
      if event.content and event.content.parts and event.content.parts[0].text
  ]
  assert 'Task completed.' in text_responses


@pytest.mark.skip(
    reason='Known framework issue with MockModel nested HITL in sub-workflow'
)
@pytest.mark.asyncio
async def test_workflow_as_tool_nested_hitl(request: pytest.FixtureRequest):
  """Parent LLM agent -> workflow -> LLM agent -> NodeTool(HITL) propagation."""
  # 1. Define the deepest node that raises RequestInput
  input_node = RequestInputNode(
      name='deep_input_node',
      message='Give me some input:',
      response_schema={
          'type': 'object',
          'properties': {'val': {'type': 'string'}},
      },
  )

  # 2. Wrap it as a NodeTool
  input_node.input_schema = DummyRequest
  node_tool = NodeTool(node=input_node, name='my_node_tool')

  # 3. Define the child agent that uses this NodeTool
  child_agent = LlmAgent(
      name='child_agent',
      model=testing_utils.MockModel.create(
          responses=[
              types.Part.from_function_call(
                  name='my_node_tool',
                  args={},
              ),
              types.Part.from_text(text='Child agent processed.'),
          ]
      ),
      tools=[node_tool],
  )

  # 4. Define the sub-workflow containing the child agent
  sub_workflow = Workflow(
      name='sub_workflow',
      edges=[
          (START, child_agent),
      ],
  )
  sub_workflow.input_schema = DummyRequest

  # 5. Wrap the sub-workflow as a WorkflowTool
  wf_tool = NodeTool(
      node=sub_workflow,
      name='my_wf_tool',
      description='Call sub workflow.',
  )

  # 6. Define the parent agent that calls the WorkflowTool
  parent_agent = LlmAgent(
      name='parent_agent',
      model=testing_utils.MockModel.create(
          responses=[
              types.Part.from_function_call(
                  name='my_wf_tool',
                  args={},
              ),
              types.Part.from_text(text='Parent agent finished successfully.'),
          ]
      ),
      tools=[wf_tool],
  )

  # 7. Wrap in App and Runner
  app = App(
      name=request.function.__name__,
      root_agent=parent_agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Turn 1: Run
  events1 = await runner.run_async(testing_utils.get_user_content('Start task'))

  # Assert Turn 1: Expect RequestInput event
  request_input_event = workflow_testing_utils.find_function_call_event(
      events1, REQUEST_INPUT_FUNCTION_CALL_NAME
  )
  assert request_input_event is not None
  args = request_input_event.content.parts[0].function_call.args
  assert args['message'] == 'Give me some input:'

  interrupt_id = get_request_input_interrupt_ids(request_input_event)[0]
  invocation_id = request_input_event.invocation_id

  # Turn 2: Resume
  user_input = create_request_input_response(interrupt_id, {'val': 'hello'})
  events2 = await runner.run_async(
      new_message=testing_utils.UserContent(user_input),
      invocation_id=invocation_id,
  )

  # Assert Turn 2: Expect completion
  text_responses = [
      event.content.parts[0].text
      for event in events2
      if event.content and event.content.parts and event.content.parts[0].text
  ]
  assert 'Parent agent finished successfully.' in text_responses


@pytest.mark.skip(
    reason='Known framework issue with MockModel multi-HITL in nested workflow'
)
@pytest.mark.asyncio
async def test_workflow_as_tool_nested_multi_hitl(
    request: pytest.FixtureRequest,
):
  """Parent LLM agent -> workflow -> LLM agent -> NodeTool(HITL) twice."""
  # 1. Define the deepest node that raises RequestInput
  input_node = RequestInputNode(
      name='deep_input_node',
      message='Give me some input:',
      response_schema={
          'type': 'object',
          'properties': {'val': {'type': 'string'}},
      },
  )

  # 2. Wrap it as a NodeTool
  input_node.input_schema = DummyRequest
  node_tool = NodeTool(node=input_node, name='my_node_tool')

  # 3. Define the child agent that uses this NodeTool twice
  child_agent = LlmAgent(
      name='child_agent',
      model=testing_utils.MockModel.create(
          responses=[
              types.Part.from_function_call(
                  name='my_node_tool',
                  args={},
              ),
              types.Part.from_function_call(
                  name='my_node_tool',
                  args={},
              ),
              types.Part.from_text(text='Child agent finished.'),
          ]
      ),
      tools=[node_tool],
  )

  # 4. Define the sub-workflow containing the child agent
  sub_workflow = Workflow(
      name='sub_workflow',
      edges=[
          (START, child_agent),
      ],
  )
  sub_workflow.input_schema = DummyRequest

  # 5. Wrap the sub-workflow as a WorkflowTool
  wf_tool = NodeTool(
      node=sub_workflow,
      name='my_wf_tool',
      description='Call sub workflow.',
  )

  # 6. Define the parent agent that calls the WorkflowTool
  parent_agent = LlmAgent(
      name='parent_agent',
      model=testing_utils.MockModel.create(
          responses=[
              types.Part.from_function_call(
                  name='my_wf_tool',
                  args={},
              ),
              types.Part.from_text(text='Parent agent finished successfully.'),
          ]
      ),
      tools=[wf_tool],
  )

  # 7. Wrap in App and Runner
  app = App(
      name=request.function.__name__,
      root_agent=parent_agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Turn 1: Run -> triggers first HITL
  events1 = await runner.run_async(testing_utils.get_user_content('Start task'))
  request_input_event1 = workflow_testing_utils.find_function_call_event(
      events1, REQUEST_INPUT_FUNCTION_CALL_NAME
  )
  assert request_input_event1 is not None
  interrupt_id1 = get_request_input_interrupt_ids(request_input_event1)[0]
  invocation_id = request_input_event1.invocation_id

  # Turn 2: Resume first HITL -> triggers second HITL
  user_input1 = create_request_input_response(interrupt_id1, {'val': 'hello'})
  events2 = await runner.run_async(
      new_message=testing_utils.UserContent(user_input1),
      invocation_id=invocation_id,
  )
  request_input_event2 = workflow_testing_utils.find_function_call_event(
      events2, REQUEST_INPUT_FUNCTION_CALL_NAME
  )
  assert request_input_event2 is not None
  interrupt_id2 = get_request_input_interrupt_ids(request_input_event2)[0]
  assert interrupt_id1 != interrupt_id2

  # Turn 3: Resume second HITL -> finishes
  user_input2 = create_request_input_response(interrupt_id2, {'val': 'world'})
  events3 = await runner.run_async(
      new_message=testing_utils.UserContent(user_input2),
      invocation_id=invocation_id,
  )

  # Assert Turn 3: Expect completion
  text_responses = [
      event.content.parts[0].text
      for event in events3
      if event.content and event.content.parts and event.content.parts[0].text
  ]
  assert 'Parent agent finished successfully.' in text_responses


@pytest.mark.skip(reason='Requires CL 2 subagent branch refactor')
@pytest.mark.asyncio
async def test_workflow_as_tool_nested_lro(request: pytest.FixtureRequest):
  """Parent LLM agent -> workflow -> LLM agent -> LRO tool."""

  # 1. Define LRO tool function
  def my_lro_func():
    return None

  # 2. Define child agent with LRO tool
  child_agent = LlmAgent(
      name='child_agent',
      model=testing_utils.MockModel.create(
          responses=[
              types.Part.from_function_call(
                  name='my_lro_func',
                  args={},
              ),
              types.Part.from_text(text='Child agent finished after LRO.'),
          ]
      ),
      tools=[LongRunningFunctionTool(func=my_lro_func)],
  )

  # 3. Define sub-workflow
  sub_workflow = Workflow(
      name='sub_workflow',
      edges=[
          (START, child_agent),
      ],
  )
  sub_workflow.input_schema = DummyRequest

  # 4. Wrap as WorkflowTool
  wf_tool = NodeTool(
      node=sub_workflow,
      name='my_wf_tool',
      description='Call sub workflow.',
  )

  # 5. Define parent agent
  parent_agent = LlmAgent(
      name='parent_agent',
      model=testing_utils.MockModel.create(
          responses=[
              types.Part.from_function_call(
                  name='my_wf_tool',
                  args={},
              ),
              types.Part.from_text(text='Parent agent finished successfully.'),
          ]
      ),
      tools=[wf_tool],
  )

  # 6. Wrap in App and Runner
  app = App(
      name=request.function.__name__,
      root_agent=parent_agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Turn 1: Run -> should pause on LRO
  events1 = await runner.run_async(testing_utils.get_user_content('Start task'))
  assert any(e.long_running_tool_ids for e in events1)

  invocation_id = events1[0].invocation_id
  fc_event = workflow_testing_utils.find_function_call_event(
      events1, 'my_lro_func'
  )
  assert fc_event is not None
  function_call_id = fc_event.content.parts[0].function_call.id

  # Turn 2: Resume with LRO response
  tool_response = testing_utils.UserContent(
      types.Part(
          function_response=types.FunctionResponse(
              id=function_call_id,
              name='my_lro_func',
              response={'result': 'LRO finished'},
          )
      )
  )
  events2 = await runner.run_async(
      new_message=tool_response,
      invocation_id=invocation_id,
  )

  # Assert Turn 2: Expect completion
  text_responses = [
      event.content.parts[0].text
      for event in events2
      if event.content and event.content.parts and event.content.parts[0].text
  ]
  assert 'Parent agent finished successfully.' in text_responses


def test_node_tool_auto_converts_function_node_binding():
  """NodeTool automatically converts FunctionNode parameter_binding to 'node_input'."""

  @node
  def my_func_node(request: str) -> str:
    """A dummy node."""
    return f'Result: {request}'

  # Originally it is 'state' mode by default
  assert my_func_node.parameter_binding == 'state'
  # input_schema is originally None
  assert getattr(my_func_node, 'input_schema', None) is None

  # Wrap it
  tool = NodeTool(node=my_func_node)

  # Check that the wrapped node copy is converted to 'node_input' mode
  assert tool.node.parameter_binding == 'node_input'
  # And input_schema is automatically inferred
  schema = tool.node.input_schema
  assert 'request' in schema['properties']


@pytest.mark.asyncio
async def test_node_tool_primitive_input_schema(request: pytest.FixtureRequest):
  """NodeTool automatically wraps primitive input_schema to object in declaration and unwraps in run."""

  def echo_func(node_input: str):
    yield Event(output=f'Echo: {node_input}')

  sub_workflow = Workflow(
      name='sub_workflow',
      edges=[
          (START, echo_func),
      ],
  )
  sub_workflow.input_schema = str
  tool = NodeTool(node=sub_workflow, name='primitive_tool')

  # 1. Check declaration is wrapped to object schema
  decl = tool._get_declaration()
  assert decl.parameters_json_schema is not None
  assert decl.parameters_json_schema['type'] == 'object'
  assert 'request' in decl.parameters_json_schema['properties']
  assert (
      decl.parameters_json_schema['properties']['request']['type'] == 'string'
  )

  # 2. Run the tool (passing wrapped argument) and check execution
  parent_agent = LlmAgent(
      name='parent_agent',
      model=testing_utils.MockModel.create(
          responses=[
              types.Part.from_function_call(
                  name='primitive_tool',
                  args={'request': 'hello_world'},
              ),
              types.Part.from_text(text='Finished.'),
          ]
      ),
      tools=[tool],
  )
  app = App(
      name=request.function.__name__,
      root_agent=parent_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('Run'))

  func_response_events = [
      e
      for e in events
      if e.content and e.content.parts and e.content.parts[0].function_response
  ]
  assert len(func_response_events) == 1
  assert func_response_events[0].content.parts[
      0
  ].function_response.response == {'result': 'Echo: hello_world'}
