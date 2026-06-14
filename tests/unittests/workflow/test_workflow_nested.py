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

import copy
from typing import Any
from typing import AsyncGenerator
import uuid

from google.adk.agents.context import Context
from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.workflow import BaseNode
from google.adk.workflow import JoinNode
from google.adk.workflow._base_node import START
from google.adk.workflow._node_status import NodeStatus
from google.adk.workflow._workflow import Workflow
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
from google.adk.workflow.utils._workflow_hitl_utils import get_request_input_interrupt_ids
from google.adk.workflow.utils._workflow_hitl_utils import has_request_input_function_call
from google.genai import types
from pydantic import ConfigDict
from pydantic import Field
import pytest
from typing_extensions import override

from .. import testing_utils
from .workflow_testing_utils import find_function_call_event
from .workflow_testing_utils import InputCapturingNode
from .workflow_testing_utils import RequestInputNode
from .workflow_testing_utils import simplify_events_with_node
from .workflow_testing_utils import TestingNode


def long_running_tool_func():
  """A test tool that simulates a long-running operation."""
  return None


@pytest.mark.asyncio
async def test_nested_workflow_as_node(request: pytest.FixtureRequest):
  """Tests that a Workflow can be used as a node in another Workflow."""

  async def nested_func(node_input: types.Content):
    return 'I am nested'

  nested_agent = Workflow(
      name='nested_agent',
      edges=[('START', nested_func)],
  )

  async def output_func(node_input: str):
    return 'I am outer'

  outer_agent = Workflow(
      name='outer_agent',
      edges=[('START', nested_agent), (nested_agent, output_func)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('hello'))

  simplified_events = simplify_events_with_node(events)
  assert simplified_events == [
      (
          'outer_agent@1/nested_agent@1/nested_func@1',
          {
              'output': 'I am nested',
          },
      ),
      (
          'outer_agent@1/output_func@1',
          {
              'output': 'I am outer',
          },
      ),
  ]


@pytest.mark.asyncio
async def test_nested_workflow_with_join_node(
    request: pytest.FixtureRequest,
):
  """Tests that a nested Workflow with JoinNode works correctly."""

  async def nested_node_a():
    return {'a': 1}

  async def nested_node_b():
    return {'b': 2}

  async def nested_node_c():
    return {'c': 3}

  nested_join_node = JoinNode(name='nested_join')

  nested_agent = Workflow(
      name='nested_agent',
      edges=[
          ('START', nested_node_a),
          ('START', nested_node_c),
          (nested_node_a, nested_join_node),
          (nested_node_c, nested_node_b),
          (nested_node_b, nested_join_node),
      ],
  )

  async def output_func(node_input: dict):
    return (
        'Joined output:'
        f' a={node_input["nested_node_a"]["a"]},'
        f' b={node_input["nested_node_b"]["b"]}'
    )

  outer_agent = Workflow(
      name='outer_agent',
      edges=[('START', nested_agent), (nested_agent, output_func)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('hello'))

  simplified_events = simplify_events_with_node(events)
  assert sorted(simplified_events[0:2], key=lambda x: x[0]) == [
      (
          'outer_agent@1/nested_agent@1/nested_node_a@1',
          {'output': {'a': 1}},
      ),
      (
          'outer_agent@1/nested_agent@1/nested_node_c@1',
          {'output': {'c': 3}},
      ),
  ]
  assert simplified_events[2:] == [
      (
          'outer_agent@1/nested_agent@1/nested_node_b@1',
          {'output': {'b': 2}},
      ),
      (
          'outer_agent@1/nested_agent@1/nested_join@1',
          {
              'output': {'nested_node_a': {'a': 1}, 'nested_node_b': {'b': 2}},
          },
      ),
      (
          'outer_agent@1/output_func@1',
          {
              'output': 'Joined output: a=1, b=2',
          },
      ),
  ]


@pytest.mark.asyncio
async def test_nested_workflow_updates_state_outer_reads(
    request: pytest.FixtureRequest,
):
  """Tests that outer workflow can read state updated by nested workflow."""

  async def nested_state_updater(ctx: Context):
    yield Event(
        state={'my_key': 'my_value'},
    )
    yield 'nested agent finished'

  nested_agent = Workflow(
      name='nested_agent',
      edges=[('START', nested_state_updater)],
  )

  def outer_state_reader(my_key: str, node_input: str):
    return f'Nested agent output: {node_input}, state value: {my_key}'

  outer_agent = Workflow(
      name='outer_agent',
      edges=[('START', nested_agent), (nested_agent, outer_state_reader)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('hello'))

  simplified_events = simplify_events_with_node(events)
  assert simplified_events == [
      (
          'outer_agent@1/nested_agent@1/nested_state_updater@1',
          {
              'output': 'nested agent finished',
          },
      ),
      (
          'outer_agent@1/outer_state_reader@1',
          {
              'output': (
                  'Nested agent output: nested agent finished, state value:'
                  ' my_value'
              ),
          },
      ),
  ]


@pytest.mark.asyncio
async def test_nested_workflow_intermediate_nodes(
    request: pytest.FixtureRequest,
):
  """Tests that only the final output of a nested workflow is passed to the outer workflow."""

  node_a = TestingNode(name='NodeA', output='Inner Intermediate')
  node_b = TestingNode(name='NodeB', output='Inner Final')

  nested_agent = Workflow(
      name='nested_agent',
      edges=[
          ('START', node_a),
          (node_a, node_b),
      ],
  )

  output_node = InputCapturingNode(name='OutputNode')

  outer_agent = Workflow(
      name='outer_agent',
      edges=[
          ('START', nested_agent),
          (nested_agent, output_node),
      ],
  )

  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  assert output_node.received_inputs == ['Inner Final']

  simplified_events = simplify_events_with_node(events)
  assert simplified_events == [
      (
          'outer_agent@1/nested_agent@1/NodeA@1',
          {'output': 'Inner Intermediate'},
      ),
      (
          'outer_agent@1/nested_agent@1/NodeB@1',
          {'output': 'Inner Final'},
      ),
      (
          'outer_agent@1/OutputNode@1',
          {'output': {'received': 'Inner Final'}},
      ),
  ]


@pytest.mark.asyncio
async def test_nested_workflow_with_hitl(request: pytest.FixtureRequest):
  """Tests that a nested Workflow with HITL works correctly."""
  # Given: A nested workflow with an LLM agent that calls a long running tool
  llm_agent = LlmAgent(
      name='llm_agent',
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

  nested_agent = Workflow(
      name='nested_agent',
      edges=[('START', llm_agent)],
  )

  async def output_func(node_input: Any):
    return 'I am outer'

  outer_agent = Workflow(
      name='outer_agent',
      edges=[('START', nested_agent), (nested_agent, output_func)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # When: Starting the workflow
  user_event = testing_utils.get_user_content('start workflow')
  events1 = await runner.run_async(user_event)

  # Then: It should yield a FunctionCall event and wait for tool execution
  ev = find_function_call_event(events1, 'long_running_tool_func')
  assert ev is not None
  function_call_id = ev.content.parts[0].function_call.id

  simplified_events1 = simplify_events_with_node(events1)
  assert simplified_events1 == [
      (
          'outer_agent@1/nested_agent@1/llm_agent@1',
          types.Part.from_function_call(name='long_running_tool_func', args={}),
      ),
  ]

  tool_response = testing_utils.UserContent(
      types.Part(
          function_response=types.FunctionResponse(
              id=function_call_id,
              name='long_running_tool_func',
              response={'result': 'Final tool output'},
          )
      )
  )

  invocation_id = events1[0].invocation_id
  events2 = await runner.run_async(
      new_message=tool_response,
      invocation_id=invocation_id,
  )

  simplified_events2 = simplify_events_with_node(events2)
  assert simplified_events2 == [
      ('outer_agent@1/nested_agent@1/llm_agent@1', 'LLM response after tool'),
      (
          'outer_agent@1/output_func@1',
          {'output': 'I am outer'},
      ),
  ]


@pytest.mark.asyncio
async def test_nested_workflow_with_request_input_event_hitl(
    request: pytest.FixtureRequest,
):
  """Nested workflow correctly propagates RequestInput and resumes.

  Setup: outer_agent -> nested_agent -> node_hitl.
  Act:
    - Run 1: start workflow, node_hitl yields RequestInput.
    - Run 2: resume with user response.
  Assert:
    - Run 1: returns RequestInput event.
    - Run 2: node_hitl receives input and completes.
  """

  class NodeHitl(BaseNode):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    rerun_on_resume: bool = Field(default=True)
    name: str = Field(default='node_hitl')

    @override
    async def _run_impl(
        self,
        *,
        ctx: Context,
        node_input: Any,
    ) -> AsyncGenerator[Any, None]:
      if resume_input := ctx.resume_inputs.get('request_input'):
        yield f'Resumed with user input: {resume_input}'
      else:
        yield RequestInput(
            message='requesting input via RequestInputEvent',
            interrupt_id='request_input',
        )

  # Given: A nested workflow where the inner node requests input
  node_hitl_instance = NodeHitl()

  nested_agent = Workflow(
      name='nested_agent',
      edges=[('START', node_hitl_instance)],
  )

  async def output_func():
    return 'I am outer'

  outer_agent = Workflow(
      name='outer_agent',
      edges=[('START', nested_agent), (nested_agent, output_func)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # When: Starting the workflow
  user_event = testing_utils.get_user_content('start workflow')
  events1 = await runner.run_async(user_event)

  # Then: It should yield a RequestInput event
  req_events = [e for e in events1 if has_request_input_function_call(e)]
  assert req_events
  interrupt_id = get_request_input_interrupt_ids(req_events[0])[0]
  assert interrupt_id == 'request_input'

  simplified_events1 = simplify_events_with_node(events1)
  assert simplified_events1 == [
      (
          'outer_agent@1/nested_agent@1/node_hitl@1',
          testing_utils.simplify_content(copy.deepcopy(req_events[0].content)),
      ),
  ]

  # When: Resuming with user response
  hitl_response = types.Content(
      parts=[
          create_request_input_response(
              interrupt_id, {'response': 'user input for hitl'}
          )
      ],
      role='user',
  )

  invocation_id = events1[0].invocation_id
  events2 = await runner.run_async(
      new_message=hitl_response,
      invocation_id=invocation_id,
  )

  # Then: It should complete and pass output to the outer workflow
  simplified_events2 = simplify_events_with_node(events2)
  assert simplified_events2 == [
      (
          'outer_agent@1/nested_agent@1/node_hitl@1',
          {
              'output': (
                  "Resumed with user input: {'response': 'user input for hitl'}"
              ),
          },
      ),
      (
          'outer_agent@1/output_func@1',
          {'output': 'I am outer'},
      ),
  ]


@pytest.mark.asyncio
async def test_nested_agent_with_request_input_piped_to_next_node(
    request: pytest.FixtureRequest,
):
  """Tests that user response to RequestInput in nested agent is piped to next node."""
  ask_user = RequestInputNode(
      name='ask_user',
      message='Please provide input',
  )
  capture_node = InputCapturingNode(name='capture_node')

  sub_agent = Workflow(
      name='sub_agent',
      edges=[
          ('START', ask_user),
          (ask_user, capture_node),
      ],
  )

  root_agent = Workflow(
      name='root_agent',
      edges=[('START', sub_agent)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=root_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)

  user_event = testing_utils.get_user_content('start workflow')
  events1 = await runner.run_async(user_event)

  req_events = [e for e in events1 if has_request_input_function_call(e)]
  assert req_events
  interrupt_id = get_request_input_interrupt_ids(req_events[0])[0]
  assert interrupt_id
  hitl_response_payload = {'response': 'user input for hitl'}
  hitl_response = types.Content(
      parts=[
          create_request_input_response(interrupt_id, hitl_response_payload)
      ],
      role='user',
  )

  invocation_id = events1[0].invocation_id
  await runner.run_async(
      new_message=hitl_response,
      invocation_id=invocation_id,
  )

  assert capture_node.received_inputs == [hitl_response_payload]


@pytest.mark.asyncio
async def test_nested_workflow_chain_input_propagation(
    request: pytest.FixtureRequest,
):

  async def create_output_a():
    return 'output of A'

  nested_agent_a = Workflow(
      name='nested_agent_a',
      edges=[('START', create_output_a)],
  )

  capture_node_b = InputCapturingNode(name='capture_node_b')
  nested_agent_b = Workflow(
      name='nested_agent_b',
      edges=[('START', capture_node_b)],
  )

  outer_agent = Workflow(
      name='outer_agent',
      edges=[
          ('START', nested_agent_a),
          (nested_agent_a, nested_agent_b),
      ],
  )

  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  await runner.run_async(testing_utils.get_user_content('hello'))

  assert capture_node_b.received_inputs == ['output of A']


@pytest.mark.asyncio
async def test_nested_workflow_with_tool_calls(
    request: pytest.FixtureRequest,
):
  """Tests that a nested Workflow works correctly with two tool calls."""
  tool_call_count = 0

  def simple_tool() -> str:
    nonlocal tool_call_count
    tool_call_count += 1
    return f'Tool output {tool_call_count}'

  # Given: A nested workflow where the inner node calls a tool
  llm_agent = LlmAgent(
      name='llm_agent',
      model=testing_utils.MockModel.create(
          responses=[
              types.Part.from_function_call(
                  name='simple_tool',
                  args={},
              ),
              types.Part.from_text(text='LLM response after tools'),
          ]
      ),
      tools=[simple_tool],
  )

  nested_agent = Workflow(
      name='nested_agent',
      edges=[('START', llm_agent)],
  )

  async def output_func():
    return 'I am outer'

  outer_agent = Workflow(
      name='outer_agent',
      edges=[
          ('START', nested_agent),
          (nested_agent, output_func),
      ],
  )

  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # When: Starting the workflow
  user_event = testing_utils.get_user_content('start workflow')
  events = await runner.run_async(user_event)

  # Then: It should call the tool and yield events
  assert tool_call_count == 1

  simplified_events = simplify_events_with_node(events)

  # Extract the dynamically generated function_call_id from events.
  ev = find_function_call_event(events, 'simple_tool')
  assert ev is not None
  function_call_id = ev.content.parts[0].function_call.id

  assert simplified_events == [
      (
          'outer_agent@1/nested_agent@1/llm_agent@1',
          types.Part.from_function_call(name='simple_tool', args={}),
      ),
      (
          'outer_agent@1/nested_agent@1/llm_agent@1',
          types.Part(
              function_response=types.FunctionResponse(
                  name='simple_tool',
                  response={'result': 'Tool output 1'},
              )
          ),
      ),
      ('outer_agent@1/nested_agent@1/llm_agent@1', 'LLM response after tools'),
      (
          'outer_agent@1/output_func@1',
          {
              'output': 'I am outer',
          },
      ),
  ]


@pytest.mark.asyncio
async def test_three_level_nested_workflow(request: pytest.FixtureRequest):
  """Tests a 3-level nested Workflow structure."""

  async def inner_func():
    return 'I am inner'

  inner_agent = Workflow(
      name='inner_agent',
      edges=[('START', inner_func)],
  )

  async def middle_func():
    return 'I am middle'

  middle_agent = Workflow(
      name='middle_agent',
      edges=[('START', inner_agent), (inner_agent, middle_func)],
  )

  async def outer_func():
    return 'I am outer'

  outer_agent = Workflow(
      name='outer_agent',
      edges=[('START', middle_agent), (middle_agent, outer_func)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('hello'))

  simplified_events = simplify_events_with_node(events)
  assert simplified_events == [
      (
          'outer_agent@1/middle_agent@1/inner_agent@1/inner_func@1',
          {
              'output': 'I am inner',
          },
      ),
      (
          'outer_agent@1/middle_agent@1/middle_func@1',
          {
              'output': 'I am middle',
          },
      ),
      (
          'outer_agent@1/outer_func@1',
          {
              'output': 'I am outer',
          },
      ),
  ]


@pytest.mark.asyncio
async def test_duplicate_grandchild_workflow_names(
    request: pytest.FixtureRequest,
):
  """Tests that grandchild workflow agents with same name can coexist."""

  # Given: A workflow hierarchy where grandchild workflows have the same name
  async def grandchild_func():
    return 'I am grandchild'

  grandchild_agent = Workflow(
      name='grandchild',
      edges=[('START', grandchild_func)],
  )

  child1_agent = Workflow(
      name='child1',
      edges=[('START', copy.deepcopy(grandchild_agent))],
  )

  child2_agent = Workflow(
      name='child2',
      edges=[('START', copy.deepcopy(grandchild_agent))],
  )

  root_agent = Workflow(
      name='root',
      edges=[('START', child1_agent), (child1_agent, child2_agent)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=root_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  # When: Starting the workflow
  user_content = testing_utils.get_user_content('hello')
  events = await runner.run_async(user_content)

  # Then: It should execute both grandchildren successfully
  simplified_events = simplify_events_with_node(events)
  assert simplified_events == [
      (
          'root@1/child1@1/grandchild@1/grandchild_func@1',
          {'output': 'I am grandchild'},
      ),
      (
          'root@1/child2@1/grandchild@1/grandchild_func@1',
          {'output': 'I am grandchild'},
      ),
  ]


@pytest.mark.asyncio
async def test_duplicate_name_in_ancestral_path(
    request: pytest.FixtureRequest,
):
  """Tests that agent with same name can exist in ancestral path (A->B->A)."""

  async def func_a():
    return 'I am A'

  agent_a_child = Workflow(
      name='A',
      edges=[('START', func_a)],
  )

  agent_b = Workflow(
      name='B',
      edges=[('START', agent_a_child)],
  )

  agent_a_root = Workflow(
      name='A',
      edges=[('START', agent_b)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=agent_a_root,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  user_content = testing_utils.get_user_content('hello')
  events = await runner.run_async(user_content)

  simplified_events = simplify_events_with_node(events)
  assert simplified_events == [
      (
          'A@1/B@1/A@1/func_a@1',
          {'output': 'I am A'},
      ),
  ]


# --- Helpers moved from test_workflow.py ---


class _OutputNode(BaseNode):
  """Yields a fixed output value."""

  value: Any = None

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    yield self.value


class _InputCapturingNode(BaseNode):
  """Captures node_input for later assertion."""

  model_config = ConfigDict(arbitrary_types_allowed=True)
  received_inputs: list[Any] = Field(default_factory=list)

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    self.received_inputs.append(node_input)
    yield {'received': node_input}


async def _run_workflow(wf, message='start'):
  """Run a Workflow through Runner, return collected events."""
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')
  msg = types.Content(parts=[types.Part(text=message)], role='user')
  events = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg
  ):
    events.append(event)
  return events, ss, session


def _outputs(events):
  """Extract non-None outputs from events."""
  return [e.output for e in events if e.output is not None]


def _output_by_node(events):
  """Extract (node_name_from_path, output) for child node events."""
  results = []
  for e in events:
    if e.output is not None and e.node_info.path and '/' in e.node_info.path:
      node_name = e.node_info.path.rsplit('/', 1)[-1]
      if '@' in node_name:
        node_name = node_name.rsplit('@', 1)[0]
      results.append((node_name, e.output))
  return results


# --- Tests moved from test_workflow.py ---


@pytest.mark.asyncio
async def test_nested_workflow_completes():
  """Inner workflow runs to completion, outer continues downstream."""
  inner_node = _OutputNode(name='inner_node', value='inner_result')
  inner_wf = Workflow(name='inner_wf', edges=[(START, inner_node)])
  before = _OutputNode(name='before', value='before_result')
  after = _InputCapturingNode(name='after')
  wf = Workflow(name='wf', edges=[(START, before, inner_wf, after)])

  events, _, _ = await _run_workflow(wf)

  by_node = _output_by_node(events)
  assert ('before', 'before_result') in by_node
  assert ('inner_node', 'inner_result') in by_node
  assert after.received_inputs == ['inner_result']


@pytest.mark.asyncio
async def test_nested_workflow_event_author():
  """Events are authored by the nearest orchestrator (workflow/agent).

  Setup: outer_wf → inner_wf → inner_node.
  Assert:
    - inner_node's events are authored by inner_wf (nearest).
    - outer_wf's direct children are authored by outer_wf.
    - inner_wf overrides the author for its subtree.
  """
  inner_node = _OutputNode(name='inner_node', value='inner_result')
  inner_wf = Workflow(name='inner_wf', edges=[(START, inner_node)])
  outer_node = _OutputNode(name='outer_node', value='outer_result')
  wf = Workflow(
      name='outer_wf',
      edges=[(START, outer_node, inner_wf)],
  )

  events, _, _ = await _run_workflow(wf)

  # outer_node's events authored by outer_wf (nearest orchestrator).
  outer_events = [
      e for e in events if e.node_info.path == 'outer_wf@1/outer_node@1'
  ]
  assert outer_events
  assert all(e.author == 'outer_wf' for e in outer_events)

  # inner_node's events authored by inner_wf (nearest orchestrator),
  # NOT outer_wf.
  inner_events = [
      e
      for e in events
      if e.node_info.path == 'outer_wf@1/inner_wf@1/inner_node@1'
  ]
  assert inner_events
  assert all(e.author == 'inner_wf' for e in inner_events)


@pytest.mark.asyncio
async def test_nested_workflow_interrupt_and_resume():
  """Inner workflow child interrupts, outer resumes on FR."""

  class _InterruptNode(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      fc_id = ctx.state.get('_nested_fc')
      if fc_id and ctx.resume_inputs and fc_id in ctx.resume_inputs:
        ctx.state['_nested_fc'] = None
        response = ctx.resume_inputs[fc_id]['answer']
        yield f'approved:{response}'
        return
      fc_id = f'fc-{uuid.uuid4().hex[:8]}'
      ctx.state['_nested_fc'] = fc_id
      yield Event(
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='inner_tool', args={}, id=fc_id
                      )
                  )
              ]
          ),
          long_running_tool_ids={fc_id},
      )

  inner_wf = Workflow(
      name='inner_wf',
      edges=[(START, _InterruptNode(name='approval'))],
  )
  before = _OutputNode(name='before', value='before_result')
  after = _InputCapturingNode(name='after')
  wf = Workflow(
      name='wf',
      edges=[(START, before, inner_wf, after)],
  )
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: before completes, inner_wf/approval interrupts
  msg1 = types.Content(parts=[types.Part(text='go')], role='user')
  events1: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg1
  ):
    events1.append(event)

  # Should have interrupt from inner's child
  interrupt_events = [e for e in events1 if e.long_running_tool_ids]
  assert len(interrupt_events) == 1
  assert interrupt_events[0].long_running_tool_ids is not None
  fc_id = list(interrupt_events[0].long_running_tool_ids)[0]

  # Workflow-level interrupt events should NOT be persisted
  # (they're _adk_internal). Only the leaf child's event at
  # 'wf/inner_wf/approval' should have interrupt ids in session.
  updated_session = await ss.get_session(
      app_name='test', user_id='u', session_id=session.id
  )
  assert updated_session is not None
  wf_interrupt_events = [
      e
      for e in updated_session.events
      if e.long_running_tool_ids and e.node_info.path in ('wf', 'wf/inner_wf')
  ]
  assert wf_interrupt_events == []

  # Run 2: resume
  msg2 = types.Content(
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name='inner_tool',
                  id=fc_id,
                  response={'answer': 'yes'},
              )
          )
      ],
      role='user',
  )
  events2: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg2
  ):
    events2.append(event)

  # Inner resumed, after should receive inner's output
  outputs = [e.output for e in events2 if e.output is not None]
  assert 'approved:yes' in outputs
  assert after.received_inputs == ['approved:yes']


@pytest.mark.asyncio
async def test_nested_workflow_partial_resume():
  """Partial FR re-runs nested Workflow, resolved child completes while unresolved stays interrupted.

  Setup: outer_wf → inner_wf → (child_a, child_b) → join.
    Both children interrupt on first run.
  Act:
    - Run 2: resolve only child_a's FR.
    - Run 3: resolve child_b's FR.
  Assert:
    - Run 2: child_a produces output, invocation still interrupted.
    - Run 3: child_b produces output, join completes, no interrupts.
  """

  class _InterruptOnce(BaseNode):
    """Interrupts on first run, yields resume response on second."""

    rerun_on_resume: bool = True

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      fc_id = f'fc-{self.name}'
      if ctx.resume_inputs and fc_id in ctx.resume_inputs:
        yield f'{self.name}:{ctx.resume_inputs[fc_id]}'
        return
      yield RequestInput(interrupt_id=fc_id)

  child_a = _InterruptOnce(name='child_a')
  child_b = _InterruptOnce(name='child_b')
  join = JoinNode(name='join', wait_for_output=True)

  inner_wf = Workflow(
      name='inner_wf',
      edges=[
          (START, child_a),
          (START, child_b),
          (child_a, join),
          (child_b, join),
      ],
  )

  outer_wf = Workflow(
      name='outer',
      edges=[(START, inner_wf)],
  )

  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=outer_wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: both children interrupt
  msg1 = types.Content(parts=[types.Part(text='go')], role='user')
  events1: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg1
  ):
    events1.append(event)

  interrupt_ids = set()
  for e in events1:
    if e.long_running_tool_ids:
      interrupt_ids.update(e.long_running_tool_ids)
  assert 'fc-child_a' in interrupt_ids
  assert 'fc-child_b' in interrupt_ids

  # Run 2: resolve only child_a
  msg2 = types.Content(
      parts=[create_request_input_response('fc-child_a', {'v': 'a'})],
      role='user',
  )
  events2: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg2
  ):
    events2.append(event)

  # child_a should have produced output
  child_a_outputs = [
      e.output
      for e in events2
      if e.node_info.path and 'child_a' in e.node_info.path and e.output
  ]
  assert any('child_a:' in str(o) for o in child_a_outputs)

  # Run 3: resolve child_b → join completes, workflow finishes
  msg3 = types.Content(
      parts=[create_request_input_response('fc-child_b', {'v': 'b'})],
      role='user',
  )
  events3: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg3
  ):
    events3.append(event)

  # child_b should have produced output
  child_b_outputs = [
      e.output
      for e in events3
      if e.node_info.path and 'child_b' in e.node_info.path and e.output
  ]
  assert any('child_b:' in str(o) for o in child_b_outputs)

  # join should have completed (no more interrupts)
  final_interrupts = set()
  for e in events3:
    if e.long_running_tool_ids:
      final_interrupts.update(e.long_running_tool_ids)
  assert not final_interrupts


@pytest.mark.asyncio
async def test_scan_child_events_ignores_descendant_run_id_resets():
  """_scan_child_events only resets run_id from direct child events."""
  from unittest.mock import MagicMock

  from google.adk.events.event import Event
  from google.adk.events.event import NodeInfo

  # We create a Workflow instance to test its private method _scan_child_events.
  wf = Workflow(name='wf', edges=[])

  # Given a direct child event and a descendant event.
  event1 = Event(
      author='node',
      node_info=NodeInfo(path='wf@1/child@1', run_id='1'),
      invocation_id='test_inv',
  )
  event2 = Event(
      author='node',
      node_info=NodeInfo(path='wf@1/child@1/grandchild@2', run_id='2'),
      invocation_id='test_inv',
  )

  ctx = MagicMock()
  ctx._invocation_context = MagicMock()
  ctx._invocation_context.invocation_id = 'test_inv'
  ctx._invocation_context.session = MagicMock()
  ctx._invocation_context.session.events = [event1, event2]
  # _scan_child_events reads ctx.node_path to determine the base workflow path.
  ctx.node_path = 'wf@1'

  children = wf._scan_child_events(ctx)

  # Assert child 'child' run_id remains '1' (not '2' from the descendant).
  assert children[0]['child@1'].run_id == '1'
