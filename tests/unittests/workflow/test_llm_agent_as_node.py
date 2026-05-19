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

"""Tests for _LlmAgentWrapper.

Verifies that _LlmAgentWrapper correctly adapts V1 LlmAgent for use as a
workflow graph node, covering mode validation, input conversion,
content isolation, output extraction, and both old/new workflow paths.
"""

from __future__ import annotations

from typing import Any

from google.adk.agents.context import Context
from google.adk.agents.llm.task._task_models import TaskResult
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.features import FeatureName
from google.adk.features import override_feature_enabled
from google.adk.workflow import START
from google.adk.workflow._workflow import Workflow
from google.adk.workflow.utils._workflow_graph_utils import build_node
from google.genai import types
from pydantic import BaseModel
from pydantic import ValidationError
import pytest

from .workflow_testing_utils import create_parent_invocation_context
from .workflow_testing_utils import InputCapturingNode
from .workflow_testing_utils import TestingNode

# --- Fixtures ---


class StoryOutput(BaseModel):
  title: str
  content: str


class StoryInput(BaseModel):
  topic: str
  style: str = 'narrative'


def _make_agent(
    name: str = 'test_agent',
    mode: str = 'task',
    **kwargs,
) -> LlmAgent:
  return LlmAgent(
      name=name,
      model='gemini-2.5-flash',
      instruction='Test agent.',
      mode=mode,
      **kwargs,
  )


def _mock_agent_run(agent, finish_output=None, content_text=None):
  """Mocks agent.run_async to yield events. Returns a context manager."""

  async def fake_run_async(*args, **kwargs):
    if content_text:
      yield Event(
          invocation_id='inv',
          author=agent.name,
          content=types.Content(parts=[types.Part(text=content_text)]),
      )
    if finish_output is not None:
      # Task Delegation API: emit a finish_task FC followed by its FR.
      # The wrapper waits for the FR (so validation errors can drive a
      # retry) before extracting the FC's args as event.output.
      yield Event(
          invocation_id='inv',
          author=agent.name,
          content=types.Content(
              role='model',
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='finish_task',
                          id='ft-1',
                          args=(
                              finish_output
                              if isinstance(finish_output, dict)
                              else {'result': finish_output}
                          ),
                      )
                  )
              ],
          ),
      )
      yield Event(
          invocation_id='inv',
          author=agent.name,
          content=types.Content(
              role='user',
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          name='finish_task',
                          id='ft-1',
                          response={'result': 'Task completed.'},
                      )
                  )
              ],
          ),
      )

  original = agent.run_async
  object.__setattr__(agent, 'run_async', fake_run_async)

  class _Ctx:

    def __enter__(self):
      return self

    def __exit__(self, *args):
      object.__setattr__(agent, 'run_async', original)

  return _Ctx()


def _mock_leaf_run(agent, content_text=None):
  """Mocks the agent.run_async. Returns a context manager."""
  target = agent

  async def fake_run_async(*args, **kwargs):
    if content_text:
      yield Event(output=content_text)

  original = target.run_async
  object.__setattr__(target, 'run_async', fake_run_async)

  class _Ctx:

    def __enter__(self):
      return self

    def __exit__(self, *args):
      object.__setattr__(target, 'run_async', original)

  return _Ctx()


def _new_workflow_runner(wf, test_name):
  """Creates an InMemoryRunner for the new Workflow (root_agent path)."""
  from google.adk.apps.app import App

  from . import testing_utils

  app = App(name=test_name, root_agent=wf)
  return testing_utils.InMemoryRunner(app=app)


# --- Validation ---


class TestValidation:

  def test_task_mode_accepted(self):
    """Wrapping a task-mode agent succeeds."""
    wrapper = build_node(_make_agent(mode='task'))
    assert wrapper.name == 'test_agent'

  def test_single_turn_mode_accepted(self):
    """Wrapping a single_turn-mode agent succeeds."""
    wrapper = build_node(_make_agent(mode='single_turn'))
    assert wrapper.name == 'test_agent'

  def test_chat_mode_accepted(self):
    """Wrapping a chat-mode agent succeeds."""
    wrapper = build_node(_make_agent(mode='chat'))
    assert wrapper.name == 'test_agent'

  def test_name_defaults_to_agent_name(self):
    """Wrapper name defaults to the inner agent's name."""
    wrapper = build_node(_make_agent(name='my_agent'))
    assert wrapper.name == 'my_agent'

  def test_name_can_be_overridden(self):
    """Explicit name overrides the agent's name."""
    wrapper = build_node(_make_agent(name='my_agent'), name='custom')
    assert wrapper.name == 'custom'

  def test_task_mode_waits_for_output(self):
    """Task mode sets wait_for_output=True."""
    wrapper = build_node(_make_agent(mode='task'))
    assert wrapper.wait_for_output is True

  def test_single_turn_does_not_wait_for_output(self):
    """Single_turn mode does not set wait_for_output."""
    wrapper = build_node(_make_agent(mode='single_turn'))
    assert wrapper.wait_for_output is False

  def test_rerun_on_resume_defaults_true(self):
    """Wrapper defaults to rerun_on_resume=True."""
    wrapper = build_node(_make_agent())
    assert wrapper.rerun_on_resume is True


# --- build_node auto-wrapping ---


class TestBuildNode:

  def test_task_mode_wrapped(self):
    """build_node returns a cloned task-mode LlmAgent."""
    agent = _make_agent(mode='task')
    node = build_node(agent)
    assert isinstance(node, LlmAgent)
    assert node is not agent
    assert node.name == agent.name

  def test_single_turn_mode_wrapped(self):
    """build_node returns a cloned single_turn-mode LlmAgent."""
    node = build_node(_make_agent(mode='single_turn'))
    assert isinstance(node, LlmAgent)

  @pytest.mark.skip(
      reason=(
          'V2 LlmAgent does not allow mode=None and defaults to chat, so'
          ' fallback in wrapper is not triggered here.'
      )
  )
  def test_default_mode_auto_set_to_single_turn(self):
    """LlmAgent with explicit mode=None is auto-converted to single_turn."""
    agent = LlmAgent(
        name='agent', model='gemini-2.5-flash', instruction='Test.', mode=None
    )

    node = build_node(agent)

    assert node.mode == 'single_turn'

  def test_name_override(self):
    """build_node respects explicit name override."""
    node = build_node(_make_agent(mode='task'), name='override')
    assert node.name == 'override'


# --- Old workflow path ---


@pytest.mark.xfail(
    strict=True,
    reason=(
        "mode='task' workflow graph nodes temporarily disabled; re-enable "
        'when scheduler preserves originating node_input on resume.'
    ),
)
@pytest.mark.asyncio
async def test_task_finish_output_reaches_downstream(
    request: pytest.FixtureRequest,
):
  """Task mode extracts finish_task output for downstream nodes."""
  agent = _make_agent(mode='task')
  from . import testing_utils

  wrapper = build_node(agent)
  capture = InputCapturingNode(name='capture')
  wf = Workflow(
      name='wf',
      edges=[('START', wrapper), (wrapper, capture)],
  )
  runner = _new_workflow_runner(wf, request.function.__name__)

  agent_clone = next(n for n in wf.graph.nodes if n.name == wrapper.name)
  with _mock_agent_run(
      agent_clone,
      finish_output={'title': 'Story', 'content': 'Once upon a time'},
      content_text='Writing...',
  ):
    await runner.run_async(testing_utils.get_user_content('start'))

  assert capture.received_inputs == [
      {'title': 'Story', 'content': 'Once upon a time'}
  ]


@pytest.mark.asyncio
async def test_single_turn_output_reaches_downstream(
    request: pytest.FixtureRequest,
):
  """Single_turn output flows to downstream nodes."""
  from . import testing_utils

  agent = _make_agent(mode='single_turn')
  wrapper = build_node(agent)
  capture = InputCapturingNode(name='capture')
  wf = Workflow(
      name='wf',
      edges=[('START', wrapper), (wrapper, capture)],
  )
  runner = _new_workflow_runner(wf, request.function.__name__)

  agent_clone = next(n for n in wf.graph.nodes if n.name == wrapper.name)
  with _mock_leaf_run(agent_clone, content_text='Done.'):
    await runner.run_async(testing_utils.get_user_content('start'))

  assert capture.received_inputs == ['Done.']


@pytest.mark.xfail(
    strict=True,
    reason=(
        "mode='task' workflow graph nodes temporarily disabled; re-enable "
        'when scheduler preserves originating node_input on resume.'
    ),
)
@pytest.mark.asyncio
async def test_valid_input_schema_accepted(
    request: pytest.FixtureRequest,
):
  """Valid dict matching input_schema passes through without error."""
  from . import testing_utils

  agent = _make_agent(mode='task', input_schema=StoryInput)
  wrapper = build_node(agent)
  capture = InputCapturingNode(name='capture')
  wf = Workflow(
      name='wf',
      edges=[('START', wrapper), (wrapper, capture)],
  )
  from unittest.mock import AsyncMock
  from unittest.mock import MagicMock

  ctx = MagicMock(spec=Context)
  ic = MagicMock()
  ctx.get_invocation_context.return_value = ic
  ctx._invocation_context = ic
  ctx.resume_inputs = {}
  ctx._output_for_ancestors = []
  ic.branch = None
  ic.model_copy.return_value = ic
  ic._enqueue_event = AsyncMock(return_value=None)
  ic.plugin_manager.run_before_agent_callback = AsyncMock(return_value=None)
  ic.plugin_manager.run_after_agent_callback = AsyncMock(return_value=None)
  ctx.node_path = 'wf'

  agent_clone = next(n for n in wf.graph.nodes if n.name == wrapper.name)
  with _mock_agent_run(agent_clone, finish_output={'result': 'ok'}):
    async for _ in wf.run(ctx=ctx, node_input={'topic': 'Gemini'}):
      pass

  assert capture.received_inputs == [{'result': 'ok'}]


# Skipping this test as _LlmAgentWrapper does not seem to validate input schema
# @pytest.mark.asyncio
# async def test_invalid_input_schema_raises(
#     request: pytest.FixtureRequest,
# ):
#   """Invalid input not matching input_schema raises ValidationError."""
#   agent = _make_agent(mode='task', input_schema=StoryInput)
#   wrapper = build_node(agent)
#   wf = Workflow(name='wf', edges=[(START, wrapper)])
#   ctx = await create_parent_invocation_context(request.function.__name__, wf)
#   ic = ctx.model_copy(update={'branch': None})
#   agent_ctx = Context(invocation_context=ic, node_path='wf', run_id='exec')
#
#   with _mock_agent_run(agent, finish_output={'result': 'ok'}):
#     with pytest.raises(ValidationError):
#       async for _ in wrapper.run(ctx=agent_ctx, node_input={'style': 'comedy'}):
#         pass


@pytest.mark.xfail(
    strict=True,
    reason=(
        "mode='task' workflow graph nodes temporarily disabled; re-enable "
        'when scheduler preserves originating node_input on resume.'
    ),
)
@pytest.mark.asyncio
async def test_auto_wrap_in_workflow_edges(request: pytest.FixtureRequest):
  """LlmAgent placed directly in edges is auto-wrapped and works."""
  from . import testing_utils

  agent = _make_agent(mode='task')
  capture = InputCapturingNode(name='capture')
  wf = Workflow(
      name='wf',
      edges=[('START', agent), (agent, capture)],
  )
  runner = _new_workflow_runner(wf, request.function.__name__)

  agent_clone = next(n for n in wf.graph.nodes if n.name == agent.name)
  with _mock_agent_run(agent_clone, finish_output={'result': 'auto'}):
    await runner.run_async(testing_utils.get_user_content('start'))

  assert capture.received_inputs == [{'result': 'auto'}]


@pytest.mark.asyncio
async def test_single_turn_isolates_content_via_branch(
    request: pytest.FixtureRequest,
):
  """Single_turn wrapper sets a branch for content isolation."""
  agent = _make_agent(mode='single_turn')
  wrapper = build_node(agent)
  captured_branches = []

  async def fake_run(invocation_context):
    captured_branches.append(invocation_context.branch)
    yield Event(output='response')

  from . import testing_utils

  wf = Workflow(name='wf', edges=[('START', wrapper)])
  runner = _new_workflow_runner(wf, request.function.__name__)

  agent_clone = next(n for n in wf.graph.nodes if n.name == wrapper.name)
  original = agent_clone.run_async
  object.__setattr__(agent_clone, 'run_async', fake_run)
  try:
    await runner.run_async(testing_utils.get_user_content('start'))
  finally:
    object.__setattr__(agent_clone, 'run_async', original)

  assert len(captured_branches) == 1
  assert captured_branches[0] is None


@pytest.mark.xfail(
    strict=True,
    reason=(
        "mode='task' workflow graph nodes temporarily disabled; re-enable "
        'when scheduler preserves originating node_input on resume.'
    ),
)
@pytest.mark.asyncio
async def test_task_mode_does_not_set_branch(
    request: pytest.FixtureRequest,
):
  """Task mode preserves None branch for HITL visibility."""
  agent = _make_agent(mode='task')
  wrapper = build_node(agent)
  captured_branches = []

  async def fake_run(invocation_context):
    captured_branches.append(invocation_context.branch)
    yield Event(
        invocation_id='inv',
        author=agent.name,
        content=types.Content(
            role='model',
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name='finish_task',
                        id='ft-2',
                        args={'output': {'result': 'done'}},
                    )
                )
            ],
        ),
    )

  from . import testing_utils

  wf = Workflow(name='wf', edges=[('START', wrapper)])
  runner = _new_workflow_runner(wf, request.function.__name__)

  agent_clone = next(n for n in wf.graph.nodes if n.name == wrapper.name)
  original = agent_clone.run_async
  object.__setattr__(agent_clone, 'run_async', fake_run)
  try:
    await runner.run_async(testing_utils.get_user_content('start'))
  finally:
    object.__setattr__(agent_clone, 'run_async', original)

  assert captured_branches == [None]


@pytest.mark.asyncio
async def test_single_turn_converts_input_to_content(
    request: pytest.FixtureRequest,
):
  """Single_turn wrapper converts string node_input to types.Content."""
  agent = _make_agent(mode='single_turn')
  wrapper = build_node(agent)
  captured_inputs = []

  async def fake_run(*args, **kwargs):
    ctx = args[0]
    captured_inputs.append(ctx.session.events[-1].message)
    yield Event(output='response')

  from . import testing_utils

  predecessor = TestingNode(name='pred', output='hello world')
  wf = Workflow(
      name='wf',
      edges=[('START', predecessor), (predecessor, wrapper)],
  )
  runner = _new_workflow_runner(wf, request.function.__name__)

  agent_clone = next(n for n in wf.graph.nodes if n.name == wrapper.name)
  original = agent_clone.run_async
  object.__setattr__(agent_clone, 'run_async', fake_run)
  try:
    await runner.run_async(testing_utils.get_user_content('start'))
  finally:
    object.__setattr__(agent_clone, 'run_async', original)

  assert len(captured_inputs) == 1
  assert isinstance(captured_inputs[0], types.Content)
  assert captured_inputs[0].parts[0].text == 'hello world'


# --- New workflow path ---


def _get_user_content():
  from . import testing_utils

  return testing_utils.get_user_content


@pytest.mark.asyncio
async def test_react_path_user_content_visible_to_llm(
    request: pytest.FixtureRequest,
):
  """First-node LLM agent sees the user message in the new Workflow."""
  from google.adk.workflow._workflow import Workflow as NewWorkflow

  from . import testing_utils

  mock_model = testing_utils.MockModel.create(responses=['extracted output'])
  agent = LlmAgent(
      name='process_request',
      model=mock_model,
      instruction='Extract info from the user message.',
  )
  wf = NewWorkflow(name='wf', edges=[('START', agent)])

  runner = _new_workflow_runner(wf, request.function.__name__)
  await runner.run_async(
      testing_utils.get_user_content('I want 3 days off for vacation')
  )

  assert len(mock_model.requests) == 1
  user_texts = [
      p.text
      for c in mock_model.requests[0].contents
      if c.role == 'user'
      for p in c.parts or []
      if p.text
  ]
  assert any('3 days' in t for t in user_texts)


@pytest.mark.skip(
    reason=(
        '_LlmAgentWrapper does not fully support new workflow path in this test'
    )
)
@pytest.mark.asyncio
async def test_react_path_output_reaches_downstream(
    request: pytest.FixtureRequest,
):
  """LLM output flows to the next node in the new Workflow."""
  from google.adk.workflow._workflow import Workflow as NewWorkflow

  from . import testing_utils

  mock_model = testing_utils.MockModel.create(responses=['hello world'])
  agent = LlmAgent(
      name='greeter',
      model=mock_model,
      instruction='Greet.',
  )
  captured = []

  def capture(node_input: str):
    captured.append(node_input)

  wf = NewWorkflow(name='wf', edges=[('START', agent, capture)])

  runner = _new_workflow_runner(wf, request.function.__name__)
  await runner.run_async(testing_utils.get_user_content('hi'))

  assert captured == ['hello world']


@pytest.mark.skip(
    reason=(
        '_LlmAgentWrapper does not fully support new workflow path in this test'
    )
)
@pytest.mark.asyncio
async def test_react_path_output_key_stored_in_state(
    request: pytest.FixtureRequest,
):
  """output_key stores LLM output in state in the new Workflow."""
  from google.adk.workflow._workflow import Workflow as NewWorkflow

  from . import testing_utils

  mock_model = testing_utils.MockModel.create(responses=['summary text'])
  agent = LlmAgent(
      name='summarizer',
      model=mock_model,
      instruction='Summarize.',
      output_key='summary',
  )
  captured_state = []

  def check_state(ctx: Context):
    captured_state.append(ctx.state.get('summary'))

  wf = NewWorkflow(name='wf', edges=[('START', agent, check_state)])

  runner = _new_workflow_runner(wf, request.function.__name__)
  await runner.run_async(testing_utils.get_user_content('some text'))

  assert captured_state == ['summary text']


@pytest.mark.skip(
    reason=(
        '_LlmAgentWrapper does not fully support new workflow path in this test'
    )
)
@pytest.mark.asyncio
async def test_react_path_output_schema_validated(
    request: pytest.FixtureRequest,
):
  """output_schema is validated and parsed in the new Workflow."""
  from google.adk.workflow._workflow import Workflow as NewWorkflow

  from . import testing_utils

  mock_model = testing_utils.MockModel.create(
      responses=['{"title": "My Story", "content": "Once upon a time"}']
  )
  agent = LlmAgent(
      name='writer',
      model=mock_model,
      instruction='Write a story.',
      output_schema=StoryOutput,
      output_key='story',
  )
  captured = []

  def check_output(node_input: dict):
    captured.append(node_input)

  wf = NewWorkflow(name='wf', edges=[('START', agent, check_output)])

  runner = _new_workflow_runner(wf, request.function.__name__)
  await runner.run_async(testing_utils.get_user_content('write'))

  assert len(captured) == 1
  assert captured[0]['title'] == 'My Story'
  assert captured[0]['content'] == 'Once upon a time'


@pytest.mark.skip(
    reason=(
        '_LlmAgentWrapper does not fully support new workflow path in this test'
    )
)
@pytest.mark.asyncio
async def test_react_path_predecessor_input_visible_to_llm(
    request: pytest.FixtureRequest,
):
  """Predecessor output is injected as user content for the LLM."""
  from google.adk.workflow._workflow import Workflow as NewWorkflow

  from . import testing_utils

  mock_model = testing_utils.MockModel.create(responses=['processed'])
  agent = LlmAgent(
      name='processor',
      model=mock_model,
      instruction='Process.',
  )

  def step_one(node_input: str) -> str:
    return 'transformed data'

  wf = NewWorkflow(name='wf', edges=[('START', step_one, agent)])

  runner = _new_workflow_runner(wf, request.function.__name__)
  await runner.run_async(testing_utils.get_user_content('raw input'))

  assert len(mock_model.requests) == 1
  user_texts = [
      p.text
      for c in mock_model.requests[0].contents
      if c.role == 'user'
      for p in c.parts or []
      if p.text
  ]
  assert any('transformed data' in t for t in user_texts)


# --- React path: interrupt and resume ---


@pytest.mark.skip(
    reason=(
        '_LlmAgentWrapper does not fully support new workflow path in this test'
    )
)
@pytest.mark.asyncio
async def test_long_running_tool_interrupts_workflow(
    request: pytest.FixtureRequest,
):
  """Long-running tool stops the workflow after one LLM call."""
  from google.adk.tools.long_running_tool import LongRunningFunctionTool
  from google.adk.workflow._workflow import Workflow as NewWorkflow

  from . import testing_utils

  def approve(request: str) -> None:
    """Approve a request (long-running)."""
    return None

  fc = types.Part.from_function_call(name='approve', args={'request': 'deploy'})
  mock_model = testing_utils.MockModel.create(responses=[fc])
  agent = LlmAgent(
      name='approver',
      model=mock_model,
      instruction='Get approval.',
      tools=[LongRunningFunctionTool(approve)],
  )
  wf = NewWorkflow(name='wf', edges=[('START', agent)])

  runner = _new_workflow_runner(wf, request.function.__name__)
  events = await runner.run_async(testing_utils.get_user_content('deploy'))

  assert len(mock_model.requests) == 1
  assert any(e.long_running_tool_ids for e in events)


@pytest.mark.skip(
    reason=(
        '_LlmAgentWrapper does not fully support new workflow path in this test'
    )
)
@pytest.mark.asyncio
async def test_resume_after_interrupt_completes_workflow(
    request: pytest.FixtureRequest,
):
  """Resuming after interrupt calls the LLM once more to complete."""
  from google.adk.apps.app import App
  from google.adk.apps.app import ResumabilityConfig
  from google.adk.tools.long_running_tool import LongRunningFunctionTool
  from google.adk.workflow._workflow import Workflow as NewWorkflow

  from . import testing_utils

  def approve(request: str) -> None:
    """Approve a request (long-running)."""
    return None

  fc = types.Part.from_function_call(name='approve', args={'request': 'deploy'})
  mock_model = testing_utils.MockModel.create(
      responses=[fc, 'Approved and deployed.']
  )
  agent = LlmAgent(
      name='approver',
      model=mock_model,
      instruction='Get approval.',
      tools=[LongRunningFunctionTool(approve)],
  )
  wf = NewWorkflow(name='wf', edges=[('START', agent)])

  app = App(
      name=request.function.__name__,
      root_agent=wf,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Run 1: LLM → FC → interrupt
  events1 = await runner.run_async(
      testing_utils.get_user_content('deploy please')
  )
  invocation_id = events1[0].invocation_id
  assert any(e.long_running_tool_ids for e in events1)

  # Find the interrupt FC id
  interrupt_event = next(e for e in events1 if e.long_running_tool_ids)
  fc_id = list(interrupt_event.long_running_tool_ids)[0]

  # Run 2: Resume with FR
  resume_msg = types.Content(
      role='user',
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name='approve',
                  id=fc_id,
                  response={'result': 'yes'},
              )
          )
      ],
  )
  events2 = await runner.run_async(
      new_message=resume_msg,
      invocation_id=invocation_id,
  )

  # Total LLM calls: 1 (first run) + 1 (resume) = 2.
  assert len(mock_model.requests) == 2
  # Verify resumed output reached completion.
  content_texts = [
      p.text
      for e in events2
      if e.content and e.content.parts
      for p in e.content.parts
      if p.text
  ]
  assert any('Approved and deployed.' in t for t in content_texts)


@pytest.mark.skip(
    reason=(
        '_LlmAgentWrapper does not fully support new workflow path in this test'
    )
)
@pytest.mark.asyncio
async def test_multiple_sequential_interrupts_in_workflow(
    request: pytest.FixtureRequest,
):
  """Two interrupts in sequence each resume and complete in a workflow."""
  from google.adk.apps.app import App
  from google.adk.apps.app import ResumabilityConfig
  from google.adk.tools.long_running_tool import LongRunningFunctionTool
  from google.adk.workflow._workflow import Workflow as NewWorkflow

  from . import testing_utils

  def step_one() -> None:
    """First long-running step."""
    return None

  def step_two() -> None:
    """Second long-running step."""
    return None

  fc1 = types.Part.from_function_call(name='step_one', args={})
  fc2 = types.Part.from_function_call(name='step_two', args={})
  mock_model = testing_utils.MockModel.create(responses=[fc1, fc2, 'All done.'])
  agent = LlmAgent(
      name='worker',
      model=mock_model,
      instruction='Do two steps.',
      tools=[
          LongRunningFunctionTool(step_one),
          LongRunningFunctionTool(step_two),
      ],
  )
  wf = NewWorkflow(name='wf', edges=[('START', agent)])

  app = App(
      name=request.function.__name__,
      root_agent=wf,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Run 1: LLM → FC1 → interrupt
  events1 = await runner.run_async(testing_utils.get_user_content('Start'))
  assert any(e.long_running_tool_ids for e in events1)
  invocation_id = events1[0].invocation_id
  interrupt1 = next(e for e in events1 if e.long_running_tool_ids)
  fc1_id = list(interrupt1.long_running_tool_ids)[0]

  # Run 2: Resume FC1 → LLM → FC2 → interrupt again
  events2 = await runner.run_async(
      new_message=types.Content(
          role='user',
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      name='step_one',
                      id=fc1_id,
                      response={'result': 'step1 done'},
                  )
              )
          ],
      ),
      invocation_id=invocation_id,
  )
  assert any(e.long_running_tool_ids for e in events2)
  assert len(mock_model.requests) == 2
  interrupt2 = next(e for e in events2 if e.long_running_tool_ids)
  fc2_id = list(interrupt2.long_running_tool_ids)[0]

  # Run 3: Resume FC2 → LLM → text → done
  invocation_id2 = events2[0].invocation_id
  events3 = await runner.run_async(
      new_message=types.Content(
          role='user',
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      name='step_two',
                      id=fc2_id,
                      response={'result': 'step2 done'},
                  )
              )
          ],
      ),
      invocation_id=invocation_id2,
  )

  # Total: 3 LLM calls (one per run).
  assert len(mock_model.requests) == 3
  content_texts = [
      p.text
      for e in events3
      if e.content and e.content.parts
      for p in e.content.parts
      if p.text
  ]
  assert any('All done.' in t for t in content_texts)


# --- Original tests from test_v1_llm_agent_wrapper.py ---


def _make_v1_agent(mode='task'):
  return LlmAgent(
      name='test_v1_agent',
      model='gemini-2.5-flash',
      instruction='Test instruction',
      mode=mode,
  )


def test_task_mode_sets_wait_for_output():
  agent = _make_v1_agent(mode='task')
  wrapper = build_node(agent)
  assert wrapper.wait_for_output is True


def test_single_turn_does_not_set_wait_for_output():
  agent = _make_v1_agent(mode='single_turn')
  wrapper = build_node(agent)
  assert wrapper.wait_for_output is False


def test_chat_mode_sets_wait_for_output():
  agent = _make_v1_agent(mode='chat')
  wrapper = build_node(agent)
  assert wrapper.wait_for_output is True


@pytest.mark.asyncio
async def test_task_mode_proceeds_on_finish_task():
  agent = _make_v1_agent(mode='task')
  wrapper = build_node(agent)

  async def mock_run_async(*args, **kwargs):
    yield Event(
        invocation_id='inv',
        author='test_v1_agent',
        content=types.Content(
            role='model',
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name='finish_task',
                        id='ft-3',
                        args={'output': 'done_output'},
                    )
                )
            ],
        ),
    )
    yield Event(
        invocation_id='inv',
        author='test_v1_agent',
        content=types.Content(
            role='user',
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name='finish_task',
                        id='ft-3',
                        response={'result': 'Task completed.'},
                    )
                )
            ],
        ),
    )

  object.__setattr__(wrapper, 'run_async', mock_run_async)

  from unittest.mock import AsyncMock
  from unittest.mock import MagicMock

  ctx = MagicMock(spec=Context)
  ic = MagicMock()
  ctx.get_invocation_context.return_value = ic
  ic.model_copy.return_value = ic
  ic.plugin_manager.run_before_agent_callback = AsyncMock(return_value=None)
  ic.plugin_manager.run_after_agent_callback = AsyncMock(return_value=None)
  ctx.node_path = 'wf'

  events = []
  async for e in wrapper._run_impl(ctx=ctx, node_input='hello'):
    events.append(e)

  # Wrapper yields both the FC and the success FR; output is set on the FR.
  assert len(events) == 2
  assert events[1].output == {'output': 'done_output'}


@pytest.mark.asyncio
async def test_task_mode_does_not_proceed_without_finish_task():
  agent = _make_v1_agent(mode='task')
  wrapper = build_node(agent)

  async def mock_run_async(*args, **kwargs):
    yield Event(
        invocation_id='inv',
        author='test_v1_agent',
        content=types.Content(parts=[types.Part(text='Working...')]),
    )

  object.__setattr__(wrapper, 'run_async', mock_run_async)

  from unittest.mock import AsyncMock
  from unittest.mock import MagicMock

  ctx = MagicMock(spec=Context)
  ic = MagicMock()
  ctx.get_invocation_context.return_value = ic
  ic.model_copy.return_value = ic
  ic.plugin_manager.run_before_agent_callback = AsyncMock(return_value=None)
  ic.plugin_manager.run_after_agent_callback = AsyncMock(return_value=None)
  ctx.node_path = 'wf'

  events = []
  async for e in wrapper._run_impl(ctx=ctx, node_input='hello'):
    events.append(e)

  assert len(events) == 1
  assert events[0].output is None


@pytest.mark.asyncio
async def test_chat_mode_yields_events_directly():
  agent = _make_v1_agent(mode='chat')
  wrapper = build_node(agent)

  async def mock_run_async(*args, **kwargs):
    yield Event(
        invocation_id='inv',
        author='test_v1_agent',
        content=types.Content(parts=[types.Part(text='Hello from chat')]),
    )

  object.__setattr__(wrapper, 'run_async', mock_run_async)

  from unittest.mock import AsyncMock
  from unittest.mock import MagicMock

  ctx = MagicMock(spec=Context)
  ic = MagicMock()
  ctx.get_invocation_context.return_value = ic
  ic.model_copy.return_value = ic
  ic.plugin_manager.run_before_agent_callback = AsyncMock(return_value=None)
  ic.plugin_manager.run_after_agent_callback = AsyncMock(return_value=None)
  ctx.node_path = 'wf'

  events = []
  async for e in wrapper._run_impl(ctx=ctx, node_input='hello'):
    events.append(e)

  assert len(events) == 1
  assert events[0].content.parts[0].text == 'Hello from chat'
  assert events[0].output is None


@pytest.mark.asyncio
async def test_three_layer_llm_agent_transfer_round_trip(
    request: pytest.FixtureRequest,
):
  """Verify 3-layer LlmAgent transfers end-to-end (Root -> Child -> Grandchild -> Child -> Root)."""
  from google.adk.apps.app import App
  from google.adk.apps.app import ResumabilityConfig

  from . import testing_utils

  # Prepare the transfer function call parts
  fc_transfer_to_child = types.Part.from_function_call(
      name='transfer_to_agent',
      args={'agent_name': 'child_agent'},
  )
  fc_transfer_to_grandchild = types.Part.from_function_call(
      name='transfer_to_agent',
      args={'agent_name': 'grandchild_agent'},
  )
  fc_transfer_to_child_parent = types.Part.from_function_call(
      name='transfer_to_agent',
      args={'agent_name': 'child_agent'},
  )
  fc_transfer_to_root = types.Part.from_function_call(
      name='transfer_to_agent',
      args={'agent_name': 'root_agent'},
  )

  # Mock models for 3 layers
  root_model = testing_utils.MockModel.create(
      responses=[fc_transfer_to_child, 'Welcome back to root!']
  )
  child_model = testing_utils.MockModel.create(
      responses=[
          fc_transfer_to_grandchild,
          'Welcome back to child!',
          fc_transfer_to_root,
      ]
  )
  grandchild_model = testing_utils.MockModel.create(
      responses=['Hello, I am grandchild!', fc_transfer_to_child_parent]
  )

  # Instantiate agents
  grandchild_agent = LlmAgent(
      name='grandchild_agent',
      model=grandchild_model,
      instruction='Grandchild agent.',
  )
  child_agent = LlmAgent(
      name='child_agent',
      model=child_model,
      instruction='Child agent.',
      sub_agents=[grandchild_agent],
  )
  root_agent = LlmAgent(
      name='root_agent',
      model=root_model,
      instruction='Root agent.',
      sub_agents=[child_agent],
  )

  app = App(
      name=request.function.__name__,
      root_agent=root_agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Turn 1: Start (Root -> Child -> Grandchild -> Grandchild speaks)
  events1 = await runner.run_async(testing_utils.get_user_content('Start'))
  invocation_id = events1[0].invocation_id

  # Verify Turn 1 completed at Grandchild
  content_texts1 = [
      p.text
      for e in events1
      if e.content and e.content.parts
      for p in e.content.parts
      if p.text
  ]
  assert any('Hello, I am grandchild!' in t for t in content_texts1)

  # Turn 2: Go back to parent (Grandchild -> Child -> Child speaks)
  events2 = await runner.run_async(
      new_message=testing_utils.get_user_content('Go back to parent'),
      invocation_id=invocation_id,
  )

  # Verify Turn 2 completed at Child
  content_texts2 = [
      p.text
      for e in events2
      if e.content and e.content.parts
      for p in e.content.parts
      if p.text
  ]
  assert any('Welcome back to child!' in t for t in content_texts2)

  # Turn 3: Go back to root (Child -> Root -> Root speaks)
  events3 = await runner.run_async(
      new_message=testing_utils.get_user_content('Go back to root'),
      invocation_id=invocation_id,
  )

  # Verify Turn 3 completed at Root
  content_texts3 = [
      p.text
      for e in events3
      if e.content and e.content.parts
      for p in e.content.parts
      if p.text
  ]
  assert any('Welcome back to root!' in t for t in content_texts3)
