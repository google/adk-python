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

"""End-to-end tests for the Task Delegation API matrix.

Covers the complete cross-product of dispatch-shape × hierarchy-depth so
the chat-coordinator wrapper, the workflow-node task path, and the
nested-delegation path are all exercised:

* LlmAgent root → single task sub-agent (basic FC delegation).
* LlmAgent root → multiple task sub-agents (sequential delegation).
* LlmAgent root → task sub-agent → nested task sub-agent (chained).
* Workflow with a task-mode node (no FC delegation).
* Workflow with a task-mode node that itself has a task sub-agent.
* Dynamic node case (task agent dispatched via ``ctx.run_node``).
"""

from __future__ import annotations

from typing import Any
from typing import AsyncGenerator

from google.adk.agents.context import Context
from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
from google.adk.events.event import Event
from google.adk.workflow import node
from google.adk.workflow import START
from google.adk.workflow._base_node import BaseNode
from google.adk.workflow._workflow import Workflow
from google.genai import types
from pydantic import BaseModel
import pytest

from tests.unittests import testing_utils

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _delegate_part(target_name: str, request_text: str) -> types.Part:
  """LLM response calling a task sub-agent (the _TaskAgentTool FC)."""
  return types.Part.from_function_call(
      name=target_name, args={'request': request_text}
  )


def _finish_part(args: dict[str, Any]) -> types.Part:
  """LLM response calling finish_task with the given args."""
  return types.Part.from_function_call(name='finish_task', args=args)


def _text_part(text: str) -> types.Part:
  return types.Part.from_text(text=text)


def _make_task_agent(
    name: str,
    responses: list,
    *,
    sub_agents: list[LlmAgent] | None = None,
) -> LlmAgent:
  return LlmAgent(
      name=name,
      model=testing_utils.MockModel.create(responses=responses),
      mode='task',
      sub_agents=sub_agents or [],
  )


def _collect_finish_outputs(events: list[Event]) -> list[Any]:
  """Pull out finish_task FC arg dicts in chronological order."""
  out = []
  for e in events:
    for fc in e.get_function_calls():
      if fc.name == 'finish_task':
        out.append(dict(fc.args or {}))
  return out


def _get_text_responses(events: list[Event]) -> list[str]:
  """Concatenate text responses from all model events."""
  texts = []
  for e in events:
    if not e.content or not e.content.parts:
      continue
    for p in e.content.parts:
      if p.text and not p.thought:
        texts.append(p.text)
  return texts


# ---------------------------------------------------------------------------
# 1. LlmAgent root → single task sub-agent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_root_with_single_task_sub_agent(
    request: pytest.FixtureRequest,
):
  """Chat coordinator delegates to one task sub-agent and reports its output."""
  child = _make_task_agent(
      name='child',
      responses=[_finish_part({'result': 'child output'})],
  )

  root = LlmAgent(
      name='root',
      model=testing_utils.MockModel.create(
          responses=[
              _delegate_part('child', 'do the thing'),
              'All done: child output.',
          ]
      ),
      sub_agents=[child],
  )

  app = App(name=request.function.__name__, root_agent=root)
  runner = testing_utils.InMemoryRunner(app=app)

  events = await runner.run_async(testing_utils.get_user_content('hi'))

  finish_args = _collect_finish_outputs(events)
  assert finish_args == [{'result': 'child output'}]
  assert any(
      'All done: child output.' in t for t in _get_text_responses(events)
  )


# ---------------------------------------------------------------------------
# 2. LlmAgent root → multiple task sub-agents (sequential)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_root_with_two_task_sub_agents_sequential(
    request: pytest.FixtureRequest,
):
  """Chat coordinator delegates to two task sub-agents in one turn."""
  collector = _make_task_agent(
      name='collector',
      responses=[_finish_part({'result': 'collected'})],
  )
  payer = _make_task_agent(
      name='payer',
      responses=[_finish_part({'result': 'paid'})],
  )

  root = LlmAgent(
      name='root',
      model=testing_utils.MockModel.create(
          responses=[
              _delegate_part('collector', 'collect'),
              _delegate_part('payer', 'pay'),
              'Order placed.',
          ]
      ),
      sub_agents=[collector, payer],
  )

  app = App(name=request.function.__name__, root_agent=root)
  runner = testing_utils.InMemoryRunner(app=app)

  events = await runner.run_async(testing_utils.get_user_content('place order'))

  finish_args = _collect_finish_outputs(events)
  assert finish_args == [{'result': 'collected'}, {'result': 'paid'}]
  assert any('Order placed.' in t for t in _get_text_responses(events))


# ---------------------------------------------------------------------------
# 3. LlmAgent root → task sub-agent → nested task sub-agent
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        'Task-mode wrapper does not dispatch task-delegation FCs (only the '
        'chat-mode wrapper does), so a task-mode middle agent cannot delegate '
        'to its task sub-agent.  Documented limitation.'
    ),
    strict=True,
)
@pytest.mark.asyncio
async def test_chat_root_with_nested_task_delegation(
    request: pytest.FixtureRequest,
):
  """Task agent itself has a task sub-agent and delegates further."""
  grandchild = _make_task_agent(
      name='grandchild',
      responses=[_finish_part({'result': 'leaf'})],
  )

  child = LlmAgent(
      name='child',
      model=testing_utils.MockModel.create(
          responses=[
              _delegate_part('grandchild', 'leaf work'),
              _finish_part({'result': 'middle wraps leaf'}),
          ]
      ),
      mode='task',
      sub_agents=[grandchild],
  )

  root = LlmAgent(
      name='root',
      model=testing_utils.MockModel.create(
          responses=[
              _delegate_part('child', 'do the thing'),
              'Top-level done.',
          ]
      ),
      sub_agents=[child],
  )

  app = App(name=request.function.__name__, root_agent=root)
  runner = testing_utils.InMemoryRunner(app=app)

  events = await runner.run_async(testing_utils.get_user_content('hi'))

  finish_args = _collect_finish_outputs(events)
  # grandchild fires first (deepest), then child.
  assert finish_args == [
      {'result': 'leaf'},
      {'result': 'middle wraps leaf'},
  ]
  assert any('Top-level done.' in t for t in _get_text_responses(events))


# ---------------------------------------------------------------------------
# 4. Workflow with a single task-mode node (no FC delegation)
# ---------------------------------------------------------------------------


class _CaptureNode(BaseNode):
  """Records its node_input for assertion."""

  received: list[Any] = []

  async def _run_impl(self, *, ctx, node_input):
    type(self).received.append(node_input)
    yield Event(output=node_input)


@pytest.mark.asyncio
async def test_workflow_rejects_task_mode_graph_node():
  """A mode='task' LlmAgent cannot be used as a static workflow graph node.

  Task agents are multi-turn and need their originating ``node_input``
  preserved across re-dispatches — which the workflow scheduler doesn't
  do yet.  Until that lands, ``Workflow`` rejects them at construction
  time.  Task agents are still supported as chat sub-agents (FC
  delegation) and via dynamic ``ctx.run_node`` dispatch (see
  test_dynamic_dispatch_of_task_agent).
  """
  intake = _make_task_agent(name='intake', responses=[])
  capture = _CaptureNode(name='capture')

  with pytest.raises(ValueError, match="mode='task'"):
    Workflow(name='wf', edges=[(START, intake), (intake, capture)])


# ---------------------------------------------------------------------------
# 6. Dynamic node: function node that dispatches a task agent via ctx.run_node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dynamic_dispatch_of_task_agent(
    request: pytest.FixtureRequest,
):
  """A custom function node can dispatch a task agent and consume its output."""
  task_agent = _make_task_agent(
      name='task_agent',
      responses=[_finish_part({'result': 'dynamic output'})],
  )

  @node(rerun_on_resume=True)
  async def driver(*, ctx: Context, node_input: Any):
    output = await ctx.run_node(task_agent, node_input='go')
    yield Event(output=f'wrapped: {output}')

  wf = Workflow(name='wf', edges=[(START, driver)])

  app = App(name=request.function.__name__, root_agent=wf)
  runner = testing_utils.InMemoryRunner(app=app)

  events = await runner.run_async(testing_utils.get_user_content('start'))

  outputs = [e.output for e in events if e.output]
  assert any(
      isinstance(o, str) and 'dynamic output' in o for o in outputs
  ), f'expected wrapped dynamic output, got: {outputs}'


# ---------------------------------------------------------------------------
# 7. Validation error -> retry: wrapper yields the error FR and lets the LLM
#    emit a corrected finish_task on the next round.
# ---------------------------------------------------------------------------


class _StrictOutput(BaseModel):
  name: str
  age: int


@pytest.mark.asyncio
async def test_task_validation_error_drives_retry(
    request: pytest.FixtureRequest,
):
  """Bad finish_task args produce an error FR; the LLM gets a retry."""
  # First finish_task call has wrong types (age as string), second is correct.
  child_model = testing_utils.MockModel.create(
      responses=[
          _finish_part({'name': 'Jane', 'age': 'thirty'}),
          _finish_part({'name': 'Jane', 'age': 30}),
      ]
  )
  child = LlmAgent(
      name='child',
      model=child_model,
      mode='task',
      output_schema=_StrictOutput,
  )

  root = LlmAgent(
      name='root',
      model=testing_utils.MockModel.create(
          responses=[
              _delegate_part('child', 'gather identity'),
              'All set.',
          ]
      ),
      sub_agents=[child],
  )

  app = App(name=request.function.__name__, root_agent=root)
  runner = testing_utils.InMemoryRunner(app=app)

  events = await runner.run_async(testing_utils.get_user_content('hi'))

  # The mock LLM was called twice for the child (the bad attempt + the
  # corrected one), proving the wrapper looped instead of terminating
  # on the first finish_task.
  assert child_model.response_index == 1
  finish_args = _collect_finish_outputs(events)
  assert finish_args == [
      {'name': 'Jane', 'age': 'thirty'},
      {'name': 'Jane', 'age': 30},
  ]
  # The validation-error FR should be present in session for the LLM
  # to see on its retry round.
  error_frs = [
      fr.response
      for e in events
      for fr in e.get_function_responses()
      if fr.name == 'finish_task'
      and isinstance(fr.response, dict)
      and 'error' in fr.response
  ]
  assert len(error_frs) == 1, f'expected one error FR, got {error_frs}'


# ---------------------------------------------------------------------------
# 8. Cross-turn resumption: an unresolved task FC from a prior turn is
#    re-dispatched by the chat coordinator on the next user turn, before
#    the LLM is called.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_coordinator_resumes_unresolved_task_fc(
    request: pytest.FixtureRequest,
):
  """Pending task FC from a prior turn is dispatched before the new LLM call."""
  child_model = testing_utils.MockModel.create(
      responses=[_finish_part({'result': 'finished after resume'})]
  )
  child = LlmAgent(name='child', model=child_model, mode='task')

  root_model = testing_utils.MockModel.create(
      responses=[
          # Only response needed: post-resume continuation after the
          # pre-LLM scan dispatches the pending task and synthesizes its FR.
          'Resumed and done.',
      ]
  )
  root = LlmAgent(
      name='root',
      model=root_model,
      sub_agents=[child],
  )

  # Seed the session with an unresolved task delegation FC authored by
  # root from a "prior turn".  No matching FR exists.
  from google.adk.sessions.in_memory_session_service import InMemorySessionService

  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name=request.function.__name__,
      user_id='u',
  )
  await session_service.append_event(
      session=session,
      event=Event(
          invocation_id='prior-inv',
          author='root',
          content=types.Content(
              role='model',
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          id='fc-pending',
                          name='child',
                          args={'request': 'leftover work'},
                      )
                  )
              ],
          ),
      ),
  )

  from google.adk.runners import Runner

  app = App(name=request.function.__name__, root_agent=root)
  runner = Runner(app=app, session_service=session_service)

  events = []
  async for ev in runner.run_async(
      user_id='u',
      session_id=session.id,
      new_message=testing_utils.get_user_content('continue'),
  ):
    events.append(ev)

  # The child must have been dispatched once (resuming the pending FC).
  assert (
      child_model.response_index == 0
  ), 'child LLM should have been called exactly once for the resumed task'
  finish_args = _collect_finish_outputs(events)
  assert {
      'result': 'finished after resume'
  } in finish_args, f'expected resumed task to finish; got {finish_args}'


# ---------------------------------------------------------------------------
# 9. Strict isolation filtering: a stranger event with a foreign
#    isolation_scope must NOT appear in the task agent's LLM context.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_strict_isolation_filter_excludes_foreign_scope(
    request: pytest.FixtureRequest,
):
  """Garbage-scoped events are excluded from the task agent's view."""
  child_model = testing_utils.MockModel.create(
      responses=[_finish_part({'result': 'ok'})]
  )
  child = LlmAgent(name='child', model=child_model, mode='task')

  root = LlmAgent(
      name='root',
      model=testing_utils.MockModel.create(
          responses=[
              _delegate_part('child', 'do the thing'),
              'Done.',
          ]
      ),
      sub_agents=[child],
  )

  from google.adk.sessions.in_memory_session_service import InMemorySessionService

  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name=request.function.__name__,
      user_id='u',
  )
  # Seed a stranger event with a different scope.
  stranger = Event(
      invocation_id='stranger-inv',
      author='someone_else',
      content=types.Content(
          role='user',
          parts=[types.Part(text='SECRET-SHOULD-NOT-LEAK')],
      ),
  )
  stranger.isolation_scope = 'garbage-scope'
  session.events.append(stranger)

  from google.adk.runners import Runner

  app = App(name=request.function.__name__, root_agent=root)
  runner = Runner(app=app, session_service=session_service)

  async for _ in runner.run_async(
      user_id='u',
      session_id=session.id,
      new_message=testing_utils.get_user_content('go'),
  ):
    pass

  # Inspect the child's LLM request: SECRET text must not appear.
  child_request = child_model.requests[0]
  rendered = '\n'.join(
      p.text or '' for c in child_request.contents or [] for p in c.parts or []
  )
  assert (
      'SECRET-SHOULD-NOT-LEAK' not in rendered
  ), 'stranger event leaked across isolation_scope filter'
