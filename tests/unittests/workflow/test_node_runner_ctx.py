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

"""Tests for NodeRunner → Context as result channel.

Verifies that NodeRunner correctly populates ctx.output, ctx.route,
and ctx.interrupt_ids from yielded events and direct assignment,
and that resume state (prior_output, prior_interrupt_ids) is carried
forward correctly.
"""

from collections.abc import AsyncGenerator
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.context import Context
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import _rewind_events
from google.adk.events.event import Event
from google.adk.events.event import NodeInfo
from google.adk.events.event_actions import EventActions
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.workflow._base_node import BaseNode
from google.adk.workflow._dynamic_node_scheduler import DynamicNodeScheduler
from google.adk.workflow._dynamic_node_scheduler import DynamicNodeState
from google.adk.workflow._node_runner import NodeRunner
from google.genai import types
import pytest

# --- Helpers ---


def _make_ctx(invocation_id='inv-test', enqueue_events=None, node_path=''):
  """Create a minimal Context mock with IC."""
  mock_agent = MagicMock(spec=BaseAgent)
  real_session = Session(
      id='test_session', app_name='test_app', user_id='test_user'
  )
  real_session_service = InMemorySessionService()

  ic = InvocationContext(
      invocation_id=invocation_id,
      agent=mock_agent,
      session=real_session,
      session_service=real_session_service,
  )

  collected = enqueue_events if enqueue_events is not None else []

  async def _enqueue(event):
    collected.append(event)

  object.__setattr__(ic, '_enqueue_event', AsyncMock(side_effect=_enqueue))

  ctx = Context(
      invocation_context=ic,
      node_path=node_path,
  )
  return ctx, collected


# =========================================================================
# Context as RESULT — fields populated by NodeRunner after execution
# =========================================================================


# --- ctx.output from yielded events ---


@pytest.mark.asyncio
async def test_yield_value_sets_ctx_output():
  """Yielding a value sets ctx.output on the returned context."""

  class _Node(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      yield 'hello'

  parent_ctx, _ = _make_ctx()
  child_ctx = await NodeRunner(
      node=_Node(name='n'), parent_ctx=parent_ctx
  ).run()

  assert child_ctx.output == 'hello'


@pytest.mark.asyncio
async def test_yield_event_output_sets_ctx_output():
  """Yielding Event(output=X) sets ctx.output."""

  class _Node(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      yield Event(output='from_event')

  parent_ctx, _ = _make_ctx()
  child_ctx = await NodeRunner(
      node=_Node(name='n'), parent_ctx=parent_ctx
  ).run()

  assert child_ctx.output == 'from_event'


@pytest.mark.asyncio
async def test_no_yield_leaves_ctx_output_none():
  """A node that yields nothing leaves ctx.output as None."""

  class _Node(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      return
      yield  # noqa: unreachable

  parent_ctx, _ = _make_ctx()
  child_ctx = await NodeRunner(
      node=_Node(name='n'), parent_ctx=parent_ctx
  ).run()

  assert child_ctx.output is None


# --- ctx.output set directly ---


@pytest.mark.asyncio
async def test_ctx_output_set_directly():
  """Setting ctx.output directly produces a deferred output event."""

  class _Node(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      ctx.output = 'direct'
      yield  # noqa: unreachable

  parent_ctx, events = _make_ctx()
  child_ctx = await NodeRunner(
      node=_Node(name='n'), parent_ctx=parent_ctx
  ).run()

  assert child_ctx.output == 'direct'
  output_events = [e for e in events if e.output is not None]
  assert len(output_events) == 1
  assert output_events[0].output == 'direct'


@pytest.mark.asyncio
async def test_ctx_output_direct_with_state_delta():
  """Deferred output bundles pending state deltas onto the same event."""

  class _Node(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      ctx.state['key'] = 'val'
      ctx.output = 'result'
      yield  # noqa: unreachable

  parent_ctx, events = _make_ctx()
  child_ctx = await NodeRunner(
      node=_Node(name='n'), parent_ctx=parent_ctx
  ).run()

  assert child_ctx.output == 'result'
  output_events = [e for e in events if e.output is not None]
  assert len(output_events) == 1
  assert output_events[0].actions.state_delta['key'] == 'val'


@pytest.mark.asyncio
async def test_deferred_output_emitted_after_intermediate():
  """ctx.output set directly emits after intermediate content events."""

  class _Node(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      ctx.output = 'deferred'
      yield Event(content=types.Content(parts=[types.Part(text='working')]))

  parent_ctx, events = _make_ctx()
  child_ctx = await NodeRunner(
      node=_Node(name='n'), parent_ctx=parent_ctx
  ).run()

  assert child_ctx.output == 'deferred'
  assert len(events) == 2
  assert events[0].content.parts[0].text == 'working'
  assert events[1].output == 'deferred'


# --- ctx.output validation ---


@pytest.mark.asyncio
async def test_double_output_raises():
  """Setting ctx.output twice raises ValueError."""

  class _Node(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      ctx.output = 'first'
      ctx.output = 'second'
      yield  # noqa: unreachable

  parent_ctx, events = _make_ctx()
  await NodeRunner(node=_Node(name='n'), parent_ctx=parent_ctx).run()
  error_events = [e for e in events if e.error_code]
  assert len(error_events) == 1
  assert error_events[0].error_code == 'ValueError'
  assert 'already set' in error_events[0].error_message


@pytest.mark.asyncio
async def test_yield_then_ctx_output_raises():
  """Yielding output then setting ctx.output raises ValueError."""

  class _Node(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      yield 'first'
      ctx.output = 'second'

  parent_ctx, events = _make_ctx()
  await NodeRunner(node=_Node(name='n'), parent_ctx=parent_ctx).run()
  error_events = [e for e in events if e.error_code]
  assert len(error_events) == 1
  assert error_events[0].error_code == 'ValueError'
  assert 'already set' in error_events[0].error_message


# --- ctx.route ---


@pytest.mark.asyncio
async def test_yield_route_sets_ctx_route():
  """Yielding Event(route=R) sets ctx.route."""

  class _Node(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      yield Event(output='out', route='next')

  parent_ctx, _ = _make_ctx()
  child_ctx = await NodeRunner(
      node=_Node(name='n'), parent_ctx=parent_ctx
  ).run()

  assert child_ctx.output == 'out'
  assert child_ctx.route == 'next'


@pytest.mark.asyncio
async def test_ctx_route_set_directly():
  """Setting ctx.route directly is readable after run."""

  class _Node(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      ctx.route = 'branch_a'
      yield 'out'

  parent_ctx, _ = _make_ctx()
  child_ctx = await NodeRunner(
      node=_Node(name='n'), parent_ctx=parent_ctx
  ).run()

  assert child_ctx.route == 'branch_a'


@pytest.mark.asyncio
@pytest.mark.parametrize('with_state', [False, True])
@pytest.mark.parametrize('with_artifact', [False, True])
async def test_deferred_route_is_emitted_with_pending_deltas(
    with_state: bool, with_artifact: bool
) -> None:
  """A final event keeps its route alongside state and artifact updates."""

  class _Node(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Event, None]:
      ctx.output = 'result'
      ctx.route = 'branch_a'
      if with_state:
        ctx.state['visited'] = True
      if with_artifact:
        ctx.actions.artifact_delta['report.txt'] = 1
      return
      yield

  parent_ctx, events = _make_ctx()
  await NodeRunner(node=_Node(name='router'), parent_ctx=parent_ctx).run()

  assert len(events) == 1
  event = events[0]
  assert event.output == 'result'
  assert event.actions.route == 'branch_a'
  assert event.actions.state_delta == ({'visited': True} if with_state else {})
  assert event.actions.artifact_delta == (
      {'report.txt': 1} if with_artifact else {}
  )


# --- ctx.interrupt_ids ---


@pytest.mark.asyncio
async def test_interrupt_sets_ctx_interrupt_ids():
  """Yielding an interrupt event populates ctx.interrupt_ids."""

  class _Node(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      yield Event(
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='tool', args={}, id='fc-1'
                      )
                  )
              ]
          ),
          long_running_tool_ids={'fc-1'},
      )

  parent_ctx, _ = _make_ctx()
  child_ctx = await NodeRunner(
      node=_Node(name='n'), parent_ctx=parent_ctx
  ).run()

  assert child_ctx.interrupt_ids == {'fc-1'}
  assert child_ctx.output is None


@pytest.mark.asyncio
async def test_output_and_interrupt_coexist():
  """Output and interrupt can coexist across separate events."""

  class _Node(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      yield 'result'
      yield Event(
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='tool', args={}, id='fc-1'
                      )
                  )
              ]
          ),
          long_running_tool_ids={'fc-1'},
      )

  parent_ctx, _ = _make_ctx()
  child_ctx = await NodeRunner(
      node=_Node(name='n'), parent_ctx=parent_ctx
  ).run()

  assert child_ctx.output == 'result'
  assert child_ctx.interrupt_ids == {'fc-1'}


@pytest.mark.asyncio
async def test_duplicate_interrupt_ids_deduplicated():
  """Duplicate interrupt IDs are deduplicated (set semantics)."""

  class _Node(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      yield Event(long_running_tool_ids={'fc-1', 'fc-2'})
      yield Event(long_running_tool_ids={'fc-2', 'fc-3'})

  parent_ctx, _ = _make_ctx()
  child_ctx = await NodeRunner(
      node=_Node(name='n'), parent_ctx=parent_ctx
  ).run()

  assert child_ctx.interrupt_ids == {'fc-1', 'fc-2', 'fc-3'}


# --- Output delegation (use_as_output) ---


@pytest.mark.asyncio
async def test_delegated_output_not_enqueued():
  """When output is delegated, the output event is not enqueued."""

  class _Node(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      ctx._output_delegated = True
      yield 'delegated_value'

  parent_ctx, events = _make_ctx()
  child_ctx = await NodeRunner(
      node=_Node(name='n'), parent_ctx=parent_ctx
  ).run()

  assert child_ctx.output == 'delegated_value'
  output_events = [e for e in events if e.output is not None]
  assert len(output_events) == 0


@pytest.mark.asyncio
async def test_delegated_ctx_output_not_emitted():
  """When output is delegated and set via ctx.output, no event emitted."""

  class _Node(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      ctx._output_delegated = True
      ctx.output = 'delegated_direct'
      yield  # noqa: unreachable

  parent_ctx, events = _make_ctx()
  child_ctx = await NodeRunner(
      node=_Node(name='n'), parent_ctx=parent_ctx
  ).run()

  assert child_ctx.output == 'delegated_direct'
  output_events = [e for e in events if e.output is not None]
  assert len(output_events) == 0


@pytest.mark.asyncio
async def test_delegated_output_preserves_event_details():
  """When output is delegated, the event is enqueued but output is suppressed."""
  from google.adk.events.event_actions import EventActions

  class _Node(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      ctx._output_delegated = True
      yield Event(
          output='delegated_value',
          actions=EventActions(state_delta={'foo': 'bar'}),
          content=types.Content(role='model', parts=[types.Part(text='hello')]),
      )

  parent_ctx, events = _make_ctx()
  child_ctx = await NodeRunner(
      node=_Node(name='n'), parent_ctx=parent_ctx
  ).run()

  assert child_ctx.output == 'delegated_value'
  assert len(events) == 1
  event = events[0]
  assert event.output is None
  assert event.actions.state_delta == {'foo': 'bar'}
  assert event.content.parts[0].text == 'hello'


# =========================================================================
# Context as INPUT — resume state provided to NodeRunner at construction
# =========================================================================


@pytest.mark.asyncio
async def test_prior_output_carried_forward():
  """Prior output from a previous run is available on ctx.output."""

  class _Node(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      return
      yield  # noqa: unreachable

  parent_ctx, _ = _make_ctx()
  child_ctx = await NodeRunner(
      node=_Node(name='n'),
      parent_ctx=parent_ctx,
      prior_output='cached_result',
  ).run()

  assert child_ctx.output == 'cached_result'


@pytest.mark.asyncio
async def test_prior_interrupt_ids_carried_forward():
  """Prior interrupt IDs from a previous run are on ctx."""

  class _Node(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      return
      yield  # noqa: unreachable

  parent_ctx, _ = _make_ctx()
  child_ctx = await NodeRunner(
      node=_Node(name='n'),
      parent_ctx=parent_ctx,
      prior_interrupt_ids={'fc-old'},
  ).run()

  assert 'fc-old' in child_ctx.interrupt_ids


@pytest.mark.asyncio
async def test_prior_and_new_interrupt_ids_merged():
  """New interrupt IDs are merged with prior ones."""

  class _Node(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      yield Event(
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='tool', args={}, id='fc-new'
                      )
                  )
              ]
          ),
          long_running_tool_ids={'fc-new'},
      )

  parent_ctx, _ = _make_ctx()
  child_ctx = await NodeRunner(
      node=_Node(name='n'),
      parent_ctx=parent_ctx,
      prior_interrupt_ids={'fc-old'},
  ).run()

  assert child_ctx.interrupt_ids == {'fc-old', 'fc-new'}


class _IdleNode(BaseNode):

  async def _run_impl(self, *, ctx, node_input):
    return
    yield  # noqa: unreachable


def _node_event(path: str, invocation_id: str = 'inv-test') -> Event:
  return Event(
      author='node',
      invocation_id=invocation_id,
      node_info=NodeInfo(path=path),
      content=types.Content(parts=[types.Part(text=path)]),
  )


@pytest.mark.asyncio
async def test_node_without_history_skips_rewind_pass():
  """A node with no event of its own in the history skips the rewind pass."""
  parent_ctx, _ = _make_ctx()
  parent_ctx.session.events.extend([
      _node_event('n@1', invocation_id='earlier-turn'),
      _node_event('other@1'),
      _node_event('n@10'),
  ])

  with mock.patch.object(
      _rewind_events, '_apply_rewinds', wraps=_rewind_events._apply_rewinds
  ) as apply_rewinds:
    child_ctx = await NodeRunner(
        node=_IdleNode(name='n'), parent_ctx=parent_ctx
    ).run()

  apply_rewinds.assert_not_called()
  assert not child_ctx.resume_inputs


@pytest.mark.asyncio
async def test_node_with_answered_interrupt_rehydrates_resume_inputs():
  """A node's answered interrupt in this invocation is still resumed."""
  parent_ctx, _ = _make_ctx()
  interrupt = _node_event('n@1')
  interrupt.content = types.Content(
      parts=[
          types.Part(
              function_call=types.FunctionCall(name='ask', args={}, id='fc-1')
          )
      ]
  )
  interrupt.long_running_tool_ids = {'fc-1'}
  answer = Event(
      author='user',
      invocation_id='inv-test',
      content=types.Content(
          role='user',
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      name='ask', id='fc-1', response={'answer': 'yes'}
                  )
              )
          ],
      ),
  )
  parent_ctx.session.events.extend([interrupt, answer])

  with mock.patch.object(
      _rewind_events, '_apply_rewinds', wraps=_rewind_events._apply_rewinds
  ) as apply_rewinds:
    child_ctx = await NodeRunner(
        node=_IdleNode(name='n'), parent_ctx=parent_ctx
    ).run()

  apply_rewinds.assert_called_once()
  assert child_ctx.resume_inputs == {'fc-1': {'answer': 'yes'}}


class _IterationCountingList(list):

  iterations = 0

  def __iter__(self):
    self.iterations += 1
    return super().__iter__()


def _interrupt_event(path: str, fc_id: str, invocation_id: str) -> Event:
  event = _node_event(path, invocation_id=invocation_id)
  event.content = types.Content(
      parts=[
          types.Part(
              function_call=types.FunctionCall(name='ask', args={}, id=fc_id)
          )
      ]
  )
  event.long_running_tool_ids = {fc_id}
  return event


def _answer_event(fc_id: str, answer: str, invocation_id: str) -> Event:
  return Event(
      author='user',
      invocation_id=invocation_id,
      content=types.Content(
          role='user',
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      name='ask', id=fc_id, response={'answer': answer}
                  )
              )
          ],
      ),
  )


def _parent_ctx_with_indexed_history(events: list[Event]) -> Context:
  """A workflow ctx whose scheduler has indexed `events`, as a run would."""
  parent_ctx, _ = _make_ctx(node_path='wf@1')
  parent_ctx.session.events = _IterationCountingList(events)
  scheduler = DynamicNodeScheduler(state=DynamicNodeState())
  scheduler._replay_manager._ensure_index(parent_ctx)
  parent_ctx._workflow_scheduler = scheduler
  return parent_ctx


def test_child_context_does_not_rescan_an_indexed_session():
  """A node run reads the scheduler's index rather than every session event."""
  parent_ctx = _parent_ctx_with_indexed_history(
      [_node_event(f'wf@1/n@{i}', invocation_id=f'turn-{i}') for i in range(50)]
      + [_node_event('wf@1/other@1')]
  )
  parent_ctx.session.events.iterations = 0

  NodeRunner(
      node=_IdleNode(name='n'), parent_ctx=parent_ctx
  )._create_child_context(resume_inputs=None)

  assert parent_ctx.session.events.iterations == 0


_REWIND_TO_TURN_1 = Event(
    author='user',
    invocation_id='inv-test',
    actions=EventActions(rewind_before_invocation_id='turn-1'),
)


@pytest.mark.parametrize(
    'history',
    [
        [],
        [_node_event('wf@1/n@1', invocation_id='turn-1')],
        [
            _interrupt_event('wf@1/n@1', 'fc-old', 'turn-1'),
            _answer_event('fc-old', 'old', 'turn-1'),
            _interrupt_event('wf@1/n@1', 'fc-1', 'inv-test'),
            _answer_event('fc-1', 'yes', 'inv-test'),
        ],
        [
            _interrupt_event('wf@1/n@1/deep@1', 'fc-1', 'inv-test'),
            _node_event('wf@1/n@10', invocation_id='inv-test'),
            _answer_event('fc-1', 'deep', 'inv-test'),
        ],
        [
            _interrupt_event('wf@1/n@1', 'fc-1', 'inv-test'),
            _answer_event('fc-1', 'first', 'inv-test'),
            _interrupt_event('wf@1/n@1', 'fc-1', 'inv-test'),
        ],
        [
            _interrupt_event('wf@1/n@1', 'fc-1', 'turn-1'),
            _answer_event('fc-1', 'rewound', 'inv-test'),
            _REWIND_TO_TURN_1,
            _interrupt_event('wf@1/n@1', 'fc-2', 'inv-test'),
            _answer_event('fc-2', 'kept', 'inv-test'),
        ],
    ],
)
@pytest.mark.parametrize('prior_interrupt_ids', [None, {'fc-1'}])
def test_child_context_rehydration_matches_a_session_scan(
    history, prior_interrupt_ids
):
  """Rehydrating from the scheduler's index matches scanning the session."""

  def _resume_inputs(parent_ctx):
    return (
        NodeRunner(
            node=_IdleNode(name='n'),
            parent_ctx=parent_ctx,
            prior_interrupt_ids=prior_interrupt_ids,
        )
        ._create_child_context(resume_inputs={'given': 1})
        ._resume_inputs
    )

  scanned_ctx, _ = _make_ctx(node_path='wf@1')
  scanned_ctx.session.events.extend(history)

  assert _resume_inputs(
      _parent_ctx_with_indexed_history(list(history))
  ) == _resume_inputs(scanned_ctx)


# =========================================================================
# event_author — parent orchestrator overrides event author
# =========================================================================


@pytest.mark.asyncio
async def test_event_author_defaults_to_node_name():
  """Without event_author, events use the node's own name."""

  class _Node(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      yield 'result'

  parent_ctx, events = _make_ctx()
  await NodeRunner(node=_Node(name='my_node'), parent_ctx=parent_ctx).run()

  assert events[0].author == 'my_node'


@pytest.mark.asyncio
async def test_event_author_overrides_node_name():
  """When parent sets event_author, events use that instead."""

  class _Node(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      yield 'result'

  parent_ctx, events = _make_ctx()
  parent_ctx.event_author = 'my_workflow'
  await NodeRunner(node=_Node(name='my_node'), parent_ctx=parent_ctx).run()

  assert events[0].author == 'my_workflow'


@pytest.mark.asyncio
async def test_event_author_overrides_preset_author():
  """event_author always wins, even over a pre-set event author."""

  class _Node(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      yield Event(author='custom_author', output='result')

  parent_ctx, events = _make_ctx()
  parent_ctx.event_author = 'my_workflow'
  await NodeRunner(node=_Node(name='my_node'), parent_ctx=parent_ctx).run()

  assert events[0].author == 'my_workflow'


# =========================================================================
# Branch propagation tests
# =========================================================================


@pytest.mark.asyncio
async def test_override_branch_used_in_node_runner():
  """NodeRunner uses override_branch if provided."""

  class _Node(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      yield Event(output='result')

  parent_ctx, events = _make_ctx()
  await NodeRunner(
      node=_Node(name='n'),
      parent_ctx=parent_ctx,
      override_branch='custom_branch',
  ).run()

  assert events[0].branch == 'custom_branch'


@pytest.mark.asyncio
async def test_use_sub_branch_appends_segment_to_branch():
  """NodeRunner appends node_name@run_id to branch when use_sub_branch is True."""

  class _Node(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      yield Event(output='result')

  parent_ctx, events = _make_ctx()
  parent_ctx._invocation_context.branch = 'parent_branch'
  await NodeRunner(
      node=_Node(name='n'),
      parent_ctx=parent_ctx,
      use_sub_branch=True,
      run_id='1',
  ).run()

  assert events[0].branch == 'parent_branch.n@1'


@pytest.mark.asyncio
async def test_delegated_output_is_published_on_claiming_parent_branch():
  """use_as_output moves only the child's output event to the parent branch."""

  class _Node(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      yield Event(content=types.ModelContent('working'))
      yield Event(output='result')

  parent_ctx, events = _make_ctx()
  parent_ctx._invocation_context.branch = 'parent_branch'

  await NodeRunner(
      node=_Node(name='n'),
      parent_ctx=parent_ctx,
      use_sub_branch=True,
      use_as_output=True,
      run_id='1',
  ).run()

  assert [e.branch for e in events] == ['parent_branch.n@1', 'parent_branch']
  assert events[1].node_info.path.endswith('n@1')


@pytest.mark.asyncio
async def test_delegated_output_on_function_response_keeps_child_branch():
  """FunctionResponse events carrying output stay paired on the child branch."""

  class _TaskLikeNode(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      yield Event(
          content=types.Content(
              role='model',
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          id='fc_finish',
                          name='finish_task',
                          args={'result': 'done'},
                      )
                  )
              ],
          )
      )
      yield Event(
          content=types.Content(
              role='user',
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          id='fc_finish',
                          name='finish_task',
                          response={'result': 'Task completed.'},
                      )
                  )
              ],
          ),
          output='done',
      )

  parent_ctx, events = _make_ctx()
  parent_ctx._invocation_context.branch = 'parent_branch'

  child_ctx = await NodeRunner(
      node=_TaskLikeNode(name='n'),
      parent_ctx=parent_ctx,
      use_sub_branch=True,
      use_as_output=True,
      run_id='1',
  ).run()

  assert child_ctx.output == 'done'
  assert [e.branch for e in events] == [
      'parent_branch.n@1',
      'parent_branch.n@1',
  ]
  assert events[1].node_info.output_for == [
      events[1].node_info.path,
      parent_ctx.node_path,
  ]


@pytest.mark.asyncio
async def test_sequential_branch_propagation():
  """NodeRunner inherits parent branch when use_sub_branch is False."""

  class _Node(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      yield Event(output='result')

  parent_ctx, events = _make_ctx()
  parent_ctx._invocation_context.branch = 'parent_branch'
  await NodeRunner(
      node=_Node(name='n'),
      parent_ctx=parent_ctx,
      use_sub_branch=False,
  ).run()

  assert events[0].branch == 'parent_branch'


@pytest.mark.asyncio
async def test_child_event_branch_does_not_mutate_parent_ic():
  """A child node altering its branch does not mutate the parent's shared InvocationContext branch."""

  class _Node(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      yield Event(output='result', branch='new_child_branch')

  parent_ctx, events = _make_ctx()
  parent_ctx._invocation_context.branch = 'parent_branch'
  await NodeRunner(
      node=_Node(name='n'),
      parent_ctx=parent_ctx,
      use_sub_branch=False,
  ).run()

  assert events[0].branch == 'new_child_branch'
  # The parent's branch must remain unchanged.
  assert parent_ctx._invocation_context.branch == 'parent_branch'


@pytest.mark.asyncio
async def test_own_event_branch_moves_node_branch():
  """An event the node authors on another branch moves the node there."""

  class _Node(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      yield Event(branch='other_branch')
      yield Event(output='result')

  parent_ctx, events = _make_ctx()
  parent_ctx._invocation_context.branch = 'parent_branch'
  child_ctx = await NodeRunner(
      node=_Node(name='n'), parent_ctx=parent_ctx
  ).run()

  assert [e.branch for e in events] == ['other_branch', 'other_branch']
  assert child_ctx._invocation_context.branch == 'other_branch'


@pytest.mark.asyncio
async def test_relayed_event_branch_does_not_move_node_branch():
  """A sub-agent's event relayed by the node does not move the node's branch.

  A structured agent run as a node, such as a ParallelAgent, re-yields events
  its sub-agents emitted on their own sub-branches. Each relayed event keeps
  its sub-branch, but the node stays on its own branch, so its later events
  and the branch its successors inherit do not depend on which sub-agent
  spoke last.
  """

  class _Node(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      yield Event(author='sub_agent', branch='parent_branch.n.sub_agent')
      yield Event(output='result')

  parent_ctx, events = _make_ctx()
  parent_ctx._invocation_context.branch = 'parent_branch'
  child_ctx = await NodeRunner(
      node=_Node(name='n'), parent_ctx=parent_ctx
  ).run()

  assert [e.branch for e in events] == [
      'parent_branch.n.sub_agent',
      'parent_branch',
  ]
  assert child_ctx._invocation_context.branch == 'parent_branch'


@pytest.mark.asyncio
async def test_override_isolation_scope_used_in_node_runner():
  """NodeRunner sets isolation_scope on child context and enriches emitted events."""

  class _Node(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      assert ctx.isolation_scope == 'task:fc-999'
      yield Event(output='result')

  parent_ctx, events = _make_ctx()
  await NodeRunner(
      node=_Node(name='n'),
      parent_ctx=parent_ctx,
      override_isolation_scope='task:fc-999',
  ).run()

  assert events[0].isolation_scope == 'task:fc-999'
