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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.context import Context
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.workflow._base_node import BaseNode
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
