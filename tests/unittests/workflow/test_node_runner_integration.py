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

"""Tests for NodeRunner ↔ node integration.

Verifies that NodeRunner correctly drives BaseNode.run(), enriches
events, flushes state/artifact deltas, and delivers events to the
session.
"""

from typing import Any
from typing import AsyncGenerator
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

# --- Test helper nodes ---


class _EchoNode(BaseNode):

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    yield node_input


class _EmptyNode(BaseNode):

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    return
    yield


class _MultiEventNode(BaseNode):

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    yield Event(author='step1')
    yield Event(author='step2')
    yield Event(author='step3')


class _InterruptNode(BaseNode):

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    yield Event(
        content=types.Content(
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name='long_tool', args={}, id='fc-1'
                    )
                )
            ]
        ),
        long_running_tool_ids={'fc-1'},
    )


class _InterruptThenMoreNode(BaseNode):

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    yield Event(
        content=types.Content(
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name='long_tool', args={}, id='fc-2'
                    )
                )
            ]
        ),
        long_running_tool_ids={'fc-2'},
    )
    yield Event(author='after_interrupt_1')
    yield Event(author='after_interrupt_2')


class _ErrorNode(BaseNode):

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    raise RuntimeError('node failure')
    yield  # pylint: disable=unreachable


class _OutputWithRouteNode(BaseNode):

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    yield Event(output='routed_output', route='next')


class _StateMutatingNode(BaseNode):

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    ctx.state['key1'] = 'value1'
    ctx.state['key2'] = 42
    yield 'done'


class _ResumeInputReadingNode(BaseNode):
  captured: list[Any] = []

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    self.captured.append(ctx.resume_inputs)
    yield 'resumed'


class _ArtifactSavingNode(BaseNode):

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    ctx.actions.artifact_delta['doc.txt'] = 1
    yield 'saved'


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


# --- Tests ---


@pytest.mark.asyncio
async def test_node_output_returned_in_result():
  """Running a node that produces output returns it in the result."""
  node = _EchoNode(name='echo')
  ctx, _ = _make_ctx()
  result = await NodeRunner(node=node, parent_ctx=ctx).run(node_input='hello')
  assert result.output == 'hello'
  assert result.interrupt_ids == set()


@pytest.mark.asyncio
async def test_no_output_returns_none():
  """Running a node that produces no output returns None."""
  node = _EmptyNode(name='empty')
  ctx, _ = _make_ctx()
  result = await NodeRunner(node=node, parent_ctx=ctx).run()
  assert result.output is None
  assert result.interrupt_ids == set()


@pytest.mark.asyncio
async def test_event_author_is_node_name():
  """Events are authored by the node's name."""
  node = _EchoNode(name='my_node')
  ctx, events = _make_ctx()
  await NodeRunner(node=node, parent_ctx=ctx).run(node_input='data')

  output_events = [e for e in events if e.output is not None]
  assert output_events[0].author == 'my_node'


@pytest.mark.asyncio
async def test_event_path_contains_node_name():
  """Event node_info.path includes the node name and execution context."""
  node = _EchoNode(name='path_test')
  ctx, events = _make_ctx(invocation_id='inv-123')
  runner = NodeRunner(node=node, parent_ctx=ctx, run_id='exec-456')
  await runner.run(node_input='data')

  output_events = [e for e in events if e.output is not None]
  event = output_events[0]
  assert event.node_info.path == 'path_test@exec-456'
  assert event.invocation_id == 'inv-123'


@pytest.mark.asyncio
async def test_interrupt_captured_in_result():
  """A node that signals an interrupt reports it in the result."""
  node = _InterruptNode(name='interrupt_node')
  ctx, _ = _make_ctx()
  result = await NodeRunner(node=node, parent_ctx=ctx).run()
  assert 'fc-1' in result.interrupt_ids


@pytest.mark.asyncio
async def test_node_continues_after_interrupt():
  """A node that interrupts can still produce more events before finishing."""
  node = _InterruptThenMoreNode(name='flag_finish')
  ctx, events = _make_ctx()
  result = await NodeRunner(node=node, parent_ctx=ctx).run()
  assert 'fc-2' in result.interrupt_ids
  assert len(events) >= 3


@pytest.mark.asyncio
async def test_state_mutations_emitted_as_delta():
  """State changes made by a node are delivered as a separate event."""
  node = _StateMutatingNode(name='state_node')
  ctx, events = _make_ctx()
  await NodeRunner(node=node, parent_ctx=ctx).run()

  all_deltas = {}
  for e in events:
    if e.actions.state_delta:
      all_deltas.update(e.actions.state_delta)
  assert all_deltas.get('key1') == 'value1'
  assert all_deltas.get('key2') == 42


@pytest.mark.asyncio
async def test_artifact_delta_emitted():
  """Artifact saves made by a node are delivered as a delta event."""
  node = _ArtifactSavingNode(name='artifact_node')
  ctx, events = _make_ctx()
  await NodeRunner(node=node, parent_ctx=ctx).run()

  artifact_deltas = {}
  for e in events:
    if e.actions.artifact_delta:
      artifact_deltas.update(e.actions.artifact_delta)
  assert 'doc.txt' in artifact_deltas


@pytest.mark.asyncio
async def test_events_enqueued_in_yield_order():
  """Multiple events from a node arrive in the order they were produced."""
  node = _MultiEventNode(name='multi')
  ctx, events = _make_ctx()
  await NodeRunner(node=node, parent_ctx=ctx).run()

  # All 3 events enqueued, authored by node name (framework overrides).
  assert len(events) == 3
  assert all(e.author == 'multi' for e in events)


@pytest.mark.asyncio
async def test_node_exception_propagates():
  """A node that raises an error surfaces it to the caller."""
  node = _ErrorNode(name='error_node')
  ctx, events = _make_ctx()
  child_ctx = await NodeRunner(node=node, parent_ctx=ctx).run()

  # Verify error is recorded on returned context
  assert child_ctx.error is not None
  assert isinstance(child_ctx.error, RuntimeError)
  assert str(child_ctx.error) == 'node failure'

  # Verify error event was enqueued
  error_events = [e for e in events if e.error_code]
  assert len(error_events) == 1
  assert error_events[0].error_code == 'RuntimeError'
  assert 'node failure' in error_events[0].error_message


@pytest.mark.asyncio
async def test_resume_inputs_available_on_context():
  """Resume inputs are accessible to the node during execution."""
  node = _ResumeInputReadingNode(name='resume_node')
  node.captured = []
  ctx, _ = _make_ctx()
  resume = {'int-1': 'user_response'}
  await NodeRunner(node=node, parent_ctx=ctx).run(resume_inputs=resume)
  assert node.captured[0] == resume


@pytest.mark.asyncio
async def test_node_path_includes_parent():
  """A child node's node_path is parent_node_path/child_name."""
  node = _EchoNode(name='child')
  ctx, events = _make_ctx(node_path='parent_path')
  runner = NodeRunner(node=node, parent_ctx=ctx)
  await runner.run(node_input='x')

  output_events = [e for e in events if e.output is not None]
  assert output_events[0].node_info.path == 'parent_path/child@1'


@pytest.mark.asyncio
async def test_run_id_generated_when_omitted():
  """Each node run gets a unique execution ID by default."""
  node = _EchoNode(name='auto_id')
  ctx, _ = _make_ctx()

  runner = NodeRunner(node=node, parent_ctx=ctx)

  assert runner.run_id
  assert isinstance(runner.run_id, str)


@pytest.mark.asyncio
async def test_explicit_run_id_used():
  """A caller-specified execution ID is used on the runner and events."""
  node = _EchoNode(name='explicit_id')
  ctx, events = _make_ctx()

  runner = NodeRunner(node=node, parent_ctx=ctx, run_id='my-exec-id')

  assert runner.run_id == 'my-exec-id'
  await runner.run(node_input='data')
  assert events[0].node_info.path == 'explicit_id@my-exec-id'


@pytest.mark.asyncio
async def test_route_captured_in_result():
  """A node's routing decision is available in the result."""
  node = _OutputWithRouteNode(name='route_node')
  ctx, _ = _make_ctx()
  result = await NodeRunner(node=node, parent_ctx=ctx).run()
  assert result.output == 'routed_output'
  assert result.route == 'next'


@pytest.mark.asyncio
async def test_preset_author_overridden_by_framework():
  """Framework always sets author — preset author is overridden."""
  node = _MultiEventNode(name='multi')
  ctx, events = _make_ctx()
  await NodeRunner(node=node, parent_ctx=ctx).run()

  # All events get node name, not the preset 'step1'/'step2'/'step3'.
  assert all(e.author == 'multi' for e in events)


class _MultiOutputNode(BaseNode):

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    yield Event(output='first')
    yield Event(output='second')


@pytest.mark.asyncio
async def test_multiple_outputs_raises():
  """A node that produces more than one output is rejected."""
  node = _MultiOutputNode(name='multi_out')
  ctx, events = _make_ctx()
  await NodeRunner(node=node, parent_ctx=ctx).run()
  error_events = [e for e in events if e.error_code]
  assert len(error_events) == 1
  assert error_events[0].error_code == 'ValueError'
  assert 'at most one output' in error_events[0].error_message


@pytest.mark.asyncio
async def test_all_events_delivered():
  """All events from a node are delivered to the session."""
  node = _EchoNode(name='enqueue_test')
  ctx, events = _make_ctx()
  await NodeRunner(node=node, parent_ctx=ctx).run(node_input='data')
  assert len(events) >= 1


# --- Delta flushing tests ---


@pytest.mark.asyncio
async def test_state_delta_bundled_with_output_event():
  """State deltas set before yield are flushed onto the output event."""

  class _Node(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      ctx.state['color'] = 'blue'
      ctx.state['count'] = 7
      yield 'result'

  ctx, events = _make_ctx()

  await NodeRunner(node=_Node(name='bundled'), parent_ctx=ctx).run()

  assert len(events) == 1
  assert events[0].output == 'result'
  assert events[0].actions.state_delta.get('color') == 'blue'
  assert events[0].actions.state_delta.get('count') == 7


@pytest.mark.asyncio
async def test_state_after_last_yield_emitted_separately():
  """State set after the last yield is emitted as a separate event."""

  class _Node(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield 'early'
      ctx.state['late_key'] = 'late_value'

  ctx, events = _make_ctx()

  await NodeRunner(node=_Node(name='late_state'), parent_ctx=ctx).run()

  assert events[0].output == 'early'
  assert events[1].actions.state_delta.get('late_key') == 'late_value'


@pytest.mark.asyncio
async def test_deltas_skip_partial_events():
  """Partial events carry no deltas — deltas flush to next non-partial."""

  class _Node(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      ctx.state['before_partial'] = True
      yield Event(
          content=types.Content(parts=[types.Part(text='streaming...')]),
          partial=True,
      )
      ctx.state['after_partial'] = True
      yield 'final'

  ctx, events = _make_ctx()

  await NodeRunner(node=_Node(name='partial_skip'), parent_ctx=ctx).run()

  assert events[0].partial is True
  assert not events[0].actions or not events[0].actions.state_delta
  assert events[1].output == 'final'
  assert events[1].actions.state_delta.get('before_partial') is True
  assert events[1].actions.state_delta.get('after_partial') is True


@pytest.mark.asyncio
async def test_artifact_and_state_bundled_together():
  """Both state and artifact deltas are flushed onto the same event."""

  class _Node(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      ctx.state['s1'] = 'v1'
      ctx.actions.artifact_delta['file.txt'] = 1
      yield 'done'

  ctx, events = _make_ctx()

  await NodeRunner(node=_Node(name='both_deltas'), parent_ctx=ctx).run()

  assert len(events) == 1
  assert events[0].output == 'done'
  assert events[0].actions.state_delta.get('s1') == 'v1'
  assert events[0].actions.artifact_delta.get('file.txt') == 1


@pytest.mark.asyncio
async def test_node_input_schema_validation():
  """NodeRunner fails if node input does not match input_schema."""
  from pydantic import BaseModel
  from pydantic import ValidationError

  class _MyInput(BaseModel):
    name: str
    age: int

  node = _EchoNode(name='schema_node', input_schema=_MyInput)
  ctx, _ = _make_ctx()

  # Valid input (dict that matches schema)
  result = await NodeRunner(node=node, parent_ctx=ctx).run(
      node_input={'name': 'Alice', 'age': 30}
  )

  # _validate_schema converts model instances to dicts!
  assert isinstance(result.output, dict)
  assert result.output['name'] == 'Alice'
  assert result.output['age'] == 30

  # Invalid input (missing field)
  result = await NodeRunner(node=node, parent_ctx=ctx).run(
      node_input={'name': 'Alice'}
  )

  assert result.error is not None
  assert isinstance(result.error, ValidationError)
