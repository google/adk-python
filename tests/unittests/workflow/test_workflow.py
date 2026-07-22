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

"""Tests for the new Workflow(BaseNode) implementation.

Migrated from test_workflow_agent.py — each test validates the same
workflow behavior through Runner(node=...) instead of agent.run_async().
"""

from collections import Counter
from typing import Any
from typing import AsyncGenerator
import uuid

from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow._base_node import BaseNode
from google.adk.workflow._base_node import START
from google.adk.workflow._join_node import JoinNode
from google.adk.workflow._workflow import Workflow
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
from google.genai import types
from pydantic import ConfigDict
from pydantic import Field
import pytest

# ---------------------------------------------------------------------------
# Shared helper nodes (used by multiple tests)
# ---------------------------------------------------------------------------


def _make_function_call_interrupt(fc_id: str, name: str = 'approve') -> Event:
  """Helper to create a raw function call interruption event."""
  return Event(
      content=types.Content(
          parts=[
              types.Part(
                  function_call=types.FunctionCall(name=name, args={}, id=fc_id)
              )
          ]
      ),
      long_running_tool_ids={fc_id},
  )


class _OutputNode(BaseNode):
  """Yields a fixed output value."""

  value: Any = None

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    yield self.value


class _PassthroughNode(BaseNode):
  """Passes node_input through as output."""

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    yield node_input


class _RouteNode(BaseNode):
  """Yields output and sets a route."""

  value: Any = None
  route_value: Any = None

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    yield Event(output=self.value, route=self.route_value)


class _InputCapturingNode(BaseNode):
  """Captures node_input for later assertion."""

  model_config = ConfigDict(arbitrary_types_allowed=True)
  received_inputs: list[Any] = Field(default_factory=list)

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    self.received_inputs.append(node_input)
    yield {'received': node_input}


class _IntermediateContentNode(BaseNode):
  """Yields intermediate content events before output."""

  contents: list[types.Content] = Field(default_factory=list)
  value: Any = None

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    for content in self.contents:
      yield Event(content=content)
    if self.value is not None:
      yield self.value


class _BytesOutputNode(BaseNode):
  """Yields bytes content or raw bytes."""

  raw_bytes: bool = False

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    data = b'\x89PNG\r\n\x1a\n'
    if self.raw_bytes:
      yield data
    else:
      yield Event(
          content=types.Content(
              parts=[types.Part.from_bytes(data=data, mime_type='image/png')]
          ),
          output='bytes_sent',
      )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tests — 1:1 mapping from test_workflow_agent.py
# ---------------------------------------------------------------------------


# 1. test_run_async → sequential A→B
@pytest.mark.asyncio
async def test_sequential_two_nodes():
  """Sequential A->B produces both outputs in order.

  Maps to: test_run_async in test_workflow_agent.py.
  """
  a = _OutputNode(name='NodeA', value='Hello')
  b = _OutputNode(name='NodeB', value='World')
  wf = Workflow(name='wf', edges=[(START, a, b)])

  events, _, _ = await _run_workflow(wf)

  by_node = _output_by_node(events)
  assert ('NodeA', 'Hello') in by_node
  assert ('NodeB', 'World') in by_node
  assert by_node.index(('NodeA', 'Hello')) < by_node.index(('NodeB', 'World'))


# 2. test_run_async_with_intermediate_content
@pytest.mark.asyncio
async def test_intermediate_content_before_output():
  """Node yields content events before output.

  Maps to: test_run_async_with_intermediate_content in test_workflow_agent.py.
  """
  a = _IntermediateContentNode(
      name='NodeA',
      contents=[
          types.Content(parts=[types.Part(text='msg1')]),
          types.Content(parts=[types.Part(text='msg2')]),
      ],
      value='A output',
  )
  wf = Workflow(name='wf', edges=[(START, a)])

  events, _, _ = await _run_workflow(wf)

  texts = [
      p.text
      for e in events
      if e.content and e.content.parts
      for p in e.content.parts
      if p.text
  ]
  assert texts.index('msg1') < texts.index('msg2')  # also asserts presence
  assert 'A output' in _outputs(events)


# 3. test_run_async_with_loop_and_break
@pytest.mark.asyncio
async def test_loop_with_conditional_break():
  """Loop iterates 3 times: A,Check repeats, then exits to B.

  Maps to: test_run_async_with_loop_and_break in test_workflow_agent.py.
  """

  class _IncrementingNode(BaseNode):

    model_config = ConfigDict(arbitrary_types_allowed=True)
    message: str = ''
    _tracker_ref: list = []

    def __init__(self, *, name: str, message: str, tracker: dict):
      super().__init__(name=name)
      object.__setattr__(self, 'message', message)
      object.__setattr__(self, '_tracker_ref', [tracker])

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      tracker = self._tracker_ref[0]
      count = tracker.get('iteration_count', 0) + 1
      tracker['iteration_count'] = count
      yield Event(
          output=self.message,
          route='continue_loop' if count < 3 else 'exit_loop',
      )

  tracker = {'iteration_count': 0}
  node_a = _OutputNode(name='NodeA', value='Looping')
  check = _IncrementingNode(
      name='CheckNode', message='Checking', tracker=tracker
  )
  node_b = _OutputNode(name='NodeB', value='Finished')
  wf = Workflow(
      name='wf',
      edges=[
          (START, node_a, check),
          (check, {'continue_loop': node_a, 'exit_loop': node_b}),
      ],
  )

  events, _, _ = await _run_workflow(wf)

  assert tracker['iteration_count'] == 3
  by_node = _output_by_node(events)
  node_names = [name for name, _ in by_node]
  # 3 iterations of (NodeA, CheckNode) then NodeB
  assert node_names.count('NodeA') == 3
  assert node_names.count('CheckNode') == 3
  assert 'NodeB' in node_names


# 4. test_resume_behavior
@pytest.mark.asyncio
async def test_resume_after_interrupt():
  """Workflow resumes from HITL interrupt and continues execution.

  Maps to: test_resume_behavior in test_workflow_agent.py.
  Old test used crash/checkpoint resume. New test uses HITL interrupt/FR.
  """

  class _InterruptOnce(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      fc_id = ctx.state.get('_fc_id')
      if fc_id and ctx.resume_inputs and fc_id in ctx.resume_inputs:
        ctx.state['_fc_id'] = None
        yield 'resumed'
        return

      fc_id = f'fc-{uuid.uuid4().hex[:8]}'
      ctx.state['_fc_id'] = fc_id
      yield Event(
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='approve', args={}, id=fc_id
                      )
                  )
              ]
          ),
          long_running_tool_ids={fc_id},
      )

  a = _OutputNode(name='NodeA', value='A')
  b = _InterruptOnce(name='NodeB')
  c = _OutputNode(name='NodeC', value='C')
  wf = Workflow(name='wf', edges=[(START, a, b, c)])
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: A completes, B interrupts
  msg1 = types.Content(parts=[types.Part(text='go')], role='user')
  events1: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg1
  ):
    events1.append(event)

  # A completed, B interrupted
  assert 'A' in [e.output for e in events1 if e.output is not None]
  assert any(e.long_running_tool_ids for e in events1)
  fc_id = None
  for e in events1:
    if e.long_running_tool_ids:
      fc_id = list(e.long_running_tool_ids)[0]

  # Run 2: resume B, then C runs
  msg2 = types.Content(
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name='approve', id=fc_id, response={'ok': True}
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

  outputs = [e.output for e in events2 if e.output is not None]
  assert 'resumed' in outputs
  assert 'C' in outputs


@pytest.mark.asyncio
async def test_resume_with_schema_validation_failure():
  """Workflow raises ValueError when resume response fails validation."""

  class _InterruptWithSchema(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      fc_id = ctx.state.get('_fc_id')
      if fc_id and ctx.resume_inputs and fc_id in ctx.resume_inputs:
        ctx.state['_fc_id'] = None
        yield ctx.resume_inputs[fc_id]
        return

      fc_id = f'fc-{uuid.uuid4().hex[:8]}'
      ctx.state['_fc_id'] = fc_id
      from google.adk.events.request_input import RequestInput

      yield RequestInput(
          interrupt_id=fc_id,
          prompt='Enter an integer',
          response_schema={'type': 'integer'},
      )

  a = _InterruptWithSchema(name='NodeA')
  wf = Workflow(name='wf', edges=[(START, a)])
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  msg1 = types.Content(parts=[types.Part(text='go')], role='user')
  events1: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg1
  ):
    events1.append(event)

  fc_id = None
  for e in events1:
    if e.long_running_tool_ids:
      fc_id = list(e.long_running_tool_ids)[0]

  msg2 = types.Content(
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name='adk_request_input', id=fc_id, response={'result': 'abc'}
              )
          )
      ],
      role='user',
  )

  with pytest.raises(ValueError, match='Validation failed for interrupt'):
    async for _ in runner.run_async(
        user_id='u', session_id=session.id, new_message=msg2
    ):
      pass


@pytest.mark.asyncio
async def test_resume_with_schema_validation_failure_nested():
  """Workflow raises ValueError when resume response fails validation in nested workflow."""

  class _InterruptWithSchema(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      fc_id = ctx.state.get('_fc_id')
      if fc_id and ctx.resume_inputs and fc_id in ctx.resume_inputs:
        ctx.state['_fc_id'] = None
        yield ctx.resume_inputs[fc_id]
        return

      fc_id = f'fc-{uuid.uuid4().hex[:8]}'
      ctx.state['_fc_id'] = fc_id
      from google.adk.events.request_input import RequestInput

      yield RequestInput(
          interrupt_id=fc_id,
          prompt='Enter an integer',
          response_schema={'type': 'integer'},
      )

  inner_wf = Workflow(
      name='inner_wf', edges=[(START, _InterruptWithSchema(name='NodeA'))]
  )
  outer_wf = Workflow(name='outer_wf', edges=[(START, inner_wf)])

  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=outer_wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  msg1 = types.Content(parts=[types.Part(text='go')], role='user')
  events1: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg1
  ):
    events1.append(event)

  fc_id = None
  for e in events1:
    if e.long_running_tool_ids:
      fc_id = list(e.long_running_tool_ids)[0]

  msg2 = types.Content(
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name='adk_request_input', id=fc_id, response={'result': 'abc'}
              )
          )
      ],
      role='user',
  )

  with pytest.raises(ValueError, match='Validation failed for interrupt'):
    async for _ in runner.run_async(
        user_id='u', session_id=session.id, new_message=msg2
    ):
      pass


@pytest.mark.asyncio
async def test_resume_with_schema_validation_failure_no_rerun():
  """Workflow raises ValueError when resume response fails validation even if rerun_on_resume is False."""

  class _InterruptWithSchema(BaseNode):
    rerun_on_resume: bool = False

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      fc_id = ctx.state.get('_fc_id')
      if fc_id and ctx.resume_inputs and fc_id in ctx.resume_inputs:
        ctx.state['_fc_id'] = None
        yield ctx.resume_inputs[fc_id]
        return

      fc_id = f'fc-{uuid.uuid4().hex[:8]}'
      ctx.state['_fc_id'] = fc_id
      from google.adk.events.request_input import RequestInput

      yield RequestInput(
          interrupt_id=fc_id,
          prompt='Enter an integer',
          response_schema={'type': 'integer'},
      )

  a = _InterruptWithSchema(name='NodeA')
  wf = Workflow(name='wf', edges=[(START, a)])
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  msg1 = types.Content(parts=[types.Part(text='go')], role='user')
  events1: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg1
  ):
    events1.append(event)

  fc_id = None
  for e in events1:
    if e.long_running_tool_ids:
      fc_id = list(e.long_running_tool_ids)[0]

  msg2 = types.Content(
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name='adk_request_input', id=fc_id, response={'result': 'abc'}
              )
          )
      ],
      role='user',
  )

  with pytest.raises(ValueError, match='Validation failed for interrupt'):
    async for _ in runner.run_async(
        user_id='u', session_id=session.id, new_message=msg2
    ):
      pass


# 5. test_agent_state_event_recorded
@pytest.mark.asyncio
async def test_internal_interrupt_event_not_persisted():
  """Workflow interrupt event is _adk_internal — not in session.

  Maps to: test_agent_state_event_recorded in test_workflow_agent.py.
  Old test verified checkpoint events. New test verifies the workflow's
  interrupt event is NOT persisted (child's event is sufficient).
  """

  class _InterruptNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
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

  wf = Workflow(name='wf', edges=[(START, _InterruptNode(name='ask'))])
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  msg = types.Content(parts=[types.Part(text='go')], role='user')
  events: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg
  ):
    events.append(event)

  # Caller should see the child's interrupt event
  assert any(e.long_running_tool_ids for e in events)

  # Session should NOT have the workflow-level interrupt event
  updated = await ss.get_session(
      app_name='test', user_id='u', session_id=session.id
  )
  wf_interrupt_events = [
      e
      for e in updated.events
      if e.long_running_tool_ids and e.node_info.path == 'wf@1'
  ]
  assert wf_interrupt_events == []
  # But child's interrupt event SHOULD be in session
  child_interrupt_events = [
      e
      for e in updated.events
      if e.long_running_tool_ids
      and e.node_info.path
      and e.node_info.path.endswith('ask@1')
  ]
  assert len(child_interrupt_events) == 1


# 6. test_run_async_with_implicit_graph
@pytest.mark.asyncio
async def test_edge_tuple_syntax():
  """Edges defined via tuples work.

  Maps to: test_run_async_with_implicit_graph in test_workflow_agent.py.
  """
  a = _OutputNode(name='NodeA', value='Hello')
  b = _OutputNode(name='NodeB', value='World')
  wf = Workflow(
      name='wf',
      edges=[
          (START, a),
          (a, b),
      ],
  )

  events, _, _ = await _run_workflow(wf)

  outputs = _outputs(events)
  assert outputs.index('Hello') < outputs.index(
      'World'
  )  # also asserts presence


# 7. test_run_async_with_string_start
@pytest.mark.asyncio
async def test_string_start_in_edges():
  """'START' string works as edge source.

  Maps to: test_run_async_with_string_start in test_workflow_agent.py.
  """
  a = _OutputNode(name='NodeA', value='Hello')
  wf = Workflow(name='wf', edges=[('START', a)])

  events, _, _ = await _run_workflow(wf)

  assert 'Hello' in _outputs(events)


# 8. test_run_async_with_implicit_graph_with_edge_combinations
@pytest.mark.asyncio
async def test_mixed_edge_and_tuple_syntax():
  """Mix of Edge objects and tuple syntax in a 3-node chain.

  Maps to: test_run_async_with_implicit_graph_with_edge_combinations
  in test_workflow_agent.py.
  """
  from google.adk.workflow._graph import Edge as GraphEdge

  a = _OutputNode(name='NodeA', value='A')
  b = _OutputNode(name='NodeB', value='B')
  c = _OutputNode(name='NodeC', value='C')
  wf = Workflow(
      name='wf',
      edges=[
          (START, a),
          GraphEdge(from_node=a, to_node=b),
          (b, c),
      ],
  )

  events, _, _ = await _run_workflow(wf)

  by_node = _output_by_node(events)
  assert by_node.index(('NodeA', 'A')) < by_node.index(
      ('NodeB', 'B')
  )  # also asserts presence
  assert by_node.index(('NodeB', 'B')) < by_node.index(('NodeC', 'C'))


# 9. test_run_async_with_update_state_event
@pytest.mark.asyncio
async def test_state_update_via_event_persisted():
  """State updates via Event(state=...) are persisted to session.

  Maps to: test_run_async_with_update_state_event in test_workflow_agent.py.
  """

  class _Node(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield Event(state={'key1': 'value1'})

  wf = Workflow(name='wf', edges=[(START, _Node(name='NodeA'))])

  events, ss, session = await _run_workflow(wf)
  updated = await ss.get_session(
      app_name='test', user_id='u', session_id=session.id
  )

  assert updated.state.get('key1') == 'value1'


# 11. test_run_async_with_raw_output_node
@pytest.mark.asyncio
async def test_raw_output_auto_wrapped():
  """Node yields raw value, auto-wrapped to Event(output=...).

  Maps to: test_run_async_with_raw_output_node in test_workflow_agent.py.
  """
  a = _OutputNode(name='NodeA', value='raw_data')
  wf = Workflow(name='wf', edges=[(START, a)])

  events, _, _ = await _run_workflow(wf)

  assert 'raw_data' in _outputs(events)


# 12. test_node_output_event_with_content_data
@pytest.mark.asyncio
async def test_content_return_produces_content_event():
  """FunctionNode returning Content yields content event, not output.

  Maps to: test_node_output_event_with_content_data
  in test_workflow_agent.py.
  """

  def content_fn() -> types.Content:
    return types.Content(parts=[types.Part(text='hello')])

  wf = Workflow(name='wf', edges=[(START, content_fn)])

  events, _, _ = await _run_workflow(wf)

  fn_events = [
      e for e in events if e.node_info.path and 'content_fn' in e.node_info.path
  ]
  assert len(fn_events) == 1
  assert fn_events[0].output is None
  assert fn_events[0].content == types.Content(parts=[types.Part(text='hello')])


# 13. test_input_propagation_linear
@pytest.mark.asyncio
async def test_input_propagation_linear():
  """Output of A becomes input to B.

  Maps to: test_input_propagation_linear in test_workflow_agent.py.
  """
  a = _OutputNode(name='NodeA', value='from_a')
  b = _InputCapturingNode(name='NodeB')
  wf = Workflow(name='wf', edges=[(START, a, b)])

  events, _, _ = await _run_workflow(wf)

  assert b.received_inputs == ['from_a']


# 14. test_input_propagation_fan_in_sequential
@pytest.mark.asyncio
async def test_input_propagation_fan_in():
  """Fan-in with asymmetric branches: C receives from A and B3.

  Maps to: test_input_propagation_fan_in_sequential
  in test_workflow_agent.py.
  """
  a = _OutputNode(name='NodeA', value='from_a')
  b = _OutputNode(name='NodeB', value='from_b')
  b2 = _PassthroughNode(name='NodeB2')
  b3 = _PassthroughNode(name='NodeB3')
  c = _InputCapturingNode(name='NodeC')
  wf = Workflow(
      name='wf',
      edges=[
          (START, a),
          (START, b, b2, b3),
          (a, c),
          (b3, c),
      ],
  )

  events, _, _ = await _run_workflow(wf)

  assert Counter(c.received_inputs) == Counter(['from_a', 'from_b'])


# 15. test_start_node_receives_user_content
@pytest.mark.asyncio
async def test_start_node_receives_user_content():
  """First node receives user message as input.

  Maps to: test_start_node_receives_user_content
  in test_workflow_agent.py.
  """
  a = _InputCapturingNode(name='NodeA')
  wf = Workflow(name='wf', edges=[(START, a)])

  events, _, _ = await _run_workflow(wf, message='hello')

  assert len(a.received_inputs) == 1
  assert a.received_inputs[0].parts[0].text == 'hello'


# 18. test_wait_for_output_suppresses_trigger


@pytest.mark.asyncio
async def test_wait_for_output_suppresses_downstream():
  """wait_for_output=True with no output prevents downstream from running.

  Maps to: test_wait_for_output_suppresses_trigger
  in test_workflow_agent.py.
  """

  class _NoOutputNode(BaseNode):
    wait_for_output: bool = True

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield Event(state={'seen': True})

  blocker = _NoOutputNode(name='blocker')
  downstream = _OutputNode(name='downstream', value='should_not_run')
  wf = Workflow(
      name='wf',
      edges=[
          (START, blocker),
          (blocker, downstream),
      ],
  )

  events, _, _ = await _run_workflow(wf)

  assert 'should_not_run' not in _outputs(events)


# 19. test_wait_for_output_retrigger_then_complete
@pytest.mark.asyncio
async def test_wait_for_output_completes_after_all():
  """Gate with wait_for_output opens after 2 triggers, fires downstream.

  Maps to: test_wait_for_output_retrigger_then_complete
  in test_workflow_agent.py.
  """

  class _GateNode(BaseNode):
    triggers_needed: int = 2
    wait_for_output: bool = True

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      count = (ctx.state.get(f'{self.name}_count') or 0) + 1
      if count >= self.triggers_needed:
        yield Event(
            output='gate_open',
            state={f'{self.name}_count': None},
        )
      else:
        yield Event(state={f'{self.name}_count': count})

  gate = _GateNode(name='Gate', triggers_needed=2)
  a = _OutputNode(name='NodeA', value='A')
  b = _OutputNode(name='NodeB', value='B')
  downstream = _OutputNode(name='Downstream', value='done')
  wf = Workflow(
      name='wf',
      edges=[
          (START, a, gate, downstream),
          (START, b, gate),
      ],
  )

  events, _, _ = await _run_workflow(wf)
  by_node = _output_by_node(events)

  # A and B run first (parallel), then Gate opens, then Downstream
  assert len(by_node) == 4
  first_two = {by_node[0][0], by_node[1][0]}
  assert first_two == {'NodeA', 'NodeB'}
  assert by_node[2] == ('Gate', 'gate_open')
  assert by_node[3] == ('Downstream', 'done')


# 25. test_run_async_with_implicit_graph_chain
@pytest.mark.asyncio
async def test_chain_syntax():
  """(START, a, b, c) creates sequential chain producing all outputs.

  Maps to: test_run_async_with_implicit_graph_chain
  in test_workflow_agent.py.
  """
  a = _OutputNode(name='a', value='A')
  b = _OutputNode(name='b', value='B')
  c = _OutputNode(name='c', value='C')
  wf = Workflow(name='wf', edges=[(START, a, b, c)])

  events, _, _ = await _run_workflow(wf)
  by_node = _output_by_node(events)

  assert [name for name, _ in by_node] == ['a', 'b', 'c']
  assert [val for _, val in by_node] == ['A', 'B', 'C']


# 26. test_run_async_with_implicit_graph_fan_out
@pytest.mark.asyncio
async def test_fan_out():
  """Fan-out to multiple terminals raises ValueError.

  Maps to: test_run_async_with_implicit_graph_fan_out
  in test_workflow_agent.py.
  """
  a = _OutputNode(name='a', value='A')
  b = _InputCapturingNode(name='b')
  c = _InputCapturingNode(name='c')
  wf = Workflow(
      name='wf',
      edges=[
          (START, a),
          (a, (b, c)),
      ],
  )

  with pytest.raises(ValueError, match='multiple terminal nodes'):
    await _run_workflow(wf)


# 27. test_run_async_with_implicit_graph_fan_in
@pytest.mark.asyncio
async def test_fan_in():
  """((a, b), c) — c triggered by both a and b.

  Maps to: test_run_async_with_implicit_graph_fan_in
  in test_workflow_agent.py.
  """
  a = _OutputNode(name='a', value='A')
  b = _OutputNode(name='b', value='B')
  c = _InputCapturingNode(name='c')
  wf = Workflow(
      name='wf',
      edges=[
          (START, (a, b)),
          ((a, b), c),
      ],
  )

  events, _, _ = await _run_workflow(wf)

  assert Counter(c.received_inputs) == Counter(['A', 'B'])


# 28. test_run_async_with_implicit_graph_fan_out_fan_in
@pytest.mark.asyncio
async def test_fan_out_fan_in():
  """S fans out to (A, B), both feed into C.

  Maps to: test_run_async_with_implicit_graph_fan_out_fan_in
  in test_workflow_agent.py.
  """
  s = _OutputNode(name='s', value='S')
  a = _PassthroughNode(name='a')
  b = _PassthroughNode(name='b')
  c = _InputCapturingNode(name='c')
  wf = Workflow(
      name='wf',
      edges=[
          (START, s),
          (s, (a, b), c),
      ],
  )

  events, _, _ = await _run_workflow(wf)

  # Both A and B receive S's output, C receives from both
  assert Counter(c.received_inputs) == Counter(['S', 'S'])


# 29. test_run_async_parallel_nodes_interleaved_events
@pytest.mark.asyncio
async def test_parallel_events_interleaved():
  """Parallel terminal nodes both producing output raises ValueError.

  Maps to: test_run_async_parallel_nodes_interleaved_events
  in test_workflow_agent.py.
  """
  a = _OutputNode(name='a', value='A')
  b = _OutputNode(name='b', value='B')
  wf = Workflow(
      name='wf',
      edges=[
          (START, a),
          (START, b),
      ],
  )

  with pytest.raises(ValueError, match='multiple terminal nodes'):
    await _run_workflow(wf)


# 30. test_buffers_events_from_parallel_nodes
@pytest.mark.asyncio
async def test_parallel_events_all_delivered():
  """All events from parallel nodes fan-in to capture node.

  Maps to: test_buffers_events_from_parallel_nodes
  in test_workflow_agent.py.
  """
  a = _OutputNode(name='a', value='A')
  b = _OutputNode(name='b', value='B')
  capture = _InputCapturingNode(name='capture')
  wf = Workflow(
      name='wf',
      edges=[
          (START, a),
          (START, b),
          (a, capture),
          (b, capture),
      ],
  )

  events, _, _ = await _run_workflow(wf)

  assert Counter(capture.received_inputs) == Counter(['A', 'B'])


# 31. test_run_id_uniqueness
@pytest.mark.asyncio
async def test_run_id_unique_per_node():
  """Each node run gets a unique run_id.

  Maps to: test_run_id_uniqueness in test_workflow_agent.py.
  """
  a = _OutputNode(name='a', value='A')
  b = _OutputNode(name='b', value='B')
  wf = Workflow(name='wf', edges=[(START, a, b)])

  events, _, _ = await _run_workflow(wf)
  paths = [e.node_info.path for e in events if e.node_info.path]

  assert len(paths) >= 2
  # paths are unique per node run (because they contain @run_id)
  assert len(set(paths)) >= 2


# 32. test_run_id_uniqueness_nested
@pytest.mark.asyncio
async def test_run_id_unique_nested():
  """Nested workflow nodes also get unique IDs.

  Maps to: test_run_id_uniqueness_nested
  in test_workflow_agent.py.
  """
  inner_a = _OutputNode(name='inner_a', value='IA')
  inner = Workflow(name='inner', edges=[(START, inner_a)])
  outer_a = _OutputNode(name='outer_a', value='OA')
  wf = Workflow(name='wf', edges=[(START, outer_a, inner)])

  events, _, _ = await _run_workflow(wf)
  paths = [e.node_info.path for e in events if e.node_info.path]

  # paths are unique per node run (because they contain @run_id)
  assert len(set(paths)) >= 2


@pytest.mark.asyncio
async def test_run_id_sequential_in_loop():
  """Looping node gets sequential run_ids: 1, 2, 3."""
  tracker = {'count': 0}

  class _LoopNode(BaseNode):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      tracker['count'] += 1
      yield Event(
          output=f'iter-{tracker["count"]}',
          route='again' if tracker['count'] < 3 else 'done',
      )

  from google.adk.workflow._graph import Edge as GraphEdge

  loop = _LoopNode(name='loop')
  wf = Workflow(
      name='wf',
      edges=[
          (START, loop),
          GraphEdge(from_node=loop, to_node=loop, route='again'),
      ],
  )

  events, _, _ = await _run_workflow(wf)

  loop_run_ids = [
      e.node_info.path.split('@')[-1]
      for e in events
      if e.node_info.path and 'loop@' in e.node_info.path
  ]
  assert loop_run_ids == ['1', '2', '3']


# 33. test_resume_with_manual_state_verifies_input_persistence
@pytest.mark.asyncio
async def test_resume_downstream_receives_output():
  """After resume, downstream node receives the resumed node's output.

  Maps to: test_resume_with_manual_state_verifies_input_persistence
  in test_workflow_agent.py. Old test verified input from checkpoint.
  New test verifies output flows to downstream after HITL resume.
  """

  class _InterruptNode(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:

      fc_id = ctx.state.get('_fc_id')
      if fc_id and ctx.resume_inputs and fc_id in ctx.resume_inputs:
        ctx.state['_fc_id'] = None
        yield f'response:{ctx.resume_inputs[fc_id]["value"]}'
        return
      fc_id = f'fc-{uuid.uuid4().hex[:8]}'
      ctx.state['_fc_id'] = fc_id
      yield Event(
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='get', args={}, id=fc_id
                      )
                  )
              ]
          ),
          long_running_tool_ids={fc_id},
      )

  a = _OutputNode(name='NodeA', value='from_a')
  b = _InterruptNode(name='NodeB')
  c = _InputCapturingNode(name='NodeC')
  wf = Workflow(name='wf', edges=[(START, a, b, c)])
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: A completes, B interrupts
  msg1 = types.Content(parts=[types.Part(text='go')], role='user')
  events1: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg1
  ):
    events1.append(event)

  # A completed, B interrupted
  assert 'from_a' in [e.output for e in events1 if e.output is not None]
  fc_id = None
  for e in events1:
    if e.long_running_tool_ids:
      fc_id = list(e.long_running_tool_ids)[0]
  assert fc_id

  # Run 2: resume B with value, C receives B's output
  msg2 = types.Content(
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name='get', id=fc_id, response={'value': 42}
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

  # NodeC should have captured NodeB's resume output
  assert c.received_inputs == ['response:42']


# 34. test_run_async_with_multiple_node_outputs_fails
@pytest.mark.asyncio
async def test_multiple_outputs_rejected():
  """Node yielding two outputs raises error.

  Maps to: test_run_async_with_multiple_node_outputs_fails
  in test_workflow_agent.py.

  NodeRunner raises ValueError but Runner swallows it in the
  background task. This test is xfail until Runner error propagation
  is implemented.
  """

  class _Node(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield Event(output='first')
      yield Event(output='second')

  wf = Workflow(name='wf', edges=[(START, _Node(name='a'))])

  with pytest.raises(ValueError, match='at most one output'):
    await _run_workflow(wf)


# 38. test_run_async_streaming_behavior
@pytest.mark.asyncio
async def test_streaming_partial_events():
  """Partial events are streamed before final output.

  Maps to: test_run_async_streaming_behavior
  in test_workflow_agent.py.
  """

  class _Node(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield Event(
          content=types.Content(parts=[types.Part(text='partial1')]),
          partial=True,
      )
      yield Event(
          content=types.Content(parts=[types.Part(text='partial2')]),
          partial=True,
      )
      yield Event(output='final')

  wf = Workflow(name='wf', edges=[(START, _Node(name='a'))])

  events, _, _ = await _run_workflow(wf)
  partials = [e for e in events if e.partial]

  assert len(partials) == 2
  assert 'final' in _outputs(events)


# 39. test_workflow_agent_unsupported_base_agent_fields
# SKIP — BaseAgent-specific, not applicable to Workflow(BaseNode).


# 40. test_node_path_generation
@pytest.mark.asyncio
async def test_node_path_correct():
  """Events have correct node_info.path.

  Maps to: test_node_path_generation in test_workflow_agent.py.
  """
  a = _OutputNode(name='NodeA', value='A')
  b = _OutputNode(name='NodeB', value='B')
  wf = Workflow(name='wf', edges=[(START, a, b)])

  events, _, _ = await _run_workflow(wf)
  paths = [e.node_info.path for e in events if e.node_info.path]

  assert any(p.endswith('NodeA@1') for p in paths)
  assert any(p.endswith('NodeB@1') for p in paths)
  assert all('None' not in p for p in paths)


# --- Additional: function node auto-wrap ---


@pytest.mark.asyncio
async def test_function_node_auto_wrap():
  """Callable in edge is auto-wrapped to FunctionNode."""

  def greet(node_input):
    return 'hi'

  wf = Workflow(name='wf', edges=[(START, greet)])

  events, _, _ = await _run_workflow(wf)

  assert 'hi' in _outputs(events)


# --- Additional: empty workflow ---


@pytest.mark.asyncio
async def test_empty_workflow():
  """Workflow with no edges produces no output."""
  wf = Workflow(name='wf', edges=[])

  events, _, _ = await _run_workflow(wf)

  assert _outputs(events) == []


# ---------------------------------------------------------------------------
# use_as_output=True (dynamic output delegation)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_use_as_output_function_to_function():
  """Node A delegates output to dynamic child B via use_as_output=True.

  B's output event appears; A's duplicate is suppressed.
  """
  from google.adk.workflow._function_node import FunctionNode

  def func_b() -> str:
    return 'from_b'

  async def func_a(ctx: Context) -> str:
    return await ctx.run_node(func_b, use_as_output=True)

  node_a = FunctionNode(func=func_a, rerun_on_resume=True)

  wf = Workflow(name='wf', edges=[(START, node_a)])
  events, _, _ = await _run_workflow(wf)

  by_node = _output_by_node(events)
  # func_b emits output; func_a's duplicate is suppressed.
  assert ('func_b', 'from_b') in by_node
  assert not any(name == 'func_a' for name, _ in by_node)

  # output_for includes child, parent, and workflow paths.
  # func_a is terminal, so its output also represents the workflow's.
  output_event = [
      e for e in events if e.node_info.name.split('@')[0] == 'func_b'
  ][0]
  assert output_event.node_info.output_for == [
      'wf@1/func_a@1/func_b@1',
      'wf@1/func_a@1',
      'wf@1',
  ]


@pytest.mark.asyncio
async def test_use_as_output_function_to_workflow():
  """Node A delegates output to a nested Workflow via use_as_output=True.

  The inner Workflow's terminal node output is used as A's output.
  """
  from google.adk.workflow._function_node import FunctionNode

  def step_1() -> str:
    return 'step_1_done'

  def step_2(node_input: str) -> str:
    return f'final:{node_input}'

  inner_wf = Workflow(
      name='inner_wf',
      edges=[(START, step_1, step_2)],
  )

  async def func_a(ctx: Context):
    return await ctx.run_node(inner_wf, use_as_output=True)

  node_a = FunctionNode(func=func_a, rerun_on_resume=True)

  wf = Workflow(name='wf', edges=[(START, node_a)])
  events, _, _ = await _run_workflow(wf)

  by_node = _output_by_node(events)
  # step_2 is the terminal; func_a's duplicate is suppressed.
  assert ('step_2', 'final:step_1_done') in by_node
  assert not any(name == 'func_a' for name, _ in by_node)


@pytest.mark.asyncio
async def test_use_as_output_custom_node():
  """Custom BaseNode delegates output via use_as_output."""
  from google.adk.workflow._function_node import FunctionNode

  def func_b() -> str:
    return 'delegated_output'

  class _Delegator(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      await ctx.run_node(func_b, use_as_output=True)
      return
      yield

  wf = Workflow(name='wf', edges=[(START, _Delegator(name='delegator'))])
  events, _, _ = await _run_workflow(wf)

  by_node = _output_by_node(events)
  assert ('func_b', 'delegated_output') in by_node
  assert not any(name == 'delegator' for name, _ in by_node)


@pytest.mark.asyncio
async def test_use_as_output_nested_delegation():
  """Chained delegation: A → B → C, all use_as_output=True.

  Only C's output event should appear; A and B are suppressed.
  """
  from google.adk.workflow._function_node import FunctionNode

  def func_c() -> str:
    return 'from_c'

  node_c = FunctionNode(func=func_c)

  async def func_b(ctx: Context) -> str:
    return await ctx.run_node(node_c, use_as_output=True)

  node_b = FunctionNode(func=func_b, rerun_on_resume=True)

  async def func_a(ctx: Context) -> str:
    return await ctx.run_node(node_b, use_as_output=True)

  node_a = FunctionNode(func=func_a, rerun_on_resume=True)

  wf = Workflow(name='wf', edges=[(START, node_a)])
  events, _, _ = await _run_workflow(wf)

  by_node = _output_by_node(events)
  assert ('func_c', 'from_c') in by_node
  assert not any(name in ('func_a', 'func_b') for name, _ in by_node)

  # output_for includes full ancestor chain plus workflow path.
  # func_a is terminal, so the chain extends to the workflow.
  output_event = [
      e for e in events if e.node_info.name.split('@')[0] == 'func_c'
  ][0]
  assert output_event.node_info.output_for == [
      'wf@1/func_a@1/func_b@1/func_c@1',
      'wf@1/func_a@1/func_b@1',
      'wf@1/func_a@1',
      'wf@1',
  ]


@pytest.mark.asyncio
async def test_use_as_output_with_downstream():
  """Delegated output flows to downstream node via graph edge.

  A delegates to B via use_as_output. downstream is after A and
  receives B's output as node_input.
  """
  from google.adk.workflow._function_node import FunctionNode

  def func_b() -> str:
    return 'from_b'

  async def func_a(ctx: Context) -> str:
    return await ctx.run_node(func_b, use_as_output=True)

  node_a = FunctionNode(func=func_a, rerun_on_resume=True)

  downstream = _InputCapturingNode(name='downstream')

  wf = Workflow(name='wf', edges=[(START, node_a, downstream)])
  events, _, _ = await _run_workflow(wf)

  assert downstream.received_inputs == ['from_b']


@pytest.mark.asyncio
async def test_use_as_output_duplicate_raises():
  """Calling use_as_output=True twice in the same node raises ValueError."""
  from google.adk.workflow._function_node import FunctionNode

  def func_b() -> str:
    return 'from_b'

  def func_c() -> str:
    return 'from_c'

  async def func_a(ctx: Context):
    await ctx.run_node(func_b, use_as_output=True)
    await ctx.run_node(func_c, use_as_output=True)

  node_a = FunctionNode(func=func_a, rerun_on_resume=True)

  wf = Workflow(name='wf', edges=[(START, node_a)])

  with pytest.raises(ValueError, match='use_as_output'):
    await _run_workflow(wf)


@pytest.mark.asyncio
async def test_without_use_as_output_parent_emits_duplicate():
  """Without use_as_output, parent re-emits child's output (duplicate)."""
  from google.adk.workflow._function_node import FunctionNode

  def func_b() -> str:
    return 'from_b'

  node_b = FunctionNode(func=func_b)

  async def func_a(ctx: Context) -> str:
    return await ctx.run_node(node_b)

  node_a = FunctionNode(func=func_a, rerun_on_resume=True)

  wf = Workflow(name='wf', edges=[(START, node_a)])
  events, _, _ = await _run_workflow(wf)

  by_node = _output_by_node(events)
  # Both func_b AND func_a emit output (duplicate).
  assert ('func_b', 'from_b') in by_node
  assert ('func_a', 'from_b') in by_node

  # func_b's output_for is just its own path (no delegation).
  b_event = [e for e in events if e.node_info.name.split('@')[0] == 'func_b'][0]
  assert b_event.node_info.output_for == ['wf@1/func_a@1/func_b@1']
  # func_a is terminal, so its output_for includes the workflow path.
  a_event = [
      e
      for e in events
      if e.node_info.name and e.node_info.name.split('@')[0] == 'func_a'
  ][0]
  assert a_event.node_info.output_for == ['wf@1/func_a@1', 'wf@1']


@pytest.mark.asyncio
async def test_terminal_node_output_dedup():
  """Terminal node output is not duplicated by the workflow.

  The terminal node's output event includes the workflow path in
  output_for, and the workflow does not emit a separate output event.
  """

  def step_a(node_input: str) -> str:
    return node_input.upper()

  def step_b(node_input: str) -> str:
    return f'final: {node_input}'

  wf = Workflow(name='wf', edges=[(START, step_a, step_b)])
  events, _, _ = await _run_workflow(wf, 'hello')

  output_events = [e for e in events if e.output is not None]

  # step_a is not terminal — its output_for is just its own path.
  a_events = [
      e
      for e in output_events
      if e.node_info.name and e.node_info.name.split('@')[0] == 'step_a'
  ]
  assert len(a_events) == 1
  assert a_events[0].node_info.output_for == ['wf@1/step_a@1']

  # step_b is terminal — its output_for includes the workflow path.
  b_events = [
      e
      for e in output_events
      if e.node_info.name and e.node_info.name.split('@')[0] == 'step_b'
  ]
  assert len(b_events) == 1
  assert b_events[0].node_info.output_for == ['wf@1/step_b@1', 'wf@1']
  assert b_events[0].output == 'final: HELLO'

  # No duplicate output event from the workflow itself.
  wf_events = [e for e in output_events if e.node_info.path == 'wf@1']
  assert len(wf_events) == 0


@pytest.mark.asyncio
async def test_terminal_node_output_dedup_nested():
  """Terminal output dedup works for nested workflows.

  Inner workflow's terminal node output propagates to the outer
  workflow without duplication.
  """

  def inner_node(node_input: str) -> str:
    return node_input.upper()

  inner_wf = Workflow(name='inner', edges=[(START, inner_node)])

  def outer_node(node_input: str) -> str:
    return f'wrapped: {node_input}'

  outer_wf = Workflow(name='outer', edges=[(START, inner_wf, outer_node)])
  events, _, _ = await _run_workflow(outer_wf, 'test')

  output_events = [e for e in events if e.output is not None]

  # inner_node is terminal in inner_wf — includes inner_wf path.
  inner_events = [
      e
      for e in output_events
      if e.node_info.name and e.node_info.name.split('@')[0] == 'inner_node'
  ]
  assert len(inner_events) == 1
  assert inner_events[0].node_info.output_for == [
      'outer@1/inner@1/inner_node@1',
      'outer@1/inner@1',
  ]

  # outer_node is terminal in outer_wf — includes outer_wf path.
  outer_events = [
      e
      for e in output_events
      if e.node_info.name and e.node_info.name.split('@')[0] == 'outer_node'
  ]
  assert len(outer_events) == 1
  assert outer_events[0].node_info.output_for == [
      'outer@1/outer_node@1',
      'outer@1',
  ]
  assert outer_events[0].output == 'wrapped: TEST'

  # No duplicate output events from the workflows themselves.
  wf_output_events = [
      e
      for e in output_events
      if e.node_info.path in ('outer@1', 'outer@1/inner@1')
  ]
  assert len(wf_output_events) == 0


# --- wait_for_output + HITL resume ---


@pytest.mark.asyncio
async def test_wait_for_output_node_preserves_state_across_resume():
  """Wait-for-output node preserves received triggers across workflow resume.

  Setup:
    START -> NodeA (completes) -> Gate (wait_for_output, needs 2 triggers).
    START -> NodeB (interrupts) -> Gate.
  Act:
    - Turn 1: Start workflow. NodeA completes, NodeB interrupts. Gate waits (1/2).
    - Turn 2: Resume with NodeB response. NodeB completes, triggers Gate (2/2).
  Assert:
    - Gate opens in Turn 2 and produces output.
  """

  class _InterruptOnce(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      fc_id = ctx.state.get('_fc_id')
      if fc_id and ctx.resume_inputs and fc_id in ctx.resume_inputs:
        ctx.state['_fc_id'] = None
        yield 'B_done'
        return
      fc_id = f'fc-{uuid.uuid4().hex[:8]}'
      ctx.state['_fc_id'] = fc_id
      yield _make_function_call_interrupt(fc_id)

  a = _OutputNode(name='NodeA', value='A')
  b = _InterruptOnce(name='NodeB')
  gate = JoinNode(name='Gate')
  downstream = _OutputNode(name='Downstream', value='done')
  wf = Workflow(
      name='wf',
      edges=[
          (START, a, gate, downstream),
          (START, b, gate),
      ],
  )
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: A completes, B interrupts, Gate triggered once (no output)
  msg1 = types.Content(parts=[types.Part(text='go')], role='user')
  events1: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg1
  ):
    events1.append(event)

  outputs1 = [e.output for e in events1 if e.output is not None]
  assert 'A' in outputs1
  assert any(e.long_running_tool_ids for e in events1)
  # Gate should NOT have opened (only 1 of 2 triggers received)
  assert not any(isinstance(o, dict) and 'NodeA' in o for o in outputs1)

  fc_id = None
  for e in events1:
    if e.long_running_tool_ids:
      fc_id = list(e.long_running_tool_ids)[0]

  # Run 2: resume B → Gate gets 2nd trigger → opens → Downstream runs
  msg2 = types.Content(
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name='approve', id=fc_id, response={'ok': True}
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

  outputs = [e.output for e in events2 if e.output is not None]
  assert 'B_done' in outputs
  # Gate opened with both inputs collected
  gate_output = [o for o in outputs if isinstance(o, dict)]
  assert len(gate_output) == 1
  assert 'NodeA' in gate_output[0]
  assert 'NodeB' in gate_output[0]
  assert 'done' in outputs


@pytest.mark.asyncio
async def test_wait_for_output_node_in_loop_generates_unique_paths():
  """Wait-for-output node in loop generates unique paths across iterations.

  Setup:
    Workflow with a loop: START -> Process -> [NodeA, NodeB] -> Join -> Handle.
    Handle loops back to Process on first iteration, exits on second.
    NodeB interrupts on first run of each iteration.
  Act:
    - Turn 1: Start workflow. NodeA completes, NodeB interrupts. Join waits.
    - Turn 2: Resume with NodeB response. Handle loops back. Process runs again.
              NodeA completes, NodeB interrupts again.
    - Turn 3: Resume with NodeB response again. Join opens.
  Assert:
    - JoinNode runs with path ending in `Join@2` in the second iteration.
  """

  class _InterruptOnce(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      fc_id = ctx.state.get('_fc_id')
      if fc_id and ctx.resume_inputs and fc_id in ctx.resume_inputs:
        ctx.state['_fc_id'] = None
        yield 'B_done'
        return
      fc_id = 'fc-interrupt'
      ctx.state['_fc_id'] = fc_id
      yield _make_function_call_interrupt(fc_id)

  class _HandleNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      # Loop back if this is the first iteration
      count = ctx.state.get('loop_count', 0) + 1
      ctx.state['loop_count'] = count
      if count == 1:
        yield Event(route='loop')
      else:
        yield Event(route='exit', output='finished')

  process = _PassthroughNode(name='Process')
  a = _OutputNode(name='NodeA', value='A')
  b = _InterruptOnce(name='NodeB')
  join = JoinNode(name='Join')
  handle = _HandleNode(name='Handle')
  exit_node = _PassthroughNode(name='Exit')

  wf = Workflow(
      name='loop_wf',
      edges=[
          (START, process),
          (process, a),
          (process, b),
          (a, join),
          (b, join),
          (join, handle),
          (handle, {'loop': process, 'exit': exit_node}),
      ],
  )

  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Turn 1: process runs, A completes, B interrupts, Join waits
  msg1 = types.Content(parts=[types.Part(text='go')], role='user')
  events1: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg1
  ):
    events1.append(event)

  # Verify paths in Turn 1
  join_events1 = [
      e for e in events1 if e.node_info and 'Join' in e.node_info.path
  ]
  # JoinNode does not yield events until all inputs are collected.

  # Turn 2: Provide response for NodeB (InterruptNode)
  msg2 = types.Content(
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name='approve', id='fc-interrupt', response={'ok': True}
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

  # Workflow loops back. NodeB interrupts again in the second iteration.

  # Turn 3: Provide response for NodeB again!
  events3: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg2
  ):
    events3.append(event)

  # JoinNode opens in the second iteration and produces output.

  join_events3 = [
      e for e in events3 if e.node_info and 'Join@2' in e.node_info.path
  ]
  assert len(join_events3) > 0, 'JoinNode should run again in loop with @2'


# --- run_id reuse on resume ---


@pytest.mark.asyncio
async def test_run_id_reused_on_resume():
  """Resumed node reuses run_id from original interrupted run."""

  class _InterruptOnce(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      fc_id = ctx.state.get('_fc_id')
      if fc_id and ctx.resume_inputs and fc_id in ctx.resume_inputs:
        ctx.state['_fc_id'] = None
        yield 'resumed'
        return
      fc_id = str(uuid.uuid4())
      ctx.state['_fc_id'] = fc_id
      yield _make_function_call_interrupt(fc_id)

  wf = Workflow(
      name='wf',
      edges=[(START, _InterruptOnce(name='ask'))],
  )
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: interrupts
  msg1 = types.Content(parts=[types.Part(text='go')], role='user')
  events1: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg1
  ):
    events1.append(event)

  fc_id = None
  original_run_id = None
  for e in events1:
    if e.long_running_tool_ids:
      fc_id = list(e.long_running_tool_ids)[0]
      original_run_id = e.node_info.run_id

  assert fc_id is not None
  assert original_run_id is not None

  # Run 2: resume
  msg2 = types.Content(
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name='approve', id=fc_id, response={'ok': True}
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

  # Resumed node should have the same run_id
  resumed_events = [
      e
      for e in events2
      if e.node_info.path
      and e.node_info.path.endswith('ask@1')
      and e.output is not None
  ]
  assert len(resumed_events) == 1
  assert resumed_events[0].node_info.run_id == original_run_id


@pytest.mark.asyncio
async def test_route_without_output_triggers_downstream_on_resume():
  """Node with route but no output triggers downstream on resume.

  Setup:
    START -> NodeA (yields route='next') -> NodeB (interrupts).
  Act:
    - Turn 1: Run workflow. NodeA completes with route, NodeB interrupts.
    - Turn 2: Resume workflow by resolving NodeB's interrupt.
  Assert:
    - Turn 1: NodeB was triggered (indicated by interrupt).
    - Turn 2: NodeB runs and produces output (proving it was triggered).
  """

  class _TestRouteNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield Event(route='next')

  class _InterruptOnce(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      if ctx.resume_inputs and 'fc-123' in ctx.resume_inputs:
        yield Event(output='done')
        return
      yield Event(
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='approve', args={}, id='fc-123'
                      )
                  )
              ]
          ),
          long_running_tool_ids={'fc-123'},
      )

  route_node = _TestRouteNode(name='route_node')
  target_node = _InterruptOnce(name='target_node')
  wf = Workflow(
      name='wf',
      edges=[
          (START, route_node),
          (route_node, {'next': target_node}),
      ],
  )
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  msg1 = types.Content(parts=[types.Part(text='go')], role='user')
  events1: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg1
  ):
    events1.append(event)

  assert any(e.long_running_tool_ids for e in events1)

  msg2 = types.Content(
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name='approve', id='fc-123', response={'ok': True}
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

  outputs = [e.output for e in events2 if e.output is not None]
  assert 'done' in outputs


@pytest.mark.asyncio
async def test_rerun_on_resume_false_preserves_route_on_resume():
  """A rerun_on_resume=False node that sets ctx.route preserves it on resume.

  Setup:
    START -> RouteAndInterruptNode (sets ctx.route='go', interrupts)
          -> {'go': TargetNode}
  Act:
    - Turn 1: RouteAndInterruptNode sets route and interrupts.
    - Turn 2: Resume by resolving the interrupt.
  Assert:
    - Turn 2: TargetNode fires (proving the route survived the resume)
              and produces output 'reached'.
  """

  class _RouteAndInterruptNode(BaseNode):
    """Sets ctx.route directly and interrupts. rerun_on_resume=False."""

    rerun_on_resume: bool = False

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      ctx.route = 'go'
      yield Event(
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='confirm', args={}, id='fc-route-1'
                      )
                  )
              ]
          ),
          long_running_tool_ids={'fc-route-1'},
      )

  class _TargetNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield Event(output='reached')

  route_node = _RouteAndInterruptNode(name='route_node')
  target_node = _TargetNode(name='target_node')
  wf = Workflow(
      name='wf',
      edges=[
          (START, route_node),
          (route_node, {'go': target_node}),
      ],
  )
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Turn 1: route_node sets route and interrupts.
  msg1 = types.Content(parts=[types.Part(text='start')], role='user')
  events1: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg1
  ):
    events1.append(event)

  assert any(e.long_running_tool_ids for e in events1)

  # Turn 2: resolve the interrupt.
  msg2 = types.Content(
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name='confirm', id='fc-route-1', response={'ok': True}
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

  # target_node should have fired via the 'go' route.
  outputs = [e.output for e in events2 if e.output is not None]
  assert 'reached' in outputs


@pytest.mark.asyncio
async def test_route_and_output_triggers_downstream_on_resume():
  """Node with route and output triggers downstream on resume.

  Setup:
    START -> NodeA (yields route='next', output='A') -> NodeB (interrupts).
  Act:
    - Turn 1: Run workflow. NodeA completes with route and output, NodeB interrupts.
    - Turn 2: Resume workflow by resolving NodeB's interrupt.
  Assert:
    - Turn 1: NodeB was triggered (indicated by interrupt).
    - Turn 2: NodeB runs and produces output (proving it was triggered).
  """

  class _RouteAndOutputNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield Event(route='next', output='A')

  class _InterruptOnce(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      if ctx.resume_inputs and 'fc-123' in ctx.resume_inputs:
        assert node_input == 'A'
        yield Event(output='done')
        return
      yield Event(
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='approve', args={}, id='fc-123'
                      )
                  )
              ]
          ),
          long_running_tool_ids={'fc-123'},
      )

  route_node = _RouteAndOutputNode(name='route_node')
  target_node = _InterruptOnce(name='target_node')
  wf = Workflow(
      name='wf',
      edges=[
          (START, route_node),
          (route_node, {'next': target_node}),
      ],
  )
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  msg1 = types.Content(parts=[types.Part(text='go')], role='user')
  events1: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg1
  ):
    events1.append(event)

  assert any(e.long_running_tool_ids for e in events1)

  msg2 = types.Content(
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name='approve', id='fc-123', response={'ok': True}
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

  outputs = [e.output for e in events2 if e.output is not None]
  assert 'done' in outputs
