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

"""Tests for dynamic node scheduling in new Workflow.

Covers the three cases from doc 18 (Dynamic Node Resume via Lazy Scan):
- Case 1: Fresh execution (no prior events)
- Case 2: Completed dedup (return cached output on rerun)
- Case 3: Interrupted resume (rerun with resume_inputs)

Plus edge cases for multiple dynamic nodes, nested dynamic nodes,
and use_as_output delegation.
"""

import asyncio
from typing import Any
from typing import AsyncGenerator

from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow import node
from google.adk.workflow import START
from google.adk.workflow._errors import DynamicNodeFailError
from google.adk.workflow._workflow import Workflow
from google.genai import types
import pytest

# --- Helpers ---


async def _run(
    runner: Runner,
    ss: InMemorySessionService,
    session: Any,
    message: str,
) -> list[Event]:
  """Send a text message and collect events."""
  msg = types.Content(parts=[types.Part(text=message)], role='user')
  events: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg
  ):
    events.append(event)
  return events


async def _resume(
    runner: Runner,
    ss: InMemorySessionService,
    session: Any,
    fc_id: str,
    response: Any,
) -> list[Event]:
  """Send a function response and collect events."""
  if not isinstance(response, dict):
    response = {'value': response}
  msg = types.Content(
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name='tool', id=fc_id, response=response
              )
          )
      ],
      role='user',
  )
  events: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg
  ):
    events.append(event)
  return events


def _outputs(events: list[Event]) -> list[Any]:
  """Extract non-None outputs."""
  return [e.output for e in events if e.output is not None]


def _interrupt_ids(events: list[Event]) -> set[str]:
  """Extract interrupt IDs from events."""
  ids: set[str] = set()
  for e in events:
    if e.long_running_tool_ids:
      ids.update(e.long_running_tool_ids)
  return ids


# =========================================================================
# Fresh execution (no resume)
# =========================================================================


@pytest.mark.asyncio
async def test_dynamic_node_fresh_execution():
  """Dynamic node runs normally on first invocation.

  Setup: Parent (rerun_on_resume=True) calls ctx.run_node(Child).
  Action: Send a text message to trigger the workflow.
  Assert: Parent receives Child's output and yields the combined result.
  """

  @node
  async def child(*, ctx, node_input):
    yield f'child got: {node_input}'

  @node(rerun_on_resume=True)
  async def parent(*, ctx, node_input):
    result = await ctx.run_node(child, node_input='hello')
    yield f'parent got: {result}'

  wf = Workflow(name='wf', edges=[(START, parent)])
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  events = await _run(runner, ss, session, 'go')

  outputs = _outputs(events)
  assert 'parent got: child got: hello' in outputs


@pytest.mark.asyncio
async def test_dynamic_node_with_downstream_static():
  """Dynamic child's output flows to a downstream static node.

  Setup: Parent calls ctx.run_node(Child). Parent is followed by
    a static After node in the graph.
  Action: Send a text message.
  Assert: After node receives Parent's output (which includes
    Child's result) as node_input.
  """

  @node
  async def child(*, ctx, node_input):
    yield f'child: {node_input}'

  @node(rerun_on_resume=True)
  async def parent(*, ctx, node_input):
    result = await ctx.run_node(child, node_input='forwarded')
    yield result

  @node
  async def after(*, ctx, node_input):
    yield f'after: {node_input}'

  wf = Workflow(
      name='wf',
      edges=[(START, parent, after)],
  )
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  events = await _run(runner, ss, session, 'hello')
  outputs = _outputs(events)
  assert 'after: child: forwarded' in outputs


# =========================================================================
# Single-level resume (interrupt → FR → resume)
# =========================================================================


@pytest.mark.asyncio
async def test_dynamic_node_interrupted_resume():
  """Dynamic child interrupts, then resumes with FR on parent rerun.

  Setup: Parent calls ctx.run_node(Approver). Approver interrupts
    with fc-1.
  Action: Send FR for fc-1 with {answer: 'yes'}.
  Assert:
    - Approver resumes and outputs 'approved: yes'.
    - Parent yields the final combined result.
  """

  @node(rerun_on_resume=True)
  async def approver(*, ctx, node_input):
    if ctx.resume_inputs and 'fc-1' in ctx.resume_inputs:
      yield f'approved: {ctx.resume_inputs["fc-1"]["answer"]}'
      return
    yield Event(
        content=types.Content(
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name='approve', args={}, id='fc-1'
                    )
                )
            ]
        ),
        long_running_tool_ids={'fc-1'},
    )

  @node(rerun_on_resume=True)
  async def parent(*, ctx, node_input):
    result = await ctx.run_node(approver)
    yield f'final: {result}'

  wf = Workflow(name='wf', edges=[(START, parent)])
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: interrupts
  events1 = await _run(runner, ss, session, 'go')
  assert 'fc-1' in _interrupt_ids(events1)

  # Run 2: resume
  events2 = await _resume(runner, ss, session, 'fc-1', {'answer': 'yes'})
  outputs = _outputs(events2)
  assert 'final: approved: yes' in outputs


@pytest.mark.asyncio
async def test_dynamic_node_completed_dedup_on_resume():
  """Completed dynamic child returns cached output when parent reruns.

  Setup: Parent calls ctx.run_node(Completer) then
    ctx.run_node(Interrupter). Completer completes, Interrupter
    interrupts with fc-1.
  Action: Send FR for fc-1 to resume.
  Assert:
    - Parent reruns (call_count increments to 2).
    - Completer is NOT re-executed (dedup returns cached output).
    - Interrupter resumes with the FR response.
    - Final output combines both results.
  """

  @node
  async def completer(*, ctx, node_input):
    yield 'completed_result'

  @node(rerun_on_resume=True)
  async def interrupter(*, ctx, node_input):
    if ctx.resume_inputs and 'fc-1' in ctx.resume_inputs:
      yield f'resumed: {ctx.resume_inputs["fc-1"]["value"]}'
      return
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

  call_count = [0]

  @node(rerun_on_resume=True)
  async def parent(*, ctx, node_input):
    call_count[0] += 1

    r1 = await ctx.run_node(completer)
    r2 = await ctx.run_node(interrupter)
    yield f'{r1} + {r2}'

  wf = Workflow(name='wf', edges=[(START, parent)])
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: completer completes, interrupter interrupts
  events1 = await _run(runner, ss, session, 'go')
  assert _interrupt_ids(events1)
  assert call_count[0] == 1

  # Run 2: resume interrupter
  events2 = await _resume(runner, ss, session, 'fc-1', 'done')

  # Parent should have rerun (call_count=2).
  # Completer should NOT re-execute (dedup returns cached).
  # Interrupter resumes with FR.
  assert call_count[0] == 2
  outputs = _outputs(events2)
  assert 'completed_result + resumed: done' in outputs


@pytest.mark.asyncio
async def test_dynamic_node_sequential_interrupts():
  """Sequential dynamic children interrupt one at a time.

  Setup: Parent calls ctx.run_node(a) then ctx.run_node(b).
    Both children interrupt, but sequentially — a interrupts first,
    parent never reaches b until a is resolved.
  Action: Resume a, then b.
  Assert:
    - Run 1: only fc-a interrupts (parent didn't reach b).
    - Run 2 (resume fc-a): a completes, parent continues to b,
      b interrupts with fc-b.
    - Run 3 (resume fc-b): b completes, parent yields combined
      output.
  """

  @node(rerun_on_resume=True)
  async def a(*, ctx, node_input):
    if ctx.resume_inputs and 'fc-a' in ctx.resume_inputs:
      yield f'a: {ctx.resume_inputs["fc-a"]["value"]}'
      return
    yield Event(
        content=types.Content(
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name='tool', args={}, id='fc-a'
                    )
                )
            ]
        ),
        long_running_tool_ids={'fc-a'},
    )

  @node(rerun_on_resume=True)
  async def b(*, ctx, node_input):
    if ctx.resume_inputs and 'fc-b' in ctx.resume_inputs:
      yield f'b: {ctx.resume_inputs["fc-b"]["value"]}'
      return
    yield Event(
        content=types.Content(
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name='tool', args={}, id='fc-b'
                    )
                )
            ]
        ),
        long_running_tool_ids={'fc-b'},
    )

  @node(rerun_on_resume=True)
  async def parent(*, ctx, node_input):
    r1 = await ctx.run_node(a, node_input=node_input)
    r2 = await ctx.run_node(b, node_input=node_input)
    yield f'{r1} + {r2}'

  wf = Workflow(name='wf', edges=[(START, parent)])
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: a interrupts, parent never reaches b
  events1 = await _run(runner, ss, session, 'go')
  ids1 = _interrupt_ids(events1)
  assert 'fc-a' in ids1
  assert 'fc-b' not in ids1

  # Run 2: resume a → a completes, parent reaches b → b interrupts
  events2 = await _resume(runner, ss, session, 'fc-a', 'done-a')
  ids2 = _interrupt_ids(events2)
  assert 'fc-b' in ids2

  # Run 3: resume b → b completes, parent yields combined output
  events3 = await _resume(runner, ss, session, 'fc-b', 'done-b')
  outputs = _outputs(events3)
  assert 'a: done-a + b: done-b' in outputs


@pytest.mark.asyncio
async def test_dynamic_node_run_id_reused_on_resume():
  """Resumed dynamic child reuses run_id from original run.

  Setup: Parent calls ctx.run_node(Interrupter). Interrupter
    interrupts with fc-1.
  Action: Send FR for fc-1.
  Assert: The resumed Interrupter's output event has the same
    run_id as the original interrupt event.
  """

  @node(rerun_on_resume=True)
  async def interrupter(*, ctx, node_input):
    if ctx.resume_inputs and 'fc-1' in ctx.resume_inputs:
      yield 'resumed'
      return
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

  @node(rerun_on_resume=True)
  async def parent(*, ctx, node_input):
    result = await ctx.run_node(interrupter)
    yield result

  wf = Workflow(name='wf', edges=[(START, parent)])
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: interrupts
  events1 = await _run(runner, ss, session, 'go')
  interrupt_event = [e for e in events1 if e.long_running_tool_ids][0]
  original_run_id = interrupt_event.node_info.run_id

  # Run 2: resume
  events2 = await _resume(runner, ss, session, 'fc-1', 'ok')
  resumed_output_events = [
      e
      for e in events2
      if e.output == 'resumed' and 'interrupter' in (e.node_info.path or '')
  ]
  assert len(resumed_output_events) == 1
  assert resumed_output_events[0].node_info.run_id == original_run_id


# =========================================================================
# Nested workflow + dynamic node combinations
# =========================================================================


@pytest.mark.asyncio
async def test_nested_static_workflow_with_dynamic_interrupt():
  """Static sub-workflow's node schedules a dynamic node that interrupts.

  Setup: outer_wf → inner_wf (static). Inside inner_wf, a static
    parent node calls ctx.run_node(Approver). Approver interrupts.
  Action: Send FR to resume.
  Assert: Approver resumes, parent completes, inner_wf completes,
    outer_wf completes.
  """

  @node(rerun_on_resume=True)
  async def approver(*, ctx, node_input):
    if ctx.resume_inputs and 'fc-1' in ctx.resume_inputs:
      yield f'approved: {ctx.resume_inputs["fc-1"]["value"]}'
      return
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

  @node(rerun_on_resume=True)
  async def parent(*, ctx, node_input):
    result = await ctx.run_node(approver)
    yield f'parent: {result}'

  inner_wf = Workflow(
      name='inner_wf',
      edges=[(START, parent)],
  )
  outer_wf = Workflow(
      name='outer_wf',
      edges=[(START, inner_wf)],
  )
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=outer_wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: dynamic approver interrupts inside inner_wf
  events1 = await _run(runner, ss, session, 'go')
  assert 'fc-1' in _interrupt_ids(events1)

  # Run 2: resume
  events2 = await _resume(runner, ss, session, 'fc-1', 'yes')
  outputs = _outputs(events2)
  assert 'parent: approved: yes' in outputs


@pytest.mark.asyncio
async def test_dynamic_workflow_with_static_interrupt():
  """Dynamic child is a Workflow whose static node interrupts.

  Setup: Parent calls ctx.run_node(inner_wf). inner_wf has a
    static Interrupter node that interrupts with fc-1.
  Action: Send FR to resume.
  Assert: Interrupter resumes, inner_wf completes, parent
    receives inner_wf's output.
  """

  @node(rerun_on_resume=True)
  async def step(*, ctx, node_input):
    if ctx.resume_inputs and 'fc-1' in ctx.resume_inputs:
      yield f'done: {ctx.resume_inputs["fc-1"]["value"]}'
      return
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

  inner_wf = Workflow(
      name='inner_wf',
      edges=[(START, step)],
  )

  @node(rerun_on_resume=True)
  async def parent(*, ctx, node_input):
    result = await ctx.run_node(inner_wf)
    yield f'parent: {result}'

  outer_wf = Workflow(
      name='outer_wf',
      edges=[(START, parent)],
  )
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=outer_wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: inner_wf's static node interrupts
  events1 = await _run(runner, ss, session, 'go')
  assert 'fc-1' in _interrupt_ids(events1)

  # Run 2: resume
  events2 = await _resume(runner, ss, session, 'fc-1', 'ok')
  outputs = _outputs(events2)
  assert 'parent: done: ok' in outputs


@pytest.mark.asyncio
async def test_dynamic_workflow_with_nested_dynamic_interrupt():
  """Dynamic Workflow's inner node schedules another dynamic node.

  Setup: Parent calls ctx.run_node(inner_wf). inner_wf has a
    static Orchestrator node that calls ctx.run_node(Approver).
    Approver interrupts with fc-1.
  Action: Send FR to resume.
  Assert: Approver resumes → Orchestrator completes → inner_wf
    completes → Parent receives output. Three levels of dynamic
    nesting resolved correctly.
  """

  @node(rerun_on_resume=True)
  async def approver(*, ctx, node_input):
    if ctx.resume_inputs and 'fc-1' in ctx.resume_inputs:
      yield f'approved: {ctx.resume_inputs["fc-1"]["value"]}'
      return
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

  @node(rerun_on_resume=True)
  async def orch(*, ctx, node_input):
    result = await ctx.run_node(approver)
    yield f'orch: {result}'

  inner_wf = Workflow(
      name='inner_wf',
      edges=[(START, orch)],
  )

  @node(rerun_on_resume=True)
  async def parent(*, ctx, node_input):
    result = await ctx.run_node(inner_wf)
    yield f'parent: {result}'

  outer_wf = Workflow(
      name='outer_wf',
      edges=[(START, parent)],
  )
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=outer_wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: deeply nested approver interrupts
  events1 = await _run(runner, ss, session, 'go')
  assert 'fc-1' in _interrupt_ids(events1)

  # Run 2: resume
  events2 = await _resume(runner, ss, session, 'fc-1', 'granted')
  outputs = _outputs(events2)
  assert 'parent: orch: approved: granted' in outputs


# =========================================================================
# Scoping: parallel parents with same-named dynamic children
# =========================================================================


@pytest.mark.asyncio
async def test_parallel_parents_same_named_dynamic_children():
  """Two static parents schedule dynamic children with the same name.

  Setup: outer_wf fans out to parent_a and parent_b (parallel).
    Both call ctx.run_node(Child(name='child')). parent_a's child
    completes, parent_b's child interrupts.
  Action: Send FR to resume parent_b's child.
  Assert:
    - parent_a's child and parent_b's child are distinct (scoped
      by parent_path: wf/parent_a/child vs wf/parent_b/child).
    - On resume, parent_a's child returns cached output (dedup),
      parent_b's child resumes with FR.
    - Both parents complete.
  """

  @node(rerun_on_resume=True)
  async def child(*, ctx, node_input):
    if ctx.resume_inputs and 'fc-b' in ctx.resume_inputs:
      yield f'resumed: {ctx.resume_inputs["fc-b"]["value"]}'
      return
    if node_input == 'interrupt':
      yield Event(
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='tool', args={}, id='fc-b'
                      )
                  )
              ]
          ),
          long_running_tool_ids={'fc-b'},
      )
    else:
      yield f'child: {node_input}'

  @node(rerun_on_resume=True)
  async def parent_a(*, ctx, node_input):
    result = await ctx.run_node(child, node_input='complete')
    yield f'a: {result}'

  @node(rerun_on_resume=True)
  async def parent_b(*, ctx, node_input):
    result = await ctx.run_node(child, node_input='interrupt')
    yield f'b: {result}'

  from google.adk.workflow import JoinNode

  join = JoinNode(name='join')
  wf = Workflow(
      name='wf',
      edges=[
          (START, parent_a, join),
          (START, parent_b, join),
      ],
  )
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: parent_a completes, parent_b's child interrupts
  events1 = await _run(runner, ss, session, 'go')
  assert 'fc-b' in _interrupt_ids(events1)
  # parent_a should have completed
  a_outputs = [
      e.output for e in events1 if e.output and 'a: child' in str(e.output)
  ]
  assert len(a_outputs) == 1

  # Run 2: resume parent_b's child
  events2 = await _resume(runner, ss, session, 'fc-b', 'done')
  outputs = _outputs(events2)
  # Both parents completed, join has both results
  assert any('b: resumed: done' in str(o) for o in outputs)


# =========================================================================
# use_as_output + interrupt
# =========================================================================


@pytest.mark.asyncio
async def test_dynamic_node_use_as_output_with_interrupt():
  """Dynamic child with use_as_output=True interrupts then resumes.

  Setup: Parent calls ctx.run_node(child, use_as_output=True).
    Child interrupts with fc-1.
  Action: Send FR for fc-1.
  Assert:
    - On resume, child resumes and its output becomes the parent's
      output (use_as_output delegation).
    - The parent's own output event is suppressed.
  """

  @node(rerun_on_resume=True)
  async def child(*, ctx, node_input):
    if ctx.resume_inputs and 'fc-1' in ctx.resume_inputs:
      yield f'child: {ctx.resume_inputs["fc-1"]["value"]}'
      return
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

  @node(rerun_on_resume=True)
  async def parent(*, ctx, node_input):
    result = await ctx.run_node(child, use_as_output=True)
    # Set on ctx so orchestrator reads it. _output_delegated
    # suppresses the output Event (child already emitted it).
    ctx.output = result
    yield  # keep as async generator

  wf = Workflow(
      name='wf',
      edges=[(START, parent)],
  )
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: child interrupts
  events1 = await _run(runner, ss, session, 'go')
  assert 'fc-1' in _interrupt_ids(events1)

  # Run 2: resume
  events2 = await _resume(runner, ss, session, 'fc-1', 'approved')
  outputs = _outputs(events2)
  assert 'child: approved' in outputs
  # Parent's output should be suppressed (use_as_output)
  parent_outputs = [
      e.output
      for e in events2
      if e.node_info.path == 'wf/parent' and e.output is not None
  ]
  assert len(parent_outputs) == 0


# =========================================================================
# None-output completion after interrupt
# =========================================================================


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason='No completion marker event for None output (event design flaw)'
)
async def test_dynamic_node_none_output_not_rerun():
  """Dynamic child that completed with None output is not re-run.

  Setup: Parent calls ctx.run_node(A) then ctx.run_node(B).
    A interrupts. On resume, A completes with no output (None).
    Parent continues to B, which also interrupts.
  Action: Resume B.
  Assert:
    - On Run 3, A should NOT re-run (it already completed).
    - A should return None (cached), B resumes.
    - Currently fails because A's None completion leaves no
      trace in session events — the lazy scan thinks A still
      needs to re-run.
  """
  run_count_a = [0]

  @node(rerun_on_resume=True)
  async def a(*, ctx, node_input):
    run_count_a[0] += 1
    if ctx.resume_inputs and 'fc-a' in ctx.resume_inputs:
      # Complete with no output.
      return
    yield Event(
        content=types.Content(
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name='tool', args={}, id='fc-a'
                    )
                )
            ]
        ),
        long_running_tool_ids={'fc-a'},
    )

  @node(rerun_on_resume=True)
  async def b(*, ctx, node_input):
    if ctx.resume_inputs and 'fc-b' in ctx.resume_inputs:
      yield f'b: {ctx.resume_inputs["fc-b"]["value"]}'
      return
    yield Event(
        content=types.Content(
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name='tool', args={}, id='fc-b'
                    )
                )
            ]
        ),
        long_running_tool_ids={'fc-b'},
    )

  @node(rerun_on_resume=True)
  async def parent(*, ctx, node_input):
    await ctx.run_node(a)
    result = await ctx.run_node(b)
    yield f'done: {result}'

  wf = Workflow(name='wf', edges=[(START, parent)])
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: A interrupts
  events1 = await _run(runner, ss, session, 'go')
  assert 'fc-a' in _interrupt_ids(events1)
  assert run_count_a[0] == 1

  # Run 2: resume A → A completes (None), parent reaches B → B interrupts
  events2 = await _resume(runner, ss, session, 'fc-a', 'ok')
  assert 'fc-b' in _interrupt_ids(events2)
  assert run_count_a[0] == 2  # A re-ran once for resume

  # Run 3: resume B → A should NOT re-run (already completed)
  events3 = await _resume(runner, ss, session, 'fc-b', 'done')
  assert run_count_a[0] == 2  # A should NOT have run again
  outputs = _outputs(events3)
  assert any('done:' in str(o) for o in outputs)


# =========================================================================
# rerun_on_resume=False for dynamic node
# =========================================================================


@pytest.mark.asyncio
async def test_dynamic_node_rerun_on_resume_false():
  """Dynamic child with rerun_on_resume=False auto-completes on resume.

  Setup: Parent calls ctx.run_node(child). Child has
    rerun_on_resume=False and interrupts with fc-1.
  Action: Send FR for fc-1.
  Assert:
    - Child does NOT re-execute _run_impl.
    - Child auto-completes with the FR response as output.
    - Parent receives the auto-completed output.
  """
  run_count = [0]

  @node(rerun_on_resume=False)
  async def child(*, ctx, node_input):
    run_count[0] += 1
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

  @node(rerun_on_resume=True)
  async def parent(*, ctx, node_input):
    result = await ctx.run_node(child)
    yield f'parent: {result}'

  wf = Workflow(
      name='wf',
      edges=[(START, parent)],
  )
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: child interrupts
  events1 = await _run(runner, ss, session, 'go')
  assert 'fc-1' in _interrupt_ids(events1)
  assert run_count[0] == 1

  # Run 2: resume
  events2 = await _resume(runner, ss, session, 'fc-1', {'answer': 42})

  # Child should NOT have re-executed (run_count stays 1).
  assert run_count[0] == 1
  # Parent receives the FR response as child's output (unwrapped).
  outputs = _outputs(events2)
  assert "parent: {'answer': 42}" in outputs


# =========================================================================
# Sequential run_id
# =========================================================================


@pytest.mark.asyncio
async def test_dynamic_nodes_get_run_id_one():
  """Each distinct dynamic child gets run_id '1' for its first run."""

  @node
  async def step_a(*, ctx, node_input):
    yield f'child: {node_input}'

  @node
  async def step_b(*, ctx, node_input):
    yield f'child: {node_input}'

  @node(rerun_on_resume=True)
  async def parent(*, ctx, node_input):
    a = await ctx.run_node(step_a, node_input='x')
    b = await ctx.run_node(step_b, node_input='y')
    yield f'{a},{b}'

  wf = Workflow(name='wf', edges=[(START, parent)])
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  events = await _run(runner, ss, session, 'go')

  child_events = [
      (e.node_name, e.node_info.path.split('@')[-1])
      for e in events
      if e.output is not None
      and e.node_name
      and e.node_name.startswith('step_')
  ]
  # Each dynamic child is a distinct path, each gets run_id '1'.
  assert child_events == [('step_a', '1'), ('step_b', '1')]


@pytest.mark.asyncio
async def test_dynamic_node_keeps_run_id_on_resume():
  """A dynamic node that interrupts and resumes keeps the same run_id."""

  @node(rerun_on_resume=True)
  async def approver(*, ctx, node_input):
    if ctx.resume_inputs and 'fc-1' in ctx.resume_inputs:
      yield f'approved: {ctx.resume_inputs["fc-1"]["answer"]}'
      return
    yield Event(
        content=types.Content(
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name='approve', args={}, id='fc-1'
                    )
                )
            ]
        ),
        long_running_tool_ids={'fc-1'},
    )

  @node(rerun_on_resume=True)
  async def parent(*, ctx, node_input):
    result = await ctx.run_node(approver)
    yield f'final: {result}'

  wf = Workflow(name='wf', edges=[(START, parent)])
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: child interrupts.
  events1 = await _run(runner, ss, session, 'go')
  approver_run_ids_1 = [
      e.node_info.path.split('@')[-1]
      for e in events1
      if e.node_info.path and 'approver@' in e.node_info.path
  ]

  # Run 2: resume with function response.
  events2 = await _resume(runner, ss, session, 'fc-1', {'answer': 'yes'})
  approver_run_ids_2 = [
      e.node_info.path.split('@')[-1]
      for e in events2
      if e.node_info.path and 'approver@' in e.node_info.path
  ]

  # Same run_id across interrupt and resume.
  assert approver_run_ids_1
  assert approver_run_ids_2
  assert approver_run_ids_1[0] == approver_run_ids_2[0]


# =========================================================================
# Custom run_id
# =========================================================================


@pytest.mark.asyncio
async def test_custom_run_id_used_on_events():
  """ctx.run_node(run_id=...) sets the custom run_id on child events."""

  @node
  async def child(*, ctx, node_input):
    yield f'done: {node_input}'

  @node(rerun_on_resume=True)
  async def parent(*, ctx, node_input):
    result = await ctx.run_node(
        child, node_input='hello', run_id='my-custom-id'
    )
    yield result

  wf = Workflow(name='wf', edges=[(START, parent)])
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  events = await _run(runner, ss, session, 'go')

  child_events = [
      e
      for e in events
      if e.node_info and e.node_info.path and 'child' in e.node_info.path
  ]
  assert child_events
  assert all(
      e.node_info.path.split('@')[-1] == 'my-custom-id' for e in child_events
  )


# =========================================================================
# Failure handling in dynamic nodes
# =========================================================================


@pytest.mark.asyncio
async def test_dynamic_node_failure_handling():
  """Dynamic node throws exception; parent catches it and continues."""

  @node
  async def failing_node(*, ctx, node_input):
    if node_input == 'fail':
      raise ValueError('Intentional Failure')
    yield f'Processed {node_input}'

  @node(rerun_on_resume=True)
  async def parent(*, ctx, node_input):
    results = []
    try:
      await ctx.run_node(failing_node, node_input='fail')
    except DynamicNodeFailError as e:
      results.append(f'Caught: {str(e.error)}')

    res = await ctx.run_node(failing_node, node_input='work')
    results.append(f'Success: {res}')
    yield results

  wf = Workflow(name='wf', edges=[(START, parent)])
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  events = await _run(runner, ss, session, 'go')
  outputs = _outputs(events)

  # Find the list output from parent
  list_outputs = [o for o in outputs if isinstance(o, list)]
  assert len(list_outputs) == 1
  results = list_outputs[0]
  assert 'Caught: Intentional Failure' in results
  assert 'Success: Processed work' in results


@pytest.mark.asyncio
async def test_workflow_resume_does_not_rerun_completed_llm_agent():
  """Completed LlmAgent node is not rerun upon workflow resumption.

  Setup: Workflow with LlmAgent node and an interrupting node.
  Act:
    - Run 1: Start workflow, LlmAgent completes, workflow interrupts.
    - Run 2: Resume workflow by resolving interrupt.
  Assert:
    - LlmAgent does not run again in Run 2.
  """
  from google.adk.agents.llm_agent import LlmAgent

  from tests.unittests import testing_utils

  # Given a workflow with an LlmAgent and a mock model
  mock_model = testing_utils.MockModel.create(
      responses=['LLM output content', 'Duplicate run output']
  )

  agent = LlmAgent(name='my_agent', model=mock_model)

  @node
  async def interrupt_node(*, ctx, node_input):
    event = Event(
        content=types.Content(
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name='tool', id='interrupt_1', args={}
                    )
                )
            ]
        )
    )
    event.long_running_tool_ids = {'interrupt_1'}
    yield event
    yield f'Resumed with {node_input}'

  @node(rerun_on_resume=True)
  async def parent(*, ctx, node_input):
    res = await ctx.run_node(agent, node_input='go')
    res2 = await ctx.run_node(interrupt_node, node_input=res)
    yield res2

  wf = Workflow(name='wf', edges=[(START, parent)])
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # When the workflow is run until it interrupts
  await _run(runner, ss, session, 'go')

  session = await ss.get_session(
      app_name='test', user_id='u', session_id=session.id
  )

  agent_events = [
      e for e in session.events if e.node_info.name == 'my_agent' and e.content
  ]
  assert len(agent_events) > 0
  agent_event = agent_events[-1]

  # Verify that runners.py cleared the output
  assert agent_event.output is None

  # When the workflow is resumed by resolving the interrupt
  resume_events = await _resume(
      runner, ss, session, fc_id='interrupt_1', response='done'
  )

  # Then the LlmAgent should not run again
  agent_runs_again = [
      e for e in resume_events if e.node_info.name == 'my_agent' and e.content
  ]
  assert (
      len(agent_runs_again) == 0
  ), 'Expected LlmAgent to NOT run again, but it did!'


# =========================================================================
# Parallel execution of dynamic nodes
# =========================================================================


@pytest.mark.asyncio
async def test_dynamic_node_parallel_execution():
  """Three parallel ctx.run_node calls via asyncio.gather return ordered results."""

  @node
  async def echo_node(*, ctx, node_input):
    yield node_input

  @node(rerun_on_resume=True)
  async def parent_node(*, ctx, node_input):
    tasks = [ctx.run_node(echo_node, node_input=f'call_{i}') for i in range(3)]
    results = await asyncio.gather(*tasks)
    yield results

  wf = Workflow(name='wf', edges=[(START, parent_node)])
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  events = await _run(runner, ss, session, 'go')
  outputs = _outputs(events)

  # Find the list output from parent
  list_outputs = [o for o in outputs if isinstance(o, list)]
  assert len(list_outputs) == 1
  results = list_outputs[0]
  assert results == ['call_0', 'call_1', 'call_2']


@pytest.mark.asyncio
async def test_inner_llm_agent_node_input_survives_tool_round_trip():
  """Inner LlmAgent's node_input reaches every LLM request across a tool round trip.

  Setup: Workflow with a single dynamic node whose body invokes an inner
    LlmAgent. The agent's mock model emits a function_call on the first
    turn and final text on the second turn.
  Act: Run the workflow once; the dynamic node passes USER_PHRASE to the
    inner agent.
  Assert:
    - The mock model is called exactly twice (pre- and post-tool).
    - USER_PHRASE appears in the user-role text of both LLM requests.
    - The second request contains both function_call and function_response
      (proving a real tool round trip occurred).
  """
  from google.adk.agents.llm_agent import LlmAgent
  from google.adk.tools import FunctionTool

  from tests.unittests import testing_utils

  USER_PHRASE = 'lookup price for token MARKER-7Z9'

  def lookup_price(token: str) -> dict:
    return {'price': '$9.99'}

  def _user_texts(req) -> list[str]:
    texts: list[str] = []
    for content in req.contents or []:
      if content.role != 'user':
        continue
      for part in content.parts or []:
        if part.text and not part.text.startswith('For context:'):
          texts.append(part.text)
    return texts

  def _part_kinds(req) -> list[str]:
    kinds: list[str] = []
    for content in req.contents or []:
      for part in content.parts or []:
        if part.function_call:
          kinds.append('function_call')
        elif part.function_response:
          kinds.append('function_response')
        else:
          kinds.append('text')
    return kinds

  mock_model = testing_utils.MockModel.create(
      responses=[
          [
              types.Part(
                  function_call=types.FunctionCall(
                      id='fc1',
                      name='lookup_price',
                      args={'token': 'MARKER-7Z9'},
                  )
              )
          ],
          'The price is $9.99.',
      ]
  )

  inner_agent = LlmAgent(
      name='inner_agent',
      model=mock_model,
      tools=[FunctionTool(lookup_price)],
  )

  @node(rerun_on_resume=True)
  async def parent(*, ctx, node_input):
    yield await ctx.run_node(inner_agent, node_input=USER_PHRASE)

  wf = Workflow(name='wf', edges=[(START, parent)])
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # When the workflow runs end-to-end through the tool round trip
  await _run(runner, ss, session, 'go')

  # Then the model is invoked exactly twice
  assert len(mock_model.requests) == 2

  # And USER_PHRASE survives into both LLM requests
  assert any(USER_PHRASE in t for t in _user_texts(mock_model.requests[0]))
  assert any(
      USER_PHRASE in t for t in _user_texts(mock_model.requests[1])
  ), 'original node_input was dropped from the second LLM request'

  # And the second request reflects a real tool round trip
  second_request_kinds = _part_kinds(mock_model.requests[1])
  assert 'function_call' in second_request_kinds
  assert 'function_response' in second_request_kinds
